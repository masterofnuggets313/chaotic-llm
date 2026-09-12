# -*- coding: utf-8 -*-
"""Построить бенчмарк «нашёлся ли факт» для ЛЮБОГО корпуса.

Зачем отдельный скрипт. Прежний бенч (`hermes_sem_bench_big.json`) built на личной
переписке владельца: её нельзя публиковать, а значит никто не мог повторить числа
из README. Число, которое нельзя повторить, для сообщества не стоит ничего.
Здесь вопросы строятся из того корпуса, который дадите ВЫ.

Что делает:
  1. Режет корпус на окна теми же параметрами, что и билдер
     (chunk 800 / overlap 160, шаг 640 — НЕ stride от длины).
  2. Выбирает n случайных окон (детерминированно по seed).
  3. Для каждого окна просит LLM придумать вопрос, ответом на который является
     именно этот фрагмент. Режим --no-llm строит запрос из редких терминов.
  4. Считает rare_terms локально (частота слов по корпусу, без сети) — это
     объективная метрика «нашлись ли специфичные для фрагмента слова».
  5. Режет утечки: если вопрос дословно цитирует фрагмент, такой вопрос
     отбрасывается — иначе метрика мерила бы поиск подстроки, а не смысла.

Выход: JSON-список [{'query','char_pos','snippet','rare_terms'}, ...], который
понимает `_bench_fact.py`.

    python make_bench.py --corpus my.txt --out bench.json --n 200
    python make_bench.py --corpus my.txt --out bench.json --n 200 --no-llm
"""
import argparse, json, os, random, re, sys
import urllib.request

# Локальные адреса — напрямую, в обход http_proxy (иначе 502 на живом сервере).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CHUNK, OVERLAP = 800, 160
STEP = CHUNK - OVERLAP

STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
уже или быть ни два ли до меня же нам чем чтобы без будто чего раз тоже собой под
будет ж себя надо там станет этом том этот того этой эти этот может есть нею него
них ним ними ней неё свою своём при про для них всего самый самую себе какая какой
какое какие кто куда где откуда почему зачем здесь тут там тогда потом опять очень
more the a an of and or to in is are was were be been this that it its for on with
as at by from not but if then than so such can could will would should may might
""".split())

PROMPT = """Вот фрагмент текста. Придумай ОДИН вопрос, ответом на который является именно он.

Требования:
- вопрос на русском, один, короткий (до 15 слов);
- НЕ цитируй фрагмент дословно, перефразируй своими словами;
- НЕ начинай с «Что говорится» и подобного;
- вопрос обязан быть специфичным: назови сервис, тикет или тему из фрагмента,
  чтобы по нему нашёлся именно этот фрагмент, а не любой похожий.

Фрагмент:
{text}

Вопрос:"""


def words(s):
    # Дефис ВНУТРИ токена: иначе из 'payment-gateway' получалось 'gatew', из
    # 'материализацией' — обрезок. Такие «редкие термины» метрику только портят.
    out = []
    for t in re.findall(r'[а-яёa-z0-9][а-яёa-z0-9-]{2,}', str(s).lower()):
        t = t.strip('-')
        if len(t) < 4 or t.isdigit():
            # Чистые числа ('0241', '2026') — нумерация, не смысл.
            continue
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', t):
            continue          # дата: редкая, но бесполезная для метрики
        out.append(t)
    return out


def corpus_freq(text, cap=4_000_000):
    """Частоты слов по корпусу (для rare_terms). Ограничение — чтобы не жевать гигабайты."""
    from collections import Counter
    c = Counter(words(text[:cap]))
    return c


def rare_terms(snippet, freq, k=5):
    """Самые редкие в корпусе слова фрагмента: именно их и надо найти."""
    seen = []
    for w in words(snippet):
        if w in STOP or w in seen:
            continue
        seen.append(w)
    seen.sort(key=lambda w: freq.get(w, 0))
    return seen[:k]


def leak(query, snippet, min_len=25):
    """Дословная цитата фрагмента в вопросе — метрика превратилась бы в поиск подстроки."""
    q = re.sub(r'\s+', ' ', str(query)).strip().lower()
    s = re.sub(r'\s+', ' ', str(snippet)).strip().lower()
    if not q or not s:
        return True
    for i in range(0, max(1, len(q) - min_len + 1), 8):
        if q[i:i + min_len] in s:
            return True
    return len(q) < 12


def ask_ollama(text, model, url, timeout=120):
    body = json.dumps({'model': model,
                       'prompt': PROMPT.format(text=text[:1200]),
                       'stream': False,
                       'options': {'temperature': 0.3, 'num_predict': 60}}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={'Content-Type': 'application/json'})
    r = json.loads(_OPENER.open(req, timeout=timeout).read().decode())
    out = str(r.get('response', '')).strip()
    out = out.split('\n')[0].strip().strip('"').strip()
    return re.sub(r'^\s*(вопрос|question)\s*[::]\s*', '', out, flags=re.I).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260912)
    # Имя модели — КАК В `ollama list`, с тегом. Ollama отдаёт 404 на /api/generate,
    # если тега нет: `qwen2.5:7b` не существует, `qwen2.5:7b-instruct` — существует.
    ap.add_argument('--gen-model',
                    default=os.environ.get('FRK_GEN_MODEL', 'qwen2.5:7b-instruct'))
    ap.add_argument('--gen-url', default=os.environ.get('FRK_GEN_URL',
                                                        'http://127.0.0.1:11434/api/generate'))
    ap.add_argument('--no-llm', action='store_true',
                    help='не спрашивать LLM: запрос = редкие термины фрагмента')
    args = ap.parse_args()

    text = open(args.corpus, encoding='utf-8', errors='replace').read()
    n_win = max(1, (len(text) - CHUNK) // STEP + 1)
    print(f'корпус {len(text):,} символов → {n_win:,} окон (шаг {STEP})')

    if n_win < args.n + 10:
        print(f'ВНИМАНИЕ: окон {n_win} меньше чем нужно ({args.n}). '
              f'Корпус слишком короткий — бенч будет бедным.')

    if not args.no_llm:
        try:
            tags = json.loads(_OPENER.open(args.gen_url.replace('/api/generate',
                                                                '/api/tags'),
                                           timeout=30).read().decode())
            have = [m['name'] for m in tags.get('models', [])]
            if args.gen_model not in have:
                print(f'модели «{args.gen_model}» нет в Ollama.')
                print('доступны:', ', '.join(have) if have else 'ни одной')
                print('укажите --gen-model <имя> либо запустите с --no-llm')
                return 2
        except Exception as e:
            print(f'не могу спросить Ollama ({type(e).__name__}: {e}). '
                  f'Запустите с --no-llm или поднимите Ollama.')
            return 2

    freq = corpus_freq(text)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(n_win), min(args.n, n_win)))

    out, dropped, failed = [], 0, 0
    for i, w in enumerate(idxs):
        a = w * STEP
        snippet = text[a:a + CHUNK].replace('\n', ' ').strip()
        if len(snippet) < 200:
            continue
        # Окно начинается посреди слова: первый и последний токены — хвосты
        # ('ерять', 'нить'). Такие «редкие термины» метрику только зашумляют,
        # поэтому частоты считаем по серединке с целыми словами.
        i0 = snippet.find(' ')
        i1 = snippet.rfind(' ')
        core = snippet[i0:i1] if 0 < i0 < i1 else snippet
        terms = rare_terms(core, freq)
        if args.no_llm:
            query = ' '.join(terms[:6]) if terms else snippet[:60]
        else:
            try:
                query = ask_ollama(snippet, args.gen_model, args.gen_url)
            except Exception as e:
                failed += 1
                print(f'  [{i}] LLM недоступен: {type(e).__name__}: {e}')
                if failed >= 5:
                    print('\nLLM не отвечает 5 раз подряд. Запустите с --no-llm '
                          'или поднимите Ollama.')
                    break
                continue
        # Редкие термины, попавшие в сам вопрос, из метрики убираем: иначе
        # rare_cov мерил бы «модель пересказала вопрос», а не «нашлась память».
        qn = str(query).lower()
        terms = [t for t in terms if t not in qn]
        if not args.no_llm and leak(query, snippet):
            dropped += 1
            continue
        out.append({'query': query, 'char_pos': int(a), 'snippet': snippet,
                    'rare_terms': terms})
        if (i + 1) % 25 == 0:
            print(f'  готово {len(out)}/{len(idxs)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\nсохранено {len(out)} вопросов → {args.out}')
    print(f'отброшено дословных цитат: {dropped}; ошибок LLM: {failed}')
    print('дальше: python _bench_fact.py  (с FRK_BENCH=<этот файл>)')


if __name__ == '__main__':
    main()
