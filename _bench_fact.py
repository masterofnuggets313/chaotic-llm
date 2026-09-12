"""Бенчмарк «нашёлся ли ФАКТ» — продуктовая метрика, а не «нашлось ли окно».

recall@k (окно) отвечает на вопрос «вернул ли поиск нужный chunk». Продукту
нужно другое: «содержится ли в возвращённом тексте сам факт, ради которого
спрашивали». Здесь три ОБЪЕКТИВНЫЕ метрики, без LLM-судьи:

  1. recall@k     — позиционная: попал ли хотя бы один золотой ЧАНК в top-k.
                    Золотые чанки: все окна, перекрывающие
                    [char_pos, char_pos + max(300, len(snippet))) при шаге 640.
  2. snippet_hit@k — нормализованный сниппет (или его 40-символьное окно)
                    встречается в объединённом тексте top-k.
  3. rare_cov@k   — доля редких терминов из rare_terms, найденных в тексте.

Контроль обязателен: запрос-бессмыслица должен давать ~0, иначе метрика
завышена (корпус узкий и «находит» что угодно).
"""
import json, os, re, time, random, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Бенч по умолчанию заменён: hermes_sem_bench_big.json построен на личной переписке
# и непубликуем. Свой бенч для ЛЮБОГО корпуса строит make_bench.py; путь сюда
# передаётся через FRK_BENCH.
BENCH = os.getenv('FRK_BENCH') or os.path.join(HERE, 'demo_bench.json')
API = os.getenv('FRK_API', 'http://127.0.0.1:8000/recall')
SEED = 20260912
CHUNK, OVERLAP = 800, 160
STEP = CHUNK - OVERLAP          # 640 — шаг, НЕ stride от длины!


# ОБХОД ПРОКСИ (обязательно): в этой среде в окружении стоит http_proxy на
# sandbox-прокси, и urllib гонит 127.0.0.1 через него → 502 Bad Gateway, хотя
# сервер жив и curl отвечает. Локальный адрес обязан идти напрямую.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(qtext, k=4, rerank=True):
    body = json.dumps({'text': qtext, 'k': k, 'mode': 'vector',
                       'rerank': rerank}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    raw = _OPENER.open(req, timeout=300).read()
    return json.loads(raw.decode()), (time.perf_counter() - t0) * 1000.0


def norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip().lower()


def text_hit(needle, hay, win=40):
    """Есть ли needle в hay: прямое вхождение, либо любое окно из win символов."""
    n, h = norm(needle), norm(hay)
    if not n:
        return False
    if n in h:
        return True
    if len(n) <= win:
        return False
    for i in range(0, len(n) - win + 1, max(1, win // 2)):
        if n[i:i + win] in h:
            return True
    return False


def gold_windows(char_pos, snippet):
    a = int(char_pos)
    b = a + max(300, len(str(snippet)))
    g = set()
    i = max(0, a // STEP - 2)
    while i * STEP < b:
        if i * STEP < b and i * STEP + CHUNK > a:
            g.add(i)
        i += 1
    return g


def boot_ci(hits, n_boot=1000, seed=SEED):
    hits = np.asarray(hits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(hits)
    if n == 0:
        return [0.0, 0.0]
    bs = np.array([hits[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run(bench, k=4, tag='осн.'):
    rec, snip, rare, lat, oks, per = [], [], [], [], [], []
    full = []
    for it in bench:
        q = it['query']
        gold = gold_windows(it['char_pos'], it['snippet'])
        r, t = post(q, k=k)
        lat.append(t)
        oks.append(1.0 if r.get('ok') else 0.0)
        res = r.get('results') or []
        pos = {int(x['position']) for x in res}
        txt = ' '.join(str(x.get('text', '')) for x in res)
        rec.append(1.0 if (pos & gold) else 0.0)
        snip.append(1.0 if text_hit(it['snippet'], txt) else 0.0)
        # СТРОГАЯ версия: весь сниппет целиком, без скидок на 40-символьные окна.
        # Зачем: текст_hit(win=40) на ОДНООБРАЗНОМ корпусе даёт 1.0 на любом
        # ответе — 40 символов шаблонной фразы находятся где угодно. Метрика
        # честна на живом тексте, но ломается на синтетике; строгая версия
        # показывает это сразу, а не выдаёт красивую единицу.
        full.append(1.0 if norm(it['snippet']) in norm(txt) else 0.0)
        terms = [t2 for t2 in (it.get('rare_terms') or []) if str(t2).strip()]
        if terms:
            found = sum(1.0 for t2 in terms if norm(t2) in norm(txt))
            rare.append(found / len(terms))
        else:
            rare.append(0.0)
        # Посерийный дамп: нужен для ПАРНОГО сравнения конфигураций (Мак-Немар).
        # Независимые CI здесь слабее: у прогонов один и тот же поиск (те же
        # позиции), различается только выдача текста — значит пары зависимы.
        per.append({'i': len(per), 'pos': sorted(int(x) for x in pos),
                    'gold': sorted(int(g) for g in gold),
                    'rec': rec[-1], 'snip': snip[-1], 'rare': rare[-1]})
    out = {'per_item': per}
    for nm, v in (('recall', rec), ('snippet_hit', snip),
                  ('snippet_full', full), ('rare_cov', rare)):
        out[nm] = {'mean': round(float(np.mean(v)), 4),
                   'ci': [round(x, 4) for x in boot_ci(v)]}
    out['latency_ms'] = {'p50': round(float(np.percentile(lat, 50)), 2),
                         'p95': round(float(np.percentile(lat, 95)), 2)}
    out['ok_rate'] = round(float(np.mean(oks)), 4)
    print(f'[{tag} k={k}] recall={out["recall"]["mean"]:.4f} '
          f'{out["recall"]["ci"]} | snippet_hit={out["snippet_hit"]["mean"]:.4f} '
          f'{out["snippet_hit"]["ci"]} | snippet_full={out["snippet_full"]["mean"]:.4f} '
          f'| rare_cov={out["rare_cov"]["mean"]:.4f} '
          f'| ok={out["ok_rate"]:.3f} | p50={out["latency_ms"]["p50"]} мс')
    return out


def main():
    bench = json.load(open(BENCH, encoding='utf-8'))
    print(f'вопросов: {len(bench)}; STEP={STEP}')

    res = {'bench': os.path.basename(BENCH), 'seed': SEED, 'step': STEP,
           'metric_note': ('recall — попал ли золотой чанк; snippet_hit — есть ли '
                           'сам сниппет в возвращённом тексте; rare_cov — доля '
                           'редких терминов. Объективно, без LLM-судьи.')}

    # прогрев
    for it in bench[:3]:
        post(it['query'], k=4)
    print('прогрев OK')

    for k in (4, 8):
        res[f'k{k}'] = run(bench, k=k, tag='осн.')

    # КОНТРОЛЬ: запрос-бессмыслица. Метрика обязана упасть, иначе она завышена.
    rng = random.Random(SEED)
    junk = ['йцукен фывапр олджщ', 'random nonsense query 12345',
            'абракадабра ерунда тест', 'щоразуміння перевірка'] * 5
    ctrl = [{'query': j, 'char_pos': 0, 'snippet': 'ZZQQXX несуществующий сниппет',
             'rare_terms': ['zzqqxx']} for j in junk[:20]]
    res['control_nonsense'] = run(ctrl, k=4, tag='КОНТР')
    print('  (контроль: snippet_hit обязан быть ~0, rare_cov ~0)')

    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    # Имя файла задаётся снаружи: прогоны при разном FRK_WIN_CTX/FRK_TXT_CAP
    # обязаны лежать рядом, иначе следующий затирает предыдущий.
    outp = os.path.join(HERE, 'runs', os.getenv('FRK_BENCH_OUT', 'bench_fact.json'))
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', outp)


if __name__ == '__main__':
    main()
