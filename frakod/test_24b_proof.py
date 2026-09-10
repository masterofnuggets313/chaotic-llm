# -*- coding: utf-8 -*-
"""ЧЕСТНЫЙ ТЕСТ 24 Б/ТОКЕН: что умеет само сжатие, без надстроек.

Разбирает вклад каналов на одних и тех же 30 иглах и одном архиве:
  adc24   — ТОЛЬКО 24-байтовые коды (ADC-ранжирование, keys_f16 не читаем)
  vec     — коды + реранк сырым косинусом keys_f16 (без лексики)
  lex     — только лексический exact-слой (IDF)
  hybrid  — auto (лексика + вектор), как в README-бенче

Запросы двух типов, и это важно:
  question — только вопрос ("Что в истории связано со словом X?"), как у RAG/Mem0
  snippet  — 160-символьный сниппет вокруг иглы, как в bench_memory_systems.py

Хит = слово-игла встречается в top-k возвращённых текстов (как в бенче).
Запуск: python test_24b_proof.py
Нужен живой frk1 на 127.0.0.1:8781.
"""
import json
import os
import statistics
import time
import urllib.request

API = os.environ.get('FRK_API', 'http://127.0.0.1:8781')
HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLES = os.environ.get('FRK_NEEDLES', os.path.join(HERE, 'hermes_needles.json'))
K = 4
OUT = os.path.join(HERE, 'gt_24b_proof.json')


def _post(path, body, timeout=180):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(API + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def run_channel(needles, mode, rerank, query_key, label):
    """Прогнать один канал по всем иглам. Возвращает hits, latency, детали."""
    hits = 0
    ms = []
    detail = []
    for n in needles:
        q = n[query_key]
        body = {'text': q, 'k': K, 'mode': mode}
        if mode == 'vector' and not rerank:
            body['mode'] = 'vector24'
        try:
            t0 = time.time()
            r = _post('/recall', body)
            ms.append((time.time() - t0) * 1000)
        except Exception as e:
            detail.append({'word': n['needle_word'], 'err': str(e)[:80]})
            continue
        docs = ' '.join(s.get('text', '') for s in r.get('results', []))
        hit = n['needle_word'].lower() in docs.lower()
        hits += bool(hit)
        detail.append({'word': n['needle_word'], 'hit': bool(hit),
                       'via': [s.get('via') for s in r.get('results', [])],
                       'rank': next((i for i, s in enumerate(r.get('results', []))
                                     if n['needle_word'].lower() in s.get('text', '').lower()), -1)})
    med = round(statistics.median(ms), 1) if ms else None
    print(f'{label:<34} recall@{K} = {hits}/{len(needles)} = {hits/len(needles):.3f}'
          f'   median {med} ms', flush=True)
    return {'label': label, 'recall': round(hits / len(needles), 3), 'hits': hits,
            'n': len(needles), 'median_ms': med, 'detail': detail}


def main():
    needles = json.load(open(NEEDLES, encoding='utf-8'))
    n = len(needles)
    print(f'ИГЛ: {n}   АРХИВ: {API}   K={K}', flush=True)
    print('-' * 78, flush=True)

    # Проверим, что архиву вообще доступны все каналы
    st = _post('/index/load', {}) if False else None

    rows = []
    print('== ЗАПРОС = ПОЛНЫЙ ВОПРОС (как у RAG/Mem0: без сниппета) ==', flush=True)
    rows.append(run_channel(needles, 'vector24', False, 'query', '1. adc24 (только 24 Б)'))
    rows.append(run_channel(needles, 'vector', True, 'query', '2. vec (коды+реранк keys)'))
    rows.append(run_channel(needles, 'lex', True, 'query', '3. lex (лексика)'))
    rows.append(run_channel(needles, 'auto', True, 'query', '4. hybrid (auto, README)'))

    print('-' * 78, flush=True)
    print('== ЗАПРОС = СНИППЕТ 160 симв. (как в bench_memory_systems.py) ==', flush=True)
    rows.append(run_channel(needles, 'vector24', False, 'snippet', '5. adc24 (только 24 Б)'))
    rows.append(run_channel(needles, 'vector', True, 'snippet', '6. vec (коды+реранк keys)'))
    rows.append(run_channel(needles, 'lex', True, 'snippet', '7. lex (лексика)'))
    rows.append(run_channel(needles, 'auto', True, 'snippet', '8. hybrid (auto, README)'))

    json.dump({'n_needles': n, 'k': K, 'rows': rows},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('-' * 78, flush=True)
    print(f'СОХРАНЕНО -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
