# -*- coding: utf-8 -*-
"""GT: 30 игл из реальной истории Hermes против архива Fracod.
Хит = редкое слово иглы вернулось в top-4 сниппетах /recall."""
import json, urllib.request, time, statistics

url = 'http://127.0.0.1:8781/recall'
needles = json.load(open('hermes_needles.json', encoding='utf-8'))
hits = 0; ms = []; results = []
for n in needles:
    body = json.dumps({'text': n['snippet'][:300], 'k': 4}).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    except Exception as e:
        results.append({'word': n['needle_word'], 'err': str(e)[:60]}); continue
    ms.append((time.time()-t0)*1000)
    word = n['needle_word'].lower()
    hit = any(word in s['text'].lower() for s in r.get('results', []))
    hits += hit
    results.append({'word': word, 'hit': hit,
                    'pos': r['results'][0]['position'] if r.get('results') else -1,
                    'score': r['results'][0]['score'] if r.get('results') else 0})
    print(f"{'+' if hit else '-'} {word:<14} pos={results[-1]['pos']:>9}", flush=True)
print(f'RESULT: {hits}/{len(needles)} hit; median {statistics.median(ms):.0f} ms; '
      f'p95 {sorted(ms)[int(len(ms)*0.95)]:.0f} ms' if ms else 'NO DATA')
json.dump({'hits': hits, 'n': len(needles), 'median_ms': statistics.median(ms) if ms else -1,
           'results': results}, open('gt_hermes_results.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
