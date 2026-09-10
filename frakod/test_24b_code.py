"""Честный тест 24-байтового канала НА КОД-КОРПУСЕ (там эмбеддер обучен).

Идея: берём 10-токенный фрагмент из середины архива, спрашиваем по нему
ADС-канал (только коды) и вектор+реранк (коды+keys). Истинная позиция известна
точно. Если и здесь 24 Б не работают — канал мёртв архитектурно, а не из-за
кириллицы. ВАЖНО: смотрим также, есть ли попадание в ±окно и в top-k.
"""
import json, os, sys, time, urllib.request
import numpy as np

IDX = os.environ.get('FRK_IDXD', 'G:/Migration/chaotic-llm/phase01/exp_vq/frakod_index_code10m')
API = os.environ.get('FRK_API', 'http://127.0.0.1:8781')


def post(path, body, timeout=600):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ids = np.load(os.path.join(IDX, 'ids.npy'), mmap_mode='r')
    N = len(ids)
    rng = np.random.default_rng(7)
    print(f'IDX={IDX}  N={N:,}', flush=True)
    for mode, rerank, tag in [('vector24', False, 'ADC по 24 Б'),
                              ('vector', True, 'коды + реранк keys')]:
        hits_k = [0, 0, 0, 0]   # top1, top4, top8, top16
        deltas = []
        for qi in range(16):
            p0 = int(rng.integers(100_000, N - 100_000))
            seq = [int(t) for t in ids[p0:p0 + 10].tolist()]
            try:
                r = post('/recall', {'tokens': seq, 'k': 16, 'mode': mode, 'rerank': rerank})
            except Exception as e:
                print('  err', str(e)[:80]); continue
            res = r.get('results', [])
            best = min(res, key=lambda h: abs(h['position'] - p0)) if res else None
            d = abs(best['position'] - p0) if best else -1
            deltas.append(d)
            for i, thr in enumerate((0, 3, 7, 15)):
                if 0 <= d <= thr:
                    hits_k[i] += 1
        n = len(deltas)
        print(f'{tag:<22} top1={hits_k[0]}/{n}  top4(+3)={hits_k[1]}/{n}  '
              f'top8(+7)={hits_k[2]}/{n}  top16(+15)={hits_k[3]}/{n}', flush=True)
        print(f'{"":22} deltas={sorted(d for d in deltas if d>=0)[:16]}', flush=True)


if __name__ == '__main__':
    main()
