"""ПРОВЕРКА СОГЛАСОВАННОСТИ ЗАПРОС-КЛЮЧ (главная гипотеза провала вектора).

Архив: keys_f16[i] = avg_pool(emb(toks)) с окном RAD=4 (context-avg, см. writer).
API:   q = E[ids_q].mean(0)  -- БЕЗ контекстного усреднения!

Если гипотеза верна, то корректный контекстный запрос найдёт СВОЮ позицию
с cos~1, а монотокенный средний -- нет. Проверяем на код-корпусе (эмбеддер
обучен) точно, без API.
"""
import os, sys, json
import numpy as np, torch

REPO = 'G:/Migration/chaotic-llm'
IDX = os.environ.get('IDX', REPO + '/phase01/exp_vq/frakod_index_code10m')
CKPT = REPO + '/results/ckpts/sts_prog_seed0.pt'
D, RAD = 192, 4

ids = np.load(IDX + '/ids.npy', mmap_mode='r')
keys = np.load(IDX + '/keys_f16.npy', mmap_mode='r')
N = len(ids)
print(f'N={N:,}', flush=True)

sd = torch.load(CKPT, map_location='cpu', weights_only=True)
E = sd['embed.weight'].float().numpy()          # (512,192)
print('E shape', E.shape, flush=True)

# контекстные ключи для 16 произвольных позиций считаем локально
rng = np.random.default_rng(7)
pos = rng.integers(100_000, N - 100_000, size=16)

res_cos, res_mono = [], []
for p0 in pos:
    p0 = int(p0)
    lo, hi = p0 - RAD, p0 + RAD + 1
    idswin = np.asarray(ids[lo:hi]).astype(np.int64)
    emb = E[np.clip(idswin, 0, 511)]            # (9,192)
    q_ctx = emb.mean(0)
    q_ctx /= (np.linalg.norm(q_ctx) + 1e-9)

    ids_q = np.asarray(ids[p0:p0 + 10]).astype(np.int64)
    q_mono = E[np.clip(ids_q, 0, 511)].mean(0)
    q_mono /= (np.linalg.norm(q_mono) + 1e-9)

    # точный косинус по 200k окрестности (для скорости) + глобально по срезу
    lo2, hi2 = max(0, p0 - 100_000), min(N, p0 + 100_000)
    K = np.asarray(keys[lo2:hi2]).astype(np.float32)
    K /= (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)

    for tag, q in (('ctx', q_ctx), ('mono', q_mono)):
        d = K @ q
        j = int(np.argmax(d))
        gpos = lo2 + j
        delta = abs(gpos - p0)
        (res_cos if tag == 'ctx' else res_mono).append((float(d[j]), delta))
        print(f'  p0={p0:>9} [{tag:4s}] best_cos={d[j]:.3f} best_pos={gpos:>9} delta={delta:>6}', flush=True)

for tag, r in (('ctx', res_cos), ('mono', res_mono)):
    ok = sum(1 for _, d in r if d <= 4)
    print(f'== {tag}: top1 within +-4: {ok}/{len(r)}  mean_cos={np.mean([c for c,_ in r]):.3f}')
