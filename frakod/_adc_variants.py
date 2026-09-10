"""Какой ADC-запрос реально работает: нормированный или нет, с mu или без.

Считаем ADC по полному архиву код-корпуса для 16 известных позиций и смотрим
top1/delta. Плюс — сравнение с точным косинусом по keys (эталон).
"""
import os
import numpy as np

IDX = os.environ.get('IDX', 'G:/Migration/chaotic-llm/phase01/exp_vq/frakod_index_code10m')
CKPT = 'G:/Migration/chaotic-llm/results/ckpts/sts_prog_seed0.pt'
D, RAD, S = 192, 4, 12
sub = D // S

import torch
E = torch.load(CKPT, map_location='cpu', weights_only=True)['embed.weight'].float().numpy()

ids = np.load(IDX + '/ids.npy', mmap_mode='r')
codes = np.load(IDX + '/codes.npy', mmap_mode='r')
keys = np.load(IDX + '/keys_f16.npy', mmap_mode='r')
cbook = np.stack([np.load(IDX + '/cbook_l0.npy'), np.load(IDX + '/cbook_l1.npy')])  # (2,S,K,sub)
N = len(ids)
print(f'N={N:,}  cbook={cbook.shape}', flush=True)


def ctx_query(p0):
    pad = list(np.asarray(ids[p0 - RAD:p0 + RAD + 1]).astype(np.int64))
    e = E[np.clip(pad, 0, 511)]
    return e.mean(0).astype(np.float32)


def adc_scores(q):
    out = np.empty(N, dtype=np.float32)
    CH = 2_000_000
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        acc = np.zeros(i1 - i0, dtype=np.float32)
        c = np.asarray(codes[i0:i1])
        for l in range(2):
            for s in range(S):
                tbl = cbook[l, s] @ q[s * sub:(s + 1) * sub]
                acc += tbl[c[:, l, s]]
        out[i0:i1] = acc
    return out


rng = np.random.default_rng(7)
pos = [int(x) for x in rng.integers(100_000, N - 100_000, size=16)]
variants = {'raw': lambda q: q,
            'unit': lambda q: q / (np.linalg.norm(q) + 1e-9),
            'centered': lambda q: q - keys[::7919].astype(np.float32).mean(0),
            'centered_unit': lambda q: (q - keys[::7919].astype(np.float32).mean(0)) /
                                       (np.linalg.norm(q - keys[::7919].astype(np.float32).mean(0)) + 1e-9)}
DELTA = {k: [] for k in variants}
for p0 in pos:
    q0 = ctx_query(p0)
    for tag, fn in variants.items():
        d = adc_scores(fn(q0))
        # top-1 из кодов
        j = int(np.argmax(d))
        DELTA[tag].append(abs(j - p0))

for tag, dl in DELTA.items():
    dl = sorted(dl)
    print(f'{tag:<14} top1 exact={sum(1 for x in dl if x==0)}/16  '
          f'within4={sum(1 for x in dl if x<=4)}/16  deltas={dl}')
