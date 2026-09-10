"""Оффлайн ADC на боевом архиве Hermes. Истина = позиция в ids архива.

Важно: архив построен BPE-512 (см. meta/max-id). Ищем иглы напрямую
в ids.npy (не через tok_v31!). Сравниваем mono-запрос и ctx-запрос.
"""
import os, sys, json
import numpy as np, torch

REPO = 'G:/Migration/chaotic-llm'
IDX = REPO + '/phase01/exp_vq/frakod_index'
CKPT = REPO + '/frakod/sts_prog_seed0.pt'
RAD, S = 4, 12

E = torch.load(CKPT, map_location='cpu', weights_only=True)['embed.weight'].float().numpy()
D = E.shape[1]; sub = D // S
ids = np.load(IDX + '/ids.npy', mmap_mode='r')
codes = np.load(IDX + '/codes.npy', mmap_mode='r')
cbook = np.stack([np.load(IDX + '/cbook_l0.npy'), np.load(IDX + '/cbook_l1.npy')])
N = len(ids)
print(f'E={E.shape} N={N:,}', flush=True)

# BPE-512, как в билдере
sys.path.insert(0, REPO + '/phase01')
import final_benchmark as fb
head = fb.load_chars(REPO + '/phase01/corpus_public.txt', 990_000)
tok = fb.make_bpe(head, vocab=512)

needles = json.load(open(REPO + '/phase01/exp_vq/hermes_needles.json', encoding='utf-8'))

# позиция иглы = первый индекс, с которого начинается её BPE-кодировка в ids
def find_pos(w):
    """Первое вхождение полной BPE-последовательности слова в ids архива.

    Посимвольный BPE-512 делает первые токены почти всегда одинаковыми (коды
    букв), поэтому ищем ВСЮ последовательность, а не только первый токен."""
    wt = np.array(tok.encode(w).ids, dtype=np.int64)
    if not len(wt):
        return -1
    ids_np = np.asarray(ids[:N])
    L = len(wt)
    cand = np.where(ids_np[:N - L + 1] == wt[0])[0]
    for i in cand:
        if np.array_equal(ids_np[i:i + L], wt):
            return int(i) + L // 2
    return -1


positions = {}
for x in needles:
    positions[x['needle_word']] = find_pos(x['needle_word'].lower())
print('found positions:', sum(1 for v in positions.values() if v >= 0), '/', len(needles), flush=True)


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


for tag in ('mono', 'ctx'):
    h1 = h4 = h8 = n_eval = 0
    for x in needles:
        p0 = positions[x['needle_word']]
        if p0 < 0:
            continue
        n_eval += 1
        if tag == 'ctx':
            pad = np.asarray(ids[max(0, p0 - RAD):p0 + RAD + 1]).astype(np.int64)
            q = E[np.clip(pad, 0, 511)].mean(0).astype(np.float32)
        else:
            ids_q = np.asarray(ids[p0 - 5:p0 + 5]).astype(np.int64)
            q = E[np.clip(ids_q, 0, 511)].mean(0).astype(np.float32)
        d = adc_scores(q)
        top = np.argsort(-d)
        h1 += int(abs(int(top[0]) - p0) <= 4)
        h4 += int(any(abs(int(t) - p0) <= 4 for t in top[:4]))
        h8 += int(any(abs(int(t) - p0) <= 4 for t in top[:8]))
    print(f'{tag:5s}: n={n_eval}  top1={h1}  top4={h4}  top8={h8}', flush=True)
