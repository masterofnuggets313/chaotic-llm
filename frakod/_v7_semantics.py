"""ГЛАВНАЯ ПРОВЕРКА: даёт ли v7-эмбеддер семантику на русском?

Раньше (seed0, vocab=512) cos между РАЗНЫМИ русскими текстами был 0.81..0.96 —
почти коллинеарно, семантики нет. Здесь то же самое, но с таблицей v7 (8192).
Плюс контекстный запрос (RAD=4) против монотокенного.

Если v7 различает тексты — пересборка архива имеет смысл.
Если нет — пересборка не спасёт, и это надо сказать прямо.
"""
import os, sys, json
import numpy as np, torch

REPO = 'G:/Migration/chaotic-llm'
CKPT = REPO + '/frakod/ckpt_v7_night_50k.pt'
TOK = REPO + '/frakod/tok_v31.json'
sys.path.insert(0, REPO + '/frakod')
from tokenizers import Tokenizer

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
E = ck['model']['embed.weight'].detach().float().numpy()
tk = Tokenizer.from_file(TOK)
print(f'E={E.shape} vocab={tk.get_vocab_size()}', flush=True)

IDS = [
    'Сервер перезапущен, проверил логи nginx',
    'Фотосинтез происходит в хлоропластах растений',
    'Купил билеты на поезд до Казани',
    'Нужно починить баг в авторизации пользователей',
    'Сегодня на улице холодно и идёт снег',
]


def mono(text):
    ids = np.array(tk.encode(text).ids, dtype=np.int64)
    return E[ids].mean(0)


def ctx(text, rad=4):
    ids = np.array(tk.encode(text).ids, dtype=np.int64)
    pad = np.concatenate([np.repeat(ids[:1], rad), ids, np.repeat(ids[-1:], rad)])
    cum = np.cumsum(np.vstack([np.zeros((1, E.shape[1]), np.float32),
                              E[pad].astype(np.float32)]), 0)
    k = 2 * rad + 1
    return ((cum[k:] - cum[:-k]) / k).mean(0)


for tag, fn in (('mono', mono), ('ctx', ctx)):
    V = np.stack([fn(t) for t in IDS])
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = Vn @ Vn.T
    off = C[~np.eye(len(IDS), dtype=bool)]
    print(f'{tag}: cos между РАЗНЫМИ текстами: '
          f'min={off.min():.3f} max={off.max():.3f} mean={off.mean():.3f}', flush=True)

# для контраста: то же на seed0 (vocab=512), как было
sd = torch.load(REPO + '/frakod/sts_prog_seed0.pt', map_location='cpu', weights_only=True)
E0 = sd['embed.weight'].float().numpy()
import final_benchmark as fb
head = fb.load_chars(REPO + '/phase01/corpus_public.txt', 990_000)
tk0 = fb.make_bpe(head, vocab=512)
V = np.stack([E0[np.array(tk0.encode(t).ids, dtype=np.int64)].mean(0) for t in IDS])
Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
C = Vn @ Vn.T
off = C[~np.eye(len(IDS), dtype=bool)]
print(f'seed0(512): cos между РАЗНЫМИ текстами: '
      f'min={off.min():.3f} max={off.max():.3f} mean={off.mean():.3f}')
