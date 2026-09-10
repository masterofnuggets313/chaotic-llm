# -*- coding: utf-8 -*-
"""Строитель архива Fracode: корпус -> 10M-токенов индекс.
Тот же движок, что stress_10m_fix (S=12, K=256, calib 1M, iters 12, seed 0).
Артефакты в frakod_index/: codes.npy (N,2,12 u8), keys_f16.npy (N,192 f16 - для
демо-возврата, НЕ часть сжатия), ids.npy (N i32 текст), cbook_l{0,1}.npy, meta.json.
"""
import sys, os, time, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, '..'); REPO = os.path.join(PHASE, '..')
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
import final_benchmark as fb
from models_pc import build_pc_model
from night_task5_fracode_forward import StreamFracode

NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
S = 12; K = 256; D = 192
CORPUS = os.environ.get('FRK_CORPUS') or os.path.join(PHASE, 'corpus5m_train.txt')
OUT = os.environ.get('FRK_OUTD') or os.path.join(HERE, 'frakod_index'); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# v8-ready: чем кодируем и какими emb строим ключи (по умолчанию — v7-историческое)
_tokfile = os.environ.get('FRK_TOK')       # путь к saved tokenizer json (v8)
if _tokfile:
    from tokenizers import Tokenizer as _Tkk
    tok = _Tkk.from_file(_tokfile)
    head = ''
else:
    head = fb.load_chars(os.path.join(PHASE, 'corpus_public.txt'), 990_000)
    tok = fb.make_bpe(head, vocab=int(os.environ.get('FRK_VOCAB', 512)))
big = fb.load_chars(CORPUS if os.path.isabs(CORPUS) else os.path.join(HERE, CORPUS), None)
t0 = time.time(); ids_all = np.array(tok.encode(big).ids, dtype=np.int64)
print(f'токенизировано {len(ids_all):,} за {time.time()-t0:.0f}s', flush=True)
N = min(NTOK, len(ids_all) - 40)
toks = torch.tensor(ids_all[:N], dtype=torch.long, device=dev)

model = build_pc_model('pc', vocab=tok.get_vocab_size(), d=D, layers=8, k_init=1.2,
    sync_steps=8, driver_mode='sts_prog', alpha=0.3, temp=0.3).to(dev).eval()
model.load_state_dict(torch.load(os.environ.get('FRK_CKPT') or
    os.path.join(REPO, 'results', 'ckpts', 'sts_prog_seed0.pt'), map_location='cpu'))
for p in model.parameters(): p.requires_grad_(False)

t0 = time.time(); CH = 1_000_000; base = torch.empty(N, D, device=dev)
# контекстные ключи: среднее emb в окне радиуса RAD (монотокен-эмбеддинг без
# контекста не даёт семантики - GT-тест это доказал)
RAD = 4
for i in range(0, N, CH):
    e = min(i+CH, N)
    lo = max(0, i-RAD); hi = min(N, e+RAD)
    emb = model.embed(toks[lo:hi]).detach()            # (len, D)
    emb_t = emb.t().unsqueeze(0)                       # (1, D, len)
    pad = torch.nn.functional.pad(emb_t, (RAD, RAD), mode='replicate')
    ctx = torch.nn.functional.avg_pool1d(pad, 2*RAD+1, stride=1).squeeze(0).t()  # (len, D)
    base[i:e] = ctx[i-lo:e-lo]
    del emb, pad, ctx
print(f'keys (context-avg r={RAD}) за {time.time()-t0:.0f}s', flush=True)

fm = StreamFracode(D, levels=2, subvecs=S, K=K, device=dev)
calib = base[:1_000_000].clone()
t0 = time.time(); fm.fit(calib, iters=12, seed=0); del calib
print(f'кодобук за {time.time()-t0:.0f}s', flush=True)
codes = torch.empty(N, 2, S, dtype=torch.uint8, device=dev)
for i in range(0, N, CH):
    e = min(i+CH, N)
    codes[i:e] = fm.encode_rows(base[i:e]).to(torch.uint8)
np.save(os.path.join(OUT, 'codes.npy'), codes.cpu().numpy())
np.save(os.path.join(OUT, 'keys_f16.npy'), base.half().cpu().numpy())
np.save(os.path.join(OUT, 'ids.npy'), ids_all[:N].astype(np.int32))
for l in range(2):
    np.save(os.path.join(OUT, f'cbook_l{l}.npy'),
            torch.stack(fm.cbooks[l]).cpu().numpy())   # (S, K, sub)
json.dump({'N': N, 'S': S, 'K': K, 'D': D, 'b_per_tok': 24,
           'fr_mb': round(codes.numel()/1024**2),
           'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'corpus': os.path.basename(CORPUS),
           'codec': 'embed-only keys (без pos) + FR 2 уровня 12x8'},
          open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'АРХИВ ГОТОВ: {N:,} токенов, коды {codes.numel()/1024**2:.0f} МБ -> {OUT}', flush=True)
