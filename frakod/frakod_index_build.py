# -*- coding: utf-8 -*-
"""Строитель архива Fracode: корпус -> индекс.
Тот же движок, что stress_10m_fix (S=12, K=256, calib 1M, iters 12, seed 0).
Артефакты в frakod_index/:
  codes.npy    (N,2,12 u8)   = 24 Б/токен — само сжатие (ADC-скрининг)
  keys_f16.npy (N,192 f16)   = 384 Б/токен — реранк кандидатов сырым косинусом
  ids.npy      (N i32)       = 4 Б/токен — лекс-слой (точный поиск слов)
  cbook_l{0,1}.npy           = кодобуки (фикс. ~0.4 МБ)
ЧЕСТНО О ХРАНИЛИЩЕ: «24 Б/токен» относится ТОЛЬКО к codes.npy. Постоянный
след на диске включает keys_f16 и ids; полный итог пишется в meta.json
(b_per_tok_total) и отдаётся /index/accounting.
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
# S (число подвекторов) зависит от размерности: d обязано делиться на S.
# d=192 -> S=12 (исторический формат, совместим со старыми архивами);
# d=256 (v7) -> S=16, иначе 256/12 не целое. API читает S из meta.json.
_S_ENV = os.environ.get('FRK_SUBVECS')
if _S_ENV:
    S = int(_S_ENV)
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

_ckpt = os.environ.get('FRK_CKPT') or os.path.join(REPO, 'results', 'ckpts', 'sts_prog_seed0.pt')
_sd = torch.load(_ckpt, map_location='cpu', weights_only=False)
# v7-чекпойнт хранит веса под 'model', seed0 — плоский state_dict
_sd = _sd.get('model', _sd) if isinstance(_sd, dict) else _sd
_w = None
for _k in ('embed.weight', 'model.embed.weight'):
    if isinstance(_sd, dict) and _k in _sd:
        _w = _sd[_k]; break
if _w is None:
    raise KeyError(f'в {_ckpt} нет embed.weight — не могу строить ключи')
EMB = _w.detach().float().cpu()                 # (vocab, d) — таблица как есть
D = int(EMB.shape[1])
if int(EMB.shape[0]) != int(tok.get_vocab_size()):
    raise ValueError(f'несовместимость: ckpt vocab={EMB.shape[0]}, '
                     f'tokenizer vocab={tok.get_vocab_size()}. '
                     f'Возьмите чекпойнт, обученный на этом словаре.')
print(f'emb-таблица из {os.path.basename(_ckpt)}: {tuple(EMB.shape)}', flush=True)
# приводим S к размерности: d % S == 0
if D % S != 0:
    for _s in (16, 8, 12, 32):
        if D % _s == 0:
            print(f'S={S} не делит d={D} -> берём S={_s}', flush=True)
            S = _s
            break
    else:
        raise ValueError(f'не найден делитель d={D} для subvecs')
EMB_gpu = EMB.to(dev) if dev == 'cuda' else EMB
t0 = time.time(); CH = 1_000_000; base = torch.empty(N, D, device=dev)
# контекстные ключи: среднее emb в окне радиуса RAD (монотокен-эмбеддинг без
# контекста не даёт семантики - GT-тест это доказал)
RAD = 4
for i in range(0, N, CH):
    e = min(i+CH, N)
    lo = max(0, i-RAD); hi = min(N, e+RAD)
    emb = EMB_gpu[toks[lo:hi]].detach()                # (len, D)
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


def _sz(name):
    p = os.path.join(OUT, name)
    return os.path.getsize(p) if os.path.exists(p) else 0


_b = {'codes.npy': _sz('codes.npy'), 'keys_f16.npy': _sz('keys_f16.npy'),
      'ids.npy': _sz('ids.npy'), 'cbook': _sz('cbook_l0.npy') + _sz('cbook_l1.npy')}
_total = sum(_b.values())
json.dump({'N': N, 'S': S, 'K': K, 'D': D,
           # честная бухгалтерия: 24 Б/токен — это ТОЛЬКО codes
           'b_per_tok_codes': round(_b['codes.npy'] / N, 3),
           'b_per_tok_keys': round(_b['keys_f16.npy'] / N, 3),
           'b_per_tok_ids': round(_b['ids.npy'] / N, 3),
           'b_per_tok_total': round(_total / N, 3),
           'bytes': _b, 'total_mb': round(_total / 1024**2, 1),
           'codes_mb': round(_b['codes.npy'] / 1024**2, 1),
           'keys_mb': round(_b['keys_f16.npy'] / 1024**2, 1),
           'b_per_tok': round(_b['codes.npy'] / N, 3),   # legacy-ключ = коды
           'fr_mb': round(_b['codes.npy'] / 1024**2),    # legacy-ключ = коды
           'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'corpus': os.path.basename(CORPUS),
           # ЧЕСТНО: чем именно токенизирован корпус — от этого зависит, каким
           # токенизатором обязан спрашивать API (иначе ids запроса не совпадут).
           'tokenizer': (os.path.basename(_tokfile) if _tokfile else 'make_bpe(corpus_public,512)'),
           'tokenizer_kind': ('external' if _tokfile else 'bpe512'),
           'tokenizer_path': (os.path.abspath(_tokfile) if _tokfile else None),
           'vocab': int(tok.get_vocab_size()),
           'key_context_rad': int(RAD),
           # ВАЖНО: пишем АБСОЛЮТНЫЙ путь. Раньше писался только basename, и если
           # чекпойнт лежал не рядом с API (напр. phase01/exp_vq/ckpt_v8_voc8k.pt),
           # API молча откатывался на sts_prog_seed0.pt — ключи архива и запрос
           # оказывались из РАЗНЫХ таблиц эмбеддингов (идентично корневой причине №1).
           'embed_ckpt': os.path.abspath(_ckpt),
           'embed_ckpt_name': os.path.basename(_ckpt),
           'codec': 'embed-only keys (без pos) + FR 2 уровня 12x8',
           'storage_note': ('24 Б/токен = codes.npy. keys_f16 используется для '
                            'реранка (можно держать memmap с диска), ids — лекс-слой.')},
          open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'АРХИВ ГОТОВ: {N:,} токенов | коды {_b["codes.npy"]/1024**2:.0f} МБ (24 Б/ток) '
      f'| keys {_b["keys_f16.npy"]/1024**2:.0f} МБ | ИТОГО {_total/1024**2:.0f} МБ '
      f'({_total/N:.0f} Б/ток) -> {OUT}', flush=True)
