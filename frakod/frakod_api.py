# -*- coding: utf-8 -*-
# FRAKOD-API (frk1): адресная память = текст + граф рёбер + байт-учёт.
# Граница честности: НЕ нейро-порт (гейт 1 закрыт, Sec.4.36). Работает механика
# R4-X (Sec.4.41): адрес + ребро -> top1 1.00, без ребра 0.01, перест. 0.01.
# Мягкий адресный abstention ЗАКРЫТ (Sec.4.42-4.45): сервер честно доводит
# любую строку до ближайшего адреса. Защита = явный /forget и точное совпадение.
import os, sys, json, time, math
import urllib.request
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DB = os.path.join(HERE, 'frakod_store.json')
_REPO = os.path.abspath(os.path.join(HERE, '..'))


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _dir_accounting(idx_dir):
    """Честный учёт ВСЕХ файлов индекса, а не только кодов.
    Ключевое: '24 Б/токен' — это codes.npy. keys_f16.npy (реранк) и ids.npy
    (лекс-слой) — тоже постоянное хранилище и их нельзя выкидывать из отчёта."""
    files = ('codes.npy', 'keys_f16.npy', 'ids.npy', 'cbook_l0.npy', 'cbook_l1.npy')
    per = {}
    total = 0
    for f in files:
        p = os.path.join(idx_dir, f)
        b = os.path.getsize(p) if os.path.exists(p) else 0
        per[f] = b
        total += b
    mp = os.path.join(idx_dir, 'meta.json')
    N = 0
    if os.path.exists(mp):
        N = int(json.load(open(mp, encoding='utf-8')).get('N') or 0)
    if not N and per['codes.npy']:
        N = per['codes.npy'] // 24
    return {
        'N': N,
        'total_bytes': total,
        'total_mb': round(total / 2**20, 1),
        'codes_bytes': per['codes.npy'],
        'codes_mb': round(per['codes.npy'] / 2**20, 1),
        'keys_bytes': per['keys_f16.npy'],
        'keys_mb': round(per['keys_f16.npy'] / 2**20, 1),
        'ids_bytes': per['ids.npy'],
        'ids_mb': round(per['ids.npy'] / 2**20, 1),
        'cbook_bytes': per['cbook_l0.npy'] + per['cbook_l1.npy'],
        # побайтово на токен, без округления до красивых чисел
        'b_per_tok_codes': round(per['codes.npy'] / N, 3) if N else None,
        'b_per_tok_keys': round(per['keys_f16.npy'] / N, 3) if N else None,
        'b_per_tok_ids': round(per['ids.npy'] / N, 3) if N else None,
        'b_per_tok_total': round(total / N, 3) if N else None,
        'note': ('24 Б/токен = только кодобуки (codes.npy). Полный постоянный '
                 'след включает keys_f16 (реранк) и ids (лекс-слой); '
                 'см. b_per_tok_total. keys можно не держать резидентно: '
                 'реранк читает их memmap-ом с диска.'),
    }

import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer

_tok = None; _EM = None; _MU = None
def load_embedder():
    """Эмбеддер и токенизатор для оперативной памяти — ПО АРХИВУ, не хардкод.

    Раньше жёстко грузились tok_v31.json + ckpt_v7_night_50k.pt. Если архив
    построен другой парой (v8: tok_v8 + d=192), dvec строил вектор из чужой
    таблицы и чужим токенизатором — ровно тот же класс ошибки, что чинился в
    _sts_embed_table. Теперь берём из meta.json (абсолютный путь), а если там
    пусто — откатываемся на исторический v7, чтобы не сломать старые индексы.
    """
    global _tok, _EM, _MU
    if _tok is not None: return
    t0 = time.time()
    m = _arch_meta()

    # --- токенизатор: из meta.json, иначе исторический tok_v31.json ---
    tf = _first_existing(
        os.environ.get('FRK_TOK'),
        m.get('tokenizer_path'),
        os.path.join(HERE, m['tokenizer']) if m.get('tokenizer') else None,
        os.path.join(HERE, 'tok_v31.json'),
    )
    _tok = Tokenizer.from_file(tf)

    # --- таблица эмбеддингов: прямо из чекпойнта (без сборки модели) ---
    _ec = m.get('embed_ckpt')
    _ec_name = m.get('embed_ckpt_name') or (os.path.basename(_ec) if _ec else None)
    ckpt = _first_existing(
        os.environ.get('FRK_STS_CKPT'),
        _ec,
        os.path.join(HERE, _ec) if _ec else None,
        os.path.join(HERE, _ec_name) if _ec_name else None,
        os.path.join(HERE, 'ckpt_v7_night_50k.pt'),
    )
    if ckpt is None:
        raise FileNotFoundError(
            'чекпойнт эмбеддера не найден: укажите FRK_STS_CKPT или положите '
            'чекпойнт из meta.json рядом с frakod_api.py')
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    emb = sd['embed.weight'] if isinstance(sd, dict) and 'embed.weight' in sd else None
    if emb is None:
        raise KeyError(f'в {ckpt} нет embed.weight')
    _EM = emb.detach().float()
    _MU = _EM.mean(0)

    # --- сверка с архивом: vocab и d обязаны совпасть ---
    _v_arch, _d_arch = int(m.get('vocab') or 0), int(m.get('D') or 0)
    if _v_arch and int(_EM.shape[0]) != _v_arch:
        raise ValueError(f'эмбеддер vocab={_EM.shape[0]}, а архив требует {_v_arch}')
    if _d_arch and int(_EM.shape[1]) != _d_arch:
        raise ValueError(f'эмбеддер d={_EM.shape[1]}, а архив требует {_d_arch}')

    print(f'[frk1] эмбеддер памяти: {os.path.basename(ckpt)} {tuple(_EM.shape)} '
          f'+ {os.path.basename(tf)} за {time.time()-t0:.1f}с', flush=True)

def dvec(text):
    ids = [t for t in _tok.encode(text, add_special_tokens=False).ids if t != 0]
    if not ids:
        return F.normalize(_MU.unsqueeze(0), dim=-1)[0]
    return F.normalize(F.normalize(_EM[torch.tensor(ids)] - _MU, dim=-1).mean(0), dim=-1)

# ---------- хранилище ----------
# Запись: {"id":N, "text":..., "meta":..., "bytes":..., "edge_to":[ids]}
# Адрес frk1: "frk1:<slot>.<id>", slot = бакет 64-слотовой карты v7 (аналог слотов модели).
_store = None
def load_store():
    global _store
    if _store is not None: return _store
    if os.path.exists(DB):
        _store = json.load(open(DB, encoding='utf-8'))
    else:
        _store = {'next_id': 1, 'records': [], 'edges': {}, 'slot_of': {}}
    return _store

def save_store():
    json.dump(_store, open(DB, 'w', encoding='utf-8'), ensure_ascii=False)

# ---------- лексический индекс: точный поиск слов по id-последовательности ----
# Лечит слепоту vocab=512 на кириллице ( emb-окно = мешок символов ) БЕЗ
# переобучения: словарь слово->позиции-в-архиве строится из того же текста,
# что токенизировал билдер (corpus из meta.json или env FRK_CORPUS).
_LEX = None
_LEX_LOCK = __import__('threading').Lock()

def _lex_corpus_path():
    """Корпус для лекс-слоя: ищем в нескольких местах (портативность)."""
    cands = []
    try:
        meta = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
        p = meta.get('corpus')
        if p:
            cands.append(p if os.path.isabs(p) else os.path.join(HERE, p))
    except Exception:
        pass
    for env in ('FRK_CORPUS', 'FRK_LEX_CORPUS'):
        p = os.environ.get(env)
        if p:
            cands.append(p if os.path.isabs(p) else os.path.join(HERE, p))
    # типовые места рядом с пакетом и в рабочем exp_vq
    for rel in ('hermes_history_dd.txt', 'hermes_history.txt', 'corpus_public.txt',
                'corpus5m_train.txt'):
        cands.append(os.path.join(HERE, rel))
        cands.append(os.path.join(_REPO, rel))
        cands.append(os.path.join(_REPO, 'phase01', rel))
        cands.append(os.path.join(_REPO, 'phase01', 'exp_vq', rel))
    return _first_existing(*cands)

def build_lex():
    global _LEX
    with _LEX_LOCK:
        if _LEX is not None:
            return _LEX
        path = _lex_corpus_path()
        if not path or not os.path.exists(path):
            _LEX = {}
            return _LEX
        import re, numpy as _np
        t0 = time.time()
        text = open(path, encoding='utf-8', errors='replace').read()
        enc = _archive_tok().encode(text)
        starts = _np.fromiter((o[0] for o in enc.offsets), dtype=_np.int64)
        lex = {}
        # \u2265\u0032 \u0441\u0438\u043c\u0432\u043e\u043b\u0430: \u0438\u0433\u043b\u044b \u0442\u0438\u043f\u0430 'alishahryar'/\u043e\u0431\u0440\u0435\u0437\u0430\u043d\u043d\u044b\u0435 \u0434\u0435\u0444\u0438\u0441\u043e\u043c 'anti-slop-'
        # \u0440\u0430\u043d\u044c\u0448\u0435 \u0432\u044b\u043f\u0430\u0434\u0430\u043b\u0438 \u0438\u0437 \u043b\u0435\u043a\u0441\u0438\u043a\u043e\u043d\u0430 (\u0431\u044b\u043b\u043e {3,} = \u043e\u0442 4 \u0441\u0438\u043c\u0432\u043e\u043b\u043e\u0432)
        for m in re.finditer(_LEX_RE, text):
            a, b = m.span()
            i0 = _np.searchsorted(starts, a, 'left')
            i1 = _np.searchsorted(starts, b, 'left')
            if i1 <= i0:
                continue
            # позиция в ТОКЕНАХ (индекс в ids архива), середина вхождения
            lex.setdefault(m.group(0).lower(), []).append((i0 + i1 - 1) // 2)
        _LEX = {w: _np.unique(_np.array(v, dtype=_np.int64)) for w, v in lex.items()}
        global _LEX_TEXT, _LEX_STARTS, _LEX_BLOB
        _LEX_TEXT, _LEX_STARTS = text, starts
        _LEX_BLOB = text.lower()
    global _LEX_W
    _LEX_W = len(_LEX) if _LEX else 0
    return _LEX

_LEX_TEXT = None     # сырой текст корпуса (для возврата сниппетов без decode)
_LEX_STARTS = None   # token_index -> char offset

# единый шаблон слова для лексикона и запроса (≥2 символа, дефис внутри)
_LEX_RE = r'[A-Za-zА-Яа-яЁё0-9_][A-Za-zА-Яа-яЁё0-9_\-]{1,}'

_LEX_W = 0
_LEX_BLOB = ''       # lower-case текст корпуса для substring-fallback


def _lex_substring_positions(w, starts, cap=2000):
    """Токен-позиции всех ВХОЖДЕНИЙ подстроки w в сырой корпус.

    Нужно, потому что иглы-«редкие слова» часто являются фрагментами токена
    ('alishahryar' внутри 'alishahryar1', 'acceptance-' внутри 'acceptance-тест').
    Точный словарь такие запросы не находит и молча возвращает [] — что выглядит
    как провал поиска, хотя позиция в корпусе есть.
    """
    if not w or len(w) < 3 or _LEX_BLOB is None:
        return []
    import numpy as _np
    out = []
    i = _LEX_BLOB.find(w)
    while i != -1 and len(out) < cap:
        out.append(int(_np.searchsorted(starts, i, 'left')))
        i = _LEX_BLOB.find(w, i + 1)
    return out


def lex_find(qtext, k, win=16):
    """[(token_position, score)] по редкости слов; позиции сливаются окнами win
    (слова фразы стоят рядом, не на одном токене). [] если лексикон недоступен.

    win=16 подобран экспериментально (свип 8..400): широкие окна (400) слипают
    всю фразу в один кластер и вытесняют иглу — snippet-recall падал до 0.30;
    win=16 даёт 1.00 на вопросах и 0.90 на сниппетах."""
    import re, math
    lex = build_lex()
    if not lex:
        return []
    words = {x.lower() for x in re.findall(_LEX_RE, qtext)}
    # короткие осколки (<=2 симв.) не несут смысла и дают шум
    words = {w for w in words if len(w) >= 3}
    pts = []   # (pos, weight)
    miss = []  # слова, которых нет в словаре -> substring-fallback
    for w in words:
        a = lex.get(w)
        if a is None or len(a) == 0:
            miss.append(w)
            continue
        if len(a) > 50_000:
            continue
        wt = 1.0 / math.log2(2.0 + len(a))
        pts.extend((int(p), wt) for p in a[:20_000])
    # fallback: запрос может быть ФРАГМЕНТОМ токена корпуса — ищем по подстроке
    if miss and _LEX_STARTS is not None:
        for w in sorted(miss, key=len, reverse=True)[:8]:
            ps = _lex_substring_positions(w, _LEX_STARTS)
            if not ps:
                continue
            wt = 1.0 / math.log2(2.0 + len(ps))
            pts.extend((int(p), wt) for p in ps[:20_000])
    if not pts:
        return []
    pts.sort()
    # Группируем позиции в окна win. Репрезентант окна = позиция САМОГО РЕДКОГО
    # слова (иначе сниппет якорится на левую границу и игла выпадает за ±window).
    # РАНЖИР — по редкости самого редкого слова, НЕ по сумме весов: в вопросе
    # «Что в истории связано со словом X?» слова что/истории/связано массовы и,
    # суммируясь, вытесняли саму иглу X из топ-k.
    acc = []   # (max_weight, rep_pos, sum_weight)
    clus = []  # [(pos, wt)]
    def _flush(clus):
        if not clus:
            return
        mx = max(w for _, w in clus)
        rep = max(clus, key=lambda x: x[1])[0]
        acc.append((mx, rep, sum(w for _, w in clus)))
    for p, wt in pts:
        if clus and p - clus[0][0] > win:
            _flush(clus)
            clus = []
        clus.append((p, wt))
    _flush(clus)
    ranked = sorted(acc, key=lambda x: (-x[0], -x[2]))[:k]
    return [(int(p), round(float(mx), 3)) for mx, p, _ in ranked]


_vec_cache = {}

# ---------- архивный индекс (10M токенов, Fracode S=12) ----------
IDX_DIR = os.environ.get('FRK_IDXD') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frakod_index')
_IDX = None            # None = не грузили; dict = жив
_idx_lock = __import__('threading').Lock()

def index_status():
    if _IDX is not None:
        return {'loaded': True, 'N': int(_IDX['N'])}
    return {'loaded': False, 'artifacts': os.path.exists(os.path.join(IDX_DIR, 'codes.npy'))}

def load_index(nmax=500_000):
    """Ленивая загрузка архива. nmax - сколько кодов держать в RAM для скрининга
    (демо-режим: топ-N сегмент; полный 10M = 229 МБ кодов + 3.8 ГБ keys memmap)."""
    global _IDX
    with _idx_lock:
        if _IDX is not None: return _IDX
        import numpy as np
        t0 = time.time()
        codes = np.load(os.path.join(IDX_DIR, 'codes.npy'), mmap_mode='r')
        keys = np.load(os.path.join(IDX_DIR, 'keys_f16.npy'), mmap_mode='r')
        cbook = [np.load(os.path.join(IDX_DIR, f'cbook_l{l}.npy')) for l in (0, 1)]
        meta = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
        _IDX = {'codes': codes, 'keys': keys, 'cbook': cbook, 'meta': meta,
                'N': codes.shape[0], 'S': meta['S'], 'load_s': time.time()-t0}
        print(f'[frk1] архив: {_IDX["N"]:,} токенов, кодобук за {_IDX["load_s"]:.1f}с', flush=True)
        return _IDX

_G = None
def _to_gpu():
    """Ленивый перенос архива на CUDA. Ошибка -> None (остаёмся на CPU)."""
    global _G
    import numpy as np
    if _G is not None: return _G if _G is not False else None
    try:
        idx = _IDX
        codes = torch.from_numpy(np.ascontiguousarray(np.asarray(idx['codes']))).to('cuda')
        keys = torch.from_numpy(np.ascontiguousarray(np.asarray(idx['keys']))).to('cuda')
        cb = torch.from_numpy(np.ascontiguousarray(np.stack(idx['cbook']))).to('cuda')  # (2,S,K,sub)
        mu = keys[::7919].float().mean(0)
        _G = {'codes': codes, 'keys': keys, 'cb': cb, 'mu': mu}
        return _G
    except Exception as e:
        print(f'[frk1] GPU-путь недоступен ({type(e).__name__}) -> CPU', flush=True)
        _G = False
        return None

def _recall_gpu(qvec, k_return, kcand):
    import numpy as np
    idx = _IDX
    G = _to_gpu()
    if G is None:
        return _recall_cpu(qvec, k_return, kcand)
    try:
        t0 = time.time()
        S = idx['S']; N = idx['N']; sub = idx['keys'].shape[1] // S
        q = qvec.to('cuda').float(); q = q / (q.norm() + 1e-9)
        adc = torch.zeros(N, device='cuda')
        for l in range(2):
            for s in range(S):
                tbl = G['cb'][l, s] @ q[s*sub:(s+1)*sub]     # (256,)
                adc += tbl[G['codes'][:, l, s].long()]
        kc = min(kcand, N)
        cand = adc.topk(kc).indices
        ck = G['keys'][cand[:min(kc, 200000)]].float()
        ck = ck - G['mu']; qs = q - G['mu']
        dot = torch.nn.functional.normalize(ck, dim=1) @ torch.nn.functional.normalize(qs.view(-1,1), dim=0).view(-1)
        top = dot.topk(k_return).indices
        torch.cuda.synchronize()
        pos = cand[top].cpu().tolist()
        return {'positions': pos, 'scores': [round(float(v), 4) for v in dot[top]],
                'ms': round((time.time()-t0)*1000, 1), 'screened': N, 'kcand': int(kc),
                'device': 'cuda'}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _recall_cpu(qvec, k_return, kcand)

def recall_vec(qvec, k_return=8, kcand=65536):
    """Точка входа: CUDA если жива, иначе numpy-CPU."""
    if torch.cuda.is_available() and _G is not False:
        return _recall_gpu(qvec, k_return, kcand)
    return _recall_cpu(qvec, k_return, kcand)


def recall_adc_only(qvec, k_return=8, kcand=65536):
    """ТОЛЬКО 24-байтовые коды: ранжируем по ADC, keys_f16 не читаем вообще.
    Служит для честного замера того, что даёт сам сжатый кодобук.

    ВАЖНО: ADC = скалярное произведение кода с запросом, а кодобуки обучены на
    НЕнормированных контекстных ключах. Поэтому запрос нормализовать нельзя
    (раньше нормализовался — это ломало масштаб и весь ранжир)."""
    import numpy as np
    idx = _IDX
    q = qvec.detach().cpu().numpy().astype(np.float32)   # БЕЗ нормализации (см. docstring)
    S = idx['S']; sub = idx['keys'].shape[1] // S
    N = idx['N']; CH = 2_000_000
    t0 = time.time()
    adc = np.empty(N, dtype=np.float32)
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        acc = np.zeros(i1 - i0, dtype=np.float32)
        cchunk = np.asarray(idx['codes'][i0:i1])
        for l in range(2):
            for s in range(S):
                tbl = idx['cbook'][l][s] @ q[s * sub:(s + 1) * sub]   # (256,)
                acc += tbl[cchunk[:, l, s]]
        adc[i0:i1] = acc
    kk = min(k_return, N)
    top = np.argpartition(-adc, kk)[:kk]
    top = top[np.argsort(-adc[top])]
    ms = (time.time() - t0) * 1000
    return {'positions': top.tolist(), 'scores': [round(float(adc[t]), 4) for t in top],
            'ms': round(ms, 1), 'screened': N, 'kcand': int(kk), 'ranking': 'adc_only'}

def _recall_cpu(qvec, k_return, kcand):
    """numpy-путь (RAM ~4 ГБ, ~320 мс на 1M; ~1.5-2 с на 10M)."""
    import numpy as np
    idx = _IDX
    q = (qvec.detach().cpu().numpy().astype(np.float32))
    q = q / (np.linalg.norm(q) + 1e-9)
    S = idx['S']; sub = idx['keys'].shape[1] // S
    N = idx['N']; CH = 2_000_000
    t0 = time.time()
    adc = np.empty(N, dtype=np.float32)
    for i0 in range(0, N, CH):
        i1 = min(i0+CH, N)
        acc = np.zeros(i1-i0, dtype=np.float32)
        cchunk = np.asarray(idx['codes'][i0:i1])
        for l in range(2):
            for s in range(S):
                qsub = q[s*sub:(s+1)*sub]
                tbl = idx['cbook'][l][s] @ qsub   # (256,)
                acc += tbl[cchunk[:, l, s]]
        adc[i0:i1] = acc
    kc = min(kcand, N)
    # ADC-оценка СИСТЕМатически смещена: u8-кодобук не центрирован, а keys при
    # переранке центрируются на mu. Смещение одинаково по всем ключам -> на
    # рангах почти не сказывается, НО на диалогах с кластерами-близнецами топит
    # истинное окно: берём кандидатов не только по ADC-топу, а с запасом, и
    # переранжируем сырым косинусом ВСЕХ (kc=large).
    cand = np.argpartition(-adc, kc)[:kc]
    cand_scores = adc[cand]
    order = np.argsort(-cand_scores)[:kc]
    cand = cand[order]
    ck = np.asarray(idx['keys'][cand[:min(kc, 200000)]], dtype=np.float32)
    if 'mu' not in idx:   # центр ключей - один раз
        idx['mu'] = np.asarray(idx['keys'][::7919], dtype=np.float32).mean(0)
    ck = ck - idx['mu']
    norm = ck / (np.linalg.norm(ck, axis=1, keepdims=True) + 1e-9)
    qs = q - idx['mu']
    qs = qs / (np.linalg.norm(qs) + 1e-9)
    dot = norm @ qs
    top = np.argsort(-dot)[:k_return]
    pos = cand[top].tolist()
    ms = (time.time()-t0)*1000
    return {'positions': pos, 'scores': [round(float(dot[t]), 4) for t in top],
            'ms': round(ms, 1), 'screened': N, 'kcand': int(kc)}
def get_vec(i, text):
    if i not in _vec_cache:
        load_embedder()
        _vec_cache[i] = dvec(text)
    return _vec_cache[i]

def assign_slot(i, text):
    st = load_store()
    if i in st['slot_of']: return st['slot_of'][i]
    v = get_vec(i, text)
    load_embedder()
    g = torch.Generator().manual_seed(777)
    protos = F.normalize(_EM[torch.randint(1, _EM.shape[0], (64,), generator=g)] - _MU, dim=-1)
    s = int((protos @ v).argmax())
    st['slot_of'][i] = s
    return s

def _ensure_slots():
    """Достроить slot_of для записей, созданных до его появления: адрес
    'frk1:<slot>.<id>' не должен отдавать -1."""
    st = load_store()
    if len(st['slot_of']) >= len(st['records']):
        return
    for r in st['records']:
        if r['id'] not in st['slot_of']:
            assign_slot(r['id'], r['text'])
    save_store()


app = FastAPI(title='frk1', version='0.1')
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class RememberIn(BaseModel):
    text: str
    meta: Optional[str] = None

def parse_ref(ref):
    """Принять id (int / '12') или адрес ('frk1:7.12') и вернуть int-id.
    None = разобрать не удалось."""
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    s = str(ref).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s.startswith('frk1:') and '.' in s:
        tail = s.split('.', 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


class LinkIn(BaseModel):
    # принимает и {"src": 3, "dst": 4}, и {"frm": "frk1:7.3", "to": "frk1:2.4"}
    src: Optional[object] = None
    dst: Optional[object] = None
    frm: Optional[object] = None
    to: Optional[object] = None

    def pair(self):
        a = self.src if self.src is not None else self.frm
        b = self.dst if self.dst is not None else self.to
        return parse_ref(a), parse_ref(b)

class ResolveIn(BaseModel):
    query: str
    hop: bool = False

@app.get('/')
def root():
    st = load_store()
    return {'service': 'frk1', 'records': len(st['records']),
            'edges': sum(len(v) for v in st['edges'].values()),
            'note': 'текстово-графовый канал; нейро-порт закрыт гейтом 1 (Sec.4.36)'}

@app.post('/remember')
def remember(body: RememberIn):
    st = load_store()
    i = st['next_id']; st['next_id'] += 1
    tb = len(body.text.encode('utf-8'))
    st['records'].append({'id': i, 'text': body.text, 'meta': body.meta or '', 'bytes': tb})
    slot = assign_slot(i, body.text)
    st.setdefault('history', []).append([int(time.time()), len(st['records']),
                                         sum(r['bytes'] for r in st['records'])])
    st['history'] = st['history'][-500:]
    save_store()
    return {'id': i, 'address': f'frk1:{slot}.{i}', 'bytes': tb}

@app.post('/link')
def link(body: LinkIn):
    st = load_store()
    src, dst = body.pair()
    if src is None or dst is None:
        raise HTTPException(422, 'не разобрал адрес: жду id или "frk1:<slot>.<id>"')
    ids = {r['id'] for r in st['records']}
    if src not in ids or dst not in ids:
        raise HTTPException(404, f'нет записи (src={src}, dst={dst})')
    st['edges'].setdefault(str(src), [])
    if dst not in st['edges'][str(src)]:
        st['edges'][str(src)].append(dst)
    save_store()
    return {'edge': f'{src} -> {dst}', 'src': src, 'dst': dst}

def _topk(qtext, k=3, exclude=None):
    st = load_store()
    if not st['records']: return []
    v = dvec(qtext)
    scored = []
    for r in st['records']:
        if exclude is not None and r['id'] == exclude: continue
        rv = get_vec(r['id'], r['text'])
        scored.append((float(v @ rv), r))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]

@app.post('/resolve')
def resolve(body: ResolveIn):
    load_embedder()  # dvec/_EM нужны до первого касания
    st = load_store()
    _ensure_slots()
    top = _topk(body.query)
    if not top: return {'hit': None}
    cos, r = top[0]
    out = {'id': r['id'], 'address': f"frk1:{st['slot_of'].get(r['id'], -1)}.{r['id']}",
           'text': r['text'], 'cos': round(cos, 4), 'via': 'top1'}
    if body.hop:
        e = st['edges'].get(str(r['id']), [])
        if e:
            dst = next((x for x in st['records'] if x['id'] == e[0]), None)
            if dst:
                out['hop'] = {'id': dst['id'], 'text': dst['text'],
                              'address': f"frk1:{st['slot_of'].get(dst['id'], -1)}.{dst['id']}"}
    out['warning'] = 'deref без "не знаю": см. Sec.4.42-4.45 (мягкий слой закрыт)'
    return out

@app.get('/chain/{start_id}')
def chain(start_id: str, max_hops: int = 8, depth: Optional[int] = None):
    # совместимость: старые клиенты шлют ?depth=3
    if depth is not None:
        max_hops = depth
    sid = parse_ref(start_id)
    if sid is None:
        raise HTTPException(422, 'не разобрал адрес: жду id или "frk1:<slot>.<id>"')
    st = load_store(); seen, cur, path = set(), sid, []
    while cur is not None and cur not in seen and len(path) < max(1, max_hops):
        seen.add(cur)
        rec = next((x for x in st['records'] if x['id'] == cur), None)
        if rec is None: break
        path.append({'id': cur, 'address': f"frk1:{st['slot_of'].get(cur, -1)}.{cur}", 'text': rec['text']})
        nxt = st['edges'].get(str(cur), [])
        cur = nxt[0] if nxt else None
    return {'path': path, 'hops': len(path) - 1 if path else 0}

@app.delete('/forget/{rec_id}')
def forget(rec_id: int):
    st = load_store()
    st['records'] = [r for r in st['records'] if r['id'] != rec_id]
    st['edges'].pop(str(rec_id), None)
    for v in st['edges'].values():
        if rec_id in v: v.remove(rec_id)
    st['slot_of'].pop(rec_id, None); _vec_cache.pop(rec_id, None)
    save_store()
    return {'forgotten': rec_id, 'records': len(st['records'])}

@app.get('/stats')
def stats():
    st = load_store()
    n = len(st['records'])
    tb = sum(r['bytes'] for r in st['records'])
    load_embedder()
    tok_bytes = int(_EM.numel() * _EM.element_size())
    return {'records': n, 'memory_bytes': tb, 'kb_per_record': round(tb / max(n, 1) / 1024, 2),
            'edges': sum(len(v) for v in st['edges'].values()),
            'slots_used': len(set(st['slot_of'].values())),
            'embedder_table_bytes': tok_bytes,
            'window_note': 'диск = таблица эмбеддера (~фикс.) + записи (растёт линейно); окно хоста не трогается вовсе'}

@app.get('/window')
def window():
    st = load_store()
    hist = st.get('history', [])
    n = len(st['records'])
    tb = sum(r['bytes'] for r in st['records'])
    load_embedder()
    tok_bytes = int(_EM.numel() * _EM.element_size())
    # сколько токенов контекста ЭТО стоило бы в window-хосте, если заливать всё текстом
    fake_tok = len(_tok.encode(' '.join(r['text'] for r in st['records']),
                               add_special_tokens=False).ids) if n else 0
    return {'records': n, 'disk_bytes': tb, 'fixed_overhead_bytes': tok_bytes,
            'tokens_if_inline': fake_tok,
            'frakod_inline_cost_per_hit': len('frk1:27.123'),
            'series': [{'t': t, 'n': nn, 'bytes': bb} for t, nn, bb in hist[-60:]]}

# ---------- /ask: оркестратор frk1 <-> LLM-хост (текстовый канал) ----------
CHAT_URL = 'http://127.0.0.1:8123/chat'
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'

class AskIn(BaseModel):
    query: str
    use_hop: bool = True
    use_store: bool = True   # False = БЕЗ памяти (для A/B-кнопки)
    use_archive: bool = False  # True = дополнить контекст находками из 10M-архива
    temperature: float = 0.8
    chat: str = 'ollama:your-chat-model'  # по умолчанию живая ...[truncated]

def _post_chat(message, temperature, chat='v7'):
    msg = message[:1800]
    if chat.startswith('ollama:'):
        payload = {'model': chat.split(':', 1)[1], 'stream': False,
                   'messages': [{'role': 'user', 'content': msg}],
                   'options': {'temperature': temperature, 'num_predict': 120}}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(OLLAMA_URL, data=data,
                                     headers={'Content-Type': 'application/json'})
        r = json.loads(urllib.request.urlopen(req, timeout=240).read())
        return r.get('message', {}).get('content', ''), None
    payload = {'message': msg, 'temperature': temperature, 'max_tokens': 64}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(CHAT_URL, data=data,
                                 headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    return r.get('reply', ''), r.get('session_id')

@app.post('/ask')
def ask(body: AskIn):
    load_embedder()
    st = load_store()
    res = resolve(ResolveIn(query=body.query, hop=body.use_hop)) if body.use_store else {}
    hit = res if res.get('id') else None
    ctx = []
    if hit:
        ctx.append(f"[frk1 {hit['address']}] {hit['text']}")
        if hit.get('hop'):
            ctx.append(f"[frk1 {hit['hop']['address']}] {hit['hop']['text']}")
    arch = []
    if body.use_archive and os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        try:
            rr = _recall_text(body.query, k_return=2)
            for h in rr['results']:
                arch.append(f"[архив {h['age_tokens_from_end']:,} ток. назад] {h['text'][:200]}")
        except Exception as e:
            arch = [f'[архив недоступен: {type(e).__name__}]']
    msg = body.query
    parts = ctx + arch
    if parts:
        msg = 'Память (frk1):\n' + chr(10).join(parts) + chr(10) + 'Вопрос: ' + body.query
    reply, sid = _post_chat(msg, body.temperature, body.chat)
    return {'reply': reply, 'answer': reply, 'chat': body.chat, 'session_id': sid,
            'address_used': hit['address'] if hit else (arch and 'archive'),
            'archive_hits': len(arch),
            'context_tokens': len(_tok.encode(chr(10).join(parts), add_special_tokens=False).ids) if parts else 0,

            'inline_tokens_would_be': sum(r['bytes'] // 3 for r in st['records']) if hit else 0}

@app.delete('/store')
def wipe():
    global _store
    _store = {'next_id': 1, 'records': [], 'edges': {}, 'slot_of': {}}
    _vec_cache.clear()
    save_store()
    return {'ok': True}

# ---------- публичные эндпоинты архива (демо-стенд) ----------
_STS_EMB = None
_ARCH_META = None
def _arch_meta():
    """meta.json архива (кэш). Определяет токенизатор, эмбеддер, S и RAD."""
    global _ARCH_META
    if _ARCH_META is None:
        try:
            _ARCH_META = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
        except Exception:
            _ARCH_META = {}
    return _ARCH_META


def _sts_embed_table():
    """Эмбеддер, СООТВЕТСТВУЮЩИЙ архиву (по meta.json.embed_ckpt), не хардкод.

    Раньше всегда грузился sts_prog_seed0.pt (d=192): при архиве на v7 (d=256)
    это давало либо ошибку формы, либо (хуже) сравнение векторов из разных
    пространств. Порядок: env -> embed_ckpt из meta -> оба имени рядом.
    """
    global _STS_EMB
    if _STS_EMB is None:
        import torch as T
        m = _arch_meta()
        _ec = m.get('embed_ckpt')
        _ec_name = m.get('embed_ckpt_name') or (os.path.basename(_ec) if _ec else None)
        cand = _first_existing(
            os.environ.get('FRK_STS_CKPT'),
            _ec,                                    # АБСОЛЮТНЫЙ путь из meta (v8+)
            os.path.join(HERE, _ec) if _ec else None,
            os.path.join(HERE, _ec_name) if _ec_name else None,
            os.path.join(_REPO, _ec_name) if _ec_name else None,
            os.path.join(HERE, 'sts_prog_seed0.pt'),
            os.path.join(_REPO, 'results', 'ckpts', 'sts_prog_seed0.pt'),
            os.path.join(_REPO, 'chaotic-llm', 'results', 'ckpts', 'sts_prog_seed0.pt'),
        )
        # ЧЕСТНО: если архив требует конкретный чекпойнт, а мы его не нашли —
        # НЕ подставляем молча seed0 (это тихо ломает вектор: ключи из одной
        # таблицы, запрос из другой). Падаем с внятной ошибкой.
        if _ec and cand is not None and os.path.abspath(cand) != os.path.abspath(_ec):
            _want = _ec_name or _ec
            if not (os.path.basename(cand).startswith('sts_prog_seed0') and
                    _want.startswith('sts_prog_seed0')):
                raise FileNotFoundError(
                    f'архив построен чекпойнтом {_want!r}, но он не найден; '
                    f'подставился {os.path.basename(cand)!r} — это другое векторное '
                    f'пространство. Положите {_want} рядом с frakod_api.py или '
                    f'задайте FRK_STS_CKPT.')
        if cand is None:
            raise FileNotFoundError(
                'эмбеддер не найден: положите чекпойнт рядом с frakod_api.py '
                'или задайте FRK_STS_CKPT (нужен для векторного канала).')
        sd = T.load(cand, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd) if isinstance(sd, dict) else sd
        _STS_EMB = sd['embed.weight'].detach().float().cpu().numpy()
        # Сверка размерности с архивом: d обязано совпасть с meta['D'].
        _d_arch = int(m.get('D') or 0)
        if _d_arch and int(_STS_EMB.shape[1]) != _d_arch:
            raise ValueError(
                f'эмбеддер {os.path.basename(cand)} имеет d={_STS_EMB.shape[1]}, '
                f'а архив требует d={_d_arch} — ключи и запрос из разных таблиц.')
        print(f'[frk1] эмбеддер архива: {os.path.basename(cand)} {_STS_EMB.shape}', flush=True)
    return _STS_EMB

_BPE = None
def _archive_tok():
    """Токенизатор, СОГЛАСОВАННЫЙ с архивом.

    Критично: ids запроса обязаны совпадать с ids архива. Архивы, построенные
    дефолтным make_bpe(corpus_public, 512), несовместимы с tok_v31.json (другой
    словарь и другое разбиение). Поэтому первым делом смотрим meta.json.
    """
    global _BPE
    if _BPE is None:
        tf = os.environ.get('FRK_TOK')
        kind = None
        try:
            _m = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
            kind = _m.get('tokenizer_kind')
            if not tf and kind == 'external' and _m.get('tokenizer'):
                cand = _first_existing(os.path.join(HERE, _m['tokenizer']),
                                       os.path.join(_REPO, _m['tokenizer']),
                                       os.path.join(HERE, 'tok_v31.json'))
                tf = cand
        except Exception:
            _m = {}
        if tf and os.path.exists(tf):
            from tokenizers import Tokenizer as _Tkk
            _BPE = _Tkk.from_file(tf)
        else:
            # архив на дефолтном BPE-512: берём тот же конструктор, что и билдер
            import final_benchmark as fb
            cp = _first_existing(os.path.join(_REPO, 'corpus_public.txt'),
                                 os.path.join(_REPO, 'chaotic-llm', 'phase01', 'corpus_public.txt'),
                                 os.path.join(_REPO, 'phase01', 'corpus_public.txt'))
            head = fb.load_chars(cp, 990_000)
            _BPE = fb.make_bpe(head, vocab=int((_m or {}).get('vocab', 512) or 512))
    return _BPE

@app.get('/index/stats')
def index_stats():
    st = index_status()
    acc = _dir_accounting(IDX_DIR)
    if not st.get('loaded') and st.get('artifacts'):
        st['meta'] = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
    st['accounting'] = acc
    # старые клиенты/бенч читают fr_mb как "размер памяти" -> отдаём честный total
    st['fr_mb'] = acc['total_mb']
    st['fr_mb_codes_only'] = acc['codes_mb']
    return st


@app.get('/index/accounting')
def index_accounting():
    """Полная побайтовая бухгалтерия постоянного следа архива."""
    return _dir_accounting(IDX_DIR)

@app.post('/index/load')
def index_load():
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        return {'ok': False, 'error': 'архив не построен (frakod_index_build.py)'}
    idx = load_index()
    acc = _dir_accounting(IDX_DIR)
    return {'ok': True, 'N': idx['N'], 'load_s': round(idx['load_s'], 1),
            'fr_mb': acc['total_mb'], 'fr_mb_codes_only': acc['codes_mb'],
            'accounting': acc}

class RecallIn(BaseModel):
    text: str = ''
    tokens: list[int] = []   # обход BPE-roundtrip (для точных тестов)
    k: int = 8
    kcand: int = 65536
    window: int = 24      # токенов контекста вокруг находки
    mode: str = 'auto'    # auto | vector | lex | vector24
    rerank: bool = True   # False = чистый ADC по 24-байтовым кодам, БЕЗ keys_f16

@app.post('/recall')
def recall(body: RecallIn):
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        return {'ok': False, 'error': 'архив не построен'}
    idx = load_index()
    import numpy as np
    tkn = _archive_tok()
    mode = getattr(body, 'mode', 'auto')   # auto | vector | lex | vector24
    if mode == 'vector24':
        # честный тест «что умеют сами 24 Б/токен»: только ADC по кодобукам,
        # без доступа к keys_f16.npy (реранк по сырым ключам отключён)
        body = body.model_copy(update={'mode': 'vector', 'rerank': False})
        mode = 'vector'
    ids_q = list(body.tokens) if body.tokens else [t for t in tkn.encode(body.text).ids]
    ids_q = [t for t in ids_q if t < _sts_embed_table().shape[0]]
    N = idx['N']
    results = []
    seen = set()
    def snip(p, score, via):
        i0 = max(0, p - body.window); i1 = min(N, p + body.window)
        txt = ''
        if _LEX_TEXT is not None and _LEX_STARTS is not None:
            # сырой текст корпуса по карте токен->символ (decodevocab=512 уродует кириллицу)
            try:
                c0 = int(_LEX_STARTS[max(0, i0)]); c1 = int(_LEX_STARTS[min(i1, len(_LEX_STARTS)-1)])
                txt = _LEX_TEXT[max(0, c0-80):c1+80].replace('\n', ' ')
            except Exception:
                txt = ''
        if not txt:
            txt = tkn.decode(np.asarray(ids_arc_load()[i0:i1].tolist())).replace('\n', ' ')
        results.append({'position': int(p), 'age_tokens_from_end': int(N - p),
                        'score': score, 'via': via, 'text': txt[:600]})
    if mode in ('auto', 'lex') and body.text and not body.tokens:
        for p, s in lex_find(body.text, body.k):
            if p >= N or p in seen:
                continue
            seen.add(p)
            snip(p, s, 'lex')
    if mode in ('auto', 'vector') and ids_q:
        E = _sts_embed_table()
        # ВАЖНО: ключи архива = context-avg с окном RAD=4 (см. frakod_index_build.py).
        # Раньше запрос считался как E[ids_q].mean(0) БЕЗ контекстного усреднения —
        # это другое пространство, cos падал с 1.000 до 0.958 и top1 уходил на +4..5
        # токенов. Теперь запрос строится тем же оператором, что и ключи.
        import numpy as _np
        # RAD берём из meta архива (билдер кладёт key_context_rad), чтобы запрос
        # строился ровно тем же оператором, что и ключи
        RAD = int(os.environ.get('FRK_RAD') or _arch_meta().get('key_context_rad') or 4)
        ids_full = [t for t in (body.tokens if body.tokens else tkn.encode(body.text).ids)]
        ids_full = [t for t in ids_full if t < E.shape[0]]
        if ids_full:
            pad = [ids_full[0]] * RAD + list(ids_full) + [ids_full[-1]] * RAD
            win = _np.asarray(pad, dtype=_np.int64)
            # среднее по скользящему окну 2RAD+1 -> (len(ids_full), D)
            cum = _np.cumsum(_np.vstack([_np.zeros((1, E.shape[1]), dtype=_np.float32),
                                         E[win].astype(_np.float32)]), axis=0)
            k = 2 * RAD + 1
            ctx_emb = (cum[k:] - cum[:-k]) / k          # (len(ids_full), D)
            q = ctx_emb.mean(0).astype(_np.float32)     # центр запроса
        else:
            q = E[ids_q].mean(0)
        if body.rerank:
            r = recall_vec(torch.from_numpy(q), k_return=body.k, kcand=body.kcand)
        else:
            r = recall_adc_only(torch.from_numpy(q), k_return=body.k, kcand=body.kcand)
        for p, sc in zip(r['positions'], r['scores']):
            if p in seen:
                continue
            seen.add(p)
            snip(p, sc, 'vec')
    else:
        r = {}
    results = results[:body.k]
    return {'ok': True, 'query': body.text[:80], 'ms_total': r.get('ms') if ids_q and mode != 'lex' else 0,
            'screened': N, 'mode': mode, 'lex_words': _LEX_W, 'results': results}

_ids_cache = None
def ids_arc_load():
    global _ids_cache
    if _ids_cache is None:
        import numpy as np
        _ids_cache = np.load(os.path.join(IDX_DIR, 'ids.npy'), mmap_mode='r')
    return _ids_cache

def _recall_text(text, k_return=2):
    return recall(RecallIn(text=text, k=k_return))

