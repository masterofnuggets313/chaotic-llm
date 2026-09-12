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

# ОБХОД ПРОКСИ (обязательно). Этот процесс ходит ТОЛЬКО на localhost (Ollama).
# urllib по умолчанию подхватывает http_proxy из окружения и гонит даже 127.0.0.1
# через него; чужой прокси отвечает 502, и это выглядит как «энкодер умер».
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({})))


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
        # fallback только если meta нет. ЧЕСТНО: НЕ «// 24» — код бывает 16 Б
        # (S=8 при d=1024), 24 Б (S=12 при d=192), 32 Б (S=16 при d=256).
        # Читаем форму массива из заголовка .npy: N = shape[0]. Без догадок.
        try:
            import numpy.lib.format as _nf
            with open(os.path.join(idx_dir, 'codes.npy'), 'rb') as _fh:
                _ver = _nf.read_magic(_fh)
                _sh, _fo, _dt = _nf._read_array_header(_fh, _ver)
            N = int(_sh[0])
        except Exception:
            N = 0
    # РЕЕСТР ЭНКОДЕРА (Путь C): статистика обязана говорить, чем читается архив.
    # Иначе «20 Б/окно» звучит как самодостаточный файл, а он не самодостаточен.
    _enc_note = {}
    try:
        _m = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else {}
        if _m.get('encoder_kind') == 'external':
            _kb = _m.get('encoder_kb') or {}
            _enc_note = {
                'encoder_kind': 'external',
                'encoder_model': _m.get('encoder_model'),
                'encoder_how': _kb.get('how') or f'ollama pull {_m.get("encoder_model")}',
                'encoder_required': True,
                'encoder_note': ('Архив читается ТОЛЬКО этим энкодером. Другой '
                                 'энкодер = другое пространство = recall 0.000 '
                                 '(измерено). Проверка: /memory/verify.'),
            }
        else:
            _enc_note = {'encoder_kind': 'internal',
                         'encoder_model': _m.get('encoder_model'),
                         'encoder_required': True,
                         'encoder_note': 'Энкодер = ckpt из meta.embed_ckpt.'}
    except Exception:
        _enc_note = {}
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
        # побайтово на ЕДИНИЦУ (токен или окно), без округления до красивых чисел.
        # ЧЕСТНО (исправлено 12.09): кодобук имеет ФИКСИРОВАННЫЙ размер, поэтому
        # включать его в «B на токен» нельзя — на маленьком архиве он даёт
        # нелепость (7934 окна -> 264 Б/окно кодобука). Разделяем:
        #   resident  — то, что растёт с архивом (это и есть цена памяти);
        #   amortized — вклад кодобука, падает с ростом N.
        'b_per_tok_codes': round(per['codes.npy'] / N, 3) if N else None,
        'b_per_tok_keys': round(per['keys_f16.npy'] / N, 3) if N else None,
        'b_per_tok_ids': round(per['ids.npy'] / N, 3) if N else None,
        'b_per_tok_total': (round((per['codes.npy'] + per['keys_f16.npy']
                                   + per['ids.npy']) / N, 3) if N else None),
        'b_per_tok_cbook_amortized': (round((per['cbook_l0.npy']
                                             + per['cbook_l1.npy']) / N, 3)
                                      if N else None),
        'b_per_tok_total_with_cbook': round(total / N, 3) if N else None,
        'unit': 'токен (внутренний архив) / окно (внешний энкодер)',
        **_enc_note,
        'note': ('B/единица = codes + keys + ids: это ЦЕНА ПАМЯТИ, растущая с '
                 'архивом. Кодобук фиксированного размера — его вклад указан '
                 'отдельно (b_per_tok_cbook_amortized) и падает с ростом N. '
                 'keys можно не держать резидентно: реранк читает memmap с диска.'),
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
            if os.path.isabs(p):
                cands.append(p)
            else:
                # meta хранит ГОЛОЕ имя файла, а корпус может лежать в phase01/.
                # Раньше искали только рядом с пакетом, не находили — и молча
                # подхватывали ЧУЖОЙ файл из списка ниже (дефект 12.09: для
                # архива 10M брался hermes_history_dd.txt, где всего 7933 окна,
                # из-за чего окна >= 7934 давали ПУСТОЙ текст, а остальные —
                # текст из совсем другого документа). Ищем во всех тех же
                # каталогах, что и типовые корпуса.
                for base in (HERE, _REPO, os.path.join(_REPO, 'phase01'),
                             os.path.join(_REPO, 'phase01', 'exp_vq')):
                    cands.append(os.path.join(base, p))
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

_CORPUS_CACHE = {'path': None, 'text': None}

def _corpus_text():
    """Сырой текст корпуса (лениво, один раз). Нужен оконным архивам: у них
    `position` — индекс ОКНА, и достать текст по нему можно только из корпуса,
    а не через токенизатор (ids.npy там = середины окон, decode даёт пустоту)."""
    p = _lex_corpus_path()
    if not p:
        return ''
    if _CORPUS_CACHE['path'] == p and _CORPUS_CACHE['text'] is not None:
        return _CORPUS_CACHE['text']
    try:
        t = open(p, encoding='utf-8', errors='replace').read()
    except Exception:
        t = ''
    _CORPUS_CACHE['path'], _CORPUS_CACHE['text'] = p, t
    return t

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
def _resolve_idx_dir():
    """Каталог архива: FRK_IDXD -> frakod_index -> свежий frakod_index_v*.

    РАНЬШЕ: жёстко `frakod_index` по умолчанию. Такого каталога в репозитории НЕТ
    (архивы лежат в frakod_index_v7 / frakod_index_v8), поэтому API и MCP не
    поднимались «из коробки» у того, кто не собирал архив сам. Ловушка была
    двойная: после первой сборки каталог frakod_index появлялся, и путь «внезапно»
    становился верным — то есть баг не воспроизводился у автора.
    Теперь: явный env -> исторический дефолт -> самый свежий по built_utc."""
    env = os.environ.get('FRK_IDXD')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    legacy = os.path.join(here, 'frakod_index')
    if os.path.exists(os.path.join(legacy, 'codes.npy')):
        return legacy
    import glob as _glob
    best, best_utc = None, ''
    for d in sorted(_glob.glob(os.path.join(here, 'frakod_index_v*'))):
        mp = os.path.join(d, 'meta.json')
        if not (os.path.exists(os.path.join(d, 'codes.npy')) and os.path.exists(mp)):
            continue
        try:
            utc = json.load(open(mp, encoding='utf-8')).get('built_utc', '')
        except Exception:
            utc = ''
        if utc >= best_utc:
            best, best_utc = d, utc
    if best:
        return best
    return legacy   # не найден: путь вернём, ошибка будет внятной при загрузке


IDX_DIR = _resolve_idx_dir()
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
        # keys_f16 — ОПЦИОНАЛЕН. Билдер штатно собирает архив без него при
        # FRK_NO_KEYS=1, и это рекомендованный продуктовый дефолт (реранк по сырым
        # ключам измеренно вредит, см. _adc_vs_rr.py). Раньше здесь стоял
        # безусловный np.load -> такой архив ронял сервер на старте, то есть
        # продукт не мог жить в рекомендованном режиме. Теперь: нет файла ->
        # работаем на ADC (24 Б/токен), реранк просто недоступен.
        kp = os.path.join(IDX_DIR, 'keys_f16.npy')
        keys = np.load(kp, mmap_mode='r') if os.path.exists(kp) else None
        cbook = [np.load(os.path.join(IDX_DIR, f'cbook_l{l}.npy')) for l in (0, 1)]
        meta = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
        _IDX = {'codes': codes, 'keys': keys, 'cbook': cbook, 'meta': meta,
                'N': codes.shape[0], 'S': meta['S'], 'load_s': time.time()-t0}
        print(f'[frk1] архив: {_IDX["N"]:,} токенов, кодобук за {_IDX["load_s"]:.1f}с'
              f'{" (без keys_f16: ADC-only)" if keys is None else ""}', flush=True)
        return _IDX

_G = None
def _to_gpu():
    """Ленивый перенос архива на CUDA. Ошибка -> None (остаёмся на CPU).
    При keys is None (архив без keys_f16) GPU-путь доступен только для ADC —
    keys/mu не переносим, `_recall_gpu` в этом случае уходит на CPU."""
    global _G
    import numpy as np
    if _G is not None: return _G if _G is not False else None
    try:
        idx = _IDX
        codes = torch.from_numpy(np.ascontiguousarray(np.asarray(idx['codes']))).to('cuda')
        cb = torch.from_numpy(np.ascontiguousarray(np.stack(idx['cbook']))).to('cuda')  # (2,S,K,sub)
        keys = None
        mu = None
        if idx['keys'] is not None:
            keys = torch.from_numpy(np.ascontiguousarray(np.asarray(idx['keys']))).to('cuda')
            mu = keys[::7919].float().mean(0)
        _G = {'codes': codes, 'keys': keys, 'cb': cb, 'mu': mu}
        return _G
    except Exception as e:
        print(f'[frk1] GPU-путь недоступен ({type(e).__name__}) -> CPU', flush=True)
        _G = False
        return None

def _recall_adc_only_gpu_or_cpu(qvec, k_return):
    """ADC-only recall (без keys_f16). GPU, если доступно, иначе numpy-CPU.
    В точности повторяет математику `recall_adc_only`: запрос НЕ нормируется
    (кодобуки обучены на ненормированных ключах), ранжир — по ADC-оценке."""
    import numpy as np
    idx = _IDX
    S = idx['S']
    sub = (idx['cbook'][0].shape[-1])   # надёжнее: sub берём из кодобука
    N = idx['N']
    t0 = time.time()
    G = _to_gpu()
    if G is not None and G.get('cb') is not None:
        try:
            q = qvec.detach().to('cuda').float()          # БЕЗ нормировки
            adc = torch.zeros(N, device='cuda')
            for l in range(2):
                for s in range(S):
                    tbl = G['cb'][l, s] @ q[s * sub:(s + 1) * sub]
                    adc += tbl[G['codes'][:, l, s].long()]
            kk = min(k_return, N)
            top = adc.topk(kk).indices
            torch.cuda.synchronize()
            return {'positions': top.cpu().tolist(),
                    'scores': [round(float(v), 4) for v in adc[top].cpu()],
                    'ms': round((time.time() - t0) * 1000, 1), 'screened': N,
                    'kcand': int(kk), 'ranking': 'adc_only', 'device': 'cuda'}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    q = qvec.detach().cpu().numpy().astype(np.float32)
    CH = 2_000_000
    adc = np.empty(N, dtype=np.float32)
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        acc = np.zeros(i1 - i0, dtype=np.float32)
        cchunk = np.asarray(idx['codes'][i0:i1])
        for l in range(2):
            for s in range(S):
                tbl = idx['cbook'][l][s] @ q[s * sub:(s + 1) * sub]
                acc += tbl[cchunk[:, l, s]]
        adc[i0:i1] = acc
    kk = min(k_return, N)
    top = np.argpartition(-adc, kk)[:kk]
    top = top[np.argsort(-adc[top])]
    return {'positions': top.tolist(), 'scores': [round(float(adc[t]), 4) for t in top],
            'ms': round((time.time() - t0) * 1000, 1), 'screened': N, 'kcand': int(kk),
            'ranking': 'adc_only', 'device': 'cpu'}


def _recall_gpu(qvec, k_return, kcand):
    import numpy as np
    idx = _IDX
    G = _to_gpu()
    # без keys_f16 реранк невозможен — этот путь только про keys, уходим на ADC
    if G is None or G.get('keys') is None:
        return _recall_adc_only_gpu_or_cpu(qvec, k_return)
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
    (раньше нормализовался — это ломало масштаб и весь ранжир).

    ВАЖНО-2: kcand здесь не имеет смысла. Никакого реранка нет, промежуточных
    кандидатов тоже: ADC-список полный, и из него сразу берётся top-k_return.
    Раньше поле 'kcand' возвращало min(k_return, N) — то есть длина ответа
    выдавалась за размер скрина. Теперь scрином честно считается N, а kcand
    явно помечается как неприменимый."""
    import numpy as np
    idx = _IDX
    q = qvec.detach().cpu().numpy().astype(np.float32)   # БЕЗ нормализации (см. docstring)
    S = idx['S']; sub = idx['cbook'][0].shape[-1]        # из кодобука, keys может не быть
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
            'ms': round(ms, 1), 'screened': N, 'kcand': None,
            'kcand_note': 'неприменимо: реранка нет, скрин = весь архив (N)',
            'ranking': 'adc_only'}

def _recall_cpu(qvec, k_return, kcand):
    """numpy-путь (RAM ~4 ГБ, ~320 мс на 1M; ~1.5-2 с на 10M).
    Если keys_f16 нет — реранка нет, честно уходим на ADC-only."""
    import numpy as np
    idx = _IDX
    if idx.get('keys') is None:
        return _recall_adc_only_gpu_or_cpu(qvec, k_return)
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

# ТЕСТ СОГЛАСОВАННОСТИ ЭНКОДЕРА (Путь C, §5 шаг 2).
# Зачем: самое дорогое наше измерение — что память, собранную одним энкодером,
# нельзя прочитать другим (recall@8 = 0.000 у 15 кандидатов). Значит, чтение
# обязано ОТКАЗАТЬ, если энкодер читателя не тот, — а не вернуть тихий ноль.
# Проверка: кодируем КОНТРОЛЬНЫЕ ОКНА корпуса, сохранённые в meta при сборке
# (`enc_probe`), и сверяем косинус с сохранёнными векторами. Это ровно тот же
# вызов энкодера, что и в recall, — значит тест меряет именно рабочий путь.
_ENC_CHECK = None
def _encoder_check(verbose=False):
    """Согласован ли энкодер читателя с энкодером, построившим архив.
    Возвращает dict. ok=None — проверить невозможно (старый архив без enc_probe)."""
    global _ENC_CHECK
    if _ENC_CHECK is not None:
        return _ENC_CHECK
    m = _arch_meta()
    res = {'ok': None, 'reason': '', 'model': m.get('encoder_model')}
    if m.get('encoder_kind') != 'external':
        res['reason'] = 'внутренний архив: энкодер = ckpt из meta, проверка не нужна'
        _ENC_CHECK = res
        return res
    probes = m.get('enc_probe') or []
    if not probes:
        res['reason'] = ('в архиве нет контрольных окон (enc_probe) — согласованность '
                         'энкодера НЕ проверена. Соберите архив билдером ≥12.09.')
        _ENC_CHECK = res
        return res
    # Векторы контрольных окон лежат в enc_probe.npy (не в JSON: meta — описание,
    # не хранилище). Порядок строк соответствует порядку probes.
    _pv_path = os.path.join(IDX_DIR, 'enc_probe.npy')
    try:
        _pv = np.load(_pv_path)
    except Exception as e:
        # НЕ ok=False. Импортированный пакет frk1x, собранный до 13.09, не нёс
        # enc_probe.npy — и такой архив становился КИРПИЧОМ: любой /recall
        # возвращал ok=False, хотя память цела и читается. «Нечем проверить»
        # и «проверка не прошла» — разные вещи; путать их = убивать перенос.
        res['ok'] = None
        res['reason'] = (f'контрольные окна есть в meta, а векторов нет '
                         f'({type(e).__name__} на enc_probe.npy) — согласованность '
                         f'энкодера НЕ проверена. Перенесите enc_probe.npy вместе '
                         f'с архивом либо соберите архив билдером ≥12.09.')
        _ENC_CHECK = res
        return res
    if int(_pv.shape[0]) != len(probes):
        res['ok'] = False
        res['reason'] = (f'рассинхрон: окон в meta {len(probes)}, векторов в '
                         f'enc_probe.npy {_pv.shape[0]}.')
        _ENC_CHECK = res
        return res
    import urllib.request as _ur
    em = os.environ.get('FRK_EMBED_MODEL') or m.get('encoder_model') or 'bge-m3'
    eu = (os.environ.get('FRK_EMBED_URL') or m.get('encoder_url')
          or 'http://localhost:11434/api/embed')
    try:
        body = json.dumps({'model': em, 'input': [p['text'] for p in probes]}).encode()
        rq = _ur.Request(eu, data=body, headers={'Content-Type': 'application/json'})
        got = json.loads(_ur.urlopen(rq, timeout=300).read().decode())['embeddings']
    except Exception as e:
        res['ok'] = False
        res['reason'] = (f'энкодер недоступен: {type(e).__name__}: {e}. Читать архив '
                         f'без энкодера {m.get("encoder_model")} НЕЛЬЗЯ — вернулся бы ноль.')
        _ENC_CHECK = res
        return res
    # cos — ДОПОЛНИТЕЛЬНАЯ диагностика, НЕ вердикт (см. ниже, почему).
    cos = []
    for _j, (p, g) in enumerate(zip(probes, got)):
        a = np.asarray(_pv[_j], dtype=np.float32)
        b = np.asarray(g, dtype=np.float32)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0 or a.shape != b.shape:
            cos.append(0.0)
            continue
        cos.append(float(np.dot(a / na, b / nb)))
    res['n'] = len(cos)
    res['min_cos'] = round(min(cos), 5)
    res['mean_cos'] = round(sum(cos) / len(cos), 5)
    res['cos'] = [round(c, 4) for c in cos]

    # ВЕРДИКТ: ЧИТАЕТСЯ ЛИ ПАМЯТЬ ЭТИМ ЭНКОДЕРОМ (а не «тот же ли вектор»).
    # ИСПРАВЛЕНО 12.09: раньше принималось min cos >= 0.999. Критерий оказался
    # НЕВЕРНЫМ и давал ЛОЖНЫЙ ОТКАЗ. Измерено: snowflake-arctic-embed2 имеет
    # min cos = 0.435 к архиву bge-m3 (порог 0.999 его отверг бы), НО при этом
    # читает архив с recall@4 = 0.227 против потолка 0.319 — 71% канала живы
    # (runs/encoder_adapter.json). Правильный вопрос: «находит ли он НУЖНОЕ ОКНО».
    # Проверка: кодируем контрольные окна энкодером читателя, ищем их в архиве
    # и смотрим, попадаем ли в ту же позицию (probes[k]['i']).
    try:
        _codes = np.load(os.path.join(IDX_DIR, 'codes.npy'), mmap_mode='r')
        _cb = [np.load(os.path.join(IDX_DIR, f'cbook_l{l}.npy')) for l in (0, 1)]
        _S = int(m.get('S') or 0)
        _D = int(m.get('D') or 0)
        _sub = _cb[0].shape[-1]
        _N = int(_codes.shape[0])
        if int(np.asarray(got[0]).shape[0]) != _D:
            res['ok'] = False
            res['reason'] = (f'энкодер вернул d={np.asarray(got[0]).shape[0]}, а архив '
                             f'требует d={_D}: прямой поиск невозможен. Нужен АДАПТЕР '
                             f'(см. exp_encoder_adapter.py) либо энкодер той же '
                             f'размерности.')
            _ENC_CHECK = res
            return res
        hits = []
        for _j, p in enumerate(probes):
            q = np.asarray(got[_j], dtype=np.float32)
            q = q / (float(np.linalg.norm(q)) + 1e-9)
            sc = np.zeros(_N, dtype=np.float32)
            CH = 2_000_000
            for i0 in range(0, _N, CH):
                i1 = min(i0 + CH, _N)
                acc = np.zeros(i1 - i0, dtype=np.float32)
                cc = np.asarray(_codes[i0:i1])
                for l in range(2):
                    for s in range(_S):
                        tbl = _cb[l][s] @ q[s * _sub:(s + 1) * _sub]
                        acc += tbl[cc[:, l, s]]
                sc[i0:i1] = acc
            top = np.argpartition(-sc, min(4, _N - 1))[:4]
            hits.append(1.0 if int(p['i']) in set(int(t) for t in top) else 0.0)
        res['probe_recall_at_4'] = round(float(sum(hits) / len(hits)), 4)
        res['probe_hits'] = [int(x) for x in hits]
        res['tolerance'] = 0.6            # 3 из 5 окон — канал живой
        res['ok'] = bool(res['probe_recall_at_4'] >= 0.6)
        if not res['ok']:
            res['reason'] = (
                f'память НЕ читается этим энкодером: контрольные окна нашлись '
                f'в {int(sum(hits))} из {len(hits)} (нужно ≥3). min cos={res["min_cos"]:.4f}. '
                f'Попробуйте энкодер {m.get("encoder_model")} или адаптер.')
    except Exception as e:
        res['ok'] = None
        res['reason'] = (f'проверка читаемости не удалась ({type(e).__name__}: {e}); '
                         f'остаётся только cos (не критерий).')
    if verbose:
        print(f'[frk1] энкодер читателя: probe_recall@4={res.get("probe_recall_at_4")} '
              f'min cos={res["min_cos"]} ok={res["ok"]}', flush=True)
    _ENC_CHECK = res
    return res
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
        # ВНЕШНИЙ ЭНКОДЕР: таблицы в архиве НЕТ и быть не может (иначе это утечка
        # готовых векторов корпуса). Роль «энкодера» играет сама модель в Ollama,
        # а query-вектор строится по ТЕКСТУ в ветке external (см. recall).
        # Здесь возвращаем заглушку нужной размерности: она нужна только для
        # проверок формы и для лекс-слоя, где таблица не используется.
        if m.get('encoder_kind') == 'external':
            _d = int(m.get('D') or 0)
            if not _d:
                raise ValueError('внешний архив без D в meta — не могу проверить '
                                 'размерность запроса')
            print(f'[frk1] внешний архив: энкодер {m.get("encoder_model")} '
                  f'(d={_d}), таблицы в памяти нет', flush=True)
            _STS_EMB = np.zeros((1, _d), dtype='float32')
            return _STS_EMB
        _ec = m.get('embed_ckpt')
        _ec_name = m.get('embed_ckpt_name') or (os.path.basename(_ec) if _ec else None)
        cand = _first_existing(
            os.environ.get('FRK_STS_CKPT'),
            _ec,                                    # АБСОЛЮТНЫЙ путь из meta (v8+)
            os.path.join(HERE, _ec) if _ec else None,
            os.path.join(HERE, _ec_name) if _ec_name else None,
            os.path.join(_REPO, _ec_name) if _ec_name else None,
            os.path.join(IDX_DIR, 'encoder_w.npy'),   # ПАКЕТ frk1x/2: энкодер внутри
            os.path.join(HERE, 'sts_prog_seed0.pt'),
            os.path.join(_REPO, 'results', 'ckpts', 'sts_prog_seed0.pt'),
            os.path.join(_REPO, 'chaotic-llm', 'results', 'ckpts', 'sts_prog_seed0.pt'),
        )
        # ЧЕСТНО: если архив требует конкретный чекпойнт, а он не найден —
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
        if str(cand).endswith('.npy'):
            # пакет frk1x/2: энкодер положен рядом как голая таблица (V,d) f16.
            # torch.load на .npy не нужен — читаем numpy напрямую.
            import numpy as _npn
            _STS_EMB = _npn.load(cand).astype('float32')
        else:
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

    ПЕРЕНОСИМОСТЬ (исправлено 11.09). Раньше путь брался из
    `meta['tokenizer_path']` — это АБСОЛЮТНЫЙ путь на машине A. После переноса
    архива на другую машину/в другой каталог он не существует, и сервер падал
    на `import final_benchmark` (ModuleNotFoundError) уже ПОСЛЕ успешной
    загрузки кодов — то есть поиск не работал, хотя память была на месте.
    Теперь порядок: `tokenizer_file` из пакета frk1x/2 (имя файла рядом с
    архивом) -> env -> абсолютный путь из meta -> fallback-конструктор.
    """
    global _BPE
    if _BPE is None:
        tf = os.environ.get('FRK_TOK')
        kind = None
        try:
            _m = json.load(open(os.path.join(IDX_DIR, 'meta.json'), encoding='utf-8'))
            kind = _m.get('tokenizer_kind')
            if not tf and kind == 'external':
                tok_name = _m.get('tokenizer') or ''
                tok_file = _m.get('tokenizer_file') or ''
                tf = _first_existing(
                    os.path.join(IDX_DIR, tok_file) if tok_file else None,  # пакет v2
                    os.path.join(IDX_DIR, tok_name) if tok_name else None,
                    os.path.join(HERE, tok_name) if tok_name else None,
                    _m.get('tokenizer_path'),                                # машина A
                    os.path.join(HERE, 'tok_v31.json'),                      # legacy
                )
        except Exception:
            _m = {}
        if tf and os.path.exists(tf):
            from tokenizers import Tokenizer as _Tkk
            _BPE = _Tkk.from_file(tf)
        elif kind == 'external':
            # Архив ТРЕБУЕТ внешний токенизатор, и он не найден. Подставить
            # дефолтный BPE-512 нельзя: ids разъедутся с архивом молча, и поиск
            # будет выдавать мусор с видом работающего. Падаем внятно.
            raise FileNotFoundError(
                f'архив {os.path.basename(IDX_DIR)} собран внешним токенизатором '
                f'{_m.get("tokenizer")!r}, но файла нет ни рядом с архивом, ни по '
                f'пути из meta. Положите файл рядом (meta.tokenizer_file) или '
                f'задайте FRK_TOK.')
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
    rerank: bool = False  # True = переранк по keys_f16 (СЫРОЙ dot). ВРЕДИТ: измерено
                          # ADC-only 0.900 против rerank 0.820 (recall@8, v8-архив,
                          # 50 запросов, _adc_vs_rr.py). Причина: сырой dot тянет
                          # векторы с большой нормой, а кодбук норму не сохраняет ->
                          # нужен косинус И ренормировка, иначе ранг-инверсия.
                          # Дефолт False = чисто 24 Б/токен, без keys_f16.

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
    # ВНЕШНИЙ архив (encoder_kind == 'external'): адрес — ОКНО текста, а не токен.
    # Тогда ids запроса не нужны вообще (и BPE Fracode в словаре чужого энкодера даёт
    # пустой список — с фильтром ниже это молча отрезало бы векторный канал).
    _EXT = (_arch_meta().get('encoder_kind') == 'external')
    if not _EXT:
        ids_q = [t for t in ids_q if t < _sts_embed_table().shape[0]]
    N = idx['N']
    results = []
    seen = set()
    # ОКОННЫЙ архив (внешний энкодер): `position` — это индекс ОКНА, а НЕ токена.
    # Символьный диапазон окна p: [p*step, p*step + chunk), step = chunk − overlap.
    # Без этой ветки текст брался через decode(ids), а ids у оконного архива =
    # середины окон → decode давал ПУСТУЮ строку, и агент получал позиции без
    # содержания (дефект найден 12.09 бенчмарком «нашёлся ли факт»).
    _m_ext = _arch_meta()
    _WIN_UNIT = (_m_ext.get('unit') == 'window')
    _WIN_CHUNK = int(_m_ext.get('encoder_chunk') or 800)
    _WIN_OVER = int(_m_ext.get('encoder_overlap') or 160)
    _WIN_STEP = max(1, _WIN_CHUNK - _WIN_OVER)
    _WIN_CTX = int(os.environ.get('FRK_WIN_CTX', 1))   # ±окон контекста
    _CTEXT = _corpus_text() if _WIN_UNIT else ''
    # Один результат наружу: для оконного архива отдаём весь захваченный span,
    # а не 600 символов (старый cap резал 200 символов из 800 и рвал факты).
    _TXT_CAP = int(os.environ.get('FRK_TXT_CAP', 2000)) if _WIN_UNIT else 600

    def snip(p, score, via):
        i0 = max(0, p - body.window); i1 = min(N, p + body.window)
        txt = ''
        if _WIN_UNIT and _CTEXT:
            # `body.window` задан в ТОКЕНАХ (24) и для оконного архива означал бы
            # 24*640 символов контекста — нелепо. Берём ±_W окна (по умолчанию 1,
            # то есть три окна ≈1.9 КБ): факт часто живёт на границе окна, и один
            # 800-символьный кусок его обрезает.
            _w = max(0, min(int(_WIN_CTX), 3))
            c0 = max(0, int(p) - _w) * _WIN_STEP
            c1 = min(len(_CTEXT), (int(p) + _w) * _WIN_STEP + _WIN_CHUNK)
            if c1 > c0:
                txt = _CTEXT[c0:c1].replace('\n', ' ')
        if not txt and _LEX_TEXT is not None and _LEX_STARTS is not None:
            # сырой текст корпуса по карте токен->символ (decodevocab=512 уродует кириллицу)
            try:
                c0 = int(_LEX_STARTS[max(0, i0)]); c1 = int(_LEX_STARTS[min(i1, len(_LEX_STARTS)-1)])
                txt = _LEX_TEXT[max(0, c0-80):c1+80].replace('\n', ' ')
            except Exception:
                txt = ''
        if not txt:
            txt = tkn.decode(np.asarray(ids_arc_load()[i0:i1].tolist())).replace('\n', ' ')
        results.append({'position': int(p), 'age_tokens_from_end': int(N - p),
                        'score': score, 'via': via, 'text': txt[:_TXT_CAP]})
    if mode in ('auto', 'lex') and body.text and not body.tokens:
        for p, s in lex_find(body.text, body.k):
            if p >= N or p in seen:
                continue
            seen.add(p)
            snip(p, s, 'lex')
    if mode in ('auto', 'vector') and (ids_q or _EXT):
        E = _sts_embed_table()
        # ВНЕШНИЙ ЭНКОДЕР (meta.encoder_kind == 'external', добавлено 12.09).
        # Ключи архива — это НЕ context-avg по таблице токенов, а векторы ОКОН от
        # готового энкодера (bge-m3). Значит и запрос обязан идти через тот же
        # энкодер по ТЕКСТУ, ровно как он шёл при сборке. Если здесь посчитать
        # context-avg по таблице, векторы попадут в другое пространство —
        # это тот самый класс ошибки, что дал нулевой recall.
        _ek = _arch_meta().get('encoder_kind')
        if _ek == 'external':
            import urllib.request as _ur
            _em = _arch_meta().get('encoder_model') or 'bge-m3'
            _eu = (os.environ.get('FRK_EMBED_URL')
                   or _arch_meta().get('encoder_url') or 'http://localhost:11434/api/embed')
            # ТЕСТ СОГЛАСОВАННОСТИ (Путь C, §5 шаг 2): не «тихий ноль», а внятный
            # отказ, если энкодер читателя не тот, которым собран архив.
            _ck = _encoder_check()
            if _ck.get('ok') is False:
                return {'ok': False, 'error': _ck.get('reason'),
                        'encoder_check': {k: v for k, v in _ck.items() if k != 'cos'},
                        'hint': 'Соберите архив тем же энкодером или включите нужный.'}
            _qtxt = body.text or ''
            if not _qtxt:
                return {'ok': False, 'error': 'внешний архив: запрос строится по '
                                              'ТЕКСТУ, поле text пустое'}
            _body = json.dumps({'model': _em, 'input': [_qtxt]}).encode()
            _rq = _ur.Request(_eu, data=_body,
                              headers={'Content-Type': 'application/json'})
            _emb = json.loads(_ur.urlopen(_rq, timeout=300).read().decode())['embeddings'][0]
            # ВАЖНО: здесь именно модульный np, а не локальный `_np` из ветки ниже
            # (внешний путь выходит раньше, чем `import numpy as _np` успевает
            # выполниться -> UnboundLocalError).
            q = np.asarray(_emb, dtype=np.float32)
            _n = float(np.linalg.norm(q))
            if _n > 0:
                q = q / _n          # сборка нормировала ключи — нормируем и запрос
            if q.shape[0] != E.shape[1]:
                return {'ok': False, 'error': f'энкодер вернул d={q.shape[0]}, '
                                              f'архив требует d={E.shape[1]} — '
                                              f'это другая модель энкодера'}
            if body.rerank:
                r = recall_vec(torch.from_numpy(q), k_return=body.k, kcand=body.kcand)
            else:
                r = recall_adc_only(torch.from_numpy(q), k_return=body.k, kcand=body.kcand)
            for p, sc in zip(r['positions'], r['scores']):
                if p in seen:
                    continue
                seen.add(p)
                snip(p, sc, 'vec')
            results = results[:body.k]
            return {'ok': True, 'query': body.text[:80], 'ms_total': r.get('ms'),
                    'screened': N, 'mode': mode, 'lex_words': _LEX_W,
                    'encoder': f'external:{_em}', 'results': results}
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



# ======================= ПЕРЕНОСИМАЯ ПАМЯТЬ (frk1x) =======================
# Единственный научный дифференциатор проекта: память — самодостаточный артефакт.
# Модель А пишет, модель Б читает; модели разные — память общая.
#
# Что именно переносимо (и почему это работает):
#   * codes.npy  — 24 Б/токен: адрес узла. Это НЕ веса модели: кодобуки обучены
#     детерминированно на корпусе, а сами коды — просто индексы. Значит, артефакт
#     не зависит от того, какой моделью он был записан.
#   * cbook_l*.npy — кодобуки (S*K*sub float32). Нужны для того, чтобы вообще
#     интерпретировать коды (без них codes.npy = шум).
#   * ids.npy — лексический слой (4 Б/токен).
#   * store.json — текстовый канал: реальные записи + рёбра + адресация по слотам.
#
# Что НЕ переносится между архитектурами и должно ехать ОТДЕЛЬНО:
#   * keys_f16.npy — 384 Б/токен, лежит в пространстве конкретного энкодера.
#   * embed_ckpt / tokenizer — их надо приложить, иначе архив не читается.
#
# Границы честности (не прятать в мелкий шрифт):
#   - Если у Б другой ТОКЕНИЗАТОР, позиции codes.npy указывают на другие куски
#     текста. Поэтому перенос работает либо при общем токенизаторе, либо
#     через ids-карту + текст в store.json.
#   - Если у Б другой ЭНКОДЕР, векторный запрос надо строить энкодером А,
#     иначе ADC-скрин уедет. Поэтому пакет несёт embed_ckpt и tokenizer_path.
#   - Переносимость по определению требует ОДИНАКОВЫХ кодобуков. Кодобуки едут
#     внутри пакета — это и есть решение.
FRKX_FORMAT = 'frk1x/1'
FRKX2_FORMAT = 'frk1x/2'

def _load_encoder_w(path):
    """Таблица эмбеддингов (V, d) из чекпойнта. Без torch в памяти надолго."""
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    W = sd['embed.weight'].detach().float().cpu().numpy().astype('float32')
    return W


def _encoder_in_package(idx_dir):
    """Путь к энкодеру, которым построен архив (из meta.json)."""
    m = _arch_meta()
    ec = m.get('embed_ckpt')
    cand = _first_existing(ec,
                           os.path.join(HERE, os.path.basename(ec)) if ec else None,
                           os.path.join(HERE, 'sts_prog_seed0.pt'),
                           os.path.join(_REPO, 'results', 'ckpts', 'sts_prog_seed0.pt'))
    return cand

def _sha256_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _export_core(idx_dir, include_keys=False, include_store=True):
    """Собрать пакет переноса В ПАМЯТЬ (не на диск). Возвращает (payload, manifest).

    payload: dict {'codes':..., 'cbook_l0':..., 'cbook_l1':..., 'ids':..., 'meta':..., 'store':...}
    """
    import numpy as np
    pay, man = {}, {'format': FRKX_FORMAT, 'files': {}}
    need = ('codes.npy', 'cbook_l0.npy', 'cbook_l1.npy', 'meta.json')
    for f in need:
        p = os.path.join(idx_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f'нет обязательного файла: {f}')
        if f.endswith('.npy'):
            pay[f[:-4]] = np.load(p)
        else:
            pay['meta'] = json.load(open(p, encoding='utf-8'))
        man['files'][f] = {'bytes': os.path.getsize(p), 'sha256': _sha256_file(p)}
    ip = os.path.join(idx_dir, 'ids.npy')
    if os.path.exists(ip):
        pay['ids'] = np.load(ip)
        man['files']['ids.npy'] = {'bytes': os.path.getsize(ip), 'sha256': _sha256_file(ip)}
    if include_keys:
        kp = os.path.join(idx_dir, 'keys_f16.npy')
        if os.path.exists(kp):
            pay['keys_f16'] = np.load(kp, mmap_mode='r')
            man['files']['keys_f16.npy'] = {'bytes': os.path.getsize(kp),
                                            'sha256': _sha256_file(kp)}
    if include_store and os.path.exists(DB):
        st = load_store()
        pay['store'] = st
        man['files']['store.json'] = {'bytes': os.path.getsize(DB),
                                      'sha256': _sha256_file(DB)}
    N = int(pay['meta'].get('N') or (pay['codes'].shape[0] if 'codes' in pay else 0))
    man['N'] = N
    man['bytes_manifest'] = sum(v['bytes'] for v in man['files'].values())
    man['mb_manifest'] = round(man['bytes_manifest'] / 2**20, 1)
    man['portable'] = {
        'codes': 'адрес узла (24 Б/токен) — архитектурно-независим; переносится',
        'cbook': 'кодобуки — обязательны для чтения кодов; переносятся',
        'ids': 'лекс-слой (4 Б/токен); переносится',
        'store': 'текстовые записи + рёбра + адресация по слотам; переносится',
        'keys_f16': ('в пространстве конкретного энкодера; МЕЖАРХИТЕКТУРНО не переносится'
                     if not include_keys else 'включён (внутриархитектурный перенос)'),
        'requires': 'энкодер (embed_ckpt) + токенизатор (tokenizer_path) из meta.json',
    }
    return pay, man

@app.get('/memory/manifest')
def memory_manifest():
    """Манифест переноса: что поедет, сколько весит, что не переносимо и почему."""
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        return {'ok': False, 'error': 'архив не построен'}
    _, man = _export_core(IDX_DIR, include_keys=False, include_store=True)
    man['ok'] = True
    man['idx_dir'] = IDX_DIR
    return man


class ExportIn(BaseModel):
    path: str = ''                 # целевой .frk1x (пусто -> ./<basename>.frk1x)
    include_keys: bool = False     # True: внутриархитектурный перенос (тяжёлый)
    include_store: bool = True
    compress: bool = True          # np.savez_compressed
    include_encoder: bool = True   # frk1x/2: вложить веса энкодера (V,d) f16
    format: str = 'v2'             # v1 = старый (без энкодера), v2 = самоописывающий


@app.post('/memory/export_v2')
def memory_export_v2(body: ExportIn):
    """САМООПИСЫВАЮЩИЙ пакет frk1x/2: память + её энкодер.

    Зачем отдельный эндпоинт: эксперимент `exp_cross_encoder.py` показал, что
    память НЕ читается чужим энкодером (recall@8 = 0.000 у всех 15 кандидатов).
    Значит «переносимая память» физически означает «память + её энкодер».
    Энкодер — это таблица (V, d), 3 МБ в float16 при V=8192, d=192: дешевле,
    чем keys_f16 (802 МБ), которые как раз выбрасываются.
    """
    import numpy as np
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        return {'ok': False, 'error': 'архив не построен'}
    t0 = time.time()
    meta = _arch_meta()
    codes = np.load(os.path.join(IDX_DIR, 'codes.npy'))
    c0 = np.load(os.path.join(IDX_DIR, 'cbook_l0.npy'))
    c1 = np.load(os.path.join(IDX_DIR, 'cbook_l1.npy'))
    npz = {'codes': codes, 'cbook_l0': c0, 'cbook_l1': c1}
    ip = os.path.join(IDX_DIR, 'ids.npy')
    if os.path.exists(ip):
        npz['ids'] = np.load(ip)
    enc_info = None
    # ВНЕШНИЙ энкодер (unit=window, запрос идёт в Ollama): таблицы энкодера в
    # архиве НЕТ и быть не должно. `_encoder_in_package` всё равно находил бы
    # sts_prog_seed0.pt (512×192) — ЧУЖУЮ таблицу внутренней модели — и пакет
    # объявлял бы себя самодостаточным при d=1024 у архива. Это тихая ложь того
    # же класса, что подмена корпуса: пакет выглядит готовым, а не работает.
    if body.include_encoder and meta.get('encoder_kind') == 'external':
        enc_info = None
    elif body.include_encoder:
        ep = _encoder_in_package(IDX_DIR)
        if ep:
            W = _load_encoder_w(ep)
            npz['encoder_w'] = W.astype(np.float16)
            enc_info = {'name': os.path.basename(ep), 'V': int(W.shape[0]),
                        'd': int(W.shape[1]), 'dtype': 'float16',
                        'note': 'таблица эмбеддингов A; без неё B не воспроизведёт запрос'}
    # КОНТРОЛЬНЫЕ ОКНА (enc_probe.npy): без них приёмник не может проверить, тем
    # ли энкодером он читает. Раньше в пакет не клались → у импортированного
    # архива meta.enc_probe был, файла не было, и _encoder_check валил ВСЕ запросы.
    pp = os.path.join(IDX_DIR, 'enc_probe.npy')
    if os.path.exists(pp):
        npz['enc_probe'] = np.load(pp)
    # ТОКЕНИЗАТОР тоже в пакет: без него позиции кодов у B не совпадут с текстом.
    tok_info = None
    try:
        tp = _first_existing(meta.get('tokenizer_path'),
                             os.path.join(HERE, str(meta.get('tokenizer') or '')))
        if tp and os.path.exists(tp):
            tb = open(tp, 'rb').read()
            npz['_tokenizer_json'] = np.frombuffer(tb, dtype=np.uint8)
            tok_info = {'name': os.path.basename(tp), 'bytes': len(tb),
                        'note': 'файл tokenizers.Tokenizer; B грузит его локально'}
    except Exception as e:
        tok_info = {'error': f'{type(e).__name__}: {e}'}
    man = {'format': FRKX2_FORMAT, 'N': int(codes.shape[0]),
           'S': int(meta['S']), 'd': int(meta['D']),
           'vocab': int(meta.get('vocab') or 0), 'encoder': enc_info,
           'source_meta': meta, 'idx_dir': IDX_DIR,
           'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'portable': {
               'codes': 'индексы (24 Б/токен): архитектурно-нейтральны',
               'cbook': 'кодобуки: обязательны, чтобы читать коды',
               'ids': 'лекс-слой (4 Б/токен)',
               'encoder_w': ('веса энкодера A включены — пакет самодостаточен'
                             if enc_info else
                             ('архив на ВНЕШНЕМ энкодере: таблицы нет по построению, '
                              'B обязан поднять тот же энкодер '
                              f'({meta.get("encoder_model") or "?"}) — см. encoder_kb'
                              if meta.get('encoder_kind') == 'external' else
                              'НЕ включены: B обязан иметь энкодер A')),
               'enc_probe': ('контрольные окна включены — B проверит, тем ли '
                             'энкодером читает' if 'enc_probe' in npz else
                             'нет контрольных окон: проверка согласованности '
                             'у B будет невозможна'),
               'keys_f16': 'не входит: производное от энкодера, воспроизводимо',
           },
           'known_limits': [
               'Память НЕ читается чужим энкодером (измерено: recall@8=0.000, 15/15).',
               'Для чужого токенизатора позиции кодов не совпадут с текстом: '
               'текст достаётся через ids.npy + store.json.',
           ]}
    if body.include_store:
        sp = os.path.join(IDX_DIR, 'store.json')
        if not os.path.exists(sp):
            sp = DB
        if os.path.exists(sp):
            man['store_json'] = json.load(open(sp, encoding='utf-8'))
    npz['_manifest_json'] = np.frombuffer(
        json.dumps(man, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    out = body.path or os.path.join(HERE, os.path.basename(IDX_DIR.rstrip('/\\')) + '.frk1x')
    if not out.lower().endswith('.npz'):
        out = out + '.npz'
    (np.savez_compressed if body.compress else np.savez)(out, **npz)
    sz = os.path.getsize(out)
    return {'ok': True, 'path': out, 'bytes': sz, 'mb': round(sz / 2**20, 1),
            'sha256': _sha256_file(out), 's': round(time.time() - t0, 1),
            'manifest': man}


@app.post('/memory/export')
def memory_export(body: ExportIn):
    """Упаковать память в единый .frk1x (npz) для переноса в агент Б.

    Почему npz: он уже есть в зависимости (numpy), стримит, хранит N массивов и
    не тянет pickle-исполнение. Формат описывается манифестом внутри архива."""
    import numpy as np
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        return {'ok': False, 'error': 'архив не построен'}
    t0 = time.time()
    pay, man = _export_core(IDX_DIR, include_keys=body.include_keys,
                            include_store=body.include_store)
    out = body.path or os.path.join(HERE, os.path.basename(IDX_DIR.rstrip('/\\')) + '.frk1x')
    # np.savez дописывает '.npz' к имени, если расширение не .npz -> приводим путь
    # к тому, что реально появится на диске (иначе getsize падает на «нет файла»).
    if not out.lower().endswith('.npz'):
        out_actual = out + '.npz'
    else:
        out_actual = out
    man['written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man['source_meta'] = pay.get('meta', {})
    npz = {k: v for k, v in pay.items() if k not in ('store', 'meta')}
    # meta и store кладём как JSON-буферы: savez на голом dict сделал бы object-array
    # и потребовал allow_pickle=True при чтении (то есть исполнение пикла из файла —
    # ровно то, чего в протоколе переноса быть не должно).
    npz['_meta_json'] = np.frombuffer(
        json.dumps(pay.get('meta', {}), ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    npz['_manifest_json'] = np.frombuffer(
        json.dumps(man, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    if 'store' in pay:
        npz['_store_json'] = np.frombuffer(
            json.dumps(pay['store'], ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    saver = np.savez_compressed if body.compress else np.savez
    saver(out_actual, **npz)
    sz = os.path.getsize(out_actual)
    return {'ok': True, 'path': out_actual, 'requested_path': out,
            'bytes': sz, 'mb': round(sz / 2**20, 1),
            'sha256': _sha256_file(out_actual), 's': round(time.time() - t0, 1),
            'manifest': man}


class ImportIn(BaseModel):
    path: str
    mode: str = 'new'          # new = положить рядом как frakod_index_imported_<ts>
    overwrite: bool = False


@app.post('/memory/import')
def memory_import(body: ImportIn):
    """Принять .frk1x от агента А и развернуть как рабочий архив.

    Ничего не перезаписываем молча: по умолчанию раскатываем НОВЫЙ каталог,
    а путь указываем в ответе. Переключение на него — через FRK_IDXD/рестарт."""
    import numpy as np
    if not os.path.exists(body.path):
        return {'ok': False, 'error': f'нет файла: {body.path}'}
    t0 = time.time()
    with np.load(body.path, allow_pickle=False) as z:
        keys = set(z.files)
        if '_manifest_json' not in keys:
            return {'ok': False, 'error': 'не frk1x: нет _manifest_json'}
        man = json.loads(bytes(z['_manifest_json']).decode('utf-8'))
        for req in ('codes', 'cbook_l0', 'cbook_l1'):
            if req not in keys:
                return {'ok': False, 'error': f'пакет неполный: нет "{req}"', 'have': sorted(keys)}
        if '_meta_json' not in keys and 'meta' not in keys:
            return {'ok': False, 'error': 'пакет неполный: нет meta (ни _meta_json, ни meta)',
                    'have': sorted(keys)}
        dst = os.path.join(
            HERE, f'frakod_index_imported_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}')
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, 'codes.npy'), z['codes'])
        np.save(os.path.join(dst, 'cbook_l0.npy'), z['cbook_l0'])
        np.save(os.path.join(dst, 'cbook_l1.npy'), z['cbook_l1'])
        if 'ids' in keys:
            np.save(os.path.join(dst, 'ids.npy'), z['ids'])
        if 'keys_f16' in keys:
            np.save(os.path.join(dst, 'keys_f16.npy'), z['keys_f16'])
        if '_meta_json' in keys:
            meta = json.loads(bytes(z['_meta_json']).decode('utf-8'))
        else:
            meta = json.loads(json.dumps(man.get('source_meta', {})))
        json.dump(meta, open(os.path.join(dst, 'meta.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        store_out = None
        if '_store_json' in keys:
            st = json.loads(bytes(z['_store_json']).decode('utf-8'))
            store_out = os.path.join(dst, 'store.json')
            json.dump(st, open(store_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return {'ok': True, 'dir': dst, 'store': store_out, 's': round(time.time() - t0, 1),
            'N': int(meta.get('N') or 0) or None,
            'manifest': man,
            'next': ('проверить: POST /memory/verify {"path": "<dir>"}; '
                     'переключиться: FRK_IDXD=<dir> и рестарт API')}


class VerifyIn(BaseModel):
    path: str
    k: int = 8
    n_probe: int = 24        # сколько запросов пробуем восстановить
    tol: int = 4             # допуск в токенах: |pos_rec - pos_true| <= tol = попадание


@app.post('/memory/verify')
def memory_verify(body: VerifyIn):
    """НЕЗАВИСИМАЯ ПРОВЕРКА ПЕРЕНОСА. Не верит метаданным пакета.

    Логика: берём ids.npy пакета, делаем из них псевдо-запросы (окно токенов),
    строим вектор запроса тем же оператором, что и ключи (context-avg окном RAD),
    гоняем ADC-only по codes.npy пакета и сравниваем найденную позицию с той,
    откуда запрос был взят. Это честный recall@k с допуском в токенах: если пакет
    собран криво или кодобуки не те, попаданий не будет — и это будет ВИДНО.

    Ничего не пишет на диск. Работает на любом каталоге, не трогая живой _IDX."""
    import numpy as np
    tkn = _archive_tok()
    E = _sts_embed_table()
    d = os.path.join(body.path, 'meta.json') if os.path.isdir(body.path) else None
    if d:
        arc = body.path
        codes = np.load(os.path.join(arc, 'codes.npy'), mmap_mode='r')
        cbooks = [np.load(os.path.join(arc, f'cbook_l{l}.npy')) for l in (0, 1)]
        meta = json.load(open(d, encoding='utf-8'))
        ip = os.path.join(arc, 'ids.npy')
        if not os.path.exists(ip):
            return {'ok': False, 'error': 'в пакете нет ids.npy — проверить нечем'}
        ids = np.load(ip, mmap_mode='r')
    else:
        return {'ok': False, 'error': f'нет каталога: {body.path}'}
    S = int(meta['S']); RAD = int(meta.get('key_context_rad') or 4)
    N = int(codes.shape[0]); sub = cbooks[0].shape[-1]
    # ВНЕШНИЙ АРХИВ (Путь C, §5 шаг 2): проверка «пакет читаем» обязана включать
    # проверку энкодера. Иначе она честно скажет «recall низкий» и не скажет
    # ПОЧЕМУ — а причина почти всегда одна: чужой энкодер.
    if meta.get('encoder_kind') == 'external':
        ck = _encoder_check(verbose=True)
        if ck.get('ok') is False:
            return {'ok': False, 'dir': arc, 'N': N,
                    'encoder_check': {k: v for k, v in ck.items() if k != 'cos'},
                    'verdict': 'ПАКЕТ НЕ ЧИТАЕТСЯ: энкодер читателя не тот, '
                               'которым собран архив. Это отказ, а не низкий recall.',
                    'reason': ck.get('reason')}
        # Согласованный энкодер: делаем ту же независимую пробу, но запрос идём
        # ТЕКСТОМ через энкодер (векторы контрольных окон воспроизводимы).
        probes = meta.get('enc_probe') or []
        _pv_path = os.path.join(arc, 'enc_probe.npy')
        try:
            _pv = np.load(_pv_path)
        except Exception:
            return {'ok': False, 'dir': arc,
                    'error': 'enc_probe.npy отсутствует — архив собран неполно'}
        hits_at_k, exact_hits, rows = 0, 0, []
        for _j, p in enumerate(probes):
            q = np.asarray(_pv[_j], dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            adc = np.zeros(N, dtype=np.float32)
            CH = 2_000_000
            for i0 in range(0, N, CH):
                i1 = min(i0 + CH, N)
                acc = np.zeros(i1 - i0, dtype=np.float32)
                cc = np.asarray(codes[i0:i1])
                for l in range(2):
                    for s in range(S):
                        tbl = cbooks[l][s] @ q[s * sub:(s + 1) * sub]
                        acc += tbl[cc[:, l, s]]
                adc[i0:i1] = acc
            kk = min(body.k, N)
            top = np.argpartition(-adc, kk)[:kk]
            best = min((abs(int(x) - int(p['i'])), int(x)) for x in top)
            hit = best[0] <= body.tol
            hits_at_k += int(hit)
            exact_hits += int(int(p['i']) in set(int(x) for x in top))
            rows.append({'probe_pos': int(p['i']), 'best_pos': best[1],
                         'delta': best[0], 'hit': hit})
        n = len(rows)
        return {'ok': True, 'dir': arc, 'N': N, 'k': body.k, 'tol': body.tol,
                'unit': 'window', 'n_probe': n,
                'encoder_check': {k: v for k, v in ck.items() if k != 'cos'},
                'recall_at_k_tol': round(hits_at_k / max(n, 1), 4),
                'exact_in_topk': round(exact_hits / max(n, 1), 4),
                'hit_rate': round(hits_at_k / max(n, 1), 4),
                'mean_abs_delta': round(float(np.mean([r['delta'] for r in rows])) if rows else -1, 2),
                'verdict': ('пакет читаем и память восстанавливается'
                            if n and hits_at_k / max(n, 1) >= 0.8 else
                            'ВНИМАНИЕ: recall низкий — пакет или кодобуки не те'),
                'rows': rows[:12],
                'note': ('Внешний архив: единица — ОКНО, tol в окнах. Энкодер сверен '
                         'с контрольными окнами сборки (min cos выше) — это и есть '
                         'проверка «энкодер едет вместе с памятью».')}
    rng = np.random.RandomState(20260911)      # фиксированный сид — воспроизводимо
    lo, hi = RAD + 8, N - RAD - 8
    hits_at_k, exact_hits, rows = 0, 0, []
    n_probe = min(body.n_probe, hi - lo)
    probe_pos = np.sort(rng.choice(np.arange(lo, hi), size=n_probe, replace=False))
    for p in probe_pos:
        ctx = np.asarray(ids[max(0, p - RAD):min(N, p + RAD + 1)], dtype=np.int64)
        ctx = ctx[ctx < E.shape[0]]
        if ctx.size == 0:
            continue
        q = E[ctx].astype(np.float32).mean(0)          # тот же оператор, что у ключей
        adc = np.zeros(N, dtype=np.float32)
        CH = 2_000_000
        for i0 in range(0, N, CH):
            i1 = min(i0 + CH, N)
            acc = np.zeros(i1 - i0, dtype=np.float32)
            cc = np.asarray(codes[i0:i1])
            for l in range(2):
                for s in range(S):
                    tbl = cbooks[l][s] @ q[s * sub:(s + 1) * sub]
                    acc += tbl[cc[:, l, s]]
            adc[i0:i1] = acc
        kk = min(body.k, N)
        top = np.argpartition(-adc, kk)[:kk]
        pos = set(int(x) for x in top)
        best = min((abs(int(x) - int(p)), int(x)) for x in top)
        hit = best[0] <= body.tol
        hits_at_k += int(hit)
        exact_hits += int(int(p) in pos)
        rows.append({'probe_pos': int(p), 'best_pos': best[1], 'delta': best[0], 'hit': hit})
    n = len(rows)
    return {'ok': True, 'dir': arc, 'N': N, 'k': body.k, 'tol': body.tol,
            'n_probe': n, 'seed': 20260911,
            'recall_at_k_tol': round(hits_at_k / max(n, 1), 4),
            'exact_in_topk': round(exact_hits / max(n, 1), 4),
            'hit_rate': round(hits_at_k / max(n, 1), 4),
            'mean_abs_delta': round(float(np.mean([r['delta'] for r in rows])) if rows else -1, 2),
            'verdict': ('пакет читаем и память восстанавливается' if n and hits_at_k / max(n, 1) >= 0.8
                        else 'ВНИМАНИЕ: recall низкий — пакет или кодобуки не те'),
            'rows': rows[:12],
            'note': ('Проверка независимая: позиции берутся из ids.npy пакета, а не из манифеста. '
                     'Допуск tol в токенах, потому что ключи = context-avg окном RAD, '
                     'точечная позиция размыта на ~RAD.')}


class ImportV2In(BaseModel):
    path: str
    name: str = ''              # имя каталога (пусто -> frakod_index_imported_<ts>)


@app.post('/memory/import_v2')
def memory_import_v2(body: ImportV2In):
    """Развернуть frk1x/2 (с энкодером) в самостоятельный архив.

    Кладём и `encoder_w.npy` — тогда `_sts_embed_table` подхватит его по
    meta.embed_ckpt, и B сможет работать БЕЗ чекпойнта A на диске. Это и делает
    пакет по-настоящему самодостаточным."""
    import numpy as np
    if not os.path.exists(body.path):
        return {'ok': False, 'error': f'нет файла: {body.path}'}
    t0 = time.time()
    with np.load(body.path, allow_pickle=False) as z:
        have = set(z.files)
        if '_manifest_json' not in have:
            return {'ok': False, 'error': 'не frk1x: нет _manifest_json'}
        man = json.loads(bytes(z['_manifest_json']).decode('utf-8'))
        for req in ('codes', 'cbook_l0', 'cbook_l1'):
            if req not in have:
                return {'ok': False, 'error': f'пакет неполный: нет "{req}"',
                        'have': sorted(have)}
        tag = body.name or f'frakod_index_imported_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}'
        dst = os.path.join(HERE, tag)
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, 'codes.npy'), z['codes'])
        np.save(os.path.join(dst, 'cbook_l0.npy'), z['cbook_l0'])
        np.save(os.path.join(dst, 'cbook_l1.npy'), z['cbook_l1'])
        if 'ids' in have:
            np.save(os.path.join(dst, 'ids.npy'), z['ids'])
        if 'keys_f16' in have:
            np.save(os.path.join(dst, 'keys_f16.npy'), z['keys_f16'])
        # КОНТРОЛЬНЫЕ ОКНА: без них у B meta.enc_probe есть, а векторов нет.
        if 'enc_probe' in have:
            np.save(os.path.join(dst, 'enc_probe.npy'), z['enc_probe'])
        # ЭНКОДЕР: сохраняем рядом и переключаем meta.embed_ckpt на локальный файл
        meta = (json.loads(bytes(z['_meta_json']).decode('utf-8'))
                if '_meta_json' in have else dict(man.get('source_meta', {})))
        if 'encoder_w' in have:
            W = z['encoder_w'].astype(np.float32)
            ep = os.path.join(dst, 'encoder_w.npy')
            np.save(ep, W)
            enc = man.get('encoder') or {}
            meta['embed_ckpt'] = ep
            meta['embed_ckpt_name'] = (enc.get('name') or 'encoder_w.npy')
            meta['encoder_bundled'] = True
        # ТОКЕНИЗАТОР: кладём рядом и прописываем ОТНОСИТЕЛЬНОЕ имя. Без этого
        # поиск у B падал на `import final_benchmark`, потому что meta хранил
        # абсолютный путь машины A (см. _archive_tok).
        if '_tokenizer_json' in have:
            tp = os.path.join(dst, 'tokenizer.json')
            with open(tp, 'wb') as fh:
                fh.write(bytes(z['_tokenizer_json']))
            meta['tokenizer_file'] = 'tokenizer.json'
            meta['tokenizer_path'] = tp
            meta['tokenizer_bundled'] = True
        meta['imported_from'] = os.path.basename(body.path)
        meta['imported_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        json.dump(meta, open(os.path.join(dst, 'meta.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        store_out = None
        st = None
        if '_store_json' in have:
            st = json.loads(bytes(z['_store_json']).decode('utf-8'))
        elif man.get('store_json'):
            st = man['store_json']
        if st is not None:
            store_out = os.path.join(dst, 'store.json')
            json.dump(st, open(store_out, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
    return {'ok': True, 'dir': dst, 'store': store_out, 'encoder': meta.get('embed_ckpt'),
            'N': int(meta.get('N') or 0) or None, 's': round(time.time() - t0, 1),
            'self_contained': bool(meta.get('encoder_bundled')),
            'next': 'FRK_IDXD=<dir> + рестарт, затем POST /memory/verify {"path": "<dir>"}'}


if __name__ == '__main__':
    # Раньше точка входа отсутствовала: `python frakod_api.py` просто импортировал
    # модуль и МОЛЧА завершался с пустым логом. Теперь прямой запуск работает;
    # эквивалентен `python -m uvicorn frakod_api:app`.
    import uvicorn
    if not os.path.exists(os.path.join(IDX_DIR, 'codes.npy')):
        raise SystemExit(f'в каталоге {IDX_DIR} нет codes.npy — задайте FRK_IDXD '
                         f'на собранный архив (например frakod_index_10m)')
    print(f'Fracod API: архив = {IDX_DIR}')
    uvicorn.run(app, host=os.environ.get('FRK_HOST', '127.0.0.1'),
                port=int(os.environ.get('FRK_PORT', '8000')))
