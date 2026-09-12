# -*- coding: utf-8 -*-
"""frk1x v2: самодостаточный пакет памяти + перенос МЕЖДУ РАЗНЫМИ энкодерами.

Отвечает на главный незакрытый вопрос проекта: память — самодостаточный
артефакт, или она тавтологически привязана к своему энкодеру?

Два уровня переноса:
  L-same  (frk1x/1 -> v2 с encoder внутри): A пишет E1, B читает E1.
          Тавтология: проверяет упаковку, не переносимость.
  L-cross (главное): A пишет кодами в кодобуке E1, B строит ЗАПРОС своим
          энкодером E2. Кодобуки едут в пакете (они не веса, а таблица),
          поэтому чтение кодов не требует E1 вовсе. Вопрос только в том,
          насколько пространство E2 позволяет попасть в кодобук E1.

Почему это вообще может работать: ADC-скрин сравнивает ЗАПРОС с КОДОБУКОМ, а не
запрос с ключами. Кодобук — 256 центроидов на подпространство, то есть очень
грубая сетка. Разные обученные энкодеры часто согласованы в грубом масштабе
(оба ловят частотность и локальный контекст), а расходятся в тонком. Если гипотеза
верна, cross-энкодерный recall должен быть заметно выше случайного, но ниже
same-энкодерного. Это И ЕСТЬ измеряемая величина.
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(HERE, '..'))
CKPTS = os.path.join(_REPO, 'results', 'ckpts')
FRKX2 = 'frk1x/2'


# ---------------------------------------------------------------- энкодеры

def load_embed(path):
    """Таблица эмбеддингов (V, d) из чекпойнта. Возвращает (np.float32, meta)."""
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    if 'embed.weight' not in sd:
        raise KeyError(f'{os.path.basename(path)}: нет embed.weight '
                       f'(есть {list(sd)[:6]})')
    W = sd['embed.weight'].detach().float().cpu().numpy()
    return W.astype(np.float32)


def discover_encoders(repo=None):
    """Все чекпойнты, пригодные как энкодер, с их d. Для A/B-перебора."""
    repo = repo or _REPO
    out = []
    for base in (os.path.join(repo, 'results', 'ckpts'),
                 os.path.join(repo, 'phase01', 'exp_vq')):
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if not fn.endswith('.pt'):
                continue
            p = os.path.join(base, fn)
            try:
                import torch
                sd = torch.load(p, map_location='cpu', weights_only=False)
                sd = sd.get('model', sd) if isinstance(sd, dict) else sd
                if 'embed.weight' not in sd:
                    continue
                V, d = map(int, sd['embed.weight'].shape)
                out.append({'name': fn, 'path': p, 'V': V, 'd': d,
                            'family': 'sts' if fn.startswith('sts') else
                                      ('tf' if fn.startswith('transformer') else 'other'),
                            'mb': round(os.path.getsize(p) / 2**20, 1)})
            except Exception:
                continue
    return out


# ---------------------------------------------------------------- ключи E1

def context_avg_keys(ids, E, rad=4):
    """Ключ архива = context-avg эмбеддингов окном RAD (как frakod_index_build).
    Это РОВНО тот оператор, которым строились ключи v8 — иначе архивы несравнимы."""
    ids = np.asarray(ids, dtype=np.int64)
    N = len(ids)
    pad = np.concatenate([np.full(rad, ids[0], np.int64), ids,
                          np.full(rad, ids[-1], np.int64)])
    k = 2 * rad + 1
    cum = np.cumsum(np.vstack([np.zeros((1, E.shape[1]), np.float32),
                               E[pad].astype(np.float32)]), 0)
    return (cum[k:] - cum[:-k]) / k, k


def query_vec_e2(ids_ctx, E2, rad=4):
    """Запрос, построенный энкодером E2 тем же оператором context-avg.
    Именно это делает агент B, если у него другой энкодер."""
    ids_ctx = np.asarray(ids_ctx, dtype=np.int64)
    ids_ctx = ids_ctx[ids_ctx < E2.shape[0]]
    if ids_ctx.size == 0:
        return None
    pad = np.concatenate([np.full(rad, ids_ctx[0], np.int64), ids_ctx,
                          np.full(rad, ids_ctx[-1], np.int64)])
    k = 2 * rad + 1
    cum = np.cumsum(np.vstack([np.zeros((1, E2.shape[1]), np.float32),
                               E2[pad].astype(np.float32)]), 0)
    return ((cum[k:] - cum[:-k]) / k).mean(0).astype(np.float32)


def query_vec_e1(ids_ctx, E1, rad=4):
    """Тот же оператор, но энкодером A — тавтологический потолок.
    Математически = query_vec_e2, имя разделено ради читаемости вердикта."""
    return query_vec_e2(ids_ctx, E1, rad)


# ---------------------------------------------------------------- ADC

def adc_scores(codes, cbook, q, chunk=2_000_000):
    """ADC-оценка всех N кодов против одного запроса. Кодобук из ПАКЕТА —
    поэтому энкодер, которым записан архив, здесь не нужен вообще."""
    N = codes.shape[0]
    S = len(cbook[0])
    sub = cbook[0].shape[-1]
    assert q.shape[0] == sub * S, f'запрос d={q.shape[0]}, кодобук ждёт {sub*S}'
    out = np.empty(N, dtype=np.float32)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        acc = np.zeros(i1 - i0, dtype=np.float32)
        cc = np.asarray(codes[i0:i1])
        for l in range(2):
            for s in range(S):
                # ВАЖНО: скобки. `@` приоритетнее индексации, и
                # `cbook[l][s] @ q[...][idx]` numpy читает как
                # `cbook[l][s] @ q[...]` , а `[idx]` применяет к РЕЗУЛЬТАТУ.
                tbl = cbook[l][s] @ q[s * sub:(s + 1) * sub]     # (K,)
                acc += tbl[cc[:, l, s].astype(np.int64)]
        out[i0:i1] = acc
    return out


def make_codes(keys, cbook_l, S, K=256):
    """Сжать точные ключи кодобуком (ближайший центроид по каждому подвектору).
    Нужно, чтобы сделать архив под ДРУГОЙ энкодер, не пересобирая билдером."""
    N, d = keys.shape
    sub = d // S
    codes = np.zeros((N, 2, S), dtype=np.uint8)
    R = keys.astype(np.float32).copy()
    for l in range(2):
        cb = cbook_l[l]                       # (S, K, sub)
        for s in range(S):
            seg = R[:, s * sub:(s + 1) * sub]                     # (N, sub)
            cbs = cb[s]                                            # (K, sub)
            # ближайший центроид: ||seg||^2 - 2 seg@cb^T + ||cb||^2
            d2 = (seg * seg).sum(1)[:, None] - 2.0 * (seg @ cbs.T) \
                + (cbs * cbs).sum(1)[None, :]
            a = d2.argmin(1).astype(np.uint8)
            codes[:, l, s] = a
            R[:, s * sub:(s + 1) * sub] = seg - cbs[a]
    return codes


# ---------------------------------------------------------------- упаковка

def export_v2(idx_dir, out_path, include_encoder=True, embed_path=None,
              include_store=True, compress=True):
    """Самодостаточный пакет: коды + кодобуки + ids + meta + ЭНКОДЕР (веса)."""
    import numpy as np
    t0 = time.time()
    meta = json.load(open(os.path.join(idx_dir, 'meta.json'), encoding='utf-8'))
    codes = np.load(os.path.join(idx_dir, 'codes.npy'))
    c0 = np.load(os.path.join(idx_dir, 'cbook_l0.npy'))
    c1 = np.load(os.path.join(idx_dir, 'cbook_l1.npy'))
    npz = {'codes': codes, 'cbook_l0': c0, 'cbook_l1': c1}
    ip = os.path.join(idx_dir, 'ids.npy')
    if os.path.exists(ip):
        npz['ids'] = np.load(ip)
    enc_path = embed_path or meta.get('embed_ckpt')
    enc_info = None
    if include_encoder and enc_path and os.path.exists(enc_path):
        W = load_embed(enc_path)
        npz['encoder_w'] = W.astype(np.float16)          # (V,d) — 17 МБ вместо 34
        enc_info = {'name': os.path.basename(enc_path), 'V': int(W.shape[0]),
                    'd': int(W.shape[1]), 'dtype': 'float16',
                    'note': 'таблица эмбеддингов; позволяет B воспроизвести запрос A'}
    man = {'format': FRKX2, 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                         time.gmtime()),
           'N': int(codes.shape[0]), 'S': int(meta['S']),
           'd': int(meta['D']), 'vocab': int(meta.get('vocab') or 0),
           'encoder': enc_info, 'source_meta': meta,
           'portable': {
               'codes': 'индексы (24 Б/токен) — архитектурно-нейтральны',
               'cbook': 'кодобуки: нужны, чтобы читать коды; едут в пакете',
               'encoder_w': ('веса энкодера A — позволяют B играть роль A' if enc_info
                             else 'НЕ включён: B обязан иметь энкодер A'),
               'keys_f16': 'не входит: производное от энкодера, воспроизводимо',
           }}
    if include_store:
        sp = os.path.join(idx_dir, 'store.json')
        if not os.path.exists(sp):
            sp = os.path.join(HERE, 'frakod_store.json')
        if os.path.exists(sp):
            man['store_json'] = json.load(open(sp, encoding='utf-8'))
    npz['_manifest_json'] = np.frombuffer(
        json.dumps(man, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
    if not out_path.lower().endswith('.npz'):
        out_path = out_path + '.npz'
    (np.savez_compressed if compress else np.savez)(out_path, **npz)
    return {'ok': True, 'path': out_path, 'mb': round(os.path.getsize(out_path) / 2**20, 1),
            's': round(time.time() - t0, 1), 'manifest': man}


def read_package(path):
    """Прочитать пакет без разворачивания на диск."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if '_manifest_json' not in keys:
            raise ValueError('не frk1x: нет _manifest_json')
        man = json.loads(bytes(z['_manifest_json']).decode('utf-8'))
        d = {'manifest': man, 'files': sorted(keys)}
        for k in ('codes', 'cbook_l0', 'cbook_l1', 'ids'):
            if k in keys:
                d[k] = z[k]
        if 'encoder_w' in keys:
            d['encoder_w'] = z['encoder_w'].astype(np.float32)
    return d
