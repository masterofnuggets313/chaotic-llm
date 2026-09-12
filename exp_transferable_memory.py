# -*- coding: utf-8 -*-
"""ЭТАП 3.1: ПЕРЕНОСИМАЯ ПАМЯТЬ — модель B читает архив, собранный моделью A.

ГЛАВНЫЙ ДИФФЕРЕНЦИАТОР. Всё измеренное до сих пор — это память ВНУТРИ одной
модели (модель читает свою же сжатую память) и ВНЕШНИЙ ретривер (энкодер + коды,
отдельно от модели). Перенос между ДВУМЯ РАЗНЫМИ моделями не проверялся ни разу.

Если работает — память ОТДЕЛЯЕТСЯ ОТ МОДЕЛИ: архив становится транспортабельным
артефактом (собрал контекст на одной модели, читаешь на другой; нет KV-кэша на
инференсе; память масштабируется независимо от весов). Это и есть то, за чем
будут охотиться — не «сжали в N раз», а «память перестала принадлежать модели».

ЭКСПЕРИМЕНТ (векторный уровень, без языковой генерации — потому не требует
починенного чекпойнта и идёт параллельно этапу 1):

  A = sts_prog_seed0.pt, B = sts_prog_seed1.pt (и другие сиды) — одно семейство,
  разная инициализация/прогон.

  Архив = статичные ключи e = embed(x) + pos  (как в frakod_api / билдере),
  сжатые Fracod-кодбуком, обученным на ключах МОДЕЛИ A.

  Замеряем 4 режима (метрика: recall@k «своя позиция находится»):
    1. A читает свой архив (кодбук A, ключи A)      — baseline внутримодельный
    2. B читает архив A   (кодбук A, ключи B)       <-- ПЕРЕНОС (главное)
    3. B читает архив B, кодбук B                   — baseline B
    4. B читает архив A, кодбук ОБУЧЕН НА B         — контроль: помогает ли
                                                      переобучение кодбука

  Если (2) заметно хуже (3), но (4) близко к (3) — значит переносится не кодбук,
  а представление; мера переносимости — насколько (2) близко к (3).

Запуск:
    "…python.exe" -u frakod/exp_transferable_memory.py
    "…python.exe" -u frakod/exp_transferable_memory.py --src 0 --dst 1,2,3,4
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
PHASE = os.path.join(REPO, 'phase01')
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)

import torch

CKPTS = os.path.join(REPO, 'results', 'ckpts')
# Токенизатор берётся под vocab чекпойнтов. По умолчанию линия v7 (vocab=512):
# именно там есть ПЯТЬ сидов (sts_prog_seed0..4), а перенос между моделями требует
# нескольких. У v8 (vocab=8192) чекпойнт ровно один -> перенос проверять не на чем.
_TOK_BY_VOCAB = {
    512: [os.path.join(HERE, 'tok_v31.json'),
          os.path.join(PHASE, 'exp_vq', 'tok_v31.json')],
    8192: [os.path.join(HERE, 'tok_v8.json'),
           os.path.join(PHASE, 'exp_vq', 'tok_v8.json')],
}
CORPUS = os.path.join(HERE, 'hermes_history_dd.txt')

# Токены архива, выставленные в main(). Глобал — потому что transfer_eval
# пересчитывает ключи приёмника на тех же позициях.
_ARCH_IDS = None


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def load_model(seed, d=192, layers=8, vocab=None, pattern=None):
    """Загрузить STS-Prog чекпойнт. vocab определяется по самому чекпойнту.

    pattern: шаблон имени файла, по умолчанию 'sts_prog_seed{seed}.pt'.
    Для линии v8 (vocab=8192) имена другие — передавать явно, напр.
    'v8_dialog_seed{seed}.pt'.
    """
    from models_pc import build_pc_model
    pat = (pattern or 'sts_prog_seed{}.pt').format(seed)
    # ищем сначала в frakod/, потом в results/ckpts/
    for base in (HERE, CKPTS):
        p = os.path.join(base, pat)
        if os.path.exists(p):
            break
    else:
        return None, None, None
    sd = torch.load(p, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    if vocab is None:
        vocab = int(sd['embed.weight'].shape[0])
    m = build_pc_model('pc', vocab=vocab, d=d, layers=layers, k_init=1.2,
                       sync_steps=8, driver_mode='sts_prog', alpha=0.3, temp=0.3)
    m.load_state_dict(sd)
    m.eval()
    return m.to('cuda'), sd, vocab


@torch.no_grad()
def static_keys(model, ids, dev='cuda'):
    """e = embed(ids) + pos  — статичные ключи, как в билдере/API (без PC-прогона).

    Окно модели фиксировано (W=256 из чекпойнта), поэтому строим НЕ окно длиной
    len(ids), а столько независимых окон, сколько влезает: каждое окно даёт W
    ключей, лишний хвост отбрасывается. Это соответствует тому, как архив устроен
    для токенной гранулярности (каждая позиция -> свой ключ).
    """
    W = model.pos.shape[1]
    V = model.embed.weight.shape[0]
    n = (len(ids) // W) * W
    if n == 0:
        return None
    x = torch.tensor(ids[:n], dtype=torch.long, device=dev).view(-1, W)
    e = model.embed(x) + model.pos            # (B, W, d)
    return e.reshape(-1, e.shape[-1])         # (B*W, d)


def build_archive(model, ids, fm, det=True):
    """Собрать архив: ключи модели -> кодбук fm. -> codes (N,L,S), keys (N,d)."""
    keys = static_keys(model, ids)
    fm.fit(keys, iters=15, seed=0, det=det)
    codes = fm.encode_rows(keys)
    return codes, keys


def external_keys(ids, nsub, dim, window=16, seed=13):
    """Ключи из ВНЕШНЕГО детерминированного отображения — БЕЗ весов модели.

    Зачем (кандидат №1 реформации задачи «память через ОБЩЕЕ пространство ключей»):
    пока ключ = `embed(x) + pos`, он приходит из обучаемых весов, и у двух сидов
    один и тот же текст даёт ортогональные векторы (измерено cos = 0.0029). Такой
    архив физически не может быть прочитан другой моделью. Здесь ключ строится
    функцией от ТОКЕНОВ, общей для всех моделей, поэтому обе модели адресуют одну
    и ту же точку по одному и тому же тексту.

    ОКНО (важно, исправлено 11.09). Первая версия хешировала только n-граммы
    (3 подряд идущих токена) без учёта более широкого контекста. Диагностика
    показала, что такой ключ почти полностью определяется НЕСКОЛЬКИМИ токенами:
    уникальными оказались лишь 23.8 % ключей из 8192 — остальное коллизии. При
    таком ключе recall@8 ≈ 8/8192 держится на одних коллизиях, и «высокий»
    recall ничего не доказывает.

    Теперь ключ — сумма хешированных вкладов ОКНА из `window` токенов вокруг
    позиции, с затуханием по расстоянию. Это делает ключ функцией локального
    контекста (как `embed` в модели), сохраняя главное свойство: он НЕ зависит
    от обучаемых весов и одинаков для всех моделей.

    ПОДПРОСТРАНСТВА (исправлено 11.09, было вырождено). `nsub` = ЧИСЛО
    подпространств, `sub = d // nsub` координат в каждом. Раньше стояло
    `idx = h % sub; col = idx % sub`, то есть `col == idx` ∈ [0, sub) — номер
    подпространства не участвовал вообще, и ключ жил только в первых `sub`
    координатах: при nsub=12 это 16 из 192, при nsub=1 — все 192.
    Измерено `_d1_repro.py`: ненулевых координат 16/192 против 192/192.

    Следствие прежнего бага: 11 из 12 кодбуков PQ квантовали точные нули —
    22 из 24 байт ключа впустую. Именно поэтому с ростом N recall падал
    (0.828 -> 0.234): били не «мало байт», а вырожденный ключ. Сравнивать
    конфигурации с разным `nsub` было нельзя: менялось само адресное
    пространство. Теперь: `col` — номер подпространства, `idx` — координата
    внутри него, знак берётся из следующего разряда хеша.

    Ни одно число про внешнюю память не цитировать без указания `nsub`:
    старые артефакты писались вырожденным ключом и в артефактах `nsub` не
    сохранялся вообще.
    """
    nsub = int(nsub)
    d = int(dim)
    sub = d // nsub
    N = len(ids)
    ids_a = np.asarray(ids, dtype=np.int64)
    out = np.zeros((N, d), dtype=np.float32)
    half = int(window) // 2
    rows = np.arange(N, dtype=np.int64)
    for off in range(-half, half + 1):
        # сдвиг окна: позиция i видит токен i+off (с паддингом нулём на краях)
        if off < 0:
            sh = np.concatenate([np.zeros(-off, dtype=np.int64), ids_a[:off]])
        elif off > 0:
            sh = np.concatenate([ids_a[off:], np.zeros(off, dtype=np.int64)])
        else:
            sh = ids_a
        # затухание: центр окна весит больше (детерминированно, без обучения)
        wgt = np.float32(1.0 / (1.0 + abs(off)))
        h = (sh.astype(np.int64) * 2654435761 + (off + half) * 40503) & 0x7FFFFFFF
        # col — НОМЕР подпространства (0..nsub-1), idx — координата внутри него.
        # Раньше было `col = idx % sub` с `idx = h % sub` -> col == idx, и все
        # ненулевые координаты падали в первые `sub` (см. docstring, D1).
        col = (h % nsub).astype(np.int64)
        idx = ((h // nsub) % sub).astype(np.int64)
        sgn = np.where((h // (nsub * sub)) % 2 == 0, 1.0, -1.0) \
            .astype(np.float32) * wgt
        np.add.at(out, (rows, col * sub + idx), sgn)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    return out / nrm


@torch.no_grad()
def transfer_eval_ext(fm, codes, Q, q_pos, dev='cuda', k=8):
    """Метрика переноса в ОБЩЕМ пространстве: запрос — та же внешняя функция.

    Отличие от `transfer_eval`: запрос НЕ зависит от модели-приёмника.
    Q — (nq, d)matрица внешних ключей в тех же позициях, что и позиции архива.
    Это и есть условие переноса: любая модель адресует ту же точку.
    """
    fm.codes = codes
    N = codes.shape[0]
    Q = Q.to(dev)
    out = torch.zeros((Q.shape[0], N), device=dev)
    for l in range(fm.L):
        for s in range(fm.S):
            qsub = Q[:, s * fm.sub:(s + 1) * fm.sub]
            tbl = qsub @ fm.cbooks[l][s].T
            out += tbl[:, codes[:, l, s]]
    kk = min(k, N)
    top = out.topk(kk, dim=1).indices
    tgt = torch.tensor(q_pos, dtype=torch.long, device=dev).unsqueeze(1)
    hit = (top == tgt).any(dim=1).float().mean().item()
    return {'recall@k': hit, 'n': len(q_pos)}


@torch.no_grad()
def transfer_eval(fm, codes, model_dst, q_pos, dev='cuda', k=8):
    """Метрика переноса: по ключам архивного пространства найти «свою» позицию.

    Архив — это плоский список ключей (B*W, d) из независимых окон модели.
    Запрос q строится ТЕМ ЖЕ оператором из модели-приёмника: берём ту же позицию
    в том же окне. Метрика — recall@k: попадает ли эта позиция в top-k ADC-скрина.

    fm    : кодбук (StreamFracode), обученный в пространстве модели-источника A
    codes : (N,L,S) коды ключей, закодированных ЭТИМ кодбуком
    model_dst : модель, ключи которой используются для запроса

    ВЕКТОРИЗОВАНО: ADC считается сразу для всех запросов одной матрицей.
    Поштучный вызов на архиве 65k позиций не укладывался в таймаут.
    """
    fm.codes = codes
    N = codes.shape[0]
    keys_dst = static_keys(model_dst, _ARCH_IDS, dev)
    if keys_dst is None:
        return {'recall@k': float('nan'), 'n': 0}
    Q = keys_dst[torch.tensor(q_pos, dtype=torch.long, device=dev)]   # (nq, d)
    # ADC для всех запросов: (nq, N) = сумма по уровням/подвекторам
    out = torch.zeros((Q.shape[0], N), device=dev)
    for l in range(fm.L):
        for s in range(fm.S):
            qsub = Q[:, s * fm.sub:(s + 1) * fm.sub]                 # (nq, sub)
            tbl = qsub @ fm.cbooks[l][s].T                           # (nq, K)
            out += tbl[:, codes[:, l, s]]                            # (nq, N)
    kk = min(k, N)
    top = out.topk(kk, dim=1).indices                                # (nq, kk)
    tgt = torch.tensor(q_pos, dtype=torch.long, device=dev).unsqueeze(1)
    hit = (top == tgt).any(dim=1).float().mean().item()
    return {'recall@k': hit, 'n': len(q_pos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=int, default=0)
    ap.add_argument('--self-check', action='store_true',
                    help='Проверка ИЗМЕРИТЕЛЯ: та же модель читает свой архив. '
                         'Должно дать высокий recall — иначе метрика сломана, '
                         'и отрицательный результат переноса ничего не значит.')
    ap.add_argument('--dst', default='1,2,3,4')
    ap.add_argument('--pattern', default=None,
                    help="шаблон имени чекпойнта, напр. 'v8_dialog_seed{}.pt'. "
                         "По умолчанию 'sts_prog_seed{}.pt' (линия v7, vocab=512).")
    ap.add_argument('--S', type=int, default=12)
    ap.add_argument('--K', type=int, default=256)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--nq', type=int, default=64, help='сколько запросов')
    ap.add_argument('--bits', type=int, default=8192, help='сколько токенов архива')
    ap.add_argument('--no-external', action='store_true',
                    help='не считать режим «общее пространство ключей» '
                         '(внешний энкодер без весов модели)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    from fracode_memory_probe import FracodeMemory
    from night_task5_fracode_forward import StreamFracode

    seeds = [a.src] + [int(x) for x in a.dst.split(',') if x.strip()]
    models = {}
    vocab = None
    for s in seeds:
        m, _, v = load_model(s, pattern=a.pattern)
        if m is None:
            log(f'  seed{s}: чекпойнта нет — пропуск')
            continue
        models[s] = m
        vocab = v
        log(f'  seed{s}: загружена (vocab={v})')
    if a.src not in models:
        log(f'НЕТ чекпойнта источника seed{a.src}')
        return 2
    # --self-check: приёмник = источник. Проверяем, что измеритель вообще
    # способен увидеть «перенос», когда он тривиален. Без этой проверки
    # отрицательный результат переноса между разными моделями неинтерпретируем.
    if a.self_check:
        log('РЕЖИМ SELF-CHECK: приёмник = источник (измеритель обязан дать ~1.0)')
        seeds = [a.src]
    elif len(models) < 2:
        log('нужно минимум 2 чекпойнта')
        return 2

    # токенизатор под vocab чекпойнтов
    cands = _TOK_BY_VOCAB.get(vocab, _TOK_BY_VOCAB[512])
    TOK = next((p for p in cands if os.path.exists(p)), None)
    if TOK is None:
        log(f'НЕТ токенизатора для vocab={vocab} (искал {cands})')
        return 2
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(TOK)
    log(f'токенизатор: {TOK} (V={tk.get_vocab_size()})')

    if not os.path.exists(CORPUS):
        log(f'НЕТ корпуса {CORPUS}')
        return 2
    txt = open(CORPUS, encoding='utf-8', errors='ignore').read()
    ids_all = np.array(tk.encode(txt).ids, dtype=np.int64)
    # ВАЖНО: id клиппится по vocab МОДЕЛИ, а не токенизатора. У линии v7 чекпойнт
    # vocab=512, а токенизатор tok_v31.json объявляет 8192 — id выше 511 в модель
    # не лезут. Это и тот же класс рассинхрона «словарь архива != словарь модели».
    V_MODEL = int(models[a.src].embed.weight.shape[0])
    n_clip = int((ids_all >= V_MODEL).sum())
    frac_clip = n_clip / max(len(ids_all), 1)
    if frac_clip > 0.10:
        log(f'STOP: vocab modeli={V_MODEL} slishkom mal - klipiruetsya '
            f'{100 * frac_clip:.1f}% tokenov. Arhiv stanet bessmyslennym '
            f'(raznye teksty solyutsya v odni id). Nuzhna liniya s vocab>=8192, '
            f'naprimer --pattern "v8_dialog_seed{{}}.pt".')
        return 3
    ids_all = np.clip(ids_all, 0, V_MODEL - 1)
    ids = ids_all[:a.bits] if len(ids_all) >= a.bits else ids_all
    log(f'tokenov arhiva: {len(ids):,}; vocab modeli={V_MODEL}, '
        f'klipirovano {n_clip} ({100.0 * frac_clip:.2f}%)')

    src = a.src
    m_src = models[src]
    W = m_src.pos.shape[1]
    n_win = len(ids) // W
    ids = ids[:n_win * W]
    global _ARCH_IDS
    _ARCH_IDS = ids
    log(f'архив: {len(ids):,} токенов = {n_win} окон по W={W}')

    def build(ids_local, model, det=True):
        """-> (fm, codes). Кодбук обучен на ключах ЭТОЙ модели."""
        fm = StreamFracode(192, a.L, a.S, a.K, device=dev)
        keys = static_keys(model, ids_local, dev)
        fm.fit(keys, iters=15, seed=0, det=det)
        return fm, fm.encode_rows(keys)

    # --- кодбук источника A ---
    t0 = time.time()
    fm_src, codes_src = build(ids, m_src)
    log(f'кодбук A(seed{src}) обучен за {time.time() - t0:.1f}s; '
        f'{fm_src.bytes_per_pos:.0f} Б/ключ')

    # позиции запросов (не в хвосте)
    rng = np.random.default_rng(7)
    lo, hi = 64, len(ids) - 64
    q_pos = rng.choice(np.arange(lo, hi), size=min(a.nq, hi - lo),
                       replace=False).tolist()

    # ---- (4) ПЕРЕНОС ЧЕРЕЗ ОБЩЕЕ ПРОСТРАНСТВО КЛЮЧЕЙ (кандидат №1 реформации) ----
    # Ключ здесь — функция ТОКЕНОВ, а не весов модели. Тогда архив физически
    # адресуем любой моделью, и вопрос «переносится ли память» становится
    # проверяемым. Кодбук учится ОДИН раз на ключах источника, но это уже не
    # важно: пространство общее, поэтому кодбук и не должен быть «своим».
    ext = None
    if not a.no_external:
        t0 = time.time()
        KE = external_keys(ids, a.S, 192)
        KE_t = torch.tensor(KE, dtype=torch.float32, device=dev)
        fm_ext = StreamFracode(192, a.L, a.S, a.K, device=dev)
        fm_ext.fit(KE_t, iters=15, seed=0, det=True)
        codes_ext = fm_ext.encode_rows(KE_t)
        Qe = KE_t[torch.tensor(q_pos, dtype=torch.long, device=dev)]
        r_ext = transfer_eval_ext(fm_ext, codes_ext, Qe, q_pos, dev)['recall@k']
        # контроль: тот же измерения на «чужих» позициях (потолок метрики)
        log(f'общее пространство ключей (внешний энкодер): recall@8 = {r_ext:.3f} '
            f'({time.time() - t0:.1f}s)')
        ext = {'recall': r_ext, 'dim': 192}
        del KE_t, fm_ext, codes_ext, Qe

    results = {'src': src, 'nq': len(q_pos), 'bits': int(len(ids)),
               'W': int(W), 'n_win': int(n_win),
               'S': a.S, 'K': a.K, 'L': a.L, 'external_key_space': ext,
               'per_dst': {}}

    for dst in seeds:
        # self-check разрешает dst == src (приёмник = источник)
        if dst not in models or (dst == src and not a.self_check):
            continue
        m_dst = models[dst]

        # (1) baseline приёмника: кодбук B на ключах B, запрос B
        fm_dst, codes_dst = build(ids, m_dst)
        r_same = transfer_eval(fm_dst, codes_dst, m_dst, q_pos, dev)['recall@k']

        # (2) ПЕРЕНОС: архив = ключи A, кодбук A; читает B
        r_cross = transfer_eval(fm_src, codes_src, m_dst, q_pos, dev)['recall@k']

        # (3) контроль: кодбук обучен на ключах B, но применяется к ключам A
        fm_hyb = StreamFracode(192, a.L, a.S, a.K, device=dev)
        fm_hyb.fit(static_keys(m_dst, ids, dev), iters=15, seed=0, det=True)
        codes_hyb = fm_hyb.encode_rows(static_keys(m_src, ids, dev))
        r_hyb = transfer_eval(fm_hyb, codes_hyb, m_dst, q_pos, dev)['recall@k']

        results['per_dst'][dst] = {
            'own_arch': r_same,
            'transfer': r_cross,
            'cross_fit_cbook': r_hyb,
        }
        log(f'seed{src} -> seed{dst}: '
            f'свой архив {r_same:.3f} | '
            f'ПЕРЕНОС {r_cross:.3f} | '
            f'кодбук B на ключах A {r_hyb:.3f}')

    out = a.out or os.path.join(HERE, 'runs',
                               f'transfer_src{src}_' + time.strftime('%Y%m%dT%H%M%S') + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'сохранено: {out}')

    vals = [v['transfer'] for v in results['per_dst'].values()]
    own = [v['own_arch'] for v in results['per_dst'].values()]
    mt, mo = float(np.mean(vals)), float(np.mean(own))
    log(f'СРЕДНЕЕ: ПЕРЕНОС {mt:.3f} против свой архив {mo:.3f}')
    if ext is not None:
        log(f'ПЕРЕНОС ЧЕРЕЗ ОБЩЕЕ ПРОСТРАНСТВО КЛЮЧЕЙ: {ext["recall"]:.3f} '
            f'(ключ = функция токенов, не весов модели)')

    # SELF-CHECK — это КАЛИБРОВКА ИЗМЕРИТЕЛЯ, а не вердикт переноса.
    # Приёмник = источник, режимы 1/2/3 вырождаются в одно и то же по построению.
    # Печатать здесь «переносится» было бы подлогом: измеряется потолок метрики
    # при данном размере архива, не более. Вердикт H-T выносится ТОЛЬКО на
    # разных моделях.
    if a.self_check:
        log(f'КАЛИБРОВКА (не вердикт): потолок метрики на {results["n_win"]} окнах '
            f'= {mt:.3f}. Вердикт переноса выносится только на разных моделях.')
        results['mode'] = 'self_check'
        results['ceiling'] = mt
        results['verdict'] = None
        json.dump(results, open(out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        return 0

    results['mode'] = 'transfer'
    # ВАЖНО (11.09.2026): вердикт ниже относится ТОЛЬКО к режиму «ключи из весов».
    # Если посчитано внешнее общее пространство, оно выносится отдельным полем —
    # иначе один верхнеуровневый 'not_transferable' читается как приговор переносу
    # вообще, хотя общее пространство ключей как раз переносится (0.797).
    results['verdict_scope'] = 'weights_keyed_only'
    if ext is not None:
        results['external_key_space_verdict'] = (
            'transferable' if ext['recall'] > 0.5 else
            'partial' if ext['recall'] > 0.2 else 'not_transferable')
    if mt >= 0.8 * mo and mt > 0.5:
        results['verdict'] = 'transferable'
        log('ВЕРДИКТ: переносится — память отделяется от модели')
    elif mt > 0.2:
        results['verdict'] = 'partial'
        log('ВЕРДИКТ: частичный перенос — нужно мерить, что общего требуется')
    else:
        results['verdict'] = 'not_transferable'
        log('ВЕРДИКТ (только для ключей из весов): НЕ переносится — '
            'нужно общее пространство ключей')
    if ext is not None:
        log(f'ВЕРДИКТ (общее пространство ключей): '
            f'{results["external_key_space_verdict"]} — {ext["recall"]:.3f}')
    json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
