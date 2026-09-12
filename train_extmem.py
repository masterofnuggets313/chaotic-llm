# -*- coding: utf-8 -*-
"""train_extmem.py — ДООБУЧЕНИЕ модели читать ВНЕШНЮЮ память Fracod.

ПОСТАНОВКА (из §13.11 / плана §4.10, п.1).

Сквозной проход при M >> W дал отрицательный результат: loss НЕ чувствует
качества выборки — разброс от идеальной выборки (`ov8` = 1.000) до сломанной
(`ov8` = 0.000) составил ~0.01 нат при SE ≈ 0.009, знак нестабилен. Механизм
понятен: чекпойнт учился с `key_codes=None`, то есть адресовал СОБСТВЕННЫЕ
ключи `embed(x)+pos`. Запрос `q0` лежит в пространстве весов модели, а архив —
в пространстве `external_keys()` (детерминированная функция токенов). Косинус
между пространствами близок к случайному → выбирается что попало → информации нет.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ. Тот же чекпойнт, но прямой проход идёт с
`key_codes = внешние ключи`, `n_keys = M`. Градиент идёт в `embed` и `pos`
через `q0 = mean(keys_range(model, x, W-nq, W))`, и в `query_proj` через
`h_last`. То есть модель УЧИТСЯ строить запрос в чужом адресном пространстве.
Если она научится — loss станет чувствителен к правильности архива.

ПОЧЕМУ ЭТО ВАЖНО (научная нагрузка внешних ключей). Ключ `external_keys()` —
функция ТОКЕНОВ, а не весов. Значит такую память может ПИСАТЬ кто угодно
(другая модель, другой сид, вообще не модель), а читать — обученная модель.
Это и есть «переносимая память»: не «общий кодбук + общие ключи» (то
опровергнуто, cos между сидами 0.0029), а «общее адресное пространство».

ГЛАВНАЯ МЕТРИКА — ЧУВСТВИТЕЛЬНОСТЬ К ПАМЯТИ:

    G = loss(shuf) - loss(ext)

`shuf` — тот же архив, строки перемешаны; `ext` — правильный архив. Если
модель читает память, перемешивание адресов должно её ломать. На исходном
чекпойнте G ≈ 0.004 нат при M = 4 096 (§13.11) — то есть НИЧЕГО.

ПРЕДРЕГИСТРАЦИЯ (записана ДО прогона; пороги не двигать постфактум):

  H-EM1  После дообучения (>= 2000 шагов, batch 24, M = 4096, nsub = 1)
         чувствительность G на удержанных окнах одновременно:
             (а) G_end >= 0.05 нат — абсолютный порог, ~7 SE при 48 окнах;
             (б) G_end - G_0 >= 0.05 нат — ПАРНЫЙ прирост к шагу 0.
         Оба условия нужны потому, что G_0 заранее НЕ положителен: §13.11
         показал, что на исходном чекпойнте испорченная выборка может давать
         МЕНЬШИЙ loss (змейка shuf/zero в §13.11 давала −0.012 нат). Условие
         «G_end >= 5 * G_0» при G_0 <= 0 вырождается в тривиальное, поэтому
         вместо кратности — прирост. Если не выполняется — в этой
         конфигурации модель не учится читать внешнюю память; это честный
         отрицательный результат.

  H-EM2  Дообучение не разрушает языковую модель: loss_own (без внешних
         ключей, нативный режим) на тех же удержанных окнах вырастает
         не более чем на 0.30 нат против шага 0.

  H-EM3  Сжатие остаётся бесплатным и ПОСЛЕ дообучения:
         loss(q24) - loss(ext) <= 0.05 нат (24 Б/ключ, L2-S12-K256).

  Вспомогательная (механизм, не вердикт): `cos_top1` — средний косинус между
  запросом `q0` и лучшим внешним ключом архива. Если модель выучила адресное
  пространство, он должен вырасти.

РЕЗУЛЬТАТ ПЕРВОЙ РУКИ (`--key-mode sym`, 1500 шагов): **H-EM1 НЕ подтверждена**.
G прошёл −0.0048 → +0.0036 → −0.0020 → +0.0003 → +0.0008, `cos_top1` замёрз на
0.2430…0.2435. H-EM2 (+0.177 ≤ 0.30) и H-EM3 (+0.0068 ≤ 0.05) подтверждены.
Причина найдена `_diag_learnable_query.py` (§13.14): симметричный ключ
зависит от 8 БУДУЩИХ токенов, а запрос строится из прошлого — ключ
непредсказуем из состояния модели (лучшая линейная карта даёт R2 = 0.017).
Это НЕ «внешняя память не работает», а «конструкция ключа неадресуема».

  H-EM1b  Вторая рука — `--key-mode bag4` (причинный порядок-независимый ключ
          по последним 4 токенам). Те же пороги, что в H-EM1: G_end >= 0.05 нат
          И G_end − G_0 >= 0.05 нат. Офлайн-граница уже известна: одна линейная
          карта поверх ЗАМОРОЖЕННОГО q0 даёт R2 = 0.305 и медиану ранга 65 из
          65 536 — то есть пространство адресуемо, и порог 0.05 нат для
          обучаемой модели не является нереалистичным.
          РЕЗУЛЬТАТ: G = −0.0242 → −0.0216 → −0.0232 на шагах 0/200/400 —
          тоже плоско. Значит адресуемость НЕДОСТАТОЧНА: второй ограничитель —
          градиент через жёсткий top-k (см. H-EM1c).

  H-EM1c  Третья рука — `--key-mode bag4 --aux-align 1.0`. Вспомогательная
          функция: `1 − cos(q0, ключ позиции T−1)`. При bag4 ключ T−1 — это
          bag токенов [T−4..T−1], то есть ровно то, что модель уже увидела;
          цель T туда не входит, утечки нет. Так запрос учится принимать вид
          адреса, а не надеемся, что он дорастёт до него через насыщенный
          softmax по восьми отобранным ключам. Те же пороги, что H-EM1.
          Это последняя рука в этой серии: если она не даёт G >= 0.05,
          вывод — « LM не учится читать внешнюю память через hard top-k
          ни при каком адресном пространстве», и надо менять механизм
          выборки, а не ключ.

  H-ADV4  Четвёртая рука — СВЯЗКА (из §13.17). Адрес строится РЕЛАКСАЦИЕЙ по
          архиву (`--relax-n`, `--relax-k`) поверх ОТДЕЛЬНОГО адресного выхода
          `addr_b` (`--addr-lr`), обучение >= 1500 шагов, M = 4096, nsub = 1,
          ключ bag4 (tau = 4).
          Пороги ТЕ ЖЕ, что в H-EM1, чтобы руки были сравнимы:
              G_end >= 0.05 нат  И  G_end - G_0 >= 0.05 нат.
          ОБОСНОВАНИЕ ПОРОГА (записано до прогона). По отдельности:
          адрес без обучения даёт -0.015 нат (§13.17, 11 конфигураций),
          обучение без адреса даёт +0.0218 прироста G (§13.16, рука 3).
          Если связка аддитивна, она даст ~0.037 — и H-ADV4 НЕ пройдёт.
          Порог 0.05 — это прямое требование НЕаддитивности: нужна именно
          связка, а не сумма двух слабых эффектов.
          Зачем это может сработать, когда части не работали: в §13.17 адрес
          сходится (медиана ранга 2205 -> 210 при k=1), но модель не умеет
          читать прочитанное; в §13.16 модель учится, но адрес у неё плохой.
          Здесь addr_b учится продолжать то, что начала релаксация, и градиент
          в него течёт (релаксация МЯГКАЯ, softmax, а не argmax — жёсткий
          argmax при k=1 стирает зависимость адреса от addr_b и обрывает
          градиент; это причина, по которой мягкая версия обязательна).

  УТОЧНЕНИЕ К H-ADV4 (внесено ДО запуска, сами пороги НЕ менялись).
  Прогон обязан идти с `--aux-align >= 1.0`. Причина найдена измерением нормы
  градиента `addr_b` (§13.18): через селекцию градиент НЕ ДОХОДИТ —

      |grad addr_b|:  без aux 0.0001 (с релаксацией) / 0.0021 (без неё)
                      aux=1.0 → 0.0477,  aux=5.0 → 0.2343

  top-k по косинусу при TEMP=0.3 — почти кусочно-постоянная операция, поэтому
  производная по адресу либо ~0, либо не определена. aux идёт в адрес МИНУЯ
  селекцию и поэтому является единственным рабочим каналом обучения. Это же
  объясняет три нуля §13.13/§13.16: там aux был приложен к `q0`, занятому
  ридаутом. Релаксация при этом градиенту почти не мешает (0.0477 против
  0.0568 без неё, −16 %).

ДИЗАЙН. Обучение и оценка идут на ОДНОМ корпусе, но на РАЗНЫХ диапазонах:
обучение T ∈ [lo_tr, TR_END), оценка T ∈ [EV_LO, EV_HI), причём
TR_END << EV_LO — окна оценки не участвовали в обучении. (Архивы оценки
могут заходить в обучающий диапазон — это допустимо: архив это не цель,
а память.)

Запуск:
    python train_extmem.py --ckpt v8_dialog_seed0.pt \
        --corpus corpus_dialog_train.txt --tok tok_v8_dialog.json \
        --steps 0                       # только базовая оценка (G_0)
    python train_extmem.py ... --steps 2000 --batch 24 --M 4096 \
        --out runs/extmem_s0_M4096.pt

ВАЖНО. Стоимость шага ~B последовательных прямых проходов (forward_general
не батчует примеры), поэтому шаг заметно дороже обычного обучения. Шаг 0
(оценка) стоит ~B*5 прямых проходов.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'phase01'))
os.chdir(HERE)

import night_task5_fracode_forward as n5          # noqa: E402
import exp_transferable_memory as etm             # noqa: E402

TEMP = 0.3
PAD = 40          # запас контекста для external_keys (window=16 -> half=8)
HALF = 8
ARMS = ['own', 'ext', 'q24', 'shuf', 'zero']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- внешние ключи
def ext_keys_torch(ids_t, d=192, window=16):
    """То же, что `etm.external_keys(ids, nsub=1, dim=d)`, но на GPU.

    ВАЖНО: реализация ДОЛЖНА совпадать с эталонной по значениям — проверяется
    `selfcheck_ext_keys()`. Расхождение означало бы, что обучение и прежние
    измерения идут в разных адресных пространствах.

    Оптимизация: `np.add.at` в эталоне — это ~1.5M операций/с, на 6M токенов
    это часы. Здесь 17 проходов `scatter_add_` по GPU — секунды.
    """
    N = int(ids_t.numel())
    half = int(window) // 2
    out = torch.zeros(N, d, dtype=torch.float32, device=ids_t.device)
    for off in range(-half, half + 1):
        sh = torch.zeros(N, dtype=torch.int64, device=ids_t.device)
        if off < 0:
            sh[-off:] = ids_t[:off]
        elif off > 0:
            sh[:-off] = ids_t[off:]
        else:
            sh = ids_t
        wgt = 1.0 / (1.0 + abs(off))
        h = (sh * 2654435761 + (off + half) * 40503) & 0x7FFFFFFF
        col = (h % d).view(N, 1)
        sgn = (torch.where((h // d) % 2 == 0, 1.0, -1.0).to(torch.float32)
               * wgt).view(N, 1)
        out.scatter_add_(1, col, sgn)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def bag_keys_torch(ids_t, d=192, w=4):
    """ПРИЧИННЫЙ порядок-независимый внешний ключ: хеши токенов [j-w+1 .. j].

    Чем отличается от `ext_keys_torch` (и почему это важно):

      * окно СДВИНУТО В ПРОШЛОЕ — ключ позиции j не содержит токенов после j.
        Симметричное окно +-8 содержало 8 БУДУЩИХ токенов, которые запрос
        (построенный из уже увиденного контекста) принципиально не может
        предсказать;
      * вклад токена НЕ зависит от его смещения в окне, поэтому ключ — сумма
        по множеству токенов, а не по последовательности. Запрос `q0` — это
        mean по последним nq embeddings, тоже нечувствительный к порядку.
        Значения выровнены по построению.

    Измерено `_diag_learnable_query.py` (120 000 позиций, архив 65 536):
    при симметричном ключе лучшая линейная карта q0 -> ключ даёт R2 = 0.017,
    медиана ранга 17 797; при этом ключе — R2 = 0.305, медиана ранга **65**.
    """
    N = int(ids_t.numel())
    out = torch.zeros(N, d, dtype=torch.float32, device=ids_t.device)
    rows = torch.arange(N, dtype=torch.int64, device=ids_t.device)
    for o in range(w):
        sh = torch.zeros(N, dtype=torch.int64, device=ids_t.device)
        if o == 0:
            sh.copy_(ids_t)
        else:
            sh[o:] = ids_t[:N - o]
        valid = rows >= o                       # иначе на краю вклад нулевого токена
        c1 = ((sh * 2654435761) & 0x7FFFFFFF) % d
        s1 = torch.where((((sh * 40503) & 0x7FFFFFFF) // d) % 2 == 0, 1.0, -1.0)
        s1 = torch.where(valid, s1, torch.zeros_like(s1)).to(torch.float32)
        out.scatter_add_(1, c1.view(N, 1), s1.view(N, 1))
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def selfcheck_ext_keys(ids_np, dev, d=192):
    """Сверить GPU-реализацию с эталонной numpy. Возвращает макс. расхождение."""
    n = 5000
    sub = np.asarray(ids_np[:n], dtype=np.int64)
    ref = etm.external_keys(sub, 1, d, window=16)
    got = ext_keys_torch(torch.tensor(sub, device=dev), d=d).cpu().numpy()
    return float(np.abs(ref - got).max())


def precompute_ext_keys(ids_np, dev, d=192, chunk=1_000_000, cache=None,
                        mode='sym', w=4):
    """Ключи для ВСЕГО корпуса -> CPU fp16 (N, d).

    Считается на GPU кусками с перекрытием HALF по краям: окно ключа заглядывает
    на +-8 токенов, и на границе куска это надо учесть, иначе край куска будет
    посчитан с нулевым контекстом, а не с реальным.

    КЭШ: на 6M токенов это минуты и 2.3 ГБ, а корпус и токенизатор между
    прогонами не меняются -> пишем .npy по хэшу входов и переиспользуем.
    Читается `mmap_mode='r'`, поэтому в RAM не загружается целиком.
    """
    N = len(ids_np)
    if cache and os.path.exists(cache):
        arr = np.load(cache, mmap_mode='r')
        if arr.shape == (N, d):
            log(f'внешние ключи из кэша: {cache}')
            return arr
        log(f'кэш не подошёл по форме ({arr.shape} != {(N, d)}) — пересчитываю')
    if mode == 'sym':
        fn, ctx_l = ext_keys_torch, HALF          # симметричное окно +-8
    else:
        # ПРИЧИННОЕ окно шириной w: ключ позиции j = bag токенов [j-w+1 .. j].
        # w — это ЗАПАЗДЫВАНИЕ tau в терминах дифференциально-разностного
        # уравнения (§13.17); свип по w ищет характерный масштаб памяти.
        fn = (lambda t, d=d: bag_keys_torch(t, d=d, w=w))
        ctx_l = w
    out = np.zeros((N, d), dtype=np.float16)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        lo = max(0, a - ctx_l)
        sub = torch.tensor(np.asarray(ids_np[lo:b], dtype=np.int64), device=dev)
        KK = fn(sub, d=d)
        off = a - lo
        out[a:b] = KK[off:off + (b - a)].to(torch.float16).cpu().numpy()
        log(f'  ключи {b:,}/{N:,}')
        del sub, KK
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, out)
        log(f'кэш внешних ключей записан: {cache}')
    return out


# ---------------------------------------------------------------- токенизация
def _tokenize(path, tok_path, chunk=4_000_000, ckpt_vocab=None):
    import tokenizers
    import final_benchmark as fb
    if ckpt_vocab is not None and ckpt_vocab != 8192:
        head = fb.load_chars(os.path.join(HERE, '..', 'phase01', 'corpus_train.txt'),
                             990_000)
        tk = fb.make_bpe(head)
        if tk.get_vocab_size() != ckpt_vocab:
            raise SystemExit(f'BPE даёт vocab {tk.get_vocab_size()} != {ckpt_vocab}')
        text = fb.load_chars(path, None)
        parts = []
        for i in range(0, len(text), chunk):
            parts.extend(tk.encode(text[i:i + chunk]).ids)
        return np.array(parts, dtype=np.int64), tk.get_vocab_size()
    tk = tokenizers.Tokenizer.from_file(tok_path)
    text = fb.load_chars(path, None)
    parts = []
    for i in range(0, len(text), chunk):
        parts.extend(tk.encode(text[i:i + chunk]).ids)
    return np.array(parts, dtype=np.int64), tk.get_vocab_size()


def _ids_cached(corpus, tok_path, ckpt_vocab, cache_dir):
    """Токенизация дорогая (~1-2 мин на 25 МБ), кэшируем по хэшу входов."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1()
    h.update(os.path.basename(corpus).encode())
    h.update(str(os.path.getsize(corpus)).encode())
    h.update(os.path.basename(tok_path).encode())
    h.update(str(ckpt_vocab).encode())
    p = os.path.join(cache_dir, f'ids_{h.hexdigest()[:16]}.npy')
    if os.path.exists(p):
        return np.load(p, mmap_mode=None), p, True
    ids, V = _tokenize(corpus, tok_path, ckpt_vocab=ckpt_vocab)
    np.save(p, ids)
    return ids, p, False


# ---------------------------------------------------------------- модель
def load_model(ckpt, V, dev):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    m = n5.build_pc_model('pc', vocab=V, d=192, layers=8, k_init=1.2,
                          sync_steps=8, driver_mode='sts_prog', alpha=0.3, temp=TEMP)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # `addr_b.*` — НОВЫЙ отдельный адресный выход (§13.16). В старых
    # чекпойнтах его нет, и это ожидаемо: он инициализируется НУЛЯМИ, то есть
    # адрес = q0 + 0 = q0, и шаг 0 сравним с прежними прогонами напрямую.
    miss_new = [k for k in missing if not k.startswith('addr_b.')]
    if miss_new or unexpected:
        raise SystemExit(f'чекпойнт не подошёл: missing={len(miss_new)} '
                         f'unexpected={len(unexpected)}: '
                         f'{miss_new[:5]} {unexpected[:5]}')
    if missing:
        log(f'чекпойнт без адресного выхода: {len(missing)} ключей addr_b.* '
            f'инициализированы НУЛЯМИ (адрес == q0, шаг 0 сравним с прежними)')
    return m.to(dev)


def relax_address_soft(q, KE, steps, k, temp=0.3, tail=None):
    """ДИФФЕРЕНЦИРУЕМАЯ релаксация адреса (§13.17). Принимает и отдаёт (1, d).

    Отличается от `_diag_advanced_relax.relax_address` — там ЖЁСТКИЙ argmax.
    ПОЧЕМУ МЯГКАЯ ВЕРСИЯ ОБЯЗАТЕЛЬНА ДЛЯ ОБУЧЕНИЯ: при k = 1 и argmax
    `q_{n+1} = KE[i*+1]` не зависит от `q_n` дифференцируемо (argmax —
    кусочно-постоянная функция), то есть релаксация ПОЛНОСТЬЮ стирает
    зависимость адреса от обученного выхода `addr_b`, и градиент обрывается.
    Softmax при temp -> 0 сколь угодно близок к argmax, но остаётся
    дифференцируемым — поэтому градиент доходит и до `addr_b`, и до `embed`.
    """
    if steps <= 0:
        return q
    lim = KE.shape[0] - (n5.TAIL if tail is None else int(tail))
    nxt = KE[1:lim + 1]                        # nxt[i] = KE[i+1] — «что дальше»
    for _ in range(int(steps)):
        sim = n5.cos_sim(KE[:lim], q).squeeze(1)
        w = torch.softmax(sim / temp, 0)
        d = (w.unsqueeze(-1) * nxt).sum(0, keepdim=True)
        q = (1.0 - k) * q + k * d
        q = q / q.norm().clamp_min(1e-9)
    return q


def _fwd(m, x, t, key_codes=None, key_fm=None, n_keys=None, Mcand=1024,
         q_init=None):
    lg = n5.forward_general(m, x, 'trained', chunk=4096, key_codes=key_codes,
                            key_fm=key_fm, Mcand=Mcand, n_keys=n_keys,
                            q_init=q_init)
    return torch.nn.functional.cross_entropy(lg, t.view(1))


def _fwd_nograd(m, x, t, **kw):
    with torch.no_grad():
        return float(_fwd(m, x, t, **kw).item())


# ---------------------------------------------------------------- оценка
@torch.no_grad()
def evaluate(m, ids_t, EK, Ts, M, dev, iters=15, seed=0, relax=None):
    """Парная оценка на фиксированном наборе целей. Возвращает средние по окнам.

    `ov8` здесь нет: он мерит согласие ПОИСКА, а не то, читает ли модель память.

    `relax = (n, k, temp)` — релаксация адреса (§13.17). Применяется ТОЛЬКО к
    рукам `ext` и `shuf`: это и есть предмет измерения (чувствительность к
    памяти при данном адресе). Руки `q24` и `zero` — контроль стоимости сжатия
    и вырожденности, их адрес НЕ трогаем, иначе метрика смешает два эффекта.
    """
    W = int(m.pos.shape[1])
    nq = int(m.nquery)
    acc = {k: [] for k in ARMS}
    cos1, cos8, cos1a = [], [], []
    for ti, T in enumerate(Ts):
        T = int(T)
        x = ids_t[T - W:T]
        tgt = ids_t[T]
        KE = EK[T - M:T].to(device=dev, dtype=torch.float32)
        q0 = n5.keys_range(m, x, W - nq, W, 'trained').mean(0)
        qa = m.addr_of(q0.unsqueeze(0))            # адрес селекции (не ридаут)
        qin = relax_address_soft(qa, KE, *relax) if relax else None
        acc['own'].append(_fwd_nograd(m, x, tgt))
        acc['ext'].append(_fwd_nograd(m, x, tgt, key_codes=KE, q_init=qin))
        fm = n5.StreamFracode(192, 2, 12, 256, device=dev)
        fm.fit(KE, iters=iters, seed=seed)
        kc = fm.encode_rows(KE)
        acc['q24'].append(_fwd_nograd(m, x, tgt, key_codes=kc, key_fm=fm))
        g = torch.Generator(device=dev).manual_seed(1234 + ti)
        perm = torch.randperm(M, generator=g, device=dev)
        KEs = KE[perm].contiguous()
        # релаксируем на ПЕРЕМЕШАННОМ архиве — иначе сравнение нечестное:
        # адрес, построенный по правильному архиву, просто переносится.
        qin_s = relax_address_soft(qa, KEs, *relax) if relax else None
        acc['shuf'].append(_fwd_nograd(m, x, tgt, key_codes=KEs, q_init=qin_s))
        acc['zero'].append(_fwd_nograd(m, x, tgt,
                                       key_codes=torch.zeros((M, 192), device=dev)))
        sim = n5.cos_sim(KE, q0.unsqueeze(0)).squeeze(1)
        lim = M - n5.TAIL
        top = sim[:lim].topk(min(8, lim)).values
        cos1.append(float(top[0]))
        cos8.append(float(top.mean()))
        # тот же косинус, но для АДРЕСА (после addr_of), а не сырого q0 —
        # это и есть величина, которую учит addr_b
        topa = n5.cos_sim(KE, qa).squeeze(1)[:lim].topk(min(8, lim)).values
        cos1a.append(float(topa[0]))
        del KE, KEs, fm, kc, perm
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out['cos_top1'] = float(np.mean(cos1))       # сырой q0 (сравнимо с руками 1-3)
    out['cos_top8'] = float(np.mean(cos8))
    out['cos_top1_a'] = float(np.mean(cos1a))    # адрес после addr_of
    # парные разницы (одно и то же окно) -> SE по окнам
    a = {k: np.array(v) for k, v in acc.items()}
    d_shuf = a['shuf'] - a['ext']
    d_q24 = a['q24'] - a['ext']
    d_zero = a['zero'] - a['ext']
    d_own = a['own'] - a['ext']
    for nm, d in (('G', d_shuf), ('q24', d_q24), ('zero', d_zero), ('own', d_own)):
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        out[nm + '_se'] = se
        out[nm + '_t'] = float(d.mean() / se) if se > 0 else float('nan')
    out['G'] = float(d_shuf.mean())
    out['q24_delta'] = float(d_q24.mean())
    out['zero_delta'] = float(d_zero.mean())
    out['own_delta'] = float(d_own.mean())
    out['n'] = len(Ts)
    return out


def fmt_ev(tag, e):
    return (f'{tag}: own={e["own"]:.4f} ext={e["ext"]:.4f} q24={e["q24"]:.4f} '
            f'shuf={e["shuf"]:.4f} zero={e["zero"]:.4f} | '
            f'G={e["G"]:+.4f}+-{e["G_se"]:.4f} (t={e["G_t"]:+.2f}) '
            f'q24-ext={e["q24_delta"]:+.4f} cos1={e["cos_top1"]:+.4f} '
            f'cos1a={e.get("cos_top1_a", float("nan")):+.4f}')


# ---------------------------------------------------------------- обучение
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--tok', required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--M', type=int, default=4096, help='размер внешнего архива')
    ap.add_argument('--aux-align', type=float, default=0.0,
                    help='вес вспомогательной функции выравнивания запроса: '
                         '1 - cos(q0, ключ позиции T-1). Имеет смысл ТОЛЬКО при '
                         '--key-mode bag4 (там ключ T-1 = bag последних 4 токенов, '
                         'то есть ровно то, что запрос ЗНАЕТ)')
    ap.add_argument('--relax-n', type=int, default=0,
                    help='шагов релаксации адреса (§13.17); 0 = выключено')
    ap.add_argument('--relax-k', type=float, default=1.0,
                    help='шаг релаксации: 1.0 = адрес полностью уходит в '
                         'прочитанное продолжение')
    ap.add_argument('--relax-temp', type=float, default=0.3,
                    help='температура мягкого argmax в релаксации')
    ap.add_argument('--addr-lr', type=float, default=1e-3,
                    help='lr для отдельного адресного выхода addr_b '
                         '(в 10 раз больше базового: он инициализирован нулями '
                         'и должен успеть сдвинуться)')
    ap.add_argument('--key-mode', choices=['sym', 'bag4'], default='sym',
                    help="sym = симметричное окно +-8 (legacy, НЕадресуемо, "
                         "см. §13.14); bag4 = причинный порядок-независимый "
                         "ключ по 4 последним токенам")
    ap.add_argument('--eval-n', type=int, default=48, help='удержанных окон')
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--iters', type=int, default=15, help='итераций кодбука')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='куда писать дообученный чекпойнт')
    ap.add_argument('--resume', default=None, help='полный resume-чекпойнт')
    ap.add_argument('--limit-tokens', type=int, default=0,
                    help='обрезать корпус (0 = весь)')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    if dev == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)

    t0 = time.time()
    sd0 = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd0 = sd0.get('model', sd0) if isinstance(sd0, dict) else sd0
    V = int(sd0['embed.weight'].shape[0])

    ids, ids_path, cached = _ids_cached(a.corpus, a.tok, V,
                                        os.path.join(HERE, 'runs', '_idcache'))
    if a.limit_tokens and len(ids) > a.limit_tokens:
        ids = ids[:a.limit_tokens]
    N = len(ids)
    log(f'корпус {os.path.basename(a.corpus)}: {N:,} ток.'
        + (' [кэш]' if cached else ''))

    W = 256
    M = a.M
    lo_tr = M + PAD + W + 8
    TR_END = N - 40_000
    EV_LO = N - 20_000
    EV_HI = N - 2_000
    if not (lo_tr < TR_END < EV_LO < EV_HI):
        raise SystemExit(f'корпус слишком короткий для M={M}: N={N:,}')

    kh = hashlib.sha1()
    kh.update(f'{os.path.basename(a.corpus)}|{len(ids)}|{a.tok}|{V}|{a.key_mode}'.encode())
    ek_cache = os.path.join(HERE, 'runs', '_idcache',
                            f'extkeys_{a.key_mode}_{kh.hexdigest()[:16]}.f16.npy')
    log(f'считаю внешние ключи корпуса (режим {a.key_mode}, GPU)...')
    EK = precompute_ext_keys(ids, dev, cache=ek_cache, mode=a.key_mode)
    if a.key_mode == 'sym':
        dchk = selfcheck_ext_keys(ids, dev)
        log(f'внешние ключи готовы ({time.time() - t0:.0f}s), '
            f'расхождение с эталоном = {dchk:.2e}')
        if dchk > 1e-4:
            raise SystemExit('GPU-реализация external_keys не совпала с эталонной')
    else:
        log(f'внешние ключи готовы ({time.time() - t0:.0f}s)')

    ids_t = torch.tensor(np.asarray(ids, dtype=np.int64), device=dev)
    EK_t = torch.tensor(EK, device='cpu')

    m = load_model(a.ckpt, V, dev)
    W = int(m.pos.shape[1])
    nq = int(m.nquery)
    relax = (a.relax_n, a.relax_k, a.relax_temp) if a.relax_n > 0 else None
    n_par = sum(p.numel() for p in m.parameters())
    log(f'модель {n_par:,} параметров; W={W}; nq={nq}; M={M}; batch={a.batch}; '
        f'lr={a.lr}; шагов={a.steps}; aux_align={a.aux_align}')

    ev_Ts = np.linspace(EV_LO, EV_HI, a.eval_n).astype(np.int64)
    log(f'удержанные окна: {len(ev_Ts)} шт., T ∈ [{EV_LO:,}, {EV_HI:,}) '
        f'(обучение заканчивается на {TR_END:,})')

    # ОТДЕЛЬНАЯ ГРУППА ПАРАМЕТРОВ для адресного выхода: он стартует с нуля и
    # базовым lr просто не успеет сдвинуться за разумное число шагов.
    # 'lr0' — запомненный базовый lr для warmup (не трогается оптимизатором).
    p_addr = [p for nm, p in m.named_parameters() if nm.startswith('addr_b.')]
    p_base = [p for nm, p in m.named_parameters() if not nm.startswith('addr_b.')]
    opt = torch.optim.AdamW(
        [{'params': p_base, 'lr': a.lr, 'lr0': a.lr},
         {'params': p_addr, 'lr': a.addr_lr, 'lr0': a.addr_lr}],
        weight_decay=0.01)
    log(f'параметров: всего {sum(p.numel() for p in m.parameters()):,}, '
        f'из них addr_b {sum(p.numel() for p in p_addr):,} (lr={a.addr_lr})')
    start_step = 1
    hist = []
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step = int(ck.get('step', 0)) + 1
        hist = list(ck.get('hist', []))
        log(f'resume с шага {start_step}')

    # ---- базовая оценка (шаг 0) ----
    m.eval()
    e0 = evaluate(m, ids_t, EK_t, ev_Ts, M, dev, iters=a.iters, relax=relax)
    log(fmt_ev('step 0    ', e0))
    hist = [h for h in hist if h['step'] != 0]
    hist.append({'step': 0, **e0})

    out = a.out or os.path.join(HERE, 'runs', f'extmem_M{M}_s{a.seed}.pt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = (-1e9, 0)
    for h in hist:
        if h['G'] > best[0]:
            best = (h['G'], h['step'])

    if a.steps <= 0:
        jp = out.replace('.pt', '_eval0.json')
        json.dump({'args': vars(a), 'N': int(N), 'M': M, 'hist': hist,
                   'prereg': PREREG, 'verdict': None},
                  open(jp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log(f'только оценка; сохранено {jp}')
        return

    rng = np.random.default_rng(a.seed)
    m.train()
    tt = time.time()
    for step in range(start_step, a.steps + 1):
        lrs = min(1.0, step / max(1, a.warmup))
        for pg in opt.param_groups:
            pg['lr'] = pg.get('lr0', a.lr) * lrs
        T = rng.integers(lo_tr, TR_END, size=a.batch)
        idx = (T[:, None] - M + np.arange(M)[None, :]).astype(np.int64)   # (B, M)
        K = EK_t[idx].to(device=dev, dtype=torch.float32)                 # (B, M, 192)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        auxs = 0.0
        for i in range(a.batch):
            x = ids_t[T[i] - W:T[i]]
            tgt = ids_t[T[i]]
            qi = None
            if relax is not None or a.aux_align > 0:
                q0 = n5.keys_range(m, x, W - nq, W, 'trained').mean(0)
                qa = m.addr_of(q0.unsqueeze(0))     # адрес, НЕ признак ридаута
                if relax is not None:
                    qi = relax_address_soft(qa, K[i], *relax)
                if a.aux_align > 0:
                    # ЦЕЛЬ БЕЗ УТЕЧКИ: при bag4 ключ позиции T-1 — это bag
                    # токенов [T-4 .. T-1], то есть ровно то, что модель уже
                    # увидела. Цель T в него не входит. Выравниваем АДРЕС (не
                    # q0 — он занят ридаутом, см. §13.16), заставляя его принять
                    # вид «адрес по последним 4 токенам», после чего выборка по
                    # незамаскированной части архива найдёт все прошлые позиции
                    # с таким же bag. Именно это и есть долгая память.
                    kstar = K[i][-1]
                    auxs = auxs + (1.0 - torch.nn.functional.cosine_similarity(
                        qa, kstar.unsqueeze(0)).squeeze())
            tot = tot + _fwd(m, x, tgt, key_codes=K[i], q_init=qi)
        loss_step = tot / a.batch
        if a.aux_align > 0:
            loss_step = loss_step + a.aux_align * (auxs / a.batch)
        loss_step.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        del K

        if step % 100 == 0 or step == 1:
            el = time.time() - tt
            eta = el / max(step - start_step + 1, 1) * (a.steps - step)
            # НОРМА ГРАДИЕНТА АДРЕСНОГО ВЫХОДА — обязательная диагностика.
            # Если она ~0, то либо addr_b не в графе, либо релаксация стёрла
            # зависимость адреса от него (жёсткий argmax при k=1 именно это
            # и делает — см. docstring relax_address_soft), и прогон впустую.
            gn = float(sum(float(p.grad.norm()) for p in p_addr
                           if p.grad is not None)) if p_addr else 0.0
            log(f'[{step}/{a.steps}] train_loss={float(tot) / a.batch:.4f} '
                f'|grad_addr_b|={gn:.4f} '
                f'({time.time() - t0:.0f}s, ETA {eta / 60:.0f} мин)')

        if step % a.eval_every == 0 or step == a.steps:
            m.eval()
            e = evaluate(m, ids_t, EK_t, ev_Ts, M, dev, iters=a.iters, relax=relax)
            log(fmt_ev(f'step {step:<6}', e))
            hist.append({'step': step, **e})
            if e['G'] > best[0]:
                best = (e['G'], step)
                torch.save(m.state_dict(), out)
                log(f'  -> лучший по G ({e["G"]:+.4f}), записан {os.path.basename(out)}')
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(),
                        'step': step, 'hist': hist},
                       out.replace('.pt', '_full.pt'))
            m.train()

    m.eval()
    efin = evaluate(m, ids_t, EK_t, ev_Ts, M, dev, iters=a.iters, relax=relax)
    log(fmt_ev('ФИНАЛ    ', efin))
    hist.append({'step': a.steps, 'final': True, **efin})
    torch.save(m.state_dict(), out.replace('.pt', '_last.pt'))

    verdict = {
        'H-EM1': {
            'criterion': 'G_end >= 0.05 нат И G_end - G_0 >= 0.05 нат',
            'G_0': hist[0]['G'], 'G_end': efin['G'], 'se': efin['G_se'],
            'gain': efin['G'] - hist[0]['G'],
            'pass': bool(efin['G'] >= 0.05
                         and (efin['G'] - hist[0]['G']) >= 0.05)},
        'H-EM2': {
            'criterion': 'loss_own выросла не более чем на 0.30 нат',
            'own_0': hist[0]['own'], 'own_end': efin['own'],
            'delta': efin['own'] - hist[0]['own'],
            'pass': bool((efin['own'] - hist[0]['own']) <= 0.30)},
        'H-EM3': {
            'criterion': 'loss(q24) - loss(ext) <= 0.05 нат',
            'delta': efin['q24_delta'], 'se': efin['q24_se'],
            'pass': bool(efin['q24_delta'] <= 0.05)},
    }
    jp = out.replace('.pt', '.json')
    json.dump({'args': vars(a), 'N': int(N), 'M': M, 'W': W, 'hist': hist,
               'prereg': PREREG, 'verdict': verdict,
               'best_G': {'G': best[0], 'step': best[1]}},
              open(jp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in verdict.items():
        log(f'{k}: {"ПОДТВЕРЖДЕНА" if v["pass"] else "НЕ ПОДТВЕРЖДЕНА"} — {v}')
    log(f'сохранено: {jp}, {out}')


PREREG = {
    'H-EM1': ('после >= 2000 шагов чувствительность G = loss(shuf) - loss(ext) '
              'на удержанных окнах >= 0.05 нат И её прирост к шагу 0 >= 0.05 нат'),
    'H-EM1b': ('то же на причинном ключе --key-mode bag4 (пространство '
               'адресуемо офлайн: R2 = 0.305, медиана ранга 65 из 65 536)'),
    'H-EM2': 'loss_own вырастает не более чем на 0.30 нат против шага 0',
    'H-EM3': 'loss(q24) - loss(ext) <= 0.05 нат (24 Б/ключ после дообучения)',
}


if __name__ == '__main__':
    main()
