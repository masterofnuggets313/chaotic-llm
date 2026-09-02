"""night_task5_fracode_forward.py — T5: FracodeMemory ВНУТРИ прямого прохода STS-Prog.

ЦЕЛЬ (из results/fracode_memory_report.md, «Следующий шаг»):
  превратить «сжатие работает» в «модель работает со сжатой памятью».
  Меряем paired next-token loss/PPL + retrieval + driverCos при 8x/16x/32x
  ПРОТИВ точной памяти на ОДНИХ И ТОЛЬКО ЖЕ ОКНАХ (парное сравнение).

=== ЧТО ВАЖНОГО ВСКРЫЛОСЬ ПРИ ВСТРОЙКЕ (в предыдущем отчёте этого НЕТ) ===

В STS-Prog НЕТ отдельной «таблицы ключей», которую можно сжать без последствий.
В прямом проходе (models_pc.PurePCLM.forward, driver_mode="sts_prog") живут ДВА
разных (W,d)-объекта:

  * e = embed(x) + pos  — СТАТИЧНЫЕ ключи селекции; `en = e/||e||` считается ОДИН
    раз до цикла и переиспользуется ВСЕМИ слоями. Только q уточняется от слоя к слою.
  * h                   — эволюционирующее скрытое состояние (W,d).

Стриминговый форвард из probe_10m_purepclm.py держит ТОЛЬКО h, пересчитывая e по
чанкам на каждом слое — оттуда и честный пол 7.26 GB @ 10M. Из этого следует:

  VARIANT "keys"  — сжать ТОЛЬКО e:  VRAM НЕ экономит (h всё ещё точный (W,d)),
                    но экономит 8-кратный пересчёт embed+pos => выигрыш по СКОРОСТИ.
                    Цена: селекция становится приближённой (ADC).
  VARIANT "state" — сжать ТОЛЬКО h:  РЕАЛЬНАЯ экономия VRAM (коды W*bytes + чанк),
                    селекция остаётся ТОЧНОЙ (ключи пересчитываем). Цена: ошибка
                    квантования накапливается по 8 слоям (перекодирование каждый слой).
  VARIANT "both"  — сжать ОБА:       и VRAM, и скорость; цена — обе ошибки сразу.
                    Это и есть конфигурация, которую реально можно деплоить.

Все четыре (exact/keys/state/both) меряются. Это и есть ответ на вопрос Германа.

=== ЧЕСТНОСТЬ ЭКСПЕРИМЕНТА ===
  * Кодбуки обучаются НА HELD-OUT КАЛИБРОВОЧНОМ УЧАСТКЕ (не на.eval-окнах) — закрывает
    ограничение #6 предыдущего отчёта.
  * Окна для exact и сжатых вариантов ОДНИ И ТЕ ЖЕ => парное сравнение, а не
    сравнение средних по разным данным.
  * При W > 256 используются ЦИКЛИЧЕСКИЕ обученные позиции (pos[i % 256]) — никаких
    случайных новых параметров, в отличие от sinusoid+pe_proj в прошлых пробах.
  * Абсолютный PPL при W > 256 НЕ является PPL обученной модели (модель обучалась
    на окне 256). Имеет смысл ТОЛЬКО парная деградация против exact при том же W.

=== ТОКЕНИЗАЦИЯ (поймано на смоуке — важно) ===
  Чекпоинт обучался на BPE-токенах (final_benchmark.make_bpe, vocab=512), НЕ на байтах.
  Первая версия скрипта кормила модель ord(c)%512 и получала PPL ~1250-2100 при
  ln(512)=6.24, т.е. ХУЖЕ случайного угадывания — верный признак неверной токенизации.
  Теперь: токенизатор строится ТОЧНО как при обучении (BPE по первым 990k символов
  corpus_train.txt), а eval идёт по УЧАСТКУ, КОТОРЫЙ МОДЕЛЬ НЕ ВИДЕЛА — corpus5m_train.txt
  начинается теми же 990k символов (md5 совпал), поэтому его хвост честно held-out.

Запуск:  cd phase01/exp_vq && py -3.13 night_task5_fracode_forward.py
Результат (инкрементально): results/night_task5_fracode.json
"""
import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)
from models_pc import build_pc_model
from fracode_memory_probe import kmeans, assign_codes

RESULTS = os.path.join(REPO, "results")
OUT_JSON = os.path.join(RESULTS, "night_task5_fracode.json")

FP32 = 4
TAIL = 8          # маска последних 8 позиций окна (запрет самовыбора, как в модели)
TEMP = 0.3        # температура селекции (как в модели)
NQUERY_FALLBACK = 4


# ---------------------------------------------------------------- ключи
def keys_at(model, x, pos, mode):
    """Статичные ключи e для произвольных позиций pos: (n,) -> (n, d)."""
    emb = model.embed(x[pos])
    P = model.pos.shape[1]
    if mode == "trained":
        return emb + model.pos[0, pos]
    return emb + model.pos[0, pos % P]


def keys_range(model, x, s, e, mode):
    """Статичные ключи e для диапазона [s, e): (e-s, d)."""
    pos = torch.arange(s, e, device=x.device)
    return keys_at(model, x, pos, mode)


def cos_sim(A, B):
    """A (N,d), B (M,d) -> (N,M) косинусы."""
    An = A / (A.norm(dim=-1, keepdim=True) + 1e-6)
    Bn = B / (B.norm(dim=-1, keepdim=True) + 1e-6)
    return An @ Bn.T


# ---------------------------------------------------------------- Fracode (потоковый)
class StreamFracode:
    """Иерархическое остаточное квантование с посекционным (чанковым) кодом.

    Отличается от FracodeMemory из fracode_memory_probe.py тем, что не хранит коды
    внутри себя: encode_rows/decode_codes работают с произвольным срезом, что нужно
    для стриминга состояния по чанкам в прямом проходе.
    """

    def __init__(self, d, levels, subvecs, K, device="cuda"):
        assert d % subvecs == 0, f"d={d} не делится на subvecs={subvecs}"
        self.d = d
        self.L = levels
        self.S = subvecs
        self.K = K
        self.sub = d // subvecs
        self.device = device
        self.bits_per_pos = levels * subvecs * int(round(math.log2(K)))
        self.bytes_per_pos = self.bits_per_pos / 8.0
        self.cbooks = None
        self.codes = None          # ставится снаружи для adc_scores

    def fit(self, X, iters=12, seed=0):
        R = X.clone()
        self.cbooks = [[None] * self.S for _ in range(self.L)]
        for l in range(self.L):
            for s in range(self.S):
                sub_x = R[:, s * self.sub:(s + 1) * self.sub].contiguous()
                self.cbooks[l][s] = kmeans(sub_x, self.K, iters=iters, seed=seed + l * 977 + s)
            R = R - self._decode(self._assign(R, l), l)
        return self

    def _assign(self, R, l):
        N = R.shape[0]
        codes = torch.empty((N, self.S), dtype=torch.long, device=self.device)
        for s in range(self.S):
            sub_x = R[:, s * self.sub:(s + 1) * self.sub].contiguous()
            codes[:, s] = assign_codes(sub_x, self.cbooks[l][s])
        return codes

    def _decode(self, codes_l, l):
        W = codes_l.shape[0]
        out = torch.zeros((W, self.d), device=self.device)
        for s in range(self.S):
            out[:, s * self.sub:(s + 1) * self.sub] = self.cbooks[l][s][codes_l[:, s]]
        return out

    def encode_rows(self, M):
        N = M.shape[0]
        codes = torch.empty((N, self.L, self.S), dtype=torch.long, device=self.device)
        R = M
        for l in range(self.L):
            codes[:, l, :] = self._assign(R, l)
            R = R - self._decode(codes[:, l, :], l)
        return codes

    def decode_codes(self, codes):
        W = codes.shape[0]
        L = codes.shape[1] if codes.dim() == 3 else self.L
        out = torch.zeros((W, self.d), device=self.device)
        for l in range(L):
            out += self._decode(codes[:, l, :], l)
        return out

    def adc_scores(self, q):
        """q: (d,) -> (W,) приближённые НЕНОРМИРОВАННЫЕ скалярные произведения.

        Ненормированность допустима: это скрининг (отбор кандидатов), точный порядок
        восстанавливается переранком по восстановленным векторам.
        """
        Q = q.unsqueeze(0)                       # (1, d)
        out = torch.zeros(self.codes.shape[0], device=self.device)
        for l in range(self.L):
            for s in range(self.S):
                qsub = Q[:, s * self.sub:(s + 1) * self.sub]        # (1, sub)
                tbl = (qsub @ self.cbooks[l][s].T).squeeze(0)       # (K,)
                out += tbl[self.codes[:, l, s]]
        return out

    def select(self, q, topk, W, Mcand, rerank=False):
        """ADC-скрининг -> точный переранк -> (значения, позиции соседей для драйвера).

        rerank=False (по умолчанию, базовый ADC): переранк по КОСИНУСУ восстановленных
          кандидатов (дёшево, нормированные векторы).
        rerank=True (совет коллеги, 02.09.2026): переранк по ТОЧНОМУ DOT-PRODUCT восстановленных
          кандидатов — лечит rank-inversion (топ-8 восстанавливается плохо на 32×). Память не
          растёт: разворачиваем лишь Mcand кандидатов, а не весь (W,d). Возвращает те же индексы
          драйвера, но в порядке точного dot-product.
        """
        sc = self.adc_scores(q)
        sc[W - TAIL:] = -1e18
        m = min(Mcand, max(1, W - TAIL))
        cand = sc.topk(m).indices                                   # (m,)
        cn = torch.clamp(cand + 1, 0, W - 2)
        pos = torch.unique(torch.cat([cand, cn]))                   # сортировано
        rec = self.decode_codes(self.codes[pos])                    # (P, d)
        if rerank:
            dot = (rec @ q.unsqueeze(0).T).squeeze(1)               # (P,) ТОЧНЫЙ dot-product
            sim = dot
        else:
            sim = cos_sim(rec, q.unsqueeze(0)).squeeze(1)           # (P,) косинус
        i_c = torch.searchsorted(pos, cand)
        sim_c = sim[i_c]
        vals, loc = sim_c.topk(min(topk, m))
        return vals, cn[loc], cand


# ---------------------------------------------------------------- селекция
def exact_select(model, x, q, mode, bounds, topk, W):
    """Точная селекция на статичных ключах, пересчитываемых по чанкам."""
    dev = x.device
    run_val = torch.full((topk,), -1e9, device=dev)
    run_next = torch.zeros((topk,), dtype=torch.long, device=dev)
    for (s, e) in bounds:
        ec = keys_range(model, x, s, e, mode)
        sim = cos_sim(ec, q).squeeze(1)                    # (C,)
        C = e - s
        if e > W - TAIL:                                   # маска хвоста окна
            m = max(0, W - TAIL - s)
            if m < C:
                sim[m:] = -1e9
        ks = min(topk, C)
        if ks <= 0:
            continue
        cs, cloc = sim.topk(ks)
        cn = torch.clamp(s + cloc + 1, 0, W - 2)
        allv = torch.cat([run_val, cs])
        alln = torch.cat([run_next, cn])
        _, o = allv.topk(topk)
        run_val = allv[o]
        run_next = alln[o]
    return run_val, run_next


# ---------------------------------------------------------------- прямой проход
def forward_general(model, x, mode, chunk=4096, fms=None, key_codes=None,
                    key_fm=None, Mcand=1024, rerank=False, capture=None):
    """Единый форвард STS-Prog с подключаемым сжатием.

    fms       : list[StreamFracode] по слоям — кодируем СОСТОЯНИЕ h  (None => точное h)
    key_codes : (W,L,S) коды статичных ключей e                      (None => пересчёт точно)
    key_fm    : StreamFracode, которому принадлежат key_codes
    capture   : список, куда положить выборку состояния НА ВХОДЕ каждого слоя
    """
    W = x.shape[0]
    d = model.d
    dev = x.device
    k_eff = torch.sigmoid(model.k)
    topk = int(model.topk)
    nq = int(model.nquery)
    bounds = [(s, min(s + chunk, W)) for s in range(0, W, chunk)]

    q0 = keys_range(model, x, W - nq, W, mode).mean(0, keepdim=True)     # (1, d)
    q = q0

    # ---- начальное состояние h_0 = статичные ключи e ----
    if fms is None:
        hchunks = [keys_range(model, x, s, e, mode) for (s, e) in bounds]
        hcodes = None
    else:
        hcodes = torch.empty((W, fms[0].L, fms[0].S), dtype=torch.long, device=dev)
        for (s, e) in bounds:
            hcodes[s:e] = fms[0].encode_rows(keys_range(model, x, s, e, mode))
        hchunks = None

    for li, blk in enumerate(model.blocks):
        if capture is not None:
            if hchunks is not None:
                capture[li] = _sample_chunks(hchunks, W, dev)
            else:
                capture[li] = fms[li].decode_codes(_sample_idx(W, dev))
        # ---- селекция драйвера ----
        if key_codes is None:
            run_val, run_next = exact_select(model, x, q, mode, bounds, topk, W)
        else:
            key_fm.codes = key_codes
            run_val, run_next, _ = key_fm.select(q.squeeze(0), topk, W, Mcand, rerank=rerank)
        if key_codes is None:
            ckey = keys_at(model, x, run_next, mode)
        else:
            ckey = key_fm.decode_codes(key_codes[run_next])
        w = torch.softmax(run_val / TEMP, 0)
        # ВАЖНО: h здесь без batch-оси — (C,d), поэтому драйвер (1,d), НЕ (1,1,d).
        # Иначе broadcast раздует состояние до (1,C,d) и всё сломается на 2-м слое.
        driver = (w.unsqueeze(-1) * ckey).sum(0).view(1, d)              # (1,d)
        # ---- обновление состояния ----
        if hchunks is not None:
            for ci in range(len(bounds)):
                hchunks[ci] = blk(hchunks[ci], driver, k_eff)
            h_last = hchunks[-1][-1].unsqueeze(0)
        else:
            for ci, (s, e) in enumerate(bounds):
                hs = fms[li].decode_codes(hcodes[s:e])
                hs = blk(hs, driver, k_eff)
                nxt = fms[li + 1] if (li + 1) < len(fms) else fms[li]
                hcodes[s:e] = nxt.encode_rows(hs)
            h_last = fms[min(li + 1, len(fms) - 1)].decode_codes(hcodes[W - 1:W])
        q = q0 + model.query_proj(h_last) * 0.5

    # ---- ридаут: g = mean по всем позициям (стримингом, без stack) ----
    if hchunks is not None:
        gsum = torch.zeros((d,), device=dev)
        for hc in hchunks:
            gsum += hc.sum(0)
    else:
        gsum = torch.zeros((d,), device=dev)
        for (s, e) in bounds:
            gsum += fms[-1].decode_codes(hcodes[s:e]).sum(0)
    g = (gsum / W).unsqueeze(0)
    if hchunks is not None:
        h_last = hchunks[-1][-1].unsqueeze(0)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))            # (1, V)


def _sample_idx(W, dev, ns=65536):
    step = max(1, W // ns)
    return torch.arange(0, W, step, device=dev)


def _sample_chunks(hchunks, W, dev, ns=65536):
    idx = _sample_idx(W, dev, ns)
    out, base = [], 0
    for hc in hchunks:
        C = hc.shape[0]
        sel = idx[(idx >= base) & (idx < base + C)] - base
        if sel.numel():
            out.append(hc[sel])
        base += C
    return torch.cat(out, 0)


# ---------------------------------------------------------------- данные
def load_bpe_ids(head_path, big_path, dev):
    """BPE-токены ТОЧНО в той кодировке, на которой обучался чекпоинт.

    Токенизатор: final_benchmark.make_bpe по первым 990k символов corpus_train.txt
    (ровно как в final_benchmark.build_data). n_head — сколько токенов занимает
    эта «обучающая» голова; всё ПОСЛЕ n_head модель не видела => честный held-out.
    """
    import numpy as np
    import final_benchmark as fb

    head = fb.load_chars(head_path, 990_000)
    tok = fb.make_bpe(head)
    V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    big = fb.load_chars(big_path, None)
    ids = np.array(tok.encode(big).ids, dtype=np.int64)
    return torch.tensor(ids, dtype=torch.long, device=dev), n_head, V


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(PHASE, "corpus5m_train.txt"),
                    help="большой корпус; его хвост (после corpus_head) — held-out")
    ap.add_argument("--corpus-head", default=os.path.join(PHASE, "corpus_train.txt"),
                    help="первые 990k символов — на них строился BPE при обучении")
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--Ws", default="256,4096,65536,262144")
    ap.add_argument("--nwin", default="256,64,32,16",
                    help="сколько окон на каждый W (должно совпадать по длине с --Ws)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--m-cand", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.Ws, args.nwin = "256,4096", "8,4"
        args.iters = 4

    dev = args.device
    torch.manual_seed(0)

    # ---- токены: BPE в кодировке обучения, eval на held-out хвосте ----
    toks, n_head, V = load_bpe_ids(args.corpus_head, args.corpus, dev)
    ho_len = len(toks) - n_head
    print(f"Корпус: {len(toks):,} BPE-токенов (V={V}); held-out с {n_head:,} "
          f"({ho_len:,}); codebook=75% | eval=25% (модель не видела ни того, ни другого)",
          flush=True)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.d
    L = len(model.blocks)
    Ppos = model.pos.shape[1]
    print(f"STS-Prog загружен: d={d} L={L} topk={model.topk} nquery={model.nquery} "
          f"pos_len={Ppos}", flush=True)

    Ws = [int(w) for w in args.Ws.split(",")]
    Ns = [int(n) for n in args.nwin.split(",")]
    assert len(Ws) == len(Ns), "--Ws и --nwin разной длины"

    # ---------------------------------------------------------------------------
    # ДИЗАЙН ПО ГЕРМАНУ (02.09.2026): три столбца и честный held-out кодбук.
    #   Столбцы:  Exact (1x) | PQ (плоский L=1) | Fracode (иерархия L=2)
    #   Режимы:   8x / 16x / 32x  — причём PQ и Fracode на РАВНОМ бюджете байт
    #   Кодбук:    первые 75% held-out участка = обучение кодбука (calib);
    #              последние 25% = eval (модель эти токены НЕ видела).
    #   Главная метрика: деградация PPL относительно Exact на ТЕХ ЖЕ окнах (парная).
    # ---------------------------------------------------------------------------
    # (имя, тип, уровни L, подвекторы S) при K=256 (8 бит/код).
    # bytes/pos = L*S ; равный бюджет: FR-8x(L2,S48)=96 <-> PQ-8x(L1,S96)=96; и т.д.
    cfgs = [
        ("PQ-8x",  "PQ", 1, 96),   ("FR-8x",  "FR", 2, 48),
        ("PQ-16x", "PQ", 1, 48),   ("FR-16x", "FR", 2, 24),
        ("PQ-32x", "PQ", 1, 24),   ("FR-32x", "FR", 2, 12),
    ]

    all_out = {"script": "night_task5_fracode_forward.py",
               "design": "по Герману: Exact|PQ|Fracode × {8x,16x,32x}; held-out codebook (75% calib / 25% eval); парная деградация PPL",
               "ckpt": args.ckpt, "K": args.K, "chunk": args.chunk,
               "device": dev, "runs": []}

    # ---- held-out сегменты (внутри общего held-out хвоста корпуса) ----
    ho_start = n_head
    ho_len = len(toks) - n_head
    seg_codebook = (ho_start, ho_start + int(0.75 * ho_len))   # 75% — обучение кодбука
    seg_eval = (ho_start + int(0.75 * ho_len), len(toks))      # 25% — eval

    for W, N in zip(Ws, Ns):
        mode = "trained" if W <= Ppos else "cyclic"
        ev_start, ev_end = seg_eval
        if ev_end - ev_start < N + W + 1:
            print(f"W={W}: в eval-сегменте ({ev_end-ev_start:,}) не хватает на {N} окон×W={W} "
                  f"— пропуск", flush=True)
            continue
        cb_start, cb_end = seg_codebook
        if cb_end - cb_start < 2 * W + 4:
            print(f"W={W}: в codebook-сегменте ({cb_end-cb_start:,}) мало для калибровки "
                  f"— пропуск", flush=True)
            continue
        print("\n" + "=" * 90, flush=True)
        print(f"W={W:,}  окон={N}  pos_mode={mode}", flush=True)
        print("=" * 90, flush=True)

        ev_base = ev_start
        Mcand = min(args.m_cand, max(1, W - TAIL))

        # ---------- калибровка кодбуков: выборки состояния h_l из codebook-сегмента ----------
        cap = [None] * L
        cb_toks = toks[cb_start:cb_start + W + 1]
        with torch.no_grad():
            _ = forward_general(model, cb_toks[:W], mode, chunk=args.chunk, capture=cap)

        # ---------- точная база (те же eval-окна, что и у сжатых) ----------
        t0 = time.time()
        losses_exact = []
        with torch.no_grad():
            for i in range(N):
                x = toks[ev_base + i:ev_base + i + W]
                tgt = toks[ev_base + i + W]
                logits = forward_general(model, x, mode, chunk=args.chunk)
                losses_exact.append(F.cross_entropy(logits, tgt.view(1)).item())
        torch.cuda.synchronize()
        dt_exact = time.time() - t0
        loss_exact = sum(losses_exact) / len(losses_exact)
        ppl_exact = math.exp(min(20.0, loss_exact))
        print(f"{'exact':<28}{'768.0':>9}{'1.0x':>7}{ppl_exact:>10.3f}"
              f"{'0.0%':>9}{'-':>11}", flush=True)

        run = {"W": W, "nwin": N, "pos_mode": mode, "loss_exact": loss_exact,
               "ppl_exact": ppl_exact, "exact_sec": dt_exact, "configs": []}

        # ---------- сжатые варианты: PQ vs Fracode на равном бюджете ----------
        for cname, ctype, Lv, S in cfgs:
            if d % S != 0:
                print(f"  пропуск {cname}: d={d} не делится на S={S}", flush=True)
                continue
            torch.cuda.empty_cache()
            t_fit = time.time()
            fms = []
            for li in range(L):
                fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
                fm.fit(cap[li], iters=args.iters, seed=li)
                fms.append(fm)
            torch.cuda.synchronize()
            t_fit = time.time() - t_fit

            bpp = fms[0].bytes_per_pos
            ratio = (d * FP32) / bpp

            # четыре режима: состояние / ключи / оба / ключи+rerank (совет коллеги)
            for vname, use_state, use_keys, rerank in [
                ("state", True, False, False),
                ("keys", False, True, False),
                ("keys+rr", False, True, True),
                ("both", True, True, False)]:
                torch.cuda.empty_cache()
                t0 = time.time()
                losses = []
                with torch.no_grad():
                    for i in range(N):
                        o = ev_base + i
                        x = toks[o:o + W]
                        tgt = toks[o + W]
                        kc = fms[0].encode_rows(_keys_all(model, x, mode, args.chunk)) if use_keys else None
                        logits = forward_general(
                            model, x, mode, chunk=args.chunk,
                            fms=(fms if use_state else None),
                            key_codes=kc, key_fm=(fms[0] if use_keys else None),
                            Mcand=Mcand, rerank=rerank)
                        losses.append(F.cross_entropy(logits, tgt.view(1)).item())
                torch.cuda.synchronize()
                dt = time.time() - t0

                lc = sum(losses) / len(losses)
                ppl = math.exp(min(20.0, lc))
                dpct = (ppl / ppl_exact - 1.0) * 100.0
                pair = [b - a for a, b in zip(losses_exact, losses)]
                dmean = sum(pair) / len(pair)
                dsd = (sum((p - dmean) ** 2 for p in pair) / max(1, len(pair) - 1)) ** 0.5

                bpp_total = (bpp * 2) if (use_state and use_keys) else bpp
                vram10 = bpp_total * 10_000_000 / 1024 ** 2

                retr = {}
                if use_keys:
                    retr = retrieval_metrics(model, toks[ev_base:ev_base + W], mode,
                                             fms[0], kc, W, Mcand, args, d,
                                             rerank=rerank)

                print(f"{vname + ' ' + cname:<28}{bpp_total:>9.1f}{ratio:>6.1f}x"
                      f"{ppl:>10.3f}{dpct:>8.1f}%{vram10:>9.1f}MB", flush=True)

                run["configs"].append({
                    "variant": "keys+rr" if rerank else vname, "cfg": cname, "type": ctype,
                    "rerank": rerank,
                    "levels": Lv, "subvecs": S, "K": args.K,
                    "bytes_per_pos": bpp_total, "ratio": ratio,
                    "loss": lc, "ppl": ppl, "loss_exact": loss_exact,
                    "delta_ppl_pct": dpct,
                    "paired_delta_mean": dmean, "paired_delta_sd": dsd,
                    "vram_10m_mb": vram10, "sec": dt, "fit_sec": t_fit,
                    "retrieval": retr,
                })

        all_out["runs"].append(dict(run))   # копия, иначе все runs будут ссылаться на один dict
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=1)

        print(f"\n  [{W:,}] точный PPL={ppl_exact:.2f}; см. деградацию по столбцам выше "
              f"(парная, на тех же окнах)", flush=True)

    print(f"\nГотово. Результаты: {OUT_JSON}", flush=True)


def _keys_all(model, x, mode, chunk):
    """Полная матрица статичных ключей e (W,d) — строится по чанкам, но материализуется.

    Нужна ТОЛЬКО для вариантов с сжатыми ключами; при больших W это временный (W,d).
    """
    W = x.shape[0]
    out = torch.empty((W, model.d), device=x.device)
    for s in range(0, W, chunk):
        e = min(s + chunk, W)
        out[s:e] = keys_range(model, x, s, e, mode)
    return out


def retrieval_metrics(model, x, mode, fm, key_codes, W, Mcand, args, d, rerank=False):
    """recall@M / final@k / driverCos на первом слое (q = q0), как в прошлом свипе.

    rerank=True: final@k считается по ТОЧНОМУ dot-product восстановленных кандидатов
    (лечит rank-inversion — плохое восстановление топ-8 на 32×, совет коллеги 02.09.2026).
    """
    if W - TAIL < model.topk + 1:
        return {}
    with torch.no_grad():
        nq = int(model.nquery)
        q0 = keys_range(model, x, W - nq, W, mode).mean(0)
        M = _keys_all(model, x, mode, args.chunk)
        Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q0 / (q0.norm() + 1e-6)
        sim_e = Mn @ qn
        sim_e[W - TAIL:] = -1e9
        topk = int(model.topk)
        gt = sim_e.topk(topk).indices

        fm.codes = key_codes
        vals, cn, cand = fm.select(q0, topk, W, Mcand, rerank=rerank)
        hit = len(set(gt.tolist()) & set(cand.tolist())) / topk
        fin = len(set(gt.tolist()) & set((cn - 1).tolist())) / topk

        rec = fm.decode_codes(key_codes[cn])
        sc = (rec / (rec.norm(dim=-1, keepdim=True) + 1e-6)) @ qn
        w = torch.softmax(sc / TEMP, 0)
        drv_c = (w.unsqueeze(-1) * rec).sum(0)
        gi = gt
        we = torch.softmax(sim_e[gi] / TEMP, 0)
        drv_e = (we.unsqueeze(-1) * M[gi]).sum(0)
        dc = F.cosine_similarity(drv_c, drv_e, dim=0).item()
    return {"recall_at_M": hit, "final_at_k": fin, "driver_cos": dc, "rerank": rerank,
            "Mcand": Mcand}


if __name__ == "__main__":
    main()
