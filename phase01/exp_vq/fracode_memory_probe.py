"""fracode_memory_probe.py — Fracode Memory Layer: сжатие памяти STS-Prog.

ИДЕЯ (от коллеги Германа): сжимать не МОДЕЛЬ, а ПАМЯТЬ (ключевую таблицу (W,d)),
иерархически, с прогрессивным восстановлением.

КРИТИЧЕСКОЕ ОТЛИЧИЕ ОТ «просто сжатия»:
  Сжатие хранилища != экономия VRAM. Если ради top-k нужно развернуть всю (W,d) обратно,
  VRAM остаётся 7.3GB. Экономия есть ТОЛЬКО при поиске В СЖАТОМ ДОМЕНЕ (ADC) с
  восстановлением лишь отобранных кандидатов. Это и проверяем.

МАППИНГ Fracode -> память модели:
  Library  (общая библиотека генераторов) -> кодбук центроидов (обучается на корпусе)
  Instructor (иерархическое дерево сборки) -> коды (уровень l, подвектор s)
  Прогрессивная реконструкция            -> x ~= sum_l decode_l(codes_l)  (остаточные уровни)
  O(1) random access                     -> код позиции независим, декод позиции без остальных
  Генератор (детерминированная функция)  -> центроид (детерминированный, воспроизводим)

ЧЕСТНО ПРО ПРИОРИТЕТ: ядровой механизм (кодбук + ADC-поиск) = product quantization
(Jegou et al., 2011 / FAISS). Это PRIOR ART, не новизна. Fracode-вклад, который здесь
ПРОВЕРЯЕТСЯ, — иерархическое остаточное уточнение (L>1) при РАВНОМ бюджете байт.
Базовые линии обязательны: fp16, int8, flat-PQ (L=1). Бить надо их, а не «ничего»
(урок Fracode Phase 0: они проиграли zstd+dict, неверно измерив конкурента).
"""
import os, sys, math, time, argparse
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE)
from models_pc import build_pc_model

D_PE = 32
FP32_BYTES = 4


# ---------------------------------------------------------------- позиции
def sinusoid(idx, d_pe):
    half = d_pe // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=idx.device) / half))
    ang = idx.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


# ---------------------------------------------------------------- k-means
def kmeans(X, K, iters=15, seed=0, chunk=65536):
    """Lloyd's на GPU через matmul-трюк (без (N,K,D) тензоров). X: (N,D)."""
    N, D_ = X.shape
    dev = X.device
    g = torch.Generator(device=dev).manual_seed(seed)
    C = X[torch.randperm(N, generator=g, device=dev)[:K]].clone()
    Cn2 = C.pow(2).sum(1)
    for _ in range(iters):
        assign = torch.empty(N, dtype=torch.long, device=dev)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Xs = X[s:e]
            d2 = Xs.pow(2).sum(1, keepdim=True) - 2.0 * (Xs @ C.T) + Cn2.unsqueeze(0)
            assign[s:e] = d2.argmin(1)
        sums = torch.zeros(K, D_, device=dev).index_add_(0, assign, X)
        cnts = torch.zeros(K, device=dev).index_add_(0, assign, torch.ones(N, device=dev))
        nz = cnts > 0
        C[nz] = sums[nz] / cnts[nz].unsqueeze(1)
        Cn2 = C.pow(2).sum(1)
    return C


def assign_codes(X, C, chunk=65536):
    """Ближайший центроид для каждой строки X: (N,D) -> (N,) long."""
    N = X.shape[0]
    dev = X.device
    out = torch.empty(N, dtype=torch.long, device=dev)
    Cn2 = C.pow(2).sum(1)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        Xs = X[s:e]
        d2 = Xs.pow(2).sum(1, keepdim=True) - 2.0 * (Xs @ C.T) + Cn2.unsqueeze(0)
        out[s:e] = d2.argmin(1)
    return out


# ---------------------------------------------------------------- Fracode Memory
class FracodeMemory:
    """Иерархическое (матрёшка) кодирование памяти + поиск в сжатом домене.

    levels=L, subvecs=S, K центроидов. Бюджет байт на позицию = L*S*log2(K)/8.
    Восстановление: x ~= sum_l decode(codes[l]).
    """

    def __init__(self, d, levels, subvecs, K, device="cuda"):
        assert d % subvecs == 0, f"d={d} не делится на subvecs={subvecs}"
        self.d, self.L, self.S, self.K = d, levels, subvecs, K
        self.sub = d // subvecs
        self.device = device
        self.bits_per_pos = levels * subvecs * int(round(math.log2(K)))
        self.bytes_per_pos = self.bits_per_pos / 8.0
        self.cbooks = None   # [L][S] -> (K, sub)
        self.codes = None    # (W, L, S) long

    # ---- обучение ----
    def fit(self, M, iters=15, seed=0, sample=100_000):
        W = M.shape[0]
        n = min(sample, W)
        idx = torch.randperm(W, generator=torch.Generator(device=M.device).manual_seed(seed),
                             device=M.device)[:n]
        R = M[idx].clone()                      # текущий остаток
        self.cbooks = [[None] * self.S for _ in range(self.L)]
        for l in range(self.L):
            for s in range(self.S):
                sub_x = R[:, s * self.sub:(s + 1) * self.sub].contiguous()
                self.cbooks[l][s] = kmeans(sub_x, self.K, iters=iters, seed=seed + l * 977 + s)
            R = R - self._decode_level(self._assign_level(R, l), l, R.shape[0])
        return self

    def _assign_level(self, R, l):
        """Коды уровня l для остатка R: (N,d) -> (N,S) long."""
        N = R.shape[0]
        codes = torch.empty((N, self.S), dtype=torch.long, device=self.device)
        for s in range(self.S):
            sub_x = R[:, s * self.sub:(s + 1) * self.sub].contiguous()
            codes[:, s] = assign_codes(sub_x, self.cbooks[l][s])
        return codes

    def _decode_level(self, codes_l, l, W):
        """Вклад уровня l в восстановление: codes_l (W,S) -> (W,d)."""
        out = torch.zeros((W, self.d), device=self.device)
        for s in range(self.S):
            out[:, s * self.sub:(s + 1) * self.sub] = self.cbooks[l][s][codes_l[:, s]]
        return out

    # ---- кодирование ----
    def encode(self, M):
        W = M.shape[0]
        self.codes = torch.empty((W, self.L, self.S), dtype=torch.long, device=M.device)
        R = M
        for l in range(self.L):
            self.codes[:, l, :] = self._assign_level(R, l)
            R = R - self._decode_level(self.codes[:, l, :], l, W)
        return self.codes

    def decode(self, idx=None):
        codes = self.codes if idx is None else self.codes[idx]
        W = codes.shape[0]
        out = torch.zeros((W, self.d), device=self.device)
        for l in range(codes.shape[1]):
            out += self._decode_level(codes[:, l, :], l, W)
        return out

    # ---- поиск в сжатом домене (ADC) ----
    def adc_scores(self, q, chunk=262144):
        """q: (d,) или (Q,d) -> (Q, W) приближённые скалярные произведения."""
        single = (q.dim() == 1)
        Q = (q.unsqueeze(0) if single else q).to(self.device)
        nq, W = Q.shape[0], self.codes.shape[0]
        out = torch.zeros((nq, W), device=self.device)
        for l in range(self.L):
            for s in range(self.S):
                # таблица: (nq, K) = Q_sub @ C^T
                qsub = Q[:, s * self.sub:(s + 1) * self.sub]
                tbl = qsub @ self.cbooks[l][s].T          # (nq, K)
                c = self.codes[:, l, s]                    # (W,)
                out += tbl[:, c]                           # (nq, W) — векторизованный ADC
        return out[0] if single else out

    def query(self, q, Mcand, k, exclude=None):
        """ADC-скрининг top-Mcand -> точный переранк -> top-k индексы."""
        sc = self.adc_scores(q)
        if exclude is not None:
            sc[exclude] = -1e18
        cand = sc.topk(min(Mcand, sc.numel()), dim=0).indices        # (Mcand,)
        rec = self.decode(cand)                                       # (Mcand, d)
        qn = q / (q.norm() + 1e-8)
        rn = rec / (rec.norm(dim=-1, keepdim=True) + 1e-8)
        exact = rn @ qn
        order = exact.topk(min(k, len(cand)), dim=0).indices
        return cand[order], cand


# ---------------------------------------------------------------- базовые линии
def baseline_quantize(M, mode):
    if mode == "fp16":
        return M.half().float(), 2.0
    if mode == "int8":
        lo, hi = M.min(), M.max()
        scale = (hi - lo) / 255.0 + 1e-12
        z = ((M - lo) / scale).round().clamp(0, 255)
        return z * scale + lo, 4.0
    return M, 1.0


# ---------------------------------------------------------------- главный прогон
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=262_144)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--m-cand", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(0)

    # ---- реальная обученная модель ----
    ck = os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt")
    model = build_pc_model("pc", vocab=512, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3)
    sd = torch.load(ck, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.d
    V = 512
    topk = int(model.topk)
    pe_proj = nn.Linear(D_PE, d).to(dev)          # производные позиции (как в probe)
    for p in pe_proj.parameters():
        p.requires_grad_(False)
    print(f"Обученный STS-Prog загружен: {ck}", flush=True)
    print(f"d={d} topk={topk} W={args.W:,}", flush=True)

    # ---- реальный текст -> токены ----
    corpus = os.path.join(HERE, "corpus_nl", "tinystories.txt")
    if os.path.exists(corpus):
        with open(corpus, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(args.W * 2)
        toks = torch.tensor([ord(c) % V for c in text[:args.W + 64]], device=dev)
        src = "tinystories (реальный текст)"
    else:
        toks = torch.randint(0, V, (args.W + 64,), device=dev)
        src = "SYNTHETIC"
    x = toks[:args.W]

    # ---- память M (ключи, как их видит STS-Prog) ----
    idx_all = torch.arange(args.W, device=dev)
    with torch.no_grad():
        M = model.embed(x) + pe_proj(sinusoid(idx_all, D_PE).to(dev))
    M = M.contiguous()
    fp32_bytes = args.W * d * FP32_BYTES
    print(f"Память построена из {src}: M=({args.W:,}, {d}) = {fp32_bytes/1024**2:.1f} MB fp32", flush=True)

    # ---- запросы: как их строит модель (q0 = mean последних nq токенов) ----
    nq = int(model.nquery)
    starts = torch.randperm(args.W - 200, generator=torch.Generator(device=dev).manual_seed(1),
                            device=dev)[:args.queries] + 100
    Q, excl = [], []
    with torch.no_grad():
        for p in starts.tolist():
            j = torch.arange(p - nq, p, device=dev)
            q = (model.embed(toks[j]) + pe_proj(sinusoid(j, D_PE).to(dev))).mean(0)
            Q.append(q)
            excl.append(j)
    Q = torch.stack(Q)

    # ---- эталон: точный top-k по полной памяти ----
    Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-8)
    Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
    SIM = Qn @ Mn.T                                # (nq, W) точные косинусы
    for i, j in enumerate(excl):
        SIM[i, j] = -1e18
    gt = SIM.topk(topk, dim=1).indices             # (nq, topk) эталон

    def recall(cand_idx):
        hits = 0
        for i in range(cand_idx.shape[0]):
            hits += len(set(gt[i].tolist()) & set(cand_idx[i].tolist()))
        return hits / (cand_idx.shape[0] * topk)

    # ---- базовые линии ----
    print("\n" + "=" * 78, flush=True)
    print("БАЗОВЫЕ ЛИНИИ (всё ещё хранят (W,d) в VRAM — экономят ТОЛЬКО хранилище)", flush=True)
    print("=" * 78, flush=True)
    for mode in ["exact", "fp16", "int8"]:
        Mq, ratio = baseline_quantize(M, mode) if mode != "exact" else (M, 1.0)
        Mqn = Mq / (Mq.norm(dim=-1, keepdim=True) + 1e-8)
        Sq = Qn @ Mqn.T
        for i, j in enumerate(excl):
            Sq[i, j] = -1e18
        cand = Sq.topk(topk, dim=1).indices
        r = recall(cand)
        vram10 = (d * FP32_BYTES / ratio) * 10_000_000 / 1024 ** 2
        print(f"  {mode:>6}: ratio {ratio:>5.1f}x  bytes/pos {d*FP32_BYTES/ratio:>7.1f}  "
              f"recall@{topk} {r*100:5.1f}%  VRAM@10M {vram10:7.1f} MB",
              flush=True)

    # ---- Fracode Memory: свип ----
    configs = []
    # (label, levels, subvecs) — подбираем бюджеты байт
    for S in [192, 96, 48, 24, 12, 6]:
        configs.append((f"flat PQ  L=1 S={S:>3}", 1, S))
    for S in [96, 48, 24, 12]:
        configs.append((f"FRACODE  L=2 S={S:>3}", 2, S))
    for S in [48, 24, 12]:
        configs.append((f"FRACODE  L=3 S={S:>3}", 3, S))

    print("\n" + "=" * 78, flush=True)
    print("FRACODE MEMORY (поиск В СЖАТОМ ДОМЕНЕ — VRAM реально экономится)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'конфиг':<22}{'bytes/pos':>10}{'ratio':>8}{'recall@M':>10}"
          f"{'final@k':>9}{'driverCos':>10}{'VRAM@10M':>11}", flush=True)
    print("-" * 88, flush=True)

    results = []
    for label, L, S in configs:
        torch.cuda.empty_cache()
        t0 = time.time()
        fm = FracodeMemory(d, levels=L, subvecs=S, K=args.K, device=dev)
        fm.fit(M, iters=args.iters, seed=0)
        codes = fm.encode(M)
        torch.cuda.synchronize()
        t_fit = time.time() - t0

        cands, candsets, dcos = [], [], []
        t1 = time.time()
        for i in range(Q.shape[0]):
            top, candset = fm.query(Q[i], args.m_cand, topk, exclude=excl[i])
            cands.append(top); candsets.append(candset)
            # КАК МОДЕЛЬ ЭТО ЕСТ: driver = взвешенная сумма top-k. Сравниваем с точным.
            rec_sel = fm.decode(top)
            sc_sel = (rec_sel / (rec_sel.norm(dim=-1, keepdim=True) + 1e-8)) @ Qn[i]
            w = torch.softmax(sc_sel / 0.3, dim=0)
            drv_c = (w.unsqueeze(-1) * rec_sel).sum(0)
            gi = gt[i]
            w_e = torch.softmax(SIM[i, gi] / 0.3, dim=0)
            drv_e = (w_e.unsqueeze(-1) * M[gi]).sum(0)
            dcos.append(torch.nn.functional.cosine_similarity(drv_c, drv_e, dim=0).item())
        torch.cuda.synchronize()
        t_q = time.time() - t1

        r_adc = recall(torch.stack(candsets))
        r_fin = recall(torch.stack(cands))
        dc = sum(dcos) / len(dcos)
        vram10 = fm.bytes_per_pos * 10_000_000 / 1024 ** 2
        ratio = (d * FP32_BYTES) / fm.bytes_per_pos
        print(f"{label:<22}{fm.bytes_per_pos:>10.1f}{ratio:>8.1f}x"
              f"{r_adc*100:>9.1f}%{r_fin*100:>8.1f}%{dc:>10.3f}{vram10:>9.1f} MB", flush=True)
        results.append({"label": label, "L": L, "S": S, "K": args.K,
                        "bytes_per_pos": fm.bytes_per_pos, "ratio": ratio,
                        "recall_adc": r_adc, "recall_final": r_fin, "driver_cos": dc,
                        "vram_10m_mb": vram10, "fit_s": t_fit, "query_s": t_q})

    print("\nrecall@M = доля истинного top-k, попавшая в ADC-кандидаты (скрининг)")
    print(f"final@k  = качество ПОСЛЕ точного переранка (M={args.m_cand} кандидатов)", flush=True)
    print("\nЧЕСТНО: механизм кодбук+ADC = product quantization (Jegou 2011, prior art).")
    print("Fracode-вклад = иерархия L>1: сравни L=2 против flat при РАВНОМ bytes/pos.", flush=True)


if __name__ == "__main__":
    main()
