"""speed_decode.py — честный стенд скорости декодинга: STS-Prog vs Transformer (KV-cache).

ЗАЧЕМ: ответить на вопрос «сколько ток/с реально» и где мы относительно трансформера.
Меряем РАЗДЕЛЬНО:
  - prefill  — стоимость приёма контекста W (STS: O(W·d·L), TF: O(W^2·d·L));
  - decode   — мс на ОДИН токен при заполненном окне (главная метрика);
  - VRAM     — пик выделения (у TF растёт с W из-за KV-кэша, у STS — нет состояния).

Модели берутся из готовых чекпоинтов v2-прогона (одинаковый корпус, BPE-512, 6000 шагов):
  STS-Prog            -> results/ckpts/sts_prog_seed0.pt             (d=192, L=8, 900 353)
  Transformer-matched -> results/ckpts/transformer_matched__d_92__seed0.pt (D=92, L=8, 940 568)

ЗАПУСК:
  py -3.13 speed_decode.py --W 4096,16384,65536 --steps 8
  py -3.13 speed_decode.py --W 262144 --steps 4 --modes ref,fast,sub
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)

import final_benchmark as fb                                    # noqa: E402
from models_pc import build_pc_model                            # noqa: E402
from night_task5_fracode_forward import TAIL, TEMP              # noqa: E402

RESULTS = os.path.join(REPO, "results")
CKPT = os.path.join(RESULTS, "ckpts")


# ------------------------------------------------------------------ утилиты
def ms(fn, warmup=2, iters=6):
    """Медианное время одного вызова, мс (CUDA sync вокруг пачки)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def ms_split(fn, warmup=2, iters=6):
    """(wall_ms, gpu_ms). Если wall >> gpu — мы уперлись в CPU/launch-overhead,
    а не в железо. Это решается CUDA-графом, а не математикой."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1000.0   # ВКЛЮЧАЯ досушивание GPU
    return wall, ev0.elapsed_time(ev1) / iters


def capture_graph(fn, warmup=3):
    """Захват CUDA-графа: убирает ВСЮ стоимость запуска ядер из Python."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def peak_mb():
    return round(torch.cuda.max_memory_allocated() / (1 << 20), 1)


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------ загрузка
def load_sts(dev, seed=0, d=192, layers=8, vocab=512):
    m = build_pc_model("pc", vocab=vocab, d=d, layers=layers, k_init=1.2,
                       sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP)
    sd = torch.load(os.path.join(CKPT, f"sts_prog_seed{seed}.pt"), map_location="cpu")
    m.load_state_dict(sd)
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_tf(dev, seed=0, D=92, layers=8, heads=4, vocab=512, W=256):
    m = fb.TransformerLM(vocab, W, D=D, HEADS=heads, LAYERS=layers)
    sd = torch.load(os.path.join(CKPT, f"transformer_matched__d_{D}__seed{seed}.pt"),
                    map_location="cpu")
    m.load_state_dict(sd)
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def token_ids(n, vocab, dev, seed=12345):
    """Реальные токены из публичного корпуса, если он есть; иначе детерминированные случайные."""
    path = os.path.join(PHASE, "corpus_public.txt")
    if os.path.exists(path):
        txt = fb.load_chars(path, None)
        tok = fb.make_bpe(txt)
        ids = tok.encode(txt).ids
        if len(ids) >= n:
            return torch.tensor(ids[:n], dtype=torch.long, device=dev)
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g, dtype=torch.long, device=dev)


# ------------------------------------------------------------------ STS: состояние
class STSState:
    """Окно токенов + буферы. Позиции ОТНОСИТЕЛЬНЫЕ (0..W-1), как при обучении.

    Ключи e = embed(ids) + pos[i % 256] — при сдвиге окна на 1 токен ВСЕ позиции
    сдвигаются, поэтому e пересчитывается каждый шаг (это 3 прохода по (W,d)).
    """

    def __init__(self, model, Wc, dev, ids):
        self.m = model
        self.W = Wc
        self.dev = dev
        self.d = model.d
        P = model.pos.shape[1]
        idx = torch.arange(Wc, device=dev) % P
        self.POSWIN = model.pos[0][idx].contiguous()          # (W, d) — константа
        self.emb_w = model.embed.weight                        # (V, d)
        self.buf = torch.empty(2 * Wc, dtype=torch.long, device=dev)
        self.buf[:Wc] = ids
        self.buf[Wc:2 * Wc] = ids
        self.start = 0
        self.e = torch.empty(Wc, self.d, device=dev)
        self.en = torch.empty(Wc, self.d, device=dev)
        self.k_eff = torch.sigmoid(model.k)
        self.pending = 0

    def push(self, tok):
        """Добавить токен в конец окна (кольцевой буфер, копия раз в W шагов)."""
        W = self.W
        self.start += 1
        if self.start == W:                     # амортизированный сдвиг
            self.buf[:W] = self.buf[W:2 * W]
            self.start = 0
        self.buf[self.start + W - 1] = tok
        self.buf[self.start + 2 * W - 1] = tok

    def window(self):
        return self.buf[self.start:self.start + self.W]

    def compute_keys(self):
        torch.add(F.embedding(self.window(), self.emb_w), self.POSWIN, out=self.e)
        return self.e


# ------------------------------------------------------------------ STS: режимы
@torch.no_grad()
def sts_ref_step(st, chunk=131072):
    """REFERENCE: точная копия логики diag_decode_profile.full_step (медленная, честная база)."""
    m, W, TAIL_ = st.m, st.W, TAIL
    e_all = st.compute_keys()
    k_eff = torch.sigmoid(m.k)
    topk, nq = int(m.topk), int(m.nquery)

    pos_q = e_all[W - nq:].mean(0, keepdim=True)
    q = pos_q
    h = e_all
    for blk in m.blocks:
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)      # <-- 8 раз!
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)                                     # <-- материализация (W,d)
        sim[W - TAIL_:] = -1e9
        kk = min(topk, W - TAIL_)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        hc = [blk(h[s:s + chunk], driver, k_eff) for s in range(0, W, chunk)]
        h = torch.cat(hc, 0)
        q = pos_q + m.query_proj(h[-1].unsqueeze(0)) * 0.5
    g = h.mean(0, keepdim=True)
    return m.readout3(torch.cat([h[-1], pos_q[0], g[0]]).unsqueeze(0))


@torch.no_grad()
def sts_fast_step(st, fused_buf=True):
    """FAST: та же математика, но
       - en считается ОДИН раз на шаг (в ref — 8 раз);
       - sim как matvec (без материализации (W,d) произведения);
       - прогон блоков по возможности без лишних аллокаций.
       Побитого совпадения нет (другой порядок редукций), логиты совпадают с точностью fp32.
    """
    m, W, TAIL_ = st.m, st.W, TAIL
    e = st.compute_keys()
    k_eff = st.k_eff
    topk, nq = int(m.topk), int(m.nquery)

    # en — один раз
    nrm = torch.linalg.vector_norm(e, dim=-1, keepdim=True).add_(1e-6)
    torch.div(e, nrm, out=st.en)

    q0 = e[W - nq:].mean(0)                       # (d,)
    q = q0
    drivers = []
    hlast = e[W - 1].unsqueeze(0).clone()         # траектория последней позиции (1,d)
    for blk in m.blocks:
        qn = q / (q.norm() + 1e-6)
        sim = torch.mv(st.en, qn)                 # (W,) — matvec, не (W,d) произведение
        sim[W - TAIL_:] = -1e9
        kk = min(topk, W - TAIL_)
        vals, loc = torch.topk(sim, kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e[nxt]).sum(0, keepdim=True)
        drivers.append(driver)
        hlast = blk(hlast, driver, k_eff)
        q = q0 + m.query_proj(hlast)[0] * 0.5

    # полный прогон по всему окну — нужен только ради g = mean(h)
    h = e
    for blk, drv in zip(m.blocks, drivers):
        h = blk(h, drv, k_eff)
    g = h.mean(0)
    return m.readout3(torch.cat([h[W - 1], q0, g]).unsqueeze(0))


@torch.no_grad()
def sts_sub_step(st, S=8192, idx=None):
    """SUB: g оценивается по ФИКСИРОВАННОЙ подвыборке позиций (Монте-Карло).

    Тождество h_L[i] = Phi(e[i]; D) (блоки построчные, драйвер общий) => чтобы
    получить h_L[i] для ЛЮБОГО подмножества позиций, весь W прогонять не нужно.
    Прогоняем только подвыборку S + последнюю позицию.
    """
    m, W, TAIL_ = st.m, st.W, TAIL
    e = st.compute_keys()
    k_eff = st.k_eff
    topk, nq = int(m.topk), int(m.nquery)

    nrm = torch.linalg.vector_norm(e, dim=-1, keepdim=True).add_(1e-6)
    torch.div(e, nrm, out=st.en)

    q0 = e[W - nq:].mean(0)
    q = q0
    drivers = []
    hlast = e[W - 1].unsqueeze(0).clone()
    for blk in m.blocks:
        qn = q / (q.norm() + 1e-6)
        sim = torch.mv(st.en, qn)
        sim[W - TAIL_:] = -1e9
        kk = min(topk, W - TAIL_)
        vals, loc = torch.topk(sim, kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e[nxt]).sum(0, keepdim=True)
        drivers.append(driver)
        hlast = blk(hlast, driver, k_eff)
        q = q0 + m.query_proj(hlast)[0] * 0.5

    # прогон только по подвыборке + последняя позиция
    if idx is None:
        idx = st.sub_idx
    sub = e[idx]                                   # (S, d)
    for blk, drv in zip(m.blocks, drivers):
        sub = blk(sub, drv, k_eff)
    g = sub.mean(0)
    h_last = hlast[0]
    return m.readout3(torch.cat([h_last, q0, g]).unsqueeze(0))


# ------------------------------------------------------------------ Transformer: KV-cache
class TFKV:
    """Стандартный инкрементальный трансформер: KV-кэш + SDPA.

    Проекции считаем вручную (in_proj_weight/out_proj из nn.MultiheadAttention),
    чтобы можно было (а) писать K/V сразу в кэш и (б) использовать SDPA вместо
    материализации матрицы W x W. Позиции циклические (i % 256) — иначе модель,
    обученная на W=256, за пределами окна не определена.
    """

    def __init__(self, model, Wc, dev, dtype=torch.float32):
        self.m = model
        self.W = Wc
        self.dev = dev
        self.dtype = dtype
        self.L = len(model.blocks)
        self.D = model.blocks[0].attn.embed_dim
        self.H = model.blocks[0].attn.num_heads
        self.Dh = self.D // self.H
        P = model.pos.shape[1]
        self.postab = model.pos[0][torch.arange(Wc, device=dev) % P].contiguous()
        self.k = [torch.empty(1, Wc, self.D, device=dev, dtype=dtype) for _ in range(self.L)]
        self.v = [torch.empty(1, Wc, self.D, device=dev, dtype=dtype) for _ in range(self.L)]
        self.T = 0

    def _x(self, ids, t0):
        T = ids.shape[0]
        return (self.m.embed(ids) + self.postab[t0:t0 + T]).unsqueeze(0).to(self.dtype)

    def _split(self, qkv, T):
        q, k, v = qkv.chunk(3, dim=-1)
        sh = lambda t: t.view(1, T, self.H, self.Dh).transpose(1, 2)
        return sh(q), sh(k), sh(v)

    @torch.no_grad()
    def prefill(self, ids):
        T = ids.shape[0]
        x = self._x(ids, 0)
        for li, blk in enumerate(self.m.blocks):
            attn = blk.attn
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = self._split(qkv, T)
            self.k[li][:, :T] = k.transpose(1, 2).reshape(1, T, self.D)
            self.v[li][:, :T] = v.transpose(1, 2).reshape(1, T, self.D)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(1, T, self.D)
            a = attn.out_proj(a)
            x = blk.norm1(x + a)
            x = blk.norm2(x + blk.ffn(x))
        self.T = T
        return self.m.head(x[:, -1, :].float())

    @torch.no_grad()
    def step(self, tok):
        T = self.T
        x = self._x(tok, T % self.W)
        for li, blk in enumerate(self.m.blocks):
            attn = blk.attn
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = self._split(qkv, 1)
            self.k[li][:, T:T + 1] = k.transpose(1, 2).reshape(1, 1, self.D)
            self.v[li][:, T:T + 1] = v.transpose(1, 2).reshape(1, 1, self.D)
            K = self.k[li][:, :T + 1].view(1, T + 1, self.H, self.Dh).transpose(1, 2)
            V = self.v[li][:, :T + 1].view(1, T + 1, self.H, self.Dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, K, V, is_causal=False)
            a = a.transpose(1, 2).reshape(1, 1, self.D)
            a = attn.out_proj(a)
            x = blk.norm1(x + a)
            x = blk.norm2(x + blk.ffn(x))
        self.T = T + 1
        return self.m.head(x[:, -1, :].float())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", default="4096,16384,65536")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--modes", default="ref,fast,sub")
    ap.add_argument("--S", type=int, default=8192, help="размер подвыборки для режима sub")
    ap.add_argument("--tf", type=int, default=1)
    ap.add_argument("--tf-dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "speed_decode.json"))
    args = ap.parse_args()

    dev = "cuda"
    Ws = [int(x) for x in args.W.split(",")]
    modes = args.modes.split(",")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    sts = load_sts(dev)
    sts.k_eff = torch.sigmoid(sts.k)
    V = sts.embed.num_embeddings
    print(f"STS-Prog: d={sts.d} L={len(sts.blocks)} V={V} "
          f"params={sum(p.numel() for p in sts.parameters()):,}")

    tfm = None
    if args.tf:
        tfm = load_tf(dev, W=256)
        print(f"Transformer-matched: D={tfm.blocks[0].attn.embed_dim} L={len(tfm.blocks)} "
              f"params={sum(p.numel() for p in tfm.parameters()):,}")

    all_ids = token_ids(max(Ws) + 4096, V, dev)
    print(f"корпус токенов: {all_ids.shape[0]:,}\n")

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "sts_params": sum(p.numel() for p in sts.parameters()),
           "tf_params": (sum(p.numel() for p in tfm.parameters()) if tfm else None),
           "S": args.S, "steps": args.steps, "rows": []}

    for W in Ws:
        row = {"W": W}
        ids = all_ids[:W]
        print(f"===== W = {W:,} =====")

        def report(tag, fn, key, extra=""):
            wall, gpu = ms_split(fn, iters=args.steps)
            row[f"{key}_ms"] = round(wall, 2)
            row[f"{key}_gpu_ms"] = round(gpu, 2)
            row[f"{key}_tok_s"] = round(1000.0 / wall, 2)
            cpu = max(0.0, wall - gpu)
            print(f"  {tag:9s}: {wall:8.2f} мс/ток  {1000/wall:8.2f} tok/s   "
                  f"[gpu {gpu:7.2f} + cpu/overhead {cpu:7.2f}]{extra}")
            return wall

        # ---------------- STS ----------------
        if "ref" in modes:
            st = STSState(sts, W, dev, ids)
            reset_peak()
            report("STS ref", lambda: sts_ref_step(st), "sts_ref")
            row["sts_vram_mb"] = peak_mb()
            del st
            torch.cuda.empty_cache()

        if "fast" in modes:
            st = STSState(sts, W, dev, ids)
            reset_peak()
            report("STS fast", lambda: sts_fast_step(st), "sts_fast")
            row.setdefault("sts_vram_mb", peak_mb())
            del st
            torch.cuda.empty_cache()

        # fast + CUDA-graph: убирает стоимость запуска ядер
        if "graph" in modes:
            st = STSState(sts, W, dev, ids)
            reset_peak()
            try:
                g, _ = capture_graph(lambda: sts_fast_step(st))
                report("STS graph", g.replay, "sts_graph")
            except RuntimeError as e:
                row["sts_graph_error"] = str(e)[:200]
                print(f"  STS graph: ОШИБКА — {str(e)[:140]}")
            del st
            torch.cuda.empty_cache()

        if "sub" in modes or "subgraph" in modes:
            st = STSState(sts, W, dev, ids)
            S = min(args.S, max(1, W // 2))
            gg = torch.Generator().manual_seed(12345)
            st.sub_idx = torch.randint(0, max(1, W - TAIL), (S,), generator=gg).to(dev)

            if "sub" in modes:
                reset_peak()
                report("STS sub", lambda: sts_sub_step(st, S=S), "sts_sub", f"   (S={S})")
                row["sts_sub_S"] = S
            if "subgraph" in modes:
                reset_peak()
                try:
                    g2, _ = capture_graph(lambda: sts_sub_step(st, S=S))
                    report("STS sub+gr", g2.replay, "sts_subgraph", f"   (S={S})")
                    row["sts_subgraph_S"] = S
                except RuntimeError as e:
                    row["sts_subgraph_error"] = str(e)[:200]
                    print(f"  STS sub+gr: ОШИБКА — {str(e)[:140]}")
            del st
            torch.cuda.empty_cache()

        # ---------------- Transformer ----------------
        if tfm is not None:
            dtype = torch.float16 if args.tf_dtype == "fp16" else torch.float32
            kv = TFKV(tfm, W + args.steps + 8, dev, dtype=dtype)
            try:
                reset_peak()
                t0 = time.perf_counter()
                kv.prefill(ids)
                torch.cuda.synchronize()
                tpre = (time.perf_counter() - t0) * 1000.0
                tok = all_ids[W:W + 1]
                row["tf_prefill_ms"] = round(tpre, 2)
                row["tf_vram_mb"] = peak_mb()
                report("TF kv", lambda: kv.step(tok), "tf")
                row["tf_dtype"] = args.tf_dtype
                print(f"             prefill {tpre:9.1f} мс   VRAM {row['tf_vram_mb']} МБ")
            except RuntimeError as e:
                row["tf_error"] = str(e)[:200]
                print(f"  TF kv    : ОШИБКА — {str(e)[:160]}")
            del kv
            torch.cuda.empty_cache()

        out["rows"].append(row)
        print()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
