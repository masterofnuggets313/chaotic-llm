"""torch.compile на ЖИРНОЙ модели d=384 l=12 (3.5M) — есть ли смысл?

Гипотеза: на больших моделях доминирует GEMM/memory-bandwidth (cuBLAS уже
оптимален) -> compile почти не помогает. Проверяем на реальном decode-пути.
"""
import os, sys, time, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, TAIL, TEMP
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 256


def gemm_fast_decode(model, en, e_all, q0, Wc, k_eff, topk, nq, K=256):
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    for blk in model.blocks:
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        sim = en @ qn.squeeze(0)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        h_last = blk(h_last, driver, k)
        hK = blk(hK, driver, k)
        q = q0 + model.query_proj(h_last) * 0.5
    g = hK.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


def timeit(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    dev = "cuda"
    print("Loading...", flush=True)
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    # ЖИРНАЯ модель: d=384, L=12 (~3.5M параметров, как chat_sts_prog)
    d, L = 384, 12
    model = build_pc_model("pc", vocab=V, d=d, layers=L, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: d={d} L={L} params={n_params:,} (~{n_params/1e6:.1f}M)", flush=True)
    for p in model.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    mode = "cyclic"
    o = 5000

    compiled = torch.compile(gemm_fast_decode, mode="reduce-overhead")

    for Wc in [16384, 65536]:
        x = toks[o:o + Wc]
        e_all = keys_at(model, x, torch.arange(Wc, device=dev), mode).detach()
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        q0 = e_all[-nq:].mean(0, keepdim=True)

        with torch.no_grad():
            lg_eager = gemm_fast_decode(model, en, e_all, q0, Wc, k_eff, topk, nq)
            lg_comp = compiled(model, en, e_all, q0, Wc, k_eff, topk, nq)
            torch.cuda.synchronize()
            cos = float(F.cosine_similarity(lg_eager, lg_comp, dim=-1).item())

        t_eager = timeit(lambda: gemm_fast_decode(model, en, e_all, q0, Wc, k_eff, topk, nq), warmup=2, iters=5)
        t_comp = timeit(lambda: compiled(model, en, e_all, q0, Wc, k_eff, topk, nq), warmup=2, iters=5)
        print(f"W={Wc}: cos={cos:.6f} | eager {t_eager*1000:6.1f}ms ({1/t_eager:6.1f} tok/s) | "
              f"compiled {t_comp*1000:6.1f}ms ({1/t_comp:6.1f} tok/s) | speedup {t_eager/t_comp:.2f}x")


if __name__ == "__main__":
    main()