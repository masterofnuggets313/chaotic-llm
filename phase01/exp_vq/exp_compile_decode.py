"""torch.compile на реальном decode-пути STS-Prog (GEMM fast-decode).

Проверяем: скомпилированный fast_decode_step vs eager на W=65536/262144.
+ корректность (cos логитов compiled vs eager).
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
    """Один decode-шаг: GEMM-селекция + блоки по K. (как exp6)"""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    for blk in model.blocks:
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        sim = en @ qn.squeeze(0)          # (Wc,) GEMM
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

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(RESULTS, "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    mode = "cyclic"
    o = 5000

    # compile версии
    compiled = torch.compile(gemm_fast_decode, mode="reduce-overhead")
    print(f"torch version: {torch.__version__}")

    for Wc in [65536, 262144]:
        x = toks[o:o + Wc]
        e_all = keys_at(model, x, torch.arange(Wc, device=dev), mode).detach()
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        q0 = e_all[-nq:].mean(0, keepdim=True)

        # корректность: compiled vs eager
        with torch.no_grad():
            lg_eager = gemm_fast_decode(model, en, e_all, q0, Wc, k_eff, topk, nq)
            lg_comp = compiled(model, en, e_all, q0, Wc, k_eff, topk, nq)
            torch.cuda.synchronize()
            cos = float(F.cosine_similarity(lg_eager, lg_comp, dim=-1).item())
            print(f"W={Wc}: cos(compiled, eager) = {cos:.6f}")

        t_eager = timeit(lambda: gemm_fast_decode(model, en, e_all, q0, Wc, k_eff, topk, nq), warmup=2, iters=5)
        t_comp = timeit(lambda: compiled(model, en, e_all, q0, Wc, k_eff, topk, nq), warmup=2, iters=5)
        print(f"W={Wc}: eager {t_eager*1000:6.1f}ms ({1/t_eager:6.2f} tok/s) | "
              f"compiled {t_comp*1000:6.1f}ms ({1/t_comp:6.2f} tok/s) | "
              f"speedup {t_eager/t_comp:.2f}x")


if __name__ == "__main__":
    main()