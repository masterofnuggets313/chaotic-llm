"""Эксперимент 8: torch.compile на fast-decode для fused kernel launch.

Идея: 76% времени STS-шага — kernel launch overhead (8 слоёв × cos + topk + gather
+ gather + blocks = ~40 отдельных kernel вызовов). torch.compile может fuser-овать
их в один compiled graph, убрав overhead.

torch.compile(fast_decode_step, mode="reduce-overhead") — триггерит Triton fusion.
"""
import os, sys, time, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, forward_general, TAIL, TEMP
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 256


def fast_decode_step_nojit(model, en, e_all, q0, Wc, topk, nq, K):
    """Базовый fast-decode с GEMM-селекцией."""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    k_eff = torch.sigmoid(model.k)
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


# Версия с @torch.compile
@torch.compile(mode="reduce-overhead", fullgraph=True)
def fast_decode_step_JIT(model, en, e_all, q0, Wc, topk, nq, K):
    return fast_decode_step_nojit(model, en, e_all, q0, Wc, topk, nq, K)


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
    print("Loading model...", flush=True)
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

    # --- проверим качество (JIT не меняет математику) ---
    # (только замер скорости)

    from exp_decode_vs_transformer import TransformerKV
    from parametric_models import TransformerLM
    from match_transformer import pick_tf_dims

    MAX_W = 262144
    D_tf = pick_tf_dims(900_000, V, 512, layers=8, heads=4)
    D_tf = max(4, (D_tf // 4) * 4)
    tf = TransformerLM(V, 512, D=D_tf, HEADS=4, LAYERS=8).to(dev).eval()
    n_pos = tf.pos.numel()
    tf.pos = torch.nn.Parameter(torch.zeros(1, MAX_W, D_tf, device=dev)); tf.pos.requires_grad_(False)
    for p in tf.parameters(): p.requires_grad_(False)
    tfkv = TransformerKV(tf)

    print(f"\nЗамер скорости JIT vs no-JIT vs TF KV (K={K}):", flush=True)
    o = 5000
    results = {}
    for Wc in [16384, 65536, 262144]:
        x = toks[o:o + Wc]
        e_all = keys_at(model, x, torch.arange(Wc, device=dev), mode).detach()
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        q0 = e_all[-nq:].mean(0, keepdim=True)

        # no-JIT
        t_nojit = timeit(lambda: fast_decode_step_nojit(model, en, e_all, q0, Wc, topk, nq, K),
                         warmup=2, iters=5)
        nojit_tps = 1 / t_nojit

        # JIT (первый вызов — compilation, не засекаем)
        _ = fast_decode_step_JIT(model, en, e_all, q0, Wc, topk, nq, K)
        torch.cuda.synchronize()
        t_jit = timeit(lambda: fast_decode_step_JIT(model, en, e_all, q0, Wc, topk, nq, K),
                       warmup=2, iters=5)
        jit_tps = 1 / t_jit

        # TF KV
        k_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        v_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        lengths = [Wc - 1] * 8
        t_tf = timeit(lambda: tfkv.decode_one(x[Wc - 1].view(1), k_cache, v_cache, Wc - 1, lengths),
                      warmup=2, iters=5)
        tf_tps = 1 / t_tf

        print(f"W={Wc:>7}: no-JIT {t_nojit*1000:6.1f} ms / {nojit_tps:6.2f} | "
              f"JIT {t_jit*1000:6.1f} ms / {jit_tps:6.2f} | "
              f"TF {t_tf*1000:6.1f} ms / {tf_tps:6.2f} | "
              f"JIT/noJIT={jit_tps/nojit_tps:.2f}x  JIT/TF={jit_tps/tf_tps:.2f}x")
        results[Wc] = {"nojit_ms": round(t_nojit * 1000, 1), "nojit_tok_s": round(nojit_tps, 1),
                       "jit_ms": round(t_jit * 1000, 1), "jit_tok_s": round(jit_tps, 1),
                       "tf_ms": round(t_tf * 1000, 1), "tf_tok_s": round(tf_tps, 1),
                       "jit_speedup": round(jit_tps / nojit_tps, 2),
                       "jit_over_tf": round(jit_tps / tf_tps, 2)}

    with open(os.path.join(RESULTS, "exp_speed_compile.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved: results/exp_speed_compile.json")


if __name__ == "__main__":
    main()