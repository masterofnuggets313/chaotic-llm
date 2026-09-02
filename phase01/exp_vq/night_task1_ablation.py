"""night_task1_ablation.py — Task 1 of the overnight run: β-prior ablation.

Scientific question (from colleague feedback): is the PPL win of STS-Prog coming
from the architecture or from the β-prior (order-3 n-gram) gating at eval time?
STS-Prog was described as "CAM without compression capacity" — so we test how
much each prior regime contributes.

This is an EVAL-ONLY study: it reloads the EXISTING trained checkpoints
(results/ckpts/sts_prog_seed{0..4}.pt and sts_prog__no_pc__seed{0..4}.pt) and
recomputes gated PPL under 4 prior regimes:

  - none    : pure model PPL (beta = 0) — the architecture alone
  - order3  : adaptive beta = tot/(tot+1) over an order-3 n-gram prior (current)
  - uniform : FLAT prior (every token count=1) — removes n-gram structure,
              isolates whether the *shape* of the prior (vs just smoothing) matters
  - frozen  : order-3 prior counts, but FIXED beta = 0.3 (not adaptive) —
              isolates whether the count-adaptive beta matters

Outputs incremental JSON to results/night_ablation.json.
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
ROOT = os.path.join(PHASE, "..")
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpts")
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))

import final_benchmark as fb  # load_chars / make_bpe / MAX_TRAIN / W

W = fb.W
LAYERS = 8
N_EVAL = 5000
CORPUS = os.path.join(ROOT, "phase01", "corpus_public.txt")
MODELS = {
    "STS-Prog": "sts_prog_",
    "STS-Prog (no-PC)": "sts_prog__no_pc__",
}
SEEDS = [0, 1, 2, 3, 4]


# --- exact copies of the repo's prior logic (no fragile imports) ---
def build_order3(train_ids):
    prior = defaultdict(dict)
    for i in range(3, len(train_ids)):
        ctx = tuple(train_ids[i - 2:i])
        w = train_ids[i]
        d = prior[ctx]
        d[w] = d.get(w, 0) + 1
    return {k: dict(v) for k, v in prior.items()}


def generalized_gated_ppl(lpm, targets, prior, V, ctx_tokens, mode, frozen_beta=0.3):
    """lpm = log-softmax logits (N, V); returns gated PPL under `mode`."""
    N = len(lpm)
    nll = np.zeros(N)
    for k in range(N):
        lp = lpm[k]
        pm = np.exp(lp[targets[k]])
        if ctx_tokens is not None:
            ctx = ctx_tokens[k]
        else:
            ctx = tuple(targets[k - 2:k]) if k >= 2 else ()
        if mode == "none":
            nll[k] = -np.log(pm)
            continue
        if mode == "uniform":
            tot = V
            c = 1.0
            beta = tot / (tot + 1.0)
        else:
            table = prior.get(ctx)
            if not table:
                nll[k] = -np.log(pm)
                continue
            tot = sum(table.values())
            c = table.get(targets[k], 0)
            if c <= 0:
                nll[k] = -np.log(pm)
                continue
            beta = frozen_beta if mode == "frozen" else tot / (tot + 1.0)
        nll[k] = -np.logaddexp(np.log1p(-beta) + np.log(pm),
                               np.log(beta) + np.log(c / tot))
    return float(np.exp(np.mean(nll)))


def eval_model(m, test_ids, device, prior, V, batch=64):
    m.eval()
    rng = np.random.default_rng(42)
    n = len(test_ids) - W - 1
    starts = np.sort(rng.choice(n, size=N_EVAL, replace=False))
    lpm_chunks = []
    y = np.zeros(N_EVAL, dtype=int)
    ctx = []
    with torch.no_grad():
        for st in range(0, N_EVAL, batch):
            en = min(st + batch, N_EVAL)
            chunk = starts[st:en]
            X = torch.tensor(np.stack([test_ids[s:s + W] for s in chunk]),
                             dtype=torch.long, device=device)
            out = m(X)
            logits = out[0] if isinstance(out, tuple) else out
            lpm_chunks.append(torch.log_softmax(logits, -1).cpu().numpy())
            for ii, s in enumerate(chunk):
                y[st + ii] = test_ids[s + W]
                ctx.append(tuple(test_ids[s + W - 2:s + W]))
    lpm = np.concatenate(lpm_chunks, axis=0)
    res = {}
    for mode in ["none", "order3", "uniform", "frozen"]:
        res[mode] = round(generalized_gated_ppl(lpm, y, prior, V, ctx, mode), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RESULTS, "night_ablation.json"))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    device = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[T1] device={device}", flush=True)

    train_text = fb.load_chars(CORPUS, fb.MAX_TRAIN)
    tok = fb.make_bpe(train_text)
    V = tok.get_vocab_size()
    ids = np.array(tok.encode(train_text).ids, dtype=np.int32)
    idx = int(len(ids) * 0.8)
    test_ids = ids[idx:]
    print(f"[T1] V={V} test={len(test_ids)}", flush=True)
    prior = build_order3(ids[:200000])

    out = {}
    if os.path.exists(a.out):
        try:
            out = json.load(open(a.out, encoding="utf-8"))
        except Exception:
            out = {}

    for name, slug in MODELS.items():
        out.setdefault(name, {})
        for seed in seeds:
            stem = f"{slug}seed{seed}"
            if stem in out.get(name, {}):
                print(f"[T1] {name} {stem} cached", flush=True)
                continue
            ckpt = os.path.join(CKPT, f"{stem}.pt")
            if not os.path.exists(ckpt):
                print(f"[T1] MISSING {ckpt}, skip", flush=True)
                continue
            driver_mode = "sts_prog" if "no_pc" not in stem else "sts_prog_nopc"
            m = fb.build_pc_model("pc", V, d=192, layers=LAYERS, k_init=1.2,
                                  sync_steps=8, driver_mode=driver_mode,
                                  alpha=0.3, temp=0.3)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m = m.to(device)
            t0 = time.time()
            res = eval_model(m, test_ids, device, prior, V)
            dt = time.time() - t0
            out[name][stem] = {"seed": seed, "ppl": res, "time_s": round(dt, 1)}
            json.dump(out, open(a.out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"[T1] {name} seed{seed}: {res} ({dt:.1f}s)", flush=True)

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[T1] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
