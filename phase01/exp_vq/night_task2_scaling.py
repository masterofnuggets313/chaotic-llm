"""night_task2_scaling.py — Task 2 of the overnight run: O(log N) validation.

Two complementary measurements of the core complexity claim:

  (a) CONTEXT-LENGTH SCALING (the memory-bandwidth claim).
      For W in {64,128,256,512,1024}, build a fresh STS-Prog and a parameter-
      matched Transformer, do one forward pass (B=1), and record peak VRAM and
      tokens/sec. The prediction: STS-Prog activation memory grows ~O(N) while
      the Transformer's grows ~O(N^2), so the gap widens with W. Any OOM on the
      Transformer side is itself a data point (recorded, not fatal).

  (b) INTERACTION REACHABILITY (the O(log N) connectivity claim).
      For N in {64,128,256,512,1024} and T in 1..Tmax, measure the fraction of
      random (query,target) pairs that become information-connected within T
      coupling steps, for schedules 'arnold' (cat map), 'shift' (poor), and
      'random'. Prediction: 'arnold' saturates to ~1.0 at T ~ O(log N), while
      'shift'/'random' need many more (or never fully connect).

Outputs incremental JSON to results/night_scaling.json.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
ROOT = os.path.join(PHASE, "..")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)
sys.path.insert(0, os.path.join(PHASE, "exp_memory_selector"))

import final_benchmark as fb
from parametric_models import TransformerLM
from chaos_lib import interaction_reachability

W_VALS = [64, 128, 256, 512, 1024]
TMAX = 24
# interaction_reachability uses the Arnold cat map, which lives on an N x N grid,
# so n_tokens MUST be a perfect square. Use squares only.
N_REACH = [64, 256, 1024, 4096]
D_TF = 92  # parameter-matched Transformer width from v2


def build_sts_for_w(Vc, Wc):
    torch.manual_seed(0)
    from models_pc import PurePCLM
    m = PurePCLM(vocab=Vc, d=192, layers=8, k_init=1.2, alpha=0.3,
                 sync_steps=8, driver_mode="sts_prog", temp=0.3)
    m.pos = nn.Parameter(torch.randn(1, Wc, 192) * 0.02)
    return m


def build_tf_for_w(Vc, Wc):
    torch.manual_seed(0)
    return TransformerLM(Vc, Wc, D=D_TF, HEADS=4, LAYERS=8)


def measure_forward(build_fn, Vc, Wc, device):
    """Return dict with vram_mb/time_ms/tok_s or {'oom': True}."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = build_fn(Vc, Wc).to(device)
        torch.cuda.synchronize()
        X = torch.randint(0, Vc, (1, Wc), device=device)
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = m(X)
        torch.cuda.synchronize()
        t = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        del m
        torch.cuda.empty_cache()
        return {"vram_mb": round(vram, 1),
                "time_ms": round(t * 1000, 2),
                "tokens_per_sec": round(Wc / t) if t > 0 else 0}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True, "W": Wc}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"error": str(e), "W": Wc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RESULTS, "night_scaling.json"))
    a = ap.parse_args()
    device = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[T2] device={device}", flush=True)

    # (a) context-length scaling
    train_text = fb.load_chars(os.path.join(ROOT, "phase01", "corpus_public.txt"), fb.MAX_TRAIN)
    tok = fb.make_bpe(train_text)
    V = tok.get_vocab_size()
    print(f"[T2] V={V}", flush=True)

    scaling = {"sts_prog": {}, "transformer": {}}
    for Wc in W_VALS:
        s = measure_forward(build_sts_for_w, V, Wc, device)
        scaling["sts_prog"][Wc] = s
        print(f"[T2] STS W={Wc}: {s}", flush=True)
        t = measure_forward(build_tf_for_w, V, Wc, device)
        scaling["transformer"][Wc] = t
        print(f"[T2] TF  W={Wc}: {t}", flush=True)

    # (b) interaction reachability
    reach = {}
    ttc = {}
    for N in N_REACH:
        reach[str(N)] = {}
        ttc[str(N)] = {}
        for sched in ["arnold", "shift", "random"]:
            curve = []
            for T in range(1, TMAX + 1):
                f = interaction_reachability(N, T, schedule=sched, n_trials=400)
                curve.append(round(f, 4))
            reach[str(N)][sched] = curve
            # smallest T at which >=99% of (query,target) pairs are connected:
            # this is the direct test of the O(log N) global-connectivity claim.
            connect = next((t for t, v in enumerate(curve, start=1) if v >= 0.99), None)
            ttc[str(N)][sched] = connect
            print(f"[T2] reach N={N} {sched}: T1={curve[0]} T8={curve[7]} "
                  f"T{TMAX}={curve[-1]} t_connect={connect}", flush=True)

    out = {
        "_meta": {"V": V, "D_TF": D_TF, "W_vals": W_VALS, "Tmax": TMAX,
                  "N_reach": N_REACH, "device": device},
        "context_scaling": scaling,
        "reachability": reach,
        "time_to_connect": ttc,
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[T2] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
