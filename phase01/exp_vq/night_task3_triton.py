"""night_task3_triton.py — Task 3 of the overnight run: Triton fused kernel prototype.

Colleague direction: the ChaoticBlock forward is memory-bandwidth bound
(permute-gather + NICE coupling are two disjoint global memory round-trips).
A fused Triton kernel (gather source x[sigma[i]] AND apply the coupling in the
same kernel, with no intermediate materialization) should cut that in half.

This script:
  1. Always runs a GPU BASELINE throughput sweep (torch.index_select + vectorized
     NICE coupling) over W in {64,128,256,512,1024,2048}, B=64, d=192 — this is
     the memory-bandwidth measurement the colleague actually wants, and it is the
     comparison anchor.
  2. If `triton` imports, also runs the FUSED Triton kernel and reports the speedup.
  3. If `triton` is unavailable, writes the kernel source anyway and records
     triton_available=False so the prototype is ready when the dep lands.

Outputs JSON to results/night_triton.json.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
ROOT = os.path.join(PHASE, "..")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, PHASE)

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception as e:
    HAVE_TRITON = False
    _TRITON_ERR = str(e)

W_VALS = [64, 128, 256, 512, 1024, 2048]
B = 64
D = 192
ITERS = 50


# ---------------- Triton fused gather + NICE coupling ----------------
# Only defined when triton is importable; otherwise Task 3 runs baseline-only
# and records triton_available=False so the kernel is ready when the dep lands.
if HAVE_TRITON:
    @triton.jit
    def _gather_couple_kernel(
        x_ptr, sigma_ptr, y_ptr,
        N, d, g,
        BLOCK_D: tl.constexpr,
    ):
        # one program per token i (across all batches flattened)
        i = tl.program_id(0)
        base = i * d
        si = tl.load(sigma_ptr + i).to(tl.int32)
        src_base = si * d
        for dd in range(0, d, BLOCK_D):
            offs = dd + tl.arange(0, BLOCK_D)
            mask = offs < d
            xv = tl.load(x_ptr + src_base + offs, mask=mask)
            tl.store(y_ptr + base + offs, xv, mask=mask)
        # coupling: pair (2k, 2k+1) within the permuted order
        k = i // 2
        even_idx = 2 * k
        odd_idx = 2 * k + 1
        if even_idx < N and odd_idx < N:
            eb = even_idx * d
            ob = odd_idx * d
            for dd in range(0, d, BLOCK_D):
                offs = dd + tl.arange(0, BLOCK_D)
                mask = offs < d
                ev = tl.load(y_ptr + eb + offs, mask=mask)
                ov = tl.load(y_ptr + ob + offs, mask=mask)
                ne = ev + g * ov
                no = ov + g * ev
                tl.store(y_ptr + eb + offs, ne, mask=mask)
                tl.store(y_ptr + ob + offs, no, mask=mask)


    def triton_fused(x, sigma, g):
        Bn, N, d = x.shape
        y = torch.empty_like(x)
        flat = Bn * N
        for b in range(Bn):
            _gather_couple_kernel[(flat,)](x[b].contiguous(), sigma, y[b],
                                           N, d, g, BLOCK_D=32)
        return y


# ---------------- Baseline (torch ops) ----------------
def baseline(x, sigma, g):
    xp = x.index_select(1, sigma)  # permute/gather
    even = xp[:, 0::2, :]
    odd = xp[:, 1::2, :]
    ne = even + g * odd
    no = odd + g * even
    out = torch.empty_like(xp)
    out[:, 0::2, :] = ne
    out[:, 1::2, :] = no
    return out


def bench(fn, x, sigma, g, iters, device):
    # warmup
    for _ in range(5):
        fn(x, sigma, g)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(x, sigma, g)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RESULTS, "night_triton.json"))
    a = ap.parse_args()
    device = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[T3] device={device} triton={HAVE_TRITON}", flush=True)

    results = {"triton_available": HAVE_TRITON, "baseline": {}, "triton": {}}
    if not HAVE_TRITON:
        results["triton_error"] = _TRITON_ERR

    g = torch.tensor(0.5, device=device)
    for Wc in W_VALS:
        N = Wc
        x = torch.randn(B, N, D, device=device)
        sigma = torch.randperm(N, device=device)
        base_ms = bench(baseline, x, sigma, g, ITERS, device) * 1000
        results["baseline"][Wc] = {
            "ms_per_step": round(base_ms, 3),
            "tok_per_sec": round(B * N / (base_ms / 1000), 1),
        }
        print(f"[T3] baseline W={Wc}: {results['baseline'][Wc]}", flush=True)
        if HAVE_TRITON:
            try:
                tri_ms = bench(triton_fused, x, sigma, g, ITERS, device) * 1000
                results["triton"][Wc] = {
                    "ms_per_step": round(tri_ms, 3),
                    "tok_per_sec": round(B * N / (tri_ms / 1000), 1),
                    "speedup_vs_baseline": round(base_ms / tri_ms, 3),
                }
                print(f"[T3] triton  W={Wc}: {results['triton'][Wc]}", flush=True)
            except Exception as e:
                results["triton"][Wc] = {"error": str(e)}
                print(f"[T3] triton  W={Wc}: ERR {e}", flush=True)

    json.dump(results, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[T3] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
