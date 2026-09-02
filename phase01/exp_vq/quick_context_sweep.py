"""quick_context_sweep.py — быстрый взгляд на форму кривых VRAM(context).

Строит STS-Prog и Transformer на ~1M параметров (d/D подобраны под цель) и
гоняет ТОЛЬКО context-sweep: свежие модели, Wc∈{256,1024,4096,16384,65536,262144},
один forward B=1, пик VRAM + tok/s. Без обучения и eval — пара минут.

Цель: увидеть на глаз, что STS ~ O(N), а TF ~ O(N^2) — и где TF упирается в OOM.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import night_task4_scaling_ladder as T

V = 512
W_TRAIN = 256
L = 4
HEADS = 4
WC_VALS = [256, 1024, 4096, 16384, 65536, 262144]


def main():
    d, p_sts = T.solve_sts_d(1_000_000, V, W_TRAIN, L)
    D, p_tf = T.solve_tf_D(1_000_000, V, W_TRAIN, L, HEADS)
    print(f"STS 1M: d={d}  params={p_sts:,}", flush=True)
    print(f"TF  1M: D={D}  params={p_tf:,}", flush=True)

    print("\n--- STS context-sweep (O(N)?) ---", flush=True)
    sts_sweep, sts_oom = T.context_sweep(d, None, "sts", V, L, HEADS, WC_VALS)
    for w in WC_VALS:
        v = sts_sweep.get(w)
        if isinstance(v, dict) and "vram_mb" in v:
            print(f"  Wc={int(w):>8}: VRAM={v['vram_mb']:>9.1f} MB   tok/s={v['tok_per_s']:>11.1f}", flush=True)
        else:
            print(f"  Wc={int(w):>8}: OOM", flush=True)

    print("\n--- TF context-sweep (O(N^2)?) ---", flush=True)
    tf_sweep, tf_oom = T.context_sweep(None, D, "tf", V, L, HEADS, WC_VALS)
    for w in WC_VALS:
        v = tf_sweep.get(w)
        if isinstance(v, dict) and "vram_mb" in v:
            print(f"  Wc={int(w):>8}: VRAM={v['vram_mb']:>9.1f} MB   tok/s={v['tok_per_s']:>11.1f}", flush=True)
        else:
            print(f"  Wc={int(w):>8}: OOM", flush=True)

    # общая точка для сравнения
    common = [w for w in WC_VALS if w in sts_sweep and w in tf_sweep
              and isinstance(sts_sweep[w], dict) and isinstance(tf_sweep[w], dict)
              and "vram_mb" in sts_sweep[w] and "vram_mb" in tf_sweep[w]]
    if common:
        w = common[-1]
        sv = sts_sweep[w]["vram_mb"]; tv = tf_sweep[w]["vram_mb"]
        print(f"\nНа общей точке Wc={w}: STS={sv:.1f}MB vs TF={tv:.1f}MB -> STS/TF={sv/tv:.3f}", flush=True)
    sts_max = max([w for w in WC_VALS if w in sts_sweep and isinstance(sts_sweep[w], dict)], default=0)
    tf_max = max([w for w in WC_VALS if w in tf_sweep and isinstance(tf_sweep[w], dict)], default=0)
    print(f"Максимальный осиленный контекст: STS={sts_max:,} токенов, TF={tf_max:,} токенов", flush=True)


if __name__ == "__main__":
    main()
