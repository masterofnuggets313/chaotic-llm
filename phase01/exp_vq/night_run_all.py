"""night_run_all.py — orchestrator for the 4-task overnight run.

Chains, in order:
  T1  night_task1_ablation.py      β-prior ablation (eval-only, fast, ~minutes)
  T2  night_task2_scaling.py       O(log N): context-length scaling + reachability
  T3  night_task3_triton.py        Triton fused-kernel prototype + throughput sweep
  T4  night_task4_scaling_ladder.py лестница параметров 1M/5M/20M/100M на TinyStories
  T5  night_task5_fracode_forward.py  FracodeMemory ВНУТРИ прямого прохода:
                                     PPL + retrieval, Exact|PQ|Fracode × {8x,16x,32x},
                                     held-out codebook (дизайн Германа, 02.09.2026)

Each task writes its own incremental JSON under results/. This orchestrator only
logs progress (with timestamps) to results/night_run.log so the run can be
monitored after launch.

Launch (do NOT run until the user signals "спокойной ночи"):
  cd phase01\\exp_vq && py -3.13 night_run_all.py

It is safe to re-run: every task caches completed work and resumes.
"""
import os
import sys
import time
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RESULTS = os.path.join(ROOT, "results")
LOG = os.path.join(RESULTS, "night_run.log")
PY = "py -3.13"

TASKS = [
    ("T1 β-prior ablation", "night_task1_ablation.py", ""),
    ("T2 O(log N) scaling", "night_task2_scaling.py", ""),
    ("T3 Triton throughput", "night_task3_triton.py", ""),
    ("T4 param ladder (TinyStories)",
     "night_task4_scaling_ladder.py",
     "--corpus corpus_nl/tinystories.txt"),
    ("T5 Fracode-in-forward (PPL+retrieval)",
     "night_task5_fracode_forward.py", ""),
]


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(script, extra=""):
    cmd = f"{PY} {script} {extra}".strip()
    log(f"START {cmd}")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=HERE, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log("   " + line.rstrip("\n"))
    p.wait()
    dt = time.time() - t0
    log(f"END {script} exit={p.returncode} ({dt/60:.1f} min)")
    return p.returncode


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    log("=" * 60)
    log("=== NIGHT RUN START ===")
    log(f"cwd={HERE}")
    try:
        cuda = subprocess.run(f"{PY} -c \"import torch;print(torch.cuda.is_available())\"",
                              shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        cuda = "?"
    log(f"cuda={cuda}")
    overall = time.time()
    for name, script, extra in TASKS:
        log(f">>> {name}")
        rc = run(script, extra)
        if rc != 0:
            log(f"!!! {name} returned rc={rc} — continuing to next task")
    total_min = (time.time() - overall) / 60
    log(f"=== NIGHT RUN COMPLETE ({total_min:.1f} min) ===")
    log("=" * 60)
