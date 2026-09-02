"""diag_risk23_retrieval_sweep.py — свип conflict-probe (Риск 2) по степеням сжатия.

ДОБИВАЕТ ДЫРУ под гипотезу «PPL/retrieval STS-Prog при 8x/16x/32x сжатой памяти
против exact»: retrieval (logit_gap conflict-probe) был замерен ТОЛЬКО на 32x
(см. risk23.json). Здесь тот же дизайн (симметрия 25/75%, 4 query-позиции,
контроль-пара, метрика logit_gap) гоняется на 8x / 16x / 32x.

Степени сжатия Fracode (bytes/tok = levels*subvecs, base exact = 768 Б):
  8x  -> L=2, S=48 -> 96 Б
  16x -> L=2, S=24 -> 48 Б
  32x -> L=2, S=12 -> 24 Б
Переиспользуем risk2_conflict из diag_risk23.py (проверенная логика).

ЗАПУСК: py -3.13 diag_risk23_retrieval_sweep.py   (интерпретатор С torch, CUDA)
"""
import os, sys, json, time
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)

from models_pc import build_pc_model
from night_task5_fracode_forward import (
    forward_general, keys_at, StreamFracode, TAIL, TEMP)
from diag_risk23 import risk2_conflict
import final_benchmark as fb

DEVICE = "cuda"
CKPT = os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt")

# (метка, subvecs) — levels=2 фиксирован => bytes = 2*subvecs
COMPRESSIONS = [("8x", 48), ("16x", 24), ("32x", 12)]


def main():
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head)
    V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(
        tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids,
        dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=DEVICE)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3,
                           temp=TEMP).to(DEVICE).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k)
    topk = int(model.topk)
    nq = int(model.nquery)

    W = 262144
    mode = "cyclic"
    # калибровочный кодбук (held-out участок, как в diag_risk23)
    with torch.no_grad():
        cb = keys_at(model, toks[W:2 * W], torch.arange(W, device=DEVICE), mode).detach()
    cb_ns = 262144
    if cb.shape[0] > cb_ns:
        cb = cb[::cb.shape[0] // cb_ns]

    out = {"W": W, "mode": mode, "base_bytes_per_tok": 768,
           "design": "conflict-probe: logit_gap(a vs b) exact vs gen; "
                     "ctrl=logit_gap(a vs c); 4 query-позиции",
           "gap_exact": None, "ctrl_gap_exact": None,
           "runs": []}

    for label, S in COMPRESSIONS:
        fm = StreamFracode(192, levels=2, subvecs=S, K=256, device=DEVICE)
        t0 = time.time()
        fm.fit(cb, iters=12, seed=0)
        r = risk2_conflict(model, toks, W, DEVICE, k_eff, topk, nq, mode, fm=fm)
        dt = time.time() - t0
        if out["gap_exact"] is None:
            out["gap_exact"] = r["gap_exact"]
            out["ctrl_gap_exact"] = r["ctrl_gap_exact"]
            out["query_positions"] = r["query_positions"]
            out["factA_pos"] = r["factA_pos"]
            out["factB_pos"] = r["factB_pos"]
            out["attr_a_idx"] = r["attr_a_idx"]
            out["attr_b_idx"] = r["attr_b_idx"]
            out["ctrl_attr_idx"] = r["ctrl_attr_idx"]
        out["runs"].append({
            "compression": label,
            "subvecs": S,
            "stored_bytes_per_tok": 2 * S,
            "ratio_x": round(768 / (2 * S), 2),
            "gap_gen": r["gap_gen"],
            "ctrl_gap_gen": r["ctrl_gap_gen"],
            "fit_sec": round(dt, 1),
        })
        print(f"  {label}: gap_gen={r['gap_gen']}  ctrl_gap_gen={r['ctrl_gap_gen']}  "
              f"({dt:.1f}s)", flush=True)

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    path = os.path.join(REPO, "results", "risk23_retrieval_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nСохранено: {path}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
