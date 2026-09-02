"""rerun_spectral_1m.py — перезапуск ТОЛЬКО спектрального диагноста на W=1M.

Контекст: в risk23.json уже валидны risk2_conflict и risk3_ppl_1m (из mVeajO).
Спектральный же кусок (risk3_spectral) был битый (E1=0/PR=0 у mVeajO из-за
кривого Welford; пересчёт 9qFmE9 упал на device-mismatch). После починки
device-бага в risk3_spectral_welford этот скрипт пересчитывает спектр на
полноценном W=1M и аккуратно ПЕРЕЗАПИСЫВАЕТ только поле risk3_spectral,
не трогая risk2_conflict и risk3_ppl_1m.

Запуск (по явному «го»):
  cd phase01/exp_vq && py -3.13 rerun_spectral_1m.py
"""
import os, sys, json, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, TEMP
from diag_risk23 import risk3_spectral_welford

RESULTS = os.path.join(REPO, "results")
CKPT = os.path.join(RESULTS, "ckpts", "sts_prog_seed0.pt")
W = 1_048_576


def main():
    import numpy as np, final_benchmark as fb
    dev = "cuda"
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"=== РИСК 3 (пересчёт): спектральный @ W={W:,} (device-fixed Welford) ===", flush=True)
    t0 = time.time()
    with torch.no_grad():
        r3 = risk3_spectral_welford(model, toks, W, dev, chunk=4096)
    dt = time.time() - t0
    print(f"  E1={r3['E1']}  PR={r3['PR']}  var={r3['interpos_var']}  "
          f"meanCos={r3['meanCos_to_driver']}  unfold={r3['unfold_sec']}s  "
          f"(wall {dt:.1f}s)", flush=True)
    print(f"SPECTRAL_RESULT: {json.dumps(r3)}", flush=True)

    # Мерджим только risk3_spectral, остальное не трогаем.
    out_path = os.path.join(RESULTS, "risk23.json")
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["risk3_spectral"] = [r3]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Перезаписано только поле risk3_spectral в {out_path}", flush=True)


if __name__ == "__main__":
    main()
