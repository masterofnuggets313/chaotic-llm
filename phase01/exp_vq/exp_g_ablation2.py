"""Эксперимент 1b: чистая g-абляция.

Изолируем эффект g: берём h_last и q0 ИЗ ЭТАЛОННОГО forward_general (точный прогон
одним чанком Wc), затем для каждого g-варианта считаем readout3([h_last, q0, g_var]).
Разница в CE — ТОЛЬКО из-за g.

Дополнительно: вариант readout без g (нужна переобученная головка? проверяем OOD).

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_g_ablation2.py
"""
import os, sys, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, exact_select)
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")


def main():
    dev = "cuda"
    Wc = 65536  # меньше окон, эталон одним чанком — быстрее

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

    print(f"Model loaded. V={V}, W={Wc}, d=192, L=8")
    print("Running clean g-ablation...", flush=True)

    g_variants = [
        "g_exact",      # эталон: mean(h) по всему W — должно дать 0%
        "g_last_1024",  # mean по последним 1024 позициям h
        "g_last_4096",
        "g_last_8192",
        "g_last_16384",
        "g_mean_e",     # mean(e) — статические ключи
        "g_h_last",     # = h_last
        "g_q0",         # = q0
    ]

    res = {v: {"ce": [], "delta": []} for v in g_variants}
    res["exact"] = {"ce": []}
    n_windows = 24

    with torch.no_grad():
        for wi in range(n_windows):
            o = 5000 + wi * Wc
            if o + Wc >= len(toks) - 1:
                break
            end = o + Wc
            x = toks[o:end]
            target = toks[end].view(1)

            # ЭТАЛОН одним чанком
            lg_exact = forward_general(model, x, mode, chunk=Wc)
            ce_ex = float(F.cross_entropy(lg_exact, target).item())
            res["exact"]["ce"].append(ce_ex)

            # Захват эталонных h_last и g из forward_general
            # ВАЖНО: пересобираем вручную, чтобы получить h_last и g
            q0 = keys_range(model, x, Wc - nq, Wc, mode).mean(0, keepdim=True)
            q = q0
            h = keys_range(model, x, 0, Wc, mode)   # (Wc, d) один чанк
            h_last = h[-1:].clone()
            for li, blk in enumerate(model.blocks):
                run_val, run_next = exact_select(model, x, q, mode, [(0, Wc)], topk, Wc)
                ckey = keys_at(model, x, run_next, mode)
                w = torch.softmax(run_val / TEMP, 0)
                driver = (w.unsqueeze(-1) * ckey).sum(0).view(1, model.d)
                h = blk(h, driver, k_eff)          # весь Wc (эталонный путь)
                h_last = h[-1:].clone()
                q = q0 + model.query_proj(h_last) * 0.5
            g_gt = h.mean(0, keepdim=True)          # эталонный g

            # базовые векторы readout
            cat_base = torch.cat([h_last, q0, g_gt], dim=-1)
            lg_gt = model.readout3(cat_base)
            # проверка: lg_gt должен совпадать с lg_exact (float noise ~1e-6)
            cos_base = float(F.cosine_similarity(lg_gt, lg_exact, dim=-1).item())

            # g-варианты
            g_map = {
                "g_exact": g_gt,
                "g_last_1024": h[-1024:].mean(0, keepdim=True),
                "g_last_4096": h[-4096:].mean(0, keepdim=True),
                "g_last_8192": h[-8192:].mean(0, keepdim=True),
                "g_last_16384": h[-16384:].mean(0, keepdim=True),
                "g_mean_e": keys_range(model, x, 0, Wc, mode).mean(0, keepdim=True),
                "g_h_last": h_last,
                "g_q0": q0,
            }
            for name, g_val in g_map.items():
                lg = model.readout3(torch.cat([h_last, q0, g_val], dim=-1))
                ce_g = float(F.cross_entropy(lg, target).item())
                res[name]["ce"].append(ce_g)
                res[name]["delta"].append((np.exp(ce_g) / np.exp(ce_ex) - 1) * 100)

            if (wi + 1) % 6 == 0:
                print(f"  window {wi+1} done (cos_base={cos_base:.6f})", flush=True)

    print("\n" + "=" * 70)
    print(f"ЧИСТАЯ g-АБЛЯЦИЯ: {len(res['exact']['ce'])} окон, W={Wc}")
    print("=" * 70)
    ce_ex = float(np.mean(res["exact"]["ce"]))
    print(f"exact (эталон) CE={ce_ex:.4f} PPL={np.exp(ce_ex):.2f}")
    print(f"{'g variant':16s} {'CE mean':>8s} {'ΔPPL mean':>10s} {'ΔPPL std':>10s}")
    print("-" * 52)
    summary = {}
    for name in g_variants:
        r = res[name]
        ce_m = float(np.mean(r["ce"]))
        dm = float(np.mean(r["delta"]))
        ds = float(np.std(r["delta"]))
        print(f"{name:16s} {ce_m:>8.4f} {dm:+9.1f}% {ds:+9.1f}%")
        summary[name] = {"ce_mean": round(ce_m, 4),
                         "delta_ppl_mean_pct": round(dm, 1),
                         "delta_ppl_std_pct": round(ds, 1)}
    print("=" * 70)
    print("ВЫВОД: если g_last_K даёт ΔPPL < 5% — g можно считать по K позициям,")
    print("и тогда на decode-шаге блоки гоняются только по K, а не по всему W.")

    out = {"W": Wc, "windows": len(res["exact"]["ce"]), "exact_ce": round(ce_ex, 4),
           "g_variants": summary}
    with open(os.path.join(RESULTS, "exp_g_ablation_clean.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_g_ablation_clean.json")


if __name__ == "__main__":
    main()