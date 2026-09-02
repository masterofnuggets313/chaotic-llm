"""Эксперимент 1: g-абляция — насколько точен должен быть g в readout3.

Гипотеза: для decode нам достаточно точных глобальных драйверов + h_last (только
последняя позиция через блоки). g = h.mean(dim=1) можно приблизить.
Если да — блоки по всему W не нужны на каждом decode-шаге.

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_g_ablation.py
"""
import os, sys, json, time, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, cos_sim, StreamFracode)
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")


def compute_drivers_and_h_last(model, e_all, q0, Wc, k_eff, topk, nq):
    """Точные драйверы по всему Wc + h_last только последней позиции через блоки.
    Возвращает (h_last, drivers_list, q_final)."""
    h_last = e_all[-1:].clone()   # h_0[-1] = e[-1]
    drivers = []
    q = q0
    for blk in model.blocks:
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        drivers.append(driver)
        h_last = blk(h_last, driver, k_eff)
        q = q0 + model.query_proj(h_last) * 0.5
    return h_last, drivers, q


def compute_full_h_from_drivers(model, e_all, Wc, k_eff, drivers):
    """Прогон блоков по ВСЕМУ Wc с зафиксированными драйверами.
    Нужен только для эталонного g = mean(h)."""
    h = e_all.clone()
    for li, blk in enumerate(model.blocks):
        driver = drivers[li]
        BLK = 131072
        hc = []
        for s in range(0, Wc, BLK):
            hc.append(blk(h[s:s + BLK], driver, k_eff))
        h = torch.cat(hc, 0)
    return h


def main():
    dev = "cuda"
    Wc = 262144

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

    print(f"Model loaded. V={V}, W={Wc}, d=192, L=8, topk={topk}, k_eff={k_eff.item():.4f}")
    print(f"Running {16} windows (held-out)...", flush=True)

    g_variants = [
        "g_exact",      # эталон: mean(h) по всему W
        "g_last_4096",  # mean по последним 4096
        "g_last_8192",  # mean по последним 8192
        "g_last_16384", # mean по последним 16384
        "g_mean_e",     # mean(e) — статические ключи
        "g_h_last",     # = h_last (последняя позиция)
        "g_q0",         # = q0 (mean последних 4 эмбеддингов)
        "g_zero",       # нулевой вектор
    ]

    # Без-слойный прогон: ещё один вариант — убрать global mean из readout,
    # но оставить exact h_last + q0.
    # readout3 = nn.Sequential(nn.Linear(3*d, d), nn.ReLU(), nn.Linear(d, vocab))
    # Заменяем g на что-то, но readout обучен на exact g...

    results = {v: {"ce": [], "cos_logits": [], "delta_ppl_pct": []} for v in g_variants}
    results["exact"] = {"ce": []}

    n_windows = 16
    with torch.no_grad():
        for wi in range(n_windows):
            o = 5000 + wi * Wc
            if o + Wc >= len(toks) - 1:
                break
            end = o + Wc

            e_all = keys_at(model, toks, torch.arange(o, end, device=dev), mode).detach()
            q0 = e_all[-nq:].mean(0, keepdim=True)
            target = toks[end].view(1)

            # --- ЭТАЛОН ---
            lg_exact = forward_general(model, toks[o:end], mode, chunk=Wc)
            ce_ex = float(F.cross_entropy(lg_exact, target).item())
            results["exact"]["ce"].append(ce_ex)

            # --- Точные драйверы + h_last ---
            h_last_exact, drivers, q_final = compute_drivers_and_h_last(
                model, e_all, q0, Wc, k_eff, topk, nq)

            # Полный h (для эталонного g)
            h_full = compute_full_h_from_drivers(model, e_all, Wc, k_eff, drivers)
            g_exact = h_full.mean(dim=0, keepdim=True)

            # Проверка h_last
            cos_check = float(F.cosine_similarity(h_last_exact, h_full[-1:], dim=-1).item())
            if wi == 0:
                print(f"cos(h_last_exact, h_full[-1]) = {cos_check:.8f}  (1.0 = exact)", flush=True)

            # --- Подготовка g-вариантов ---
            g_map = {
                "g_exact": g_exact,
                "g_last_4096": h_full[-4096:].mean(0, keepdim=True) if Wc >= 4096 else g_exact,
                "g_last_8192": h_full[-8192:].mean(0, keepdim=True) if Wc >= 8192 else g_exact,
                "g_last_16384": h_full[-16384:].mean(0, keepdim=True) if Wc >= 16384 else g_exact,
                "g_mean_e": e_all.mean(0, keepdim=True),
                "g_h_last": h_last_exact,
                "g_q0": q0,
                "g_zero": torch.zeros_like(q0),
            }

            for name, g_val in g_map.items():
                logits_g = model.readout3(torch.cat([h_last_exact, q0, g_val], dim=-1))
                ce_g = float(F.cross_entropy(logits_g, target).item())
                cos_lg = float(F.cosine_similarity(logits_g, lg_exact, dim=-1).item())
                results[name]["ce"].append(ce_g)
                results[name]["cos_logits"].append(cos_lg)
                results[name]["delta_ppl_pct"].append((np.exp(ce_g) / np.exp(ce_ex) - 1) * 100)

            if (wi + 1) % 4 == 0:
                print(f"  window {wi+1}/{n_windows} done", flush=True)

    # --- ИТОГИ ---
    print("\n" + "=" * 70)
    print(f"g-АБЛЯЦИЯ: {n_windows} окон, W={Wc}")
    print("=" * 70)
    print(f"{'g variant':20s} {'CE mean':>8s} {'CE std':>8s} {'ΔPPL%':>10s} {'ΔPPL std':>10s} {'cos_logits':>10s}")
    print("-" * 70)

    summary = {}
    ce_ex = np.mean(results["exact"]["ce"])
    ppl_ex = np.exp(ce_ex)
    print(f"{'exact (эталон)':20s} {ce_ex:>8.4f} {np.std(results['exact']['ce']):>8.4f} {'—':>10s} {'—':>10s} {'—':>10s}")

    for name in g_variants:
        r = results[name]
        ce_m = np.mean(r["ce"])
        ce_s = np.std(r["ce"])
        deltas = r["delta_ppl_pct"]
        dp_mean = np.mean(deltas)
        dp_std = np.std(deltas)
        cos_m = np.mean(r["cos_logits"])
        cos_s = np.std(r["cos_logits"])
        print(f"{name:20s} {ce_m:>8.4f} {ce_s:>8.4f} {dp_mean:+9.1f}% {dp_std:>+9.1f}% {cos_m:>10.4f}")
        summary[name] = {
            "ce_mean": round(float(ce_m), 4),
            "ce_std": round(float(ce_s), 4),
            "delta_ppl_mean_pct": round(float(dp_mean), 1),
            "delta_ppl_std_pct": round(float(dp_std), 1),
            "cos_logits_mean": round(float(cos_m), 4),
        }

    print("-" * 70)
    print(f"Эталон PPL: {ppl_ex:.2f}")
    print("ВЫВОД: если ΔPPL < 5% и cos_logits > 0.99 — g можно смело приближать.")
    print("Тогда decode: драйверы (O(W·d)) + h_last (1 позиция) + g_local (K позиций).")
    print("=" * 70)

    out = {"W": Wc, "windows": n_windows, "exact_ce": round(float(ce_ex), 4),
           "exact_ppl": round(float(ppl_ex), 2), "g_variants": summary}
    with open(os.path.join(RESULTS, "exp_g_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_g_ablation.json")


if __name__ == "__main__":
    main()