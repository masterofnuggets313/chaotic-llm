"""diag_ci_and_spectral.py — две задачи:

(A) ЧЕСТНЫЕ ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ на ΔPPL (в LOSS-пространстве).
    Старый скрипт считал CI в PPL-пространстве (delta_ppl_ci95_pct в JSON) — это
    взрывоустойчиво: PPL = e^loss, мизерный шум loss => гигантский разброс PPL.
    Здесь пересчитываем paired Δloss по окнам, 1.96*SE в loss, и переводим в
    асимметричный CI на PPL через e^(Δloss ± 1.96·SE). Проверяем, пересекает ли
    ноль интервал Δloss (=> «улучшение» — шум).

(B) СПЕКТРАЛЬНЫЙ/НАПРАВЛЕННЫЙ ДИАГНОСТИКА — ответ на вопрос комментатора:
    насколько STS-Prog способен удерживать КОНФЛИКТУЮЩИЕ локальные представления
    позиций при ЕДИНОМ глобальном драйвере.
    Блок: h <- (1-k)*h + k*driver  +  alpha*tanh(h@W+b)
    driver — ОДИН вектор (1,d) на ВСЁ окно. Глобальное стягивание всех позиций
    к driver с силой k.
    Метрики на матрице движений D=(W,d):
      * Δh_per_layer = h_{l+1} - h_l  (движение каждой позиции за слой)
      * спектр сингулярных чисел D (SVD): если 1-е σ >> остальных -> коллапс в
        одно/несколько доминирующих направлений (опасение комментатора верно);
        если σ спадает медленно -> позиции сохраняют независимые направления.
      * alignment с драйвером: cos(Δh_i, driver) для каждой позиции; доля
        позиций, чьё движение ЗНАЧИМО направлено ВДОЛЬ/ПРОТИВ драйвера.
      * степень «расслоения» h по позициям: Frobenius-норма вариации между
        позициями (inter-position variance) на каждом слое — падает ли она
        (схлопывание) или держится.
"""
import os, sys, json, math, time, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import (
    forward_general, keys_at, keys_range, TAIL, TEMP)

RESULTS = os.path.join(REPO, "results")
FP32 = 4

def _ci_loss(losses):
    n = len(losses); m = sum(losses) / n
    if n < 2: return 0.0, 0.0, 0.0
    se = (sum((x - m) ** 2 for x in losses) / (n - 1)) ** 0.5 / n ** 0.5
    return m, 1.96 * se, n

def recompute_ci(json_path):
    """Пересчёт paired Δloss-CI из сохранённых per-window loss-массивов."""
    with open(json_path) as f:
        data = json.load(f)
    out = {}
    for run in data["runs"]:
        W = run["W"]; le = run["loss_exact_per_window"]
        ex_m, ex_c, ex_n = _ci_loss(le)
        row = {"W": W, "nwin": run["nwin"], "loss_exact_mean": round(le and sum(le)/len(le),4),
               "exact_ci_loss_halfwidth": round(ex_c, 4), "configs": []}
        for cfg in run["configs"]:
            lg = cfg["loss_per_window"]
            # paired Δloss по окнам
            dl = [lg[j] - le[j] for j in range(min(len(lg), len(le)))]
            dlm, dlc, dn = _ci_loss(dl)
            # перевод в PPL через e^(Δloss): точечная оценка и асимметричный CI
            ppl_lo = (math.exp(dlm - dlc) - 1.0) * 100.0
            ppl_hi = (math.exp(dlm + dlc) - 1.0) * 100.0
            crosses_zero = -dlc <= dlm <= dlc
            row["configs"].append({
                "cfg": cfg["cfg"], "variant": cfg["variant"],
                "delta_loss_mean": round(dlm, 4),
                "delta_loss_ci95_hw": round(dlc, 4),
                "ci_ppl_low_pct": round(ppl_lo, 1),
                "ci_ppl_high_pct": round(ppl_hi, 1),
                "crosses_zero_loss": bool(crosses_zero),
                "old_delta_ppl_pct": cfg["delta_ppl_pct"],
                "old_ci_ppl_pct_wrong": cfg["delta_ppl_ci95_pct"],
            })
        out[f"W{W}"] = row
    return out

def spectral_diagnostic(model, toks, W, mode, device, seed=0):
    """Извлекает per-layer h и driver, строит спектр движений и коллинеарность."""
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    L = len(model.blocks); d = model.d
    pos_all = torch.arange(W, device=device)
    with torch.no_grad():
        e_all = keys_at(model, toks[:W], pos_all, mode).detach()

    # capture per-layer h (вход слоя) и driver
    Hs = [None] * L            # h на ВХОДЕ слоя l
    drivers = [None] * L       # driver, использованный перед обновлением слоя l
    q0 = e_all[W - nq:].mean(0, keepdim=True)
    q = q0; h = e_all
    for li, blk in enumerate(model.blocks):
        Hs[li] = h.detach().clone()
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q / (q.norm() + 1e-6)                  # (1, d) — уже с batch-dim
        sim = (en * qn).sum(-1)                      # (W,) — как в models_pc.forward (B=1)
        sim[W - TAIL:] = -1e9
        kk = min(topk, W - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)    # (1,d)
        drivers[li] = driver.detach().clone()
        h = blk(h, driver, k_eff)
        h_last = h[-1].unsqueeze(0)
        q = q0 + model.query_proj(h_last) * 0.5

    res = {"W": W, "mode": mode, "d": d, "layers": L,
           "k_eff": round(float(k_eff.item()), 4),
           "layers_data": []}

    # норма драйвера (для масштаба)
    drv_norm = float(torch.cat(drivers, 0).norm(dim=-1).mean().item())

    for li in range(L):
        h_in = Hs[li]         # (W,d) состояние на ВХОДЕ слоя li
        h_out = Hs[li + 1] if li + 1 < L else None
        # движение за слой: h_{l+1} - h_l  (для последнего слоя — это выход readout-модуля)
        if h_out is not None:
            D = (h_out - h_in).double()      # (W,d)
        else:
            D = (h_in - Hs[li - 1]).double() if li > 0 else torch.zeros_like(h_in).double()

        # --- SVD спектр движений ---
        # центрируем по позициям, чтобы убрать доминирующий «средний» сдвиг
        Dc = D - D.mean(0, keepdim=True)
        # SVD через матрицу ковариации (d x d) — быстрее, чем (W,d)
        # используем нецентрированную для «энергии направлений», центрированную для формы
        G = (Dc.T @ Dc) / (W - 1)            # (d,d) ковариация движений
        try:
            eig = torch.linalg.eigvalsh(G)   # возрастающие
            eig = eig.clamp(min=0)
            svals = torch.sqrt(eig)
            svals_desc = svals.flip(0)
            total = svals_desc.sum().item() + 1e-12
            energy1 = svals_desc[0].item() / total
            energy_top3 = svals_desc[:3].sum().item() / total
            # эффективная размерность (participation ratio) спектра движений
            pr = (svals_desc ** 2).sum() / ((svals_desc ** 4).sum() + 1e-12)
        except Exception as ex:
            energy1 = energy_top3 = pr = float('nan')

        # --- alignment с драйвером ---
        drv = drivers[li].squeeze(0).double()        # (d,)
        drv_n = drv / (drv.norm() + 1e-12)
        # cos каждой позиции со своим движением и с драйвером
        Dn = D / (D.norm(dim=-1, keepdim=True) + 1e-12)
        cos_drv = (Dn @ drv_n).cpu()                 # (W,)
        # доля позиций, чьё |cos| > 0.5 (сильно коллинеарно драйверу)
        frac_align = float((cos_drv.abs() > 0.5).float().mean().item())
        mean_cos = float(cos_drv.mean().item())

        # --- межпозиционная вариация (расслоение) ---
        # норма отклонения позиций от их среднего по слою (показывает схлопывание)
        h_in_c = h_in.double() - h_in.double().mean(0, keepdim=True)
        inter_var = float((h_in_c ** 2).mean().item() ** 0.5)   # RMS отклонение позиций
        h_out_c = h_out.double() - h_out.double().mean(0, keepdim=True) if h_out is not None else h_in_c
        inter_var_out = float((h_out_c ** 2).mean().item() ** 0.5)

        res["layers_data"].append({
            "layer": li,
            "move_energy1": round(energy1, 4),
            "move_energy_top3": round(energy_top3, 4),
            "move_participation_ratio": round(float(pr), 2),
            "frac_aligned_with_driver": round(frac_align, 4),
            "mean_cos_to_driver": round(mean_cos, 4),
            "interpos_var_in": round(inter_var, 3),
            "interpos_var_out": round(inter_var_out, 3),
        })
    res["driver_norm"] = round(drv_norm, 3)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(RESULTS, "fold_unfold_ppl_seed0_ci.json"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--Ws", default="65536,262144")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(RESULTS, "ci_and_spectral.json"))
    args = ap.parse_args()
    dev = args.device

    # ---------- (A) CI ----------
    print("#" * 90, flush=True)
    print("# (A) ЧЕСТНЫЕ CI НА ΔPPL (loss-пространство)", flush=True)
    print("#" * 90, flush=True)
    ci = recompute_ci(args.json)
    for wk, row in ci.items():
        print(f"\n=== {wk}  (окон={row['nwin']}, loss_exact_mean={row['loss_exact_mean']}, "
              f"±{row['exact_ci_loss_halfwidth']} loss) ===", flush=True)
        print(f"  {'cfg':<8}{'variant':<26}{'Δloss':>8}{'±95%':>8}"
              f"{'PPL CI':>16}{'0?':>5}", flush=True)
        for c in row["configs"]:
            print(f"  {c['cfg']:<8}{c['variant']:<26}{c['delta_loss_mean']:>8.4f}"
                  f"{c['delta_loss_ci95_hw']:>8.4f}"
                  f"  [{c['ci_ppl_low_pct']:>7.1f}%, {c['ci_ppl_high_pct']:>7.1f}%]"
                  f"{('ДА' if c['crosses_zero_loss'] else 'нет'):>5}"
                  f"   (старый: {c['old_delta_ppl_pct']:>6.1f}% ±{c['old_ci_ppl_pct_wrong']:.1f}%)",
                  flush=True)

    # ---------- (B) SPECTRAL ----------
    print("\n" + "#" * 90, flush=True)
    print("# (B) СПЕКТРАЛЬНЫЙ ДИАГНОСТ — конфликтующие локальные представления", flush=True)
    print("#" * 90, flush=True)
    import numpy as np, final_benchmark as fb
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)

    spectral = []
    for W in [int(w) for w in args.Ws.split(",")]:
        mode = "trained" if W <= model.pos.shape[1] else "cyclic"
        t0 = time.time()
        r = spectral_diagnostic(model, toks, W, mode, dev)
        torch.cuda.synchronize()
        spectral.append(r)
        print(f"\n=== W={W:,} mode={mode}  k_eff={r['k_eff']}  driver_norm={r['driver_norm']} "
              f"({time.time()-t0:.1f}s) ===", flush=True)
        print(f"  {'L':>2}{'E1':>8}{'Etop3':>8}{'PR':>8}{'fracAlign':>11}{'meanCos':>9}"
              f"{'varIn':>9}{'varOut':>9}", flush=True)
        for l in r["layers_data"]:
            print(f"  {l['layer']:>2}{l['move_energy1']:>8.3f}{l['move_energy_top3']:>8.3f}"
                  f"{l['move_participation_ratio']:>8.1f}{l['frac_aligned_with_driver']:>11.3f}"
                  f"{l['mean_cos_to_driver']:>9.3f}{l['interpos_var_in']:>9.2f}"
                  f"{l['interpos_var_out']:>9.2f}", flush=True)

    # сводка-вердикт
    print("\n--- ВЕРДИКТ (интерпретация) ---", flush=True)
    print("  E1 -> доля энергии движения в 1-м сингулярном направлении.", flush=True)
    print("  Если E1 близко к 1.0 и PR мала -> позиции движутся почти все ВДОЛЬ ОДНОГО", flush=True)
    print("  направления (коллапс, опасение комментатора ПОДТВЕРЖДАЕТСЯ).", flush=True)
    print("  Если E1 умеренное и PR велика -> направления движения РАЗНЫЕ у позиций", flush=True)
    print("  (локальные представления выживают, аргумент ПРОТИВ опасения).", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ci_loss_space": ci, "spectral": spectral}, f, indent=1)
    print(f"\nСохранено: {args.out}", flush=True)

if __name__ == "__main__":
    main()
