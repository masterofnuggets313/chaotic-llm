"""two_tier_proto.py — прототип Two-Tier Memory для STS-Prog (чистая версия).

Проверяет в ТРИ фазы, что Two-Tier даёт и что теряет:

ФАЗА A — качество (teacher-forced, один target-токен):
  Прогнозируем токен в позиции o+W по окну [o, o+W) тремя способами и сравниваем CE:
    1) exact    — полный окно W через точные ключи (эталон, как обучалась модель);
    2) window   — ТОЛЬКО активное окно (последние N_local), без глубины
                  (g = g_local, драйвер из локальных ключей) — нижняя граница качества;
    3) two-tier — активное окно + глубинная Fracode-память: драйвер = объединение
                  локальных top-k и ADC-кандидатов из глубины; g = g_local + λ·g_deep.
  Цель: измерить, сколько качества возвращает глубина поверх окна (и не сломали ли мы readout).

ФАЗА B — скорость:
  Время одного 8-слойного forward на N_local против full-W + время ADC-поиска по глубине.

ЗАПУСК: cd phase01/exp_vq && py -3.13 two_tier_proto.py [--N 8192] [--windows 4]
"""
import os, sys, time, json, argparse, torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, cos_sim, StreamFracode)

RESULTS = os.path.join(REPO, "results")
LAMBDA_G = 0.5  # вес g_deep в гибриде g


def _forward_window(model, e_win, q0, k_eff, topk, nq):
    """8-слойный sts_prog forward по окну e_win (N,d). Драйвер из локальных ключей.
    Возвращает (logits, g_local). q0 фиксирован (из хвоста окна)."""
    N = e_win.shape[0]
    d = model.d
    q_cur = q0
    h = e_win
    g_sum = torch.zeros(d, device=e_win.device)
    for blk in model.blocks:
        en = e_win / (e_win.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[N - TAIL:] = -1e9
        kk = min(topk, N - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, N - 2)
        driver = (w.unsqueeze(-1) * e_win[nxt]).sum(0, keepdim=True)  # (1,d)
        h = blk(h, driver, k_eff)
        g_sum = g_sum + h.sum(0)
        q_cur = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    g_local = (g_sum / N).unsqueeze(0)          # (1,d)
    logits = model.readout3(torch.cat([h[-1].unsqueeze(0), q0, g_local], dim=-1))
    return logits, g_local


def _deep_driver_candidates(model, fm, depth_codes, depth, q_cur, topk, Mcand):
    """ADC-поиск по глубокой памяти: возвращает (e_deep_top, sim_vals_top) —
    развёрнутые ключи соседей top-k кандидатов из глубины и их значения близости."""
    fm.codes = depth_codes
    vals, cn, _ = fm.select(q_cur.squeeze(0), topk, depth, Mcand, rerank=False)  # (topk,)
    e_deep = fm.decode_codes(depth_codes[cn])  # (topk, d)
    return e_deep, vals


def _forward_two_tier(model, fm, e_win, depth_codes, depth, q0, k_eff, topk, nq, Mcand):
    """8-слойный forward: активное окно + глубинная память.
    На каждом слое драйвер = объединение локальных top-k и ADC-кандидатов глубины
    (мягкое объединение через concat + softmax по 2*topk значениям).
    g = g_local + λ·g_deep (g_deep — средний развёрнутый кандидат глубины на каждом слое).
    Возвращает (logits, g_local, g_deep)."""
    N = e_win.shape[0]
    d = model.d
    q_cur = q0
    h = e_win
    g_sum = torch.zeros(d, device=e_win.device)
    g_deep_sum = torch.zeros(d, device=e_win.device)
    n_deep = 0
    for blk in model.blocks:
        en = e_win / (e_win.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim_loc = (en * qn).sum(-1)
        sim_loc[N - TAIL:] = -1e9
        kk_loc = min(topk, N - TAIL)
        vals_loc, loc_loc = sim_loc.topk(kk_loc)
        nxt_loc = torch.clamp(loc_loc + 1, 0, N - 2)
        e_loc_cand = e_win[nxt_loc]                        # (kk,d)
        # --- глубина: ADC-кандидаты ---
        e_deep_cand, vals_deep = _deep_driver_candidates(
            model, fm, depth_codes, depth, q_cur, topk, Mcand)
        # --- объединение ---
        vals_all = torch.cat([vals_loc, vals_deep])        # (2*topk,)
        e_all_cand = torch.cat([e_loc_cand, e_deep_cand], 0)  # (2*topk, d)
        w = torch.softmax(vals_all / TEMP, 0)
        driver = (w.unsqueeze(-1) * e_all_cand).sum(0, keepdim=True)  # (1,d)
        # --- блок ---
        h = blk(h, driver, k_eff)
        g_sum = g_sum + h.sum(0)
        g_deep_sum = g_deep_sum + (e_deep_cand * w[:topk].unsqueeze(-1)).sum(0)
        n_deep += 1
        q_cur = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    g_local = (g_sum / N).unsqueeze(0)
    g_deep = (g_deep_sum / n_deep).unsqueeze(0) if n_deep > 0 else torch.zeros(1, d, device=e_win.device)
    g_hybrid = g_local + LAMBDA_G * g_deep
    logits = model.readout3(torch.cat([h[-1].unsqueeze(0), q0, g_hybrid], dim=-1))
    return logits, g_local, g_deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192, help="активное окно")
    ap.add_argument("--windows", type=int, default=4, help="число окон для усреднения")
    ap.add_argument("--steps-per-window", type=int, default=4, help="число target-токенов на окно")
    args = ap.parse_args()

    import final_benchmark as fb
    dev = "cuda"
    Wc = 262144

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
    N_local = args.N

    # ---- Fracode-кодбук (калибровка на первых 262k токенах) ----
    print("Fitting Fracode codebook (calib on 262k)...", flush=True)
    e_calib = keys_at(model, toks[:Wc], torch.arange(Wc, device=dev), mode).detach()
    sub = e_calib[::max(1, e_calib.shape[0] // 65536)]
    fm = StreamFracode(model.d, levels=2, subvecs=12, K=256, device=dev)
    fm.fit(sub, iters=12, seed=0)
    del e_calib, sub
    print(f"Fracode: {fm.bytes_per_pos:.0f} B/pos", flush=True)

    # ---- фаза A+B: по окнам ----
    ce = {"exact": [], "window": [], "two_tier": []}
    logit_cos = {"win_vs_exact": [], "tt_vs_exact": []}
    speed = {"exact_full_s": None, "window_s": None, "adc_s": None}

    with torch.no_grad():
        for wi in range(args.windows):
            o = 20000 + wi * (Wc + 4000)  # непересекающиеся окна
            x_win = toks[o:o + Wc]        # (Wc,)
            # глубокий префикс = всё до активного окна
            depth = Wc - N_local
            t0 = time.time()
            depth_codes = fm.encode_rows(keys_at(model, toks,
                                                 torch.arange(o, o + depth, device=dev), mode))
            t_enc = time.time() - t0

            for st in range(args.steps_per_window):
                # target: токен в позиции end = o+depth+N_local+st = o+Wc+st
                end = o + Wc + st
                # активное окно заканчивается перед target: [end-N_local, end)
                loc = toks[end - N_local:end]
                e_win = keys_at(model, toks,
                                torch.arange(end - N_local, end, device=dev), mode).detach()
                q0 = e_win[-nq:].mean(0, keepdim=True)  # как в forward_general
                target = toks[end].view(1)

                # 1) exact: полный окно [o, end) — для честного сравнения нужен forward_general
                #    на [o,end) длиной Wc+st (растёт) — но st мал, ок
                lg_ex = forward_general(model, toks[o:end], mode, chunk=end - o)
                ce["exact"].append(float(torch.nn.functional.cross_entropy(lg_ex, target).item()))

                # 2) window-only
                lg_win, _ = _forward_window(model, e_win, q0, k_eff, topk, nq)
                ce["window"].append(float(torch.nn.functional.cross_entropy(lg_win, target).item()))

                # 3) two-tier
                lg_tt, _, _ = _forward_two_tier(model, fm, e_win, depth_codes, depth,
                                                q0, k_eff, topk, nq, Mcand=1024)
                ce["two_tier"].append(float(torch.nn.functional.cross_entropy(lg_tt, target).item()))

                logit_cos["win_vs_exact"].append(float(cos_sim(lg_win, lg_ex).item()))
                logit_cos["tt_vs_exact"].append(float(cos_sim(lg_tt, lg_ex).item()))

            if wi == 0:
                # фаза B: скорость
                t0 = time.time()
                _ = forward_general(model, toks[o:o + Wc], mode, chunk=Wc)
                torch.cuda.synchronize()
                speed["exact_full_s"] = time.time() - t0
                t0 = time.time()
                _ = _forward_window(model, e_win, q0, k_eff, topk, nq)
                torch.cuda.synchronize()
                speed["window_s"] = time.time() - t0
                t0 = time.time()
                _ = _deep_driver_candidates(model, fm, depth_codes, depth, q0, topk, 1024)
                torch.cuda.synchronize()
                speed["adc_s"] = time.time() - t0
                del depth_codes

    mean = lambda x: float(np.mean(x)) if x else float("nan")
    out = {
        "N_local": N_local, "windows": args.windows, "steps_per_window": args.steps_per_window,
        "ce_exact": mean(ce["exact"]), "ce_window": mean(ce["window"]),
        "ce_two_tier": mean(ce["two_tier"]),
        "delta_window_vs_exact_ppl": (np.exp(mean(ce["window"])) / np.exp(mean(ce["exact"])) - 1) * 100,
        "delta_two_tier_vs_exact_ppl": (np.exp(mean(ce["two_tier"])) / np.exp(mean(ce["exact"])) - 1) * 100,
        "logit_cos_win_vs_exact": mean(logit_cos["win_vs_exact"]),
        "logit_cos_tt_vs_exact": mean(logit_cos["tt_vs_exact"]),
        "speed": speed,
        "speedup_exact_over_window": (speed["exact_full_s"] / speed["window_s"]) if speed["window_s"] else None,
    }
    print("\n=== Two-Tier PROTO ===")
    print(f"N_local={N_local}  windows={args.windows}  targets/window={args.steps_per_window}")
    print(f"CE  exact     : {out['ce_exact']:.4f}")
    print(f"CE  window    : {out['ce_window']:.4f}  (ΔPPL {out['delta_window_vs_exact_ppl']:+.1f}%)")
    print(f"CE  two-tier  : {out['ce_two_tier']:.4f}  (ΔPPL {out['delta_two_tier_vs_exact_ppl']:+.1f}%)")
    print(f"logit cos: window {out['logit_cos_win_vs_exact']:.4f} | two-tier {out['logit_cos_tt_vs_exact']:.4f}")
    print(f"speed: exact-full {speed['exact_full_s']:.3f}s | window {speed['window_s']*1000:.1f}ms "
          f"| adc {speed['adc_s']*1000:.1f}ms | speedup {out['speedup_exact_over_window']:.0f}x")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"two_tier_proto_N{N_local}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nСохранено: two_tier_proto_N{N_local}.json")


if __name__ == "__main__":
    main()
