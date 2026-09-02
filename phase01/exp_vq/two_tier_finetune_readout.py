"""two_tier_finetune_readout.py — доводка readout3 для Two-Tier Memory.

Логика: модель заморожена, кроме readout3 и опционально query_proj.
Обучаем readout на teacher-forced данных: exact (full-W) logits = target,
а на вход подаём g = гибрид (активное окно + глубина).

Цель: readout3 должен научиться выдавать ≈ те же логиты, получая
на вход другое g (не mean по всему W, а g_local + λ·g_deep).

ЗАПУСК: cd phase01/exp_vq && py -3.13 two_tier_finetune_readout.py
"""
import os, sys, time, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, StreamFracode)

RESULTS = os.path.join(REPO, "results")
LAMBDA_G = 0.5
N_LOCAL = 8192


def _make_hybrid_g(model, fm, e_win, depth_codes, depth, q0, k_eff, topk, nq, Mcand=1024):
    """Строит hybrid g = g_local + λ·g_deep для teacher-forced обучения.
    Это тот же двухслойный forward, что в two_tier_proto, но возвращает только
    g_hybrid и h_last (для readout)."""
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
        kk = min(topk, N - TAIL)
        vals_loc, loc_loc = sim_loc.topk(kk)
        nxt_loc = torch.clamp(loc_loc + 1, 0, N - 2)
        e_loc_cand = e_win[nxt_loc]
        # глубина
        if depth > 0:
            fm.codes = depth_codes
            vals_deep, cn_deep, _ = fm.select(q_cur.squeeze(0), topk, depth, Mcand, rerank=False)
            e_deep_cand = fm.decode_codes(depth_codes[cn_deep])  # (topk, d)
        else:
            vals_deep = torch.zeros(0, device=e_win.device)
            e_deep_cand = torch.zeros(0, d, device=e_win.device)
        # объединение
        vals_all = torch.cat([vals_loc, vals_deep])
        e_all_cand = torch.cat([e_loc_cand, e_deep_cand], 0)
        w = torch.softmax(vals_all / TEMP, 0)
        driver = (w.unsqueeze(-1) * e_all_cand).sum(0, keepdim=True)
        h = blk(h, driver, k_eff)
        g_sum = g_sum + h.sum(0)
        if depth > 0:
            g_deep_sum = g_deep_sum + (e_deep_cand * w[:kk].unsqueeze(-1)).sum(0)
            n_deep += 1
        q_cur = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    g_local = (g_sum / N).unsqueeze(0)
    g_deep = (g_deep_sum / max(n_deep, 1)).unsqueeze(0) if n_deep > 0 else torch.zeros(1, d, device=e_win.device)
    g_hybrid = g_local + LAMBDA_G * g_deep
    return g_hybrid, h[-1].unsqueeze(0)


def main():
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

    # ---- разморозить только readout3 (query_proj не трогаем — он внутри цикла блоков) ----
    for p in model.readout3.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(
        model.readout3.parameters(),
        lr=1e-4, weight_decay=1e-5)

    # ---- Fracode-кодбук ----
    print("Fitting Fracode codebook...", flush=True)
    e_calib = keys_at(model, toks[:Wc], torch.arange(Wc, device=dev), mode).detach()
    sub = e_calib[::max(1, e_calib.shape[0] // 65536)]
    fm = StreamFracode(model.d, levels=2, subvecs=12, K=256, device=dev)
    fm.fit(sub, iters=12, seed=0)
    del e_calib, sub
    print(f"Fracode: {fm.bytes_per_pos:.0f} B/pos, trainable params: "
          f"{sum(p.numel() for p in model.readout3.parameters() if p.requires_grad)}", flush=True)

    # ---- генерация данных: teacher-forced exact logits + hybrid g ----
    print("Generating training data (teacher-forced)...", flush=True)
    n_windows = 24      # 24 окна × 1 target = 24 семпла (toks ~11M, окошки непересекающиеся)
    data = []
    ws = Wc             # отступ = размер окна (непересекающиеся окна, ~42 окна в 11M)
    for wi in range(n_windows):
        o = 5000 + wi * ws
        if o + Wc >= len(toks):
            break
        end = o + Wc  # target-позиция
        depth = Wc - N_LOCAL
        # exact logits (target)
        with torch.no_grad():
            lg_ex = forward_general(model, toks[o:end], mode, chunk=end - o)
            target = toks[end].view(1)
        # hybrid g (всё в no_grad — вход для readout учим как фиксированные признаки)
        with torch.no_grad():
            e_win = keys_at(model, toks, torch.arange(end - N_LOCAL, end, device=dev), mode).detach()
            q0 = e_win[-nq:].mean(0, keepdim=True).detach()
            depth_codes = fm.encode_rows(keys_at(model, toks, torch.arange(o, o + depth, device=dev), mode))
            g_hybrid, h_last = _make_hybrid_g(model, fm, e_win, depth_codes, depth, q0, k_eff, topk, nq)
            g_hybrid = g_hybrid.detach()
            h_last = h_last.detach()
        data.append((h_last, q0, g_hybrid, lg_ex, target))
        if (wi + 1) % 8 == 0:
            print(f"  {wi+1}/{n_windows}", flush=True)

    # ---- обучение ----
    n_epochs = 3
    print(f"\nTraining readout3 ({n_epochs} epochs, {len(data)} samples)...", flush=True)
    model.readout3.train()
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for h_last, q0, g_hybrid, lg_ex, target in data:
            # forward с гибридным g: обучается ТОЛЬКО readout3 (входы — фикс. признаки)
            logits = model.readout3(torch.cat([h_last, q0, g_hybrid], dim=-1))
            # KL-дивергенция к exact logits (soft target — распределение, не только argmax)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(lg_ex, dim=-1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout3.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data))
        print(f"  epoch {epoch+1}: KL={losses[-1]:.6f}", flush=True)

    # ---- оценка: CE после обучения на held-out окнах ----
    # (before = из прошлого прогона two_tier_proto: CE 7.63, +465% PPL)
    model.eval()
    print("\nEvaluating on held-out windows...", flush=True)
    ce_after = {"exact": [], "two_tier": []}
    for wi in range(8):
        o = 5000 + (n_windows + wi) * ws
        if o + Wc >= len(toks):
            break
        end = o + Wc
        depth = Wc - N_LOCAL
        with torch.no_grad():
            lg_ex = forward_general(model, toks[o:end], mode, chunk=end - o)
            target = toks[end].view(1)
            ce_after["exact"].append(float(F.cross_entropy(lg_ex, target).item()))
            # two-tier ПОСЛЕ обучения readout
            e_win = keys_at(model, toks, torch.arange(end - N_LOCAL, end, device=dev), mode).detach()
            q0 = e_win[-nq:].mean(0, keepdim=True)
            depth_codes = fm.encode_rows(keys_at(model, toks, torch.arange(o, o + depth, device=dev), mode))
            g_hybrid, h_last = _make_hybrid_g(model, fm, e_win, depth_codes, depth, q0, k_eff, topk, nq)
            logits_after = model.readout3(torch.cat([h_last, q0, g_hybrid], dim=-1))
            ce_after["two_tier"].append(float(F.cross_entropy(logits_after, target).item()))

    mean = lambda x: float(np.mean(x)) if x else float("nan")
    out = {
        "train_losses": losses,
        "ce_before": {"exact": mean(ce_after["exact"]), "two_tier": 7.63},
        "ce_after": {"exact": mean(ce_after["exact"]),
                     "two_tier": mean(ce_after["two_tier"])},
        "delta_before_ppl": (np.exp(7.63) / np.exp(mean(ce_after["exact"])) - 1) * 100,
        "delta_after_ppl": (np.exp(mean(ce_after["two_tier"])) / np.exp(mean(ce_after["exact"])) - 1) * 100,
    }
    print(f"\n=== Two-Tier READOUT FINETUNE ===")
    print(f"CE exact         : {out['ce_before']['exact']:.4f}")
    print(f"CE two-tier ДО   : {out['ce_before']['two_tier']:.4f}  (ΔPPL {out['delta_before_ppl']:+.1f}%)")
    print(f"CE two-tier ПОСЛЕ: {out['ce_after']['two_tier']:.4f}  (ΔPPL {out['delta_after_ppl']:+.1f}%)")
    print(f"KL train loss: {losses[-1]:.6f} (final)")

    # ---- сохраняем чекпоинт доводки ----
    ckpt_path = os.path.join(RESULTS, "ckpts", "sts_prog_seed0_twotier_readout.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "two_tier_finetune.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"Data saved: two_tier_finetune.json")


if __name__ == "__main__":
    main()