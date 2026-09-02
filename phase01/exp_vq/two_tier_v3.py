"""two_tier_v3.py — Вариант 3: истинный g (mean по всему W), драйверы из активного окна.

Отделяет эффект драйвера от эффекта g:
  - g = h.mean(dim=1) по ВСЕМУ W (честно, стримингом, как в forward_general).
  - Драйверы на каждом слое — из АКТИВНОГО окна (N_local=8192, topk по 8192).
  - h эволюционирует по всему W (через блоки), но драйверы дёшевы.

Если PPL ≈ exact — механика активного окна работает, проблема была только в g.
Если PPL далёк — проблема в драйверах, и Two-Tier требует переобучения.

ЗАПУСК: cd phase01/exp_vq && py -3.13 two_tier_v3.py
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
    keys_at, keys_range, forward_general, TAIL, TEMP, cos_sim, StreamFracode)

RESULTS = os.path.join(REPO, "results")
N_LOCAL = 8192


def _forward_local_driver_global_g(model, e_all, q0, Wc, N_local, k_eff, topk, nq):
    """8-слойный sts_prog forward: драйверы из активного окна (последние N_local),
    g = h.mean(dim=1) по ВСЕМУ Wc (честно стримингом).
    Возвращает (logits, g_true)."""
    q_cur = q0
    h = e_all       # (Wc, d)
    g_sum = torch.zeros(model.d, device=e_all.device)
    for blk in model.blocks:
        # ---- драйвер из АКТИВНОГО окна (последние N_local позиций) ----
        e_local = e_all[-N_local:]
        en = e_local / (e_local.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[N_local - TAIL:] = -1e9
        kk = min(topk, N_local - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, N_local - 2)
        driver = (w.unsqueeze(-1) * e_local[nxt]).sum(0, keepdim=True)  # (1,d)
        # ---- прогон блоков по ВСЕМУ Wc ----
        BLK = 131072
        hc = []
        for s in range(0, Wc, BLK):
            hc.append(blk(h[s:s + BLK], driver, k_eff))
        h = torch.cat(hc, 0)
        g_sum = g_sum + h.sum(0)
        q_cur = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    g_true = (g_sum / Wc).unsqueeze(0)  # (1, d) — честный глобальный g
    logits = model.readout3(torch.cat([h[-1].unsqueeze(0), q0, g_true], dim=-1))
    return logits, g_true


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

    # ---- PPL на 8 held-out окнах ----
    ws = Wc  # непересекающиеся, как в finetune
    ce = {"exact": [], "local_driver": []}
    speed = {"exact_full_s": None, "local_driver_s": None}

    with torch.no_grad():
        for wi in range(8):
            o = 5000 + wi * ws
            if o + Wc >= len(toks):
                break
            end = o + Wc
            # exact: полный Wc, всё честно
            lg_ex = forward_general(model, toks[o:end], mode, chunk=end - o)
            target = toks[end].view(1)
            ce["exact"].append(float(F.cross_entropy(lg_ex, target).item()))

            # local_driver + global g: весь Wc, но драйверы из последних N_local
            e_all = keys_at(model, toks, torch.arange(o, end, device=dev), mode).detach()
            q0 = e_all[-nq:].mean(0, keepdim=True)
            lg_loc, _ = _forward_local_driver_global_g(
                model, e_all, q0, Wc, N_LOCAL, k_eff, topk, nq)
            ce["local_driver"].append(float(F.cross_entropy(lg_loc, target).item()))

            if wi == 0:
                # скорость
                t0 = time.time()
                _ = forward_general(model, toks[o:end], mode, chunk=end - o)
                torch.cuda.synchronize()
                speed["exact_full_s"] = time.time() - t0
                t0 = time.time()
                _ = _forward_local_driver_global_g(
                    model, e_all, q0, Wc, N_LOCAL, k_eff, topk, nq)
                torch.cuda.synchronize()
                speed["local_driver_s"] = time.time() - t0

    mean = lambda x: float(np.mean(x)) if x else float("nan")
    out = {
        "N_local": N_LOCAL, "windows": len(ce["exact"]),
        "ce_exact": mean(ce["exact"]),
        "ce_local_driver": mean(ce["local_driver"]),
        "delta_ppl": (np.exp(mean(ce["local_driver"])) / np.exp(mean(ce["exact"])) - 1) * 100,
        "speed": speed,
        "speedup": speed["exact_full_s"] / speed["local_driver_s"] if speed["local_driver_s"] else None,
    }
    print(f"\n=== Two-Tier V3: local driver, global g ===")
    print(f"N_local={N_LOCAL}  windows={out['windows']}")
    print(f"CE exact          : {out['ce_exact']:.4f}")
    print(f"CE local_driver+g : {out['ce_local_driver']:.4f}  (ΔPPL {out['delta_ppl']:+.1f}%)")
    print(f"speed: exact {speed['exact_full_s']*1000:.1f}ms | local_driver {speed['local_driver_s']*1000:.1f}ms "
          f"| speedup {out['speedup']:.0f}x" if out['speedup'] else "")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "two_tier_v3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"Data saved: two_tier_v3.json")


if __name__ == "__main__":
    main()