"""diag_h_update.py — проверка: можно ли обновить h аналитически при смене драйвера?

Гипотеза: h на слое l меняется как h' = h + k·(driver' - driver) для всех позиций,
если driver поменялся, а хаотическая часть (h@W+b) осталась той же.
НО на следующем слое хаотическая часть применяется к ОБНОВЛЁННОМУ h' → tanh(h'@W+b) ≠ tanh(h@W+b).
Вопрос: насколько сильно расходится аппроксимация?

Измеряем: cos(h_t1_real, h_t1_approx) где h_t1_approx = h_t + k·(driver' - driver) на каждом слое.
Если cos ~ 0.9+, то update работает, и мы можем ускорить decode без пересчёта блоков.
"""
import os, sys, torch, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, cos_sim)

RESULTS = os.path.join(REPO, "results")
N = 4096  # для проверки берём N=4096, а не 262k (чтобы считать быстро)


def _get_h_and_drivers(model, e, Wc, nq, k_eff, topk):
    """Строим h и список драйверов для каждого слоя."""
    q0 = e[-nq:].mean(0, keepdim=True)
    q = q0
    en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
    h = e
    drivers = []
    hs = [h.clone()]
    for blk in model.blocks:
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e[nxt]).sum(0, keepdim=True)
        drivers.append(driver)
        h = blk(h, driver, k_eff)
        hs.append(h.clone())
        q = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    return hs, drivers


def main():
    import final_benchmark as fb
    dev = "cuda"
    Wc = N

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
    k_val = float(k_eff.item())  # sigmoid(1.2) ≈ 0.77

    o = 10000
    with torch.no_grad():
        e_t = keys_at(model, toks, torch.arange(o, o+Wc, device=dev), mode).detach()
        hs_t, drivers_t = _get_h_and_drivers(model, e_t, Wc, nq, k_eff, topk)

        e_t1 = keys_at(model, toks, torch.arange(o+1, o+1+Wc, device=dev), mode).detach()
        hs_t1, drivers_t1 = _get_h_and_drivers(model, e_t1, Wc, nq, k_eff, topk)

    # --- Аналитический update: h' = h + k·(driver' - driver) на каждом слое ---
    print(f"k (sigmoid) = {k_val:.4f}")
    print(f"\n=== Аналитический update h через Δdriver (W={Wc}) ===\n")
    print(f"  {'layer':<8} {'cos(h_approx, h_real)':<25} {'||Δdriver||':<15}")
    print(f"  {'-'*8} {'-'*25} {'-'*15}")

    h_approx = e_t1.clone()  # h_0 = e_t1 (начальное состояние — новые ключи, реальные)
    # Реальный h_t1 на слое 0 — это e_t1 (совпадает с approx)
    h_real_layer0 = hs_t1[0]  # e_t1
    cos0 = float(cos_sim(h_approx, h_real_layer0).mean().item())
    print(f"  {'0 (init)':<8} {cos0:<25.6f}")

    for li in range(len(model.blocks)):
        # реальный h на этом слое
        h_real = hs_t1[li + 1]
        # реальные драйверы
        d_old = drivers_t[li]
        d_new = drivers_t1[li]
        delta_d = d_new - d_old
        norm_dd = float(delta_d.norm().item())

        # 1) хаотическая часть: h_chaos = h_approx + α·tanh(h_approx @ W + b)
        blk = model.blocks[li]
        h_chaos = h_approx + blk.alpha * torch.tanh(h_approx @ blk.W + blk.b)
        # 2) PC-синхронизация с НОВЫМ драйвером: h_approx = (1-k)·h_chaos + k·driver_new
        h_approx = (1 - k_eff) * h_chaos + k_eff * d_new

        # косинус с реальным
        cos_real = float(cos_sim(h_approx, h_real).mean().item())
        print(f"  {li+1:<8} {cos_real:<25.6f} {norm_dd:<15.6f}")

    # --- Итог ---
    An = h_approx / (h_approx.norm(dim=-1, keepdim=True) + 1e-6)
    Bn = hs_t1[-1] / (hs_t1[-1].norm(dim=-1, keepdim=True) + 1e-6)
    final_cos = float((An * Bn).sum(-1).mean().item())
    print(f"\n  Финальный cos (слой 8): {final_cos:.6f}")
    print(f"  Δdriver || в среднем по слоям: {float(torch.stack([(drivers_t1[i]-drivers_t[i]).norm() for i in range(len(drivers_t))]).mean().item()):.6f}")


if __name__ == "__main__":
    main()