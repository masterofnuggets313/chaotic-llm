"""diag_h_drift.py — насколько меняется h при смене одного токена в окне?

Возвращает cos(h_t, h_{t+1}) для каждого слоя — чтобы понять, можно ли
кэшировать h и не пересчитывать его полностью на каждом шаге.

Если cos ~ 1.0 — h почти не меняется, кэширование работает.
Если cos падает сильно — каждый новый токен перетряхивает всё состояние.
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


def _get_h(model, x, Wc, mode):
    """Строим h (Wc, d) после 8 слоёв sts_prog.
    Возвращает список h по слоям [h_0, h_1, ..., h_8]."""
    dev = x.device
    e = keys_at(model, x, torch.arange(Wc, device=dev), mode)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    q0 = e[-nq:].mean(0, keepdim=True)
    q = q0
    en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
    h = e
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
        h = blk(h, driver, k_eff)
        hs.append(h.clone())
        q = q0 + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    return hs


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
    mode = "cyclic"

    # окно [o, o+Wc) и следующее окно [o+1, o+1+Wc) — сдвиг на 1 токен
    o = 10000
    with torch.no_grad():
        hs_t = _get_h(model, toks[o:o+Wc], Wc, mode)
        hs_t1 = _get_h(model, toks[o+1:o+1+Wc], Wc, mode)

    print(f"\n=== H-drift при сдвиге на 1 токен (W={Wc}) ===")
    print(f"  cos(h_t_layer, h_t1_layer) для общего (W-1, d) и последней позиции:\n")
    print(f"  {'layer':<8} {'cos(общий)':<15} {'cos(посл.поз)':<15} {'norm(Δdriver)':<15}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*15}")
    for li in range(len(hs_t)):
        # общий: первые W-1 позиций (сдвиг, t+1 позиция 0 = t позиция 1 и т.д.)
        A = hs_t[li][:-1]; B = hs_t1[li][1:]
        # поэлементный косинус (не pairwise — OOM!)
        An = A / (A.norm(dim=-1, keepdim=True) + 1e-6)
        Bn = B / (B.norm(dim=-1, keepdim=True) + 1e-6)
        c_common = float((An * Bn).sum(-1).mean().item()) if A.shape[0] > 0 else 0
        # последняя позиция (новая)
        c_last = float(cos_sim(hs_t[li][-1].unsqueeze(0), hs_t1[li][-1].unsqueeze(0)).item())
        print(f"  {li:<8} {c_common:<15.6f} {c_last:<15.6f}")

    # ACE (cos среднего по всем позициям и слоям — поэлементно, без хранения All)
    cos_sum = 0.0
    cos_n = 0
    for li in range(len(hs_t)):
        A = hs_t[li][:-1]; B = hs_t1[li][1:]
        An = A / (A.norm(dim=-1, keepdim=True) + 1e-6)
        Bn = B / (B.norm(dim=-1, keepdim=True) + 1e-6)
        cos_sum += float((An * Bn).sum(-1).sum().item())
        cos_n += A.shape[0]
    cos_mean = cos_sum / cos_n if cos_n > 0 else 0
    print(f"\n  Средний cos по всем позициям и слоям: {cos_mean:.6f}")


if __name__ == "__main__":
    main()