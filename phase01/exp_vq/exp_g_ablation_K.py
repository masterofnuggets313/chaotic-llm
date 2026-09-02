"""Эксперимент 5a: g-абляция по K — насколько мал K можно брать без потери качества?
g_last_K при K=256/512/1024/2048/4096. Если K=512 работает — блоки в 8× дешевле.
"""
import os, sys, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, forward_general, TAIL, TEMP
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")


def fast_decode_step(model, e_all, q0, Wc, k_eff, topk, nq, K):
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
    for blk in model.blocks:
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        h_last = blk(h_last, driver, k)
        hK = blk(hK, driver, k)
        q = q0 + model.query_proj(h_last) * 0.5
    g = hK.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


def main():
    dev = "cuda"
    Wc = 65536
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

    print(f"Сбор эталонных CE на 16 окнах W={Wc}...", flush=True)
    cex = []
    with torch.no_grad():
        for wi in range(16):
            o = 5000 + wi * Wc
            if o + Wc >= len(toks) - 1: break
            end = o + Wc
            x = toks[o:end]; target = toks[end].view(1)
            lg = forward_general(model, x, mode, chunk=Wc)
            cex.append(float(F.cross_entropy(lg, target).item()))
    ce_ref = float(np.mean(cex))
    print(f"Эталон CE={ce_ref:.4f}  PPL={np.exp(ce_ref):.1f}  (n={len(cex)} окон)")

    res = {}
    for K in [256, 512, 1024, 2048, 4096, 8192]:
        ce = []
        with torch.no_grad():
            for wi in range(len(cex)):
                o = 5000 + wi * Wc
                end = o + Wc
                x = toks[o:end]; target = toks[end].view(1)
                e_all = keys_at(model, x, torch.arange(Wc, device=dev), mode).detach()
                q0 = e_all[-nq:].mean(0, keepdim=True)
                lg = fast_decode_step(model, e_all, q0, Wc, k_eff, topk, nq, K)
                ce.append(float(F.cross_entropy(lg, target).item()))
        ce_m = float(np.mean(ce))
        dppl = (np.exp(ce_m) / np.exp(ce_ref) - 1) * 100
        res[K] = {"ce": round(ce_m, 4), "delta_ppl_pct": round(dppl, 1)}
        print(f"K={K:>5}: CE={ce_m:.4f}  ΔPPL={dppl:+.1f}%")

    with open(os.path.join(RESULTS, "exp_g_ablation_K.json"), "w", encoding="utf-8") as f:
        json.dump({"ce_ref": ce_ref, "K": res}, f, indent=1)
    print(f"\nSaved: results/exp_g_ablation_K.json")


if __name__ == "__main__":
    main()