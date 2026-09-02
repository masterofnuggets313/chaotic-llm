"""Эксперимент 9a: драйвер из ВОССТАНОВЛЕННЫХ Fracode-векторов vs точных e_all.

Ключ к победе на W=10M: если драйвер можно строить из decode_codes (коды 24B/поз),
то e_all (768B/поз) не нужен на decode-шаге. Тогда STS-Prog работает на W=10M
с состоянием 240MB, а TF KV (56GB) — нет.

Проверяем на W=65536: ΔPPL драйвера из восстановленных vs точных.
"""
import os, sys, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, forward_general, TAIL, TEMP, StreamFracode
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 256
Mcand = 512


def decode_step(model, fr, e_all, q0, Wc, k_eff, topk, nq, K, driver_src):
    """driver_src: 'exact' (e_all[nxt]) или 'recon' (decode_codes[nxt])."""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    codes3 = fr.codes
    flat = codes3.reshape(Wc, -1)
    for blk in model.blocks:
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        if qn.dim() == 2:
            qn = qn.squeeze(0)
        tabs = torch.empty(fr.L * fr.S, fr.K, device=qn.device)
        j = 0
        for l in range(fr.L):
            for s in range(fr.S):
                tabs[j] = qn[s * fr.sub:(s + 1) * fr.sub] @ fr.cbooks[l][s].T
                j += 1
        g = tabs.gather(1, flat.t())
        scores = g.sum(0)
        scores[Wc - TAIL:] = -1e18
        m = min(Mcand, max(1, Wc - TAIL))
        cand = scores.topk(m).indices
        cn = torch.clamp(cand + 1, 0, Wc - 2)
        pos = torch.unique(torch.cat([cand, cn]))
        rec = fr.decode_codes(fr.codes[pos])
        recn = rec / (rec.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (recn * qn.unsqueeze(0)).sum(-1)
        i_c = torch.searchsorted(pos, cand)
        sim_c = sim[i_c]
        vals, loc = sim_c.topk(min(topk, m))
        nxt = cn[loc]
        w = torch.softmax(vals / TEMP, 0)
        if driver_src == "exact":
            driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        else:
            dr = fr.decode_codes(fr.codes[nxt])
            driver = (w.unsqueeze(-1) * dr).sum(0, keepdim=True)
        h_last = blk(h_last, driver, k)
        hK = blk(hK, driver, k)
        q = q0 + model.query_proj(h_last) * 0.5
    g = hK.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


def main():
    dev = "cuda"
    Wq = 65536
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

    res = {"exact": [], "recon": []}
    ref = []
    with torch.no_grad():
        for wi in range(8):
            o = 5000 + wi * Wq
            if o + Wq >= len(toks) - 1: break
            end = o + Wq
            x = toks[o:end]; target = toks[end].view(1)
            lg_ex = forward_general(model, x, mode, chunk=Wq)
            ref.append(float(F.cross_entropy(lg_ex, target).item()))
            e_all = keys_at(model, x, torch.arange(Wq, device=dev), mode).detach()
            en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
            fr = StreamFracode(d=192, levels=2, subvecs=12, K=256, device=dev)
            fr.fit(en, iters=12, seed=0)
            fr.codes = fr.encode_rows(en)
            q0 = e_all[-nq:].mean(0, keepdim=True)
            for src in ["exact", "recon"]:
                lg = decode_step(model, fr, e_all, q0, Wq, k_eff, topk, nq, K, src)
                res[src].append(float(F.cross_entropy(lg, target).item()))
            print(f"  win {wi}: exact={res['exact'][-1]:.4f} recon={res['recon'][-1]:.4f}", flush=True)

    ref_m = float(np.mean(ref))
    print(f"\nЭталон (forward_general): CE={ref_m:.4f}")
    for src in ["exact", "recon"]:
        ce = float(np.mean(res[src]))
        d = (np.exp(ce) / np.exp(ref_m) - 1) * 100
        print(f"ADC {src:>5}: CE={ce:.4f}  ΔPPL={d:+.1f}%")

    out = {"ref_ce": round(ref_m, 4),
           "exact_ce": round(float(np.mean(res["exact"])), 4),
           "recon_ce": round(float(np.mean(res["recon"])), 4),
           "exact_dppl": round((np.exp(np.mean(res["exact"])) / np.exp(ref_m) - 1) * 100, 1),
           "recon_dppl": round((np.exp(np.mean(res["recon"])) / np.exp(ref_m) - 1) * 100, 1)}
    with open(os.path.join(RESULTS, "exp_recon_driver.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_recon_driver.json")


if __name__ == "__main__":
    main()