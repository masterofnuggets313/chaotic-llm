"""diag_unfold.py — послойная сверка моей unfold_h с эталонным forward_general."""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import forward_general, keys_at, TEMP, TAIL
from fold_unfold import unfold_h

dev = "cuda"; torch.manual_seed(0)
import numpy as np, final_benchmark as fb
head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
tok = fb.make_bpe(head); V = tok.get_vocab_size(); nh = len(tok.encode(head).ids)
ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
toks = torch.tensor(ids[nh:], dtype=torch.long, device=dev)

model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                       driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
model.load_state_dict(torch.load(os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
for p in model.parameters(): p.requires_grad_(False)

W = 65536
Ppos = model.pos.shape[1]
mode = "trained" if W <= Ppos else "cyclic"
L = len(model.blocks)
caps = [None] * L
with torch.no_grad():
    _ = forward_general(model, toks[:W], mode, chunk=4096, capture=caps)
    e_all = keys_at(model, toks[:W], torch.arange(W, device=dev), mode).detach()
    # моя развёртка с захватом КАЖДОГО слоя
    h_mine = [None] * L
    q0 = e_all[W - int(model.nquery):].mean(0)
    en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
    q = q0; h = e_all
    k_eff = torch.sigmoid(model.k); topk = int(model.topk)
    for li, blk in enumerate(model.blocks):
        h_mine[li] = h.detach().clone()
        qn = q / (q.norm() + 1e-6)
        sim = en @ qn
        sim[W - TAIL:] = -1e9
        vals, loc = sim.topk(min(topk, W - TAIL))
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        h = blk(h, driver, k_eff)
        q = q0 + model.query_proj(h[-1]) * 0.5

print(f"W={W} mode={mode} Ppos={Ppos}")
print(f"{'слой':>5} {'cos(мой,эталон)':>16} {'relL2':>10} {'||мой||':>9} {'||этал||':>9}")
for li in range(L):
    a = h_mine[li].float(); b = caps[li].float().to(dev)
    c = F.cosine_similarity(a, b, dim=-1).mean().item()
    r = ((a - b).norm() / b.norm()).item()
    print(f"{li:>5} {c:>16.6f} {r:>10.4f} {a.norm().item():>9.2f} {b.norm().item():>9.2f}")
# финальный h (после всех 8 слоёв) — сравниваем через лосс
h_fin = h
print("\nфинальный h: норма", round(h_fin.norm().item(), 3))
