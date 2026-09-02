"""diag_noise.py — неустойчива ли САМА модель к float-шуму?

Тест: один и тот же форвард, один и тот же вход, но разное разбиение на чанки
(=> другой порядок редукций в matmul => другой float-шум ~1e-7).
Если состояние на выходе расходится — это ХАОТИЧЕСКАЯ НЕУСТОЙЧИВОСТЬ модели,
а не ошибка развёртки. Это критично и для честности вывода "h не надо хранить".
"""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import forward_general

dev = "cuda"; torch.manual_seed(0)
import numpy as np, final_benchmark as fb
head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
tok = fb.make_bpe(head); V = tok.get_vocab_size(); nh = len(tok.encode(head).ids)
ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
toks = torch.tensor(ids[nh:], dtype=torch.long, device=dev)

model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                       driver_mode="sts_prog", alpha=0.3, temp=0.3).to(dev).eval()
model.load_state_dict(torch.load(os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
for p in model.parameters(): p.requires_grad_(False)
L = len(model.blocks); Ppos = model.pos.shape[1]

for W in (16384, 65536):
    mode = "trained" if W <= Ppos else "cyclic"
    runs = {}
    for ch in (1024, 4096, W):
        caps = [None] * L
        with torch.no_grad():
            lg = forward_general(model, toks[:W], mode, chunk=ch, capture=caps)
        runs[ch] = ([c.float() for c in caps], lg.float().clone())
    keys = sorted(runs)
    a, la = runs[keys[0]]
    print(f"\n=== W={W} mode={mode} ===")
    print(f"{'слой':>5} " + " ".join(f"cos(ch{keys[0]} vs ch{c})" for c in keys[1:]))
    for li in range(L):
        row = [F.cosine_similarity(a[li], runs[c][0][li], dim=-1).mean().item() for c in keys[1:]]
        print(f"{li:>5} " + " ".join(f"{v:>18.6f}" for v in row))
    kl = [F.kl_div(F.log_softmax(la, -1), F.log_softmax(runs[c][1], -1), reduction="sum").item()
          for c in keys[1:]]
    print(f"KL(logits) между chunk={keys[0]} и " +
          ", ".join(f"chunk={c}: {v:.4f}" for c, v in zip(keys[1:], kl)))
    print("совпадение argmax: " + ", ".join(
        str(bool(la.argmax(-1).item() == runs[c][1].argmax(-1).item())) for c in keys[1:]))
