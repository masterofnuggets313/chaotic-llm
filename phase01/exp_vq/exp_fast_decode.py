"""Эксперимент 2: fast-decode прототип.

Гипотеза (из g-абляции exp1b): блоки по ВСЕМУ W не нужны на decode-шаге.
Нужно: (a) глобальные драйверы (селекция по всему W), (b) h_last (1 позиция
через блоки), (c) g по последним K позициям. Всё остальное — лишние вычисления.

Текущий decode (chat_sts_prog) пересчитывает forward_general по ВСЕМУ окну
каждым шагом -> O(W) на шаг. Мы устраняем доминирующий блок-бутылнек:
блоки гоняем только по K последних позиций + h_last, драйверы оставляем
точными (O(W*d) — это 17% времени, уберём в exp4 через Fracode).

Проверяем:
1. cos(logits_fast, logits_exact) ~ 1.0 — качество не теряется (доказано в abl2
   для g; тут доказываем для полного fast-шага).
2. tok/s fast vs exact (полный forward окна как сейчас).

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_fast_decode.py
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
    keys_at, forward_general, TAIL, TEMP)
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 4096


def fast_decode_step(model, e_all, q0, Wc, k_eff, topk, nq, TEMP=TEMP, TAIL=TAIL):
    """Быстрый decode-шаг: драйверы по всему Wc + h_last + g по K.
    e_all: (Wc, d) — статичные ключи всего окна (кэшируются между шагами)."""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
    for li, blk in enumerate(model.blocks):
        k = k_eff  # один скаляр на все слои
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


def timeit(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    dev = "cuda"
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

    # --- 1. Качество: cos logits fast vs exact на N окнах (W=65536) ---
    Wq = 65536
    print(f"Качество fast vs exact (K={K}) на 8 окнах W={Wq}...", flush=True)
    cos_list, ce_fast, ce_ex = [], [], []
    with torch.no_grad():
        for wi in range(8):
            o = 5000 + wi * Wq
            if o + Wq >= len(toks) - 1: break
            end = o + Wq
            x = toks[o:end]; target = toks[end].view(1)
            lg_ex = forward_general(model, x, mode, chunk=Wq)
            ce_ex.append(float(F.cross_entropy(lg_ex, target).item()))
            e_all = keys_at(model, x, torch.arange(Wq, device=dev), mode).detach()
            q0 = e_all[-nq:].mean(0, keepdim=True)
            lg_fast = fast_decode_step(model, e_all, q0, Wq, k_eff, topk, nq)
            ce_fast.append(float(F.cross_entropy(lg_fast, target).item()))
            cos_list.append(float(F.cosine_similarity(lg_fast, lg_ex, dim=-1).item()))
    cos_m = float(np.mean(cos_list))
    ce_ex_m = float(np.mean(ce_ex)); ce_fast_m = float(np.mean(ce_fast))
    delta_ppl = (np.exp(ce_fast_m) / np.exp(ce_ex_m) - 1) * 100
    print(f"cos(logits_fast, logits_exact) = {cos_m:.6f}")
    print(f"CE exact={ce_ex_m:.4f}  fast={ce_fast_m:.4f}  ΔPPL={delta_ppl:+.1f}%")

    # --- 2. Скорость decode-шага ---
    print(f"\nЗамер скорости decode-шага (exact=полный forward окна, как сейчас)...", flush=True)
    o = 5000

    # exact на W=65536 (полный forward — текущий decode-шаг)
    x = toks[o:o + Wq]
    t_exact = timeit(lambda: forward_general(model, x, mode, chunk=Wq), warmup=2, iters=3)
    print(f"exact decode W={Wq:>7}: {t_exact*1000:8.1f} ms -> {1/t_exact:8.2f} tok/s")

    # fast на W=65536 и W=262144
    scaling = {}
    for Wc2 in [65536, 262144]:
        x2 = toks[o:o + Wc2]
        e2 = keys_at(model, x2, torch.arange(Wc2, device=dev), mode).detach()
        q0_2 = e2[-nq:].mean(0, keepdim=True)
        t2 = timeit(lambda: fast_decode_step(model, e2, q0_2, Wc2, k_eff, topk, nq), warmup=2, iters=3)
        scaling[Wc2] = round(1 / t2, 3)
        sp = (t_exact / t2) if Wc2 == 65536 else None
        sp_s = f"  (vs exact W=65536: {sp:.1f}x)" if sp else ""
        print(f"fast  decode W={Wc2:>7}: {t2*1000:8.1f} ms -> {1/t2:8.2f} tok/s{sp_s}")

    out = {
        "K": K,
        "cos_logits": round(cos_m, 6),
        "ce_exact": round(ce_ex_m, 4), "ce_fast": round(ce_fast_m, 4),
        "delta_ppl_pct": round(delta_ppl, 1),
        "speed_exact_W65536_tok_s": round(1 / t_exact, 2),
        "speed_fast_tok_s": scaling,
        "speedup_vs_exact_W65536": round(t_exact / scaling[65536], 1),
    }
    with open(os.path.join(RESULTS, "exp_fast_decode.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_fast_decode.json")


if __name__ == "__main__":
    main()