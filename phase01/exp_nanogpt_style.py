"""exp_nanogpt_style.py — контр-эксперимент: трансформер в nanoGPT-режиме.

Комментатор утверждает: наш TransformerLM обучен слабо (один токен на окно).
Проверяем: трансформер с лоссом по ВСЕМ позициям (как nanoGPT) vs STS-Prog.

Сравнение: PPL + retrieval (как в final_benchmark).
"""
import os, sys, json, time, math
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_vq"))
from parametric_models import TransformerLM, count_params
from exp_vq.final_benchmark import build_data, build_order3, eval_ppl, eval_retrieval, W
from exp_vq.models_pc import build_pc_model
from exp_vq.match_transformer import pick_tf_dims

HERE = os.path.dirname(os.path.abspath(__file__))
EXPVQ = os.path.join(HERE, "exp_vq")
STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000


def train_gpt_style(model, train_ids, seed=0, steps=STEPS, batch=BATCH, lr=LR):
    """Тренировка nanoGPT-стиля: цель = каждый следующий токен на каждой позиции."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    t0 = time.time()
    for step in range(1, steps + 1):
        lr_scale = min(1.0, step / WARMUP)
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_scale
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        # цель: сдвинутые вправо токены (все позиции)
        Y_all = torch.tensor(np.stack([train_ids[i + 1:i + W + 1] for i in s]), dtype=torch.long, device="cuda")
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            # полные logits по всем позициям
            h = model.embed(X) + model.pos
            for blk in model.blocks:
                h = blk(h)
            h = model.ln_f(h)
            logits_all = model.head(h)          # (B, W, V)
            loss = lossf(logits_all.reshape(-1, logits_all.shape[-1]), Y_all.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 1000 == 0:
            print(f"  [{step}/{steps}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return model


def train_last_token(model, train_ids, seed=0, steps=STEPS, batch=BATCH, lr=LR):
    """Тренировка как в final_benchmark: цель = только последний токен окна."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    t0 = time.time()
    for step in range(1, steps + 1):
        lr_scale = min(1.0, step / WARMUP)
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_scale
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device="cuda")
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(X)
            loss = lossf(logits, Y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 1000 == 0:
            print(f"  [{step}/{steps}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return model


def main():
    print("Loading data...", flush=True)
    tok, V, train_ids = build_data()
    n = len(train_ids)
    tr_ids = train_ids[:n - n // 20]
    te_ids = train_ids[n - n // 20:]
    print(f"train={len(tr_ids):,} test={len(te_ids):,} V={V}", flush=True)

    # STS-Prog эталон (как в final_benchmark: d=192, layers=8)
    ck_sts = os.path.join(HERE, "exp_vq", "ckpt_nano_sts.pt")
    ck_tf_last = os.path.join(HERE, "exp_vq", "ckpt_nano_tf_last.pt")
    ck_tf_gpt = os.path.join(HERE, "exp_vq", "ckpt_nano_tf_gpt.pt")

    print("\n=== STS-Prog (последний токен, как в бенчмарке) ===", flush=True)
    sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                         driver_mode="sts_prog", alpha=0.3, temp=0.3).to("cuda")
    if os.path.exists(ck_sts):
        sts.load_state_dict(torch.load(ck_sts, map_location="cuda", weights_only=False))
        print("STS: loaded from checkpoint", flush=True)
    else:
        sts = train_last_token(sts, tr_ids, seed=0)
        torch.save(sts.state_dict(), ck_sts)
    print(f"STS params={count_params(sts):,}", flush=True)

    # Трансформер в режиме final_benchmark (последний токен) — D из pick_tf_dims
    print("\n=== Transformer (последний токен, как в бенчмарке) ===", flush=True)
    D_tf = pick_tf_dims(900_000, V, W, layers=8, heads=4)
    tf_last = TransformerLM(V, W, D=D_tf, HEADS=4, LAYERS=8).to("cuda")
    print(f"TF params={count_params(tf_last):,} (D={D_tf})", flush=True)
    if os.path.exists(ck_tf_last):
        tf_last.load_state_dict(torch.load(ck_tf_last, map_location="cuda", weights_only=False))
        print("TF (last): loaded from checkpoint", flush=True)
    else:
        tf_last = train_last_token(tf_last, tr_ids, seed=0)
        torch.save(tf_last.state_dict(), ck_tf_last)

    # Трансформер в nanoGPT-режиме (все позиции)
    print("\n=== Transformer (nanoGPT-стиль: все позиции) ===", flush=True)
    tf_gpt = TransformerLM(V, W, D=D_tf, HEADS=4, LAYERS=8).to("cuda")
    if os.path.exists(ck_tf_gpt):
        tf_gpt.load_state_dict(torch.load(ck_tf_gpt, map_location="cuda", weights_only=False))
        print("TF (gpt): loaded from checkpoint", flush=True)
    else:
        tf_gpt = train_gpt_style(tf_gpt, tr_ids, seed=0)
        torch.save(tf_gpt.state_dict(), ck_tf_gpt)

    # Eval: PPL + retrieval
    print("\n=== EVAL ===", flush=True)
    results = {"sts_last_token": {}, "tf_last_token": {}, "tf_gpt_style": {}}
    for name, m in [("sts_last_token", sts), ("tf_last_token", tf_last), ("tf_gpt_style", tf_gpt)]:
        ppl = eval_ppl(m, te_ids)
        ret = eval_retrieval(m, te_ids, distances=(16, 64, 256))
        ret_acc = {str(k): round(v["accuracy"], 3) for k, v in ret.items()}
        print(f"{name}: ppl={ppl:.3f} ret={ret_acc}", flush=True)
        results[name] = {"ppl": round(ppl, 3), "retrieval": ret_acc}

    with open(os.path.join(HERE, "exp_vq", "results", "exp_nanogpt_style.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/exp_nanogpt_style.json", flush=True)


if __name__ == "__main__":
    main()