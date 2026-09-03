"""exp_fair_protocol.py — честный протокол v2 (ответ на критику комментатора).

Претензии: (1) nanoGPT гонялся без рецепта Карпати -> PPL 210 artefact,
(2) отбор моделей асимметричен (STS best-val, GPT последний шаг).

Fair-протокол:
- train 90% / val 10% / test (test не трогаем до финала)
- ВСЕ модели: cosine decay 5e-4 -> 5e-5, warmup 1000, clip 1.0, best-val checkpoint
- STS/TF: wd 0.01 uniform (как в бенчмарке)
- nanoGPT: karpathy recipe — dropout 0.1, wd 0.1 ТОЛЬКО на ndim>=2 (его схема групп)
  - вариант A: все позиции (родной лосс model(X, Y) — karpathy сам решейпит)
  - вариант B: последний токен (наш режим)
- Resume: ckpt_fair_sts.pt / ckpt_fair_tf.pt уже обучены в предыдущем прогоне
"""
import os, sys, json, time, math
import importlib.util
import copy
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "exp_vq"))
sys.path.insert(0, os.path.join(HERE, "exp_vq", "nanogpt_ref"))

import final_benchmark as fb
from final_benchmark import eval_ppl, eval_retrieval
from models_pc import build_pc_model
from parametric_models import count_params, TransformerLM

W = 256
STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000
EVAL_EVERY = 500

def load_corpus():
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    corpus_path = os.path.join(ROOT, "phase01", "corpus_public.txt")
    text = fb.load_chars(corpus_path, fb.MAX_TRAIN)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["<pad>"], min_frequency=2)
    tok.train_from_iterator([text], trainer)
    ids = tok.encode(text).ids
    n = len(ids)
    tr = ids[:int(n * 0.8)]
    te = ids[int(n * 0.8):]
    v = int(len(tr) * 0.1)
    return len(tok.get_vocab()), tr[:-v], tr[-v:], te

def cosine_lr(step):
    if step < WARMUP:
        return LR * step / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 5e-5 + 0.5 * (LR - 5e-5) * (1 + math.cos(math.pi * t))

def logits_last(model, X):
    """Унифицированно: last-token logits (B, V) для любой модели."""
    out = model(X)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 3:
        out = out[:, -1, :]
    return out

class LastTokenAdapter(nn.Module):
    """Обёртка: forward -> last-token logits; делегирует eval()/train()/state_dict()."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, X):
        return logits_last(self.m, X)

class GPTTupleAdapter(nn.Module):
    """nanoGPT возвращает (logits, loss); eval_* ждут logits."""
    def __init__(self, g):
        super().__init__()
        self.gpt = g
    def forward(self, X):
        logits, _ = self.gpt(X)
        return logits

def last_adapter(m):
    return LastTokenAdapter(m).to(next(m.parameters()).device).eval()

def make_opt(model, wd, karpathy_groups):
    if karpathy_groups:
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([
            {"params": decay, "weight_decay": wd},
            {"params": nodecay, "weight_decay": 0.0},
        ], lr=LR)
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

def train_generic(model, tr, val, loss_mode, tag, wd=0.01, karpathy_groups=False):
    """Единый цикл: cosine decay, best-val selection.
    loss_mode: 'last' (одна позиция) | 'all' (родной лосс GPT по всем позициям)."""
    print(f"\n=== {tag} (wd={wd}, groups={karpathy_groups}) ===", flush=True)
    model = model.to("cuda")
    opt = make_opt(model, wd, karpathy_groups)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss()
    n = len(tr) - W - 1
    rng = np.random.default_rng(0)
    best_val, best_state, best_step = float("inf"), None, 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        lr_now = cosine_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        s = rng.integers(0, n, size=BATCH)
        X = torch.tensor(np.stack([tr[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            if loss_mode == "last":
                Y = torch.tensor([tr[i + W] for i in s], dtype=torch.long, device="cuda")
                loss = lossf(logits_last(model, X), Y)
            else:
                Y = torch.tensor(np.stack([tr[i + 1:i + W + 1] for i in s]), dtype=torch.long, device="cuda")
                _, loss = model(X, Y)  # родной лосс karpathy по всем позициям
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                vp = eval_ppl(last_adapter(model), val)
            model.train()
            if vp < best_val:
                best_val, best_step = vp, step
                best_state = copy.deepcopy(model.state_dict())
            star = "*" if vp <= best_val else " "
            print(f"  [{step}/{STEPS}] loss={loss.item():.3f} val_ppl={vp:.2f}{star} ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_val, best_step

def eval_all(model, te, tag):
    ppl = eval_ppl(last_adapter(model), te)
    ret = eval_retrieval(model, te, distances=(16, 64, 256))
    ret_acc = {str(k): round(v["accuracy"], 3) for k, v in ret.items()}
    print(f"{tag}: test_ppl={ppl:.3f} ret={ret_acc}", flush=True)
    return round(ppl, 3), ret_acc

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    V, tr, val, te = load_corpus()
    print(f"train={len(tr):,} val={len(val):,} test={len(te):,} V={V}", flush=True)

    spec = importlib.util.spec_from_file_location("nanogpt_model",
        os.path.join(HERE, "exp_vq", "nanogpt_ref", "model.py"))
    ngmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ngmod)
    GPT, GPTConfig = ngmod.GPT, ngmod.GPTConfig

    results = {}
    os.makedirs(os.path.join(HERE, "exp_vq", "results"), exist_ok=True)

    # ---- 1) STS-Prog: resume из чекпоинта (обучен в пред. прогоне, best-val state) ----
    ck_sts = os.path.join(HERE, "exp_vq", "ckpt_fair_sts.pt")
    sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                         driver_mode="sts_prog", alpha=0.3, temp=0.3)
    if os.path.exists(ck_sts):
        print("\n=== STS-Prog: resume из ckpt_fair_sts.pt ===", flush=True)
        sts.load_state_dict(torch.load(ck_sts, map_location="cuda"))
        sts = sts.to("cuda").eval()
        bv, bs = eval_ppl(last_adapter(sts), val), -1
    else:
        bv, bs = train_generic(sts, tr, val, "last", "STS-Prog (900K)")
        torch.save(sts.state_dict(), ck_sts)
    ppl, ret_acc = eval_all(sts, te, "STS-Prog")
    results["sts_prog"] = {"test_ppl": ppl, "retrieval": ret_acc,
                           "val_ppl": round(bv, 3), "best_step": bs, "params": count_params(sts)}

    # ---- 2) TF-last-token: resume ----
    ck_tf = os.path.join(HERE, "exp_vq", "ckpt_fair_tf.pt")
    tf = TransformerLM(V, W, D=88, HEADS=4, LAYERS=4)
    if os.path.exists(ck_tf):
        print("\n=== TF-last: resume из ckpt_fair_tf.pt ===", flush=True)
        tf.load_state_dict(torch.load(ck_tf, map_location="cuda"))
        tf = tf.to("cuda").eval()
        bv2, bs2 = eval_ppl(last_adapter(tf), val), -1
    else:
        bv2, bs2 = train_generic(tf, tr, val, "last", "TF-last-token (наша копия)")
        torch.save(tf.state_dict(), ck_tf)
    ppl2, ret2_acc = eval_all(tf, te, "TF-last")
    results["tf_last"] = {"test_ppl": ppl2, "retrieval": ret2_acc,
                          "val_ppl": round(bv2, 3), "best_step": bs2, "params": count_params(tf)}

    # ---- 3) nanoGPT-tiny A: все позиции, karpathy recipe ----
    cfgA = GPTConfig(block_size=W, vocab_size=V, n_layer=4, n_head=4, n_embd=128,
                     dropout=0.1, bias=True)
    gA = GPT(cfgA)
    bvA, bsA = train_generic(gA, tr, val, "all", "nanoGPT-tiny ALL-positions",
                             wd=0.1, karpathy_groups=True)
    torch.save(gA.state_dict(), os.path.join(HERE, "exp_vq", "ckpt_fair_gpt_all.pt"))
    gA_eval = GPTTupleAdapter(gA).to("cuda").eval()
    pplA, retA_acc = eval_all(gA_eval, te, "nanoGPT-A(all)")
    results["nanoGPT_all"] = {"test_ppl": pplA, "retrieval": retA_acc,
                              "val_ppl": round(bvA, 3), "best_step": bsA, "params": count_params(gA)}

    # ---- 4) nanoGPT-tiny B: последний токен, karpathy recipe ----
    cfgB = GPTConfig(block_size=W, vocab_size=V, n_layer=4, n_head=4, n_embd=128,
                     dropout=0.1, bias=True)
    gB = GPT(cfgB)
    bvB, bsB = train_generic(gB, tr, val, "last", "nanoGPT-tiny LAST-token",
                             wd=0.1, karpathy_groups=True)
    torch.save(gB.state_dict(), os.path.join(HERE, "exp_vq", "ckpt_fair_gpt_last.pt"))
    gB_eval = GPTTupleAdapter(gB).to("cuda").eval()
    pplB, retB_acc = eval_all(gB_eval, te, "nanoGPT-B(last)")
    results["nanoGPT_last"] = {"test_ppl": pplB, "retrieval": retB_acc,
                               "val_ppl": round(bvB, 3), "best_step": bsB, "params": count_params(gB)}

    results["meta"] = {
        "protocol": "FAIR v2: train90/val10/test; cosine 5e-4->5e-5; warmup 1000; clip 1.0; best-val ckpt для ВСЕХ",
        "nanoGPT_recipe": "dropout=0.1; wd=0.1 на ndim>=2, 0 на biases/norms (karpathy grouping); оба режима обучения",
        "STS_TF_recipe": "wd=0.01 uniform (как в бенчмарке репо)",
        "corpus": "corpus_public.txt, BPE-512, W=256, 6000 шагов, seed 0",
        "note": "Ответ на критику: nanoGPT с полным родным рецептом; отбор best-val симметричен; last-token PPL у всех одинаковый",
    }
    out = os.path.join(HERE, "exp_vq", "results", "exp_fair_protocol.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out}", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
