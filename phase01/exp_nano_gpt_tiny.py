"""exp_nano_gpt_tiny.py — nanoGPT (karpathy, код как есть) с МАЛЫМ конфигом ~900K.

Финальный бой в равном весе: та же архитектура nanoGPT (karpathy/model.py),
но конфиг d=128, L=4, h=4 -> ~957K параметров (STS-Prog = 900K).
Родное обучение nanoGPT (все позиции), тот же корпус, тот же протокол.
"""
import os, sys, json, time, math
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "exp_vq"))
sys.path.insert(0, os.path.join(HERE, "exp_vq", "nanogpt_ref"))

import final_benchmark as fb
from final_benchmark import eval_ppl, eval_retrieval
from models_pc import build_pc_model
from parametric_models import count_params

W = 256
STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000

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
    return len(tok.get_vocab()), ids[:int(n * 0.8)], ids[int(n * 0.8):]

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    V, tr, te = load_corpus()
    print(f"train={len(tr):,} test={len(te):,} V={V}", flush=True)

    spec = importlib.util.spec_from_file_location("nanogpt_model",
        os.path.join(HERE, "exp_vq", "nanogpt_ref", "model.py"))
    ngmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ngmod)
    GPT, GPTConfig = ngmod.GPT, ngmod.GPTConfig

    # nanoGPT в равном весе: d=128, L=4, h=4 -> ~957K (STS 900K)
    ck = os.path.join(HERE, "exp_vq", "ckpt_nano_gpt_tiny.pt")
    cfg = GPTConfig(block_size=W, vocab_size=V, n_layer=4, n_head=4, n_embd=128,
                    dropout=0.0, bias=True)
    gpt = GPT(cfg).to("cuda")
    nparams = count_params(gpt)
    print(f"nanoGPT-tiny params={nparams:,} (STS=900,353)", flush=True)

    if os.path.exists(ck):
        print("Загружаю чекпоинт nanoGPT-tiny...", flush=True)
        gpt.load_state_dict(torch.load(ck, map_location="cuda"))
        gpt.eval()
    else:
        opt = torch.optim.AdamW(gpt.parameters(), lr=LR, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        n = len(tr) - W - 1
        rng = np.random.default_rng(0)
        t0 = time.time()
        for step in range(1, STEPS + 1):
            lr_scale = min(1.0, step / WARMUP)
            for pg in opt.param_groups:
                pg["lr"] = LR * lr_scale
            s = rng.integers(0, n, size=BATCH)
            X = torch.tensor(np.stack([tr[i:i + W] for i in s]), dtype=torch.long, device="cuda")
            Y = torch.tensor(np.stack([tr[i + 1:i + W + 1] for i in s]), dtype=torch.long, device="cuda")
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, loss = gpt(X, Y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 1000 == 0:
                print(f"  [{step}/{STEPS}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        gpt.eval()
        torch.save(gpt.state_dict(), ck)

    # ---- STS-Prog из чекпоинта (для самодостаточного JSON) ----
    sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                         driver_mode="sts_prog", alpha=0.3, temp=0.3).to("cuda")
    sts.load_state_dict(torch.load(os.path.join(HERE, "exp_vq", "ckpt_nano_ref_sts.pt"),
                                   map_location="cuda"))

    # ---- EVAL ----
    print("\n=== EVAL ===", flush=True)
    results = {}

    ppl_sts = eval_ppl(sts, te)
    ret_sts = eval_retrieval(sts, te, distances=(16, 64, 256))
    ret_sts_acc = {str(k): round(v["accuracy"], 3) for k, v in ret_sts.items()}
    print(f"STS-Prog: ppl={ppl_sts:.3f} ret={ret_sts_acc}", flush=True)
    results["sts_prog"] = {"ppl": round(ppl_sts, 3), "retrieval": ret_sts_acc, "params": 900353}

    # PPL nanoGPT-tiny: последний токен окна
    gp = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(te) - W - 1, W * 4):
            X = torch.tensor([te[i:i + W]], dtype=torch.long, device="cuda")
            Y = torch.tensor([te[i + W]], dtype=torch.long, device="cuda")
            logits, _ = gpt(X)
            p = F.softmax(logits[0, -1], dim=-1)
            gp += -math.log(p[Y[0]].item() + 1e-12)
            cnt += 1
    gp = math.exp(gp / cnt)
    print(f"nanoGPT-tiny PPL={gp:.3f}", flush=True)

    class GPTAdapter(nn.Module):
        def __init__(self, g):
            super().__init__()
            self.gpt = g
        def forward(self, X):
            logits, _ = self.gpt(X)
            return logits

    ret_gpt = eval_retrieval(GPTAdapter(gpt), te, distances=(16, 64, 256))
    ret_gpt_acc = {str(k): round(v["accuracy"], 3) for k, v in ret_gpt.items()}
    print(f"nanoGPT-tiny: ret={ret_gpt_acc}", flush=True)
    results["nano_gpt_tiny"] = {"ppl": round(gp, 3), "retrieval": ret_gpt_acc, "params": nparams}

    results["meta"] = {
        "corpus": "corpus_public.txt (TheAlgorithms, публичный)",
        "protocol": "BPE-512, split 80/20, W=256, eval last-token PPL, 6000 steps both",
        "nanoGPT_tiny_config": "karpathy model.py as-is, n_embd=128, n_layer=4, n_head=4 (equal budget ~957K)",
        "note": "equal-budget fight: karpathy code, native all-positions training"
    }

    out = os.path.join(HERE, "exp_vq", "results", "exp_nano_gpt_tiny.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}", flush=True)

if __name__ == "__main__":
    main()
