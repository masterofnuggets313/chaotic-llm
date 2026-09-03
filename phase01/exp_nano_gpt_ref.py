"""exp_nano_gpt_ref.py — nanoGPT (karpathy) на нашем корпусе.

Берём GPT из nanoGPT/model.py как есть (архитектура официальная).
Обучаем на публичном корпусе (TheAlgorithms) с нашим BPE-512.
Сравниваем со STS-Prog по PPL и retrieval.

Конфиг: n_embd=384, n_layer=8, n_head=8, vocab=512, ~14.7M params
(в 16× больше чем STS-Prog 900K — Solaris vs наша машина)

С чекпоинтами + resume: если чекпоинт есть — загружаем, не переобучаем.
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

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from parametric_models import TransformerLM, count_params
import final_benchmark as fb
from final_benchmark import build_order3, eval_ppl, eval_retrieval, W
from models_pc import build_pc_model

def make_bpe(text, vocab=512):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<pad>"], min_frequency=2)
    tok.train_from_iterator([text], trainer)
    return tok

def encode(tok, text):
    return tok.encode(text).ids

STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000
W = 256

def load_corpus():
    ROOT = os.path.abspath(os.path.join(HERE, ".."))
    corpus_path = os.path.join(ROOT, "phase01", "corpus_public.txt")
    text = fb.load_chars(corpus_path, fb.MAX_TRAIN)
    tok = make_bpe(text, 512)
    ids = encode(tok, text)
    n = len(ids)
    tr = ids[:int(n * 0.8)]
    te = ids[int(n * 0.8):]
    return tok, len(tok.get_vocab()), tr, te

CONTEXTS = [
    "def binary_search(arr, target):",
    "import numpy as np",
    "class Stack:",
    "def quicksort(lst):",
    "x = [1, 2, 3]",
    "for i in range(10):",
    "def factorial(n):",
    "print('hello world')",
]

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    ck_sts = os.path.join(HERE, "exp_vq", "ckpt_nano_ref_sts.pt")
    ck_ng = os.path.join(HERE, "exp_vq", "ckpt_nano_ref_gpt.pt")

    print("Loading public corpus...", flush=True)
    tok, V, tr, te = load_corpus()
    print(f"train={len(tr):,} test={len(te):,} V={V}", flush=True)

    # 1) STS-Prog эталон (900K, один токен)
    if os.path.exists(ck_sts):
        print("\n=== STS-Prog: загружаю чекпоинт ===", flush=True)
        sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                             driver_mode="sts_prog", alpha=0.3, temp=0.3).to("cuda")
        sts.load_state_dict(torch.load(ck_sts, map_location="cuda"))
        sts_ppl = eval_ppl(sts, te)
        print(f"STS PPL={sts_ppl:.3f}", flush=True)
    else:
        print("\n=== STS-Prog (эталон, 900K) ===", flush=True)
        sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                             driver_mode="sts_prog", alpha=0.3, temp=0.3).to("cuda")
        from final_benchmark import train_model
        old_W = fb.W
        fb.W = W
        sts_info = train_model(sts, tr, seed=0, steps=STEPS, batch=BATCH, lr=LR)
        fb.W = old_W
        sts_ppl = sts_info["best_ppl"]
        print(f"STS PPL={sts_ppl:.3f}", flush=True)
        torch.save(sts.state_dict(), ck_sts)

    # 2) nanoGPT (их архитектура как есть, их родное обучение — все позиции)
    spec = importlib.util.spec_from_file_location("nanogpt_model",
        os.path.join(HERE, "exp_vq", "nanogpt_ref", "model.py"))
    ngmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ngmod)
    GPT, GPTConfig = ngmod.GPT, ngmod.GPTConfig

    if os.path.exists(ck_ng):
        print("\n=== nanoGPT: загружаю чекпоинт ===", flush=True)
        cfg = GPTConfig(block_size=W, vocab_size=V, n_layer=8, n_head=8, n_embd=384,
                        dropout=0.0, bias=True)
        gpt = GPT(cfg).to("cuda")
        gpt.load_state_dict(torch.load(ck_ng, map_location="cuda"))
        gpt.eval()
    else:
        print("\n=== nanoGPT (код karpathy как есть, ~14.7M) ===", flush=True)
        cfg = GPTConfig(block_size=W, vocab_size=V, n_layer=8, n_head=8, n_embd=384,
                        dropout=0.0, bias=True)
        gpt = GPT(cfg).to("cuda")
        print(f"nanoGPT params={count_params(gpt):,}", flush=True)

        opt = torch.optim.AdamW(gpt.parameters(), lr=LR, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        lossf = nn.CrossEntropyLoss()
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
        torch.save(gpt.state_dict(), ck_ng)

    # 3) Eval: PPL (последний токен окна) + retrieval
    print("\n=== EVAL ===", flush=True)
    results = {"sts_prog": {}, "nano_gpt_ref": {}}

    ppl_sts = eval_ppl(sts, te)
    ret_sts = eval_retrieval(sts, te, distances=(16, 64, 256))
    ret_acc_sts = {str(k): round(v["accuracy"], 3) for k, v in ret_sts.items()}
    print(f"STS-Prog: ppl={ppl_sts:.3f} ret={ret_acc_sts}", flush=True)
    results["sts_prog"] = {"ppl": round(ppl_sts, 3), "retrieval": ret_acc_sts, "params": count_params(sts)}

    # nanoGPT: считаем PPL в режиме "последний токен" (как STS для честности сравнения)
    gpt_ppl = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(te) - W - 1, W * 4):
            X = torch.tensor([te[i:i + W]], dtype=torch.long, device="cuda")
            Y = torch.tensor([te[i + W]], dtype=torch.long, device="cuda")
            logits, _ = gpt(X)
            logit = logits[0, -1]
            p = F.softmax(logit, dim=-1)
            gpt_ppl += -math.log(p[Y[0]].item() + 1e-12)
            cnt += 1
    gpt_ppl = math.exp(gpt_ppl / cnt)
    print(f"nanoGPT PPL={gpt_ppl:.3f}", flush=True)
    results["nano_gpt_ref"] = {"ppl": round(gpt_ppl, 3), "params": count_params(gpt)}

    # retrieval для nanoGPT (используем eval_retrieval, который ждёт model с .predict)
    # делаем адаптер
    class GPTAdapter:
        def __init__(self, gpt):
            self.gpt = gpt
        def predict(self, ctx, target_idx):
            with torch.no_grad():
                X = torch.tensor([ctx], dtype=torch.long, device="cuda")
                logits, _ = self.gpt(X)
                return logits[0, -1].detach().cpu().numpy()
    ret_gpt = eval_retrieval(GPTAdapter(gpt), te, distances=(16, 64, 256))
    ret_acc_gpt = {str(k): round(v["accuracy"], 3) for k, v in ret_gpt.items()}
    print(f"nanoGPT: ret={ret_acc_gpt}", flush=True)
    results["nano_gpt_ref"]["retrieval"] = ret_acc_gpt

    os.makedirs(os.path.join(HERE, "exp_vq", "results"), exist_ok=True)
    out = os.path.join(HERE, "exp_vq", "results", "exp_nano_gpt_ref.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
