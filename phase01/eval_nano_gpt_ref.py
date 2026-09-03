"""eval_nano_gpt_ref.py — eval из чекпоинтов (без переобучения).

Грузит ckpt_nano_ref_sts.pt и ckpt_nano_ref_gpt.pt,
считает PPL (последний токен) + retrieval для обоих, сохраняет results/exp_nano_gpt_ref.json.
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
from final_benchmark import build_order3, eval_ppl, eval_retrieval, W
from models_pc import build_pc_model
from parametric_models import count_params

W = 256

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
    return len(tok.get_vocab()), tr, te

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    V, tr, te = load_corpus()
    print(f"train={len(tr):,} test={len(te):,} V={V}", flush=True)

    results = {}

    # ---- STS-Prog ----
    ck_sts = os.path.join(HERE, "exp_vq", "ckpt_nano_ref_sts.pt")
    sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                         driver_mode="sts_prog", alpha=0.3, temp=0.3).to("cuda")
    sts.load_state_dict(torch.load(ck_sts, map_location="cuda"))
    ppl_sts = eval_ppl(sts, te)
    ret_sts = eval_retrieval(sts, te, distances=(16, 64, 256))
    ret_acc_sts = {str(k): round(v["accuracy"], 3) for k, v in ret_sts.items()}
    print(f"STS-Prog: ppl={ppl_sts:.3f} ret={ret_acc_sts}", flush=True)
    results["sts_prog"] = {"ppl": round(ppl_sts, 3), "retrieval": ret_acc_sts, "params": count_params(sts)}

    # ---- nanoGPT ----
    spec = importlib.util.spec_from_file_location("nanogpt_model",
        os.path.join(HERE, "exp_vq", "nanogpt_ref", "model.py"))
    ngmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ngmod)
    GPT, GPTConfig = ngmod.GPT, ngmod.GPTConfig

    ck_ng = os.path.join(HERE, "exp_vq", "ckpt_nano_ref_gpt.pt")
    cfg = GPTConfig(block_size=W, vocab_size=V, n_layer=8, n_head=8, n_embd=384,
                    dropout=0.0, bias=True)
    gpt = GPT(cfg).to("cuda")
    gpt.load_state_dict(torch.load(ck_ng, map_location="cuda"))
    gpt.eval()

    # PPL: последний токен окна (как STS)
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

    # retrieval: адаптер с eval() и model(X) -> logits
    class GPTAdapter(nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt
        def forward(self, X):
            logits, _ = self.gpt(X)
            return logits

    adapter = GPTAdapter(gpt)
    ret_gpt = eval_retrieval(adapter, te, distances=(16, 64, 256))
    ret_acc_gpt = {str(k): round(v["accuracy"], 3) for k, v in ret_gpt.items()}
    print(f"nanoGPT: ret={ret_acc_gpt}", flush=True)

    results["nano_gpt_ref"] = {"ppl": round(gpt_ppl, 3), "retrieval": ret_acc_gpt,
                               "params": count_params(gpt)}
    results["meta"] = {
        "corpus": "corpus_public.txt (TheAlgorithms, публичный)",
        "protocol": "BPE-512, split 80/20, W=256, both eval last-token PPL",
        "nanoGPT_config": "n_embd=384, n_layer=8, n_head=8, vocab=512 (karpathy code as-is)",
        "nanoGPT_steps": 6000, "STS_steps": 6000,
        "note": "nanoGPT 16x params; trained in native all-positions mode"
    }

    os.makedirs(os.path.join(HERE, "exp_vq", "results"), exist_ok=True)
    out = os.path.join(HERE, "exp_vq", "results", "exp_nano_gpt_ref.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {out}", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
