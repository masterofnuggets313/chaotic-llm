"""train_chat.py — обучение STS-Prog на русском чат-датасете (ru_turbo_alpaca).

Формат: <user>: {instruction} \\n <bot>: {output}
BPE-токенизатор обучен на русском тексте (V=2048 — русский требует больше).

Использование:
  python train_chat.py --d 384 --layers 12 --steps 20000 --batch 32
"""
import os, sys, json, time, math, argparse
import torch, torch.nn as nn
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)
from models_pc import PurePCLM

W = 256
VOCAB = 2048
LR = 5e-4
WARMUP = 1000
STEPS = 20000
N_EVAL = 2000
D_MODEL = 384
LAYERS = 12
K_INIT = 1.2
ALPHA = 0.3
BATCH = 32
SYNC_STEPS = 8

def make_bpe(text, vocab=VOCAB):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>", "<user>", "<bot>"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok

def format_example(instruction, output):
    return f"<user>: {instruction}\n<bot>: {output}\n<|endoftext|>"

def build_dataset(rows):
    texts = []
    for r in rows:
        instr = r["instruction"]
        if r.get("input"):
            instr = f"{instr}\n{r['input']}"
        texts.append(format_example(instr, r["output"]))
    return texts

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    data_path = os.path.join(HERE, DATA_FILE)
    print(f"Loading {data_path}...", flush=True)
    with open(data_path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"rows: {len(rows)}", flush=True)

    texts = build_dataset(rows)
    full_text = "\n".join(texts)
    print(f"total text: {len(full_text):,} chars", flush=True)

    tok = make_bpe(full_text[:10_000_000])
    V = tok.get_vocab_size()
    print(f"V={V}", flush=True)

    # токенизация чанками
    CH = 2_000_000
    chunks = []
    for i in range(0, len(full_text), CH):
        chunk = tok.encode(full_text[i:i + CH]).ids
        chunks.append(np.array(chunk, dtype=np.int32))
    ids = np.concatenate(chunks)
    del chunks
    print(f"total tokens: {len(ids):,}", flush=True)

    n = len(ids)
    train_ids = ids[:n - n // 20]
    test_ids = ids[n - n // 20:]
    print(f"train: {len(train_ids):,} test: {len(test_ids):,}", flush=True)

    model = PurePCLM(V, d=D_MODEL, layers=LAYERS, k_init=K_INIT, alpha=ALPHA,
                     sync_steps=SYNC_STEPS, driver_mode="sts_prog").to("cuda")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam:,} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(1)
    t0 = time.time()
    N = len(train_ids) - W - 1

    # aux loss для селекции (мультиголовая: равномерная + ближайшее)
    def aux_target(sim, X):
        B = X.shape[0]
        last_tok = X[:, -1]
        pos_mask = (X == last_tok.unsqueeze(1)).float()
        pos_mask[:, W - 8:] = 0.0
        pos_sum = pos_mask.sum(dim=1, keepdim=True)
        valid = (pos_sum > 0).squeeze(1)
        gauss = torch.zeros_like(sim)
        if valid.any():
            pos_idx = torch.arange(W, device=X.device).unsqueeze(0)
            dist = (pos_idx - (W - 9)).abs()
            nnd = dist + (1 - pos_mask) * 1e9
            nearest = nnd.argmin(dim=1, keepdim=True)
            sigma = 2.0
            gauss = torch.exp(-(dist - nearest).pow(2) / (2 * sigma ** 2))
            gauss = gauss / (gauss.sum(dim=1, keepdim=True) + 1e-6)
        uniform = pos_mask / pos_sum.clamp(min=1e-6)
        uniform[~valid] = 0.0
        target = 0.7 * gauss + 0.3 * uniform
        # КРИТИЧНО: обнуляем target на замаскированных позициях (W-8:W),
        # там sim=-1e9 → log_softmax даёт огромные отрицательные значения,
        # и target*log_sm взрывается (это и был баг: aux=98M).
        target[:, W - 8:] = 0.0
        return target, valid

    for step in range(1, STEPS + 1):
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * step / WARMUP
        idx = rng.integers(0, N, size=BATCH)
        X = np.stack([train_ids[i:i + W] for i in idx])
        Y = np.stack([train_ids[i + 1:i + W + 1] for i in idx])
        Xt = torch.tensor(X, dtype=torch.long, device="cuda")
        Yt = torch.tensor(Y, dtype=torch.long, device="cuda")

        with torch.amp.autocast("cuda"):
            logits = model(Xt)
            main_loss = lossf(logits, Yt[:, -1])  # предсказываем последний токен
            aux_loss = torch.tensor(0.0, device="cuda")
            if model._last_sim is not None:
                sim = model._last_sim
                target, valid = aux_target(sim, Xt)
                if valid.any():
                    log_sm = torch.log_softmax(sim / 0.3, dim=1)
                    aux_loss = -(target * log_sm).sum(dim=1)[valid].mean()
            loss = main_loss + 0.5 * aux_loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 2000 == 0 or step == 1:
            print(f"  [{step}/{STEPS}] loss={loss.item():.3f} main={main_loss.item():.3f} aux={aux_loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
            torch.save(model.state_dict(), os.path.join(HERE, f"chat_ckpt_{step}.pt"))

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)

    # eval: PPL на тесте
    model.eval()
    rng2 = np.random.default_rng(42)
    N2 = len(test_ids) - W - 1
    total_nll = 0.0
    total_tok = 0
    for _ in range(N_EVAL):
        i = int(rng2.integers(0, N2))
        X = torch.tensor([test_ids[i:i + W]], dtype=torch.long, device="cuda")
        Y = torch.tensor([test_ids[i + 1:i + W + 1]], dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = model(X)
            loss = lossf(logits, Y[:, -1])
        total_nll += loss.item() * 1
        total_tok += 1
    ppl = math.exp(total_nll / total_tok)
    print(f"[chat] mixer_ppl={ppl:.3f} params={nparam:,} time={time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(HERE, "model_chat.pt"))
    with open(os.path.join(HERE, "results_chat.json"), "w") as f:
        json.dump({"mixer_ppl": ppl, "params": nparam, "time": time.time()-t0}, f)
    print("saved model_chat.pt + results_chat.json", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=D_MODEL)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--data", default="ru_chat.json", help="файл датасета в phase01/exp_vq/")
    args = ap.parse_args()
    D_MODEL, LAYERS, STEPS, BATCH = args.d, args.layers, args.steps, args.batch
    DATA_FILE = args.data
    main()