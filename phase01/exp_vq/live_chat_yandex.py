"""live_chat_yandex.py — проверка новой русской модели (обучена на ru_chat_yandex.json).

Воспроизводит токенизатор ТОЧНО как train_chat.py: тот же датасет, тот же формат,
первые 10M символов. Затем генерирует ответы на вопросы.
"""
import os, sys, time, json
import torch
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)
from models_pc import build_pc_model

W = 256
DATA_FILE = "ru_chat_yandex.json"
CKPT = os.path.join(HERE, "model_chat.pt")


def make_bpe(text, vocab=2048):
    from tokenizers import decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab,
                                  special_tokens=["<|endoftext|>", "<user>", "<bot>"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok


def build_dataset(rows):
    texts = []
    for r in rows:
        instr = r["instruction"]
        if r.get("input"):
            instr = f"{instr}\n{r['input']}"
        texts.append(f"<user>: {instr}\n<bot>: {r['output']}\n<|endoftext|>")
    return texts


def generate(model, tok, prompt, steps=100, temp=0.8, top_k=50):
    device = next(model.parameters()).device
    ids = tok.encode(prompt).ids[:W - 1]
    if not ids:
        ids = [0]
    for _ in range(steps):
        if len(ids) < W:
            x = torch.tensor([[0] * (W - len(ids)) + ids], dtype=torch.long, device=device)
        else:
            x = torch.tensor([ids[-W:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        logits = logits[0, :] / temp
        if top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < vals[-1]] = -1e9
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        if next_id == tok.token_to_id("<|endoftext|>"):
            break
    return ids


def main():
    print(f"Loading {DATA_FILE}...", flush=True)
    with open(os.path.join(HERE, DATA_FILE), encoding="utf-8") as f:
        rows = json.load(f)
    texts = build_dataset(rows)
    full_text = "\n".join(texts)
    print(f"total text: {len(full_text):,} chars", flush=True)

    print("Training BPE (первые 10M символов, как train_chat.py)...", flush=True)
    tok = make_bpe(full_text[:10_000_000])
    V = tok.get_vocab_size()
    print(f"V={V}", flush=True)

    model = build_pc_model("pc", V, d=384, layers=12, driver_mode="sts_prog",
                           k_init=1.2, sync_steps=8, alpha=0.3).to("cuda")
    model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=False), strict=False)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    questions = [
        "Привет! Как дела?",
        "Что такое гравитация?",
        "Напиши короткое стихотворение про осень.",
        "Как приготовить борщ?",
    ]
    for q in questions:
        print(f"\n>>> {q}", flush=True)
        t0 = time.time()
        ids = generate(model, tok, q, steps=100, temp=0.8, top_k=50)
        text = tok.decode(ids)
        dt = time.time() - t0
        print(f"<<< {text}", flush=True)
        print(f"    [{len(ids)} токенов за {dt:.1f}s = {len(ids)/max(dt,0.01):.1f} tok/s]", flush=True)


if __name__ == "__main__":
    main()