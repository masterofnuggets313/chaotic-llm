"""chat_sts_prog.py — инференс STS-Prog: генерация текста.

Использование:
  python chat_sts_prog.py --prompt "def hello():" --steps 100 --temp 0.8
  python chat_sts_prog.py --prompt "Привет! Как дела?" --ckpt model_chat.pt --v 2048 --chat

Модель: STS-Prog d=384 l=12. По умолчанию — Stack (V=512), с --v 2048 — русский чат.
"""
import os, sys, json, argparse
import torch, torch.nn as nn
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
sys.path.insert(0, HERE)
from models_pc import build_pc_model, W

def make_bpe(text, vocab=512):
    from tokenizers import decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok

def load_model(ckpt_path=None, d=256, layers=12, device="cuda", vocab=512, chat=False):
    """Загружает модель и токенизатор. chat=True → русский чат (V=2048, ru_chat.json)."""
    if chat:
        print("Loading chat tokenizer (ru_chat.json, V=2048)...", flush=True)
        with open(os.path.join(HERE, "ru_chat.json"), encoding="utf-8") as f:
            rows = json.load(f)
        texts = []
        for r in rows[:2000]:
            texts.append(f"<user>: {r['instruction']}\n<bot>: {r['output']}\n<|endoftext|>")
        tok = make_bpe("\n".join(texts), vocab=2048)
        V = tok.get_vocab_size()
        model = build_pc_model("pc", V, d=d, layers=layers, driver_mode="sts_prog",
                               k_init=1.2, sync_steps=8, alpha=0.3)
    else:
        print("Loading corpus for tokenizer...", flush=True)
        text = open(os.path.join(PHASE, "corpus_stack_train.txt"), "rb").read(10_000_000)
        text = text.decode("utf-8", errors="ignore")
        tok = make_bpe(text, vocab=vocab)
        V = tok.get_vocab_size()
        model = build_pc_model("pc", V, d=d, layers=layers, driver_mode="sts_prog",
                               k_init=1.2, sync_steps=8, alpha=0.3)
    print(f"V={V}", flush=True)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)
    model.to(device).eval()
    return model, tok

def generate(model, tok, prompt, steps=200, temp=0.9, top_k=50):
    device = next(model.parameters()).device
    ids = tok.encode(prompt).ids[:W-1]
    if not ids:
        ids = [0]
    for _ in range(steps):
        # паддинг до W слева нулями (модель требует полное окно W)
        if len(ids) < W:
            x = torch.tensor([[0] * (W - len(ids)) + ids], dtype=torch.long, device=device)
        else:
            x = torch.tensor([ids[-W:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)          # (B, V) — уже для последней позиции
        logits = logits[0, :] / temp
        if top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < vals[-1]] = -1e9
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        yield next_id
        if next_id == tok.token_to_id("<|endoftext|>"):
            return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def hello():")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "night_ckpt_10000.pt"))
    ap.add_argument("--v", type=int, default=512, dest="vocab")
    ap.add_argument("--chat", action="store_true", help="русский чат (V=2048, ru_chat.json)")
    args = ap.parse_args()

    model, tok = load_model(args.ckpt, d=384, layers=12, vocab=args.vocab, chat=args.chat)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    print(f"Prompt: {args.prompt}")
    print("---", flush=True)
    gen_ids = []
    for tid in generate(model, tok, args.prompt, args.steps, args.temp, args.top_k):
        gen_ids.append(tid)
        print(tok.decode(gen_ids), end="\r", flush=True)
    print("\n---", flush=True)
    print("FINAL:", tok.decode(gen_ids), flush=True)