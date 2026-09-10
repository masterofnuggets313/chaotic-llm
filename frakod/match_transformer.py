"""match_transformer.py — честный матч: TransformerLM vs sts_prog (наш лучший).

Тот же корпус (corpus_train), W=256, 6000 шагов, eval тот же:
  - PPL (mixer = PPL трансформера)
  - gated PPL (order-3 prior, для справки)
  - retrieval honest induction (L=16/64/128/256)
Параметры трансформера подобраны ~900K (как sts_prog d192_l8).

Автономный: НЕ импортирует experiment.py (у того тяжёлый top-level код),
копирует только load_chars / make_bpe.
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, torch.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
W = 256
LR = 5e-4
WARMUP = 1000
BATCH = 64
STEPS = 6000
N_EVAL = 5000
VOCAB = 512
MAX_TRAIN = 990_000
MAX_TEST = 2_400_000


def load_chars(path, limit=None):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    if limit:
        s = s[:limit]
    return s


def make_bpe(text, vs=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=vs, special_tokens=["<|endoftext|>"])
    tok.train_from_iterator([text], trainer=tr)
    return tok


def build_data():
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    test_text = load_chars(os.path.join(PHASE, "corpus_test.txt"))
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = np.array(tok.encode(train_text).ids, dtype=np.int32)
    test_ids = np.array(tok.encode(test_text).ids, dtype=np.int32)
    return tok, V, train_ids, test_ids


from parametric_models import TransformerLM, count_params


def tf_params_analytic(V, W, D, layers=4, heads=4):
    embed = V * D
    pos = W * D
    attn = 4 * D * D + 4 * D
    ln = 4 * D
    ffn = 8 * D * D + 5 * D
    per_block = attn + ln + ffn
    head = D * V
    return embed + pos + layers * per_block + head


def pick_tf_dims(budget, V, W, layers=4, heads=4):
    lo, hi = 16, 1024
    while lo < hi:
        mid = (lo + hi) // 2
        mid = max(heads, (mid // heads) * heads)
        mid = max(mid, lo)  # гарантируем прогресс: mid >= lo
        if mid >= hi:
            mid = hi
        if tf_params_analytic(V, W, mid, layers, heads) < budget:
            lo = mid + 1
        else:
            hi = mid
    best = max(heads, (lo // heads) * heads)
    best_cnt = tf_params_analytic(V, W, best, layers, heads)
    for cand_step in (-heads, heads * 2):
        cand = max(heads, best + cand_step)
        if abs(tf_params_analytic(V, W, cand, layers, heads) - budget) < abs(best_cnt - budget):
            best = cand
            best_cnt = tf_params_analytic(V, W, best, layers, heads)
    return best


def build_order3(ids, ORDER=3):
    from collections import defaultdict
    prior = defaultdict(dict)
    for i in range(ORDER - 1, len(ids)):
        ctx = tuple(int(t) for t in ids[i - ORDER + 1:i])
        prior[ctx][int(ids[i])] = prior[ctx].get(int(ids[i]), 0) + 1
    return prior


def gated_ppl(lpm, targets, prior, ctx_tokens):
    N = len(lpm)
    nll = np.zeros(N)
    for k in range(N):
        pm = lpm[k, targets[k]]
        ctx = ctx_tokens[k]
        table = prior.get(ctx)
        if table:
            tot = sum(table.values())
            c = table.get(int(targets[k]), 0)
            if c > 0:
                beta = tot / (tot + 1.0)
                nll[k] = -np.logaddexp(np.log1p(-beta) + pm, np.log(beta) + np.log(c / tot))
                continue
        nll[k] = -pm
    return float(np.exp(np.mean(nll)))


def induction_retrieval(model, test_ids, distances=(16, 64, 128, 256), n_trials=200):
    model.eval()
    rng = np.random.default_rng(0)
    res = {}
    for L in distances:
        hits, miss = 0, 0
        for _ in range(n_trials):
            found = False
            for _try in range(200):
                i = int(rng.integers(L + 2, len(test_ids) - L - 2))
                A = int(test_ids[i]); B = int(test_ids[i + 1])
                j = i + L
                if j < len(test_ids) and test_ids[j - 1] == A:
                    found = True
                    break
            if not found:
                continue
            window = test_ids[j - W:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits = model(X)
            pred = int(logits[0].argmax().item())
            if pred == B:
                hits += 1
            miss += 1
        res[L] = hits / max(1, miss)
    return res


def main():
    train_transformer(budget=900_000, layers=8, heads=4)


def train_transformer(budget=900_000, layers=8, heads=4):
    """Матч трансформера, ПРОТОКОЛ ВЫРОВНЕН под sts_prog:
    LR 5e-4 + warmup 1000 + grad clip 1.0 + N_EVAL 5000 + та же глубина (8 слоёв)."""
    tok, V, train_ids, test_ids = build_data()
    n = len(train_ids) - W - 1
    D = pick_tf_dims(budget, V, W, layers=layers, heads=heads)
    model = TransformerLM(V, W, D=D, HEADS=heads, LAYERS=layers).to("cuda")
    nparam = count_params(model)
    print(f"Transformer: D={D} L={layers} H={heads} params={nparam:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # warmup как в experiment_pc
        if step < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = LR * step / WARMUP
        else:
            for pg in opt.param_groups:
                pg["lr"] = LR
        s = rng.integers(0, n, size=BATCH)
        X = np.stack([train_ids[i:i + W] for i in s])
        Y = np.array([train_ids[i + W] for i in s])
        Xt = torch.tensor(X, dtype=torch.long, device="cuda")
        Yt = torch.tensor(Y, dtype=torch.long, device="cuda")
        opt.zero_grad()
        logits = model(Xt)
        loss = lossf(logits, Yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0:
            print(f"  [{step}/{STEPS}] loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0

    # eval: N_EVAL = 5000 (как в experiment_pc)
    model.eval()
    rng2 = np.random.default_rng(42)
    te_starts = np.sort(rng2.choice(len(test_ids) - W - 1, size=N_EVAL, replace=False))
    logits_te = np.zeros((N_EVAL, V), dtype=np.float32)
    y_te = np.zeros(N_EVAL, dtype=np.int64)
    ctx_tokens = []
    with torch.no_grad():
        for k, s in enumerate(te_starts):
            Xt = torch.tensor([test_ids[s:s + W]], dtype=torch.long, device="cuda")
            logits_te[k] = model(Xt)[0].cpu().numpy()
            y_te[k] = test_ids[s + W]
            ctx_tokens.append(tuple(test_ids[s + W - 2:s + W]))
    lpm = torch.log_softmax(torch.tensor(logits_te), -1).numpy()
    mixer_ppl = float(np.exp(np.mean([-lpm[k, y_te[k]] for k in range(N_EVAL)])))
    prior = build_order3(train_ids)
    gated = gated_ppl(lpm, y_te, prior, ctx_tokens)
    retrieval = induction_retrieval(model, test_ids)
    print(f"[tf] mixer={mixer_ppl:.3f} gated={gated:.3f} retrieval={retrieval} params={nparam:,} time={dt:.0f}s", flush=True)
    res = {"kind": "transformer", "D": D, "L": layers, "H": heads, "params": nparam,
           "mixer_ppl": round(mixer_ppl, 3), "gated_ppl": round(gated, 3),
           "retrieval": {str(k): round(v, 3) for k, v in retrieval.items()},
           "time_s": round(dt, 1)}
    with open(os.path.join(HERE, "results_transformer.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved results_transformer.json", flush=True)


if __name__ == "__main__":
    main()