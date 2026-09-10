"""final_benchmark.py — единый launcher для честного benchmark STS-Prog vs Transformer.

Протокол: FINAL_BENCHMARK.md
Результаты: ../results/final_benchmark.json и ../results/final_benchmark.md

Запуск:
  cd phase01/exp_vq && python final_benchmark.py
"""
import os, sys, json, time, math, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
EXPERIMENTS = os.path.join(HERE, "..", "..", "experiments")
RESULTS = os.path.join(HERE, "..", "..", "results")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(EXPERIMENTS, exist_ok=True)

# Пути к данным
sys.path.insert(0, PHASE)
from parametric_models import TransformerLM, count_params
from models_pc import build_pc_model

MAX_TRAIN = 990_000
VOCAB = 512
W = 256
D = 192
BLOCKS = 4

def load_chars(path, max_chars=None):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]
    return text

def make_bpe(text, vocab=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"])
    tok.train_from_iterator([text], trainer)
    tok.enable_padding(length=None)
    return tok

# ========== ПРОТОКОЛ ==========
STEPS = 6000
BATCH = 64
LR = 5e-4
WARMUP = 1000
N_EVAL = 5000
SEEDS = [0, 1, 2]
W_VALS = [64, 128, 256, 512]  # для scaling
RETRIEVAL_L = [16, 32, 64, 128, 256]
RETRIEVAL_N = 200  # trials per distance

# ========== ДАННЫЕ ==========
def build_data():
    train_text = load_chars(os.path.join(PHASE, "corpus_train.txt"), MAX_TRAIN)
    tok = make_bpe(train_text)
    V = tok.get_vocab_size()
    train_ids = np.array(tok.encode(train_text).ids, dtype=np.int32)
    return tok, V, train_ids

# ========== ПОРЯДОК-3 (для gated PPL) ==========
def build_order3(train_ids):
    prior = defaultdict(dict)
    for i in range(3, len(train_ids)):
        ctx = tuple(train_ids[i - 2:i])
        w = train_ids[i]
        d = prior[ctx]
        d[w] = d.get(w, 0) + 1
    return {k: dict(v) for k, v in prior.items()}

# ========== ОБУЧЕНИЕ ==========
def train_model(model, train_ids, seed=0, steps=None, batch=None, lr=None, warmup=None, n_eval=None, desc=""):
    global STEPS, BATCH, LR, WARMUP, N_EVAL
    if steps is None: steps = STEPS
    if batch is None: batch = BATCH
    if lr is None: lr = LR
    if warmup is None: warmup = WARMUP
    if n_eval is None: n_eval = N_EVAL
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    best_ppl = float("inf")
    best_step = 0
    t0 = time.time()
    for step in range(1, steps + 1):
        lr_scale = min(1.0, step / warmup) if warmup else 1.0
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_scale
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device="cuda")
        with torch.amp.autocast("cuda"):
            logits = model(X)
            loss = lossf(logits, Y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % n_eval == 0 or step == steps:
            ppl = float(torch.exp(loss).item())
            if ppl < best_ppl:
                best_ppl = ppl
                best_step = step
    elapsed = time.time() - t0
    return {"final_ppl": float(torch.exp(loss).item()),
            "best_ppl": best_ppl, "best_step": best_step, "time": elapsed}

# ========== EVAL PPL ==========
def eval_ppl(model, test_ids, batch=None):
    global BATCH
    if batch is None: batch = BATCH
    model.eval()
    n = len(test_ids) - W - 1
    rng = np.random.default_rng(42)
    nll = 0.0
    cnt = 0
    with torch.no_grad():
        for _ in range(0, n, W):
            s = rng.integers(0, n, size=batch)
            X = torch.tensor(np.stack([test_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
            Y = torch.tensor([test_ids[i + W] for i in s], dtype=torch.long, device="cuda")
            logits = model(X)
            nll += nn.CrossEntropyLoss(reduction="sum")(logits, Y).item()
            cnt += batch
    return np.exp(nll / cnt) if cnt > 0 else float("inf")

# ========== RETRIEVAL ==========
def eval_retrieval(model, test_ids, distances=RETRIEVAL_L, n_trials=RETRIEVAL_N):
    """Правильный индукционный тест: A→B, второй A — последний токен окна, предсказываем B."""
    model.eval()
    W0 = W
    results = {}
    for L in distances:
        hits = 0
        trials = 0
        attempts = 0
        max_attempts = n_trials * 100
        while trials < n_trials and attempts < max_attempts:
            # i должно быть достаточно велико, чтобы j-W0 >= 0 (окно валидно)
            lo = max(L + 2, W0 - L + 2)
            if lo >= len(test_ids) - L - 3:
                break
            i = int(np.random.randint(lo, len(test_ids) - L - 3))
            A = int(test_ids[i])
            B = int(test_ids[i + 1])
            j = i + L
            if j >= len(test_ids):
                attempts += 1
                continue
            if test_ids[j - 1] != A:
                attempts += 1
                continue
            window = test_ids[j - W0:j]
            X = torch.tensor([window], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits = model(X)
            pred = logits.argmax(dim=-1).item()
            if pred == B:
                hits += 1
            trials += 1
            attempts += 1
        results[L] = {"trials": trials, "hits": hits, "accuracy": hits / trials if trials > 0 else 0.0}
    return results

# ========== SCALING ==========
def eval_scaling(build_sts, build_tf, V, W_vals=(64, 128, 256, 512)):
    """Измеряет VRAM/time forward при разных W.
    Для каждого W строится СВЕЖАЯ модель с pos под этот W (случайный init, seed 0),
    чтобы изолировать зависимость от длины контекста."""
    res = {"sts_prog": {}, "transformer": {}}
    for Wc in W_vals:
        torch.cuda.reset_peak_memory_stats()
        m = build_sts(V, Wc)
        m = m.to("cuda")
        torch.cuda.synchronize()
        X = torch.randint(0, V, (1, Wc), device="cuda")
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = m(X)
        torch.cuda.synchronize()
        t = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**2
        res["sts_prog"][Wc] = {"vram_mb": round(vram, 1), "time_ms": round(t * 1000, 2), "tokens_per_sec": round(Wc / t) if t > 0 else 0}
        del m
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        m = build_tf(V, Wc)
        m = m.to("cuda")
        torch.cuda.synchronize()
        X = torch.randint(0, V, (1, Wc), device="cuda")
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _ = m(X)
        torch.cuda.synchronize()
        t = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**2
        res["transformer"][Wc] = {"vram_mb": round(vram, 1), "time_ms": round(t * 1000, 2), "tokens_per_sec": round(Wc / t) if t > 0 else 0}
        del m
        torch.cuda.empty_cache()
    return res

# ========== MAIN ==========
def main():
    print("=" * 60, flush=True)
    print("FINAL BENCHMARK: STS-Prog vs Transformer", flush=True)
    print("=" * 60, flush=True)
    print(f"Commit: {os.popen('git rev-parse HEAD').read().strip()}", flush=True)
    
    tok, V, train_ids = build_data()
    print(f"V={V} train_ids={len(train_ids)}", flush=True)
    prior = build_order3(train_ids[:200000])
    idx = int(len(train_ids) * 0.8)
    train_sub = train_ids[:idx]
    test_sub = train_ids[idx:]

    # Конфигурации моделей
    configs = {
        "STS-Prog": lambda: build_pc_model("pc", V, d=192, layers=8, k_init=1.2, sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3),
        "STS-Prog (no-PC)": lambda: build_pc_model("pc", V, d=192, layers=8, k_init=1.2, sync_steps=8, driver_mode="sts_prog_nopc", alpha=0.3, temp=0.3),
    }
    
    # Подбор D для Transformer под ~900K
    from match_transformer import pick_tf_dims
    D_tf = pick_tf_dims(900_000, V, W, layers=8, heads=4)
    configs[f"Transformer (D={D_tf})"] = lambda: TransformerLM(V, W, D=D_tf, HEADS=4, LAYERS=8)

    all_results = {}
    for name, build_fn in configs.items():
        print(f"\n--- {name} ---", flush=True)
        model_seeds = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", flush=True)
            m = build_fn()
            info = train_model(m, train_sub, seed=seed)
            ppl = eval_ppl(m, test_sub)
            ret = eval_retrieval(m, test_sub)
            params = sum(p.numel() for p in m.parameters())
            entry = {"params": params, "seed": seed, "ppl": round(ppl, 3), "train": info, "retrieval": ret}
            model_seeds.append(entry)
            print(f"    PPL={ppl:.3f} params={params:,} time={info['time']:.1f}s", flush=True)
            if ret:
                print(f"    Retrieval: {dict((k, round(v['accuracy'], 3)) for k, v in ret.items())}", flush=True)
        # Mean/std
        ppls = [s["ppl"] for s in model_seeds]
        stats = {"mean_ppl": round(np.mean(ppls), 3), "std_ppl": round(np.std(ppls), 3), "min_ppl": round(min(ppls), 3), "max_ppl": round(max(ppls), 3)}
        all_results[name] = {"seeds": model_seeds, "stats": stats}

    # Scaling
    print("\n--- Scaling curves ---", flush=True)
    def build_sts_for_w(Vc, Wc):
        torch.manual_seed(0)
        from models_pc import PurePCLM
        m = PurePCLM(vocab=Vc, d=192, layers=8, k_init=1.2, alpha=0.3, sync_steps=8, driver_mode="sts_prog", temp=0.3)
        # позиции под Wc (переопределяем, т.к. pos = W×d)
        m.pos = nn.Parameter(torch.randn(1, Wc, 192) * 0.02)
        return m
    def build_tf_for_w(Vc, Wc):
        torch.manual_seed(0)
        m = TransformerLM(Vc, Wc, D=D_tf, HEADS=4, LAYERS=8)
        return m
    scaling = eval_scaling(build_sts_for_w, build_tf_for_w, V, W_VALS)
    all_results["_scaling"] = scaling
    print(json.dumps(scaling, indent=2), flush=True)

    # Сохранение
    commit = os.popen('git rev-parse HEAD').read().strip()
    all_results["_meta"] = {"commit": commit, "protocol": "FINAL_BENCHMARK.md", "seeds": SEEDS, "steps": STEPS, "batch": BATCH, "lr": LR, "warmup": WARMUP}
    with open(os.path.join(RESULTS, "final_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to results/final_benchmark.json", flush=True)

    # Генерация Markdown
    md = ["# Final Benchmark Results", f"**Commit:** `{commit}`", f"**Protocol:** FINAL_BENCHMARK.md", f"**Seeds:** {SEEDS}", ""]
    md.append("## Parameter-matched benchmark")
    md.append("| Model | Params | PPL |")
    md.append("| --- | --- | --- |")
    for name, data in all_results.items():
        if name.startswith("_"):
            continue
        s = data["stats"]
        md.append(f"| {name} | {data['seeds'][0]['params']:,} | {s['mean_ppl']}±{s['std_ppl']} |")
    md.append("")
    md.append("## Retrieval")
    md.append("| Model | L | Trials | Hits | Accuracy |")
    md.append("| --- | --- | --- | --- | --- |")
    for name, data in all_results.items():
        if name.startswith("_"):
            continue
        for seed_data in data["seeds"]:
            for L, ret in seed_data["retrieval"].items():
                md.append(f"| {name} seed={seed_data['seed']} | L={L} | {ret['trials']} | {ret['hits']} | {ret['accuracy']:.3f} |")
    md.append("")
    md.append("## Scaling")
    md.append("| Model | W | VRAM MB | Time ms | Tok/s |")
    md.append("| --- | --- | --- | --- | --- |")
    for name, data in all_results.items():
        if name.startswith("_") or "scaling" not in data:
            continue
        for Wc, sc in data["scaling"].items():
            md.append(f"| {name} | {Wc} | {sc['vram_mb']} | {sc['time_ms']} | {sc['tokens_per_sec']} |")
    md_text = "\n".join(md)
    with open(os.path.join(RESULTS, "final_benchmark.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved to results/final_benchmark.md", flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()