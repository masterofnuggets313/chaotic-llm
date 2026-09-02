"""night_task4_scaling_ladder.py — Task 4: лестница параметров на реальном ЯЗЫКЕ.

Масштабируем STS-Prog и параметр-матченный Transformer по лестнице
1M / 5M / 20M / 100M на TinyStories (EN, естественный язык) и на каждом
размере меряем 4 вещи:

  1. КАЧЕСТВО ЯЗЫКА — PPL на тесте + флаг "развала" (NaN/inf loss или PPL>1e3).
     ВАЖНО: на маленьком корпусе PPL больших моделей = запоминание, поэтому
     имеет смысл только ОТНОШЕНИЕ STS-Prog / Transformer, а не абсолют.
  2. VRAM — пиковая память forward-прохода на тренировочном окне (W=256),
     и тренд роста по контексту (см. п.4).
  3. СКОРОСТЬ — tok/s в обучении + tok/s в context-sweep.
  4. КОНТЕКСТ 128K → 1M → 10M+ — context-sweep на свежих моделях того же
     размера: Wc ∈ {256, 1024, 4096, 16384, 65536, 262144}, один forward B=1,
     пик VRAM + время. OOM ловим и ставим флаг; дальше не идём. По измеренным
     точкам экстраполируем VRAM на 128K/1M/10M (STS ~ O(N), TF ~ O(N^2)) —
     это и есть честный ответ про "контекст 10M без катастрофы".

Архитектурная оговорка (честно): позиции self.pos = (1, W, d) — O(W) параметров.
При W=256 это ничтожно (~0.5MB даже при d=2000). Настоящий лимит контекста на
текущей архитектуре — активации (STS O(N) на слой, TF O(N^2) на слой) и память
pos-параметра при экстремальных W. Истинный контекст 10M требует вынесения
позиций в производную кодировку (будущая инженерия), а не просто измерения.
Здесь мы ИЗМЕРЯЕМ тренд до упора железа (12GB) и экстраполируем O(N) vs O(N^2).

Resumable: пишет инкрементальный JSON (results/night_task4.json). Каждая
(размер, модель) пишется отдельно и при перезапуске пропускается.
--smoke — быстрая проверка «разводки» на синтетике, без скачивания корпуса.

Запуск (реальный, ночью):
  cd phase01\\exp_vq && py -3.13 night_task4_scaling_ladder.py --corpus corpus_nl/tinystories.txt
"""
import os
import sys
import json
import time
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")          # phase01
sys.path.insert(0, HERE)
sys.path.insert(0, PHASE)

from models_pc import build_pc_model          # STS-Prog (Pecora–Carroll driver)
from parametric_models import TransformerLM, count_params

W_TRAIN = 256
LR = 5e-4
WARMUP = 1000
VOCAB = 512
HEADS = 4

# Лестница параметров + протокол на размер
TARGETS = [1_000_000, 5_000_000, 20_000_000, 100_000_000]
CFG = {
    1_000_000:   dict(L=4,  steps=6000, batch=64, heads=4),
    5_000_000:   dict(L=8,  steps=4000, batch=64, heads=4),
    20_000_000:  dict(L=12, steps=2000, batch=32, heads=4),
    100_000_000: dict(L=16, steps=1000, batch=16, heads=4),
}
# Context-sweep: до упора железа (RTX 3060 12GB); OOM — стоп.
WC_VALS = [256, 1024, 4096, 16384, 65536, 262144]
# Точки для честной экстраполяции
WC_PROJECT = [128_000, 1_000_000, 10_000_000]


# ===================== BPE / корпус =====================
def make_bpe(text, vocab=VOCAB):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|endoftext|>"])
    tok.train_from_iterator([text], trainer=tr)
    tok.enable_padding(length=None)
    return tok


def load_corpus(path, max_chars, smoke):
    if smoke:
        words = ["the", "cat", "sat", "on", "the", "mat", "and", "a", "dog", "ran",
                 "to", "the", "red", "box", "she", "opened", "it", "with", "a", "key"]
        unit = " ".join(words) + " "
        rep = (max_chars // len(unit)) + 1
        return (unit * rep)[:max_chars]
    if path and os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    raise FileNotFoundError(f"corpus not found: {path} (запусти подготовку или укажи --corpus)")


# ===================== Подбор размеров =====================
def make_sts(V, d, L):
    return build_pc_model("pc", vocab=V, d=d, layers=L, k_init=1.2,
                          sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3)


def solve_sts_d(target, V, W, L):
    lo, hi = 16, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        c = count_params(make_sts(V, mid, L))
        if c < target:
            lo = mid + 1
        else:
            hi = mid
    cands = [lo, lo - 1, lo + 1]
    best = min(cands, key=lambda d: abs(count_params(make_sts(V, d, L)) - target))
    return best, count_params(make_sts(V, best, L))


def solve_tf_D(target, V, W, L, heads):
    def cnt(D):
        D = max(heads, (D // heads) * heads)
        return count_params(TransformerLM(V, W, D, heads, L))
    lo, hi = heads, 2048
    while lo < hi:
        mid = (lo + hi) // 2
        if cnt(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    cands = [lo, lo - heads, lo + heads]
    best = min(cands, key=lambda d: abs(cnt(d) - target))
    best = max(heads, (best // heads) * heads)
    return best, cnt(best)


# ===================== Обучение =====================
def train_one(model, train_ids, V, W, steps, batch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss()
    n = len(train_ids) - W - 1
    rng = np.random.default_rng(seed)
    t0 = time.time()
    last_loss = None
    collapsed = False
    nan_seen = 0
    for step in range(1, steps + 1):
        lr_scale = min(1.0, step / WARMUP)
        for pg in opt.param_groups:
            pg["lr"] = LR * lr_scale
        s = rng.integers(0, n, size=batch)
        X = torch.tensor(np.stack([train_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
        Y = torch.tensor([train_ids[i + W] for i in s], dtype=torch.long, device="cuda")
        with torch.amp.autocast("cuda"):
            logits = model(X)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = lossf(logits, Y)
        if not torch.isfinite(loss):
            nan_seen += 1
            if nan_seen >= 3:
                collapsed = True
                break
            opt.zero_grad()
            continue
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last_loss = float(loss.item())
        if step % 500 == 0:
            print(f"    [step {step}/{steps}] loss={last_loss:.3f}", flush=True)
    dt = time.time() - t0
    tok_per_s = (steps * batch * W) / dt if dt > 0 else 0.0
    # пик VRAM за обучение
    vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return dict(time_s=round(dt, 1), tok_per_s=round(tok_per_s, 1),
                last_loss=last_loss, collapsed=collapsed,
                train_vram_mb=round(vram_mb, 1), steps=steps, batch=batch)


def eval_ppl(model, test_ids, V, W, batch=64):
    model.eval()
    n = len(test_ids) - W - 1
    rng = np.random.default_rng(42)
    nll = 0.0
    cnt = 0
    with torch.no_grad():
        for _ in range(0, max(1, n), W):
            s = rng.integers(0, n, size=batch)
            X = torch.tensor(np.stack([test_ids[i:i + W] for i in s]), dtype=torch.long, device="cuda")
            Y = torch.tensor([test_ids[i + W] for i in s], dtype=torch.long, device="cuda")
            with torch.amp.autocast("cuda"):
                logits = model(X)
                if isinstance(logits, tuple):
                    logits = logits[0]
            nll += nn.CrossEntropyLoss(reduction="sum")(logits, Y).item()
            cnt += batch
    return float(np.exp(nll / cnt)) if cnt > 0 else float("inf")


# ===================== Context-sweep =====================
def context_sweep(d, D, kind, V, L, heads, Wc_vals):
    res = {}
    for Wc in Wc_vals:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            if kind == "sts":
                m = make_sts(V, d, L)
                m.pos = nn.Parameter(torch.randn(1, Wc, d) * 0.02)
            else:
                m = TransformerLM(V, Wc, D, heads, L)
            m = m.to("cuda")
            torch.cuda.synchronize()
            X = torch.randint(0, V, (1, Wc), device="cuda")
            t0 = time.time()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = m(X)
            torch.cuda.synchronize()
            t = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1024 ** 2
            res[Wc] = {"vram_mb": round(vram, 1),
                       "time_ms": round(t * 1000, 2),
                       "tok_per_s": round(Wc / t, 1) if t > 0 else 0.0}
            del m
            torch.cuda.empty_cache()
        except RuntimeError as e:
            s = str(e).lower()
            # CUDA-OOM ("out of memory") И CPU-аллокатор ("not enough memory"/"alloc")
            # — оба означают, что модель/forward не влезает в память при этом Wc.
            if "out of memory" in s or "not enough memory" in s or "alloc" in s:
                res[Wc] = {"oom": True}
                torch.cuda.empty_cache()
                # монотонный рост памяти — дальше тоже OOM
                return res, Wc
            else:
                raise
    return res, None


def project_vram(sweep, kind):
    """Экстраполяция VRAM на 128K/1M/10M. STS~O(N) (линейно), TF~O(N^2) (квадратично)."""
    pts = sorted([(w, v["vram_mb"]) for w, v in sweep.items()
                  if isinstance(v, dict) and "vram_mb" in v and v["vram_mb"] > 0])
    if len(pts) < 2:
        return {tw: None for tw in WC_PROJECT}, "insufficient_points"
    ws = np.array([p[0] for p in pts], dtype=float)
    vs = np.array([p[1] for p in pts], dtype=float)
    out = {}
    if kind == "sts":
        # линейная аппроксимация по последним двум точкам (через начало координат)
        slope = vs[-1] / ws[-1]
        for tw in WC_PROJECT:
            out[tw] = round(slope * tw, 1)
        return out, "linear_O(N)"
    else:
        # квадратичная: v = a * w^2 (через начало), a = mean(v/w^2)
        a = float(np.mean(vs / (ws ** 2)))
        for tw in WC_PROJECT:
            out[tw] = round(a * tw * tw, 1)
        return out, "quadratic_O(N^2)"


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "night_task4.json"))
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus_nl", "tinystories.txt"))
    ap.add_argument("--max-chars", type=int, default=60_000_000)
    ap.add_argument("--smoke", action="store_true", help="быстрая проверка разводки на синтетике")
    ap.add_argument("--targets", default="")
    a = ap.parse_args()

    device = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("[T4] WARNING: CUDA недоступна, считаем на CPU (медленно)", flush=True)

    if a.smoke:
        targets = [1_000_000]
        wc_vals = [64, 128, 256]
        max_chars = 200_000
        step_scale = 0.01     # ~30-60 шагов
        corpus_path = None
    else:
        targets = [int(t) for t in a.targets.split(",") if t] if a.targets else TARGETS
        wc_vals = WC_VALS
        max_chars = a.max_chars
        step_scale = 1.0
        corpus_path = a.corpus

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {}
    if os.path.exists(a.out):
        try:
            out = json.load(open(a.out, encoding="utf-8"))
        except Exception:
            out = {}

    print(f"[T4] device={device} targets={targets} smoke={a.smoke}", flush=True)
    text = load_corpus(corpus_path, max_chars, a.smoke)
    print(f"[T4] corpus chars={len(text):,}", flush=True)
    tok = make_bpe(text)
    V = tok.get_vocab_size()
    ids = np.array(tok.encode(text).ids, dtype=np.int32)
    idx = max(1, int(len(ids) * 0.9))
    train_ids = ids[:idx]
    test_ids = ids[idx:]
    print(f"[T4] V={V} train={len(train_ids):,} test={len(test_ids):,}", flush=True)

    for target in targets:
        cfg = CFG[target]
        L = cfg["L"]
        heads = cfg["heads"]
        steps = max(20, int(cfg["steps"] * step_scale))
        batch = cfg["batch"]
        print(f"\n[T4] === size {target:,}  L={L} steps={steps} batch={batch} ===", flush=True)

        # ---- STS-Prog ----
        key = f"{target}_sts"
        if key in out and out[key].get("done"):
            print(f"[T4] {key} cached", flush=True)
        else:
            d, p_sts = solve_sts_d(target, V, W_TRAIN, L)
            print(f"[T4] STS d={d} params={p_sts:,} (target {target:,})", flush=True)
            m = make_sts(V, d, L).to(device)
            tr = train_one(m, train_ids, V, W_TRAIN, steps, batch, seed=0)
            ppl = eval_ppl(m, test_ids, V, W_TRAIN)
            sweep, oom_at = context_sweep(d, None, "sts", V, L, heads, wc_vals)
            proj, proj_kind = project_vram(sweep, "sts")
            torch.cuda.empty_cache()
            out[key] = dict(model="STS-Prog", target=target, L=L, d=d, params=p_sts,
                            ppl=round(ppl, 3), collapsed=tr["collapsed"] or (ppl > 1e3),
                            train=tr, context_sweep=sweep, oom_at=oom_at,
                            projected_vram_mb=proj, projection=proj_kind, done=True)
            json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[T4] STS PPL={ppl:.3f} collapsed={out[key]['collapsed']} "
                  f"sweep_max={max([w for w,v in sweep.items() if isinstance(v,dict)], default=0)}", flush=True)

        # ---- Transformer ----
        key = f"{target}_tf"
        if key in out and out[key].get("done"):
            print(f"[T4] {key} cached", flush=True)
        else:
            D, p_tf = solve_tf_D(target, V, W_TRAIN, L, heads)
            print(f"[T4] TF D={D} params={p_tf:,} (target {target:,})", flush=True)
            m = TransformerLM(V, W_TRAIN, D, heads, L).to(device)
            tr = train_one(m, train_ids, V, W_TRAIN, steps, batch, seed=0)
            ppl = eval_ppl(m, test_ids, V, W_TRAIN)
            sweep, oom_at = context_sweep(None, D, "tf", V, L, heads, wc_vals)
            proj, proj_kind = project_vram(sweep, "tf")
            torch.cuda.empty_cache()
            out[key] = dict(model="Transformer", target=target, L=L, D=D, heads=heads, params=p_tf,
                            ppl=round(ppl, 3), collapsed=tr["collapsed"] or (ppl > 1e3),
                            train=tr, context_sweep=sweep, oom_at=oom_at,
                            projected_vram_mb=proj, projection=proj_kind, done=True)
            json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[T4] TF  PPL={ppl:.3f} collapsed={out[key]['collapsed']} "
                  f"sweep_max={max([w for w,v in sweep.items() if isinstance(v,dict)], default=0)}", flush=True)

    # ---- Итоговая сводка ----
    print("\n[T4] ============ СВОДКА ============", flush=True)
    for target in targets:
        s = out.get(f"{target}_sts", {})
        t = out.get(f"{target}_tf", {})
        ratio = (s.get("ppl") / t.get("ppl")) if (s.get("ppl") and t.get("ppl")) else None
        print(f"  {target:>10,} : STS PPL={s.get('ppl')} (p={s.get('params'):,}) | "
              f"TF PPL={t.get('ppl')} (p={t.get('params'):,}) | "
              f"ratio={round(ratio,3) if ratio else 'n/a'} | "
              f"STS tok/s={s.get('train',{}).get('tok_per_s')} TF tok/s={t.get('train',{}).get('tok_per_s')}", flush=True)
    print(f"[T4] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
