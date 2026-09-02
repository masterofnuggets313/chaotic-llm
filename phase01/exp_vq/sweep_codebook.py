"""sweep_codebook.py — ПОЧЕМУ 8x ХУЖЕ 32x? Свип структуры кодбука.

Наблюдение: сжатие НЕ монотонно. 8x (L=2,S=48 => 96 Б/ток) систематически ХУЖЕ
32x (L=2,S=12 => 24 Б/ток), хотя байт больше и ошибка квантования вектора меньше.

ГИПОТЕЗА (ADC rank noise):
  Оценка сходимости в ADC — это СУММА L*S частичных скалярных произведений.
  8x:  подвектор dim=4,  слагаемых L*S = 96
  32x: подвектор dim=16, слагаемых L*S = 24
  Ошибка суммы растёт как sqrt(L*S) * ошибка одного слагаемого, поэтому при прочих равных
  МЕНЬШЕ подвекторов => НАДЁЖНЕЕ РАНЖИРОВАНИЕ, даже если сам вектор восстановлен грубее.
  Косвенное подтверждение: в T5 rerank (точный dot-product по развёрнутым кандидатам)
  лечил именно ранг и на W=262144 давал -8.6% против +8.4% без него.

ПРОВЕРКА: свип (L, S, K) при фиксированном бюджете 24 Б/ток = 192 бит/ток.
  (1,24,256) dim 8   (2,12,256) dim 16  (3,8,256) dim 24
  (4,6,256)  dim 32  (6,4,256)  dim 48  (8,3,256) dim 64
Если гипотеза верна, качество должно монотонно улучшаться с ростом dim (убыванием S).
И два варианта на каждом: gen (ADC) и gen+rr (rerank). Если rerank УБИРАЕТ
немонотонность — гипотеза подтверждена.

Плюс свип K при фиксированной структуре (2,12): K = 64 / 128 / 256 / 512.
Плюс исходная 8x-конфигурация (2,48,256) как точка отсчёта.

Запуск: cd phase01/exp_vq && py -3.13 sweep_codebook.py --Ws 65536 --nwin 32
"""
import os, sys, math, time, json, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import (
    forward_general, load_bpe_ids, StreamFracode, _keys_all, TAIL, TEMP)
from fold_unfold_ppl import forward_generative

RESULTS = os.path.join(REPO, "results")
FP32 = 4

# (L, S, K) — все делители d=192
CFGS = [
    (2, 48, 256),   # исходная 8x  : dim 4,  96 Б/ток  (точка отсчёта)
    (1, 24, 256),   # dim 8,  24 Б/ток
    (2, 12, 256),   # dim 16, 24 Б/ток
    (3, 8, 256),    # dim 24, 24 Б/ток
    (4, 6, 256),    # dim 32, 24 Б/ток
    (6, 4, 256),    # dim 48, 24 Б/ток
    (8, 3, 256),    # dim 64, 24 Б/ток
    (2, 12, 64),    # K-свип
    (2, 12, 128),
    (2, 12, 512),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ws", default="65536")
    ap.add_argument("--nwin", default="32")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--m-cand", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--cb-tokens", type=int, default=0,
                    help="сколько токенов калибровочного сегмента отдать под кодбук; 0 = весь сегмент W")
    ap.add_argument("--corpus", default=os.path.join(PHASE, "corpus5m_train.txt"))
    ap.add_argument("--corpus-head", default=os.path.join(PHASE, "corpus_train.txt"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--cfgs", default="",
                    help="переопределить сетку: 'L-S-K,L-S-K,...', напр. '1-12-256,2-12-256,4-12-256'")
    ap.add_argument("--out", default=os.path.join(RESULTS, "sweep_codebook.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    cfgs = CFGS
    if args.cfgs:
        cfgs = [tuple(int(x) for x in c.split("-")) for c in args.cfgs.split(",")]

    toks, n_head, V = load_bpe_ids(args.corpus_head, args.corpus, dev)
    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                           driver_mode="sts_prog", alpha=0.3, temp=TEMP)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    d = model.d; Ppos = model.pos.shape[1]
    print(f"STS-Prog: d={d} L={len(model.blocks)} topk={model.topk}", flush=True)

    ho_start = n_head; ho_len = len(toks) - n_head
    cb_start = ho_start; cb_end = ho_start + int(0.75 * ho_len)
    ev_start = cb_end

    out = {"script": "sweep_codebook.py",
           "hypothesis": ("ADC rank noise растёт как sqrt(L*S): при фиксированном бюджете байт "
                          "меньше подвекторов (больше dim) => надёжнее ранг. rerank должен снять эффект."),
           "m_cand": args.m_cand, "iters": args.iters, "runs": []}

    for W, N in zip([int(w) for w in args.Ws.split(",")],
                    [int(n) for n in args.nwin.split(",")]):
        mode = "trained" if W <= Ppos else "cyclic"
        if len(toks) - ev_start < N + W + 1:
            print(f"W={W}: eval-сегмент мал — пропуск"); continue
        Mcand = min(args.m_cand, max(1, W - TAIL))
        print("\n" + "=" * 100, flush=True)
        print(f"W={W:,}  окон={N}  pos_mode={mode}", flush=True)
        print("=" * 100, flush=True)
        print(f"{'L':>2}{'S':>4}{'K':>6}{'dim':>5}{'Б/ток':>8}{'слаг.':>7}"
              f"{'PPL gen':>11}{'Δ%':>8}{'PPL gen+rr':>12}{'Δ%':>8}{'сек':>7}", flush=True)

        t0 = time.time(); le = []
        with torch.no_grad():
            for i in range(N):
                o = ev_start + i
                lg = forward_general(model, toks[o:o + W], mode, chunk=args.chunk)
                le.append(F.cross_entropy(lg, toks[o + W].view(1)).item())
        torch.cuda.synchronize(); dt_ex = time.time() - t0
        loss_ex = sum(le) / len(le); ppl_ex = math.exp(min(20.0, loss_ex))
        print(f"{'exact':<25}{d*FP32:>8.0f}{'':>7}{ppl_ex:>11.3f}{'0.0':>8}", flush=True)
        run = {"W": W, "nwin": N, "pos_mode": mode, "ppl_exact": ppl_ex,
               "loss_exact": loss_ex, "exact_sec": dt_ex, "configs": []}

        n_cb = args.cb_tokens if args.cb_tokens else W
        n_cb = min(n_cb, len(toks) - cb_start - 1)
        cb_keys = _keys_all(model, toks[cb_start:cb_start + n_cb], mode, args.chunk)
        print(f"  калибровка кодбука: {cb_keys.shape[0]:,} строк "
              f"(запрошено {n_cb:,})", flush=True)

        for Lv, S, K in cfgs:
            if d % S: continue
            fm = StreamFracode(d, levels=Lv, subvecs=S, K=K, device=dev)
            with torch.no_grad():
                fm.fit(cb_keys.clone(), iters=args.iters, seed=0)
            dim = d // S
            bpp = fm.bytes_per_pos
            nterms = Lv * S
            row = {"L": Lv, "S": S, "K": K, "subvec_dim": dim,
                   "bytes_per_tok": round(bpp, 2), "adc_terms": nterms}
            t_all = 0.0
            for vname, rr in (("gen", False), ("gen+rr", True)):
                t0 = time.time(); ls = []
                with torch.no_grad():
                    for i in range(N):
                        o = ev_start + i
                        kc = fm.encode_rows(_keys_all(model, toks[o:o + W], mode, args.chunk))
                        lg = forward_generative(model, toks[o:o + W], fm, kc, Mcand, rerank=rr)
                        ls.append(F.cross_entropy(lg, toks[o + W].view(1)).item())
                torch.cuda.synchronize(); dt = time.time() - t0; t_all += dt
                lc = sum(ls) / len(ls); ppl = math.exp(min(20.0, lc))
                dp = (ppl / ppl_ex - 1.0) * 100.0
                row[f"ppl_{vname}"] = round(ppl, 3)
                row[f"d_pct_{vname}"] = round(dp, 2)
                row[f"sec_{vname}"] = round(dt, 1)
            print(f"{Lv:>2}{S:>4}{K:>6}{dim:>5}{bpp:>8.1f}{nterms:>7}"
                  f"{row['ppl_gen']:>11.3f}{row['d_pct_gen']:>8.1f}"
                  f"{row['ppl_gen+rr']:>12.3f}{row['d_pct_gen+rr']:>8.1f}{t_all:>7.1f}", flush=True)
            run["configs"].append(row)
            del fm; torch.cuda.empty_cache()
        del cb_keys; torch.cuda.empty_cache()
        out["runs"].append(run)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nСохранено: {args.out}", flush=True)


if __name__ == "__main__":
    main()
