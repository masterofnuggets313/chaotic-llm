"""fold_unfold_ppl.py — решающий тест генеративного fold/unfold: PPL end-to-end.

Идея (Fracode): ключи e — это Instructor, хаотическая карта модели — это Library,
состояние h — развёрнутый фрактал. h = G(e) (доказано в fold_unfold.py: cos=1.000000,
rel.L2=6.9e-08). Значит состояние НЕ НАДО ХРАНИТЬ: unfold = обычный форвард.

Три столбца на ОДНИХ И ТОМ ЖЕ held-out окнах (парное сравнение):
  exact     : e(F32) + h(F32)                     = 1536 Б/ток   1.0x
  keys-T5   : коды ключей + h в FP32              = 792 Б/ток    1.94x   (режим T5 `keys`)
  gen (NEW) : ТОЛЬКО коды ключей, h разворачиваем = 24 Б/ток    64x      (генеративный unfold)

Протокол (как в T5, «по Герману»): held-out хвост корпуса, 75% — калибровка кодбука,
25% — eval; модель не видела ни того, ни другого. Метрика — парная деградация PPL.

Запуск: cd phase01/exp_vq && py -3.13 fold_unfold_ppl.py --Ws 65536,262144 --nwin 16,8
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

RESULTS = os.path.join(REPO, "results")
OUT_JSON = os.path.join(RESULTS, "fold_unfold_ppl.json")
FP32 = 4


def forward_generative(model, x, key_fm, key_codes, Mcand, rerank=False):
    """ГЕНЕРАТИЧЕСКИЙ ФОРВАРД: храним только коды ключей, h разворачиваем картой.

    Отличие от T5-режима `keys`: там h_0 брался из ТОЧНЫХ ключей и хранился в FP32,
    здесь h_0 = decode(codes) — то есть ключи в памяти нет вообще, ни одного FP32-вектора.
    """
    W = x.shape[0]; d = model.d
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    key_fm.codes = key_codes
    e_comp = key_fm.decode_codes(key_codes)          # (W, d) — ключи, восстановленные из кодов
    q0 = e_comp[W - nq:].mean(0, keepdim=True)       # (1, d)
    q = q0
    h = e_comp                                       # h_0 = e (развёртка начинается с ключей)
    for li, blk in enumerate(model.blocks):
        run_val, run_next, _ = key_fm.select(q.squeeze(0), topk, W, Mcand, rerank=rerank)
        ckey = key_fm.decode_codes(key_codes[run_next])
        w = torch.softmax(run_val / TEMP, 0)
        driver = (w.unsqueeze(-1) * ckey).sum(0).view(1, d)     # (1, d)
        h = blk(h, driver, k_eff)
        h_last = h[-1].unsqueeze(0)
        q = q0 + model.query_proj(h_last) * 0.5
    g = h.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ws", default="65536,262144")
    ap.add_argument("--nwin", default="16,8")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--m-cand", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--corpus", default=os.path.join(PHASE, "corpus5m_train.txt"))
    ap.add_argument("--corpus-head", default=os.path.join(PHASE, "corpus_train.txt"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--cb-ns", type=int, default=0,
                    help="сколько строк калибровочных ключей брать для обучения кодбука; "
                         "0 = все. T5 брал подвыборку ~65536 (см. _sample_chunks).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed чекпоинта STS-Prog (файл sts_prog_seed{N}.pt); для второго чекпоинта =1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    # если задан --seed, подменяем путь к чекпоинту
    if args.seed is not None:
        args.ckpt = os.path.join(REPO, "results", "ckpts", f"sts_prog_seed{args.seed}.pt")

    toks, n_head, V = load_bpe_ids(args.corpus_head, args.corpus, dev)
    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2, sync_steps=8,
                           driver_mode="sts_prog", alpha=0.3, temp=TEMP)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    d = model.d; L = len(model.blocks); Ppos = model.pos.shape[1]
    print(f"STS-Prog: d={d} L={L} topk={model.topk} nquery={model.nquery} pos_len={Ppos}", flush=True)

    ho_start = n_head; ho_len = len(toks) - n_head
    seg_cb = (ho_start, ho_start + int(0.75 * ho_len))
    seg_ev = (ho_start + int(0.75 * ho_len), len(toks))
    print(f"Held-out {ho_len:,} токенов: кодбук {seg_cb[0]:,}..{seg_cb[1]:,} | "
          f"eval {seg_ev[0]:,}..{seg_ev[1]:,}", flush=True)

    cfgs = [("FR-8x", 2, 48), ("FR-16x", 2, 24), ("FR-32x", 2, 12)]
    all_out = {"script": "fold_unfold_ppl.py",
               "design": ("exact(e+h,F32) | keys-T5(коды+h F32) | gen(только коды, h разворачивается); "
                          "held-out кодбук 75/25; парная деградация PPL"),
               "chunk": args.chunk, "K": args.K, "m_cand": args.m_cand, "runs": []}

    for W, N in zip([int(w) for w in args.Ws.split(",")],
                    [int(n) for n in args.nwin.split(",")]):
        mode = "trained" if W <= Ppos else "cyclic"
        ev_start, ev_end = seg_ev
        if ev_end - ev_start < N + W + 1:
            print(f"W={W}: eval-сегмент мал — пропуск"); continue
        cb_start, cb_end = seg_cb
        if cb_end - cb_start < 2 * W + 4:
            print(f"W={W}: кодбук-сегмент мал — пропуск"); continue
        Mcand = min(args.m_cand, max(1, W - TAIL))
        print("\n" + "=" * 104, flush=True)
        print(f"W={W:,}  окон={N}  pos_mode={mode}", flush=True)
        print("=" * 104, flush=True)
        print(f"{'режим':<22}{'Б/ток':>9}{'сжатие':>9}{'PPL':>10}{'ΔPPL':>9}{'сек':>8}", flush=True)

        # ---- точная база ----
        t0 = time.time(); le = []
        with torch.no_grad():
            for i in range(N):
                o = ev_start + i
                lg = forward_general(model, toks[o:o + W], mode, chunk=args.chunk)
                le.append(F.cross_entropy(lg, toks[o + W].view(1)).item())
        torch.cuda.synchronize(); dt_ex = time.time() - t0
        loss_ex = sum(le) / len(le); ppl_ex = math.exp(min(20.0, loss_ex))
        print(f"{'exact (только h FP32)':<22}{d*FP32:>9.0f}{'1.0x':>9}{ppl_ex:>10.3f}"
              f"{'0.0%':>9}{dt_ex:>8.1f}", flush=True)
        import statistics as _st
        def _ci(losses):
            n = len(losses); m = sum(losses) / n
            if n < 2: return 0.0
            se = (sum((x - m) ** 2 for x in losses) / (n - 1)) ** 0.5 / n ** 0.5
            return 1.96 * se  # 95% CI половина ширины (в единицах loss)
        run = {"W": W, "nwin": N, "pos_mode": mode, "ppl_exact": ppl_ex,
               "loss_exact": loss_ex, "exact_sec": dt_ex,
               "loss_exact_per_window": le, "exact_ci_loss": _ci(le), "configs": []}

        for cname, Lv, S in cfgs:
            if d % S: continue
            bpp = Lv * S                      # байт/ток на коды ключей
            fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
            with torch.no_grad():
                cb_keys = _keys_all(model, toks[cb_start:cb_start + W], mode, args.chunk)
                if args.cb_ns and cb_keys.shape[0] > args.cb_ns:
                    step = cb_keys.shape[0] // args.cb_ns
                    cb_keys = cb_keys[::step]
                print(f"  кодбук: обучаем на {cb_keys.shape[0]:,} строках ключей", flush=True)
                fm.fit(cb_keys, iters=args.iters, seed=0)
            del cb_keys; torch.cuda.empty_cache()

            variants = []
            # (1) режим T5 `keys`: коды + h в FP32
            t0 = time.time(); lk = []
            with torch.no_grad():
                for i in range(N):
                    o = ev_start + i
                    kc = fm.encode_rows(_keys_all(model, toks[o:o + W], mode, args.chunk))
                    lg = forward_general(model, toks[o:o + W], mode, chunk=args.chunk,
                                         key_codes=kc, key_fm=fm, Mcand=Mcand, rerank=False)
                    lk.append(F.cross_entropy(lg, toks[o + W].view(1)).item())
            torch.cuda.synchronize(); dt_k = time.time() - t0
            variants.append(("keys-T5 (коды+h F32)", bpp + d * FP32, lk, dt_k))

            # (2) НОВОЕ: только коды, h разворачивается картой
            t0 = time.time(); lg_list = []
            with torch.no_grad():
                for i in range(N):
                    o = ev_start + i
                    kc = fm.encode_rows(_keys_all(model, toks[o:o + W], mode, args.chunk))
                    lg_list.append(F.cross_entropy(
                        forward_generative(model, toks[o:o + W], fm, kc, Mcand, rerank=False),
                        toks[o + W].view(1)).item())
            torch.cuda.synchronize(); dt_g = time.time() - t0
            variants.append(("gen (только коды, h=0Б)", bpp, lg_list, dt_g))

            for vname, stored, losses, dt in variants:
                lc = sum(losses) / len(losses); ppl = math.exp(min(20.0, lc))
                dp = (ppl / ppl_ex - 1.0) * 100.0
                # 95% CI на ΔPPL: дельты по окнам, затем 1.96*SE
                pair_d = [(math.exp(min(20.0, l)) / math.exp(min(20.0, le[j])) - 1.0) * 100.0
                          for j, l in enumerate(losses)]
                ci_dp = _ci(pair_d)
                print(f"{cname+' '+vname:<22}{stored:>9.0f}{(d*FP32)/stored:>8.1f}x"
                      f"{ppl:>10.3f}{dp:>8.1f}%{dt:>8.1f}", flush=True)
                run["configs"].append({"cfg": cname, "variant": vname,
                                       "stored_bytes_per_tok": stored,
                                       "ratio_x": round((d * FP32) / stored, 2),
                                       "ppl": ppl, "loss": round(lc, 4),
                                       "delta_ppl_pct": round(dp, 2),
                                       "delta_ppl_ci95_pct": round(ci_dp, 2),
                                       "loss_per_window": [round(x, 4) for x in losses],
                                       "sec": round(dt, 1)})
            del fm; torch.cuda.empty_cache()
        all_out["runs"].append(run)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=1)
    print(f"\nСохранено: {args.out}", flush=True)


if __name__ == "__main__":
    main()
