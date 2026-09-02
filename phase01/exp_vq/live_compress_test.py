"""live_compress_test.py — ЖИВОЙ тест сжатия памяти STS-Prog.

НЕ на формулах, а на реальных данных модели:
  1) генерируем базу СТАТИЧНЫХ ключей e = embed+pos ровно как в forward STS-Prog
     (models_pc.PurePCLM, тот же чекпоинт, что в T5);
  2) сжимаем Fracode (L=2) при 8x/16x/32x, меряем ФАКТИЧЕСКИЙ размер в байтах;
  3) для каждого окна ищем соседей: точно vs сжато (ADC+rerank) -> recall@M, driver_cos;
  4) пересчитываем, сколько токенов влезет в 12 GB по реальному байт/токен.

Запуск:  cd phase01/exp_vq && py -3.13 live_compress_test.py --ntok 2000000
"""
import os, sys, math, time, json, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from fracode_memory_probe import kmeans, assign_codes
from night_task5_fracode_forward import StreamFracode, keys_at, keys_range, cos_sim

RESULTS = os.path.join(REPO, "results")
OUT_JSON = os.path.join(RESULTS, "live_compress_test.json")

FP32 = 4
TAIL = 8
TEMP = 0.3
TOPK = 8            # сколько соседей берёт модель (driver_mode=sts_prog)

def make_key_base(model, toks, s, e, mode, Ppos, dev):
    """База ключей e для [s,e) — ТОЧНО как в forward (static keys). (N,d)."""
    pos = torch.arange(s, e, device=dev)
    emb = model.embed(toks[s:e])
    if mode == "trained":
        return (emb + model.pos[0, pos]).detach()
    return (emb + model.pos[0, pos % Ppos]).detach()

def topk_exact(keys, q, W, topk):
    sim = cos_sim(keys, q.unsqueeze(0)).squeeze(1)      # (W,)
    sim[W-TAIL:] = -1e9
    vals, loc = sim.topk(topk)
    return vals, loc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=2_000_000, help="размер базы ключей в токенах")
    ap.add_argument("--corpus", default=os.path.join(PHASE, "corpus5m_train.txt"))
    ap.add_argument("--corpus-head", default=os.path.join(PHASE, "corpus_train.txt"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--W", type=int, default=262144, help="окно запроса (рабочая точка T5)")
    ap.add_argument("--nq", type=int, default=2000, help="сколько окон-запросов промерить")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    # ---- токены (BPE, как при обучении); берём ХВОСТ, который модель не видела ----
    import numpy as np, final_benchmark as fb
    head = fb.load_chars(args.corpus_head, 990_000)
    tok = fb.make_bpe(head)
    V = tok.get_vocab_size(); n_head = len(tok.encode(head).ids)
    big = fb.load_chars(args.corpus, None)
    ids = np.array(tok.encode(big).ids, dtype=np.int64)
    toks_full = torch.tensor(ids, dtype=torch.long, device=dev)
    # хвост held-out
    toks = toks_full[n_head:]
    print(f"База токенов: {len(toks_full):,} всего; берём held-out хвост {len(toks):,}", flush=True)
    if args.ntok > len(toks) - args.W - 1:
        args.ntok = len(toks) - args.W - 1
        print(f"  (урезаем базу до {args.ntok:,} — столько помещается в held-out хвост)", flush=True)

    # ---- модель ----
    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    sd = torch.load(args.ckpt, map_location="cpu"); model.load_state_dict(sd)
    for p in model.parameters(): p.requires_grad_(False)
    d = model.d; Ppos = model.pos.shape[1]
    mode = "trained" if args.W <= Ppos else "cyclic"
    print(f"Модель: d={d} L={len(model.blocks)} topk={TOPK} pos_len={Ppos} windows={args.W:,} ({mode})", flush=True)

    # ---- 1) СТРОИМ БАЗУ КЛЮЧЕЙ e на NTOK токенах ----
    # база должна быть >= W (окно) + запас на калибровку (50%) и окна-запросы (после 60%)
    minN = args.W + 200_000
    if args.ntok < minN:
        args.ntok = minN
        print(f"  (база увеличена до {args.ntok:,} — нужно >= окно {args.W:,} + запас)", flush=True)
    if args.ntok > len(toks) - 1:
        args.ntok = len(toks) - 1
    N = args.ntok
    t0 = time.time()
    base = make_key_base(model, toks, 0, N, mode, Ppos, dev)
    torch.cuda.synchronize()
    print(f"База ключей: {base.shape} за {time.time()-t0:.1f}s; "
          f"FP32-размер = {base.numel()*FP32/1024**3:.2f} GB", flush=True)

    # ---- 2) СЖИМАЕМ Fracode при 8x/16x/32x, меряем ФАКТ байты ----
    # бюджет байт/поз: FR L=2 => bytes = 2*S ; ratio = d*4 / (2*S)
    # K=256 умещается в 1 байт (uint8) — это и есть деплойный формат.
    plans = [("FR-8x",2,48),("FR-16x",2,24),("FR-32x",2,12)]
    calib = base[: int(0.5*N)].clone()     # кодбук учим на первой половине базы
    results = {"N_tokens": N, "d": d, "W": args.W, "mode": mode,
               "fp32_bytes_per_tok": d*FP32, "compression": []}

    for name, Lv, S in plans:
        assert d % S == 0, f"d={d} % S={S} != 0"
        fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
        tf = time.time()
        fm.fit(calib, iters=args.iters, seed=0)
        torch.cuda.synchronize()
        fit_t = time.time() - tf
        codes = fm.encode_rows(base)              # (N, L, S) int коды
        torch.cuda.synchronize()
        # ФАКТИЧЕСКИЙ размер: коды как uint8 (K=256 < 256 -> 1 байт/код) +
        # кодбуки (L*S*K*sub, FP32). Кодбук крошечный (единицы МБ) — им пренебрегаем в ёмкости.
        codes_bytes = codes.numel() * 1            # uint8 = 1 байт на код
        cbook_bytes = sum(c.numel()*FP32 for lvl in fm.cbooks for c in lvl)
        total_bytes = codes_bytes + cbook_bytes
        bytes_per_tok = total_bytes / N
        ratio = (d*FP32) / bytes_per_tok
        # ёмкость 12 GB (оставляем 1 GB на активации/модель)
        cap_11gb = int((11*1024**3) / bytes_per_tok)
        print(f"\n[{name}] S={S} fit={fit_t:.1f}s  коды={codes_bytes/1024**2:.1f}MB "
              f"кодбук={cbook_bytes/1024**2:.1f}MB | {bytes_per_tok:.2f} байт/ток "
              f"=> {ratio:.1f}x | в 11GB ~ {cap_11gb:,} токенов", flush=True)
        results["compression"].append({
            "name": name, "levels": Lv, "subvecs": S, "ratio": round(ratio,2),
            "bytes_per_tok": round(bytes_per_tok,3),
            "codes_mb": round(codes_bytes/1024**2,1), "codebook_mb": round(cbook_bytes/1024**2,1),
            "capacity_11gb_tokens": cap_11gb})

        # ---- 3) ВОССТАНОВЛЕНИЕ: recall@M + driver_cos на окнах-запросах ----
        # Семантика STS-Prog: память для предсказания позиции o+W — это окно o:o+W.
        # Ищем соседей ВНУТРИ окна и точно, и в сжатии (честно).
        # окна-запросы: из диапазона [0.6N, N-W) — внутри базы, вне калибровки
        idx0 = int(0.6*N)
        max_q = N - args.W - 1
        q0 = min(args.nq, max(1, max_q - idx0))
        if q0 < 1:
            print("  !! недостаточно места для окон-запросов в базе — пропуск retrieval", flush=True)
            results["compression"][-1].update({"recall_at_1_pct": None, f"recall_at_{TOPK}_pct": None, "driver_cos": None})
            del codes, fm; torch.cuda.empty_cache()
            continue
        recalls = {m:0.0 for m in (1, TOPK)}
        hitrate = 0.0          # доля окон, где ХОТЯ БЫ 1 из top-8 сжатого ∈ top-8 точного
        qcos_exact = []        # cosine(query, выбранный ключ) для точного поиска
        qcos_comp = []         # ... и для сжатого
        dcs = []
        t1 = time.time()
        for i in range(q0):
            o = idx0 + i
            x = toks[o:o+args.W]
            q = keys_at(model, x, torch.arange(args.W-1, args.W, device=dev), mode).mean(0)  # (d,)
            # точный ответ внутри окна
            ke = make_key_base(model, toks, o, o+args.W, mode, Ppos, dev)
            _, exact_loc = topk_exact(ke, q, args.W, TOPK)
            exact_set = set(exact_loc.tolist())
            qn = q/ (q.norm()+1e-6)
            qcos_exact.append((ke[exact_loc] * qn).sum(-1).mean().item())
            # сжатый поиск: коды ТОЛЬКО окна
            cw = codes[o:o+args.W]
            fm.codes = cw
            Mcand = min(2048, args.W - TAIL)
            sc = fm.adc_scores(q)
            sc[args.W-TAIL:] = -1e18
            cand = sc.topk(Mcand).indices
            rec = fm.decode_codes(cw[cand])
            dot = (rec @ q.unsqueeze(0).T).squeeze(1)
            _, cloc = dot.topk(TOPK)
            comp_set = set(cand[cloc].tolist())
            qcos_comp.append((fm.decode_codes(cw[list(comp_set)]) * qn).sum(-1).mean().item())
            # recall@1: попал ли точный top1 в сжатый топ-TOPK
            recalls[1] += (exact_loc[0].item() in comp_set)
            # recall@TOPK: доля точных топ-TOPK, попавших в сжатый топ-TOPK
            recalls[TOPK] += len(exact_set & comp_set) / max(1,len(exact_set))
            # hitrate: хотя бы один точный top-8 среди сжатого top-8
            if len(exact_set & comp_set) > 0:
                hitrate += 1
            # driver_cos: средний ключ топ-соседей точный vs сжатый (внутри окна)
            drv_exact = ke[exact_loc].mean(0)
            drv_comp = (fm.decode_codes(cw[list(comp_set)]) if comp_set else torch.zeros(d,device=dev)).mean(0)
            dcs.append(cos_sim(drv_exact.unsqueeze(0), drv_comp.unsqueeze(0)).item())
        fm.codes = None
        torch.cuda.synchronize()
        r1 = 100*recalls[1]/q0; rk = 100*recalls[TOPK]/q0
        hr = 100*hitrate/q0
        qce = sum(qcos_exact)/len(qcos_exact); qcc = sum(qcos_comp)/len(qcos_comp)
        dc = sum(dcs)/len(dcs)
        r1 = 100*recalls[1]/q0; rk = 100*recalls[TOPK]/q0; dc = sum(dcs)/len(dcs)
        print(f"  recall@1={r1:.1f}%  recall@{TOPK}={rk:.1f}%  hitrate@{TOPK}={hr:.1f}%  "
              f"driver_cos={dc:.3f}  qcos: exact={qce:.3f} comp={qcc:.3f} "
              f"({q0} окон, {time.time()-t1:.1f}s)", flush=True)
        results["compression"][-1].update({
            "recall_at_1_pct": round(r1,1), f"recall_at_{TOPK}_pct": round(rk,1),
            "hitrate_at_TOPK_pct": round(hr,1), "driver_cos": round(dc,3),
            "qcos_exact": round(qce,3), "qcos_comp": round(qcc,3)})
        del codes, fm; torch.cuda.empty_cache()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nСохранено: {OUT_JSON}", flush=True)
    # ---- итог по ёмкости ----
    print("\n=== ЁМКОСТЬ 12 GB (реальные байты/токен) ===")
    for c in results["compression"]:
        print(f"  {c['name']:8s}: {c['bytes_per_tok']:.2f} Б/ток -> {c['capacity_11gb_tokens']:,} токенов "
              f"в 11GB  (~{c['capacity_11gb_tokens']/1e6:.0f} млн)")

if __name__ == "__main__":
    main()
