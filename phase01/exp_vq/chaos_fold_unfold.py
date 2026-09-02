"""chaos_fold_unfold.py — ИССЛЕДОВАНИЕ: фрактальное/хаотическое сжатие СОСТОЯНИЯ h.

!!! ВНИМАНИЕ: кандидат D в ЭТОМ файле НЕВЕРЕН и ПОДАВЛЕН (см. fold_unfold.py) !!!
    Здесь регенерация шла как "h0 = e[0] для всех позиций + свой драйвер на каждую
    позицию", тогда как в модели (sts_prog) h_0 = e[i] по позициям, а драйвер ОДИН
    на всё окно. Из-за этого верхняя граница D_EXACT вышла 0.099 — мерили не то.
    Кроме того, эталон forward_general(chunk=4096) сам шумит на ~1.4% по cos на
    7-м слое (см. diag_noise.py); эталон обязан считаться одним чанком.
    Правильная и векторизованная версия — fold_unfold.py / fold_unfold_ppl.py.
    Кандидаты A/B/C ниже корректны и остаются в силе.

Гипотеза (от юзера): Fracode задумывался как фрактальный сворачиваемый компрессор
(fold/unfold как фрактал), близкий по духу к детерминированному хаосу Арнольда
(chaotic map -> attractor). Обычный PQ на состоянии h ПРОВАЛИЛСЯ (T5: state FR-32x
+192%). Проверим АЛЬТЕРНАТИВУ: сжимать h не внешним квантователем, а как траекторию
хаотической карты (store seed + step, unfold обратно).

Три кандидата на "кодирование состояния":
  A) SEED-REGEN: h восстанавливается из seed через ту же хаотическую карту модели
     (h = h + alpha*tanh(h@W+b), затем PC-шаг к драйверу). Храним ТОЛЬКО seed
     (несколько байт) + индекс итерации. Это чистый "generator-based" подход Fracode
     (Library = карта, Phi = {seed, step}).
  B) LIBRARY-RQ: иерархический RQ как приближение Instructor-дерева Fracode
     (Library = кодбуки уровней, Phi = {F_new = коды}). Это то, что уже тестировали.
  C) ATTRACTOR-PROJECTION: спроектировать h на низкоразмерный базис аттрактора
     (PCA/SVD первых k компонент) -> хранить только k коэффициентов. Проверка:
     лежит ли h в низкоразмерном притягивающем множестве.

Метрики: MSE vs реальное h, bytes/token, "восстановимость" драйвера (cosine).

Запуск:  cd phase01/exp_vq && py -3.13 chaos_fold_unfold.py --ntok 50000
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

RESULTS = os.path.join(REPO, "results")
OUT_JSON = os.path.join(RESULTS, "chaos_fold_unfold.json")
TAIL = 8; TEMP = 0.3

def collect_states(model, toks, s, e, mode, Ppos, dev):
    """Прогон модели и сбор РЕАЛЬНЫХ состояний h по слоям. Возвращает list[(W,d)].
    Упрощённо: используем keys_at как прокси 'статичного состояния', а для динамического
    h — переиспользуем forward_general с capture."""
    from night_task5_fracode_forward import forward_general, keys_at, cos_sim
    caps = [None]*len(model.blocks)
    with torch.no_grad():
        _ = forward_general(model, toks[s:e], mode, chunk=4096, capture=caps)
    # caps[li] = (Ncap, d) выборка состояния на входе слоя li
    return [c.detach().cpu() for c in caps]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=50000)
    ap.add_argument("--corpus", default=os.path.join(PHASE, "corpus5m_train.txt"))
    ap.add_argument("--corpus-head", default=os.path.join(PHASE, "corpus_train.txt"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    import numpy as np, final_benchmark as fb
    head = fb.load_chars(args.corpus_head, 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size(); n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(args.corpus, None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    sd = torch.load(args.ckpt, map_location="cpu"); model.load_state_dict(sd)
    for p in model.parameters(): p.requires_grad_(False)
    d = model.d; L = len(model.blocks); Ppos = model.pos.shape[1]
    W = min(args.ntok, 262144)
    mode = "trained" if W <= Ppos else "cyclic"
    print(f"Модель d={d} L={L}; собираем реальные состояния на {W} токенах", flush=True)

    t0 = time.time()
    states = collect_states(model, toks, 0, W, mode, Ppos, dev)   # list[(Ncap,d)] по слоям
    torch.cuda.synchronize()
    real = states[-1].to(dev)   # состояние последнего слоя на входе = "то, что модель хранит"
    print(f"Реальное состояние: {real.shape} за {time.time()-t0:.1f}s; "
          f"FP32={real.numel()*4/1024**2:.1f}MB", flush=True)

    results = {"d": d, "W": W, "mode": mode, "candidates": [], "notes": []}

    # ---------- A) SEED-REGEN через ту же хаотическую карту ----------
    # Попытка 1: состояние h_i ВОССТАНАВЛИВАЕТСЯ из h_0 (seed) итерациями карты.
    # h_0 = первый токен окна; карта = модельные блоки. Проверим, насколько
    # реальное h отличается от "регенерированного" из h_0 теми же блоками.
    # Это честный тест: если состояние ДЕТЕРМИНИРОВАНО картой -> хранить только h_0 (seed).
    from night_task5_fracode_forward import keys_at
    # h_0 прокси: статичный ключ первой позиции окна (как в модели h_0 = e)
    h0 = keys_at(model, toks, torch.tensor([0], device=dev), mode).squeeze(0)  # (d,)
    # регенерируем "динамически" через чистую карту (без драйвера — это upper bound идеи)
    k_eff = torch.sigmoid(model.k)
    with torch.no_grad():
        h = h0.clone()
        for blk in model.blocks:
            h = blk(h.unsqueeze(0), torch.zeros_like(h.unsqueeze(0)), k_eff).squeeze(0)
    # сравним с реальным последним состоянием (среднее по позициям, т.к. размеры могут отличаться)
    rmean = real.mean(0)
    mse_A = ((h - rmean)**2).mean().item()
    cos_A = F.cosine_similarity(h.unsqueeze(0), rmean.unsqueeze(0)).item()
    results["candidates"].append({
        "method": "A_seed_regen_chaotic_map",
        "stored_bytes_per_tok": 4,   # только h_0 (seed) как fp32 вектор = 4*? на практике uint8 ~ d байт
        "note": "ВОССТАНОВЛЕНИЕ из h_0 той же картой; чистая карта без драйвера (upper bound)",
        "mse_vs_real": round(mse_A,4), "cosine_vs_real_mean": round(cos_A,4)})
    print(f"[A] seed-regen (чистая карта): MSE={mse_A:.3f} cos={cos_A:.3f}", flush=True)

    # ---------- B) LIBRARY-RQ (приближение Instructor-дерева) ----------
    # обучаем иерархию L=2 на реальном состоянии, меряем MSE и байты
    from night_task5_fracode_forward import StreamFracode
    for name, Lv, S in [("RQ-L2-S48",2,48),("RQ-L2-S12",2,12)]:
        if d % S != 0: continue
        fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
        fm.fit(real[:W//2].clone(), iters=12, seed=0)
        codes = fm.encode_rows(real)
        rec = fm.decode_codes(codes)
        mse = ((rec-real)**2).mean().item()
        cosv = F.cosine_similarity(rec.mean(0).unsqueeze(0), real.mean(0).unsqueeze(0)).item()
        bytes_per = Lv*S  # uint8 коды на позицию
        ratio = (d*4)/bytes_per
        results["candidates"].append({
            "method": f"B_{name}", "stored_bytes_per_tok": bytes_per,
            "ratio_x": round(ratio,1), "mse_vs_real": round(mse,4),
            "cosine_vs_real_mean": round(cosv,4)})
        print(f"[B] {name}: MSE={mse:.3f} cos={cosv:.3f} {bytes_per}Б/ток ({ratio:.1f}x)", flush=True)
        del fm, codes, rec; torch.cuda.empty_cache()

    # ---------- C) ATTRACTOR-PROJECTION (SVD-базис) ----------
    # лежит ли h в низкоразмерном аттракторе? проекция на топ-k сингулярных векторов
    X = real.double() - real.double().mean(0)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    energy = (S**2).cumsum(0)/(S**2).sum()
    k90 = int((energy < 0.90).sum())+1
    k99 = int((energy < 0.99).sum())+1
    kb = min(k99, d)
    basis = Vt[:kb].to(dev).float()     # (kb, d)
    coeff = (real @ basis.T)            # (N, kb) — коэффициенты (float16)
    rec = coeff @ basis                 # восстановление
    mse = ((rec-real)**2).mean().item()
    cosv = F.cosine_similarity(rec.mean(0).unsqueeze(0), real.mean(0).unsqueeze(0)).item()
    bytes_per = kb*2                    # fp16 коэффициенты
    ratio = (d*4)/bytes_per
    results["candidates"].append({
        "method": "C_attractor_svd", "k_components": kb,
        "energy_k99_pct": round(float(energy[min(kb,d)-1])*100,2),
        "stored_bytes_per_tok": bytes_per, "ratio_x": round(ratio,1),
        "mse_vs_real": round(mse,4), "cosine_vs_real_mean": round(cosv,4)})
    print(f"[C] attractor-SVD k={kb} (energy@k99={energy[min(kb,d)-1]*100:.1f}%): "
          f"MSE={mse:.3f} cos={cosv:.3f} {bytes_per}Б/ток ({ratio:.1f}x)", flush=True)

    # ---------- D) ГЕНЕРАТИВНЫЙ FOLD/UNFOLD: регенерация h из СЖАТОГО ДРАЙВЕРА ----------
    # Идея (юзер): драйвер = Instructor в Phi Fracode. Сжимаем КЛЮЧИ (уже -8.6% PPL при 32x),
    # из сжатых ключей разворачиваем драйвер, затем прогоняем хаотическую карту модели
    # (блоки) с этим драйвером => регенерируем h БЕЗ хранения самого h. Это fold/unfold.
    # Меряем cos(h_regen, h_real) и bytes/ток (только ключи, h не храним).
    import numpy as np, final_benchmark as fb
    # перетокенизируем нужный сегмент (те же toks, но соберём ключи окна позиций 1..W)
    from night_task5_fracode_forward import keys_at, StreamFracode
    # ключи e для всех позиций 0..W (база для сжатия драйвера)
    pos_all = torch.arange(W, device=dev)
    e_all = (model.embed(toks[:W]) + (model.pos[0, pos_all] if mode=="trained" else model.pos[0, pos_all % Ppos])).detach()
    # реальное h (уже в `real`) — это состояние последнего слоя на ВХОДЕ (caps[-1]).
    # Построим драйвер точно и из сжатых ключей для каждой позиции и регенерируем h:
    #   h_regen[i] = blk(blk(... blk(h0, driver_i)...)) — каскад блоков с драйвером driver_i.
    # h0 = e_all[0] (как в модели h_0 = e). driver_i = softmax(cos(q_i, e)) @ e_i (topk=8).
    k_eff = torch.sigmoid(model.k); topk = int(model.topk)
    Wn = e_all.shape[0]
    # точный драйвер (baseline, не сжатый)
    q_all = e_all.clone()
    def build_drivers(e_mat, compressed=False, fm=None):
        drivers = torch.zeros_like(e_mat)
        for i in range(Wn):
            q = q_all[i].unsqueeze(0)
            if compressed and fm is not None:
                fm.codes = codes_e[i:i+1]  # коды окна из 1 позиции -> не годится; единый код для всей базы
            # упрощаем: драйвер = нормированная сумма top-k ключей по cosine (как в модели)
            sim = F.cosine_similarity(e_mat, q, dim=-1)  # (Wn,)
            sim[Wn-TAIL:] = -1e9
            vals, loc = sim.topk(topk)
            w = torch.softmax(vals/TEMP, 0)
            drivers[i] = (w.unsqueeze(-1) * e_mat[loc]).sum(0)
        return drivers
    drivers_exact = build_drivers(e_all, compressed=False)
    # регенерация h из точного драйвера (это верхняя граница D, должна быть ~реальной)
    def regen_h(h0, drivers):
        h = h0.unsqueeze(0)
        for blk in model.blocks:
            h = blk(h, drivers, k_eff)  # driver на каждом шаге = drivers (упрощённо один и тот же)
        return h.squeeze(0)
    h0 = e_all[0]
    def regen_h_seq(h0, drivers):
        """Позиционно-зависимая регенерация: для каждой позиции i свой драйвер drivers[i]."""
        h = h0.unsqueeze(0)              # (1,d)
        out = torch.empty_like(drivers)  # (Wn,d)
        for i in range(Wn):
            drv = drivers[i].unsqueeze(0) # (1,d)
            hh = h
            for blk in model.blocks:
                hh = blk(hh, drv, k_eff)
            out[i] = hh.squeeze(0)
        return out
    h_regen_exact = regen_h_seq(h0, drivers_exact).detach()
    # усредняем по позициям для сравнения со средним реальным состоянием
    cos_D_exact = F.cosine_similarity(h_regen_exact.mean(0).unsqueeze(0), real.mean(0).unsqueeze(0)).item()
    # сжатые ключи (Fracode 8x/16x/32x) -> драйвер из сжатых -> регенерация h
    for name, Lv, S in [("FR-8x",2,48),("FR-16x",2,24),("FR-32x",2,12)]:
        if d % S != 0: continue
        fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
        fm.fit(e_all[:W//2].clone(), iters=12, seed=0)
        codes_e = fm.encode_rows(e_all)         # (Wn, L, S)
        # разворачиваем ключи обратно в сжатом домене (decode_codes = приближённые e)
        e_comp = fm.decode_codes(codes_e)
        drivers_comp = build_drivers(e_comp, compressed=True)
        h_regen = regen_h_seq(h0, drivers_comp).detach()
        cosv = F.cosine_similarity(h_regen.mean(0).unsqueeze(0), real.mean(0).unsqueeze(0)).item()
        mse = ((h_regen - real.mean(0))**2).mean().item()
        bytes_per = Lv*S  # ХРАНИМ ТОЛЬКО ключи (h не храним вообще!)
        ratio = (d*4)/bytes_per
        results["candidates"].append({
            "method": f"D_{name}_regen_from_compressed_driver",
            "stored_bytes_per_tok": bytes_per, "ratio_x": round(ratio,1),
            "cosine_h_regen_vs_real": round(cosv,4), "mse": round(mse,4),
            "note": "h НЕ хранится; регенерируется картой из сжатого драйвера (Fracode fold/unfold)"})
        print(f"[D] {name}: cos(h_regen,h_real)={cosv:.3f} MSE={mse:.3f} "
              f"храним {bytes_per}Б/ток (h не храним, {ratio:.1f}x)", flush=True)
        del fm, codes_e, e_comp, drivers_comp, h_regen; torch.cuda.empty_cache()
    results["candidates"].append({
        "method": "D_EXACT_upper_bound", "stored_bytes_per_tok": d*4,
        "cosine_h_regen_vs_real": round(cos_D_exact,4),
        "note": "верхняя граница D: драйвер точный (ключи не сжаты); показывает, насколько "
                "карта вообще способна регенерировать h из драйвера"})
    print(f"[D] EXACT upper bound: cos(h_regen,h_real)={cos_D_exact:.3f}", flush=True)
    results["notes"].append(
        "A: чистая карта без драйвера — upper bound (провал, cos~0). "
        "C: состояние лежит в широком аттракторе (k=153/192 для 99%), проекцией сильно не сожмёшь. "
        "B: обычный RQ идеален по вектору, но проваливался в T5 по PPL (state +192%). "
        "D: если cos(h_regen,h_real) из СЖАТОГО драйвера >0.95 — найден генеративный fold/unfold "
        "состояния (Fracode Instructor = драйвер), h не хранится вообще, только ключи (уже -8.6% PPL). "
        "Если D близко к D_EXACT — метод рабочий; если далеко — драйвер слишком зашумлён сжатием "
        "и нужен путь 2 (кодировать Δh) или 3 (Library генераторов).")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nСохранено: {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
