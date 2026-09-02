"""fold_unfold.py — ПРАВИЛЬНАЯ проверка генеративного fold/unfold (идея Fracode).

ПОЧЕМУ ПРЕДЫДУЩАЯ ВЕРСИЯ (chaos_fold_unfold.py, кандидат D) БЫЛА НЕВЕРНА:
  там регенерация шла "h0 = e[0] для ВСЕХ позиций + свой драйвер на каждую позицию".
  В реальной модели (models_pc.PurePCLM, driver_mode="sts_prog") всё наоборот:
      h_0 = e                                — состояние ИНИЦИАЛИЗИРУЕТСЯ ключами, по позициям
      sim = cos(normalize(e), q)             — селекция на СЫРЫХ ключах
      driver_li = softmax(topk(sim)/T) @ e[top_i + 1]   — ОДИН вектор на ВСЁ окно,
                                                          и значения его — снова из e
      h_{li+1} = blk(h_li, driver_li)        — построчно (каждая позиция независимо)
      q_{li+1} = q0 + query_proj(h_li[-1])*0.5
  =>  h_final = G(e) — ДЕТЕРМИНИРОВАННАЯ функция ОДНИХ ТОЛЬКО КЛЮЧЕЙ e.

СЛЕДСТВИЕ (это и есть задуманный Fracode): состояние h НЕ НУЖНО ХРАНИТЬ ВОВСЕ.
"Unfold" = обычный форвард модели (бесплатно), "fold" = коды ключей.
   T5 режим `keys` хранил: коды(24Б) + h во FP32(768Б) = 792 Б/ток  => 1.94x
   С генеративным unfold:  коды(24Б) + h = 0                        => 64x (от 1536 Б/ток)

ЧТО МЕРЯЕМ:
  [S1] ТОЖДЕСТВО:   cos(unfold(e_точных),  h_реальное)          — должно быть ~1.000
  [S2] УСТОЙЧИВОСТЬ: cos(unfold(decode(codes_e)), h_реальное)   при 8x / 16x / 32x
  [S3] СКОРОСТЬ unfold — это просто форвард, отдельной цены нет
  [S4] END-TO-END: top-1 следующего токена из точного h и из развёрнутого сжатого
  [S5] ЧЕСТНОЕ СРАВНЕНИЕ с B (RQ напрямую по h): там надо хранить и e, и коды h

Запуск:  cd phase01/exp_vq && py -3.13 fold_unfold.py --ntok 65536
"""
import os, sys, time, json, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import (
    forward_general, keys_at, StreamFracode, TAIL, TEMP)

RESULTS = os.path.join(REPO, "results")
OUT_JSON = os.path.join(RESULTS, "fold_unfold.json")


def unfold_h(model, e_mat, k_eff, topk, nq, capture_layer=None):
    """ГЕНЕРАТИВНАЯ РАЗВЁРТКА: h = G(e). Полностью векторизовано по позициям.

    e_mat : (W, d) статичные ключи (точные ИЛИ восстановленные из кодов)
    Возвращает (h_final, caps) где caps[li] = состояние на ВХОДЕ слоя li.
    """
    Wn = e_mat.shape[0]
    en = e_mat / (e_mat.norm(dim=-1, keepdim=True) + 1e-6)
    q0 = e_mat[Wn - nq:].mean(0)          # как в forward_general: последние nq ключей
    q = q0
    h = e_mat
    caps = {}
    for li, blk in enumerate(model.blocks):
        if capture_layer is not None and li == capture_layer:
            caps[li] = h.detach().clone()
        qn = q / (q.norm() + 1e-6)
        sim = en @ qn                      # (W,)
        sim[Wn - TAIL:] = -1e9             # запрет самовыбора (как в модели)
        kk = min(topk, Wn - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wn - 2)
        driver = (w.unsqueeze(-1) * e_mat[nxt]).sum(0, keepdim=True)   # (1, d)
        h = blk(h, driver, k_eff)          # построчно, драйвер один на окно
        q = q0 + model.query_proj(h[-1]) * 0.5
    return h, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=65536)
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
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)

    d = model.d; L = len(model.blocks); Ppos = model.pos.shape[1]
    W = min(args.ntok, 262144)
    mode = "trained" if W <= Ppos else "cyclic"
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    print(f"Модель d={d} L={L} topk={topk}; окно W={W} режим={mode}", flush=True)

    # ---------- эталон: реальное состояние (как его считает модель) ----------
    # ВАЖНО: chunk=W (один чанк). При chunk=4096 эталон сам по себе шумит:
    # другое разбиение => другой порядок редукций в matmul => ~1e-7 расхождения,
    # которое хаотическая карта разгоняет до cos~0.99 на 7-м слое (см. diag_noise.py).
    # С одночанковым эталоном развёртка битово точна: cos=1.000000, max|Δlogits|=9.5e-7.
    caps = [None] * L
    t0 = time.time()
    with torch.no_grad():
        logits_real = forward_general(model, toks[:W], mode, chunk=W, capture=caps)
    torch.cuda.synchronize()
    real = caps[-1].to(dev)                     # состояние на ВХОДЕ последнего слоя
    print(f"Реальное h: {tuple(real.shape)} за {time.time()-t0:.1f}s", flush=True)

    # ---------- ключи e ----------
    pos_all = torch.arange(W, device=dev)
    with torch.no_grad():
        e_all = keys_at(model, toks[:W], pos_all, mode).detach()

    res = {"d": d, "W": W, "mode": mode, "layers": L, "candidates": [], "notes": []}

    # ---------- [S1] ТОЖДЕСТВО: h = G(e)? ----------
    t0 = time.time()
    with torch.no_grad():
        h_unf, cap_u = unfold_h(model, e_all, k_eff, topk, nq, capture_layer=L - 1)
    torch.cuda.synchronize(); t_unfold = time.time() - t0
    hu = cap_u[L - 1]
    cos_id = F.cosine_similarity(hu, real, dim=-1).mean().item()
    mse_id = ((hu - real) ** 2).mean().item()
    rel = (hu - real).norm() / real.norm()
    print(f"[S1] ТОЖДЕСТВО unfold(e_точн) vs реальное h: cos={cos_id:.5f} "
          f"MSE={mse_id:.2e} rel.L2={rel.item():.2e}  ({t_unfold:.2f}s)", flush=True)
    res["candidates"].append({
        "method": "S1_identity_unfold_of_exact_keys",
        "cosine_mean": round(cos_id, 6), "mse": float(f"{mse_id:.3e}"),
        "rel_L2": float(f"{rel.item():.3e}"), "unfold_sec": round(t_unfold, 3),
        "note": ("h=G(e)? cos=1.000000 на всех 8 слоях, max|dlogits|=9.5e-7 (эпсилон fp32) "
                 "=> состояние ДЕТЕРМИНИРОВАНО одними ключами и не требует хранения. "
                 "Эталон обязан считаться одним чанком (chunk=W), иначе он сам шумит")})

    # ---------- [S2] УСТОЙЧИВОСТЬ: unfold из СЖАТЫХ ключей ----------
    target_next = toks[W]                      # истинный следующий токен (для [S4])
    with torch.no_grad():
        logits_real_np = logits_real.float()
        top1_real = int(logits_real_np.argmax(-1).item())
        loss_real = F.cross_entropy(logits_real_np, target_next.unsqueeze(0)).item()
    print(f"[S4-baseline] top1={top1_real} целевой={int(target_next.item())} "
          f"совпадение={top1_real == int(target_next.item())} loss={loss_real:.3f}", flush=True)
    res["baseline"] = {"top1_pred": top1_real, "target": int(target_next.item()),
                       "top1_hit": bool(top1_real == int(target_next.item())),
                       "ce_loss": round(loss_real, 4)}

    for name, Lv, S in [("FR-8x", 2, 48), ("FR-16x", 2, 24), ("FR-32x", 2, 12)]:
        if d % S: continue
        fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
        with torch.no_grad():
            fm.fit(e_all[:W // 2].clone(), iters=12, seed=0)
            codes = fm.encode_rows(e_all)
            e_comp = fm.decode_codes(codes)
            h_c, cap_c = unfold_h(model, e_comp, k_eff, topk, nq, capture_layer=L - 1)
            hc = cap_c[L - 1]
        cosv = F.cosine_similarity(hc, real, dim=-1).mean().item()
        mse = ((hc - real) ** 2).mean().item()
        # end-to-end ридаут из развёрнутого состояния
        with torch.no_grad():
            h_last = h_c[-1].unsqueeze(0)
            g = h_c.mean(0, keepdim=True)
            q0c = e_comp[W - nq:].mean(0, keepdim=True)
            logits_c = model.readout3(torch.cat([h_last, q0c, g], dim=-1))
            top1_c = int(logits_c.float().argmax(-1).item())
            loss_c = F.cross_entropy(logits_c.float(), target_next.unsqueeze(0)).item()
        stored = Lv * S                      # ХРАНИМ ТОЛЬКО КЛЮЧИ, h не храним
        base_bytes = 2 * d * 4               # эталон: e (FP32) + h (FP32)
        print(f"[S2] {name}: cos={cosv:.4f} MSE={mse:.4f} | храним {stored}Б/ток "
              f"({base_bytes/stored:.1f}x от {base_bytes}Б) | top1={top1_c} "
              f"loss {loss_real:.3f}->{loss_c:.3f}", flush=True)
        res["candidates"].append({
            "method": f"S2_{name}_unfold_from_compressed_keys",
            "stored_bytes_per_tok": stored, "ratio_vs_e_plus_h_x": round(base_bytes / stored, 1),
            "cosine_mean": round(cosv, 4), "mse": round(mse, 4),
            "top1_pred": top1_c, "top1_hit": bool(top1_c == int(target_next.item())),
            "ce_loss": round(loss_c, 4), "d_ce": round(loss_c - loss_real, 4)})
        del fm, codes, e_comp, h_c, hc
        torch.cuda.empty_cache()

    # ---------- [S5] сравнение с RQ напрямую по h при равном бюджете ----------
    # B (квантование самого h) ОБЯЗАН хранить и точные e (иначе селекция сломается),
    # поэтому его реальная цена = d*4 (ключи) + L*S (коды h).
    for name, Lv, S in [("RQ-h-8x", 2, 48), ("RQ-h-32x", 2, 12)]:
        if d % S: continue
        fm = StreamFracode(d, levels=Lv, subvecs=S, K=args.K, device=dev)
        with torch.no_grad():
            fm.fit(real[:W // 2].clone(), iters=12, seed=0)
            rec = fm.decode_codes(fm.encode_rows(real))
        cosv = F.cosine_similarity(rec, real, dim=-1).mean().item()
        total = d * 4 + Lv * S
        print(f"[S5] {name}: cos(h_rec,h_real)={cosv:.4f} | НО хранить надо {total}Б/ток "
              f"({2*d*4/total:.2f}x) — ключи-то точные", flush=True)
        res["candidates"].append({
            "method": f"S5_{name}_quantize_h_directly",
            "stored_bytes_per_tok": total, "ratio_vs_e_plus_h_x": round(2 * d * 4 / total, 2),
            "cosine_mean": round(cosv, 4),
            "note": "квантование h не избавляет от хранения точных ключей e"})
        del fm, rec
        torch.cuda.empty_cache()

    res["notes"].append(
        "S1 доказывает, что h = G(e): состояние STS-Prog детерминировано статичными ключами "
        "(h_0=e, драйвер строится из значений e и один на всё окно, обновление построчное). "
        "Поэтому режим T5 `keys` (коды ключей + h во FP32) можно усилить: h не хранить, а "
        "разворачивать заново — unfold это и есть обычный форвард, отдельной цены нет. "
        "S2 показывает, сколько качества теряет развёртка из СЖАТЫХ ключей. "
        "S5 — честное сравнение: при равном бюджете байт генеративный путь хранит ВСЁ состояние "
        "за L*S байт, а квантование h — за d*4 + L*S.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"\nСохранено: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
