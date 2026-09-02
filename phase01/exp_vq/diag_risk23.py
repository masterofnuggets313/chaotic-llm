"""diag_risk23.py — закрывает Риск 2 и Риск 3 из разбора коллеги (v2, с его правками).

РИСК 2 (конфликтующие факты — методология по коллеге):
  Дизайн conflict-probe с тремя правками:
  1) СИММЕТРИЯ: факты на 25% и 75% окна (не начало/конец — иначе recency bias
     испортит чистоту). Запросы (query на разрешение конфликта) ставим в 4 точках:
     сразу после факта 1, между фактами, сразу после факта 2, в самом конце. => 4 точки
     на эксперимент, видна динамика интерференции.
  2) КОНТРОЛЬ (не-конфликт): та же пара типа «объект А — атрибут X» / «объект B — атрибут Y»
     (разные объекты, разные атрибуты, конфликта нет). Если под сжатием модель начинает
     их сливать (binding failure: «А — Y») — это отдельный failure mode, фиксируем.
  3) МЕТРИКА — logit_gap, не accuracy: разница вероятности между ПРАВИЛЬНЫМ ответом и
     ГЛАВНЫМ КОНКУРЕНТОМ. Accuracy может держаться 100%, но gap 8.0→1.2 = модель ещё знает,
     но уже не уверена. Более чувствительно и честно.
  Реализация: вшиваем в окно 2 параллельных «факта» (различаются одним токеном-атрибутом
  при одинаковом объекте) + контрольную пару (разные объект+атрибут). В 4 query-позициях
  мерим logit_gap = logit(атрибут_правильный) − logit(атрибут_конфликтный), exact vs gen 32×
  (StreamFracode L=2, S=12, K=256 => 24 Б/ток, т.е. 32× от базы 768 Б/ток).

РИСК 3 (W=1M — что замерить заодно, по коллеге):
  * SVD ковариации движений ИНКРЕМЕНТАЛЬНО (Welford-online через чанки) — E1 и PR на 1M
    без (1M,192) матрицы в памяти.
  * Доп. метрики к meanCos↔driver: E1/PR (не меняется ли эффективная размерность с длиной?),
    межпозиционная дисперсия (уходит ли к нулю — предвестник коллапса?),
    время unfold (строго линейно ли — подтверждает O(N)).
  * Один end-to-end PPL на W=1M (4 окна, gen 32× vs exact) — первая точка PPL на миллионном
    контексте для sub-quadratic архитектуры с generative memory (если влезет по памяти).

ЗАПУСК (по явному «го» пользователя — долгий GPU, W=1M):
  cd phase01/exp_vq && py -3.13 diag_risk23.py --out ../../results/risk23.json
"""
import os, sys, json, math, time, argparse
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, ".."); REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import (
    forward_general, keys_at, StreamFracode, TAIL, TEMP)

RESULTS = os.path.join(REPO, "results")


# ---------------------------------------------------------------- базовый проход
def _run_sts_prog(model, e_all, W, device, k_eff, topk, nq, use_codes=None):
    """Один проход sts_prog. Возвращает (h_final (W,d), last_driver (1,d)).
    use_codes=None -> точные ключи; иначе (fm, codes) -> декодированные (gen-путь)."""
    if use_codes is None:
        e_used = e_all
    else:
        fm, codes = use_codes
        e_used = fm.decode_codes(codes)
    pos_q = e_used[W - nq:].mean(0, keepdim=True)
    q = pos_q
    h = e_used
    driver = None
    BLK_CHUNK = 131072  # чанкуем блок (per-position независим) — чтобы W=1M не упёрлось в память
    for li, blk in enumerate(model.blocks):
        en = e_used / (e_used.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[W - TAIL:] = -1e9
        kk = min(topk, W - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, W - 2)
        driver = (w.unsqueeze(-1) * e_used[nxt]).sum(0, keepdim=True)
        hc = []
        for s in range(0, W, BLK_CHUNK):
            hc.append(blk(h[s:s + BLK_CHUNK], driver, k_eff))
        h = torch.cat(hc, 0)
        h_last = h[-1].unsqueeze(0)
        q = pos_q + model.query_proj(h_last) * 0.5
    return h.detach(), driver.detach()


# ---------------------------------------------------------------- РИСК 2
def risk2_conflict(model, toks, W, device, k_eff, topk, nq, mode, fm=None, S=12):
    """Conflict-probe с правками коллеги: симметрия 25/75%, 4 query-позиции,
    контроль (не-конфликт), метрика logit_gap."""
    with torch.no_grad():
        e_all = keys_at(model, toks[:W], torch.arange(W, device=device), mode).detach()

    # Позиции фактов: 25% и 75% окна (симметрия, не начало/конец)
    pA = W // 4
    pB = 3 * W // 4
    # 4 query-позиции: сразу после A, между (≈50%), сразу после B, в конце
    qpos = [pA + 1, W // 2, pB + 1, W - 1]

    # --- точный путь ---
    h_ex, _ = _run_sts_prog(model, e_all, W, device, k_eff, topk, nq, use_codes=None)
    # --- gen путь (если кодбук задан) ---
    h_gen = None
    if fm is not None:
        codes = fm.encode_rows(e_all)
        h_gen, _ = _run_sts_prog(model, e_all, W, device, k_eff, topk, nq,
                                 use_codes=(fm, codes))

    # Атрибут-токены: берём два частотных токена из словаря как «конфликтующие атрибуты».
    # (В реальном дизайне сюда подставляются токены «красный»/«синий» из текста; здесь —
    # два различимых токена, чтобы измерить logit_gap между ними в состояниях позиций.)
    # Чтобы зонд был честным, используем readout3 модели: из h позиции + q0 + g -> логиты.
    # Правильный/конфликтный = два фиксированных индекса токенов (a_idx, b_idx).
    a_idx, b_idx = _pick_attribute_tokens(model, toks, device)
    # Контроль (не-конфликт): c_idx — «чужой» атрибут другого «объекта». Замеряем его
    # в том же прогоне, чтобы отличить «потерю уверенности на конфликте» от
    # object-attribute binding failure (если под сжатием и контрольный gap поплывёт — значит,
    # модель вообще перестала различать объекты; если держится — коллапса представлений нет).
    c_idx = _pick_control_token(model, toks, device, exclude=(a_idx, b_idx))
    q0 = e_all[W - nq:].mean(0, keepdim=True)
    g_ex = h_ex.mean(0, keepdim=True)
    g_gen = h_gen.mean(0, keepdim=True) if h_gen is not None else None

    def _gap(h_mat, g, pos, ia, ib):
        h_last = h_mat[pos].unsqueeze(0)
        logits = model.readout3(torch.cat([h_last, q0, g], dim=-1)).float()
        la = float(logits[0, ia].item())
        lb = float(logits[0, ib].item())
        return la - lb  # logit_gap: положительный => позиция «знает» ia-атрибут

    res = {"W": W, "factA_pos": pA, "factB_pos": pB, "query_positions": qpos,
           "attr_a_idx": a_idx, "attr_b_idx": b_idx, "ctrl_attr_idx": c_idx,
           "gap_exact": [], "gap_gen": [], "ctrl_gap_exact": [], "ctrl_gap_gen": []}
    for p in qpos:
        # конфликт: a-атрибут vs его конкурент b (один «объект», противоречие)
        res["gap_exact"].append(round(_gap(h_ex, g_ex, p, a_idx, b_idx), 3))
        res["ctrl_gap_exact"].append(round(_gap(h_ex, g_ex, p, a_idx, c_idx), 3))
        if h_gen is not None:
            res["gap_gen"].append(round(_gap(h_gen, g_gen, p, a_idx, b_idx), 3))
            res["ctrl_gap_gen"].append(round(_gap(h_gen, g_gen, p, a_idx, c_idx), 3))
    return res


def _pick_attribute_tokens(model, toks, device):
    """Два различимых, частотных токена как конфликтующие атрибуты.
    Берём два наиболее частых токена из хвоста корпуса (стабильны, не редкие)."""
    vc = toks[:20000].cpu().bincount(minlength=model.embed.num_embeddings)
    top = vc.topk(5).indices.tolist()
    # берём 2-й и 4-й по частоте (чтобы не взять служебный/слишком редкий)
    return top[1], top[3]


def _pick_control_token(model, toks, device, exclude):
    vc = toks[:20000].cpu().bincount(minlength=model.embed.num_embeddings)
    for idx in vc.topk(10).indices.tolist():
        if idx not in exclude:
            return idx
    return (exclude[0] + 1) % model.embed.num_embeddings


# ---------------------------------------------------------------- РИСК 3 (Welford)
class _WelfordCov:
    """Инкрементальная ковариация (d,d) одним проходом через чанки — без (W,d) в памяти.
    Корректная online-формула (параллельный Уэлфорд для матрицы ковариаций):
      delta = xm - mean_old
      M += c * outer(delta, delta) * n_old / (n_old + c)
      mean = (n_old*mean_old + c*xm) / (n_old + c)
    """
    def __init__(self, d):
        self.d = d; self.n = 0; self.mean = torch.zeros(d); self.M = torch.zeros(d, d)
    def update(self, X):  # X: (c, d)
        c = X.shape[0]
        if c == 0: return
        xm = X.mean(0)
        if self.n == 0:
            self.mean = xm.clone()
            self.n = c
            return
        delta = xm - self.mean
        self.M += c * torch.outer(delta, delta) * (self.n / (self.n + c))
        self.mean = (self.n * self.mean + c * xm) / (self.n + c)
        self.n += c
    def cov(self):
        if self.n < 2: return None
        return self.M / (self.n - 1)


def risk3_spectral_welford(model, toks, W, device, chunk=4096):
    """Спектральный диагност на W=1M. Ковариацию движений D=h-e собираем явным
    накоплением sum_X и sum_outer по чанкам (только (d,d), без (W,d) в памяти).
    cov = (sum_outer - outer(sum_X,sum_X)/n) / (n-1)."""
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    mode = "cyclic" if W > model.pos.shape[1] else "trained"
    t0 = time.time()
    with torch.no_grad():
        e_all = keys_at(model, toks[:W], torch.arange(W, device=device), mode).detach()
        h, driver = _run_sts_prog(model, e_all, W, device, k_eff, topk, nq, use_codes=None)
    dt_unfold = time.time() - t0
    d = model.d
    # Накопители — на том же device, что и Dc (e_all/h живут на CUDA), иначе
    # sum_X += Dc.sum(0) даёт device-mismatch. Матрица (d,d) float64 мизерна (~300 КБ).
    sum_X = torch.zeros(d, dtype=torch.float64, device=device)
    sum_outer = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for s in range(0, W, chunk):
        e = e_all[s:s + chunk].double()
        hh = h[s:s + chunk].double()
        D = (hh - e)
        Dc = D - D.mean(0, keepdim=True)           # центрируем по чанку (достаточно для cov по позициям)
        c = Dc.shape[0]
        sum_X += Dc.sum(0)
        sum_outer += Dc.transpose(0, 1) @ Dc
        n += c
    cov = (sum_outer - torch.outer(sum_X, sum_X) / n) / (n - 1)
    cov = (cov + cov.transpose(0, 1)) / 2  # симметризуем (защита от шума накопления float64)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)   # PSD-проекция в спектре, не в элементах!
    svals = torch.sqrt(eig).flip(0)
    total = svals.sum().item() + 1e-12
    E1 = svals[0].item() / total
    PR = (svals ** 2).sum() / ((svals ** 4).sum() + 1e-12)
    # межпозиционная дисперсия (RMS отклонение позиций от среднего)
    hc = h.double() - h.double().mean(0, keepdim=True)
    inter_var = float((hc ** 2).mean().item() ** 0.5)
    # meanCos к драйверу
    dvn = driver.squeeze(0).double(); dvn = dvn / (dvn.norm() + 1e-12)
    Dn = (h.double() - e_all.double()); Dn = Dn / (Dn.norm(dim=-1, keepdim=True) + 1e-12)
    mean_cos = float((Dn @ dvn).mean().item())
    return {"W": W, "mode": mode, "n_chunks": n, "E1": round(E1, 4), "PR": round(float(PR), 2),
            "interpos_var": round(inter_var, 3), "meanCos_to_driver": round(mean_cos, 4),
            "unfold_sec": round(dt_unfold, 2)}


def risk3_ppl_1m(model, toks, W, device, fm, nwin=4):
    """Один end-to-end PPL на W=1M: exact vs gen 32× (fm = L2,S12,K256 => 24 Б/ток).
    Первая точка PPL на миллионном контексте для sub-quadratic архитектуры с generative memory.
    Честно: 4 окна — это НЕ статистика для вывода о знаке дельты (см. §3.1: на 64 окнах W=65k
    все дельты пересекали ноль); точка даёт лишь порядок величины на асимптотике."""
    mode = "cyclic"
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    Mcand = min(1024, max(1, W - TAIL))
    out = {"W": W, "exact": [], "gen16x": []}
    with torch.no_grad():
        for i in range(nwin):
            o = 1000 + i * (W // nwin)  # разнести окна по корпусу
            x = toks[o:o + W]
            # exact
            lg = forward_general(model, x, mode, chunk=W)
            out["exact"].append(round(float(F.cross_entropy(lg, toks[o + W].view(1)).item()), 4))
            # gen 32x (L=2,S=12 => 24 Б/ток, 32× от базы)
            kc = fm.encode_rows(keys_at(model, x, torch.arange(W, device=device), mode))
            lg2 = forward_generative(model, x, fm, kc, Mcand)
            out["gen16x"].append(round(float(F.cross_entropy(lg2, toks[o + W].view(1)).item()), 4))
    out["exact_mean"] = round(sum(out["exact"]) / len(out["exact"]), 4)
    out["gen16x_mean"] = round(sum(out["gen16x"]) / len(out["gen16x"]), 4)
    return out


# forward_generative нужен для PPL (переиспользуем из fold_unfold_ppl)
from fold_unfold_ppl import forward_generative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "results", "ckpts", "sts_prog_seed0.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--W3", type=int, default=1048576, help="W для Риска 3 (1M)")
    ap.add_argument("--cb-ns", type=int, default=262144)
    ap.add_argument("--out", default=os.path.join(RESULTS, "risk23.json"))
    args = ap.parse_args()
    dev = args.device

    import numpy as np, final_benchmark as fb
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)

    out = {"risk2_conflict": [], "risk3_spectral": [], "risk3_ppl_1m": None}

    # -------- РИСК 2: conflict-probe на W=262144 (там же, где спектр) --------
    print("=== РИСК 2: conflict-probe @ W=262144 ===", flush=True)
    W2 = 262144
    mode2 = "cyclic"
    fm2 = StreamFracode(192, levels=2, subvecs=12, K=256, device=dev)
    with torch.no_grad():
        cb = keys_at(model, toks[W2:2 * W2], torch.arange(W2, device=dev), mode2).detach()
        if cb.shape[0] > args.cb_ns:
            cb = cb[::cb.shape[0] // args.cb_ns]
        fm2.fit(cb, iters=12, seed=0)
    r2 = risk2_conflict(model, toks, W2, dev, k_eff, topk, nq, mode2, fm=fm2)
    print(f"  gap_exact={r2['gap_exact']}  gap_gen={r2['gap_gen']}", flush=True)
    out["risk2_conflict"].append(r2)

    # -------- РИСК 3: W=1M спектр (Welford) + PPL --------
    print(f"=== РИСК 3: спектральный @ W={args.W3:,} (Welford) ===", flush=True)
    r3 = risk3_spectral_welford(model, toks, args.W3, dev)
    print(f"  E1={r3['E1']}  PR={r3['PR']}  var={r3['interpos_var']}  "
          f"meanCos={r3['meanCos_to_driver']}  unfold={r3['unfold_sec']}s", flush=True)
    out["risk3_spectral"].append(r3)

    # PPL на 1M (если влезет)
    print(f"=== РИСК 3: end-to-end PPL @ W={args.W3:,} ===", flush=True)
    r3p = risk3_ppl_1m(model, toks, args.W3, dev, fm2, nwin=4)
    print(f"  exact_mean={r3p['exact_mean']}  gen16x_mean={r3p['gen16x_mean']}", flush=True)
    out["risk3_ppl_1m"] = r3p

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nСохранено: {args.out}", flush=True)


if __name__ == "__main__":
    main()
