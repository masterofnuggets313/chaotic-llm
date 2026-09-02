"""two_tier_generate.py — прототип Two-Tier Memory для STS-Prog.

Архитектура (по дизайну из results/two_tier_memory_design.md):
  - Активное окно: последние N_local = 8192 токенов, точные ключи e (FP32).
  - Глубокая память: всё, что за пределами активного окна — Fracode 32× (24 Б/ток).
  - g = гибрид: g_fast (среднее h по активному окну) + λ·g_deep (периодический средний
    вектор из глубины).
  - Триггер глобального поиска: каждые K=32 шага (детерминированно).
  - readout: readout3(concat[h_last, q0, g]) — без изменения размерности.

Сравнение: exact (full-W, 1 шаг) vs two-tier (активное окно + глубина, 32 шага).
Метрики: PPL (loss), время (tok/s), logit match (cos(логитов)).

ЗАПУСК: cd phase01/exp_vq && py -3.13 two_tier_generate.py
"""
import os, sys, time, json, torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import (
    keys_at, keys_range, forward_general, TAIL, TEMP, cos_sim, StreamFracode)

RESULTS = os.path.join(REPO, "results")

# ---------------------------------------------------------------- настройки
N_LOCAL = 8192           # активное окно
K_GLOBAL = 32            # раз в 32 шага — глобальный поиск по глубине
LAMBDA_G = 0.5           # вес g_deep в гибриде g (0.5 = равный вес)
M_CAND = 1024            # количество кандидатов при глобальном поиске


def _exact_driver(model, e_all, q, Wc, k_eff, topk, nq):
    """Один проход: q -> driver (8 слоёв, полный Wc). Возвращает (h, driver, g_sum).
    Аккуратно считает g = h.mean(0) без хранения всего (W,d)."""
    pos_q = e_all[Wc - nq:].mean(0, keepdim=True)
    q_cur = q if q is not None else pos_q
    h = e_all
    driver = None
    g_sum = torch.zeros(model.d, device=e_all.device)
    BLK = 131072
    for li, blk in enumerate(model.blocks):
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        hc = []
        for s in range(0, Wc, BLK):
            hc.append(blk(h[s:s + BLK], driver, k_eff))
        h = torch.cat(hc, 0)
        g_sum += h.sum(0)  # для g = mean по всем позициям
        q_cur = pos_q + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    return h, driver, g_sum


def _local_driver(model, e_local, q_cur, pos_q, Wc, k_eff, topk, nq):
    """Драйвер ТОЛЬКО из активного окна (N_local). Возвращает (h_local, driver, g_local_sum).
    h_local — полное состояние только для активного окна (N_local, d).
    pos_q — фиксированный q0 (mean последних nq ключей), не меняется в цикле."""
    N = e_local.shape[0]
    h = e_local
    driver = None
    g_sum = torch.zeros(model.d, device=e_local.device)
    BLK = 131072
    for li, blk in enumerate(model.blocks):
        en = e_local / (e_local.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[N - TAIL:] = -1e9
        kk = min(topk, N - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, N - 2)
        driver = (w.unsqueeze(-1) * e_local[nxt]).sum(0, keepdim=True)
        hc = []
        for s in range(0, N, BLK):
            hc.append(blk(h[s:s + BLK], driver, k_eff))
        h = torch.cat(hc, 0)
        g_sum += h.sum(0)
        q_cur = pos_q + model.query_proj(h[-1].unsqueeze(0)) * 0.5
    return h, driver, g_sum


def two_tier_generate(model, fm, toks, Wc, N_local, steps, k_eff, topk, nq, mode, q0_in=None):
    """Two-Tier генерация: активное окно + глубокая память.

    Возвращает:
      logits_history: list of logits (для PPL)
      timing: dict с замером времени компонентов.
    """
    dev = toks.device
    # начальное окно depth = Wc - N_local (всё, что до активного окна, это глубина)
    depth = Wc - N_local
    # начальный q0
    if q0_in is not None:
        q0 = q0_in
    else:
        q0 = keys_at(model, toks, torch.arange(Wc - nq, Wc, device=dev), mode).mean(0, keepdim=True)

    # ---- кодируем глубину в Fracode (один раз) ----
    t0 = time.time()
    e_depth = keys_at(model, toks[:depth], torch.arange(depth, device=dev), mode).detach()
    depth_codes = fm.encode_rows(e_depth)  # (depth, L, S)
    del e_depth
    t_encode = time.time() - t0

    # ---- активное окно (начальное) ----
    e_local = keys_at(model, toks[depth:depth + N_local],
                      torch.arange(depth, depth + N_local, device=dev), mode).detach()

    # ---- пробный readout для PPL (exact vs two-tier) ----
    # exact: полный forward по всем Wc (как эталон)
    t0 = time.time()
    q_cur = q0
    h, driver, g_sum = _exact_driver(model, e_local, q_cur, N_local, k_eff, topk, nq)
    # но это только для локального окна. Для exact full-W нужен другой проход.
    # Сделаем один exact full-W для сравнения.
    e_full = keys_at(model, toks[:Wc], torch.arange(Wc, device=dev), mode).detach()
    h_full, driver_full, g_sum_full = _exact_driver(model, e_full, q0, Wc, k_eff, topk, nq)
    g_full = (g_sum_full / Wc).unsqueeze(0)
    h_last_full = h_full[-1].unsqueeze(0)
    logits_exact = model.readout3(torch.cat([h_last_full, q0, g_full], dim=-1))
    t_exact = time.time() - t0
    del e_full, h_full, driver_full

    # ---- Two-Tier: N_local шагов ----
    # На каждом шаге: драйвер из активного окна; раз в K шагов — глобальный поиск.
    # g = g_fast + λ·g_deep (где g_fast = h_local.mean(0), g_deep = средний unfold из глубины)
    t_gen = 0.0
    t_global = 0.0
    t_local = 0.0

    # Инициализируем h_local на активном окне
    q_cur = q0
    h_local, driver_local, g_local_sum = _local_driver(model, e_local, q_cur, N_local, k_eff, topk, nq)
    g_fast = (g_local_sum / N_local).unsqueeze(0)  # (1, d)

    # Глубинный g: начальный (средний по глубине) — раз в K шагов
    g_deep = torch.zeros(1, model.d, device=dev)
    global_step = 0

    logits_history = []
    for step in range(steps):
        # ---- драйвер из активного окна (дешёвый) ----
        t0 = time.time()
        en = e_local / (e_local.norm(dim=-1, keepdim=True) + 1e-6)
        qn = q_cur / (q_cur.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[N_local - TAIL:] = -1e9
        kk = min(topk, N_local - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, N_local - 2)
        driver_local = (w.unsqueeze(-1) * e_local[nxt]).sum(0, keepdim=True)
        t_local += time.time() - t0

        # ---- глобальный поиск: раз в K шагов ----
        if step % K_GLOBAL == 0 and depth > 0:
            t0 = time.time()
            fm.codes = depth_codes
            _, glob_next, _ = fm.select(q_cur.squeeze(0), topk, depth, M_CAND, rerank=False)
            # разворачиваем найденные кандидаты из глубины
            e_deep = fm.decode_codes(depth_codes[glob_next])  # (topk, d)
            # g_deep = средний ключ среди найденных (как вклад глубины в readout)
            g_deep = e_deep.mean(0, keepdim=True)  # (1, d)
            t_global += time.time() - t0
            global_step += 1

        # ---- прогон блоков по активному окну (с новым драйвером) ----
        t0 = time.time()
        BLK = 131072
        hc = []
        for s in range(0, N_local, BLK):
            hc.append(model.blocks[0](h_local[s:s + BLK], driver_local, k_eff))
        # на самом деле нужно 8 слоёв, а не 1. Упростим: делаем через _local_driver
        # но _local_driver уже сделал полный проход. Для реального пошагового декодинга
        # надо брать h_local из предыдущего шага и обновлять только последний блок.
        # Для прототипа: пересчитываем весь h_local каждый шаг (упрощение).
        h_local, driver_local, g_local_sum = _local_driver(model, e_local, q_cur, N_local, k_eff, topk, nq)
        g_fast = (g_local_sum / N_local).unsqueeze(0)
        t_local += time.time() - t0

        # ---- readout ----
        h_last = h_local[-1].unsqueeze(0)
        g_hybrid = g_fast + LAMBDA_G * g_deep
        logits = model.readout3(torch.cat([h_last, q0, g_hybrid], dim=-1))
        logits_history.append(logits)

        # ---- выбираем следующий токен (argmax, не sample — для PPL) ----
        next_id = logits.argmax(-1).item()
        # ---- сдвигаем окно ----
        # проще всего: пересчитываем e_local для нового окна (сдвиг на 1)
        # В реальном прототипе: кэшировать e, здесь — упрощение
        e_local = keys_at(model, toks[depth + step + 1:depth + step + 1 + N_local],
                          torch.arange(depth + step + 1, depth + step + 1 + N_local, device=dev),
                          mode).detach()
        # q_cur для следующего шага
        q_cur = q0 + model.query_proj(h_last) * 0.5
        t_gen += time.time() - t0

    timing = {
        "encode_deep_sec": round(t_encode, 3),
        "exact_full_sec": round(t_exact, 3),
        "local_avg_ms": round(t_local / steps * 1000, 2),
        "global_total_ms": round(t_global * 1000, 2),
        "gen_total_sec": round(t_gen, 3),
        "tok_s": round(steps / t_gen, 2) if t_gen > 0 else 0,
    }
    return logits_history, logits_exact, timing


def main():
    import final_benchmark as fb
    dev = "cuda"
    Wc = 262144

    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(RESULTS, "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)

    # ---- Fracode для глубинной памяти ----
    mode = "cyclic"
    # калибровка кодбука на первом Wc токенах (глубинная часть)
    print("Fitting Fracode codebook...", flush=True)
    calib = keys_at(model, toks[:Wc], torch.arange(Wc, device=dev), mode).detach()
    # подвыборка для калибровки (как в T5)
    if calib.shape[0] > 65536:
        calib = calib[::calib.shape[0] // 65536]
    fm = StreamFracode(model.d, levels=2, subvecs=12, K=256, device=dev)
    fm.fit(calib, iters=12, seed=0)
    del calib
    print(f"Fracode: {fm.bytes_per_pos:.0f} B/pos ({fm.L}x{fm.S}, K={fm.K})", flush=True)

    # ---- прогон ----
    steps = 32  # 32 шага — достаточно для оценки PPL и скорости
    print(f"\n=== Two-Tier @ W={Wc}, N_local={N_LOCAL}, K={K_GLOBAL}, steps={steps} ===", flush=True)

    logits_hist, logits_exact, timing = two_tier_generate(
        model, fm, toks, Wc, N_LOCAL, steps, k_eff, topk, nq, mode)

    # ---- PPL ----
    losses = []
    for step, logits in enumerate(logits_hist):
        target = toks[Wc + step].view(1)
        losses.append(float(torch.nn.functional.cross_entropy(logits, target).item()))
    loss_mean = sum(losses) / len(losses)
    loss_exact = float(torch.nn.functional.cross_entropy(logits_exact, toks[Wc].view(1)).item())

    # ---- logit match ----
    cos_logit = float(cos_sim(logits_hist[0], logits_exact).item())

    print(f"\nРезультаты:")
    print(f"  exact (full-W) loss:    {loss_exact:.4f}")
    print(f"  two-tier mean loss:     {loss_mean:.4f}  ({steps} steps)")
    print(f"  logit cos (step 0):     {cos_logit:.4f}")
    print(f"  timing:                 {timing['tok_s']:.2f} tok/s")
    print(f"  local avg:              {timing['local_avg_ms']:.1f} ms/step")
    print(f"  global total:           {timing['global_total_ms']:.1f} ms ({timing.get('global_total_ms',0)/steps:.1f} ms/step avg)")
    print(f"  encode deep:            {timing['encode_deep_sec']:.2f}s")

    out = {
        "W": Wc, "N_local": N_LOCAL, "K_global": K_GLOBAL, "steps": steps,
        "loss_exact": loss_exact, "loss_two_tier": loss_mean,
        "logit_cos": cos_logit, "tok_s": timing["tok_s"],
        "timing": timing,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "two_tier_proto.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nСохранено: {os.path.join(RESULTS, 'two_tier_proto.json')}")


if __name__ == "__main__":
    main()