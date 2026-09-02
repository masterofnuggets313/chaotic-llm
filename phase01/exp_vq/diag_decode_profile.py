"""diag_decode_profile.py — Шаг 0: профилировщик ОДНОГО decode-шага STS-Prog.

Отвечает на вопрос коллеги «куда уходят миллисекунды?» цифрами.
Моделирует генерацию одного токена при уже заполненном окне W: 
  x -> e = embed(x)+pos  ->  sts_prog forward (8 слоёв: драйвер + блоки + q).

Замеряет по-компонентно, где время:
  (a) e = embed+pos  — пересчёт ключей (неизбежно? есть ли кэш?)
  (b) пересчёт драйверов на каждом слое (topk по W + сборка)  — 8×
  (c) прогон блоков по всему W (8 слоёв × chank)
  (d) Python-overhead  (мелкие kernel'ы, построение тензора, sync)

Плюс — сколько даёт КЭШИРОВАНИЕ e (если бы мы не пересчитывали ключи каждый шаг):
  exact-decode (всё заново) vs cached-e (e зафиксирован, только h эволюционирует).

ЗАПУСК: cd phase01/exp_vq && py -3.13 diag_decode_profile.py
"""
import os, sys, time, torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model, W
from night_task5_fracode_forward import keys_at, TAIL, TEMP

RESULTS = os.path.join(REPO, "results")


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


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

    # Окно Wc токенов (полный контекст, как в decode-шаге)
    x = toks[:Wc].unsqueeze(0)  # (1, Wc)

    with torch.no_grad():
        # --- (a) пересчёт ключей e = embed+pos ---
        mode = "cyclic" if Wc > model.pos.shape[1] else "trained"
        te = timeit(lambda: keys_at(model, x[0], torch.arange(Wc, device=dev), mode), warmup=3, iters=5)

        # --- полный decode-шаг: e -> 8 драйверов -> 8 блоков -> q ---
        e_all = keys_at(model, x[0], torch.arange(Wc, device=dev), mode).detach()
        def full_step():
            # полный sts_prog forward на окне (как chat_sts_prog.generate: model(x))
            pos_q = e_all[Wc - nq:].mean(0, keepdim=True)
            q = pos_q; h = e_all
            for li, blk in enumerate(model.blocks):
                en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
                qn = q / (q.norm() + 1e-6)
                sim = (en * qn).sum(-1)
                sim[Wc - TAIL:] = -1e9
                kk = min(topk, Wc - TAIL)
                vals, loc = sim.topk(kk)
                w = torch.softmax(vals / TEMP, 0)
                nxt = torch.clamp(loc + 1, 0, Wc - 2)
                driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
                # прогон блоков чанками
                BLK = 131072
                hc = []
                for s in range(0, Wc, BLK):
                    hc.append(blk(h[s:s + BLK], driver, k_eff))
                h = torch.cat(hc, 0)
                q = pos_q + model.query_proj(h[-1].unsqueeze(0)) * 0.5
        t_full = timeit(full_step, warmup=3, iters=5)

        # --- (b) только пересчёт драйверов (без прогона блоков) ---
        def drivers_only():
            pos_q = e_all[Wc - nq:].mean(0, keepdim=True)
            q = pos_q
            for li in range(len(model.blocks)):
                en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
                qn = q / (q.norm() + 1e-6)
                sim = (en * qn).sum(-1)
                sim[Wc - TAIL:] = -1e9
                kk = min(topk, Wc - TAIL)
                vals, loc = sim.topk(kk)
                w = torch.softmax(vals / TEMP, 0)
                nxt = torch.clamp(loc + 1, 0, Wc - 2)
                driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
                # имитация q-обновления без реального блока
                q = pos_q + torch.zeros_like(driver) * 0.5
        t_drivers = timeit(drivers_only, warmup=3, iters=5)

        # --- (c) только прогон блоков (драйвер зафиксирован) ---
        driver_fixed = torch.zeros(1, model.d, device=dev)
        def blocks_only():
            h = e_all
            for li, blk in enumerate(model.blocks):
                BLK = 131072
                hc = []
                for s in range(0, Wc, BLK):
                    hc.append(blk(h[s:s + BLK], driver_fixed, k_eff))
                h = torch.cat(hc, 0)
        t_blocks = timeit(blocks_only, warmup=3, iters=5)

    tok_s = 1.0 / t_full
    print(f"\n=== ПРОФИЛЬ DECODE-ШАГА @ W={Wc:,} (d=192, L=8) ===")
    print(f"полный шаг (exact, всё заново):     {t_full*1000:8.1f} ms  ->  {tok_s:8.2f} tok/s")
    print(f"  (a) ключи e=embed+pos:             {te*1000:8.1f} ms  ({100*te/t_full:5.1f}%)")
    print(f"  (b) пересчёт драйверов (8 слоёв):  {t_drivers*1000:8.1f} ms  ({100*t_drivers/t_full:5.1f}%)")
    print(f"  (c) прогон блоков (8 слоёв):       {t_blocks*1000:8.1f} ms  ({100*t_blocks/t_full:5.1f}%)")
    print(f"  (d) python-overhead + sync:        {(t_full-te-t_drivers-t_blocks)*1000:8.1f} ms  ({100*(t_full-te-t_drivers-t_blocks)/t_full:5.1f}%)")

    out = {
        "W": Wc, "full_ms": round(t_full*1000, 2), "tok_s": round(tok_s, 3),
        "keys_ms": round(te*1000, 2), "drivers_ms": round(t_drivers*1000, 2),
        "blocks_ms": round(t_blocks*1000, 2),
        "overhead_ms": round((t_full-te-t_drivers-t_blocks)*1000, 2),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "decode_profile.json"), "w", encoding="utf-8") as f:
        import json; json.dump(out, f, indent=1)
    print(f"\nСохранено: {os.path.join(RESULTS, 'decode_profile.json')}")


if __name__ == "__main__":
    main()
