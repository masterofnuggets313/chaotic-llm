"""Эксперимент 5: полный оптимизированный decode = ADC-драйверы (кэш кодов 24B/поз)
+ блоки по K=256 + h_last + g_last_256.

Цель: превзойти Transformer KV-cache на tok/s при равном качестве.

Компоненты (все доказаны по отдельности):
- g_last_256: ΔPPL −0.0% (exp_g_ablation_K)
- Fracode L=2,S=12,K=256 на НОРМАЛИЗОВАННЫХ ключах: recall 100%, 24B/поз (exp3)
- ADC-селекция: gather из маленьких таблиц вместо O(W·d) cosine — кэш кодов 24B/поз
  вместо e_all 768B/поз

Векторизованный ADC-скоринг: tbls (L*S, K), codes_flat (W, L*S)
score = sum_j tbls[j, codes_flat[:, j]]  — один gather, нет Python-циклов.

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_adc_fast_decode.py
"""
import os, sys, time, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, forward_general, TAIL, TEMP, StreamFracode
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 256          # блоки по K позиций
Mcand = 512      # кандидатов ADC
FR_L, FR_S, FR_CB = 2, 12, 256   # Fracode


def adc_score_tables(fr, qn):
    """Таблицы (L*S, K) скоринга для нормированного запроса qn (d,) или (1, d).
    tbl[j, k] = dot(qn_sub, centroid[j][k]) — один проход по кодбукам."""
    L, S = fr.L, fr.S
    sub = fr.sub
    if qn.dim() == 2:
        qn = qn.squeeze(0)  # (d,)
    tabs = torch.empty(L * S, fr.K, device=qn.device)
    j = 0
    for l in range(L):
        for s in range(S):
            qsub = qn[s * sub:(s + 1) * sub]
            tabs[j] = qsub @ fr.cbooks[l][s].T          # (K,)
            j += 1
    return tabs


def adc_select(fr, codes, qn, topk, Wc, Mcand=512):
    """ADC-скрининг + rerank. Возвращает (vals, loc_next) как в exact_select.
    codes: (Wc, L*S) int. qn: (d,) или (1, d)."""
    if qn.dim() == 2:
        qn = qn.squeeze(0)
    tabs = adc_score_tables(fr, qn)
    # score[w] = sum_j tabs[j, codes[w, j]]  — один gather, нет Python-циклов
    g = tabs.gather(1, codes.t())                        # (L*S, Wc)
    scores = g.sum(0)                                    # (Wc,)
    scores[Wc - TAIL:] = -1e18
    m = min(Mcand, max(1, Wc - TAIL))
    cand = scores.topk(m).indices                        # (m,)
    cn = torch.clamp(cand + 1, 0, Wc - 2)
    pos = torch.unique(torch.cat([cand, cn]))
    rec = fr.decode_codes(fr.codes[pos])                 # (P, d) восстановленные
    recn = rec / (rec.norm(dim=-1, keepdim=True) + 1e-6)
    sim = (recn * qn.unsqueeze(0)).sum(-1)               # (P,) точный косинус
    i_c = torch.searchsorted(pos, cand)
    sim_c = sim[i_c]
    vals, loc = sim_c.topk(min(topk, m))
    return vals, cn[loc]


def adc_fast_decode(model, fr, codes, e_all, q0, Wc, k_eff, topk, nq, K, Mcand):
    """Один decode-шаг: ADC-драйверы по кодам + блоки по K + h_last + g."""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    for blk in model.blocks:
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        vals, nxt = adc_select(fr, codes, qn, topk, Wc, Mcand)
        w = torch.softmax(vals / TEMP, 0)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        h_last = blk(h_last, driver, k)
        hK = blk(hK, driver, k)
        q = q0 + model.query_proj(h_last) * 0.5
    g = hK.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


def timeit(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    dev = "cuda"
    print("Loading model...", flush=True)
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
    mode = "cyclic"

    Wq = 65536
    o = 5000

    # --- 1. Качество: ADC+fast vs exact на 8 окнах W=65536 ---
    print(f"Качество ADC+fast (K={K}, Mcand={Mcand}) на 8 окнах W={Wq}...", flush=True)
    cos_list, ce_fast, ce_ex = [], [], []
    with torch.no_grad():
        for wi in range(8):
            oo = 5000 + wi * Wq
            if oo + Wq >= len(toks) - 1: break
            end = oo + Wq
            x = toks[oo:end]; target = toks[end].view(1)
            lg_ex = forward_general(model, x, mode, chunk=Wq)
            ce_ex.append(float(F.cross_entropy(lg_ex, target).item()))
            e_all = keys_at(model, x, torch.arange(Wq, device=dev), mode).detach()
            en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
            fr = StreamFracode(d=192, levels=FR_L, subvecs=FR_S, K=FR_CB, device=dev)
            fr.fit(en, iters=12, seed=0)
            codes3 = fr.encode_rows(en)                     # (Wc, L, S)
            flat = codes3.reshape(Wq, FR_L * FR_S)
            fr.codes = codes3
            q0 = e_all[-nq:].mean(0, keepdim=True)
            lg = adc_fast_decode(model, fr, flat, e_all, q0, Wq, k_eff, topk, nq, K, Mcand)
            ce_fast.append(float(F.cross_entropy(lg, target).item()))
            cos_list.append(float(F.cosine_similarity(lg, lg_ex, dim=-1).item()))
    cos_m = float(np.mean(cos_list))
    ce_ex_m = float(np.mean(ce_ex)); ce_fast_m = float(np.mean(ce_fast))
    dppl = (np.exp(ce_fast_m) / np.exp(ce_ex_m) - 1) * 100
    print(f"cos={cos_m:.6f}  CE exact={ce_ex_m:.4f} fast={ce_fast_m:.4f}  ΔPPL={dppl:+.1f}%")

    # --- 2. Скорость vs TF KV на W=16384/65536/262144 ---
    from exp_decode_vs_transformer import TransformerKV, fast_decode_step, timeit as _t2
    from parametric_models import TransformerLM, count_params
    from match_transformer import pick_tf_dims

    MAX_W = 262144
    D_tf = pick_tf_dims(900_000, V, 512, layers=8, heads=4)
    D_tf = max(4, (D_tf // 4) * 4)
    tf = TransformerLM(V, 512, D=D_tf, HEADS=4, LAYERS=8).to(dev).eval()
    n_pos = tf.pos.numel()
    tf.pos = torch.nn.Parameter(torch.zeros(1, MAX_W, D_tf, device=dev)); tf.pos.requires_grad_(False)
    for p in tf.parameters(): p.requires_grad_(False)
    tfkv = TransformerKV(tf)

    print(f"\nЗамер скорости (ADC+fast vs TF KV):", flush=True)
    results = {}
    for Wc in [16384, 65536, 262144]:
        x = toks[o:o + Wc]
        # ADC+fast (Fracode обучен один раз на окно — амортизировано; тайминг только шаг)
        e_all = keys_at(model, x, torch.arange(Wc, device=dev), mode).detach()
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        fr = StreamFracode(d=192, levels=FR_L, subvecs=FR_S, K=FR_CB, device=dev)
        fr.fit(en, iters=12, seed=0)
        codes3 = fr.encode_rows(en)                     # (Wc, L, S)
        flat = codes3.reshape(Wc, FR_L * FR_S)
        fr.codes = codes3
        q0 = e_all[-nq:].mean(0, keepdim=True)
        t_adc = timeit(lambda: adc_fast_decode(model, fr, flat, e_all, q0, Wc, k_eff, topk, nq, K, Mcand),
                       warmup=2, iters=5)
        adc_tps = 1 / t_adc

        # TF KV decode
        k_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        v_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        lengths = [Wc - 1] * 8
        t_tf = timeit(lambda: tfkv.decode_one(x[Wc - 1].view(1), k_cache, v_cache, Wc - 1, lengths),
                      warmup=2, iters=5)
        tf_tps = 1 / t_tf

        print(f"W={Wc:>7}: ADC+fast {t_adc*1000:6.1f} ms / {adc_tps:6.2f} tok/s | "
              f"TF KV {t_tf*1000:6.1f} ms / {tf_tps:6.2f} tok/s | "
              f"ADC/TF = {adc_tps/tf_tps:.2f}x")
        results[Wc] = {"adc_ms": round(t_adc * 1000, 1), "adc_tok_s": round(adc_tps, 1),
                       "tf_ms": round(t_tf * 1000, 1), "tf_tok_s": round(tf_tps, 1),
                       "adc_over_tf": round(adc_tps / tf_tps, 2)}

    out = {"K": K, "Mcand": Mcand, "cos": round(cos_m, 6),
           "delta_ppl_pct": round(dppl, 1), "results": results}
    with open(os.path.join(RESULTS, "exp_adc_fast_decode.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_adc_fast_decode.json")


if __name__ == "__main__":
    main()