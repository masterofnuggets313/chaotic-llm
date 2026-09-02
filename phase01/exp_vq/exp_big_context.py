"""Эксперимент 9: демонстрация decode STS-Prog на W=2M+ через Fracode-коды.

Состояние: токены (int32) + Fracode-коды (24B/поз). Точные e_all для драйвера
вычисляются на лету из embed(x[nxt]) + pos[nxt % P] — только для top-8 позиций.

TF KV на W=2M = 11.3GB — не влезает в 12GB (с моделью) → TF не может.
STS-Prog с кодами = 48MB → работает.

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_big_context.py 2>&1
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
K = 256
Mcand = 512
FR_L, FR_S, FR_CB = 2, 12, 256


def load_big_data(path, max_bytes=50 * 1024 * 1024):
    """Загрузка до max_bytes из файла."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes)


def main():
    dev = "cuda"
    print("Loading model + tokenizer...", flush=True)
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()

    # Загружаем большой корпус
    stack = load_big_data(os.path.join(PHASE, "corpus_stack_train.txt"), 80_000_000)
    train = fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)
    big_text = stack + train
    print(f"Big text: {len(big_text):,} chars", flush=True)
    ids = np.array(tok.encode(big_text).ids, dtype=np.int64)
    del big_text
    print(f"Total tokens: {len(ids):,}", flush=True)

    W = min(2_000_000, len(ids) - 1000)
    toks = torch.tensor(ids[:W], dtype=torch.long, device=dev)
    print(f"W={W:,} ({(W * 24) / 1024**2:.0f}MB codes)", flush=True)

    model = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                           sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(RESULTS, "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
    for p in model.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(model.k); topk = int(model.topk); nq = int(model.nquery)
    mode = "cyclic"
    P = model.pos.shape[1]

    # --- Шаг 1: чанкованный encode Fracode ---
    print("Encoding Fracode codes (chunked)...", flush=True)
    chunk = 32768
    fr = None
    codes_chunks = []
    # Собираем подвыборку для обучения (первые 500K точек)
    fit_samples = []
    for s in range(0, min(500_000, W), chunk):
        e = min(s + chunk, W)
        xs = toks[s:e]
        pos = torch.arange(s, e, device=dev)
        e_all = model.embed(xs) + model.pos[0, pos % P]
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        fit_samples.append(en.cpu())
        del e_all, en
    fit_X = torch.cat(fit_samples, 0).to(dev)
    print(f"  fit samples: {fit_X.shape[0]:,}", flush=True)
    fr = StreamFracode(d=192, levels=FR_L, subvecs=FR_S, K=FR_CB, device=dev)
    fr.fit(fit_X, iters=12, seed=0)
    del fit_X, fit_samples
    torch.cuda.empty_cache()
    print(f"  Fracode trained", flush=True)

    # Кодируем все чанки
    for s in range(0, W, chunk):
        e = min(s + chunk, W)
        xs = toks[s:e]
        pos = torch.arange(s, e, device=dev)
        e_all = model.embed(xs) + model.pos[0, pos % P]
        en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)
        c3 = fr.encode_rows(en)
        codes_chunks.append(c3.cpu())
        del e_all, en, c3
        if (s // chunk) % 10 == 0:
            print(f"  encoded {e:,}/{W:,}", flush=True)
    codes = torch.cat(codes_chunks, 0).to(dev)  # (W, L, S)
    fr.codes = codes
    flat = codes.reshape(W, FR_L * FR_S)
    print(f"  codes: {codes.shape}, {codes.numel() * 4 / 1024**2:.0f}MB", flush=True)

    # --- Шаг 2: измеряем decode-шаг ---
    # Для точного драйвера нужны e_all[nxt] — вычисляем на лету из токенов
    # Токены уже есть в toks. embed + pos доступны.
    # q0 = mean(e_all[-nq:]) — нужны e_all последних nq позиций
    # Вычисляем e_all для последних nq позиций (для q0)
    pos_q = torch.arange(W - nq, W, device=dev)
    e_q = model.embed(toks[W - nq:W]) + model.pos[0, pos_q % P]
    q0 = e_q.mean(0, keepdim=True)
    del e_q

    def e_at(pos_idx):
        """Вычисление точных e_all для заданных позиций (n,) -> (n, d)."""
        p = pos_idx.clamp(0, W - 1)
        return model.embed(toks[p]) + model.pos[0, p % P]

    def adc_fast_decode():
        q = q0
        h_last = e_at(torch.tensor([W - 1], device=dev))
        hK = e_at(torch.tensor(range(W - K, W), device=dev))
        for blk in model.blocks:
            k = k_eff
            qn = q / (q.norm() + 1e-6)
            if qn.dim() == 2: qn = qn.squeeze(0)
            tabs = torch.empty(FR_L * FR_S, FR_CB, device=dev)
            j = 0
            for l in range(FR_L):
                for s in range(FR_S):
                    tabs[j] = qn[s * fr.sub:(s + 1) * fr.sub] @ fr.cbooks[l][s].T
                    j += 1
            g = tabs.gather(1, flat.t())
            scores = g.sum(0)
            scores[W - TAIL:] = -1e18
            m = min(Mcand, max(1, W - TAIL))
            cand = scores.topk(m).indices
            cn = torch.clamp(cand + 1, 0, W - 2)
            pos = torch.unique(torch.cat([cand, cn]))
            rec = fr.decode_codes(fr.codes[pos])
            recn = rec / (rec.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (recn * qn.unsqueeze(0)).sum(-1)
            i_c = torch.searchsorted(pos, cand)
            sim_c = sim[i_c]
            vals, loc = sim_c.topk(min(topk, m))
            nxt = cn[loc]
            w = torch.softmax(vals / TEMP, 0)
            # Точный driver из e_all[nxt] (вычисляем на лету)
            ed = e_at(nxt)
            driver = (w.unsqueeze(-1) * ed).sum(0, keepdim=True)
            h_last = blk(h_last, driver, k)
            hK = blk(hK, driver, k)
            q = q0 + model.query_proj(h_last) * 0.5
        g = hK.mean(0, keepdim=True)
        return model.readout3(torch.cat([h_last, q0, g], dim=-1))

    # warmup
    for _ in range(2):
        adc_fast_decode()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        adc_fast_decode()
    torch.cuda.synchronize()
    t = (time.time() - t0) / 5
    tps = 1 / t
    print(f"\n=== W={W:,} ===", flush=True)
    print(f"ADC decode: {t*1000:.1f} ms -> {tps:.1f} tok/s", flush=True)

    # Память
    cache_mb = (W * 24) / 1024**2  # Fracode codes
    tokens_mb = (W * 4) / 1024**2  # tokens
    tf_kv = (W * 88 * 2 * 8 * 4) / 1024**3  # TF KV at D=88
    print(f"Memory: codes={cache_mb:.0f}MB + tokens={tokens_mb:.0f}MB = {cache_mb+tokens_mb:.0f}MB")
    print(f"TF KV would need: {tf_kv:.1f}GB (doesn't fit in 12GB)")
    print(f"TF KV on 12GB max W: {12e9 / (88*2*8*4):.0f} tokens")

    out = {"W": W, "ms": round(t * 1000, 1), "tok_s": round(tps, 1),
           "sts_mb": round(cache_mb + tokens_mb, 0),
           "tf_kv_gb": round(tf_kv, 1)}
    with open(os.path.join(RESULTS, "exp_big_context.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_big_context.json")


if __name__ == "__main__":
    main()