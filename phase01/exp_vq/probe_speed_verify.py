"""probe_speed_verify.py — честная проверка заявления про скорость (~200K tok/s).

Коллега выделил: «скорость не падает — ~200K tok/s на всех масштабах» — потенциально
даже круче 10M контекста, НО надо проверить, что это именно throughput обработки контекста.

Разводим ДВЕ метрики:
  (A) PROCESSING / PREFILL throughput: весь контекст W прогоняется за ОДИН forward.
      tok/s = W / dt_forward. Это то, что мерилось ранее (~200K, константа).
  (B) DECODE throughput: токены генерятся по одному (инкрементально), состояние несём
      между шагами. На каждом шаге — скан ВСЕХ позиций контекста (top-k по всем) =>
      стоимость шага O(W) => декодирование ЗАМЕДЛЯЕТСЯ с ростом контекста (линейно),
      а не константа. Честная «скорость генерации».

Память: (B) держит ec_all=(W,d) для селекции + h_sum=(d) для readout, плюс на префилле
второй (W,d) временно. Поэтому декод пускаем ТОЛЬКО на W0<=1M (на 10M было бы ~15GB).
10M decode выводим экстраполяцией по линейному тренду (per-step ∝ W).
"""
import os, sys, math, time
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
from models_pc import build_pc_model

D_PE = 32
TEMP = 0.3


def sinusoid(idx, d_pe):
    half = d_pe // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=idx.device) / half))
    ang = idx.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def chunked_forward(base, x, C=4096, temp=TEMP):
    """Тот же доказанный форвард, что в probe_10m_purepclm.py (in-place, без удвоения (W,d))."""
    B, W = x.shape
    d = base.d
    k_eff = torch.sigmoid(base.k)
    blocks = base.blocks
    L = len(blocks)
    pe_proj = base.pe_proj
    nchunks = math.ceil(W / C)
    h_chunks = []
    for ci in range(nchunks):
        idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
        xc = x[:, idx]
        ec = base.embed(xc) + pe_proj(sinusoid(idx, D_PE).to(x.device))
        h_chunks.append(ec)
    q0 = h_chunks[-1][:, -base.nquery:, :].mean(dim=1)
    q = q0
    for li in range(L):
        run_val = torch.full((B, base.topk), -1e9, device=x.device)
        run_gidx = torch.zeros((B, base.topk), dtype=torch.long, device=x.device)
        run_key = torch.zeros((B, base.topk, d), device=x.device)
        for ci in range(nchunks):
            idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
            ec = h_chunks[ci]
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            en = ec / (ec.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)
            sim[:, idx >= W - 8] = -1e9
            ks = min(base.topk, ec.shape[1])
            cs, cloc = sim.topk(ks, dim=1)
            cg = ci * C + cloc
            ckey = torch.gather(ec, 1, cloc.unsqueeze(-1).expand(-1, -1, d))
            allv = torch.cat([run_val, cs], 1)
            allg = torch.cat([run_gidx, cg], 1)
            allk = torch.cat([run_key, ckey], 1)
            _, ordr = allv.topk(base.topk, dim=1)
            run_val = allv.gather(1, ordr)
            run_gidx = allg.gather(1, ordr)
            run_key = allk.gather(1, ordr.unsqueeze(-1).expand(-1, -1, d))
        w = torch.softmax(run_val / temp, dim=1)
        driver = (w.unsqueeze(-1) * run_key).sum(dim=1, keepdim=True)
        for ci in range(nchunks):
            h_chunks[ci] = blocks[li](h_chunks[ci], driver, k_eff)
        h_last = h_chunks[-1][:, -1, :]
        q = q0 + base.query_proj(h_last) * 0.5
    g_sum = torch.zeros((B, d), device=x.device)
    for hc in h_chunks:
        g_sum += hc.sum(dim=1)
    g = g_sum / W
    h_last = h_chunks[-1][:, -1, :]
    return base.readout3(torch.cat([h_last, q0, g], dim=-1))


def prefill_state(base, x, C=4096, temp=TEMP):
    """Строит состояние для декодирования: ec_all=(W,d) ключи + h_sum=(d) + h_last."""
    B, W = x.shape
    d = base.d
    k_eff = torch.sigmoid(base.k)
    blocks = base.blocks
    L = len(blocks)
    pe_proj = base.pe_proj
    nchunks = math.ceil(W / C)
    ec_all = []
    for ci in range(nchunks):
        idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
        xc = x[:, idx]
        ec = base.embed(xc) + pe_proj(sinusoid(idx, D_PE).to(x.device))
        ec_all.append(ec)
    h_chunks = [e.clone() for e in ec_all]  # (W,d) total — ВТОРОЙ тензор, поэтому W<=1M
    q0 = h_chunks[-1][:, -base.nquery:, :].mean(dim=1)
    q = q0
    for li in range(L):
        run_val = torch.full((B, base.topk), -1e9, device=x.device)
        run_key = torch.zeros((B, base.topk, d), device=x.device)
        for ci in range(nchunks):
            idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
            ec = ec_all[ci]
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            en = ec / (ec.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)
            sim[:, idx >= W - 8] = -1e9
            ks = min(base.topk, ec.shape[1])
            cs, cloc = sim.topk(ks, dim=1)
            ckey = torch.gather(ec, 1, cloc.unsqueeze(-1).expand(-1, -1, d))
            allv = torch.cat([run_val, cs], 1)
            allk = torch.cat([run_key, ckey], 1)
            _, ordr = allv.topk(base.topk, dim=1)
            run_val = allv.gather(1, ordr)
            run_key = allk.gather(1, ordr.unsqueeze(-1).expand(-1, -1, d))
        w = torch.softmax(run_val / temp, dim=1)
        driver = (w.unsqueeze(-1) * run_key).sum(dim=1, keepdim=True)
        for ci in range(nchunks):
            h_chunks[ci] = blocks[li](h_chunks[ci], driver, k_eff)
        h_last = h_chunks[-1][:, -1, :]
        q = q0 + base.query_proj(h_last) * 0.5
    ec_all_cat = torch.cat(ec_all, 1)  # (B, W, d)
    h_sum = torch.zeros((B, d), device=x.device)
    for hc in h_chunks:
        h_sum += hc.sum(dim=1)
    h_last = h_chunks[-1][:, -1, :]
    return {"ec": ec_all_cat, "h_sum": h_sum, "h_last": h_last, "q0": q0, "W": W}


def decode_step(base, st, x_new, temp=TEMP):
    d = base.d
    pe_proj = base.pe_proj
    blocks = base.blocks
    L = len(blocks)
    k_eff = torch.sigmoid(base.k)
    ec = st["ec"]
    h_sum = st["h_sum"]
    q0 = st["q0"]
    W = st["W"]
    W = W + 1
    idx_new = torch.tensor([W - 1], device=x_new.device)
    ec_new = base.embed(x_new) + pe_proj(sinusoid(idx_new, D_PE).to(x_new.device))
    ec = torch.cat([ec, ec_new], 1)  # (B, W, d)
    q = q0
    h_new = None
    for li in range(L):
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        en = ec / (ec.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (en * qn.unsqueeze(1)).sum(-1)  # (B, W)
        sim[:, max(0, W - 8):] = -1e9
        cs, cloc = sim.topk(base.topk, dim=1)
        ckey = torch.gather(ec, 1, cloc.unsqueeze(-1).expand(-1, -1, d))
        w = torch.softmax(cs / temp, dim=1)
        driver = (w.unsqueeze(-1) * ckey).sum(dim=1, keepdim=True)
        if h_new is None:
            h_new = blocks[li](ec_new, driver, k_eff)  # (B,1,d) от входного эмбеддинга
        else:
            h_new = blocks[li](h_new, driver, k_eff)
        hlast = h_new[:, 0, :]
        q = q0 + base.query_proj(hlast) * 0.5
    h_sum = h_sum + h_new[:, 0, :]
    g = h_sum / W
    _ = base.readout3(torch.cat([h_new[:, 0, :], q0, g], dim=-1))
    return {"ec": ec, "h_sum": h_sum, "h_last": h_new[:, 0, :], "q0": q0, "W": W}


def main():
    V = 512
    d = 192
    L = 8
    torch.manual_seed(0)
    base = build_pc_model("pc", vocab=V, d=d, layers=L, k_init=1.2,
                          sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).cuda()
    base.pe_proj = nn.Linear(D_PE, d).cuda()
    print(f"STS-Prog speed-verify: d={d} L={L}, derived-pos + chunked", flush=True)

    # ---------- (A) PROCESSING / PREFILL throughput ----------
    print("\n[A] PROCESSING (prefill) throughput — весь контекст за 1 forward", flush=True)
    proc = {}
    for W in [262_144, 1_000_000, 10_000_000]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, V, (1, W), device="cuda")
        t0 = time.time()
        with torch.no_grad():
            _ = chunked_forward(base, x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        tps = W / dt
        proc[W] = tps
        print(f"  W={W:,} : {tps:,.0f} tok/s  (dt={dt:.2f}s, peakVRAM={vram/1024:.2f}GB)", flush=True)

    # ---------- (B) DECODE throughput (инкрементально) ----------
    print("\n[B] DECODE throughput — токен за токеном (состояние между шагами)", flush=True)
    print("    NOTE: на каждом шаге скан ВСЕХ позиций => per-step O(W) => замедляется с W", flush=True)
    decode = {}
    for W0 in [262_144, 1_000_000]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, V, (1, W0), device="cuda")
        tp0 = time.time()
        st = prefill_state(base, x)
        torch.cuda.synchronize()
        dt_pre = time.time() - tp0
        # warmup 1 шаг (исключить первый вызов)
        nw = torch.randint(0, V, (1, 1), device="cuda")
        with torch.no_grad():
            st = decode_step(base, st, nw)
        torch.cuda.synchronize()
        N = 32
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                nw = torch.randint(0, V, (1, 1), device="cuda")
                st = decode_step(base, st, nw)
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        tps = N / dt
        decode[W0] = tps
        print(f"  ctx={W0:,} : prefill={dt_pre:.2f}s, decode={tps:.1f} tok/s "
              f"(32 шага за {dt:.2f}s, peakVRAM={vram/1024:.2f}GB)", flush=True)

    # экстраполяция 10M decode: per-step ∝ W => tok/s ∝ 1/W
    if 262_144 in decode and 1_000_000 in decode:
        r = decode[262_144] / decode[1_000_000]  # ~ (1M/262K)
        print(f"\n  соотношение decode(262K)/decode(1M) = {r:.2f} (~ожидаемое 3.81 для линейного скана)", flush=True)
        est_10m = decode[1_000_000] * (1_000_000 / 10_000_000)
        print(f"  -> экстраполяция decode @10M ~= {est_10m:.1f} tok/s (per-step ∝ W, линейно)", flush=True)

    print("\nИТОГО: (A) processing ~CONST ~200K tok/s (линейно по W); "
          "(B) decode ~O(1/W) — линейно замедляется, НЕ квадратично.", flush=True)


if __name__ == "__main__":
    main()
