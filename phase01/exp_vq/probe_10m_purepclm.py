"""probe_10m_purepclm.py — proof-of-concept: влезает ли STS-Prog в 12GB при 10M контекста.

Два рычага из разговора:
  (1) ПРОИЗВОДНЫЕ позиции вместо self.pos=(1,W,d): позиция вычисляется из индекса
      (синусоида + маленький learned проектор O(d_pe*d)), параметры НЕ зависят от W.
  (2) ЧАНКОВЫЙ форвард: последовательность гонится кусками C; top-k выборка и обновление
      блоков — построчно по чанкам, без материализации лишних (W,d)-тензоров.

Честно: скрытый слой h=(W,d) неизбежен — это пол любой модели, держащей все позиции.
Значит память ~ O(W*d): при d=192 и W=10M это ~7.7GB (один (W,d) тензор) -> ВЛЕЗАЕТ в 12GB.
При широких моделях (d=2046, 100M) тот же (W,d) -> 82GB -> НЕ влезает (нужна меньшая d
или факторизация). Probe это покажет.

Один forward-проход, fp32 (консервативно), пик VRAM через torch.cuda.max_memory_allocated.
"""
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
sys.path.insert(0, PHASE)
from models_pc import build_pc_model

D_PE = 32  # размер позиционного признака (независим от W)


def sinusoid(idx, d_pe):
    # idx: (C,) long -> (C, d_pe) синусоидальных признаков
    half = d_pe // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=idx.device) / half))
    ang = idx.float().unsqueeze(1) * freqs.unsqueeze(0)        # (C, half)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (C, d_pe)


def chunked_forward(base, x, C=4096, temp=0.3):
    """Версия STS-Prog форварда: производные позиции + чанковая top-k выборка.
    Использует подмодули реального PurePCLM (embed, blocks, readout3, query_proj)."""
    B, W = x.shape
    d = base.d
    k_eff = torch.sigmoid(base.k)
    nq = base.nquery
    blocks = base.blocks
    L = len(blocks)
    pe_proj = base.pe_proj  # добавлен снаружи (O(d_pe*d) параметров)
    nchunks = math.ceil(W / C)

    # начальный скрытый слой h_chunks = сырые ключи e (по чанкам, без полного (W,d))
    h_chunks = []
    for ci in range(nchunks):
        idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
        xc = x[:, idx]
        ec = base.embed(xc) + pe_proj(sinusoid(idx, D_PE).to(x.device))
        h_chunks.append(ec)

    q0 = h_chunks[-1][:, -nq:, :].mean(dim=1)  # raw query из последних nq токенов
    q = q0

    for li in range(L):
        # --- top-k выборка по ВСЕМ позициям, стриминг чанками (running top-k) ---
        run_val = torch.full((B, base.topk), -1e9, device=x.device)
        run_gidx = torch.zeros((B, base.topk), dtype=torch.long, device=x.device)
        run_key = torch.zeros((B, base.topk, d), device=x.device)
        for ci in range(nchunks):
            idx = torch.arange(ci * C, min((ci + 1) * C, W), device=x.device)
            xc = x[:, idx]
            ec = base.embed(xc) + pe_proj(sinusoid(idx, D_PE).to(x.device))  # (B, C, d)
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            en = ec / (ec.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)              # (B, C)
            sim[:, idx >= W - 8] = -1e9                        # маска самовыбора
            ks = min(base.topk, ec.shape[1])
            cs, cloc = sim.topk(ks, dim=1)                    # (B, ks)
            cg = ci * C + cloc                                # глобальные индексы (B, ks)
            ckey = torch.gather(ec, 1, cloc.unsqueeze(-1).expand(-1, -1, d))  # (B, ks, d)
            allv = torch.cat([run_val, cs], 1)
            allg = torch.cat([run_gidx, cg], 1)
            allk = torch.cat([run_key, ckey], 1)
            _, ordr = allv.topk(base.topk, dim=1)
            run_val = allv.gather(1, ordr)
            run_gidx = allg.gather(1, ordr)
            run_key = allk.gather(1, ordr.unsqueeze(-1).expand(-1, -1, d))
        w = torch.softmax(run_val / temp, dim=1)
        driver = (w.unsqueeze(-1) * run_key).sum(dim=1, keepdim=True)  # (B, 1, d)

        # --- обновление h построчно по чанкам, НА МЕСТЕ (без удвоения (W,d)) ---
        for ci in range(nchunks):
            h_chunks[ci] = blocks[li](h_chunks[ci], driver, k_eff)

        # рефайн q для следующего слоя
        h_last = h_chunks[-1][:, -1, :]
        q = q0 + base.query_proj(h_last) * 0.5

    # --- readout: среднее по всем позициям, СТРИМИНГ (без torch.stack -> без удвоения (W,d)) ---
    g_sum = torch.zeros((B, d), device=x.device)
    for hc in h_chunks:
        g_sum += hc.sum(dim=1)
    g = g_sum / W
    h_last = h_chunks[-1][:, -1, :]
    logits = base.readout3(torch.cat([h_last, q0, g], dim=-1))
    return logits


def main():
    V = 512
    d = 192
    L = 8
    torch.manual_seed(0)
    base = build_pc_model("pc", vocab=V, d=d, layers=L, k_init=1.2,
                          sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3).cuda()
    # производная позиционная кодировка (O(d_pe*d) параметров, НЕ зависит от W)
    base.pe_proj = nn.Linear(D_PE, d).cuda()
    nparam = sum(p.numel() for p in base.parameters())
    print(f"STS-Prog (chunked, derived-pos): d={d} L={L} params={nparam:,} "
          f"(без O(W) позиционного параметра)", flush=True)

    for W in [262_144, 1_000_000, 10_000_000]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, V, (1, W), device="cuda")
        t0 = time.time()
        with torch.no_grad():
            _ = chunked_forward(base, x, C=4096)
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  W={W:,} : peak VRAM={vram:,.1f} MB ({vram/1024:.2f} GB)  "
              f"tok/s={W/dt:,.0f}  time={dt:.2f}s", flush=True)
        del x
        torch.cuda.empty_cache()

    # честная проверка широкой модели: d=600 при 10M -> (W,d) уже ~24GB, должно не влезть
    print("\n-- контраст: широкая модель d=600 при 10M (ожидаем OOM / >12GB) --", flush=True)
    try:
        base2 = build_pc_model("pc", vocab=V, d=600, layers=L, k_init=1.2,
                               sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=0.3).cuda()
        base2.pe_proj = nn.Linear(D_PE, 600).cuda()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, V, (1, 10_000_000), device="cuda")
        with torch.no_grad():
            _ = chunked_forward(base2, x, C=4096)
        torch.cuda.synchronize()
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  d=600 W=10M : peak VRAM={vram:,.1f} MB ({vram/1024:.2f} GB)", flush=True)
    except RuntimeError as e:
        print(f"  d=600 W=10M : OOM/error -> {str(e)[:80]}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import time
    main()
