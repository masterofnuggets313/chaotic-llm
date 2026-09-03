"""models_pc.py — ЧИСТЫЙ PC-микшер + Lightweight Address Selection (LAS).

Архитектор: «убираем Арнольда из уравнения, ставим синхронизацию Пекоры–Кэрролла».
НЕТ permute_indices, НЕТ even/odd coupling, НЕТ Arnold.

Динамика:
  1. Хаотическая диссипативная карта: h = h + tanh(h @ W + b) (спектр.радиус W > 1)
  2. PC-синхронизация: h = h + k * (driver - h)

Селекция драйвера (lightweight address, O(W·d)):
  mean  — глобальный пул (базовая линия, разбавляет KEY в 256×)
  last  — контроль: driver = последняя позиция (без селекции)
  top1  — query=последняя позиция, keys=все, cosine, берём 1 позицию argmax
  soft  — softmax(cosine/temp) по позициям, взвешенная сумма
  crt   — КТО-сумма (провалился ранее, оставлен для сравнения)
"""
import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

W = 256  # window size

def count_params(m):
    return sum(p.numel() for p in m.parameters())


class PurePCBlock(nn.Module):
    """Чистый PC-блок: НЕТ Arnold. Хаотическая карта + PC-синхронизация к драйверу."""
    def __init__(self, d, alpha=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d) * 1.5 + torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.alpha = alpha   # сила хаотической динамики (используется!)

    def forward(self, h, driver, k):
        h = h + self.alpha * torch.tanh(h @ self.W + self.b)   # диссипативный хаос
        h = (1 - k) * h + k * driver                           # PC-синхронизация (k∈[0,1])
        return h


class PurePCLM(nn.Module):
    def __init__(self, vocab=512, d=128, layers=4, k_init=1.2, alpha=0.3,
                 sync_steps=1, driver_mode="mean", temp=0.3, primes=(3, 5, 7, 11),
                 W=256):
        super().__init__()
        self.d = d
        self.layers = layers
        self.sync_steps = sync_steps
        self.driver_mode = driver_mode
        self.temp = temp
        self.alpha = alpha
        self.primes = list(primes)
        self.W = W
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, W, d) * 0.02)
        self.blocks = nn.ModuleList([PurePCBlock(d, alpha=alpha) for _ in range(layers)])
        self.k = nn.Parameter(torch.tensor([k_init]))  # обучаемый, sigmoid → [0,1]
        if driver_mode == "crt":
            self.crt_proj = nn.ModuleList([nn.Linear(d, d) for _ in primes])
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, vocab))
        self.readout3 = nn.Sequential(  # для sts_prog (3 входа: h + q0 + g)
            nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, vocab))
        self.last_driver_pos = None  # для анализа распределения выбранных позиций
        self.query_proj = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.topk = 8
        self.nquery = 4  # последние N токенов как multi-query для sts_prog
        self._last_sim = None  # для auxiliary loss (локализация повтора)

    def _address_weights(self, h):
        """Query=последняя позиция, keys=все позиции, cosine.
        Маскируем позицию запроса и ближайшие 8 (самовыбор = тривиальный максимум,
        не даёт найти ДРУГОЙ драйвер)."""
        B = h.shape[0]
        q = h[:, -1, :]                                   # [B, d]
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6)    # [B, W, d]
        sim = (hn * qn.unsqueeze(1)).sum(-1)              # [B, W] cosine
        # маскируем последние 8 позиций (включая саму query) — запрещаем самовыбор
        sim[:, h.shape[1] - 8:] = -1e9
        return sim

    def _select_driver(self, h):
        """Выбор драйвера для readout-позиции (W-1). Записывает позиции для анализа."""
        B = h.shape[0]
        m = self.driver_mode
        if m == "mean":
            driver = h.mean(dim=1, keepdim=True)
            self.last_driver_pos = torch.full((B,), -1, dtype=torch.long, device=h.device)
        elif m == "last":
            driver = h[:, -1:, :]                          # контроль: сам последний токен
            self.last_driver_pos = torch.full((B,), h.shape[1] - 1, dtype=torch.long, device=h.device)
        elif m == "top1":
            sim = self._address_weights(h)                 # [B, W]
            idx = sim.argmax(dim=1)                        # [B]
            self.last_driver_pos = idx
            driver = h[torch.arange(B, device=h.device), idx][:, None, :]
        elif m == "soft":
            sim = self._address_weights(h) / self.temp
            w = torch.softmax(sim, dim=1)                  # [B, W]
            self.last_driver_pos = w.argmax(dim=1)
            driver = (w.unsqueeze(-1) * h).sum(dim=1, keepdim=True)
        elif m == "crt":
            i = h.shape[1] - 1
            buckets = []
            for pi, p in enumerate(self.primes):
                mask = (torch.arange(h.shape[1], device=h.device) % p) == (i % p)
                sel = h[:, mask]
                b = self.crt_proj[pi](sel).sum(dim=1, keepdim=True)
                buckets.append(b)
            driver = torch.cat(buckets, dim=1).mean(dim=1, keepdim=True)
            self.last_driver_pos = torch.full((B,), -2, dtype=torch.long, device=h.device)
        else:
            raise ValueError(f"unknown driver_mode {m}")
        return driver

    def forward(self, x, return_aux=False):
        e = self.embed(x) + self.pos          # сырые эмбеддинги — идентичность жива
        Bn = e.shape[0]
        Wc = x.shape[1]                        # фактическая длина окна (для scaling)
        k_eff = torch.sigmoid(self.k)         # обучаемый, всегда в [0,1] — стягивание
        # ============ STS-MQ: multi-query single-shot ============
        if self.driver_mode == "sts_mq":
            q = e[:, -self.nquery:, :].mean(dim=1)            # [B, d] query из 4 токенов
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)              # селекция на СЫРЫХ ключах
            sim[:, Wc - 8:] = -1e9
            kk = min(self.topk, Wc - 8)
            top_w, top_i = torch.topk(sim, kk, dim=1)
            w = torch.softmax(top_w / self.temp, dim=1)
            top_next = torch.clamp(top_i + 1, 0, Wc - 2)
            idx = torch.arange(Bn, device=e.device).unsqueeze(1).expand(Bn, kk)
            neigh = e[idx, top_next]                          # соседи из СЫРЫХ (B жива)
            self._sts_driver = (w.unsqueeze(-1) * neigh).sum(dim=1, keepdim=True)
            self.last_driver_pos = top_next[:, 0]
            self._last_sim = sim
            h = e
            for blk in self.blocks:
                h = blk(h, h.mean(dim=1, keepdim=True), k_eff)   # хаотическая динамика
            driver = self._sts_driver
            k = torch.clamp(self.k, 0.0, 2.0)
            h_sync = h[:, -1:, :]
            for _ in range(self.sync_steps):
                h_sync = h_sync + k * (driver - h_sync)
            h_last = h_sync[:, -1, :]
            g = h.mean(dim=1)
            logits = self.readout(torch.cat([h_last, g], dim=-1))
            return logits
        # ============ STS-PROG: прогрессивное уточнение селекции ============
        if self.driver_mode in ("sts_prog", "sts_prog_nopc"):
            q0 = e[:, -self.nquery:, :].mean(dim=1)            # [B, d] raw query
            q = q0
            en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)     # keys СЫРЫЕ (идентичность жива)
            h = e
            for li, blk in enumerate(self.blocks):
                qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
                sim = (en * qn.unsqueeze(1)).sum(-1)           # селекция на СЫРЫХ ключах
                sim[:, Wc - 8:] = -1e9
                kk = min(self.topk, Wc - 8)
                top_w, top_i = torch.topk(sim, kk, dim=1)
                w = torch.softmax(top_w / self.temp, dim=1)
                top_next = torch.clamp(top_i + 1, 0, Wc - 2)
                idx = torch.arange(Bn, device=e.device).unsqueeze(1).expand(Bn, kk)
                neigh = e[idx, top_next]                       # соседи из СЫРЫХ (B жива)
                driver = (w.unsqueeze(-1) * neigh).sum(dim=1, keepdim=True)
                self.last_driver_pos = top_next[:, 0]
                if self.driver_mode == "sts_prog_nopc":
                    h = h + driver * 0.0                       # абляция: без хаоса (identity)
                else:
                    h = blk(h, driver, k_eff)                         # PC-синхронизация
                q = q0 + self.query_proj(h[:, -1, :]) * 0.5
            self._last_sim = sim  # для aux loss
            h_last = h[:, -1, :]
            g = h.mean(dim=1)
            logits = self.readout3(torch.cat([h_last, q0, g], dim=-1))
            return logits
        # ============ SELECT-THEN-SYNC: селекция на сырых ============
        if self.driver_mode in ("sts_emb", "sts_h", "sts_lq", "sts_lqk"):
            Bn = e.shape[0]
            if self.driver_mode in ("sts_lq", "sts_lqk"):
                # learned query: проекция последнего токена (план п.1)
                q = self.query_proj(e[:, -1, :])
            else:
                q = e[:, -1, :]
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            en = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (en * qn.unsqueeze(1)).sum(-1)
            sim[:, Wc - 8:] = -1e9              # маска самовыбора
            if self.driver_mode in ("sts_lq", "sts_lqk"):
                # soft top-k (план п.3): взвешенная сумма соседей лучших k позиций
                kk = min(self.topk, Wc - 8)
                top_w, top_i = torch.topk(sim, kk, dim=1)     # [B, k]
                w = torch.softmax(top_w / self.temp, dim=1)   # [B, k]
                top_next = torch.clamp(top_i + 1, 0, Wc - 2)   # соседи (B)
                idx = torch.arange(Bn, device=e.device).unsqueeze(1).expand(Bn, kk)
                neigh = e[idx, top_next]                       # [B, k, d]
                self.last_driver_pos = top_next[:, 0]
                self._sts_driver = (w.unsqueeze(-1) * neigh).sum(dim=1, keepdim=True)
            else:
                sel = sim.argmax(dim=1)            # [B] похожая позиция (KEY)
                sel_next = torch.clamp(sel + 1, 0, Wc - 2)  # сосед KEY (содержит B)
                self.last_driver_pos = sel_next
            self._last_sim = sim  # для auxiliary loss (локализация повтора)
            # хаотическая динамика (смысл)
            h = e
            for blk in self.blocks:
                h = blk(h, h.mean(dim=1, keepdim=True), k_eff)
            if self.driver_mode in ("sts_emb", "sts_h"):
                if self.driver_mode == "sts_emb":
                    self._sts_driver = e[torch.arange(Bn, device=e.device), sel_next][:, None, :]
                else:
                    self._sts_driver = h[torch.arange(Bn, device=h.device), sel_next][:, None, :]
            driver = self._sts_driver
            k = torch.clamp(self.k, 0.0, 2.0)
            h_sync = h[:, -1:, :]
            for _ in range(self.sync_steps):
                h_sync = h_sync + k * (driver - h_sync)
            h_last = h_sync[:, -1, :]
            g = h.mean(dim=1)
            logits = self.readout(torch.cat([h_last, g], dim=-1))
            if return_aux:
                return logits, h_sync
            return logits

        # обычный путь: динамика + селекция после
        h = e
        for blk in self.blocks:
            driver = self._select_driver(h)
            h = blk(h, driver, k_eff)
        driver = self._select_driver(h)
        k = torch.clamp(self.k, 0.0, 2.0)
        h_sync = h[:, -1:, :]
        for _ in range(self.sync_steps):
            h_sync = h_sync + k * (driver - h_sync)
        h_last = h_sync[:, -1, :]
        g = h.mean(dim=1)
        logits = self.readout(torch.cat([h_last, g], dim=-1))
        if return_aux:
            return logits, h_sync
        return logits


def build_pc_model(config, vocab=512, d=128, alpha=0.3, k_init=1.2,
                   sync_steps=1, driver_mode="mean", temp=0.3, layers=4):
    if config == "pc":
        return PurePCLM(vocab=vocab, d=d, layers=layers, k_init=k_init, alpha=alpha,
                        sync_steps=sync_steps, driver_mode=driver_mode, temp=temp)
    raise ValueError(f"unknown {config}")


if __name__ == "__main__":
    for mode in ["mean", "last", "top1", "soft", "crt"]:
        m = build_pc_model("pc", driver_mode=mode)
        x = torch.randint(0, 512, (2, W))
        y = m(x)
        pos = m.last_driver_pos
        print(f"{mode}: params={count_params(m):,} out={tuple(y.shape)} last_driver_pos={pos.tolist()}")
