"""models_pc_v3.py — STS-Prog v3 Production Architecture.

Продакшн-реализация не-attention архитектуры STS-Prog с генеративной памятью:
1. Multi-Head Sparse Retrieval: многоголовый поиск совпадений по подпространствам.
2. Channel-Wise Vector Gating: векторный гейтинг p-c синхронизации (отдельный баланс для каждого канала).
3. SwiGLU FFN (Channel Mixer): блок емкого хранения фактологических знаний.
4. RMSNorm: числовая стабильность в глубоких сетях.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    """Channel Mixer: SwiGLU FFN для фактологической емкости без увеличения памяти контекста."""
    def __init__(self, d, multiplier=2.6667):
        super().__init__()
        hidden = int(d * multiplier)
        self.up = nn.Linear(d, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class MultiHeadSparseRetrieval(nn.Module):
    """Multi-Head Sparse Retrieval по сырым ключам.

    Разбивает вектор поиска на H голов. Каждая голова ищет свои top-k совпадений
    в своём подпространстве (синтаксические, семантические, логические связи).
    """
    def __init__(self, d, num_heads=4, topk=8, temp=0.3):
        super().__init__()
        assert d % num_heads == 0, f"d={d} должно делиться на num_heads={num_heads}"
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.topk = topk
        self.temp = temp

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, raw_e, q_current, length, pad_mask=None):
        """
        raw_e: [B, W, D] — сырые эмбеддинги (идентичность токенов)
        q_current: [B, D] — текущая query
        length: фактическая длина текущего контекста W
        pad_mask: [B, W] bool tensor, True для падинг токенов
        """
        B, W, D = raw_e.shape
        H, Dh = self.num_heads, self.head_dim

        # Проекция на H голов
        q_h = self.q_proj(q_current).view(B, H, Dh)                          # [B, H, Dh]
        k_h = self.k_proj(raw_e).view(B, W, H, Dh).transpose(1, 2)            # [B, H, W, Dh]
        v_h = self.v_proj(raw_e).view(B, W, H, Dh).transpose(1, 2)            # [B, H, W, Dh]

        # Нормализация для косинусной близости по головам
        q_norm = F.normalize(q_h, dim=-1)                                    # [B, H, Dh]
        k_norm = F.normalize(k_h, dim=-1)                                    # [B, H, W, Dh]
        sim = (k_norm * q_norm.unsqueeze(2)).sum(dim=-1)                      # [B, H, W]

        # Маскирование падинга
        if pad_mask is not None:
            sim = sim.masked_fill(pad_mask.unsqueeze(1), -1e9)

        # Маскирование хвоста (8 позиций) для предотвращения тривиального самовыбора
        sim[:, :, length - 8:] = -1e9

        k = min(self.topk, max(1, length - 8))
        top_val, top_idx = sim.topk(k, dim=-1)                               # [B, H, K]
        weights = torch.softmax(top_val / self.temp, dim=-1)                 # [B, H, K]

        # Следующий токен (top_idx + 1) — ассоциированное значение A -> B
        next_idx = (top_idx + 1).clamp_max(length - 2)                       # [B, H, K]

        # Сборка значений по индексам
        batch_indices = torch.arange(B, device=raw_e.device).view(B, 1, 1).expand(B, H, k)
        head_indices = torch.arange(H, device=raw_e.device).view(1, H, 1).expand(B, H, k)
        selected_v = v_h[batch_indices, head_indices, next_idx]               # [B, H, K, Dh]

        # Взвешенное суммирование голов
        driver_h = (weights.unsqueeze(-1) * selected_v).sum(dim=2)            # [B, H, Dh]
        driver = self.out_proj(driver_h.reshape(B, D)).unsqueeze(1)           # [B, 1, D]
        return driver, top_idx


class VectorGatedPCBlock(nn.Module):
    """Блок STS-Prog v3: Диссипативный хаос + Векторный PC-гейтинг + SwiGLU FFN."""
    def __init__(self, d, num_heads=4, topk=8, alpha=0.3, temp=0.3, ffn_multiplier=2.6667):
        super().__init__()
        self.alpha = alpha
        self.norm_dyn = RMSNorm(d)
        self.dynamics = nn.Linear(d, d)
        nn.init.eye_(self.dynamics.weight)
        with torch.no_grad():
            self.dynamics.weight.mul_(1.5)
        nn.init.zeros_(self.dynamics.bias)

        self.retrieval = MultiHeadSparseRetrieval(d, num_heads=num_heads, topk=topk, temp=temp)

        self.norm_gate = RMSNorm(d)
        self.gate_h = nn.Linear(d, d, bias=False)
        self.gate_driver = nn.Linear(d, d, bias=True)
        nn.init.zeros_(self.gate_h.weight)
        nn.init.zeros_(self.gate_driver.weight)
        nn.init.constant_(self.gate_driver.bias, -1.0)  # Старт с умеренного гейтинга

        self.norm_ffn = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, multiplier=ffn_multiplier)

    def forward(self, h, raw_e, q_current, length):
        # 1. Хаотическая динамика (Sequence Mixer)
        z = h + self.alpha * torch.tanh(self.dynamics(self.norm_dyn(h)))

        # 2. Многоголовая селекция драйвера
        driver, top_idx = self.retrieval(raw_e, q_current, length)

        # 3. Покомпонентный векторный PC-гейтинг (Channel-wise Gated Sync)
        gate = torch.sigmoid(self.gate_h(self.norm_gate(z)) + self.gate_driver(driver))
        h_sync = z + gate * (driver - z)

        # 4. SwiGLU FFN (Channel Mixer)
        out = h_sync + self.ffn(self.norm_ffn(h_sync))
        return out, top_idx


class PCSTSLMv3(nn.Module):
    """Продакшн-модель STS-Prog v3.

    Объединяет Multi-Head Sparse Retrieval, Vector-Gated PC Chaos и SwiGLU FFN.
    Сохраняет тождество h = G(e) и полную совместимость с 28B/pos Fracode Memory.
    """
    def __init__(self, vocab=2048, d=256, layers=8, window=512, num_heads=4, topk=8,
                 nquery=4, alpha=0.3, temp=0.3, ffn_multiplier=2.6667):
        super().__init__()
        if window <= 8:
            raise ValueError("Размер окна должен быть больше 8 (маска самовыбора)")
        self.d = d
        self.layers = layers
        self.window = window
        self.topk = topk
        self.nquery = nquery

        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, window, d) * 0.02)
        self.query_norm = RMSNorm(d)
        self.query_proj = nn.Linear(d, d, bias=False)

        self.blocks = nn.ModuleList([
            VectorGatedPCBlock(d, num_heads=num_heads, topk=topk, alpha=alpha,
                               temp=temp, ffn_multiplier=ffn_multiplier)
            for _ in range(layers)
        ])

        self.readout_norm = RMSNorm(2 * d)
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, vocab)
        )
        self._last_selection = None

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] > self.window:
            raise ValueError(f"Ожидался вход [batch, length <= {self.window}], получено {tuple(x.shape)}")
        length = x.shape[1]
        if length <= 8:
            raise ValueError("Длина входа должна превосходить маскируемый хвост в 8 токенов")

        # Сырые ключи (идентичность токенов не искажена)
        e = self.embed(x) + self.pos[:, :length]
        h = e
        q0 = e[:, -self.nquery:].mean(dim=1)
        q = q0

        last_indices = None
        for block in self.blocks:
            h, top_idx = block(h, e, q, length)
            q = q0 + self.query_proj(self.query_norm(h[:, -1])) * 0.5
            last_indices = top_idx

        self._last_selection = {"indices": last_indices.detach()}

        features = torch.cat([h[:, -1], q0], dim=-1)
        return self.readout(self.readout_norm(features))


def build_pc_v3_model(vocab=2048, d=256, layers=8, window=512, num_heads=4, topk=8):
    return PCSTSLMv3(vocab=vocab, d=d, layers=layers, window=window, num_heads=num_heads, topk=topk)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_pc_v3_model(vocab=512, d=192, layers=4, window=256, num_heads=4)
    x = torch.randint(0, 512, (2, 64))
    y = m(x)
    print(f"STS-Prog v3: params={count_params(m):,} out_shape={tuple(y.shape)}")
