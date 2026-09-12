"""parametric_models.py — scalable ChaoticLLM mixer + Transformer baseline.

Оба класса имеют одинаковый интерфейс (forward(x) -> (N,V) logits),
чтобы сравнивать их на одном протоколе.

Масштабирование:
  ChaoticMixer:  параметры ~ LAYERS * BLOCKS_PER_LAYER * (4 * D^2 * 2 + 2D)
                 (каждый ChaoticBlock = 2 permutation-Net (2*D*D) + 2 projection (D))
  Transformer:   параметры ~ LAYERS * (12 D^2 + 8 D^2) + V*D + W*D  (d=128 style)

Конфиги (цель 50-100M):
  50M:  D=256, BLOCKS_PER_LAYER=4,  LAYERS=6   (chaos)  / D=256, LAYERS=8, HEADS=8 (tf)
  100M: D=512, BLOCKS_PER_LAYER=4,  LAYERS=6   (chaos)  / D=512, LAYERS=8, HEADS=8 (tf)
"""
import math
import torch
import torch.nn as nn
import numpy as np

# ---- chaos primitives (from chaos_lib, inlined to avoid path issues) ----
A_MATRIX = np.array([[1, 1], [1, 2]], dtype=np.int64)
A_INV = np.array([[2, -1], [-1, 1]], dtype=np.int64)


def _nearest_square(n):
    r = int(math.ceil(math.sqrt(n)))
    return r * r


def _permute_indices(n_tokens, t):
    N = int(math.sqrt(_nearest_square(n_tokens)))
    N = max(1, N)
    idx = np.arange(n_tokens, dtype=np.int64)
    pos = np.stack([idx // N, idx % N], axis=-1)
    x, y = pos[..., 0], pos[..., 1]
    for _ in range(t):
        nx = (2 * x - y) % N
        ny = (-x + y) % N
        x, y = nx, ny
    flat = x * N + y
    return flat.astype(np.int64)


def _schedule(n_tokens, step, perm_buf):
    """Permutation for a given step; cache in perm_buf to avoid recompute."""
    if step not in perm_buf:
        perm_buf[step] = _permute_indices(n_tokens, step + 1)
    return perm_buf[step]


class ChaoticBlock(nn.Module):
    """One chaotic mixing block: backward perm + Net + forward perm + Net.
    Mirrors exp52 BidirectionalMixer structure but parameterised by D."""
    def __init__(self, D, n_tokens):
        super().__init__()
        self.D = D
        self.n_tokens = n_tokens
        # permutation networks (value mixing)
        self.net_b = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        self.net_f = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        self.perm_buf = {}

    def _permute(self, X, step):
        sigma = _schedule(self.n_tokens, step, self.perm_buf)
        if X.device.type == 'cuda':
            sigma = torch.as_tensor(sigma, device=X.device)
        else:
            sigma = torch.as_tensor(sigma)
        return X[:, sigma, :]

    def forward(self, X):
        n = X.shape[1]
        # expand to even for coupling split
        if n % 2 == 1:
            X = torch.cat([X, X[:, -1:, :]], dim=1)
        half = X.shape[1] // 2
        # BACKWARD branch
        X_b = self._permute(X, 1)
        src_b, dst_b = X_b[:, :half], X_b[:, half:]
        mixed_b = self.net_b(torch.cat([src_b, dst_b], dim=-1)) + dst_b
        X_b2 = torch.cat([src_b, mixed_b], dim=1)
        # FORWARD branch
        X_f = self._permute(X, 0)
        src_f, dst_f = X_f[:, :half], X_f[:, half:]
        mixed_f = self.net_f(torch.cat([src_f, dst_f], dim=-1)) + dst_f
        X_f2 = torch.cat([src_f, mixed_f], dim=1)
        # recombine (average of two branches)
        return 0.5 * (X_b2 + X_f2)


class BidirectionalMixer(nn.Module):
    """Stack of ChaoticBlocks (bidirectional mixing) with pre-norm + residual
    for deep stability (32+ layers would otherwise explode)."""
    def __init__(self, D=128, BLOCKS=4, LAYERS=1, n_tokens=256):
        super().__init__()
        self.D = D
        self.n_tokens = n_tokens
        self.blocks = nn.ModuleList([
            ChaoticBlock(D, n_tokens) for _ in range(BLOCKS * LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(BLOCKS * LAYERS)])

    def forward(self, X):
        for blk, norm in zip(self.blocks, self.norms):
            X = X + blk(norm(X))
        return X


class ChaoticMixerLM(nn.Module):
    """Full LM: embed + pos + stacked BidirectionalMixer + head.
    Interface matches TinyGPT (forward(x) -> (N,V) logits)."""
    def __init__(self, V, W, D=256, BLOCKS_PER_LAYER=4, LAYERS=6):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.mixer = BidirectionalMixer(D, BLOCKS_PER_LAYER, LAYERS, n_tokens=W)
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.mixer(h)
        h = self.ln_f(h)
        return self.head(h[:, -1, :])


# ---------------------------------------------------------------------------
# Transformer baseline (scaled TinyGPT from exp45)
# ---------------------------------------------------------------------------
class TFBlock(nn.Module):
    def __init__(self, d, heads, W):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        mask = torch.triu(torch.full((W, W), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=self.mask)
        x = x + a
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, V, W, D=256, HEADS=8, LAYERS=8):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, W, D) * 0.02)
        self.blocks = nn.ModuleList([TFBlock(D, HEADS, W) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h[:, -1, :])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_matched_pair(budget_m, V=512, W=256, heads=8, tf_layers=12, chaos_layers=12):
    """Build (chaos_model, tf_model) with ~equal param count near budget_m (millions).
    Fixed depth=12; width D chosen from precomputed table to hit the budget."""
    # precomputed D for budget (from sweep): chaos/tf D for 50M and 100M
    table = {
        50: (384, 576),   # (chaos_D, tf_D)
        100: (576, 832),
    }
    cd, td = table.get(budget_m, (384, 576))
    chaos_model = ChaoticMixerLM(V, W, D=cd, BLOCKS_PER_LAYER=4, LAYERS=chaos_layers)
    tf_model = TransformerLM(V, W, D=td, HEADS=heads, LAYERS=tf_layers)
    return chaos_model, tf_model, cd, td


if __name__ == "__main__":
    for budget in [50, 100]:
        cm, tm, d_c, d_t = build_matched_pair(budget)
        print(f"[{budget}M budget] chaos D={d_c} ({count_params(cm):,}) | tf D={d_t} L=12 ({count_params(tm):,})")
