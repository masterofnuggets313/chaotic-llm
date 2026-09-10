# -*- coding: utf-8 -*-
"""STS-Prog v4 — Этап 1 по рецензии коллеги (2026-09-05, день).

Чеклист §7 рецензии — что изменилось против v3.3:
  1. DENSE causal loss: логиты по ВСЕМ позициям хвоста [B, L, V], не одна
     цель на пример. (Коллега: 99.6% FLOPs раньше шло в мусор.)
  2. Retrieval каузальный: кандидаты позиции i = все валидные слоты +
     локальные j<i. Гэзер value: j+1 для локальных (индукция), j для слотов
     (преемник уже подставлен ADC). val_idx = where(is_local, j+1, j).
  3. Ключи e = embed(x) + маркер — БЕЗ pos (и для слотов, и для локальных).
     Позиция живёт только в h.
  4. Слоты: seg-маркер hop ∈ {empty,1,2,3} (nn.Embedding(4,d)) вместо pos;
     slot_valid = hop>0 участвует и в retrieval, и в пуле ридаута.
  5. PC-резидуальность ВЕРНУТА: z = h + alpha*tanh(dynamics(norm(h))),
     alpha learnable, init 0.1 (в v3.1→v3.3 z была мёртвым кодом — регрессия).
  6. Readout попозиционный: cat(h[i], attention_pool(h[:, :K], q=h[i],
     mask=slot_valid)) → MLP → [B, L, V].
  7. ADC-запрос на обучении = ids[start-8:start] (ДО хвоста) — утечки нет;
     start = граница <bot>: (правило в тренере).
  8. Copy-задача УДАЛЕНА (заменится синт. сессиями Этапа 2).

Конфиги гейта: --no-slots (A: чистый локальный LM) vs полный (B).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_pc_v3 import RMSNorm


class HistoryStore:
    """Фракод-стор: token ids (~11 бит) + позиция (~17 бит) = ~28 бит/позиция.
    Честно: 4 БАЙТА/позиция (RQ-коды v4 не использует — см. NOT_WIN_YET §6.1)."""

    def __init__(self):
        self.tokens = []

    def append(self, token_ids):
        self.tokens.extend(int(t) for t in token_ids)

    def __len__(self):
        return len(self.tokens)


class PCBlockV4(nn.Module):
    """PC-блок v4: резидуальная хаос-динамика + каузальный multi-hop retrieval
    c induction-сдвигом для локальных кандидатов."""

    def __init__(self, d, num_heads=4, topk=4, alpha_init=0.1, alpha_max=None):
        super().__init__()
        self.d = d
        self.alpha_max = alpha_max   # v5-gate: клэмп alpha (напр. 1.0); None = без клэмпа
        self.H = num_heads
        self.topk = topk
        self.Dh = d // num_heads
        self.norm_dyn = nn.RMSNorm(d)
        self.dynamics = nn.Linear(d, d)          # diag 1.5*I — в init_weights
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.retrieval = nn.ModuleDict({
            "q_proj": nn.Linear(d, d, bias=False),
            "k_proj": nn.Linear(d, d, bias=False),
            "v_proj": nn.Linear(d, d, bias=False),
            "out_proj": nn.Linear(d, d, bias=False),
        })
        self.norm_gate = nn.RMSNorm(d)
        self.gate_h = nn.Linear(d, d)
        self.gate_driver = nn.Linear(d, d)
        self.norm_ffn = nn.RMSNorm(d)
        self.ffn = nn.ModuleDict({
            "up": nn.Linear(d, int(d * 2.6667)),
            "down": nn.Linear(int(d * 2.6667), d),
        })

    def forward(self, h, e, allowed):
        """h,e: [B,W,D]. allowed: [B,1,W,W] bool — можно ли позиции i смотреть
        на кандидата j (слоты валидные + локальные j<i)."""
        B, W, D = h.shape
        # 1) PC-хаос динамика — РЕЗИДУАЛЬНО (возврат из v3; alpha малый, learnable)
        a = self.alpha if self.alpha_max is None else self.alpha.clamp(max=self.alpha_max)
        z = h + a * torch.tanh(self.dynamics(self.norm_dyn(h)))
        # 2) каузальный retrieval (raw e — ключи, без pos)
        q = self.retrieval["q_proj"](z)
        k = self.retrieval["k_proj"](e)
        v = self.retrieval["v_proj"](e)
        qh = q.view(B, W, self.H, self.Dh).permute(0, 2, 1, 3)
        kh = k.view(B, W, self.H, self.Dh).permute(0, 2, 1, 3)
        vh = v.view(B, W, self.H, self.Dh).permute(0, 2, 1, 3)
        scores = torch.einsum("bhid,bhjd->bhij", F.normalize(qh, dim=-1),
                              F.normalize(kh, dim=-1))
        # -1e9 (не -inf): при пустой истории topk берёт fillers с весом ~0,
        # softmax не даёт NaN
        scores = scores.masked_fill(~allowed, -1e9)
        k_eff = min(self.topk, W)
        top_scores, top_idx = scores.topk(k_eff, dim=-1)
        # induction-сдвиг: локальный кандидат j → value j+1; слот → value j
        K = self._num_slots
        is_local_c = torch.arange(W, device=h.device) >= K          # [W]
        val_idx = torch.where(is_local_c,
                              torch.arange(W, device=h.device) + 1,
                              torch.arange(W, device=h.device)).clamp(max=W - 1)
        sel_idx = val_idx[top_idx]                                   # [B,H,W,k]
        b_idx = torch.arange(B, device=h.device).view(B, 1, 1, 1)
        h_idx = torch.arange(self.H, device=h.device).view(1, self.H, 1, 1)
        sel_v = vh[b_idx, h_idx, sel_idx]                            # [B,H,W,k,Dh]
        weights = top_scores.softmax(dim=-1).unsqueeze(-1)
        mixed = (sel_v * weights).sum(dim=3)
        driver = self.retrieval["out_proj"](mixed.permute(0, 2, 1, 3).reshape(B, W, D))
        # 3) векторный гейт
        gate = torch.sigmoid(self.gate_h(self.norm_gate(h)) + self.gate_driver(self.norm_gate(driver)))
        h = h + gate * (driver - h)
        # 4) SwiGLU FFN
        h = h + self.ffn["down"](F.silu(self.ffn["up"](self.norm_ffn(h))))
        return h


class PCSTSLMv4(nn.Module):
    """v4: окно = [K слотов (hop-маркеры, без pos) | L локальных (pos есть)].
    forward → h; logits_local(h, slot_valid) → [B, L, V] dense-логиты хвоста."""

    def __init__(self, vocab=8192, d=256, layers=8, window=256, num_heads=4,
                 topk=4, num_slots=64, alpha_max=None):
        super().__init__()
        assert window > num_slots
        self.vocab, self.d, self.layers = vocab, d, layers
        self.window = window
        self.num_slots = num_slots
        self.local_len = window - num_slots

        self.embed = nn.Embedding(vocab, d)
        self.pos_local = nn.Parameter(torch.randn(1, self.local_len, d) * 0.02)
        self.local_marker = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.hop_emb = nn.Embedding(4, d)          # 0=empty,1,2,3 — вместо pos у слотов

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            b = PCBlockV4(d, num_heads=num_heads, topk=topk, alpha_max=alpha_max)
            b._num_slots = num_slots
            self.blocks.append(b)
        # dense causal mask (буфер): allowed[i,j] = слот j (всегда, валидность
        # добавляется снаружи) ИЛИ локальный j<i
        i = torch.arange(window).view(-1, 1)
        j = torch.arange(window).view(1, -1)
        local_j = j >= num_slots
        causal = torch.where(local_j, j < i, torch.ones_like(j < i))
        self.register_buffer("causal_base", causal.view(1, 1, window, window))

        self.norm_out = nn.RMSNorm(2 * d)
        self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, vocab))
        self.pool_tau = nn.Parameter(torch.tensor(1.0))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def init_weights(self):
        for b in self.blocks:
            nn.init.eye_(b.dynamics.weight)
            with torch.no_grad():
                b.dynamics.weight.mul_(1.5)
            nn.init.zeros_(b.dynamics.bias)

    # ---------- WindowManager (Фракод) ----------
    @torch.no_grad()
    def adc_slots(self, store_tokens, query_ids, k=None, hops=3,
                  bigram=False, center=True):
        """v4: запрос = query_ids (хвост ДО start), ответ = преемники, цепочкой
        по hops. Ключи: центрированные эмбеддинги (снимает анизотропию),
        опц. биграммы. Возвращает (slot_ids [K], hop_ids [K])."""
        K = self.num_slots or k
        k = k or K
        dev = self.embed.weight.device
        emb = self.embed.weight
        if center:
            emb = emb - emb.mean(0, keepdim=True)
        store = list(store_tokens)
        query = list(query_ids)[-8:]
        if not store or not query:
            return torch.zeros(K, dtype=torch.long, device=dev), torch.zeros(K, dtype=torch.long, device=dev)
        st = torch.tensor(store, device=dev)
        qt = torch.tensor(query, device=dev)
        ks = F.normalize(emb[st], dim=-1)
        qv = F.normalize(emb[qt], dim=-1)
        if bigram and len(store) > 1:
            kb = F.normalize(emb[st] + torch.cat([emb[st[:1]], emb[st[:-1]]]), dim=-1)
            qb_prev = torch.cat([qt[:1], qt[:-1]])
            qb = F.normalize(emb[qt] + emb[qb_prev], dim=-1)
        slots, hop_ids = [], []
        cur_q = qv
        used = set()
        for hop in range(1, hops + 1):
            if bigram and hop == 1 and len(store) > 1:
                s = qb @ kb.T
            else:
                s = cur_q @ ks.T
            best = s.max(dim=0).values if hop == 1 else s.max(dim=1).values
            match_pos = (s.max(dim=0 if hop == 1 else 1).indices)
            # совпадение на ПОСЛЕДНЕМ токене стора преемника не имеет — исключаем
            succ_pos = match_pos + 1
            kk = max(1, k // (2 ** (hop - 1))) if hop < 3 else max(1, k - len(slots))
            order = best.topk(min(kk, len(best))).indices
            take_pos = succ_pos[order]
            # дубликаты и позиции вне реального стора — не слоты
            keep = []
            for p in take_pos.tolist():
                if p < len(store) and p not in used:
                    used.add(p); keep.append(p)
            if not keep:
                break
            new_tokens = st[torch.tensor(keep, device=dev)]
            cur_q = F.normalize(emb[new_tokens], dim=-1)
            slots.append(new_tokens)
            hop_ids.append(torch.full_like(new_tokens, hop))
        sid = torch.cat(slots)[:K]
        hid = torch.cat(hop_ids)[:K]
        if len(sid) < K:
            sid = torch.cat([sid, torch.zeros(K - len(sid), dtype=torch.long, device=dev)])
            hid = torch.cat([hid, torch.zeros(K - len(hid), dtype=torch.long, device=dev)])
        return sid, hid

    # ---------- forward ----------
    def forward(self, x, hop_ids, slot_valid):
        """x: [B,W] (слоты в 0..K-1, локальные в K..W-1). hop_ids: [B,K].
        slot_valid: [B,K] bool. → h [B,W,D]."""
        B, W = x.shape
        K = self.num_slots
        e = self.embed(x)
        e = torch.cat([e[:, :K], e[:, K:] + self.local_marker], dim=1)   # ключи: без pos
        h = torch.cat([
            self.embed(x[:, :K]) + self.hop_emb(hop_ids),                 # слоты: hop-маркер
            self.embed(x[:, K:]) + self.pos_local + self.local_marker,    # локальные: pos
        ], dim=1)
        # allowed[i,j] = causal_base И (слот валидный ИЛИ локальный)
        cand_ok = torch.cat([slot_valid, torch.ones(B, W - K, dtype=torch.bool, device=x.device)], dim=1)
        allowed = self.causal_base & cand_ok.view(B, 1, 1, W)
        for blk in self.blocks:
            h = blk(h, e, allowed)
        return h

    def logits_local(self, h, slot_valid):
        """Dense-ридаут: для каждой локальной позиции i —
        cat(h[i], attention_pool(h[:, :K], q=h[i])) → [B, L, V]."""
        K = self.num_slots
        hs = h[:, K:]                                   # [B,L,D]
        keys = h[:, :K]                                 # [B,K,D]
        qn = F.normalize(hs, dim=-1)
        kn = F.normalize(keys, dim=-1)
        s = torch.einsum("bld,bkd->blk", qn, kn) * self.pool_tau.exp()
        self._last_pool_s = s.detach()                # diag: скоры ДО маски (valid vs empty)
        s = s.masked_fill(~slot_valid.unsqueeze(1), -1e9)
        w = s.softmax(dim=-1)
        pooled = torch.einsum("blk,bkd->bld", w, keys)
        any_valid = slot_valid.any(-1).view(-1, 1, 1).float()
        pooled = pooled * any_valid                     # пустая история → нулевой пул
        cat = torch.cat([hs, pooled], dim=-1)
        return self.readout(self.norm_out(cat))
