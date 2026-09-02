"""Эксперимент 4: честное сравнение decode-шага STS-Prog(fast) vs Transformer(KV-cache).

Вопрос: кто быстрее генерирует 1 токен при длине контекста W?
- STS-Prog fast: драйверы = селекция по всему W (e_all кэшируется, меняется только q),
  блоки только по K последних позиций + h_last. НЕТ FFN.
- Transformer: KV-кэш decode, attention нового токена ко всем W ключам + FFN.

Запуск: cd phase01/exp_vq && C:/Python313/python.exe exp_decode_vs_transformer.py
"""
import os, sys, time, json, torch
import torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.join(HERE, "..")
REPO = os.path.join(PHASE, "..")
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
from models_pc import build_pc_model
from night_task5_fracode_forward import keys_at, TEMP, TAIL
from parametric_models import TransformerLM, count_params
import final_benchmark as fb

RESULTS = os.path.join(REPO, "results")
K = 4096


def fast_decode_step(model, e_all, q0, Wc, k_eff, topk, nq):
    """Один decode-шаг STS-Prog: драйверы по Wc + h_last + g по K. e_all кэширован."""
    q = q0
    h_last = e_all[-1:].clone()
    hK = e_all[-K:].clone()
    en = e_all / (e_all.norm(dim=-1, keepdim=True) + 1e-6)   # кэшируемая нормировка
    for li, blk in enumerate(model.blocks):
        k = k_eff
        qn = q / (q.norm() + 1e-6)
        sim = (en * qn).sum(-1)
        sim[Wc - TAIL:] = -1e9
        kk = min(topk, Wc - TAIL)
        vals, loc = sim.topk(kk)
        w = torch.softmax(vals / TEMP, 0)
        nxt = torch.clamp(loc + 1, 0, Wc - 2)
        driver = (w.unsqueeze(-1) * e_all[nxt]).sum(0, keepdim=True)
        h_last = blk(h_last, driver, k)
        hK = blk(hK, driver, k)
        q = q0 + model.query_proj(h_last) * 0.5
    g = hK.mean(0, keepdim=True)
    return model.readout3(torch.cat([h_last, q0, g], dim=-1))


class TransformerKV:
    """Инкрементальный decode Transformer с KV-кэшем (честный decode-шаг)."""
    def __init__(self, model):
        self.m = model
        self.nhead = model.blocks[0].attn.num_heads
        self.d = model.blocks[0].attn.embed_dim
        self.hd = self.d // self.nhead

    def _kv(self, blk, x):
        """Достаём проекции K/V из MultiheadAttention in_proj_weight (d, 3d).
        x: (1, W, d) или (1, d). Возвращает (1, n, W, hd) или (1, n, 1, hd)."""
        w = blk.attn.in_proj_weight          # (3d, d)
        b = blk.attn.in_proj_bias            # (3d,)
        Wq, Wk, Wv = w.chunk(3, 0); bq, bk, bv = b.chunk(3)
        q = x @ Wq.T + bq
        k = x @ Wk.T + bk
        v = x @ Wv.T + bv
        # reshape to (1, nhead, W, hd)
        if q.dim() == 2:  # (1, d) — decode_one
            q = q.view(1, self.nhead, self.hd)
            k = k.view(1, self.nhead, self.hd)
            v = v.view(1, self.nhead, self.hd)
        else:  # (1, W, d) — prefix
            q = q.view(1, self.nhead, -1, self.hd)
            k = k.view(1, self.nhead, -1, self.hd)
            v = v.view(1, self.nhead, -1, self.hd)
        return q, k, v

    def decode_one(self, x_tok, k_cache, v_cache, pos, lengths=None):
        """Один decode-шаг с преаллоцированным KV-кэшем (без O(W) cat).
        k_cache/v_cache: списки (1,n,MAXL,hd); lengths: [len] списки текущей длины."""
        d, nhead, hd = self.d, self.nhead, self.hd
        h = self.m.embed(x_tok) + self.m.pos[:, pos, :]      # (1, d)
        if lengths is None:
            lengths = [k.shape[2] - 1 for k in k_cache]  # текущая длина до вставки
        for li, blk in enumerate(self.m.blocks):
            ln = blk.ln1(h)
            q, k, v = self._kv(blk, ln)
            if q.dim() == 3:  # decode_one: (1,n,hd) -> (1,n,1,hd)
                q = q.unsqueeze(2); k = k.unsqueeze(2); v = v.unsqueeze(2)
            L = lengths[li]
            k_cache[li][:, :, L:L+1, :] = k      # запись в буфер
            v_cache[li][:, :, L:L+1, :] = v
            kc = k_cache[li][:, :, :L+1, :]      # (1,n,L+1,hd) view — без копии
            vc = v_cache[li][:, :, :L+1, :]
            scores = (q @ kc.transpose(-1, -2)) / (hd ** 0.5)  # (1,n,1,L+1)
            attn = torch.softmax(scores, -1)
            a = (attn @ vc).reshape(1, d)
            a = a @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
            h = h + a
            h = h + blk.ffn(blk.ln2(h))
        return self.m.head(self.m.ln_f(h)), k_cache, v_cache

    def prefix(self, x, chunk=256):
        """Префилл с преаллокацией KV-кэша (без O(W²) cat)."""
        import torch.nn.functional as F
        W = x.shape[0]
        k_cache = [torch.zeros(1, self.nhead, W, self.hd, device=x.device) for _ in self.m.blocks]
        v_cache = [torch.zeros(1, self.nhead, W, self.hd, device=x.device) for _ in self.m.blocks]
        h = self.m.embed(x) + self.m.pos[:, :W, :]             # (1, W, d)
        for s in range(0, W, chunk):
            e = min(s + chunk, W)
            B = e - s
            # маска (B, s+B): allow j <= s+i
            i_row = torch.arange(B, device=x.device).unsqueeze(1)
            j_col = torch.arange(s + B, device=x.device).unsqueeze(0)
            mask = (j_col > s + i_row)                         # (B, s+B) bool
            for li, blk in enumerate(self.m.blocks):
                ln = blk.ln1(h[:, s:e, :])
                q, k, v = self._kv(blk, ln)                    # (1,n,B,hd)
                k_cache[li][:, :, s:e, :] = k
                v_cache[li][:, :, s:e, :] = v
                kc = k_cache[li][:, :, :s+B, :]                # (1,n,s+B,hd)
                vc = v_cache[li][:, :, :s+B, :]
                a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
                a = a.reshape(1, B, self.d)
                a = a @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
                h[:, s:e, :] = h[:, s:e, :] + a
                h[:, s:e, :] = h[:, s:e, :] + blk.ffn(blk.ln2(h[:, s:e, :]))
        return k_cache, v_cache


def timeit(fn, warmup=2, iters=5):
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
    print("Loading data + models...", flush=True)
    head = fb.load_chars(os.path.join(PHASE, "corpus_train.txt"), 990_000)
    tok = fb.make_bpe(head); V = tok.get_vocab_size()
    n_head = len(tok.encode(head).ids)
    ids = np.array(tok.encode(fb.load_chars(os.path.join(PHASE, "corpus5m_train.txt"), None)).ids, dtype=np.int64)
    toks = torch.tensor(ids[n_head:], dtype=torch.long, device=dev)

    # STS-Prog
    sts = build_pc_model("pc", vocab=V, d=192, layers=8, k_init=1.2,
                         sync_steps=8, driver_mode="sts_prog", alpha=0.3, temp=TEMP).to(dev).eval()
    sts.load_state_dict(torch.load(os.path.join(RESULTS, "ckpts", "sts_prog_seed0.pt"), map_location="cpu"))
    for p in sts.parameters(): p.requires_grad_(False)
    k_eff = torch.sigmoid(sts.k); topk = int(sts.topk); nq = int(sts.nquery)
    mode = "cyclic"

    # Transformer под ~900K параметров (как в final_benchmark)
    from match_transformer import pick_tf_dims
    MAX_W = 262144
    D_tf = pick_tf_dims(900_000, V, 512, layers=8, heads=4)  # W=512 (учебное окно, не inflate)
    D_tf = max(4, (D_tf // 4) * 4)  # кратно heads=4
    print(f"Choosing D_tf={D_tf} for ~900K params")
    # Создаём с малым W (TFBlock материализует маску (W,W) в __init__ — не нужна,
    # своя prefix использует is_causal=True), pos оставляем малый (W=512) — для speed
    # benchmark pos не влияет, используем только для размера.
    tf = TransformerLM(V, 512, D=D_tf, HEADS=4, LAYERS=8).to(dev).eval()
    # pos-таблица на MAX_W (конструктор уже не пересоздаст маску; pos не влияет на
    # скорость шага — gather+add, в отчёт параметров не входит)
    n_pos = tf.pos.numel()
    tf.pos = torch.nn.Parameter(torch.zeros(1, MAX_W, D_tf, device=dev))
    tf.pos.requires_grad_(False)
    for p in tf.parameters(): p.requires_grad_(False)
    tfkv = TransformerKV(tf)
    print(f"STS params={count_params(sts):,} | TF params={count_params(tf)-n_pos:,} + pos-таблица {n_pos:,} (D={D_tf})")

    o = 5000
    results = {}
    for Wc in [16384, 65536, 262144]:
        x = toks[o:o + Wc]
        print(f"\n=== W={Wc} ===", flush=True)

        # --- STS fast decode ---
        e_all = keys_at(model=sts, x=x, pos=torch.arange(Wc, device=dev), mode=mode).detach()
        q0 = e_all[-nq:].mean(0, keepdim=True)
        t_sts = timeit(lambda: fast_decode_step(sts, e_all, q0, Wc, k_eff, topk, nq),
                       warmup=2, iters=5)
        sts_tps = 1 / t_sts
        print(f"STS fast:   {t_sts*1000:8.1f} ms -> {sts_tps:8.2f} tok/s")

        # --- Transformer KV decode ---
        # KV-кэш преаллоцирован (MAXL=Wc-1) и заполнен СЛУЧАЙНО: стоимость decode-шага
        # зависит от длины кэша, не от значений. Префилл — one-time cost, не засекаем.
        k_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        v_cache = [torch.randn(1, tfkv.nhead, Wc - 1, tfkv.hd, device=dev) for _ in range(8)]
        lengths = [Wc - 1] * 8   # кэш уже "заполнен" до Wc-1
        # decode-шаг (засекаем)
        t_tf = timeit(lambda: tfkv.decode_one(x[Wc - 1].view(1), k_cache, v_cache, Wc - 1, lengths),
                      warmup=2, iters=5)
        tf_tps = 1 / t_tf
        print(f"TF KV:      {t_tf*1000:8.1f} ms -> {tf_tps:8.2f} tok/s")

        ratio = tf_tps / sts_tps
        print(f"ratio TF/STS: {ratio:.2f}x  ({'TF быстрее' if ratio > 1 else 'STS быстрее'})")
        results[Wc] = {
            "sts_ms": round(t_sts * 1000, 2), "sts_tok_s": round(sts_tps, 2),
            "tf_ms": round(t_tf * 1000, 2), "tf_tok_s": round(tf_tps, 2),
            "tf_over_sts": round(ratio, 2),
        }

    out = {"K": K, "results": results}
    with open(os.path.join(RESULTS, "exp_decode_vs_transformer.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: results/exp_decode_vs_transformer.json")


if __name__ == "__main__":
    main()