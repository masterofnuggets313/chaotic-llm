# -*- coding: utf-8 -*-
"""ГЕЙТ 1 (ТЗ Universal Frakod Connector, §5): hardcoded K2 oracle KV bridge.

Цепочка (§5.3, единственный засчитываемый вариант):
  lossless Frakod span (утверждение+реплика, tok_v31)
  → v7 embedding table → PCSTSLMv4 adc_slots+forward → h[:, K:K+L, :] (512-d, bf16, без пулинга)
  → MemoryEncoder (fp32: LN → trunk 512 → SiLU → послоевые K/V heads)
  → gated параллельная KV-ветка в attention K2, continuation RoPE (§4.34a.2, §4.34b)
  → замороженная K2-Horizon-0.9B.

Предрегистрация: Desktop/NOT_WIN_YET.md §4.34, §4.34a, §4.34b. Пороги §5.9 зафиксированы ДО запуска.
--mode invariants | probe | explore | confirm [--from-run TAG]
"""
import os, sys, json, time, random, hashlib, argparse, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

A = argparse.ArgumentParser()
A.add_argument('--mode', choices=['invariants', 'probe', 'explore', 'confirm'], required=True)
A.add_argument('--steps', type=int, default=3000)
A.add_argument('--batch', type=int, default=8)
A.add_argument('--lr', type=float, default=3e-4)
A.add_argument('--lam', type=float, default=1.0)
A.add_argument('--probe-steps', type=int, default=200)
A.add_argument('--from-run', type=str, default=None)
ARGS = A.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
K2_DIR = os.path.join(HERE, 'k2_model')
RUNS = os.path.join(HERE, 'gate1_runs')
DEV = 'cuda'; SEED = 20260905; PAD = 64255
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

PROV = dict(representation_source='native_frakod_pc_adc', frakod_tokenizer='tok_v31',
            frakod_checkpoint='ckpt_v7_night_50k.pt', host_embedding_reencode=False)

# ═════════════ 1. Frakod v7 (frozen) — нативный пайплайн ═════════════
from models_pc_v4 import PCSTSLMv4
v7tok = Tokenizer.from_file(os.path.join(HERE, 'tok_v31.json'))
_ck = torch.load(os.path.join(HERE, 'ckpt_v7_night_50k.pt'), map_location=DEV, weights_only=False)
cfg7 = _ck['cfg']
v7 = PCSTSLMv4(vocab=v7tok.get_vocab_size(), d=cfg7['d'], layers=cfg7['layers'],
               window=cfg7['window'], num_heads=cfg7['heads'], topk=cfg7['topk'],
               num_slots=cfg7['slots']).to(DEV)
v7.load_state_dict(_ck['model'], strict=False); v7.eval()
for p in v7.parameters(): p.requires_grad_(False)
K7, W7, DPC = cfg7['slots'], cfg7['window'], 2 * cfg7['d']
SPAN_MAX = 32
PROV['frakod_checkpoint_hash'] = hashlib.sha256(
    open(os.path.join(HERE, 'ckpt_v7_night_50k.pt'), 'rb').read()).hexdigest()[:16]
PROV['native_output_schema'] = dict(fields='cat(h_i_local, attention_pool(h_slots,q=h_i))', shape='[B, L<=32, 512]',
    order='token order', pooling='slot-attention pool (readout path, = вход head генерации)', dtype='bfloat16',
    slots='adc_slots(k=64,hops=3,bigram=True,center=True,store=real span tokens)',
    pc_cfg=dict(cfg7), vocab=v7tok.get_vocab_size())

@torch.no_grad()
def frakod_repr(span_text):
    ids = list(v7tok.encode(span_text, add_special_tokens=False).ids)[:SPAN_MAX]
    real = [t for t in ids if t != 0]                    # <user>:==0==pad — вне стора
    assert real, 'пустой span'
    L = len(real)
    sid, hid = v7.adc_slots(store_tokens=real, query_ids=real[:8], k=K7, hops=3,
                            bigram=True, center=True)
    x = torch.zeros(1, W7, dtype=torch.long, device=DEV)
    x[0, :K7] = sid.to(DEV); x[0, K7:K7 + L] = torch.tensor(real, device=DEV)
    sv = (sid.to(DEV) != 0).view(1, K7)
    h = v7(x, hid.to(DEV).view(1, K7), sv)
    # readout-представление = ровно то, что модель подаёт в head для генерации
    hs = h[:, K7:K7 + L]                                # [1,L,D]
    keys = h[:, :K7]                                    # [1,K,D]
    qn = F.normalize(hs, dim=-1); kn = F.normalize(keys, dim=-1)
    s = torch.einsum('bld,bkd->blk', qn, kn) * v7.pool_tau.exp()
    s = s.masked_fill(~sv.unsqueeze(1), -1e9)
    pooled = torch.einsum('blk,bkd->bld', s.softmax(dim=-1), keys) * sv.any(-1).view(-1, 1, 1).float()
    rep = torch.cat([hs, pooled], dim=-1)[0]            # [L, 512]
    return rep.to(torch.bfloat16).contiguous()

# ═════════════ 2. Forced-memory эпизоды (100% memory-required) ═════════════
UP, BP = '<user>:', '<bot>:'
SMALL_TALK = [("Привет!", "Привет! Рад(а) тебя видеть!"), ("Привет, как дела?", "Привет! Всё отлично, а у тебя?"),
 ("Как дела?", "Неплохо, спасибо! А как у тебя?"), ("Как ты?", "Хорошо! Спасибо, что спросил(а)."),
 ("Что делаешь?", "Да вот, отдыхаю. А ты?"), ("Чем занимаешься?", "Ничего особенного, скучаю. А ты как?"),
 ("Как настроение?", "Отличное! А у тебя?"), ("Что нового?", "Да всё по-старому. А у тебя что нового?"),
 ("Как жизнь?", "Жизнь идёт! Рассказывай, как ты."), ("Как прошёл день?", "Нормально. А у тебя как?"),
 ("Как самочувствие?", "Хорошо, спасибо! А у тебя?"), ("Здравствуй!", "Здравствуй! Очень рад(а) тебя видеть!"),
 ("Ну что, как оно?", "Да нормально! А как у тебя дела?"), ("Эй, привет!", "Привет! Как дела?")]
ST_ALL = ' '.join(u + ' ' + b for u, b in SMALL_TALK).lower()
FACT_T = {'name': [("Меня зовут {f}.", "Приятно познакомиться, {f}!"), ("Моё имя — {f}.", "Красивое имя, {f}!")],
 'city': [("Я из города {f}.", "О, {f} — хороший город!"), ("Живу в {f}.", "Никогда не был в {f}.")],
 'age':  [("Мне {f} лет.", "Ого, {f} — отличный возраст!"), ("Мой возраст — {f}.", "Понятно, {f} лет.")],
 'food': [("Люблю есть {f}.", "Я тоже люблю {f}!"), ("Моя любимая еда — {f}.", "{f} — вкусно!")]}
Q_T = {'name': ["Назови моё имя.", "Как меня зовут?", "Скажи, как меня зовут."],
 'city': ["Из какого я города?", "Где я живу?", "Назови мой город."],
 'age':  ["Сколько мне лет?", "Какой мой возраст?", "Сколько мне лет на самом деле?"],
 'food': ["Что я люблю есть?", "Какая моя любимая еда?", "Что я люблю из еды?"]}
TRAIN_NAMES = ["Алиса","Максим","Даша","Тимур","Оля","Егор","Соня","Марк","Вера","Лев","Настя","Глеб",
 "Юля","Роман","Полина","Кирилл","Милана","Захар","Ева","Артур","Ксюша","Дима","Аня","Стас"]
NEEDLE_NAMES = ["Клава","Борис","Зина","Феликс","Раиса","Эдуард","Валентина","Геннадий","Людмила","Степан"]
CITIES = ["Москва","Питер","Казань","Новосибирск","Екатеринбург","Самара","Омск","Тверь"]
FOODS = ["пицца","борщ","паста","пельмени","шаурма","гречка","суши","блины","плов","салат"]
POOLS = {'train': dict(name=TRAIN_NAMES, city=CITIES[:4], age=list(map(str, range(14, 61))), food=FOODS[:5]),
         'confirm': dict(name=NEEDLE_NAMES, city=CITIES[4:], food=FOODS[5:])}   # age — нет (§4.34a.6)

def make_episode(rng, split, etype, seq):
    pool = POOLS['confirm' if split == 'confirm' else 'train'][etype]
    fact = rng.choice(pool); foil = rng.choice([v for v in pool if v != fact])
    assert foil not in (fact, '') and fact not in foil
    ti = 1 if split == 'confirm' else 0
    qi = 2 if split == 'confirm' else rng.randint(0, 1)
    stmt, reply = FACT_T[etype][ti]
    s_text = f"{UP} {stmt.format(f=fact)}\n{BP} {reply.format(f=fact)}"
    f_text = f"{UP} {stmt.format(f=foil)}\n{BP} {reply.format(f=foil)}"
    assert fact not in f_text                                   # foil чистый (§5.5)
    prompt = "".join(f"{UP} {u}\n{BP} {b}\n" for u, b in rng.sample(SMALL_TALK, rng.randint(4, 8)))
    prompt += f"{UP} {Q_T[etype][qi]}\n{BP} "
    assert fact.lower() not in prompt.lower() and fact.lower() not in ST_ALL   # local_copy=0
    return dict(id=f'{split}-{etype}-{seq}', type=etype, fact=fact, foil=foil, prompt=prompt,
                answer=f'{fact}.', span_text=s_text, foil_text=f_text)

def build_split(split, n, types):
    # фикс по второму аудиту (19.09): встроенный hash() рандомизирован между
    # процессами (PYTHONHASHSEED) -> контент сплитов плыл между прогонами.
    # sha256 детерминирован.
    import hashlib as _h
    seed_off = int(_h.sha256(split.encode()).hexdigest()[:8], 16) % 997
    rng = random.Random(SEED + seed_off)
    return [make_episode(rng, split, types[i % len(types)], i) for i in range(n)]

train_eps   = build_split('train', 600, ['name', 'city', 'age', 'food'])
expl_eval   = build_split('explore', 150, ['name', 'city', 'age', 'food'])
confirm_eps = build_split('confirm', 300, ['name', 'city', 'food'])
for e in confirm_eps:                                            # дизъюнктность значений
    assert e['fact'] not in POOLS['train'][e['type']], e
SPLHASH = hashlib.sha256(json.dumps([[e['id'] for e in s] for s in (train_eps, expl_eval, confirm_eps)],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
print(f"эпизоды: train {len(train_eps)} | expl-eval {len(expl_eval)} | confirm {len(confirm_eps)} | splithash {SPLHASH[:12]}", flush=True)

# ═════════════ 3. K2 (frozen) + Bridge + injection ═════════════
from transformers import AutoModelForCausalLM
k2 = AutoModelForCausalLM.from_pretrained(K2_DIR, dtype=torch.bfloat16, trust_remote_code=True,
                                          attn_implementation='eager').to(DEV)
k2.eval()
for p in k2.parameters(): p.requires_grad_(False)
CFG = k2.config; N_L = CFG.num_hidden_layers; HQ = CFG.num_attention_heads
HKV = CFG.num_key_value_heads; HD = CFG.head_dim; NREP = HQ // HKV
ROTARY = k2.model.rotary_emb
PROV.update(host='K2-Horizon-0.9B', host_layers=N_L, host_gqa=f'{HQ}/{HKV}', head_dim=HD,
            host_frozen=True, host_attn='eager', injection='parallel-gated-KV(§4.34a.2)')
k2tok = Tokenizer.from_file(os.path.join(K2_DIR, 'tokenizer.json'))

GROUPS = dict(early=list(range(0, 7)), middle=list(range(7, 14)), late=list(range(14, 21)),
              all=list(range(0, N_L)))

class Bridge(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)
        self.pos = {id(l): None for l in []}                    # заполняется в patch (по module-object)
        self.ln = nn.LayerNorm(DPC); self.trunk = nn.Linear(DPC, 512)
        self.kv = nn.ModuleDict({str(l): nn.ModuleDict({
            'k': nn.Linear(512, HKV * HD), 'v': nn.Linear(512, HKV * HD)}) for l in self.layers})
        self.gates = nn.Parameter(torch.zeros(len(self.layers)))   # tanh(0)=0 ⇒ точный no-op
        self.index = {}                                              # id(module) -> (slot_index, layer_id)

BR = None; MEM = None
_ORIG_ATTN = None

def _attn_wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    assert getattr(module, 'gate_func', None) is None, 'gate на attn_out ломает аддитивную ветку'
    assert getattr(module, 'rope_head_dim', HD) == module.head_dim, 'partial RoPE не реализован'
    assert getattr(module, 'sliding_window', None) is None, 'sliding window не реализован'
    out, w = _ORIG_ATTN(module, query, key, value, attention_mask, scaling, dropout, **kw)
    if MEM is None or BR is None: return out, w
    slot = BR.index.get(id(module))
    if slot is None: return out, w
    i_slot, layer_id = slot
    g = torch.tanh(BR.gates[i_slot])
    if (not torch.is_grad_enabled()) and g.detach().item() == 0.0:
        return out, w                                            # точный no-op на eval
    km, vm = MEM['kv'][i_slot]
    kr = km.repeat_interleave(NREP, dim=1).to(torch.float32)
    vr = vm.repeat_interleave(NREP, dim=1)
    s = (query.float() @ kr.transpose(2, 3)) * scaling + MEM['mmask'].float()
    p = F.softmax(s, dim=-1).to(query.dtype)
    out = (out + g.to(out.dtype) * (p @ vr).transpose(1, 2))
    return out, w

def patch_k2():
    global _ORIG_ATTN
    mod = importlib.import_module(type(k2.model.layers[0].self_attn).__module__)
    if _ORIG_ATTN is None:
        _ORIG_ATTN = mod.eager_attention_forward
    mod.eager_attention_forward = _attn_wrapper
    BR.index = {id(k2.model.layers[l].self_attn): (i, l) for i, l in enumerate(BR.layers)}

def init_bridge(group):
    global BR
    BR = Bridge(GROUPS[group]).to(DEV)                          # fp32
    return BR

def prep_ids(e):
    p = list(k2tok.encode(e['prompt'], add_special_tokens=False).ids)
    a = list(k2tok.encode(e['answer'], add_special_tokens=False).ids)
    e['_ids'] = torch.tensor([p + a], device=DEV)
    e['_T'] = len(p); e['_A'] = len(a)
    e['_rep_c'] = frakod_repr(e['span_text']); e['_rep_f'] = frakod_repr(e['foil_text'])

print('кэширую frakod-представления…', flush=True)
t0 = time.time()
for e in train_eps + expl_eval + confirm_eps: prep_ids(e)
print(f'готово за {time.time()-t0:.0f}с', flush=True)

def pad_ids(eps):
    Tmax = max(int(e['_ids'].shape[1]) for e in eps)
    X = torch.full((len(eps), Tmax), PAD, dtype=torch.long, device=DEV)
    for i, e in enumerate(eps):
        L = e['_ids'].shape[1]; X[i, :L] = e['_ids'][0]
    return X

def _stable_id(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:6], 'big')

def build_mem_pack(eps, mode):
    reps = []
    for e in eps:
        if mode == 'correct': reps.append(e['_rep_c'])
        elif mode == 'foil':  reps.append(e['_rep_f'])
        else:
            r = random.Random(_stable_id(e['id']) + (7 if mode == 'random' else 13))
            if mode == 'wrongowner':
                alt = [x for x in confirm_eps + expl_eval if x['type'] == e['type'] and x['fact'] != e['fact']]
                reps.append(r.choice(alt)['_rep_c'])
            else:
                reps.append(r.choice(train_eps)['_rep_c'])
    B = len(eps); M = max(int(r.shape[0]) for r in reps)
    t = torch.zeros(B, M, DPC, dtype=torch.bfloat16, device=DEV)
    lens = []
    for i, r in enumerate(reps):
        t[i, :r.shape[0]] = r; lens.append(int(r.shape[0]))
    h = F.silu(BR.trunk(BR.ln(t.float())))                       # [B,M,512] fp32
    pos0 = torch.tensor([e['_T'] for e in eps], device=DEV)
    pos_ids = pos0.view(-1, 1) + torch.arange(M, device=DEV).view(1, -1)   # continuation
    cos, sin = ROTARY(torch.zeros(B, M, 1, dtype=torch.bfloat16, device=DEV), pos_ids)
    cos = cos.unsqueeze(1).float(); sin = sin.unsqueeze(1).float()
    def rot(x): return torch.cat((-x[..., HD // 2:], x[..., :HD // 2]), dim=-1)
    kv = {}
    for slot, li in enumerate(BR.layers):
        m = BR.kv[str(li)]
        km = m['k'](h).view(B, M, HKV, HD).transpose(1, 2)
        vm = m['v'](h).view(B, M, HKV, HD).transpose(1, 2)
        km = ((km.float() * cos) + (rot(km.float()) * sin)).to(torch.bfloat16)
        kv[slot] = (km, vm.to(torch.bfloat16))
    mm = torch.stack([torch.tensor([0.0] * ln + [-1e9] * (M - ln), device=DEV) for ln in lens])
    return dict(kv=kv, mmask=mm.view(B, 1, 1, M).to(torch.bfloat16))

def nll_batch(eps, mode):
    """средний NLL ответных токенов на строку. mode empty ⇒ MEM=None."""
    global MEM
    MEM = None if mode == 'empty' else build_mem_pack(eps, mode)
    X = pad_ids(eps)
    if torch.is_grad_enabled():
        lg = k2(input_ids=X).logits
    else:
        with torch.no_grad():
            lg = k2(input_ids=X).logits
    MEM = None
    out = []
    for i, e in enumerate(eps):
        a0, aL = e['_T'] - 1, e['_A']
        tgt = X[i, a0 + 1: a0 + 1 + aL]
        out.append(F.cross_entropy(lg[i, a0: a0 + aL].float(), tgt))
    return out

def episode_metrics(eps, batch=16, modes=('correct', 'foil', 'empty')):
    res = {}
    with torch.no_grad():
        for mode in modes:
            vals = []
            for i in range(0, len(eps), batch):
                vals += [float(x) for x in nll_batch(eps[i:i+batch], mode)]
            res[mode] = np.array(vals)
    return dict(contrast=res['foil'] - res['correct'], gain=res['empty'] - res['correct'], nll=res)

def boot_ci(d, seed=0, n=5000):
    d = np.asarray(d); r = np.random.default_rng(seed)
    bs = np.array([r.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# ═════════════ 4. Инварианты (§12; поправка 12.9 — §4.34b.2) ═════════════
def run_invariants(group='all'):
    init_bridge(group); patch_k2()
    R = {}; eps = expl_eval[:8]
    tg = sum(1 for p in k2.parameters() if p.requires_grad)
    R['12.1_frozen_host'] = dict(host_trainable=tg, ok=tg == 0)
    def host_hash():
        return {n: hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:12]
                for n, t in k2.named_parameters()}
    H0 = host_hash()                                        # ДО любых backward/step
    bare = np.array([float(x) for x in nll_batch(eps, 'empty')])
    zero = np.array([float(x) for x in nll_batch(eps, 'correct')])   # gate=0, no_grad ⇒ skip
    d = np.abs(zero - bare).max()
    R['12.2_noop_parity'] = dict(max_abs_diff=float(d), ok=bool(d == 0.0))
    # живой градиент: gate ≠0 на 1-м backward; encoder — на 2-м (после opt.step), §4.34b.2
    opt = torch.optim.AdamW(BR.parameters(), lr=ARGS.lr)
    loss1 = torch.stack(nll_batch(eps, 'correct')).mean()
    opt.zero_grad(set_to_none=True); loss1.backward()
    gg = float(BR.gates.grad.abs().sum())
    opt.step()
    loss2 = torch.stack(nll_batch(eps, 'correct')).mean()
    opt.zero_grad(set_to_none=True); loss2.backward()
    tr = float(BR.trunk.weight.grad.abs().sum()); tk = float(BR.kv[str(BR.layers[0])]['k'].weight.grad.abs().sum())
    host_grad = any(p.grad is not None for p in k2.parameters())
    BR.zero_grad(set_to_none=True)
    with torch.no_grad(): BR.gates.zero_()
    R['12.9_zero_grad_live'] = dict(gate_grad_step1=gg, trunk_grad_step2=tr, kv_grad_step2=tk,
                                    host_leak_grad=bool(host_grad),
                                    ok=(gg > 0 and tr > 0 and tk > 0 and not host_grad))
    globals()['MEM'] = None
    # ненулевой gate ⇒ вклад меняется (sanity аддитивной ветки)
    with torch.no_grad(): BR.gates.add_(0.5)
    m1 = np.array([float(x) for x in nll_batch(eps, 'correct')])
    with torch.no_grad(): BR.gates.zero_()
    m0 = np.array([float(x) for x in nll_batch(eps, 'correct')])
    R['injection_active'] = dict(diff=float(np.abs(m1 - m0).max()), ok=bool(np.abs(m1 - m0).max() > 1e-4))
    e = confirm_eps[0]
    pa = list(k2tok.encode(e['prompt'], add_special_tokens=False).ids)
    ta = pa + list(k2tok.encode('Клава.', add_special_tokens=False).ids)
    tb = pa + list(k2tok.encode('Зина.', add_special_tokens=False).ids)
    L = max(len(ta), len(tb)); Z = 0                        # pad-0, формы равны ⇒ округление идентично
    ib = torch.tensor([ta + [Z] * (L - len(ta)), tb + [Z] * (L - len(tb))], device=DEV)
    with torch.no_grad():
        lg2 = k2(input_ids=ib).logits
    la, lb = lg2[0, len(pa) - 1], lg2[1, len(pa) - 1]
    R['12.6_causal_leak'] = dict(max_diff=float((la - lb).abs().max()), ok=bool(torch.equal(la, lb)),
                                 note='одинаковые формы батча ⇒ любое расхождение = не-каузальность')
    mm = build_mem_pack(eps, 'correct'); km = mm['kv'][0][0]
    R['12.4_dtype_device'] = dict(dtype=str(km.dtype), cuda=km.is_cuda,
                                  ok=str(km.dtype) == 'torch.bfloat16' and km.is_cuda)
    R['12.5_shape'] = dict(k=tuple(km.shape), ok=(km.shape[1] == HKV and km.shape[3] == HD))
    bad = 0; contig = 0; n_all = len(train_eps) + len(expl_eval) + len(confirm_eps)
    for e in train_eps + expl_eval + confirm_eps:
        # гарантированное владение: все токены ответа лежат в oracle span (слово факта —
        # ответ и есть; §5.4: span = утверждение+реплика, оба содержат факт).
        f_word = e['fact']
        ok_span = f_word in e['span_text'] and f' {f_word}' in e['span_text']
        ok_foil = f_word not in e['foil_text']            # foil не содержит факт (§5.5)
        ft = [t for t in v7tok.encode(e['fact'], add_special_tokens=False).ids if t != 0]
        sid = v7tok.encode(e['span_text'], add_special_tokens=False).ids
        contig += int(any(list(sid[i:i + len(ft)]) == ft for i in range(len(sid) - len(ft) + 1)))
        if not (ok_span and ok_foil): bad += 1
    R['12.8_replacement'] = dict(bad=bad, n=n_all, token_contig_frac=contig / n_all, ok=bad == 0,
        note='владение по тексту (кириллица дробится BPE контекстно — континуум инфо, не гейт)')
    lg = torch.zeros(1, 5, 8); lg[0, 4, 3] = 30.   # позиция T-1 предсказывает класс 3
    n1 = float(F.cross_entropy(lg[0, 4:5, :], torch.tensor([3])))
    ok127 = n1 < 1e-3 and float(F.cross_entropy(lg[0, 3:4, :], torch.tensor([3]))) > 1.0
    R['12.7_target_align'] = dict(nll_aligned=n1, ok=bool(ok127),
                                  note='a0=_T-1: logits[T-1]→target[0]; соседняя позиция НЕ согласована')
    H1 = host_hash()                                        # ПОСЛЕ backward+opt.step
    diffp = [n for n in H0 if H0[n] != H1[n]]
    R['12.1b_host_hash_after_backward'] = dict(n_params=len(H0), changed=diffp[:5], n_changed=len(diffp),
                                               ok=(len(diffp) == 0))
    ok = all(v.get('ok', False) for v in R.values())
    for k, v in R.items(): print(' ', k, json.dumps(v, ensure_ascii=False, default=str)[:200])
    print('INVARIANTS:', 'ALL PASS' if ok else 'FAIL')
    os.makedirs(RUNS, exist_ok=True)
    json.dump(R, open(os.path.join(HERE, 'gate1_invariants.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    return ok

# ═════════════ 5. Обучение и прогоны ═════════════
def train(group_name, steps, eval_every, tag):
    init_bridge(group_name); patch_k2()
    opt = torch.optim.AdamW(BR.parameters(), lr=ARGS.lr)
    rng = random.Random(SEED + 1); t0 = time.time(); hist = []
    for st in range(1, steps + 1):
        be = rng.sample(train_eps, ARGS.batch)
        n_c = torch.stack(nll_batch(be, 'correct')).mean()
        n_f = torch.stack(nll_batch(be, 'foil')).mean()
        n_e = torch.stack(nll_batch(be, 'empty')).mean()
        loss = n_c + ARGS.lam * (F.softplus(n_f - n_c) + F.softplus(n_e - n_c))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st % eval_every == 0 or st == steps:
            m = episode_metrics(expl_eval[:64])
            c, ci = boot_ci(m['contrast'])
            hist.append(dict(step=st, loss=float(loss), contrast=c, ci=ci,
                             gain=float(m['gain'].mean()), mins=round((time.time()-t0)/60, 1)))
            print(f'  [{tag} {st}/{steps}] loss={float(loss):.3f} contrast_expl={c:+.2f} '
                  f'CI[{ci[0]:+.2f},{ci[1]:+.2f}] gain={m["gain"].mean():+.2f} {hist[-1]["mins"]}мин', flush=True)
    best = max(hist, key=lambda h: h['contrast'])
    os.makedirs(RUNS, exist_ok=True)
    torch.save(dict(bridge=BR.state_dict(), cfg=dict(group=group_name, steps=steps, seed=SEED,
                                                     best_step=best['step'])),
               os.path.join(RUNS, f'{tag}.pt'))
    run = dict(tag=tag, group=group_name, steps=steps, hist=hist, best=best,
               gates=[round(float(torch.tanh(g)), 4) for g in BR.gates],
               trainable=sum(p.numel() for p in BR.parameters()))
    json.dump(run, open(os.path.join(RUNS, f'{tag}.json'), 'w'), indent=1)
    return run

def do_probe():
    print('== PROBE: диагностика живости градиентов (НЕ гейт-вердикт) ==')
    if not run_invariants(): print('INVARIANTS FAIL — СТОП'); return
    r = train('all', ARGS.probe_steps, 50, 'probe')
    print('probe hist:', json.dumps([h['contrast'] for h in r['hist']]))
    print('probe best:', json.dumps(r['best']))

def do_explore():
    inv = json.load(open(os.path.join(HERE, 'gate1_invariants.json'), encoding='utf-8'))
    if not all(v.get('ok', False) for v in inv.values()):
        print('инварианты не пройдены — СТОП (ТЗ §18.14)'); return
    sel = {}
    for gname in GROUPS:
        r = train(gname, ARGS.steps, max(1, ARGS.steps // 4), f'expl_{gname}')
        sel[gname] = r['best']['contrast']
        print(f'group {gname}: contrast(expl)={sel[gname]:+.2f}', flush=True)
    chosen = max(sel, key=lambda g: (sel[g], -len(GROUPS[g])))
    json.dump(dict(selection=sel, chosen=chosen, rule='max contrast on exploration; tie→fewer layers',
                   splithash=SPLHASH, prov=PROV,
                   manifest=dict(contrast_threshold_nat=3.0, bootstrap_samples=5000,
                                 minimum_confirmatory_episodes=250, exploration_episode_count=150,
                                 confirmatory_episode_count=len(confirm_eps),
                                 primary_eval_split='confirmatory_heldout', primary_memory_slots=1,
                                 primary_distractor_spans=0,
                                 layer_candidates=['early', 'middle', 'late', 'all'],
                                 confirmatory_episode_ids=SPLHASH)),
              open(os.path.join(RUNS, 'layer_choice.json'), 'w'), indent=1)
    print(f'LAYER GROUP CHOSEN (exploration-only): {chosen}')

def do_confirm():
    if ARGS.from_run:
        tag = ARGS.from_run
    else:
        tag = 'expl_' + json.load(open(os.path.join(RUNS, 'layer_choice.json')))['chosen']
    ck = torch.load(os.path.join(RUNS, tag + '.pt'), map_location=DEV, weights_only=False)
    init_bridge(ck['cfg']['group']); BR.load_state_dict(ck['bridge']); patch_k2(); BR.eval()
    m = episode_metrics(confirm_eps)
    mo = episode_metrics(confirm_eps[:64], modes=('correct', 'wrongowner'))
    c, ci = boot_ci(m['contrast']); gain, gci = boot_ci(m['gain'])
    woc, _ = boot_ci(mo['contrast'])
    verdict = ('UNDERPOWERED_CONFIRMATION' if len(confirm_eps) < 250 else
               'GATE_1_PASS' if (c >= 3.0 and ci[0] > 0 and gain > 0) else
               'GATE_1_PARTIAL' if (c > 0.3 and ci[0] > 0) else 'GATE_1_FAIL')
    rep = dict(run_id=f'confirm_{tag}', verdict=verdict, contrast=c, contrast_ci=ci,
               gain=gain, gain_ci=gci, n_confirm=len(confirm_eps), splithash=SPLHASH, prov=PROV,
               layer_group=ck['cfg']['group'], mode='native_frakod_oracle',
               primary_memory_slots=1, primary_distractor_spans=0,
               historical_e1_contrast=1.85, contrast_delta_vs_e1=c - 1.85,
               nll_means={k: float(v.mean()) for k, v in m['nll'].items()},
               per_type_contrast={t: float(np.mean([m['contrast'][i] for i, e in enumerate(confirm_eps) if e['type'] == t]))
                                  for t in ('name', 'city', 'food')},
               diag_wrongowner_contrast_64=woc,
               gates=[round(float(torch.tanh(g)), 4) for g in BR.gates],
               per_episode=dict(ids=[e['id'] for e in confirm_eps],
                                contrast=[float(x) for x in m['contrast']],
                                gain=[float(x) for x in m['gain']]))
    json.dump(rep, open(os.path.join(HERE, 'gate1_confirm.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_episode'}, indent=1, default=float))
    print('VERDICT:', verdict)

if __name__ == '__main__':
    {'invariants': run_invariants, 'probe': do_probe, 'explore': do_explore,
     'confirm': do_confirm}[ARGS.mode]()
