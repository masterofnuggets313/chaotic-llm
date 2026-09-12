# -*- coding: utf-8 -*-
"""Строитель архива Fracode: корпус -> индекс.
Тот же движок, что stress_10m_fix (S=12, K=256, calib 1M, iters 12, seed 0).
Артефакты в frakod_index/:
  codes.npy    (N,2,12 u8)   = 24 Б/токен — само сжатие (ADC-скрининг)
  keys_f16.npy (N,192 f16)   = 384 Б/токен — реранк кандидатов сырым косинусом
  ids.npy      (N i32)       = 4 Б/токен — лекс-слой (точный поиск слов)
  cbook_l{0,1}.npy           = кодобуки (фикс. ~0.4 МБ)
ЧЕСТНО О ХРАНИЛИЩЕ: «24 Б/токен» относится ТОЛЬКО к codes.npy. Постоянный
след на диске включает keys_f16 и ids; полный итог пишется в meta.json
(b_per_tok_total) и отдаётся /index/accounting.
"""
import sys, os, time, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
# ВАЖНО: этот файл лежит в frakod/, а не в phase01/exp_vq/ (откуда он был
# скопирован), поэтому относительные пути считаются иначе. Раньше здесь было
# PHASE=HERE/.. (= chaotic-llm) и REPO=PHASE/.. (= G:/Migration), из-за чего все
# ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ были битыми: corpus5m_train.txt и corpus_public.txt
# искались в chaotic-llm/, а дефолтный чекпойнт — в G:/Migration/results/ckpts/.
# Пайплайн спасался тем, что всегда передаёт FRK_TOK/FRK_CKPT/FRK_CORPUS/FRK_OUTD.
REPO = os.path.abspath(os.path.join(HERE, '..'))        # chaotic-llm
PHASE = os.path.join(REPO, 'phase01')                   # chaotic-llm/phase01
sys.path.insert(0, PHASE); sys.path.insert(0, HERE)
import final_benchmark as fb
from models_pc import build_pc_model
from night_task5_fracode_forward import StreamFracode

NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
S = 12; K = 256; D = 192
# S (число подвекторов) зависит от размерности: d обязано делиться на S.
# d=192 -> S=12 (исторический формат, совместим со старыми архивами);
# d=256 (v7) -> S=16, иначе 256/12 не целое. API читает S из meta.json.
_S_ENV = os.environ.get('FRK_SUBVECS')
if _S_ENV:
    S = int(_S_ENV)
CORPUS = os.environ.get('FRK_CORPUS') or os.path.join(PHASE, 'corpus5m_train.txt')
OUT = os.environ.get('FRK_OUTD') or os.path.join(HERE, 'frakod_index'); os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# v8-ready: чем кодируем и какими emb строим ключи (по умолчанию — v7-историческое)
_tokfile = os.environ.get('FRK_TOK')       # путь к saved tokenizer json (v8)
if _tokfile:
    from tokenizers import Tokenizer as _Tkk
    tok = _Tkk.from_file(_tokfile)
    head = ''
else:
    head = fb.load_chars(os.path.join(PHASE, 'corpus_public.txt'), 990_000)
    tok = fb.make_bpe(head, vocab=int(os.environ.get('FRK_VOCAB', 512)))
big = fb.load_chars(CORPUS if os.path.isabs(CORPUS) else os.path.join(HERE, CORPUS), None)
t0 = time.time(); ids_all = np.array(tok.encode(big).ids, dtype=np.int64)
print(f'токенизировано {len(ids_all):,} за {time.time()-t0:.0f}s', flush=True)
N = min(NTOK, len(ids_all) - 40)
# ЧЕСТНО: на коротком корпусе кодбук (K=256 на подвектор) физически недоучивается —
# кластеров больше, чем точек. Архив валиден и читается, но сжатие не показательно.
# kmeans теперь переживает N<K (см. fracode_memory_probe.kmeans), но смысла в нём нет.
if N < 10_000:
    print(f'ВНИМАНИЕ: корпус мал (N={N} позиций). Кодбук K={K} на уровень '
          f'недоучен, метрики сжатия на таком архиве не интерпретировать. '
          f'Для честной сборки нужно >= ~200k позиций.', flush=True)
toks = torch.tensor(ids_all[:N], dtype=torch.long, device=dev)

_ckpt = os.environ.get('FRK_CKPT') or os.path.join(REPO, 'results', 'ckpts', 'sts_prog_seed0.pt')
# ---------------------------------------------------------------------------
# ДВА ПУТИ КОДИРОВАНИЯ (добавлено 12.09, ПУТЬ А).
#
# Исторический путь (ниже): ключи = context-avg по СТАТИЧЕСКОЙ таблице токенов
# внутренней модели (`embed.weight`). Он и есть корень нулевого recall: энкодер
# обучен не был, cos между эмбеддингами разных текстов = 0.000.
#
# Внешний путь (FRK_ENCODER=ollama): ключи приходят от ГОТОВОГО энкодера
# (bge-m3 через Ollama). Измерено на продуктовом протоколе 12.09:
#   наша модель (сырой) 0.087 -> bge-m3 (сырой) 0.517 -> bge-m3 (24 Б) 0.309.
# То есть пункт «достать инфу» закрывается сменой энкодера, а НЕ переделкой
# памяти: формат архива (codes/cbook/ids/meta) не меняется.
#
# ВАЖНО О ГРАНУЛЯРНОСТИ. Внутренний путь строит КЛЮЧ НА КАЖДЫЙ ТОКЕН (таблица
# токенов). Внешний энкодер даёт вектор на ТЕКСТ, не на токен: разумная
# гранулярность — символьные окна того же размера, что продуктовый чанк
# (FRK_ENC_CHUNK, по умолчанию 800 с перекрытием 160). Тогда вектор запроса
# приходит от того же оператора, и API попадает в ту же геометрию.
# Ограничение (ЧЕСТНО): архив на символьных окнах и архив на токенах — разная
# гранулярность адреса. Для продуктового канала «вопрос -> фрагмент» верна
# оконная; токенный режим остаётся для совместимости со старыми архивами.
# ---------------------------------------------------------------------------
_ENC = (os.environ.get('FRK_ENCODER') or '').strip().lower()
if _ENC in ('ollama', 'api', 'http'):
    import urllib.request
    _emodel = os.environ.get('FRK_EMBED_MODEL', 'bge-m3')
    _eurl = os.environ.get('FRK_EMBED_URL', 'http://localhost:11434/api/embed')
    _echunk = int(os.environ.get('FRK_ENC_CHUNK', 800))
    _eover = int(os.environ.get('FRK_ENC_OVERLAP', 160))
    if _eover >= _echunk:
        raise ValueError(f'FRK_ENC_OVERLAP={_eover} >= FRK_ENC_CHUNK={_echunk}: '
                         f'шаг окна <= 0, архив будет битым')
    _text_full = big
    _estep = _echunk - _eover
    _spans = [(i, min(i + _echunk, len(_text_full)))
              for i in range(0, len(_text_full), _estep)]
    _spans = [(s, e) for s, e in _spans if e - s > 50]
    _texts = [_text_full[s:e] for s, e in _spans]
    # КЭШ векторов: кодирование 7934 окон через Ollama занимает ~6 минут. Без
    # кэша каждый повторный прогон (а их будет много при отладке) платит заново.
    # Ключ кэша — модель + размер/перекрытие окна + длина корпуса, чтобы смена
    # любого из них не подсунула чужие векторы.
    import hashlib as _hl
    _ckey = _hl.md5(f'{_emodel}|{_echunk}|{_eover}|{len(_text_full)}|'
                    f'{len(_spans)}'.encode()).hexdigest()[:12]
    # КЭШ — НЕ ВНУТРИ АРХИВА. Здесь был смысловой промах: файл кэша (32 МБ на
    # 7934 окна) лежал в самом каталоге архива, который весь весит 2.2 МБ.
    # Это и утечка (векторы окон корпуса), и абсурд в любой проверке размера.
    # Кладём рядом с каталогом архива (общий рабочий кэш на все прогоны).
    _cdir = os.environ.get('FRK_ENCCACHE') or os.path.join(os.path.dirname(
        OUT.rstrip('/\\')), 'enc_cache')
    os.makedirs(_cdir, exist_ok=True)
    _cf = os.path.join(_cdir, f'enc_cache_{_emodel.replace(":", "-")}_{_ckey}.npy')
    if os.path.exists(_cf):
        base = torch.tensor(np.load(_cf), device=dev)
        print(f'ВНЕШНИЙ ЭНКОДЕР {_emodel}: векторы из кэша {os.path.basename(_cf)} '
              f'{tuple(base.shape)}', flush=True)
    else:
        _vecs = []
        t0 = time.time()
        for _i0 in range(0, len(_texts), 16):
            _body = json.dumps({'model': _emodel,
                                'input': _texts[_i0:_i0 + 16]}).encode()
            _req = urllib.request.Request(_eurl, data=_body,
                                          headers={'Content-Type': 'application/json'})
            _r = json.loads(urllib.request.urlopen(_req, timeout=600).read().decode())
            _vecs.extend(_r['embeddings'])
            if _i0 % 1600 == 0:
                print(f'  энкодер: {_i0 + len(_texts[_i0:_i0+16])}/{len(_texts)} окон '
                      f'({time.time()-t0:.0f}s)', flush=True)
        base = torch.tensor(np.asarray(_vecs, dtype=np.float32), device=dev)
        np.save(_cf, np.asarray(_vecs, dtype=np.float32))
        print(f'  векторы закэшированы -> {os.path.basename(_cf)} '
              f'({time.time()-t0:.0f}s)', flush=True)
    D = int(base.shape[1])
    N_WIN = int(base.shape[0])
    # S подбираем под d (d=1024 -> 1,2,4,8,16,32,64; S=12 от d=192 НЕ подходит)
    if D % S != 0:
        for _s in (8, 4, 16, 32, 2, 1):
            if D % _s == 0:
                S = _s
                break
        else:
            raise ValueError(f'не найден делитель d={D} для subvecs')
    print(f'ВНЕШНИЙ ЭНКОДЕР {_emodel}: {N_WIN} окон d={D}, S={S}, '
          f'окно {_echunk}/{_eover}, за {time.time()-t0:.0f}s', flush=True)
    _ckpt = None            # внутренний путь не используется
    _EXT_ENC = {'kind': 'external', 'model': _emodel, 'url': _eurl,
                'chunk': _echunk, 'overlap': _eover, 'd': D}
else:
    _EXT_ENC = None
    _sd = torch.load(_ckpt, map_location='cpu', weights_only=False)
    # v7-чекпойнт хранит веса под 'model', seed0 — плоский state_dict
    _sd = _sd.get('model', _sd) if isinstance(_sd, dict) else _sd
    _w = None
    for _k in ('embed.weight', 'model.embed.weight'):
        if isinstance(_sd, dict) and _k in _sd:
            _w = _sd[_k]; break
    if _w is None:
        raise KeyError(f'в {_ckpt} нет embed.weight — не могу строить ключи')
    EMB = _w.detach().float().cpu()                 # (vocab, d) — таблица как есть
    D = int(EMB.shape[1])
    _NT = int(tok.get_vocab_size()) if hasattr(tok, 'get_vocab_size') else int(tok.vocab_size)
    if int(EMB.shape[0]) != _NT:
        raise ValueError(f'несовместимость: ckpt vocab={EMB.shape[0]}, '
                         f'tokenizer vocab={_NT}. '
                         f'Возьмите чекпойнт, обученный на этом словаре.')
    print(f'emb-таблица из {os.path.basename(_ckpt)}: {tuple(EMB.shape)}', flush=True)
    # приводим S к размерности: d % S == 0
    if D % S != 0:
        for _s in (16, 8, 12, 32, 4, 2, 1):
            if D % _s == 0:
                print(f'S={S} не делит d={D} -> берём S={_s}', flush=True)
                S = _s
                break
        else:
            raise ValueError(f'не найден делитель d={D} для subvecs')
    EMB_gpu = EMB.to(dev) if dev == 'cuda' else EMB
t0 = time.time(); CH = 1_000_000
# ВНЕШНИЙ режим: base уже построен энкодером (N_WIN, D) и нормирован. Здесь
# создавать его заново НЕЛЬЗЯ — это затрёт окна пустым тензором на N токенов
# (именно так упало: codes[7934] против base[300]). Аллоцируем только внутренний.
if _EXT_ENC is None:
    base = torch.empty(N, D, device=dev)
# контекстные ключи: среднее emb в окне радиуса RAD (монотокен-эмбеддинг без
# контекста не даёт семантики - GT-тест это доказал)
RAD = 4
if _EXT_ENC is None:
    for i in range(0, N, CH):
        e = min(i+CH, N)
        lo = max(0, i-RAD); hi = min(N, e+RAD)
        emb = EMB_gpu[toks[lo:hi]].detach()                # (len, D)
        emb_t = emb.t().unsqueeze(0)                       # (1, D, len)
        pad = torch.nn.functional.pad(emb_t, (RAD, RAD), mode='replicate')
        ctx = torch.nn.functional.avg_pool1d(pad, 2*RAD+1, stride=1).squeeze(0).t()  # (len, D)
        base[i:e] = ctx[i-lo:e-lo]
        del emb, pad, ctx
    print(f'keys (context-avg r={RAD}) за {time.time()-t0:.0f}s', flush=True)
else:
    # ВНЕШНИЙ путь: ключ уже готов (вектор окна от энкодера). Дополнительное
    # контекстное усреднение НЕ применяется: перекрытие окон (_eover) уже даёт
    # контекст, а усреднение по окнам размыло бы адрес. Векторы нормируем —
    # энкодеры обучены на косинус, и API в внешнем режиме нормирует запрос так же.
    base = base / (base.norm(dim=1, keepdim=True) + 1e-9)
    RAD = 0                    # meta: точечная позиция НЕ размыта -> tol против 0
    # ВАЖНО: в внешнем режиме адресуемая единица — ОКНО, а не токен. Всё, что ниже
    # (кодобук, codes, ids, бухгалтерия Б/единица), должно считаться по N окон,
    # иначе codes не совпадут по длине с base и архив будет битым.
    if N_WIN != N:
        print(f'внешний режим: адресов {N:,} токенов -> {N_WIN:,} окон', flush=True)
    N = N_WIN
    # КОНТРОЛЬНЫЕ ОКНА (Путь C, §5 шаг 2). Пишем в meta 5 окон и их векторы:
    # читатель повторит энкодер на этих же строках и сверит косинус. Без этого
    # нельзя отличить «энкодер тот» от «энкодер другой» — а разница между ними
    # измеренно фатальна (recall@8 0.000 у 15 чужих энкодеров).
    # Стоимость: 5 окон * (d + текст 800 симв.) ≈ 25 КБ — пренебрежимо на фоне
    # кодобука.
    # Содержимое: это ФРАГМЕНТЫ корпуса, которые и так лежат в ids.npy/сниппетах,
    # поэтому утечкой это не является (вектор — не ответ на запрос).
    # ВАЖНО (исправлено): текст сохраняем ЦЕЛИКОМ (_echunk символов). Здесь был
    # баг: я писал _texts[i][:400], а вектор — от полного окна 800. Проверка
    # кодировала обрезанный текст и получала cos 0.9559 вместо 1.000, из-за чего
    # верный энкодер выглядел «чужим». Что сохраняем — то и должно кодироваться.
    _probe_i = [int(x) for x in np.linspace(0, N - 1, 5).astype(int)] if N >= 5 else list(range(N))
    # СТРАХОВКА: _texts и base обязаны быть выровнены по индексу. Если фильтр
    # спанов что-то выбросил, _texts[i] — не то окно, и тест энкодера начнёт
    # врать (именно этот класс ошибки я только что поймал).
    assert len(_texts) == int(base.shape[0]), (
        f'рассинхрон: текстов окон {len(_texts)}, векторов {int(base.shape[0])} — '
        f'enc_probe будет указывать на чужие окна')
    # Векторы контрольных окон — в отдельный .npy, НЕ в JSON. Здесь был промах:
    # 5×1024 float в meta.json раздували её до 57 КБ текстом. meta — описание
    # архива, она должна читаться глазами, а не быть хранилищем векторов.
    _probe_vec = np.stack([base[i].cpu().numpy().astype(np.float32) for i in _probe_i])
    np.save(os.path.join(OUT, 'enc_probe.npy'), _probe_vec)
    _probes = [{'i': int(i), 'text': str(_texts[i])} for i in _probe_i]
    print(f'контрольные окна для теста энкодера: {_probe_i} '
          f'(векторы -> enc_probe.npy {_probe_vec.nbytes} Б)', flush=True)
    print(f'keys (external encoder, окон {N}, d={D}) готовы', flush=True)

fm = StreamFracode(D, levels=2, subvecs=S, K=K, device=dev)
calib = base[:1_000_000].clone()
t0 = time.time(); fm.fit(calib, iters=12, seed=0); del calib
print(f'кодобук за {time.time()-t0:.0f}s', flush=True)
codes = torch.empty(N, 2, S, dtype=torch.uint8, device=dev)
for i in range(0, N, CH):
    e = min(i+CH, N)
    codes[i:e] = fm.encode_rows(base[i:e]).to(torch.uint8)
np.save(os.path.join(OUT, 'codes.npy'), codes.cpu().numpy())
# keys_f16 (384 Б/токен) нужны ТОЛЬКО для реранка, а реранк ИЗМЕРЕННО ВРЕДИТ
# (ADC-only 0.900 против rerank 0.820, recall@8, v8-архив, _adc_vs_rr.py).
# FRK_NO_KEYS=1 -> архив без этого файла: 24 Б/токен СТАНОВИТСЯ реальным следом,
# а не только codes.npy. Для продукта это дефолт-рекомендация.
NO_KEYS = os.environ.get('FRK_NO_KEYS', '').strip() not in ('', '0', 'false', 'False')
if NO_KEYS:
    print('FRK_NO_KEYS=1: keys_f16 НЕ сохраняются (реранк не используется, '
          'след = 24 Б/токен + ids)', flush=True)
else:
    np.save(os.path.join(OUT, 'keys_f16.npy'), base.half().cpu().numpy())
del base   # 800 МБ fp16 могут быть уже не нужны
if _EXT_ENC is None:
    np.save(os.path.join(OUT, 'ids.npy'), ids_all[:N].astype(np.int32))
else:
    # ВНЕШНИЙ режим: адресуем ОКНА. Лекс-слой нужен для точного поиска слов, и
    # его ставим на середину каждого окна — тогда ids.npy лежит в том же
    # координатном пространстве, что codes (позиция = номер окна), и лекс-канал
    # возвращает те же позиции. Раньше здесь молча писался ids_all[:N] токенов —
    # при N окон это ДРУГИЕ позиции, и лекс-результаты уехали бы.
    _mid = np.array([(s + e) // 2 for s, e in _spans], dtype=np.int32)[:N]
    np.save(os.path.join(OUT, 'ids.npy'), _mid)
    print(f'ids.npy (внешний режим): {len(_mid)} позиций-середин окон', flush=True)
for l in range(2):
    np.save(os.path.join(OUT, f'cbook_l{l}.npy'),
            torch.stack(fm.cbooks[l]).cpu().numpy())   # (S, K, sub)


def _sz(name):
    p = os.path.join(OUT, name)
    return os.path.getsize(p) if os.path.exists(p) else 0


_b = {'codes.npy': _sz('codes.npy'), 'keys_f16.npy': _sz('keys_f16.npy'),
      'ids.npy': _sz('ids.npy'), 'cbook': _sz('cbook_l0.npy') + _sz('cbook_l1.npy')}
_total = sum(_b.values())
# ЧЕСТНАЯ БУХГАЛТЕРИЯ ПАМЯТИ (исправлено 12.09). Раньше `b_per_tok_total` считался
# как (коды+keys+ids+кодобук)/N и на любом архиве давал нелепость: кодобук имеет
# ФИКСИРОВАННЫЙ размер (S*L*K*sub*4 Б), поэтому на архиве из 7934 окон он давал
# 264 Б/окно и «итого 284 Б/окно» — число, которое пугает читателя и не имеет
# отношения к делу. Правильно показывать ДВЕ величины:
#   per_unit_resident — то, что растёт с архивом (это и есть «цена памяти»);
#   per_unit_amortized — цена кодобука, размазанная по N (падает с ростом архива).
_per_unit = round((_b['codes.npy'] + _b['keys_f16.npy'] + _b['ids.npy']) / N, 3)
_amort = round(_b['cbook'] / N, 3)
_total_amortized = round(_per_unit + _amort, 3)
_enc_meta = ({'encoder_kind': 'external',
              'encoder_model': _EXT_ENC['model'],
              'encoder_url': _EXT_ENC['url'],
              'encoder_chunk': _EXT_ENC['chunk'],
              'encoder_overlap': _EXT_ENC['overlap'],
              'emb_table_path': None,
              'unit': 'window',
              # РЕЕСТР ЭНКОДЕРОВ (Путь C, §5 шаг 1). Читатель обязан знать, ГДЕ
              # взять энкодер, иначе требование «энкодер едет с памятью» —
              # невыполнимое пожелание. Здесь — достаточная инструкция, чтобы
              # поднять нужную модель с нуля на чистой машине.
              'encoder_kb': {
                  'model': _EXT_ENC['model'],
                  'backend': 'ollama',
                  'how': f'ollama pull {_EXT_ENC["model"]}',
                  'api': _EXT_ENC['url'],
                  'd': int(_EXT_ENC['d']),
                  'normalize': True,       # ключи и запрос нормируются по L2
                  'unit': 'window',
                  'chunk': _EXT_ENC['chunk'],
                  'overlap': _EXT_ENC['overlap'],
                  'note': ('Память адресуется ВЕКТОРАМИ ЭТОГО энкодера. Другой '
                           'энкодер = другое пространство = recall 0.000 '
                           '(это измерено, не предположение).')},
              # контрольные окна: читатель повторит энкодер и сверит косинус
              # (API: _encoder_check -> отказ вместо тихого нуля)
              'enc_probe': _probes,
              # честная единица: внешний архив адресует ОКНА, не токены. Поля
              # b_per_tok_* ниже считаются на окно и названы так для совместимости.
              'unit_note': ('внешний энкодер: адрес = окно FRK_ENC_CHUNK; '
                            'B/единица = B/окно, не B/токен')}
             if _EXT_ENC is not None else
             {'encoder_kind': 'internal',
              'unit': 'token',
              'encoder_model': os.path.basename(_ckpt)})
json.dump({'N': N, 'S': S, 'K': K, 'D': D,
           'levels': 2,
           # честная бухгалтерия: 24 Б/токен — это ТОЛЬКО codes
           'b_per_tok_codes': round(_b['codes.npy'] / N, 3),
           'b_per_tok_keys': round(_b['keys_f16.npy'] / N, 3),
           'b_per_tok_ids': round(_b['ids.npy'] / N, 3),
           'b_per_tok_total': _total_amortized,
           # ЧЕСТНО: две отдельные величины вместо одной обманчивой.
           'b_per_unit_resident': _per_unit,
           'b_per_unit_cbook_amortized': _amort,
           'memory_note': ('b_per_unit_resident — то, что растёт с архивом (это '
                           'цена памяти). Кодобук фиксированного размера; его '
                           'вклад указан отдельно и падает с ростом N.'),
           # legacy: считалось с кодобуком -> на маленьком архиве завышает
           'b_per_tok_total_legacy': round(_total / N, 3),
           'bytes': _b, 'total_mb': round(_total / 1024**2, 1),
           'codes_mb': round(_b['codes.npy'] / 1024**2, 1),
           'keys_mb': round(_b['keys_f16.npy'] / 1024**2, 1),
           'b_per_tok': round(_b['codes.npy'] / N, 3),   # legacy-ключ = коды
           'fr_mb': round(_b['codes.npy'] / 1024**2),    # legacy-ключ = коды
           'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           # АБСОЛЮТНЫЙ путь: reader не должен угадывать, где лежит корпус.
           # Раньше писалось голое имя, и frakod_api подхватывал первый файл с
           # таким именем рядом с пакетом — то есть ЧУЖОЙ корпус (дефект 12.09).
           'corpus': os.path.abspath(CORPUS),
           'corpus_basename': os.path.basename(CORPUS),
           # ЧЕСТНО: чем именно токенизирован корпус — от этого зависит, каким
           # токенизатором обязан спрашивать API (иначе ids запроса не совпадут).
           'tokenizer': (os.path.basename(_tokfile) if _tokfile else 'make_bpe(corpus_public,512)'),
           'tokenizer_kind': ('external' if _tokfile else 'bpe512'),
           'tokenizer_path': (os.path.abspath(_tokfile) if _tokfile else None),
           'vocab': int(tok.get_vocab_size()),
           'key_context_rad': int(RAD),
           # ВНЕШНИЙ РЕЖИМ: ключи адресуют окна энкодера, а не таблицу токенов.
           # API обязан узнать об этом из meta (и не строить context-avg запрос).
           **_enc_meta,
           # ВАЖНО: пишем АБСОЛЮТНЫЙ путь. Раньше писался только basename, и если
           # чекпойнт лежал не рядом с API (напр. phase01/exp_vq/ckpt_v8_voc8k.pt),
           # API молча откатывался на sts_prog_seed0.pt — ключи архива и запрос
           # оказывались из РАЗНЫХ таблиц эмбеддингов (идентично корневой причине №1).
           'embed_ckpt': (os.path.abspath(_ckpt) if _ckpt
                          else f'external:{_EXT_ENC["model"]}' if _EXT_ENC else None),
           'embed_ckpt_name': (os.path.basename(_ckpt) if _ckpt
                               else (f'external:{_EXT_ENC["model"]}' if _EXT_ENC else None)),
           'codec': ('external encoder keys (нормированные, без pos) + FR 2 уровня'
                     if _EXT_ENC is not None else
                     'embed-only keys (без pos) + FR 2 уровня 12x8'),
           'storage_note': ('24 Б/токен = codes.npy. keys_f16 используется для '
                            'реранка (можно держать memmap с диска), ids — лекс-слой.')},
          open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
if _EXT_ENC is not None:
    # САМОДОСТАТОЧНОСТЬ — ЧЕСТНО. Тут была ошибка: я клал рядом `encoder_w.npy` =
    # ВСЕ векторы окон корпуса (15.5 МБ). Это дважды неверно:
    #   1) это утечка — архив хранил бы готовые ответы на свои же запросы;
    #   2) это 4 КБ/окно вместо 24 Б, весь смысл сжатия обнуляется.
    # Энкодер — это ВНЕШНЯЯ модель (bge-m3 в Ollama). Архив на внешнем энкодере
    # самодостаточным быть не может по определению: чтобы прочитать адрес, нужен
    # тот же энкодер. Поэтому meta фиксирует модель и URL, а не притворяется
    # автономным. Честная формулировка: «энкодер едет вместе с памятью» —
    # как зависимость, а не как копия корпуса.
    if os.path.exists(_cf):
        print(f'ЧЕСТНО: архив требует внешний энкодер {_emodel} ({_eurl}); '
              f'векторы окон НЕ сохраняются в архив (утечка + 4 КБ/окно). '
              f'Кэш {os.path.basename(_cf)} оставлен только как рабочий файл.',
              flush=True)
    _cf = None   # чтобы не оставлять в каталоге архива рабочую копию векторов
_unit = 'окно' if _EXT_ENC is not None else 'токен'
print(f'АРХИВ ГОТОВ: {N:,} {_unit}(-ов) | коды {_b["codes.npy"]/1024**2:.1f} МБ '
      f'| ids {_b["ids.npy"]/1024**2:.1f} МБ | кодобук {_b["cbook"]/1024**2:.1f} МБ '
      f'| ИТОГО {_total/1024**2:.1f} МБ\n'
      f'  ЦЕНА ПАМЯТИ: {_per_unit:.1f} Б/{_unit} (растёт с архивом) + '
      f'{_amort:.1f} Б/{_unit} кодобука (амортизируется; на 10M {_unit}ов '
      f'станет {_b["cbook"]/10_485_760:.2f}) -> {OUT}', flush=True)
