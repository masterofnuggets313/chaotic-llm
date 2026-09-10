# -*- coding: utf-8 -*-
"""НОЧНОЙ ПАЙПЛАЙН: v8 (d=192, vocab=8192) -> 24 Б/токен + живой вектор.

ОДНА КОМАНДА (запускать ночью, по явной команде владельца):

    python night_v8_pipeline.py --steps 10000

Что делает по шагам:

  1. Обучает v8-uplift (PurePCLM d=192 l=8, BPE vocab=8192, смешанный корпус
     код+реальная история) на указанное число шагов. Проверяет, что чекпойнт
     грузится в модель БЕЗ missing/unexpected (иначе — стоп, не тратим ночь).
  2. Пересобирает архив на паре (ckpt_v8 + tok_v8) через frakod_index_build.py.
  3. Поднимает API на этом архиве, ждёт готовности.
  4. Прогоняет test_24b_proof.py и test_24b_code.py.
  5. Кладёт отчёт night_v8_report.json и печатает честную сводку:
     ppl до/после, b_per_tok_codes (должно быть РОВНО 24.0), recall по каналам.

ПОЧЕМУ ЭТО НУЖНО: на 2026-09-10 чекпойнт v8 — smoke на 100 шагов (ppl ~8254),
поэтому его архив даёт вектор 0.00. Пара (d=192, V=8192) архитектурно лучшая
(cos между разными текстами mean 0.04 против 0.30 у v7), не хватает обучения.
Это ЕДИНСТВЕННЫЙ известный путь получить 24 Б/токен и семантику одновременно.

Оценка времени (~1.09 с/шаг, RTX 3060 12GB, batch=48):
    8000 шагов  ~2.4 ч   (ppl ~100, уровень v7)
   10000 шагов  ~3.0 ч   (рекомендуемый минимум)
   20000 шагов  ~6.1 ч
   40000 шагов  ~12.1 ч  (полная рецептура)

ВАЖНО: скрипт train_v8_uplift.py перезаписывает ckpt_v8_voc8k.pt по лучшему ppl.
Пайплайн сам делает бэкап исходного чекпойнта перед обучением.
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
EVQ = os.path.join(REPO, 'phase01', 'exp_vq')
PY = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'hermes', 'hermes-agent',
                  'venv', 'Scripts', 'python.exe')
if not os.path.exists(PY):                       # фолбэк: любой python с uvicorn
    PY = sys.executable

CKPT = os.path.join(EVQ, 'ckpt_v8_voc8k.pt')
TOK = os.path.join(EVQ, 'tok_v8.json')
CORPUS = os.path.join(EVQ, 'hermes_history_dd.txt')
TRAIN = os.path.join(EVQ, 'train_v8_uplift.py')
IDXD = os.path.join(HERE, 'frakod_index_v8')
NEEDLES = os.path.join(EVQ, 'hermes_needles.json')
REPORT = os.path.join(HERE, 'night_v8_report.json')
PORT = 8790


def sh(cmd, **kw):
    print(f'\n$ {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, cwd=HERE, **kw)


def ppl_of(ckpt, tok, d, layers, corpus, ctx=40):
    """Честный ppl на фиксированном срезе корпуса, на верной конфигурации."""
    code = f'''
import sys, torch, numpy as np
sys.path.insert(0, r"{REPO}/phase01"); sys.path.insert(0, r"{EVQ}")
import final_benchmark as fb
from models_pc import build_pc_model
from tokenizers import Tokenizer
W = fb.W
txt = fb.load_chars(r"{corpus}", None)
tk = Tokenizer.from_file(r"{tok}")
ids = np.array(tk.encode(txt[:400000]).ids, dtype=np.int64)
m = build_pc_model('pc', vocab=tk.get_vocab_size(), d={d}, layers={layers},
                   k_init=1.2, sync_steps=8, driver_mode='sts_prog',
                   alpha=0.3, temp=0.3)
ck = torch.load(r"{ckpt}", map_location='cpu', weights_only=False)
sd = ck.get('model', ck) if isinstance(ck, dict) else ck
ld = m.load_state_dict(sd, strict=False)
assert not ld.missing_keys and not ld.unexpected_keys, (
    f'shape mismatch: {{len(ld.missing_keys)}} missing, {{len(ld.unexpected_keys)}} unexpected')
m.eval()
X = torch.tensor(np.stack([ids[i:i+W] for i in range(0,{ctx}*W,W)]), dtype=torch.long)
Y = torch.tensor([ids[i+W] for i in range(0,{ctx}*W,W)], dtype=torch.long)
with torch.no_grad():
    l = torch.nn.functional.cross_entropy(m(X), Y)
print(f'PPL={{float(torch.exp(l)):.1f}}')
'''
    r = subprocess.run([PY, '-c', code], capture_output=True, text=True, cwd=EVQ)
    for line in (r.stdout or '').splitlines():
        if line.startswith('PPL='):
            return float(line.split('=')[1])
    print(r.stdout[-800:], r.stderr[-800:], flush=True)
    return None


def wait_api(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + '/stats', timeout=3).read()
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--skip-train', action='store_true',
                    help='не обучать, только пересобрать и проверить текущий чекпойнт')
    a = ap.parse_args()

    rep = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'steps': a.steps, 'batch': a.batch}

    for p in (CKPT, TOK, CORPUS, TRAIN):
        if not os.path.exists(p):
            print(f'НЕТ ФАЙЛА: {p}', flush=True)
            return 1

    # --- 0. ppl ДО ---
    print('== ppl ДО ==', flush=True)
    rep['ppl_before'] = ppl_of(CKPT, TOK, 192, 8, CORPUS)

    # --- 1. бэкап + обучение ---
    if not a.skip_train:
        bak = CKPT + time.strftime('_%Y%m%d_%H%M%S.bak')
        shutil.copy2(CKPT, bak)
        print(f'бэкап чекпойнта -> {bak}', flush=True)
        rep['backup'] = os.path.basename(bak)
        r = sh(f'"{PY}" train_v8_uplift.py --steps {a.steps} --batch {a.batch}')
        if r.returncode != 0:
            print('ОБУЧЕНИЕ УПАЛО — стоп', flush=True)
            return 2

    # --- 2. ppl ПОСЛЕ (и проверка, что чекпойнт грузится без mismatch) ---
    print('\n== ppl ПОСЛЕ ==', flush=True)
    rep['ppl_after'] = ppl_of(CKPT, TOK, 192, 8, CORPUS)
    if rep['ppl_after'] is None:
        print('чекпойнт не грузится в PurePCLM(d=192,l=8) — архитектура не та, стоп', flush=True)
        return 3
    if rep['ppl_after'] > 300:
        print(f'ВНИМАНИЕ: ppl={rep["ppl_after"]:.0f} всё ещё высок (у v7 ~40-100). '
              f'Шагов, скорее всего, мало — вектор может не ожить.', flush=True)

    # --- 3. пересборка архива с абсолютным embed_ckpt ---
    env = dict(os.environ, FRK_TOK=TOK, FRK_CKPT=CKPT, FRK_CORPUS=CORPUS,
               FRK_OUTD=IDXD)
    r = subprocess.run([PY, 'frakod_index_build.py', '3000000'],
                       cwd=HERE, env=env, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-500:], flush=True)
    if r.returncode != 0:
        print('СБОРКА АРХИВА УПАЛА — стоп', flush=True)
        return 4
    meta = json.load(open(os.path.join(IDXD, 'meta.json'), encoding='utf-8'))
    rep['b_per_tok_codes'] = meta['b_per_tok_codes']
    rep['N'] = meta['N']
    rep['embed_ckpt'] = meta.get('embed_ckpt')
    rep['tokenizer'] = meta.get('tokenizer')
    if abs(meta['b_per_tok_codes'] - 24.0) > 1e-6:
        print(f'ВНИМАНИЕ: b_per_tok_codes={meta["b_per_tok_codes"]} != 24.0 '
              f'(d={meta["D"]}, S={meta["S"]}) — это НЕ 24 Б/токен', flush=True)

    # --- 4. API + доказательство ---
    srv = subprocess.Popen(
        [PY, '-m', 'uvicorn', 'frakod_api:app', '--port', str(PORT)],
        cwd=HERE, env=dict(os.environ, FRK_IDXD=IDXD, FRK_CORPUS=CORPUS,
                           PYTHONPATH=REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_api(f'http://127.0.0.1:{PORT}'):
            print('API не поднялся — стоп', flush=True)
            return 5
        r = subprocess.run([PY, 'test_24b_proof.py'], cwd=HERE,
                           env=dict(os.environ, FRK_API=f'http://127.0.0.1:{PORT}',
                                    FRK_NEEDLES=NEEDLES),
                           capture_output=True, text=True)
        print('\n--- proof (v8-архив) ---\n' + r.stdout[-2500:], flush=True)
        rep['proof_stdout'] = r.stdout[-4000:]
    finally:
        srv.terminate()

    # --- 4b. КОД-АРХИВ: отдельный сервер (у него свой токенизатор и ids!) ---
    # ВАЖНО: test_24b_code.py читает ids из код-архива; гонять его против
    # v8-сервера нельзя — там другой словарь, попаданий не будет by construction.
    code_idxd = os.path.join(REPO, 'phase01', 'exp_vq', 'frakod_index_code10m')
    if os.path.exists(code_idxd):
        code_corpus = os.path.join(REPO, 'phase01', 'exp_vq', 'corpus_code.txt')
        srv2 = subprocess.Popen(
            [PY, '-m', 'uvicorn', 'frakod_api:app', '--port', str(PORT + 1)],
            cwd=HERE, env=dict(os.environ, FRK_IDXD=code_idxd, FRK_CORPUS=code_corpus,
                               PYTHONPATH=REPO),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if wait_api(f'http://127.0.0.1:{PORT + 1}'):
                r = subprocess.run([PY, 'test_24b_code.py'], cwd=HERE,
                                   env=dict(os.environ,
                                            FRK_API=f'http://127.0.0.1:{PORT + 1}',
                                            FRK_IDXD=code_idxd),
                                   capture_output=True, text=True)
                print('\n--- code (10M-архив, ровно 24 Б/токен) ---\n' + r.stdout[-2500:],
                      flush=True)
                rep['code_stdout'] = r.stdout[-4000:]
            else:
                print('код-сервер не поднялся — пропускаю code-тест', flush=True)
        finally:
            srv2.terminate()

    # --- 5. отчёт ---
    rep['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(rep, open(REPORT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n== СВОДКА ==', flush=True)
    print(f'ppl: {rep.get("ppl_before")} -> {rep.get("ppl_after")}', flush=True)
    print(f'код: {rep.get("b_per_tok_codes")} Б/токен (N={rep.get("N"):,})', flush=True)
    print(f'эмбеддер: {rep.get("embed_ckpt")}', flush=True)
    print(f'отчёт -> {REPORT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
