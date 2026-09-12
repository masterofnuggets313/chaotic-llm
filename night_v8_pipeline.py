# -*- coding: utf-8 -*-
"""НОЧНОЙ ПАЙПЛАЙН: v8 (d=192, vocab=8192) -> 24 Б/токен + живой вектор.

ОДНА КОМАНДА (запускать ночью, по явной команде владельца):

    python night_v8_pipeline.py --steps 40000

Что делает по шагам:

  1. Предполётный гейт (night_v8_preflight.py): зависимости, GPU, диск, живые
     конкурирующие процессы, загрузка чекпойнта, сверка meta.json архива.
     Не прошёл — стоп, ночь не тратим (--force чтобы всё равно запустить).
  2. Обучает v8-uplift (PurePCLM d=192 l=8, BPE vocab=8192, смешанный корпус
     код+реальная история) на указанное число шагов.
  3. Меряет честный ppl ДО и ПОСЛЕ — на УДЕРЖАННОМ корпусе
     (phase01/corpus5m_test.txt, в обучение не входит).
  4. Пересобирает архив на паре (ckpt_v8 + tok_v8) через frakod_index_build.py.
  5. Поднимает API на этом архиве, ждёт готовности, гоняет test_24b_proof.py
     (и, если мета код-архива актуальна, test_24b_code.py).
  6. Кладёт отчёт night_v8_report.json и печатает честную сводку:
     ppl до/после, b_per_tok_codes (должно быть РОВНО 24.0), recall по каналам.

ПОЧЕМУ ЭТО НУЖНО: на 2026-09-10 чекпойнт v8 был smoke на 100 шагов (ppl ~8254),
поэтому его архив давал вектор 0.00. Пара (d=192, V=8192) архитектурно лучшая
(cos между разными текстами mean 0.04 против 0.30 у v7), не хватает обучения.

Оценка времени (ФАКТ, RTX 3060 12GB, batch=48): ~0.05 с/шаг, то есть
    8000 шагов  ~7 мин      40000 шагов ~40 мин      200000 шагов ~3.2 ч
(В прежней версии docstring стояло «1.09 с/шаг, 40000 шагов ~12.1 ч» — завышено
примерно в 20 раз; из-за этого ночь планировалась неверно.)

ИСПРАВЛЕНО 2026-09-11 (разбор падения, см. NIGHT_V8_READINESS_2026-09-11.md):
  I2  stderr нигде не сохранялся -> падения были «тихими». Теперь stdout+stderr
      каждого шага построчно пишутся в runs/<stamp>/*.log, включая лог API
      (раньше он уходил в DEVNULL, и крэш API выглядел как «API не поднялся»).
  I3  прогоны перезаписывали общие ids_v8.memmap и ckpt_v8_voc8k.pt. Теперь у
      каждого прогона своя папка runs/<stamp>/, memmap живёт в ней.
  I4  отчёт писался ТОЛЬКО на успехе, поэтому провал выглядел как успех
      (в night_v8_report.json лежал результат прошлого удачного прогона).
      Теперь отчёт пишется ВСЕГДА и содержит status/stage/exit_code.
  I5  добавлен --resume (полный чекпойнт: model+opt+step).
  I6  добавлен lock-файл: второй прогон не стартует, пока жив первый.
  I8  лог API сохраняется в файл.
  Дочерние процессы убиваются деревом (taskkill /T) — нет осиротевших обучений.
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
EVQ = os.path.join(REPO, 'phase01', 'exp_vq')
PHASE = os.path.join(REPO, 'phase01')
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
PREFLIGHT = os.path.join(HERE, 'night_v8_preflight.py')
PPL_CORPUS = os.path.join(PHASE, 'corpus5m_test.txt')   # удержан, в обучение не входит
CODE_IDXD = os.path.join(EVQ, 'frakod_index_code10m')
RUNS = os.path.join(HERE, 'runs')
LOCK = os.path.join(RUNS, '.night_v8.lock')
PORT = 8790

CHILDREN = []


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run_logged(cmd, logpath, env=None, cwd=None, label=''):
    """Запуск с построчным tee stdout+stderr в лог (I2). -> (rc, elapsed)."""
    t0 = time.time()
    log(f'$ {label or " ".join(cmd)}')
    with open(logpath, 'a', encoding='utf-8', errors='replace') as lf:
        lf.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} $ {" ".join(cmd)}\n')
        lf.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding='utf-8', errors='replace')
        CHILDREN.append(p)
        try:
            for line in p.stdout:
                lf.write(line); lf.flush()
                sys.stdout.write(line); sys.stdout.flush()
        finally:
            rc = p.wait()
    el = time.time() - t0
    log(f'rc={rc} за {el:.0f}с -> {os.path.basename(logpath)}')
    return rc, el


def kill_tree(p):
    if p is None or p.poll() is not None:
        return
    try:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)],
                       capture_output=True, timeout=60)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass


def ppl_of(run_dir, ckpt, tok, corpus, tag):
    """Честный ppl: на УДЕРЖАННОМ корпусе (I4/B2). Возвращает float|None."""
    code = f'''
import sys, torch, numpy as np
sys.path.insert(0, r"{PHASE}"); sys.path.insert(0, r"{EVQ}")
import final_benchmark as fb
from models_pc import build_pc_model
from tokenizers import Tokenizer
W = fb.W
txt = fb.load_chars(r"{corpus}", 4_000_000)
tk = Tokenizer.from_file(r"{tok}")
ids = np.array(tk.encode(txt).ids, dtype=np.int64)
V = tk.get_vocab_size()
ids = np.clip(ids, 0, V - 1)
m = build_pc_model('pc', vocab=V, d=192, layers=8, k_init=1.2,
                   sync_steps=8, driver_mode='sts_prog', alpha=0.3, temp=0.3)
ck = torch.load(r"{ckpt}", map_location='cpu', weights_only=False)
sd = ck.get('model', ck) if isinstance(ck, dict) else ck
ld = m.load_state_dict(sd, strict=False)
assert not ld.missing_keys and not ld.unexpected_keys, (
    f'shape mismatch: {{len(ld.missing_keys)}} missing, {{len(ld.unexpected_keys)}} unexpected')
m.eval()
rng = np.random.default_rng(1234)
s = rng.integers(0, len(ids) - W - 1, size=32)
X = torch.tensor(np.stack([ids[i:i+W] for i in s]), dtype=torch.long)
Y = torch.tensor(ids[s + W], dtype=torch.long)
with torch.no_grad():
    l = torch.nn.functional.cross_entropy(m(X), Y)
print(f'PPL={{float(torch.exp(l)):.1f}}')
'''
    lp = os.path.join(run_dir, f'ppl_{tag}.log')
    with open(lp, 'w', encoding='utf-8', errors='replace') as lf:
        r = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                           cwd=EVQ, timeout=1800)
        lf.write((r.stdout or '') + '\n' + (r.stderr or ''))
    for line in (r.stdout or '').splitlines():
        if line.startswith('PPL='):
            return float(line.split('=')[1])
    log(f'ppl({tag}) не посчитан — смотри {os.path.basename(lp)}')
    return None


def wait_api(url, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url + '/stats', timeout=3).read()
            return True
        except Exception:
            time.sleep(3)
    return False


def code_archive_is_modern():
    """legacy-мета код-архива -> code-тест врёт (нет tokenizer_path/embed_ckpt)."""
    mp = os.path.join(CODE_IDXD, 'meta.json')
    if not os.path.exists(mp):
        return False
    try:
        m = json.load(open(mp, encoding='utf-8'))
    except Exception:
        return False
    return (bool(m.get('tokenizer_path')) and bool(m.get('embed_ckpt'))
            and str(m.get('corpus', '')).endswith('.txt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--select-by', choices=['val', 'train'], default='val')
    ap.add_argument('--skip-train', action='store_true',
                    help='не обучать, только пересобрать и проверить текущий чекпойнт')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-preflight', action='store_true')
    ap.add_argument('--force', action='store_true', help='игнорировать FAIL preflight')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--with-code-test', action='store_true')
    a = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    run_dir = os.path.join(RUNS, stamp)
    os.makedirs(run_dir, exist_ok=True)

    rep = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'status': 'RUNNING',
           'stage': 'init', 'run_dir': run_dir, 'steps': a.steps, 'batch': a.batch,
           'seed': a.seed, 'select_by': a.select_by, 'warnings': [],
           'ppl_corpus': PPL_CORPUS}

    def finish(status, stage, rc=0):
        rep.update(status=status, stage=stage, exit_code=rc,
                   finished=time.strftime('%Y-%m-%dT%H:%M:%S'))
        for p in (os.path.join(run_dir, 'report.json'), REPORT):
            json.dump(rep, open(p, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
        log(f'ОТЧЁТ ({status}/{stage}) -> {run_dir}/report.json')
        return rc

    # --- I6: lock ---
    if os.path.exists(LOCK):
        try:
            old = json.load(open(LOCK, encoding='utf-8'))
            pid = int(old.get('pid', 0))
            alive = False
            if pid:
                r = subprocess.run(['powershell', '-NoProfile', '-Command',
                                    f'(Get-Process -Id {pid} -ErrorAction SilentlyContinue)'
                                    f' | Measure-Object | Select-Object -ExpandProperty Count'],
                                   capture_output=True, text=True, timeout=60)
                alive = (r.stdout or '').strip() not in ('', '0')
            if alive:
                log(f'УЖЕ ИДЁТ прогон PID {pid} (с {old.get("started")}). Второй прогон '
                    f'сломает общий memmap/ckpt -> выход')
                return finish('ABORTED', 'lock', 9)
            log(f'снят протухший lock от PID {pid}')
        except Exception as e:
            log(f'lock нечитаем ({e}) — перезаписываю')
    json.dump({'pid': os.getpid(), 'started': rep['started']},
              open(LOCK, 'w', encoding='utf-8'))

    try:
        if not os.path.exists(PY):
            log(f'НЕТ ИНТЕРПРЕТАТОРА: {PY}')
            return finish('FAILED', 'interpreter', 3)
        rep['python'] = PY
        log(f'интерпретатор: {PY}')

        for p in (CKPT, TOK, CORPUS, TRAIN):
            if not os.path.exists(p):
                log(f'НЕТ ФАЙЛА: {p}')
                return finish('FAILED', 'files', 3)

        # --- 1. предполётный гейт ---
        if not a.no_preflight:
            rep['stage'] = 'preflight'
            jf = os.path.join(run_dir, 'preflight.json')
            rc, _ = run_logged([PY, PREFLIGHT, '--with-ppl', '--json', jf],
                               os.path.join(run_dir, 'preflight.log'), cwd=HERE,
                               label='PREFLIGHT')
            if os.path.exists(jf):
                pf = json.load(open(jf, encoding='utf-8'))
                rep['preflight_fails'] = pf.get('fails')
                rep['preflight_warns'] = pf.get('warns')
            if rc != 0 and not a.force:
                log('PREFLIGHT НЕ ПРОЙДЕН — стоп (--force чтобы всё равно запустить)')
                return finish('FAILED', 'preflight', 4)
        if a.preflight_only:
            return finish('OK', 'preflight-only', 0)

        # --- 2. ppl ДО ---
        log('== ppl ДО ==')
        rep['ppl_before'] = ppl_of(run_dir, CKPT, TOK, PPL_CORPUS, 'before')

        # --- 3. бэкап + обучение ---
        if not a.skip_train:
            rep['stage'] = 'train'
            bak = os.path.join(run_dir, 'ckpt_before.pt')
            shutil.copy2(CKPT, bak)
            rep['ckpt_before'] = bak
            cmd = [PY, TRAIN, '--steps', str(a.steps), '--batch', str(a.batch),
                   '--lr', str(a.lr), '--seed', str(a.seed),
                   '--select-by', a.select_by, '--workdir', run_dir,
                   '--tok', TOK, '--ckpt-out', CKPT]
            if a.resume:
                cmd.append('--resume')
            rc, el = run_logged(cmd, os.path.join(run_dir, 'train.log'),
                                cwd=EVQ, label='TRAIN')
            rep['train_rc'], rep['train_s'] = rc, round(el)
            summ = os.path.join(run_dir, 'train_summary.json')
            if os.path.exists(summ):
                rep['train_summary'] = json.load(open(summ, encoding='utf-8'))
            if rc != 0:
                log('ОБУЧЕНИЕ УПАЛО — стоп. Полный трейсбек: train.log')
                return finish('FAILED', 'train', 2)

        # --- 4. ppl ПОСЛЕ (+ проверка, что чекпойнт грузится без mismatch) ---
        log('== ppl ПОСЛЕ ==')
        rep['stage'] = 'ppl_after'
        rep['ppl_after'] = ppl_of(run_dir, CKPT, TOK, PPL_CORPUS, 'after')
        if rep['ppl_after'] is None:
            log('чекпойнт не грузится в PurePCLM(d=192,l=8) — архитектура не та, стоп')
            return finish('FAILED', 'ppl_after', 3)
        if rep['ppl_after'] > 300:
            w = (f'ppl={rep["ppl_after"]:.0f} высок (у v7 ~40-100) — '
                 f'вектор может не ожить')
            rep['warnings'].append(w)
            log('ВНИМАНИЕ: ' + w)

        # --- 5. пересборка архива (канонический билдер, абсолютный embed_ckpt) ---
        rep['stage'] = 'index'
        env = dict(os.environ, FRK_TOK=TOK, FRK_CKPT=CKPT, FRK_CORPUS=CORPUS,
                   FRK_OUTD=IDXD)
        rc, _ = run_logged([PY, 'frakod_index_build.py', '3000000'],
                           os.path.join(run_dir, 'index_build.log'), env=env,
                           cwd=HERE, label='INDEX-BUILD')
        rep['index_rc'] = rc
        if rc != 0:
            log('СБОРКА АРХИВА УПАЛА — стоп')
            return finish('FAILED', 'index', 4)
        meta = json.load(open(os.path.join(IDXD, 'meta.json'), encoding='utf-8'))
        rep['index'] = {k: meta.get(k) for k in
                        ('N', 'S', 'K', 'D', 'b_per_tok_codes', 'b_per_tok_total',
                         'embed_ckpt', 'tokenizer', 'key_context_rad', 'built_utc')}
        # допуск: 128 Б заголовка .npy -> +128/N к «Б/токен». На N=3M это 4e-5,
        # на коротком архиве заметно; раньше порог 1e-6 давал ложное предупреждение.
        _tol = 128.0 / max(1, int(meta.get('N') or 1)) + 1e-6
        if abs(float(meta['b_per_tok_codes']) - 24.0) > _tol:
            w = f'b_per_tok_codes={meta["b_per_tok_codes"]} != 24.0 (D={meta["D"]}, S={meta["S"]})'
            rep['warnings'].append(w)
            log('ВНИМАНИЕ: ' + w)
        for k, want in (('embed_ckpt', CKPT), ('tokenizer_path', TOK)):
            if os.path.abspath(str(meta.get(k) or '')) != os.path.abspath(want):
                w = f'meta.{k}={meta.get(k)!r} != {want} — ключи и запрос из разных таблиц'
                rep['warnings'].append(w)
                log('ВНИМАНИЕ: ' + w)

        # --- 6. API + доказательство ---
        rep['stage'] = 'proof'
        srv = subprocess.Popen(
            [PY, '-m', 'uvicorn', 'frakod_api:app', '--port', str(PORT)],
            cwd=HERE, env=dict(os.environ, FRK_IDXD=IDXD, FRK_CORPUS=CORPUS,
                               PYTHONPATH=REPO),
            stdout=open(os.path.join(run_dir, 'api.out.log'), 'w', encoding='utf-8'),
            stderr=subprocess.STDOUT)
        CHILDREN.append(srv)
        try:
            if not wait_api(f'http://127.0.0.1:{PORT}'):
                log('API не поднялся — смотри api.out.log')
                return finish('FAILED', 'api', 5)
            rc, _ = run_logged([PY, 'test_24b_proof.py'],
                               os.path.join(run_dir, 'proof.log'),
                               env=dict(os.environ, FRK_API=f'http://127.0.0.1:{PORT}',
                                        FRK_NEEDLES=NEEDLES),
                               cwd=HERE, label='PROOF')
            rep['proof_rc'] = rc
            gt = os.path.join(HERE, 'gt_24b_proof.json')
            if os.path.exists(gt):
                g = json.load(open(gt, encoding='utf-8'))
                rep['proof_recall'] = {r['label']: r['recall'] for r in g.get('rows', [])}
        finally:
            kill_tree(srv)

        # --- 6b. КОД-АРХИВ: только если мета актуальна (иначе числа ложные) ---
        if a.with_code_test:
            if not code_archive_is_modern():
                w = ('code-тест пропущен: legacy-мета код-архива (нет tokenizer_path/'
                     'embed_ckpt, corpus не путь) — числа были бы ложные')
                rep['warnings'].append(w)
                log('code-тест ПРОПУЩЕН: ' + w)
            elif os.path.exists(CODE_IDXD):
                rep['stage'] = 'code'
                srv2 = subprocess.Popen(
                    [PY, '-m', 'uvicorn', 'frakod_api:app', '--port', str(PORT + 1)],
                    cwd=HERE, env=dict(os.environ, FRK_IDXD=CODE_IDXD,
                                       PYTHONPATH=REPO),
                    stdout=open(os.path.join(run_dir, 'api_code.out.log'), 'w',
                                encoding='utf-8'), stderr=subprocess.STDOUT)
                CHILDREN.append(srv2)
                try:
                    if wait_api(f'http://127.0.0.1:{PORT + 1}'):
                        rc, _ = run_logged([PY, 'test_24b_code.py'],
                                           os.path.join(run_dir, 'code.log'),
                                           env=dict(os.environ,
                                                    FRK_API=f'http://127.0.0.1:{PORT + 1}',
                                                    FRK_IDXD=CODE_IDXD),
                                           cwd=HERE, label='CODE')
                        rep['code_rc'] = rc
                    else:
                        rep['warnings'].append('код-сервер не поднялся')
                finally:
                    kill_tree(srv2)

        # --- 7. сводка ---
        rep['stage'] = 'done'
        log('=' * 70)
        log(f'ppl (удержанный корпус): {rep.get("ppl_before")} -> {rep.get("ppl_after")}')
        if rep.get('index'):
            log(f'код: {rep["index"].get("b_per_tok_codes")} Б/токен '
                f'(N={rep["index"].get("N"):,}), полный след '
                f'{rep["index"].get("b_per_tok_total")} Б/ток')
        log(f'recall: {rep.get("proof_recall")}')
        if rep['warnings']:
            log('предупреждения: ' + '; '.join(rep['warnings']))
        return finish('OK', 'done', 0)
    except KeyboardInterrupt:
        log('прервано пользователем')
        return finish('ABORTED', rep.get('stage', '?'), 130)
    except Exception as e:
        import traceback
        traceback.print_exc()
        rep['exception'] = f'{type(e).__name__}: {e}'
        return finish('FAILED', rep.get('stage', '?'), 1)
    finally:
        for p in CHILDREN:
            kill_tree(p)
        # снятие lock: сначала пробуем удалить, но если удаление недоступно
        # (политика ФС/антивирус), помечаем файл как освобождённый — stale-lock
        # всё равно распознаётся по мёртвому PID
        try:
            os.remove(LOCK)
        except Exception:
            try:
                json.dump({'pid': 0, 'released': True,
                           'at': time.strftime('%Y-%m-%dT%H:%M:%S')},
                          open(LOCK, 'w', encoding='utf-8'))
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
