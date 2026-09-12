# -*- coding: utf-8 -*-
"""ПРЕДПОЛЁТНАЯ ПРОВЕРКА ночного пайплайна v8 (Fracode / STS-Prog).

Ничего не обучает и не пишет в артефакты (кроме отчёта, если попросить).
Отвечает на вопрос: «заведётся ли ночной прогон и не соврёт ли он?»

Запуск:
    python night_v8_preflight.py            # быстрый гейт
    python night_v8_preflight.py --with-ppl # + честный ppl текущего чекпойнта
    python night_v8_preflight.py --json out.json

Код возврата: 0 = все проверки пройдены (WARN допустимы), 1 = есть FAIL.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
EVQ = os.path.join(REPO, 'phase01', 'exp_vq')
PHASE = os.path.join(REPO, 'phase01')

CKPT = os.path.join(EVQ, 'ckpt_v8_voc8k.pt')
TOK = os.path.join(EVQ, 'tok_v8.json')
CORPUS_TRAIN = os.path.join(EVQ, 'hermes_history_dd.txt')
CORPUS_CODE = os.path.join(PHASE, 'corpus_stack_train.txt')
CORPUS_5M = os.path.join(PHASE, 'corpus5m_train.txt')
NEEDLES = os.path.join(EVQ, 'hermes_needles.json')
PPL_CORPUS = os.path.join(PHASE, 'corpus5m_test.txt')   # удержан, в обучение не входит
TRAIN = os.path.join(EVQ, 'train_v8_uplift.py')
BUILDER = os.path.join(HERE, 'frakod_index_build.py')
IDXD_V8 = os.path.join(HERE, 'frakod_index_v8')
IDXD_CODE = os.path.join(EVQ, 'frakod_index_code10m')

D_EXPECT = 192
LAYERS_EXPECT = 8
VOC_EXPECT = 8192

RESULTS = []


def rec(name, status, detail=''):
    RESULTS.append({'check': name, 'status': status, 'detail': str(detail)})
    icon = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL', 'INFO': 'INFO'}[status]
    print(f'[{icon}] {name}' + (f' — {detail}' if detail else ''), flush=True)


def check_deps():
    missing = []
    for m in ('numpy', 'torch', 'tokenizers', 'fastapi', 'uvicorn', 'pydantic'):
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    if missing:
        rec('зависимости интерпретатора', 'FAIL',
            f'{sys.executable} не имеет: {", ".join(missing)}')
        return False
    rec('зависимости интерпретатора', 'PASS',
        f'{sys.executable} (torch {__import__("torch").__version__})')
    return True


def check_files():
    ok = True
    for label, p in (('ckpt v8', CKPT), ('токенизатор v8', TOK),
                     ('корпус история', CORPUS_TRAIN), ('корпус stack-train', CORPUS_CODE),
                     ('корпус 5m', CORPUS_5M), ('иглы', NEEDLES),
                     ('удержанный корпус для ppl', PPL_CORPUS),
                     ('train_v8_uplift.py', TRAIN), ('frakod_index_build.py (frakod/)', BUILDER)):
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1024 ** 2
            rec(f'файл: {label}', 'PASS', f'{mb:.1f} МБ')
        else:
            rec(f'файл: {label}', 'FAIL', f'нет: {p}')
            ok = False
    return ok


def check_interpreter_choice():
    """Тот же выбор, что делает night_v8_pipeline.py, но с явной диагностикой."""
    py = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'hermes', 'hermes-agent',
                      'venv', 'Scripts', 'python.exe')
    if os.path.exists(py):
        rec('интерпретатор пайплайна', 'PASS', f'hermes venv: {py}')
    else:
        rec('интерпретатор пайплайна', 'WARN',
            f'hermes venv не найден, фолбэк на sys.executable: {sys.executable}')
    bad = r'C:\Python313\python.exe'
    if os.path.abspath(sys.executable).lower() == bad.lower():
        rec('лаунчер run_v8_night.bat', 'FAIL',
            'C:\\Python313 не имеет numpy/tokenizers/fastapi/uvicorn — этот лаунчер мёртв')
    else:
        rec('лаунчер run_v8_night.bat', 'WARN',
            'не использовать: он зовёт C:\\Python313\\python.exe (нет numpy/tokenizers)')
    return True


def check_gpu():
    try:
        import torch
    except Exception as e:
        rec('GPU', 'FAIL', f'torch не импортируется: {e}')
        return False
    if not torch.cuda.is_available():
        rec('GPU', 'FAIL', 'torch.cuda.is_available() = False — обучение уйдёт на CPU')
        return False
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024 ** 3
    rec('GPU', 'PASS' if used < 2.0 else 'WARN',
        f'{torch.cuda.get_device_name(0)}: занято {used:.2f} ГБ из {total / 1024 ** 3:.1f} ГБ'
        + ('' if used < 2.0 else ' — освободите VRAM (ollama/llama), иначе OOM'))
    return True


def check_disk():
    need = 8 * 1024 ** 3
    for label, p in (('G:', HERE), ('C:', os.environ.get('LOCALAPPDATA', 'C:\\'))):
        try:
            free = shutil.disk_usage(p).free
        except Exception as e:
            rec(f'диск {label}', 'WARN', str(e))
            continue
        rec(f'диск {label}', 'PASS' if free > need else 'FAIL',
            f'свободно {free / 1024 ** 3:.1f} ГБ (нужно ~8 ГБ на memmap+ckpt+лог)')
    return True


def check_stale_processes():
    """Параллельные прогоны — главный подозреваемый по «тихим» падениям."""
    try:
        ps = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=60)
        data = json.loads(ps.stdout or '[]')
        if isinstance(data, dict):
            data = [data]
    except Exception as e:
        rec('живые прогоны v8', 'WARN', f'не смог опросить процессы: {e}')
        return True
    me = os.getpid()
    live = []
    for p in data:
        pid = p.get('ProcessId')
        cl = (p.get('CommandLine') or '')
        if pid == me:
            continue
        # Опасны только те, кто пишет общий memmap/ckpt. Оркестраторы
        # (night_v8_run/_preflight) себя защищают lock-файлом, их не считаем.
        if 'train_v8_uplift' in cl or 'frakod_index_build' in cl:
            live.append(f"PID {pid}: {cl[:90]}")
    if live:
        rec('живые прогоны v8', 'FAIL',
            'уже запущено (перезапись общего memmap/ckpt => тихое падение): ' + ' | '.join(live))
        return False
    rec('живые прогоны v8', 'PASS', 'других train_v8/index_build/night_v8 нет')
    return True


def check_memmap_stale():
    for name in ('ids_v8.memmap',):
        p = os.path.join(EVQ, name)
        if os.path.exists(p):
            rec(f'устаревший {name}', 'WARN',
                f'есть ({os.path.getsize(p) / 1024 ** 2:.0f} МБ) — будет перезаписан; '
                f'удалите перед прогоном, если он остался от убитого процесса')
        else:
            rec(f'устаревший {name}', 'PASS', 'нет')
    return True


def check_ckpt_loads():
    """Самое важное: чекпойнт должен грузиться в ту же архитектуру без missing/unexpected."""
    try:
        import torch
        sys.path.insert(0, PHASE)
        sys.path.insert(0, EVQ)
        from models_pc import build_pc_model
        from tokenizers import Tokenizer
    except Exception as e:
        rec('чекпойнт -> архитектура', 'FAIL', f'импорт: {e}')
        return False
    try:
        tk = Tokenizer.from_file(TOK)
        v = tk.get_vocab_size()
        m = build_pc_model('pc', vocab=v, d=D_EXPECT, layers=LAYERS_EXPECT, k_init=1.2,
                           sync_steps=8, driver_mode='sts_prog', alpha=0.3, temp=0.3)
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        sd = ck.get('model', ck) if isinstance(ck, dict) else ck
        ld = m.load_state_dict(sd, strict=False)
        if ld.missing_keys or ld.unexpected_keys:
            rec('чекпойнт -> архитектура', 'FAIL',
                f'{len(ld.missing_keys)} missing / {len(ld.unexpected_keys)} unexpected '
                f'(пример: {(ld.missing_keys or ld.unexpected_keys)[:2]})')
            return False
        emb = sd.get('embed.weight')
        if emb is None:
            rec('чекпойнт -> архитектура', 'FAIL', 'в чекпойнте нет embed.weight')
            return False
        rec('чекпойнт -> архитектура', 'PASS',
            f'PurePCLM(d={D_EXPECT},l={LAYERS_EXPECT}) без missing/unexpected; '
            f'embed {tuple(emb.shape)}')
        if int(emb.shape[0]) != int(v):
            rec('vocab чекпойнт == vocab токенизатора', 'FAIL',
                f'ckpt {emb.shape[0]} != tokenizer {v} — ids запроса не совпадут с архивом')
            return False
        rec('vocab чекпойнт == vocab токенизатора', 'PASS', f'{v}')
        if int(emb.shape[1]) != D_EXPECT:
            rec('d чекпойнта', 'FAIL', f'{emb.shape[1]} != {D_EXPECT}')
            return False
        rec('d чекпойнта', 'PASS', f'{emb.shape[1]} (S=12 делит 192 -> ровно 24 Б/токен)')
        extra = [k for k in ('step', 'best_ppl', 'opt') if isinstance(ck, dict) and k in ck]
        rec('формат чекпойнта', 'PASS',
            ('расширенный (resume-совместимый): ' + ', '.join(extra)) if extra
            else 'плоский state_dict — так и задумано: его читают API и билдер; '
                 'resume-чекпойнт (model+opt+step) тренер кладёт в workdir прогона')
        return True
    except Exception as e:
        rec('чекпойнт -> архитектура', 'FAIL', f'{type(e).__name__}: {e}')
        return False


def check_index():
    """Согласованность архива v8 с парой (ckpt, tok). Именно тут был класс ошибки №1."""
    mp = os.path.join(IDXD_V8, 'meta.json')
    if not os.path.exists(mp):
        rec('архив v8', 'WARN', f'нет {mp} — пайплайн пересоберёт')
        return True
    try:
        m = json.load(open(mp, encoding='utf-8'))
    except Exception as e:
        rec('архив v8', 'FAIL', f'meta.json не читается: {e}')
        return False
    ok = True
    if int(m.get('D', 0)) != D_EXPECT:
        rec('архив v8: D', 'FAIL', f"D={m.get('D')} != {D_EXPECT}")
        ok = False
    else:
        rec('архив v8: D', 'PASS', f"{m['D']}, S={m.get('S')}")
    b = m.get('b_per_tok_codes')
    if b is None or abs(float(b) - 24.0) > 1e-6:
        rec('архив v8: 24 Б/токен', 'FAIL', f'b_per_tok_codes={b}')
        ok = False
    else:
        rec('архив v8: 24 Б/токен', 'PASS', f"{b} (полный след {m.get('b_per_tok_total')} Б/ток)")
    for key, want, label in (('embed_ckpt', CKPT, 'эмбеддер'),
                             ('tokenizer_path', TOK, 'токенизатор')):
        got = m.get(key)
        if got and os.path.abspath(got) == os.path.abspath(want):
            rec(f'архив v8: {label}', 'PASS', os.path.basename(got))
        else:
            rec(f'архив v8: {label}', 'FAIL',
                f'meta {key}={got!r} != ожидаемого {want} — ключи и запрос из разных таблиц')
            ok = False
    if not m.get('key_context_rad'):
        rec('архив v8: RAD', 'WARN', 'key_context_rad не записан -> API возьмёт 4 по умолчанию')
    else:
        rec('архив v8: RAD', 'PASS', f"key_context_rad={m['key_context_rad']}")
    return ok


def check_code_archive():
    """Код-архив отстал по формату: meta без tokenizer/embed_ckpt, corpus — проза."""
    mp = os.path.join(IDXD_CODE, 'meta.json')
    if not os.path.exists(mp):
        rec('код-архив', 'WARN', 'нет — шаг code-теста будет пропущен')
        return True
    m = json.load(open(mp, encoding='utf-8'))
    problems = []
    if not m.get('tokenizer_path') and not m.get('tokenizer'):
        problems.append('нет tokenizer_path -> API свалится на tok_v31.json (чужой словарь)')
    if not m.get('embed_ckpt'):
        problems.append('нет embed_ckpt -> векторный канал берёт sts_prog_seed0.pt')
    c = str(m.get('corpus', ''))
    if not c.endswith('.txt') or ' ' in c:
        problems.append(f'corpus={c!r} — это не путь к файлу, лекс-слой молча выключится')
    if problems:
        rec('код-архив (legacy meta)', 'WARN',
            '; '.join(problems) + ' => code-тест сегодня ГЛУШИТЬ или пересобрать frakod/frakod_index_build.py')
        return False
    rec('код-архив', 'PASS', 'meta в актуальном формате')
    return True


def check_builder_copy():
    """Копия билдера в exp_vq должна быть делегатом, а не legacy-реализацией."""
    a = os.path.join(EVQ, 'frakod_index_build.py')
    if not os.path.exists(a):
        rec('копия билдера в exp_vq', 'PASS', 'отсутствует (есть только каноническая)')
        return True
    src = open(a, encoding='utf-8', errors='replace').read()
    if 'DEPRECATED' in src and 'run_path' in src:
        rec('копия билдера в exp_vq', 'PASS', 'делегат на frakod/frakod_index_build.py')
        return True
    sa, sb = os.path.getsize(a), os.path.getsize(BUILDER)
    rec('копия билдера в exp_vq', 'FAIL',
        f'{sa} Б против {sb} Б — это САМОСТОЯТЕЛЬНАЯ СТАРАЯ версия (пишет legacy-meta '
        f'без tokenizer_path/embed_ckpt). Заменить делегатом или удалить.')
    return False


def check_api_copy():
    """Копия API в exp_vq должна быть делегатом: своя версия падала на /stats."""
    a = os.path.join(EVQ, 'frakod_api.py')
    if not os.path.exists(a):
        rec('копия API в exp_vq', 'PASS', 'отсутствует')
        return True
    src = open(a, encoding='utf-8', errors='replace').read()
    if 'DEPRECATED' in src and 'app = _mod.app' in src:
        rec('копия API в exp_vq', 'PASS', 'делегат на frakod/frakod_api.py')
        return True
    if 'load_embedder()' not in src.split("@app.get('/stats')")[-1][:400]:
        rec('копия API в exp_vq', 'FAIL',
            'самостоятельная старая версия: /stats обращается к _EM без load_embedder() '
            '-> 500 -> wait_api() решает, что API не поднялся. Заменить делегатом.')
        return False
    rec('копия API в exp_vq', 'WARN', 'самостоятельная копия (может отстать от канонической)')
    return True


def check_ppl():
    """Честный ppl на УДЕРЖАННОМ корпусе — та же метрика, что печатает пайплайн.

    Раньше пайплайн мерил ppl на txt[:400000] из hermes_history_dd.txt, а этот
    файл входит в обучение (вес x4) -> число было на увиденных данных.
    """
    try:
        import numpy as np
        import torch
        sys.path.insert(0, PHASE)
        sys.path.insert(0, EVQ)
        import final_benchmark as fb
        from models_pc import build_pc_model
        from tokenizers import Tokenizer
    except Exception as e:
        rec('ppl чекпойнта', 'FAIL', f'импорт: {e}')
        return False
    t0 = time.time()
    try:
        tk = Tokenizer.from_file(TOK)
        txt = fb.load_chars(PPL_CORPUS, 4_000_000)
        ids = np.array(tk.encode(txt).ids, dtype=np.int64)
        V = tk.get_vocab_size()
        ids = np.clip(ids, 0, V - 1)
        W = fb.W
        NCTX = 32
        m = build_pc_model('pc', vocab=V, d=D_EXPECT, layers=LAYERS_EXPECT,
                           k_init=1.2, sync_steps=8, driver_mode='sts_prog',
                           alpha=0.3, temp=0.3)
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        sd = ck.get('model', ck) if isinstance(ck, dict) else ck
        m.load_state_dict(sd, strict=False)
        m.eval()
        rng = np.random.default_rng(1234)          # тот же сид, что в пайплайне
        s = rng.integers(0, len(ids) - W - 1, size=NCTX)
        X = torch.tensor(np.stack([ids[i:i + W] for i in s]), dtype=torch.long)
        Y = torch.tensor(ids[s + W], dtype=torch.long)
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(m(X), Y)
        ppl = float(torch.exp(loss))
        lvl = 'PASS' if ppl < 300 else 'WARN'
        rec('ppl чекпойнта (удержанный корпус)', lvl,
            f'{ppl:.1f} за {time.time() - t0:.0f}с на {os.path.basename(PPL_CORPUS)} '
            f'({"уровень v7 ~40-100" if ppl < 300 else "высок: вектора, скорее всего, не оживут"})')
        return True
    except Exception as e:
        rec('ppl чекпойнта', 'FAIL', f'{type(e).__name__}: {e}')
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-ppl', action='store_true')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    print('=' * 78, flush=True)
    print('PREDFLIGHT v8 — ночной пайплайн Fracode/STS-Prog', flush=True)
    print(f'время: {time.strftime("%Y-%m-%d %H:%M:%S")}   python: {sys.executable}', flush=True)
    print('=' * 78, flush=True)

    check_deps()
    check_files()
    check_interpreter_choice()
    check_gpu()
    check_disk()
    check_stale_processes()
    check_memmap_stale()
    check_ckpt_loads()
    check_index()
    check_code_archive()
    check_builder_copy()
    check_api_copy()
    if a.with_ppl:
        check_ppl()

    n_fail = sum(1 for r in RESULTS if r['status'] == 'FAIL')
    n_warn = sum(1 for r in RESULTS if r['status'] == 'WARN')
    print('-' * 78, flush=True)
    print(f'ИТОГО: {len(RESULTS)} проверок, FAIL={n_fail}, WARN={n_warn}', flush=True)
    print('ВЕРДИКТ: ' + ('НЕ ЗАПУСКАТЬ, пока не исправлены FAIL' if n_fail else
                         'МОЖНО ЗАПУСКАТЬ (WARN прочитать глазами)'), flush=True)
    if a.json:
        json.dump({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'python': sys.executable,
                   'results': RESULTS, 'fails': n_fail, 'warns': n_warn},
                  open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'отчёт -> {a.json}', flush=True)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
