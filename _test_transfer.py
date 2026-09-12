"""Сквозной тест переноса памяти: агент А экспортировал -> агент Б импортировал -> читает.

Пятая продуктовая заявка: «память переезжает от агента А к агенту Б». Проверяется
на ВНЕШНЕМ архиве (bge-m3 через Ollama) — то есть на том, что реально продаётся.

Схема:
  1) POST /memory/export_v2  -> пакет .frk1x на диске
  2) POST /memory/import_v2  -> разворачивает пакет в НОВЫЙ каталог-архив
  3) поднимается ВТОРОЙ API на этом каталоге (эмулирует машину Б) и делает recall
  4) сравниваются позиции и текст с ответом исходного API

Важная честность: для внешнего энкодера пакет НЕ самодостаточен — таблица
энкодера в архиве намеренно не лежит. Поэтому у Б должен быть тот же Ollama +
bge-m3. Тест это явно проверяет и явно пишет в отчёт.
"""
import json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:/Users/Geroin/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe'
API = os.environ.get('FRK_TEST_API', 'http://127.0.0.1:8781')
PORT_B = int(os.environ.get('FRK_TEST_PORT_B', '8782'))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None, method='POST'):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    return json.loads(OPENER.open(req, timeout=300).read().decode())


def recall(port, q, k=4):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/recall',
                                 data=json.dumps({'text': q, 'k': k,
                                                  'mode': 'vector',
                                                  'rerank': True}).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(OPENER.open(req, timeout=300).read().decode())


def main():
    out = {'api_a': API}
    tmp = os.path.join(HERE, 'runs', 'transfer_test')
    os.makedirs(tmp, exist_ok=True)
    pkg = os.path.join(tmp, 'pkg.frk1x').replace('\\', '/')

    t0 = time.time()
    e = call('/memory/export_v2', {'path': pkg, 'include_encoder': True,
                                   'include_store': True, 'compress': True})
    # ВАЖНО: сервер дописывает '.npz', если имя так не кончается (frakod_api.py:1591).
    # Брать надо путь ИЗ ОТВЕТА, а не тот, что посылали — иначе «экспорт ok»,
    # а файл ищется не там и импорт падает с «нет файла».
    pkg = e.get('path') or pkg
    out['export'] = {'ok': e.get('ok'), 'ms': round((time.time() - t0) * 1000),
                     'path': pkg,
                     'bytes': os.path.getsize(pkg) if os.path.exists(pkg) else 0,
                     'encoder_bundled': bool((e.get('manifest') or {}).get('encoder')),
                     'error': e.get('error')}
    print('экспорт:', out['export'])
    if not e.get('ok'):
        raise SystemExit('экспорт не удался')

    t0 = time.time()
    imp = call('/memory/import_v2', {'path': pkg, 'name': 'frakod_index_transfer_B'})
    out['import'] = {'ok': imp.get('ok'), 'ms': round((time.time() - t0) * 1000),
                     'dir': imp.get('dir') or imp.get('idx_dir'),
                     'error': imp.get('error')}
    print('импорт:', out['import'])
    if not imp.get('ok'):
        print('ПОЛНЫЙ ОТВЕТ ИМПОРТА:', json.dumps(imp, ensure_ascii=False)[:800])
        raise SystemExit('импорт не удался')

    dst = os.path.join(HERE, 'frakod_index_transfer_B')
    out['b_dir'] = dst
    env = dict(os.environ)
    env.update({'FRK_IDXD': dst, 'FRK_WIN_CTX': '1', 'FRK_TXT_CAP': '2000',
                'FRK_PORT': str(PORT_B)})
    p = subprocess.Popen([PY, '-u', os.path.join(HERE, 'frakod_api.py')],
                         env=env, cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace')
    started = False
    for _ in range(60):
        time.sleep(1)
        try:
            r = recall(PORT_B, 'тест', k=1)
            if r.get('ok'):
                started = True
                break
        except Exception:
            pass
    out['b_started'] = started
    print('API агента Б поднят:', started)
    if not started:
        p.kill()
        raise SystemExit('API Б не поднялся')

    bench = json.load(open(os.path.join(HERE, 'hermes_sem_bench_big.json'),
                           encoding='utf-8'))[:20]
    same_pos, same_txt, hits = 0, 0, 0
    for it in bench:
        ra = recall(8781, it['query'], k=4)
        rb = recall(PORT_B, it['query'], k=4)
        pa = [x['position'] for x in (ra.get('results') or [])]
        pb = [x['position'] for x in (rb.get('results') or [])]
        ta = (ra.get('results') or [{}])[0].get('text', '')
        tb = (rb.get('results') or [{}])[0].get('text', '')
        if pa == pb:
            same_pos += 1
        if ta and ta == tb:
            same_txt += 1
        if ta:
            hits += 1
    n = len(bench)
    out['compare'] = {'n_queries': n, 'same_positions': f'{same_pos}/{n}',
                      'same_text': f'{same_txt}/{n}', 'a_nonempty_text': f'{hits}/{n}'}
    print('сравнение А vs Б:', out['compare'])
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()

    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    outp = os.path.join(HERE, 'runs', 'transfer_roundtrip.json')
    json.dump(out, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', outp)


if __name__ == '__main__':
    main()
