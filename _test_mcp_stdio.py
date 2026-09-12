"""Сквозной тест MCP-слоя: настоящий stdio-запуск `frakod_mcp.py` + JSON-RPC.

Зачем: продукт — это MCP-память для агента, а не HTTP-API. Можно сто раз
проверить /recall через curl и всё равно продать сломанный MCP: другой порт
по умолчанию (8781 против 8000), другой транспорт, свои обёртки инструментов.
Здесь поднимается настоящий процесс сервера и по нему идут настоящие вызовы.

Проверяется: initialize → tools/list → tools/call recall (текст НЕ пустой) →
index_stats → memory_manifest. Плюс негативный тест: битый FRK_API обязан
вернуть ok=False, а не уронить сервер.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:/Users/Geroin/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe'
ENV = dict(os.environ)
ENV['FRK_API'] = os.environ.get('FRK_TEST_API', 'http://127.0.0.1:8781')


def session(questions, want_text=True):
    p = subprocess.Popen([PY, '-u', os.path.join(HERE, 'frakod_mcp.py')],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=ENV, text=True,
                         encoding='utf-8', bufsize=1, cwd=HERE)

    def send(obj):
        p.stdin.write(json.dumps(obj, ensure_ascii=False) + '\n')
        p.stdin.flush()

    def recv(tag, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f'[{tag}] сервер закрыл stdout')
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f'[{tag}] таймаут {timeout}с')

    out = {}
    send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
          'params': {'protocolVersion': '2024-11-05',
                     'capabilities': {},
                     'clientInfo': {'name': 'test', 'version': '1'}}})
    r = recv('initialize')
    out['server'] = (r.get('result', {}).get('serverInfo') or {}).get('name')
    send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})

    send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
    r = recv('tools/list')
    tools = [t['name'] for t in r.get('result', {}).get('tools', [])]
    out['n_tools'] = len(tools)
    out['tools'] = tools

    for i, q in enumerate(questions):
        send({'jsonrpc': '2.0', 'id': 10 + i, 'method': 'tools/call',
              'params': {'name': 'recall', 'arguments': {'query': q, 'k': 4}}})
        r = recv(f'recall{i}')
        res = r.get('result', {})
        txt = ''
        try:
            payload = res.get('content', [{}])[0].get('text', '')
            d = json.loads(payload)
            txt = ' '.join(str(x.get('text', '')) for x in (d.get('results') or []))
            out[f'q{i}'] = {'ok': d.get('ok'), 'n_res': len(d.get('results') or []),
                            'chars': len(txt)}
        except Exception as e:
            out[f'q{i}'] = {'parse_error': f'{type(e).__name__}: {e}'}
        if want_text:
            assert out[f'q{i}'].get('chars', 0) > 0, f'пустой текст на q{i}: {q!r}'

    for name, tid in (('index_stats', 30), ('memory_manifest', 31)):
        send({'jsonrpc': '2.0', 'id': tid, 'method': 'tools/call',
              'params': {'name': name, 'arguments': {}}})
        r = recv(name)
        try:
            d = json.loads(r.get('result', {}).get('content', [{}])[0].get('text', ''))
            out[name] = {'ok': d.get('ok'), 'keys': len(d)}
        except Exception as e:
            out[name] = {'parse_error': f'{type(e).__name__}: {e}'}
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return out


def main():
    # Личный бенч (hermes_sem_bench_big.json) в публикацию не едет — он построен
    # на чужой переписке. Берём FRK_BENCH или демо, иначе захардкоженные строки:
    # тест обязан запускаться там, где нет никаких данных, кроме репозитория.
    bench_path = os.environ.get('FRK_BENCH') or os.path.join(HERE, 'demo_bench.json')
    if os.path.exists(bench_path):
        qs = [it['query'] for it in
              json.load(open(bench_path, encoding='utf-8'))[:5]]
    else:
        qs = ['какой тикет описывает проблему с очередью',
              'что решили по мониторингу метрики',
              'какой риск обсуждали в инциденте'][:3]
    print('вопросы из:', os.path.basename(bench_path) if os.path.exists(bench_path)
          else 'harcode')
    print('== MCP stdio,FRK_API =', ENV['FRK_API'])
    r = session(qs)
    print('сервер:', r['server'], '| инструментов:', r['n_tools'])
    print('инструменты:', ', '.join(r['tools']))
    for i in range(len(qs)):
        print(f"  q{i}: {r[f'q{i}']}  [{qs[i][:48]}]")
    print('  index_stats:', r['index_stats'])
    print('  memory_manifest:', r['memory_manifest'])

    print('\n== НЕГАТИВ: FRK_API на мёртвый порт')
    ENV['FRK_API'] = 'http://127.0.0.1:9'
    r2 = session(qs[:1], want_text=False)
    print('  q0:', r2['q0'])
    ok_neg = (r2['q0'].get('ok') is False) or ('error' in r2['q0'])
    print('  негативный тест пройден:', ok_neg)

    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    outp = os.path.join(HERE, 'runs', 'mcp_stdio_test.json')
    json.dump({'positive': r, 'negative': r2, 'negative_passed': bool(ok_neg)},
              open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', outp)


if __name__ == '__main__':
    main()
