"""Замер задержки поиска через API/MCP (продуктовая метрика №1).

Меряем две величины РАЗДЕЛЬНО:
  t_http   — полный круг «агент → API → ответ» (то, что видит пользователь);
  ms_total — внутреннее время ADC-поиска (вклад Fracode, отдаёт сам /recall).
Разница — время внешнего энкодера (Ollama bge-m3), которое не относится к Fracode.

Контроль: доля ok-ответов обязана быть 1.0; иначе цифры не о чём.
"""
import json, os, time, statistics, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'hermes_sem_bench_big.json')
API = os.getenv('FRK_API', 'http://127.0.0.1:8000/recall')
SEED = 20260912


def post(qtext, k=8, rerank=False):
    body = json.dumps({'text': qtext, 'k': k, 'mode': 'vector',
                       'rerank': rerank}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    raw = urllib.request.urlopen(req, timeout=300).read()
    t1 = time.perf_counter()
    return json.loads(raw.decode()), (t1 - t0) * 1000.0


def pct(v, p):
    v = sorted(v)
    if not v:
        return 0.0
    i = min(len(v) - 1, max(0, int(round((p / 100.0) * (len(v) - 1)))))
    return float(v[i])


def stats(v):
    return {'n': len(v), 'mean': round(float(statistics.mean(v)), 2),
            'p50': round(pct(v, 50), 2), 'p90': round(pct(v, 90), 2),
            'p95': round(pct(v, 95), 2), 'p99': round(pct(v, 99), 2),
            'min': round(float(min(v)), 2), 'max': round(float(max(v)), 2)}


def run(tag, rerank=False, warmup=5):
    bench = json.load(open(BENCH, encoding='utf-8'))
    qs = [it['query'] for it in bench]
    print(f'[{tag}] вопросов: {len(qs)}, rerank={rerank}')

    cold = None
    for i, q in enumerate(qs[:warmup]):
        r, t = post(q, rerank=rerank)
        if i == 0:
            cold = (t, float(r.get('ms_total') or 0.0))
    print(f'[{tag}] прогрев OK; холодный первый вызов: http={cold[0]:.1f} мс, '
          f'adc={cold[1]:.1f} мс')

    http_ms, adc_ms, oks = [], [], []
    t_start = time.perf_counter()
    for q in qs:
        r, t = post(q, rerank=rerank)
        http_ms.append(t)
        adc_ms.append(float(r.get('ms_total') or 0.0))
        oks.append(1.0 if r.get('ok') else 0.0)
    wall = time.perf_counter() - t_start

    ok_rate = sum(oks) / len(oks)
    enc_ms = [h - a for h, a in zip(http_ms, adc_ms)]
    out = {'http_ms': stats(http_ms), 'adc_ms': stats(adc_ms),
           'encoder_ms_derived': stats(enc_ms),
           'ok_rate': round(ok_rate, 4),
           'throughput_qps': round(len(qs) / wall, 2),
           'wall_s': round(wall, 1)}
    print(f'[{tag}] ok={ok_rate:.3f}  http p50={out["http_ms"]["p50"]} '
          f'p95={out["http_ms"]["p95"]}  adc p50={out["adc_ms"]["p50"]} '
          f'p95={out["adc_ms"]["p95"]}  энкодер p50={out["encoder_ms_derived"]["p50"]}')
    return out


def archive_info():
    """N и размер архива — чтобы точка замера была привязана к масштабу."""
    try:
        base = API.rsplit('/', 1)[0]
        acc = json.loads(urllib.request.urlopen(base + '/index/accounting',
                                                timeout=30).read().decode())
        return int(acc.get('N') or 0), acc.get('total_mb'), acc.get('b_per_tok_total')
    except Exception:
        return 0, None, None


def main():
    N, mb, bpt = archive_info()
    print(f'архив: N={N}, {mb} МБ, {bpt} Б/токен')
    res = {'bench': os.path.basename(BENCH), 'api': API, 'seed': SEED,
           'N_windows': N, 'archive_mb': mb, 'b_per_tok_total': bpt,
           'note': ('http_ms — полный круг агент→API→ответ; adc_ms — только Fracode '
                    'ADC-поиск (из поля ms_total); encoder_ms_derived = http − adc, '
                    'это время Ollama bge-m3, к Fracode не относится.')}
    res['adc_only'] = run('adc', rerank=False)
    res['with_rerank'] = run('rerank', rerank=True)

    if res['adc_only']['ok_rate'] < 1.0 or res['with_rerank']['ok_rate'] < 1.0:
        print('ВНИМАНИЕ: ok_rate < 1.0 — цифры задержки неполные, часть запросов '
              'отвергнута. Сначала починить канал.')

    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    # имя по N, чтобы замеры на разных архивах НЕ перезаписывали друг друга
    outp = os.path.join(HERE, 'runs', f'latency_N{N}.json')
    json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', outp)


if __name__ == '__main__':
    main()
