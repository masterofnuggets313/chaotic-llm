"""Парное сравнение конфигураций выдачи: CTX=1 (±1 окно) против CTX=0 (одно окно).

Почему ПАРНОЕ, а не по непересекающимся CI: поиск в обоих прогонах ОДИН И ТОТ ЖЕ
(детерминированный архив, те же позиции — recall совпал до 4-го знака). Различается
только текст, который API возвращает вокруг найденной позиции. Значит выборки
зависимые, и независимые интервалы здесь избыточно консервативны.

Два теста:
  1. Мак-Немар (точный): b = стало находиться, c = перестало. При c = 0 эффект
     однонаправленный по построению (более широкий span ⊇ более узкий).
  2. Парный бутстрап по 207 сериям: средняя разность и 95% CI.

Вход: runs/bench_fact_ctx0.json и runs/bench_fact_ctx1.json (поле per_item).
"""
import json, os
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
SEED = 20260912


def load(name):
    return json.load(open(os.path.join(RUNS, name), encoding='utf-8'))


def mcnemar_exact(b, c):
    """Точный двусторонний Мак-Немар."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def paired_boot(a, b, n_boot=10000, seed=SEED):
    """a, b — посерийные 0/1 или доли. Возвращает (diff, CI)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))]


def main():
    c0, c1 = load('bench_fact_ctx0.json'), load('bench_fact_ctx1.json')
    out = {'note': ('CTX=1 (±1 окно вокруг найденной позиции) против CTX=0 (одно окно). '
                    'Поиск идентичен, меняется только выдача текста. '
                    'n = 207 серий, seed = 20260912.'),
           'files': ['runs/bench_fact_ctx0.json', 'runs/bench_fact_ctx1.json']}
    for k in ('k4', 'k8'):
        p0 = c0[k]['per_item']
        p1 = c1[k]['per_item']
        assert len(p0) == len(p1), 'разная длина прогонов'
        row = {}
        # САНПРОВЕРКА: поиск обязан быть тем же.
        same_pos = sum(1 for a, b in zip(p0, p1) if a['pos'] == b['pos'])
        row['identical_search_positions'] = f'{same_pos}/{len(p0)}'
        for m in ('rec', 'snip', 'rare'):
            a = [x[m] for x in p0]
            b = [x[m] for x in p1]
            d, ci = paired_boot(a, b)
            row[m] = {'ctx0': round(float(np.mean(a)), 4),
                      'ctx1': round(float(np.mean(b)), 4),
                      'diff': round(d, 4), 'ci95': [round(x, 4) for x in ci]}
        # Мак-Немар только для бинарной метрики сниппета
        a = [x['snip'] for x in p0]
        b = [x['snip'] for x in p1]
        bb = sum(1 for x, y in zip(a, b) if y > x)
        cc = sum(1 for x, y in zip(a, b) if y < x)
        row['mcnemar'] = {'gained': bb, 'lost': cc,
                          'p_exact': float(f'{mcnemar_exact(bb, cc):.3e}')}
        out[k] = row
        print(f'[{k}] поиск совпал: {row["identical_search_positions"]}')
        for m in ('rec', 'snip', 'rare'):
            r = row[m]
            print(f'   {m:5s}: {r["ctx0"]:.4f} → {r["ctx1"]:.4f}  '
                  f'Δ={r["diff"]:+.4f} {r["ci95"]}')
        print(f'   Мак-Немар (snip): выиграно {bb}, потеряно {cc}, '
              f'p = {row["mcnemar"]["p_exact"]:.3e}')
    outp = os.path.join(RUNS, 'fact_paired_ctx1_vs_ctx0.json')
    json.dump(out, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('сохранено:', outp)


if __name__ == '__main__':
    main()
