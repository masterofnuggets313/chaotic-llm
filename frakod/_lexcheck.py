"""Проверка ложных срабатываний substring-fallback.

Берём 30 игл, для каждой считаем lex_find. Хит = игла реально внутри окна.
Смотрим: (a) сколько окон на иглу (лишние = шум), (b) precision@4 — доля
возвращённых окон, содержащих иглу (а не мусор).
"""
import os, sys, json
os.environ.setdefault('FRK_IDXD', 'G:/Migration/chaotic-llm/phase01/exp_vq/frakod_index')
os.environ.setdefault('FRK_TOK', 'G:/Migration/chaotic-llm/frakod/tok_v31.json')
os.environ.setdefault('FRK_CORPUS', 'G:/Migration/chaotic-llm/phase01/exp_vq/hermes_history_dd.txt')
sys.path.insert(0, 'G:/Migration/chaotic-llm/frakod')
import frakod_api as A

n = json.load(open('G:/Migration/chaotic-llm/phase01/exp_vq/hermes_needles.json', encoding='utf-8'))
A.build_lex()

tot_win = 0
tot_hitwin = 0
recall = 0
for x in n:
    ps = A.lex_find(x['query'], 4)
    tot_win += len(ps)
    inwin = 0
    for p, _ in ps:
        i0 = max(0, p - 24); i1 = min(len(A._LEX_STARTS) - 1, p + 24)
        c0 = int(A._LEX_STARTS[i0]); c1 = int(A._LEX_STARTS[i1])
        if x['needle_word'].lower() in A._LEX_TEXT[c0:c1].lower():
            inwin += 1
    tot_hitwin += inwin
    recall += inwin > 0
print(f'recall@4 (needle in a returned window) = {recall}/{len(n)} = {recall/len(n):.3f}')
print(f'precision: {tot_hitwin}/{tot_win} returned windows contain the needle'
      f' = {tot_hitwin/max(1,tot_win):.3f}')
print(f'avg windows returned per query = {tot_win/len(n):.2f}')
