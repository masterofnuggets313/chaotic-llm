"""Проверка ранжирования лекс-слоя: где именно стоит игла в топ-4 и куда
раньше вытеснялась. Плюс — сколько окон содержат иглу на позиции #1."""
import os, sys, json
os.environ.setdefault('FRK_IDXD', 'G:/Migration/chaotic-llm/phase01/exp_vq/frakod_index')
os.environ.setdefault('FRK_TOK', 'G:/Migration/chaotic-llm/frakod/tok_v31.json')
os.environ.setdefault('FRK_CORPUS', 'G:/Migration/chaotic-llm/phase01/exp_vq/hermes_history_dd.txt')
sys.path.insert(0, 'G:/Migration/chaotic-llm/frakod')
import frakod_api as A

n = json.load(open('G:/Migration/chaotic-llm/phase01/exp_vq/hermes_needles.json', encoding='utf-8'))
A.build_lex()


def win_has_needle(p, word):
    i0 = max(0, p - 24); i1 = min(len(A._LEX_STARTS) - 1, p + 24)
    c0 = int(A._LEX_STARTS[i0]); c1 = int(A._LEX_STARTS[i1])
    return word in A._LEX_TEXT[c0:c1].lower()


top1 = 0
ranks = []
for x in n:
    ps = A.lex_find(x['query'], 4)
    r = next((i + 1 for i, (p, _) in enumerate(ps) if win_has_needle(p, x['needle_word'].lower())), 99)
    ranks.append(r)
    top1 += r == 1
print(f'needle at rank #1: {top1}/{len(n)} = {top1/len(n):.3f}')
print(f'rank distribution: {sorted(ranks)}')
print(f'mean rank (hits only): {sum(r for r in ranks if r<99)/max(1,len([r for r in ranks if r<99])):.2f}')
