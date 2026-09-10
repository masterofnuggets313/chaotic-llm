# -*- coding: utf-8 -*-
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r = json.load(open('bench_results.json', encoding='utf-8'))
names = ['None', 'RAG', 'Mem0', 'Fracode']
keys = ['none', 'rag', 'mem0', 'fracod']
c = ['#9aa0a6', '#4c8bf5', '#f07c5a', '#3fb950']
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=150)
fig.patch.set_facecolor('white')
def bars(a, vals, title, fmt):
    b = a.bar(names, vals, color=c, width=0.62)
    a.set_title(title, fontsize=11, weight='bold')
    a.spines[['top', 'right']].set_visible(False)
    a.grid(axis='y', alpha=0.25)
    a.set_ylim(0, max(vals) * 1.25 or 1)
    for rect, v in zip(b, vals):
        a.text(rect.get_x() + rect.get_width() / 2, v, fmt(v),
               ha='center', va='bottom', fontsize=9)
bars(ax[0], [r[k]['recall'] for k in keys], 'recall@4 (higher is better)', lambda v: f'{v:.2f}')
bars(ax[1], [r[k]['prompt_tok_mean'] for k in keys], 'prompt tokens per query', lambda v: f'{v:.0f}')
w = [max(r[k].get('write_tokens_llm', 0), 0.001) for k in keys]
b = ax[2].bar(names, w, color=c, width=0.62)
ax[2].set_yscale('log'); ax[2].set_ylim(0.001, 1e8)
ax[2].set_title('LLM tokens on write (log)', fontsize=11, weight='bold')
ax[2].spines[['top', 'right']].set_visible(False)
ax[2].grid(axis='y', alpha=0.25)
for rect, v in zip(b, w):
    lbl = '0' if v < 1 else f'{v:,.0f}'
    ax[2].text(rect.get_x() + rect.get_width() / 2, v, lbl, ha='center', va='bottom', fontsize=9)
fig.suptitle('One shared history (2.9M tokens, 313 questions) - four memory systems', fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig('bench_chart.png', bbox_inches='tight')
print('saved')
