# -*- coding: utf-8 -*-
"""Регенерация игл под НОВЫЙ dd-корпус: слова с 1 вхождением, игла+вопрос внутри dd.
Пишет hermes_needles.json (30 игл)."""
import json
import re

corpus = open('hermes_history_dd.txt', encoding='utf-8').read()
low = corpus.lower()
words = re.findall(r"[А-Яа-яЁё][а-яёA-Za-z-]{7,}|[A-Za-z][a-zA-Z-]{9,}", corpus)
cnt = {}
for w in words:
    lw = w.lower()
    cnt[lw] = cnt.get(lw, 0) + 1
uniq = sorted({w.lower() for w, c in cnt.items() if c == 1})

needles = []
step = max(1, len(corpus) // 30)
for i in range(30):
    lo, hi = i * step, (i + 1) * step
    picked = next((w for w in uniq if lo <= low.find(w, lo) < hi), None)
    if not picked:
        continue
    p = low.find(picked, lo)
    start = max(p - 150, 0)
    snip = corpus[start:p + 250].replace('\n', ' ')
    needles.append({'needle_word': picked, 'snippet': snip,
                    'query': f'Что в истории связано со словом "{picked}"? Краткий факт.',
                    'char_pos': p})

json.dump(needles, open('hermes_needles.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('игл:', len(needles), '| все в dd:', all(n['snippet'][:60] in corpus for n in needles))
