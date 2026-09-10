# -*- coding: utf-8 -*-
"""Слив ИСТОРИИ HERMES во Fracod: state.db -> hermes_history.txt + GT-иглы.
Ночной прогон 2026-09-10/11. Выходы: hermes_history.txt, hermes_needles.json."""
import sqlite3, json, random, os

DB = os.path.expanduser('~/AppData/Local/hermes/state.db')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

db = sqlite3.connect(DB)
rows = db.execute(
    "select m.id, m.role, m.content, m.timestamp from messages m "
    "join sessions s on s.id=m.session_id "
    "where m.role in ('user','assistant') and m.content is not null "
    "and length(m.content) between 30 and 4000 "
    "and m.content not like '%<system>%' and m.content not like '%[OUT-OF-BAND%' "
    "and m.content not like '%COMPACTION%' and s.source != 'cron' "
    "order by m.id").fetchall()
print(f'реплик отобрано: {len(rows)}')

out = []; pos = 0
lines = []
for mid, role, content, ts in rows:
    import datetime
    t = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
    txt = ' '.join(content.split())[:1500]
    # фильтр бинарного мусора (Replacement-фрагменты tool-выводов бьют поиск)
    junk = txt.count('\ufffd') + txt.count('\x00')
    if junk > max(2, len(txt) // 20):
        continue
    # hygiene v4: шаблонные вставки векторно сливаются в мусор-кластер и топят ADC
    if txt.startswith('[Система:') or txt.endswith('(restored)') or 'active context reference' in txt:
        continue
    lines.append(f'[{t} #{mid}] {role}: {txt}')
corpus = '\n'.join(lines)
with open(os.path.join(OUT_DIR, 'hermes_history.txt'), 'w', encoding='utf-8') as f:
    f.write(corpus)
n_tok_est = len(corpus) / 3.6
print(f'корпус: {len(corpus):,} симв. ~{n_tok_est:,.0f} токенов')

# --- иглы: юзер-сообщения с редкими словами, из середины истории ---
user_rows = [(mid, ' '.join(c.split()), ts) for mid, r, c, ts in rows if r == 'user' and 80 < len(c) < 600]
random.seed(7)
import re
def rare_words(s):
    ws = [w.lower() for w in re.findall(r'[а-яА-Яa-zA-Z]{6,}', s)]
    return ws
picked = []
seen = set()
for mid, txt, ts in random.sample(user_rows, min(400, len(user_rows))):
    rws = rare_words(txt)
    if not rws: continue
    key = max(rws, key=lambda w: len(w))
    if key in seen: continue
    seen.add(key)
    picked.append((mid, txt, key))
    if len(picked) == 30: break
needles = [{'msg_id': mid, 'needle_word': kw,
            'query': f'что я говорил про «{kw}»?',
            'snippet': txt[:160]} for mid, txt, kw in picked]
with open(os.path.join(OUT_DIR, 'hermes_needles.json'), 'w', encoding='utf-8') as f:
    json.dump(needles, f, ensure_ascii=False, indent=1)
print(f'игл сохранено: {len(needles)}; первая: {needles[0]["needle_word"]}')
