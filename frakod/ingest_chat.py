# -*- coding: utf-8 -*-
"""UNIVERSAL history import into Frakod (cross-agent portability).
Formats: 'a' (conversations.json экспорт диалогов), 'b' (export.json),
jsonl ({"role","content"} per line), txt (plain dump).
Output: same pipeline as Hermes dump -> dedup -> frakod_index -> frk1.

Usage:
  python ingest_chat.py a my_conversations.json [-o history_a.txt]
  python ingest_chat.py b export.json
  python ingest_chat.py jsonl   history.jsonl
  python ingest_chat.py txt     raw_chat.txt
"""
import argparse
import datetime
import json
import sys


def _emit(f, role, text, ts=None):
    text = ' '.join(str(text).split())
    if not text:
        return 0
    junk = text.count('\ufffd') + text.count('\x00')
    if junk > max(2, len(text) // 20):
        return 0
    if text.lstrip().startswith('[Система:') or '(restored)' in text:
        return 0
    t = ''
    if ts:
        try:
            t = ' ' + datetime.datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M')
        except (ValueError, OSError, TypeError):
            pass
    f.write(f'[{t.strip()}] {role}: {text[:1500]}\n')
    return 1


def _flatten_content(content):
    """Экспорт-форматы крупных ассистентов: content: str | list[{type:text,text:...}] | dict."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get('type') == 'text':
                parts.append(c.get('text', ''))
            elif isinstance(c, str):
                parts.append(c)
        return ' '.join(parts)
    if isinstance(content, dict):
        if isinstance(content.get('parts'), list):
            return ' '.join(str(x) for x in content['parts'] if x)
        return str(content.get('text') or content.get('content') or '')
    return str(content)


def load_a(path, f):
    data = json.load(open(path, encoding='utf-8'))
    n = 0
    for conv in data:
        mapping = conv.get('mapping', {})
        for node in mapping.values():
            msg = node.get('message')
            if not msg or msg.get('role') not in ('user', 'assistant'):
                continue
            ts = (msg.get('timestamp') or {}).get('timestamp') if isinstance(msg.get('timestamp'), dict) else msg.get('create_time')
            n += _emit(f, msg['role'], _flatten_content(msg.get('content', '')), ts)
    return n


def load_b(path, f):
    data = json.load(open(path, encoding='utf-8'))
    convos = data.get('conversations', data if isinstance(data, list) else [])
    n = 0
    for conv in convos:
        for msg in conv.get('chat_messages', conv.get('messages', [])) or []:
            role = msg.get('sender') or msg.get('role')
            if role not in ('human', 'user', 'assistant', 'bot'):
                continue
            role = 'user' if role in ('human', 'user') else 'assistant'
            ts = msg.get('created_at')
            n += _emit(f, role, _flatten_content(msg.get('text') or msg.get('content', '')),
                       ts if isinstance(ts, (int, float)) else None)
    return n


def load_jsonl(path, f):
    n = 0
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n += _emit(f, 'user', line)
            continue
        role = obj.get('role', 'user')
        n += _emit(f, 'assistant' if role in ('assistant', 'model', 'bot') else 'user',
                   _flatten_content(obj.get('content', '')), obj.get('ts') or obj.get('timestamp'))
    return n


def load_txt(path, f):
    n = 0
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if s:
            n += _emit(f, 'user', s)
    return n


LOADERS = {'a': load_a, 'b': load_b,
           'jsonl': load_jsonl, 'txt': load_txt}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('format', choices=sorted(LOADERS))
    ap.add_argument('file')
    ap.add_argument('-o', '--out', default='hermes_history_in.txt')
    a = ap.parse_args()
    cnt = 0
    with open(a.out, 'w', encoding='utf-8') as f:
        cnt = LOADERS[a.format](a.file, f)
    size = len(open(a.out, encoding='utf-8').read())
    print(f'INTO {a.out}: {cnt} valid replicas, {size:,} chars (~{int(size*0.57):,} tok)')
    print('NEXT: FRK_CORPUS=' + a.out + ' bash run_rebuild.sh (or full nightly)')
