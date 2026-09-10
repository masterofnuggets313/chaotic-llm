# -*- coding: utf-8 -*-
"""Ночной бенч: NONE vs RAG(chroma+эмбедер) vs Mem0 vs FRACOD.
Один корпус (срез истории Hermes), одна LLM-нода (openai-совместимый хост),
одинаковые вопросы-иглы. Метрики: recall@4, prompt-токены ответа,
LLM-токены записи (через proxy-учёт для Mem0), латентность поиска, вес хранилища.
"""
import json, os, sqlite3, time, statistics, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = 'http://127.0.0.1:11434'
FRK = 'http://127.0.0.1:8781'
MODEL = __import__('os').environ.get('BENCH_MODEL', 'your-chat-model')
SLICE_MSGS = 2000          # столько скармливаем всем системам
N_QUESTIONS = 30
ANSWER_TOKEN_BUDGET = 120

# ---------------- corpus: тот же, что сливает dump_hermes_history ----------
def load_messages():
    rows = []
    with open(os.path.join(HERE, 'hermes_history_dd.txt'), encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln.startswith('['): continue
            head, rest = ln[1:].split('] ', 1)
            msg_id = head.split(' #')[-1]
            role, txt = rest.split(': ', 1)
            rows.append({'id': int(msg_id), 'role': role, 'text': txt})
    return rows

msgs = load_messages()[:SLICE_MSGS]
print(f'срез: {len(msgs)} реплик, {sum(len(m["text"]) for m in msgs)/1e6:.1f} МБ текста', flush=True)
slice_ids = {m['id'] for m in msgs}

# иглы генерим ВНУТРИ среза (чтобы все 4 системы видели и иглу, и вопрос)
import random, re
random.seed(11)
def rare_key(s):
    ws = re.findall(r'[а-яА-Яa-zA-Z]{6,}', s)
    return max(ws, key=len).lower() if ws else None
cands = [m for m in msgs if m['role'] == 'user' and 80 < len(m['text']) < 600]
needles = []; seen = set()
for m in random.sample(cands, len(cands)):
    kw = rare_key(m['text'])
    if not kw or kw in seen: continue
    seen.add(kw)
    needles.append({'msg_id': m['id'], 'needle_word': kw,
                    'query': f'что я говорил про «{kw}»?', 'snippet': m['text'][:160]})
    if len(needles) == 25: break
print(f'вопросов из среза: {len(needles)}', flush=True)

def ollama_chat(prompt):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'stream': False, 'options': {'temperature': 0.2, 'num_predict': ANSWER_TOKEN_BUDGET}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(OLLAMA + '/api/chat',
        data=body, headers={'Content-Type': 'application/json'}), timeout=600).read())
    return r.get('response', ''), int(r.get('prompt_eval_count', 0)), int(r.get('eval_count', 0))

def answer_prompt(question, ctx_snips):
    ctx = '\n\n'.join(f'— {s}' for s in ctx_snips) if ctx_snips else '(нет сохранённых воспоминаний)'
    return (f'Ты — ассистент с памятью. Используй ФАКТЫ из блока «воспоминания» '
            f'(это фрагменты старых диалогов пользователя), отвечай по ним кратко и по делу; '
            f'если фактов нет — так и скажи.\n\nВОСПОМИНАНИЯ:\n{ctx}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}')

# ---------------- proxy-учёт LLM-токенов записи (для Mem0) ------------------
write_tokens = {'prompt': 0, 'completion': 0, 'calls': 0}
class Proxy(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('content-length', 0)); data = self.rfile.read(n)
        req = urllib.request.Request(OLLAMA + self.path, data=data,
            headers={'Content-Type': 'application/json'})
        try:
            resp = urllib.request.urlopen(req, timeout=600).read()
            j = json.loads(resp)
            write_tokens['prompt'] += int(j.get('prompt_eval_count', 0) or 0)
            write_tokens['completion'] += int(j.get('eval_count', 0) or 0)
            write_tokens['calls'] += 1
            self.send_response(200); self.end_headers(); self.wfile.write(resp)
        except Exception as e:
            self.send_error(502, str(e))
    def do_GET(self):
        try:
            resp = urllib.request.urlopen(OLLAMA + self.path, timeout=30).read()
            self.send_response(200); self.end_headers(); self.wfile.write(resp)
        except Exception: self.send_error(502)
    def log_message(self, *a): pass
srv = ThreadingHTTPServer(('127.0.0.1', 11435), Proxy)
threading.Thread(target=srv.serve_forever, daemon=True).start()

def embed(texts):
    body = json.dumps({'model': __import__('os').environ.get('BENCH_EMBED', 'your-embed-model'), 'input': texts}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(OLLAMA + '/api/embed',
        data=body, headers={'Content-Type': 'application/json'}), timeout=600).read())
    return r['embeddings']

def dir_mb(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: tot += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return round(tot / 1024**2, 1)

results = {}

# ============ 1) NONE: память не подключена =================================
print('=== NONE ===', flush=True)
none_prompts = []
for nd in needles:
    txt, pe, ce = ollama_chat(answer_prompt(nd['query'], []))
    none_prompts.append(pe)
results['none'] = {'recall': 0.0, 'prompt_tok_mean': round(statistics.mean(none_prompts), 1),
                   'search_ms': 0, 'storage_mb': 0, 'write_tokens_llm': 0}
print('none ok', flush=True)

# ============ 2) RAG: chroma + эмбедер, чанк=реплика, top-4 ===================
print('=== RAG (chroma) ===', flush=True)
import chromadb
rc = chromadb.PersistentClient(path=os.path.join(HERE, 'bench_rag_chroma'))
try: rc.delete_collection('bench')
except Exception: pass
col = rc.create_collection('bench', metadata={'hnsw:space': 'cosine'})
t0 = time.time()
for i in range(0, len(msgs), 64):
    batch = msgs[i:i+64]
    # ОДИН И ТОТ ЖЕ текст идёт и в эмбеддинг, и в документ (было [:2000] vs [:300] —
    # неравный стандарт внутри одной системы)
    docs_ = [m['text'][:2000] for m in batch]
    col.add(ids=[str(m['id']) for m in batch], embeddings=embed(docs_), documents=docs_)
rag_write_s = time.time()-t0
hits = 0; ptoks = []; sms = []
for nd in needles:
    t1 = time.time()
    q = embed([nd['query'][:2000]])[0]
    r = col.query(query_embeddings=[q], n_results=4)
    sms.append((time.time()-t1)*1000)
    docs = r['documents'][0]
    txt, pe, ce = ollama_chat(answer_prompt(nd['query'], docs))
    ptoks.append(pe); hits += nd['needle_word'].lower() in ' '.join(docs).lower()
results['rag'] = {'recall': round(hits/len(needles), 3), 'prompt_tok_mean': round(statistics.mean(ptoks), 1),
                  'search_ms': round(statistics.median(sms), 1), 'storage_mb': dir_mb(os.path.join(HERE, 'bench_rag_chroma')),
                  'write_tokens_llm': 0, 'write_wall_s': round(rag_write_s)}
print('rag', results['rag'], flush=True)

# ============ 3) MEM0: ollama LLM-экстракция + своя векторка ================
print('=== MEM0 ===', flush=True)
try:
    from mem0 import Memory
    m0 = Memory.from_config({
        'llm': {'provider': 'ollama', 'config': {'model': MODEL, 'ollama_base_url': 'http://127.0.0.1:11435',
                'temperature': 0.0, 'max_tokens': 300}},
        'embedder': {'provider': 'ollama', 'config': {'model': __import__('os').environ.get('BENCH_EMBED', 'your-embed-model'),
                     'ollama_base_url': OLLAMA}},
        'vector_store': {'provider': 'chroma', 'config': {'collection_name': 'mem0bench_v3',
                       'path': os.path.join(HERE, 'bench_mem0_chroma_v3')}},
        'history_db_path': os.path.join(HERE, 'bench_mem0_history_v4.db'),
        'version': 'v1.1'})
    t0 = time.time(); wt_before = dict(write_tokens)
    for i in range(0, len(msgs), 12):
        conv = [{'role': m['role'], 'content': m['text'][:800]} for m in msgs[i:i+12]]
        try: m0.add(conv, user_id='bench')
        except Exception as e: print('add err', str(e)[:60], flush=True)
        if (i//12) % 25 == 0: print(f'mem0 ingest {i}/{len(msgs)}', flush=True)
    mem0_write_s = time.time()-t0
    mem0_write_tok = write_tokens['prompt'] - wt_before['prompt'] + write_tokens['completion'] - wt_before['completion']
    hits = 0; ptoks = []; sms = []
    for nd in needles:
        t1 = time.time(); r = m0.search(nd['query'], limit=4, filters={'user_id': 'bench'}); sms.append((time.time()-t1)*1000)
        docs = [x['memory'] for x in r.get('results', [])]
        txt, pe, ce = ollama_chat(answer_prompt(nd['query'], docs))
        ptoks.append(pe); hits += nd['needle_word'].lower() in ' '.join(docs).lower()
    results['mem0'] = {'recall': round(hits/len(needles), 3), 'prompt_tok_mean': round(statistics.mean(ptoks), 1),
                       'search_ms': round(statistics.median(sms), 1), 'storage_mb': dir_mb(os.path.join(HERE, 'bench_mem0_chroma')),
                       'write_tokens_llm': mem0_write_tok, 'write_wall_s': round(mem0_write_s),
                       'write_llm_calls': write_tokens['calls']}
    print('mem0', results['mem0'], flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    results['mem0'] = {'error': str(e)[:200]}
    print('MEM0 FAILED (isolated):', str(e)[:120], flush=True)

# ============ 4) FRACOD: архив через /recall + 0 LLM-токенов на запись ======
# ЧЕСТНО: два протокола запроса, потому что они меряют разное.
#   query   — только вопрос ("что я говорил про «X»?"), как у RAG/Mem0 -> сравнение 1:1
#   snippet — 160-симв. фрагмент вокруг иглы: проверка, находит ли система
#             фрагмент по его же тексту (у Fracod это задействует лекс-слой)
# Плюс строки vector-only: что даёт САМ 24-байтовый код без реранка и лексики.
print('=== FRACOD ===', flush=True)


def fracod_query(qtext, mode='auto'):
    body = json.dumps({'text': qtext[:300], 'k': 4, 'mode': mode}).encode()
    req = urllib.request.Request(FRK + '/recall', data=body,
                                 headers={'Content-Type': 'application/json'})
    t1 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return r, (time.time() - t1) * 1000


def fracod_eval(query_field, mode, label):
    hits = 0; ptoks = []; sms = []
    for nd in needles:
        r, ms = fracod_query(nd[query_field], mode)
        sms.append(ms)
        docs = [s['text'] for s in r.get('results', [])]
        txt, pe, ce = ollama_chat(answer_prompt(nd['query'], docs))
        ptoks.append(pe); hits += nd['needle_word'].lower() in ' '.join(docs).lower()
    out = {'recall': round(hits / len(needles), 3),
           'prompt_tok_mean': round(statistics.mean(ptoks), 1),
           'search_ms': round(statistics.median(sms), 1)}
    print(f'fracod[{label}] {out}', flush=True)
    return out


st = json.load(open(os.path.join(HERE, 'frakod_index', 'meta.json'), encoding='utf-8'))
# честный вес хранилища: коды + keys(реранк) + ids(лекс), а не только коды
_storage_files = ('codes.npy', 'keys_f16.npy', 'ids.npy')
_store_mb = round(sum(os.path.getsize(os.path.join(HERE, 'frakod_index', f))
                      for f in _storage_files
                      if os.path.exists(os.path.join(HERE, 'frakod_index', f))) / 1024**2, 1)

q_main = fracod_eval('query', 'auto', 'query·hybrid')
results['fracod'] = {**q_main, 'storage_mb': _store_mb,
                     'storage_mb_codes_only': st.get('codes_mb', st.get('fr_mb')),
                     'archive_tokens': st['N'], 'write_tokens_llm': 0,
                     'protocol': 'query (same as RAG/Mem0)'}
# дополнительные строки-разрезы
results['fracod_snippet_hybrid'] = {**fracod_eval('snippet', 'auto', 'snippet·hybrid'),
                                    'protocol': 'snippet 160 chars (bench-memory style)'}
results['fracod_query_vector24'] = {**fracod_eval('query', 'vector24', 'query·24B-only'),
                                    'protocol': 'query, ADC over 24 B codes only, no keys/lex'}
results['fracod_query_lex'] = {**fracod_eval('query', 'lex', 'query·lex-only'),
                               'protocol': 'query, lexical layer only'}
print('fracod storage_mb (codes+keys+ids) =', _store_mb, flush=True)

json.dump(results, open(os.path.join(HERE, 'bench_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('BENCH DONE -> bench_results.json', flush=True)
