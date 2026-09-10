# -*- coding: utf-8 -*-
"""frk1-mcp: MCP-сервер поверх Fracode-памяти (stdio).
Подключение в любой MCP-клиент (stdio):
  {"mcpServers": {"frakod": {"command": "python", "args": ["frakod_mcp.py"]}}}
Инструменты: remember / link / resolve / chain / ask_llm.
"""
import json, os, urllib.request, urllib.error
from mcp.server.fastmcp import FastMCP

FRK = 'http://127.0.0.1:8781'

def _call(method, path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
    req = urllib.request.Request(FRK + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {'ok': False, 'error': e.read().decode('utf-8', 'replace')[:300]}
    except Exception as e:
        # frk1 недоступен: ошибка как результат инструмента, MCP-сервер живёт дальше
        return {'ok': False, 'error': f'frk1 недоступен: {type(e).__name__}: {e}'}

mcp = FastMCP('frakod', instructions=(
    'Fracode-память: запиши факт - получи адрес, свяжи рёбрами, спроси - '
    'найдёт по адресу и вернёт контекст. ask_llm отдаёт вопрос связанной LLM.'))

@mcp.tool()
def remember(text: str, meta: str = '') -> dict:
    """Сохранить факт в постоянную память. Вернёт адрес вида frk1:<slot>.<id>."""
    return _call('POST', '/remember', {'text': text, 'meta': meta or None})

@mcp.tool()
def link(frm: str, to: str) -> dict:
    """Связать два адреса ребром (причина->следствие, упоминание->факт)."""
    return _call('POST', '/link', {'frm': frm, 'to': to})

@mcp.tool()
def resolve(query: str, hop: bool = True) -> dict:
    """Найти в памяти ближайший адрес по вопросу; hop=True - разыменовать ребро."""
    return _call('POST', '/resolve', {'query': query, 'hop': hop})

@mcp.tool()
def chain(start: str, depth: int = 3) -> dict:
    """Обойти цепочку рёбер от адреса (A->B->C)."""
    return _call('GET', f'/chain/{start}?depth={depth}')

@mcp.tool()
def recall(query: str, k: int = 8) -> dict:
    """Поиск по 10M-токенов АРХИВУ (два года истории сжаты в 229 МБ).
    Вернёт куски текста с позицией и возрастом в токенах от конца."""
    return _call('POST', '/recall', {'text': query, 'k': k})

@mcp.tool()
def ask_llm(question: str, chat: str = os.environ.get('FRK_CHAT', 'ollama:your-model'),
            use_archive: bool = False) -> dict:
    """Задать вопрос LLM-хосту с контекстом из Fracode-памяти;
    use_archive=True добавит находки из 10M-архива."""
    return _call('POST', '/ask', {'query': question, 'chat': chat,
                                  'use_archive': use_archive})

@mcp.tool()
def index_stats() -> dict:
    """Размер и состояние архива (N токенов, МБ кодов, скорость загрузки)."""
    return _call('GET', '/index/stats')

if __name__ == '__main__':
    mcp.run()
