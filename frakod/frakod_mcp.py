# -*- coding: utf-8 -*-
"""frk1-mcp: MCP-сервер поверх Fracode-памяти (stdio).
Подключение в любой MCP-клиент (stdio):
  {"mcpServers": {"frakod": {"command": "python", "args": ["frakod_mcp.py"]}}}
Инструменты: remember / link / resolve / chain / recall / recall_vector_only /
ask_llm / index_stats.
"""
import json, os, urllib.request, urllib.error

# FastMCP переезжал между версиями mcp: сначала mcp.server.fastmcp, потом mcp.server.MCPServer
try:
    from mcp.server.fastmcp import FastMCP          # mcp < 1.10 (классический путь)
except ImportError:                                  # pragma: no cover
    try:
        from mcp.server import FastMCP               # промежуточные версии
    except ImportError:
        from mcp.server.mcpserver import (            # mcp >= 1.10
            MCPServer as FastMCP,
        )

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
    """Связать два адреса ребром (причина->следствие, упоминание->факт).
    Принимает и адреса 'frk1:<slot>.<id>', и голые id."""
    return _call('POST', '/link', {'frm': frm, 'to': to})

@mcp.tool()
def resolve(query: str, hop: bool = False) -> dict:
    """Найти в памяти ближайший адрес по вопросу; hop=True - разыменовать ребро.
    По умолчанию hop=False (как у сервера): явное поведение, без лишних ребёр."""
    return _call('POST', '/resolve', {'query': query, 'hop': hop})

@mcp.tool()
def chain(start: str, depth: int = 3) -> dict:
    """Обойти цепочку рёбер от адреса (A->B->C). depth = сколько шагов."""
    return _call('GET', f'/chain/{start}?max_hops={depth}')

@mcp.tool()
def recall(query: str, k: int = 8) -> dict:
    """Поиск по архиву (гибрид: векторный ADC + лексический exact-слой).
    Вернёт куски текста с позицией и возрастом в токенах от конца."""
    return _call('POST', '/recall', {'text': query, 'k': k})

@mcp.tool()
def recall_vector_only(query: str, k: int = 8) -> dict:
    """Честный замер 24-байтового канала: поиск ТОЛЬКО по кодобукам (ADC),
    без реранка по keys_f16 и без лексического слоя. Показывает, что умеет
    само сжатие, а не серверные надстройки."""
    return _call('POST', '/recall', {'text': query, 'k': k, 'mode': 'vector24'})

@mcp.tool()
def ask_llm(question: str, chat: str = os.environ.get('FRK_CHAT', 'ollama:your-model'),
            use_archive: bool = False) -> dict:
    """Задать вопрос LLM-хосту с контекстом из Fracode-памяти;
    use_archive=True добавит находки из архива."""
    return _call('POST', '/ask', {'query': question, 'chat': chat,
                                  'use_archive': use_archive})

@mcp.tool()
def index_stats() -> dict:
    """Размер и состояние архива с ПОЛНОЙ бухгалтерией: коды (24 Б/токен),
    keys_f16 (реранк), ids (лекс-слой), честный итог в МБ и Б/токен."""
    return _call('GET', '/index/stats')

if __name__ == '__main__':
    mcp.run()
