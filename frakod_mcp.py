# -*- coding: utf-8 -*-
"""Fracode MCP — переносимая долговременная память для LLM-агентов (stdio).

Подключение в любой MCP-клиент (stdio):
  {"mcpServers": {"fracode": {"command": "python", "args": ["frakod_mcp.py"]}}}

Инструменты: remember / link / resolve / chain / recall / recall_vector_only /
ask_llm / index_stats / memory_manifest / memory_export / memory_import /
memory_verify.

Почему файл называется frakod_mcp.py, а продукт — Fracode MCP: внутренняя
инфраструктура (имена файлов, переменные FRK_*, формат пакета frk1x) сложилась
исторически и переименование ломает ночные пайплайны. Наружу продукт называется
только Fracode.
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

# Порт канонический — 8781 (как в README и test_24b_*.py), но переопределяемый:
# без этого MCP жёстко бил в 8781, тогда как ночной пайплайн поднимает API на 8790,
# и «MCP не работает» выглядело как поломка памяти, а не как несовпадение портов.
FRK = os.environ.get('FRK_API', 'http://127.0.0.1:8781').rstrip('/')

# ОБХОД ПРОКСИ (обязательно). urllib по умолчанию берёт http_proxy из окружения и
# гонит даже 127.0.0.1 через него: в этой среде так прокси отвечает 502 Bad Gateway,
# и «память сломалась», хотя API жив и curl отвечает 200. Локальный адрес — напрямую.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _call(method, path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
    req = urllib.request.Request(FRK + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with _OPENER.open(req, timeout=120) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {'ok': False, 'error': e.read().decode('utf-8', 'replace')[:300]}
    except Exception as e:
        # API недоступен: ошибка как результат инструмента, MCP-сервер живёт дальше
        return {'ok': False, 'error': f'Fracode API недоступен: {type(e).__name__}: {e}'}

mcp = FastMCP('fracode', instructions=(
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
    """Размер и состояние архива с ЧЕСТНОЙ бухгалтерией.

    Единица адреса бывает разной: у архива на ВНУТРЕННЕЙ модели это ТОКЕН
    (коды 24 Б/токен при d=192, S=12), у архива на ВНЕШНЕМ энкодере — ОКНО
    (коды 16 Б/окно при d=1024, S=8; см. meta.unit). Поэтому возвращаются
    раздельно: b_per_tok_total (то, что растёт с архивом — цена памяти) и
    b_per_tok_cbook_amortized (вклад кодобука фиксированного размера, падает
    с ростом N). Не цитируйте одно число без другого."""
    return _call('GET', '/index/stats')


# ---------- переносимая память (перенос из агента А в агент Б) ----------
# Единственный научный дифференциатор: память — самодостаточный артефакт,
# а не часть весов модели. Модель А записала -> модель Б прочитала.

@mcp.tool()
def memory_manifest() -> dict:
    """ЧТО ПЕРЕНОСИМО, а что нет. Отдаёт манифест будущего пакета: список файлов
    с sha256 и размерами, честный итог в МБ, и отдельно — что НЕ поедет между
    разными архитектурами (keys_f16, энкодер). Вызывать перед экспортом."""
    return _call('GET', '/memory/manifest')

@mcp.tool()
def memory_export(path: str = '', include_keys: bool = False) -> dict:
    """Упаковать всю память в один файл .frk1x (перенос в другой агент).
    path пустой -> положит рядом с архивом. include_keys=True добавит
    keys_f16 (нужен только при переносе внутрь одной архитектуры; тяжелее)."""
    return _call('POST', '/memory/export',
                 {'path': path, 'include_keys': include_keys, 'include_store': True})

@mcp.tool()
def memory_import(path: str) -> dict:
    """Развернуть .frk1x, полученный от другого агента, как рабочий архив.
    Ничего не перезаписывает: создаёт новый каталог и возвращает его путь."""
    return _call('POST', '/memory/import', {'path': path, 'mode': 'new'})

@mcp.tool()
def memory_verify(path: str, k: int = 8, n_probe: int = 24) -> dict:
    """НЕЗАВИСИМАЯ ПРОВЕРКА перенесённой памяти: гоняет реальный поиск по пакету
    и сравнивает с тем, откуда запрос взят. Не верит манифесту.

    Для пакета на ВНЕШНЕМ энкодере сначала сверяется САМ ЭНКОДЕР: читатель
    повторяет энкодер на 5 контрольных окнах сборки (meta.enc_probe) и должен
    получить cos >= 0.999. Если энкодер чужой — вернётся ok=False и явная
    причина, а не тихий нулевой recall. Единица адреса — токен или окно
    (см. поле unit в ответе); для внешнего архива просьба считать в окнах."""
    return _call('POST', '/memory/verify', {'path': path, 'k': k, 'n_probe': n_probe})

@mcp.tool()
def memory_export_selfcontained(path: str = '') -> dict:
    """САМООПИСЫВАЮЩИЙ пакет (frk1x/2): память + её ЭНКОДЕР + токенизатор.

    Почему так, а не иначе: измерено, что память НЕ читается чужим энкодером
    (recall@8 = 0.000 у всех 15 проверенных кандидатов). Поэтому «переносимая
    память» = «память вместе с энкодером». Энкодер весит ~3 МБ (float16), так
    что пакет остаётся лёгким. Использовать ЭТО, а не memory_export."""
    return _call('POST', '/memory/export_v2',
                 {'path': path, 'include_encoder': True, 'include_store': True})

@mcp.tool()
def memory_import_selfcontained(path: str, name: str = '') -> dict:
    """Развернуть самодостаточный пакет frk1x/2. Энкодер кладётся рядом как
    encoder_w.npy и прописывается в meta — архив работает БЕЗ чекпойнта A."""
    return _call('POST', '/memory/import_v2', {'path': path, 'name': name})

if __name__ == '__main__':
    mcp.run()
