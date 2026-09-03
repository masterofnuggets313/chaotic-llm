"""convert_to_chat.py — универсальный конвертер русских датасетов в чат-формат.

Формат на входе: JSONL/parquet с любыми полями (instruction, question, text, conversations, role...)
Формат на выходе: строки "<user>: {вопрос}\n<bot>: {ответ}\n<|endoftext|>"

Использование:
  python convert_to_chat.py --input ru_sharegpt_cleaned.jsonl --output ru_chat_merged.json

Поддерживаемые датасеты (пробует схемы автоматически):
  - yandex_q_full (question/answer)
  - pikabu (text — посты, title — заголовки)
  - oasst1_ru (role/text — диалоговые цепочки)
  - ru_sharegpt_cleaned (instruction/output)
  - ru_turbo_alpaca/evol_instruct (instruction/input/output)
  - ru_turbo_saiga (instruction/output)
"""
import os, sys, json, argparse


def try_extract(item, field_variants):
    """Пробует разные варианты имени поля."""
    for v in field_variants:
        val = item.get(v)
        if val and isinstance(val, str) and len(val) > 3:
            return val.strip()
    return None


def extract_pairs(rows, fmt=None):
    """Извлекает пары (user, bot) из списка записей."""
    pairs = []
    for item in rows[:5]:
        print(f"  пример полей: {list(item.keys())[:8]}", flush=True)
        break

    for item in rows:
        # Схема 1: instruction / output (alpaca, sharegpt, saiga)
        instr = try_extract(item, ["instruction", "question", "title", "text"])
        out = try_extract(item, ["output", "answer", "summary", "description"])
        if instr and out:
            pairs.append((instr, out))
            continue

        # Схема 2: conversations (массив {role, content})
        conv = item.get("conversations")
        if isinstance(conv, list) and len(conv) >= 2:
            user = bot = None
            for m in conv:
                role = m.get("role", "")
                content = m.get("content", m.get("text", ""))
                if "user" in role.lower() or "human" in role.lower():
                    user = content
                elif "assistant" in role.lower() or "bot" in role.lower():
                    bot = content
            if user and bot:
                pairs.append((user, bot))
                continue

        # Схема 3: messages (аналогично)
        msgs = item.get("messages")
        if isinstance(msgs, list) and len(msgs) >= 2:
            user = bot = None
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", m.get("text", ""))
                if "user" in role.lower() or "human" in role.lower():
                    user = content
                elif "assistant" in role.lower() or "bot" in role.lower():
                    bot = content
            if user and bot:
                pairs.append((user, bot))
                continue

        # Схема 4: question / answer
        q = try_extract(item, ["question", "query"])
        a = try_extract(item, ["answer", "reply"])
        if q and a:
            pairs.append((q, a))
            continue

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="путь к входному файлу (.jsonl / .json / .parquet)")
    ap.add_argument("--output", default="ru_chat_merged.json", help="выходной JSON")
    ap.add_argument("--format", default=None, help="принудительная схема: alpaca, oasst, qa, sharegpt")
    args = ap.parse_args()

    print(f"Загрузка: {args.input}", flush=True)

    # Читаем
    ext = os.path.splitext(args.input)[1]
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(args.input)
            rows = [dict(zip(table.column_names, row)) for row in zip(*table.to_pydict().values())]
        except ImportError:
            print("pyarrow not installed! pip install pyarrow", flush=True)
            sys.exit(1)
    elif ext == ".jsonl":
        rows = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext in (".zst",):
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(args.input, "rb") as f:
                data = dctx.decompress(f.read())
            if data[0] == ord("["):
                rows = json.loads(data.decode("utf-8"))
            else:
                rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
        except ImportError:
            print("zstandard not installed! pip install zstandard", flush=True)
            sys.exit(1)
    else:
        # plain JSON
        with open(args.input, "r", encoding="utf-8") as f:
            rows = json.load(f)

    print(f"Записей: {len(rows):,}", flush=True)

    # Конвертируем
    pairs = extract_pairs(rows, args.format)
    print(f"Извлечено пар: {len(pairs):,}", flush=True)

    # Форматируем в чат
    chat_rows = []
    for user, bot in pairs:
        chat_rows.append({
            "instruction": user,
            "input": "",
            "output": bot,
            "source": os.path.basename(args.input),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chat_rows, f, ensure_ascii=False, indent=1)
    print(f"Сохранено: {args.output} ({len(chat_rows):,} пар)", flush=True)
    print(f"Пример: {json.dumps(chat_rows[0], ensure_ascii=False)[:120]}", flush=True)


if __name__ == "__main__":
    main()