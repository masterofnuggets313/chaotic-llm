"""prep_yandex_chat.py — подготовка русских чат-данных из yandex_q_full (v2).

Структура записи:
  question = text_plain или title
  answers = {id: [], text_plain: [], votes: [], quality: [], ...}
  approved_answer = id одобренного ответа (часто None)

Алгоритм: берём лучший ответ по голосам/качеству, чистим HTML, фильтруем мусор.

Запуск:
  python prep_yandex_chat.py --input C:/Users/Geroin/Downloads/yandex_q.jsonl.zst --max-pairs 300000
"""
import os, sys, json, re, random, argparse

try:
    import zstandard as zstd
except ImportError:
    print("pip install zstandard", flush=True)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(s):
    if not s:
        return ""
    s = TAG_RE.sub(" ", s)
    s = s.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_best_answer(item):
    """Выбирает лучший ответ: одобренный → самый залайканный → первый."""
    ans = item.get("answers") or {}
    texts = ans.get("text_plain") or []
    if not texts:
        return None
    votes = ans.get("votes") or [0] * len(texts)
    quality = ans.get("quality") or [0] * len(texts)
    approved = item.get("approved_answer")
    ids = ans.get("id") or []
    # 1) одобренный
    if approved and approved in ids:
        i = ids.index(approved)
        t = clean_html(texts[i])
        if len(t) >= 40:
            return t
    # 2) лучший по (votes, quality)
    best_i, best_key = -1, (-1, -1)
    for i in range(len(texts)):
        t = clean_html(texts[i])
        if len(t) < 40:
            continue
        key = (votes[i] if i < len(votes) else 0, quality[i] if i < len(quality) else 0)
        if key > best_key:
            best_key, best_i = key, i
    if best_i >= 0:
        return clean_html(texts[best_i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="C:/Users/Geroin/Downloads/yandex_q.jsonl.zst")
    ap.add_argument("--output", default=os.path.join(HERE, "ru_chat_yandex.json"))
    ap.add_argument("--max-pairs", type=int, default=300_000)
    ap.add_argument("--min-q", type=int, default=15)
    ap.add_argument("--min-a", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Читаю: {args.input}", flush=True)
    pairs = []
    n = 0
    with open(args.input, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        buf = b""
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                n += 1
                q = clean_html(item.get("text_plain") or item.get("title") or "")
                a = pick_best_answer(item)
                if len(q) < args.min_q or not a or len(a) < args.min_a:
                    continue
                if q[:40] in a or a[:40] in q:
                    continue
                pairs.append((q, a))
                if len(pairs) >= args.max_pairs * 2:
                    break
            if len(pairs) >= args.max_pairs * 2:
                break

    print(f"Всего записей: {n:,}, валидных пар: {len(pairs):,}", flush=True)

    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[:args.max_pairs]
    print(f"После выборки: {len(pairs):,}", flush=True)

    rows = [{"instruction": q, "input": "", "output": a, "source": "yandex_q_full"}
            for q, a in pairs]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"Сохранено: {args.output} ({len(rows):,} пар)", flush=True)
    for r in rows[:2]:
        print("  Q:", r["instruction"][:100], flush=True)
        print("  A:", r["output"][:100], flush=True)


if __name__ == "__main__":
    main()