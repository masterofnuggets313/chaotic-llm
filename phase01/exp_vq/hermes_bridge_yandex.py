"""hermes_bridge_yandex.py — Hermes-сервер для русской модели (yandex_q_full).

Запуск:
  python hermes_bridge_yandex.py --port 8080

Hermes: custom provider → http://127.0.0.1:8080/v1/chat/completions
"""
import os, sys, json, time, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from live_chat_yandex import make_bpe, build_dataset, generate
from models_pc import build_pc_model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "ru_chat_yandex.json")
CKPT = os.path.join(HERE, "model_chat.pt")


class YandexHandler(BaseHTTPRequestHandler):
    model = None
    tok = None

    def _send(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "sts-prog-russian", "object": "model"}]})
        elif path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8", errors="replace")) if length else {}
        messages = body.get("messages", [])
        # последние 3 сообщения в формат <user>/<bot>
        prompt_parts = []
        for m in messages[-4:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                prompt_parts.append(f"<bot>: {content}")
            else:
                prompt_parts.append(f"<user>: {content}")
        prompt = "\n".join(prompt_parts) + "\n<bot>:"
        steps = min(body.get("max_tokens", 200), 500)
        temp = body.get("temperature", 0.7)
        top_k = body.get("top_k", 40)

        ids = []
        t0 = time.time()
        for tid in generate(self.__class__.model, self.__class__.tok, prompt, steps, temp, top_k):
            ids.append(tid)
        dt = time.time() - t0
        text = self.__class__.tok.decode(ids)
        self._send({
            "id": "chatcmpl-sts",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "sts-prog-russian",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids),
                      "total_tokens": len(prompt) + len(ids), "time_s": round(dt, 1)},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print("Loading Russian chat model (yandex_q_full)...", flush=True)
    # токенизатор — как в train_chat.py: первые 10M символов
    print(f"Loading {DATA_FILE}...", flush=True)
    with open(DATA_FILE, encoding="utf-8") as f:
        rows = json.load(f)
    texts = []
    for r in rows:
        instr = r["instruction"]
        if r.get("input"):
            instr = f"{instr}\n{r['input']}"
        texts.append(f"<user>: {instr}\n<bot>: {r['output']}\n<|endoftext|>")
    tok = make_bpe("\n".join(texts)[:10_000_000])
    V = tok.get_vocab_size()
    print(f"Tokenizer: V={V}", flush=True)

    model = build_pc_model("pc", V, d=384, layers=12, driver_mode="sts_prog",
                           k_init=1.2, sync_steps=8, alpha=0.3).to("cuda")
    model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=False), strict=False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {n:,} params", flush=True)

    YandexHandler.model = model
    YandexHandler.tok = tok

    server = HTTPServer((args.host, args.port), YandexHandler)
    print(f"\nHermes bridge ready: http://{args.host}:{args.port}/v1/chat/completions", flush=True)
    print("Add as custom provider in Hermes with this URL", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()