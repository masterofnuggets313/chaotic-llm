# Fracode MCP — long-term memory for agents

[Russian](README.md) | [English](README.en.md)

External long-term memory for LLM agents. The archive of chats and code lives on
disk in compressed form, and the agent pulls the context it needs over MCP —
instead of re-uploading the whole history into the prompt on every query.

## Where the idea came from

A micro-model (STS-Prog, ~11M params) was trained to compress sequences — the
experiment lives in this same repo, branch `pc-carroll`. The model predicted the
next token through synchronous driver layers, and its byproduct — a compact token
embedding table — became the retrieval key. The memory grew out of a model that
learns to compress.

## How much memory it takes

Two independent mechanisms, and they must not be conflated:

| | Archive (`codes.npy`) | Operational memory (`/remember`) |
|---|---|---|
| Stores | 24 B/token of codes + keys for rerank | the record text, in full |
| Written | offline, from the corpus | at runtime, by the agent |
| Compresses | yes | no |
| File | `frakod_index/*.npy` | `frakod_store.json` |

**"24 B/token" refers to `codes.npy`** — that is the compression itself, and it
powers ADC candidate screening. The archive's full on-disk footprint is larger,
because exact reranking reads `keys_f16.npy` and the lexical layer reads
`ids.npy`:

```
N = 2,927,882 tokens
codes.npy     =    67.0 MB    24.0 B/token   <- this is the "24 B"
keys_f16.npy  = 1,072.2 MB   384.0 B/token   <- candidate rerank
ids.npy       =    11.2 MB     4.0 B/token   <- lexical exact layer
-------------------------------------------------------------
total         = 1,150.8 MB   412.1 B/token
```

`keys_f16` need not be resident: the rerank reads them via `mmap` from disk.
Without it, the permanent footprint is 28 B/token of archive.

Code size: `24 B/token` means **d=192, S=12**. Other configurations give a
different code — at d=256, S=16 the code is `(N,2,16)` = 32 B/token. The
cheaper/more-accurate trade-off is a deliberate choice.

## How fast and how precise

One archive, the same questions, different search channels — `test_24b_proof.py`.
Archive built on v7: BPE-8192, d=256, S=16, N=2,386,967 tokens.

| search channel | recall@4 | median |
|---|---|---|
| ADC over codes, question query | 0.30 | 638 ms |
| codes + `keys_f16` rerank, question query | 0.60 | 49 ms |
| lexical exact layer | 1.00 | 17 ms |
| hybrid (`auto`) | 1.00 | 64 ms |
| ADC over codes, snippet query | 0.00 | 648 ms |
| codes + rerank, snippet query | 0.17 | 53 ms |

`recall@4` = "the needle word appears in the text of any of the 4 returned
windows". Strict precision@4 of the lexical layer is 0.25 on questions, 0.29 on
snippets.

How to read it: 24-byte ADC finds candidates (0.30), the `keys_f16` rerank lifts
that to 0.60. The hybrid reaches 1.00 via exact lexical search over `ids.npy` —
an inverted index over rare words. On short snippets without a template the
vector barely works (0.00–0.17): the averaged window blurs a short query.

## 24 B/token on the code archive

Archive `frakod_index_code10m` — 10,000,000 tokens of code, BPE-8192. The true
position of the target fragment is known; the metric is a hit within ±7 tokens.

| query method | top-1 | top-4 | cos |
|---|---|---|---|
| context query, window RAD=4 (as in the builder) | 14/16 | 14/16 | 1.000 |
| monotoken query `E[ids].mean(0)` | 8/16 | 8/16 | 0.958 |

Through the live API on the same archive, 16 positions out of 10M, ADC over 24
bytes:

| channel | top-1 | top-4 | top-8 |
|---|---|---|---|
| ADC over 24 B only | 0/16 | 3/16 | 7/16 |
| 24 B + `keys_f16` rerank | 0/16 | 3/16 | 7/16 |

7 out of 16 queries land within ±7 tokens, picking 16 candidates out of
10,000,000 — 625,000× denser than chance. 24-byte compression carries usable
information.

## Bench against RAG and Mem0

History slice, 30 needle questions (25 actually used after dedup), one chat node.
Artifact — `bench_results.json`, reproduction — `bench_memory_systems.py`.

| | recall@4 | prompt tokens | search | LLM tokens on write | storage |
|---|---|---|---|---|---|
| no memory | 0.00 | 151 | — | 0 | — |
| RAG (chroma) | 0.24 | 263 | 27 ms | 0 | 20.3 MB |
| Mem0 | 0.08 | 684 | 63 ms | 1,686,418 | — |
| **Fracode** | **1.00** | 560 | 17 ms | **0** | **1,150.8 MB** |

Honest caveats:

- **The systems get different queries.** RAG and Mem0 receive a short question;
  Fracode in the original bench received a 160-char fragment literally
  containing the needle word. The bench now measures both protocols; the table
  above uses equal conditions (`query`).
- **Fracode wins by the lexical layer**, not by the vector. That is expected for
  the task "a rare word occurs once in the corpus".
- **Mem0 solves a different task** — it extracts structured facts with an LLM,
  hence 1.68M write tokens. That is the honest price of LLM ingest.

## How it works

1. **Archive** — BPE-tokenized text cut into windows; each window is quantized by
   a product quantizer over the micro-model's embeddings.
   `frakod_index_build.py`.
2. **Retrieval** — ADC screening over 24-B codes, rerank over `keys_f16`, plus a
   lexical exact layer (rare query words matched as exact token-id subsequences
   in `ids.npy`, IDF-ranked). Hybrid via `/recall`, mode `auto`. `frakod_api.py`.
3. **MCP** — stdio server `frakod_mcp.py`, 8 tools: `remember`, `link`,
   `resolve`, `chain`, `recall`, `recall_vector_only`, `ask_llm`, `index_stats`.
   Addresses are `frk1:<slot>.<id>`; `link`/`chain` accept both an address and a
   bare id.
   ```json
   {"mcpServers": {"frakod": {"command": "python", "args": ["frakod_mcp.py"]}}}
   ```
4. **History portability** — `ingest_chat.py` imports chat exports from other
   agents (formats `a`/`b`/jsonl/txt) into the same archive.

## Quick start

```bash
pip install -r requirements.txt
# 1) build the archive from your own text
FRK_CORPUS=my_history.txt FRK_OUTD=frakod_index python frakod_index_build.py
# 2) start the API (port 8781)
python -m uvicorn frakod_api:app --port 8781
# 3) point any MCP client at frakod_mcp.py
```

The vector channel needs an embedder checkpoint; the lexical layer needs the
corpus you encoded. Set `FRK_CORPUS`, otherwise it is searched next to the
package.

## Why it is cool

- **Writing is free**: zero LLM calls per ingest — unlike schemes that run every
  fact through a model.
- **Memory on disk is 16× more compact than the keys**: 24 B/token of codes
  against 384 B/token of f16 — and those 24 bytes genuinely carry retrieval
  information.
- **The hybrid gives recall@4 1.00** at a median of ~64 ms over 2.4M tokens: the
  vector brings candidates, the lexical layer finishes exactly.
- **Runs as a plain MCP server** — plugs into any agent; the memory lives beside
  the model, not inside it.
- **All local**, GPL-3.0, ~1.5k lines of Python. No external APIs.
