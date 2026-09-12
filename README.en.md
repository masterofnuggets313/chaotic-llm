# Fracode MCP — portable long-term memory for agents

[Russian](README.md) | [English](README.en.md)

External long-term memory for LLM agents. The archive of chats and code lives on
disk in compressed form, and the agent pulls the context it needs over MCP —
instead of re-uploading the whole history into the prompt on every query.

**The difference from plain RAG and the KV cache:** the memory is a single file
that can be handed to another agent. Model A wrote it, model B reads it. Nothing
is rebuilt from raw text on B's side.

```
A:  memory_export_selfcontained(path="mem.frk1x")   -> 48.7 MB, 3.5 s
    (move the file by any means)
B:  memory_import_selfcontained(path="mem.frk1x.npz", name="b_archive")
    memory_verify(path="b_archive")                 -> recall@8 = 0.875
```

Verified on a clean machine: agent B's directory contained **only** these files
and two Python files. No file from machine A, no checkpoint from the project.

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
| Stores | 24 B/token of codes | the record text, in full |
| Written | offline, from the corpus | at runtime, by the agent |
| Compresses | yes | no |
| File | `frakod_index/*.npy` | `frakod_store.json` |

**"24 B/token" refers to `codes.npy`** — that is the compression itself, and it
powers ADC candidate screening. The archive's full on-disk footprint is larger:
exact reranking historically read `keys_f16.npy`, and the lexical layer reads
`ids.npy`.

Current archive (v8, N = 2,189,372 tokens):

```
codes.npy      =   50.1 MB    24.0 B/token   <- this is the "24 B"
keys_f16.npy   =  801.8 MB   384.0 B/token   <- rerank; NOT NEEDED (see below)
ids.npy        =    8.4 MB     4.0 B/token   <- lexical exact layer
cbook (L0+L1)  =    0.4 MB
-------------------------------------------------------------
core (codes + codebooks + ids + meta) = 58.8 MB = 28.2 B/token
total footprint                       = 860.6 MB = 412.2 B/token
```

**I do not use `keys_f16`.** It lives in one specific encoder's space and is
16× heavier than the codes themselves. Reranking on the raw keys measurably
**hurts**: ADC-only 0.900 vs rerank 0.820 (`recall@8`, v8 archive). It is not
included in the transferable package at all.

Code size: `24 B/token` means **d=192, S=12, K=256, L=2**. Other configurations
give a different code — at d=256, S=16 the code is `(N,2,16)` = 32 B/token.

## How fast and how precise

### 24 B/token on a real archive (v8, 2,189,372 tokens)

Full-rank key, ADC screening, metric `recall@8` with a tolerance of 4 tokens
(keys are `context-avg` over a RAD=4 window, so the point position is physically
blurred). Seed 7, nq=64.

| N | recall@8 (24 B/key) | distinguishability ceiling |
|---|---|---|
| 4,096 | **1.000** | 0.938 |
| 8,192 | **1.000** | 0.812 |
| 16,384 | **1.000** | 0.938 |
| 32,768 | **0.984** | 0.875 |
| 65,536 | **0.984** | 0.781 |
| 131,072 | **0.969** | 0.656 |
| 262,144 | **0.969** | 0.688 |
| 524,288 | **0.969** | 0.797 |

*The earlier numbers in this series (0.828 → 0.234 from 4k to 64k) were obtained
on a **degenerate key** (`external_keys` used only 16 of 192 coordinates because
of `col = idx % sub`). The bug was fixed on 09-11; see
`evidence/D1_CLOSED_2026-09-11.md`. On the fixed key there is almost no degradation.*

### What `recall@8` actually measures

Diagnostics (30 probes): the nearest candidate equals the truth in **0.83** of
cases, while on average **3 of 8** candidates lie within 4 tokens. The vector
finds the **window** around the truth, not the exact token — a direct consequence
of `context-avg` keys. A shuffled-code control gives **exactly 0.00** at all
radii: the codebook genuinely addresses, rather than "any query lands in the
top-8".

## The encoder is the bottleneck — and the fix

I have to say this plainly, because it is measured rather than assumed.

The archive above is built on the **internal** embedding table of the micro-model.
It has a fundamental defect: **the cosine between embeddings of two different
texts is zero** (mean −0.000, max|·| 0.256). The content is simply absent from the
representation, and the address rests on the positional term alone. Verified by
four independent tests (`addr_diag.py`, `path_b_addr_signal.py`).

Swapping the encoder changes everything. One and the same honest protocol
(207 questions, dialogue corpus, gold = all overlapping chunks, bootstrap 1000,
seed 20260912):

| encoder | budget | recall@4 | 95% CI |
|---|---|---|---|
| my model `v8_dialog_seed0.pt` | 4096 B/vec | **0.087** | [0.053, 0.126] |
| `bge-m3`, raw | 4096 B/vec | **0.517** | [0.449, 0.585] |
| `bge-m3` + compression to 20 B/window | 20 B | **0.319** | [0.261, 0.382] |
| ceiling (query = its own chunk) | — | 1.000 | — |

Read it this way:

* **Swapping the encoder is ×6** (0.087 → 0.517). Not a fine-tuning knob — the
  difference between "does not work" and "works".
* **Compression at 205× costs 1.6×** (0.517 → 0.319). A price, not a cliff: the
  channel survives.
* Controls (SHUFFLED, RANDOM, ceiling with gold excluded) all give 0.000 — the
  measuring instrument is validated, the numbers can be trusted.

### How to enable the external encoder

```bash
FRK_ENCODER=ollama FRK_EMBED_MODEL=bge-m3 \
FRK_TOK=tok_v8_dialog.json FRK_NO_KEYS=1 \
python frakod_index_build.py 10000000
```

What changes (and why this is an honest model rather than a hack):

| | internal archive | external archive |
|---|---|---|
| unit of address | token | **window** (default 800 chars, overlap 160) |
| key | context-avg of the token table | window vector from the encoder |
| `key_context_rad` | 4 | **0** (window overlap already supplies context) |
| S (subvectors) | 12 at d=192 | **chosen for d** (d=1024 → 8) |
| cost | 24 B/token | **20 B/window** (16 codes + 4 lexical) |
| readable without the encoder | yes | **no** — the encoder is a dependency |

**The archive does NOT store the corpus embedding table.** The temptation to put
it alongside is strong (and I tested it): it would cost 4 KB/window instead of
20 B and, worse, turn the archive into a leak — ready-made answers to its own
queries inside the file. The external encoder is a dependency of the archive, the
way libc is for a binary.

### Encoder registry and consistency check

Since the encoder is a dependency, a reader must know **where to get it** and be
able to **verify it got the right one**. Both mechanisms are built in:

1. **`meta.json` → `encoder_kb`** — a registry: model, backend, install command,
   URL, dimensionality. Enough to bring up the required encoder on a clean machine:

   ```json
   "encoder_kb": {
     "model": "bge-m3", "backend": "ollama",
     "how": "ollama pull bge-m3",
     "api": "http://localhost:11434/api/embed",
     "d": 1024, "normalize": true, "unit": "window"
   }
   ```

2. **Control windows `meta.json` → `enc_probe`** — at build time, 5 corpus windows
   and their vectors are stored (≈25 KB). Before reading, Fracode encodes those same
   windows with the reader's encoder and **searches for them in the archive**: if it
   finds the right window, the channel is alive.

   **The criterion is READABILITY, not "is it the same vector".** Fixed 2026-09-12:
   the check used to accept `cos ≥ 0.999`, and that produced a **false rejection**.
   Measured: `snowflake-arctic-embed2` has `min cos = 0.435` against a bge-m3 archive
   (the 0.999 threshold would reject it), yet it **reads the archive at
   `recall@4 = 0.227` vs a ceiling of 0.319** — 71% of the channel is alive. The
   right question is "does it find the right window", not "does the vector match".

   | reader encoder | probe_recall@4 | min cos | verdict |
   |---|---|---|---|
   | bge-m3 (own) | **1.0** | 1.000 | reads |
   | snowflake-arctic-embed2 | **0.8** | 0.435 | **reads** (old threshold lied) |
   | nomic-embed-text (d=768) | — | — | refused: different dim, needs an adapter |

   Threshold: `probe_recall@4 ≥ 0.6` (3 of 5 windows). `cos` stays in the response
   as diagnostics, but **no verdict is based on it**.

   If the encoder is wrong, `recall` and `memory_verify` return an **explicit
   refusal** with the reason, instead of a silent zero recall:

   ```json
   {"ok": false,
    "error": "encoder returned d=768 but the archive requires d=1024: direct search is impossible. An ADAPTER is needed …",
    "hint": "Use an encoder of the same dimension (see meta.encoder_kb), or train an adapter."}
   ```

This is "the encoder travels with the memory" not as a slogan but as a protocol:
memory + the specification of how to read it + a check that the specification is met.

## Transferring memory between agents

Format `.frk1x` (npz). What travels:

| Component | B/token | Why |
|---|---|---|
| `codes.npy` | 24.0 | node addresses — architecturally neutral |
| `cbook_l0/l1.npy` | — | without them the codes are noise |
| `ids.npy` | 4.0 | lexical layer |
| `encoder_w.npy` | — | **encoder A**: without it B cannot build a query. **Internal encoder only** |
| `enc_probe.npy` | — | **probe windows**: lets B check it is reading with the right encoder |
| `tokenizer.json` | — | **tokenizer A**: otherwise positions drift from the text |
| `meta.json` + `store.json` | — | archive description + text records and edges |

Size depends on the archive: **2.1 MB** for a 10M-scale archive (31,595 windows),
48.7 MB for a v8 archive with an internal encoder (its embedding table ships
along). For an **external** encoder (bge-m3 via Ollama) `encoder_w.npy` is
deliberately absent — the package states plainly in its manifest that B must run
the same encoder.

### End-to-end transfer check

`_test_transfer.py` runs the full round: export → import into a new directory →
a **second API** (emulating machine B) → the same queries. Result
(`evidence/transfer_roundtrip.json`):

| | |
|---|---|
| positions identical to agent A | **20/20** |
| text identical | **20/20** |
| cross-check via `memory_verify` | `recall = 1.0`, position delta **0.0** |
| encoder check at B | `ok=true`, `min cos = 1.0` |

The MCP layer is checked separately with a real stdio launch of the server
(`_test_mcp_stdio.py`, `evidence/mcp_stdio_test.json`): 14 tools, `ok=true` answers,
and with a dead API a tool returns `ok=false` **without crashing the server**.

### A negative result worth knowing

I tested the hypothesis that the memory would be readable by a **foreign**
encoder. It is **refuted**: 15 candidates, `recall@8 = 0.000` for every one of
them, median miss 87k–172k tokens. The reason: the codebook is a derivative of
the encoder — 256 centroids describe one specific space.

So the honest phrasing is:

> **Different models — shared memory. But the encoder travels with the memory.**

That is weaker than "a pure artifact anyone can read", but it is measured.
Artifact: `evidence/cross_encoder_NEGATIVE_2026-09-11.md`.

## Architecture

1. **Archive** — BPE-tokenized text cut into windows; each window is quantized by
   a product quantizer over the micro-model's embeddings.
   `frakod_index_build.py`.
2. **Retrieval** — ADC screening over 24-B codes, plus a lexical exact layer
   (rare query words matched as exact token-id subsequences in `ids.npy`,
   IDF-ranked). Hybrid via `/recall`, mode `auto`. `frakod_api.py`.
3. **MCP** — stdio server `frakod_mcp.py`, 14 tools. Addresses are
   `frk1:<slot>.<id>`; `link`/`chain` accept both an address and a bare id.

| Tool | What it does |
|---|---|
| `remember` / `link` / `resolve` / `chain` | text memory + edge graph |
| `recall` / `recall_vector_only` | search the archive (hybrid / pure 24 B) |
| `index_stats` | byte-level accounting of the archive |
| `ask_llm` | ask an LLM with context pulled from memory |
| `memory_manifest` | what is transferable, what is not, and why |
| `memory_export` / `memory_import` | transfer without the encoder (`frk1x/1`) |
| **`memory_export_selfcontained`** | **transfer with the encoder (`frk1x/2`)** |
| **`memory_import_selfcontained`** | **deploy a standalone archive** |
| `memory_verify` | **independent verification of a transfer** |

```json
{"mcpServers": {"fracode": {
  "command": "python",
  "args": ["frakod_mcp.py"],
  "env": {"FRK_API": "http://127.0.0.1:8781"}
}}}
```

## Quick start

```bash
pip install -r requirements.txt

# 1) build the archive on an EXTERNAL encoder (recommended path — see
#    "The encoder is the main bottleneck"): 20 B/window, an order of magnitude
#    better retrieval than the in-house v8 encoder
FRK_ENCODER=ollama FRK_EMBED_MODEL=bge-m3 FRK_TOK=tok_v8_dialog.json \
FRK_NO_KEYS=1 FRK_CORPUS=my_history.txt FRK_OUTD=frakod_index \
  python frakod_index_build.py

# 2) start the API. Direct launch now works (entry point added 2026-09-12):
FRK_IDXD=frakod_index FRK_PORT=8781 python frakod_api.py
#   equivalent to: python -m uvicorn frakod_api:app --port 8781

# 3) point any MCP client at frakod_mcp.py
```

Ollama with `ollama pull bge-m3` on `localhost:11434` is required. If `FRK_IDXD`
contains no `codes.npy`, startup fails with an explicit message instead of
silently doing nothing. Before 2026-09-12 `python frakod_api.py` exited silently
with an empty log — there was simply no entry point; now there is.

The vector channel needs an embedder checkpoint; the lexical layer needs the
corpus you encoded. Set `FRK_CORPUS`, otherwise it is searched next to the
package. The archive directory can be set via `FRK_IDXD`; without it `frakod_index`
is used, and if that is absent, the newest `frakod_index_v*` by build time.

**Measured latency** (207 questions, live HTTP, `evidence/latency_N*.json`): full
round trip **90.7 ms** at 7,934 windows and **104.5 ms** at 31,595; of that the search
itself is **1.1 and 3.0 ms** — the rest is the external encoder.

**How much text `/recall` returns.** For a window archive, ±`FRK_WIN_CTX` windows
around the hit are returned (default 1, i.e. 1,920 characters; truncation via
`FRK_TXT_CAP`, default 2000). This is not cosmetic: with text truncated to 600
characters the fact appeared in the output 44% of the time; with ±1 window it is
**59%** (`evidence/bench_fact_ctx*.json`, gain +0.140, p = 3.7e-9, paired test).

**Two versions of the "fact found" metric — quote the strict one.**

| | k=4 | k=8 |
|---|---|---|
| `snippet_hit` — soft, any 40 consecutive characters | 0.4783 | 0.5894 |
| **`snippet_full` — strict, the whole fragment** | **0.3913** | **0.5121** |

The soft one overstates by 0.077. Sanity check: on the synthetic demo corpus it
reports **exactly 1.0000** at recall 0.21 (40 characters of a templated phrase are
found anywhere), while the strict one gives 0.208 = recall. A metric that shouts
"found it" at every answer must never be shown alone.

## Try it out of the box (without your data)

Every number in this README comes from corpora that cannot be published. So the
repository ships a synthetic demo corpus and a ready-made bench for it — five
minutes and you have a working archive and **your own** numbers:

```bash
# 1) demo corpus (1.2 MB) and questions for it
python make_demo_corpus.py --out demo_corpus.txt --n 2000
python make_bench.py --corpus demo_corpus.txt --out demo_bench.json --n 120
#   with no generative model: add --no-llm (queries built from rare terms)

# 2) build the archive and start the API
FRK_ENCODER=ollama FRK_EMBED_MODEL=bge-m3 FRK_TOK=tok_v8_dialog.json \
FRK_NO_KEYS=1 FRK_CORPUS=demo_corpus.txt FRK_OUTD=frakod_index_demo \
  python frakod_index_build.py
FRK_IDXD=frakod_index_demo FRK_PORT=8781 python frakod_api.py

# 3) measure
FRK_BENCH=demo_bench.json python _bench_fact.py
```

Expected on the demo (`evidence/bench_fact_demo.json`): recall **0.208**,
`snippet_full` **0.208**, nonsense control **0.0000**. The numbers are deliberately
lower than on real text: the corpus is templated and its fragments are hard to
tell apart. The demo proves the pipeline is alive, not that retrieval is good.

For your own numbers: `make_bench.py --corpus yours.txt`.

**Proxies.** If `http_proxy` is set in the environment, `urllib` sends even
`127.0.0.1` through it and the local API answers 502 — while it is alive and
`curl` works. The API, the MCP server and the benchmark bypass the proxy for
local addresses (`ProxyHandler({})`); do the same for any new call.

**Fixed 2026-09-12:** with a window-based archive `/recall` returned an **empty
`text`** — positions were found but the content was blank, because a window index
was treated as a token index. Text for `unit == 'window'` is now read from the
corpus by the window's character range.

## Repository layout

```
frakod_api.py  frakod_mcp.py  frakod_index_build.py   product: API, MCP, archive builder
make_bench.py  make_demo_corpus.py                    bench for any corpus + demo data
_bench_fact.py  _test_mcp_stdio.py  _test_transfer.py tests and benchmarks (runnable)
_diag_fact_paired.py  _bench_latency.py               statistics and latency
evidence/                                             source file for every number here
docs/                                                 roadmap, audit, transfer protocol
models_pc.py  train_extmem.py  frk1x.py               model and training (research track)
```

**Why the files are `frakod_*` while the product is Fracode MCP.** The internal
infrastructure (file names, `FRK_*` variables, the `frk1x` package format) grew
historically, and renaming it breaks the night pipelines and the tests. Outward,
the product is called Fracode only: that is the name it reports over MCP
(`serverInfo.name = fracode`) and the name used throughout the documentation.

## What is honestly NOT shown

The list is short but fundamental:

- **A model trained with Fracode memory vs a KV transformer at equal memory** —
  not shown.
- **Speed.** Fracode is **slower** than the KV cache: 1.63× with fast ADC (at the
  cost of +2.2% PPL) and up to 5.2× on the exact path. Speedup cannot be claimed
  in any form — only the trade "memory for speed".
- **10 million tokens** — the promised "~280 MB" is an extrapolation
  (24 B × 10M); no archive of that size has actually been built.
- **`recall@8` ≠ "the needed thought was found"** — it is position recovery with
  a tolerance. No product-level benchmark exists yet.
- **The memory is not readable by a foreign encoder** — a measured negative
  result, see above.
- **Transport** — a file moved by hand. No object storage, no hash addressing,
  no incremental deltas.

## Why it is cool (no exaggeration)

- **Writing is free**: zero LLM calls per ingest — unlike schemes that run every
  fact through a model.
- **A standalone archive is 17.7× smaller than the full footprint**: 48.7 MB
  against 860.6 MB, with `keys_f16` (802 MB) discarded and no loss of quality.
- **The memory is a file.** It can be handed over, copied, dropped into storage.
  It does not live inside a model and is not rebuilt from text on a new host.
- **Runs as a plain MCP server** — plugs into any agent.
- **All local**, GPL-3.0, pure Python, ~1.5k lines. No external APIs.
