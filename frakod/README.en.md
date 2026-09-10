# Fracode MCP — long-term memory for agents

[Russian](README.md) | [English](README.en.md)

External long-term memory for LLM agents: the archive of chats and code lives in
compressed form, and the agent pulls context from it over MCP instead of
re-uploading history into the prompt on every query.

## Where the idea came from

A micro-model (STS-Prog, ~11M params) was trained to compress sequences — the
experiment lives in this repo, branch `pc-carroll`. The model predicted the next
token through synchronous driver layers, and its byproduct — a compact token
embedding table — served as a retrieval key. The memory grew out of a model that
learns to compress.

## What is honest here and what is not

There are two independent mechanisms, and they must not be conflated:

| | Archive (`codes.npy`) | Operational memory (`/remember`) |
|---|---|---|
| Stores | 24 B/token of codes + keys for rerank | the raw record text, in full |
| Written | offline, from the corpus | at runtime, by the agent |
| Uses the compression | yes | **no** |
| File | `frakod_index/*.npy` | `frakod_store.json` |

**"24 B/token" refers to `codes.npy` only.** That is the compression itself, and
it powers ADC candidate screening. But the archive's on-disk footprint is larger,
because exact reranking reads `keys_f16.npy` (384 B/token) and the lexical layer
reads `ids.npy` (4 B/token). The full honest total comes from `/index/accounting`:

```
N = 2,927,882 tokens
codes.npy     =    67.0 MB    24.0 B/token   <- this is the "24 B"
keys_f16.npy  = 1,072.2 MB   384.0 B/token   <- candidate rerank
ids.npy       =    11.2 MB     4.0 B/token   <- lexical exact layer
--------------------------------------------------------------
total         = 1,150.8 MB   412.1 B/token
```

`keys_f16` need not be resident: the rerank reads them via `mmap` from disk.

## Channel attribution (the number that matters)

One archive, the same questions, different search channels — `test_24b_proof.py`
(after the lexical fix, cluster window `win=16`). Two archives: the old one
(BPE-512, per-character) and the **one rebuilt on v7** (BPE-8192, d=256, S=16).

| search channel | old archive | **v7 archive** | median |
|---|---|---|---|
| ADC over codes, question query | 0.00 | **0.30** | 638 ms |
| codes + `keys_f16` rerank, question query | 0.00 | **0.60** | 49 ms |
| lexical exact layer | 1.00 | **1.00** | 17 ms |
| hybrid (`auto`) | 1.00 | **1.00** | 64 ms |
| ADC over codes, snippet query | 0.00 | **0.00** | 648 ms |
| codes + rerank, snippet query | 0.00 | **0.17** | 53 ms |

**What this honestly means.** The rebuild on a consistent tokenizer **turned the
vector channel on**: 0.00 → **0.60** (question + rerank). This confirms the
diagnosis — the channel was not killed by the "cost of 24 bytes" but by a
misalignment between tokenizer, embedder and query. But:

- **on snippets the vector barely works** (0.00–0.17) — a short query without the
  template loses the needle in the averaged window;
- **ADC alone gives 0.30** — 24-byte compression finds it, but not precisely
  enough; the `keys_f16` rerank lifts it to 0.60;
- **the hybrid is still held up by the lexical layer** (1.00).

Strict metric caveat: `recall@4` = "the needle word appears in the text of any of
the 4 returned windows". Strict precision@4 of the lexical layer is **0.25** on
questions, **0.29** on snippets.

A note on size: `24 B/token` is achievable only at **d=192, S=12**. The v7 archive
(d=256, S=16) has a code of `(N,2,16)` = **32 B/token**; keys are 512 B/token.
The cheaper/more-accurate trade-off is a deliberate choice.

## 24 B/token: proven on a consistent archive

The scheme works when **the query tokenizer matches the archive tokenizer** and the
query is built by the same operator as the keys. Verification — `test_24b_code.py`
and `_qkey_check.py` on the `frakod_index_code10m` archive (10M tokens, code
corpus, BPE-8192):

| query method (true position known) | top-1 | top-4 | cos |
|---|---|---|---|
| **context query, window RAD=4** (as in the builder) | 14/16 | 14/16 | **1.000** |
| monotoken query `E[ids].mean(0)` (old API path) | 8/16 | 8/16 | 0.958 |

Through the live API on the same archive, 16 positions out of 10M, ADC over 24
bytes:

| channel | top-1 | top-4 | top-8 |
|---|---|---|---|
| ADC over 24 B only | 0/16 | 3/16 | 7/16 |
| 24 B + `keys_f16` rerank | 0/16 | 3/16 | 7/16 |

Conclusion: **24-byte compression carries usable information** — 7 out of 16
queries land within ±7 tokens, picking 16 candidates out of 10,000,000 (625,000×
denser than chance). The zero on the Hermes archive is explained by two concrete
defects, not by the physics of compression (see below).

## Four root causes found during the audit

1. **The Hermes archive was built with a per-character BPE-512.** Before the fix
   `meta.json` carried neither the tokenizer nor its kind; `ids.npy` holds only
   values 0…511 and Cyrillic is encoded one letter at a time (`фотосинтез` → 2
   fragment tokens). The embedder `sts_prog_seed0.pt` is (512, 192) — consistent
   with the archive, but there is no semantics in a bag of characters.
   **Verified and fixed:** separation of unrelated Russian texts by cos — seed0
   (512): 0.47…0.90 (mean 0.70, near-collinear) vs v7 (8192): 0.17…0.38 (mean
   0.30). The archive was rebuilt on v7 → the vector channel came alive (0.60).
2. **Query/key asymmetry in `frakod_api.py`.** Archive keys are a context average
   with window RAD=4, while the query was computed as `E[ids_q].mean(0)` with no
   averaging. On the code archive this dropped cos from 1.000 to 0.958 and pushed
   top-1 by +4…5 tokens (8/16 instead of 14/16). **Fixed:** the query now uses the
   same sliding average, and `RAD` is read from `meta.json`.
3. **Two different embedders in one process.** `recall` (archive) ran on
   `sts_prog_seed0.pt` (d=192, vocab=512), while `dvec` (`/resolve`, operational
   memory) ran on `ckpt_v7_night_50k.pt` (d=256, vocab=8192).    **Fixed:**
   `_sts_embed_table()` takes the checkpoint from `meta.embed_ckpt`, and
   `load_embedder()` (which serves `dvec`, `/stats`, `/window`) now also reads
   both the table and the tokenizer from `meta.json` — the table is pulled
   straight from the checkpoint, no model built. `vocab`/`d` are checked against
   the archive, with a clear error on mismatch. Verified: v7 archive →
   `(8192, 256)` + `tok_v31`, v8 archive → `(8192, 192)` + `tok_v8`. The
   `/resolve` and `/recall` vector spaces now coincide.
4. **`embed_ckpt` was written as a basename, not a path.** The builder put only
   the file name into `meta.json`, so if the checkpoint did not sit next to
   `frakod_api.py` (e.g. `phase01/exp_vq/ckpt_v8_voc8k.pt`), the API **silently
   fell back** to `sts_prog_seed0.pt` — putting the archive keys and the query
   back into different embedding tables, exactly like root cause #1.
   **Fixed:** `meta.json` now carries the absolute path (`embed_ckpt`) plus the
   name (`embed_ckpt_name`); the API resolves both and, if the required
   checkpoint is missing, **fails with a clear error** instead of substituting a
   different embedder. A check that the embedder's `d` matches the archive's `D`
   was added. Verified: the v8 archive now loads exactly `ckpt_v8_voc8k.pt`
   (8192, 192).

## Bench against RAG and Mem0

Hermes history slice, 30 needle questions (25 actually used — that is what the
generator produced after dedup), one chat node. Artifact — `bench_results.json`,
reproduction — `bench_memory_systems.py`.

| | recall@4 | prompt tokens | search | LLM tokens on write | storage |
|---|---|---|---|---|---|
| no memory | 0.00 | 151 | — | 0 | — |
| RAG (chroma) | 0.24 | 263 | 27 ms | 0 | 20.3 MB |
| Mem0 | 0.08 | 684 | 63 ms | 1,686,418 | — |
| **Fracode** | **1.00** | 560 | 17 ms | **0** | **1,150.8 MB** |

Honest caveats:

- **The queries are not equivalent.** RAG and Mem0 receive a short question.
  Fracode in the original bench received a 160-char fragment literally containing
  the needle word — and the needle occurs exactly once in the corpus by
  construction. The bench now measures both protocols (`query` and `snippet`);
  the table above uses `query`, i.e. equal conditions. On `snippet`, Fracode also
  scores 1.00.
- **Fracode wins by the lexical exact layer, not by the vector.** That is exact
  word lookup over `ids.npy` — effectively an inverted index. This result is
  expected for the task "a rare word occurs once in the corpus" and does not mean
  the vector compression is superior. Strict precision@4 is 0.25.
- **Mem0 solves a different task.** Mem0 does not search the corpus; it extracts
  structured facts with an LLM — hence 1.68M write tokens (~105% of the corpus
  size). Comparing 0.08 vs 1.00 as two implementations of one task is invalid;
  Mem0 honestly shows the price of LLM ingest.
- **Fracode storage is now counted in full** (codes + keys + ids), not codes only
  as before.

## How it works

1. **Archive** — BPE-tokenized text cut into windows; each window is quantized by
   a product quantizer over the micro-model's embeddings. `frakod_index_build.py`.
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

The vector channel needs the `sts_prog_seed0.pt` checkpoint; the lexical layer
needs the corpus you encoded. Set `FRK_CORPUS`, otherwise it is searched next to
the package.

## What to do next

1. **Lift snippet mode.** The vector gives 0.60 on a full question but 0.00–0.17
   on a 160-char snippet. The cause is the context average blurring a short query.
   Options: build the query as an average over several anchor windows, or move to
   a multi-level key (not one averaged vector per position).
2. **Pre-train the d=192 embedder with a LARGE vocabulary — overnight. The
   pipeline is ready.** The pair (d=192, vocab=8192) is **already implemented**:
   `phase01/exp_vq/train_v8_uplift.py` builds a mixed corpus (code + real
   history), trains PurePCLM d=192 l=8 and saves `ckpt_v8_voc8k.pt` +
   `tok_v8.json`. That is exactly 24 B/token of codes. Its geometry is **better**
   than v7's: cos between unrelated Russian texts min=−0.13 / max=0.23 /
   mean=0.04 versus 0.17…0.38 / mean=0.30 for v7 — the embedder discriminates
   between texts instead of pulling everything into one point. **But the
   checkpoint on disk is a smoke run** (ppl 6492–8254; v7 is ~40–100), so an
   archive built on this pair does not revive the vector channel even with the
   correct embedder: adc24 0.00, vec+rerank 0.00, lex 0.90.

   The checkpoint loads into `PurePCLM(d=192, l=8)` with **no
   missing/unexpected keys** — the architecture is right, only training is
   missing. One command:

   ```bash
   python night_v8_pipeline.py --steps 10000     # ~3 h; 8000 → ~2.4 h
   ```

   The pipeline backs up the checkpoint, trains, measures ppl before/after,
   rebuilds the archive, starts the API, runs `test_24b_proof.py` and
   `test_24b_code.py`, writes `night_v8_report.json`, and asserts
   `b_per_tok_codes == 24.0`. Validated in `--skip-train` mode — the whole chain
   passes. **This is the only known path to 24 B/token and semantics at once.**

   ppl guide (v7 fell to 454 within the first 500 steps and to 40 by 60k): at
   least 8000 steps are needed for the vector to have a chance.
3. **Drop the `keys_f16` dependency.** ADC alone gives 0.30 versus 0.60 with
   rerank — the gap is still wide, but better codebook training can close it, and
   then the permanent footprint falls from 548 to ~36 B/token.
4. **Replace the linear scan.** `_recall_cpu` walks all N tokens (~640 ms). An
   ANN/HNSW over code prefixes is needed.
5. **Decide the distribution model.** Currently GPL-3.0. Settle this before
   external PRs.

## Why it is still cool

- **Writing is free**: zero LLM calls per archive ingest.
- **The hybrid gives recall@4 1.00** at a median of ~64 ms over 2.4M tokens.
- **24-byte compression is verified and revived**: on a consistent archive (v7,
  vocab=8192) ADC alone gives 0.30; with rerank, 0.60. On the code archive through
  the live API, top-8 is 7/16 out of 10M. The scheme works once the four
  integration problems are removed.
- **The (d=192, vocab=8192) pair exists and is valid** — its geometry is better
  than v7's; only training is missing. It is the one that yields 24 B/token and
  semantics together, and its overnight pipeline is ready to run.
- **All local**, GPL-3.0, ~1.5k lines of Python.
