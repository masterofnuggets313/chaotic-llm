# PC-STS v2: preserving the PC + Fracode core

PC-STS v2 retains explicit Pecora-Carroll synchronization and the canonical
Fracode memory vector `e = embed(token) + position`.

It adds RMS normalization, a vector content gate, a SwiGLU residual FFN, and a
learned retrieval score that starts at zero contribution. The raw score therefore
remains the initial mechanism.

For long-context decode, use Fracode ADC with raw keys to retrieve `M`
candidates. Reconstruct their raw keys exactly from stored token ids, add the
learned reranking score, and take final top-k. Report both candidate recall@M
and final recall@k.

## Required experiment matrix

Keep tokenizer, token budget, window, optimizer schedule and evaluation seed
fixed across: baseline; baseline with auxiliary weights 0/.05/.1/.5; v2 without
learned score; v2 with learned score; and a parameter-matched Transformer.

Use the fixed checkpoint audit for held-out PPL, plus response-only PPL,
retrieval accuracy, candidate/final Fracode recall, bytes per token, and decode
tokens/s. No long-context claim is valid until all of those measurements agree.
