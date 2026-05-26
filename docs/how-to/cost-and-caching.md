# Cost and prompt caching

LLM calls dominate the running cost of `hierokeryx`. The good news is the
pipeline is designed around Anthropic's
[prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
— a stable system prompt + schema block sits in the cache prefix and
amortises across documents.

## Per-document cost model

For a typical document with `M` mentions and `K` resulting clusters:

- **One coref call**, with input ≈ `len(document) + sum(mention lengths)`
  tokens and a structured tool-use output ≈ `K × 50` tokens.
- **Zero to a few cross-doc tie-break calls**, only if the document
  contributes borderline entities. Each call is small (a few hundred
  tokens in and out).

The coref call dominates. For an English document of ~2000 words with
~20 mentions, expect roughly:

- Input: 3000 tokens (mostly cached after warm-up).
- Output: 500 tokens.
- One call per document.

Concrete back-of-envelope on Claude Sonnet pricing
(see <https://www.anthropic.com/pricing>): roughly $0.003 per typical
document on cache hits, $0.015 per document on cache misses.

## How caching works in `hierokeryx`

[`hierokeryx.llm.prompts`](../reference/api/llm.md) keeps the system
prompt invariant across calls. Specifically, `COREF_SYSTEM_PROMPT` +
`render_schema_block(schema)` together form the cached prefix.

What lives in the cache:

```text
┌─────────────────────────────────────────────┐
│ COREF_SYSTEM_PROMPT      (~1500 tokens)     │  ← cache_control: ephemeral
│ render_schema_block(...) (~200-500 tokens)  │  ← cache_control: ephemeral
└─────────────────────────────────────────────┘
   ↓ per-document message ↓
┌─────────────────────────────────────────────┐
│ document text + mentions list               │  ← not cached
└─────────────────────────────────────────────┘
```

You get the cache discount across all documents within a 5-minute window
(Anthropic's TTL). For a batch run of 100 docs, that's effectively one
cache write and 99 cache reads.

To verify caching is working:

```python
from hierokeryx.llm.anthropic_client import AnthropicClient

llm = AnthropicClient()
# ... run pipeline ...
print(f"Cache hit ratio: {llm.cache_hit_ratio():.1%}")
```

After the first document, the ratio should climb above 90%.

## Things that bust the cache

The cache key is the full prefix verbatim. Changing any of these
invalidates the prefix and forces a fresh write:

- **Editing the schema** — even adding a single example.
- **Editing the prompt strings** in
  [`hierokeryx.llm.prompts`](../reference/api/llm.md).
- **Switching Claude models** (e.g., from Sonnet to Opus).
- **A 5-minute gap** between calls.

If you're tuning the schema iteratively, expect every iteration to start
cold. Once the schema is stable, you'll see steady-state cache hits.

## When caching does not apply

The cross-document tie-break uses a different system prompt
(`CROSSDOC_SYSTEM_PROMPT`) with its own cache prefix. It also caches —
but since tie-breaks are sparse, you'll see fewer hits in absolute terms.

## Budgeting a batch run

A rough formula:

```text
cost ≈ N_docs × (input_tokens × $rate_input + output_tokens × $rate_output)
     × (1 − cache_hit_ratio × cache_discount)
```

For Claude Sonnet 4.5 / 4.6 with default settings:

- Input: ~$3 per million tokens (`$3e-6` per token).
- Cached input: ~$0.30 per million tokens (~90% discount).
- Output: ~$15 per million tokens.

So for 1000 documents at ~3000 input + 500 output each, with 95% cache
hit rate:

```text
cost ≈ 1000 × (3000 × $3e-6 + 500 × $15e-6) × (1 − 0.95 × 0.9)
     ≈ 1000 × ($0.009 + $0.0075) × 0.145
     ≈ $2.39
```

The 95% cache hit is realistic for a batch you run end-to-end; it falls
to 0% if you fan out across many short-lived processes that each cold-
start the cache.

## Cutting cost further

- **Skip the LLM tie-break.** Use `--no-llm-tiebreak` if your corpus
  has distinctive entity names. Most cross-doc clusters resolve on
  embedding similarity alone.
- **Use a smaller Claude model for coref.** Pass
  `coref_model="claude-haiku-4-5"` to
  [`AnthropicClient`][hierokeryx.llm.anthropic_client.AnthropicClient]
  for cheaper but slightly noisier clustering. Tie-break with the larger
  model (`tiebreak_model`) on the rare borderline calls.
- **Pre-filter documents.** GLiNER runs locally and is essentially free.
  If you only need entities of one type and your documents are mixed,
  filter on GLiNER output before paying for the LLM.

## What `hierokeryx` will not do for you

- **Streaming.** The pipeline buffers per-document. If you need first-
  entity latency, you're in the wrong tool.
- **Sub-second total latency.** A typical document takes 1–3 seconds end
  to end. Most of that is the LLM round-trip.
- **Free.** GLiNER + embeddings are CPU/GPU; the LLM is paid API. You
  can swap the LLM for a self-hosted model behind the
  [`LLMClient`][hierokeryx.llm.protocol.LLMClient] Protocol if you need
  zero per-call cost (see [Use a custom LLM backend](custom-llm-backend.md)).
