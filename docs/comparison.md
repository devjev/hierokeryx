# Comparison with alternatives

`hierokeryx` overlaps with several existing tools. Honest summary of
when each is the right pick.

## TL;DR

| Tool                             | Pick when…                                                                       |
|----------------------------------|----------------------------------------------------------------------------------|
| **`hierokeryx`**                 | You need domain-specific entities + coreference + HITL on hundreds–thousands of docs. |
| **spaCy**                        | You need fast, deterministic NER on a closed schema and don't need coreference.  |
| **GLiNER (used directly)**       | You only need zero-shot spans and will resolve / cluster yourself.               |
| **LangChain extraction**         | You're already deep in LangChain and want quick LLM-only structured extraction.  |
| **stanza / flair**               | Academic NER benchmarks, well-supported European languages, classic CoNLL types. |

The rest of the page goes deeper.

## vs spaCy

spaCy is mature, fast, and excellent at the things it's designed for:
classic CoNLL-style NER (`PERSON`, `ORG`, `LOC`, `DATE`, etc.) on text
where the entities are similar to its training distribution.

**spaCy is better when:**

- You're processing tens of thousands of documents per minute.
- Your entity types match its trained labels.
- You need a pipeline that runs without an internet connection.
- You don't care about coreference.

**`hierokeryx` is better when:**

- Your entity types are custom (drugs, parts, parties, gene symbols)
  and you don't want to label training data.
- You need within-document coreference resolution.
- You need cross-document entity clustering.
- You want a JSONL review loop that doesn't require building a UI.

You can also use both: spaCy for the easy types where its precision is
excellent, `hierokeryx` for the domain-specific extras. Output formats
won't line up out of the box, but the offsets are interoperable.

## vs GLiNER used directly

[GLiNER](https://github.com/urchade/GLiNER) is what `hierokeryx` uses
under the hood for the span extraction step. If your only need is "give
me zero-shot character spans for these types", call GLiNER directly —
you'll save the cost of the LLM and the Pydantic layer.

**GLiNER alone is better when:**

- You only need spans, not entities.
- You will do your own coreference / clustering / canonicalisation.
- You don't care about a workdir or HITL.

**`hierokeryx` is better when:**

- You want pronouns and short forms resolved to their referents.
- You want cross-document clustering with calibrated confidence.
- You want the JSONL HITL loop, the linter, and the workdir layout.

`hierokeryx`'s value-add over GLiNER is largely the resolution and review
layers — if you don't need those, GLiNER raw is a leaner choice.

## vs LangChain / LlamaIndex extraction

LangChain's `with_structured_output` and LlamaIndex's `Extractor` both
let you ask an LLM to extract structured data from text given a Pydantic
schema. They're popular because they're simple.

**LangChain / LlamaIndex extraction is better when:**

- You're already using their broader framework.
- You can tolerate the LLM hallucinating character offsets (or you
  don't need offsets at all).
- Your input documents are short enough that the LLM can extract in
  one pass.

**`hierokeryx` is better when:**

- You need accurate character offsets you can highlight in a UI.
- You need coreference and cross-document clustering.
- You want HITL with a stable file format.
- You can't tolerate "hallucinated spans" silently corrupting your
  output.

The core difference: those frameworks treat span extraction as just
another LLM output. `hierokeryx` treats character offsets as a fact
produced by a different model, then asks the LLM to do *only* the
reasoning. See
[Why GLiNER + LLM](concepts/why-gliner-plus-llm.md) for the rationale.

## vs stanza / flair / transformers NER

The HuggingFace ecosystem has many trained NER models — `stanza`,
`flair`, fine-tuned BERTs from `transformers`. They're typically very
good at the labels they were trained on and very bad at anything else.

**A fine-tuned model is better when:**

- You have labeled training data.
- Your entity types are stable across years.
- You can spend the time to train, evaluate, and re-train per
  drift event.

**`hierokeryx` is better when:**

- You don't have training data.
- Your entity types shift between corpora.
- You'd rather edit a schema than re-train.

For most practical purposes you don't have to choose — `hierokeryx`'s
GLiNER step does what a fine-tuned model would, with a schema swap
instead of retraining.

## vs commercial NER APIs

Google Document AI, AWS Comprehend, Azure Cognitive Services all offer
NER as a service. They're rock-solid for their built-in entity sets and
have great SLAs.

**A commercial NER API is better when:**

- You need an SLA-backed service.
- You're inside the cloud provider's ecosystem.
- The built-in entity types cover your domain.

**`hierokeryx` is better when:**

- You need custom entity types.
- You need coreference, not just NER.
- You don't want your text leaving your perimeter (run GLiNER + a
  self-hosted LLM).
- You want to control the review tooling.

## What `hierokeryx` is *not* good at

To stay honest:

- **Streaming / sub-second latency.** Every LLM round-trip dominates.
  If you need real-time, build something else.
- **Open-vocabulary discovery.** You must declare a schema. There's no
  "find all the interesting things in this document" mode.
- **Relations and events.** Out of scope at v0.1. We extract entities,
  not edges between them.
- **Sub-token spans.** Token boundaries are GLiNER's unit. If you need
  to highlight a single character (a morpheme, a Chinese character
  within a compound), this isn't the tool.
- **Free / zero-marginal-cost runs.** GLiNER and embeddings are local;
  the LLM is a paid API. You can swap the LLM for a self-hosted model
  but you'll spend more wall-clock time.

## Migration paths

Coming from another tool:

- **From spaCy**: keep spaCy for high-volume base NER, add `hierokeryx`
  on top for custom types and HITL. The character offsets compose.
- **From LangChain extraction**: replace the `with_structured_output`
  call with `pipeline.run_one`. The schema translates directly.
- **From a hand-rolled GLiNER pipeline**: drop the GLiNER call and
  use [`GLiNERExtractor`][hierokeryx.extract.gliner_runner.GLiNERExtractor]
  — it handles overlap resolution, token snapping, and lazy model
  loading.
