# Why GLiNER + LLM

`hierokeryx` splits entity extraction into two phases on purpose: a small
deterministic NER model finds the spans, an LLM does the reasoning. This
page explains why.

## The two halves of "extract entities from text"

Naively, "extract entities" is one task. In practice it's two very
different ones:

| Task                         | Hard part                                   |
|------------------------------|---------------------------------------------|
| Finding character offsets    | Tokenisation, recall over rare types        |
| Resolving who refers to whom | Pronouns, short forms, world knowledge      |

The first task wants a small, fast model whose outputs are mechanical
properties of the text. The second wants reasoning over the document as a
whole.

## What happens when an LLM does both

If you ask Claude or GPT to "extract entities with character offsets",
you'll see two failure modes within a few thousand documents:

1. **Off-by-one offsets.** The model knows the entity is there but
   picks indices that don't line up with the source text — typically off
   by one or two characters around whitespace and punctuation.
2. **Hallucinated text.** The model returns a span whose `text` field
   doesn't appear verbatim in the document. This is especially common
   for names with non-ASCII characters or hyphenation.

Both are silent failures: nothing crashes, your downstream code happily
indexes into the text and shows a highlighting offset that drifts by one
character.

We took the position that **character offsets are facts, not judgments.**
A model that doesn't know the offset shouldn't be asked to invent one.

## Why GLiNER for the span half

[GLiNER](https://github.com/urchade/GLiNER) is a zero-shot NER model
designed to be small (a few hundred MB), fast (CPU-runnable), and
schema-driven — you give it the type names and descriptions you care
about and it returns spans with character offsets. Mechanical things:

- The offsets are guaranteed to be valid indices into the input text.
- It handles non-ASCII and whitespace correctly because it operates over
  tokens, not generated text.
- It's deterministic given the same model and threshold.

It is *not* good at: pronouns, short forms ("she", "the doctor"), or
cross-sentence coreference. That's fine — we're not asking it to.

## Why an LLM for the reasoning half

Pronoun resolution, picking a canonical form, deciding "is 'M. Curie' in
doc B the same as 'Marie Curie' in doc A" — these are reasoning problems.
A reasoning model is the right tool.

Specifically, the LLM is given:

- The full document text (or a cross-doc context window).
- The list of spans GLiNER already found, with their offsets.
- A schema describing the entity types.

And asked: cluster these spans into entities. Pick canonical forms. Set
a calibrated confidence. Never invent an offset that isn't already in
the input.

The LLM literally cannot break offsets in this pipeline, because it
never produces them — it produces `mention_ids` referring to spans
GLiNER already found.

## What we lose by not letting the LLM extract spans

A few rare-tail cases. GLiNER's recall isn't perfect — if your text
contains an entity type GLiNER doesn't recognise at all, the LLM never
gets to "rescue" it.

In practice this happens mostly for:

- Entity types where the description is too vague (fix: improve the
  schema).
- Multi-word entity boundaries that span across GLiNER's token
  boundaries (fix: lower `--threshold`).

For these cases the human review loop is the safety net. A reviewer can
add a missed entity with `op: add` in the JSONL.

## When this design is wrong

If your problem is **structured information extraction** (date + amount +
account from an invoice; ingredient + dose from a label), `hierokeryx`
is overkill. You want a smaller pipeline: an LLM with a tightly-
constrained output schema, no NER step needed, because the fields are
known up front and there's no coreference to do.

If your problem is **open-domain entity discovery** with no schema at
all, `hierokeryx` won't help — the schema is required.

## Further reading

- [Why JSONL HITL](why-jsonl-hitl.md) — the complementary "humans review
  what the model is unsure about" half of the design.
- [Confidence math](confidence-math.md) — how scores from both phases
  combine into a single per-entity confidence.
- [GLiNER paper](https://arxiv.org/abs/2311.08526) — the underlying NER
  model.
