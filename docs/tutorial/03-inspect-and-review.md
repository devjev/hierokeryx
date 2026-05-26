# 3 · Inspect and review

You have a workdir. Now look at what you got and round-trip one entity
through the JSONL review loop.

## Inspect

```bash
hkx inspect workdir/
```

Output (your numbers will differ):

```
Run at: 2026-05-26T14:08:00.123456
  schema fingerprint: a1b2c3d4...
  documents: 3
  entities:  7
  flagged:   1

                Top entities by mention count
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ doc      ┃ entity        ┃ type    ┃ mentions ┃ confidence ┃ cluster   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ curie_1  │ Marie Curie   │ Person  │ 3        │ 0.92       │ pers_abc  │
│ curie_2  │ M. Curie      │ Person  │ 3        │ 0.65       │ pers_abc  │
│ curie_3  │ Curie         │ Person  │ 2        │ 0.88       │ pers_abc  │
│ ...      │ ...           │ ...     │ ...      │ ...        │ ...       │
└──────────┴───────────────┴─────────┴──────────┴────────────┴───────────┘

Schema: Person, Organization (fingerprint a1b2c3d4)
```

All three documents' "Curie" entities ended up in the same cross-document
cluster (`pers_abc`). One of them — the `M. Curie` in `curie_2` — has
confidence below the review threshold, so it got flagged.

## Open the flagged review file

```bash
$EDITOR workdir/review/curie_2.jsonl
```

The first line is the header (schema marker, doc id, text hash, schema
version). Each subsequent line is one entity. Compact rendering:

```jsonl
{"$schema":"hierokeryx/review/v1","doc_id":"curie_2","text_sha":"...","schema_version":"1"}
{"op":"keep","id":"ent_...","type":"Person","canonical":"M. Curie","surface_canonical":"M. Curie","confidence":0.65,"mentions":[{"id":"...","start":29,"end":37,"text":"M. Curie","score":0.71,"source":"gliner"}, ...],"cluster_id":"pers_abc","reason":"low_within_doc_confidence"}
```

The `op` field controls what happens on import:

| `op`     | Effect                                                  |
|----------|---------------------------------------------------------|
| `keep`   | Accept the entity as-is (default).                      |
| `reject` | Drop the entity entirely.                               |
| `edit`   | Replace mutable fields with this line's content.        |
| `add`    | Insert a new human-curated entity (id must start with `human_`). |

VS Code, neovim+coc, and any editor that respects the `$schema` header
will validate inline — bad field types and missing required keys light up
as you type.

## Make a change

Suppose the model picked `M. Curie` as the canonical, but you want
`Marie Curie` everywhere for consistency. Edit the line:

```jsonl
{"op":"edit","id":"ent_...","type":"Person","canonical":"Marie Curie","surface_canonical":"M. Curie","confidence":1.0,"mentions":[...],"cluster_id":"pers_abc"}
```

A few things to note:

- `surface_canonical` is the literal span that appeared in the document —
  leave it as `M. Curie` (it must quote the document verbatim).
- `canonical` is your chosen display form. It does not need to appear in
  the text.
- Set `confidence: 1.0` to signal that a human has signed off.
- Do not change mention `start`/`end`/`text` unless you are also editing
  the source document.

## Lint before importing

```bash
hkx review lint workdir/review --workdir workdir/
```

The linter cross-references each mention span against the original
document text, validates `op` semantics, and checks that types come from
your schema. A clean run prints:

```
OK 1 file(s)
```

A broken run lists every error and exits non-zero — wire this into CI if
you ship review files via PRs. See the [Troubleshooting](../troubleshooting.md)
page for common lint failures.

## Import the edits

```bash
hkx review import workdir/review --workdir workdir/
```

This replays your edits onto each
[`ExtractionResult`][hierokeryx.models.ExtractionResult] in the workdir.
The pipeline's load-bearing invariant — every mention span must quote the
source verbatim — is re-checked here, so a bad offset is rejected at
import time rather than corrupting downstream artifacts.

Now re-inspect:

```bash
hkx inspect workdir/
```

`Marie Curie` should now show up as the canonical for that entity, with
confidence `1.00`.

## You've finished the tutorial

You've done a full loop: schema → extraction → resolution → review →
import. From here:

- :material-arrow-right: For the **API**, see the
  [reference](../reference/api/index.md).
- :material-arrow-right: To wire `hierokeryx` into an application **without**
  the CLI, see [Library mode](../how-to/library-mode.md).
- :material-arrow-right: To plug in **a different LLM provider** behind the
  same pipeline, see
  [Use a custom LLM backend](../how-to/custom-llm-backend.md).
- :material-arrow-right: To understand **why the design is the way it is**,
  read the [Concepts](../concepts/why-gliner-plus-llm.md) section.
