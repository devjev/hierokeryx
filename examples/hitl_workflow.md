# Human-in-the-loop workflow

The end-to-end loop, command by command.

## 1. Declare the schema

```bash
nix develop                                    # enter the dev shell (only on NixOS)
uv run hkx schema init --out schema.yaml
$EDITOR schema.yaml                            # edit to declare your entity types
uv run hkx schema validate schema.yaml
```

## 2. Run the pipeline

```bash
export ANTHROPIC_API_KEY=...
uv run hkx pipeline path/to/docs \
    --schema schema.yaml \
    --out workdir/
```

This creates:

- `workdir/schema.yaml` — the schema you used
- `workdir/extractions/<doc_id>.json` — per-doc results
- `workdir/registry.json` — cross-doc cluster registry
- `workdir/review/<doc_id>.jsonl` — flagged entities for your review
- `workdir/manifest.json` — run metadata

By default only flagged (low-confidence) entities are exported for review.
Adjust the threshold with `--review-threshold 0.6` (lower = fewer flagged).

## 3. Review

Open the review files in your editor of choice. Each line is one entity. The
first character of the line is `{"op":"keep",…}` — change `op` to:

| op       | effect                                                   |
|----------|----------------------------------------------------------|
| `keep`   | accept the entity as-is (default)                        |
| `reject` | drop the entity entirely                                 |
| `edit`   | replace the entity with this line's content              |
| `add`    | insert a new entity (id must start with `human_`)        |

Editor support: VS Code, neovim+coc, and similar tools pick up the JSON Schema
from the `$schema` header on line 1 and validate as you type.

Run the linter to catch span-text mismatches and unknown types before importing:

```bash
uv run hkx review lint workdir/review --workdir workdir
```

## 4. Re-import the edits

```bash
uv run hkx review import workdir/review --workdir workdir
uv run hkx inspect workdir
```

This replays your edits onto each extraction. The pipeline's load-bearing
invariant — every mention span must quote the document verbatim — is re-checked
on import, so a bad offset is rejected immediately rather than corrupting
downstream artifacts.
