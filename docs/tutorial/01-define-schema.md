# 1 · Define a schema

The schema tells `hierokeryx` what to look for. It is a list of entity
types, each with a name, a short description, and a few examples. Both
GLiNER (the zero-shot extractor) and the LLM (the resolver) read the
description and examples, so make them crisp.

## Generate a starter schema

```bash
hkx schema init --out schema.yaml
```

This writes a default schema with `Person`, `Organization`, and `Location`.
Open it up:

```yaml
version: "1"
types:
  - name: Person
    description: A named individual human being.
    examples:
      - Marie Curie
      - Albert Einstein
  - name: Organization
    description: A company, government agency, university, or similar institution.
    examples:
      - Apple Inc.
      - University of Cambridge
  - name: Location
    description: A geographic place — country, city, region, address.
    examples:
      - Paris
      - 221B Baker Street
```

For this tutorial keep `Person` and `Organization`, drop `Location`:

```yaml
version: "1"
types:
  - name: Person
    description: A named individual human being.
    examples:
      - Marie Curie
      - Albert Einstein
  - name: Organization
    description: A company, government agency, university, or similar institution.
    examples:
      - Apple Inc.
      - University of Cambridge
```

## Validate it

```bash
hkx schema validate schema.yaml
```

You should see something like:

```
OK: schema.yaml — 2 type(s), fingerprint a1b2c3d4...
  - Person: A named individual human being.
  - Organization: A company, government agency, university, or similar institution.
```

The `fingerprint` is a stable hash of the schema content. Every artifact
written under your workdir is stamped with this fingerprint, so you can
detect "did this extraction run against the same schema?" without
hand-comparing YAML. See [Determinism](../concepts/determinism.md) for
why this matters.

## Writing a good schema

A few rules of thumb learned from running this pipeline on real
documents:

- **One concept per type.** `Person` not `PersonOrCharacter`. If you need to
  distinguish, declare two types.
- **Description as a definition, not a hint.** Both models read it as the
  spec for what should be included. "A company, government agency,
  university, or similar institution" is better than "businesses".
- **3–5 examples covering the variety.** Examples grounded in your actual
  domain beat generic ones. For a legal corpus, use `Person: Justice
  Sotomayor / John Doe (plaintiff) / opposing counsel` rather than
  `Albert Einstein`.
- **Keep schemas small.** GLiNER's recall starts dropping past ~15 types
  per call. If you need 30 types, run two passes with two schemas and
  merge the workdirs.

For a worked example of a domain-specific schema, see
`examples/custom_schema.yaml` in the repo (drugs / doses / conditions for
a medical extraction).

## Library API equivalent

If you'd rather build the schema in Python:

```python
from hierokeryx import EntitySchema, EntityType
from hierokeryx.schema import save_schema

schema = EntitySchema(types=[
    EntityType(
        name="Person",
        description="A named individual human being.",
        examples=["Marie Curie", "Albert Einstein"],
    ),
    EntityType(
        name="Organization",
        description="A company, government agency, university, or similar institution.",
        examples=["Apple Inc.", "University of Cambridge"],
    ),
])

save_schema(schema, "schema.yaml")
```

See the [`hierokeryx.schema`](../reference/api/schema.md) and
[`hierokeryx.models`](../reference/api/models.md) reference for the
full API.

[Next: run the pipeline :material-arrow-right:](02-run-pipeline.md){ .md-button .md-button--primary }
