# `hierokeryx.review`

File-based HITL review: JSONL read/write, lint, and apply.

For the design rationale see [Why JSONL HITL](../../concepts/why-jsonl-hitl.md).
For a worked example of using these helpers from your own code see
[Roundtrip JSONL programmatically](../../how-to/programmatic-jsonl.md).

## `hierokeryx.review.jsonl`

::: hierokeryx.review.jsonl
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - ReviewHeader
        - ReviewMention
        - ReviewEntityLine
        - write_review
        - write_review_dir
        - read_review
        - read_review_dir
        - review_mention_to_mention

## `hierokeryx.review.lint`

::: hierokeryx.review.lint
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - LintError
        - lint_review_file
        - lint_review_dir

## `hierokeryx.review.apply`

::: hierokeryx.review.apply
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - apply_review
