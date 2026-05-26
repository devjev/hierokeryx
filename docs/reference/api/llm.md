# `hierokeryx.llm`

The LLM Protocol, the Anthropic Claude backend, and the prompt + tool
definitions everyone reuses.

To plug in a different provider, implement
[`LLMClient`][hierokeryx.llm.protocol.LLMClient] — see
[Use a custom LLM backend](../../how-to/custom-llm-backend.md) for a
worked OpenAI example.

## `hierokeryx.llm.protocol`

::: hierokeryx.llm.protocol
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - LLMClient
        - LLMError

## `hierokeryx.llm.anthropic_client`

::: hierokeryx.llm.anthropic_client
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - AnthropicClient

## `hierokeryx.llm.prompts`

The two stable system prompts (`COREF_SYSTEM_PROMPT` and
`CROSSDOC_SYSTEM_PROMPT`) and the schema-block renderer. The prompt
strings are not rendered here for readability — see the source if you
need the full text.

::: hierokeryx.llm.prompts
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - render_schema_block

## `hierokeryx.llm.tools`

JSON-schema tool definitions for Anthropic tool-use. Defined as plain
dicts so they can be converted to other providers' shapes without a
class hierarchy — see
[Use a custom LLM backend](../../how-to/custom-llm-backend.md).

Two tools are exported: `RECORD_CLUSTERS_TOOL` (used by the within-doc
coref call) and `RECORD_MERGE_DECISIONS_TOOL` (used by the cross-doc
tie-break call). Both are loaded via `from hierokeryx.llm.tools import
...`.

::: hierokeryx.llm.tools
    options:
      show_root_heading: true
      show_root_full_path: true
      filters:
        - "!^_"
