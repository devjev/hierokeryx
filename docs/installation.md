# Installation

`hierokeryx` requires **Python 3.13** (the typing features used internally
need the 3.13 syntax). The package itself is a pure-Python install; the
heavier ML dependencies it pulls in (`gliner`, `sentence-transformers`,
`torch`) are imported lazily so a fresh `import hierokeryx` is cheap.

## With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is the fastest way to get a working
environment.

```bash
# In an existing project:
uv add hierokeryx

# Standalone playground:
uv init hkx-playground && cd hkx-playground
uv add hierokeryx
```

To use the `hkx` CLI without committing the dep, run it ad-hoc:

```bash
uvx --from hierokeryx hkx schema init --out schema.yaml
```

## With `pip`

```bash
pip install hierokeryx
```

## GPU extra

The default install runs GLiNER and the sentence-transformer embedder on
CPU, which is fine for documents up to a few hundred KB. For larger batches:

```bash
uv add 'hierokeryx[gpu]'
# or
pip install 'hierokeryx[gpu]'
```

This pulls in `torch>=2.5` with CUDA support. See
[Run on GPU](how-to/run-on-gpu.md) for environment variables, Nix
considerations, and Apple Silicon notes.

## NixOS / Nix devshell

PyPI ML wheels are linked against `libstdc++` / `libgcc_s` / `libgomp` and
will not run on NixOS without help. The repo ships a `flake.nix` that
provides Python 3.13, `uv`, `ruff` (the PyPI ruff binary needs patching),
and the C++ runtime libs on the loader path. If you're hacking on
`hierokeryx` itself or running it on a NixOS host:

```bash
git clone https://github.com/jevtarassov/hierokeryx
cd hierokeryx
nix develop          # or: direnv allow, if you use direnv
uv sync --all-groups
```

A `.envrc` is included for direnv users.

## Anthropic API key

Coreference and cross-document tie-breaks call Claude. Set one of these
before running the pipeline:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or, if you only want to extract spans without resolution, pass
`--no-llm`-style flags (see the [CLI reference](reference/cli.md)) or call
[`GLiNERExtractor.extract`][hierokeryx.extract.gliner_runner.GLiNERExtractor]
directly without an `LLMClient`.

## First-run model download

The first call to GLiNER downloads the underlying model (`urchade/gliner_*`,
~1.7 GB) to your HuggingFace cache. The Nix devshell sets `HF_HOME` to
`./.models/` so the download stays inside the project tree; otherwise it
goes to `~/.cache/huggingface/`. The download has no progress indicator in
`hkx` v0.1 — see [Troubleshooting](troubleshooting.md) if it appears to
hang.

## Verify the install

```bash
hkx --help                            # CLI is on PATH
python -c "import hierokeryx; print(hierokeryx.__version__)"
```
