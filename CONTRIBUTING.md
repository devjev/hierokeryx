# Contributing to hierokeryx

Thanks for considering a contribution. `hierokeryx` is alpha — the public
surface is small and changes are still cheap, so this is a great time to
file an issue, propose an API change, or send a PR.

## Quick setup

The project assumes Python 3.13 and is developed inside a Nix devshell that
also provides `uv` and the C++ runtime libraries ML wheels need.

```bash
git clone https://github.com/jevtarassov/hierokeryx
cd hierokeryx

# direnv users: the .envrc activates the devshell automatically.
# Otherwise:
nix develop

uv sync --all-groups        # runtime + dev + docs deps
uv run pytest               # unit tests
uv run mkdocs serve          # docs at http://127.0.0.1:8000/
```

If you're not on NixOS, plain `uv sync --all-groups` works on macOS / glibc
Linux distros — the flake is for NixOS, not a hard requirement.

## Running the integration tests

The integration tests under `tests/integration/` exercise GLiNER and (in
`test_pipeline_smoke.py`) the Anthropic API. They are gated by the
`integration` pytest marker and need:

- A downloaded GLiNER model (~1.7 GB on first run; cached under `.models/`
  thanks to the `HF_HOME` set in `flake.nix`).
- `ANTHROPIC_API_KEY` exported for tests that hit Claude.

```bash
uv run pytest -m integration
```

VCR cassettes under `tests/integration/cassettes/` record real API responses
so subsequent runs are offline; only re-record when prompts or tool schemas
change.

## Code style

- **Formatting & lint**: `ruff` (provided by the Nix devshell — the PyPI ruff
  binary won't run on NixOS without patching). Run `ruff check src tests`
  and `ruff format src tests` before pushing.
- **Types**: `mypy --strict` over `src/`. CI gates on this.
- **Docstrings**: Google style on every public function, class, and module.
  Module-level constants and private helpers can be terser.
- **Imports**: heavy ML deps (`gliner`, `sentence_transformers`, `anthropic`)
  are imported lazily inside the functions that need them — see
  `extract/gliner_runner.py` for the pattern. The `PLC0415` ruff rule is
  per-file-ignored for these modules.

## PR conventions

- One logical change per PR. Bug fixes and feature work in separate PRs.
- Commit messages follow Conventional Commits prefixes (`feat:`, `fix:`,
  `docs:`, `chore:`, `refactor:`, `test:`).
- Update `CHANGELOG.md` under the `[Unreleased]` heading.
- Add or update tests covering the change. New CLI flags get a unit test
  that calls the Typer app via `CliRunner`.

## Filing issues

A reproducible report beats a vague one. Include:

- The exact `hkx` command or library call you ran.
- The schema (paste the YAML).
- The documents (or a redacted snippet) that triggered the bug.
- The `workdir/` layout after the run, if relevant.
- `hkx --version`, Python version, OS.

For LLM-related bugs (wrong cluster, hallucinated id), please also attach
the JSONL workdir output for the affected document so we can replay it
against the VCR fixtures.
