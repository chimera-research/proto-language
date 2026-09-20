# Development Guide

Dev workflow for `proto-language` contributors: commands, initial setup, independent fork checkouts, worktrees, the export-chain validator, and the CI workflows that gate PRs. For testing specifics see `notes/testing.md`; for batching see `notes/batching.md`.

## Quick Reference

```bash
ruff check proto_language tests        # lint (mirrors checks.yml lint job)
ruff format --check                    # formatting (mirrors checks.yml lint job)
mypy proto_language/                   # types (mirrors checks.yml mypy job)
pytest --cpu-only                      # CPU unit tests (mirrors unit-tests.yml)
pytest --cpu-only --skip-ci            # additionally skip skip_ci tests, hide CUDA
pytest --integration --cpu-only -v     # external-tool tests (mirrors integration-tests.yml)
python .github/scripts/validate_exports.py --verbose  # export-chain consistency
```

## Initial Setup

Follow [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup). The tools source lives
in the independent sibling `../diablo-tools` repository. `scripts/bootstrap-tools.sh`
clones the tested tools revision when that directory is absent. Existing checkouts
are left in place so local branches and work survive repeated setup.

Use the [Diablo Lang workspace](https://github.com/chimera-research/diablo-lang) for
Flox + moon + uv and the integration checks. `uv` resolves the language package's
`diablo-tools` dependency from the sibling through `[tool.uv.sources]`.

## Git Worktrees

Each fork has its own history. Create a worktree for either fork with ordinary
`git worktree add`, then provide the tools checkout at `../diablo-tools` or install
both chosen checkouts together using `uv pip install -e PATH_TO_TOOLS -e .`.
For tool changes, follow that repository's own instructions and `notes/`.

## Logging

`proto_language/utils/logging_config.py` centralizes logging: `setup_logging()`
installs a console handler plus an optional timestamped file handler under `logs/`.
Use `get_logger(__name__)` in framework code; never `print()`. The console handler
is bar-aware (`_BarAwareStreamHandler`), so logs route through `tqdm.write` while a
proto-tools spinner is active instead of corrupting it; spinner helpers are reused
from proto-tools via `proto_language/utils/spinner.py`.

## Export Chain Validator

`.github/scripts/validate_exports.py` runs AST-based checks on `__init__.py` files in both repos to catch missing or stale exports before they surface as runtime `ImportError`s. The script is stdlib-only and never executes the target modules.

```bash
python .github/scripts/validate_exports.py                # all domains
python .github/scripts/validate_exports.py --domain Tools # single domain
python .github/scripts/validate_exports.py --verbose      # show every check
```

Exit code is `0` on pass, `1` on errors. The script's module docstring is the canonical list of which checks run per domain. Intentional omissions live in the `exceptions` section of `.github/scripts/export_config.json`; add new exceptions there rather than mutating `__all__` to satisfy the check.

## CI Workflows

| Workflow | Trigger | What it runs |
|---|---|---|
| `unit-tests.yml` | non-draft PR + manual `workflow_dispatch` | `pytest --cpu-only -q --override-ini="log_cli=false" --cov --cov-report=term-missing` |
| `checks.yml` | non-draft PR | Three parallel jobs: `ruff check` + `ruff format --check`; `mypy proto_language/`; `python .github/scripts/validate_exports.py --verbose` |
| `integration-tests.yml` | scheduled (daily 06:00 UTC) + `workflow_dispatch` | Install MAFFT; `pytest --integration --cpu-only -v`. **Not** PR-triggered |
| `claude.yml` | `@claude` in issue/PR/review comment | Code review or scoped question response |

See `notes/testing.md` for the test-side details (markers, fixtures, mocks) that these workflows exercise.

## Documentation

Reference docs are generated externally from this repo's registries, docstrings, and field descriptions plus `proto-tools` tool READMEs. Update source inputs — not generated pages — and merge; downstream regeneration picks up the change. There is no `docs_autogen.yml` workflow in this repo.

### Docstring standard

Google convention (ruff `D`, `convention = "google"`). `tests/test_docstring_consistency.py` checks that class/function `Args`/`Attributes`/`Returns` types match signatures, and holds `proto_language/core/` to a stricter standard: every `core/*.py` component module has a detailed header with an `Examples:` section, and every public behavioral core class has an `Examples:` section. Pydantic models (`BaseModel`) and enums are exempt — they document shape via `Attributes:`/values — as is the package `__init__` aggregator.

Module-header template:

```python
"""<one-line summary ending with a period>.

<2-5 sentences: what the module provides and its role in the data model /
optimization loop.>

Examples:
    >>> from proto_language.core import Thing
    >>> thing = Thing(...)
    >>> thing.attr  # expected value
"""
```

Examples are illustrative, not executed (no `--doctest-modules`): use `>>> expr  # result` inline comments, never a separate expected-output line. ruff's `docstring-code-format` reformats the `>>>` blocks, so keep them valid, canonically-formatted Python. Module headers serve source readers; the generated reference docs are driven by class/function and `ConfigField` docstrings via `proto_language/utils/docs_api.py`, so user-facing classes need their own `Examples:`. The test gates `core/` only — apply the same pattern to new modules and components elsewhere.
