# Integration Guide: Model → Tool → Component → Program → Release

End-to-end path for integrating **any** new model into the Diablo platform: a
generative model, a predictive/scoring model, a classical bioinformatics binary,
or a remote database/API. You wrap it once as a **proto-tools tool**, expose it
in **proto-language** as a generator / constraint / optimizer, use it in a
**program**, and let CI publish it to the **registry**.

> This guide teaches the path and inlines the load-bearing contracts. Canonical,
> always-current detail lives in the linked skills and notes. **When this guide
> and a linked source disagree, the source code wins** — and this guide must be
> updated in the same PR that changes the behavior.
>
> The two forward references below are being built in parallel and do not exist
> yet in full: the `python -m proto_language.provenance` CLI verbs, and the notes
> [`notes/registry.md`](registry.md) and [`notes/upstream-control.md`](upstream-control.md).
> Everything else here is grounded in current source.

---

## 1. What this guide is / TL;DR

```
          upstream model (code + weights, a GitHub repo / HF model / binary / API)
                               │  pin exact versions, commits, weight revisions
                               ▼
  proto-tools TOOLKIT  proto_tools/tools/{category}/{toolkit}/
   ├── {tool_key_snake}.py   @tool(key="{toolkit}-{suffix}")  Input → Config → run_* → Output
   ├── standalone/           isolated micromamba env (setup.sh, requirements.txt, python_version.txt,
   │                          inference.py|run.py)  — NEVER imports proto_tools
   └── cite.bib · license.yaml · links.yaml · README.md · examples/example.ipynb
                               │  one registered operation = one "tool"; its key is the dependency unit
                               ▼
  proto-language COMPONENT  proto_language/{generator|constraint|optimizer}/
   @generator / @constraint / @optimizer   +  BaseConfig(ConfigField)  +  tools_called=[...]
                               │  tools_called is the edge the registry versions on
                               ▼
  PROGRAM   Python (Segment → Construct → Generator → Constraint → Optimizer → Program)
            or declarative JSON (examples/jsons/*.json, the optimization_stages schema)
                               │
                               ▼
  REGISTRY RELEASE   CI builds registry.lock.json (Merkle digests) and publishes
                     immutable artifacts: upstream → tool → component → program
```

### Fast path (10-line checklist)

1. **Decide the shape** (§2): which tool category, and generator vs constraint vs both.
2. **Scaffold the toolkit** under `proto_tools/tools/{category}/{toolkit}/` (§3). Copy the nearest sibling.
3. **Write the core tool file**: `Input → Config → run_*() → Output` + `@tool(...)` (§3).
4. **Write `standalone/`** (setup.sh, requirements.txt, python_version.txt, inference.py|run.py) — no `proto_tools` imports (§3).
5. **Add metadata**: `cite.bib`, `license.yaml`, `links.yaml`, `README.md`, `examples/example.ipynb`; pin all upstream sources (§3).
6. **Wire the export chain** (4 `__init__.py` levels) and run proto-tools tests + `style_consistency_tests` (§3, §6).
7. **Expose in proto-language**: config + decorated component, set `tools_called=[...]` to every tool key it calls (§4).
8. **Wire that export chain** (constraint/generator/optimizer `__init__.py`) and write component tests (§4).
9. **Use it in a program** (Python or JSON); run it end to end (§5).
10. **Validate locally** (ruff, mypy, pytest, `validate_exports.py`, provenance build), bump the submodule, open PRs (§6–§7).

---

## 2. Decide what you are integrating

Pick the tool category and the component kind(s) from what the model consumes and produces. Directories cited are real; browse the nearest sibling before writing.

| Upstream model type | proto-tools category (dir under `proto_tools/tools/`) | proto-language component | Copy from |
|---|---|---|---|
| Generative / masked LM (fills or samples sequence) | `masked_models/`, `causal_models/` | **generator** (`mutation` or `autoregressive`) | tool `masked_models/esm2`; gen `proto_language/generator/esm2_generator.py` |
| Inverse folding (structure → sequence) | `inverse_folding/` | **generator** (`inverse_folding`, `input_type=STRUCTURE`) | tool `inverse_folding/ligandmpnn`; gen `proteinmpnn_generator.py` |
| Structure predictor (sequence → structure + confidences) | `structure_prediction/` | **constraint** (scores pLDDT/pTM/pAE); optionally **gradient** constraint | tool `structure_prediction/boltz2`; constraint `protein_structure/structure_confidence_constraint.py` |
| Scorer / oracle / predictive model (sequence → scalar(s)) | `sequence_scoring/`, `structure_scoring/`, `rna_splicing/`, `gene_annotation/` | **constraint** (discrete scoring; `gradient`/`dual` if differentiable) | constraint skill; `protein_quality/protein_complexity_constraint.py` |
| Classical binary (BLAST, MMseqs2, DSSP, TMalign…) | `sequence_alignment/`, `structure_alignment/`, `gene_annotation/` | **constraint** (usually filter/scoring) | tool `sequence_alignment/mmseqs2`; compile-from-source: `structure_alignment/tmalign` |
| Remote DB / API (lookup, no local weights) | `database_retrieval/`, `gene_annotation/` | **constraint** or lookup used by one | `proto-tools/notes/finding-tools.md`; set `local_only=` / `remote_unsupported_reason` |
| New search algorithm (not a model) | — (no tool) | **optimizer** | `proto_language/optimizer/mcmc_optimizer.py` |

Deciding **generator vs constraint**: a **generator proposes** sequence (it writes `proposal_sequences`), a **constraint scores** sequence (it returns `ConstraintOutput`). A model that both samples and scores (e.g. ESM2 does both) gets wrapped as *separate tools* (`esm2-sample`, `esm2-score`) and *separate components*. **Gradient/dual constraints**: register a differentiable scorer as a `dual` constraint (forward scoring + `backward=`), so both discrete optimizers (MCMC/Beam/RejSamp) and `GradientOptimizer` can use it (§4).

An **optimizer** wraps no tool; it is pure search logic over the dual-pool architecture.

---

## 3. Step 1 — Wrap it as a proto-tools tool

Canonical workflow: [`proto-tools/.claude/skills/implement-tool/SKILL.md`](../proto-tools/.claude/skills/implement-tool/SKILL.md) (+ `TEMPLATES.md`, `PATTERNS.md`). Conventions: [`proto-tools/CLAUDE.md`](../proto-tools/CLAUDE.md), [`proto-tools/CONTRIBUTING.md`](../proto-tools/CONTRIBUTING.md).

### Vocabulary (strict — these are not interchangeable)

- **toolkit** = `{toolkit}` snake_case directory, one family sharing code/model/env/worker (e.g. `boltz2`, `esm2`).
- **tool** = one *registered operation*; its registry key is `{toolkit}-{suffix}` kebab-case (e.g. `boltz2-prediction`, `esm2-sample`). One `{tool_key_snake}.py` per operation.
- Run function is `run_{tool_key_snake}`; classes are `{ToolName}Input/Config/Output`.

### Toolkit directory layout

```text
proto_tools/tools/{category}/{toolkit}/
  __init__.py
  {tool_key_snake}.py        # core tool file: Input/Config/Output + @tool + run_*
  README.md                  # 4 canonical H2s: Overview, Background, Tools, Toolkit Notes
  cite.bib                   # required (BibTeX, verified DOI)
  license.yaml               # required — enforced by test_license_consistency.py
  links.yaml                 # required — canonical upstream links
  helpers.py                 # optional, self-contained plain-type converters (no proto_tools imports)
  examples/example.ipynb     # required, kernelspec {"name":"python3","display_name":"proto-tools"}
  standalone/                # optional, for isolated env (most GPU/ML & binary tools)
    setup.sh
    requirements.txt
    python_version.txt       # required when standalone/ exists
    inference.py             # AI models  (or run.py for CPU tools)
    uv_version.txt           # optional uv pin override
    env_vars.txt             # optional passthrough/set/no_passthrough
    binary_config.py         # optional, for downloaded binaries
    shared_env.txt           # optional, contents name a proto_tools/shared_envs/<name>/ env
```

A pure-Python in-process CPU tool (e.g. `mutagenesis/random_nucleotide`) has **no** `standalone/` dir — it runs in-process. `ToolSpec.local_cpu` is `True` iff `uses_gpu=False` and no `standalone/`.

### The Input/Config/run_*/Output contract

Field helpers: `Field()` / `InputField` on `BaseToolInput`, `ConfigField` on `BaseConfig` (both `extra="forbid"`), bare `pydantic.Field` on `BaseToolOutput` (`extra="ignore"`). All fields need `title=` + `description=` (description **< 100 chars** — enforced by style tests).

```python
"""{tool_display_name} {operation} tool."""

import logging
from typing import Literal

from pydantic import Field, computed_field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)


class MyToolInput(BaseToolInput):
    """Input for {tool_display_name}.

    Attributes:
        sequences (list[str]): Primary per-item input.
    """
    sequences: list[str] = Field(description="Protein sequence(s)")

    @field_validator("sequences", mode="before")
    @classmethod
    def _normalize(cls, v):
        return [v] if isinstance(v, str) else v


class MyToolConfig(BaseConfig):
    """Config for {tool_display_name}. Inherits device/verbose/timeout/seed from BaseConfig."""
    # GPU tools MUST override device to "cuda" (and set include_in_key=False); CPU tools omit it.
    device: str = ConfigField(title="Device", default="cuda", include_in_key=False,
                              description="Device to run on, e.g. 'cuda'")
    model_checkpoint: str = ConfigField(title="Checkpoint", default="default",
        description="Model checkpoint", reload_on_change=True)          # restart worker on change
    batch_size: int = ConfigField(title="Batch Size", default=1, ge=1,
        description="Items per GPU forward pass")


class MyToolOutput(BaseToolOutput):
    """Output. Inherits tool_id/execution_time/success/errors/metadata from BaseToolOutput."""
    results: list[str] = Field(default_factory=list, description="Per-item results")

    @computed_field
    @property
    def num_results(self) -> int:
        return len(self.results)

    @property
    def output_format_options(self) -> list[str]: return ["csv", "json"]
    @property
    def output_format_default(self) -> str: return "csv"
    def _export_output(self, export_path, file_format): ...   # writes the file


def example_input() -> MyToolInput:
    return MyToolInput(sequences=["MKTL"])


@tool(
    key="mytoolkit-sample",             # {toolkit}-{suffix} kebab-case  (required)
    label="My Tool Sampler",            # (required)
    category="masked_models",           # matches the directory (required)
    input_class=MyToolInput,            # (required)
    config_class=MyToolConfig,          # (required)
    output_class=MyToolOutput,          # (required)
    description="One line of what it does",   # (required)
    uses_gpu=True,                      # conventionally set
    device_count="1",                   # "1" | "1-2" | ">=1" | "<=2"
    example_input=example_input,        # factory for parametrized infra tests
    iterable_input_fields=["sequences"],# index-parallel list fields, primary first
    iterable_output_field="results",    # 1:1 with the primary input list, by position
    max_chunk_size=1,                   # REQUIRED to be explicit when config has batch_size
    cacheable=True,                     # output is a deterministic fn of (input, config)
    stochastic=True,                    # outputs depend on config.seed
    # metrics_class=MyToolMetrics,      # when the tool emits scalar metrics (plDDT, perplexity…)
)
def run_mytoolkit_sample(inputs: MyToolInput, config: MyToolConfig) -> MyToolOutput:
    """Run the tool via the standalone worker. No try/except — the decorator owns errors."""
    out = ToolInstance.dispatch(
        "mytoolkit",
        {"operation": "sample", "sequences": inputs.sequences,
         "device": config.device, "seed": config.seed},   # raw Optional[int]
        script_path=Path(__file__).parent / "standalone" / "inference.py",
        config=config,
    )
    return MyToolOutput(results=out["results"], metadata={"model": config.model_checkpoint})
```

The complete `@tool` kwargs (from `ToolRegistry.register`): required `key, label, category, input_class, config_class, output_class, description`; optional `metrics_class, uses_gpu, gpu_only, pin_visible_devices, device_count, example_input, iterable_input_fields, iterable_output_field, max_chunk_size, local_only, cacheable, stochastic, post_process_iterable`.

**Key rules (behavioral):**
- **Iterable cardinality is strictly 1:1**: `len(output.{iterable_output_field})` equals the primary input length, by position. N-samples-per-item must bundle N inside a per-input wrapper, never flatten (see `ProteinMPNNSequences`). PATTERNS.md → "Iterable cardinality".
- **Input = per-item, Config = shared.** A per-batch param or a search DB goes in Config (keeps the per-item cache key correct); a per-item value goes in Input.
- `max_chunk_size` is only valid for iterable tools and is **required** to be explicit when the config carries `batch_size`.
- Outputs must be JSON-serializable Pydantic types — no DataFrames/numpy/arbitrary objects as fields.
- Biological coordinates are **1-indexed, inclusive**.

### Standalone env contract

`standalone/` runs in an isolated micromamba subprocess where **`proto_tools` is NOT importable**. See [`proto-tools/notes/tool-environments.md`](../proto-tools/notes/tool-environments.md).

- **Never import `proto_tools`** in standalone code. Use `standalone_helpers` (published on `PYTHONPATH`) for `get_logger`, `resolve_weights_dir`, `serialize_output`, `set_torch_seed`, `get_random_int`, device/OOM helpers.
- **Every `.py` under `standalone/` (except `__init__.py`) must declare `logger = get_logger(__name__)`** from `standalone_helpers` — plain `logging.getLogger` lands outside the worker bridge and is silently dropped (enforced by `test_standalone_logger_consistency.py`).
- Heavy imports (torch, model libs) go **inside functions**, never module level.
- Implement two module-level protocol functions: `to_device(device: str) -> dict` and `get_memory_stats() -> dict` (for `DeviceManager`).
- `inference.py` entry is `dispatch(input_dict) -> dict`; route by `input_dict["operation"]`; a `if __name__ == "__main__":` block reads/writes JSON files.
- `setup.sh` sources `standalone_helpers.sh` and uses the preinstalled `uv` (pinned by `UV_VERSION` in `tool_instance.py`): call `uv pip install` directly, never `pip install uv`. Helpers: `proto_install_pytorch`, `proto_install_jax`, `proto_install_cuda_toolkit`, `proto_resolve_weights_dir <toolkit>`, `proto_check_gated_hf_repo`.
- `python_version.txt` is **required** when `standalone/` exists (keyed format, required `default:` key).
- **Shared envs**: if two toolkits share deps + Python version, point `standalone/shared_env.txt` at `proto_tools/shared_envs/<name>/`; `inference.py` still lives per-tool. A tool with both `shared_env.txt` and `setup.sh` raises.

### Weights & storage

All persistent data lives under `PROTO_HOME` (default `~/.proto/`). See [`proto-tools/notes/storage.md`](../proto-tools/notes/storage.md).

| Weight source | What to do |
|---|---|
| HuggingFace `from_pretrained` | Nothing — `HF_HOME` is set automatically (`PROTO_MODEL_CACHE/huggingface/`). Example: `masked_models/esm2`. |
| Direct download / foundry | In `setup.sh` call `proto_resolve_weights_dir {toolkit}` → `$WEIGHTS_DIR`, download there; at runtime `resolve_weights_dir("{toolkit}")`. |
| Gated HF repo | `proto_check_gated_hf_repo "org/model" "<license_url>"` before install; user accepts license + sets `HF_TOKEN`. See [`proto-tools/notes/gated-models.md`](../proto-tools/notes/gated-models.md). |
| Not auto-fetchable (too large / manual approval) | Precheck in `setup.sh`: print `[proto-tools] ASSET_NOT_AVAILABLE: <toolkit>:<asset_kind>` and `exit 64` → `MissingAssetError` → pytest skip. Example: `structure_prediction/x3dna`. |

Per-tool override `PROTO_{TOOLKIT}_WEIGHTS_DIR` always wins. `PROTO_MODEL_CACHE` is safe to share across a team; `PROTO_HOME` is not.

### Seeding / stochastic / cacheable

See [`proto-tools/notes/seeding.md`](../proto-tools/notes/seeding.md).
- Set `stochastic=True` **only** when outputs depend on `config.seed`. It drives three effects: unseeded calls skip cache; iterable dedup is skipped (duplicates diverge); cacheable+seeded uses the whole-call cache.
- The framework does **not** unroll multi-item dispatches — the tool advances its own RNG per item (`base_seed + i`, as `run_boltz2` does).
- Pass `config.seed` (raw `Optional[int]`) through the dispatch dict; `set_torch_seed(seed)` is called unconditionally (no-ops on `None`); use `seed if seed is not None else get_random_int()` where a concrete int is needed.
- Hardware/float noise is **not** seed-controllable and does not warrant `stochastic=True`.

### Error handling (never catch in tool fns)

The `@tool` decorator **raises by default**; it owns error policy. Do **not** write try/except around tool logic. `PROTO_CAPTURE_ERRORS=1` packs exceptions into `success=False` outputs. `MissingAssetError` always raises regardless. OOM via `standalone_helpers.oom`. Full contract: [`proto-tools/notes/error-handling.md`](../proto-tools/notes/error-handling.md).

### Metadata files

- `cite.bib` — verified DOI/BibTeX.
- `license.yaml` — **required**; SPDX id from the allowlist (or `Custom (...)` with inline `text:`); keys `code`, `commercial_use`, `redistribution`, `last_updated` (+ optional `weights`/`data` blocks). README `> [!NOTE] License:` callout must agree.
- `links.yaml` — **required**; verified `github`/`website`/`paper`/`preprint`/`huggingface`/`organizations`.
- `README.md` — four canonical H2 sections exactly: `Overview`, `Background`, `Tools`, `Toolkit Notes` (parser keys on these). Per-tool H3 is `{Tool Display Name} (\`{tool-key}\`)`.
- `examples/example.ipynb` — kernelspec exactly `{"name": "python3", "display_name": "proto-tools"}`; use `display_api_reference(...)` for Input/Config/Output tables.

### Export chain (4 levels — all required)

1. `tools/{category}/{toolkit}/__init__.py` — export `{ToolName}Input/Config/Output`, `run_{tool_key_snake}`.
2. `tools/{category}/__init__.py` — add import block + `__all__`.
3. `tools/__init__.py` — add import block + `__all__`.
4. `proto_tools/__init__.py` — uses `from proto_tools.tools import *`; no edit needed if level 3 `__all__` is correct.

Sort every `__all__` alphabetically.

### Upstream provenance (pin everything)

Integrators must pin every upstream source so a release lock captures exactly what was built:
- **pip**: exact `==` versions (e.g. `boltz[cuda]==2.2.1`, `transformers==5.12.1`).
- **git**: exact commit SHAs, never branches.
- **weights**: explicit HF `revision=<commit-sha>`, or a download URL with a stable digest — never a floating `main`.

Then declare those sources so they are captured by the upstream lock. **See [`notes/upstream-control.md`](upstream-control.md) for the declaration format and the `python -m proto_language.provenance upstream discover`/`resolve` workflow.** Each toolkit gets one entry in `registry/upstream/declarations.yaml`.

### Tests + markers + style suite

- Place tool tests at `tests/{category}_tests/test_{tool_key_snake}.py`; **flat functions only**, no test classes, no trivial Pydantic tests.
- Markers: `@pytest.mark.uses_gpu` (GPU dispatch), `@pytest.mark.integration` (CPU dispatch to real env), `@pytest.mark.slow`, `@pytest.mark.skip_ci`. See [`proto-tools/notes/testing.md`](../proto-tools/notes/testing.md).
- **Before any PR, run the full `tests/style_consistency_tests/` suite** (fast, CPU, no env): docstring style, the <100-char field-description limit, license/README/version consistency, I/O round-trip, standalone logger, `python_version.txt`. `ruff` ignores `E501`, so these catch what lint/mypy miss.

---

## 4. Step 2 — Expose it in proto-language

Each component is a registered class/function with a `BaseConfig` config and a decorator. The decorator's `tools_called=[...]` is the **dependency edge**: it must list **every proto-tools key the component invokes**. The registry uses it to version components — when a tool changes, every component that lists it (and every program using them) re-versions. Keep it exact and complete.

Skills: [`implement-constraint`](../.claude/skills/implement-constraint/SKILL.md), [`implement-generator`](../.claude/skills/implement-generator/SKILL.md) (+ `TEMPLATES.md`), [`implement-optimizer`](../.claude/skills/implement-optimizer/SKILL.md) (+ `TEMPLATES.md`). Config classes inherit the local `BaseConfig` and use `ConfigField` (`from proto_language.utils.base import BaseConfig, ConfigField`).

### Constraint (scores sequences)

Signature is fixed: `(input_sequences: list[tuple[Sequence, ...]], config) -> list[ConstraintOutput]`. Score convention: `0.0` = `MIN_ENERGY` (perfect), `1.0` = `MAX_ENERGY` (worst); clamp with `min(MAX_ENERGY, ...)`.

```python
from proto_tools import run_mytool, MyToolInput, MyToolConfig
from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

@constraint(
    key="my-score",                         # kebab-case (required)
    label="My Score",                       # (required)
    config=MyScoreConfig,                   # BaseConfig subclass (required)
    description="Scores sequences by ...",  # (required)
    uses_gpu=True,                          # True if it calls GPU tools
    tools_called=["mytoolkit-sample"],      # EVERY tool key this calls — the dependency edge
    category="protein_structure",           # matches the subdirectory
    supported_sequence_types=["protein"],   # MUST be non-empty
    input_labels=None,                      # None = any number of interchangeable inputs
    # backward=my_backward, backward_config=MyScoreConfig,   # for dual-mode (see below)
)
def my_score(input_sequences, config) -> list[ConstraintOutput]:
    tool_out = run_mytool(MyToolInput(sequences=[s.sequence for (s,) in input_sequences]),
                          MyToolConfig(batch_size=config.batch_size))
    results = []
    for (seq,), value in zip(input_sequences, tool_out.results, strict=True):
        results.append(ConstraintOutput(score=min(MAX_ENERGY, _norm(value)),
                                        metadata={"my_metric": value}))
    return results
```

- **Modes**: `discrete` (forward scoring only), `gradient` (decorated fn returns `list[GradientConstraintOutput]`, auto-detected), `dual` (forward + `backward=`). Prefer **dual** when the computation supports both — one key, one config, and MCMC/Beam/RejSamp and `GradientOptimizer` each route to the right path.
- **Per-proposal data** goes in `ConstraintOutput.metadata`; predicted structures/logits go in `.structures`/`.logits` as tuples aligned to the input tuple length (use `None` for untouched slots).
- **Error handling** ([`notes/error-handling.md`](error-handling.md)): raise on shared/batch-level failures (bad config, tool crash, missing asset); **soft-fail only a proposal-local failure**, appending `ConstraintOutput(score=MAX_ENERGY, metadata={"<name>_error": ...})` and continuing. A real structure-predictor-backed example is `protein_structure/structure_confidence_constraint.py` (`tools_called=["boltz2-prediction", ...]`, returns `MAX_ENERGY` when the metric is missing).
- **Batching**: add `batch_size` to the config and pass it to the tool config; **never** write a chunking loop in the constraint — the tool owns batching ([`notes/batching.md`](batching.md)).
- **Export chain** (2 edits): category `__init__.py` imports the function; `constraint/__init__.py` imports it and adds to `__all__`.

### Generator (proposes sequences)

A `Generator` subclass with the `input_type` classvar, decorated and `@final`. It mutates `self.segment.proposal_sequences` **in place** inside `_sample()`. `category` is **auto-derived from `input_type`** — it is *not* a decorator argument.

```python
from typing import final
from proto_tools import run_mytool, MyToolInput, MyToolConfig
from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField

@generator(
    key="my-gen",                       # (required)
    label="My Generator",               # (required)
    config=MyGenConfig,                 # (required)
    description="Samples via ...",      # (required)
    uses_gpu=True,
    tools_called=["mytoolkit-sample"],  # EVERY tool key this calls
    supported_sequence_types=["protein"],  # [] = all; never list "ligand" (immutable)
)
@final
class MyGenerator(Generator):
    input_type = GeneratorInputType.STARTING_SEQUENCE   # STARTING_SEQUENCE|PROMPT|STRUCTURE|LOGITS

    def __init__(self, config: MyGenConfig) -> None:
        super().__init__()
        self.config = config

    def _sample(self) -> None:
        self._validate_generator()          # first line, always
        seqs = [s.sequence for s in self.segment.proposal_sequences]
        out = run_mytool(MyToolInput(sequences=seqs),
                         MyToolConfig(batch_size=self.config.batch_size, seed=self._next_seed()))
        for proposal, new in zip(self.segment.proposal_sequences, out.results, strict=True):
            proposal.sequence = new
```

- `input_type` sources: `STARTING_SEQUENCE` (mutation / de-novo), `PROMPT` (autoregressive), `STRUCTURE` (inverse folding), `LOGITS` (gradient). De-novo generation sets `allows_empty_starting_sequence=True`, not `PROMPT`.
- **Seeding**: pass `seed=self._next_seed()` so tool output joins program determinism; never pass `seed_per_item`.
- **No batching loop** — pass `batch_size` through; the tool chunks.
- Real example: `proto_language/generator/esm2_generator.py` (`tools_called=["esm2-sample"]`).
- **Export chain**: add class + config to `generator/__init__.py` and `__all__`.

### Optimizer (search strategy, wraps no tool)

An `Optimizer` subclass, decorated and `@final`, over the dual-pool architecture (`result_sequences` = top-K persisted, `proposal_sequences` = per-step). Config inherits `BaseOptimizerConfig`.

```python
from typing import final
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.core import Optimizer

@optimizer(
    key="my-opt",                       # (required)
    label="My Optimizer",               # (required)
    config=MyOptConfig,                 # BaseOptimizerConfig subclass (required)
    description="...",                  # (required)
    uses_gpu=False,
    targets_single_segment=False,       # True → also take target_segment as first __init__ arg
    compatible_generators=None,         # None = all unclaimed generators
    required_constraint_mode=None,      # "gradient" accepts gradient/dual; "discrete" accepts discrete/dual
)
@final
class MyOptimizer(Optimizer):
    def __init__(self, constructs, generators, constraints, config: MyOptConfig) -> None:
        self.config = config
        super().__init__(constructs, generators, constraints, seed=config.seed)
    def run(self) -> None:
        self._prepare_run(); self._initialize_sequence_pools()
        # ... score_energy(), _update_results(), _save_progress_snapshot() ...
```

- Optimizers have **no `tools_called`** (they call constraints/generators, not tools directly). Filter constraints (threshold set) run before scoring constraints.
- Seeds: `Program(seed=…)` derives each optimizer's `config.seed`. Reuse the **same construct objects by identity** across stages.
- **Export chain**: add class + config to `optimizer/__init__.py` and `__all__`.

### Component export validation

`.github/scripts/validate_exports.py` checks that every `@tool`/`@constraint`/`@generator`/`@optimizer` is exported by its parent `__init__.py`, and that `__all__` entries are real. Intentional omissions go in `.github/scripts/export_config.json`.

---

## 5. Step 3 — Use it in a program

Program flow (6 steps): Segments → Constructs → Generators → Constraints → Optimizers → Program. Skill: [`write-program`](../.claude/skills/write-program/SKILL.md). Runnable examples: `examples/scripts/`.

### Python

```python
from proto_language.core import Construct, Program, Segment, Constraint
from proto_language.generator import MyGenerator, MyGeneratorConfig
from proto_language.constraint import my_score            # the registered function
from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig

seg = Segment(length=100, sequence_type="protein")
construct = Construct([seg])

gen = MyGenerator(MyGeneratorConfig()); gen.assign(seg)    # assign every variable segment
scorer = Constraint(inputs=[seg], function=my_score, function_config={"batch_size": 1})

opt = MCMCOptimizer(constructs=[construct], generators=[gen], constraints=[scorer],
                    config=MCMCOptimizerConfig(num_results=1, num_steps=100))
program = Program(optimizers=[opt], num_results=1)
program.run()                                              # program.run(device="modal") for remote
for s in construct.joined_sequences:
    print(s.sequence)
```

Rules: a `Constraint` with `threshold=` is a **filter** (pass/fail gate); without it, a **scorer** (`weight=` multiplies). Create **fresh** generators/constraints per optimizer stage. In a multi-stage program, **share the same construct objects by identity** across stages — that is how state flows.

### Declarative JSON

JSON files under `examples/jsons/` express the same shape (the `optimization_stages` schema) and **illustrate structure; they are not loaded by a Python consumer** today. A minimal stage referencing a new constraint key:

```json
{
  "name": "my-design", "version": "1.0", "num_results": 1,
  "constructs": [{ "id": "c1", "type": "protein",
    "segments": [{ "id": "variable", "label": "100 aa", "length": 100 }] }],
  "optimization_stages": [{
    "generators":  [{ "key": "my-gen",   "targets": ["variable"], "config": {} }],
    "constraints": [{ "key": "my-score", "targets": ["variable"], "config": { "batch_size": 1 } }],
    "optimizer":   { "method": "mcmc",   "config": { "num_steps": 100, "num_results": 1 } }
  }]
}
```

Component keys in JSON are the registry keys (`my-gen`, `my-score`, `mcmc`). See `examples/jsons/toy.json` for a full working example. Program-level `seed` owns run determinism and derives downstream seeds.

---

## 6. Step 4 — Validate locally

Run both repos' checks. proto-tools first (the submodule), then proto-language.

**proto-tools** (in `proto-tools/`):

```bash
ruff check proto_tools tests
ruff format --check
mypy proto_tools                                     # strict; every # type: ignore needs a code
pytest tests/style_consistency_tests/                # ALWAYS before a PR (fast, CPU, no env)
pytest --all --ext -k "{tool_key}" -v                # the new tool's functional + infra tests
python scripts/run_example_notebooks.py --only {toolkit}   # execute the example notebook
```

**proto-language** (repo root):

```bash
ruff check proto_language tests
ruff format --check
mypy proto_language/
pytest --cpu-only -x                                 # fast CPU feedback
pytest --integration --cpu-only -v                   # external-tool tests
pytest --gpu-only -k "mygen" -x                      # GPU-marked, if applicable
python .github/scripts/validate_exports.py --verbose # export-chain consistency (both repos)
python -m proto_language.provenance build --out registry.lock.json   # release-lock dry run (§7)
```

**Submodule bump** (parent repo tracks `proto-tools@main`; `submodule-check.yml` fails on drift):

```bash
cd proto-tools && git add -A && git commit && git push      # land the tool PR on proto-tools main first
cd .. && git submodule update --remote proto-tools
git add proto-tools && git commit -m "Bump proto-tools submodule"
```

Editable install order matters — install the submodule **last** so edits take effect; if `import proto_tools` fails with a namespace-package shadow, see the recovery in [`notes/dev.md`](dev.md) / `CONTRIBUTING.md`.

---

## 7. Step 5 — Register & publish

Once merged to `main`, CI builds the registry lock and publishes **immutable** releases — upstream sources, tools, components, and programs — each identified by a Merkle digest (an artifact's digest covers its own record plus its dependencies', so changing a tool re-versions every component and program built on it). Publication order is `framework → upstream → tool → generator → constraint → optimizer → program`. For the full flow (Postgres registry, release process), see [`notes/registry.md`](registry.md).

Local dry-runs / audits:

```bash
python -m proto_language.provenance build --out registry.lock.json   # build the release lock locally
python -m proto_language.provenance upstream discover                # static scan of toolkits for upstream sources
python -m proto_language.provenance upstream resolve                 # pin them into registry/upstream/upstream.lock.json
```

Curated upstream declarations live in `registry/upstream/declarations.yaml` (one entry per toolkit); see [`notes/upstream-control.md`](upstream-control.md).

---

## 8. Definition of done

**proto-tools (the tool):**
- [ ] `{tool_key_snake}.py` with `Input → Config → run_*() → Output`; `@tool(...)` has all 7 required kwargs (+ `uses_gpu`).
- [ ] Config inherits `BaseConfig`/`ConfigField`; GPU tools override `device="cuda"` (`include_in_key=False`); every field has `title=` + `description=` (< 100 chars).
- [ ] Output implements `output_format_options`, `output_format_default`, `_export_output()`; fields JSON-serializable.
- [ ] No try/except in tool logic; `logging.getLogger(__name__)`, never `print()`.
- [ ] Iterable 1:1 cardinality; `max_chunk_size` set when config has `batch_size`; `stochastic=True` iff output depends on `config.seed` (and RNG advanced per item).
- [ ] `standalone/` (if any): no `proto_tools` imports; `get_logger` in every `.py`; `to_device`/`get_memory_stats`; `python_version.txt` present; weights via `resolve_weights_dir`/HF.
- [ ] `cite.bib`, `license.yaml`, `links.yaml`, `README.md` (4 H2s), `examples/example.ipynb` (correct kernelspec) all present and verified.
- [ ] All upstream pins exact (`==`, commit SHAs, weight `revision=`).
- [ ] **Toolkit has an entry in `registry/upstream/declarations.yaml` with every code/weights source pinned (no floating refs).**
- [ ] Export chain updated at all 4 levels (`__all__` sorted).
- [ ] `tests/style_consistency_tests/` green; tool tests green; example notebook executes.

**proto-language (the component):**
- [ ] Config inherits `BaseConfig`/`ConfigField`; `@constraint`/`@generator`/`@optimizer` with kebab-case key.
- [ ] **`tools_called=[...]` lists every proto-tools key the component invokes.**
- [ ] Constraint returns `list[ConstraintOutput]`, score in `[0,1]`; `supported_sequence_types` non-empty. Generator sets `input_type`, is `@final`, mutates in place. Optimizer is `@final`, shares constructs by identity.
- [ ] Raise vs soft-fail follows `notes/error-handling.md`; batching delegated to the tool; seeds via `_next_seed()` / program-derived.
- [ ] Export chain updated; component tests cover scoring/sampling, type validation, config validation, edge cases.

**Both repos / release:**
- [ ] `ruff check`, `ruff format --check`, `mypy`, `pytest --cpu-only`, `validate_exports.py` all green in each repo.
- [ ] Submodule pointer bumped to the merged proto-tools SHA.
- [ ] **`python -m proto_language.provenance build` succeeds with no edge-audit warnings for your component (`tools_called` matches the `run_*` functions it imports).**

---

## 9. Pitfalls (real gotchas)

1. **Stale `proto_tools` namespace package.** A partial `site-packages/proto_tools/` from a pre-submodule install silently shadows the editable install (`proto_tools.__file__ is None`); `pip uninstall` won't remove it. Delete it manually and reinstall. (`notes/dev.md`, `CONTRIBUTING.md`.)
2. **Editable install order.** Install `./proto-tools[dev]` **last**, or the git-installed copy wins and your submodule edits don't take effect.
3. **Description length limit.** proto-tools field `description=` must be **< 100 chars** — `ruff` ignores `E501`, so only `style_consistency_tests` catch it, and an over-long description in a nested config (e.g. `Mmseqs2HomologySearchConfig`) fails **every** tool that embeds it.
4. **1-indexed, inclusive coordinates** everywhere biological — don't pass 0-indexed offsets through a dispatch dict.
5. **Never `print()`** — use `logging.getLogger(__name__)` (framework) / `get_logger(__name__)` from `standalone_helpers` (standalone). Plain `logging.getLogger` in a standalone is silently dropped.
6. **`# type: ignore` needs an error code** (mypy strict, both repos); prefer `assert` narrowing over `cast()`/`Protocol`/`TYPE_CHECKING`.
7. **Standalone can't import `proto_tools`.** `from proto_tools...` or `from standalone_helpers import get_subprocess_device_env`-style mistakes fail at runtime; use the published `standalone_helpers` surface only.
8. **Iterable cardinality violations.** Flattening N samples per input breaks per-item cache stitching and dedup; bundle N in a per-input wrapper and keep `iterable_output_field` 1:1 with the primary input.
9. **`tools_called` drift.** Missing or stale keys break registry versioning — the provenance edge audit warns when `tools_called` doesn't match the imported `run_*` functions. Keep it exact.
10. **Wrong `stochastic` flag.** Setting it on a deterministic tool makes seeded calls skip per-item cache and dedup for no reason; omitting it on a sampler makes duplicate iterable items collapse. Flag iff `config.seed` changes the output.
11. **`category` is not a generator decorator arg.** It is derived from the generator's `input_type` classvar; passing it is wrong. (Constraints *do* take `category`.)
12. **Reusing components across stages / copying constructs.** Generators and constraints can't be reused across optimizer stages (create fresh ones); multi-stage programs must share the **same construct objects by identity** or state won't flow.
13. **Catching exceptions in a tool function.** The `@tool` decorator owns error policy; a try/except hides failures and defeats capture mode and `MissingAssetError` skips.
14. **Notebook kernelspec.** Must be exactly `{"name": "python3", "display_name": "proto-tools"}`; a custom name raises `NoSuchKernel` off the author's machine.

---

## 10. Where to go deeper

| Topic | Canonical file |
|---|---|
| Tool implementation lifecycle | [`proto-tools/.claude/skills/implement-tool/SKILL.md`](../proto-tools/.claude/skills/implement-tool/SKILL.md) (+ `TEMPLATES.md`, `PATTERNS.md`) |
| `@tool` signature / `ToolSpec` | `proto-tools/proto_tools/tools/tool_registry.py` |
| Standalone envs, compute deps, `to_device`, shared envs | [`proto-tools/notes/tool-environments.md`](../proto-tools/notes/tool-environments.md) |
| Weights, `PROTO_HOME`, `PROTO_MODEL_CACHE`, databases | [`proto-tools/notes/storage.md`](../proto-tools/notes/storage.md) |
| Gated models / HF auth | [`proto-tools/notes/gated-models.md`](../proto-tools/notes/gated-models.md) |
| Tool error policy / capture mode | [`proto-tools/notes/error-handling.md`](../proto-tools/notes/error-handling.md) |
| Seeding / caching / dedup | [`proto-tools/notes/seeding.md`](../proto-tools/notes/seeding.md) |
| Worker logging bridge | [`proto-tools/notes/logging.md`](../proto-tools/notes/logging.md) |
| Discovery / CLI / MCP (`ToolRegistry`) | [`proto-tools/notes/finding-tools.md`](../proto-tools/notes/finding-tools.md) |
| Model/validator selection for design | [`proto-tools/notes/model-taste.md`](../proto-tools/notes/model-taste.md) |
| Remote execution (`device="modal"`) | [`proto-tools/notes/modal-deployment.md`](../proto-tools/notes/modal-deployment.md) |
| Tool tests / markers | [`proto-tools/notes/testing.md`](../proto-tools/notes/testing.md) |
| Constraint lifecycle | [`.claude/skills/implement-constraint/SKILL.md`](../.claude/skills/implement-constraint/SKILL.md) |
| Generator lifecycle | [`.claude/skills/implement-generator/SKILL.md`](../.claude/skills/implement-generator/SKILL.md) |
| Optimizer lifecycle | [`.claude/skills/implement-optimizer/SKILL.md`](../.claude/skills/implement-optimizer/SKILL.md) |
| Composing programs | [`.claude/skills/write-program/SKILL.md`](../.claude/skills/write-program/SKILL.md) (+ `PATTERNS.md`) |
| Component registries | `proto_language/{constraint,generator,optimizer}/*_registry.py` |
| Raise vs soft-fail (language side) | [`notes/error-handling.md`](error-handling.md) |
| Batching ownership across layers | [`notes/batching.md`](batching.md) |
| Where files live / package layout | [`notes/filesystem.md`](filesystem.md) |
| Dev workflow, submodule, export validator, CI | [`notes/dev.md`](dev.md) |
| Component tests / markers / mocks | [`notes/testing.md`](testing.md) |
| Registry release / publication | [`notes/registry.md`](registry.md) *(forthcoming)* |
| Upstream declarations / provenance workflow | [`notes/upstream-control.md`](upstream-control.md) *(forthcoming)* |
