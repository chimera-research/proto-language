# DrugFlow — proto-tools side

The DrugFlow ligand generator spans two repositories. The proto-language half (the
`DrugFlowGenerator` and the example generative-screen program) lives in this repo. The proto-tools
half — the `drugflow-generate` tool, its standalone environment, and its Modal deployment — belongs
in the [`proto-tools`](https://github.com/evo-design/proto-tools) repository (vendored here as the
`proto-tools/` submodule), so it must land there as its own pull request before the submodule is
bumped.

`0001-drugflow-proto-tools.patch` is that change, exported with `git format-patch`.

## What it adds to proto-tools

- The `drugflow` toolkit under the `molecular_generation` category:
  - `proto_tools/tools/molecular_generation/drugflow/drugflow_generate.py` — the `drugflow-generate`
    tool (Input/Config/Output/Metrics + `run_drugflow_generate`). Pocket-conditioned: given a protein
    structure and a reference ligand marking the pocket, it generates 3D posed molecules.
  - `standalone/` — clones [LPDI-EPFL/DrugFlow](https://github.com/LPDI-EPFL/DrugFlow) and shells out
    to `src/generate.py`. Unlike SynFlowNet, DrugFlow's checkpoints are **public** (Zenodo record
    14919171), so `setup.sh` auto-downloads them — no training or manual provisioning required.
  - `README.md`, `cite.bib`, `license.yaml` (MIT), `links.yaml`, and `examples/example.ipynb`.
- The Modal deployment: `proto_tools/modal/molecular_generation/drugflow_deployment/`, the
  `manifest.py` wiring (`APP_BUCKETS`, `GPU_SERVICES`, `SERVICE_TIERS`, `SERVICE_TO_MODULE`), and the
  regenerated `tool_map.py`.
- Tests: `tests/molecular_generation_tests/test_drugflow_generate.py` (incl. a device-agnostic
  benchmark), and the README tool catalog entry.

## Relationship to the SynFlowNet PR

Both this and the SynFlowNet integration add the `molecular_generation` category. If both land, the
category's `__init__.py`, the master `tools/__init__.py`, `manifest.py`, `tool_map.py`, and the
README catalog are the union of the two (each adds its own toolkit). Resolve those at merge time;
whichever merges first, the other rebases to add only its toolkit.

## Applying it

From a checkout of the proto-tools repo:

```bash
git checkout -b drugflow main
git am /path/to/0001-drugflow-proto-tools.patch   # or: git apply <patch>
```

Then push and open a PR against proto-tools. Once merged, bump this repo's `proto-tools` submodule.

## Caveat (needs GPU-host validation)

DrugFlow's upstream environment is conda-heavy (torch 2.2.1/cu121, PyG 2.5.1, openbabel) and
`generate.py` is CLI-only, so the standalone clones the repo and runs it as a subprocess. The
`setup.sh`/`inference.py` are grounded in the upstream `environment.yaml` and `src/generate.py` CLI
but were not built/run here (no GPU); validate the env build on a GPU host (the proto-tools
`fix-env` skill covers this) and pin `DRUGFLOW_COMMIT` in `setup.sh` to a reviewed upstream commit
before shipping. The tool contract, generator, example program, tests, `ruff`, `mypy`, and the full
`tests/style_consistency_tests/` suite (incl. benchmark-coverage and license/field guards) all pass.
