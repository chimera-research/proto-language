# SynFlowNet — proto-tools side

The SynFlowNet ligand generator spans two repositories. The proto-language half
(the `SynFlowNetGenerator`, the `boltz2-affinity` constraint, and the example
generative-screen programs) lives in this repo. The proto-tools half — the
`synflownet-sample` tool, its standalone environment, and its Modal deployment —
belongs in the [`proto-tools`](https://github.com/evo-design/proto-tools)
repository (vendored here as the `proto-tools/` submodule), so it must land there
as its own pull request before the submodule is bumped.

`0001-synflownet-proto-tools.patch` is that change, exported with `git format-patch`.

## What it adds to proto-tools

- A new `molecular_generation` tool category with the `synflownet` toolkit:
  - `proto_tools/tools/molecular_generation/synflownet/synflownet_sample.py` —
    the `synflownet-sample` tool (Input/Config/Output/Metrics + `run_synflownet_sample`).
  - `standalone/` — the isolated inference environment. SynFlowNet ships no public
    weights, so the checkpoint is user-provisioned via the `ASSET_NOT_AVAILABLE`
    pattern (`setup.sh` skips cleanly when no `model_state.pt` is present).
  - `README.md`, `cite.bib`, `license.yaml`, `links.yaml`, and `examples/example.ipynb`.
- The Modal deployment: `proto_tools/modal/molecular_generation/synflownet_deployment/`,
  the `manifest.py` wiring (`APP_BUCKETS`, `GPU_SERVICES`, `SERVICE_TIERS`,
  `SERVICE_TO_MODULE`), and the regenerated `tool_map.py`.
- Tests: `tests/molecular_generation_tests/test_synflownet_sample.py` (incl. a
  device-agnostic benchmark), and the README tool catalog entry.

## Applying it

From a checkout of the proto-tools repo:

```bash
git checkout -b synflownet main
git am /path/to/0001-synflownet-proto-tools.patch   # or: git apply <patch>
```

Then push and open a PR against proto-tools. Once merged, bump this repo's
`proto-tools` submodule to include it, exactly as the existing
"Bump proto-tools to latest main" commits do.

## Validated

Against the pinned submodule commit, the patch passes `ruff`, `mypy` on the tool
file, the full `tests/style_consistency_tests/` suite (including the benchmark
coverage and license/field-metadata guards), and the `molecular_generation` tests.
The GPU/checkpoint-dependent integration and benchmark tests skip cleanly without
a provisioned checkpoint.
