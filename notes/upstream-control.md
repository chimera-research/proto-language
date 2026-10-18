# Upstream control: owning the code and weights every tool depends on

Every proto-tools toolkit pulls things from outside our repositories: PyPI
packages, git repositories, Hugging Face model repos, checkpoint tarballs on
Google Cloud Storage or a lab's file server, prebuilt binaries. Today those are
fetched at setup or first run from wherever upstream keeps them, often by a
moving reference. This note describes how we find all of them, pin them, mirror
them into infrastructure we control, and (next) run, fine-tune, and modify them
entirely from our own git repos and data layer.

It is a tool-level concern because tools are the only layer that touches
upstream. Generators, constraints, and programs reach upstream only through
tools, and the [registry](registry.md) carries every upstream pin up through
them by digest.

Code: `proto_language/provenance/upstream/`. Data: `registry/upstream/`.

---

## 1. Where we are (first full scan)

`registry/upstream/AUDIT.md` is the generated, per-toolkit answer. The first
scan of all 82 toolkits (proto-tools `43868eb`) found:

| | pinned | floating | unresolved here* | notes |
|---|---:|---:|---:|---|
| PyPI requirements | 60 | 125 | 0 | `==` pins resolve to file SHA-256s from PyPI |
| git repositories | 6 | 61 | 0 | most `git clone`s take the default branch |
| Hugging Face models | 0 | 0 | 50 | only 2 of 50 references carry an explicit revision SHA (Chai-1, LLaMA-2-7B-32K) |
| URLs (weights, binaries) | 0 | 12 | 78 | none has a recorded content hash yet |

\* The scan ran in a sandbox whose proxy blocks the Hub and many file hosts. The
weekly CI job (§6) resolves these with network access and an `HF_TOKEN`.

**133 weights sources, none pinned.** Two examples from the audit:

- **Boltz-2.** proto-tools pins `boltz==2.2.1`, and that is all it says. The
  recursive scan opened the `boltz` wheel and found where the weights really come
  from: `boltz/main.py` downloads `boltz2_conf.ckpt` and `boltz2_aff.ckpt` from
  `model-gateway.boltz.bio` and references `boltz-community/boltz-2` on the Hub.
  None of that is visible in proto-tools. The links-only `github.com/jwohlwend/boltz`
  is matched to the `v2.2.1` tag (`cb04aec`) so the mirror can carry the exact
  source of the installed release.
- **AlphaFold2.** `setup.sh` curls `alphafold_params_2022-12-06.tar` (a mutable
  URL) and installs a ColabDesign fork from `germinal@1e1c1a5`, an abbreviated SHA;
  the resolver expands it to the full commit.

## 2. Method

```
 proto-tools toolkits ──▶ discover ──▶ declarations.yaml ──▶ resolve (+ recurse) ──▶ upstream.lock.json
   (static, offline)        │          (human corrections)     (network)                  │   AUDIT.md
                            └──────────────── carry over mirror hashes ◀── mirror ◀───────┘
```

### 2.1 Discover (static, no network, no imports)

`discover.py` walks `proto_tools/tools/{category}/{toolkit}/` and any shared env
it points to (`standalone/shared_env.txt`), reading:

| Channel | What it yields |
|---|---|
| `links.yaml` | `github` → git (reference); `huggingface` → HF repos; `weights`, `model`, `model_gcs`, `zenodo`, `figshare` → weight URLs |
| `standalone/requirements.txt` | PyPI requirements (role `code` when the name matches the toolkit or its repo, else `dependency`); `pkg @ git+...@ref` → git |
| `setup.sh` | `git clone` (+ a following `git checkout <ref>` pins it); `uv pip install` tokens; `curl`/`wget` URLs; inline `hf_hub_download`/`snapshot_download`. Shell variables assigned in the script are expanded; comments and `echo` lines are skipped. |
| toolkit Python | `hf_hub_download` / `snapshot_download` / `from_pretrained("org/name")` / `repo_id=`; `revision="<sha>"`; `huggingface.co/...` URLs (with `/resolve/<sha>` pins); weight-like URLs (by extension or host), with module constants expanded inside f-string URLs |
| pass two | bare `"owner/name"` string literals whose owner was seen in an explicit HF reference anywhere (catches model ids that only travel through variables, e.g. ESM-2 checkpoints in a config `Literal`) |

### 2.2 Declarations (`registry/upstream/declarations.yaml`)

The human-owned correction layer, applied after discovery: `add` sources
discovery cannot see (or exact pins: `hf:org/model@<sha>` replaces the floating
`hf:org/model`), `ignore` false positives, record `services` (live APIs such as
NCBI E-utilities that cannot be mirrored; runs using them are reproducible only up
to the service's own versioning), and `notes`. The file's header documents the
format.

### 2.3 Resolve, recursively (`resolve.py`)

| Type | Resolution | Pinned when |
|---|---|---|
| `pypi` | PyPI JSON API → version and every file's SHA-256 | requirement is `==` |
| `git` | `git ls-remote` → commit for the ref; a ref-less repo is matched to the tag of the toolkit's primary PyPI version (`v2.2.1`) | the toolkit references a full 40-hex commit |
| `hf-model` / `hf-dataset` | Hub revision API → commit SHA, per-file LFS SHA-256 and sizes, gated flag, license (honors `HF_ENDPOINT`, `HF_TOKEN`) | the toolkit passes `revision=<sha>` |
| `url` | `HEAD` → ETag, size, final URL | the mirror recorded its SHA-256 (§3) |

**Recursion (the important part).** For each *primary* code source a toolkit
actually installs or clones (not mere `links.yaml` references), the resolver
downloads the pinned wheel/sdist from PyPI (verifying its SHA-256) or shallow-clones
the repo at the resolved commit, and scans it with the same rules as discovery.
Archives are read member by member and never extracted or executed; tests,
examples, and docs directories are skipped. Anything found is attributed to the
toolkit with evidence like `pypi:boltz==2.2.1!boltz/main.py:45`. Depth is one
level: enough to find weights fetched by the model package itself.

Statuses: **pinned** (an immutable id is recorded), **floating** (reachable, but
the toolkit fetches a moving ref), **unresolved** (not reachable from this
network, or a template URL with runtime variables), **missing** (the host says it
does not exist). A source this network could not reach keeps its last known
resolution from the committed lock.

### 2.4 Outputs

- `registry/upstream/upstream.lock.json`: every toolkit → source ids; every
  source → evidence, status, resolved identifiers, mirror locations. Committed.
- `registry/upstream/AUDIT.md`: the human view: unpinned weights first, then each
  toolkit's code/weights/data with evidence. Committed, reviewed in PRs.
- Every source becomes an `upstream` artifact in each registry release, and every
  tool depends on its toolkit's sources, so a run's lineage reaches weight bytes.

## 3. Mirroring into our infrastructure (`mirror.py`)

| What | Where | How |
|---|---|---|
| git repositories | one mirror repo per upstream, e.g. `chimera-research/upstream--jwohlwend--boltz` (`mirror.git_remote_template` in `registry/config.yaml`) | `git clone --mirror` + `git push --mirror`: all branches and tags, full history |
| weights, binaries, data files | content-addressed store (CAS): `s3://chimera-upstream-cas/sha256/aa/bb/<sha256>` (`mirror.blob_store`; `file://` and `gs://` also work) | download, verify against the upstream-published SHA-256 when there is one (HF LFS, PyPI), upload by hash |
| primary PyPI distributions | same CAS | same, verified against PyPI's SHA-256 |

After a mirror run the lock records `sha256` and `mirror` URIs, URL sources become
**pinned by content**, and `publish` indexes them in `registry.blobs` /
`registry.blob_locations`. Bytes never go into Postgres; Postgres holds the index
and the lineage.

`plan` is the default (prints actions); `--execute` performs them. Mirror repos
must exist before pushing (create them under the org first).

## 4. Roadmap: from "we know what we pull" to "we own it"

**Phase 0 (this change): visibility.** Discovery, recursive resolution, audit,
lock, registry edges, mirror tooling. Nothing at run time changes.

**Phase 1: pin and mirror.** Work `AUDIT.md` top-down, in proto-tools:

1. Weights: pass `revision=<sha>` to every Hub call; replace mutable URLs with
   mirrored ones; record SHA-256s via the mirror job.
2. Code: full commit SHAs for every `git clone`/`git+` requirement; `==` for every
   primary package.
3. Run `upstream mirror --execute` for all toolkits, then set
   `require_pinned_upstream: true` in `registry/config.yaml` so a release cannot
   contain an unpinned code or weights source.
4. Move each toolkit's declarations next to its `links.yaml` in proto-tools
   (e.g. `upstream.yaml`) so the pin lives with the code that uses it; discovery
   already reads the toolkit directory.

**Phase 2: run from our infrastructure only.** proto-tools already has the hooks:

- Weights: populate `PROTO_MODEL_CACHE` (or `PROTO_{TOOL}_WEIGHTS_DIR`) from the
  CAS by digest, and set `HF_HUB_OFFLINE=1`; or serve an internal Hub mirror and
  set `HF_ENDPOINT`, which the resolver already honors.
- Python environments: compile each standalone env to a hash-locked
  requirements file (`uv pip compile --generate-hashes`), install with
  `--require-hashes` from an internal index, and add the env lock to the tool's
  `env` digest.
- Git: rewrite `git clone` / `git+` URLs to the mirror repos.
- CI check: a tool's setup must succeed with outbound network restricted to our
  hosts.

**Phase 3: modify, fine-tune, and train in-house.** Mirror repos are where our
changes live: a `chimera/main` branch per fork for patches, kernels, and training
code (today's `sed` patches in `setup.sh`, such as AlphaFold2's Germinal fork
fixes, become commits). Weights we produce (fine-tunes, distillations,
quantizations) are published to the CAS and registered as `upstream` sources with
`origin: internal` and lineage fields: base weights digest, training code commit,
dataset digests, and the `lineage.runs` id of the training run. A tool then points
at our fork commit and our weights digest, and the registry chain stays intact:
program → constraint → tool → our weights → our training run → base weights and
data. Training runs are recorded in `lineage.runs` exactly like design runs.

## 5. Commands

```bash
python -m proto_language.provenance upstream discover --toolkit boltz2      # what discovery sees (offline)
python -m proto_language.provenance upstream resolve                         # rewrite lock + AUDIT.md (network)
python -m proto_language.provenance upstream resolve --skip-types hf-model,hf-dataset   # no Hub access
python -m proto_language.provenance upstream resolve --check                 # CI: lock covers discovery (offline)
python -m proto_language.provenance upstream mirror --toolkit boltz2         # plan
python -m proto_language.provenance upstream mirror --toolkit boltz2 --execute --blob-store file:///data/cas
```

## 6. CI

- **Every PR** (`registry.yml`): `upstream resolve --check` fails when proto-tools
  references a source the committed lock lacks (typically after a submodule bump).
  Fix by running `upstream resolve` and committing the lock and audit.
- **Weekly + manual** (`upstream.yml`): full resolve with network and `HF_TOKEN`;
  opens a PR when anything moved (a branch advanced, a file was re-uploaded, a
  new source appeared). With `mirror: true` it also mirrors to the CAS (AWS OIDC
  role) and mirror repos (`UPSTREAM_MIRROR_TOKEN`) and commits the recorded hashes.

## 7. Limits to know

- Discovery is static and heuristic. A URL assembled at run time from config
  shows up as a `template` source (or not at all) and needs a declaration. The
  audit's evidence column shows exactly what was matched.
- The recursion goes one level deep. Weights fetched by a dependency of a
  dependency need a declaration.
- The lock records what the toolkit *asks for*. Until Phase 2 hash-locks the
  environments, transitive PyPI dependencies are captured only through the
  `requirements.txt` specifiers and the env digest.
- Live services are recorded, not pinned.
