# Registry: immutable releases of tools, components, and programs

How every tool, generator, constraint, optimizer, and published program becomes
an immutable, content-addressed record in Postgres (TigerData / Timescale), how
CI publishes it, and how a historical run traces back to the exact code, weights,
and seeds it used.

Code: `proto_language/provenance/`. CLI: `python -m proto_language.provenance`.
Upstream pinning (the bottom layer of the graph) is in
[`upstream-control.md`](upstream-control.md). The integration path for new models
is in [`integration-guide.md`](integration-guide.md).

---

## 1. The idea in one picture

The Python registries (`ToolRegistry`, `ConstraintRegistry`, ...) are ephemeral:
they exist only inside a process, and they describe whatever code is checked
out. The registry described here freezes them. Each release is a closed set of
**artifacts**, and each artifact's id is a hash of its content *and of the ids of
everything it depends on* (a Merkle DAG, like git commits):

```
  upstream ──▶ tool ──▶ generator / constraint ──▶ program ◀── optimizer
 (weights,      ▲              ▲                                  ▲
  pypi, git)    │              │                                  │
          framework:      framework: proto-language ──────────────┘
          proto-tools
```

Consequences:

- **Change anything, get a new id, upward.** Re-uploaded Boltz-2 weights change
  the `hf:boltz-community/boltz-2` record, which changes `boltz2-prediction`,
  which changes every constraint that calls it, which changes every program that
  uses those constraints. Nothing else moves.
- **Same content, same id, forever.** Rebuilding an unchanged tree produces
  byte-identical records; publishing them again is a no-op.
- **History is never rewritten.** Rows are insert-only. "Latest" is a pointer
  (a *channel*), and pointers move by appending events.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **artifact** | One immutable record: `(kind, key, digest, local_digest, record, deps)`. |
| **kind** | `framework`, `upstream`, `tool`, `generator`, `constraint`, `optimizer`, `program`. |
| **key** | Registry key within a kind: `boltz2-prediction`, `structure-plddt`, `mcmc`, `toy`, `pypi:boltz==2.2.1`. Keys repeat across versions; digests never do. |
| **digest** | `sha256:` over the canonical JSON of `{kind, key, local_digest, sorted deps}`. The identity a run pins to. |
| **local_digest** | `sha256:` over `{kind, key, record}` only. Unchanged when only a dependency moved, so you can tell "this tool changed" from "this tool was re-versioned because its weights changed". |
| **record** | The artifact's own content, as canonical JSON (see §4). |
| **lock** | `registry.lock.json`: every artifact for one release, plus source commits, warnings, and the release digest. Deterministic, built offline. |
| **release** | A published lock. Id `YYYY.MM.DD-<12 hex of release digest>`, e.g. `2026.10.18-ed6002edf4c8`. |
| **channel** | A named moving pointer to a release: `main` (every merge), `stable` (GitHub releases). |

### Postgres terms (they collide with ours)

| Postgres word | What it is here |
|---|---|
| **database** | The Tiger Cloud service's database (`tsdb` by default). Use one per environment (prod, staging). |
| **SQL schema** | A namespace inside the database. We create two: **`registry`** (what exists) and **`lineage`** (what ran). |
| **table** | Rows of one shape in a SQL schema: `registry.artifacts`, `lineage.runs`, ... (§5). |
| **JSON schema** | A Pydantic `model_json_schema()`. Stored *as data* inside `registry.artifacts.record`; never turned into SQL columns. A tool's Input/Config/Output schemas are part of its record. |
| **migration** | A numbered SQL file in `proto_language/provenance/sql/` that creates or changes SQL schemas and tables. Applied once, tracked in `registry.schema_migrations`, never edited after it ships. |

## 3. Where everything is picked up from

`python -m proto_language.provenance build` (`collect.py`) imports both packages
and reads:

| Artifact kind | Source | Dependency edges |
|---|---|---|
| `framework` | `proto_language/{core,utils}/*.py`, `optimizer/constraint_compiler/`, the three `*_registry.py`; `proto_tools/{utils,entities,transforms}/`, `tool_registry.py` | none |
| `upstream` | `registry/upstream/upstream.lock.json` (see [`upstream-control.md`](upstream-control.md)) | none |
| `tool` | `ToolRegistry.list_all()` + the toolkit directory | `framework:proto-tools`; every upstream source of the tool's toolkit (role `upstream`) |
| `generator` / `constraint` / `optimizer` | `GeneratorRegistry` / `ConstraintRegistry` / `OptimizerRegistry` | `framework:proto-language`; each key in `tools_called` (role `tools_called`); each tool whose `run_*` / Input / Config the component's code imports but does not declare (role `imports`) |
| `program` | JSON files matched by `programs:` globs in `registry/config.yaml` (default `examples/jsons/*.json`); key = the file's `"name"` | every generator, constraint, and optimizer key in its `optimization_stages` |

The `imports` edges are a safety net: `tools_called` is hand-written, so the
builder also parses each component's first-party import closure for proto-tools
names and adds an edge for any tool it finds. Over-approximating is safe (a tool
upgrade can only re-version too much, never too little). Those cases are listed
in the lock's `edge_audit`; fix them by declaring the key. Tools reached only
through a dispatcher such as `predict_structures` are invisible to the scan, so
`tools_called` still matters.

Programs must validate: each referenced key must exist and each `config` must
pass the component's config model, or the build fails (`strict_programs: true`).
Python-only programs (`examples/scripts/`) are not published; give a program a
declarative JSON form to publish it.

## 4. What a record contains

| Kind | `record` fields |
|---|---|
| `tool` | label, category, toolkit, description, `uses_gpu`/`gpu_only`/`device_count`, `cacheable`, `stochastic`, iterable fields; full Input/Config/Output JSON schemas; `code_digest` over the first-party files that define the tool (same file set as `proto_tools.modal.fingerprint`); `env` = standalone env name + digest over every file in its env directory (`setup.sh`, `requirements.txt`, `inference.py`, ...); parsed `license.yaml` and `links.yaml`; `cite.bib` hash; the 12-hex Modal `fingerprint` for cross-referencing deployments. |
| `generator` / `constraint` / `optimizer` | `spec.model_dump()` (key, label, description, config JSON schema with field docs, `tools_called`, sequence types, mode, ...); `code_digest` and `code_files` = the component's first-party import closure, excluding framework files. |
| `program` | `{"program": <the JSON exactly as written>}`, seeds included. |
| `upstream` | type, role, locator, requested ref, pin status, and everything the resolver recorded (commit SHAs, file SHA-256s, sizes, mirror URIs). |
| `framework` | file count and a digest over the framework files. |

Canonical JSON (`canonical.py`): sorted keys, compact separators, UTF-8, NaN
forbidden. `verify` re-hashes rows read back from Postgres, which proves the
`jsonb` round-trip is lossless for our records.

## 5. The Postgres layout

Created by `proto_language/provenance/sql/0001_registry.sql`, applied
automatically by `publish` or explicitly by `migrate`.

### SQL schema `registry`

| Table / view | Columns (key ones) | Notes |
|---|---|---|
| `artifacts` | `digest` PK, `kind` (enum), `key`, `local_digest`, `record` jsonb, `first_release`, `created_at` | One row per distinct content, ever. GIN index on `record` for JSON queries. |
| `artifact_deps` | `artifact_digest`, `dep_digest`, `role` | Merkle edges. Written once with the artifact. |
| `releases` | `release_id` PK, `release_digest` unique, `proto_language_commit`, `proto_tools_commit`, `source` jsonb, `warnings`, `edge_audit`, `ci_run_url`, `published_by`, `published_at` | |
| `release_artifacts` | `release_id`, `kind`, `key`, `artifact_digest` | Membership: which version of each key a release contains. |
| `channel_events` | `channel`, `release_id`, `moved_by`, `moved_at`, `reason` | Append-only pointer log. |
| `blobs`, `blob_locations` | `sha256`, `size_bytes`; `uri`, `source_id`, `origin_url` | Index of mirrored files (weights, binaries, wheels). Bytes live in the content-addressed store, not Postgres. |
| `schema_migrations` | `name`, `sha256`, `applied_at` | Which migrations ran, and their content hash. |
| view `channels` | current release per channel | Latest `channel_events` row. |
| view `current_artifacts` | channel, kind, key, digest, record | What `main` / `stable` contain now. |
| view `artifact_versions` | kind, key, digest, `version` (1, 2, ...), first_release | Version history per key. |
| fn `dependency_closure(digest)` | everything an artifact depends on, with depth | program → components → tools → upstream weights. |
| fn `dependents_closure(digest)` | everything that depends on an artifact | "what does this weights change touch?" |

### SQL schema `lineage`

| Table | Columns | Notes |
|---|---|---|
| `runs` | `run_id`, `started_at` (PK together), `finished_at`, `status`, `release_id`, `program_digest`, `program` jsonb, `seed`, `environment` jsonb, `results_uri` | Timescale **hypertable** on `started_at` when the `timescaledb` extension is present (Tiger Cloud), plain table otherwise. A row may move from `started` to a terminal status exactly once; nothing else may change. |
| `run_artifacts` | `run_id`, `started_at`, `artifact_digest`, `role` | Exact component and program digests a run used. Hypertable too. |

### Enforcement (not convention)

- Every `registry` table has `BEFORE UPDATE OR DELETE` and `BEFORE TRUNCATE`
  triggers that raise. Even a superuser's accidental `UPDATE` fails.
- Roles (created `NOLOGIN`; grant them to real logins):
  `registry_publisher` (INSERT + SELECT on `registry`, CI only),
  `registry_reader` (SELECT), `lineage_writer` (record runs, read registry).
- `publish` refuses locks built from dirty checkouts, and refuses a lock whose
  `release_digest` does not match its contents.
- If an incoming artifact's digest already exists with a different
  `local_digest`, kind, or key, the publish transaction aborts.

## 6. What happens in CI

`.github/workflows/registry.yml`:

| Trigger | Steps |
|---|---|
| Pull request | Install → `upstream resolve --check` (every source discovery sees is in the committed upstream lock) → `build` (fails on invalid programs or broken `tools_called` keys) → upload `registry.lock.json` as an artifact → job summary: `diff --against main` (added / changed / re-versioned per key) and the edge audit. Nothing is written to the database. |
| Push to `main` | Same build, then `publish --channel main` and `verify` of the stored release, in the `registry` GitHub environment. |
| GitHub release published | Same, with `--channel stable`. The tagged commit was already published from `main`, so this only appends a `stable` channel event pointing at the existing release. |

Publishes queue (`concurrency: registry-publish`) so releases land in commit
order. A retry of a successful publish is a no-op.

`.github/workflows/upstream.yml` refreshes the upstream lock weekly and opens a
PR; see [`upstream-control.md`](upstream-control.md).

## 7. Setting up TigerData (once per environment)

1. Create a Tiger Cloud service (Postgres with the TimescaleDB extension). Note
   the connection string: `postgres://tsdbadmin:<pw>@<host>:<port>/tsdb?sslmode=require`.
2. Apply migrations as the admin:
   `REGISTRY_DATABASE_URL=<admin dsn> python -m proto_language.provenance migrate`.
3. Create logins and grant roles:

   ```sql
   CREATE ROLE ci_publisher LOGIN PASSWORD '...' IN ROLE registry_publisher;
   CREATE ROLE registry_ro LOGIN PASSWORD '...' IN ROLE registry_reader;
   CREATE ROLE runner LOGIN PASSWORD '...' IN ROLE lineage_writer;
   ```

   New migrations that add tables must re-run the `GRANT` statements for them.
4. In GitHub, create a `registry` environment (add required reviewers if you
   want a human gate on publishes) with secret `REGISTRY_DATABASE_URL`
   (`ci_publisher`), and add repository secret `REGISTRY_READ_URL` (`registry_ro`)
   for PR diffs.

## 8. Local use

```bash
pip install -e ".[dev]"                                      # includes psycopg
python -m proto_language.provenance build --out registry.lock.json
python -m proto_language.provenance diff registry.lock.json --old previous.lock.json --markdown

# against a disposable Postgres
export REGISTRY_DATABASE_URL=postgresql://postgres@localhost:5432/postgres
python -m proto_language.provenance publish --lock registry.lock.json --allow-dirty --channel dev
python -m proto_language.provenance verify --release <release_id>
REGISTRY_TEST_DATABASE_URL=$REGISTRY_DATABASE_URL pytest tests/provenance_tests --integration --cpu-only
```

## 9. Queries you will want

```sql
-- What does main contain for Boltz-2, and which weights does it depend on?
SELECT d.depth, d.kind, d.key, a.record->'resolved'->>'commit' AS revision
FROM registry.current_artifacts c,
     registry.dependency_closure(c.digest) d
JOIN registry.artifacts a ON a.digest = d.digest
WHERE c.channel = 'main' AND c.kind = 'tool' AND c.key = 'boltz2-prediction'
ORDER BY d.depth;

-- Everything a weights change re-versions.
SELECT kind, key FROM registry.dependents_closure('sha256:...') ORDER BY kind, key;

-- Version history of one constraint.
SELECT version, digest, first_release, published_at
FROM registry.artifact_versions WHERE kind = 'constraint' AND key = 'structure-plddt';

-- Every run in the last 30 days that used a given tool version, through any component.
SELECT DISTINCT r.run_id, r.started_at, r.program->>'name'
FROM lineage.runs r
JOIN lineage.run_artifacts ra USING (run_id, started_at)
WHERE r.started_at > now() - interval '30 days'
  AND 'sha256:<tool digest>' IN (SELECT digest FROM registry.dependency_closure(ra.artifact_digest));
```

## 10. Recording runs (lineage)

`proto_language.provenance.lineage` records a run against the release matching
the running checkout's commits:

```python
from proto_language.provenance.lineage import finish_run, start_run

run_id = start_run(dsn, program_json, repo_root=Path("."))   # finds the release by commits
...  # run the program
finish_run(dsn, run_id, status="succeeded", results_uri="s3://runs/<run_id>/")
```

The run row stores the program as executed (seeds included), host/GPU/Python,
both commits, and `PROTO_*` / `HF_*` environment. If the checkout is dirty or
unreleased, the run is still recorded with `release_id` null, so nothing is lost,
but it cannot be resolved to artifact digests. Run from released commits for
full lineage.

What "reproducible" means here: the registry pins code, environments, schemas,
configs, seeds, and (once mirrored) weight bytes. It does not remove
nondeterminism inside GPU kernels, and live services (NCBI, UniProt, ...) are
recorded as such in the upstream lock rather than pinned.

## 11. Changing the system

- **New table or column:** add `sql/0002_<name>.sql`. Never edit a shipped
  migration; `publish` refuses if an applied file's hash changed.
- **New field in a record:** fine, but every artifact of that kind gets a new
  digest in the next release (expected; the content changed).
- **New artifact kind:** add it to `ArtifactKind`/`ARTIFACT_KINDS` in `models.py`
  and to the `registry.artifact_kind` enum in a new migration
  (`ALTER TYPE ... ADD VALUE`).
- **Renaming a registry key** creates a new key with version 1; the old key's
  history remains under the old name.
