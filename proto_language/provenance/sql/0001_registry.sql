-- Proto / Diablo registry: immutable, content-addressed releases.
--
-- Terminology (Postgres overloads "schema"):
--   database         the Tiger Cloud service's database (e.g. tsdb). One per environment.
--   SQL schema       a namespace inside the database. This file creates two:
--                      registry  - what exists (artifacts, releases, channels, blobs)
--                      lineage   - what ran (runs and the exact artifacts each used)
--   table            rows of one shape inside a SQL schema (registry.artifacts, ...).
--   JSON schema      a Pydantic model's model_json_schema(); stored *as data* in
--                    registry.artifacts.record, never as SQL structure.
--
-- Invariants enforced here, not by convention:
--   * Every registry table is append-only: UPDATE, DELETE, and TRUNCATE raise.
--   * Artifacts are keyed by their Merkle digest, so re-publishing identical
--     content is a no-op and different content can never overwrite.
--   * A release is a closed set: its member rows are inserted in the same
--     transaction as the release row and can never change afterwards.
--   * Channels ("main", "stable") are the only moving pointers, and they move by
--     appending an event; the current value is a view over the latest event.
--
-- Applied by `python -m proto_language.provenance publish` (idempotent; each
-- migration runs once, recorded in registry.schema_migrations).

CREATE SCHEMA IF NOT EXISTS registry;
CREATE SCHEMA IF NOT EXISTS lineage;

-- ---------------------------------------------------------------------------
-- Immutability guard shared by every table below.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION registry.forbid_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'registry table %.% is append-only (% rejected)', TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'insufficient_privilege';
END;
$$;

-- ---------------------------------------------------------------------------
-- Artifacts: one row per distinct content, ever.
-- ---------------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE registry.artifact_kind AS ENUM
        ('framework', 'upstream', 'tool', 'generator', 'constraint', 'optimizer', 'program');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS registry.artifacts (
    digest          text PRIMARY KEY CHECK (digest ~ '^sha256:[0-9a-f]{64}$'),
    kind            registry.artifact_kind NOT NULL,
    key             text NOT NULL,
    local_digest    text NOT NULL CHECK (local_digest ~ '^sha256:[0-9a-f]{64}$'),
    record          jsonb NOT NULL,
    first_release   text NOT NULL,               -- release that introduced this digest
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS artifacts_kind_key_idx ON registry.artifacts (kind, key);
CREATE INDEX IF NOT EXISTS artifacts_local_digest_idx ON registry.artifacts (local_digest);
CREATE INDEX IF NOT EXISTS artifacts_record_gin ON registry.artifacts USING gin (record jsonb_path_ops);

-- Dependency edges of the Merkle DAG. Part of an artifact's identity, so they
-- are written once with the artifact and never change.
CREATE TABLE IF NOT EXISTS registry.artifact_deps (
    artifact_digest text NOT NULL REFERENCES registry.artifacts (digest),
    dep_digest      text NOT NULL REFERENCES registry.artifacts (digest),
    role            text NOT NULL,   -- tools_called | imports | upstream | framework | generator | constraint | optimizer
    PRIMARY KEY (artifact_digest, dep_digest, role)
);
CREATE INDEX IF NOT EXISTS artifact_deps_reverse_idx ON registry.artifact_deps (dep_digest);

-- ---------------------------------------------------------------------------
-- Releases: an immutable, named set of artifacts built from known commits.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS registry.releases (
    release_id              text PRIMARY KEY,         -- e.g. 2026.10.18-ed6002edf4c8
    release_digest          text NOT NULL UNIQUE,
    proto_language_commit   text NOT NULL,
    proto_tools_commit      text NOT NULL,
    source                  jsonb NOT NULL,           -- SourceInfo: toolchain versions, pins
    warnings                jsonb NOT NULL DEFAULT '[]',
    edge_audit              jsonb NOT NULL DEFAULT '[]',
    ci_run_url              text,
    published_by            text NOT NULL DEFAULT current_user,
    published_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.release_artifacts (
    release_id      text NOT NULL REFERENCES registry.releases (release_id),
    kind            registry.artifact_kind NOT NULL,
    key             text NOT NULL,
    artifact_digest text NOT NULL REFERENCES registry.artifacts (digest),
    PRIMARY KEY (release_id, kind, key)
);
CREATE INDEX IF NOT EXISTS release_artifacts_digest_idx ON registry.release_artifacts (artifact_digest);

-- first_release must name a real release once the transaction commits.
DO $$ BEGIN
    ALTER TABLE registry.artifacts
        ADD CONSTRAINT artifacts_first_release_fk FOREIGN KEY (first_release)
        REFERENCES registry.releases (release_id) DEFERRABLE INITIALLY DEFERRED;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- Channels: the only moving pointers, as an append-only event log.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS registry.channel_events (
    channel     text NOT NULL,
    release_id  text NOT NULL REFERENCES registry.releases (release_id),
    moved_by    text NOT NULL DEFAULT current_user,
    moved_at    timestamptz NOT NULL DEFAULT clock_timestamp(),
    reason      text,
    PRIMARY KEY (channel, moved_at)
);

CREATE OR REPLACE VIEW registry.channels AS
SELECT DISTINCT ON (channel) channel, release_id, moved_by, moved_at, reason
FROM registry.channel_events
ORDER BY channel, moved_at DESC;

-- ---------------------------------------------------------------------------
-- Blobs: bytes live in the content-addressed store; Postgres holds the index.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS registry.blobs (
    sha256      text PRIMARY KEY CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes  bigint,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.blob_locations (
    sha256      text NOT NULL REFERENCES registry.blobs (sha256),
    uri         text NOT NULL,                 -- s3://bucket/sha256/aa/bb/<sha256>
    source_id   text,                          -- upstream source that produced it
    origin_url  text,                          -- where it was downloaded from
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (sha256, uri)
);

-- ---------------------------------------------------------------------------
-- Append-only enforcement.
-- ---------------------------------------------------------------------------
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['artifacts', 'artifact_deps', 'releases', 'release_artifacts',
                             'channel_events', 'blobs', 'blob_locations']
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I_append_only ON registry.%I', t, t);
        EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON registry.%I '
                       'FOR EACH ROW EXECUTE FUNCTION registry.forbid_mutation()', t, t);
        EXECUTE format('DROP TRIGGER IF EXISTS %I_no_truncate ON registry.%I', t, t);
        EXECUTE format('CREATE TRIGGER %I_no_truncate BEFORE TRUNCATE ON registry.%I '
                       'FOR EACH STATEMENT EXECUTE FUNCTION registry.forbid_mutation()', t, t);
    END LOOP;
END $$;

-- ---------------------------------------------------------------------------
-- Query helpers.
-- ---------------------------------------------------------------------------

-- Artifacts of the release a channel currently points at.
CREATE OR REPLACE VIEW registry.current_artifacts AS
SELECT c.channel, ra.release_id, ra.kind, ra.key, a.digest, a.local_digest, a.record
FROM registry.channels c
JOIN registry.release_artifacts ra ON ra.release_id = c.release_id
JOIN registry.artifacts a ON a.digest = ra.artifact_digest;

-- Version history of each (kind, key): one row per distinct digest, oldest first.
CREATE OR REPLACE VIEW registry.artifact_versions AS
SELECT a.kind, a.key, a.digest, a.local_digest, a.first_release, r.published_at,
       row_number() OVER (PARTITION BY a.kind, a.key ORDER BY r.published_at, a.digest) AS version
FROM registry.artifacts a
JOIN registry.releases r ON r.release_id = a.first_release;

-- Everything an artifact transitively depends on (program -> components -> tools -> upstream).
CREATE OR REPLACE FUNCTION registry.dependency_closure(root text)
RETURNS TABLE (digest text, kind registry.artifact_kind, key text, depth int, role text)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walk(digest, depth, role) AS (
        SELECT root, 0, 'self'::text
        UNION
        SELECT d.dep_digest, w.depth + 1, d.role
        FROM walk w JOIN registry.artifact_deps d ON d.artifact_digest = w.digest
        WHERE w.depth < 16
    )
    SELECT w.digest, a.kind, a.key, min(w.depth), min(w.role)
    FROM walk w JOIN registry.artifacts a ON a.digest = w.digest
    GROUP BY w.digest, a.kind, a.key;
$$;

-- Everything that transitively depends on an artifact ("what does a Boltz-2 weights change touch?").
CREATE OR REPLACE FUNCTION registry.dependents_closure(root text)
RETURNS TABLE (digest text, kind registry.artifact_kind, key text, depth int)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walk(digest, depth) AS (
        SELECT root, 0
        UNION
        SELECT d.artifact_digest, w.depth + 1
        FROM walk w JOIN registry.artifact_deps d ON d.dep_digest = w.digest
        WHERE w.depth < 16
    )
    SELECT w.digest, a.kind, a.key, min(w.depth)
    FROM walk w JOIN registry.artifacts a ON a.digest = w.digest
    GROUP BY w.digest, a.kind, a.key;
$$;

-- ---------------------------------------------------------------------------
-- Lineage: what ran, pinned to exact artifact digests.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lineage.runs (
    run_id          uuid NOT NULL,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz,
    status          text NOT NULL DEFAULT 'started',   -- started | succeeded | failed
    release_id      text REFERENCES registry.releases (release_id),
    program_digest  text REFERENCES registry.artifacts (digest),
    program         jsonb NOT NULL,                    -- program as executed (seeds included)
    seed            bigint,
    environment     jsonb NOT NULL DEFAULT '{}',       -- host, GPU, driver, PROTO_* env, commits
    results_uri     text,
    PRIMARY KEY (run_id, started_at)
);

-- Exact artifact versions a run used. Rows exist only for digests the
-- registry knows, so every run is traceable down to upstream weights.
CREATE TABLE IF NOT EXISTS lineage.run_artifacts (
    run_id          uuid NOT NULL,
    started_at      timestamptz NOT NULL,
    artifact_digest text NOT NULL REFERENCES registry.artifacts (digest),
    role            text NOT NULL,                     -- program | generator | constraint | optimizer | tool
    PRIMARY KEY (run_id, started_at, artifact_digest, role)
);

-- Time-series layout on TigerData / Timescale; plain tables elsewhere (local dev, CI).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('lineage.runs', 'started_at', if_not_exists => TRUE,
                                  chunk_time_interval => INTERVAL '7 days');
        PERFORM create_hypertable('lineage.run_artifacts', 'started_at', if_not_exists => TRUE,
                                  chunk_time_interval => INTERVAL '7 days');
    END IF;
END $$;

-- Runs finish exactly once: only the started -> terminal transition is allowed.
CREATE OR REPLACE FUNCTION lineage.finish_once() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'lineage.runs is append-only' USING ERRCODE = 'insufficient_privilege';
    END IF;
    IF OLD.status <> 'started'
       OR NEW.run_id <> OLD.run_id OR NEW.started_at <> OLD.started_at
       OR NEW.program IS DISTINCT FROM OLD.program OR NEW.program_digest IS DISTINCT FROM OLD.program_digest
       OR NEW.release_id IS DISTINCT FROM OLD.release_id OR NEW.seed IS DISTINCT FROM OLD.seed THEN
        RAISE EXCEPTION 'lineage.runs rows only move from started to a terminal status'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS runs_finish_once ON lineage.runs;
CREATE TRIGGER runs_finish_once BEFORE UPDATE OR DELETE ON lineage.runs
    FOR EACH ROW EXECUTE FUNCTION lineage.finish_once();
DROP TRIGGER IF EXISTS run_artifacts_append_only ON lineage.run_artifacts;
CREATE TRIGGER run_artifacts_append_only BEFORE UPDATE OR DELETE ON lineage.run_artifacts
    FOR EACH ROW EXECUTE FUNCTION registry.forbid_mutation();

-- ---------------------------------------------------------------------------
-- Roles. Created NOLOGIN; grant them to real login roles per environment.
--   registry_publisher: CI only. Can insert, never modify.
--   registry_reader:    everyone. Read-only.
--   lineage_writer:     program runners. Record runs, read the registry.
-- ---------------------------------------------------------------------------
DO $$ BEGIN CREATE ROLE registry_publisher NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE registry_reader NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE lineage_writer NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

GRANT USAGE ON SCHEMA registry, lineage TO registry_publisher, registry_reader, lineage_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA registry TO registry_publisher, registry_reader, lineage_writer;
GRANT INSERT ON ALL TABLES IN SCHEMA registry TO registry_publisher;
GRANT SELECT ON ALL TABLES IN SCHEMA lineage TO registry_reader, lineage_writer;
GRANT INSERT ON lineage.runs, lineage.run_artifacts TO lineage_writer;
GRANT UPDATE (finished_at, status, results_uri, environment) ON lineage.runs TO lineage_writer;
GRANT EXECUTE ON FUNCTION registry.dependency_closure(text), registry.dependents_closure(text)
    TO registry_publisher, registry_reader, lineage_writer;
