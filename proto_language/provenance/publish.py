"""Publish a release lock into the Postgres registry (TigerData / Timescale or plain Postgres).

Publishing is one transaction:

1. Apply pending SQL migrations from ``proto_language/provenance/sql/`` (each
   recorded once in ``registry.schema_migrations`` with its SHA-256; an edited
   migration is refused, since migrations are immutable too).
2. Insert the release row, then every artifact and dependency edge with
   ``ON CONFLICT DO NOTHING`` (content addressing makes repeats no-ops),
   then the release membership rows, mirrored blobs, and a channel event.
3. Verify that every artifact already present under the same digest has the
   same ``local_digest`` (a canonicalization bug would otherwise go unnoticed).

Re-running for a lock whose ``release_digest`` is already published is a no-op,
so CI retries are safe. ``psycopg`` (v3) is imported lazily; install it with
``pip install "psycopg[binary]"``.

Examples:
    >>> from proto_language.provenance.publish import publish
    >>> result = publish(dsn, lock, channel="main")  # doctest: +SKIP
    >>> result.release_id  # '2026.10.18-ed6002edf4c8'
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from importlib import resources
from typing import Any

from pydantic import BaseModel, Field

from proto_language.provenance.canonical import canonical_json, digest, sha256_bytes
from proto_language.provenance.models import Lock

__all__ = ["PublishResult", "fetch_release_index", "migrate", "publish", "release_id_for", "verify_release"]

logger = logging.getLogger(__name__)

_ADVISORY_LOCK_KEY = 0x70726F746F  # "proto"


class PublishResult(BaseModel):
    """Outcome of a publish call.

    Attributes:
        release_id (str): Release identifier (new or pre-existing).
        created (bool): False when the release digest was already published.
        new_artifacts (int): Artifacts first introduced by this release.
        total_artifacts (int): Release members.
        channel (str | None): Channel moved to this release, if any.
    """

    release_id: str = Field(description="Release identifier.")
    created: bool = Field(description="False when the release digest was already published.")
    new_artifacts: int = Field(description="Artifacts first introduced by this release.")
    total_artifacts: int = Field(description="Release members.")
    channel: str | None = Field(default=None, description="Channel moved to this release.")


def _connect(dsn: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise RuntimeError('Publishing needs psycopg 3: pip install "psycopg[binary]"') from exc
    return psycopg.connect(dsn)


def _migrations() -> list[tuple[str, str]]:
    sql_dir = resources.files("proto_language.provenance") / "sql"
    files = sorted((p for p in sql_dir.iterdir() if p.name.endswith(".sql")), key=lambda p: p.name)
    return [(p.name, p.read_text()) for p in files]


def migrate(conn: Any) -> list[str]:
    """Apply pending migrations inside the caller's transaction.

    Args:
        conn (Any): Open ``psycopg.Connection``.

    Returns:
        list[str]: Names of migrations applied by this call.

    Raises:
        RuntimeError: If an applied migration's file content has changed.
    """
    applied_now: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
        cur.execute("CREATE SCHEMA IF NOT EXISTS registry")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS registry.schema_migrations ("
            " name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT name, sha256 FROM registry.schema_migrations")
        applied = dict(cur.fetchall())
        for name, sql in _migrations():
            sha = sha256_bytes(sql.encode())
            if name in applied:
                if applied[name] != sha:
                    raise RuntimeError(f"Migration {name} was edited after being applied; add a new migration instead")
                continue
            cur.execute(sql)
            cur.execute("INSERT INTO registry.schema_migrations (name, sha256) VALUES (%s, %s)", (name, sha))
            applied_now.append(name)
    return applied_now


def release_id_for(lock: Lock, when: datetime | None = None) -> str:
    """Human-readable release id: UTC date plus the first 12 hex of the release digest.

    Args:
        lock (Lock): Release lock.
        when (datetime | None): Publication time; defaults to now.

    Returns:
        str: For example ``2026.10.18-ed6002edf4c8``.
    """
    stamp = (when or datetime.now(UTC)).strftime("%Y.%m.%d")
    return f"{stamp}-{lock.release_digest.removeprefix('sha256:')[:12]}"


def _blob_rows(lock: Lock) -> list[tuple[str, int | None, str, str, str]]:
    """``(sha256, size, uri, source_id, origin_url)`` for every mirrored upstream file."""
    rows: list[tuple[str, int | None, str, str, str]] = []
    for art in lock.artifacts:
        if art.kind != "upstream":
            continue
        resolved = art.record.get("resolved") or {}
        if resolved.get("sha256") and resolved.get("mirror"):
            rows.append((resolved["sha256"], resolved.get("size"), resolved["mirror"], art.key, art.record["locator"]))
        rows.extend(
            (f["sha256"], f.get("size"), f["mirror"], art.key, f.get("url") or f.get("path", ""))
            for f in resolved.get("files") or []
            if f.get("sha256") and f.get("mirror")
        )
    return rows


def _move_channel(cur: Any, channel: str, release_id: str, reason: str | None) -> bool:
    """Append a channel event unless the channel already points at ``release_id``."""
    cur.execute("SELECT release_id FROM registry.channels WHERE channel = %s", (channel,))
    current = cur.fetchone()
    if current and current[0] == release_id:
        return False
    cur.execute(
        "INSERT INTO registry.channel_events (channel, release_id, reason) VALUES (%s, %s, %s)",
        (channel, release_id, reason or "publish"),
    )
    return True


def fetch_release_index(dsn: str, ref: str) -> dict[tuple[str, str], tuple[str, str]]:
    """Members of a published release, by channel name or release id.

    Args:
        dsn (str): Postgres connection string (read-only role is enough).
        ref (str): Channel (``main``) or release id.

    Returns:
        dict[tuple[str, str], tuple[str, str]]: ``(kind, key)`` to ``(digest, local_digest)``;
            empty when nothing is published under ``ref``.
    """
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ra.kind::text, ra.key, a.digest, a.local_digest FROM registry.release_artifacts ra"
            " JOIN registry.artifacts a ON a.digest = ra.artifact_digest"
            " WHERE ra.release_id = COALESCE((SELECT release_id FROM registry.channels WHERE channel = %s), %s)",
            (ref, ref),
        )
        return {(kind, key): (dig, local) for kind, key, dig, local in cur.fetchall()}


def publish(
    dsn: str,
    lock: Lock,
    *,
    channel: str | None = "main",
    ci_run_url: str | None = None,
    allow_dirty: bool = False,
    release_id: str | None = None,
) -> PublishResult:
    """Publish ``lock`` as an immutable release.

    Args:
        dsn (str): Postgres connection string (``postgres://user:pass@host:port/db?sslmode=require``).
        lock (Lock): Release lock from :func:`~proto_language.provenance.collect.build_lock`.
        channel (str | None): Channel to point at the release (``None`` leaves channels alone).
        ci_run_url (str | None): Link to the CI run that published.
        allow_dirty (bool): Permit locks built from uncommitted checkouts (never in CI).
        release_id (str | None): Override the generated release id.

    Returns:
        PublishResult: What happened.

    Raises:
        ValueError: If the lock was built from a dirty or unknown checkout.
        RuntimeError: If stored content disagrees with the lock under the same digest.
    """
    src = lock.source
    if not allow_dirty and (src.proto_language_dirty or src.proto_tools_dirty):
        raise ValueError("Refusing to publish a lock built from a dirty checkout")
    if not src.proto_language_commit or not src.proto_tools_commit:
        raise ValueError("Lock has no source commits; build it from git checkouts")
    if Lock.compute_release_digest(src, lock.artifacts) != lock.release_digest:
        raise ValueError("Lock release_digest does not match its contents")
    rid = release_id or release_id_for(lock)

    with _connect(dsn) as conn:
        migrate(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT release_id FROM registry.releases WHERE release_digest = %s", (lock.release_digest,))
            existing = cur.fetchone()
            if existing:
                moved = _move_channel(cur, channel, existing[0], ci_run_url) if channel else False
                logger.info("Release digest already published as %s (channel moved: %s)", existing[0], moved)
                return PublishResult(
                    release_id=existing[0],
                    created=False,
                    new_artifacts=0,
                    total_artifacts=len(lock.artifacts),
                    channel=channel if moved else None,
                )
            cur.execute(
                "INSERT INTO registry.releases (release_id, release_digest, proto_language_commit,"
                " proto_tools_commit, source, warnings, edge_audit, ci_run_url)"
                " VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)",
                (
                    rid,
                    lock.release_digest,
                    src.proto_language_commit,
                    src.proto_tools_commit,
                    canonical_json(src.model_dump(mode="json")),
                    canonical_json(lock.warnings),
                    canonical_json(lock.edge_audit),
                    ci_run_url,
                ),
            )
            cur.execute(
                "CREATE TEMP TABLE incoming (digest text, kind registry.artifact_kind, key text,"
                " local_digest text, record jsonb) ON COMMIT DROP"
            )
            with cur.copy("COPY incoming (digest, kind, key, local_digest, record) FROM STDIN") as copy:
                for art in lock.artifacts:
                    copy.write_row((art.digest, art.kind, art.key, art.local_digest, canonical_json(art.record)))
            cur.execute(
                "SELECT i.digest FROM incoming i JOIN registry.artifacts a USING (digest)"
                " WHERE a.local_digest <> i.local_digest OR a.kind <> i.kind OR a.key <> i.key"
            )
            clashes = [row[0] for row in cur.fetchall()]
            if clashes:
                raise RuntimeError(f"Stored artifacts disagree with the lock under the same digest: {clashes[:5]}")
            cur.execute(
                "INSERT INTO registry.artifacts (digest, kind, key, local_digest, record, first_release)"
                " SELECT digest, kind, key, local_digest, record, %s FROM incoming"
                " ON CONFLICT (digest) DO NOTHING",
                (rid,),
            )
            new_artifacts = cur.rowcount
            # Stage edges so re-published artifacts' existing edges are tolerated.
            cur.execute("CREATE TEMP TABLE incoming_deps (a text, d text, role text) ON COMMIT DROP")
            with cur.copy("COPY incoming_deps (a, d, role) FROM STDIN") as copy:
                for art in lock.artifacts:
                    for dep in art.deps:
                        copy.write_row((art.digest, dep.digest, dep.role))
            cur.execute(
                "INSERT INTO registry.artifact_deps (artifact_digest, dep_digest, role)"
                " SELECT a, d, role FROM incoming_deps ON CONFLICT DO NOTHING"
            )
            cur.execute(
                "INSERT INTO registry.release_artifacts (release_id, kind, key, artifact_digest)"
                " SELECT %s, kind, key, digest FROM incoming",
                (rid,),
            )
            blobs = _blob_rows(lock)
            if blobs:
                cur.executemany(
                    "INSERT INTO registry.blobs (sha256, size_bytes) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(b[0], b[1]) for b in blobs],
                )
                cur.executemany(
                    "INSERT INTO registry.blob_locations (sha256, uri, source_id, origin_url)"
                    " VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    [(b[0], b[2], b[3], b[4]) for b in blobs],
                )
            if channel:
                _move_channel(cur, channel, rid, ci_run_url)
    logger.info("Published %s: %d artifacts (%d new)", rid, len(lock.artifacts), new_artifacts)
    return PublishResult(
        release_id=rid, created=True, new_artifacts=new_artifacts, total_artifacts=len(lock.artifacts), channel=channel
    )


def verify_release(dsn: str, release_id: str) -> list[str]:
    """Recompute every member's digests from stored rows and report mismatches.

    Proves the stored records still hash to their keys, and that the Merkle
    edges stored in ``artifact_deps`` reproduce each ``digest``.

    Args:
        dsn (str): Postgres connection string.
        release_id (str): Release to verify.

    Returns:
        list[str]: Problems; empty when the release verifies.
    """
    problems: list[str] = []
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT a.digest, a.kind::text, a.key, a.local_digest, a.record"
            " FROM registry.release_artifacts ra JOIN registry.artifacts a ON a.digest = ra.artifact_digest"
            " WHERE ra.release_id = %s",
            (release_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return [f"release {release_id} not found or empty"]
        cur.execute(
            "SELECT d.artifact_digest, a.kind::text, a.key, d.role, d.dep_digest FROM registry.artifact_deps d"
            " JOIN registry.artifacts a ON a.digest = d.dep_digest"
            " WHERE d.artifact_digest IN (SELECT artifact_digest FROM registry.release_artifacts WHERE release_id = %s)",
            (release_id,),
        )
        edges: dict[str, list[list[str]]] = {}
        for owner, kind, key, role, dep in cur.fetchall():
            edges.setdefault(owner, []).append([kind, key, role, dep])
    for dig, kind, key, local, record in rows:
        if digest({"kind": kind, "key": key, "record": record}) != local:
            problems.append(f"{kind} {key}: record no longer hashes to local_digest")
        merkle = digest({"kind": kind, "key": key, "local_digest": local, "deps": sorted(edges.get(dig, []))})
        if merkle != dig:
            problems.append(f"{kind} {key}: stored edges do not reproduce digest")
    return problems
