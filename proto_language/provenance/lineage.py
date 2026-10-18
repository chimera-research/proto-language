"""Record program runs against the registry so every result traces to exact artifact versions.

A run row stores the program exactly as executed (seeds included) and the
release it ran against; ``lineage.run_artifacts`` lists the digest of every
component the program used. From there ``registry.dependency_closure`` walks
down to tools, environments, and upstream code and weights.

The release is found by matching the running checkout's commits to a
published release. A run from an unreleased checkout is still recorded, with
``release_id`` null and the commits in ``environment``, so nothing is lost;
it simply cannot be resolved to artifact digests.

Examples:
    >>> from proto_language.provenance.lineage import finish_run, start_run
    >>> run_id = start_run(dsn, program_json, repo_root=Path("."))  # doctest: +SKIP
    >>> finish_run(dsn, run_id, status="succeeded", results_uri="s3://runs/abc/")  # doctest: +SKIP
"""

from __future__ import annotations

import os
import platform
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from proto_language.provenance.canonical import canonical_json
from proto_language.provenance.collect import program_refs, source_info
from proto_language.provenance.models import Artifact, Dependency
from proto_language.provenance.publish import _connect

__all__ = ["finish_run", "start_run"]


def _environment(repo_root: Path) -> dict[str, Any]:
    info = source_info(repo_root)
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commits": info.model_dump(mode="json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "proto_env": {k: v for k, v in os.environ.items() if k.startswith(("PROTO_", "HF_HUB_OFFLINE", "HF_ENDPOINT"))},
    }


def start_run(
    dsn: str,
    program: dict[str, Any],
    *,
    repo_root: Path,
    seed: int | None = None,
    release_id: str | None = None,
) -> uuid.UUID:
    """Insert a ``started`` run and its artifact usage rows.

    Args:
        dsn (str): Postgres connection string (a ``lineage_writer`` login).
        program (dict[str, Any]): Declarative program JSON as executed.
        repo_root (Path): proto-language checkout, used to find the matching release.
        seed (int | None): Program-level seed; defaults to ``program["seed"]``.
        release_id (str | None): Release to attribute the run to; looked up by commits when omitted.

    Returns:
        uuid.UUID: The new run id.
    """
    run_id = uuid.uuid4()
    started_at = datetime.now(UTC)
    env = _environment(repo_root)
    commits = env["commits"]
    with _connect(dsn) as conn, conn.cursor() as cur:
        if release_id is None and not (commits["proto_language_dirty"] or commits["proto_tools_dirty"]):
            cur.execute(
                "SELECT release_id FROM registry.releases WHERE proto_language_commit = %s AND proto_tools_commit = %s"
                " ORDER BY published_at DESC LIMIT 1",
                (commits["proto_language_commit"], commits["proto_tools_commit"]),
            )
            row = cur.fetchone()
            release_id = row[0] if row else None

        usage: list[tuple[str, str]] = []
        program_digest: str | None = None
        if release_id is not None:
            cur.execute(
                "SELECT kind::text, key, artifact_digest FROM registry.release_artifacts WHERE release_id = %s",
                (release_id,),
            )
            members = {(kind, key): dig for kind, key, dig in cur.fetchall()}
            deps = []
            for kind, key, _config in program_refs(program):
                dig = members.get((kind, key))
                if dig is not None:
                    usage.append((dig, kind))
                    deps.append(Dependency(kind=kind, key=key, digest=dig, role=kind))
            name = str(program.get("name") or "")
            candidate = Artifact.build(kind="program", key=name, record={"program": program}, deps=deps).digest
            if members.get(("program", name)) == candidate:
                program_digest = candidate
                usage.append((candidate, "program"))

        cur.execute(
            "INSERT INTO lineage.runs (run_id, started_at, release_id, program_digest, program, seed, environment)"
            " VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)",
            (
                run_id,
                started_at,
                release_id,
                program_digest,
                canonical_json(program),
                seed if seed is not None else program.get("seed"),
                canonical_json(env),
            ),
        )
        for dig, role in sorted(set(usage)):
            cur.execute(
                "INSERT INTO lineage.run_artifacts (run_id, started_at, artifact_digest, role) VALUES (%s, %s, %s, %s)",
                (run_id, started_at, dig, role),
            )
    return run_id


def finish_run(
    dsn: str,
    run_id: uuid.UUID,
    *,
    status: Literal["succeeded", "failed"],
    results_uri: str | None = None,
) -> None:
    """Move a run from ``started`` to a terminal status (allowed exactly once).

    Args:
        dsn (str): Postgres connection string.
        run_id (uuid.UUID): Run returned by :func:`start_run`.
        status (Literal["succeeded", "failed"]): Terminal status.
        results_uri (str | None): Where results were exported.
    """
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE lineage.runs SET status = %s, finished_at = now(), results_uri = %s WHERE run_id = %s",
            (status, results_uri, run_id),
        )
