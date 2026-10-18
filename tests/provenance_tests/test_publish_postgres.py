"""Publish, re-publish, verify, and immutability against a real Postgres.

Opt-in: set ``REGISTRY_TEST_DATABASE_URL`` to a disposable database and run with
``--integration``. The test creates the ``registry`` and ``lineage`` SQL schemas.
"""

import os

import pytest

from proto_language.provenance.models import Artifact, Lock, SourceInfo

DSN = os.environ.get("REGISTRY_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="REGISTRY_TEST_DATABASE_URL not set"),
]


def _lock(weights_rev: str, commit: str) -> Lock:
    source = SourceInfo(
        proto_language_commit=commit,
        proto_language_dirty=False,
        proto_tools_commit="b" * 40,
        proto_tools_dirty=False,
        proto_tools_pinned_commit="b" * 40,
        python_version="3.12",
        pydantic_version="2",
    )
    weights = Artifact.build("upstream", "hf:test/model", {"resolved": {"commit": weights_rev}, "locator": "x"}, [])
    tool = Artifact.build("tool", "test-tool", {"schemas": {"x": 1.5}}, [weights.as_dependency("upstream")])
    constraint = Artifact.build("constraint", "test-c", {"cfg": None}, [tool.as_dependency("tools_called")])
    members = [weights, tool, constraint]
    return Lock(source=source, artifacts=members, release_digest=Lock.compute_release_digest(source, members))


def test_publish_is_idempotent_verifiable_and_append_only() -> None:
    import psycopg

    from proto_language.provenance.publish import fetch_release_index, publish, verify_release

    assert DSN is not None
    nonce = os.urandom(4).hex()
    first = _lock("1" * 40, nonce.ljust(40, "0"))
    result = publish(DSN, first, channel=f"test-{nonce}", release_id=f"test-{nonce}-1")
    assert result.created and result.total_artifacts == 3
    assert not publish(DSN, first, channel=f"test-{nonce}").created
    assert verify_release(DSN, f"test-{nonce}-1") == []

    second = _lock("2" * 40, nonce.ljust(40, "1"))
    result = publish(DSN, second, channel=f"test-{nonce}", release_id=f"test-{nonce}-2")
    assert result.new_artifacts == 3  # weights changed, so tool and constraint are re-versioned
    index = fetch_release_index(DSN, f"test-{nonce}")
    assert index[("constraint", "test-c")][0] == second.index()[("constraint", "test-c")].digest

    with psycopg.connect(DSN) as conn, conn.cursor() as cur, pytest.raises(psycopg.errors.InsufficientPrivilege):
        cur.execute("DELETE FROM registry.releases WHERE release_id = %s", (f"test-{nonce}-1",))
