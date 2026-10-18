"""Canonical encoding, artifact digests, and lock diffs (no registries, network, or DB)."""

from proto_language.provenance.canonical import canonical_json, digest, strip_presentation
from proto_language.provenance.cli import diff_locks
from proto_language.provenance.models import Artifact, Lock, SourceInfo


def _source(commit: str = "a" * 40) -> SourceInfo:
    return SourceInfo(
        proto_language_commit=commit,
        proto_language_dirty=False,
        proto_tools_commit="b" * 40,
        proto_tools_dirty=False,
        proto_tools_pinned_commit="b" * 40,
        python_version="3.12",
        pydantic_version="2.12.0",
    )


def _chain(weights_rev: str) -> dict[str, Artifact]:
    """Upstream -> tool -> constraint -> program, plus an unrelated optimizer."""
    weights = Artifact.build("upstream", "hf:org/model", {"resolved": {"commit": weights_rev}}, [])
    tool = Artifact.build("tool", "model-predict", {"schemas": {}}, [weights.as_dependency("upstream")])
    constraint = Artifact.build("constraint", "model-score", {"config": {}}, [tool.as_dependency("tools_called")])
    optimizer = Artifact.build("optimizer", "mcmc", {"config": {}}, [])
    program = Artifact.build(
        "program",
        "demo",
        {"program": {"name": "demo"}},
        [constraint.as_dependency("constraint"), optimizer.as_dependency("optimizer")],
    )
    return {a.key: a for a in (weights, tool, constraint, optimizer, program)}


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": {"y": 2, "x": [3, 1]}}) == '{"a":{"x":[3,1],"y":2},"b":1}'
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    assert digest({"a": [1, 2]}) != digest({"a": [2, 1]})


def test_strip_presentation_keeps_defaults_and_drops_docs() -> None:
    schema = {"properties": {"x": {"type": "integer", "default": 3, "title": "X", "description": "doc"}}}
    assert strip_presentation(schema) == {"properties": {"x": {"default": 3, "type": "integer"}}}


def test_dependency_order_does_not_change_digest() -> None:
    a = Artifact.build("tool", "a", {}, [])
    b = Artifact.build("tool", "b", {}, [])
    first = Artifact.build("constraint", "c", {}, [a.as_dependency("tools_called"), b.as_dependency("tools_called")])
    second = Artifact.build("constraint", "c", {}, [b.as_dependency("tools_called"), a.as_dependency("tools_called")])
    assert first.digest == second.digest


def test_upstream_change_propagates_through_merkle_chain() -> None:
    before, after = _chain("1" * 40), _chain("2" * 40)
    for key in ("hf:org/model", "model-predict", "model-score", "demo"):
        assert before[key].digest != after[key].digest, key
    # Only the upstream record itself changed; everything above was re-versioned by an edge.
    assert before["hf:org/model"].local_digest != after["hf:org/model"].local_digest
    for key in ("model-predict", "model-score", "demo"):
        assert before[key].local_digest == after[key].local_digest, key
    assert before["mcmc"].digest == after["mcmc"].digest


def test_diff_classifies_changed_and_reversioned() -> None:
    def lock(arts: dict[str, Artifact]) -> Lock:
        members = list(arts.values())
        return Lock(source=_source(), artifacts=members, release_digest=Lock.compute_release_digest(_source(), members))

    from proto_language.provenance.cli import lock_index

    diff = diff_locks(lock_index(lock(_chain("1" * 40))), lock_index(lock(_chain("2" * 40))))
    assert diff["changed"] == ["upstream:hf:org/model"]
    assert diff["reversioned"] == ["constraint:model-score", "program:demo", "tool:model-predict"]
    assert diff["added"] == diff["removed"] == []


def test_release_digest_depends_on_commits_and_members() -> None:
    members = list(_chain("1" * 40).values())
    base = Lock.compute_release_digest(_source(), members)
    assert base == Lock.compute_release_digest(_source(), list(reversed(members)))
    assert base != Lock.compute_release_digest(_source("c" * 40), members)
    assert base != Lock.compute_release_digest(_source(), members[:-1])
