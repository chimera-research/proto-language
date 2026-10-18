"""Build a release lock from the real registries (offline; no network or database)."""

from pathlib import Path

import pytest

from proto_language.provenance.collect import RegistryConfig, build_lock
from proto_language.provenance.models import Lock
from proto_language.provenance.upstream.sources import ToolkitUpstream, UpstreamLock, UpstreamSource

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lock() -> Lock:
    config = RegistryConfig(programs=["examples/jsons/toy.json"])
    upstream = UpstreamLock()
    upstream.toolkits["random_nucleotide"] = ToolkitUpstream(toolkit="random_nucleotide", category="mutagenesis")
    upstream.add("random_nucleotide", UpstreamSource.pypi("numpy", "==2.0.0"))
    return build_lock(REPO_ROOT, config, upstream)


def test_lock_covers_every_registry(lock: Lock) -> None:
    from proto_tools.tools import ToolRegistry

    from proto_language.constraint.constraint_registry import ConstraintRegistry
    from proto_language.generator.generator_registry import GeneratorRegistry
    from proto_language.optimizer.optimizer_registry import OptimizerRegistry

    kinds = {kind: {a.key for a in lock.artifacts if a.kind == kind} for kind in ("tool", "constraint", "generator")}
    assert kinds["tool"] == {s.key for s in ToolRegistry.list_all()}
    assert kinds["constraint"] == {s.key for s in ConstraintRegistry.list_all()}
    assert kinds["generator"] == {s.key for s in GeneratorRegistry.list_all()}
    assert {a.key for a in lock.artifacts if a.kind == "optimizer"} == {s.key for s in OptimizerRegistry.list_all()}
    assert lock.release_digest == Lock.compute_release_digest(lock.source, lock.artifacts)


def test_program_edges_reach_tools_and_upstream(lock: Lock) -> None:
    index = lock.index()
    program = index[("program", "toy")]
    edges = {(d.kind, d.key) for d in program.deps}
    assert edges == {
        ("generator", "random-nucleotide"),
        ("constraint", "gc-content"),
        ("constraint", "max-homopolymer"),
        ("optimizer", "mcmc"),
    }
    generator = index[("generator", "random-nucleotide")]
    assert ("tool", "random-nucleotide-sample") in {(d.kind, d.key) for d in generator.deps}
    tool = index[("tool", "random-nucleotide-sample")]
    assert ("upstream", "pypi:numpy==2.0.0") in {(d.kind, d.key) for d in tool.deps}
    assert all(d.digest == index[(d.kind, d.key)].digest for a in lock.artifacts for d in a.deps)


def test_build_is_deterministic(lock: Lock) -> None:
    config = RegistryConfig(programs=["examples/jsons/toy.json"])
    upstream = UpstreamLock()
    upstream.toolkits["random_nucleotide"] = ToolkitUpstream(toolkit="random_nucleotide", category="mutagenesis")
    upstream.add("random_nucleotide", UpstreamSource.pypi("numpy", "==2.0.0"))
    again = build_lock(REPO_ROOT, config, upstream)
    assert again.release_digest == lock.release_digest


def test_invalid_program_fails_strict_build(tmp_path: Path) -> None:
    bad = REPO_ROOT / "registry" / "_test_invalid_program.json"
    bad.write_text('{"name": "bad", "optimization_stages": [{"constraints": [{"key": "no-such-constraint"}]}]}')
    try:
        with pytest.raises(ValueError, match="no-such-constraint"):
            build_lock(REPO_ROOT, RegistryConfig(programs=["registry/_test_invalid_program.json"]), UpstreamLock())
    finally:
        bad.unlink()
