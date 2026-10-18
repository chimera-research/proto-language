"""Registry releases, upstream control, and run lineage for proto-language and proto-tools.

``provenance`` turns the in-process registries (tools, generators, constraints,
optimizers) and published programs into immutable, content-addressed releases
in Postgres, pins every upstream code and weights source a toolkit pulls, and
records which exact artifact versions each program run used. See
``notes/registry.md`` and ``notes/upstream-control.md``.

Examples:
    >>> from proto_language.provenance import build_lock
    >>> lock = build_lock(Path("."))  # doctest: +SKIP
    >>> lock.release_digest  # 'sha256:...'
"""

from proto_language.provenance.canonical import canonical_json, digest
from proto_language.provenance.collect import RegistryConfig, build_lock, load_config
from proto_language.provenance.models import Artifact, Dependency, Lock, SourceInfo

__all__ = [
    "Artifact",
    "Dependency",
    "Lock",
    "RegistryConfig",
    "SourceInfo",
    "build_lock",
    "canonical_json",
    "digest",
    "load_config",
]
