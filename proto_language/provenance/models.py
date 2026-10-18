"""Data models for the release lock: artifacts, dependency edges, and source info.

A *lock* is the deterministic, network-free description of one registry
release. It lists every artifact (upstream source, tool, generator, constraint,
optimizer, program, framework) with a Merkle digest: an artifact's digest
covers its own record plus the digests of everything it depends on, so a
change to a tool re-versions every component and program built on it.

Examples:
    >>> from proto_language.provenance.models import Artifact
    >>> art = Artifact.build(kind="optimizer", key="mcmc", record={"schema": {}}, deps=[])
    >>> art.digest.startswith("sha256:")  # True
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from proto_language.provenance.canonical import digest

__all__ = ["ARTIFACT_KINDS", "Artifact", "ArtifactKind", "Dependency", "Lock", "SourceInfo"]

ArtifactKind = Literal["framework", "upstream", "tool", "generator", "constraint", "optimizer", "program"]

# Publication order: every kind depends only on kinds that appear before it.
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = (
    "framework",
    "upstream",
    "tool",
    "generator",
    "constraint",
    "optimizer",
    "program",
)

LOCK_FORMAT_VERSION = 1


class Dependency(BaseModel):
    """One edge from an artifact to an artifact it depends on.

    Attributes:
        kind (ArtifactKind): Kind of the dependency.
        key (str): Registry key of the dependency.
        digest (str): Merkle digest of the exact dependency version.
        role (str): Why the edge exists (``"tools_called"``, ``"imports"``, ``"upstream"``,
            ``"framework"``, ``"generator"``, ``"constraint"``, ``"optimizer"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ArtifactKind = Field(description="Kind of the dependency.")
    key: str = Field(description="Registry key of the dependency.")
    digest: str = Field(description="Merkle digest of the exact dependency version.")
    role: str = Field(description="Why the edge exists.")


class Artifact(BaseModel):
    """One immutable registry entry.

    Attributes:
        kind (ArtifactKind): Artifact kind.
        key (str): Registry key, unique within ``kind`` for one release.
        digest (str): Merkle digest over ``kind``, ``key``, ``record`` and the
            dependency digests. This is the identity runs pin to.
        local_digest (str): Digest over ``kind``, ``key`` and ``record`` only.
            Unchanged when only a dependency moved.
        record (dict[str, Any]): Canonical, JSON-serializable description.
        deps (list[Dependency]): Edges to the exact dependency versions.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind = Field(description="Artifact kind.")
    key: str = Field(description="Registry key, unique within kind for one release.")
    digest: str = Field(description="Merkle digest over the record and dependency digests.")
    local_digest: str = Field(description="Digest over the record only.")
    record: dict[str, Any] = Field(description="Canonical, JSON-serializable description.")
    deps: list[Dependency] = Field(default_factory=list, description="Edges to exact dependency versions.")

    @classmethod
    def build(cls, kind: ArtifactKind, key: str, record: dict[str, Any], deps: list[Dependency]) -> Artifact:
        """Construct an artifact and compute both digests.

        Args:
            kind (ArtifactKind): Artifact kind.
            key (str): Registry key.
            record (dict[str, Any]): Canonical description.
            deps (list[Dependency]): Dependency edges; order does not matter.

        Returns:
            Artifact: The artifact with ``digest`` and ``local_digest`` filled in.
        """
        ordered = sorted(set(deps), key=lambda d: (d.kind, d.key, d.role))
        local = digest({"kind": kind, "key": key, "record": record})
        merkle = digest(
            {
                "kind": kind,
                "key": key,
                "local_digest": local,
                "deps": [[d.kind, d.key, d.role, d.digest] for d in ordered],
            }
        )
        return cls(kind=kind, key=key, digest=merkle, local_digest=local, record=record, deps=ordered)

    def as_dependency(self, role: str) -> Dependency:
        """Return an edge pointing at this artifact.

        Args:
            role (str): Edge role recorded on the dependent artifact.

        Returns:
            Dependency: Edge carrying this artifact's kind, key, and digest.
        """
        return Dependency(kind=self.kind, key=self.key, digest=self.digest, role=role)


class SourceInfo(BaseModel):
    """Where a lock was built from.

    Attributes:
        proto_language_commit (str | None): ``HEAD`` of the proto-language checkout.
        proto_language_dirty (bool): Whether tracked files had uncommitted changes.
        proto_tools_commit (str | None): Commit of the proto-tools checkout actually imported.
        proto_tools_dirty (bool): Whether the proto-tools checkout had uncommitted changes.
        proto_tools_pinned_commit (str | None): Submodule pointer recorded in proto-language.
        python_version (str): ``major.minor`` of the building interpreter.
        pydantic_version (str): Pydantic version, which shapes every JSON schema.
    """

    model_config = ConfigDict(extra="forbid")

    proto_language_commit: str | None = Field(description="HEAD of the proto-language checkout.")
    proto_language_dirty: bool = Field(description="Whether tracked files had uncommitted changes.")
    proto_tools_commit: str | None = Field(description="Commit of the imported proto-tools checkout.")
    proto_tools_dirty: bool = Field(description="Whether proto-tools had uncommitted changes.")
    proto_tools_pinned_commit: str | None = Field(description="Submodule pointer recorded in proto-language.")
    python_version: str = Field(description="major.minor of the building interpreter.")
    pydantic_version: str = Field(description="Pydantic version used to render schemas.")


class Lock(BaseModel):
    """A complete, deterministic release description.

    Attributes:
        format_version (int): Lock file format version.
        source (SourceInfo): Checkout and toolchain the lock was built from.
        artifacts (list[Artifact]): Every artifact, sorted by kind order then key.
        warnings (list[str]): Non-fatal findings (missing upstream entries, skipped programs).
        edge_audit (list[str]): Tools a component's code imports but its ``tools_called`` omits.
            Those tools are still dependency edges (role ``imports``); the list is advisory.
        release_digest (str): Digest over the ``(kind, key, digest)`` set and ``source`` commits.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: int = Field(default=LOCK_FORMAT_VERSION, description="Lock file format version.")
    source: SourceInfo = Field(description="Checkout and toolchain the lock was built from.")
    artifacts: list[Artifact] = Field(description="Every artifact, sorted by kind order then key.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal findings.")
    edge_audit: list[str] = Field(default_factory=list, description="Imported but undeclared tool edges.")
    release_digest: str = Field(description="Digest over the artifact set and source commits.")

    @staticmethod
    def compute_release_digest(source: SourceInfo, artifacts: list[Artifact]) -> str:
        """Digest the artifact set together with the commits it was built from.

        Args:
            source (SourceInfo): Build source.
            artifacts (list[Artifact]): Release members.

        Returns:
            str: Prefixed digest.
        """
        members = sorted([a.kind, a.key, a.digest] for a in artifacts)
        return digest(
            {
                "format_version": LOCK_FORMAT_VERSION,
                "proto_language_commit": source.proto_language_commit,
                "proto_tools_commit": source.proto_tools_commit,
                "members": members,
            }
        )

    def index(self) -> dict[tuple[str, str], Artifact]:
        """Map ``(kind, key)`` to artifact.

        Returns:
            dict[tuple[str, str], Artifact]: Lookup table.
        """
        return {(a.kind, a.key): a for a in self.artifacts}
