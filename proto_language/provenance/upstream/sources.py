"""Data models for upstream sources: what a toolkit pulls from outside our repos.

An *upstream source* is anything a proto-tools toolkit fetches at setup or run
time from a third party: a git repository, a PyPI distribution, a Hugging Face
model or dataset repository, or a plain URL (weights tarballs, binaries). The
discovery pass records candidates; the resolve pass pins each one to an
immutable identifier (commit SHA, file SHA-256, HF revision SHA).

Examples:
    >>> from proto_language.provenance.upstream.sources import UpstreamSource
    >>> src = UpstreamSource.pypi("boltz", "==2.2.1", role="code")
    >>> src.id  # 'pypi:boltz==2.2.1'
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PinStatus",
    "SourceRole",
    "SourceType",
    "ToolkitUpstream",
    "UpstreamLock",
    "UpstreamSource",
    "normalize_git_url",
]

SourceType = Literal["git", "pypi", "hf-model", "hf-dataset", "url"]
SourceRole = Literal["code", "weights", "data", "binary", "dependency"]
# pinned: an immutable identifier is recorded (commit SHA, file SHA-256, HF revision SHA).
# floating: resolvable, but the toolkit fetches a moving ref (branch, range, mutable URL).
# unresolved: not yet resolved (offline build, network denied, template URL).
# missing: the resolver reached the host and the source does not exist.
PinStatus = Literal["pinned", "floating", "unresolved", "missing"]

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def normalize_git_url(url: str) -> str:
    """Normalize a git remote to ``host/owner/repo`` (no scheme, no ``.git``).

    Args:
        url (str): Remote URL in https, ssh, or ``git@host:owner/repo`` form.

    Returns:
        str: Normalized locator, lowercase host.
    """
    url = url.strip().removeprefix("git+")
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    url = re.sub(r"^(https?|ssh|git)://([^@/]+@)?", "", url)
    url = url.split("#", 1)[0].rstrip("/").removesuffix(".git")
    host, _, path = url.partition("/")
    return f"{host.lower()}/{path}"


class UpstreamSource(BaseModel):
    """One upstream dependency of one or more toolkits.

    Attributes:
        id (str): Stable identifier (``git:``, ``pypi:``, ``hf:``, ``hf-dataset:``, ``url:`` prefix).
        type (SourceType): Source protocol.
        role (SourceRole): What the toolkit uses it for.
        locator (str): Repository URL, distribution name, HF repo id, or URL.
        ref (str | None): Requested ref: git branch/tag/SHA, PEP 440 specifier, or HF revision.
        found_in (list[str]): ``path:line`` evidence for where the source is referenced.
        template (bool): The reference contains unexpanded variables, so the exact
            target depends on runtime configuration.
        status (PinStatus): Pinning status after resolution.
        resolved (dict[str, Any]): Immutable identifiers recorded by the resolver.
        notes (str | None): Curator notes from ``declarations.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier with a type prefix.")
    type: SourceType = Field(description="Source protocol.")
    role: SourceRole = Field(description="What the toolkit uses it for.")
    locator: str = Field(description="Repository URL, distribution name, HF repo id, or URL.")
    ref: str | None = Field(default=None, description="Requested ref, specifier, or revision.")
    found_in: list[str] = Field(default_factory=list, description="Evidence of where it is referenced.")
    template: bool = Field(default=False, description="Reference contains unexpanded variables.")
    status: PinStatus = Field(default="unresolved", description="Pinning status after resolution.")
    resolved: dict[str, Any] = Field(default_factory=dict, description="Immutable identifiers from the resolver.")
    notes: str | None = Field(default=None, description="Curator notes.")

    @classmethod
    def git(cls, url: str, ref: str | None, role: SourceRole = "code", found_in: str | None = None) -> UpstreamSource:
        """Build a git source.

        Args:
            url (str): Remote URL.
            ref (str | None): Branch, tag, or commit; ``None`` means the default branch.
            role (SourceRole): Usage role.
            found_in (str | None): Evidence location.

        Returns:
            UpstreamSource: The source.
        """
        locator = normalize_git_url(url)
        sid = f"git:{locator}" + (f"@{ref}" if ref else "")
        return cls(id=sid, type="git", role=role, locator=locator, ref=ref, found_in=[found_in] if found_in else [])

    @classmethod
    def pypi(cls, name: str, spec: str, role: SourceRole = "dependency", found_in: str | None = None) -> UpstreamSource:
        """Build a PyPI source.

        Args:
            name (str): Distribution name (normalized to PEP 503 form).
            spec (str): PEP 440 specifier, empty for unconstrained.
            role (SourceRole): Usage role.
            found_in (str | None): Evidence location.

        Returns:
            UpstreamSource: The source.
        """
        norm = re.sub(r"[-_.]+", "-", name).lower()
        return cls(
            id=f"pypi:{norm}{spec}",
            type="pypi",
            role=role,
            locator=norm,
            ref=spec or None,
            found_in=[found_in] if found_in else [],
        )

    @classmethod
    def hf(
        cls,
        repo_id: str,
        revision: str | None,
        dataset: bool = False,
        role: SourceRole | None = None,
        found_in: str | None = None,
    ) -> UpstreamSource:
        """Build a Hugging Face model or dataset source.

        Args:
            repo_id (str): ``owner/name`` repository id.
            revision (str | None): Revision requested by the code, if any.
            dataset (bool): Dataset repository instead of model repository.
            role (SourceRole | None): Usage role; defaults to ``weights`` or ``data``.
            found_in (str | None): Evidence location.

        Returns:
            UpstreamSource: The source.
        """
        prefix = "hf-dataset" if dataset else "hf"
        return cls(
            id=f"{prefix}:{repo_id}" + (f"@{revision}" if revision else ""),
            type="hf-dataset" if dataset else "hf-model",
            role=role or ("data" if dataset else "weights"),
            locator=repo_id,
            ref=revision,
            found_in=[found_in] if found_in else [],
        )

    @classmethod
    def url(cls, url: str, role: SourceRole, template: bool = False, found_in: str | None = None) -> UpstreamSource:
        """Build a plain-URL source.

        Args:
            url (str): Download URL.
            role (SourceRole): Usage role.
            template (bool): URL contains unexpanded variables.
            found_in (str | None): Evidence location.

        Returns:
            UpstreamSource: The source.
        """
        return cls(
            id=f"url:{url}",
            type="url",
            role=role,
            locator=url,
            template=template,
            found_in=[found_in] if found_in else [],
        )

    @property
    def ref_is_immutable(self) -> bool:
        """Whether the requested ref alone is an immutable pin (full SHA or ``==`` version)."""
        if self.ref is None:
            return False
        if self.type in ("git", "hf-model", "hf-dataset"):
            return bool(_SHA_RE.match(self.ref))
        if self.type == "pypi":
            return self.ref.startswith("==") and "*" not in self.ref and "," not in self.ref
        return False

    def merge(self, other: UpstreamSource) -> None:
        """Fold another sighting of the same source into this one (union of evidence).

        Args:
            other (UpstreamSource): Another source with the same ``id``.
        """
        for loc in other.found_in:
            if loc not in self.found_in:
                self.found_in.append(loc)
        self.template = self.template or other.template
        # Prefer the more specific role when sightings disagree (weights beats dependency).
        rank = {"weights": 0, "data": 1, "code": 2, "binary": 3, "dependency": 4}
        if rank[other.role] < rank[self.role]:
            self.role = other.role


class ToolkitUpstream(BaseModel):
    """Upstream sources referenced by one toolkit.

    Attributes:
        toolkit (str): Toolkit directory name (e.g. ``boltz2``).
        category (str): Tool category directory (e.g. ``structure_prediction``).
        env_name (str | None): Standalone environment name, if the toolkit owns or shares one.
        license (dict[str, Any]): Parsed ``license.yaml``.
        sources (list[str]): Source ids, sorted.
        services (list[str]): Live remote services called at run time (cannot be mirrored).
        notes (str | None): Curator notes from ``declarations.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    toolkit: str = Field(description="Toolkit directory name.")
    category: str = Field(description="Tool category directory.")
    env_name: str | None = Field(default=None, description="Standalone environment name.")
    license: dict[str, Any] = Field(default_factory=dict, description="Parsed license.yaml.")
    sources: list[str] = Field(default_factory=list, description="Source ids, sorted.")
    services: list[str] = Field(default_factory=list, description="Live remote services called at run time.")
    notes: str | None = Field(default=None, description="Curator notes.")


class UpstreamLock(BaseModel):
    """The checked-in upstream lock: every toolkit and every source it pulls.

    Attributes:
        format_version (int): File format version.
        proto_tools_commit (str | None): proto-tools commit the scan ran against.
        toolkits (dict[str, ToolkitUpstream]): Toolkit name to its sources.
        sources (dict[str, UpstreamSource]): Source id to source.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: int = Field(default=1, description="File format version.")
    proto_tools_commit: str | None = Field(default=None, description="proto-tools commit scanned.")
    toolkits: dict[str, ToolkitUpstream] = Field(default_factory=dict, description="Toolkit to sources.")
    sources: dict[str, UpstreamSource] = Field(default_factory=dict, description="Source id to source.")

    def add(self, toolkit: str, source: UpstreamSource) -> None:
        """Attach ``source`` to ``toolkit``, merging with an existing sighting.

        Args:
            toolkit (str): Toolkit name; must already be present in ``toolkits``.
            source (UpstreamSource): Source to attach.
        """
        existing = self.sources.get(source.id)
        if existing is None:
            self.sources[source.id] = source
        else:
            existing.merge(source)
        entry = self.toolkits[toolkit]
        if source.id not in entry.sources:
            entry.sources.append(source.id)
            entry.sources.sort()

    def summary(self) -> dict[str, int]:
        """Count sources by status.

        Returns:
            dict[str, int]: Status to count.
        """
        counts: dict[str, int] = {}
        for src in self.sources.values():
            counts[src.status] = counts.get(src.status, 0) + 1
        return counts
