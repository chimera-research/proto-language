"""Curated upstream declarations and carry-over between lock generations.

Discovery is heuristic. ``registry/upstream/declarations.yaml`` is the
human-owned correction layer: it adds sources discovery cannot see, pins
floating references to exact revisions, drops false positives, and records
live services a toolkit calls at run time (which cannot be mirrored). A
declaration that pins a source (``hf:org/model@<sha>``) replaces the floating
sighting of the same base id.

Examples:
    >>> from proto_language.provenance.upstream.declarations import apply_declarations
    >>> apply_declarations(lock, {"toolkits": {"boltz2": {"ignore": ["git:github.com/x/y"]}}})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from proto_language.provenance.upstream.sources import UpstreamLock, UpstreamSource

__all__ = ["apply_declarations", "carry_over", "declared_source", "load_declarations"]

_PRESERVED_RESOLVED_KEYS = ("sha256", "mirror", "size", "mirrored_at")


def load_declarations(path: Path) -> dict[str, Any]:
    """Load ``declarations.yaml``; a missing file means no declarations.

    Args:
        path (Path): File path.

    Returns:
        dict[str, Any]: Parsed mapping with a ``toolkits`` key.

    Raises:
        ValueError: If the top level is not a mapping.
    """
    if not path.exists():
        return {"toolkits": {}}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    data.setdefault("toolkits", {})
    return data


def declared_source(entry: dict[str, Any], where: str) -> UpstreamSource:
    """Build a source from one ``add:`` entry.

    Args:
        entry (dict[str, Any]): Mapping with ``type``, ``locator``, and optional
            ``ref``, ``role``, ``notes``, ``sha256``.
        where (str): Evidence string (declarations file and toolkit).

    Returns:
        UpstreamSource: The declared source.

    Raises:
        ValueError: If ``type`` is unknown or required fields are missing.
    """
    kind = entry.get("type")
    locator = entry.get("locator")
    if not kind or not locator:
        raise ValueError(f"{where}: declarations need 'type' and 'locator'")
    ref = entry.get("ref")
    if kind == "git":
        src = UpstreamSource.git(f"https://{locator}" if "://" not in locator else locator, ref, found_in=where)
    elif kind == "pypi":
        src = UpstreamSource.pypi(locator, ref or "", role="code", found_in=where)
    elif kind in ("hf-model", "hf-dataset"):
        src = UpstreamSource.hf(locator, ref, dataset=kind == "hf-dataset", found_in=where)
    elif kind == "url":
        src = UpstreamSource.url(locator, role="weights", found_in=where)
        if entry.get("sha256"):
            src.resolved["sha256"] = entry["sha256"]
    else:
        raise ValueError(f"{where}: unknown source type {kind!r}")
    if entry.get("role"):
        src.role = entry["role"]
    src.notes = entry.get("notes")
    return src


def apply_declarations(lock: UpstreamLock, declarations: dict[str, Any]) -> list[str]:
    """Merge curated declarations into a freshly discovered lock, in place.

    Args:
        lock (UpstreamLock): Discovered lock.
        declarations (dict[str, Any]): Parsed ``declarations.yaml``.

    Returns:
        list[str]: Problems found (unknown toolkits, ignores that matched nothing).
    """
    problems: list[str] = []
    for toolkit, raw_decl in (declarations.get("toolkits") or {}).items():
        entry = lock.toolkits.get(toolkit)
        if entry is None:
            problems.append(f"declarations.yaml: unknown toolkit {toolkit!r}")
            continue
        decl = raw_decl or {}
        for sid in decl.get("ignore") or []:
            if sid in entry.sources:
                entry.sources.remove(sid)
            else:
                problems.append(f"declarations.yaml: {toolkit} ignores {sid!r}, which was not discovered")
        for raw in decl.get("add") or []:
            src = declared_source(raw, f"registry/upstream/declarations.yaml#{toolkit}")
            base = src.id.split("@", 1)[0] if "@" in src.id.split(":", 1)[1] else None
            if base and base in entry.sources:
                entry.sources.remove(base)
            lock.add(toolkit, src)
            lock.sources[src.id].notes = src.notes
            if src.role != lock.sources[src.id].role:
                lock.sources[src.id].role = src.role
        entry.services = sorted(set(entry.services) | set(decl.get("services") or []))
        if decl.get("notes"):
            entry.notes = str(decl["notes"]).strip()
    used = {sid for e in lock.toolkits.values() for sid in e.sources}
    for sid in list(lock.sources):
        if sid not in used:
            del lock.sources[sid]
    return problems


def carry_over(lock: UpstreamLock, previous: UpstreamLock | None) -> None:
    """Keep mirror-recorded content hashes from the previous lock generation.

    URL sources only become ``pinned`` after ``mirror.py`` downloads them and
    records a SHA-256; rediscovery must not forget that.

    Args:
        lock (UpstreamLock): New lock (modified in place).
        previous (UpstreamLock | None): Previously checked-in lock.
    """
    if previous is None:
        return
    for sid, src in lock.sources.items():
        old = previous.sources.get(sid)
        if old is None:
            continue
        for key in _PRESERVED_RESOLVED_KEYS:
            if key in old.resolved and key not in src.resolved:
                src.resolved[key] = old.resolved[key]
