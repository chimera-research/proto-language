"""Canonical JSON encoding and content digests for registry artifacts.

Every artifact the registry publishes is identified by the SHA-256 of its
canonical JSON form, so two builds of the same source produce byte-identical
records and identical digests. Canonical form means sorted keys, no
insignificant whitespace, UTF-8, and no floats that differ only by repr.

Examples:
    >>> from proto_language.provenance.canonical import digest
    >>> digest({"b": 1, "a": [1, 2]}) == digest({"a": [1, 2], "b": 1})  # True
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "PRESENTATION_SCHEMA_KEYS",
    "canonical_json",
    "digest",
    "files_digest",
    "sha256_bytes",
    "sha256_file",
    "strip_presentation",
]

# JSON-schema keys that describe presentation rather than behavior. Mirrors the
# ignore-set in ``proto_tools.modal.fingerprint`` (minus ``default``: a registry
# record must change when a default changes, because a program that omits the
# field silently picks up the new value).
PRESENTATION_SCHEMA_KEYS = frozenset(
    {"description", "title", "examples", "x-proto-doc", "include_in_key", "reload_on_change", "_field_type"}
)


def canonical_json(value: Any) -> str:
    """Encode ``value`` as canonical JSON (sorted keys, compact separators, UTF-8).

    Args:
        value (Any): JSON-serializable value.

    Returns:
        str: Canonical JSON text.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``.

    Args:
        data (bytes): Bytes to hash.

    Returns:
        str: 64-character lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """Return the ``sha256:<hex>`` digest of ``value``'s canonical JSON.

    Args:
        value (Any): JSON-serializable value.

    Returns:
        str: Prefixed digest, e.g. ``"sha256:9f86d0..."``.
    """
    return "sha256:" + sha256_bytes(canonical_json(value).encode())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream ``path`` through SHA-256 without loading it into memory.

    Args:
        path (Path): File to hash.
        chunk_size (int): Read size in bytes.

    Returns:
        str: 64-character lowercase hex digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def files_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash a set of files by root-relative path and content.

    Paths are sorted and made relative to ``root`` so the digest does not move
    with the checkout location.

    Args:
        paths (Iterable[Path]): Files to include.
        root (Path): Directory the recorded paths are relative to.

    Returns:
        str: Prefixed digest over ``(relative path, content)`` pairs.
    """
    hasher = hashlib.sha256()
    for path in sorted({p.resolve() for p in paths}):
        hasher.update(path.relative_to(root.resolve()).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def strip_presentation(node: Any) -> Any:
    """Recursively drop presentation-only schema keys and sort mappings.

    Args:
        node (Any): JSON-schema fragment.

    Returns:
        Any: The fragment without keys in :data:`PRESENTATION_SCHEMA_KEYS`.
    """
    if isinstance(node, dict):
        return {k: strip_presentation(v) for k, v in sorted(node.items()) if k not in PRESENTATION_SCHEMA_KEYS}
    if isinstance(node, list):
        return [strip_presentation(v) for v in node]
    return node
