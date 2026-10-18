"""Mirror upstream code and weights into infrastructure we control.

Two kinds of mirror, matching the two kinds of upstream artifact:

* **Git repositories** are mirrored with ``git clone --mirror`` and pushed, all
  refs included, to a remote built from ``git_remote_template`` (e.g. one repo
  per upstream under our GitHub org). Forks we modify live on branches there,
  so upstream history and our changes share one repository.
* **Files** (URL downloads, Hugging Face files, PyPI distributions) go to a
  content-addressed store (CAS) at ``{blob_store}/sha256/{aa}/{bb}/{sha256}``.
  The SHA-256 is recorded in the upstream lock, which turns a mutable URL into
  a ``pinned`` source. Postgres stores only metadata; bytes live in the CAS.

Blob stores: ``file:///abs/path`` is handled natively; ``s3://`` and ``gs://``
shell out to ``aws s3 cp`` / ``gcloud storage cp`` so credentials come from the
CI runner's standard configuration.

Examples:
    >>> from proto_language.provenance.upstream.mirror import MirrorConfig, plan
    >>> actions = plan(lock, MirrorConfig(blob_store="file:///data/cas"))  # doctest: +SKIP
    >>> actions[0].kind  # 'git' or 'blob'
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from proto_language.provenance.upstream.resolve import _hf_endpoint, _open
from proto_language.provenance.upstream.sources import UpstreamLock, UpstreamSource

__all__ = ["MirrorAction", "MirrorConfig", "cas_uri", "execute", "plan"]

logger = logging.getLogger(__name__)


class MirrorConfig(BaseModel):
    """Where mirrors go.

    Attributes:
        git_remote_template (str | None): Push target for git mirrors; ``{owner}`` and
            ``{repo}`` are substituted. ``None`` disables git mirroring.
        blob_store (str | None): CAS root (``file://``, ``s3://``, or ``gs://``). ``None``
            disables file mirroring.
        max_blob_bytes (int): Skip files larger than this (guards accidental 100 GB pulls).
    """

    model_config = ConfigDict(extra="forbid")

    git_remote_template: str | None = Field(default=None, description="Push target template for git mirrors.")
    blob_store: str | None = Field(default=None, description="Content-addressed store root.")
    max_blob_bytes: int = Field(default=50 << 30, description="Skip files larger than this many bytes.")


class MirrorAction(BaseModel):
    """One unit of mirroring work.

    Attributes:
        kind (Literal["git", "blob"]): Repository mirror or file mirror.
        source_id (str): Upstream source this action belongs to.
        url (str): Where to fetch from.
        target (str | None): Push remote (git) or file path within the source (blob).
        expected_sha256 (str | None): Hash to verify against, when the upstream publishes one.
        size (int | None): Expected size in bytes, when known.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["git", "blob"] = Field(description="Repository mirror or file mirror.")
    source_id: str = Field(description="Upstream source this action belongs to.")
    url: str = Field(description="Where to fetch from.")
    target: str | None = Field(default=None, description="Push remote (git) or file path (blob).")
    expected_sha256: str | None = Field(default=None, description="Hash to verify against.")
    size: int | None = Field(default=None, description="Expected size in bytes.")


def cas_uri(store: str, sha256: str) -> str:
    """Content-addressed location for a blob.

    Args:
        store (str): CAS root URI.
        sha256 (str): Hex digest.

    Returns:
        str: ``{store}/sha256/{aa}/{bb}/{sha256}``.
    """
    return f"{store.rstrip('/')}/sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"


def plan(lock: UpstreamLock, config: MirrorConfig, toolkits: set[str] | None = None) -> list[MirrorAction]:
    """List the mirror actions needed for ``lock`` (or a subset of toolkits).

    Already-mirrored files (``resolved.mirror`` set) are skipped, so the plan
    is the remaining work.

    Args:
        lock (UpstreamLock): Resolved upstream lock.
        config (MirrorConfig): Mirror targets.
        toolkits (set[str] | None): Restrict to these toolkits.

    Returns:
        list[MirrorAction]: Actions, sorted for stable output.
    """
    wanted = {sid for name, e in lock.toolkits.items() if toolkits is None or name in toolkits for sid in e.sources}
    actions: list[MirrorAction] = []
    for sid in sorted(wanted):
        src = lock.sources[sid]
        if src.type == "git" and config.git_remote_template and src.locator.count("/") == 2:
            _host, owner, repo = src.locator.split("/")
            target = config.git_remote_template.format(owner=owner, repo=repo)
            actions.append(MirrorAction(kind="git", source_id=sid, url=f"https://{src.locator}", target=target))
        elif config.blob_store:
            actions.extend(_blob_actions(src, config))
    return actions


def _blob_actions(src: UpstreamSource, config: MirrorConfig) -> list[MirrorAction]:
    if src.type == "url" and not src.template and not src.resolved.get("mirror"):
        return [MirrorAction(kind="blob", source_id=src.id, url=src.locator, size=src.resolved.get("size"))]
    if src.type in ("hf-model", "hf-dataset") and src.resolved.get("commit"):
        prefix = "datasets/" if src.type == "hf-dataset" else ""
        out = []
        for f in src.resolved.get("files") or []:
            if f.get("mirror") or (f.get("size") or 0) > config.max_blob_bytes:
                continue
            url = f"{_hf_endpoint()}/{prefix}{src.locator}/resolve/{src.resolved['commit']}/{f['path']}"
            out.append(
                MirrorAction(
                    kind="blob",
                    source_id=src.id,
                    url=url,
                    target=f["path"],
                    expected_sha256=f.get("sha256"),
                    size=f.get("size"),
                )
            )
        return out
    if src.type == "pypi" and src.role == "code":
        return [
            MirrorAction(kind="blob", source_id=src.id, url=f["url"], target=f["filename"], expected_sha256=f["sha256"])
            for f in src.resolved.get("files") or []
            if not f.get("mirror")
        ]
    return []


def _run(*args: str, cwd: Path | None = None, timeout: float = 3600.0) -> None:
    subprocess.run(list(args), cwd=cwd, check=True, timeout=timeout, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})  # noqa: S603


def _upload(local: Path, uri: str) -> None:
    if uri.startswith("file://"):
        dest = Path(uri.removeprefix("file://"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(local, dest)
    elif uri.startswith("s3://"):
        _run("aws", "s3", "cp", "--only-show-errors", str(local), uri)
    elif uri.startswith("gs://"):
        _run("gcloud", "storage", "cp", str(local), uri)
    else:
        raise ValueError(f"Unsupported blob store URI: {uri}")


def _download(url: str, dest: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with _open(url, timeout=600.0) as response, dest.open("wb") as handle:
        while chunk := response.read(1 << 20):
            hasher.update(chunk)
            handle.write(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _mirror_blob(action: MirrorAction, config: MirrorConfig, workdir: Path) -> tuple[str, int, str]:
    assert config.blob_store is not None  # noqa: S101 -- plan() only emits blob actions with a store
    local = workdir / "blob"
    sha, size = _download(action.url, local)
    if action.expected_sha256 and sha != action.expected_sha256:
        raise ValueError(f"{action.url}: SHA-256 {sha} does not match upstream-published {action.expected_sha256}")
    uri = cas_uri(config.blob_store, sha)
    _upload(local, uri)
    local.unlink()
    return sha, size, uri


def _mirror_git(action: MirrorAction, workdir: Path) -> None:
    assert action.target is not None  # noqa: S101 -- plan() always sets a git push target
    bare = workdir / "repo.git"
    _run("git", "clone", "--mirror", "--quiet", action.url, str(bare))
    _run("git", "-C", str(bare), "push", "--mirror", "--quiet", action.target)
    shutil.rmtree(bare)


def execute(lock: UpstreamLock, actions: list[MirrorAction], config: MirrorConfig) -> list[str]:
    """Run mirror actions and record the results in ``lock`` (in place).

    Files are verified against upstream-published hashes before upload. On
    success the source's ``resolved`` gains ``sha256``/``mirror`` (URLs) or a
    per-file ``mirror`` (HF, PyPI), and URL sources become ``pinned``.

    Args:
        lock (UpstreamLock): Lock to update.
        actions (list[MirrorAction]): Output of :func:`plan`.
        config (MirrorConfig): Mirror targets.

    Returns:
        list[str]: Failures, one line each; empty when everything mirrored.
    """
    failures: list[str] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for action in actions:
        src = lock.sources[action.source_id]
        with tempfile.TemporaryDirectory(prefix="provenance-mirror-") as tmp:
            try:
                if action.kind == "git":
                    _mirror_git(action, Path(tmp))
                    src.resolved["git_mirror"] = action.target
                    continue
                sha, size, uri = _mirror_blob(action, config, Path(tmp))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                failures.append(f"{action.source_id} ({action.target or action.url}): {exc}")
                continue
        if src.type == "url":
            src.resolved.update({"sha256": sha, "size": size, "mirror": uri, "mirrored_at": now})
            src.status = "pinned"
        else:
            key = "path" if src.type.startswith("hf") else "filename"
            for f in src.resolved.get("files") or []:
                if f.get(key) == action.target:
                    f.update({"sha256": f.get("sha256") or sha, "mirror": uri})
        logger.info("Mirrored %s -> %s", action.url, uri)
    return failures
