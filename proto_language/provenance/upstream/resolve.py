"""Resolve discovered upstream sources to immutable identifiers, recursing one level into upstream code.

For each source the resolver records what is *actually* behind the reference
today and whether the toolkit's reference is pinned:

* ``pypi``: release files with their SHA-256 from the PyPI JSON API. ``==``
  pins are ``pinned``; ranges resolve to the newest matching release and stay
  ``floating``.
* ``git``: ``git ls-remote`` maps the ref (or default branch) to a commit. Only
  a full 40-hex commit in the toolkit is ``pinned``. A repository referenced
  without a ref is matched to the tag of the toolkit's primary PyPI release
  when one exists (``v2.2.1`` for ``boltz==2.2.1``).
* ``hf-model`` / ``hf-dataset``: the Hub revision API returns the commit SHA and
  per-file LFS SHA-256. Honors ``HF_ENDPOINT`` for internal mirrors.
* ``url``: ``HEAD`` metadata (ETag, size). A URL becomes ``pinned`` only once
  ``mirror.py`` has downloaded it and recorded its SHA-256.

With ``recurse=True`` the resolver downloads each primary code package (wheel
or sdist from PyPI, shallow clone for git) and scans it with the same rules as
discovery, so weights fetched *inside* upstream code (Boltz-2 pulling its
checkpoint from the Hub, for example) are attributed to the toolkit. Archives
are read member by member and never extracted or executed.

Examples:
    >>> from proto_language.provenance.upstream.resolve import resolve
    >>> lock = resolve(lock, recurse=True)  # doctest: +SKIP
    >>> lock.summary()  # {'pinned': 120, 'floating': 180, 'unresolved': 30}
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import ssl
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from proto_language.provenance.canonical import sha256_bytes
from proto_language.provenance.upstream.discover import scan_text_for_sources
from proto_language.provenance.upstream.sources import UpstreamLock, UpstreamSource

__all__ = ["resolve"]

logger = logging.getLogger(__name__)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SCAN_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg")
# Upstream paths that hold tests, examples, docs, or benchmarks rather than the inference path.
_SKIP_UPSTREAM_PARTS = frozenset(
    {"test", "tests", "testing", "example", "examples", "doc", "docs", "notebooks", "benchmark", "benchmarks"}
)
_MAX_SCAN_BYTES = 1 << 20
_MAX_ARCHIVE_BYTES = 200 << 20
_USER_AGENT = "proto-language-provenance/1"


def _ssl_context() -> ssl.SSLContext:
    """TLS context honoring ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` (corporate proxies)."""
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()


def _open(url: str, method: str = "GET", timeout: float = 30.0) -> Any:
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Refusing non-HTTP URL: {url}")
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface" in urllib.parse.urlparse(url).netloc:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, method=method, headers=headers)  # noqa: S310 - scheme checked above
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())  # noqa: S310


def _get_json(url: str) -> Any:
    with _open(url) as response:
        return json.load(response)


def _get_bytes(url: str, limit: int = _MAX_ARCHIVE_BYTES) -> bytes:
    with _open(url, timeout=120.0) as response:
        data: bytes = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url} exceeds {limit} bytes; not scanning")
    return data


def _git(*args: str, cwd: Path | None = None, timeout: float = 120.0) -> str:
    result = subprocess.run(  # noqa: S603 - fixed binary, argument list
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout


# =============================================================================
# Per-type resolvers
# =============================================================================


class _PyPICache:
    """One JSON fetch per distribution, shared across toolkits."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def project(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = _get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
        return self._cache[name]


def _resolve_pypi(src: UpstreamSource, pypi: _PyPICache) -> None:
    project = pypi.project(src.locator)
    releases: dict[str, list[dict[str, Any]]] = project.get("releases", {})
    try:
        spec = SpecifierSet(src.ref or "")
    except InvalidSpecifier:
        src.resolved = {"error": f"invalid specifier {src.ref!r}"}
        return
    candidates: list[Version] = []
    for raw, files in releases.items():
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        allow_pre = bool(spec.prereleases)
        if files and not all(f.get("yanked") for f in files) and spec.contains(version, prereleases=allow_pre):
            candidates.append(version)
    if not candidates:
        src.status = "missing"
        src.resolved = {"error": "no release satisfies the specifier"}
        return
    chosen = max(candidates)
    files = (
        releases[str(chosen)]
        if str(chosen) in releases
        else next(v for k, v in releases.items() if Version(k) == chosen)
    )
    info = project.get("info", {})
    src.resolved = {
        "version": str(chosen),
        "files": sorted(
            (
                {
                    "filename": f["filename"],
                    "sha256": f["digests"]["sha256"],
                    "size": f.get("size"),
                    "packagetype": f.get("packagetype"),
                    "url": f["url"],
                }
                for f in files
                if not f.get("yanked")
            ),
            key=lambda f: f["filename"],
        ),
        "license": info.get("license_expression") or (info.get("license") or "")[:120] or None,
        "project_urls": {k: v for k, v in (info.get("project_urls") or {}).items() if "github" in (v or "")},
    }
    src.status = "pinned" if src.ref_is_immutable else "floating"


def _ls_remote(url: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in _git("ls-remote", "--tags", "--heads", "--symref", url, timeout=60.0).splitlines():
        if line.startswith("ref:"):
            continue
        sha, _, ref = line.partition("\t")
        refs[ref] = sha
    head = _git("ls-remote", url, "HEAD", timeout=60.0).split()
    if head:
        refs["HEAD"] = head[0]
    return refs


def _resolve_git(src: UpstreamSource, version_hints: list[str]) -> None:
    url = f"https://{src.locator}"
    refs = _ls_remote(url)
    ref = src.ref
    if ref and _SHA40.match(ref):
        src.resolved = {"commit": ref}
        src.status = "pinned"
        return
    if ref and re.fullmatch(r"[0-9a-f]{7,39}", ref):
        src.resolved = {"abbreviated_commit": ref, "note": "expand to a full 40-hex SHA"}
        src.status = "floating"
        return
    if ref:
        for candidate in (f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}", f"refs/heads/{ref}"):
            if candidate in refs:
                src.resolved = {"commit": refs[candidate], "ref": candidate.removesuffix("^{}")}
                src.status = "floating"
                return
        src.status = "missing"
        src.resolved = {"error": f"ref {ref!r} not found"}
        return
    resolved: dict[str, Any] = {"default_branch_commit": refs.get("HEAD")}
    for version in version_hints:
        for tag in (f"v{version}", version):
            for candidate in (f"refs/tags/{tag}^{{}}", f"refs/tags/{tag}"):
                if candidate in refs:
                    resolved["release_tag"] = tag
                    resolved["release_commit"] = refs[candidate]
                    break
            if "release_tag" in resolved:
                break
    src.resolved = resolved
    src.status = "floating"


def _hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _resolve_hf(src: UpstreamSource) -> None:
    kind = "datasets" if src.type == "hf-dataset" else "models"
    revision = src.ref or "main"
    data = _get_json(f"{_hf_endpoint()}/api/{kind}/{src.locator}/revision/{revision}?blobs=true")
    files = [
        {
            "path": s["rfilename"],
            "size": s.get("size"),
            "sha256": (s.get("lfs") or {}).get("sha256"),
            "git_blob": s.get("blobId"),
        }
        for s in data.get("siblings", [])
    ]
    src.resolved = {
        "commit": data.get("sha"),
        "gated": data.get("gated", False),
        "license": (data.get("cardData") or {}).get("license"),
        "files": sorted(files, key=lambda f: f["path"]),
    }
    src.status = "pinned" if src.ref_is_immutable else "floating"


def _resolve_url(src: UpstreamSource) -> None:
    if src.template:
        src.status = "unresolved"
        src.resolved = {**src.resolved, "error": "URL contains unexpanded variables; declare the concrete URL"}
        return
    if src.resolved.get("sha256"):
        src.status = "pinned"
        return
    with _open(src.locator, method="HEAD") as response:
        headers = response.headers
        src.resolved = {
            "final_url": response.geturl(),
            "etag": headers.get("ETag"),
            "size": int(headers["Content-Length"]) if headers.get("Content-Length") else None,
            "last_modified": headers.get("Last-Modified"),
        }
    src.status = "floating"


# =============================================================================
# Recursive scan of primary upstream code
# =============================================================================


def _scannable(name: str, size: int) -> bool:
    parts = set(name.lower().split("/")[:-1])
    return name.endswith(_SCAN_SUFFIXES) and size < _MAX_SCAN_BYTES and not parts & _SKIP_UPSTREAM_PARTS


def _iter_archive_texts(blob: bytes, filename: str) -> Iterator[tuple[str, str]]:
    """Yield ``(member, text)`` for scannable members without extracting anything."""
    if filename.endswith((".whl", ".zip")):
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for info in archive.infolist():
                if _scannable(info.filename, info.file_size):
                    yield info.filename, archive.read(info).decode("utf-8", errors="replace")
    elif filename.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and _scannable(member.name, member.size):
                    handle = archive.extractfile(member)
                    if handle is not None:
                        yield member.name, handle.read().decode("utf-8", errors="replace")


def _scan_pypi(src: UpstreamSource) -> list[UpstreamSource]:
    files = src.resolved.get("files") or []
    ranked = sorted(
        files,
        key=lambda f: (
            not f["filename"].endswith("-none-any.whl"),
            not f["filename"].endswith(".whl"),
            f["filename"],
        ),
    )
    if not ranked:
        return []
    chosen = ranked[0]
    blob = _get_bytes(chosen["url"])
    if sha256_bytes(blob) != chosen["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {chosen['filename']}")
    found: list[UpstreamSource] = []
    for member, text in _iter_archive_texts(blob, chosen["filename"]):
        found.extend(scan_text_for_sources(text, f"{src.id}!{member}", is_python=member.endswith(".py")))
    return found


def _scan_git(src: UpstreamSource, workdir: Path) -> list[UpstreamSource]:
    commit = src.resolved.get("commit") or src.resolved.get("release_commit")
    target = workdir / re.sub(r"[^\w.-]", "_", src.locator)
    _git("init", "-q", str(target))
    _git("-C", str(target), "remote", "add", "origin", f"https://{src.locator}")
    ref = commit or src.ref or "HEAD"
    try:
        _git("-C", str(target), "fetch", "-q", "--depth", "1", "--filter=blob:limit=1m", "origin", ref, timeout=300.0)
        _git("-C", str(target), "checkout", "-q", "FETCH_HEAD")
    except subprocess.CalledProcessError:
        if not re.fullmatch(r"[0-9a-f]{7,39}", ref):
            raise
        # Servers refuse to fetch abbreviated SHAs; fetch history without blobs and resolve locally.
        _git("-C", str(target), "fetch", "-q", "--filter=blob:none", "origin", timeout=600.0)
        _git("-C", str(target), "checkout", "-q", ref)
    full = _git("-C", str(target), "rev-parse", "HEAD").strip()
    src.resolved.setdefault("scanned_commit", full)
    if src.resolved.get("abbreviated_commit"):
        src.resolved["commit"] = full
    found: list[UpstreamSource] = []
    for path in sorted(target.rglob("*")):
        rel = path.relative_to(target).as_posix()
        if ".git" in path.parts or not path.is_file() or not _scannable(rel, path.stat().st_size):
            continue
        found.extend(
            scan_text_for_sources(path.read_text(errors="replace"), f"{src.id}!{rel}", is_python=rel.endswith(".py"))
        )
    return found


def _recurse(lock: UpstreamLock, workers: int) -> None:
    """Scan each toolkit's primary code sources and attach what they fetch."""
    owners_by_toolkit: dict[str, list[UpstreamSource]] = {}
    jobs: list[tuple[str, UpstreamSource]] = []
    for name, entry in lock.toolkits.items():
        for sid in entry.sources:
            src = lock.sources[sid]
            if src.role == "code" and src.type in ("pypi", "git") and src.status in ("pinned", "floating"):
                if src.found_in and all(loc.endswith("links.yaml") for loc in src.found_in):
                    continue  # a reference link, not code the toolkit installs or clones
                jobs.append((name, src))

    with tempfile.TemporaryDirectory(prefix="provenance-scan-") as tmp:
        workdir = Path(tmp)

        def run(job: tuple[str, UpstreamSource]) -> tuple[str, list[UpstreamSource]]:
            toolkit, src = job
            try:
                found = _scan_pypi(src) if src.type == "pypi" else _scan_git(src, workdir / toolkit)
            except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile, tarfile.TarError) as exc:
                logger.warning("Recursive scan of %s failed: %s", src.id, exc)
                return toolkit, []
            return toolkit, found

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for toolkit, found in pool.map(run, jobs):
                owners_by_toolkit.setdefault(toolkit, []).extend(found)

    for toolkit, found in owners_by_toolkit.items():
        for src in found:
            if src.type in ("hf-model", "hf-dataset", "url") and src.role != "code":
                src.notes = "found inside upstream code"
                lock.add(toolkit, src)


# =============================================================================
# Entry point
# =============================================================================


def _version_hints(lock: UpstreamLock) -> dict[str, list[str]]:
    """Map each git source to versions of primary PyPI packages in the same toolkits."""
    hints: dict[str, list[str]] = {}
    for entry in lock.toolkits.values():
        versions = [
            str(lock.sources[sid].resolved.get("version"))
            for sid in entry.sources
            if lock.sources[sid].type == "pypi" and lock.sources[sid].role == "code" and lock.sources[sid].resolved
        ]
        for sid in entry.sources:
            if lock.sources[sid].type == "git":
                hints.setdefault(sid, []).extend(v for v in versions if v != "None")
    return hints


def _run_all(
    sources: list[UpstreamSource], fn: Callable[[UpstreamSource], None], workers: int, offline_types: set[str]
) -> None:
    def guarded(src: UpstreamSource) -> None:
        if src.type in offline_types:
            return
        try:
            fn(src)
        except urllib.error.HTTPError as exc:
            src.status = "missing" if exc.code == 404 else "unresolved"
            src.resolved = {**src.resolved, "error": f"HTTP {exc.code}"}
        except (OSError, ValueError, subprocess.SubprocessError, KeyError) as exc:
            src.status = "unresolved"
            src.resolved = {**src.resolved, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(guarded, sources))


def resolve(
    lock: UpstreamLock,
    *,
    recurse: bool = True,
    workers: int = 16,
    skip_types: set[str] | None = None,
) -> UpstreamLock:
    """Resolve every source in ``lock`` in place and return it.

    Args:
        lock (UpstreamLock): Discovered (and declaration-merged) lock.
        recurse (bool): Scan primary code packages/repos for weights they fetch.
        workers (int): Concurrent network requests.
        skip_types (set[str] | None): Source types to leave unresolved (e.g. ``{"hf-model"}``
            when the Hub is unreachable from this network).

    Returns:
        UpstreamLock: The same lock, with ``status`` and ``resolved`` filled in.
    """
    skip = skip_types or set()
    pypi = _PyPICache()
    by_type: dict[str, list[UpstreamSource]] = {}
    for src in lock.sources.values():
        by_type.setdefault(src.type, []).append(src)

    _run_all(by_type.get("pypi", []), lambda s: _resolve_pypi(s, pypi), workers, skip)
    hints = _version_hints(lock)
    _run_all(by_type.get("git", []), lambda s: _resolve_git(s, hints.get(s.id, [])), workers, skip)

    if recurse:
        _recurse(lock, workers=max(1, workers // 2))
        by_type = {}
        for src in lock.sources.values():
            by_type.setdefault(src.type, []).append(src)

    for kind in ("hf-model", "hf-dataset"):
        _run_all(by_type.get(kind, []), _resolve_hf, workers, skip)
    _run_all(by_type.get("url", []), _resolve_url, workers, skip)
    logger.info("Resolved upstream sources: %s", lock.summary())
    return lock
