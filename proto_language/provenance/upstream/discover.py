"""Static discovery of upstream code, weights, and data referenced by proto-tools toolkits.

Walks every toolkit directory under ``proto_tools/tools/{category}/{toolkit}/``
(plus any shared environment it redirects to) without importing or executing
anything, and records each third-party source it can see:

* ``links.yaml``: ``github``, ``huggingface``, and weight/model links.
* ``standalone/requirements.txt`` and ``uv pip install`` lines in ``setup.sh``.
* ``git clone`` / ``git checkout``, ``curl`` / ``wget`` downloads, and inline
  ``hf_hub_download`` / ``snapshot_download`` calls in ``setup.sh``.
* Hugging Face repo ids and weight-like URLs in the toolkit's Python sources.

Discovery runs in two passes. Pass one collects explicit references and learns
the set of Hugging Face owners (``facebook``, ``biohub``...). Pass two matches
bare string literals such as ``"facebook/esm2_t33_650M_UR50D"`` against those
learned owners, which catches model ids that are only ever passed through
variables. The resolver (``resolve.py``) then recurses one level into the
primary packages and repositories to find weights fetched by upstream code.

Examples:
    >>> from pathlib import Path
    >>> from proto_language.provenance.upstream.discover import discover
    >>> lock = discover(Path("proto-tools"))
    >>> "boltz2" in lock.toolkits  # True
"""

from __future__ import annotations

import ast
import logging
import re
import shlex
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement

from proto_language.provenance.upstream.sources import (
    SourceRole,
    ToolkitUpstream,
    UpstreamLock,
    UpstreamSource,
    normalize_git_url,
)

__all__ = ["discover", "iter_toolkits", "scan_text_for_sources"]

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s'\"<>)`\]\\,]+")
HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/(?:(datasets)/)?([A-Za-z0-9][\w.-]*)/([\w.-]+)(?:/(?:resolve|tree|blob)/([^/]+))?"
)
HF_CALL_RE = re.compile(
    r"(?:hf_hub_download|snapshot_download|from_pretrained)\(\s*(?:repo_id\s*=\s*)?[\"']([A-Za-z0-9][\w.-]*/[\w.-]+)[\"']"
)
REPO_ID_KW_RE = re.compile(r"repo_id\s*=\s*[\"']([A-Za-z0-9][\w.-]*/[\w.-]+)[\"']")
REVISION_RE = re.compile(r"revision\s*=\s*[\"']([0-9a-f]{40})[\"']")
HF_ID_LITERAL_RE = re.compile(r"^([A-Za-z0-9][\w.-]*)/([\w.-]+)$")
SHELL_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s;#]+)")
SHELL_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
PY_FMT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

ARTIFACT_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".pkl",
    ".onnx",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".zip",
    ".gz",
    ".whl",
    ".jar",
    ".json.gz",
)
ARTIFACT_HOSTS = (
    "storage.googleapis.com",
    "dl.fbaipublicfiles.com",
    "zenodo.org",
    "figshare.com",
    "s3.amazonaws.com",
    "amazonaws.com",
    "dropbox.com",
    "drive.google.com",
    "ftp.ncbi.nlm.nih.gov",
)
# Hosts that are documentation or APIs called at run time, not artifacts to mirror.
IGNORED_URL_HOSTS = (
    "doi.org",
    "arxiv.org",
    "biorxiv.org",
    "www.biorxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "docs.python.org",
    "pydantic.dev",
    "localhost",
    "127.0.0.1",
)
# Hugging Face path segments that are not model repositories.
HF_NON_REPO_OWNERS = frozenset({"docs", "spaces", "collections", "settings", "join", "login", "api", "blog", "papers"})
SKIP_DIRS = frozenset({"__pycache__", "examples", "tests", ".ipynb_checkpoints"})
_PIP_FLAGS_WITH_VALUE = frozenset(
    {
        "-r",
        "--requirement",
        "--target",
        "-t",
        "--index-url",
        "-i",
        "--extra-index-url",
        "-c",
        "--constraint",
        "--index-strategy",
        "--find-links",
        "-f",
        "--python",
        "-p",
        "--prefix",
    }
)
LINK_WEIGHT_KEYS = ("weights", "model", "model_gcs", "zenodo", "figshare")


def iter_toolkits(proto_tools_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(category, toolkit, path)`` for every toolkit directory.

    Args:
        proto_tools_root (Path): proto-tools checkout root (contains ``proto_tools/``).

    Yields:
        Iterator[tuple[str, str, Path]]: Category name, toolkit name, toolkit directory.
    """
    tools_dir = proto_tools_root / "proto_tools" / "tools"
    for category_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        if category_dir.name == "testing":
            continue
        for toolkit_dir in sorted(p for p in category_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            if (toolkit_dir / "__init__.py").exists():
                yield category_dir.name, toolkit_dir.name, toolkit_dir


def _tokens(name: str) -> set[str]:
    """Alphabetic tokens of a name, digits stripped (``boltz2`` -> ``{'boltz'}``)."""
    return {t for t in re.split(r"[^a-z]+", re.sub(r"\d+", " ", name.lower())) if len(t) >= 3}


def _is_primary(dist_name: str, toolkit: str, repo_names: Iterable[str]) -> bool:
    """Whether a distribution looks like the toolkit's own model package rather than a generic dependency."""
    dist = _tokens(dist_name)
    anchors = _tokens(toolkit)
    for repo in repo_names:
        anchors |= _tokens(repo)
    return any(a in d or d in a for a in anchors for d in dist)


def _expand_shell(text: str, env: dict[str, str]) -> tuple[str, bool]:
    """Expand ``$VAR`` / ``${VAR:-default}`` from known assignments; report leftovers."""
    unresolved = False

    def repl(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(1) or match.group(3)
        default = match.group(2)
        if name in env:
            return env[name]
        if default is not None and "$" not in default:
            return default
        unresolved = True
        return match.group(0)

    return SHELL_VAR_RE.sub(repl, text), unresolved


def _url_role(url: str) -> SourceRole | None:
    """Classify a URL as an artifact worth tracking, or ``None`` to ignore it."""
    host = re.sub(r"^https?://", "", url).split("/", 1)[0].lower()
    if host in IGNORED_URL_HOSTS or host.endswith(("readthedocs.io", "readthedocs.org")):
        return None
    path = url.split("?", 1)[0].lower()
    if "github.com" in host and "/releases/download/" in path:
        return "code" if path.endswith(".whl") else "binary"
    if path.endswith(".whl"):
        return "code"
    if path.endswith(ARTIFACT_SUFFIXES) or any(host == h or host.endswith("." + h) for h in ARTIFACT_HOSTS):
        return "weights"
    return None


def _hf_from_url(url: str, found_in: str) -> UpstreamSource | None:
    """Parse a ``huggingface.co`` URL into a model/dataset source."""
    match = HF_URL_RE.match(url)
    if not match:
        return None
    datasets, owner, name, revision = match.groups()
    if owner in HF_NON_REPO_OWNERS or name in ("resolve", "tree", "blob"):
        return None
    rev = revision if revision and re.fullmatch(r"[0-9a-f]{40}", revision) else None
    return UpstreamSource.hf(f"{owner}/{name}", rev, dataset=bool(datasets), found_in=found_in)


def _parse_requirement(line: str, toolkit: str, repos: list[str], found_in: str) -> UpstreamSource | None:
    """Turn one requirements line (or ``uv pip install`` token) into a source."""
    line = line.split(" #", 1)[0].strip().strip("\"'")
    if not line or line.startswith(("#", "-")):
        return None
    direct = re.match(r"^(?:([A-Za-z0-9][\w.-]*)(?:\[[^\]]*\])?\s*@\s*)?(git\+\S+)$", line)
    if direct:
        url = direct.group(2)
        ref_match = re.search(r"\.git@([^#\s]+)|@([0-9a-f]{7,40})(?:#|$)", url)
        ref = (ref_match.group(1) or ref_match.group(2)) if ref_match else None
        base = re.sub(r"@[^@#/]+(#.*)?$", "", url.removeprefix("git+"))
        return UpstreamSource.git(base, ref, role="code", found_in=found_in)
    if line.startswith(("http://", "https://")):
        return UpstreamSource.url(line, role="code", found_in=found_in)
    try:
        req = Requirement(line)
    except InvalidRequirement:
        logger.debug("Skipping unparseable requirement %r in %s", line, found_in)
        return None
    if req.url:
        return UpstreamSource.url(req.url, role="code", found_in=found_in)
    role: SourceRole = "code" if _is_primary(req.name, toolkit, repos) else "dependency"
    return UpstreamSource.pypi(req.name, str(req.specifier), role=role, found_in=found_in)


def _parse(source: str) -> ast.Module | None:
    """Parse Python source, tolerating upstream files that do not parse or warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError):
            return None


def _module_constants(source: str) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` assignments, for expanding f-string URLs."""
    tree = _parse(source)
    if tree is None:
        return {}
    consts: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            consts[node.targets[0].id] = node.value.value
    return consts


def _string_literals(source: str) -> Iterator[tuple[int, str]]:
    """Yield ``(lineno, value)`` for every plain string constant in a Python module."""
    tree = _parse(source)
    if tree is None:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200:
            yield node.lineno, node.value


def scan_text_for_sources(text: str, rel: str, *, is_python: bool) -> list[UpstreamSource]:
    """Find Hugging Face repos and artifact URLs in one source file.

    Shared by toolkit discovery and the resolver's recursive scan of upstream
    packages, so both apply identical rules.

    Args:
        text (str): File contents.
        rel (str): Path used in ``found_in`` evidence.
        is_python (bool): Expand module-level constants inside f-string URLs.

    Returns:
        list[UpstreamSource]: Sources found, possibly with duplicates.
    """
    found: list[UpstreamSource] = []
    consts = _module_constants(text) if is_python else {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        where = f"{rel}:{lineno}"
        for match in HF_CALL_RE.finditer(line):
            rev = REVISION_RE.search(line)
            found.append(UpstreamSource.hf(match.group(1), rev.group(1) if rev else None, found_in=where))
        found.extend(UpstreamSource.hf(m.group(1), None, found_in=where) for m in REPO_ID_KW_RE.finditer(line))
        for raw in URL_RE.findall(line):
            url = raw.rstrip(".;:'\"")
            template = False
            if is_python and "{" in url:
                url = PY_FMT_RE.sub(lambda m: consts.get(m.group(1), m.group(0)), url)
                template = "{" in url
            if "huggingface.co" in url:
                hf = _hf_from_url(url, where)
                if hf is not None:
                    found.append(hf)
                continue
            role = _url_role(url)
            if role is not None:
                found.append(UpstreamSource.url(url, role=role, template=template, found_in=where))
    return found


class _ToolkitScan:
    """Accumulates sources for one toolkit across its files."""

    def __init__(self, proto_tools_root: Path, category: str, toolkit: str, path: Path) -> None:
        self.root = proto_tools_root
        self.category = category
        self.toolkit = toolkit
        self.path = path
        self.sources: list[UpstreamSource] = []
        self.repos: list[str] = []
        self.env_name: str | None = None
        self.license: dict[str, Any] = {}

    def rel(self, path: Path, lineno: int | None = None) -> str:
        rel = path.relative_to(self.root).as_posix()
        return f"{rel}:{lineno}" if lineno else rel

    def run(self) -> None:
        self._links()
        self._license()
        env_dirs = self._env_dirs()
        for env_dir in env_dirs:
            self._requirements(env_dir / "requirements.txt")
            for script in sorted(env_dir.glob("*.sh")):
                self._shell(script)
        for py in sorted(self._python_files()):
            self.sources.extend(scan_text_for_sources(py.read_text(errors="replace"), self.rel(py), is_python=True))

    def _python_files(self) -> Iterator[Path]:
        for py in self.path.rglob("*.py"):
            if not SKIP_DIRS.intersection(py.relative_to(self.path).parts):
                yield py

    def _links(self) -> None:
        links_path = self.path / "links.yaml"
        if not links_path.exists():
            return
        links = yaml.safe_load(links_path.read_text()) or {}
        where = self.rel(links_path)
        for url in _as_list(links.get("github")):
            if normalize_git_url(url).count("/") != 2:
                continue  # an organization page, not a repository
            src = UpstreamSource.git(url, None, role="code", found_in=where)
            self.repos.append(src.locator.rsplit("/", 1)[-1])
            self.sources.append(src)
        for url in _as_list(links.get("huggingface")):
            hf = _hf_from_url(url, where)
            if hf is not None:
                self.sources.append(hf)
        for key in LINK_WEIGHT_KEYS:
            for url in _as_list(links.get(key)):
                hf = _hf_from_url(url, where) if "huggingface.co" in url else None
                self.sources.append(hf or UpstreamSource.url(url, role="weights", found_in=where))

    def _license(self) -> None:
        license_path = self.path / "license.yaml"
        if license_path.exists():
            self.license = yaml.safe_load(license_path.read_text()) or {}

    def _env_dirs(self) -> list[Path]:
        standalone = self.path / "standalone"
        dirs: list[Path] = []
        shared = standalone / "shared_env.txt"
        if shared.exists():
            name = shared.read_text().strip().splitlines()[0].strip()
            shared_dir = self.root / "proto_tools" / "shared_envs" / name
            if shared_dir.is_dir():
                dirs.append(shared_dir)
                self.env_name = name
        if standalone.is_dir():
            dirs.append(standalone)
            self.env_name = self.env_name or self.toolkit
        return dirs

    def _requirements(self, path: Path) -> None:
        if not path.exists():
            return
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            src = _parse_requirement(line, self.toolkit, self.repos, self.rel(path, lineno))
            if src is not None:
                self.sources.append(src)

    def _shell(self, path: Path) -> None:
        env: dict[str, str] = {}
        last_clone: UpstreamSource | None = None
        text = path.read_text(errors="replace").replace("\\\n", " ")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", "echo", "printf")):
                continue
            assign = SHELL_ASSIGN_RE.match(raw)
            if assign:
                value, _ = _expand_shell(assign.group(2).strip("\"'"), env)
                env[assign.group(1)] = value
            line, unresolved = _expand_shell(raw, env)
            where = self.rel(path, lineno)
            if "git clone" in line:
                last_clone = self._git_clone(line, where) or last_clone
                continue
            checkout = re.search(r"git (?:-C \S+ )?(?:checkout|reset --hard) ([0-9a-f]{7,40}|v?[\w.-]+)", line)
            if checkout and last_clone is not None:
                pinned = UpstreamSource.git(f"https://{last_clone.locator}", checkout.group(1), found_in=where)
                self.sources = [s for s in self.sources if s is not last_clone]
                self.sources.append(pinned)
                last_clone = pinned
                continue
            if re.search(r"\buv pip install\b|\bpip install\b", line):
                self._pip_tokens(line, where)
            for src in scan_text_for_sources(line, where.rsplit(":", 1)[0], is_python=False):
                src.found_in = [where]
                src.template = src.template or (unresolved and "$" in src.locator)
                self.sources.append(src)

    def _git_clone(self, line: str, where: str) -> UpstreamSource | None:
        try:
            parts = shlex.split(line[line.index("git clone") :], comments=True)
        except ValueError:
            parts = line[line.index("git clone") :].split()
        ref = None
        url = None
        it = iter(parts[2:])
        for tok in it:
            if tok in ("--branch", "-b"):
                ref = next(it, None)
            elif tok in ("--depth", "--filter", "--origin", "-o", "--reference"):
                next(it, None)
            elif tok.startswith("-"):
                continue
            elif url is None:
                url = tok
        if not url or "$" in url:
            return None
        src = UpstreamSource.git(url, ref, role="code", found_in=where)
        self.repos.append(src.locator.rsplit("/", 1)[-1])
        self.sources.append(src)
        return src

    def _pip_tokens(self, line: str, where: str) -> None:
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            return
        start = next((i for i, t in enumerate(tokens) if t == "install"), None)
        if start is None:
            return
        skip_next = False
        for tok in tokens[start + 1 :]:
            if skip_next:
                skip_next = False
                continue
            if tok in _PIP_FLAGS_WITH_VALUE:
                skip_next = True
                continue
            if tok.startswith("-") or tok in ("&&", "||", ";", "|") or "$" in tok:
                continue
            src = _parse_requirement(tok, self.toolkit, self.repos, where)
            if src is not None:
                self.sources.append(src)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _learned_owner_literals(scan: _ToolkitScan, owners: set[str]) -> Iterator[UpstreamSource]:
    """Pass two: bare ``owner/name`` literals whose owner was seen in an explicit HF reference."""
    for py in sorted(scan._python_files()):
        for lineno, value in _string_literals(py.read_text(errors="replace")):
            match = HF_ID_LITERAL_RE.match(value)
            if match and match.group(1) in owners and not value.endswith((".py", ".json", ".yaml", ".txt")):
                yield UpstreamSource.hf(value, None, found_in=scan.rel(py, lineno))


def discover(proto_tools_root: Path, toolkits: Iterable[str] | None = None) -> UpstreamLock:
    """Scan proto-tools toolkits and return an unresolved upstream lock.

    Args:
        proto_tools_root (Path): proto-tools checkout root.
        toolkits (Iterable[str] | None): Restrict to these toolkit names.

    Returns:
        UpstreamLock: Every toolkit with its discovered sources (status ``unresolved``).
    """
    wanted = set(toolkits) if toolkits else None
    lock = UpstreamLock()
    scans: list[_ToolkitScan] = []
    for category, toolkit, path in iter_toolkits(proto_tools_root):
        if wanted is not None and toolkit not in wanted:
            continue
        scan = _ToolkitScan(proto_tools_root, category, toolkit, path)
        scan.run()
        scans.append(scan)

    owners = {s.locator.split("/", 1)[0] for scan in scans for s in scan.sources if s.type.startswith("hf")}
    for scan in scans:
        scan.sources.extend(_learned_owner_literals(scan, owners))
        lock.toolkits[scan.toolkit] = ToolkitUpstream(
            toolkit=scan.toolkit, category=scan.category, env_name=scan.env_name, license=scan.license
        )
        for src in scan.sources:
            lock.add(scan.toolkit, src)
    _collapse_unpinned_duplicates(lock)
    logger.info("Discovered %d sources across %d toolkits", len(lock.sources), len(lock.toolkits))
    return lock


def _collapse_unpinned_duplicates(lock: UpstreamLock) -> None:
    """Drop ``hf:x`` when ``hf:x@<sha>`` (or ``git:x`` when ``git:x@ref``) is also referenced by the same toolkit."""
    for entry in lock.toolkits.values():
        refs = {sid.split("@", 1)[0] for sid in entry.sources if "@" in sid.split(":", 1)[1]}
        keep = [sid for sid in entry.sources if not ("@" not in sid.split(":", 1)[1] and sid in refs)]
        entry.sources = keep
    used = {sid for entry in lock.toolkits.values() for sid in entry.sources}
    for sid in list(lock.sources):
        if sid not in used:
            del lock.sources[sid]
