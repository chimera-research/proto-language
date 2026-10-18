"""Collect registry artifacts from the live registries, the upstream lock, and program files.

Builds one :class:`~proto_language.provenance.models.Artifact` per framework,
upstream source, tool, generator, constraint, optimizer, and published program,
wiring dependency edges so digests form a Merkle DAG::

    upstream ─▶ tool ─▶ generator / constraint ─▶ program ◀─ optimizer
                 ▲              ▲                                ▲
    proto-tools framework   proto-language framework ────────────┘

Edges come from declared metadata: a tool depends on its toolkit's upstream
sources, a component on its ``tools_called``, a program on the components its
stages reference. The *edge audit* cross-checks ``tools_called`` against the
proto-tools names each component's code actually imports.

Examples:
    >>> from proto_language.provenance.collect import build_lock
    >>> lock = build_lock(repo_root=Path("."))  # doctest: +SKIP
    >>> lock.index()[("constraint", "gc-content")].digest  # 'sha256:...'
"""

from __future__ import annotations

import ast
import functools
import glob
import inspect
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pydantic
import yaml
from pydantic import BaseModel

from proto_language.provenance.canonical import files_digest, sha256_file
from proto_language.provenance.models import ARTIFACT_KINDS, Artifact, ArtifactKind, Dependency, Lock, SourceInfo
from proto_language.provenance.upstream.sources import UpstreamLock

__all__ = ["RegistryConfig", "build_lock", "load_config"]

logger = logging.getLogger(__name__)

PL_FRAMEWORK_KEY = "proto-language"
PT_FRAMEWORK_KEY = "proto-tools"

# Shared framework code. A change here re-versions every component (or tool):
# the honest answer, since base classes and registries shape every run.
_PL_FRAMEWORK_GLOBS = (
    "proto_language/core/*.py",
    "proto_language/utils/*.py",
    "proto_language/optimizer/constraint_compiler/**/*.py",
    "proto_language/constraint/constraint_registry.py",
    "proto_language/generator/generator_registry.py",
    "proto_language/optimizer/optimizer_registry.py",
)
_PT_FRAMEWORK_GLOBS = (
    "proto_tools/utils/**/*.py",
    "proto_tools/utils/standalone_helpers_source/**/*",
    "proto_tools/tools/tool_registry.py",
    "proto_tools/entities/**/*.py",
    "proto_tools/transforms/**/*.py",
)


class RegistryConfig(BaseModel):
    """``registry/config.yaml``: what a release contains and how strict the build is.

    Attributes:
        programs (list[str]): Repo-relative globs of declarative program JSON files to publish.
        strict_programs (bool): Fail the build when a program references an unknown key or
            an invalid config (otherwise warn and skip the program).
        require_pinned_upstream (bool): Fail when a tool's upstream sources are not all pinned.
        mirror (dict[str, Any]): ``MirrorConfig`` fields (git remote template, blob store).
    """

    programs: list[str] = pydantic.Field(default_factory=list, description="Program JSON globs.")
    strict_programs: bool = pydantic.Field(default=True, description="Fail on invalid programs.")
    require_pinned_upstream: bool = pydantic.Field(default=False, description="Fail on unpinned upstream.")
    mirror: dict[str, Any] = pydantic.Field(default_factory=dict, description="MirrorConfig fields.")


def load_config(path: Path) -> RegistryConfig:
    """Load ``registry/config.yaml`` (defaults when absent).

    Args:
        path (Path): Config file path.

    Returns:
        RegistryConfig: Parsed config.
    """
    if not path.exists():
        return RegistryConfig()
    return RegistryConfig.model_validate(yaml.safe_load(path.read_text()) or {})


# =============================================================================
# Source info
# =============================================================================


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def proto_tools_root() -> Path:
    """Checkout root of the imported ``proto_tools`` package.

    Returns:
        Path: Directory containing the ``proto_tools`` package.
    """
    import proto_tools

    assert proto_tools.__file__ is not None, "proto_tools is a namespace package; reinstall it"  # noqa: S101 -- type narrowing
    return Path(proto_tools.__file__).resolve().parents[1]


def source_info(repo_root: Path) -> SourceInfo:
    """Describe the checkouts a lock is built from.

    Args:
        repo_root (Path): proto-language checkout root.

    Returns:
        SourceInfo: Commits, dirty flags, and toolchain versions.
    """
    pt_root = proto_tools_root()
    pinned = _git(repo_root, "ls-tree", "HEAD", "proto-tools")
    return SourceInfo(
        proto_language_commit=_git(repo_root, "rev-parse", "HEAD"),
        proto_language_dirty=bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no", "--", ".")),
        proto_tools_commit=_git(pt_root, "rev-parse", "HEAD"),
        proto_tools_dirty=bool(_git(pt_root, "status", "--porcelain", "--untracked-files=no")),
        proto_tools_pinned_commit=pinned.split()[2] if pinned and len(pinned.split()) >= 3 else None,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        pydantic_version=pydantic.VERSION,
    )


# =============================================================================
# Framework
# =============================================================================


def _glob_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(str(root / pattern), recursive=True):
            path = Path(match)
            if path.is_file() and "__pycache__" not in path.parts and path.name != "__init__.py":
                files.add(path.resolve())
    return sorted(files)


@functools.cache
def _pl_framework_files(repo_root: Path) -> frozenset[Path]:
    return frozenset(_glob_files(repo_root, _PL_FRAMEWORK_GLOBS))


def _framework_artifacts(repo_root: Path, pt_root: Path) -> dict[str, Artifact]:
    pl_files = _glob_files(repo_root, _PL_FRAMEWORK_GLOBS)
    pt_files = _glob_files(pt_root, _PT_FRAMEWORK_GLOBS)
    return {
        PL_FRAMEWORK_KEY: Artifact.build(
            kind="framework",
            key=PL_FRAMEWORK_KEY,
            record={"files": len(pl_files), "code_digest": files_digest(pl_files, repo_root)},
            deps=[],
        ),
        PT_FRAMEWORK_KEY: Artifact.build(
            kind="framework",
            key=PT_FRAMEWORK_KEY,
            record={"files": len(pt_files), "code_digest": files_digest(pt_files, pt_root)},
            deps=[],
        ),
    }


# =============================================================================
# Upstream
# =============================================================================


def _upstream_artifacts(upstream: UpstreamLock | None) -> dict[str, Artifact]:
    if upstream is None:
        return {}
    out: dict[str, Artifact] = {}
    for sid, src in upstream.sources.items():
        record = src.model_dump(mode="json", exclude={"found_in", "notes", "id"})
        out[sid] = Artifact.build(kind="upstream", key=sid, record=record, deps=[])
    return out


# =============================================================================
# Tools
# =============================================================================


def _yaml_or_empty(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.exists() else None


def _tool_artifacts(
    framework: Artifact, upstream_arts: dict[str, Artifact], upstream: UpstreamLock | None, warnings: list[str]
) -> dict[str, Artifact]:
    from proto_tools.modal.fingerprint import _defining_files, fingerprint
    from proto_tools.tools import ToolRegistry
    from proto_tools.utils.tool_instance import ToolInstance

    pt_root = proto_tools_root()
    out: dict[str, Artifact] = {}
    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        toolkit_dir = Path(spec.source_file).resolve().parent
        toolkit = toolkit_dir.name
        try:
            env_dir, env_name = ToolInstance._resolve_env_def(toolkit)
            env_files = [p for p in Path(env_dir).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
            env = {"name": env_name, "digest": files_digest(env_files, Path(env_dir))}
        except ValueError:
            env = {"name": None, "digest": None}
        record: dict[str, Any] = {
            "label": spec.label,
            "category": spec.category,
            "toolkit": toolkit,
            "description": spec.description,
            "uses_gpu": spec.uses_gpu,
            "gpu_only": spec.gpu_only,
            "device_count": spec.device_count,
            "cacheable": spec.cacheable,
            "stochastic": spec.stochastic,
            "iterable_input_fields": spec.iterable_input_fields,
            "iterable_output_field": spec.iterable_output_field,
            "schemas": {
                "input": spec.input_model.model_json_schema(),
                "config": spec.config_model.model_json_schema(),
                "output": spec.output_model.model_json_schema(),
            },
            "code_digest": files_digest(_defining_files(spec.key), pt_root),
            "env": env,
            "license": _yaml_or_empty(toolkit_dir / "license.yaml"),
            "links": _yaml_or_empty(toolkit_dir / "links.yaml"),
            "citation_sha256": sha256_file(toolkit_dir / "cite.bib") if (toolkit_dir / "cite.bib").exists() else None,
            "modal_fingerprint": fingerprint(spec.key).model_dump(),
        }
        deps = [framework.as_dependency("framework")]
        tk_entry = upstream.toolkits.get(toolkit) if upstream else None
        if upstream is not None and tk_entry is None and env["name"] is not None:
            warnings.append(f"tool {spec.key}: toolkit {toolkit!r} missing from the upstream lock")
        deps.extend(upstream_arts[sid].as_dependency("upstream") for sid in (tk_entry.sources if tk_entry else []))
        out[spec.key] = Artifact.build(kind="tool", key=spec.key, record=record, deps=deps)
    return out


# =============================================================================
# Components
# =============================================================================


def _source_file(obj: Any) -> Path | None:
    try:
        path = inspect.getsourcefile(obj)
    except TypeError:
        return None
    return Path(path).resolve() if path else None


def _module_file(module: str) -> Path | None:
    mod = sys.modules.get(module)
    file = getattr(mod, "__file__", None)
    return Path(file).resolve() if file else None


def _import_closure(seeds: Iterable[Path], repo_root: Path) -> list[Path]:
    """First-party, non-framework files reachable from ``seeds`` through ``proto_language`` imports."""
    pkg_root = (repo_root / "proto_language").resolve()
    framework = _pl_framework_files(repo_root)
    seen: set[Path] = set()
    stack = [p for p in seeds if p.is_relative_to(pkg_root)]
    while stack:
        path = stack.pop()
        if path in seen or path in framework or path.name == "__init__.py":
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("proto_language"):
                modules = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names if a.name.startswith("proto_language")]
            for module in modules:
                file = _module_file(module)
                if file is not None and file.is_relative_to(pkg_root):
                    stack.append(file)
    return sorted(seen)


def _proto_tools_names(files: Iterable[Path]) -> set[str]:
    """Names each file imports from ``proto_tools`` (``from proto_tools... import X``)."""
    names: set[str] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("proto_tools"):
                names.update(alias.name for alias in node.names)
    return names


def _tool_name_index() -> dict[str, set[str]]:
    """Map run-function and model class names to the tool keys that own them."""
    from proto_tools.tools import ToolRegistry

    index: dict[str, set[str]] = {}
    for spec in ToolRegistry.list_all():
        for name in (spec.function.__name__, spec.input_model.__name__, spec.config_model.__name__):
            index.setdefault(name, set()).add(spec.key)
    return index


def _component_artifacts(
    repo_root: Path,
    framework: Artifact,
    tools: dict[str, Artifact],
    audit: list[str],
) -> dict[tuple[str, str], Artifact]:
    from proto_language.constraint.constraint_registry import ConstraintRegistry
    from proto_language.generator.generator_registry import GeneratorRegistry
    from proto_language.optimizer.optimizer_registry import OptimizerRegistry

    name_index = _tool_name_index()
    out: dict[tuple[str, str], Artifact] = {}
    registries: list[tuple[ArtifactKind, Any, tuple[str, ...]]] = [
        ("generator", GeneratorRegistry, ("generator_class", "config_model")),
        ("constraint", ConstraintRegistry, ("function", "backward", "config_model", "backward_config_model")),
        ("optimizer", OptimizerRegistry, ("optimizer_class", "config_model")),
    ]
    for kind, registry, code_attrs in registries:
        for spec in sorted(registry.list_all(), key=lambda s: s.key):
            seeds: list[Path] = []
            for attr in code_attrs:
                obj = getattr(spec, attr, None)
                if obj is None:
                    continue
                for klass in getattr(obj, "__mro__", (obj,)):
                    path = _source_file(klass)
                    if path is not None:
                        seeds.append(path)
            files = _import_closure(seeds, repo_root)
            record = spec.model_dump(mode="json")
            record["code_digest"] = files_digest(files, repo_root)
            record["code_files"] = [p.relative_to(repo_root).as_posix() for p in files]
            declared = list(getattr(spec, "tools_called", []) or [])
            deps = [framework.as_dependency("framework")]
            for tool_key in declared:
                if tool_key not in tools:
                    raise ValueError(f"{kind} {spec.key}: tools_called references unknown tool {tool_key!r}")
                deps.append(tools[tool_key].as_dependency("tools_called"))
            implied = {
                next(iter(keys)) for name in _proto_tools_names(files) if len(keys := name_index.get(name, set())) == 1
            }
            # Over-approximate: depend on every tool the component's code imports, not only the
            # declared ones, so a tool upgrade can never fail to re-version a component that uses it.
            # Imports are module-granular (several components can share a module), hence advisory.
            undeclared = sorted(implied - set(declared))
            deps.extend(tools[tool_key].as_dependency("imports") for tool_key in undeclared)
            if undeclared:
                audit.append(f"{kind} {spec.key}: imports {undeclared} without declaring them in tools_called")
            out[(kind, spec.key)] = Artifact.build(kind=kind, key=spec.key, record=record, deps=deps)
    return out


# =============================================================================
# Programs
# =============================================================================


def program_refs(program: dict[str, Any]) -> list[tuple[ArtifactKind, str, dict[str, Any]]]:
    refs: list[tuple[ArtifactKind, str, dict[str, Any]]] = []
    for stage in program.get("optimization_stages", []):
        refs.extend(("generator", g["key"], g.get("config") or {}) for g in stage.get("generators", []))
        refs.extend(("constraint", c["key"], c.get("config") or {}) for c in stage.get("constraints", []))
        opt = stage.get("optimizer") or {}
        if "method" in opt:
            refs.append(("optimizer", opt["method"], opt.get("config") or {}))
    return refs


def _program_artifacts(
    repo_root: Path,
    config: RegistryConfig,
    components: dict[tuple[str, str], Artifact],
    warnings: list[str],
) -> dict[str, Artifact]:
    from proto_language.constraint.constraint_registry import ConstraintRegistry
    from proto_language.generator.generator_registry import GeneratorRegistry
    from proto_language.optimizer.optimizer_registry import OptimizerRegistry

    registries: dict[str, Any] = {
        "generator": GeneratorRegistry,
        "constraint": ConstraintRegistry,
        "optimizer": OptimizerRegistry,
    }
    out: dict[str, Artifact] = {}
    origins: dict[str, str] = {}
    paths = sorted({Path(p) for pattern in config.programs for p in glob.glob(str(repo_root / pattern))})
    for path in paths:
        rel = path.relative_to(repo_root).as_posix()
        program = json.loads(path.read_text())
        key = str(program.get("name") or path.stem)
        if key in origins:
            raise ValueError(f"Program name {key!r} is used by both {origins[key]} and {rel}")
        problems: list[str] = []
        deps: list[Dependency] = []
        for kind, ref_key, ref_config in program_refs(program):
            art = components.get((kind, ref_key))
            if art is None:
                problems.append(f"unknown {kind} {ref_key!r}")
                continue
            try:
                registries[kind].get(ref_key).config_model.model_validate(ref_config)
            except pydantic.ValidationError as exc:
                err = exc.errors()[0]
                problems.append(f"{kind} {ref_key!r} config invalid at {'.'.join(map(str, err['loc']))}: {err['msg']}")
            deps.append(art.as_dependency(kind))
        if problems:
            message = f"program {key} ({rel}): " + "; ".join(problems)
            if config.strict_programs:
                raise ValueError(message)
            warnings.append(message + " (skipped)")
            continue
        origins[key] = rel
        out[key] = Artifact.build(kind="program", key=key, record={"program": program}, deps=deps)
    return out


# =============================================================================
# Entry point
# =============================================================================


def build_lock(repo_root: Path, config: RegistryConfig | None = None, upstream: UpstreamLock | None = None) -> Lock:
    """Build the complete, deterministic release lock.

    Args:
        repo_root (Path): proto-language checkout root.
        config (RegistryConfig | None): Release configuration; defaults to ``registry/config.yaml``.
        upstream (UpstreamLock | None): Upstream lock; defaults to ``registry/upstream/upstream.lock.json``.

    Returns:
        Lock: Every artifact with Merkle digests, plus warnings.

    Raises:
        ValueError: On broken edges, invalid programs (strict mode), or unpinned upstream
            sources when ``require_pinned_upstream`` is set.
    """
    repo_root = repo_root.resolve()
    config = config or load_config(repo_root / "registry" / "config.yaml")
    if upstream is None:
        lock_path = repo_root / "registry" / "upstream" / "upstream.lock.json"
        upstream = UpstreamLock.model_validate_json(lock_path.read_text()) if lock_path.exists() else None
    warnings: list[str] = []
    if upstream is None:
        warnings.append("no upstream lock found; tools carry no upstream edges")
    elif config.require_pinned_upstream:
        unpinned = sorted(sid for sid, s in upstream.sources.items() if s.status != "pinned" and s.role != "dependency")
        if unpinned:
            raise ValueError(f"{len(unpinned)} upstream sources are not pinned, e.g. {unpinned[:5]}")

    source = source_info(repo_root)
    if source.proto_tools_pinned_commit and source.proto_tools_commit != source.proto_tools_pinned_commit:
        warnings.append(
            f"imported proto-tools {source.proto_tools_commit} differs from the submodule pin "
            f"{source.proto_tools_pinned_commit}"
        )
    framework = _framework_artifacts(repo_root, proto_tools_root())
    upstream_arts = _upstream_artifacts(upstream)
    tools = _tool_artifacts(framework[PT_FRAMEWORK_KEY], upstream_arts, upstream, warnings)
    audit: list[str] = []
    components = _component_artifacts(repo_root, framework[PL_FRAMEWORK_KEY], tools, audit)
    programs = _program_artifacts(repo_root, config, components, warnings)

    artifacts = [
        *framework.values(),
        *upstream_arts.values(),
        *tools.values(),
        *components.values(),
        *programs.values(),
    ]
    order = {kind: i for i, kind in enumerate(ARTIFACT_KINDS)}
    artifacts.sort(key=lambda a: (order[a.kind], a.key))
    return Lock(
        source=source,
        artifacts=artifacts,
        warnings=sorted(warnings),
        edge_audit=sorted(audit),
        release_digest=Lock.compute_release_digest(source, artifacts),
    )
