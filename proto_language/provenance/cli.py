"""Command line for the registry and upstream control: ``python -m proto_language.provenance``.

Verbs:

* ``build``: introspect registries and write the release lock (no network, no DB).
* ``diff``: compare two locks by ``(kind, key)`` and digest.
* ``migrate`` / ``publish`` / ``verify``: apply SQL, publish a lock, re-hash a stored release.
* ``upstream discover``: static scan of toolkits (prints a summary or JSON).
* ``upstream resolve``: discover + declarations + resolve (+ recursive scan); writes
  ``registry/upstream/upstream.lock.json`` and ``AUDIT.md``.
* ``upstream mirror``: plan (default) or ``--execute`` mirroring of code and weights.

The database DSN comes from ``--dsn`` or ``REGISTRY_DATABASE_URL``.

Examples:
    >>> # python -m proto_language.provenance build --out registry.lock.json
    >>> # python -m proto_language.provenance upstream resolve --skip-types hf-model,hf-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from proto_language.provenance.models import Lock

__all__ = ["DigestIndex", "diff_locks", "lock_index", "main"]

logger = logging.getLogger(__name__)

_DEFAULT_UPSTREAM_LOCK = Path("registry/upstream/upstream.lock.json")
_DEFAULT_DECLARATIONS = Path("registry/upstream/declarations.yaml")
_DEFAULT_AUDIT = Path("registry/upstream/AUDIT.md")


DigestIndex = dict[tuple[str, str], tuple[str, str]]


def lock_index(lock: Lock) -> DigestIndex:
    """Reduce a lock to ``(kind, key) -> (digest, local_digest)``.

    Args:
        lock (Lock): Release lock.

    Returns:
        DigestIndex: Digest index.
    """
    return {(a.kind, a.key): (a.digest, a.local_digest) for a in lock.artifacts}


def diff_locks(before: DigestIndex, after: DigestIndex) -> dict[str, list[str]]:
    """Summarize what changed between two releases.

    Args:
        before (DigestIndex): Baseline (``lock_index`` or ``fetch_release_index``).
        after (DigestIndex): Candidate.

    Returns:
        dict[str, list[str]]: ``added``, ``removed``, ``changed`` (own content moved), and
        ``reversioned`` (only a dependency moved) entries as ``kind:key``.
    """
    out: dict[str, list[str]] = {"added": [], "removed": [], "changed": [], "reversioned": []}
    for ident in sorted(after.keys() - before.keys()):
        out["added"].append(":".join(ident))
    for ident in sorted(before.keys() - after.keys()):
        out["removed"].append(":".join(ident))
    for ident in sorted(before.keys() & after.keys()):
        (old_digest, old_local), (new_digest, new_local) = before[ident], after[ident]
        if old_digest == new_digest:
            continue
        out["changed" if old_local != new_local else "reversioned"].append(":".join(ident))
    return out


def _markdown_diff(diff: dict[str, list[str]], limit: int = 40) -> str:
    lines = ["| change | count | examples |", "|---|---:|---|"]
    for name, items in diff.items():
        sample = ", ".join(f"`{i}`" for i in items[:limit]) + (" ..." if len(items) > limit else "")
        lines.append(f"| {name} | {len(items)} | {sample} |")
    return "\n".join(lines) + "\n"


def _dsn(args: argparse.Namespace) -> str:
    dsn = args.dsn or os.environ.get("REGISTRY_DATABASE_URL")
    if not dsn:
        raise SystemExit("No database: pass --dsn or set REGISTRY_DATABASE_URL")
    return str(dsn)


def _cmd_build(args: argparse.Namespace) -> int:
    from proto_language.provenance.collect import build_lock, load_config

    config = load_config(args.repo / "registry" / "config.yaml")
    if args.no_strict:
        config.strict_programs = False
    lock = build_lock(args.repo, config)
    Path(args.out).write_text(lock.model_dump_json(indent=1) + "\n")
    counts: dict[str, int] = {}
    for art in lock.artifacts:
        counts[art.kind] = counts.get(art.kind, 0) + 1
    logger.info("Wrote %s (%s) release_digest=%s", args.out, counts, lock.release_digest)
    for warning in lock.warnings:
        logger.warning("%s", warning)
    if lock.edge_audit:
        logger.info(
            "Edge audit: %d components import tools they do not declare (see lock.edge_audit)", len(lock.edge_audit)
        )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    new = lock_index(Lock.model_validate_json(Path(args.new).read_text()))
    if args.against:
        from proto_language.provenance.publish import fetch_release_index

        old = fetch_release_index(_dsn(args), args.against)
    elif args.old:
        old = lock_index(Lock.model_validate_json(Path(args.old).read_text()))
    else:
        raise SystemExit("diff needs --old LOCK or --against CHANNEL_OR_RELEASE")
    diff = diff_locks(old, new)
    sys.stdout.write(_markdown_diff(diff) if args.markdown else json.dumps(diff, indent=2) + "\n")
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    from proto_language.provenance.publish import _connect, migrate

    with _connect(_dsn(args)) as conn:
        applied = migrate(conn)
    logger.info("Applied migrations: %s", applied or "none pending")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    from proto_language.provenance.publish import publish

    lock = Lock.model_validate_json(Path(args.lock).read_text())
    result = publish(
        _dsn(args),
        lock,
        channel=args.channel or None,
        ci_run_url=args.ci_run_url,
        allow_dirty=args.allow_dirty,
        release_id=args.release_id,
    )
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    if args.github_output:
        with Path(args.github_output).open("a") as handle:
            handle.write(f"release_id={result.release_id}\ncreated={str(result.created).lower()}\n")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from proto_language.provenance.publish import verify_release

    problems = verify_release(_dsn(args), args.release)
    for problem in problems:
        logger.error("%s", problem)
    logger.info("Release %s: %s", args.release, "verified" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def _cmd_upstream_discover(args: argparse.Namespace) -> int:
    from proto_language.provenance.collect import proto_tools_root
    from proto_language.provenance.upstream.discover import discover

    lock = discover(args.proto_tools or proto_tools_root(), toolkits=args.toolkit or None)
    if args.json:
        sys.stdout.write(lock.model_dump_json(indent=1) + "\n")
        return 0
    for name, entry in sorted(lock.toolkits.items()):
        interesting = [sid for sid in entry.sources if lock.sources[sid].role != "dependency"]
        sys.stdout.write(f"{name}: {len(entry.sources)} sources\n")
        for sid in interesting:
            sys.stdout.write(f"    {lock.sources[sid].role:8s} {sid}\n")
    return 0


def _cmd_upstream_resolve(args: argparse.Namespace) -> int:
    from proto_language.provenance.collect import _git, proto_tools_root
    from proto_language.provenance.upstream.declarations import apply_declarations, carry_over, load_declarations
    from proto_language.provenance.upstream.discover import discover
    from proto_language.provenance.upstream.report import render_audit
    from proto_language.provenance.upstream.resolve import resolve
    from proto_language.provenance.upstream.sources import UpstreamLock

    root = args.proto_tools or proto_tools_root()
    out = Path(args.out)
    previous = UpstreamLock.model_validate_json(out.read_text()) if out.exists() else None
    lock = discover(root)
    lock.proto_tools_commit = _git(root, "rev-parse", "HEAD")
    problems = apply_declarations(lock, load_declarations(Path(args.declarations)))
    for problem in problems:
        logger.warning("%s", problem)
    carry_over(lock, previous)

    if args.check:
        if previous is None:
            logger.error("No committed upstream lock at %s", out)
            return 1
        added = sorted(set(lock.sources) - set(previous.sources))
        removed = sorted(set(previous.sources) - set(lock.sources))
        for sid in added:
            logger.error("new upstream source not in lock: %s", sid)
        for sid in removed:
            logger.error("lock lists a source no longer referenced: %s", sid)
        if added or removed:
            logger.error("Run `python -m proto_language.provenance upstream resolve` and commit the result")
            return 1
        logger.info("Upstream lock covers every discovered source (%d)", len(lock.sources))
        return 0

    skip = set(filter(None, (args.skip_types or "").split(",")))
    resolve(lock, recurse=not args.no_recurse, workers=args.workers, skip_types=skip)
    if previous is not None:
        # Keep last-known resolutions for sources this network could not reach.
        for sid, src in lock.sources.items():
            old = previous.sources.get(sid)
            if src.status == "unresolved" and old is not None and old.status != "unresolved":
                src.status, src.resolved = old.status, old.resolved
    out.write_text(lock.model_dump_json(indent=1) + "\n")
    Path(args.audit).write_text(render_audit(lock))
    logger.info("Wrote %s and %s: %s", out, args.audit, lock.summary())
    return 0


def _cmd_upstream_mirror(args: argparse.Namespace) -> int:
    from proto_language.provenance.collect import load_config
    from proto_language.provenance.upstream.mirror import MirrorConfig, execute, plan
    from proto_language.provenance.upstream.report import render_audit
    from proto_language.provenance.upstream.sources import UpstreamLock

    lock_path = Path(args.lock)
    lock = UpstreamLock.model_validate_json(lock_path.read_text())
    config = MirrorConfig.model_validate(load_config(args.repo / "registry" / "config.yaml").mirror)
    if args.blob_store:
        config.blob_store = args.blob_store
    if args.git_remote_template:
        config.git_remote_template = args.git_remote_template
    actions = plan(lock, config, toolkits=set(args.toolkit) if args.toolkit else None)
    if not args.execute:
        for action in actions:
            sys.stdout.write(f"{action.kind:4s} {action.source_id}  {action.target or action.url}\n")
        logger.info("%d actions planned; pass --execute to run them", len(actions))
        return 0
    failures = execute(lock, actions, config)
    lock_path.write_text(lock.model_dump_json(indent=1) + "\n")
    Path(args.audit).write_text(render_audit(lock))
    for failure in failures:
        logger.error("%s", failure)
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m proto_language.provenance", description=__doc__.split("\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(), help="proto-language checkout root")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="write the release lock")
    p.add_argument("--out", default="registry.lock.json")
    p.add_argument("--no-strict", action="store_true", help="skip invalid programs instead of failing")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("diff", help="compare a lock with another lock or a published release")
    p.add_argument("new", help="candidate lock file")
    p.add_argument("--old", default=None, help="baseline lock file")
    p.add_argument("--against", default=None, help="baseline channel or release id in the database")
    p.add_argument("--dsn", default=None, help="Postgres DSN (default: $REGISTRY_DATABASE_URL)")
    p.add_argument("--markdown", action="store_true", help="render a Markdown table (for CI summaries)")
    p.set_defaults(func=_cmd_diff)

    for name, func, help_text in (
        ("migrate", _cmd_migrate, "apply pending SQL migrations"),
        ("publish", _cmd_publish, "publish a lock as an immutable release"),
        ("verify", _cmd_verify, "re-hash a stored release"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dsn", default=None, help="Postgres DSN (default: $REGISTRY_DATABASE_URL)")
        p.set_defaults(func=func)
        if name == "publish":
            p.add_argument("--lock", default="registry.lock.json")
            p.add_argument("--channel", default="main", help="channel to move ('' to leave channels alone)")
            p.add_argument("--ci-run-url", default=None)
            p.add_argument("--release-id", default=None)
            p.add_argument("--allow-dirty", action="store_true")
            p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
        if name == "verify":
            p.add_argument("--release", required=True)

    up = sub.add_parser("upstream", help="upstream code/weights control").add_subparsers(dest="upstream", required=True)
    p = up.add_parser("discover", help="static scan of toolkits")
    p.add_argument("--proto-tools", type=Path, default=None)
    p.add_argument("--toolkit", action="append")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_upstream_discover)

    p = up.add_parser("resolve", help="discover, apply declarations, resolve, write lock + audit")
    p.add_argument("--proto-tools", type=Path, default=None)
    p.add_argument("--out", default=str(_DEFAULT_UPSTREAM_LOCK))
    p.add_argument("--audit", default=str(_DEFAULT_AUDIT))
    p.add_argument("--declarations", default=str(_DEFAULT_DECLARATIONS))
    p.add_argument("--no-recurse", action="store_true", help="skip scanning upstream packages/repos")
    p.add_argument("--skip-types", default="", help="comma-separated source types to leave unresolved")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--check", action="store_true", help="offline: fail if the lock misses discovered sources")
    p.set_defaults(func=_cmd_upstream_resolve)

    p = up.add_parser("mirror", help="plan or execute mirroring")
    p.add_argument("--lock", default=str(_DEFAULT_UPSTREAM_LOCK))
    p.add_argument("--audit", default=str(_DEFAULT_AUDIT))
    p.add_argument("--toolkit", action="append")
    p.add_argument("--blob-store", default=None)
    p.add_argument("--git-remote-template", default=None)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=_cmd_upstream_mirror)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv (list[str] | None): Arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    code: int = args.func(args)
    return code
