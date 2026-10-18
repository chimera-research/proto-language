"""Mirror planning and content-addressed layout (no network)."""

import pytest

from proto_language.provenance.upstream.mirror import MirrorConfig, cas_uri, plan
from proto_language.provenance.upstream.sources import ToolkitUpstream, UpstreamLock, UpstreamSource


def _lock() -> UpstreamLock:
    lock = UpstreamLock()
    lock.toolkits["t"] = ToolkitUpstream(toolkit="t", category="c")
    lock.add("t", UpstreamSource.git("https://github.com/lab/repo", None))
    url = UpstreamSource.url("https://x.org/w.pt", role="weights")
    lock.add("t", url)
    hf = UpstreamSource.hf("lab/model", None)
    hf.resolved = {"commit": "c" * 40, "files": [{"path": "m.safetensors", "size": 10, "sha256": "d" * 64}]}
    lock.add("t", hf)
    return lock


def test_cas_uri_layout() -> None:
    sha = "ab" + "c" * 62
    assert cas_uri("s3://bucket/", sha) == f"s3://bucket/sha256/ab/cc/{sha}"


def test_plan_covers_git_urls_and_hf_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    config = MirrorConfig(git_remote_template="git@github.com:org/up--{owner}--{repo}.git", blob_store="file:///cas")
    actions = plan(_lock(), config)
    git = [a for a in actions if a.kind == "git"]
    blobs = {a.url: a for a in actions if a.kind == "blob"}
    assert git[0].target == "git@github.com:org/up--lab--repo.git"
    assert "https://x.org/w.pt" in blobs
    hf_url = "https://huggingface.co/lab/model/resolve/" + "c" * 40 + "/m.safetensors"
    assert blobs[hf_url].expected_sha256 == "d" * 64


def test_plan_skips_already_mirrored() -> None:
    lock = _lock()
    lock.sources["url:https://x.org/w.pt"].resolved["mirror"] = "file:///cas/sha256/x"
    actions = plan(lock, MirrorConfig(blob_store="file:///cas"))
    assert all(a.url != "https://x.org/w.pt" for a in actions)
