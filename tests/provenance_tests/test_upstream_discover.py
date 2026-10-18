"""Static upstream discovery and declarations over a synthetic proto-tools tree."""

import textwrap
from pathlib import Path

import pytest

from proto_language.provenance.upstream.declarations import apply_declarations, carry_over
from proto_language.provenance.upstream.discover import discover, scan_text_for_sources
from proto_language.provenance.upstream.sources import UpstreamLock, UpstreamSource, normalize_git_url

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def proto_tools_tree(tmp_path: Path) -> Path:
    """One toolkit exercising every discovery channel."""
    toolkit = tmp_path / "proto_tools" / "tools" / "structure_prediction" / "foldy2"
    (toolkit / "standalone").mkdir(parents=True)
    (toolkit / "__init__.py").write_text("")
    (toolkit / "links.yaml").write_text(
        "github: https://github.com/Lab/Foldy\n"
        "huggingface:\n  - https://huggingface.co/lab/foldy-2\n"
        "organizations:\n  - Lab\n"
    )
    (toolkit / "license.yaml").write_text("code:\n  spdx: MIT\n")
    (toolkit / "standalone" / "requirements.txt").write_text(
        "foldy==2.0.1\nnumpy>=1.24\ncolabdesign @ git+https://github.com/sokrypton/ColabDesign.git@" + SHA + "\n"
    )
    (toolkit / "standalone" / "setup.sh").write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            REPO_URL="https://github.com/lab/extras.git"
            # git clone https://github.com/ignored/comment.git
            echo "git clone https://github.com/ignored/echo.git"
            git clone --depth 1 $REPO_URL "$DIR"
            git -C "$DIR" checkout v1.2
            uv pip install --index-strategy unsafe-best-match "torch==2.6.0"
            curl -fsSL https://storage.googleapis.com/lab/foldy_params_2024.tar | tar x
            """
        )
    )
    (toolkit / "foldy2.py").write_text(
        textwrap.dedent(
            f'''\
            _VERSION = "1.2"
            MODELS = {{"small": "lab/foldy-2-small", "unrelated": "not/a-model"}}

            def load():
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id="lab/foldy-2", filename="m.pt", revision="{SHA}")
                return f"https://example.org/weights/v{{_VERSION}}/foldy.ckpt"
            '''
        )
    )
    return tmp_path


def test_normalize_git_url_variants() -> None:
    expected = "github.com/Lab/Repo"
    for url in ("https://github.com/Lab/Repo.git", "git@github.com:Lab/Repo.git", "git+https://GitHub.com/Lab/Repo/"):
        assert normalize_git_url(url) == expected


def test_discover_finds_every_channel(proto_tools_tree: Path) -> None:
    lock = discover(proto_tools_tree)
    entry = lock.toolkits["foldy2"]
    ids = set(entry.sources)
    assert entry.category == "structure_prediction"
    assert entry.env_name == "foldy2"
    assert entry.license == {"code": {"spdx": "MIT"}}
    assert "git:github.com/Lab/Foldy" in ids
    assert "pypi:foldy==2.0.1" in ids
    assert lock.sources["pypi:foldy==2.0.1"].role == "code"
    assert lock.sources["pypi:numpy>=1.24"].role == "dependency"
    assert f"git:github.com/sokrypton/ColabDesign@{SHA}" in ids
    # A clone followed by a checkout records the checked-out ref, not the floating clone.
    assert "git:github.com/lab/extras@v1.2" in ids
    assert "git:github.com/lab/extras" not in ids
    assert not any("ignored" in sid for sid in ids)
    assert "pypi:torch==2.6.0" in ids
    assert not any("unsafe-best-match" in sid for sid in ids)
    assert "url:https://storage.googleapis.com/lab/foldy_params_2024.tar" in ids
    # The revision-pinned call supersedes the unpinned links.yaml sighting.
    assert f"hf:lab/foldy-2@{SHA}" in ids
    assert "hf:lab/foldy-2" not in ids
    # Pass two: a bare literal under an owner learned from an explicit HF reference.
    assert "hf:lab/foldy-2-small" in ids
    assert not any("not/a-model" in sid for sid in ids)
    # Module constants expand inside f-string URLs.
    assert "url:https://example.org/weights/v1.2/foldy.ckpt" in ids


def test_pinned_status_rules() -> None:
    assert UpstreamSource.pypi("x", "==1.0").ref_is_immutable
    assert not UpstreamSource.pypi("x", ">=1.0").ref_is_immutable
    assert UpstreamSource.git("https://github.com/a/b", SHA).ref_is_immutable
    assert not UpstreamSource.git("https://github.com/a/b", "main").ref_is_immutable
    assert not UpstreamSource.hf("a/b", None).ref_is_immutable


def test_scan_ignores_docs_urls() -> None:
    found = scan_text_for_sources("see https://doi.org/10.1/x and https://arxiv.org/abs/1", "f.py", is_python=False)
    assert found == []


def test_declarations_pin_ignore_and_services(proto_tools_tree: Path) -> None:
    lock = discover(proto_tools_tree)
    problems = apply_declarations(
        lock,
        {
            "toolkits": {
                "foldy2": {
                    "add": [{"type": "url", "locator": "https://example.org/extra.pt", "sha256": "f" * 64}],
                    "ignore": ["git:github.com/Lab/Foldy", "git:github.com/never/seen"],
                    "services": ["https://api.example.org"],
                    "notes": "test",
                },
                "nope": {},
            }
        },
    )
    entry = lock.toolkits["foldy2"]
    assert "git:github.com/Lab/Foldy" not in entry.sources
    assert "git:github.com/Lab/Foldy" not in lock.sources
    assert lock.sources["url:https://example.org/extra.pt"].resolved["sha256"] == "f" * 64
    assert entry.services == ["https://api.example.org"]
    assert entry.notes == "test"
    assert any("nope" in p for p in problems)
    assert any("never/seen" in p for p in problems)


def test_carry_over_keeps_mirrored_hashes() -> None:
    old = UpstreamLock()
    src = UpstreamSource.url("https://x.org/w.pt", role="weights")
    src.resolved = {"sha256": "a" * 64, "mirror": "s3://cas/sha256/aa/aa/" + "a" * 64}
    old.sources[src.id] = src
    new = UpstreamLock()
    new.sources[src.id] = UpstreamSource.url("https://x.org/w.pt", role="weights")
    carry_over(new, old)
    assert new.sources[src.id].resolved["sha256"] == "a" * 64
