from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

publisher = importlib.import_module("publish_github_pages")
PUBLISHER = publisher.PUBLISHER
publish_site = publisher.publish_site


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
        yield Path(temp_dir)


def _make_source(root: Path, files: dict[str, bytes]) -> Path:
    source = root / "build-source"
    source.mkdir(parents=True, exist_ok=True)
    records = []
    for relative, content in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (source / "export-integrity.json").write_text(
        json.dumps({"files": records}, ensure_ascii=False), encoding="utf-8"
    )
    return source


def test_publish_preserves_course_docs_skips_workflows_and_cleans_stale_assets(temp_root):
    docs = temp_root / "docs"
    docs.mkdir()
    course_doc = docs / "annotation_decisions.md"
    course_doc.write_text("原有课程说明\n", encoding="utf-8")
    reused_index = docs / "index.html"
    reused_index.write_text("same site page", encoding="utf-8")
    workflow = docs / ".github/workflows/course.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("原有工作流\n", encoding="utf-8")

    source = _make_source(
        temp_root,
        {
            "index.html": b"same site page",
            ".nojekyll": b"",
            "static/old.bin": b"old generated asset",
            ".github/workflows/deploy.yml": b"must not be published",
        },
    )
    first = publish_site(source, docs)
    assert first["reused"] == 1
    assert (docs / ".nojekyll").is_file()
    assert (docs / "static/old.bin").read_bytes() == b"old generated asset"
    assert not (docs / ".github/workflows/deploy.yml").exists()
    assert workflow.read_text(encoding="utf-8") == "原有工作流\n"
    first_manifest = json.loads((docs / "export-integrity.json").read_text(encoding="utf-8"))
    listed = {row["path"]: row for row in first_manifest["files"]}
    assert ".github/workflows/deploy.yml" not in listed
    assert listed["index.html"]["owned"] is False
    assert first_manifest["publisher"] == PUBLISHER

    source = _make_source(
        temp_root,
        {
            "index.html": b"same site page",
            ".nojekyll": b"",
            "static/new.bin": b"new generated asset",
        },
    )
    second = publish_site(source, docs)
    assert second["removed"] == 1
    assert not (docs / "static/old.bin").exists()
    assert (docs / "static/new.bin").read_bytes() == b"new generated asset"
    assert reused_index.read_text(encoding="utf-8") == "same site page"
    assert course_doc.read_text(encoding="utf-8") == "原有课程说明\n"
    assert workflow.read_text(encoding="utf-8") == "原有工作流\n"


def test_publish_rejects_conflicting_unowned_file_before_copying(temp_root):
    source = _make_source(
        temp_root,
        {"index.html": b"generated page", "static/model.json": b"model"},
    )
    docs = temp_root / "docs"
    docs.mkdir()
    (docs / "index.html").write_bytes(b"course document")

    with pytest.raises(FileExistsError, match="non-publisher file: index.html"):
        publish_site(source, docs)

    assert (docs / "index.html").read_bytes() == b"course document"
    assert not (docs / "static/model.json").exists()
    assert not (docs / "export-integrity.json").exists()


def test_publish_rejects_source_and_previous_manifest_path_traversal(temp_root):
    source = _make_source(temp_root, {"index.html": b"page"})
    source_manifest = json.loads(
        (source / "export-integrity.json").read_text(encoding="utf-8")
    )
    source_manifest["files"].append({"path": "../outside.txt", "bytes": 1, "sha256": "0"})
    (source / "export-integrity.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    docs = temp_root / "docs"
    docs.mkdir()
    outside = temp_root / "outside.txt"
    outside.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="path escapes source root"):
        publish_site(source, docs)
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert not (docs / "index.html").exists()

    source = _make_source(temp_root, {"index.html": b"page"})
    old_manifest = {
        "publisher": PUBLISHER,
        "files": [{"path": "../outside.txt", "owned": True}],
    }
    (docs / "export-integrity.json").write_text(
        json.dumps(old_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="path escapes destination root"):
        publish_site(source, docs)
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert not (docs / "index.html").exists()
