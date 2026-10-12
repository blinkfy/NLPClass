from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.templating import Jinja2Templates

from nlplab.core.exceptions import ResourceUnavailableError

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

export_client = importlib.import_module("static_export_client")
exporter = importlib.import_module("export_static_data")
StaticExportClient = export_client.StaticExportClient


def test_static_client_parses_query_and_path_params_without_network():
    app = FastAPI()

    @app.get("/items/{item_id}")
    def item(item_id: int, active: bool = True, label: str = ""):
        return {"item_id": item_id, "active": active, "label": label}

    response = StaticExportClient(app).get(
        "/items/42", {"active": "false", "label": "中文 & NLP"}
    )

    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "active": False, "label": "中文 & NLP"}


def test_static_client_preserves_utf8_template_json_and_file_response():
    app = FastAPI()
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
        temp_path = Path(temp_dir)
        file_path = temp_path / "冻结结果.json"
        file_path.write_bytes('{"状态":"就绪"}'.encode("utf-8"))
        (temp_path / "page.html").write_text("<p>{{ title }}</p>", encoding="utf-8")
        templates = Jinja2Templates(directory=str(temp_path))

        @app.get("/template")
        def template(request):
            return templates.TemplateResponse(
                request=request,
                name="page.html",
                context={"title": "中文页面"},
            )

        @app.get("/json")
        def json_response():
            return JSONResponse({"分词": ["自然语言", "处理"]})

        @app.get("/file")
        def file_response():
            return FileResponse(file_path)

        client = StaticExportClient(app)
        html = client.get("/template")
        data = client.get("/json")
        file = client.get("/file")

        assert html.text == "<p>中文页面</p>"
        assert "text/html" in html.headers["content-type"]
        assert data.json() == {"分词": ["自然语言", "处理"]}
        assert file.json() == {"状态": "就绪"}
        assert file.content == file_path.read_bytes()
        assert "application/json" in file.headers["content-type"]


def test_static_client_returns_route_exceptions_as_json_without_network():
    app = FastAPI()

    @app.get("/unavailable")
    def unavailable():
        raise ResourceUnavailableError("本地模型未准备")

    @app.get("/conflict")
    def conflict():
        raise HTTPException(status_code=409, detail="版本已变化")

    client = StaticExportClient(app)
    unavailable_response = client.get("/unavailable")
    conflict_response = client.get("/conflict")

    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["error"]["code"] == "resource_unavailable"
    assert unavailable_response.json()["error"]["message"] == "本地模型未准备"
    assert conflict_response.status_code == 409
    assert conflict_response.json() == {"detail": "版本已变化"}


def test_errors_are_exported_as_one_full_response_and_paths_are_scrubbed(monkeypatch):
    class FakeResponse:
        def __init__(self, value, status_code=200, content_type="application/json"):
            self.status_code = status_code
            self.headers = {"content-type": content_type}
            self.content = json.dumps(value, ensure_ascii=False).encode("utf-8")

        def json(self):
            return json.loads(self.content)

    class FakeClient:
        def __init__(self, app):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, path, params=None):
            params = params or {}
            if path == "/api/v1/experiments/1/runs":
                value = {"items": [{"run_id": "frozen-a", "run_type": "frozen-final-benchmark"}]}
            elif path == "/api/v1/experiments/1/extensions":
                value = {"items": []}
            elif path == "/api/v1/experiments/1/errors":
                page = int(params.get("page", 1))
                items = [
                    {
                        "sample_id": f"S{index}",
                        "text": f"错误原文{index}",
                        "path": r"C:\Users\private\run.json",
                        "user_email": "person@example.com",
                    }
                    for index in range((page - 1) * 100, min(page * 100, 101))
                ]
                value = {"total": 101, "page": page, "page_size": 100, "items": items}
            elif path == "/api/v1/views/cws/errors":
                value = {"total": 0, "page": 1, "page_size": 100, "items": []}
            elif path == "/api/v1/experiments/2/runs":
                value = {"items": []}
            elif path in {
                "/api/v1/experiments/3/runs",
                "/api/v1/experiments/3/stage4/runs",
            }:
                value = {"items": []}
            else:
                value = {}
            return FakeResponse(value)

    monkeypatch.setattr(exporter, "StaticExportClient", FakeClient)
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
        destination = Path(temp_dir)
        outputs = destination / "outputs"
        app = SimpleNamespace(
            state=SimpleNamespace(
                clustering=SimpleNamespace(settings=SimpleNamespace(outputs=outputs)),
                store=SimpleNamespace(settings=SimpleNamespace(outputs=outputs)),
            )
        )
        monkeypatch.setattr(exporter, "_frozen_clustering_reads", lambda _: nullcontext())
        result = exporter.export_data(app, destination)

        full_key = "/api/v1/experiments/1/errors?run_id=frozen-a"
        assert full_key in result["responses"]
        error_keys = (
            key
            for key in result["responses"]
            if key.endswith("/errors") or "/errors?" in key
        )
        assert not any(
            "page=" in key or "page_size=" in key for key in error_keys
        )
        full_file = destination / result["responses"][full_key]["file"]
        full = json.loads(full_file.read_text(encoding="utf-8"))
        assert full["total"] == 101
        assert len(full["items"]) == 101
        assert full["page"] == 1 and full["page_size"] == 100000
        assert full["items"][100]["text"] == "错误原文100"
        assert full["items"][0]["path"] == "[本机路径已移除]"
        assert "user_email" not in full["items"][0]


def test_canonical_key_normalizes_bool_and_omits_empty_parameters():
    assert exporter.canonical_key(
        "/api/v1/example", {"z": "", "flag": False, "none": None, "count": 2}
    ) == "/api/v1/example?count=2&flag=false"


def test_scrub_keeps_errors_text_and_removes_paths_even_with_corpus_included():
    value = {
        "items": [
            {
                "sample_id": "S1",
                "text": "错误案例原文",
                "path": r"C:\Users\private\case.json",
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 100000,
    }

    clean = exporter._scrub(value, include_corpus=True, redact_corpus=False)

    assert clean["items"][0]["text"] == "错误案例原文"
    assert clean["total"] == 1
    assert clean["items"][0]["path"] == "[本机路径已移除]"


def test_export_local_hash_cache_invalidates_and_restores_module_aliases(monkeypatch):
    course_module = importlib.import_module("nlplab.data.course")
    original_digest = export_client.file_sha256
    calls = []

    def counted_digest(path):
        calls.append(Path(path))
        return original_digest(path)

    monkeypatch.setattr(export_client, "file_sha256", counted_digest)
    monkeypatch.setattr(course_module, "file_sha256", counted_digest)
    client = StaticExportClient(FastAPI())

    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
        path = Path(temp_dir) / "frozen.json"
        path.write_bytes(b"first")
        with client:
            assert course_module.file_sha256 is client._cached_digest
            first = course_module.file_sha256(path)
            assert course_module.file_sha256(path) == first
            assert len(calls) == 1

            previous = path.stat()
            path.write_bytes(b"other")  # same size, changed mtime invalidates the digest
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
            second = course_module.file_sha256(path)
            assert second != first
            assert len(calls) == 2

        assert course_module.file_sha256 is counted_digest


def test_compact_snapshots_shares_large_utf8_metadata_and_restores_values():
    compact_module = importlib.import_module("compact_static_snapshots")
    chi2_metadata = {
        f"中文特征{i:04d}": {"chi2_score": i / 7.0, "display_rank": i + 1}
        for i in range(1800)
    }
    assert len(
        json.dumps(chi2_metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ) >= 32768
    snapshots = {
        "data/first.json": {
            "run_id": "冻结一",
            "representation": {
                "chi2_metadata": chi2_metadata,
                "selected_features": ["中文", "特征"],
            },
        },
        "data/second.json": {
            "run_id": "冻结二",
            "representation": {
                "chi2_metadata": chi2_metadata,
                "selected_features": ["中文", "特征"],
            },
        },
    }

    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp_dir:
        destination = Path(temp_dir)
        for relative_path, value in snapshots.items():
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        manifest = {"responses": {key: {"file": key} for key in snapshots}}

        result = compact_module.compact_snapshots(destination, manifest)

        assert result["shared_files"] == 1
        assert len(list((destination / "data/shared").glob("*.json"))) == 1
        shared_file = next((destination / "data/shared").glob("*.json"))

        def restore(value):
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, dict):
                reference = value.get("__nlplab_static_ref__")
                if reference:
                    return json.loads((destination / reference).read_text(encoding="utf-8"))
                return {key: restore(item) for key, item in value.items()}
            return value

        assert shared_file.is_file()
        for relative_path, expected in snapshots.items():
            compacted = json.loads((destination / relative_path).read_text(encoding="utf-8"))
            assert "__nlplab_static_ref__" in compacted["representation"]["chi2_metadata"]
            assert restore(compacted) == expected
