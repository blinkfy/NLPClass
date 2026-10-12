"""Export read-only API responses for the GitHub Pages build.

This module never trains, predicts from user supplied text, builds caches, or
updates frozen runs. JSON responses have licensed corpus text reduced to
short previews by default.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

from static_export_client import StaticExportClient

# OpenAPI defaults used by the static runtime when matching a URL with its
# canonical snapshot key after removing parameters equal to their defaults.
DEFAULTS: dict[str, dict[str, Any]] = {
    "/api/v1/experiments/1/metrics": {"run_id": "latest"},
    "/api/v1/experiments/1/errors": {"run_id": "latest", "page": 1, "page_size": 10},
    "/api/v1/experiments/1/trace": {"model": "hmm1-smooth", "run_id": "latest"},
    "/api/v1/models/hmm/parameters": {"run_id": "latest", "topk": 20, "model": "hmm1-smooth"},
    "/api/v1/views/cws/errors": {"model": "hmm1-smooth", "page": 1, "page_size": 10},
    "/api/v1/experiments/2/overview": {"run_id": "latest"},
    "/api/v1/experiments/2/result": {"run_id": "latest", "k": 5},
    "/api/v1/experiments/2/projection": {"run_id": "latest", "dimensions": 2},
    "/api/v1/experiments/2/inspection": {"run_id": "latest", "k": 5},
    "/api/v1/experiments/2/document": {"run_id": "latest", "k": 5},
    "/api/v1/experiments/2/summary": {"k": 5},
    "/api/v1/experiments/2/comparison-view": {"k": 5},
    "/api/v1/experiments/3/overview": {"run_id": "latest"},
    "/api/v1/experiments/3/predictions": {
        "run_id": "latest",
        "model_id": "C0",
        "errors_only": False,
    },
    "/api/v1/experiments/3/features": {"run_id": "latest", "model_id": "C0"},
    "/api/v1/experiments/3/stage2/predictions": {"run_id": "latest", "errors_only": False},
    "/api/v1/experiments/3/stage2/features": {"run_id": "latest"},
    "/api/v1/experiments/3/stage2/paired": {"run_id": "latest"},
    "/api/v1/experiments/3/stage3/predictions": {"run_id": "latest", "errors_only": False},
    "/api/v1/experiments/3/stage3/features": {"run_id": "latest"},
    "/api/v1/experiments/3/stage3/paired": {"run_id": "latest", "old_model": "C2"},
    "/api/v1/experiments/3/stage4/runs": {},
    "/api/v1/experiments/3/stage4/overview": {"run_id": "latest"},
    "/api/v1/experiments/3/stage5/overview": {"run_id": "latest"},
    "/api/v1/experiments/3/stage5/flips": {"run_id": "latest"},
    "/api/v1/experiments/3/stage6/overview": {"run_id": "latest"},
    "/api/v1/experiments/3/workbench/document": {
        "run_id": "latest",
        "source": "test",
        "condition": "OA",
    },
    "/api/v1/experiments/3/visualization/c3-trace": {"kernel": 3, "filter_id": 0},
    "/api/v1/experiments/3/visualization/shortcut-document": {"run_id": "latest"},
    "/api/v1/comprehensive/collections/{collection_id}/documents/{document_id}": {"analyze": False},
}


def canonical_key(
    path: str, params: dict[str, Any] | Iterable[tuple[str, Any]] | None = None
) -> str:
    pairs = list((params or {}).items()) if isinstance(params, dict) else list(params or [])

    def value_text(value: Any) -> str:
        return str(value).lower() if isinstance(value, bool) else str(value)

    normalized = sorted(
        (str(k), value_text(v)) for k, v in pairs if v is not None and value_text(v) != ""
    )
    return path + ("?" + urlencode(normalized) if normalized else "")


def _scrub(value: Any, include_corpus: bool, redact_corpus: bool, *, parent_key: str = "") -> Any:
    """Redact licensed corpus fields and machine-local identifiers recursively."""
    import re

    private_keys = {
        "username",
        "user_name",
        "email",
        "user_email",
        "user_id",
        "home",
        "home_dir",
        "user_home",
        "personal_key",
        "api_key",
        "access_token",
        "hf_token",
        "password",
        "secret",
        "hostname",
        "computer_name",
        "machine_id",
    }
    local_path = re.compile(
        r"(?i)(?:[a-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\?)+|\\\\[^\\\s]+\\[^\\\s]+(?:\\[^\\\s]+)*)"
    )
    if isinstance(value, list):
        return [
            _scrub(item, include_corpus, redact_corpus, parent_key=parent_key) for item in value
        ]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        redacted = False
        for key, item in value.items():
            normalized_key = key.casefold()
            if normalized_key in private_keys:
                continue
            if (
                normalized_key in {"original_text", "raw_text", "document_text", "full_text"}
                and redact_corpus
                and not include_corpus
            ):
                cleaned[key] = None
                redacted = True
                continue
            if (
                normalized_key == "text"
                and redact_corpus
                and not include_corpus
                and isinstance(item, str)
                and parent_key not in {"entities", "entity", "lexical"}
            ):
                cleaned[key] = item[:160] + "…（静态版省略原文）"
                redacted = True
                continue
            cleaned[key] = _scrub(item, include_corpus, redact_corpus, parent_key=normalized_key)
        if redacted:
            cleaned["corpus_text_redacted"] = True
        return cleaned
    if isinstance(value, str):
        value = local_path.sub("[本机路径已移除]", value)
        return re.sub(
            r"(?<![A-Za-z0-9:])/(?:home|Users|root|tmp|mnt|workspace)/[^\s\"']+",
            "[本机路径已移除]",
            value,
        )
    return value


def export_data(app: Any, destination: Path, include_corpus: bool = False) -> dict[str, Any]:
    """Call safe GET endpoints and write canonical response JSON under data/.

    ``include_corpus=False`` reduces full-text fields in selected corpus APIs to
    short previews or null values to avoid
    redistributing corpus content with uncertain licensing. Binary/file
    responses are preserved as returned. Calls that fail are listed in issues.
    """
    destination = Path(destination)
    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    responses: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    response_cache: dict[str, Any] = {}
    with StaticExportClient(app) as client, _frozen_clustering_reads(app.state.clustering):

        def save(key: str, response) -> dict[str, Any]:
            content_type = response.headers.get("content-type", "application/octet-stream")
            body = getattr(response, "content", getattr(response, "body", b""))
            redact_corpus = key.startswith(
                ("/api/v1/experiments/2/", "/api/v1/experiments/3/", "/api/v1/comprehensive/")
            )
            if "json" in content_type.casefold():
                try:
                    parsed = json.loads(body)
                    clean = _scrub(parsed, include_corpus, redact_corpus)
                    body = json.dumps(
                        clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                except (ValueError, TypeError):
                    pass
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            ext = (
                ".json"
                if "json" in content_type.casefold()
                else (mimetypes.guess_extension(content_type.split(";", 1)[0]) or ".bin")
            )
            target = data_dir / f"{digest}{ext}"
            target.write_bytes(body)
            item = {
                "file": f"data/{target.name}",
                "status": response.status_code,
                "content_type": content_type,
            }
            responses[key] = item
            if response.status_code >= 400:
                issues.append(
                    {
                        "key": key,
                        "error": (
                            f"HTTP {response.status_code}: {body[:300].decode('utf-8', 'replace')}"
                        ),
                    }
                )
            return item

        def get(path: str, params: dict[str, Any] | None = None, *, key: str | None = None):
            params = params or {}
            canonical = key or canonical_key(path, params)
            try:
                return client.get(path, params=params), canonical
            except Exception as exc:  # keep the export moving and report exactly what failed
                issues.append({"key": canonical, "error": f"{type(exc).__name__}: {exc}"})
                return None, canonical

        def request(
            path: str,
            params: dict[str, Any] | None = None,
            *,
            key: str | None = None,
            store: bool = True,
        ):
            canonical = key or canonical_key(path, params or {})
            response = response_cache.get(canonical)
            if response is None:
                response, canonical = get(path, params, key=canonical)
            if response is None:
                return None
            response_cache[canonical] = response
            if response.status_code >= 400:
                issues.append(
                    {
                        "key": canonical,
                        "error": (
                            f"HTTP {response.status_code}: "
                            f"{response.content[:300].decode('utf-8', 'replace')}"
                        ),
                    }
                )
                return None
            if store:
                save(canonical, response)
            try:
                return response.json()
            except ValueError:
                return None

        def pages(
            path: str,
            params: dict[str, Any],
            *,
            page_size: int = 100,
            total_key: str = "total",
            items_key: str = "items",
        ):
            first = request(path, {**params, "page": 1, "page_size": page_size}, store=False)
            if not isinstance(first, dict):
                return []
            items = list(first.get(items_key, []))
            total = int(first.get(total_key, len(items)))
            for page in range(2, max(2, (total + page_size - 1) // page_size + 1)):
                page_value = request(
                    path, {**params, "page": page, "page_size": page_size}, store=False
                )
                if not isinstance(page_value, dict):
                    break
                items.extend(page_value.get(items_key, []))
            full = {**first, "page": 1, "page_size": 100000, "items": items}
            key = canonical_key(path, params)
            save(key, _json_response(full))
            return items

        # Default public GET surface. POST inference, PATCH annotation, SSE,
        # system tests and index building are deliberately excluded.
        for path in (
            "/api/v1/health",
            "/api/v1/datasets/public/status",
            "/api/v1/course/resources",
            "/api/v1/course/modules/resources",
            "/api/v1/comprehensive/status",
            "/api/v1/comprehensive/course",
            "/api/v1/comprehensive/cases",
        ):
            request(path)
        evaluation = app.state.store.settings.outputs / "course_dependency/evaluation_latest.json"
        if evaluation.is_file():
            request("/api/v1/course/dependency/evaluation")
        else:
            issues.append(
                {
                    "key": "/api/v1/course/dependency/evaluation",
                    "error": (
                        "skipped: no persisted evaluation; "
                        "export does not invoke dependency inference"
                    ),
                }
            )

        # Experiment 1: formal runs and all paginated read-only records.
        e1 = request("/api/v1/experiments/1/runs") or {}
        e1_extensions = request("/api/v1/experiments/1/extensions") or {"items": []}
        e1_comparison = request("/api/v1/experiments/1/comparison")
        e1_items = e1.get("items", []) if isinstance(e1, dict) else []
        publish_e1 = [
            run
            for run in e1_items
            if run.get("run_type") in {"frozen-final-benchmark", "course_samples"}
        ]
        for run in publish_e1:
            run_id = run.get("run_id")
            if not run_id:
                continue
            request(
                "/api/v1/models/hmm/parameters",
                {"run_id": run_id, "model": "hmm1-smooth", "topk": 100},
            )
        for model in ("longest", "bigram", "hmm2-smooth"):
            request("/api/v1/models/hmm/parameters", {"model": model, "topk": 100})
        for run in publish_e1:
            run_id = run.get("run_id")
            if not run_id:
                continue
            base = {"run_id": run_id}
            request("/api/v1/experiments/1/metrics", base)
            pages("/api/v1/experiments/1/errors", base)
        for model in (
            "longest",
            "bigram",
            "hmm1-smooth",
            "hmm1-mle",
            "hmm2-smooth",
            "hmm2-mle",
            "bilstm-crf",
        ):
            pages("/api/v1/views/cws/errors", {"model": model})

        # Experiment 2: fixed runs only; skip cache-only views when their
        # display cache is absent so GET cannot create a cache as a side effect.
        e2 = request("/api/v1/experiments/2/runs") or {}
        e2_items = (
            e2.get("items", []) if isinstance(e2, dict) else e2 if isinstance(e2, list) else []
        )
        for k in range(2, 11):
            request("/api/v1/experiments/2/summary", {"k": k})
            request("/api/v1/experiments/2/comparison-view", {"k": k})
        for run in e2_items:
            run_id = run.get("run_id") if isinstance(run, dict) else None
            if not run_id:
                continue
            request("/api/v1/experiments/2/overview", {"run_id": run_id})
            for k in range(2, 11):
                request("/api/v1/experiments/2/result", {"run_id": run_id, "k": k})
            cache_dir = app.state.clustering.settings.outputs / "exp2/visualizations" / run_id
            inspection_cache = cache_dir / "inspection.json"
            if inspection_cache.is_file():
                for k in range(2, 11):
                    request("/api/v1/experiments/2/inspection", {"run_id": run_id, "k": k})
            else:
                issues.append(
                    {
                        "key": f"/api/v1/experiments/2/inspection?run_id={run_id}",
                        "error": (
                            "skipped: existing inspection cache is absent; "
                            "export does not create caches"
                        ),
                    }
                )
            request("/api/v1/experiments/2/projection", {"run_id": run_id, "dimensions": 2})
            # dimensions=3 is cache_only; preflight only the expected cache path.
            cache_dir = app.state.clustering.settings.outputs / "exp2/visualizations" / run_id
            if (cache_dir / "projection3d.json").is_file():
                request("/api/v1/experiments/2/projection", {"run_id": run_id, "dimensions": 3})
            # Reuse the existing inspection cache data to enumerate document IDs;
            # do not call the API if doing so would trigger cache generation.
            latest_e2 = e2.get("latest") if isinstance(e2, dict) else None
            primary_e2 = e2.get("primary") if isinstance(e2, dict) else None
            if inspection_cache.is_file() and (
                run_id in {latest_e2, primary_e2}
                or (latest_e2 is None and e2_items and run_id == e2_items[-1].get("run_id"))
            ):
                try:
                    ids = [
                        doc["id"]
                        for doc in json.loads(inspection_cache.read_text(encoding="utf-8")).get(
                            "documents", []
                        )
                    ]
                    for document_id in ids:
                        request(
                            "/api/v1/experiments/2/document",
                            {"run_id": run_id, "k": 5, "document_id": document_id},
                        )
                except (OSError, ValueError, KeyError) as exc:
                    issues.append(
                        {
                            "key": f"/api/v1/experiments/2/document?run_id={run_id}&k=5",
                            "error": f"cannot enumerate existing inspection cache: {exc}",
                        }
                    )

        # Experiment 3: frozen classification Runs. Endpoints internally verify
        # frozen manifests but do not write. No prediction POSTs are made.
        e3 = request("/api/v1/experiments/3/runs") or {}
        e3_items = (
            e3.get("items", []) if isinstance(e3, dict) else e3 if isinstance(e3, list) else []
        )
        stage4_runs = request("/api/v1/experiments/3/stage4/runs") or []
        for stage_run in stage4_runs:
            stage_run_id = stage_run.get("run_id") if isinstance(stage_run, dict) else None
            if stage_run_id:
                request("/api/v1/experiments/3/stage4/overview", {"run_id": stage_run_id})
                for name in (
                    "protocol",
                    "subsets",
                    "curves",
                    "raw",
                    "per-class",
                    "diagnostics",
                    "runtimes",
                    "manifest",
                ):
                    request(f"/api/v1/experiments/3/stage4/export/{name}", {"run_id": stage_run_id})
        request("/api/v1/experiments/3/stage5/overview")
        for model_id in ("C0", "C1", "C2", "C3"):
            request("/api/v1/experiments/3/stage5/flips", {"model_id": model_id})
        request("/api/v1/experiments/3/stage6/overview")
        for stage, names in {
            "stage5/": ("results", "flips", "rules", "prevalence", "manifest", "protocol"),
            "stage6/": ("protocol", "statistics", "alignment", "manifest", "bootstrap", "aligned"),
        }.items():
            for name in names:
                request(f"/api/v1/experiments/3/{stage}export/{name}")
        for run in e3_items:
            run_id = run.get("run_id") if isinstance(run, dict) else None
            if not run_id:
                continue
            base = {"run_id": run_id}
            overview = request("/api/v1/experiments/3/overview", base) or {}
            available = set(overview.get("available_models", overview.get("test", {})))
            for model in ("C0", "C1"):
                request("/api/v1/experiments/3/predictions", {**base, "model_id": model})
                request("/api/v1/experiments/3/features", {**base, "model_id": model})
            for stage, models in (("stage2", ("C2",)), ("stage3", ("C3",))):
                if not available.intersection(models):
                    continue
                request(f"/api/v1/experiments/3/{stage}/predictions", base)
                request(f"/api/v1/experiments/3/{stage}/features", base)
                request(f"/api/v1/experiments/3/{stage}/paired", base)
            for stage, names in {
                "": ("metrics", "validation", "manifest", "config", "environment", "dataset"),
                "stage2/": ("metrics", "validation", "manifest", "config", "paired", "coverage"),
                "stage3/": (
                    "metrics",
                    "validation",
                    "manifest",
                    "config",
                    "preregistration",
                    "selection",
                    "paired-c1",
                    "paired-c2",
                    "representation",
                    "runtime",
                ),
            }.items():
                if stage == "stage2/" and "C2" not in available:
                    continue
                if stage == "stage3/" and "C3" not in available:
                    continue
                for name in names:
                    request(f"/api/v1/experiments/3/{stage}export/{name}", base)
            # Test IDs are obtained from predictions already exported above.
            test_ids: set[str] = set()
            for model in ("C0", "C1"):
                key = canonical_key(
                    "/api/v1/experiments/3/predictions", {**base, "model_id": model}
                )
                file = responses.get(key, {}).get("file")
                if file:
                    try:
                        test_ids.update(
                            row.get("document_id", row.get("sample_id", row.get("id")))
                            for row in json.loads((destination / file).read_text(encoding="utf-8"))
                            if row.get("document_id", row.get("sample_id", row.get("id")))
                        )
                    except (OSError, ValueError, TypeError):
                        pass
            for document_id in sorted(test_ids):
                request(
                    "/api/v1/experiments/3/workbench/document",
                    {
                        "document_id": document_id,
                        "run_id": run_id,
                        "source": "test",
                        "condition": "OA",
                    },
                )
                for model_id in ("C1", "C2"):
                    if model_id not in available:
                        continue
                    request(
                        "/api/v1/experiments/3/visualization/decision",
                        {"document_id": document_id, "run_id": run_id, "model_id": model_id},
                    )
                # This GET starts a Python TextCNN forward when its in-memory
                # cache is cold. Export only persisted evidence; never call it.
                # Shortcut records use Stage5 Validation IDs, not formal Test IDs.

        # Comprehensive fixed course collection and pre-existing cases only.
        course = request("/api/v1/comprehensive/course")
        if isinstance(course, dict):
            collection_id = course.get("id") or course.get("collection", {}).get("id")
            documents = course.get("documents", course.get("collection", {}).get("documents", []))
            if collection_id:
                request(f"/api/v1/comprehensive/collections/{collection_id}/workspace")
                for document in documents:
                    doc_id = document.get("id")
                    if doc_id:
                        request(
                            f"/api/v1/comprehensive/collections/{collection_id}/documents/{doc_id}",
                            {"analyze": "false"},
                        )

    issues.append(
        {
            "key": "/api/v1/experiments/3/visualization/c3-trace",
            "error": (
                "excluded: this GET can execute a new TextCNN forward; static export is read-only"
            ),
        }
    )
    models = {
        "experiment1": {
            "runs": [r.get("run_id") for r in publish_e1],
            "extensions": e1_extensions.get("items", [])
            if isinstance(e1_extensions, dict)
            else e1_extensions,
            "comparison": e1_comparison,
        },
        "experiment2": {"runs": e2_items},
        "experiment3": {"runs": e3_items, "stage4_runs": stage4_runs},
    }
    corpus_policy = (
        "Default export truncates corpus text fields to a 160-codepoint preview "
        "(or nulls original_text), marks corpus_text_redacted, and removes local "
        "machine paths/private keys. Set include_corpus=True only after confirming "
        "redistribution rights; local paths and private keys remain scrubbed."
    )
    defaults = _scrub(DEFAULTS, True, False)
    models = _scrub(models, True, False)
    issues = _scrub(issues, True, False)
    issues = list({(item["key"], item["error"]): item for item in issues}.values())
    corpus_policy = _scrub(corpus_policy, True, False)
    manifest = {
        "schema_version": 1,
        "include_corpus": include_corpus,
        "corpus_policy": corpus_policy,
        "defaults": defaults,
        "models": models,
        "responses": responses,
        "issues": issues,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {
        "responses": responses,
        "issues": issues,
        "defaults": defaults,
        "models": models,
        "include_corpus": include_corpus,
        "corpus_policy": corpus_policy,
    }


def _json_response(value: Any):
    """Create the tiny response-like object accepted by the shared save helper."""
    from starlette.responses import JSONResponse

    response = JSONResponse(value)
    return type(
        "SavedResponse",
        (),
        {"content": response.body, "headers": response.headers, "status_code": 200},
    )()


@contextmanager
def _frozen_clustering_reads(service):
    """Share verified frozen arrays during export instead of parsing per document."""
    inspection, calculation, result = (
        service._inspection_cache,
        service._document_calculation,
        service.result,
    )
    calculations = {}

    def document_calculation(run_id, document_id):
        if run_id not in calculations:
            # The original reader checks the cache hash and every source hash.
            # Only after that succeeds do we retain the complete immutable file.
            first = calculation(run_id, document_id)
            if first is None:
                calculations[run_id] = None
            else:
                path = (
                    service.settings.outputs / "exp2/visualizations" / run_id / "calculation.json"
                )
                calculations[run_id] = json.loads(path.read_text(encoding="utf-8"))
        value = calculations[run_id]
        if value is None:
            return None
        return {
            **value["documents"][document_id],
            "document_count": value["document_count"],
            "config": value["config"],
            "provenance": value["provenance"],
        }

    service._inspection_cache = lru_cache(maxsize=3)(inspection)
    service.result = lru_cache(maxsize=32)(result)
    service._document_calculation = document_calculation
    try:
        yield
    finally:
        service._inspection_cache, service._document_calculation, service.result = (
            inspection,
            calculation,
            result,
        )


if __name__ == "__main__":
    from app.main import app

    result = export_data(app, Path("build/static-data-test"))
    print(
        json.dumps(
            {"responses": len(result["responses"]), "issues": len(result["issues"])},
            ensure_ascii=False,
        )
    )
