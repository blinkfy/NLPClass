import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.evaluation.errors import CATEGORIES

router = APIRouter(prefix="/api/v1")


@router.get("/datasets/public/status")
def public_data_status(request: Request) -> dict:
    from nlplab.data.public_audit import audit_public

    result = audit_public(request.app.state.store.settings.root)
    return {
        name: {
            "status": item["status"],
            "count": item.get("manifest", {}).get("count", 0),
            "category_counts": item.get("category_counts"),
            "split_counts": {
                key: value["count"]
                for key, value in item.get("split_manifest", {}).get("splits", {}).items()
            },
        }
        for name, item in result["datasets"].items()
    }


def read_errors(path: Path) -> list[dict]:
    errors = [
        json.loads(line)
        for line in (path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    sidecar = path / "error_annotations.json"
    categories = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    for error in errors:
        error["manual_category"] = categories.get(error["sample_id"])
    return errors


@router.get("/experiments/1/runs")
def runs(request: Request) -> dict:
    store = request.app.state.store
    items = store.list_runs()
    return {"items": items, "latest": store.resolve().name if items else None}


@router.get("/models/hmm/parameters")
def parameters(
    request: Request,
    run_id: str = "latest",
    topk: int = Query(20, ge=1, le=100),
    model: str = "hmm1-smooth",
) -> dict:
    if model != "hmm1-smooth":
        run_path, info = request.app.state.nlp.extension_run(model)
        segmenter = request.app.state.nlp._load_extension_segmenter(model)
        return {
            "run": info,
            "parameters": segmenter.parameters(topk)
            if model == "hmm2-smooth"
            else segmenter.parameters(),
            "training_history": json.loads(
                (run_path / "training_history.json").read_text(encoding="utf-8")
            )
            if (run_path / "training_history.json").is_file()
            else None,
        }
    store = request.app.state.store
    path = store.resolve(run_id)
    return {
        "run": store.read(path.name, "run.json"),
        "parameters": store.model(path.name).parameters(topk),
    }


@router.get("/experiments/1/metrics")
def metrics(request: Request, run_id: str = "latest") -> dict:
    store = request.app.state.store
    path = store.resolve(run_id)
    return {
        "run": store.read(path.name, "run.json"),
        "metrics": store.read(path.name, "metrics.json"),
    }


@router.get("/experiments/1/extensions")
def extension_status(request: Request) -> dict:
    """只返回拓展 Run 元数据，不加载神经模型权重。"""
    from importlib.util import find_spec

    root = request.app.state.store.settings.outputs / "exp1/extensions"
    items = []
    for path in sorted(root.glob("*/run.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("phase") == "final":
            model_path = Path(info["model_path"])
            reason = None
            if not model_path.is_dir():
                reason = "模型资源缺失"
            elif info["model"] == "bilstm-crf" and (
                find_spec("torch") is None or find_spec("torchcrf") is None
            ):
                reason = "当前 Web 环境缺少 torch / pytorch-crf"
            items.append({**info, "available": reason is None, "unavailable_reason": reason})
    return {"items": items}


@router.get("/experiments/1/comparison")
def comparison(request: Request) -> dict:
    """统一 CWS final 表；MSRA NER 不混入。"""
    from app.api.comparison_display import with_derived_tags
    from nlplab.extensions.comparison import collect

    root = request.app.state.store.settings.root
    return with_derived_tags(collect(root), root)


@router.get("/experiments/1/trace")
def trace(
    request: Request,
    text: str = Query(min_length=1, max_length=30),
    model: Literal["hmm1-smooth", "longest", "bigram", "hmm2-smooth", "bilstm-crf"] = "hmm1-smooth",
    run_id: str = "latest",
) -> dict:
    from app.api.trace_display import display_trace
    from nlplab.extensions.traces import algorithm_trace

    service = request.app.state.nlp
    result = algorithm_trace(service, text, model, run_id)
    dictionary = service._load_extension_segmenter(model).words if model == "longest" else None
    return display_trace(result, text, model, dictionary)


@router.get("/experiments/1/errors")
def errors(
    request: Request,
    run_id: str = "latest",
    category: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
) -> dict:
    store = request.app.state.store
    path = store.resolve(run_id)
    items = read_errors(path)
    if category:
        if category not in CATEGORIES:
            raise DataValidationError("错误类别无效")
        items = [
            item for item in items if (item["manual_category"] or item["category"]) == category
        ]
    start = (page - 1) * page_size
    return {
        "run": store.read(path.name, "run.json"),
        "total": len(items),
        "page": page,
        "page_size": page_size,
        "items": items[start : start + page_size],
    }


class ManualCategory(BaseModel):
    category: (
        Literal["OOV", "Proper noun", "Ambiguity", "Number/date", "English mixed", "Other"] | None
    )


@router.patch("/experiments/1/errors/{sample_id}")
def annotate_error(
    sample_id: str, body: ManualCategory, request: Request, run_id: str = "latest"
) -> dict:
    path = request.app.state.store.resolve(run_id)
    info = request.app.state.store.read(path.name, "run.json")
    if info.get("read_only") or info.get("run_type") == "frozen-final-benchmark":
        raise DataValidationError("冻结 benchmark/课程正式 Run 只读")
    if sample_id not in {item["sample_id"] for item in read_errors(path)}:
        raise DataValidationError("错误 sample_id 不存在")
    sidecar = path / "error_annotations.json"
    categories = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    categories[sample_id] = body.category
    temporary = path / "error_annotations.tmp"
    write_json(temporary, categories)
    temporary.replace(sidecar)
    return {"sample_id": sample_id, "manual_category": body.category}


@router.get("/experiments/1/export/{name}")
def export(
    name: Literal["metrics", "predictions", "errors", "config"],
    request: Request,
    run_id: str = "latest",
) -> FileResponse:
    path = request.app.state.store.resolve(run_id)
    filename = {
        "metrics": "metrics.json",
        "predictions": "predictions.jsonl",
        "errors": "errors.jsonl",
        "config": "config.yaml",
    }[name]
    return FileResponse(path / filename, filename=f"{path.name}_{filename}")
