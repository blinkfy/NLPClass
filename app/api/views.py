"""只读课程页面聚合；复用既有 RunStore、状态和错误文件读取。"""

from typing import Literal

from fastapi import APIRouter, Query, Request

from app.api.experiment import extension_status, read_errors
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.evaluation.errors import CATEGORIES

router = APIRouter(prefix="/api/v1/views")


@router.get("/cws/errors")
def cws_errors(
    request: Request,
    model: Literal[
        "longest", "bigram", "hmm1-smooth", "hmm1-mle", "hmm2-smooth", "hmm2-mle", "bilstm-crf"
    ] = "hmm1-smooth",
    run_id: str | None = None,
    category: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
) -> dict:
    if category and category not in CATEGORIES:
        raise DataValidationError("无效错误类别")
    store = request.app.state.store
    if model.startswith("hmm1"):
        candidates = [
            r
            for r in store.list_runs()
            if r.get("run_type") == "frozen-final-benchmark"
            and r["run_id"].endswith("_smooth" if model.endswith("smooth") else "_mle")
        ]
        selected = run_id or (candidates[0]["run_id"] if candidates else None)
        if selected is None:
            raise ResourceUnavailableError("尚无 CWS Run")
        path = store.resolve(selected)
        info = store.read(selected, "run.json")
        if info.get("run_type") == "course_samples":
            raise ResourceUnavailableError("MSRA 课程样本没有 CWS Gold 错误分析")
        if info.get("run_type") != "frozen-final-benchmark" or selected not in {
            candidate["run_id"] for candidate in candidates
        }:
            raise DataValidationError("CWS 错误展示必须使用对应模型的冻结 MSR Run")
    else:
        matches = [r for r in extension_status(request)["items"] if r["model"] == model]
        if not matches:
            raise ResourceUnavailableError("尚无该模型 final Run")
        info = matches[-1]
        path = (store.directory / "extensions" / info["run_id"]).resolve()
        if not path.is_relative_to((store.directory / "extensions").resolve()):
            raise ResourceUnavailableError("无效 Run 路径")
    if not (path / "errors.jsonl").is_file():
        raise ResourceUnavailableError("该 Run 未保存错误案例")
    items = read_errors(path)
    if category in CATEGORIES:
        items = [r for r in items if (r.get("manual_category") or r["category"]) == category]
    start = (page - 1) * page_size
    return {
        "run": info,
        "items": items[start : start + page_size],
        "total": len(items),
        "page": page,
        "page_size": page_size,
        "read_only": True,
        "evaluation_task": "cws",
    }
