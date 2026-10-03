"""读取冻结聚类 Run 和只读检视缓存；Web 不触发模型或显示投影拟合。"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from nlplab.core.exceptions import DataValidationError

router = APIRouter(prefix="/api/v1/experiments/2")


@router.get("/runs")
def runs(request: Request):
    return request.app.state.clustering.list_runs()


@router.get("/overview")
def overview(request: Request, run_id: str = "latest"):
    return request.app.state.clustering.overview(run_id)


@router.get("/summary")
def summary(request: Request, k: int = Query(5, ge=2, le=10)):
    return request.app.state.clustering.final_summary(k)


@router.get("/result")
def result(request: Request, run_id: str = "latest", k: int = Query(5, ge=2, le=10)):
    return request.app.state.clustering.result(run_id, k)


@router.get("/projection")
def projection(request: Request, run_id: str = "latest", dimensions: int = Query(2, ge=2, le=3)):
    return request.app.state.clustering.projection(run_id, dimensions, cache_only=True)


@router.get("/inspection")
def inspection(request: Request, run_id: str = "latest", k: int = Query(5, ge=2, le=10)):
    return request.app.state.clustering.inspection(run_id, k)


@router.get("/document")
def document(
    request: Request,
    run_id: str = "latest",
    k: int = Query(5, ge=2, le=10),
    document_id: str = Query(..., min_length=1, max_length=128),
):
    return request.app.state.clustering.document(run_id, k, document_id)


@router.get("/comparison-view")
def comparison_view(request: Request, k: int = Query(5, ge=2, le=10)):
    return request.app.state.clustering.comparison_view(k)


@router.get("/export/{name}")
def export(request: Request, name: str, run_id: str = "latest"):
    allowed = {
        "sweep": "sweep.json",
        "config": "config.json",
        "projection": "projection.json",
        "reference": "reference/k5.json",
        "manifest": "artifact_manifest.json",
        "stability": "stability.json",
        "environment": "environment.json",
    }
    if name not in allowed:
        raise DataValidationError("不支持的聚类导出文件")
    path = request.app.state.clustering.store.resolve(run_id) / allowed[name]
    return FileResponse(path, media_type="application/json", filename=f"{run_id}-{name}.json")
