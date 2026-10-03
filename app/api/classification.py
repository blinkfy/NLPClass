"""分类 API 只展示冻结结果和预测，禁止从 Web 拟合模型。"""

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nlplab.core.exceptions import DataValidationError
from nlplab.services.classification_stage3_service import ClassificationStage3Service

router = APIRouter(prefix="/api/v1/experiments/3")


def stage2_service(request):
    # 测试或插件注入的 Stage1 service 继续原样使用；真实服务仅扩展 dispatch。
    service = request.app.state.classification
    if not hasattr(service, "settings"):
        return service
    return ClassificationStage3Service(service.settings)


class PredictionInput(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_id: Literal["C0", "C1"] = "C0"
    run_id: str = "latest"


@router.get("/runs")
def runs(request: Request):
    return request.app.state.classification.list_runs()


@router.get("/overview")
def overview(request: Request, run_id: str = "latest"):
    return stage2_service(request).overview(run_id)


@router.post("/predict")
def predict(request: Request, payload: PredictionInput):
    return stage2_service(request).predict(payload.text, payload.model_id, payload.run_id)


@router.get("/predictions")
def predictions(
    request: Request,
    run_id: str = "latest",
    model_id: Literal["C0", "C1"] = "C0",
    errors_only: bool = False,
):
    return stage2_service(request).predictions(run_id, model_id, errors_only)


@router.get("/features")
def features(request: Request, run_id: str = "latest", model_id: Literal["C0", "C1"] = "C0"):
    return stage2_service(request).features(run_id, model_id)


@router.get("/export/{name}")
def export(request: Request, name: str, run_id: str = "latest"):
    allowed = {
        "metrics": "test_metrics.json",
        "validation": "validation_selection.json",
        "manifest": "artifact_manifest.json",
        "config": "config.json",
        "environment": "environment.json",
        "dataset": "dataset.json",
    }
    if name not in allowed:
        raise DataValidationError("不支持的分类导出文件")
    service = request.app.state.classification
    service.verify_artifacts(run_id)
    return FileResponse(
        service.store.resolve(run_id) / allowed[name],
        media_type="application/json",
        filename=f"exp3-{name}.json",
    )


class C2PredictionInput(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_id: Literal["C2"] = "C2"
    run_id: str = "latest"


@router.post("/stage2/predict")
def c2_predict(request: Request, payload: C2PredictionInput):
    return stage2_service(request).predict(payload.text, "C2", payload.run_id)


@router.get("/stage2/predictions")
def c2_predictions(request: Request, run_id: str = "latest", errors_only: bool = False):
    return stage2_service(request).predictions(run_id, "C2", errors_only)


@router.get("/stage2/features")
def c2_features(request: Request, run_id: str = "latest"):
    return stage2_service(request).features(run_id, "C2")


@router.get("/stage2/paired")
def paired(request: Request, run_id: str = "latest"):
    return stage2_service(request).paired(run_id)


@router.get("/stage2/export/{name}")
def c2_export(request: Request, name: str, run_id: str = "latest"):
    allowed = {
        "metrics": "comparison_metrics.json",
        "validation": "validation_selection.json",
        "manifest": "artifact_manifest.json",
        "config": "config.json",
        "paired": "paired_c1_c2.json",
        "coverage": "coverage.json",
    }
    if name not in allowed:
        raise DataValidationError("不支持的 C2 导出文件")
    service = stage2_service(request)
    directory = service.stage2_directory(run_id)
    service.verify_artifacts(directory.name)
    return FileResponse(
        directory / allowed[name],
        media_type="application/json",
        filename=f"exp3-stage2-{name}.json",
    )


class C3PredictionInput(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_id: Literal["C3"] = "C3"
    run_id: str = "latest"


@router.post("/stage3/predict")
def c3_predict(request: Request, payload: C3PredictionInput):
    return stage2_service(request).predict(payload.text, "C3", payload.run_id)


@router.get("/stage3/predictions")
def c3_predictions(request: Request, run_id: str = "latest", errors_only: bool = False):
    return stage2_service(request).predictions(run_id, "C3", errors_only)


@router.get("/stage3/features")
def c3_features(request: Request, run_id: str = "latest"):
    return stage2_service(request).features(run_id, "C3")


@router.get("/stage3/paired")
def c3_paired(request: Request, run_id: str = "latest", old_model: Literal["C1", "C2"] = "C2"):
    return stage2_service(request).paired_stage3(run_id, old_model)


@router.get("/stage3/export/{name}")
def c3_export(request: Request, name: str, run_id: str = "latest"):
    allowed = {
        "metrics": "comparison_metrics.json",
        "validation": "validation_selection.json",
        "manifest": "artifact_manifest.json",
        "config": "config.json",
        "preregistration": "preregistration.json",
        "selection": "selection_freeze.json",
        "paired-c1": "paired_c1_c3.json",
        "paired-c2": "paired_c2_c3.json",
        "representation": "representation_stats.json",
        "runtime": "runtime.json",
    }
    if name not in allowed:
        raise DataValidationError("不支持的 C3 导出文件")
    service = stage2_service(request)
    directory = service.stage3_directory(run_id)
    service.verify_artifacts(directory.name)
    return FileResponse(
        directory / allowed[name],
        media_type="application/json",
        filename=f"exp3-stage3-{name}.json",
    )


def stage4_service(request):
    from nlplab.services.classification_stage4_service import ClassificationStage4Service

    return ClassificationStage4Service(request.app.state.classification.settings)


@router.get("/stage4/runs")
def stage4_runs(request: Request):
    return stage4_service(request).list_runs()


@router.get("/stage4/overview")
def stage4_overview(
    request: Request,
    run_id: str = "latest",
    model_id: Literal["C0", "C1", "C2", "C3"] | None = None,
):
    return stage4_service(request).overview(run_id, model_id)


@router.get("/stage4/export/{name}")
def stage4_export(request: Request, name: str, run_id: str = "latest"):
    return FileResponse(
        stage4_service(request).export_path(run_id, name), media_type="application/json"
    )


def stage5_service(request):
    from nlplab.services.classification_stage5_service import ClassificationStage5Service

    return ClassificationStage5Service(request.app.state.classification.settings)


@router.get("/stage5/overview")
def stage5_overview(request: Request, run_id: str = "latest"):
    return stage5_service(request).overview(run_id)


@router.get("/stage5/flips")
def stage5_flips(
    request: Request,
    run_id: str = "latest",
    model_id: Literal["C0", "C1", "C2", "C3"] | None = None,
    condition: Literal["OS", "OT", "OA", "AA"] | None = None,
):
    return stage5_service(request).flips(run_id, model_id, condition)


@router.get("/stage5/export/{name}")
def stage5_export(request: Request, name: str, run_id: str = "latest"):
    return FileResponse(
        stage5_service(request).export_path(run_id, name), media_type="application/json"
    )


def stage6_service(request):
    from nlplab.services.classification_stage6_service import ClassificationStage6Service

    return ClassificationStage6Service(request.app.state.classification.settings)


@router.get("/stage6/overview")
def stage6_overview(request: Request, run_id: str = "latest"):
    return stage6_service(request).overview(run_id)


@router.get("/stage6/export/{name}")
def stage6_export(request: Request, name: str, run_id: str = "latest"):
    service = stage6_service(request)
    if name == "aligned":
        from fastapi.responses import JSONResponse

        return JSONResponse(service.aligned_labels(run_id))
    return FileResponse(service.export_path(run_id, name), media_type="application/json")


@router.get("/workbench/document")
def workbench_document(
    request: Request,
    document_id: str,
    run_id: str = "latest",
    source: Literal["test", "stage5"] = "test",
    condition: Literal["OS", "OT", "OA", "AA"] = "OA",
):
    from nlplab.services.classification_workbench_service import ClassificationWorkbenchService

    return ClassificationWorkbenchService(request.app.state.classification.settings).document(
        run_id, document_id, source, condition
    )


@router.get("/visualization/decision")
def visualization_decision(request: Request, document_id: str, run_id: str, model_id: Literal["C1", "C2"]):
    from nlplab.services.classification_visualization_service import ClassificationVisualizationService
    return ClassificationVisualizationService(request.app.state.classification.settings).decision(run_id, document_id, model_id)


@router.get("/visualization/c3-trace")
def visualization_c3_trace(request: Request, document_id: str, run_id: str, kernel: int = 3, filter_id: int = 0):
    from nlplab.services.classification_visualization_service import ClassificationVisualizationService
    return ClassificationVisualizationService(request.app.state.classification.settings).trace(run_id, document_id, kernel, filter_id)


@router.get("/visualization/shortcut-document")
def visualization_shortcut_document(request: Request, document_id: str, model_id: Literal["C0", "C1", "C2", "C3"], run_id: str = "latest"):
    from nlplab.services.classification_shortcut_visualization_service import ClassificationShortcutVisualizationService
    return ClassificationShortcutVisualizationService(request.app.state.classification.settings).document(run_id, document_id, model_id)
