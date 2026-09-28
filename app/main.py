"""应用工厂；算法实现全部位于 nlplab 包。"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import analysis, annotation, experiment, views
from nlplab import __version__
from nlplab.core.config import Settings, default_settings
from nlplab.core.exceptions import DataValidationError, NoValidPathError, ResourceUnavailableError
from nlplab.data.sequence_annotations import AnnotationStore
from nlplab.ner.hanlp_adapter import HanLPRecognizer
from nlplab.services.experiment_service import RunStore
from nlplab.services.nlp_service import NLPService


def create_app(settings: Settings | None = None, service: NLPService | None = None) -> FastAPI:
    """创建可用于开发和 API 测试的应用。"""
    application = FastAPI(title="NLPLab", version=__version__)
    settings = settings or default_settings()
    application.state.store = RunStore(settings)
    application.state.annotations = AnnotationStore(settings.root)
    application.state.nlp = service or NLPService(
        settings, ner=HanLPRecognizer.from_settings(settings)
    )
    application.include_router(analysis.router)
    application.include_router(experiment.router)
    application.include_router(annotation.router)
    application.include_router(views.router)
    resources = Path(__file__).parent
    application.mount("/static", StaticFiles(directory=resources / "static"), name="static")
    templates = Jinja2Templates(directory=resources / "templates")

    @application.exception_handler(ResourceUnavailableError)
    async def unavailable(request: Request, error: ResourceUnavailableError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "resource_unavailable", "message": str(error)}}, status_code=503
        )

    @application.exception_handler(DataValidationError)
    @application.exception_handler(NoValidPathError)
    async def invalid(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "invalid_data", "message": str(error)}}, status_code=422
        )

    pages = {
        "/": ("index.html", "首页"),
        "/segmentation": ("segmentation.html", "中文分词"),
        "/sequence-labeling": ("sequence_labeling.html", "序列标注"),
        "/tagging": ("tagging.html", "词性与命名实体"),
        "/information-extraction": ("future.html", "信息抽取"),
        "/clustering": ("future.html", "文本聚类"),
        "/classification": ("future.html", "文本分类"),
        "/dependency-parsing": ("future.html", "依存句法分析"),
        "/deep-learning": ("deep_learning.html", "深度学习与大模型"),
        "/integrated-analysis": ("analysis.html", "综合分析"),
        "/annotation/sequence": ("annotation_sequence.html", "人工 Sequence Gold"),
    }
    tabs = {
        "/segmentation": [
            ("interactive", "交互分析"),
            ("process", "算法过程"),
            ("evaluation", "实验评测"),
            ("errors", "错误分析"),
        ],
        "/sequence-labeling": [
            ("overview", "模型概览"),
            ("hmm", "HMM"),
            ("perceptron", "感知机"),
            ("crf", "CRF"),
            ("comparison", "模型对比"),
        ],
        "/tagging": [
            ("pos", "词性标注"),
            ("ner", "命名实体识别"),
            ("evaluation", "实体评测"),
            ("errors", "错误分析"),
        ],
        "/information-extraction": [
            ("interactive", "交互抽取"),
            ("relations", "实体与关系"),
            ("results", "抽取结果"),
            ("evaluation", "实验评测"),
        ],
        "/clustering": [
            ("data", "数据与表示"),
            ("clustering", "聚类分析"),
            ("visualization", "可视化"),
            ("evaluation", "实验评测"),
        ],
        "/classification": [
            ("data", "数据与表示"),
            ("prediction", "分类预测"),
            ("comparison", "模型对比"),
            ("errors", "错误分析"),
        ],
        "/dependency-parsing": [
            ("interactive", "交互分析"),
            ("tree", "依存树"),
            ("process", "算法过程"),
            ("evaluation", "实验评测"),
        ],
        "/deep-learning": [
            ("bilstm", "BiLSTM-CRF"),
            ("transformer", "Transformer"),
            ("llm", "大语言模型"),
            ("extensions", "扩展应用"),
        ],
        "/integrated-analysis": [
            ("single", "单文本综合分析"),
            ("documents", "多文档分析"),
            ("overview", "全课程结果总览"),
        ],
    }

    def make_page(template: str, title: str):
        def page(request: Request):
            return templates.TemplateResponse(
                request=request,
                name=template,
                context={
                    "title": title,
                    "page": request.url.path,
                    "module_tabs": tabs.get(request.url.path, []),
                },
            )

        return page

    for route, (template, title) in pages.items():
        application.add_api_route(
            route, make_page(template, title), methods=["GET"], include_in_schema=False
        )

    @application.get("/sequence", include_in_schema=False)
    def legacy_sequence():
        return RedirectResponse("/segmentation", status_code=307)

    @application.get("/analysis", include_in_schema=False)
    def legacy_analysis():
        return RedirectResponse("/integrated-analysis", status_code=307)

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return application


app = create_app()
