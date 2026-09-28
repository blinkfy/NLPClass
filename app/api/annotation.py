"""本地人工标注 API；建议端点不保存任何 Gold。"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from nlplab.core.exceptions import NLPLabError, ResourceUnavailableError
from nlplab.data.course import file_sha256, load_plan
from nlplab.data.sequence_annotations import DraftConflictError

router = APIRouter(prefix="/api/v1/annotation/sequence", tags=["manual annotation"])


class GoldEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    type: str
    start: StrictInt
    end: StrictInt


class DraftUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: StrictInt = Field(ge=0)
    boundaries: list[StrictInt] | None = None
    entities: list[GoldEntity] | None = None
    notes: str | None = None
    status: str | None = None
    review_confirmation: bool = False


@router.get("")
def queue(
    request: Request,
    label: str | None = None,
    source_type: str | None = None,
    status: str | None = None,
) -> dict:
    return request.app.state.annotations.queue(label, source_type, status)


@router.get("/{sample_id}")
def sample(sample_id: str, request: Request) -> dict:
    return request.app.state.annotations.get(sample_id)


@router.patch("/{sample_id}")
def update(sample_id: str, body: DraftUpdate, request: Request) -> dict:
    value = body.model_dump(exclude_unset=True)
    version = value.pop("expected_version")
    try:
        return request.app.state.annotations.save(sample_id, value, version)
    except DraftConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/{sample_id}/suggestions")
def suggestions(sample_id: str, request: Request) -> dict:
    store = request.app.state.annotations
    value = store.get(sample_id)
    result = {"notice": "参考建议，非 Gold", "writes_gold": False}
    settings = load_plan(store.config).get("suggestions", {})
    try:
        from nlplab.segmentation.hmm_segmenter import HMMSegmenter
        from nlplab.sequence_labeling.hmm import HMMTagger

        path, freeze_path = settings.get("hmm_model_path"), settings.get("hmm_freeze_path")
        if not path or not freeze_path:
            raise ResourceUnavailableError("未配置 frozen Smooth 模型和冻结指纹；人工标注仍可使用")
        import json

        model_path, frozen = store.root / Path(path), store.root / Path(freeze_path)
        hashes = json.loads(frozen.read_text(encoding="utf-8"))["models"]["smooth"]
        if not all(file_sha256(model_path / name) == digest for name, digest in hashes.items()):
            raise ResourceUnavailableError("模型与冻结 Smooth 指纹不一致")
        model = HMMTagger.load(model_path)
        if model.config.smoothing != "add_alpha" or model.config.alpha != 0.1:
            raise ResourceUnavailableError("参考模型必须是冻结的 Smooth α=0.1")
        segmentation = HMMSegmenter(model).segment(value["text"])
        result["hmm"] = {"status": "ok", "tokens": segmentation.tokens, "tags": segmentation.tags}
    except (NLPLabError, OSError, ValueError, KeyError) as error:
        result["hmm"] = {"status": "unavailable", "error": str(error)}
    try:
        ner = request.app.state.nlp.ner
        if ner is None:
            raise ResourceUnavailableError("HanLP NER 未准备；人工标注仍可使用")
        entities = ner.recognize(value["text"])
        result["ner"] = {
            "status": "ok",
            "entities": [
                {"text": e.text, "type": e.type, "start": e.start, "end": e.end} for e in entities
            ],
        }
    except (NLPLabError, OSError, ValueError, ImportError, RuntimeError) as error:
        result["ner"] = {"status": "unavailable", "error": str(error)}
    return result
