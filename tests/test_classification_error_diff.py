"""Read-only toy contracts for the Stage5 error-diff document projection."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nlplab.services.classification_shortcut_visualization_service as shortcut_module
from app.api.classification import router as classification_router
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.services.classification_shortcut_visualization_service import (
    ClassificationShortcutVisualizationService,
    retained_segments,
)
from nlplab.services.classification_stage3_service import ClassificationStage3Service
from nlplab.services.classification_stage5_service import ClassificationStage5Service


DOCUMENT_ID = "toy-validation-🙂"
RUN_ID = "toy-stage5"
TEXT = "甲🙂来源乙模板丙"
SOURCE_SPAN = {"start": TEXT.index("来源"), "end": TEXT.index("来源") + 2,
               "kind": "source", "rule": "toy-source", "text": "来源"}
TEMPLATE_SPAN = {"start": TEXT.index("模板"), "end": TEXT.index("模板") + 2,
                 "kind": "template", "rule": "toy-template", "text": "模板"}
SPANS = [SOURCE_SPAN, TEMPLATE_SPAN]


def _segments(text, spans):
    return retained_segments(text, spans)


def _toy_artifacts(model_id="C3"):
    clean_segments = _segments(TEXT, SPANS)
    predictions = {}
    results = []
    for condition in ("OO", "OS", "OT", "OA", "AA"):
        row = {
            "document_id": DOCUMENT_ID,
            "gold": "社会",
            "prediction": "社会" if condition != "OS" else "教育",
            "condition": condition,
            "score_type": "logit" if model_id == "C3" else "decision",
        }
        if model_id == "C3":
            row["all_channels"] = [
                {"channel": index, "max_activation": index / 1000.0,
                 "argmax_position": index % 7, "contribution": index / 10_000.0}
                for index in range(384)
            ]
        predictions[f"{model_id}_{condition}_predictions.json"] = [row]
        results.append({
            "model_id": model_id,
            "condition": condition,
            "model_reference": "stage4-original-model" if condition != "AA" else "stage5-aa-model",
        })
    return {
        "flips.json": [{"document_id": DOCUMENT_ID, "model_id": model_id,
                         "condition": "OS", "gold": "社会"}],
        "validation_cleaning.json": [{
            "document_id": DOCUMENT_ID,
            "gold": "社会",
            "text": TEXT,
            "char_text": "\n".join(segment["text"] for segment in clean_segments),
            "removed_spans": copy.deepcopy(SPANS),
            "segments": clean_segments,
        }],
        "validation_open_record.json": {"document_ids": [DOCUMENT_ID]},
        "results.json": results,
        "run.json": {"parent_stage4_run": "stage4-parent"},
        **predictions,
    }


def _install_toy_artifacts(monkeypatch, artifacts):
    calls = []
    directory = SimpleNamespace(name=RUN_ID)

    def resolve(self, run_id="latest"):
        assert run_id == RUN_ID
        return directory

    def read_json(self, _directory, name):
        calls.append(name)
        assert "test" not in name.lower(), "Stage5 Validation view must not open Test artifacts"
        if name not in artifacts:
            raise AssertionError(f"unexpected saved artifact read: {name}")
        return copy.deepcopy(artifacts[name])

    monkeypatch.setattr(ClassificationStage5Service, "resolve", resolve)
    monkeypatch.setattr(ClassificationShortcutVisualizationService, "_json", read_json)
    return calls


def _service():
    return ClassificationShortcutVisualizationService(SimpleNamespace(outputs=Path(".")))


def test_saved_offsets_are_unicode_codepoints_and_validate_exact_text():
    text = "甲🙂é尾"
    span = {"start": 1, "end": 2, "text": "🙂"}
    assert retained_segments(text, [span]) == [
        {"start": 0, "end": 1, "text": "甲"},
        {"start": 2, "end": 4, "text": "é尾"},
    ]
    for bad_span in (
        {"start": 1, "end": 2, "text": "x"},
        {"start": -1, "end": 2, "text": "甲🙂"},
        {"start": 2, "end": 2, "text": ""},
        {"start": 3, "end": 8, "text": "尾"},
    ):
        with pytest.raises(DataValidationError):
            retained_segments(text, [bad_span])


def test_conditions_keep_independent_segments_and_saved_model_identity(monkeypatch):
    artifacts = _toy_artifacts()
    calls = _install_toy_artifacts(monkeypatch, artifacts)
    result = _service().document(RUN_ID, DOCUMENT_ID, "C3")

    conditions = result["conditions"]
    assert conditions["OO"]["removed_spans"] == []
    assert conditions["OO"]["segments"] == [{"start": 0, "end": len(TEXT), "text": TEXT}]
    assert conditions["OO"]["input_text"] == TEXT
    assert conditions["OO"]["input_identity"] == "original"
    assert conditions["OO"]["record"] == artifacts["C3_OO_predictions.json"][0]

    assert [segment["text"] for segment in conditions["OS"]["segments"]] == ["甲🙂", "乙模板丙"]
    assert conditions["OS"]["input_text"] == "甲🙂\n乙模板丙"
    assert "甲🙂乙" not in conditions["OS"]["input_text"]
    assert [segment["text"] for segment in conditions["OT"]["segments"]] == ["甲🙂来源乙", "丙"]
    assert [segment["text"] for segment in conditions["OA"]["segments"]] == ["甲🙂", "乙", "丙"]
    assert conditions["OA"]["segments"] == conditions["AA"]["segments"]
    assert conditions["OS"]["input_identity"] == "cleaned"

    for condition in ("OO", "OS", "OT", "OA"):
        assert conditions[condition]["model_reference"] == "stage4-original-model"
        assert conditions[condition]["model_identity"] == "original"
    assert conditions["AA"]["model_reference"] == "stage5-aa-model"
    assert conditions["AA"]["model_identity"] == "clean-trained"
    assert result["original_model_run"] == "stage4-parent"
    assert result["provenance"] == {
        "source_run_id": RUN_ID,
        "saved_explanations_only": True,
        "cleaning_rules_executed": False,
        "inference_executed": False,
        "fit_executed": False,
    }
    assert "C3_OO_predictions.json" in calls


def test_bad_saved_cleaning_boundaries_fail_closed(monkeypatch):
    artifacts = _toy_artifacts()
    artifacts["validation_cleaning.json"][0]["removed_spans"][0]["text"] = "bad"
    _install_toy_artifacts(monkeypatch, artifacts)
    with pytest.raises(DataValidationError, match="保存的删除区间"):
        _service().document(RUN_ID, DOCUMENT_ID, "C1")

    artifacts = _toy_artifacts()
    artifacts["validation_cleaning.json"][0]["char_text"] = "人为拼接文本"
    _install_toy_artifacts(monkeypatch, artifacts)
    with pytest.raises(DataValidationError, match="清理边界"):
        _service().document(RUN_ID, DOCUMENT_ID, "C1")


def test_unknown_model_or_document_never_reads_test_or_runs_model_code(monkeypatch):
    artifacts = _toy_artifacts()
    calls = _install_toy_artifacts(monkeypatch, artifacts)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("document projection must not call prediction/inference")

    monkeypatch.setattr(ClassificationStage3Service, "predict", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(DataValidationError, match="未知模型"):
        _service().document(RUN_ID, DOCUMENT_ID, "C9")
    assert calls == []

    with pytest.raises(ResourceUnavailableError, match="冻结 Stage5 flip 文档"):
        _service().document(RUN_ID, "unknown-document", "C3")
    assert calls == ["flips.json"]


def test_c3_preserves_all_384_saved_channel_rows_for_each_condition(monkeypatch):
    _install_toy_artifacts(monkeypatch, _toy_artifacts())
    result = _service().document(RUN_ID, DOCUMENT_ID, "C3")

    by_condition = result["conditions"]
    for condition in ("OO", "OS", "OT", "OA", "AA"):
        channels = by_condition[condition]["record"]["all_channels"]
        assert len(channels) == 384
        assert [row["channel"] for row in channels] == list(range(384))
        assert channels[173] == {
            "channel": 173,
            "max_activation": 0.173,
            "argmax_position": 5,
            "contribution": 0.0173,
        }


def test_shortcut_document_get_route_returns_selected_model_and_provenance(monkeypatch):
    calls = _install_toy_artifacts(monkeypatch, _toy_artifacts("C2"))
    app = FastAPI()
    app.include_router(classification_router)
    app.state.classification = SimpleNamespace(settings=SimpleNamespace(outputs=Path(".")))

    response = TestClient(app).get(
        "/api/v1/experiments/3/visualization/shortcut-document",
        params={"run_id": RUN_ID, "document_id": DOCUMENT_ID, "model_id": "C2"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "C2"
    assert body["original_model_run"] == "stage4-parent"
    assert body["conditions"]["AA"]["model_reference"] == "stage5-aa-model"
    assert body["provenance"]["inference_executed"] is False
    assert "C2_AA_predictions.json" in calls
