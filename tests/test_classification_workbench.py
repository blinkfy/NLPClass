"""Toy readonly workbench service contracts using frozen artifact-shaped fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nlplab.services.classification_workbench_service as workbench_service_module
from app.api.classification import router as classification_router
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.course import file_sha256
from nlplab.services.classification_stage3_service import ClassificationStage3Service
from nlplab.services.classification_workbench_service import ClassificationWorkbenchService


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(directory, names):
    files = {name: file_sha256(directory / name) for name in names}
    _write_json(directory / "artifact_manifest.json", {"files": files})


def _write_csr(path, dimension, indices, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.asarray(b"csr"),
        shape=np.asarray([1, dimension], dtype=np.int64),
        indptr=np.asarray([0, len(indices)], dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(values, dtype=np.float64),
    )


def test_stage3_document_projects_frozen_full_text_encoding_and_explanations(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only workbench must not run inference")

    monkeypatch.setattr(ClassificationStage3Service, "predict", forbidden)

    root = tmp_path
    output = root / "outputs/exp3"
    stage1 = output / "stage1"
    stage2 = output / "stage2"
    stage3 = output / "stage3"
    doc_id = "toy-document"
    text = "完整冻结文本" * 150
    gold = "社会"
    test_path = root / "data/splits/thucnews_600/test.jsonl"
    _write_jsonl(test_path, [{"id": doc_id, "text": text, "primary_label": gold}])
    dataset = {
        "splits": {"test": {"ids": [doc_id]}},
        "file_sha256": {"test.jsonl": file_sha256(test_path)},
        "dataset_fingerprint": "toy-same-corpus",
    }

    _write_json(stage1 / "run.json", {"run_type": "formal-exp3-stage1", "run_id": "stage1"})
    _write_json(stage1 / "dataset.json", dataset)
    for model in ("C0", "C1"):
        _write_jsonl(
            stage1 / f"{model}_test_predictions.jsonl",
            [{"document_id": doc_id, "gold": gold, "prediction": gold}],
        )
    word_vocabulary = {"title-term": 0, "body-term": 1}
    _write_json(
        stage1 / "final/tfidf/vocabulary.json",
        {"vocabulary": word_vocabulary},
    )
    _write_csr(
        stage1 / "final/test.npz",
        2,
        [0, 1],
        [0.6, 0.8],
    )
    _write_manifest(
        stage1,
        [
            "run.json",
            "dataset.json",
            "C0_test_predictions.jsonl",
            "C1_test_predictions.jsonl",
            "final/tfidf/vocabulary.json",
            "final/test.npz",
        ],
    )

    _write_json(
        stage2 / "run.json",
        {"run_type": "formal-exp3-stage2", "run_id": "stage2", "stage1_run_id": "stage1"},
    )
    _write_json(stage2 / "dataset.json", dataset)
    _write_jsonl(
        stage2 / "C2_test_predictions.jsonl",
        [{"document_id": doc_id, "gold": gold, "prediction": gold}],
    )
    selected = [
        {"column": 0, "term": "完整", "chi2_score": 1.5},
        {"column": 1, "term": "冻结", "chi2_score": 2.5},
    ]
    raw_vocabulary = {"完整": 0, "冻结": 1, "文本": 2}
    _write_json(stage2 / "final/representation/selected_features.json", selected)
    _write_json(
        stage2 / "final/representation/representation.json",
        {"vocabulary": raw_vocabulary},
    )
    _write_csr(
        stage2 / "final/test.npz",
        2,
        [0, 1],
        [0.4, 0.9],
    )
    _write_manifest(
        stage2,
        [
            "run.json",
            "dataset.json",
            "C2_test_predictions.jsonl",
            "final/representation/selected_features.json",
            "final/representation/representation.json",
            "final/test.npz",
        ],
    )

    _write_json(
        stage3 / "run.json",
        {
            "run_type": "formal-exp3-stage3",
            "stage1_run_id": "stage1",
            "stage2_run_id": "stage2",
        },
    )
    channels = [
        {"channel": 0, "max_activation": 0.25},
        {"channel": 1, "max_activation": 0.75},
    ]
    _write_jsonl(
        stage3 / "C3_test_predictions.jsonl",
        [
            {
                "document_id": doc_id,
                "gold": gold,
                "prediction": gold,
                "text": text,
                "sequence_diagnostic": {
                    "original_length": 3,
                    "retained_length": 2,
                    "truncated": True,
                },
                "raw_logits": [0.1, 0.2],
                "all_channels": channels,
            }
        ],
    )
    _write_json(
        stage3 / "test_encoding.json",
        {
            "document_ids": [doc_id],
            "tokens": [["新闻", "热点"]],
            "original_positions": [[0, 2]],
            "stats": {"per_document": []},
        },
    )
    np.savez_compressed(
        stage3 / "test_encoding.npz",
        ids=np.asarray([[2, 3, 0]], dtype=np.int64),
        lengths=np.asarray([2], dtype=np.int64),
    )
    # This is the frozen run's actual vocabulary location.
    _write_json(
        stage3 / "final_vocabulary/vocabulary.json",
        {"terms": ["<PAD>", "<UNK>", "新闻", "热点"], "pad": 0, "unk": 1},
    )
    _write_json(stage3 / "dataset.json", dataset)
    _write_manifest(
        stage3,
        [
            "run.json",
            "dataset.json",
            "C3_test_predictions.jsonl",
            "test_encoding.json",
            "test_encoding.npz",
            "final_vocabulary/vocabulary.json",
        ],
    )

    service = ClassificationWorkbenchService(Settings(root=root))
    result = service.document("stage3", doc_id)

    assert result["document"]["text"] == text
    assert result["records"]["C3"]["raw_logits"] == [0.1, 0.2]
    assert "all_channels" not in result["records"]["C3"]
    assert result["representations"]["C3"]["sequence"] == {
        "tokens": ["新闻", "热点"],
        "original_positions": [0, 2],
        "token_ids": [2, 3],
        "model_tokens": ["新闻", "热点"],
        "diagnostic": {"original_length": 3, "retained_length": 2, "truncated": True},
        "source_run": "stage3",
        "pooled": [0.25, 0.75],
        "embedding_values_saved": False,
    }
    assert result["representations"]["C2"]["selected_features"] == selected
    assert result["representations"]["C2"]["raw_vocabulary"] == list(raw_vocabulary)
    for model in ("C0", "C1"):
        word_row = result["representations"][model]
        assert word_row["top_tfidf_features"] == [
            {"term": "body-term", "tfidf": 0.8, "column": 1},
            {"term": "title-term", "tfidf": 0.6, "column": 0},
        ]
        assert (word_row["nonzero_count"], word_row["dimension"], word_row["sparsity"]) == (
            2,
            2,
            0.0,
        )
    assert result["representations"]["C2"]["top_tfidf_features"] == [
        {"term": "冻结", "tfidf": 0.9, "column": 1},
        {"term": "完整", "tfidf": 0.4, "column": 0},
    ]
    complete_feature = next(
        row
        for row in result["representations"]["C2"]["feature_membership"]
        if row["term"] == "完整"
    )
    assert complete_feature == {
        "term": "完整",
        "count": 150,
        "in_raw_vocabulary": True,
        "retained": True,
        "chi2_score": 1.5,
    }
    assert result["provenance"]["inference_executed"] is False

    # Stage1/2 views do not open a Stage3 model/prediction/encoding.
    def no_future(*args, **kwargs):
        raise AssertionError("historical Run must remain independent of Stage3")

    monkeypatch.setattr(service.formal, "stage3_directory", no_future)
    historical = service.document("stage1", doc_id)
    assert set(historical["records"]) == {"C0", "C1"}
    assert historical["document"]["text"] == text
    assert "sequence" not in historical["representations"]["C1"]
    historical2 = service.document("stage2", doc_id)
    assert set(historical2["records"]) == {"C0", "C1", "C2"}
    # Valid file hash alone cannot mask disagreement between stored full texts.
    _write_jsonl(test_path, [{"id": doc_id, "text": "changed", "primary_label": gold}])
    with pytest.raises(DataValidationError, match="正文来源"):
        service.document("stage1", doc_id)


def test_saved_csr_row_rejects_document_and_feature_name_mismatches(tmp_path):
    directory = tmp_path / "outputs/exp3/stage1"
    _write_json(directory / "dataset.json", {"splits": {"test": {"ids": ["toy-id"]}}})
    _write_csr(directory / "final/test.npz", 2, [0, 1], [0.25, 0.75])
    _write_manifest(directory, ["dataset.json", "final/test.npz"])
    service = ClassificationWorkbenchService(Settings(root=tmp_path))

    with pytest.raises(DataValidationError, match="document IDs 不匹配"):
        service._sparse_row(directory, "different-id", ["left", "right"])
    with pytest.raises(DataValidationError, match="特征名和矩阵列不一致"):
        service._sparse_row(directory, "toy-id", ["only-one-name"])


def _stage5_fixture(root):
    stage5 = root / "outputs/exp3/stage5/toy"
    doc_id = "validation-document"
    full_text = "Validation 正文" * 80
    snippet = full_text[:40]
    prediction_files = []

    _write_json(stage5 / "run.json", {"run_id": "toy"})
    _write_json(
        stage5 / "flips.json",
        [
            {
                "document_id": doc_id,
                "gold": "社会",
                "condition": "OA",
                "model_id": "C3",
                "text_snippet": snippet,
                "removed_spans": [],
            }
        ],
    )
    for model in ("C0", "C1", "C2", "C3"):
        for condition in ("OO", "OA"):
            name = f"{model}_{condition}_predictions.json"
            row = {
                "document_id": doc_id,
                "gold": "社会",
                "prediction": "社会",
                "text_snippet": snippet,
            }
            if model == "C3":
                row["all_channels"] = [{"max_activation": 0.4}]
            _write_json(stage5 / name, [row])
            prediction_files.append(name)
    val = root / "data/splits/thucnews_600/val.jsonl"
    _write_jsonl(val, [{"id": doc_id, "text": full_text, "primary_label": "社会"}])
    _write_json(
        stage5 / "validation_open_record.json",
        {
            "document_ids": [doc_id],
            "train_ids": [],
            "validation_sha256": file_sha256(val),
        },
    )
    _write_manifest(
        stage5,
        [
            "run.json",
            "flips.json",
            "validation_open_record.json",
            *prediction_files,
        ],
    )
    _write_json(root / "outputs/exp3/stage5_latest.json", {"run_id": "toy"})
    return ClassificationWorkbenchService(Settings(root=root)), doc_id, full_text, val


def test_stage5_document_reads_json_array_predictions_and_full_validation_text(tmp_path):
    service, doc_id, full_text, _ = _stage5_fixture(tmp_path)

    result = service.document("ignored-for-stage5", doc_id, source="stage5", condition="OA")

    assert result["document"]["text"] == full_text
    assert result["records"]["C3"]["prediction"] == "社会"
    assert "all_channels" not in result["records"]["C3"]
    assert result["ablation"]["changes"]["C3"]["prediction"] == "社会"
    assert result["provenance"]["inference_executed"] is False


def test_stage5_document_rejects_changed_validation_source_hash(tmp_path):
    service, doc_id, full_text, val = _stage5_fixture(tmp_path)
    val.write_text(
        json.dumps(
            [{"id": doc_id, "text": full_text + "changed", "primary_label": "社会"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="Validation 正文来源不一致"):
        service.document("ignored-for-stage5", doc_id, source="stage5", condition="OA")


def test_workbench_document_get_route_returns_schema_without_predict(monkeypatch):
    settings = object()
    calls = {"document": [], "predict": 0}
    payload = {
        "document": {
            "document_id": "toy-document",
            "gold": "topic",
            "text": "A frozen document.",
            "title": "A frozen document.",
            "split": "Test",
            "source_run": "toy-stage3",
        },
        "records": {
            "C3": {"prediction": "topic", "raw_logits": [0.2, 0.8]},
        },
        "representations": {
            "C3": {
                "sequence": {
                    "tokens": ["frozen", "text"],
                    "original_positions": [0, 1],
                    "token_ids": [4, 5],
                    "model_tokens": ["frozen", "text"],
                    "pooled": [0.3, 0.9],
                    "embedding_values_saved": False,
                }
            }
        },
        "provenance": {
            "view_run_id": "toy-stage3",
            "saved_predictions_only": True,
            "inference_executed": False,
        },
    }

    class FakeWorkbenchService:
        def __init__(self, actual_settings):
            assert actual_settings is settings

        def document(self, run_id, document_id, source, condition):
            calls["document"].append((run_id, document_id, source, condition))
            return payload

        def predict(self, *args, **kwargs):
            calls["predict"] += 1
            raise AssertionError("GET workbench document must not predict")

    monkeypatch.setattr(
        workbench_service_module,
        "ClassificationWorkbenchService",
        FakeWorkbenchService,
    )
    app = FastAPI()
    app.include_router(classification_router)
    app.state.classification = SimpleNamespace(settings=settings)

    response = TestClient(app).get(
        "/api/v1/experiments/3/workbench/document",
        params={
            "document_id": "toy-document",
            "run_id": "toy-stage3",
            "source": "test",
            "condition": "OA",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"document", "records", "representations", "provenance"}
    assert body["document"]["document_id"] == "toy-document"
    assert body["records"]["C3"]["raw_logits"] == [0.2, 0.8]
    assert body["representations"]["C3"]["sequence"]["token_ids"] == [4, 5]
    assert body["provenance"]["inference_executed"] is False
    assert calls["document"] == [("toy-stage3", "toy-document", "test", "OA")]
    assert calls["predict"] == 0
