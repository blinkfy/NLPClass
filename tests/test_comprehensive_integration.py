"""Toy-only contracts for the frozen C2 application index integration.

All inference calls are stubs. These tests never load formal weights, fit a
model, inspect Test text, or invoke the NLP/NER runtimes.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.comprehensive.classification_application import (
    attach_evidence,
    evidence_occurrences,
)
from nlplab.comprehensive.contracts import aggregate, document_id
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.services.comprehensive_service import ComprehensiveAnalysisService


class StubClustering:
    def __init__(self):
        self.assignments = [
            {"id": "train-a", "cluster_id": 0, "snippet": "Train 摘要", "distance": 0.1},
            {"id": "val-b", "cluster_id": 0, "snippet": "Validation 摘要", "distance": 0.2},
            {"id": "test-c", "cluster_id": 0, "snippet": "Test 摘要", "distance": 0.3},
        ]
        self.calls = []

    def verify_artifacts(self, run_id):
        self.calls.append(("verify", run_id))

    def result(self, run_id, k):
        self.calls.append(("result", run_id, k))
        return {"assignments": self.assignments, "clusters": [{"cluster_id": 0}]}

    def projection(self, run_id, dimensions, *, cache_only=False):
        self.calls.append(("projection", run_id, dimensions, cache_only))
        assert cache_only is True
        return {
            "points": [
                {"id": row["id"], "x": index, "y": -index}
                for index, row in enumerate(self.assignments)
            ],
            "method": "toy cached projection",
        }


def _write_sources(root):
    split_dir = root / "data/splits/thucnews_600"
    split_dir.mkdir(parents=True)
    split_rows = {
        "train.jsonl": [{"id": "train-a", "text": "Train 正文Ａ", "label": "secret-train-label"}],
        "val.jsonl": [{"id": "val-b", "text": "Validation 正文Ｂ", "label": "secret-val-label"}],
        "test.jsonl": [{"id": "test-c", "text": "Test 正文不可用于分类", "label": "secret-test-label"}],
    }
    split_hashes = {}
    for name, rows in split_rows.items():
        path = split_dir / name
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        split_hashes[name] = file_sha256(path)
    (split_dir / "manifest.json").write_text(
        json.dumps({"file_sha256": split_hashes}), encoding="utf-8"
    )

    run = root / "outputs/exp3/20261001_133829_68ee50d7"
    toy_artifacts = (
        "artifact_manifest.json",
        "final/C2/model.json",
        "final/C2/parameters.npz",
        "final/representation/representation.json",
        "final/representation/parameters.npz",
        "final/representation/observed_df.json",
    )
    for name in toy_artifacts:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("toy hash source: " + name).encode("utf-8"))


def _service(root, *, initialize=True):
    if initialize:
        _write_sources(root)
    return ComprehensiveAnalysisService(
        Settings(root),
        nlp=object(),
        course=object(),
        clustering=StubClustering(),
        neural=object(),
    )


def _install_classify_worker(service):
    calls = []

    def worker(kind, payload, **_kwargs):
        calls.append((kind, copy.deepcopy(payload)))
        assert kind == "classify"
        records = []
        for doc in payload["documents"]:
            records.append(
                {
                    "id": doc["id"],
                    "text_id": document_id(doc["text"]),
                    "classification": {
                        "model_id": "C2",
                        "run_id": service.CLASSIFIER_RUN,
                        "prediction": CATEGORIES[0],
                        "score_type": "decision_score",
                        "top_contributing_features": [],
                        "top_relative_features": [],
                    },
                }
            )
        return {
            "records": records,
            "fit_executed": False,
            "ner_executed": False,
            "gold_labels_used": False,
            "formal_evaluation_executed": False,
            "formal_run_written": False,
        }

    service._run_worker = worker
    return calls


def test_index_promotes_partial_detail_without_losing_lexical(tmp_path):
    service = _service(tmp_path)
    service.course_collection()
    collection = service._collection("course-r2-k5")
    lexical = {"tokens": [{"form": "Train", "start": 0, "end": 5}], "pos": [], "entities": []}
    collection["analyses"]["train-a"] = {
        "status": "partial", "lexical": copy.deepcopy(lexical), "classification": {},
        "modules": {"segmentation": {"status": "ok"}, "classification": {"status": "unavailable"}},
    }
    _install_classify_worker(service)
    service.build_classification_index()
    detail = service.profile("course-r2-k5", "train-a")
    assert detail["analysis"]["classification"]["prediction"] == CATEGORIES[0]
    assert detail["analysis"]["modules"]["classification"]["status"] == "ok"
    assert detail["analysis"]["lexical"] == lexical


def _recompute_records_hash(value):
    payload = json.dumps(
        value["records"], sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    value["records_sha256"] = hashlib.sha256(payload).hexdigest()


def test_build_index_batches_only_eligible_train_validation_text_without_evaluation_or_ner(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    calls = _install_classify_worker(service)
    test_path = tmp_path / "data/splits/thucnews_600/test.jsonl"
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == test_path.resolve():
            raise AssertionError("classification index must not open the Test split")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    summary = service.build_classification_index()

    assert len(calls) == 1
    kind, payload = calls[0]
    assert kind == "classify"
    assert payload["run_id"] == service.CLASSIFIER_RUN
    assert payload["documents"] == [
        {"id": "train-a", "text": "Train 正文Ａ"},
        {"id": "val-b", "text": "Validation 正文Ｂ"},
    ]
    assert all(set(row) == {"id", "text"} for row in payload["documents"])
    assert all("label" not in row and "gold" not in row for row in payload["documents"])
    assert summary["eligible"] == summary["classification_known"] == 2
    assert summary["new_inferences"] == 2
    assert summary["ner_executed"] is False
    assert summary["test_text_accessed"] is False
    assert summary["gold_labels_used"] is False
    assert summary["formal_run_written"] is False
    assert summary["formal_evaluation_executed"] is False
    assert "metrics" not in summary
    cache = json.loads(service.classification_index_path.read_text(encoding="utf-8"))
    assert cache["test_text_accessed"] is False
    assert cache["ner_executed"] is False
    assert cache["gold_labels_used"] is False
    assert cache["formal_evaluation_executed"] is False
    assert cache["formal_run_written"] is False
    assert "metrics" not in cache
    assert all("test-c" != row["id"] for row in cache["records"])


def test_classification_cache_uses_independent_path_and_restores_after_service_restart(tmp_path):
    service = _service(tmp_path)
    calls = _install_classify_worker(service)
    service.build_classification_index()

    expected = tmp_path / "outputs/applications/lab4/classification-index.json"
    assert service.classification_index_path == expected
    assert expected.is_file()
    assert not (tmp_path / "outputs/exp3/stage2_latest.json").exists()

    restored = _service(tmp_path, initialize=False)
    restored_calls = _install_classify_worker(restored)
    collection = restored.course_collection()
    state = restored._collection(collection["id"])
    assert set(state["application_predictions"]) == {"train-a", "val-b"}
    assert restored_calls == []
    assert len(calls) == 1


def test_second_index_build_reuses_persisted_results_without_new_inference(tmp_path):
    service = _service(tmp_path)
    calls = _install_classify_worker(service)

    first = service.build_classification_index()
    second = service.build_classification_index()

    assert first["new_inferences"] == 2
    assert second["new_inferences"] == 0
    assert second["classification_known"] == 2
    assert len(calls) == 1


@pytest.mark.parametrize("tamper", ["signature", "records_hash", "text_id"])
def test_persisted_index_rejects_signature_content_and_text_tampering(tmp_path, tamper):
    service = _service(tmp_path)
    _install_classify_worker(service)
    service.build_classification_index()
    index_path = service.classification_index_path

    if tamper == "signature":
        model_path = (
            tmp_path
            / "outputs/exp3/20261001_133829_68ee50d7/final/C2/parameters.npz"
        )
        model_path.write_bytes(b"tampered source hash")
    else:
        cache = json.loads(index_path.read_text(encoding="utf-8"))
        if tamper == "records_hash":
            cache["records"][0]["classification"]["prediction"] = CATEGORIES[1]
        else:
            cache["records"][0]["text_id"] = "text-wrong-body-hash"
            _recompute_records_hash(cache)
        index_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    reloaded = _service(tmp_path, initialize=False)
    with pytest.raises(DataValidationError, match="指纹不符|正文不符"):
        reloaded.course_collection()


def test_aggregate_keeps_classification_known_when_ner_is_unknown(tmp_path):
    service = _service(tmp_path)
    _install_classify_worker(service)
    service.build_classification_index()
    collection = service._collection("course-r2-k5")

    result = aggregate(collection)
    assert result["classification_document_counts"] == [2]
    assert result["classification_matrix"] == [[2, 0, 0, 0, 0]]
    assert result["ner_document_counts"] == [0]
    assert result["entity_matrix"] == [[None, None, None]]
    rows = {row["id"]: row for row in result["documents"]}
    assert rows["train-a"]["classification_known"] is True
    assert rows["train-a"]["ner_known"] is False
    assert rows["val-b"]["classification_known"] is True
    assert rows["val-b"]["ner_known"] is False
    assert "test-c" not in rows


def test_classification_index_build_api_schema_has_no_request_body_and_dispatches(tmp_path):
    service = _service(tmp_path)
    calls = _install_classify_worker(service)
    application = create_app(settings=Settings(tmp_path), service=object())
    application.state.comprehensive = service

    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()
        path = "/api/v1/comprehensive/classification-index/build"
        operation = schema["paths"][path]["post"]
        assert "requestBody" not in operation
        response = client.post(path)

    assert response.status_code == 200
    assert response.json()["new_inferences"] == 2
    assert response.json()["formal_evaluation_executed"] is False
    assert len(calls) == 1


def test_evidence_occurrences_map_nfkc_spans_and_fail_closed_on_composition_or_punctuation():
    text = "😀Ａa-Ａa"
    assert evidence_occurrences(text, "aa") == [
        {"start": 1, "end": 3, "text": "Ａa"},
        {"start": 4, "end": 6, "text": "Ａa"},
    ]
    assert evidence_occurrences("e\u0301ab", "ab") == []
    assert evidence_occurrences("甲-乙", "甲乙") == []
    assert evidence_occurrences("abab", "a") == []


def test_attach_evidence_keeps_direct_and_relative_contributions_separate_and_copies_input():
    prediction = {
        "top_contributing_features": [
            {"term": "aa", "tfidf": 0.5, "weight": 2.0, "direct_contribution": 1.0}
        ],
        "top_relative_features": [
            {"term": "aa", "tfidf": 0.5, "contribution": 0.75, "coefficient_difference": 1.5},
            {"term": "bb", "tfidf": 0.25, "contribution": -0.2, "coefficient_difference": -0.8},
        ],
    }
    original = copy.deepcopy(prediction)

    result = attach_evidence("Ａa bb", prediction)

    assert prediction == original
    assert result is not prediction
    assert result["evidence"] is not prediction["top_relative_features"]
    by_feature = {row["feature"]: row for row in result["evidence"]}
    assert by_feature["aa"]["relative_contribution"] == 0.75
    assert by_feature["aa"]["coefficient_difference"] == 1.5
    assert by_feature["aa"]["direct_contribution"] == 1.0
    assert by_feature["aa"]["weight"] == 2.0
    assert by_feature["bb"]["relative_contribution"] == -0.2
    assert "direct_contribution" not in by_feature["bb"]
    assert "weight" not in by_feature["bb"]
    assert result["offset_space"] == "original_text Unicode code-point"
