"""Toy-only Stage5 freeze, cleaning-boundary, and API contracts."""

import json
import re
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.classification as classification_api
from app.api.classification import router as classification_router
from nlplab.classification.stage5 import (
    CONDITIONS,
    PROTOCOL,
    build_flips,
    feature_positions,
    word_representation,
)
from nlplab.core.exceptions import DataValidationError
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.classification_stage5 import CleaningFreeze, Stage5Data, install_stage5_guard
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.char_chi2 import char_bigrams
from nlplab.features.shortcut_cleaning import clean_view, segmented_word_features, tokenize_view
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig


def _write_manifest(root):
    directory = root / "data/splits/thucnews_600"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"splits": {}, "file_sha256": {}}), encoding="utf-8"
    )
    return directory


def test_stage5_test_split_is_lazy_and_forbidden(tmp_path):
    root = tmp_path
    split_dir = _write_manifest(root)
    test_path = split_dir / "test.jsonl"
    test_path.write_text('{"toy": true}\n', encoding="utf-8")
    install_stage5_guard(root)

    data = Stage5Data(root)
    with pytest.raises(DataValidationError, match="Train/Validation only"):
        data.load("test")
    assert data.opened_splits == []
    with pytest.raises(PermissionError, match="prohibits Test"):
        test_path.open("rb")


def test_validation_and_stage4_validation_artifacts_require_freeze(tmp_path):
    root = tmp_path
    split_dir = _write_manifest(root)
    val_path = split_dir / "val.jsonl"
    val_path.write_text('{"toy": true}\n', encoding="utf-8")
    stage4_result = root / "outputs/exp3/stage4/toy/C0_result.json"
    stage4_result.parent.mkdir(parents=True)
    stage4_result.write_text('{"toy": true}\n', encoding="utf-8")
    install_stage5_guard(root)

    data = Stage5Data(root)
    with pytest.raises(DataValidationError, match="Cleaning freeze required"):
        data.load("val")
    assert data.opened_splits == []
    with pytest.raises(PermissionError, match="Validation cannot open"):
        val_path.open("rb")
    with pytest.raises(PermissionError, match="Validation cannot open"):
        stage4_result.open("rb")


def test_cleaning_freeze_rejects_rule_mutation(tmp_path):
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    rules = freeze_dir / "cleaning_rules.json"
    rules.write_text('{"rules": []}\n', encoding="utf-8")
    seal = {
        "root": str(tmp_path),
        "files": {rules.name: file_sha256(rules)},
        "source_hashes": {},
    }
    (freeze_dir / "cleaning_freeze.json").write_text(json.dumps(seal), encoding="utf-8")
    freeze = CleaningFreeze(freeze_dir)
    assert freeze.verify()["files"][rules.name] == file_sha256(rules)

    rules.write_text('{"rules": [{"id": "changed"}]}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="frozen cleaning rules/source changed"):
        freeze.verify()


def test_oo_features_match_original_analyzer_and_keep_deletion_boundaries():
    rules = {"rules": [{"id": "toy-source", "kind": "source", "pattern": "source"}]}
    view = clean_view("alphaSOURCEbeta", rules, {"source"})

    processor = SimpleNamespace(
        pattern=re.compile(r"[A-Za-z]+"),
        config=SimpleNamespace(min_token_length=1),
        stopwords=frozenset(),
        segment=lambda chunk: [chunk],
    )
    token_row = tokenize_view(processor, view)
    assert token_row["sections"] == [["alpha"], ["beta"]]
    assert segmented_word_features(token_row["sections"]) == ["alpha", "beta"]
    assert "alpha beta" not in segmented_word_features(token_row["sections"])

    config = TFIDFConfig(min_df=1, max_df=1.0, max_features=None, ngram_range=(1, 2))
    original_analyzer = TextTFIDF(config).vectorizer.build_analyzer()
    clean_analyzer = word_representation(asdict(config)).vectorizer.build_analyzer()
    assert clean_analyzer([["alpha", "beta"]]) == original_analyzer(["alpha", "beta"])
    assert clean_analyzer(token_row["sections"]) == segmented_word_features(token_row["sections"])

    char_view = clean_view("甲source乙", rules, {"source"})
    assert char_view["char_text"] == "甲\n乙"
    assert "甲乙" not in char_bigrams(char_view["char_text"])
    assert set(feature_positions({}, char_view["text"], "C2")) == set(
        char_bigrams(char_view["text"])
    )


def test_conditions_are_train_frozen_and_have_no_extra_median_model():
    assert CONDITIONS == {
        "OO": [],
        "OS": ["source"],
        "OT": ["template"],
        "OA": ["source", "template"],
    }
    assert PROTOCOL["rules_design_split"] == "train"
    assert PROTOCOL["rules_freeze_before_validation"] is True
    assert PROTOCOL["hyperparameter_selection"] is False
    assert PROTOCOL["AA"].startswith("clean Train420 refit")
    assert not any("median" in name.lower() for name in CONDITIONS)


def test_flip_directions_and_stage5_model_selector_api(tmp_path, monkeypatch):
    labels = CATEGORIES[:3]
    documents = [
        ClassificationDocument(f"toy-{i}", f"toy text {i}", label) for i, label in enumerate(labels)
    ]
    evaluations = {}
    for model_id in ("C0", "C1", "C2", "C3"):
        evaluations[(model_id, "OO")] = {
            "predictions": [{"prediction": labels[0]} for _ in documents]
        }
        for condition in ("OS", "OT", "OA", "AA"):
            predictions = [{"prediction": labels[0]} for _ in documents]
            if condition == "OS":
                predictions = [{"prediction": labels[1]} for _ in documents]
            evaluations[(model_id, condition)] = {"predictions": predictions}

    flips = build_flips(evaluations, documents, [[], [], []])
    c0_os = [row for row in flips if row["model_id"] == "C0" and row["condition"] == "OS"]
    assert [row["direction"] for row in c0_os] == [
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ]
    assert len(flips) == 12

    class FakeStage5Service:
        def flips(self, run_id="latest", model_id=None, condition=None):
            return [
                row
                for row in flips
                if (model_id is None or row["model_id"] == model_id)
                and (condition is None or row["condition"] == condition)
            ]

    monkeypatch.setattr(classification_api, "stage5_service", lambda request: FakeStage5Service())
    app = FastAPI()
    app.include_router(classification_router)
    with TestClient(app) as client:
        response = client.get("/api/v1/experiments/3/stage5/flips?model_id=C0&condition=OS")
        assert response.status_code == 200
        assert len(response.json()) == 3
        assert client.get("/api/v1/experiments/3/stage5/flips?model_id=C4").status_code == 422
        assert client.get("/api/v1/experiments/3/stage5/flips?model_id=median").status_code == 422
