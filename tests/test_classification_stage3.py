"""Toy-only Stage3 controller contracts and read-only pinned lineage checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.classification.stage3 import save_encoding
from nlplab.classification.stage3_config import ARCHITECTURE, PROTOCOL, TRAINING
from nlplab.classification.stage3_experiment import (
    median_final_epochs,
    paired_stage3,
    select_learning_rate,
    summarize_seeds,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.word_sequence import WordSequenceVocabulary
from nlplab.services import classification_service
from nlplab.services.classification_stage3_service import ClassificationStage3Service

STAGE1_RUN_ID = "20261001_123337_9c1e7341"
STAGE1_MANIFEST_SHA256 = "ddb329ec1bc5f9e0020ecff5fa35c6832091b7b8981fd9d3ac40fc2d3991aeee"
STAGE1_LATEST_SHA256 = "3988fe4fb98bedbb3ad8d70dc0300148ecefa0a143406323bb56a6dd7063f8dd"
STAGE2_RUN_ID = "20261001_133829_68ee50d7"
STAGE2_MANIFEST_SHA256 = "9838910fab286e7705b3d4db3290936895d58db188cb03823eda0a695f43c564"
STAGE2_LATEST_SHA256 = "61779febcdd92dfeb7382f7ac5dafa0c6d4c4b4ebdc794df6fbe1042284ecd99"


def test_stage3_protocol_pins_frozen_sources_and_fixed_cnn_contract():
    assert PROTOCOL["stage1_run_id"] == STAGE1_RUN_ID
    assert PROTOCOL["stage2_run_id"] == STAGE2_RUN_ID
    assert PROTOCOL["vocabulary"] == {
        "pad": 0,
        "unk": 1,
        "min_freq": 2,
        "max_vocab": None,
        "order": "frequency desc, lexical asc",
    }
    assert PROTOCOL["max_len_rule"] == "max(128, min(768, ceil(P95_train / 32) * 32))"
    assert PROTOCOL["final_fit_splits"] == ["train", "val"]
    assert PROTOCOL["test_policy"].startswith("one formal evaluation")
    assert ARCHITECTURE["embedding_dim"] == 128
    assert ARCHITECTURE["kernels"] == [3, 4, 5]
    assert ARCHITECTURE["channels_per_kernel"] == 128
    assert ARCHITECTURE["pooled_dimension"] == 384
    assert ARCHITECTURE["output_classes"] == 5
    assert TRAINING["patience"] == 5
    assert TRAINING["seed"] == [42, 43, 44]


def test_stage1_and_stage2_manifest_and_pointer_hashes_are_pinned_read_only():
    root = Path(__file__).resolve().parents[1]
    stage1 = root / "outputs/exp3" / STAGE1_RUN_ID
    stage2 = root / "outputs/exp3" / STAGE2_RUN_ID
    stage1_manifest = stage1 / "artifact_manifest.json"
    stage1_latest = root / "outputs/exp3/latest.json"
    stage2_manifest = stage2 / "artifact_manifest.json"
    stage2_latest = root / "outputs/exp3/stage2_latest.json"
    protected = (stage1_manifest, stage1_latest, stage2_manifest, stage2_latest)
    before = tuple(file_sha256(path) for path in protected)

    assert before == (
        STAGE1_MANIFEST_SHA256,
        STAGE1_LATEST_SHA256,
        STAGE2_MANIFEST_SHA256,
        STAGE2_LATEST_SHA256,
    )
    assert json.loads(stage1_latest.read_text("utf-8")) == {"run_id": STAGE1_RUN_ID}
    assert json.loads(stage2_latest.read_text("utf-8")) == {"run_id": STAGE2_RUN_ID}
    assert tuple(file_sha256(path) for path in protected) == before


def test_seed_summary_selects_mean_then_population_std_then_lower_learning_rate():
    results = [
        {"learning_rate": 0.001, "seed": seed, "best_validation_macro_f1": score}
        for seed, score in zip((42, 43, 44), (0.5, 0.5, 0.8))
    ] + [
        {"learning_rate": 0.0005, "seed": seed, "best_validation_macro_f1": 0.6}
        for seed in (42, 43, 44)
    ]
    summary = summarize_seeds(results)
    assert [row["learning_rate"] for row in summary] == [0.001, 0.0005]
    assert summary[0]["mean_macro_f1"] == pytest.approx(0.6)
    assert summary[0]["std_macro_f1"] == pytest.approx(np.std([0.5, 0.5, 0.8], ddof=0))
    assert summary[0]["std_ddof"] == 0
    assert summary[1]["mean_macro_f1"] == pytest.approx(0.6)
    assert summary[1]["std_macro_f1"] == 0
    assert select_learning_rate(summary) == 0.0005

    equal_summary = [
        {"learning_rate": 0.001, "mean_macro_f1": 0.7, "std_macro_f1": 0.0},
        {"learning_rate": 0.0005, "mean_macro_f1": 0.7, "std_macro_f1": 0.0},
    ]
    assert select_learning_rate(equal_summary) == 0.0005
    with pytest.raises(DataValidationError, match="seed42/43/44"):
        summarize_seeds(results[:-1])


def test_final_epochs_use_the_median_of_three_selected_lr_best_epochs():
    assert median_final_epochs([2, 8, 5]) == 5
    assert median_final_epochs([4, 4, 9]) == 4
    with pytest.raises(DataValidationError, match="三个正整数"):
        median_final_epochs([3, 7])
    with pytest.raises(DataValidationError, match="三个正整数"):
        median_final_epochs([3, 0, 7])


def test_post_test_pairing_keeps_ids_lineage_and_validates_input_rows():
    old = [
        {
            "document_id": "both",
            "gold": CATEGORIES[0],
            "prediction": CATEGORIES[0],
            "correct": True,
        },
        {
            "document_id": "fixed",
            "gold": CATEGORIES[1],
            "prediction": CATEGORIES[0],
            "correct": False,
        },
        {
            "document_id": "regressed",
            "gold": CATEGORIES[2],
            "prediction": CATEGORIES[2],
            "correct": True,
        },
        {
            "document_id": "wrong",
            "gold": CATEGORIES[3],
            "prediction": CATEGORIES[4],
            "correct": False,
        },
    ]
    c3 = [
        {
            "document_id": "both",
            "gold": CATEGORIES[0],
            "prediction": CATEGORIES[0],
            "correct": True,
        },
        {
            "document_id": "fixed",
            "gold": CATEGORIES[1],
            "prediction": CATEGORIES[1],
            "correct": True,
            "all_channels": [{"channel": 0}],
            "top_convolution_spans": [{"channel": 0}],
        },
        {
            "document_id": "regressed",
            "gold": CATEGORIES[2],
            "prediction": CATEGORIES[0],
            "correct": False,
        },
        {
            "document_id": "wrong",
            "gold": CATEGORIES[3],
            "prediction": CATEGORIES[4],
            "correct": False,
        },
    ]

    paired = paired_stage3(old, c3, "C2")
    assert paired["phase"] == "post_test_analysis_only"
    assert paired["old_model_id"] == "C2"
    assert paired["new_model_id"] == "C3"
    assert paired["sample_count"] == 4
    assert paired["counts"] == {
        "both_correct": 1,
        "old_only_correct": 1,
        "c3_only_correct": 1,
        "both_wrong": 1,
    }
    assert paired["document_ids"]["c3_only_correct"] == ["fixed"]
    assert paired["document_ids"]["old_only_correct"] == ["regressed"]
    assert paired["disagreement_ids"] == ["fixed", "regressed"]
    assert paired["samples"]["corrected_old"][0]["document_id"] == "fixed"
    assert "all_channels" not in paired["samples"]["corrected_old"][0]["c3"]

    with pytest.raises(DataValidationError, match="ID 重复"):
        paired_stage3(old + [dict(old[0])], c3, "C2")
    changed_gold = [dict(row) for row in c3]
    changed_gold[0]["gold"] = CATEGORIES[1]
    changed_gold[0]["correct"] = False
    with pytest.raises(DataValidationError, match="gold"):
        paired_stage3(old, changed_gold, "C2")


def test_toy_encoding_saves_numeric_arrays_and_document_ids_in_order(tmp_path):
    docs = [
        ClassificationDocument("toy-a", "alpha beta alpha", CATEGORIES[0]),
        ClassificationDocument("toy-b", "beta gamma beta", CATEGORIES[1]),
    ]
    tokenized = [["alpha", "beta", "alpha"], ["beta", "gamma", "beta"]]
    vocabulary = WordSequenceVocabulary(max_len=6).fit(tokenized)
    encoded = vocabulary.encode(tokenized)
    path = tmp_path / "toy-train"

    save_encoding(path, encoded, docs)

    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert set(arrays.files) == {"ids", "lengths", "labels"}
        assert arrays["ids"].shape == (2, 6)
        assert arrays["lengths"].tolist() == [3, 3]
        assert arrays["labels"].tolist() == [0, 1]
        assert arrays["ids"].dtype == np.int64
    metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
    assert metadata["document_ids"] == ["toy-a", "toy-b"]
    assert metadata["tokens"] == encoded["tokens"]


def test_stage3_predict_route_requires_c3_and_preserves_result_schema(tmp_path):
    class FakeClassificationService:
        def __init__(self):
            self.calls = []

        def predict(self, text, model_id, run_id):
            self.calls.append((text, model_id, run_id))
            return {
                "model_id": model_id,
                "run_id": run_id,
                "prediction": CATEGORIES[0],
                "score_type": "softmax",
                "scores": [{"label": CATEGORIES[0], "value": 1.0}],
            }

    fake = FakeClassificationService()
    application = create_app(Settings(root=tmp_path), service=fake)
    application.state.classification = fake
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/experiments/3/stage3/predict",
            json={"text": "toy news", "run_id": "toy-stage3"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "model_id": "C3",
            "run_id": "toy-stage3",
            "prediction": CATEGORIES[0],
            "score_type": "softmax",
            "scores": [{"label": CATEGORIES[0], "value": 1.0}],
        }
        assert fake.calls == [("toy news", "C3", "toy-stage3")]
        assert (
            client.post(
                "/api/v1/experiments/3/stage3/predict",
                json={"text": "toy news", "model_id": "C2"},
            ).status_code
            == 422
        )


def test_stage3_service_dispatches_old_models_to_parent_runs_and_blocks_c3_on_parents(
    tmp_path, monkeypatch
):
    run_info = {
        "stage3-run": {
            "run_type": "formal-exp3-stage3",
            "stage1_run_id": "stage1-run",
            "stage2_run_id": "stage2-run",
        },
        "stage1-run": {"run_type": "formal-exp3-stage1"},
        "stage2-run": {"run_type": "formal-exp3-stage2"},
    }

    class FakeRunStore:
        directory = tmp_path / "outputs/exp3"

        def resolve(self, run_id):
            return tmp_path / "outputs/exp3" / run_id

        def read(self, run_id, filename):
            assert filename == "run.json"
            return run_info[run_id]

    service = ClassificationStage3Service.__new__(ClassificationStage3Service)
    service.settings = SimpleNamespace(root=tmp_path, max_text_length=5000)
    service.store = FakeRunStore()
    calls = []

    def fake_subprocess_run(command, **kwargs):
        calls.append(list(command))
        model = "C1" if "--model-id" in command else "C2"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"model_id": model}), stderr="")

    monkeypatch.setattr(classification_service.subprocess, "run", fake_subprocess_run)
    assert service.predict("toy news", "C0", "stage3-run") == {"model_id": "C1"}
    assert service.predict("toy news", "C1", "stage3-run") == {"model_id": "C1"}
    assert service.predict("toy news", "C2", "stage3-run") == {"model_id": "C2"}
    assert len(calls) == 3
    assert all(command[command.index("--run-id") + 1] == "stage1-run" for command in calls[:2])
    assert calls[2][calls[2].index("--run-id") + 1] == "stage2-run"
    assert "nlplab.classification.stage2" in calls[2]

    with pytest.raises(DataValidationError, match="Stage3 Run"):
        service.predictions("stage1-run", "C3")
    with pytest.raises(DataValidationError, match="Stage3 Run"):
        service.predictions("stage2-run", "C3")
