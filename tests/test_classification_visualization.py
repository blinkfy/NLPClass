"""Toy contracts for saved linear explanations and frozen TextCNN traces."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nlplab.classification.textcnn import TextCNN, explain_batch
from nlplab.classification.visualization_trace import trace_model
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.services.classification_visualization_service import (
    ClassificationVisualizationService,
)


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


def _write_csr(path, dimension, columns, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.asarray(b"csr"),
        shape=np.asarray([1, dimension], dtype=np.int64),
        indptr=np.asarray([0, len(columns)], dtype=np.int32),
        indices=np.asarray(columns, dtype=np.int32),
        data=np.asarray(values, dtype=np.float64),
    )


def _scores_for_document(weights, intercept, columns, values):
    return intercept + (weights[:, columns] * values).sum(axis=1)


def _prediction_row(document_id, weights, intercept, columns, values):
    scores = _scores_for_document(weights, intercept, columns, values)
    order = sorted(range(len(CATEGORIES)), key=lambda i: (-scores[i], CATEGORIES[i]))
    return {
        "document_id": document_id,
        "prediction": CATEGORIES[order[0]],
        "runner_up": CATEGORIES[order[1]],
        "scores": [
            {"label": label, "value": float(scores[i])}
            for i, label in enumerate(CATEGORIES)
        ],
    }


@pytest.fixture
def frozen_visualization_runs(tmp_path):
    output = tmp_path / "outputs/exp3"
    stage1, stage2, stage3 = (output / name for name in ("stage1", "stage2", "stage3"))
    known_id, unknown_id = "toy-test-id", "record-outside-test-split"
    dataset = {"splits": {"test": {"ids": [known_id]}}}
    classes = len(CATEGORIES)

    # Small frozen C1 CSR and numeric classifier parameters.
    c1_columns = np.asarray([0, 2], dtype=np.int32)
    c1_values = np.asarray([0.5, 1.25], dtype=np.float64)
    c1_weights = np.asarray(
        [
            [0.25, 0.0, -0.1],
            [0.2, 0.4, 0.1],
            [-0.1, 0.2, 0.7],
            [0.3, -0.2, 0.4],
            [-0.5, 0.1, 0.3],
        ],
        dtype=np.float64,
    )
    c1_intercept = np.asarray([0.1, 0.0, 0.2, -0.1, 0.05], dtype=np.float64)
    _write_json(stage1 / "run.json", {"run_id": "stage1", "run_type": "formal-exp3-stage1"})
    _write_json(stage1 / "dataset.json", dataset)
    _write_json(
        stage1 / "final/tfidf/vocabulary.json",
        {"vocabulary": {"alpha": 0, "beta": 1, "gamma": 2}},
    )
    np.save(stage1 / "final/tfidf/idf.npy", np.asarray([1.1, 1.2, 1.3]))
    _write_json(
        stage1 / "final/C1/model.json",
        {"model_id": "C1", "label_order": list(CATEGORIES)},
    )
    np.savez_compressed(
        stage1 / "final/C1/parameters.npz",
        weights=c1_weights,
        intercept=c1_intercept,
    )
    _write_csr(stage1 / "final/test.npz", 3, c1_columns, c1_values)
    _write_jsonl(
        stage1 / "C1_test_predictions.jsonl",
        [
            _prediction_row(doc_id, c1_weights, c1_intercept, c1_columns, c1_values)
            for doc_id in (known_id, unknown_id)
        ],
    )
    _write_manifest(
        stage1,
        [
            "run.json",
            "dataset.json",
            "final/tfidf/vocabulary.json",
            "final/tfidf/idf.npy",
            "final/C1/model.json",
            "final/C1/parameters.npz",
            "final/test.npz",
            "C1_test_predictions.jsonl",
        ],
    )

    # C2 selection order differs from raw vocabulary order to exercise index alignment.
    c2_columns = np.asarray([0, 1], dtype=np.int32)
    c2_values = np.asarray([1.5, 0.25], dtype=np.float64)
    c2_weights = np.asarray(
        [
            [0.1, 0.2],
            [1.0, -0.5],
            [0.3, 0.4],
            [-0.1, 0.2],
            [0.2, -0.3],
        ],
        dtype=np.float64,
    )
    c2_intercept = np.asarray([0.0, 0.1, 0.2, -0.1, 0.3], dtype=np.float64)
    raw_vocabulary = {"甲乙": 0, "乙丙": 1, "丙丁": 2, "丁戊": 3}
    selected = [
        {"column": 0, "term": "丙丁", "chi2_score": 5.2},
        {"column": 1, "term": "甲乙", "chi2_score": 9.0},
    ]
    _write_json(
        stage2 / "run.json",
        {"run_id": "stage2", "run_type": "formal-exp3-stage2", "stage1_run_id": "stage1"},
    )
    _write_json(stage2 / "dataset.json", dataset)
    _write_json(
        stage2 / "final/representation/representation.json",
        {
            "k": 2,
            "selector_tie_break": "score descending, term ascending",
            "vocabulary": raw_vocabulary,
        },
    )
    _write_json(stage2 / "final/representation/selected_features.json", selected)
    np.savez_compressed(
        stage2 / "final/representation/parameters.npz",
        selected_indices=np.asarray([2, 0], dtype=np.int64),
        chi2_scores=np.asarray([9.0, 1.1, 5.2, 0.4], dtype=np.float64),
        df=np.asarray([4, 2, 3, 1], dtype=np.int64),
        idf=np.asarray([1.4, 1.1], dtype=np.float64),
    )
    _write_json(
        stage2 / "final/C2/model.json",
        {"model_id": "C2", "label_order": list(CATEGORIES)},
    )
    np.savez_compressed(
        stage2 / "final/C2/parameters.npz",
        weights=c2_weights,
        intercept=c2_intercept,
    )
    _write_csr(stage2 / "final/test.npz", 2, c2_columns, c2_values)
    _write_jsonl(
        stage2 / "C2_test_predictions.jsonl",
        [
            _prediction_row(doc_id, c2_weights, c2_intercept, c2_columns, c2_values)
            for doc_id in (known_id, unknown_id)
        ],
    )
    _write_manifest(
        stage2,
        [
            "run.json",
            "dataset.json",
            "final/representation/representation.json",
            "final/representation/selected_features.json",
            "final/representation/parameters.npz",
            "final/C2/model.json",
            "final/C2/parameters.npz",
            "final/test.npz",
            "C2_test_predictions.jsonl",
        ],
    )

    _write_json(
        stage3 / "run.json",
        {
            "run_id": "stage3",
            "run_type": "formal-exp3-stage3",
            "stage1_run_id": "stage1",
            "stage2_run_id": "stage2",
        },
    )
    service = ClassificationVisualizationService(Settings(root=tmp_path))
    return SimpleNamespace(
        service=service,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        known_id=known_id,
        unknown_id=unknown_id,
        c1_columns=c1_columns,
        c1_values=c1_values,
        c1_weights=c1_weights,
        c1_intercept=c1_intercept,
        c2_columns=c2_columns,
        c2_values=c2_values,
        c2_weights=c2_weights,
        c2_intercept=c2_intercept,
        selected=selected,
    )


@pytest.mark.parametrize("model_id", ["C1", "C2"])
def test_decision_decomposes_frozen_csr_and_coefficients_without_fit(
    frozen_visualization_runs, model_id
):
    runs = frozen_visualization_runs

    result = runs.service.decision("stage3", runs.known_id, model_id)
    weights = runs.c1_weights if model_id == "C1" else runs.c2_weights
    intercept = runs.c1_intercept if model_id == "C1" else runs.c2_intercept
    columns = runs.c1_columns if model_id == "C1" else runs.c2_columns
    values = runs.c1_values if model_id == "C1" else runs.c2_values
    scores = _scores_for_document(weights, intercept, columns, values)
    order = sorted(range(len(CATEGORIES)), key=lambda i: (-scores[i], CATEGORIES[i]))

    assert result["view_run_id"] == "stage3"
    assert result["source_run_id"] == ("stage1" if model_id == "C1" else "stage2")
    assert result["prediction"] == CATEGORIES[order[0]]
    assert result["runner_up"] == CATEGORIES[order[1]]
    assert result["predicted_score"] == pytest.approx(scores[order[0]])
    assert result["runner_up_score"] == pytest.approx(scores[order[1]])
    assert result["margin"] == pytest.approx(scores[order[0]] - scores[order[1]])
    assert result["provenance"]["fit_executed"] is False
    assert result["provenance"]["classifier_inference_executed"] is False
    assert {row["column"] for row in result["features"]} == set(columns)
    predicted = CATEGORIES.index(result["prediction"])
    runner_up = CATEGORIES.index(result["runner_up"])
    for feature in result["features"]:
        column = feature["column"]
        position = list(columns).index(column)
        assert feature["tfidf"] == pytest.approx(values[position])
        assert feature["direct_contribution"] == pytest.approx(
            values[position] * weights[predicted, column]
        )
        assert feature["relative_contribution"] == pytest.approx(
            values[position] * (weights[predicted, column] - weights[runner_up, column])
        )

    if model_id == "C2":
        assert result["chi2_metadata"]["甲乙"] == {
            "raw_column": 0,
            "chi2_score": 9.0,
            "df": 4,
            "display_rank": 1,
        }
        assert result["chi2_metadata"]["丙丁"]["display_rank"] == 2
        assert result["rank_definition"] == "descending frozen chi2_score; ties by raw column; display rank only"
    else:
        assert result["chi2_metadata"] == {}


@pytest.mark.parametrize("model_id", ["C1", "C2"])
def test_decision_rejects_record_outside_frozen_test_ids(frozen_visualization_runs, model_id):
    with pytest.raises(DataValidationError, match="当前冻结 Test ID"):
        frozen_visualization_runs.service.decision(
            "stage3", frozen_visualization_runs.unknown_id, model_id
        )


@pytest.mark.parametrize(
    ("model_id", "directory"), [("C1", "stage1"), ("C2", "stage2")]
)
def test_decision_rejects_wrong_saved_label_order(
    frozen_visualization_runs, model_id, directory
):
    runs = frozen_visualization_runs
    source = getattr(runs, directory)
    model_path = source / f"final/{model_id}/model.json"
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    metadata["label_order"] = list(reversed(CATEGORIES))
    _write_json(model_path, metadata)

    manifest_path = source / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][f"final/{model_id}/model.json"] = file_sha256(model_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(DataValidationError, match="类别顺序不符"):
        runs.service.decision("stage3", runs.known_id, model_id)


def _textcnn_fixture():
    torch.manual_seed(27)
    model = TextCNN(vocabulary_size=12).eval()
    with torch.no_grad():
        model.embedding.weight[2:].fill_(-1.0)
        for conv in model.convs:
            conv.weight.fill_(1.0)
            conv.bias.fill_(400.0)

    tokens = [f"tok-{i}" for i in range(7)]
    token_ids = np.asarray([2, 3, 4, 5, 6, 7, 8, 0, 0, 0], dtype=np.int64)
    encoded = {
        "ids": token_ids[None, :],
        "lengths": np.asarray([len(tokens)], dtype=np.int64),
        "tokens": [tokens],
        "original_positions": [list(range(len(tokens)))],
    }
    vocabulary = SimpleNamespace(terms=["<PAD>", "<UNK>", *tokens, "unused-1", "unused-2", "unused-3"])
    saved = explain_batch(model, encoded, vocabulary)[0]
    return model, token_ids, len(tokens), encoded, vocabulary, saved


def test_trace_model_checks_every_valid_position_pooling_contributions_and_padding():
    model, token_ids, length, encoded, vocabulary, saved = _textcnn_fixture()
    trace = trace_model(model, token_ids, length, saved)

    assert model.training is False
    assert len(trace["pooled"]) == len(trace["argmax"]) == 384
    assert trace["verification"]["logits_match"] is True
    assert trace["verification"]["pooled_match"] is True
    assert trace["verification"]["argmax_match"] is True
    assert trace["fit_executed"] is False
    assert trace["formal_evaluation_executed"] is False

    embedded = model.embedding(torch.as_tensor(encoded["ids"], dtype=torch.long)).transpose(1, 2)
    group_start = 0
    for kernel, conv in zip((3, 4, 5), model.convs):
        valid_positions = length - kernel + 1
        with torch.inference_mode():
            expected = torch.relu(conv(embedded))[0, :, :valid_positions].cpu().numpy()
        actual = np.asarray(trace["activations"][str(kernel)], dtype=np.float32)
        assert actual.shape == (128, valid_positions)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(trace["pooled"][group_start : group_start + 128]),
            expected.max(axis=1),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(trace["argmax"][group_start : group_start + 128]),
            expected.argmax(axis=1),
        )

        # Deliberately positive biases make padded tail windows larger than valid
        # windows; only the true-length activation positions may enter the trace.
        all_positions = torch.relu(conv(embedded))[0, :, :].detach().cpu().numpy()
        assert np.any(all_positions[:, valid_positions:] > expected.max(axis=1)[:, None])
        group_start += 128

    weights = np.asarray(trace["weights"], dtype=np.float64)
    pooled = np.asarray(trace["pooled"], dtype=np.float64)
    bias = np.asarray(trace["bias"], dtype=np.float64)
    logits = np.asarray(trace["logits"], dtype=np.float64)
    contributions = weights * pooled[None, :]
    assert contributions.shape == (5, 384)
    np.testing.assert_allclose(bias + contributions.sum(axis=1), logits, atol=2e-5, rtol=2e-5)

    with torch.inference_mode():
        padded_logits, padded_pooled, padded_positions = model(
            torch.as_tensor(token_ids[None, :]), torch.tensor([length]), True
        )
        clean_logits, clean_pooled, clean_positions = model(
            torch.as_tensor(token_ids[:length][None, :]), torch.tensor([length]), True
        )
    torch.testing.assert_close(padded_logits, clean_logits)
    torch.testing.assert_close(padded_pooled, clean_pooled)
    torch.testing.assert_close(padded_positions, clean_positions)
    assert torch.equal(model.embedding.weight[0], torch.zeros(128))


def test_trace_model_rejects_mismatched_saved_logits():
    model, token_ids, length, _, _, saved = _textcnn_fixture()
    corrupted = deepcopy(saved)
    corrupted["raw_logits"][0] += 0.25

    with pytest.raises(DataValidationError, match="diagnostic logits/prediction"):
        trace_model(model, token_ids, length, corrupted)


def test_c3_trace_route_casts_integer_query_parameters_without_running_trace(monkeypatch, tmp_path):
    calls = []

    def fake_trace(self, run_id, document_id, kernel, filter_id):
        calls.append((run_id, document_id, kernel, filter_id))
        return {"run_id": run_id, "document_id": document_id}

    monkeypatch.setattr(ClassificationVisualizationService, "trace", fake_trace)
    app = FastAPI()
    from app.api.classification import router as classification_router

    app.include_router(classification_router)
    app.state.classification = SimpleNamespace(settings=Settings(root=tmp_path))

    response = TestClient(app).get(
        "/api/v1/experiments/3/visualization/c3-trace",
        params={"run_id": "toy-run", "document_id": "toy-id", "kernel": "3", "filter_id": "17"},
    )

    assert response.status_code == 200
    assert response.json() == {"run_id": "toy-run", "document_id": "toy-id"}
    assert calls == [("toy-run", "toy-id", 3, 17)]
    assert all(type(value) is int for value in calls[0][2:])
