"""Toy-only contracts for Stage6 frozen-prediction uncertainty analysis."""

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import classification as classification_api
from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data import frozen_prediction_projection
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.evaluation.classification_uncertainty import (
    MODELS,
    align_predictions,
    bootstrap_macro_f1,
    exact_mcnemar,
    holm_adjust,
    stratified_indices,
    wilson,
)
from nlplab.services.classification_stage6_service import ClassificationStage6Service


def _toy_fixture():
    rows_by_model = {model_id: [] for model_id in MODELS}
    gold = []
    for class_index, label in enumerate(CATEGORIES):
        for within_class in range(18):
            document_id = f"toy-{class_index}-{within_class}"
            gold.append(label)
            for model_index, model_id in enumerate(MODELS):
                prediction = label
                if within_class < model_index + 1:
                    prediction = CATEGORIES[(class_index + 1) % len(CATEGORIES)]
                rows_by_model[model_id].append(
                    {"document_id": document_id, "gold": label, "prediction": prediction}
                )

    metrics = {
        model_id: classification_metrics(
            gold, [row["prediction"] for row in rows_by_model[model_id]]
        )
        for model_id in MODELS
    }
    return rows_by_model, metrics


def test_alignment_joins_by_document_id_and_reproduces_frozen_metrics_exactly():
    rows_by_model, metrics = _toy_fixture()
    rows_by_model["C1"] = rows_by_model["C1"][18:] + rows_by_model["C1"][:18]
    rows_by_model["C2"] = list(reversed(rows_by_model["C2"]))

    aligned, points, audit = align_predictions(rows_by_model, metrics)

    assert [row["document_id"] for row in aligned[:3]] == ["toy-0-0", "toy-0-1", "toy-0-2"]
    assert aligned[0] == {
        "document_id": "toy-0-0",
        "gold": CATEGORIES[0],
        "C0": CATEGORIES[1],
        "C1": CATEGORIES[1],
        "C2": CATEGORIES[1],
        "C3": CATEGORIES[1],
    }
    assert points == metrics
    assert audit["support"] == {label: 18 for label in CATEGORIES}
    assert audit["original_order_identical"] == {
        "C0": True,
        "C1": False,
        "C2": False,
        "C3": True,
    }


@pytest.mark.parametrize("problem", ["duplicate", "id_set", "gold", "support", "metrics"])
def test_alignment_rejects_invalid_frozen_inputs(problem):
    rows_by_model, metrics = _toy_fixture()
    if problem == "duplicate":
        rows_by_model["C1"][0]["document_id"] = rows_by_model["C1"][1]["document_id"]
    elif problem == "id_set":
        rows_by_model["C3"][0]["document_id"] = "different-id"
    elif problem == "gold":
        rows_by_model["C2"][0]["gold"] = CATEGORIES[1]
    elif problem == "support":
        for model_id in MODELS:
            rows_by_model[model_id][0]["gold"] = CATEGORIES[1]
    else:
        metrics["C3"]["accuracy"] += 0.01

    with pytest.raises(DataValidationError):
        align_predictions(rows_by_model, metrics)


def test_existing_classification_metrics_fixture_remains_exact():
    gold = ["finance", "education", "science", "sports", "entertainment"]
    predicted = ["finance", "science", "science", "entertainment", "entertainment"]

    metrics = classification_metrics(gold, predicted)

    assert metrics["label_order"] == list(CATEGORIES)
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    assert metrics["sample_count"] == 5
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["micro_f1"] == pytest.approx(0.6)
    assert metrics["macro_precision"] == pytest.approx(0.4)
    assert metrics["macro_recall"] == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(7 / 15)
    assert metrics["weighted_f1"] == pytest.approx(7 / 15)


def test_wilson_bounds_cover_zero_and_all_successes():
    zero = wilson(0, 18)
    all_successes = wilson(18, 18)

    assert zero["point"] == 0
    assert zero["low"] == 0
    assert zero["high"] == pytest.approx(0.1759, abs=0.0001)
    assert all_successes["point"] == 1
    assert all_successes["high"] == 1
    assert all_successes["low"] == pytest.approx(0.8241, abs=0.0001)


def test_bootstrap_uses_seeded_shared_class_stratified_indices_and_matches_metrics():
    gold_indices = np.repeat(np.arange(len(CATEGORIES)), 18)
    indices = stratified_indices(gold_indices, replicates=12, seed=606)

    np.testing.assert_array_equal(indices, stratified_indices(gold_indices, 12, 606))
    assert indices.shape == (12, 90)
    for replicate in indices:
        np.testing.assert_array_equal(np.bincount(gold_indices[replicate]), np.full(5, 18))

    predictions = np.stack(
        [
            gold_indices,
            np.where(np.tile(np.arange(18), 5) < 2, (gold_indices + 1) % 5, gold_indices),
            np.where(np.tile(np.arange(18), 5) < 3, (gold_indices + 1) % 5, gold_indices),
            np.where(np.tile(np.arange(18), 5) < 4, (gold_indices + 1) % 5, gold_indices),
        ]
    )
    bootstrap = bootstrap_macro_f1(gold_indices, predictions, indices)
    labels = np.asarray(CATEGORIES)
    expected = np.empty_like(bootstrap)
    for replicate_index, sampled in enumerate(indices):
        sampled_gold = labels[gold_indices[sampled]].tolist()
        for model_index, prediction in enumerate(predictions):
            sampled_predictions = labels[prediction[sampled]].tolist()
            expected[replicate_index, model_index] = classification_metrics(
                sampled_gold, sampled_predictions
            )["macro_f1"]
    np.testing.assert_allclose(bootstrap, expected, rtol=0, atol=1e-15)

    identical = bootstrap_macro_f1(gold_indices, np.repeat(predictions[:1], 2, axis=0), indices)
    np.testing.assert_array_equal(identical[:, 0], identical[:, 1])
    np.testing.assert_array_equal(identical[:, 1] - identical[:, 0], np.zeros(len(indices)))


@pytest.mark.parametrize(
    ("a_only", "b_only", "expected"),
    [(0, 0, 1.0), (5, 0, 0.0625), (3, 1, 0.625)],
)
def test_exact_mcnemar_uses_two_sided_binomial_counts(a_only, b_only, expected):
    assert exact_mcnemar(a_only, b_only) == pytest.approx(expected)


def test_holm_adjustment_preserves_input_order_and_handles_ties():
    assert holm_adjust([0.01, 0.04, 0.01, 0.5]) == pytest.approx([0.04, 0.08, 0.04, 0.5])


def test_projection_skips_irrelevant_values_without_decoding_them(monkeypatch):
    raw = (
        b'{"document_id":"toy-1","gold":"finance","prediction":"finance",'
        b'"raw_text":"toy private text","features":{"term":3},"scores":[1,2],'
        b'"nested":{"x":[true,null]}}'
    )
    decoded = []
    original_loads = frozen_prediction_projection.json.loads

    def recording_loads(value, *args, **kwargs):
        decoded.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(frozen_prediction_projection.json, "loads", recording_loads)
    projected = frozen_prediction_projection.project_record(raw)

    assert projected == {"document_id": "toy-1", "gold": "finance", "prediction": "finance"}
    assert decoded == [
        b'"document_id"',
        b'"toy-1"',
        b'"gold"',
        b'"finance"',
        b'"prediction"',
        b'"finance"',
        b'"raw_text"',
        b'"features"',
        b'"scores"',
        b'"nested"',
    ]


def test_projection_hashes_the_source_container_and_rejects_a_mismatch(tmp_path):
    source_bytes = (
        b'{"document_id":"toy-1","gold":"finance","prediction":"science",'
        b'"raw_text":"ignored toy text","scores":[0.8,0.2]}\n'
    )
    source = tmp_path / "toy-predictions.jsonl"
    source.write_bytes(source_bytes)

    rows = frozen_prediction_projection.load_projection(
        source, hashlib.sha256(source_bytes).hexdigest()
    )

    assert rows == [{"document_id": "toy-1", "gold": "finance", "prediction": "science"}]
    with pytest.raises(DataValidationError, match="container hash mismatch"):
        frozen_prediction_projection.load_projection(source, "0" * 64)


def test_stage6_read_only_api_routes_use_a_fake_service(tmp_path, monkeypatch):
    application = create_app(Settings(root=tmp_path), service=object())
    output = tmp_path / "protocol.json"
    output.write_text('{"toy":true}\n', encoding="utf-8")
    aligned = [{"document_id": "toy-1", "gold": "finance", **dict.fromkeys(MODELS, "finance")}]
    fake_service = type(
        "FakeStage6Service",
        (),
        {
            "overview": lambda self, run_id: {"run_id": run_id, "statistics": {"toy": True}},
            "aligned_labels": lambda self, run_id: aligned,
            "export_path": lambda self, run_id, name: output,
        },
    )()
    monkeypatch.setattr(classification_api, "stage6_service", lambda request: fake_service)

    with TestClient(application) as client:
        overview = client.get("/api/v1/experiments/3/stage6/overview")
        aligned_response = client.get("/api/v1/experiments/3/stage6/export/aligned")
        protocol = client.get("/api/v1/experiments/3/stage6/export/protocol")
        schema = client.get("/openapi.json").json()

    assert overview.status_code == 200
    assert overview.json() == {"run_id": "latest", "statistics": {"toy": True}}
    assert aligned_response.json() == aligned
    assert protocol.status_code == 200
    assert protocol.json() == {"toy": True}
    assert "get" in schema["paths"]["/api/v1/experiments/3/stage6/overview"]
    assert "get" in schema["paths"]["/api/v1/experiments/3/stage6/export/{name}"]


def test_stage6_controller_source_guards_against_model_fitting_and_inference():
    source_path = Path(__file__).resolve().parents[1] / "src/nlplab/classification/stage6.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        name.startswith("nlplab.classification.")
        or name.split(".", 1)[0] in {"sklearn", "torch", "transformers"}
        for name in imports
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fit", "predict", "predict_proba", "decision_function", "scores"}
        for node in ast.walk(tree)
    )

    guarded_entrypoints = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"preregister", "run"}
    }
    assert set(guarded_entrypoints) == {"preregister", "run"}
    for function in guarded_entrypoints.values():
        first_statement = function.body[0]
        assert isinstance(first_statement, ast.Expr)
        assert isinstance(first_statement.value, ast.Call)
        assert isinstance(first_statement.value.func, ast.Name)
        assert first_statement.value.func.id == "install_statistics_guard"


def test_stage6_guard_allows_preregistration_pointer_and_blocks_legacy_input_writes(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import json
import sys
from pathlib import Path

from nlplab.classification.stage6 import PARENTS, install_statistics_guard, publish_pointer

root = Path(sys.argv[1]).resolve()
base = root / "outputs/exp3"
legacy_prediction = base / PARENTS["C0"] / "C0_test_predictions.jsonl"
legacy_metrics = base / PARENTS["C0"] / "test_metrics.json"
for path in (legacy_prediction, legacy_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("toy input", encoding="utf-8")

directory = base / "stage6" / "toy-run"
directory.mkdir(parents=True)
install_statistics_guard(root)
publish_pointer(base, directory, "stage6_preregistration.json", {"run_id": "toy-run"})
pointer = base / "stage6_preregistration.json"
assert json.loads(pointer.read_text(encoding="utf-8")) == {"run_id": "toy-run"}

for path in (legacy_prediction, legacy_metrics):
    try:
        path.write_text("forbidden overwrite", encoding="utf-8")
    except PermissionError:
        pass
    else:
        raise AssertionError(f"guard allowed a write to {path.name}")
print("toy guard contract passed")
"""
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, str(project_root), environment.get("PYTHONPATH", "")) if path
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "toy guard contract passed"


def test_stage6_manifest_export_does_not_require_a_self_hash(tmp_path):
    directory = tmp_path / "outputs/exp3/stage6/toy-run"
    directory.mkdir(parents=True)
    run_path = directory / "run.json"
    run_bytes = b'{"run_id":"toy-run"}\n'
    run_path.write_bytes(run_bytes)
    manifest_path = directory / "artifact_manifest.json"
    manifest_path.write_text(
        '{"algorithm":"sha256","files":{"run.json":"'
        + hashlib.sha256(run_bytes).hexdigest()
        + '"}}\n',
        encoding="utf-8",
    )
    service = ClassificationStage6Service(Settings(root=tmp_path))

    assert service.export_path("toy-run", "manifest") == manifest_path
    with pytest.raises(DataValidationError, match="不支持的 Stage6 分析导出"):
        service.export_path("toy-run", "unknown")
