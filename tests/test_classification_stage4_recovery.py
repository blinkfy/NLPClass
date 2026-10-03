"""Toy-only regressions for metadata publication and checkpoint adoption; no training."""

import json
import subprocess
import sys

import numpy as np
import pytest

from nlplab.classification.stage4_finalize import adopt_neural_result
from nlplab.core.exceptions import DataValidationError
from nlplab.data.classification_development import forbidden_development_path


def test_metadata_pointer_exemption_keeps_test_files_blocked(tmp_path):
    for name in ("stage4_latest.json", "stage4_preregistration.json", "stage4_latest.json.tmp"):
        assert not forbidden_development_path(tmp_path, tmp_path / "outputs/exp3" / name)
    assert forbidden_development_path(tmp_path, tmp_path / "data/splits/thucnews_600/test.jsonl")
    assert forbidden_development_path(tmp_path, tmp_path / "data/course/thucnews_600.jsonl")
    assert forbidden_development_path(tmp_path, tmp_path / "outputs/exp3/legacy/test_metrics.json")
    script = (
        "from pathlib import Path; "
        "from nlplab.data.classification_development import install_development_io_guard; "
        f"root=Path({str(tmp_path)!r}); "
        "p=root/'data/splits/thucnews_600/test.jsonl'; p.parent.mkdir(parents=True); "
        "p.write_text('TOY ONLY'); install_development_io_guard(root); p.read_text()"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode != 0 and "PermissionError" in result.stderr


def recovery_fixture(root):
    subset = {
        "subset_seed": 303,
        "train_size": 5,
        "ids_sha256": "toy",
        "document_ids": [f"toy-{i}" for i in range(5)],
    }
    point = root / "points/ladder303_n5"
    output = point / "C3_recovery"
    output.mkdir(parents=True)
    stats = {
        "vocabulary_size": 10,
        "unk_count": 2,
        "unk_rate": 0.2,
        "retained_unk_rate": 0.2,
        "truncated_count": 0,
        "truncation_rate": 0.0,
        "retained_token_ratio": 1.0,
    }
    for name in ("train", "val"):
        np.savez_compressed(point / f"sequence_{name}.npz", labels=np.arange(5))
        (point / f"sequence_{name}.json").write_text(
            json.dumps({"document_ids": subset["document_ids"], "stats": stats}), encoding="utf-8"
        )
    result = {
        "training": {
            "seed": 42,
            "learning_rate": 0.001,
            "epochs_executed": 12,
            "early_stopping_used": False,
            "parameter_count": 100,
            "history": [{"epoch": i, "train_loss": 0.1} for i in range(1, 13)],
            "runtime_seconds": 1.0,
        },
        "state_sha256": "toy-parameters",
        "validation_prediction_count": 1,
        "test_evaluation_count": 0,
    }
    for name in ("train", "val"):
        result[name] = {"prediction_indices": list(range(5)), "inference_seconds": 0.01}
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return subset, point, result


def test_adoption_uses_saved_predictions_and_preserves_missing_timer(tmp_path):
    subset, point, _ = recovery_fixture(tmp_path)
    result = adopt_neural_result(tmp_path, subset)
    assert result["train_metrics"]["macro_f1"] == 1.0
    assert result["validation_metrics"]["macro_f1"] == 1.0
    assert result["formal_test_evaluation_count"] == 0
    assert result["validation_prediction_count"] == 1
    assert result["runtime"]["representation_fit_seconds"] is None
    assert result["model_reference"] == "C3_recovery/model"
    assert (point / "C3_result.json").is_file()


def test_adoption_rejects_epoch_or_subset_changes(tmp_path):
    subset, point, result = recovery_fixture(tmp_path)
    result["training"]["epochs_executed"] = 11
    path = point / "C3_recovery/result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(DataValidationError, match="training protocol"):
        adopt_neural_result(tmp_path, subset)
    result["training"]["epochs_executed"] = 12
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(DataValidationError, match="subset IDs"):
        adopt_neural_result(tmp_path, {**subset, "document_ids": ["changed"]})
