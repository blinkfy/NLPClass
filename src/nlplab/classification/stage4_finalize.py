"""Finalize already-fitted development points after an interrupted worker; never trains."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nlplab.classification.stage4 import (
    MODEL_IDS,
    PROTOCOL,
    frozen_references,
    signature,
    source_hashes,
)
from nlplab.classification.stage4_analysis import aggregate_points, verify_full_size
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.classification_development import install_development_io_guard
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.word_sequence import read_json


def adopt_neural_result(directory, subset):
    point_dir = directory / "points" / f"ladder{subset['subset_seed']}_n{subset['train_size']}"
    output = point_dir / "C3_recovery"
    result = read_json(output / "result.json")
    training = result["training"]
    if (
        training["seed"],
        training["learning_rate"],
        training["epochs_executed"],
        training["early_stopping_used"],
    ) != (42, 0.001, 12, False):
        raise DataValidationError("Recovery model violates frozen training protocol")
    if result["validation_prediction_count"] != 1 or result["test_evaluation_count"] != 0:
        raise DataValidationError("Recovery evaluation protocol changed")
    train_meta = read_json(point_dir / "sequence_train.json")
    val_meta = read_json(point_dir / "sequence_val.json")
    if train_meta["document_ids"] != subset["document_ids"]:
        raise DataValidationError("Recovery subset IDs changed")
    with np.load(point_dir / "sequence_train.npz", allow_pickle=False) as data:
        train_gold = [CATEGORIES[i] for i in data["labels"]]
    with np.load(point_dir / "sequence_val.npz", allow_pickle=False) as data:
        val_gold = [CATEGORIES[i] for i in data["labels"]]
    train_pred = [CATEGORIES[i] for i in result["train"]["prediction_indices"]]
    val_pred = [CATEGORIES[i] for i in result["val"]["prediction_indices"]]
    train_metrics = classification_metrics(train_gold, train_pred)
    val_metrics = classification_metrics(val_gold, val_pred)
    stats = val_meta["stats"]
    representation = {
        "vocabulary_size": train_meta["stats"]["vocabulary_size"],
        "parameter_count": training["parameter_count"],
        "max_len": 768,
        "min_freq": 2,
        "model_seed": 42,
        "epochs": 12,
        "validation_unk_count": stats["unk_count"],
        "validation_unk_rate": stats["unk_rate"],
        "validation_retained_unk_rate": stats["retained_unk_rate"],
        "validation_truncated_count": stats["truncated_count"],
        "validation_truncation_rate": stats["truncation_rate"],
        "validation_retained_token_ratio": stats["retained_token_ratio"],
        "final_training_loss": training["history"][-1]["train_loss"],
        "train_encoding_stats": train_meta["stats"],
        "val_encoding_stats": stats,
    }
    point = {
        "model_id": "C3",
        "subset_seed": subset["subset_seed"],
        "train_size": subset["train_size"],
        "training_ids_sha256": subset["ids_sha256"],
        "representation_fit_ids": subset["document_ids"],
        "representation": representation,
        "representation_reference": "sequence_vocabulary",
        "model_reference": "C3_recovery/model",
        "recovered_after_interruption": True,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "train_predictions": train_pred,
        "validation_predictions": val_pred,
        "training_history": training["history"],
        "early_stopping_used": False,
        "runtime": {
            "representation_fit_seconds": None,
            "fit_seconds": None,
            "classifier_fit_seconds": training["runtime_seconds"],
            "train_inference_seconds": result["train"]["inference_seconds"],
            "validation_inference_seconds": result["val"]["inference_seconds"],
            "note": (
                "Subset vocabulary/encoding already fitted before interruption; "
                "its timer was not persisted"
            ),
        },
        "formal_test_evaluation_count": 0,
        "validation_prediction_count": 1,
        "deterministic_signature": signature(
            subset["document_ids"],
            train_metrics,
            val_metrics,
            train_pred,
            val_pred,
            result["state_sha256"],
        ),
    }
    write_json(point_dir / "C3_result.json", point)
    return point


def finalize(root, run_id):
    install_development_io_guard(root)
    base = root / "outputs/exp3"
    pointer = read_json(base / "stage4_preregistration.json")
    if run_id != pointer["run_id"]:
        raise DataValidationError("Recovery can only finalize the preregistered analysis")
    directory = base / "stage4" / run_id
    guard_path = directory / "execution.json"
    if read_json(guard_path)["status"] != "started":
        raise DataValidationError("Do not finalize completed analysis again")
    started_timestamp = guard_path.stat().st_mtime
    registration = read_json(directory / "preregistration.json")
    if file_sha256(directory / "preregistration.json") != pointer["preregistration_sha256"]:
        raise DataValidationError("Preregistration changed")
    if file_sha256(directory / "subset_ladders.json") != pointer["subset_ladders_sha256"]:
        raise DataValidationError("Subset ladder changed")
    references, _ = frozen_references(root)
    if (
        references != registration["frozen_config_references"]
        or registration["protocol"] != PROTOCOL
    ):
        raise DataValidationError("Frozen configurations/protocol changed")
    current_hashes = source_hashes(root)
    changed = [p for p, sha in current_hashes.items() if sha != registration["source_hashes"][p]]
    permitted = "src/nlplab/data/classification_development.py"
    if changed != [permitted]:
        raise DataValidationError("Unexpected source amendment")
    original = (
        (root / permitted)
        .read_bytes()
        .replace(
            b'    stage4_metadata = (\n        relative.startswith("outputs/exp3/stage4/")\n'
            b'        or (path.parent == root / "outputs/exp3" '
            b'and path.name.startswith("stage4_"))\n'
            b'    )\n    if relative.startswith("outputs/exp3/") and not stage4_metadata:',
            b'    if relative.startswith("outputs/exp3/") '
            b'and not relative.startswith("outputs/exp3/stage4/"):',
            1,
        )
    )
    if hashlib.sha256(original).hexdigest() != registration["source_hashes"][permitted]:
        raise DataValidationError("Guard amendment is more than the documented metadata exception")
    (directory / "original_development_guard.py").write_bytes(original)
    write_json(
        directory / "source_amendment.json",
        {
            "path": permitted,
            "before_sha256": registration["source_hashes"][permitted],
            "after_sha256": current_hashes[permitted],
            "reason": (
                "The substring test in latest blocked Stage4-only metadata; "
                "allow stage4_ pointers"
            ),
            "training_or_representation_configuration_changed": False,
            "finalizer_source_sha256": file_sha256(Path(__file__)),
            "finalizer_trains_models": False,
        },
    )
    ladders = read_json(directory / "subset_ladders.json")
    points = []
    for subset in ladders:
        point_dir = directory / "points" / f"ladder{subset['subset_seed']}_n{subset['train_size']}"
        for model in MODEL_IDS:
            path = point_dir / f"{model}_result.json"
            if path.is_file():
                points.append(read_json(path))
            elif model == "C3" and subset["subset_seed"] == 303 and subset["train_size"] == 420:
                points.append(adopt_neural_result(directory, subset))
            else:
                raise DataValidationError("Missing curve point; finalization must not retrain")
    sanity = verify_full_size(points)
    aggregates = aggregate_points(points)
    write_json(directory / "raw_results.json", points)
    write_json(directory / "learning_curves.json", aggregates)
    write_json(
        directory / "per_class_curves.json",
        [
            {
                "model_id": p["model_id"],
                "train_size": p["train_size"],
                "per_class_f1": p["per_class_f1"],
            }
            for p in aggregates
        ],
    )
    write_json(
        directory / "representation_diagnostics.json",
        [
            {
                "model_id": p["model_id"],
                "train_size": p["train_size"],
                "representation": p["representation"],
            }
            for p in aggregates
        ],
    )
    write_json(directory / "sanity_420.json", sanity)
    wall = datetime.now(timezone.utc).timestamp() - started_timestamp
    write_json(
        directory / "runtimes.json",
        {
            "formal_wall_seconds": wall,
            "wall_definition": "execution guard start through recovery/finalization",
            "model_fit_seconds": {
                m: sum(p["runtime"]["classifier_fit_seconds"] for p in points if p["model_id"] == m)
                for m in MODEL_IDS
            },
            "representation_fit_seconds_shared_word_once": sum(
                p["runtime"]["representation_fit_seconds"] for p in points if p["model_id"] == "C0"
            ),
            "missing_encoding_runtime_points": 1,
            "points": [
                {
                    "model_id": p["model_id"],
                    "subset_seed": p["subset_seed"],
                    "train_size": p["train_size"],
                    **p["runtime"],
                }
                for p in points
            ],
        },
    )
    run = {
        "run_id": run_id,
        "run_type": "exp3-stage4-development-learning-curves",
        "models": list(MODEL_IDS),
        "train_sizes": PROTOCOL["train_sizes"],
        "subset_seeds": list(PROTOCOL["subset_seeds"]),
        "model_fits": 60,
        "c3_training_runs": 15,
        "formal_test_evaluation_count": 0,
        "allowed_splits": ["train", "val"],
        "historical_validation_reused": True,
        "config_references": references,
        "sanity_420_passed": True,
    }
    write_json(directory / "run.json", run)
    write_json(
        directory / "recovery_record.json",
        {
            "completed_points_preserved": 59,
            "recovered_point": "C3 ladder303 n420",
            "interrupted_attempts_without_checkpoint": 1,
            "rejected_pretraining_retries": 1,
            "successful_c3_training_runs": 15,
            "c3_worker_invocations": 17,
            "initial_controller_import_failure": "WinError1450; before execution guard or any fit",
            "worker_failure": (
                "Nonzero exit before checkpoint; original worker exit code was not persisted"
            ),
            "preserved_empty_output": "points/ladder303_n420/C3",
            "new_output": "points/ladder303_n420/C3_recovery",
            "configuration_changed": False,
        },
    )
    atomic_json(
        guard_path,
        {
            "status": "complete",
            "model_fits": 60,
            "c3_training_runs": 15,
            "formal_test_evaluation_count": 0,
            "test_accessed": False,
            "rerun_allowed": False,
            "recovery_record": "recovery_record.json",
        },
    )
    manifest = {
        p.relative_to(directory).as_posix(): file_sha256(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    write_json(directory / "artifact_manifest.json", {"algorithm": "sha256", "files": manifest})
    atomic_json(
        base / "stage4_latest.json", {"run_id": run_id, "analysis_type": PROTOCOL["analysis_type"]}
    )
    print(f"Finalized {run_id}: 60 completed curve points, Test=0, all 420 identical", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    finalize(args.root.resolve(), args.run_id)


if __name__ == "__main__":
    main()
