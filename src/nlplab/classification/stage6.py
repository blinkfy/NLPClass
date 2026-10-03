"""Offline statistics over frozen labels/predictions, without model or text access."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from nlplab.core.io import write_json
from nlplab.data.course import file_sha256
from nlplab.data.frozen_prediction_projection import load_projection
from nlplab.data.sequence_annotations import atomic_json
from nlplab.evaluation.classification_uncertainty import (
    BOOTSTRAP,
    MODELS,
    PRIMARY_PAIRS,
    align_predictions,
    analyze,
)
from nlplab.features.word_sequence import read_json

PARENTS = {
    "C0": "20261001_123337_9c1e7341",
    "C1": "20261001_123337_9c1e7341",
    "C2": "20261001_133829_68ee50d7",
    "C3": "20261001_153153_c99a8d6f",
}
PROTOCOL = {
    "analysis_type": "uncertainty and paired comparisons of frozen formal Test predictions",
    "primary_pairs": [list(pair) for pair in PRIMARY_PAIRS],
    "primary_family": "three two-sided exact McNemar tests; Holm familywise correction",
    "difference_direction": "B minus A for Accuracy and Macro-F1",
    "accuracy_interval": "95% Wilson score interval; no continuity correction",
    "recall_interval": "95% Wilson per class, n=18; marginal intervals, not simultaneous",
    "macro_f1_interval": "95% class-stratified percentile bootstrap",
    "bootstrap": BOOTSTRAP,
    "model_fit_count": 0,
    "new_test_inference_count": 0,
    "selection_or_retuning": False,
    "secondary_comparisons": [],
    "permitted_prediction_fields": ["document_id", "gold", "prediction"],
    "data_access": (
        "Narrowest existing prediction JSONL containers are projected at root fields. "
        "Other values skipped without decoding/extracting/retaining text or explanations; "
        "container bytes hashed during that single projection read. No model/data-split access."
    ),
    "limitations": [
        "This fixed Test90 was previously evaluated; no new independent benchmark is created.",
        "Intervals describe Test sampling conditional on class support18 and frozen models.",
        "Training randomness, dataset changes and distribution drift are excluded.",
        "Bootstrap difference intervals are marginal, not multiplicity-adjusted tests.",
    ],
}


def prediction_paths(root):
    base = root / "outputs/exp3"
    return {m: base / run / f"{m}_test_predictions.jsonl" for m, run in PARENTS.items()}


def install_statistics_guard(root):
    root = Path(root).resolve()
    allowed = set(prediction_paths(root).values())
    for run in set(PARENTS.values()):
        allowed.update(
            root / "outputs/exp3" / run / name
            for name in ("artifact_manifest.json", "test_metrics.json")
        )

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(root):
            return
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("data/", "artifacts/models/")):
            raise PermissionError("Stage6 forbids data text/features and model files")
        if relative.startswith("outputs/"):
            own = relative.startswith("outputs/exp3/stage6/") or (
                path.parent == root / "outputs/exp3" and path.name.startswith("stage6_")
            )
            if not own and path not in allowed:
                raise PermissionError(
                    "Stage6 only permits frozen prediction projection/metrics/manifests"
                )
            if not own and args[2] & (os.O_WRONLY | os.O_RDWR):
                raise PermissionError("Stage6 frozen prediction references are read-only")

    sys.addaudithook(guard)


def source_hashes(root):
    paths = [
        "classification/stage6.py",
        "evaluation/classification_uncertainty.py",
        "data/frozen_prediction_projection.py",
        "evaluation/classification.py",
    ]
    return {"src/nlplab/" + p: file_sha256(root / "src/nlplab" / p) for p in paths}


def publish_pointer(base, directory, name, value):
    # Reuse the existing atomic writer, with its temporary file inside our own run.
    staged = directory / name
    atomic_json(staged, value)
    os.replace(staged, base / name)


def references(root):
    result = {}
    for model, path in prediction_paths(root).items():
        manifest_path = path.parent / "artifact_manifest.json"
        manifest = read_json(manifest_path)["files"]
        metrics_path = path.parent / "test_metrics.json"
        if file_sha256(metrics_path) != manifest["test_metrics.json"]:
            raise ValueError("Frozen formal metrics hash changed; stop")
        result[model] = {
            "run_id": PARENTS[model],
            "prediction_path": path.relative_to(root).as_posix(),
            "prediction_sha256": manifest[path.name],
            "metrics_path": metrics_path.relative_to(root).as_posix(),
            "metrics_sha256": manifest["test_metrics.json"],
            "parent_manifest_sha256": file_sha256(manifest_path),
        }
    return result


def preregister(root):
    install_statistics_guard(root)
    base = root / "outputs/exp3"
    if (base / "stage6_preregistration.json").exists():
        raise ValueError("Stage6 already preregistered; no overwrite")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = base / "stage6" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", PROTOCOL)
    write_json(directory / "frozen_prediction_references.json", references(root))
    write_json(directory / "bootstrap_configuration.json", BOOTSTRAP)
    write_json(directory / "source_hashes.json", source_hashes(root))
    write_json(
        directory / "preregistration.json",
        {
            "run_id": run_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "prediction_analysis_started": False,
            "files": {
                name: file_sha256(directory / name)
                for name in (
                    "protocol.json",
                    "bootstrap_configuration.json",
                    "source_hashes.json",
                    "frozen_prediction_references.json",
                )
            },
        },
    )
    publish_pointer(
        base,
        directory,
        "stage6_preregistration.json",
        {
            "run_id": run_id,
            "preregistration_sha256": file_sha256(directory / "preregistration.json"),
        },
    )
    print(
        json.dumps({"run_id": run_id, "primary_pairs_frozen": PROTOCOL["primary_pairs"]}),
        flush=True,
    )


def run(root):
    install_statistics_guard(root)
    base = root / "outputs/exp3"
    pointer = read_json(base / "stage6_preregistration.json")
    directory = base / "stage6" / pointer["run_id"]
    if file_sha256(directory / "preregistration.json") != pointer["preregistration_sha256"]:
        raise ValueError("Stage6 preregistration changed")
    registration = read_json(directory / "preregistration.json")
    for name, digest in registration["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Stage6 protocol/config changed")
    refs = references(root)
    if (
        refs != read_json(directory / "frozen_prediction_references.json")
        or source_hashes(root) != read_json(directory / "source_hashes.json")
        or read_json(directory / "protocol.json") != PROTOCOL
    ):
        raise ValueError("Stage6 sources or frozen references changed")
    execution = directory / "execution.json"
    with execution.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "started",
                "model_fit_count": 0,
                "new_test_inference_count": 0,
                "rerun_allowed": False,
            },
            stream,
        )
    start = perf_counter()
    rows, metrics = {}, {}
    for m, info in refs.items():
        rows[m] = load_projection(root / info["prediction_path"], info["prediction_sha256"])
        metrics[m] = read_json(root / info["metrics_path"])[m]
    # Any ID/gold/support/metric inconsistency aborts here, before resampling.
    aligned, points, audit = align_predictions(rows, metrics)
    audit["prediction_projection"] = {
        "fields": ["document_id", "gold", "prediction"],
        "irrelevant_values_decoded": False,
        "source_container_hashes_verified": True,
    }
    write_json(directory / "alignment_audit.json", audit)
    write_json(directory / "aligned_predictions.json", aligned)
    write_json(directory / "point_metrics.json", points)
    statistics, boot, indices = analyze(aligned, points)
    write_json(directory / "statistics.json", statistics)
    write_json(
        directory / "bootstrap_summaries.json",
        {
            "models": statistics["models"],
            "pairs": statistics["pairs"],
            "shared_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            "replicate_count": 20000,
            "column_order": list(MODELS),
        },
    )
    np.savez_compressed(directory / "bootstrap_macro_f1.npz", macro_f1=boot)
    write_json(directory / "pairwise_outcomes.json", statistics["pairs"])
    write_json(
        directory / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")
            },
            "runtime": str(sys.executable),
            "device": "CPU statistics only",
        },
    )
    write_json(
        directory / "run.json",
        {
            "run_id": directory.name,
            "analysis_type": PROTOCOL["analysis_type"],
            "sample_count": 90,
            "primary_pair_count": 3,
            "bootstrap_replicates": 20000,
            "model_fit_count": 0,
            "new_test_inference_count": 0,
            "frozen_point_metrics_reproduced": True,
            "runtime_seconds": perf_counter() - start,
        },
    )
    write_json(
        execution,
        {
            "status": "complete",
            "model_fit_count": 0,
            "new_test_inference_count": 0,
            "rerun_allowed": False,
        },
    )
    write_json(
        directory / "artifact_manifest.json",
        {
            "algorithm": "sha256",
            "files": {
                p.relative_to(directory).as_posix(): file_sha256(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p.name != "artifact_manifest.json"
            },
        },
    )
    publish_pointer(
        base,
        directory,
        "stage6_latest.json",
        {"run_id": directory.name, "analysis_type": PROTOCOL["analysis_type"]},
    )
    print(
        json.dumps(
            {
                "run_id": directory.name,
                "alignment": "passed",
                "model_fit_count": 0,
                "new_test_inference_count": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--analyze-frozen", action="store_true")
    args = parser.parse_args()
    if args.preregister:
        preregister(args.root.resolve())
    elif args.analyze_frozen:
        run(args.root.resolve())
    else:
        parser.error("Explicit preregistration or frozen statistics execution required")
