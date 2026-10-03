"""Frozen-method development learning curves; no Test input or model selection."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from nlplab.classification.__main__ import artifact_manifest, processor_for
from nlplab.classification.c2 import C2LinearClassifier
from nlplab.classification.linear import LinearTextClassifier
from nlplab.classification.stage3 import runtime_information, save_encoding, subprocess_environment
from nlplab.classification.stage4_analysis import (
    MODEL_IDS,
    PER_CLASS_SIZES,
    SUBSET_SEEDS,
    aggregate_points,
    build_ladders,
    effective_k,
    ids_hash,
    verify_full_size,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.classification_development import (
    DevelopmentClassificationData,
    install_development_io_guard,
)
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.char_chi2 import (
    COUNT_CONFIG,
    TFIDF_CONFIG,
    CharacterChi2TFIDF,
    CharacterCounts,
    char_bigrams,
    sparse_stats,
)
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig
from nlplab.features.word_sequence import WordSequenceVocabulary, read_json
from nlplab.services.experiment_service import RunStore

PARENTS = {
    "stage1": "20261001_123337_9c1e7341",
    "stage2": "20261001_133829_68ee50d7",
    "stage3": "20261001_153153_c99a8d6f",
}
PROTOCOL = {
    "analysis_type": "post-hoc development-set learning-curve analysis",
    "interpretation": (
        "Frozen methods at different training sizes on the previously-used development set; "
        "not independent generalization estimates"
    ),
    "subset_seeds": list(SUBSET_SEEDS),
    "per_class_sizes": list(PER_CLASS_SIZES),
    "train_sizes": [n * 5 for n in PER_CLASS_SIZES],
    "validation_count": 90,
    "subset_rule": 'per class sort by SHA256(f"{seed}:{document_id}"); take prefixes',
    "fit_order": "frozen Train manifest order within each selected subset (including all 420)",
    "representation_fit": "from scratch on current subset only; C0/C1 share one matrix object",
    "c2_effective_k": "min(10000, current subset raw dimension)",
    "c3": {
        "max_len": 768,
        "min_freq": 2,
        "learning_rate": 0.001,
        "seed": 42,
        "epochs": 12,
        "early_stopping": False,
        "validation_predictions": "one forward pass after epoch12",
    },
    "std_ddof": 0,
    "statistical_tests": False,
    "model_selection": False,
    "allowed_splits": ["train", "val"],
    "test_access_permitted": False,
    "formal_test_evaluation_count": 0,
}


def source_hashes(root):
    modules = [
        "data/classification_development.py",
        "classification/stage4_analysis.py",
        "classification/stage4_neural.py",
        "classification/stage4.py",
        "features/text_tfidf.py",
        "features/char_chi2.py",
        "features/word_sequence.py",
        "classification/linear.py",
        "classification/c2.py",
        "classification/__main__.py",
        "classification/textcnn.py",
        "classification/textcnn_training.py",
        "classification/stage3_neural.py",
        "classification/stage3_config.py",
        "classification/stage3.py",
        "evaluation/classification.py",
        "segmentation/hmm_segmenter.py",
    ]
    return {"src/nlplab/" + p: file_sha256(root / "src/nlplab" / p) for p in modules}


def frozen_references(root):
    references, configs, selections = {}, {}, {}
    for stage, run_id in PARENTS.items():
        path = root / "outputs/exp3" / run_id
        references[stage] = {
            "run_id": run_id,
            "files": {
                name: {
                    "path": (path / name).relative_to(root).as_posix(),
                    "sha256": file_sha256(path / name),
                }
                for name in ("config.json", "selection_freeze.json")
            },
        }
        configs[stage] = read_json(path / "config.json")
        selections[stage] = read_json(path / "selection_freeze.json")["selected"]
    if selections["stage1"] != {"C0": 0.5, "C1": 2.0}:
        raise DataValidationError("Stage1 frozen parameters changed")
    if (selections["stage2"]["k"], selections["stage2"]["C"]) != (10000, 2.0):
        raise DataValidationError("Stage2 frozen parameters changed")
    selected = selections["stage3"]
    if tuple(selected[k] for k in ("max_len", "learning_rate", "seed", "final_epochs")) != (
        768,
        0.001,
        42,
        12,
    ):
        raise DataValidationError("Stage3 frozen parameters changed")
    if configs["stage2"]["count"] != COUNT_CONFIG or configs["stage2"]["tfidf"] != TFIDF_CONFIG:
        raise DataValidationError("Character representation differs from frozen configuration")
    return references, configs


def preregister(root):
    install_development_io_guard(root)
    base = RunStore(Settings(root), "exp3").directory
    pointer = base / "stage4_preregistration.json"
    if pointer.exists():
        raise DataValidationError("Stage4 already preregistered; do not repeat")
    references, configs = frozen_references(root)
    environment = runtime_information(root)
    data = DevelopmentClassificationData(root)
    train = data.load("train")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = base / "stage4" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", PROTOCOL)
    registration = {
        "run_id": run_id,
        "preregistered_at": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "frozen_config_references": references,
        "source_hashes": source_hashes(root),
        "environment": environment,
        "development_data": data.provenance(),
    }
    write_json(directory / "preregistration.json", registration)
    # Subsets are built only after the protocol has been written.
    ladders = build_ladders(train)
    write_json(directory / "subset_ladders.json", ladders)
    write_json(directory / "frozen_config_references.json", references)
    write_json(directory / "environment.json", environment)
    atomic_json(
        pointer,
        {
            "run_id": run_id,
            "preregistration_sha256": file_sha256(directory / "preregistration.json"),
            "subset_ladders_sha256": file_sha256(directory / "subset_ladders.json"),
        },
    )
    print(
        json.dumps({"preregistered": run_id, "subset_count": len(ladders), "test_accessed": False}),
        flush=True,
    )
    return run_id


def coverage(candidates, vocabulary):
    features = [f for row in candidates for f in row]
    types = set(features)
    return {
        "occurrences": len(features),
        "hit_occurrences": sum(f in vocabulary for f in features),
        "occurrence_coverage": sum(f in vocabulary for f in features) / len(features)
        if features
        else None,
        "types": len(types),
        "hit_types": len(types & set(vocabulary)),
        "type_coverage": len(types & set(vocabulary)) / len(types) if types else None,
    }


def metrics_and_predictions(model, matrix, documents):
    start = perf_counter()
    prediction = model.predict(matrix).tolist()
    elapsed = perf_counter() - start
    return classification_metrics([d.label for d in documents], prediction), prediction, elapsed


def signature(ids, train_metrics, val_metrics, train_prediction, val_prediction, numeric_hash):
    payload = [
        ids_hash(ids),
        train_metrics,
        val_metrics,
        train_prediction,
        val_prediction,
        numeric_hash,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def linear_point(
    model,
    train_matrix,
    val_matrix,
    docs,
    val,
    subset,
    representation,
    output,
    representation_seconds,
    shared_representation,
):
    model.fit(train_matrix, [d.label for d in docs])
    train_metrics, train_pred, train_time = metrics_and_predictions(model, train_matrix, docs)
    val_metrics, val_pred, val_time = metrics_and_predictions(model, val_matrix, val)
    model.save(output / model.model_id)
    numeric_hash = hashlib.sha256(model.weights.tobytes() + model.intercept.tobytes()).hexdigest()
    point = {
        "model_id": model.model_id,
        "subset_seed": subset["subset_seed"],
        "train_size": len(docs),
        "training_ids_sha256": subset["ids_sha256"],
        "representation_fit_ids": [d.id for d in docs],
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "train_predictions": train_pred,
        "validation_predictions": val_pred,
        "representation": representation,
        "representation_reference": shared_representation,
        "runtime": {
            "representation_fit_seconds": representation_seconds,
            "classifier_fit_seconds": model.fit_seconds,
            "fit_seconds": representation_seconds + model.fit_seconds,
            "train_inference_seconds": train_time,
            "validation_inference_seconds": val_time,
        },
        "formal_test_evaluation_count": 0,
        "deterministic_signature": signature(
            [d.id for d in docs], train_metrics, val_metrics, train_pred, val_pred, numeric_hash
        ),
    }
    write_json(output / f"{model.model_id}_result.json", point)
    return point


def fit_sparse_points(directory, docs, val, tokens, val_tokens, subset, config):
    start = perf_counter()
    word = TextTFIDF(TFIDFConfig(**config["representation"]))
    train_matrix = word.fit_transform(tokens)
    word_fit = perf_counter() - start
    start = perf_counter()
    val_matrix = word.transform(val_tokens)
    word_transform = perf_counter() - start
    word.save(directory / "word_representation")
    word_candidates = [word.vectorizer.build_analyzer()(t) for t in val_tokens]
    word_coverage = coverage(word_candidates, set(word.terms))
    word_stats = {
        "vocabulary_size": len(word.terms),
        "dimension": len(word.terms),
        "train_sparsity": sparse_stats(train_matrix)["sparsity"],
        "validation_sparsity": sparse_stats(val_matrix)["sparsity"],
        "train_zero_vectors": sparse_stats(train_matrix)["zero_vector_count"],
        "validation_zero_vectors": sparse_stats(val_matrix)["zero_vector_count"],
        "validation_coverage": word_coverage["occurrence_coverage"],
        "train_matrix": sparse_stats(train_matrix),
        "val_matrix": sparse_stats(val_matrix),
        "coverage_detail": word_coverage,
        "validation_transform_seconds": word_transform,
    }
    points = [
        linear_point(
            LinearTextClassifier(model_id, 0.5 if model_id == "C0" else 2.0),
            train_matrix,
            val_matrix,
            docs,
            val,
            subset,
            word_stats,
            directory,
            word_fit,
            "word_representation",
        )
        for model_id in ("C0", "C1")
    ]
    # Both classifiers above receive the exact same matrix objects.
    start = perf_counter()
    raw = CharacterCounts()
    counts = raw.fit_transform([d.text for d in docs])
    k = effective_k(counts.shape[1])
    char = CharacterChi2TFIDF(k)
    char_train = char.fit_from_counts(counts, [d.label for d in docs], raw)
    char_fit = perf_counter() - start
    start = perf_counter()
    val_counts = raw.transform([d.text for d in val])
    char_val, diagnostic = char.transform_counts(val_counts)
    char_transform = perf_counter() - start
    char.save(directory / "char_representation")
    candidates = [char_bigrams(d.text) for d in val]
    char_coverage = coverage(candidates, set(char.terms))
    raw_coverage = coverage(candidates, set(raw.terms))
    char_stats = {
        "raw_dimension": counts.shape[1],
        "target_k": 10000,
        "effective_k": k,
        "selected_dimension": char_train.shape[1],
        "dimension": char_train.shape[1],
        "train_sparsity": sparse_stats(char_train)["sparsity"],
        "validation_sparsity": sparse_stats(char_val)["sparsity"],
        "raw_train_sparsity": sparse_stats(counts)["sparsity"],
        "raw_validation_sparsity": diagnostic["before_selection"]["sparsity"],
        "train_zero_vectors": sparse_stats(char_train)["zero_vector_count"],
        "validation_zero_vectors": sparse_stats(char_val)["zero_vector_count"],
        "validation_coverage": char_coverage["occurrence_coverage"],
        "validation_raw_coverage": raw_coverage["occurrence_coverage"],
        "coverage_detail": char_coverage,
        "raw_coverage_detail": raw_coverage,
        "validation_transform_seconds": char_transform,
        "raw_train_matrix": sparse_stats(counts),
        "train_matrix": sparse_stats(char_train),
        "val_matrix": sparse_stats(char_val),
    }
    points.append(
        linear_point(
            C2LinearClassifier(parameter=2.0),
            char_train,
            char_val,
            docs,
            val,
            subset,
            char_stats,
            directory,
            char_fit,
            "char_representation",
        )
    )
    return points


def fit_neural_point(root, directory, docs, val, tokens, val_tokens, subset):
    start = perf_counter()
    vocab = WordSequenceVocabulary(768, min_freq=2).fit(tokens)
    train_encoded, val_encoded = vocab.encode(tokens), vocab.encode(val_tokens)
    encoding_seconds = perf_counter() - start
    vocab.save(directory / "sequence_vocabulary")
    save_encoding(directory / "sequence_train", train_encoded, docs)
    save_encoding(directory / "sequence_val", val_encoded, val)
    job = {
        "root": str(root),
        "train": str(directory / "sequence_train"),
        "evaluation_val": str(directory / "sequence_val"),
        "output": str(directory / "C3"),
        "vocabulary_size": len(vocab.terms),
        "max_len": 768,
        "learning_rate": 0.001,
        "seed": 42,
        "epochs": 12,
        "toy": False,
    }
    job_path = directory / "neural_job.json"
    write_json(job_path, job)
    start = perf_counter()
    completed = subprocess.run(
        [
            str(root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"),
            "-m",
            "nlplab.classification.stage4_neural",
            "--job",
            str(job_path),
        ],
        cwd=root,
        env=subprocess_environment(root),
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    process_seconds = perf_counter() - start
    if completed.returncode:
        raise DataValidationError("Stage4 fixed-epoch neural job failed; do not silently rerun")
    result = read_json(directory / "C3/result.json")
    training = result["training"]
    train_pred = [CATEGORIES[i] for i in result["train"]["prediction_indices"]]
    val_pred = [CATEGORIES[i] for i in result["val"]["prediction_indices"]]
    train_metrics = classification_metrics([d.label for d in docs], train_pred)
    val_metrics = classification_metrics([d.label for d in val], val_pred)
    stats = val_encoded["stats"]
    representation = {
        "vocabulary_size": len(vocab.terms),
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
        "train_encoding_stats": train_encoded["stats"],
        "val_encoding_stats": stats,
    }
    point = {
        "model_id": "C3",
        "subset_seed": subset["subset_seed"],
        "train_size": len(docs),
        "training_ids_sha256": subset["ids_sha256"],
        "representation_fit_ids": [d.id for d in docs],
        "representation": representation,
        "representation_reference": "sequence_vocabulary",
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "train_predictions": train_pred,
        "validation_predictions": val_pred,
        "training_history": training["history"],
        "early_stopping_used": False,
        "runtime": {
            "representation_fit_seconds": encoding_seconds,
            "classifier_fit_seconds": training["runtime_seconds"],
            "fit_seconds": encoding_seconds + training["runtime_seconds"],
            "worker_process_seconds": process_seconds,
            "train_inference_seconds": result["train"]["inference_seconds"],
            "validation_inference_seconds": result["val"]["inference_seconds"],
        },
        "formal_test_evaluation_count": 0,
        "validation_prediction_count": 1,
        "deterministic_signature": signature(
            [d.id for d in docs],
            train_metrics,
            val_metrics,
            train_pred,
            val_pred,
            result["state_sha256"],
        ),
    }
    write_json(directory / "C3_result.json", point)
    return point


def run_formal(root):
    install_development_io_guard(root)
    base = RunStore(Settings(root), "exp3").directory
    pointer = read_json(base / "stage4_preregistration.json")
    run_id = pointer["run_id"]
    directory = base / "stage4" / run_id
    registration = read_json(directory / "preregistration.json")
    if (
        file_sha256(directory / "preregistration.json") != pointer["preregistration_sha256"]
        or file_sha256(directory / "subset_ladders.json") != pointer["subset_ladders_sha256"]
    ):
        raise DataValidationError("Preregistration/subsets changed")
    references, configs = frozen_references(root)
    if (
        references != registration["frozen_config_references"]
        or source_hashes(root) != registration["source_hashes"]
        or registration["protocol"] != PROTOCOL
    ):
        raise DataValidationError("Preregistered sources/config changed")
    guard_path = directory / "execution.json"
    with guard_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {"status": "started", "formal_test_evaluation_count": 0, "rerun_allowed": False}, stream
        )
    start = perf_counter()
    data = DevelopmentClassificationData(root)
    train, val = data.load("train"), data.load("val")
    ladders = read_json(directory / "subset_ladders.json")
    if ladders != build_ladders(train):
        raise DataValidationError("Subsets differ from frozen Train")
    processor, tokenizer = processor_for(root, configs["stage1"])
    tokens = {d.id: processor.tokens(d.text) for d in train}
    val_tokens = [processor.tokens(d.text) for d in val]
    write_json(directory / "tokenizer.json", tokenizer)
    write_json(directory / "development_data.json", data.provenance())
    write_jsonl(
        directory / "train_tokens.jsonl",
        [{"document_id": d.id, "tokens": tokens[d.id]} for d in train],
    )
    write_jsonl(
        directory / "validation_tokens.jsonl",
        [{"document_id": d.id, "tokens": t} for d, t in zip(val, val_tokens)],
    )
    lookup = {d.id: d for d in train}
    points, full_signatures = [], {}
    for subset in ladders:
        name = f"ladder{subset['subset_seed']}_n{subset['train_size']}"
        point_dir = directory / "points" / name
        point_dir.mkdir(parents=True)
        docs = [lookup[i] for i in subset["document_ids"]]
        current_tokens = [tokens[d.id] for d in docs]
        print(f"START {name}: from-scratch C0/C1/C2/C3, fixed epochs=12", flush=True)
        new = fit_sparse_points(
            point_dir, docs, val, current_tokens, val_tokens, subset, configs["stage1"]
        )
        new.append(fit_neural_point(root, point_dir, docs, val, current_tokens, val_tokens, subset))
        for point in new:
            if point["train_size"] == 420:
                model = point["model_id"]
                previous = full_signatures.setdefault(model, point["deterministic_signature"])
                if previous != point["deterministic_signature"]:
                    raise DataValidationError(
                        f"STOP: {model} 420 deterministic mismatch; no aggregation"
                    )
        points.extend(new)
        atomic_json(
            directory / "progress.json",
            {
                "completed_subsets": len(points) // 4,
                "completed_model_fits": len(points),
                "formal_test_evaluation_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "completed": name,
                    "validation_f1": {
                        p["model_id"]: p["validation_metrics"]["macro_f1"] for p in new
                    },
                }
            ),
            flush=True,
        )
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
    runtimes = {
        "formal_wall_seconds": perf_counter() - start,
        "model_fit_seconds": {
            m: sum(p["runtime"]["classifier_fit_seconds"] for p in points if p["model_id"] == m)
            for m in MODEL_IDS
        },
        "representation_fit_seconds_shared_word_once": sum(
            p["runtime"]["representation_fit_seconds"] for p in points if p["model_id"] == "C0"
        ),
        "points": [
            {
                "model_id": p["model_id"],
                "subset_seed": p["subset_seed"],
                "train_size": p["train_size"],
                **p["runtime"],
            }
            for p in points
        ],
    }
    write_json(directory / "runtimes.json", runtimes)
    run = {
        "run_id": run_id,
        "run_type": "exp3-stage4-development-learning-curves",
        "models": list(MODEL_IDS),
        "train_sizes": PROTOCOL["train_sizes"],
        "subset_seeds": list(SUBSET_SEEDS),
        "model_fits": len(points),
        "c3_training_runs": 15,
        "formal_test_evaluation_count": 0,
        "allowed_splits": ["train", "val"],
        "historical_validation_reused": True,
        "config_references": references,
        "sanity_420_passed": True,
    }
    write_json(directory / "run.json", run)
    atomic_json(
        guard_path,
        {
            "status": "complete",
            "model_fits": 60,
            "c3_training_runs": 15,
            "formal_test_evaluation_count": 0,
            "test_accessed": False,
            "rerun_allowed": False,
        },
    )
    write_json(directory / "artifact_manifest.json", artifact_manifest(directory))
    atomic_json(
        base / "stage4_latest.json", {"run_id": run_id, "analysis_type": PROTOCOL["analysis_type"]}
    )
    print(
        json.dumps(
            {
                "complete": run_id,
                "model_fits": 60,
                "formal_test_evaluation_count": 0,
                "formal_wall_seconds": runtimes["formal_wall_seconds"],
            }
        ),
        flush=True,
    )
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preregister", action="store_true")
    group.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    (preregister if args.preregister else run_formal)(args.root.resolve())


if __name__ == "__main__":
    main()
