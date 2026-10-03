"""Stage3 两环境协调器：冻结分词/统一评价复用现有实现，神经子进程独立。"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from nlplab.classification.__main__ import artifact_manifest, processor_for, verify_artifacts
from nlplab.classification.stage3_config import ARCHITECTURE, PROTOCOL, TRAINING
from nlplab.classification.stage3_experiment import (
    annotate_history,
    median_final_epochs,
    paired_stage3,
    select_learning_rate,
    summarize_seeds,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.classification_input import FrozenClassificationData
from nlplab.data.corpus import iter_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.word_sequence import (
    WordSequenceVocabulary,
    length_statistics,
    max_len_from_train,
    read_json,
)
from nlplab.services.experiment_service import RunStore


def source_hashes(root):
    relative = [
        "features/word_sequence.py",
        "classification/textcnn.py",
        "classification/textcnn_training.py",
        "classification/stage3_neural.py",
        "classification/stage3_config.py",
        "classification/stage3_experiment.py",
        "classification/stage3.py",
        "features/text_tfidf.py",
        "classification/__main__.py",
        "evaluation/classification.py",
        "data/classification_input.py",
        "segmentation/hmm_segmenter.py",
    ]
    return {"src/nlplab/" + p: file_sha256(root / "src/nlplab" / p) for p in relative}


def neural_runtime(root):
    executable = root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
    if not executable.is_file():
        raise DataValidationError("没有已存在的 PyTorch runtime；正式训练未开始")
    return executable


def subprocess_environment(root):
    return {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONIOENCODING": "utf-8"}


def runtime_information(root):
    probe = (
        "import torch,sys,json,numpy; print(json.dumps({'python':sys.version,"
        "'executable':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__,"
        "'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,"
        "'device':'cpu','threads':2,'deterministic_algorithms':True}))"
    )
    result = subprocess.run(
        [str(neural_runtime(root)), "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        env=subprocess_environment(root),
        timeout=60,
    )
    return {
        "neural": json.loads(result.stdout),
        "controller": {
            "python": sys.version,
            "executable": sys.executable,
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "PyYAML")
            },
        },
        "platform": platform.platform(),
        "device": "cpu",
        "dependency_changes": False,
    }


def save_encoding(path, encoded, docs=None):
    arrays = {"ids": encoded["ids"], "lengths": encoded["lengths"]}
    if docs is not None:
        arrays["labels"] = np.asarray([CATEGORIES.index(d.label) for d in docs], dtype=np.int64)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write_json(
        path.with_suffix(".json"),
        {
            "tokens": encoded["tokens"],
            "original_positions": encoded["original_positions"],
            "stats": encoded["stats"],
            "document_ids": [d.id for d in docs] if docs is not None else [],
        },
    )


def run_neural(root, job_path, job, quiet=False):
    write_json(job_path, job)
    args = [
        str(neural_runtime(root)),
        "-m",
        "nlplab.classification.stage3_neural",
        "--job",
        str(job_path),
    ]
    result = subprocess.run(
        args,
        cwd=root,
        env=subprocess_environment(root),
        capture_output=quiet,
        text=True,
        encoding="utf-8",
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise DataValidationError(f"Stage3 neural job failed: {result.returncode} {result.stderr}")


def validate_parent_assets(store):
    bindings = {}
    for name, run_id in (
        ("stage1", PROTOCOL["stage1_run_id"]),
        ("stage2", PROTOCOL["stage2_run_id"]),
    ):
        directory = store.resolve(run_id)
        verify_artifacts(directory)
        bindings[name] = {
            "run_id": run_id,
            "manifest_sha256": file_sha256(directory / "artifact_manifest.json"),
            "retraining_executed": False,
        }
    return bindings


def preregister(root):
    store = RunStore(Settings(root), "exp3")
    pointer = store.directory / "stage3_preregistration.json"
    if pointer.exists() or (store.directory / "stage3_formal_execution.json").exists():
        raise DataValidationError("Stage3 已预注册/执行，不允许重复")
    parents = validate_parent_assets(store)
    env = runtime_information(root)
    data = FrozenClassificationData(root)
    base = read_json(store.resolve(PROTOCOL["stage1_run_id"]) / "config.json")
    processor, tokenizer = processor_for(root, base)
    train = data.load("train")
    tokens = [processor.tokens(d.text) for d in train]
    stats = length_statistics(tokens)
    max_len = max_len_from_train(stats)
    vocabulary = WordSequenceVocabulary(max_len).fit(tokens)
    encoded = vocabulary.encode(tokens)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = store.directory / run_id
    directory.mkdir()
    (directory / "selection").mkdir()
    vocabulary.save(directory / "selection/vocabulary")
    save_encoding(directory / "selection/train", encoded, train)
    write_jsonl(
        directory / "selection/train_tokens.jsonl",
        [{"id": d.id, "tokens": t} for d, t in zip(train, tokens)],
    )
    write_json(directory / "train_length_statistics.json", stats)
    write_json(directory / "environment.json", env)
    write_json(directory / "dataset.json", data.provenance())
    write_json(directory / "parent_bindings.json", parents)
    write_json(directory / "tokenizer.json", tokenizer)
    write_json(directory / "source_hashes.json", source_hashes(root))
    config = {
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "training": TRAINING,
        "tokenizer_run_id": base["tokenizer_run_id"],
        "preprocessing": base["preprocessing"],
        "max_len": max_len,
        "train_vocabulary_size": len(vocabulary.terms),
    }
    write_json(directory / "config.json", config)
    freeze = {
        "run_id": run_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "opened_splits": data.opened_splits,
        "formal_training_started": False,
        "max_len": max_len,
        "max_len_source": "Train420 only",
        "length_stats_sha256": file_sha256(directory / "train_length_statistics.json"),
        "config_sha256": file_sha256(directory / "config.json"),
        "sources": source_hashes(root),
        "tokenizer": tokenizer,
        "environment": env,
        "train_fit_ids": [d.id for d in train],
        "historical_test_used": True,
        "candidate_design_from_historical_errors": False,
        "files_frozen_before_training": artifact_manifest(directory)["files"],
    }
    write_json(directory / "preregistration.json", freeze)
    with pointer.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "run_id": run_id,
                "preregistration_sha256": file_sha256(directory / "preregistration.json"),
            },
            stream,
        )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "max_len": max_len,
                "train_vocabulary_size": len(vocabulary.terms),
                "train_stats": {k: v for k, v in stats.items() if k != "lengths"},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return freeze


def verify_preregistration(root, directory, pointer):
    freeze = read_json(directory / "preregistration.json")
    if file_sha256(directory / "preregistration.json") != pointer["preregistration_sha256"]:
        raise DataValidationError("preregistration 被修改")
    if source_hashes(root) != freeze["sources"]:
        raise DataValidationError("正式训练代码与 preregistration hash 不同")
    for name, digest in freeze["files_frozen_before_training"].items():
        if file_sha256(directory / name) != digest:
            raise DataValidationError(f"预注册资产修改: {name}")
    return freeze


def run_formal(root):
    store = RunStore(Settings(root), "exp3")
    pointer = read_json(store.directory / "stage3_preregistration.json")
    directory = store.directory / pointer["run_id"]
    freeze = verify_preregistration(root, directory, pointer)
    parents = validate_parent_assets(store)
    guard = store.directory / "stage3_formal_execution.json"
    with guard.open("x", encoding="utf-8") as stream:
        json.dump(
            {"run_id": pointer["run_id"], "status": "started", "rerun_allowed": False}, stream
        )
    start = perf_counter()
    data = FrozenClassificationData(root)
    config = read_json(directory / "config.json")
    base = read_json(store.resolve(PROTOCOL["stage1_run_id"]) / "config.json")
    processor, tokenizer = processor_for(root, base)
    if tokenizer != freeze["tokenizer"]:
        raise DataValidationError("冻结 tokenizer hash 不符")
    train, val = data.load("train"), data.load("val")
    vocab = WordSequenceVocabulary.load(directory / "selection/vocabulary")
    train_tokens = [
        r["tokens"] for r in iter_jsonl(directory / "selection/train_tokens.jsonl", lambda r: r)
    ]
    val_tokens = [processor.tokens(d.text) for d in val]
    val_encoded = vocab.encode(val_tokens)
    save_encoding(directory / "selection/val", val_encoded, val)
    results = []
    for lr in TRAINING["learning_rates"]:
        for seed in TRAINING["seed"]:
            candidate = directory / f"validation_lr{lr}_seed{seed}"
            job = {
                "mode": "validation",
                "train": str(directory / "selection/train"),
                "validation": str(directory / "selection/val"),
                "output": str(candidate),
                "vocabulary_size": len(vocab.terms),
                "learning_rate": lr,
                "seed": seed,
            }
            run_neural(root, directory / f"job_validation_lr{lr}_seed{seed}.json", job)
            raw = read_json(candidate / "training_result.json")
            history, metrics, predictions = annotate_history(raw, val)
            write_json(candidate / "epoch_history.json", history)
            write_json(candidate / "best_validation_metrics.json", metrics)
            write_jsonl(candidate / "best_validation_predictions.jsonl", predictions)
            results.append(
                {
                    "learning_rate": lr,
                    "seed": seed,
                    "best_epoch": raw["best_epoch"],
                    "best_validation_macro_f1": metrics["macro_f1"],
                    "metrics": metrics,
                    "epochs_executed": raw["epochs_executed"],
                    "runtime_seconds": raw["runtime_seconds"],
                    "history": history,
                    "candidate_directory": candidate.name,
                }
            )
    summary = summarize_seeds(results)
    selected_lr = select_learning_rate(summary)
    best_epochs = [r["best_epoch"] for r in results if r["learning_rate"] == selected_lr]
    final_epochs = median_final_epochs(best_epochs)
    write_json(
        directory / "validation_selection.json",
        {
            "phase": "validation_only",
            "training_runs": results,
            "seed_summary": summary,
            "selected_learning_rate": selected_lr,
            "fit_ids": [d.id for d in train],
            "validation_ids": [d.id for d in val],
            "test_opened": False,
        },
    )
    selected = {
        "learning_rate": selected_lr,
        "seed": 42,
        "max_len": config["max_len"],
        "best_epochs_at_selected_lr": best_epochs,
        "final_epochs": final_epochs,
        "derivation": "median of three best epochs; not best seed",
    }
    write_json(
        directory / "selection_freeze.json",
        {
            "selected": selected,
            "opened_splits": data.opened_splits,
            "frozen_before_test_open": True,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": file_sha256(directory / "validation_selection.json"),
        },
    )
    print(f"STAGE3 SELECTION FROZEN {selected}; Test unopened", flush=True)
    final_vocab = WordSequenceVocabulary(config["max_len"]).fit(train_tokens + val_tokens)
    final_vocab.save(directory / "final_vocabulary")
    final_encoded = final_vocab.encode(train_tokens + val_tokens)
    save_encoding(directory / "final_train_val", final_encoded, train + val)
    run_neural(
        root,
        directory / "job_final.json",
        {
            "mode": "final",
            "train": str(directory / "final_train_val"),
            "output": str(directory / "final"),
            "vocabulary_size": len(final_vocab.terms),
            "learning_rate": selected_lr,
            "seed": 42,
            "final_epochs": final_epochs,
        },
    )
    final_info = read_json(directory / "final/training_result.json")
    write_json(
        directory / "final_fit_metadata.json",
        {
            "fit_ids": [d.id for d in train + val],
            "opened_splits": data.opened_splits,
            "max_len_from_stage_a": config["max_len"],
            "epochs_executed": final_info["epochs_executed"],
            "parameter_count": final_info["parameter_count"],
            "early_stopping_used": False,
            "model_state_sha256": file_sha256(directory / "final/model/state.npz"),
            "vocabulary_sha256": file_sha256(directory / "final_vocabulary/vocabulary.json"),
        },
    )
    write_json(
        directory / "test_evaluation_started.json",
        {
            "count": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "selection_freeze_sha256": file_sha256(directory / "selection_freeze.json"),
            "final_metadata_sha256": file_sha256(directory / "final_fit_metadata.json"),
        },
    )
    test = data.load("test")
    test_tokens = [processor.tokens(d.text) for d in test]
    test_encoded = final_vocab.encode(test_tokens)
    save_encoding(directory / "test_encoding", test_encoded, test)
    test_start = perf_counter()
    run_neural(
        root,
        directory / "job_test_inference.json",
        {
            "mode": "infer",
            "model": str(directory / "final/model"),
            "vocabulary": str(directory / "final_vocabulary"),
            "encoding": str(directory / "test_encoding"),
            "output": str(directory / "test_inference.json"),
            "runtime_output": str(directory / "test_inference_runtime.json"),
        },
    )
    test_process_seconds = perf_counter() - test_start
    predictions = [
        {
            "document_id": d.id,
            "gold": d.label,
            "correct": d.label == r["prediction"],
            "text_snippet": " ".join(d.text.split())[:240],
            "text": d.text,
            "sequence_diagnostic": test_encoded["stats"]["per_document"][i],
            **r,
        }
        for i, (d, r) in enumerate(zip(test, read_json(directory / "test_inference.json")))
    ]
    metrics = classification_metrics(
        [d.label for d in test], [r["prediction"] for r in predictions]
    )
    write_json(directory / "test_metrics.json", {"C3": metrics})
    write_jsonl(directory / "C3_test_predictions.jsonl", predictions)
    write_json(directory / "C3_test_errors.json", [r for r in predictions if not r["correct"]])
    # C1/C2 Test predictions 至此才用于配对，不影响配置。
    for old_id, parent_key in (("C1", "stage1"), ("C2", "stage2")):
        old_dir = store.resolve(parents[parent_key]["run_id"])
        old = list(iter_jsonl(old_dir / f"{old_id}_test_predictions.jsonl", lambda r: r))
        write_json(
            directory / f"paired_{old_id.lower()}_c3.json", paired_stage3(old, predictions, old_id)
        )
    frozen_comparison = read_json(
        store.resolve(PROTOCOL["stage2_run_id"]) / "comparison_metrics.json"
    )
    write_json(directory / "comparison_metrics.json", {**frozen_comparison, "C3": metrics})
    write_json(
        directory / "representation_stats.json",
        {
            "train_lengths": read_json(directory / "train_length_statistics.json"),
            "selection_train": vocab.encode(train_tokens)["stats"],
            "selection_val": val_encoded["stats"],
            "final_train_val": final_encoded["stats"],
            "final_test": test_encoded["stats"],
            "max_len": config["max_len"],
            "parameter_count": final_info["parameter_count"],
            "fit_ids": [d.id for d in train + val],
            "transform_only_test_ids": [d.id for d in test],
        },
    )
    write_json(
        directory / "runtime.json",
        {
            "total_formal_seconds": perf_counter() - start,
            "validation_training_seconds": sum(r["runtime_seconds"] for r in results),
            "final_training_seconds": final_info["runtime_seconds"],
            "test_inference_process_seconds": test_process_seconds,
            "test_evaluation_count": 1,
        },
    )
    write_json(directory / "label_mapping.json", {"label_order": list(CATEGORIES)})
    info = {
        "run_id": directory.name,
        "run_type": "formal-exp3-stage3",
        "experiment": "exp3",
        "stage1_run_id": PROTOCOL["stage1_run_id"],
        "stage2_run_id": PROTOCOL["stage2_run_id"],
        "dataset_fingerprint": data.provenance()["dataset_fingerprint"],
        "selected": selected,
        "formal_exp3_stage3_validation_training_runs": 6,
        "formal_exp3_stage3_final_fit_executed": True,
        "formal_exp3_stage3_test_evaluation_count": 1,
        "stage1_modified": False,
        "stage2_modified": False,
        "exp1_modified": False,
        "exp2_modified": False,
        "lab4_started": False,
    }
    write_json(directory / "run.json", info)
    write_json(directory / "artifact_manifest.json", artifact_manifest(directory))
    verify_artifacts(directory)
    validate_parent_assets(store)
    atomic_json(store.directory / "stage3_latest.json", {"run_id": directory.name})
    atomic_json(
        guard,
        {
            "run_id": directory.name,
            "status": "complete",
            "rerun_allowed": False,
            "validation_training_runs": 6,
            "final_fit_count": 1,
            "test_evaluation_count": 1,
        },
    )
    print(
        json.dumps(
            {"run_id": directory.name, "selected": selected, "test": metrics}, ensure_ascii=False
        ),
        flush=True,
    )
    return info


def predict_saved(root, run_id, text):
    store = RunStore(Settings(root), "exp3")
    directory = store.resolve(run_id)
    verify_artifacts(directory)
    config = read_json(directory / "config.json")
    base = {
        "tokenizer_run_id": config["tokenizer_run_id"],
        "preprocessing": config["preprocessing"],
    }
    processor, tokenizer = processor_for(root, base)
    if tokenizer != read_json(directory / "tokenizer.json"):
        raise DataValidationError("C3 frozen tokenizer 已修改")
    vocab = WordSequenceVocabulary.load(directory / "final_vocabulary")
    tokens = processor.tokens(text)
    encoded = vocab.encode([tokens])
    with tempfile.TemporaryDirectory(prefix="nlplab-c3-") as scratch:
        temp = Path(scratch)
        save_encoding(temp / "input", encoded)
        run_neural(
            root,
            temp / "job.json",
            {
                "mode": "infer",
                "model": str(directory / "final/model"),
                "vocabulary": str(directory / "final_vocabulary"),
                "encoding": str(temp / "input"),
                "output": str(temp / "result.json"),
            },
            quiet=True,
        )
        result = read_json(temp / "result.json")[0]
    return {
        **result,
        "run_id": run_id,
        "token_count": len(tokens),
        "sequence_diagnostic": encoded["stats"]["per_document"][0],
        "notice": "Softmax 输出未校准；卷积片段不是 attention 或因果解释",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preregister", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--predict", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.preregister:
        preregister(args.root.resolve())
    elif args.formal:
        run_formal(args.root.resolve())
    elif args.run_id:
        result = predict_saved(
            args.root.resolve(), args.run_id, json.loads(sys.stdin.read())["text"]
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        parser.error("predict requires --run-id")


if __name__ == "__main__":
    main()
