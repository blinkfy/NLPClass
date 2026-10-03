"""C2 正式 Run 与只读推理；Stage1 latest、模型和所有结果保持不变。"""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml
from scipy.sparse import save_npz

from nlplab.classification.__main__ import artifact_manifest, verify_artifacts
from nlplab.classification.c2 import C2LinearClassifier
from nlplab.classification.stage2_experiment import (
    evaluate_stage2_test,
    fit_stage2_final,
    paired_outcomes,
    select_stage2,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.classification_input import FrozenClassificationData
from nlplab.data.corpus import iter_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.char_chi2 import (
    ANALYZER_RULE,
    COUNT_CONFIG,
    TFIDF_CONFIG,
    CharacterChi2TFIDF,
    char_bigrams,
    feature_coverage,
)
from nlplab.services.experiment_service import RunStore

STAGE1_RUN_ID = "20261001_123337_9c1e7341"


def test_coverage(test, representation, stage1_directory):
    raw_vocab, selected_vocab = set(representation.raw.terms), set(representation.terms)
    word_vocab = set(
        json.loads((stage1_directory / "final/tfidf/vocabulary.json").read_text("utf-8"))[
            "vocabulary"
        ]
    )
    test_ids = {d.id for d in test}
    tokens = {
        r["id"]: r["tokens"]
        for r in iter_jsonl(stage1_directory / "tokens.jsonl", lambda r: r)
        if r["id"] in test_ids
    }
    result = []
    char_all, word_all = [], []
    for doc in test:
        chars = char_bigrams(doc.text)
        words = tokens[doc.id]
        word_features = words + [a + " " + b for a, b in zip(words, words[1:])]
        char_all.extend(chars)
        word_all.extend(word_features)
        result.append(
            {
                "document_id": doc.id,
                "c1_final_vocabulary": feature_coverage(word_features, word_vocab),
                "c2_raw_count_vocabulary": feature_coverage(chars, raw_vocab),
                "c2_selected_vocabulary": feature_coverage(chars, selected_vocab),
            }
        )
    return {
        "definition": (
            "candidate feature occurrences/types present in final fitted vocabulary / "
            "all extracted candidate occurrences/types; empty denominator -> null"
        ),
        "comparability_note": (
            "word 与 char 的候选单位不同；比例不能解释为 OOV Recall 或直接性能优势"
        ),
        "c1_tokens_source": "frozen Stage1 tokens.jsonl; HMM not called",
        "aggregate": {
            "c1_final_vocabulary": feature_coverage(word_all, word_vocab),
            "c2_raw_count_vocabulary": feature_coverage(char_all, raw_vocab),
            "c2_selected_vocabulary": feature_coverage(char_all, selected_vocab),
        },
        "per_document": result,
    }


def run_formal(root):
    store = RunStore(Settings(root), "exp3")
    config_path = root / "configs/classification_stage2.yaml"
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if (
        config["stage1_run_id"] != STAGE1_RUN_ID
        or config["analyzer"] != ANALYZER_RULE
        or config["count"] != COUNT_CONFIG
        or config["tfidf"] != TFIDF_CONFIG
        or config["candidates"] != {"k": [2000, 5000, 10000, "all"], "C": [0.1, 0.5, 1.0, 2.0]}
        or config["tie_break"]
        != ["macro_f1_desc", "accuracy_desc", "selected_feature_count_asc", "C_asc"]
        or config["svm"]
        != {
            "multi_class": "ovr",
            "penalty": "l2",
            "loss": "squared_hinge",
            "dual": True,
            "random_state": 42,
            "tol": 1e-4,
            "max_iter": 10000,
        }
    ):
        raise DataValidationError("Stage2 预注册配置不符")
    stage1_dir = store.resolve(STAGE1_RUN_ID)
    verify_artifacts(stage1_dir)
    guard = store.directory / "stage2_formal_execution.json"
    with guard.open("x", encoding="utf-8") as stream:
        json.dump({"status": "started", "rerun_allowed": False}, stream)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = store.directory / run_id
    directory.mkdir()
    start = perf_counter()
    data = FrozenClassificationData(root)
    write_json(directory / "config.json", config)
    write_json(directory / "dataset.json", data.provenance())
    stage1_info = json.loads((stage1_dir / "run.json").read_text("utf-8"))
    if stage1_info["dataset_fingerprint"] != data.provenance()["dataset_fingerprint"]:
        raise DataValidationError("Stage1/Stage2 数据源不一致")
    write_json(
        directory / "stage1_binding.json",
        {
            "run_id": STAGE1_RUN_ID,
            "artifact_manifest_sha256": file_sha256(stage1_dir / "artifact_manifest.json"),
            "files": json.loads((stage1_dir / "artifact_manifest.json").read_text("utf-8"))[
                "files"
            ],
            "read_only": True,
            "retraining_executed": False,
        },
    )
    train, val = data.load("train"), data.load("val")
    rep, rawtrain, rawval, xtrain, xval, selection = select_stage2(train, val, config)
    selected_dir = directory / "selection"
    selected_dir.mkdir()
    rep.save(selected_dir / "representation")
    for name, matrix in (
        ("train_counts", rawtrain),
        ("val_counts", rawval),
        ("train", xtrain),
        ("val", xval),
    ):
        save_npz(selected_dir / f"{name}.npz", matrix)
    write_json(directory / "validation_selection.json", selection)
    freeze = {
        "selected": selection["selected"],
        "opened_splits": list(data.opened_splits),
        "frozen_before_test_open": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(directory / "config.json"),
        "selection_sha256": file_sha256(directory / "validation_selection.json"),
    }
    write_json(directory / "selection_freeze.json", freeze)
    print(f"STAGE2 SELECTION FROZEN: {freeze['selected']}; Test unopened", flush=True)
    final_rep, final_vectors, model, repr_seconds = fit_stage2_final(
        train + val, freeze["selected"]
    )
    final_dir = directory / "final"
    final_dir.mkdir()
    final_rep.save(final_dir / "representation")
    model.save(final_dir / "C2")
    save_npz(final_dir / "train_val.npz", final_vectors)
    write_json(
        final_dir / "C2_global_features.json",
        model.global_features(final_rep.terms, final_rep.selected_scores),
    )
    write_json(directory / "top_chi2_features.json", final_rep.top_chi2())
    write_json(
        directory / "test_evaluation_started.json",
        {
            "count": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "selection_freeze_sha256": file_sha256(directory / "selection_freeze.json"),
        },
    )
    test = data.load("test")
    xtest, metrics, predictions, test_stats = evaluate_stage2_test(test, final_rep, model)
    save_npz(final_dir / "test.npz", xtest)
    write_json(directory / "test_metrics.json", {"C2": metrics})
    write_jsonl(directory / "C2_test_predictions.jsonl", predictions)
    write_json(directory / "C2_test_errors.json", [r for r in predictions if not r["correct"]])
    # 至此 Test 评价已完成；随后才读取 frozen C1 predictions 做配对和 coverage。
    c1 = list(iter_jsonl(stage1_dir / "C1_test_predictions.jsonl", lambda r: r))
    paired = paired_outcomes(c1, predictions)
    write_json(directory / "paired_c1_c2.json", paired)
    write_json(directory / "coverage.json", test_coverage(test, final_rep, stage1_dir))
    baseline_metrics = json.loads((stage1_dir / "test_metrics.json").read_text("utf-8"))
    write_json(directory / "comparison_metrics.json", {**baseline_metrics, "C2": metrics})
    write_json(
        directory / "representation_stats.json",
        {
            "selection": selection["train_stats"],
            "validation": selection["validation_stats"],
            "final": final_rep.fit_stats,
            "test": test_stats,
            "final_fit_ids": [d.id for d in train + val],
            "test_transform_only_ids": [d.id for d in test],
        },
    )
    write_json(
        directory / "runtime.json",
        {
            "total_seconds": perf_counter() - start,
            "final_representation_seconds": repr_seconds,
            "model_fit_seconds": model.fit_seconds,
            "test_inference_seconds": test_stats["inference_seconds"],
            "test_evaluation_count": 1,
        },
    )
    write_json(
        directory / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "PyYAML")
            },
        },
    )
    write_json(
        directory / "label_mapping.json",
        {
            "label_order": list(CATEGORIES),
            "label_to_index": {c: i for i, c in enumerate(CATEGORIES)},
        },
    )
    source_paths = [
        Path(__file__),
        root / "src/nlplab/classification/stage2_experiment.py",
        root / "src/nlplab/classification/c2.py",
        root / "src/nlplab/classification/linear.py",
        root / "src/nlplab/features/char_chi2.py",
        root / "src/nlplab/evaluation/classification.py",
        root / "src/nlplab/data/classification_input.py",
        config_path,
    ]
    write_json(
        directory / "source_hashes.json",
        {p.relative_to(root).as_posix(): file_sha256(p) for p in source_paths},
    )
    info = {
        "run_id": run_id,
        "run_type": "formal-exp3-stage2",
        "experiment": "exp3",
        "stage1_run_id": STAGE1_RUN_ID,
        "selected": freeze["selected"],
        "dataset_fingerprint": data.provenance()["dataset_fingerprint"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "formal_exp3_stage2_validation_fits_executed": True,
        "formal_exp3_stage2_final_fit_executed": True,
        "formal_exp3_stage2_test_evaluation_executed": True,
        "exp3_stage1_modified": False,
        "exp1_modified": False,
        "exp2_modified": False,
        "c3_started": False,
        "lab4_started": False,
    }
    write_json(directory / "run.json", info)
    write_json(directory / "artifact_manifest.json", artifact_manifest(directory))
    verify_artifacts(directory)
    verify_artifacts(stage1_dir)
    # 独立 Stage2 指针；Stage1 latest.json 不改变。
    atomic_json(store.directory / "stage2_latest.json", {"run_id": run_id})
    atomic_json(guard, {"status": "complete", "run_id": run_id, "rerun_allowed": False})
    print(
        json.dumps(
            {"run_id": run_id, "metrics": metrics, "paired": paired["counts"]}, ensure_ascii=False
        ),
        flush=True,
    )
    return info


def predict_saved(root, run_id, text):
    directory = RunStore(Settings(root), "exp3").resolve(run_id)
    verify_artifacts(directory)
    representation = CharacterChi2TFIDF.load(directory / "final/representation")
    model = C2LinearClassifier.load(directory / "final/C2")
    row = representation.transform([text])
    features = char_bigrams(text)
    return {
        "run_id": directory.name,
        "model_id": "C2",
        **model.explain(row, representation.terms),
        "token_count": len(features),
        "known_feature_count": row.nnz,
        "coverage": feature_coverage(features, set(representation.terms)),
        "notice": "零向量：未命中选择后的词表，预测由截距决定" if row.nnz == 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.formal and not args.predict:
        run_formal(args.root.resolve())
    elif args.predict and not args.formal and args.run_id:
        print(
            json.dumps(
                predict_saved(
                    args.root.resolve(), args.run_id, json.loads(sys.stdin.read())["text"]
                ),
                ensure_ascii=False,
            )
        )
    else:
        parser.error("显式选择 --formal 或 --predict --run-id")


if __name__ == "__main__":
    main()
