"""正式选参、最终训练和只读推理入口。需显式 --formal，不由 Web 触发训练。"""

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

from nlplab.classification.experiment import (
    evaluate_final,
    fit_final,
    matrix_stats,
    select_candidates,
)
from nlplab.classification.linear import LinearTextClassifier
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.classification_input import FrozenClassificationData
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.text_tfidf import ChinesePreprocessor, PreprocessingConfig, TextTFIDF
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.services.experiment_service import RunStore


def processor_for(root, config):
    store = RunStore(Settings(root))
    info = store.read(config["tokenizer_run_id"], "run.json")
    model_path = Path(info["model_path"])
    provenance = {
        "name": "frozen-hmm1-smooth",
        "run_id": config["tokenizer_run_id"],
        "training_executed": False,
        "model_files": {
            str(p.relative_to(root)): file_sha256(p)
            for p in sorted(model_path.glob("*"))
            if p.is_file()
        },
    }
    segmenter = HMMSegmenter(store.model(config["tokenizer_run_id"]))
    processor = ChinesePreprocessor(
        lambda text: segmenter.segment(text).tokens,
        PreprocessingConfig(**config["preprocessing"]),
    )
    return processor, provenance


def tokenize(processor, docs, progress=print):
    result = []
    for i, doc in enumerate(docs, 1):
        result.append(processor.tokens(doc.text))
        if i % 60 == 0 or i == len(docs):
            progress(f"tokenized {i}/{len(docs)}", flush=True)
    return result


def artifact_manifest(directory):
    return {
        "algorithm": "sha256",
        "files": {
            path.relative_to(directory).as_posix(): file_sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }


def verify_artifacts(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
            raise DataValidationError(f"分类 artifact 指纹不符: {name}")
    return manifest


def run_formal(root):
    settings = Settings(root)
    store = RunStore(settings, "exp3")
    store.directory.mkdir(parents=True, exist_ok=True)
    # exclusive 持久标记：已开始的正式阶段不能被 Web/CLI 意外重新训练或重评。
    guard = store.directory / "stage1_formal_execution.json"
    with guard.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "started",
                "rerun_allowed": False,
            },
            stream,
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = store.directory / run_id
    directory.mkdir()
    start = perf_counter()
    config_path = root / "configs/classification.yaml"
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if (
        config["selection_metric"] != "macro_f1"
        or config["final_fit_splits"] != ["train", "val"]
        or config["tie_break"] != "first_in_declared_candidate_order"
    ):
        raise DataValidationError("实验三协议不符")
    write_json(directory / "config.json", config)
    data = FrozenClassificationData(root)
    write_json(directory / "dataset.json", data.provenance())
    processor, tokenizer_info = processor_for(root, config)
    write_json(directory / "tokenizer.json", tokenizer_info)
    train, val = data.load("train"), data.load("val")
    train_tokens, val_tokens = tokenize(processor, train), tokenize(processor, val)
    representation, xtrain, xval, selection = select_candidates(
        train, val, train_tokens, val_tokens, config
    )
    selection_dir = directory / "selection"
    selection_dir.mkdir()
    representation.save(selection_dir / "tfidf")
    save_npz(selection_dir / "train.npz", xtrain)
    save_npz(selection_dir / "val.npz", xval)
    write_json(directory / "validation_selection.json", selection)
    freeze = {
        "selected": selection["selected"],
        "frozen_before_test_open": True,
        "opened_splits": list(data.opened_splits),
        "config_sha256": file_sha256(directory / "config.json"),
        "selection_sha256": file_sha256(directory / "validation_selection.json"),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(directory / "selection_freeze.json", freeze)
    print(f"SELECTION FROZEN: {freeze['selected']}; Test still unopened", flush=True)
    final_rep, xfinal, models, final_repr_seconds = fit_final(
        train + val, train_tokens + val_tokens, config, freeze["selected"]
    )
    final_dir = directory / "final"
    final_dir.mkdir()
    final_rep.save(final_dir / "tfidf")
    save_npz(final_dir / "train_val.npz", xfinal)
    for model_id, model in models.items():
        model.save(final_dir / model_id)
        write_json(
            final_dir / f"{model_id}_global_features.json", model.global_features(final_rep.terms)
        )
    # 先将 final model 落盘，才打开 Test 文本；单次评价开始标记不能被覆盖。
    write_json(
        directory / "test_evaluation_started.json",
        {
            "selection_freeze_sha256": file_sha256(directory / "selection_freeze.json"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "count_per_model": 1,
        },
    )
    test = data.load("test")
    test_tokens = tokenize(processor, test)
    xtest, metrics, predictions, runtime = evaluate_final(test, test_tokens, final_rep, models)
    save_npz(final_dir / "test.npz", xtest)
    write_json(directory / "test_metrics.json", metrics)
    for model_id, rows in predictions.items():
        write_jsonl(directory / f"{model_id}_test_predictions.jsonl", rows)
        write_json(
            directory / f"{model_id}_test_errors.json", [r for r in rows if not r["correct"]]
        )
    all_docs, all_tokens = train + val + test, train_tokens + val_tokens + test_tokens
    write_jsonl(
        directory / "tokens.jsonl",
        [
            {
                "id": doc.id,
                "split": "train" if i < 420 else "val" if i < 510 else "test",
                "tokens": tokens,
            }
            for i, (doc, tokens) in enumerate(zip(all_docs, all_tokens))
        ],
    )
    write_json(
        directory / "representation_stats.json",
        {
            "selection": selection["matrix_stats"],
            "final": {"train_val": matrix_stats(xfinal), "test": matrix_stats(xtest)},
            "shared_C0_C1_matrix": True,
            "final_fit_ids": [d.id for d in train + val],
            "test_transform_only_ids": [d.id for d in test],
        },
    )
    write_json(
        directory / "runtime.json",
        {
            **runtime,
            "final_representation_seconds": final_repr_seconds,
            "total_seconds": perf_counter() - start,
        },
    )
    write_json(
        directory / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "PyYAML", "threadpoolctl")
            },
        },
    )
    source_paths = [
        *sorted((root / "src/nlplab/classification").glob("*.py")),
        root / "src/nlplab/features/text_tfidf.py",
        root / "src/nlplab/data/classification_input.py",
        root / "src/nlplab/evaluation/classification.py",
        config_path,
    ]
    write_json(
        directory / "source_hashes.json",
        {path.relative_to(root).as_posix(): file_sha256(path) for path in source_paths},
    )
    write_json(
        directory / "label_mapping.json",
        {
            "label_order": list(CATEGORIES),
            "label_to_index": {c: i for i, c in enumerate(CATEGORIES)},
        },
    )
    info = {
        "run_id": run_id,
        "experiment": "exp3",
        "run_type": "formal-exp3-stage1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selected": freeze["selected"],
        "protocol": "420 train -> 90 validation; 510 final -> 90 test",
        "formal_exp3_validation_fits_executed": True,
        "formal_exp3_final_fits_executed": True,
        "formal_exp3_test_evaluation_executed": True,
        "exp1_modified": False,
        "exp2_modified": False,
        "lab4_started": False,
        "dataset_fingerprint": data.provenance()["dataset_fingerprint"],
    }
    write_json(directory / "run.json", info)
    write_json(directory / "artifact_manifest.json", artifact_manifest(directory))
    verify_artifacts(directory)
    atomic_json(store.directory / "latest.json", {"run_id": run_id})
    atomic_json(guard, {"run_id": run_id, "status": "complete", "rerun_allowed": False})
    print(json.dumps({"run_id": run_id, "test_metrics": metrics}, ensure_ascii=False), flush=True)
    return info


def predict_saved(root, run_id, model_id, text):
    directory = RunStore(Settings(root), "exp3").resolve(run_id)
    verify_artifacts(directory)
    config = json.loads((directory / "config.json").read_text("utf-8"))
    processor, tokenizer_info = processor_for(root, config)
    if tokenizer_info != json.loads((directory / "tokenizer.json").read_text("utf-8")):
        raise DataValidationError("冻结 tokenizer 已改变")
    start = perf_counter()
    tokens = processor.tokens(text)
    representation = TextTFIDF.load(directory / "final/tfidf")
    model = LinearTextClassifier.load(directory / "final" / model_id)
    row = representation.transform([tokens])
    return {
        "run_id": directory.name,
        "model_id": model_id,
        **model.explain(row, representation.terms),
        "token_count": len(tokens),
        "known_feature_count": row.nnz,
        "runtime_seconds": perf_counter() - start,
        "notice": "零向量：未命中词表，预测由先验/截距决定" if row.nnz == 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--model-id", choices=["C0", "C1"], default="C0")
    args = parser.parse_args()
    if args.formal and not args.predict:
        run_formal(args.root.resolve())
    elif args.predict and not args.formal:
        text = json.loads(sys.stdin.read())["text"]
        print(
            json.dumps(
                predict_saved(args.root.resolve(), args.run_id, args.model_id, text),
                ensure_ascii=False,
            )
        )
    else:
        parser.error("显式选择 --formal 或 --predict")


if __name__ == "__main__":
    main()
