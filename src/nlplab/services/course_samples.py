"""冻结 Smooth 的样例预测及 MSRA test 独立 exact-span NER Run。"""

import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import yaml

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.core.schema import Entity
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.msra_ner import gold_support
from nlplab.data.personal_selection import read_object, subset_manifest, utc_now, verify_subset
from nlplab.ner.hanlp_adapter import HanLPRecognizer
from nlplab.segmentation.bmes import encode
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.sequence_labeling.hmm import HMMTagger


def frozen_smooth(
    settings: Settings, model_path: Path, freeze_path: Path
) -> tuple[HMMTagger, dict]:
    freeze = read_object(freeze_path)
    hashes = {
        name: file_sha256(model_path / name) for name in ("metadata.json", "probabilities.npz")
    }
    if hashes != freeze["models"]["smooth"]:
        raise DataValidationError("HMM 模型不是冻结 Smooth 模型或模型指纹已改变")
    for name, digest in freeze["implementation_sha256"].items():
        path = settings.root / name
        if any(
            part in path.parts for part in ("sequence_labeling", "segmentation")
        ) or path.name in {"msr.py", "hmm_mle.yaml", "hmm_smooth.yaml"}:
            if file_sha256(path) != digest:
                raise DataValidationError(f"冻结 HMM/Viterbi/MSR/config 指纹改变: {name}")
    model = HMMTagger.load(model_path)
    return model, hashes


def course_predictions(rows: list[dict], segmenter, ner) -> tuple[list[dict], dict]:
    """只保存预测 BMES，无 CWS Gold 或 CWS 性能；使用现有 exact-span evaluator。"""
    from nlplab.evaluation.ner import evaluate_ner

    predictions, gold, predicted = [], [], []
    for row in rows:
        text = row["text"]
        entities = [Entity(**value) for value in row["ner_gold"]]
        for entity in entities:
            entity.validate(text)
        result = segmenter.segment(text)
        if "".join(result.tokens) != text or encode(result.tokens) != result.tags:
            raise DataValidationError("HMM 分词/BMES 不满足原文 roundtrip")
        output = ner.recognize(text)
        for entity in output:
            entity.validate(text)
        gold.append(entities)
        predicted.append(output)
        predictions.append(
            {
                **row,
                "segmentation": {"tokens": result.tokens, "bmes": result.tags},
                "ner_prediction": [asdict(entity) for entity in output],
            }
        )
    return predictions, {
        "segmentation": None,
        "cws_note": "无 MSRA CWS Gold；CWS benchmark 保留 SIGHAN 2005 MSR 冻结结果",
        "ner": evaluate_ner([row["text"] for row in rows], gold, predicted),
        "ner_protocol": "exact match (sample, Python code-point start, end, PER/LOC/ORG)",
        "dataset": "MSRA NER test personal subset",
    }


def run_course_samples(
    settings: Settings, source: Path, model_path: Path, freeze_path: Path
) -> dict:
    rows, manifest = verify_subset(source, dataset="MSRA-NER", count=200)
    if manifest.get("source_split") != "test" or any(row["source_split"] != "test" for row in rows):
        raise DataValidationError("NER 正式评价只允许冻结 MSRA test 子集")
    if not all(gold_support(rows).values()):
        raise DataValidationError("冻结子集存在 Gold 支持为 0 的类别")
    model, model_hashes = frozen_smooth(settings, model_path, freeze_path)
    setup_path = settings.root / "artifacts/cache/ner_setup.json"
    setup = read_object(setup_path)
    if setup["model"] != "MSRA_NER_BERT_BASE_ZH" or setup["label_map"] != {
        "NR": "PER",
        "NS": "LOC",
        "NT": "ORG",
    }:
        raise DataValidationError("正式 NER 必须使用指定 HanLP MSRA BERT 及已核验映射")
    ner = HanLPRecognizer.from_settings(settings)
    # selection_manifest 在第一次模型调用前已完整冻结/校验。
    ner._load()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    path = settings.outputs / "exp1/course_samples" / run_id
    path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(subset_manifest(source), path / "selection_manifest.json")
    shutil.copyfile(source, path / "samples.jsonl")
    shutil.copyfile(setup_path, path / "ner_setup.json")
    config = {
        "hmm": asdict(model.config),
        "ner": setup,
        "no_test_tuning": True,
        "cws_gold_available": False,
        "selection_frozen_at": manifest["frozen_at"],
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "ner_chunking": (
            "original sentence splitting; <=200 code points AND actual tokenizer "
            "max_seq_length minus special tokens; no truncation"
        ),
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
    try:
        predictions, metrics = course_predictions(rows, HMMSegmenter(model), ner)
    except Exception as error:
        write_json(
            path / "failure.json",
            {"at": utc_now(), "error": str(error), "status": "incomplete; no benchmark metrics"},
        )
        raise
    examples = {"correct": [], "errors": []}
    errors = []
    for row in predictions:
        expected = {(entity["start"], entity["end"], entity["type"]) for entity in row["ner_gold"]}
        predicted = {
            (entity["start"], entity["end"], entity["type"]) for entity in row["ner_prediction"]
        }
        if expected == predicted:
            if expected and len(examples["correct"]) < 5:
                examples["correct"].append(row)
        else:
            record = {
                **row,
                "evaluation_task": "ner",
                "category": "Other",
                "false_positive": sorted(predicted - expected),
                "false_negative": sorted(expected - predicted),
            }
            errors.append(record)
            if len(examples["errors"]) < 10:
                examples["errors"].append(record)
    write_jsonl(path / "predictions.jsonl", predictions)
    write_jsonl(path / "errors.jsonl", errors)
    write_json(path / "metrics.json", metrics)
    write_json(path / "examples.json", examples)
    runtime = {}
    for package in ("hanlp", "tensorflow", "keras", "numpy", "transformers", "torch"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "not installed"
    ner_path = Path(setup["model_path"])
    fingerprints = {
        "hmm_model": model_hashes,
        "ner_model": {
            file.relative_to(ner_path).as_posix(): file_sha256(file)
            for file in ner_path.rglob("*")
            if file.is_file()
        },
        "selected_data_sha256": file_sha256(source),
        "selection_manifest_sha256": file_sha256(subset_manifest(source)),
        "source_hashes": manifest["source_hashes"],
        "selection_key_fingerprint": manifest["selection_key_fingerprint"],
        "runtime": runtime,
        "ner_adapter_sha256": file_sha256(settings.root / "src/nlplab/ner/hanlp_adapter.py"),
    }
    write_json(path / "fingerprints.json", fingerprints)
    info = {
        "run_id": run_id,
        "created_at": utc_now(),
        "is_demo": False,
        "dataset": "MSRA NER test personal subset",
        "sample_count": 200,
        "model": "frozen HMM-Smooth + HanLP MSRA_NER_BERT_BASE_ZH",
        "model_path": str(model_path.resolve()),
        "run_type": "course_samples",
        "read_only": True,
        "source_split": "test",
        "cws_gold_available": False,
        "test_inference_passes": 1,
        "metrics": metrics,
        "gold_support": gold_support(rows),
    }
    report = [
        "# 实验一课程 200 样本 / NER benchmark",
        "",
        "数据：MSRA NER test personal subset。CWS benchmark：SIGHAN 2005 MSR（冻结）。",
        "MSRA 没有 CWS Gold；分词/BMES 仅预测展示，不计算 CWS P/R/F1。",
        "",
        "| 类别 | P | R | F1 | Gold support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("PER", "LOC", "ORG", "Overall"):
        values = metrics["ner"][label]
        report.append(
            f"| {label} | {values['precision']:.6%} | {values['recall']:.6%} "
            f"| {values['f1']:.6%} | {values['support']} |"
        )
    report.extend(
        [
            "",
            "选择在推理前冻结，无预测质量过滤、无结果调参。",
            f"全部 {len(rows)} 条分词 roundtrip；实体采用 Python code-point exact-span。",
            f"实体不完全匹配的句子：{len(errors)}。",
            f"版本：{manifest['dataset_version']}",
            f"选择键 fingerprint：{manifest['selection_key_fingerprint']}",
        ]
    )
    (path / "report.md").write_text("\n".join(report) + "\n", "utf-8")
    write_json(path / "run.json", info)  # 完成标记最后写，不更新 development latest。
    return {"run_id": run_id, "path": str(path.resolve()), "metrics": metrics}
