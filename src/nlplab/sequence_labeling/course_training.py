"""课程扩展训练入口：只从已核验的 MSR official training 内部选择固定子集。"""

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import yaml

from nlplab.core.exceptions import DataValidationError
from nlplab.data.dataset import load_samples
from nlplab.data.msr import load_msr_provenance
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.linear_crf import CRFConfig, LinearChainCRF
from nlplab.sequence_labeling.perceptron import (
    AveragedStructuredPerceptron,
    PerceptronConfig,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def course_subset(root: Path, config: dict) -> tuple[list, list, dict]:
    source = (root / config["source"]).resolve()
    if not source.is_relative_to((root / "data" / "splits").resolve()):
        raise DataValidationError("课程模型只能读取本地已导入的 MSR training split")
    if source.name != "train.jsonl":
        raise DataValidationError("课程扩展训练只允许 MSR train.jsonl；禁止 official test")
    records = load_samples(source)
    provenance = load_msr_provenance(source, records)
    if provenance is None or provenance["protocol"] != "standard":
        raise DataValidationError("课程扩展需要经过 manifest 验证的 MSR standard training")
    seed = int(config["seed"])
    train_size, dev_size = int(config["train_size"]), int(config["dev_size"])
    if not 0 < train_size <= 20000 or not 0 < dev_size <= 2000:
        raise DataValidationError("课程扩展 train/dev 大小超出固定预算")
    ranked = sorted(
        records,
        key=lambda sample: (
            sha256(f"{seed}\0{sample.sample_id}".encode()).hexdigest(),
            sample.sample_id,
        ),
    )
    selected, seen_text = [], set()
    for sample in ranked:
        text_hash = sha256(sample.text.encode("utf-8")).hexdigest()
        if text_hash in seen_text:
            continue
        seen_text.add(text_hash)
        selected.append(sample)
        if len(selected) == train_size + dev_size:
            break
    if len(selected) != train_size + dev_size:
        raise DataValidationError("MSR training 中不同文本数量不足课程子集配额")
    train, dev = selected[:train_size], selected[train_size:]
    manifest = {
        "dataset": "SIGHAN Bakeoff 2005 MSR",
        "source_split": "official training only",
        "source_file": str(source),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_manifest_sha256": provenance["manifest_sha256"],
        "seed": seed,
        "selection": "SHA256(seed + NUL + sample_id), sorted ascending; duplicate text excluded",
        "train_ids": [sample.sample_id for sample in train],
        "dev_ids": [sample.sample_id for sample in dev],
        "train_size": len(train),
        "dev_size": len(dev),
        "official_test_read": False,
    }
    return train, dev, manifest


def train_course_model(root: Path, model_name: str) -> dict:
    if model_name not in {"perceptron", "crf"}:
        raise DataValidationError("课程扩展模型只能为 perceptron 或 crf")
    config = yaml.safe_load((root / "configs/course_sequence.yaml").read_text(encoding="utf-8"))
    train, dev, subset = course_subset(root, config)
    subset_path = root / "artifacts/cache/course_sequence_subset.json"
    if subset_path.exists():
        if json.loads(subset_path.read_text(encoding="utf-8")) != subset:
            raise DataValidationError("课程子集已冻结且与当前来源或配置不一致")
    else:
        _write_json(subset_path, subset)
    output = root / "artifacts/models/course_sequence" / f"{model_name}_v1"
    if output.exists():
        raise DataValidationError(f"课程扩展模型已存在，拒绝覆盖: {output}")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise DataValidationError(f"存在上次未完成训练目录，请先检查: {temporary}")
    observations = [list(sample.text) for sample in train]
    labels = [encode(sample.gold_tokens) for sample in train]
    metadata = {
        "corpus_source": subset["source_file"],
        "corpus_sha256": subset["source_sha256"],
        "subset_manifest_sha256": sha256(subset_path.read_bytes()).hexdigest(),
        "seed": subset["seed"],
        "train_size": len(train),
        "dev_size": len(dev),
        "feature_template": (
            "bias, c[-1], c[0], c[+1], adjacent characters, character type, BOS/EOS"
        ),
    }
    started = perf_counter()
    if model_name == "perceptron":
        model = AveragedStructuredPerceptron(PerceptronConfig(**config["perceptron"]))
        model.training_metadata = metadata
        model.fit(observations, labels)
        model.save(temporary)
        model_file = "weights.json"
    else:
        model = LinearChainCRF(CRFConfig(**config["crf"]))
        model.training_metadata = metadata
        model.fit(observations, labels, temporary)
        model_file = "model.crfsuite"
    training_seconds = round(perf_counter() - started, 2)
    correct = total = failed = 0
    for sample in dev:
        gold = encode(sample.gold_tokens)
        try:
            predicted = model.predict(list(sample.text))
        except DataValidationError:
            failed += 1
            continue
        correct += sum(left == right for left, right in zip(gold, predicted))
        total += len(gold)
    temporary.rename(output)
    run = {
        "model": model_name,
        "kind": "course-extension",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(output),
        "model_sha256": sha256((output / model_file).read_bytes()).hexdigest(),
        "config": config[model_name],
        "training_metadata": metadata,
        "training_seconds": training_seconds,
        "dev_tag_accuracy": correct / total if total else None,
        "dev_decoding_failures": failed,
        "official_test_inference_executed": False,
        "official_scorer_executed": False,
    }
    _write_json(output / "course_run.json", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="只训练新的课程序列标注扩展模型")
    parser.add_argument("model", choices=["perceptron", "crf"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            train_course_model(args.root.resolve(), args.model), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
