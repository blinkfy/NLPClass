"""MSR 官方 standard 与去污染 clean protocol 的导入及可追溯清单。"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.core.schema import SequenceSample
from nlplab.data.corpus import duplicate_key, write_jsonl
from nlplab.data.dataset import read_msr, split_training

MSR_DATASET = "SIGHAN Bakeoff 2005 MSR"


def sample_fingerprint(samples: list[SequenceSample]) -> str:
    """包含顺序、重复次数和 Gold 的记录指纹，不能仅用文本集合代替。"""
    digest = sha256()
    for sample in samples:
        digest.update(
            json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def import_msr(
    source: Path,
    destination: Path,
    *,
    protocol: str = "standard",
    seed: int = 42,
    dev_ratio: float = 0.1,
) -> dict[str, Any]:
    """standard 保留全部官方 train/test；clean 才清理并从训练集划分 dev。"""
    if protocol not in {"standard", "clean"}:
        raise DataValidationError("MSR protocol 必须为 standard 或 clean")
    if destination.exists():
        raise DataValidationError(f"目标划分目录已存在，拒绝覆盖: {destination}")
    train_file = source / "training" / "msr_training.utf8"
    test_file = source / "gold" / "msr_test_gold.utf8"
    training = read_msr(train_file, "MSR_TRAIN")
    test = read_msr(test_file, "MSR_TEST")
    train_keys = [duplicate_key(sample.text) for sample in training]
    test_keys = [duplicate_key(sample.text) for sample in test]
    test_texts = set(test_keys)
    overlap_count = sum(key in test_texts for key in train_keys)
    # 检测只产生清单信息，standard 不调用去重/分割函数，也不修改顺序或 Gold。
    if protocol == "standard":
        train, dev, duplicate_count, removed_overlap = training, [], 0, 0
    else:
        clean_training = [
            sample for sample, key in zip(training, train_keys) if key not in test_texts
        ]
        train, dev, duplicate_count = split_training(clean_training, dev_ratio, seed)
        removed_overlap = overlap_count
    destination.mkdir(parents=True)
    splits = {}
    for name, samples in (("train", train), ("dev", dev), ("test", test)):
        path = destination / f"{name}.jsonl"
        write_jsonl(path, [sample.to_dict() for sample in samples])
        splits[name] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "sample_fingerprint": sample_fingerprint(samples),
            "count": len(samples),
        }
    manifest = {
        "dataset": MSR_DATASET,
        "protocol": protocol,
        "seed": seed,
        "dev_ratio": dev_ratio if protocol == "clean" else 0.0,
        "original_train_count": len(training),
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
        "potential_training_duplicates": len(train_keys) - len(set(train_keys)),
        "potential_test_duplicates": len(test_keys) - len(test_texts),
        "potential_training_test_overlap": overlap_count,
        "duplicate_detection": "NFC and whitespace normalization; original records unchanged",
        "removed_training_duplicates": duplicate_count,
        "removed_training_test_overlap": removed_overlap,
        "splits": splits,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in (train_file, test_file)
        ],
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_msr_provenance(path: Path, samples: list[SequenceSample]) -> dict[str, Any] | None:
    """校验 importer 清单和实际记录；旧/修改后的划分不能冒充 standard。"""
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise DataValidationError("MSR manifest 必须为 JSON mapping")
        if manifest.get("dataset") != MSR_DATASET:
            return None
        if manifest.get("protocol") not in {"standard", "clean"}:
            raise DataValidationError("旧或无效 MSR protocol 清单；请重新导入至新目录")
        split = path.stem
        if path.name != f"{split}.jsonl" or split not in {"train", "dev", "test"}:
            raise DataValidationError("MSR 输入必须是清单中的 train/dev/test.jsonl")
        record = manifest["splits"][split]
        if (
            record["sha256"] != sha256(path.read_bytes()).hexdigest()
            or record["sample_fingerprint"] != sample_fingerprint(samples)
            or record["count"] != len(samples)
        ):
            raise DataValidationError("MSR 划分与 manifest 指纹/数量不一致")
        if manifest["protocol"] == "standard" and (
            manifest["train_count"] != manifest["original_train_count"]
            or manifest["dev_count"] != 0
            or manifest["removed_training_duplicates"] != 0
            or manifest["removed_training_test_overlap"] != 0
        ):
            raise DataValidationError("standard 必须保留全部官方训练样本，不执行清理或 dev 划分")
        return {
            "protocol": manifest["protocol"],
            "manifest_sha256": sha256(raw).hexdigest(),
            "split": split,
            "sample_fingerprint": record["sample_fingerprint"],
            "sources": manifest["sources"],
        }
    except (ValueError, TypeError, KeyError) as error:
        raise DataValidationError(f"MSR 清单损坏: {manifest_path}") from error
