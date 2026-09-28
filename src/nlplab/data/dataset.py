"""序列数据加载与可复现划分，保护原始外部文件。"""

import random
from pathlib import Path

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.schema import SequenceSample
from nlplab.data.corpus import duplicate_key, read_jsonl


def load_samples(path: Path) -> list[SequenceSample]:
    return read_jsonl(path, SequenceSample.from_dict)


def read_msr(path: Path, prefix: str) -> list[SequenceSample]:
    """读取 MSR .utf8 分词 Gold；仅移除词间分隔符，不改写源文件。"""
    if not path.is_file():
        raise ResourceUnavailableError(f"MSR 文件缺失: {path}；参见 README 的数据准备说明")
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            words = line.split()
            if not words:
                continue
            result.append(
                SequenceSample(
                    f"{prefix}_{number:06}", f"{prefix}_{number:06}", "".join(words), words
                )
            )
    if not result:
        raise DataValidationError(f"MSR 文件为空: {path}")
    return result


def split_training(
    samples: list[SequenceSample], dev_ratio: float = 0.1, seed: int = 42
) -> tuple[list[SequenceSample], list[SequenceSample], int]:
    """先去重再划分，同文不同 Gold 直接报错；不使用官方测试调参。"""
    if not 0 < dev_ratio < 1:
        raise DataValidationError("dev_ratio 必须在 (0,1) 内")
    unique: dict[str, SequenceSample] = {}
    for sample in samples:
        key = duplicate_key(sample.text)
        if key in unique and unique[key].gold_tokens != sample.gold_tokens:
            raise DataValidationError("训练集中存在同文不同 Gold，必须人工核查")
        unique.setdefault(key, sample)
    records = list(unique.values())
    if len(records) < 2:
        raise DataValidationError("划分 train/dev 至少需要两条不同文本")
    random.Random(seed).shuffle(records)
    dev_count = max(1, min(len(records) - 1, round(len(records) * dev_ratio)))
    return records[dev_count:], records[:dev_count], len(samples) - len(records)
