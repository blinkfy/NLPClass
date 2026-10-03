"""只读、按需加载已冻结的分类 split；不生成或修改划分。"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.data.corpus import iter_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES


@dataclass(frozen=True)
class ClassificationDocument:
    id: str
    text: str
    label: str


class FrozenClassificationData:
    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "data/splits/thucnews_600"
        self.manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        selected = root / "data/course/thucnews_600.jsonl"
        selection_manifest = selected.with_name("thucnews_600_manifest.json")
        if (
            file_sha256(selected) != self.manifest["subset_sha256"]
            or file_sha256(selection_manifest) != self.manifest["selection_manifest_sha256"]
        ):
            raise DataValidationError("冻结 THUCNews 来源指纹不符")
        ids = []
        for split, per_class in (("train", 84), ("val", 18), ("test", 18)):
            info = self.manifest["splits"][split]
            if (
                info["count"] != per_class * 5
                or len(info["ids"]) != per_class * 5
                or info["category_counts"] != {label: per_class for label in CATEGORIES}
            ):
                raise DataValidationError("固定 84/18/18 split 不符")
            ids.extend(info["ids"])
        if len(set(ids)) != 600:
            raise DataValidationError("split ID 重复或未覆盖 600 篇")
        # 完整性检查读取字节，不解码 Test 文本，不构建任何特征。
        for name, digest in self.manifest["file_sha256"].items():
            if file_sha256(self.directory / name) != digest:
                raise DataValidationError("冻结 split 文件指纹不符")
        self.opened_splits = []

    def load(self, split: str) -> list[ClassificationDocument]:
        if split not in {"train", "val", "test"}:
            raise DataValidationError("未知 split")
        rows = list(iter_jsonl(self.directory / f"{split}.jsonl", lambda row: row))
        if [row["id"] for row in rows] != self.manifest["splits"][split]["ids"]:
            raise DataValidationError("split ID/顺序与清单不符")
        if Counter(row["primary_label"] for row in rows) != Counter(
            self.manifest["splits"][split]["category_counts"]
        ):
            raise DataValidationError("split 类别不符")
        from nlplab.core.schema import text_sha256

        if any(text_sha256(row["text"]) != row["sha256"] for row in rows):
            raise DataValidationError("文档文本指纹不符")
        self.opened_splits.append(split)
        return [
            ClassificationDocument(row["id"], row["text"], row["primary_label"]) for row in rows
        ]

    def provenance(self):
        return {
            "name": "THUCNews personal-600 fulltext",
            "dataset_fingerprint": self.manifest["subset_sha256"],
            "split_manifest_sha256": file_sha256(self.directory / "manifest.json"),
            "selection_manifest_sha256": self.manifest["selection_manifest_sha256"],
            "file_sha256": self.manifest["file_sha256"],
            "splits": self.manifest["splits"],
        }
