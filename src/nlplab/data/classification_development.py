"""Stage4 development-only access; never opens the held-out split or full 600 corpus."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.corpus import iter_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES


def forbidden_development_path(root, value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    path = Path(os.fsdecode(value)).resolve()
    root = Path(root).resolve()
    if not path.is_relative_to(root):
        return False
    relative = path.relative_to(root).as_posix().lower()
    if relative.startswith("data/splits/thucnews_600/test"):
        return True
    if relative.startswith("data/course/thucnews_600"):
        return True
    stage4_metadata = (
        relative.startswith("outputs/exp3/stage4/")
        or (path.parent == root / "outputs/exp3" and path.name.startswith("stage4_"))
    )
    if relative.startswith("outputs/exp3/") and not stage4_metadata:
        return (
            "test" in path.name.lower()
            or path.name
            in {"comparison_metrics.json", "representation_statistics.json", "dataset.json"}
            or path.name.startswith("paired_")
        )
    return False


def install_development_io_guard(root):
    def guard(event, args):
        if event == "open" and forbidden_development_path(root, args[0]):
            raise PermissionError("Stage4 forbids Test/full-corpus/historical Test asset access")

    sys.addaudithook(guard)


class DevelopmentClassificationData:
    allowed_splits = frozenset({"train", "val"})

    def __init__(self, root):
        self.directory = Path(root) / "data/splits/thucnews_600"
        manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        self.info = {s: manifest["splits"][s] for s in ("train", "val")}
        self.hashes = {}
        self.opened_splits = []
        for split, per_class in (("train", 84), ("val", 18)):
            info = self.info[split]
            if (
                info["count"] != per_class * 5
                or len(info["ids"]) != per_class * 5
                or info["category_counts"] != {c: per_class for c in CATEGORIES}
            ):
                raise DataValidationError("Frozen Train/Validation counts changed")
            name = f"{split}.jsonl"
            digest = file_sha256(self.directory / name)
            if digest != manifest["file_sha256"][name]:
                raise DataValidationError("Frozen development split hash changed")
            self.hashes[name] = digest
        ids = self.info["train"]["ids"] + self.info["val"]["ids"]
        if len(set(ids)) != 510:
            raise DataValidationError("Development IDs overlap")
        self.manifest_sha256 = file_sha256(self.directory / "manifest.json")

    def load(self, split):
        if split not in self.allowed_splits:
            raise DataValidationError("Stage4 only permits train/val")
        rows = list(iter_jsonl(self.directory / f"{split}.jsonl", lambda row: row))
        if (
            [r["id"] for r in rows] != self.info[split]["ids"]
            or Counter(r["primary_label"] for r in rows)
            != Counter(self.info[split]["category_counts"])
            or any(text_sha256(r["text"]) != r["sha256"] for r in rows)
        ):
            raise DataValidationError("Development split records changed")
        self.opened_splits.append(split)
        return [ClassificationDocument(r["id"], r["text"], r["primary_label"]) for r in rows]

    def provenance(self):
        return {
            "split_manifest_sha256": self.manifest_sha256,
            "file_sha256": self.hashes,
            "splits": self.info,
            "opened_splits": self.opened_splits,
            "test_access_permitted": False,
            "full_corpus_opened": False,
        }
