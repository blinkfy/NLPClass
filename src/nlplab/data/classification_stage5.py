"""Lazy Train/Validation loader with a mandatory cleaning freeze before Validation I/O."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256
from nlplab.data.classification_development import forbidden_development_path
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.corpus import iter_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES


class CleaningFreeze:
    def __init__(self, directory):
        self.directory = Path(directory)

    def verify(self):
        seal = json.loads((self.directory / "cleaning_freeze.json").read_text("utf-8"))
        for name, digest in seal["files"].items():
            if file_sha256(self.directory / name) != digest:
                raise DataValidationError("Stage5 frozen cleaning rules/source changed")
        for name, digest in seal.get("source_hashes", {}).items():
            if file_sha256(Path(seal["root"]) / name) != digest:
                raise DataValidationError("Stage5 frozen cleaning implementation changed")
        return seal


def install_stage5_guard(root, freeze=None):
    root = Path(root).resolve()

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(root):
            return
        relative = path.relative_to(root).as_posix().lower()
        own = relative.startswith("outputs/exp3/stage5/") or (
            path.parent == root / "outputs/exp3" and path.name.startswith("stage5_")
        )
        if not own and forbidden_development_path(root, path):
            raise PermissionError("Stage5 prohibits Test/full600/historical Test assets")
        if relative.startswith("data/splits/thucnews_600/val") or (
            relative.startswith("outputs/exp3/stage4/")
            and ("sequence_val" in path.name or path.name.endswith("_result.json"))
        ):
            if freeze is None:
                raise PermissionError("Validation cannot open before Stage5 cleaning freeze")
            freeze.verify()

    sys.addaudithook(guard)


class Stage5Data:
    def __init__(self, root, freeze=None):
        self.directory = Path(root) / "data/splits/thucnews_600"
        self.manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        self.freeze = freeze
        self.opened_splits = []

    def load(self, split):
        if split not in {"train", "val"}:
            raise DataValidationError("Stage5 permits Train/Validation only")
        if split == "val":
            if self.freeze is None:
                raise DataValidationError("Cleaning freeze required before Validation")
            self.freeze.verify()
        info = self.manifest["splits"][split]
        name = f"{split}.jsonl"
        if file_sha256(self.directory / name) != self.manifest["file_sha256"][name]:
            raise DataValidationError("Frozen development split changed")
        rows = list(iter_jsonl(self.directory / name, lambda row: row))
        count = 84 if split == "train" else 18
        if (
            [r["id"] for r in rows] != info["ids"]
            or Counter(r["primary_label"] for r in rows) != Counter({c: count for c in CATEGORIES})
            or any(text_sha256(r["text"]) != r["sha256"] for r in rows)
        ):
            raise DataValidationError("Frozen development records changed")
        self.opened_splits.append(split)
        return [ClassificationDocument(r["id"], r["text"], r["primary_label"]) for r in rows]
