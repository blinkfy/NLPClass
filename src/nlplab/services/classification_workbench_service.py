"""Read saved documents/encodings/explanations; never run a classifier or tokenizer."""

import json
import unicodedata
from collections import Counter

import numpy as np

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.services.classification_stage3_service import ClassificationStage3Service
from nlplab.services.classification_stage5_service import ClassificationStage5Service


class ClassificationWorkbenchService:
    _verified = set()
    _array_records = {}

    def __init__(self, settings):
        self.settings = settings
        self.formal = ClassificationStage3Service(settings)

    def _path(self, directory, name):
        directory = directory.resolve()
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise DataValidationError("工作台 artifact 路径无效")
        manifest = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))["files"]
        if name not in manifest or not path.is_file():
            raise ResourceUnavailableError(f"冻结 artifact 不存在: {name}")
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, manifest[name])
        if key not in self._verified:
            if file_sha256(path) != manifest[name]:
                raise DataValidationError(f"工作台 artifact hash 不符: {name}")
            self._verified.add(key)
        return path

    def _json(self, directory, name):
        return json.loads(self._path(directory, name).read_text("utf-8"))

    def _record(self, directory, name, document_id):
        if name.endswith(".json"):
            path = self._path(directory, name)
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key not in self._array_records:
                self._array_records[key] = {
                    row.get("document_id", row.get("id")): {
                        k: v for k, v in row.items() if k != "all_channels"
                    }
                    for row in json.loads(path.read_text("utf-8"))
                }
            if document_id in self._array_records[key]:
                return self._array_records[key][document_id]
            raise ResourceUnavailableError("冻结样本中没有此 document ID")
        with self._path(directory, name).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("document_id", row.get("id")) == document_id:
                    return row
        raise ResourceUnavailableError("冻结样本中没有此 document ID")

    def _sparse_row(self, directory, document_id, terms):
        # Copy coordinates from the already-saved CSR matrix. No transform/fit.
        ids = self._json(directory, "dataset.json")["splits"]["test"]["ids"]
        if document_id not in ids:
            raise DataValidationError("冻结矩阵 document IDs 不匹配")
        index = ids.index(document_id)
        with np.load(self._path(directory, "final/test.npz"), allow_pickle=False) as matrix:
            if matrix["format"].item() != b"csr" or int(matrix["shape"][0]) != len(ids):
                raise DataValidationError("冻结 Test 矩阵形状/格式不匹配")
            start, end = matrix["indptr"][index : index + 2]
            columns, values = matrix["indices"][start:end], matrix["data"][start:end]
            dimension = int(matrix["shape"][1])
            if len(terms) != dimension:
                raise DataValidationError("冻结特征名和矩阵列不一致")
            nonzero = [
                {"term": terms[int(c)], "tfidf": float(v), "column": int(c)}
                for c, v in zip(columns, values)
            ]
        return {
            "top_tfidf_features": sorted(nonzero, key=lambda r: (-r["tfidf"], r["column"]))[:20],
            "nonzero_count": len(nonzero),
            "dimension": dimension,
            "sparsity": 1 - len(nonzero) / dimension,
            "matrix_source": "final/test.npz · saved CSR",
        }

    @staticmethod
    def _character_membership(text, raw_vocabulary, selected):
        # Display candidate occurrences and look up frozen membership only.
        # This does not create a representation matrix or run a model.
        counts, previous = Counter(), None
        for char in unicodedata.normalize("NFKC", text).lower():
            content = (
                "a" <= char <= "z"
                or unicodedata.category(char) == "Nd"
                or unicodedata.name(char, "").startswith(
                    ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
                )
            )
            if content:
                if previous is not None:
                    counts[previous + char] += 1
                previous = char
            else:
                previous = None
        scores = {r["term"]: r["chi2_score"] for r in selected}
        return [
            {
                "term": term,
                "count": count,
                "in_raw_vocabulary": term in raw_vocabulary,
                "retained": term in scores,
                "chi2_score": scores.get(term),
            }
            for term, count in sorted(
                counts.items(), key=lambda r: (r[0] not in scores, -r[1], r[0])
            )
        ]

    def document(self, run_id, document_id, source="test", condition="OA"):
        if source == "stage5":
            return self._development_document(document_id, condition)
        directory = self.formal.store.resolve(run_id)
        info = self._json(directory, "run.json")
        kind = info["run_type"]
        stage1 = self.formal.store.resolve(self.formal.parent_run(directory.name))
        records = {
            m: self._record(stage1, f"{m}_test_predictions.jsonl", document_id)
            for m in ("C0", "C1")
        }
        if kind in ("formal-exp3-stage2", "formal-exp3-stage3"):
            stage2 = self.formal.stage2_directory(directory.name)
            records["C2"] = self._record(stage2, "C2_test_predictions.jsonl", document_id)
        gold = records["C0"]["gold"]
        if any(row["gold"] != gold for row in records.values()):
            raise DataValidationError("工作台冻结样本 gold 不一致")
        dataset = self._json(stage1, "dataset.json")
        test_path = self.settings.root / "data/splits/thucnews_600/test.jsonl"
        if (
            not test_path.is_file()
            or file_sha256(test_path) != dataset["file_sha256"]["test.jsonl"]
        ):
            raise DataValidationError("冻结 Test 正文来源不一致")
        sample = None
        with test_path.open(encoding="utf-8") as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate.get("id") == document_id:
                    sample = candidate
                    break
        if sample is None or sample["primary_label"] != gold:
            raise DataValidationError("冻结 Test 文档/gold 不一致")
        text = sample["text"]
        if kind in ("formal-exp3-stage2", "formal-exp3-stage3"):
            for child in {stage2, directory} if kind == "formal-exp3-stage3" else {stage2}:
                child_dataset = self._json(child, "dataset.json")
                if child_dataset.get("dataset_fingerprint") != dataset.get("dataset_fingerprint"):
                    raise DataValidationError("历史 Run 数据指纹不一致")
        representations = {m: {} for m in records}
        source_run = directory.name
        if kind == "formal-exp3-stage3":
            stage3 = self.formal.stage3_directory(directory.name)
            c3 = self._record(stage3, "C3_test_predictions.jsonl", document_id)
            if c3["gold"] != gold or c3["text"] != text:
                raise DataValidationError("工作台冻结样本正文/gold 不一致")
            records["C3"] = {k: v for k, v in c3.items() if k != "all_channels"}
            text = c3["text"]
            source_run = stage3.name
            encoding = self._json(stage3, "test_encoding.json")
            index = encoding["document_ids"].index(document_id)
            with np.load(self._path(stage3, "test_encoding.npz"), allow_pickle=False) as arrays:
                length = int(arrays["lengths"][index])
                ids = arrays["ids"][index, :length].tolist()
            vocab = self._json(stage3, "final_vocabulary/vocabulary.json")
            sequence = {
                "tokens": encoding["tokens"][index],
                "original_positions": encoding["original_positions"][index],
                "token_ids": ids,
                "model_tokens": [vocab["terms"][i] for i in ids],
                "diagnostic": c3["sequence_diagnostic"],
                "source_run": stage3.name,
                "pooled": [row["max_activation"] for row in c3["all_channels"]],
                "embedding_values_saved": False,
            }
            representations = {m: {"sequence": sequence} for m in records}
        word_vocabulary = self._json(stage1, "final/tfidf/vocabulary.json")["vocabulary"]
        word_terms = [term for term, _ in sorted(word_vocabulary.items(), key=lambda r: r[1])]
        word_row = self._sparse_row(stage1, document_id, word_terms)
        for model in ("C0", "C1"):
            representations[model].update(word_row)
        if "C2" in records:
            selected = self._json(stage2, "final/representation/selected_features.json")
            raw = self._json(stage2, "final/representation/representation.json")
            representations["C2"]["selected_features"] = selected
            representations["C2"]["raw_vocabulary"] = list(raw.get("vocabulary", {}))
            representations["C2"].update(
                self._sparse_row(
                    stage2,
                    document_id,
                    [r["term"] for r in sorted(selected, key=lambda r: r["column"])],
                )
            )
            representations["C2"]["feature_membership"] = self._character_membership(
                text, raw["vocabulary"], selected
            )
        return {
            "document": {
                "document_id": document_id,
                "gold": gold,
                "text": text,
                "title": text.splitlines()[0][:65],
                "split": "Test",
                "source_run": source_run,
            },
            "records": records,
            "representations": representations,
            "provenance": {
                "view_run_id": directory.name,
                "saved_predictions_only": True,
                "inference_executed": False,
            },
        }

    def _development_document(self, document_id, condition):
        if condition not in ("OS", "OT", "OA", "AA"):
            raise DataValidationError("Stage5 条件无效")
        directory = ClassificationStage5Service(self.settings).resolve()
        validation = self._json(directory, "validation_open_record.json")
        val_path = self.settings.root / "data/splits/thucnews_600/val.jsonl"
        if (
            document_id not in validation["document_ids"]
            or file_sha256(val_path) != validation["validation_sha256"]
        ):
            raise DataValidationError("冻结 Validation 正文来源不一致")
        flips = self._json(directory, "flips.json")
        matches = [r for r in flips if r["document_id"] == document_id]
        if not matches:
            raise ResourceUnavailableError("Stage5 flip 中没有此样本")
        records, changes = {}, {}
        for model in ("C0", "C1", "C2", "C3"):
            original = self._record(directory, f"{model}_OO_predictions.json", document_id)
            clean = self._record(directory, f"{model}_{condition}_predictions.json", document_id)
            records[model] = {k: v for k, v in original.items() if k != "all_channels"}
            changes[model] = {k: v for k, v in clean.items() if k != "all_channels"}
        sample = matches[0]
        text = None
        with val_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("id") == document_id:
                    text = row["text"]
                    break
        if text is None:
            raise ResourceUnavailableError("冻结 Validation split 中没有此文档")
        return {
            "document": {
                "document_id": document_id,
                "gold": sample["gold"],
                "text": text,
                "title": text.splitlines()[0][:65],
                "split": "Validation · Stage5",
                "source_run": directory.name,
            },
            "records": records,
            "representations": {},
            "ablation": {
                "condition": condition,
                "changes": changes,
                "flips": matches,
                "removed_spans": next(
                    (r.get("removed_spans", []) for r in matches if r["condition"] == condition), []
                ),
            },
            "provenance": {
                "view_run_id": directory.name,
                "saved_predictions_only": True,
                "inference_executed": False,
                "model_fit_split": "Train420; Stage5 original model",
            },
        }
