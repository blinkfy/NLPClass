"""人工草稿、显式复核、严格导出。BMES 只派生，不独立保存。"""

import json
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.schema import Entity
from nlplab.data.corpus import load_documents, read_jsonl
from nlplab.data.course import DEFAULT_PLAN, file_sha256, load_plan, quota, save_quality_report
from nlplab.data.sequence_sampling import difficult_features
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.tags import is_legal_sequence

SAMPLE_ID = re.compile(r"SEQ[0-9]{5,}\Z")
STATUSES = ("pending", "in_progress", "done", "reviewed")
GOLD_SCHEMA_PATH = DEFAULT_PLAN.parent.parent / "docs/sequence_gold.schema.json"
GOLD_SCHEMA = json.loads(GOLD_SCHEMA_PATH.read_text(encoding="utf-8"))
GOLD_VALIDATOR = Draft202012Validator(GOLD_SCHEMA)


class DraftConflictError(DataValidationError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    """同目录临时文件 + fsync + replace；异常时原草稿保持不变。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".draft-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def boundary_tokens(text: str, boundaries: list[int]) -> list[str]:
    if not isinstance(boundaries, list) or any(
        type(i) is not int or not 0 < i < len(text) for i in boundaries
    ):
        raise DataValidationError("分词边界必须是原文内部的 code-point 整数位置")
    if boundaries != sorted(set(boundaries)):
        raise DataValidationError("分词边界必须升序且不重复")
    cuts = [0, *boundaries, len(text)]
    return [text[a:b] for a, b in zip(cuts, cuts[1:])]


def token_boundaries(tokens: list[str]) -> list[int]:
    position, result = 0, []
    for token in tokens[:-1]:
        position += len(token)
        result.append(position)
    return result


def validate_record(value: dict, document_map: dict, *, complete: bool = False) -> None:
    problem = next(GOLD_VALIDATOR.iter_errors(value), None)
    if problem is not None:
        location = "/".join(str(part) for part in problem.absolute_path)
        raise DataValidationError(f"sequence_gold.schema: {location}: {problem.message}")
    if not isinstance(value.get("sample_id"), str) or not SAMPLE_ID.fullmatch(value["sample_id"]):
        raise DataValidationError("sample_id 必须符合 SEQ + 至少五位数字")
    if value.get("doc_id") not in document_map:
        raise DataValidationError("doc_id 不存在于主语料")
    doc = document_map[value["doc_id"]]
    text, start, end = value.get("text"), value.get("source_start"), value.get("source_end")
    if not isinstance(text, str) or not text or type(start) is not int or type(end) is not int:
        raise DataValidationError("必须提供原文与 Python code-point source offsets")
    if not 0 <= start < end <= len(doc.text) or doc.text[start:end] != text:
        raise DataValidationError("document.text[source_start:source_end] != sample.text")
    if (
        value.get("primary_label") != doc.primary_label
        or value.get("source_type") != doc.source_type
    ):
        raise DataValidationError("样本 label/source_type 与文档 metadata 不一致")
    if value.get("is_demo", False) != doc.is_demo:
        raise DataValidationError("样本 demo 标记与原文不一致")
    if value.get("status") not in STATUSES:
        raise DataValidationError("未知标注状态")
    if type(value.get("revision_count", 0)) is not int or value.get("revision_count", 0) < 0:
        raise DataValidationError("revision_count 必须为非负整数")
    if not isinstance(value.get("notes", ""), str):
        raise DataValidationError("notes 必须为字符串")
    if not isinstance(value.get("selection_reasons", []), list) or any(
        not isinstance(x, str) for x in value.get("selection_reasons", [])
    ):
        raise DataValidationError("selection_reasons 必须为字符串列表")
    tokens = value.get("gold_tokens")
    if tokens is not None:
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(t, str) or not t for t in tokens)
        ):
            raise DataValidationError("gold_tokens 必须为非空字符串列表或 null")
        if "".join(tokens) != text:
            raise DataValidationError("tokens roundtrip 与原文不一致")
        if not is_legal_sequence(encode(tokens)):
            raise DataValidationError("非法 BMES")
    entities = value.get("entities")
    if entities is not None:
        if not isinstance(entities, list):
            raise DataValidationError("entities 必须为列表或 null")
        previous_end = 0
        for item in sorted(
            entities, key=lambda e: e.get("start", -1) if isinstance(e, dict) else -1
        ):
            if not isinstance(item, dict) or set(item) != {"text", "type", "start", "end"}:
                raise DataValidationError("Gold entity 仅允许 text/type/start/end，不包含 score")
            Entity(**item).validate(text)
            if item["start"] < previous_end:
                raise DataValidationError("实体跨度重叠")
            previous_end = item["end"]
    if complete and (tokens is None or entities is None):
        raise DataValidationError("done/reviewed 必须人工确认分词和 NER；[] 表示确认无实体")


class AnnotationStore:
    def __init__(
        self,
        root: Path,
        *,
        corpus: Path | None = None,
        candidates: Path | None = None,
        work: Path | None = None,
        config: Path | None = None,
    ):
        self.root = root
        self.corpus = corpus or root / "data/raw/documents.jsonl"
        self.candidates = candidates or root / "data/annotations/candidates.jsonl"
        self.work = work or root / "data/annotations/work"
        self.config = config or (
            root / "configs/course_corpus.yaml"
            if (root / "configs/course_corpus.yaml").is_file()
            else DEFAULT_PLAN
        )
        self.lock = threading.RLock()

    def base_records(self) -> tuple[list[dict], dict]:
        documents = load_documents(self.corpus)
        doc_map = {doc.id: doc for doc in documents}
        if len(doc_map) != len(documents):
            raise DataValidationError("主语料存在重复 document id")
        records = read_jsonl(self.candidates, lambda value: value)
        sample_ids, doc_ids = set(), set()
        plan = load_plan(self.config)
        sampling = plan["sequence_gold_sampling"]
        for record in records:
            for name, default in (
                ("status", "pending"),
                ("gold_tokens", None),
                ("entities", None),
                ("revision_count", 0),
                ("notes", ""),
                ("selection_reasons", []),
                ("is_demo", False),
            ):
                record.setdefault(name, default)
            if record.get("doc_id") in doc_map:
                doc = doc_map[record["doc_id"]]
                record.setdefault("primary_label", doc.primary_label)
                record.setdefault("source_type", doc.source_type)
            validate_record(record, doc_map)
            if (
                record["status"] != "pending"
                or record.get("gold_tokens") is not None
                or record.get("entities") is not None
            ):
                raise DataValidationError(
                    "候选输入必须 pending、Gold 为 null；人工标注仅保存在 work 草稿"
                )
            if (
                not sampling["candidate_min_codepoints"]
                <= len(record["text"])
                <= sampling["candidate_max_codepoints"]
            ):
                raise DataValidationError("候选长度必须为 15–80 code point")
            if (
                record["primary_label"] not in sampling["primary_label_targets"]
                or record["source_type"] not in sampling["source_type_targets"]
            ):
                raise DataValidationError("候选 metadata 不在正式计划中")
            if record["sample_id"] in sample_ids or record["doc_id"] in doc_ids:
                raise DataValidationError("candidate sample_id/doc_id 必须唯一")
            sample_ids.add(record["sample_id"])
            doc_ids.add(record["doc_id"])
        manifest_path = self.candidates.with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not manifest.get("success")
                or manifest.get("candidate_sha256") != file_sha256(self.candidates)
                or manifest.get("corpus_sha256") != file_sha256(self.corpus)
            ):
                raise DataValidationError("采样后 corpus/candidates 指纹变化；请核对 manifest")
        return records, doc_map

    def draft_path(self, sample_id: str) -> Path:
        if not SAMPLE_ID.fullmatch(sample_id):
            raise DataValidationError("非法 sample_id")
        return self.work / (sample_id + ".json")

    def merge(self, base: dict, doc_map: dict) -> dict:
        path = self.draft_path(base["sample_id"])
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            for name in (
                "sample_id",
                "doc_id",
                "text",
                "source_start",
                "source_end",
                "primary_label",
                "source_type",
                "is_demo",
                "selection_reasons",
            ):
                if value.get(name) != base.get(name):
                    raise DataValidationError(f"草稿不可改变来源字段: {name}")
        else:
            value = {**base, "draft_version": 0, "review_history": []}
        value.setdefault("review_history", [])
        validate_record(value, doc_map, complete=value["status"] in {"done", "reviewed"})
        return value

    def get(self, sample_id: str) -> dict:
        self.draft_path(sample_id)
        records, doc_map = self.base_records()
        base = next((r for r in records if r["sample_id"] == sample_id), None)
        if base is None:
            raise ResourceUnavailableError("样本不存在于候选队列")
        value = self.merge(base, doc_map)
        tokens = value.get("gold_tokens")
        return {
            **value,
            "boundaries": token_boundaries(tokens) if tokens is not None else None,
            "bmes": encode(tokens) if tokens is not None else None,
            "source_title": doc_map[value["doc_id"]].source_title,
            "source_url": doc_map[value["doc_id"]].source_url,
        }

    def queue(
        self, label: str | None = None, source_type: str | None = None, status: str | None = None
    ) -> dict:
        records, doc_map = self.base_records()
        values = [self.merge(r, doc_map) for r in records]
        counts = {name: sum(r["status"] == name for r in values) for name in STATUSES}
        selected = [
            r
            for r in values
            if (not label or r["primary_label"] == label)
            and (not source_type or r["source_type"] == source_type)
            and (not status or r["status"] == status)
        ]
        return {
            "items": [
                {k: r[k] for k in ("sample_id", "doc_id", "primary_label", "source_type", "status")}
                for r in selected
            ],
            "total": len(values),
            "filtered_count": len(selected),
            "progress": counts,
            "labels": sorted({r["primary_label"] for r in values}),
            "source_types": sorted({r["source_type"] for r in values}),
            "is_demo": any(r.get("is_demo") for r in values),
        }

    def save(self, sample_id: str, changes: dict, expected_version: int) -> dict:
        self.draft_path(sample_id)
        allowed = {"boundaries", "entities", "notes", "status", "review_confirmation"}
        if set(changes) - allowed:
            raise DataValidationError(
                "仅允许人工边界、实体、备注和状态；不可直接修改来源或复核次数"
            )
        with self.lock:
            records, doc_map = self.base_records()
            base = next((r for r in records if r["sample_id"] == sample_id), None)
            if base is None:
                raise ResourceUnavailableError("样本不存在")
            current = self.merge(base, doc_map)
            if type(expected_version) is not int or expected_version != current["draft_version"]:
                raise DraftConflictError("草稿已被更新；请保留当前编辑并重新加载，禁止覆盖他人版本")
            value = dict(current)
            if "boundaries" in changes:
                value["gold_tokens"] = (
                    boundary_tokens(value["text"], changes["boundaries"])
                    if changes["boundaries"] is not None
                    else None
                )
            for name in ("entities", "notes"):
                if name in changes:
                    value[name] = changes[name]
            changed_gold = any(value.get(k) != current.get(k) for k in ("gold_tokens", "entities"))
            status = changes.get("status")
            if status is None:
                value["status"] = (
                    "in_progress"
                    if current["status"] == "pending" or changed_gold
                    else current["status"]
                )
            elif status == "reviewed" and current["status"] != "reviewed":
                if (
                    current["status"] != "done"
                    or changed_gold
                    or changes.get("review_confirmation") is not True
                ):
                    raise DataValidationError("先保存 done，再显式确认第二遍复核才能 reviewed")
                value["status"] = "reviewed"
                history = list(current.get("review_history", []))
                history.append(
                    {
                        "action": "second_pass_review",
                        "previous_status": "done",
                        "confirmed": True,
                        "draft_version": expected_version + 1,
                        "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                value["review_history"] = history
                value["revision_count"] = len(history)
            else:
                if status not in STATUSES or status == "pending":
                    raise DataValidationError("保存后状态仅允许 in_progress/done/reviewed")
                if status == "reviewed" and changed_gold:
                    raise DataValidationError("修改 Gold 后必须重新 done 和复核")
                value["status"] = status
            validate_record(value, doc_map, complete=value["status"] in {"done", "reviewed"})
            value["draft_version"] = expected_version + 1
            value["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(self.draft_path(sample_id), value)
        return self.get(sample_id)

    def validation(self) -> tuple[list[dict], dict]:
        bases, doc_map = self.base_records()
        values, errors = [], []
        for base in bases:
            try:
                values.append(self.merge(base, doc_map))
            except (DataValidationError, ValueError, KeyError, TypeError) as error:
                errors.append({"sample_id": base["sample_id"], "message": str(error)})
        reviewed = [r for r in values if r["status"] == "reviewed"]
        plan = load_plan(self.config)
        sampling = plan["sequence_gold_sampling"]
        if len(reviewed) != 200:
            errors.append(
                {"check": "exactly_200_reviewed", "actual": len(reviewed), "required": 200}
            )
        labels = Counter(r["primary_label"] for r in reviewed)
        types = Counter(r["source_type"] for r in reviewed)
        quotas = {
            "labels": quota(labels, sampling["primary_label_targets"]),
            "source_types": quota(
                types, sampling["source_type_targets"], sampling["source_type_tolerance"]
            ),
        }
        if any(not item["met"] for group in quotas.values() for item in group.values()):
            errors.append({"check": "sample_quotas", "message": "label/source_type 配额不满足"})
        entity_counts = Counter(e["type"] for r in reviewed for e in r["entities"])
        with_entity = sum(bool(r["entities"]) for r in reviewed)
        difficult = sum(bool(difficult_features(r["text"])) for r in reviewed)
        second_pass = 0
        for record in reviewed:
            if record.get("is_demo") or doc_map[record["doc_id"]].is_demo:
                errors.append({"sample_id": record["sample_id"], "check": "demo_not_formal_gold"})
            history = record.get("review_history", [])
            valid_history = isinstance(history, list) and all(
                isinstance(event, dict)
                and event.get("action") == "second_pass_review"
                and event.get("previous_status") == "done"
                and event.get("confirmed") is True
                and isinstance(event.get("reviewed_at"), str)
                and type(event.get("draft_version")) is int
                and 0 < event["draft_version"] <= record["draft_version"]
                for event in history
            )
            if not valid_history or record["revision_count"] != len(history):
                errors.append(
                    {"sample_id": record["sample_id"], "check": "review_history_inconsistent"}
                )
            elif history:
                second_pass += 1
        targets = sampling["post_annotation_quality_targets"]
        quality = {
            "with_entity": {
                "actual": with_entity,
                "minimum": targets["samples_with_any_entity_min"],
            },
            "difficult_samples": {"actual": difficult, "minimum": targets["difficult_samples_min"]},
            "second_pass_samples": {
                "actual": second_pass,
                "minimum": plan["quality_control"]["review"]["minimum_second_pass_samples"],
            },
        }
        quality.update(
            {
                name: {"actual": entity_counts[name], "minimum": minimum}
                for name, minimum in targets["entity_support_min"].items()
            }
        )
        for name, check in quality.items():
            if check["actual"] < check["minimum"]:
                errors.append({"check": name, **check})
        return reviewed, {
            "kind": "sequence_annotation_validation",
            "valid": not errors,
            "errors": errors,
            "reviewed_count": len(reviewed),
            "status_distribution": dict(Counter(r["status"] for r in values)),
            "primary_label_distribution": dict(labels),
            "source_type_distribution": dict(types),
            "quotas": quotas,
            "quality_targets": quality,
            "corpus_sha256": file_sha256(self.corpus),
            "candidate_sha256": file_sha256(self.candidates),
            "config_sha256": file_sha256(self.config),
            "original_data_modified": False,
        }

    def export(self, output: Path, reports: Path) -> dict:
        reviewed, result = self.validation()
        report = save_quality_report(result, reports, "Sequence Gold 正式导出校验")
        if not result["valid"]:
            raise DataValidationError(
                f"正式导出拒绝：{json.dumps(result['errors'], ensure_ascii=False)}；"
                f"报告 {report / 'report.md'}"
            )
        manifest_path = output.with_name(output.stem + "_manifest.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output.parent / ("." + output.name + ".export.lock")
        descriptor = None
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            if output.exists() or manifest_path.exists():
                raise DataValidationError("正式 Gold 或 manifest 已存在，拒绝覆盖冻结数据")
            names = (
                "sample_id",
                "doc_id",
                "text",
                "source_start",
                "source_end",
                "primary_label",
                "source_type",
                "status",
                "gold_tokens",
                "entities",
                "is_demo",
                "revision_count",
                "notes",
                "selection_reasons",
                "review_history",
            )
            exported = [
                {name: record[name] for name in names}
                for record in sorted(reviewed, key=lambda r: r["sample_id"])
            ]
            payload = "".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in exported
            )
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
            ) as stream:
                temp = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temp, output)
            finally:
                temp.unlink(missing_ok=True)
            manifest = {
                **result,
                "gold_sha256": file_sha256(output),
                "gold_samples": 200,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "gold_origin": "manual-reviewed-drafts",
                "sampling_manifest_sha256": file_sha256(
                    self.candidates.with_suffix(".manifest.json")
                )
                if self.candidates.with_suffix(".manifest.json").is_file()
                else None,
                "draft_sha256": {
                    r["sample_id"]: file_sha256(self.draft_path(r["sample_id"])) for r in reviewed
                },
                "report_path": str(report),
                "model_evaluation_performed": False,
            }
            atomic_json(manifest_path, manifest)
        except FileExistsError as error:
            raise DataValidationError("另一个正式导出正在进行，拒绝并发覆盖") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
                lock_path.unlink(missing_ok=True)
        return {
            "output": str(output),
            "manifest": str(manifest_path),
            "report": str(report),
            "samples": 200,
        }
