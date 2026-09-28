"""公开语料的持久个人键、稳定哈希选择和冻结清单；不接收模型预测。"""

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256
from nlplab.data.corpus import iter_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selection_score(version: str, key: str, record_id: str) -> str:
    return text_sha256(version + key + record_id)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataValidationError(f"清单无法读取: {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"清单不是 object: {path}")
    return value


def personal_key(path: Path, purpose: str, binding: dict) -> tuple[str, dict]:
    """首次原子创建；已有 purpose 必须与来源绑定一致，不能重抽逃避低分。"""
    value = read_object(path) if path.exists() else {"schema_version": 1, "selections": {}}
    if purpose in value["selections"]:
        entry = value["selections"][purpose]
        if entry["binding"] != binding:
            raise DataValidationError("selection key 的 source hash/version 绑定不一致")
    else:
        entry = {
            "key": secrets.token_hex(16),
            "binding": binding,
            "created_at": utc_now(),
            "rejected_selections": [],
        }
        value["selections"][purpose] = entry
        atomic_json(path, value)
    return entry["key"], value


def source_binding(manifest: dict) -> dict:
    return {
        "dataset_version": manifest["dataset_version"],
        "source_hashes": manifest["source_hashes"],
        "data_sha256": manifest["data_sha256"],
    }


def verify_import(path: Path, manifest_path: Path) -> dict:
    manifest = read_object(manifest_path)
    if file_sha256(path) != manifest["data_sha256"]:
        raise DataValidationError("数据 SHA256 与导入清单不符")
    for record in iter_jsonl(path, lambda row: row):
        if (
            record.get("dataset_version") != manifest["dataset_version"]
            or record.get("source_hashes") != manifest["source_hashes"]
            or record.get("sha256") != text_sha256(record["text"])
        ):
            raise DataValidationError("record 的 source hash/version/text SHA256 与清单不符")
    return manifest


def subset_manifest(path: Path) -> Path:
    return path.with_name(path.stem + "_manifest.json")


def save_frozen_subset(path: Path, rows: list[dict], manifest: dict) -> dict:
    """相同绑定重复执行只验证旧结果；部分输出或不同来源拒绝覆盖。"""
    manifest_path = subset_manifest(path)
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise DataValidationError("冻结输出不完整，拒绝覆盖；请使用新的输出路径")
        old = read_object(manifest_path)
        if any(old.get(key) != manifest.get(key) for key in manifest):
            raise DataValidationError("已有 selection manifest 与此次选择不一致，拒绝覆盖")
        if file_sha256(path) != old["data_sha256"]:
            raise DataValidationError("冻结子集文件被修改")
        return old
    write_jsonl(path, rows)
    manifest = {**manifest, "frozen_at": utc_now(), "data_sha256": file_sha256(path)}
    atomic_json(manifest_path, manifest)
    return manifest


def verify_subset(path: Path, *, dataset: str, count: int) -> tuple[list[dict], dict]:
    manifest = read_object(subset_manifest(path))
    if manifest.get("dataset_name") != dataset or file_sha256(path) != manifest["data_sha256"]:
        raise DataValidationError("冻结子集数据集或 SHA256 不匹配")
    rows = list(iter_jsonl(path, lambda row: row))
    ids = [row["source_record_id"] for row in rows]
    if len(rows) != count or len(set(ids)) != count:
        raise DataValidationError(f"子集必须恰好 {count} 条唯一 source record")
    own_ids = [row["sample_id" if dataset == "MSRA-NER" else "id"] for row in rows]
    if len(set(own_ids)) != count or any(row["source_dataset"] != dataset for row in rows):
        raise DataValidationError("子集 ID 重复或 source dataset 不一致")
    selected = manifest["selected_records"]
    if ids != [row["source_record_id"] for row in selected]:
        raise DataValidationError("冻结 selected IDs 与文件不一致")
    for row, saved in zip(rows, selected):
        if (
            row["sha256"] != text_sha256(row["text"])
            or row["sha256"] != saved["text_sha256"]
            or row["dataset_version"] != manifest["dataset_version"]
            or row["source_hashes"] != manifest["source_hashes"]
            or row["selection_score"] != saved["selection_score"]
        ):
            raise DataValidationError("冻结 record 指纹、来源版本或 selection score 不一致")
    return rows, manifest


def selection_records(rows: list[dict]) -> list[dict]:
    return [
        {
            "source_record_id": row["source_record_id"],
            "text_sha256": row["sha256"],
            "selection_score": row["selection_score"],
        }
        for row in rows
    ]
