"""固定 MSRA BIO 导入及 test-only 个人 200 子集；不提供 CWS Gold。"""

import secrets
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Entity, text_sha256
from nlplab.data.corpus import read_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.personal_selection import (
    personal_key,
    read_object,
    save_frozen_subset,
    selection_records,
    selection_score,
    source_binding,
    utc_now,
    verify_import,
)
from nlplab.data.sequence_annotations import atomic_json

LABELS = {"PER", "LOC", "ORG"}


def bio_entities(chars: list[str], tags: list[str]) -> list[Entity]:
    """严格 BIO2；I 不可开始实体，也不可接在不同类型后。"""
    if not chars or len(chars) != len(tags) or any(len(char) != 1 for char in chars):
        raise DataValidationError("BIO 必须每个 Unicode code point 对应一个标签")
    text = "".join(chars)
    entities, active, start = [], None, 0
    for index, tag in enumerate([*tags, "O"]):
        prefix, label = (
            ("O", None) if tag == "O" else tuple(tag.split("-", 1)) if "-" in tag else ("?", None)
        )
        if (
            prefix not in {"O", "B", "I"}
            or (prefix == "O" and label is not None)
            or (prefix != "O" and label not in LABELS)
        ):
            raise DataValidationError(f"非法 BIO 标签: {tag!r} @ {index}")
        if prefix == "I" and active != label:
            raise DataValidationError(f"非法 BIO I 标签: {tag!r} @ {index}")
        if active is not None and prefix != "I":
            entity = Entity(text[start:index], active, start, index)
            entity.validate(text)
            entities.append(entity)
            active = None
        if prefix == "B":
            active, start = label, index
    return entities


def read_bio(path: Path, split: str, version: str, hashes: dict) -> list[dict]:
    rows, chars, tags = [], [], []

    def finish() -> None:
        if not chars:
            return
        record_id = f"{split}:{len(rows):06d}"
        try:
            entities = bio_entities(chars, tags)
        except DataValidationError as error:
            raise DataValidationError(f"{path}: record {record_id}: {error}") from error
        text = "".join(chars)
        rows.append(
            {
                "sample_id": f"MSRA-{record_id.replace(':', '-')}",
                "source_dataset": "MSRA-NER",
                "source_split": split,
                "source_record_id": record_id,
                "dataset_version": version,
                "source_hashes": hashes,
                "text": text,
                "sha256": text_sha256(text),
                "ner_gold": [asdict(entity) for entity in entities],
            }
        )
        chars.clear()
        tags.clear()

    # 只按物理 LF/CRLF 行边界解析，Unicode 分隔符不得充当文件行边界。
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.rstrip("\r\n")
            if not value:
                finish()
                continue
            parts = value.split("\t")
            if len(parts) != 2 or len(parts[0]) != 1:
                raise DataValidationError(f"{path}:{line_number}: 需要 character TAB BIO tag")
            chars.append(parts[0])
            tags.append(parts[1])
    finish()
    if not rows:
        raise DataValidationError("BIO 数据为空")
    return rows


def prepare_msra(root: Path, config: dict, source: Path | None = None) -> dict:
    target = root / "data/external/msra_ner"
    target.mkdir(parents=True, exist_ok=True)
    inputs, sources = {}, {}
    for split, item in config["sources"].items():
        path = (source or target) / f"msra_{split}_bio.txt"
        if not path.exists() and source is None:
            temporary = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(item["url"], timeout=90) as response:
                    temporary.write_bytes(response.read())
                if file_sha256(temporary) != item["sha256"]:
                    raise DataValidationError(f"下载的 {split} source SHA256 不匹配")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha256(path) != item["sha256"]:
            raise DataValidationError(f"{split} source SHA256/version 不匹配")
        inputs[split] = path
        sources[split] = {**item, "path": str(path.resolve())}
    hashes = {split: value["sha256"] for split, value in sources.items()}
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        old = read_object(manifest_path)
        if old["dataset_version"] != config["version"] or old["source_hashes"] != hashes:
            raise DataValidationError("MSRA 导入版本/源哈希改变，拒绝覆盖")
        for split, item in old["splits"].items():
            if item["validation_status"] != "valid":
                continue
            verify_import(target / f"{split}.jsonl", target / f"{split}_manifest.json")
            if file_sha256(target / f"{split}.jsonl") != item["data_sha256"]:
                raise DataValidationError("已导入 MSRA 被修改")
        return old
    splits = {}
    rows_by_split, errors = {}, {}
    for split, path in inputs.items():
        # train 只记录规模/校验状态；非法 split 整体拒绝，不补 O、不跳过记录。
        count, active = 0, False
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.rstrip("\r\n"):
                    active = True
                elif active:
                    count += 1
                    active = False
        count += int(active)
        try:
            rows_by_split[split] = read_bio(path, split, config["version"], hashes)
        except DataValidationError as error:
            errors[split] = str(error)
            splits[split] = {"count": count, "validation_status": "rejected", "error": str(error)}
    if errors:
        atomic_json(target / "bio_import_error.json", {"at": utc_now(), "errors": errors})
    if "test" in errors:
        raise DataValidationError(errors["test"])
    provenance = {
        "dataset_name": "MSRA-NER",
        "dataset_version": config["version"],
        "variant": "BIO2-PER-ORG-LOC",
        "sources": sources,
        "source_hashes": hashes,
        "script_url": config["script_url"],
        "downloaded_at": utc_now(),
        "license": config["license"],
        "usage_note": config["usage_note"],
    }
    for split, rows in rows_by_split.items():
        path = target / f"{split}.jsonl"
        if path.exists():
            raise DataValidationError("MSRA 部分输出已存在，拒绝覆盖")
        write_jsonl(path, rows)
        splits[split] = {
            "count": len(rows),
            "data_sha256": file_sha256(path),
            "validation_status": "valid",
        }
        atomic_json(
            target / f"{split}_manifest.json", {**provenance, **splits[split], "split": split}
        )
    manifest = {**provenance, "splits": splits}
    atomic_json(manifest_path, manifest)
    return manifest


def gold_support(rows: list[dict]) -> dict[str, int]:
    counts = Counter(entity["type"] for row in rows for entity in row["ner_gold"])
    return {label: counts[label] for label in sorted(LABELS)}


def select_exp1(source: Path, output: Path, key_path: Path) -> dict:
    imported = verify_import(source, source.with_name(source.stem + "_manifest.json"))
    if imported["dataset_name"] != "MSRA-NER" or imported["split"] != "test":
        raise DataValidationError("正式 200 条只允许来自 MSRA test")
    rows = read_jsonl(source, lambda row: row)
    if len(rows) < 200 or len({row["source_record_id"] for row in rows}) != len(rows):
        raise DataValidationError("MSRA test 不足 200 或 source IDs 重复")
    binding = source_binding(imported)
    key, keys = personal_key(key_path, "exp1-msra-test-200", binding)
    entry = keys["selections"]["exp1-msra-test-200"]
    for attempt in range(2):
        selected = sorted(
            [
                {
                    **row,
                    "selection_score": selection_score(
                        imported["dataset_version"], key, row["source_record_id"]
                    ),
                }
                for row in rows
            ],
            key=lambda row: (row["selection_score"], row["source_record_id"]),
        )[:200]
        support = gold_support(selected)
        if all(support.values()):
            break
        if attempt or entry["rejected_selections"]:
            raise DataValidationError("一次重新生成后仍有实体类别无支持；停止，不进行模型推理")
        entry["rejected_selections"].append(
            {
                "key_fingerprint": text_sha256(key),
                "reason": "Gold class has zero support before inference",
                "gold_support": support,
            }
        )
        key = entry["key"] = secrets.token_hex(16)
        atomic_json(key_path, keys)
    manifest = {
        "dataset_name": "MSRA-NER",
        "source_split": "test",
        **binding,
        "provenance": imported,
        "selection_key_fingerprint": text_sha256(key),
        "rejected_selections": entry["rejected_selections"],
        "selection_method": "SHA256(dataset_version + selection_key + stable_record_id)",
        "selection_uses_model_predictions": False,
        "count": 200,
        "gold_support": support,
        "selected_records": selection_records(selected),
    }
    # binding.data_sha256 是导入指纹；子集指纹在 save 中生成，单独保存来源。
    manifest["import_data_sha256"] = manifest.pop("data_sha256")
    return save_frozen_subset(output, selected, manifest)
