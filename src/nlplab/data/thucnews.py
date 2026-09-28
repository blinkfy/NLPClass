"""THUCNews 原文导入、5×120 个人选择及固定 84/18/18 划分。"""

import zipfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterator

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256
from nlplab.data.corpus import duplicate_key, iter_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.personal_selection import (
    personal_key,
    read_object,
    save_frozen_subset,
    selection_records,
    selection_score,
    source_binding,
    subset_manifest,
    utc_now,
    verify_import,
    verify_subset,
)
from nlplab.data.sequence_annotations import atomic_json

CATEGORY_MAP = {
    "财经": "finance",
    "教育": "education",
    "科技": "science",
    "体育": "sports",
    "娱乐": "entertainment",
    **{label: label for label in ("finance", "education", "science", "sports", "entertainment")},
}
CATEGORIES = ("finance", "education", "science", "sports", "entertainment")


def category_label(category: str) -> str | None:
    return CATEGORY_MAP.get(category)


def zip_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


def official_records(source: Path) -> Iterator[tuple[str, str, str]]:
    if source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            yield path.relative_to(source).as_posix(), path.parent.name, path.read_text("utf-8")
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in sorted(archive.infolist(), key=zip_name):
                name = zip_name(info)
                if (
                    info.is_dir()
                    or not name.endswith(".txt")
                    or name.startswith("__MACOSX/")
                    or Path(name).name.startswith("._")
                ):
                    continue
                parts = Path(name).parts
                if len(parts) < 2 or ".." in parts or Path(name).is_absolute():
                    raise DataValidationError(f"归档文件路径不合法: {name}")
                yield (
                    name,
                    parts[-2],
                    archive.read(info).decode("utf-8-sig") if category_label(parts[-2]) else "",
                )
    else:
        raise DataValidationError("官方 THUCNews 需要分类目录或官方 ZIP 全文归档")


def cnews_records(source: Path) -> Iterator[tuple[str, str, str]]:
    paths = sorted(source.glob("cnews.*.txt")) if source.is_dir() else [source]
    if not paths:
        raise DataValidationError("未找到 cnews.*.txt")
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) != 2:
                    raise DataValidationError(f"{path}:{number}: cnews 需要 category TAB fulltext")
                yield f"{path.name}:{number:06d}", parts[0], parts[1]


def import_thucnews(source: Path, output: Path, provenance: dict) -> dict:
    required = {"variant", "version", "source_url", "downloaded_at", "license", "usage_note"}
    if not required <= provenance.keys():
        raise DataValidationError(
            f"THUCNews provenance 缺字段: {sorted(required - provenance.keys())}"
        )
    variant = provenance["variant"]
    if variant not in {"thucnews-official-14", "cnews-fulltext-10"}:
        raise DataValidationError("只支持官方全文或 provenance 核验的 cnews-fulltext-10")
    if variant == "cnews-fulltext-10" and not provenance.get("derivation_evidence"):
        raise DataValidationError("cnews fallback 必须提供可核验的全文来源 derivation_evidence")
    files = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    hashes = {
        path.relative_to(source).as_posix() if source.is_dir() else path.name: file_sha256(path)
        for path in files
    }
    expected = provenance.get("source_hashes")
    if expected and expected != hashes:
        raise DataValidationError("THUCNews source hash/version 不匹配")
    if not source.is_dir() and provenance.get("source_file_sha256") not in {
        None,
        hashes[source.name],
    }:
        raise DataValidationError("THUCNews 归档 source SHA256 不匹配")
    manifest_path = subset_manifest(output)
    if output.exists() or manifest_path.exists():
        old = verify_import(output, manifest_path)
        if (
            old["source_hashes"] != hashes
            or old["dataset_version"] != provenance["version"]
            or old["provenance"] != provenance
        ):
            raise DataValidationError("THUCNews 来源改变，拒绝覆盖")
        return old
    counts, lengths, ids = Counter(), Counter(), set()
    iterator = official_records if variant == "thucnews-official-14" else cnews_records
    imported = 0

    def records() -> Iterator[dict]:
        nonlocal imported
        for record_id, category, text in iterator(source):
            counts[category] += 1
            label = category_label(category)
            if label is None:
                continue
            if record_id in ids:
                raise DataValidationError("THUCNews source ID 重复")
            ids.add(record_id)
            lengths[len(text)] += 1
            imported += 1
            yield {
                "id": "THU-" + text_sha256(record_id)[:24],
                "text": text,
                "primary_label": label,
                "source_dataset": "THUCNews",
                "source_variant": variant,
                "source_category": category,
                "source_record_id": record_id,
                "source_url": provenance["source_url"],
                "sha256": text_sha256(text),
                "dataset_version": provenance["version"],
                "source_hashes": hashes,
            }

    temporary = output.with_suffix(".importing.jsonl")
    try:
        write_jsonl(temporary, records())
        if not imported:
            raise DataValidationError("五类全文候选为空")
        middle = median(list(lengths.elements()))
        if middle < 80:
            raise DataValidationError(f"声明 fulltext 的来源中位长度仅 {middle}；拒绝标题版")
        manifest = {
            "dataset_name": "THUCNews",
            "dataset_variant": variant,
            "dataset_version": provenance["version"],
            "source_hashes": hashes,
            "provenance": provenance,
            "imported_at": utc_now(),
            "source_category_counts": dict(sorted(counts.items())),
            "imported_categories": list(CATEGORIES),
            "count": imported,
            "length_median": middle,
            "data_sha256": file_sha256(temporary),
            "raw_text_preserved": True,
        }
        temporary.replace(output)
        atomic_json(manifest_path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def select_thucnews(source: Path, output: Path, key_path: Path, min_length: int = 80) -> dict:
    if min_length < 80:
        raise DataValidationError("全文选择 min_length 不能低于 80 code point")
    imported = verify_import(source, subset_manifest(source))
    if imported["dataset_name"] != "THUCNews":
        raise DataValidationError("来源不是 THUCNews")
    key, _ = personal_key(
        key_path, "thucnews-600", {**source_binding(imported), "min_length": min_length}
    )
    representatives, excluded, ids = {}, Counter(), set()
    # 先确定每个 normalized duplicate 的稳定代表；跨类重复也只保留一个。
    for row in iter_jsonl(source, lambda row: row):
        record_id = row["source_record_id"]
        if record_id in ids:
            raise DataValidationError("THUCNews source IDs 重复")
        ids.add(record_id)
        if row["primary_label"] not in CATEGORIES:
            raise DataValidationError("THUCNews 候选存在非目标类别")
        if not row["text"].strip():
            excluded["empty"] += 1
            continue
        if len(row["text"]) < min_length:
            excluded["short"] += 1
            continue
        normalized_hash = text_sha256(duplicate_key(row["text"]))
        rank = (selection_score(imported["dataset_version"], key, record_id), record_id)
        if normalized_hash in representatives:
            excluded["normalized_duplicate"] += 1
        if normalized_hash not in representatives or rank < representatives[normalized_hash]:
            representatives[normalized_hash] = rank
    bins = {label: [] for label in CATEGORIES}
    eligible = Counter()
    for row in iter_jsonl(source, lambda row: row):
        rank = (
            selection_score(imported["dataset_version"], key, row["source_record_id"]),
            row["source_record_id"],
        )
        normalized_hash = text_sha256(duplicate_key(row["text"]))
        if rank != representatives.get(normalized_hash):
            continue
        label = row["primary_label"]
        eligible[label] += 1
        # 最多持有每类 120 条完整文本，归档全文不整体加载到内存。
        candidates = bins[label]
        candidates.append({**row, "selection_score": rank[0]})
        if len(candidates) > 120:
            candidates.sort(key=lambda item: (item["selection_score"], item["source_record_id"]))
            candidates.pop()
    shortages = {label: 120 - len(bins[label]) for label in CATEGORIES if len(bins[label]) < 120}
    if shortages:
        raise DataValidationError(f"全文配额不足，不放宽规则: {shortages}")
    rows = [
        row
        for label in CATEGORIES
        for row in sorted(
            bins[label], key=lambda item: (item["selection_score"], item["source_record_id"])
        )
    ]
    return save_frozen_subset(
        output,
        rows,
        {
            "dataset_name": "THUCNews",
            "dataset_variant": imported["dataset_variant"],
            "dataset_version": imported["dataset_version"],
            "source_hashes": imported["source_hashes"],
            "import_data_sha256": imported["data_sha256"],
            "provenance": imported["provenance"],
            "selection_key_fingerprint": text_sha256(key),
            "min_length": min_length,
            "selection_method": "SHA256(dataset_version + selection_key + stable_record_id)",
            "selection_uses_model_predictions": False,
            "count": 600,
            "category_counts": {label: 120 for label in CATEGORIES},
            "eligible_counts": dict(sorted(eligible.items())),
            "excluded": dict(excluded),
            "selected_records": selection_records(rows),
        },
    )


def clustering_view(rows: list[dict]) -> list[dict]:
    """算法输入视图仅含 ID/text；primary_label 只能单独用于结果评价。"""
    return [{"id": row["id"], "text": row["text"]} for row in rows]


def split_thucnews(source: Path, output: Path, key_path: Path) -> dict:
    rows, selected = verify_subset(source, dataset="THUCNews", count=600)
    if Counter(row["primary_label"] for row in rows) != Counter(
        {label: 120 for label in CATEGORIES}
    ):
        raise DataValidationError("冻结 600 条必须严格每类 120")
    binding = {
        "dataset_version": selected["dataset_version"],
        "subset_sha256": selected["data_sha256"],
    }
    key, _ = personal_key(key_path, "thucnews-split-84-18-18", binding)
    split_rows = {split: [] for split in ("train", "val", "test")}
    for label in CATEGORIES:
        ranked = sorted(
            (row for row in rows if row["primary_label"] == label),
            key=lambda row: (
                selection_score(selected["dataset_version"], key, row["source_record_id"]),
                row["source_record_id"],
            ),
        )
        for split, start, end in (("train", 0, 84), ("val", 84, 102), ("test", 102, 120)):
            split_rows[split].extend(ranked[start:end])
    manifest = {
        "dataset_name": "THUCNews",
        **binding,
        "split_key_fingerprint": text_sha256(key),
        "selection_manifest_sha256": file_sha256(subset_manifest(source)),
        "method": "per-category SHA256(dataset_version + split_key + stable_record_id)",
        "splits": {
            split: {
                "count": len(items),
                "category_counts": dict(Counter(row["primary_label"] for row in items)),
                "ids": [row["id"] for row in items],
            }
            for split, items in split_rows.items()
        },
        "clustering_input": "clustering_all.jsonl contains only id/text; use all 600",
    }
    path = output / "manifest.json"
    if path.exists():
        old = read_object(path)
        if any(old.get(key) != value for key, value in manifest.items()):
            raise DataValidationError("固定 split 清单与此次来源/键不一致，拒绝覆盖")
        for name, digest in old["file_sha256"].items():
            if file_sha256(output / name) != digest:
                raise DataValidationError("冻结 split 文件已修改")
        return old
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DataValidationError("split 目录非空，拒绝覆盖")
    for split, items in split_rows.items():
        write_jsonl(output / f"{split}.jsonl", items)
    write_jsonl(output / "clustering_all.jsonl", clustering_view(rows))
    manifest["file_sha256"] = {file.name: file_sha256(file) for file in output.glob("*.jsonl")}
    manifest["frozen_at"] = utc_now()
    atomic_json(path, manifest)
    return manifest
