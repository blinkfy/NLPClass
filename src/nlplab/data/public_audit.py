"""公开数据来源与冻结子集/split 的只读审计。"""

import json
from collections import Counter
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.data.course import file_sha256, save_quality_report
from nlplab.data.personal_selection import read_object, verify_subset


def audit_public(root: Path) -> dict:
    result = {"datasets": {}}
    msra = root / "data/course/exp1_msra_200.jsonl"
    thucnews = root / "data/course/thucnews_600.jsonl"
    for name, path, count in (("MSRA-NER", msra, 200), ("THUCNews", thucnews, 600)):
        if not path.exists():
            result["datasets"][name] = {"status": "waiting for dataset", "expected_path": str(path)}
            continue
        rows, manifest = verify_subset(path, dataset=name, count=count)
        item = {"status": "ready", "manifest": manifest}
        if name == "THUCNews":
            item["category_counts"] = dict(Counter(row["primary_label"] for row in rows))
            item["lengths"] = {
                "min": min(len(row["text"]) for row in rows),
                "max": max(len(row["text"]) for row in rows),
            }
            split = root / "data/splits/thucnews_600/manifest.json"
            if split.exists():
                split_manifest = read_object(split)
                if split_manifest["subset_sha256"] != file_sha256(path):
                    raise DataValidationError("split 与 600 条子集指纹不符")
                for filename, digest in split_manifest["file_sha256"].items():
                    if file_sha256(split.parent / filename) != digest:
                        raise DataValidationError("split 文件被修改")
                item["split_manifest"] = split_manifest
            else:
                item["split_status"] = "not prepared"
        result["datasets"][name] = item
    source_manifest = root / "data/external/msra_ner/manifest.json"
    if source_manifest.exists():
        result["msra_import"] = read_object(source_manifest)
    return result


def save_public_audit(root: Path) -> dict:
    result = audit_public(root)
    report = save_quality_report(
        result, root / "outputs/data_quality", "公共数据集来源与冻结子集审计"
    )
    lines = [
        "# 公共数据集审计",
        "",
        "只读检查来源、版本、SHA256、selected IDs 与固定 split；不运行模型。",
        "",
    ]
    for name, item in result["datasets"].items():
        lines.extend([f"## {name}", "", f"状态：{item['status']}", ""])
        if item["status"] != "ready":
            lines.append(f"预期位置：{item['expected_path']}")
            continue
        manifest = item["manifest"]
        lines.extend(
            [
                f"版本：{manifest['dataset_version']}",
                f"数量：{manifest['count']}",
                f"文件 SHA256：{manifest['data_sha256']}",
                "",
                "来源/许可：",
                "```json",
                json.dumps(manifest["provenance"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        if name == "MSRA-NER":
            lines.append(f"split=test；Gold 实体 support：{manifest['gold_support']}")
            lines.append("没有 CWS Gold，不计算课程 200 条 CWS F1。")
        else:
            lines.append(f"类别数量：{item['category_counts']}；长度 code point：{item['lengths']}")
            if "split_manifest" in item:
                lines.append(
                    "split："
                    + str(
                        {
                            key: value["count"]
                            for key, value in item["split_manifest"]["splits"].items()
                        }
                    )
                )
            lines.append(
                "K-means 输入视图只有 id/text，实验二使用全部 600。尚未训练聚类/分类模型。"
            )
        lines.append("")
    if "msra_import" in result:
        lines.extend(
            [
                "## MSRA 原始 split 校验",
                "",
                "```json",
                json.dumps(result["msra_import"]["splits"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    (report / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return {
        "report": str(report),
        "datasets": {name: value["status"] for name, value in result["datasets"].items()},
    }
