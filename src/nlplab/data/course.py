"""课程语料计划、只读审计与数据质量报告；不依赖任何模型。"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse
from uuid import uuid4

import yaml

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256
from nlplab.data.corpus import duplicate_key, read_jsonl

DEFAULT_PLAN = Path(__file__).resolve().parents[3] / "configs/course_corpus.yaml"


def load_plan(path: Path = DEFAULT_PLAN) -> dict:
    try:
        plan = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = plan["primary_labels"]
        sampling = plan["sequence_gold_sampling"]
        assert sum(item["target"] for item in labels.values()) == plan["target_documents"]
        assert all(sum(v["source_type_quota"].values()) == v["target"] for v in labels.values())
        assert all(
            sum(v["source_type_quota"].get(kind, 0) for v in labels.values()) == target
            for kind, target in plan["source_type_totals"].items()
        )
        assert sampling["target_samples"] == 200
        assert sampling["freeze_policy"]["random_seed"] == 42
        assert sampling["max_samples_per_document"] == 1
        assert sampling["candidate_min_codepoints"] == 15
        assert sampling["candidate_max_codepoints"] == 80
        assert sum(sampling["primary_label_targets"].values()) == 200
        assert sum(sampling["source_type_targets"].values()) == 200
        assert set(sampling["primary_label_targets"]) == set(labels)
        assert set(sampling["source_type_targets"]) == set(plan["source_type_totals"])
        assert sampling["selection_must_not_use_model_error"] is True
        assert plan["annotation"]["entity_types"] == ["PER", "LOC", "ORG"]
        assert plan["annotation"]["span_indexing"] == "python_unicode_codepoint_half_open"
        assert plan["annotation"]["workflow_statuses"] == [
            "pending",
            "in_progress",
            "done",
            "reviewed",
        ]
        assert plan["annotation"]["final_export_requires_status"] == "reviewed"
        assert type(sampling["source_type_tolerance"]) is int
        assert sampling["source_type_tolerance"] >= 0
        assert sampling["source_type_tolerance"] <= 3
        targets = sampling["post_annotation_quality_targets"]
        assert targets["samples_with_any_entity_min"] >= 100
        assert targets["difficult_samples_min"] >= 40
        assert all(targets["entity_support_min"][kind] >= 40 for kind in ("PER", "LOC", "ORG"))
        assert plan["quality_control"]["review"]["minimum_second_pass_samples"] >= 40
    except (AssertionError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise DataValidationError(
            f"无效课程计划: {path}；需保持正式 200 条、seed=42 约束"
        ) from error
    return plan


def file_sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def quota(actual: dict, expected: dict, tolerance: int = 0) -> dict:
    return {
        key: {
            "actual": actual.get(key, 0),
            "target": target,
            "shortfall": max(0, target - tolerance - actual.get(key, 0)),
            "excess": max(0, actual.get(key, 0) - target - tolerance),
            "met": abs(actual.get(key, 0) - target) <= tolerance,
        }
        for key, target in expected.items()
    }


def audit_corpus(path: Path, plan: dict) -> dict:
    records = read_jsonl(path, lambda value: value)
    labels, types, domains, urls = Counter(), Counter(), Counter(), Counter()
    cross = defaultdict(Counter)
    exact, normalized, ids = {}, {}, {}
    duplicates, missing, invalid, lengths = [], [], [], []
    required = (
        "id",
        "text",
        "primary_label",
        "source_type",
        "source_title",
        "source_url",
        "sha256",
    )
    limits = plan["document_constraints"]
    for line, record in enumerate(records, 1):
        absent = [
            key
            for key in required
            if not isinstance(record.get(key), str) or not record[key].strip()
        ]
        if absent:
            missing.append({"line": line, "id": record.get("id"), "fields": absent})
        label = record.get("primary_label") or "<missing>"
        kind = record.get("source_type") or "<missing>"
        label = str(label)
        kind = str(kind)
        labels[label] += 1
        types[kind] += 1
        cross[label][kind] += 1
        issues = []
        text = record.get("text")
        if isinstance(text, str):
            lengths.append(len(text))
            if not limits["min_text_codepoints"] <= len(text) <= limits["max_text_codepoints"]:
                issues.append("text_length")
            digest, key = text_sha256(text), duplicate_key(text)
            if digest in exact or key in normalized:
                duplicates.append(
                    {
                        "line": line,
                        "duplicate_of_line": exact.get(digest, normalized.get(key)),
                        "exact": digest in exact,
                        "normalized": key in normalized,
                    }
                )
            exact.setdefault(digest, line)
            normalized.setdefault(key, line)
            if record.get("sha256") and record["sha256"] != digest:
                issues.append("sha256_mismatch")
        else:
            issues.append("invalid_text")
        identifier = record.get("id")
        if isinstance(identifier, str):
            if identifier in ids:
                issues.append("duplicate_id")
            ids[identifier] = line
        source_url = record.get("source_url")
        if isinstance(source_url, str):
            parsed = urlparse(source_url)
            domains[parsed.hostname or "<invalid>"] += 1
            urls[urldefrag(source_url)[0]] += 1
            if limits["require_http_url"] and not (
                parsed.scheme in {"http", "https"} and parsed.hostname
            ):
                issues.append("invalid_url")
        if label not in plan["primary_labels"]:
            issues.append("unknown_primary_label")
        if kind not in plan["source_type_totals"]:
            issues.append("unknown_source_type")
        if record.get("is_demo"):
            issues.append("demo")
        if issues:
            invalid.append({"line": line, "id": identifier, "issues": issues})
    label_quota = quota(labels, {k: v["target"] for k, v in plan["primary_labels"].items()})
    type_quota = quota(types, plan["source_type_totals"])
    cells = {
        label: quota(cross[label], values["source_type_quota"])
        for label, values in plan["primary_labels"].items()
    }
    excess_urls = {
        url: count
        for url, count in sorted(urls.items())
        if count > limits["one_url_may_supply_at_most"]
    }
    ordered = sorted(lengths)
    total = len(records)
    achieved = (
        total == plan["target_documents"]
        and not (duplicates or missing or invalid or excess_urls)
        and all(q["met"] for q in label_quota.values())
        and all(q["met"] for q in type_quota.values())
        and all(q["met"] for row in cells.values() for q in row.values())
    )
    return {
        "kind": "course_corpus_audit",
        "input_sha256": file_sha256(path),
        "total": total,
        "target": plan["target_documents"],
        "unique_normalized_texts": len(normalized),
        "primary_label_distribution": dict(sorted(labels.items())),
        "source_type_distribution": dict(sorted(types.items())),
        "cross_table": {
            label: dict(sorted(counts.items())) for label, counts in sorted(cross.items())
        },
        "length_codepoints": {
            "count": len(lengths),
            "min": min(lengths, default=None),
            "max": max(lengths, default=None),
            "mean": sum(lengths) / len(lengths) if lengths else None,
            "median": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
            if ordered
            else None,
            "values": lengths,
        },
        "domains": dict(sorted(domains.items())),
        "documents_per_url": dict(sorted(urls.items())),
        "url_limit_violations": excess_urls,
        "duplicates": duplicates,
        "missing_metadata": missing,
        "invalid_records": invalid,
        "quotas": {"labels": label_quota, "source_types": type_quota, "cross_table": cells},
        "quota_achieved": achieved,
        "original_data_modified": False,
    }


def save_quality_report(result: dict, base: Path, title: str) -> Path:
    directory = base / (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"# {title}", "", "全部长度/跨度使用 Python Unicode code point。审计不修改输入。", ""]
    for name in (
        "total",
        "target",
        "quota_achieved",
        "valid",
        "reviewed_count",
        "errors",
        "shortfalls",
    ):
        if name in result:
            lines += [
                f"## {name}",
                "",
                "```json",
                json.dumps(result[name], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
    for name in (
        "primary_label_distribution",
        "source_type_distribution",
        "cross_table",
        "length_codepoints",
        "domains",
        "documents_per_url",
        "url_limit_violations",
        "duplicates",
        "missing_metadata",
        "invalid_records",
        "quotas",
        "quality_targets",
        "status_distribution",
    ):
        if name in result:
            value = result[name]
            if name == "length_codepoints":
                value = {k: v for k, v in value.items() if k != "values"}
            lines += [
                f"## {name}",
                "",
                "```json",
                json.dumps(value, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return directory
