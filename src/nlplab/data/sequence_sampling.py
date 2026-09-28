"""可复现的原文切句和双边配额抽样；完全不使用模型输出。"""

import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.data.corpus import duplicate_key, load_documents, write_jsonl
from nlplab.data.course import file_sha256, quota


def sentence_spans(text: str) -> list[tuple[int, int]]:
    pairs = {"“": "”", "‘": "’", "「": "」", "『": "』", '"': '"', "'": "'"}
    closers = set(pairs.values()) | set("）)]】》")
    terminals = set("。！？!?；;")
    stack, spans = [], []
    start = i = 0
    while i < len(text):
        char = text[i]
        apostrophe = (
            char == "'"
            and 0 < i < len(text) - 1
            and text[i - 1].isascii()
            and text[i - 1].isalnum()
            and text[i + 1].isascii()
            and text[i + 1].isalnum()
        )
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs and not apostrophe:
            stack.append(pairs[char])
        period = char == "." and (i + 1 == len(text) or text[i + 1].isspace())
        boundary = not stack and (char in terminals or char in "\r\n" or period)
        if char in closers and not stack:
            j = i - 1
            while j >= start and text[j] in closers:
                j -= 1
            boundary = boundary or (j >= start and text[j] in terminals)
        if boundary:
            end = i + 1
            while end < len(text) and text[end] in closers:
                end += 1
            if char == "\r" and end < len(text) and text[end] == "\n":
                end += 1
            spans.append((start, end))
            start = end
            i = end
        else:
            i += 1
    if start < len(text):
        spans.append((start, len(text)))
    assert "".join(text[a:b] for a, b in spans) == text
    return spans


def difficult_features(text: str) -> list[str]:
    flags = {
        "number": any(c.isdigit() for c in text),
        "ascii": any(ord(c) < 128 and not c.isspace() for c in text),
        "fullwidth": any("\uff01" <= c <= "\uff60" or "\uffe0" <= c <= "\uffe6" for c in text),
        "quotation": any(c in "“”‘’「」『』\"'" for c in text),
        "mixed_chinese_english": bool(
            re.search(r"[\u3400-\u9fff]", text) and re.search(r"[A-Za-z]", text)
        ),
        "date_or_percentage": bool(
            re.search(
                r"[0-9０-９]{2,4}年|[0-9０-９]+月[0-9０-９]+日|[0-9０-９]{1,4}[-/][0-9０-９]{1,2}[-/][0-9０-９]{1,2}|[0-9０-９.．]+[%％]",
                text,
            )
        ),
    }
    return [name for name, detected in flags.items() if detected]


def allocate(cells: dict, label_targets: dict, type_targets: dict, tolerance: int) -> dict | None:
    """有下界的整数流，保证两个分层边际和每格实际文档容量。"""
    graph = defaultdict(dict)
    demand = Counter()
    edges = {}

    def edge(left, right, lower, upper):
        if lower > upper:
            return False
        graph[left][right] = upper - lower
        graph[right].setdefault(left, 0)
        demand[left] -= lower
        demand[right] += lower
        return True

    total = sum(label_targets.values())
    for label, count in label_targets.items():
        edge("source", "label:" + label, count, count)
        for kind in type_targets:
            a, b = "label:" + label, "type:" + kind
            capacity = cells.get((label, kind), 0)
            edge(a, b, 0, capacity)
            edges[label, kind] = (a, b, capacity)
    for kind, count in type_targets.items():
        edge("type:" + kind, "sink", max(0, count - tolerance), count + tolerance)
    edge("sink", "source", total, total)
    required = 0
    for node, count in list(demand.items()):
        if count > 0:
            edge("super_source", node, 0, count)
            required += count
        elif count < 0:
            edge(node, "super_sink", 0, -count)
    flow = 0
    while True:
        previous = {"super_source": None}
        queue = deque(["super_source"])
        while queue and "super_sink" not in previous:
            node = queue.popleft()
            for neighbor, capacity in graph[node].items():
                if capacity > 0 and neighbor not in previous:
                    previous[neighbor] = node
                    queue.append(neighbor)
        if "super_sink" not in previous:
            break
        amount, node = required, "super_sink"
        while previous[node] is not None:
            amount = min(amount, graph[previous[node]][node])
            node = previous[node]
        node = "super_sink"
        while previous[node] is not None:
            parent = previous[node]
            graph[parent][node] -= amount
            graph[node][parent] += amount
            node = parent
        flow += amount
    if flow != required:
        return None
    return {key: capacity - graph[a][b] for key, (a, b, capacity) in edges.items()}


def sample_candidates(
    corpus: Path, plan: dict, *, allow_demo: bool = False
) -> tuple[list[dict], dict]:
    documents = load_documents(corpus)
    sampling = plan["sequence_gold_sampling"]
    seed = sampling["freeze_policy"]["random_seed"]
    rng = random.Random(seed)
    pools, excluded = defaultdict(list), []
    seen_ids, seen_text = set(), set()
    for doc in sorted(documents, key=lambda d: d.id):
        if doc.id in seen_ids:
            raise DataValidationError(f"重复 doc_id: {doc.id}")
        key = duplicate_key(doc.text)
        if key in seen_text:
            raise DataValidationError("主语料包含重复文本；先核对审计结果并导入去重语料")
        seen_ids.add(doc.id)
        seen_text.add(key)
        reasons = []
        if doc.is_demo and not allow_demo:
            reasons.append("demo_not_allowed")
        if (
            doc.primary_label not in sampling["primary_label_targets"]
            or doc.source_type not in sampling["source_type_targets"]
        ):
            reasons.append("metadata_outside_plan")
        limits = plan["document_constraints"]
        if not limits["min_text_codepoints"] <= len(doc.text) <= limits["max_text_codepoints"]:
            reasons.append("document_length_outside_plan")
        candidates = [
            (a, b)
            for a, b in sentence_spans(doc.text)
            if sampling["candidate_min_codepoints"] <= b - a <= sampling["candidate_max_codepoints"]
        ]
        if not candidates:
            reasons.append("no_sentence_in_length_range")
        if reasons:
            excluded.append({"doc_id": doc.id, "reasons": reasons})
            continue
        # 同文档先在表面特征最丰富的句中随机选择，不使用模型质量。
        best = max(len(difficult_features(doc.text[a:b])) for a, b in candidates)
        options = [(a, b) for a, b in candidates if len(difficult_features(doc.text[a:b])) == best]
        a, b = rng.choice(options)
        pools[doc.primary_label, doc.source_type].append((doc, a, b))
    capacities = {key: len(value) for key, value in pools.items()}
    labels, types = sampling["primary_label_targets"], sampling["source_type_targets"]
    allocation = allocate(capacities, labels, types, 0)
    used_tolerance = 0
    if allocation is None:
        used_tolerance = sampling["source_type_tolerance"]
        allocation = allocate(capacities, labels, types, used_tolerance)
    eligible_labels = Counter()
    eligible_types = Counter()
    for (label, kind), count in capacities.items():
        eligible_labels[label] += count
        eligible_types[kind] += count
    manifest = {
        "version": 1,
        "algorithm": "metadata-surface-lower-bound-flow-v1",
        "seed": seed,
        "corpus_sha256": file_sha256(corpus),
        "target": sampling["target_samples"],
        "eligible_documents": sum(capacities.values()),
        "excluded_documents": excluded,
        "eligible_cross_table": {
            label: {kind: capacities.get((label, kind), 0) for kind in types} for label in labels
        },
        "uses_model_predictions": False,
        "allow_demo": allow_demo,
        "source_type_tolerance_used": used_tolerance,
        "success": allocation is not None,
    }
    if allocation is None:
        manifest["shortfalls"] = {
            "labels": quota(eligible_labels, labels),
            "source_types": quota(eligible_types, types, sampling["source_type_tolerance"]),
            "joint_quota_feasible": False,
            "message": "联合配额不可满足；检查 eligible_cross_table。未放宽长度或配额。",
        }
        return [], manifest
    selected = []
    for cell, count in allocation.items():
        pool = pools[cell][:]
        rng.shuffle(pool)
        selected.extend(pool[:count])
    result = []
    for number, (doc, a, b) in enumerate(sorted(selected, key=lambda item: item[0].id), 1):
        text = doc.text[a:b]
        result.append(
            {
                "sample_id": f"SEQ{number:05d}",
                "doc_id": doc.id,
                "text": text,
                "source_start": a,
                "source_end": b,
                "primary_label": doc.primary_label,
                "source_type": doc.source_type,
                "is_demo": doc.is_demo,
                "status": "pending",
                "gold_tokens": None,
                "entities": None,
                "revision_count": 0,
                "notes": "",
                "selection_reasons": [
                    f"seed:{seed}",
                    f"label:{doc.primary_label}",
                    f"source_type:{doc.source_type}",
                    "length:15-80-codepoints",
                    "one-per-document",
                    *difficult_features(text),
                ],
            }
        )
    assert len(result) == sampling["target_samples"] and len({r["doc_id"] for r in result}) == len(
        result
    )
    actual_types = Counter(r["source_type"] for r in result)
    manifest.update(
        {
            "selected_count": len(result),
            "primary_label_distribution": dict(Counter(r["primary_label"] for r in result)),
            "source_type_distribution": dict(actual_types),
            "source_type_deviations": {
                kind: actual_types[kind] - count for kind, count in types.items()
            },
            "difficult_samples": sum(bool(difficult_features(r["text"])) for r in result),
            "selection": [
                {
                    "sample_id": r["sample_id"],
                    "doc_id": r["doc_id"],
                    "source_start": r["source_start"],
                    "source_end": r["source_end"],
                }
                for r in result
            ],
            "document_fingerprints": {doc.id: doc.sha256 for doc, _, _ in selected},
        }
    )
    return result, manifest


def save_candidates(output: Path, samples: list[dict], manifest: dict) -> Path:
    import json

    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise DataValidationError("候选或 sampling manifest 已存在，拒绝覆盖；请使用新路径")
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest["success"]:
        write_jsonl(output, samples)
        manifest["candidate_sha256"] = file_sha256(output)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path
