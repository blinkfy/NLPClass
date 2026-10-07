"""Read-only workspace selection plans and original-text contextual offsets."""

import copy
from collections import Counter
from hashlib import sha256

from nlplab.core.exceptions import DataValidationError
from nlplab.comprehensive.contracts import validate_annotations

SAMPLING_SEED = "lab4-cluster-balanced-v1"


def sampling_plan(collection, scope_ids=None, limit=10):
    if type(limit) is not int or not 1 <= limit <= 10:
        raise DataValidationError("每批选择1–10篇")
    docs = collection["documents"]
    known = {doc["id"] for doc in docs}
    scope = known if scope_ids is None else set(scope_ids)
    if not scope <= known:
        raise DataValidationError("选择包含语料集合之外的文档")
    groups, completed, counts = {}, Counter(), {}
    analyses = collection.get("analyses", {})
    for doc in docs:
        cid = doc["cluster_id"]
        counts.setdefault(cid, {"cluster_id": cid, "total": 0, "eligible": 0, "analyzed": 0,
                                "scope_eligible": 0, "scope_analyzed": 0})
        counts[cid]["total"] += 1
        if not doc["eligible"]:
            continue
        done = analyses.get(doc["id"], {}).get("status") == "ok"
        counts[cid]["eligible"] += 1
        counts[cid]["analyzed"] += int(done)
        if doc["id"] not in scope:
            continue
        counts[cid]["scope_eligible"] += 1
        counts[cid]["scope_analyzed"] += int(done)
        if done:
            completed[cid] += 1
        else:
            groups.setdefault(cid, []).append(doc["id"])
    for cid, values in groups.items():
        values.sort(key=lambda doc_id: (sha256(f"{SAMPLING_SEED}:{cid}:{doc_id}".encode()).hexdigest(), doc_id))
    selected, allocated = [], Counter()
    while len(selected) < limit:
        available = [cid for cid, values in groups.items() if values]
        if not available:
            break
        cid = min(available, key=lambda c: (completed[c] + allocated[c], c))
        selected.append(groups[cid].pop(0))
        allocated[cid] += 1
    return {"document_ids": selected, "limit": limit, "seed": SAMPLING_SEED,
            "strategy": "eligible-only; SHA256 order within cluster; minimum completed+allocated across clusters",
            "scope_count": len(scope), "clusters": [counts[c] for c in sorted(counts)],
            "gold_labels_used": False, "inference_executed": False}


def contextual_result(original, result, kind, start, end):
    """Only translate native character spans; leave scores, heads and labels intact."""
    value = copy.deepcopy(result)
    if kind == "dependency":
        for sentence in value.get("sentences", []):
            sentence["start"] += start
            sentence["end"] += start
            for item in sentence.get("items", []):
                item["start"] += start
                item["end"] += start
            validate_annotations(original, sentence.get("items", []), "form")
    elif kind == "information_extraction":
        rows = value.get("entities", [])
        for entity in rows:
            entity["start"] += start
            entity["end"] += start
        validate_annotations(original, rows, "text")
        for relation in value.get("relations", []):
            for field in ("subject", "object", "evidence", "trigger"):
                span = relation.get(field)
                if isinstance(span, dict) and "start" in span:
                    span["start"] += start
                    span["end"] += start
                    validate_annotations(original, [span], "text")
    return value


def replay_cases():
    from nlplab.comprehensive.system_tests import SAMPLE, CUSTOM_EXAMPLES
    return [
        {"id": "multi-entity", "title": "多实体新闻", "workspace": "single", "text": SAMPLE,
         "expected": "观察真实词项、POS、NER和新闻预测；不预写实体结果"},
        {"id": "punctuation", "title": "纯标点", "workspace": "single", "text": "，。！？",
         "expected": "提示输入正文，不调用模型"},
        {"id": "no-entity", "title": "无实体短文", "workspace": "single", "text": "生活越来越美好。",
         "expected": "观察真实NER；若为空保留空集合，不手工修补"},
        {"id": "too-few", "title": "文档数量不足", "workspace": "documents", "documents": CUSTOM_EXAMPLES[:2], "k": 2,
         "expected": "提示至少3篇；不拟合模型"},
        {"id": "custom-topics", "title": "短新闻集合", "workspace": "documents", "documents": CUSTOM_EXAMPLES, "k": 3,
         "expected": "显式执行临时聚类后探索真实分配"},
    ]
