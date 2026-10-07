"""Canonical original-text spans and descriptive application aggregation."""

from collections import Counter
from hashlib import sha256

from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES

LABEL_NAMES = dict(zip(CATEGORIES, ("财经", "教育", "科技", "体育", "娱乐")))
ENTITY_TYPES = ("PER", "LOC", "ORG")


def validate_text(text, maximum=5000):
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError("请输入非空文本")
    if len(text) > maximum:
        raise DataValidationError(f"输入最多 {maximum} 字，不自动截断")
    if not any(char.isalnum() for char in text):
        raise DataValidationError("文本只有空白或标点，请输入可分析的正文")


def document_id(text):
    return "text-" + sha256(text.encode("utf-8")).hexdigest()[:24]


def token_spans(text, tokens):
    if "".join(tokens) != text:
        raise DataValidationError("分词不能无损映射回原文")
    rows, position = [], 0
    for index, token in enumerate(tokens):
        rows.append({"id": index + 1, "form": token, "start": position,
                     "end": position + len(token), "whitespace": token.isspace()})
        position += len(token)
    return rows


def validate_annotations(text, rows, field):
    for row in rows:
        start, end = row.get("start"), row.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise DataValidationError("原文 annotation 跨度无效")
        if text[start:end] != row[field]:
            raise DataValidationError("annotation 与原文不一致，拒绝显示错位结果")


def aggregate(collection):
    clusters = sorted({doc["cluster_id"] for doc in collection["documents"]})
    classified = [[0] * len(CATEGORIES) for _ in clusters]
    entities = [[0] * len(ENTITY_TYPES) for _ in clusters]
    frequencies, documents, errors = Counter(), [], []
    classified_observations, ner_observations = [0] * len(clusters), [0] * len(clusters)
    analyses = collection.get("analyses", {})
    for doc in collection["documents"]:
        record = analyses.get(doc["id"])
        application = collection.get("application_predictions", {}).get(doc["id"])
        if record is None and application is None:
            continue
        if record is None:
            record = {"status": "classification_only", "modules": {}, "lexical": {"entities": [], "pos": []}}
        if application:
            record = {**record, "classification": application,
                      "modules": {**record["modules"], "classification": {"status": "ok"}}}
        ci = clusters.index(doc["cluster_id"])
        prediction = record.get("classification", {}).get("prediction")
        if record["modules"].get("classification", {}).get("status") == "ok":
            classified[ci][CATEGORIES.index(prediction)] += 1
            classified_observations[ci] += 1
        ner_ok = record["modules"].get("ner", {}).get("status") == "ok"
        if ner_ok:
            ner_observations[ci] += 1
        mentions = record["lexical"]["entities"] if ner_ok else []
        for mention in mentions:
            ti = ENTITY_TYPES.index(mention["type"])
            entities[ci][ti] += 1
            frequencies[(doc["cluster_id"], mention["type"], mention["text"])] += 1
        for module, value in record["modules"].items():
            if value["status"] != "ok":
                errors.append({"id": doc["id"], "module": module, "error": value.get("error")})
        documents.append({"id": doc["id"], "prediction": prediction,
                          "entities": mentions if ner_ok else None, "ner_known": ner_ok, "classification_known": record["modules"].get("classification", {}).get("status") == "ok",
                          "pos_known": record["modules"].get("pos", {}).get("status") == "ok",
                          "pos_counts": dict(Counter(row["pos"] for row in record["lexical"]["pos"])) if record["modules"].get("pos", {}).get("status") == "ok" else None,
                          "entity_count": len(mentions) if ner_ok else None, "ner_status": record["modules"].get("ner", {}).get("status"), "entity_types": sorted({mention["type"] for mention in mentions}), "status": record["status"]})
    return {"cluster_ids": clusters, "label_order": list(CATEGORIES), "label_names": LABEL_NAMES,
            "classification_matrix": [row if classified_observations[i] else [None]*len(CATEGORIES) for i, row in enumerate(classified)],
            "classification_document_counts": classified_observations, "entity_types": list(ENTITY_TYPES),
            "entity_matrix": [row if ner_observations[i] else [None]*len(ENTITY_TYPES) for i, row in enumerate(entities)],
            "ner_document_counts": ner_observations, "top_entities": [
                {"cluster_id": key[0], "type": key[1], "text": key[2], "count": value}
                for key, value in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:50]],
            "analyzed_count": len(analyses), "total_count": len(collection["documents"]),
            "eligible_count": sum(bool(doc["eligible"]) for doc in collection["documents"]),
            "classification_count": sum(map(sum, classified)),
            "ner_document_count": sum(record["modules"].get("ner", {}).get("status") == "ok"
                                      for record in analyses.values()),
            "documents": documents, "errors": errors,
            "notice": "只统计实际已分析文档；未处理/不可用不代表零实体。Cluster 不等于类别；应用预测分布不是独立泛化估计。"}
