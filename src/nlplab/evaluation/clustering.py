"""内部指标只接收向量/簇；真实标签只在拟合结束后的显式 join 使用。"""

from collections import Counter

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import text_sha256


def internal_metrics(vectors, dense_vectors, labels, model):
    sizes = np.bincount(labels, minlength=model.config.n_clusters).tolist()
    return {
        "inertia": model.inertia,
        "silhouette": float(silhouette_score(vectors, labels, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(dense_vectors, labels)),
        "davies_bouldin": float(davies_bouldin_score(dense_vectors, labels)),
        "n_iter": model.n_iter,
        "cluster_sizes": sizes,
    }


def join_gold_posthoc(documents, assignments, gold_records):
    if len(assignments) != len(documents) or any(
        item["id"] != doc.id for doc, item in zip(documents, assignments)
    ):
        raise DataValidationError("事后评价要求已完成、与输入顺序一致的 assignments")
    gold = {row["id"]: row for row in gold_records}
    if len(gold) != len(gold_records) or set(gold) != {doc.id for doc in documents}:
        raise DataValidationError("事后标签必须按唯一 id 完整关联，不允许按文件顺序猜测")
    labels = []
    for doc in documents:
        row = gold[doc.id]
        if text_sha256(row["text"]) != text_sha256(doc.text):
            raise DataValidationError("事后标签来源文本与聚类输入不匹配")
        label = row.get("primary_label")
        if not isinstance(label, str) or not label:
            raise DataValidationError("事后评价缺失真实类别")
        labels.append(label)
    return labels


def external_metrics(cluster_labels, gold_labels):
    h, c, v = homogeneity_completeness_v_measure(gold_labels, cluster_labels)
    return {
        "ari": float(adjusted_rand_score(gold_labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(gold_labels, cluster_labels)),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
    }


def add_posthoc_composition(result, gold_labels):
    for item, label in zip(result["assignments"], gold_labels):
        item["gold_category"] = label
    for cluster in result["clusters"]:
        counts = Counter(
            item["gold_category"]
            for item in result["assignments"]
            if item["cluster_id"] == cluster["cluster_id"]
        )
        cluster["category_counts"] = dict(sorted(counts.items()))
        cluster["dominant_category"] = sorted(counts, key=lambda key: (-counts[key], key))[0]
    result["category_order"] = sorted(set(gold_labels))
    result["external"] = external_metrics(
        [item["cluster_id"] for item in result["assignments"]], gold_labels
    )
