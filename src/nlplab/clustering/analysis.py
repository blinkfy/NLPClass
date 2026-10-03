"""用原 TF-IDF 空间的中心权重和距离解释簇，不读取真实类别。"""

import re

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def snippet(text: str, limit: int = 140):
    value = re.sub(r"\s+", " ", text).strip()
    return value[:limit] + ("…" if len(value) > limit else "")


def summarize_clusters(
    documents, vectors, labels, centers, terms, *, top_n=12, examples=3, term_centers=None
):
    distances = euclidean_distances(vectors, centers)
    assigned_distances = distances[np.arange(len(documents)), labels]
    assignments = [
        {
            "id": doc.id,
            "cluster_id": int(label),
            "distance": float(distance),
            "snippet": snippet(doc.text),
        }
        for doc, label, distance in zip(documents, labels, assigned_distances)
    ]
    clusters = []
    for cluster_id, center in enumerate(centers if term_centers is None else term_centers):
        members = [item for item in assignments if item["cluster_id"] == cluster_id]
        members.sort(key=lambda item: (item["distance"], item["id"]))
        ranked = sorted(range(len(terms)), key=lambda i: (-center[i], terms[i]))[:top_n]

        def readable(items):
            return [{key: item[key] for key in ("id", "snippet", "distance")} for item in items]

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "top_terms": [
                    {"term": str(terms[i]), "weight": float(center[i])}
                    for i in ranked
                    if center[i] > 0
                ],
                "representatives": readable(members[:examples]),
                "boundary_documents": readable(list(reversed(members[-examples:]))),
            }
        )
    return assignments, clusters
