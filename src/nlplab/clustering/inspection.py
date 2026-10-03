"""冻结向量与中心的只读派生：真实距离、词权重与中心显示坐标，无任何 fit。"""

import argparse
import json
import platform
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.sparse import issparse, load_npz
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.course import file_sha256


def vector_preview(vector, limit=10):
    if issparse(vector):
        row = vector.tocsr()
        ranked = sorted(zip(row.indices, row.data), key=lambda pair: (-abs(pair[1]), pair[0]))[
            :limit
        ]
        return [{"dimension": int(index) + 1, "value": float(value)} for index, value in ranked]
    return [
        {"dimension": index + 1, "value": float(value)}
        for index, value in enumerate(np.asarray(vector).ravel()[:limit])
    ]


def full_vector_values(vector):
    """Return real dense coordinates only for the requested 100-D display case."""
    if issparse(vector):
        return None
    values = np.asarray(vector).ravel()
    if values.size != 100:
        return None
    return [float(value) for value in values]


def inspect_centers(vectors, centers, labels, projection2d, components2d, projection3d):
    """三维复用既有线性投影；先验证存储中心确为成员均值，再取投影均值。"""
    labels = np.asarray(labels)
    if vectors.shape[0] != len(labels) or vectors.shape[1] != centers.shape[1]:
        raise DataValidationError("向量、分配与中心维度不一致")
    if projection2d.shape != (len(labels), 2) or projection3d.shape != (len(labels), 3):
        raise DataValidationError("投影必须与文档顺序及显示维度一致")
    if components2d.shape != (2, vectors.shape[1]):
        raise DataValidationError("二维投影基底与实际空间维度不一致")
    with threadpool_limits(limits=1):
        distances = euclidean_distances(vectors, centers)
    nearest = distances.argmin(axis=1)
    if not np.array_equal(nearest, labels):
        raise DataValidationError("保存分配与最近存储质心不一致，禁止伪造判定过程")
    if not np.allclose(vectors @ components2d.T, projection2d, atol=1e-10, rtol=1e-9):
        raise DataValidationError("保存二维坐标与投影基底不一致")
    projected2d = centers @ components2d.T
    rows = []
    for cluster_id, center in enumerate(centers):
        members = labels == cluster_id
        if not members.any():
            raise DataValidationError("保存分配含空簇")
        mean = np.asarray(vectors[members].mean(axis=0)).ravel()
        if not np.allclose(center, mean, atol=1e-10, rtol=1e-9):
            raise DataValidationError("中心不是已分配成员均值，无法复用三维线性投影")
        rows.append(
            {
                "cluster_id": cluster_id,
                "dimension": centers.shape[1],
                "norm": float(np.linalg.norm(center)),
                "vector_preview": vector_preview(center),
                "projection2d": projected2d[cluster_id].tolist(),
                "projection3d": projection3d[members].mean(axis=0).tolist(),
                "member_mean_max_error": float(np.max(np.abs(mean - center))),
            }
        )
    return {"centroids": rows, "distances": distances.tolist(), "nearest": nearest.tolist()}


def save_inspection_cache(settings, run_id):
    from nlplab.services.clustering_service import ClusteringService

    service = ClusteringService(settings)
    directory = service.store.resolve(run_id)
    service.verify_artifacts(directory.name)
    model_path = settings.models / "exp2" / directory.name
    sources = {}

    def track(path):
        sources[str(path.relative_to(settings.root))] = file_sha256(path)
        return path

    def read(path):
        return json.loads(track(path).read_text(encoding="utf-8"))

    ids = read(directory / "input_ids.json")
    config = read(directory / "config.json")
    overview = read(directory / "overview.json")
    tokens_path = track(directory / "tokens.jsonl")
    tokens = [
        json.loads(line) for line in tokens_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if [item["id"] for item in tokens] != ids:
        raise DataValidationError("切词缓存 ID 顺序不一致")
    vector_path = model_path / "document_vectors.npy"
    if vector_path.is_file():
        vectors = np.load(track(vector_path), allow_pickle=False)
        word_vectors = load_npz(track(model_path / "word_vectors.npz"))
    else:
        vectors = load_npz(track(model_path / "document_vectors.npz"))
        word_vectors = vectors
    vocabulary = read(model_path / "tfidf/vocabulary.json")["vocabulary"]
    terms = [term for term, _ in sorted(vocabulary.items(), key=lambda pair: pair[1])]
    if word_vectors.shape != (len(ids), len(terms)) or vectors.shape[0] != len(ids):
        raise DataValidationError("冻结词表、向量与 ID 维度不一致")
    projection2d = read(directory / "projection.json")
    cache_path = settings.outputs / "exp2/visualizations" / directory.name
    projection3d = read(cache_path / "projection3d.json")
    projection_manifest = read(cache_path / "manifest.json")
    if file_sha256(cache_path / "projection3d.json") != projection_manifest["projection3d_sha256"]:
        raise DataValidationError("三维显示源指纹不符")
    if projection3d["source"]["vectors_sha256"] != file_sha256(
        track(vector_path if vector_path.is_file() else model_path / "document_vectors.npz")
    ):
        raise DataValidationError("三维显示源向量指纹不符")
    if projection3d["source"]["ids_sha256"] != file_sha256(directory / "input_ids.json"):
        raise DataValidationError("三维显示源文档 ID 指纹不符")
    for projection in (projection2d, projection3d):
        if [point["id"] for point in projection["points"]] != ids:
            raise DataValidationError("显示坐标 ID 顺序不一致")
    p2 = np.array([[point["x"], point["y"]] for point in projection2d["points"]])
    p3 = np.array([[point["x"], point["y"], point["z"]] for point in projection3d["points"]])
    components = np.load(track(model_path / "svd_components.npy"), allow_pickle=False)
    documents = []
    for index, document_id in enumerate(ids):
        row = word_vectors.getrow(index)
        ranked = sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], terms[pair[0]]))[
            :12
        ]
        actual = vectors[index] if not issparse(vectors) else vectors.getrow(index)
        norm = (
            np.sqrt(actual.multiply(actual).sum()) if issparse(actual) else np.linalg.norm(actual)
        )
        document = {
            "id": document_id,
            "tokens": tokens[index]["tokens"][:40],
            "token_count": len(tokens[index]["tokens"]),
            "top_terms": [
                {"term": terms[column], "weight": float(weight)} for column, weight in ranked
            ],
            "vector_preview": vector_preview(actual),
            "vector_norm": float(norm),
            "tfidf_dimension": word_vectors.shape[1],
            "input_dimension": vectors.shape[1],
        }
        values = full_vector_values(actual)
        if values is not None:
            document["vector_values"] = values
        documents.append(document)
    results = {}
    for k in config["k_values"]:
        name = f"reference/k{k}.json" if k == config["reference_k"] else f"results/k{k}.json"
        result = read(directory / name)
        assignments = result["assignments"]
        if [assignment["id"] for assignment in assignments] != ids:
            raise DataValidationError("冻结分配 ID 顺序不一致")
        centers = np.load(track(model_path / f"k{k}/centers.npy"), allow_pickle=False)
        labels = [item["cluster_id"] for item in assignments]
        derived = inspect_centers(vectors, centers, labels, p2, components, p3)
        actual_distances = np.array(derived["distances"])[np.arange(len(ids)), labels]
        if not np.allclose(
            actual_distances, [item["distance"] for item in assignments], atol=1e-10, rtol=1e-9
        ):
            raise DataValidationError("派生距离与冻结分配距离不一致")
        results[str(k)] = derived
    value = {
        "run_id": directory.name,
        "space": overview["representation"].get("space", "tfidf_l2"),
        "dimension": vectors.shape[1],
        "document_count": len(ids),
        "documents": documents,
        "results": results,
        "provenance": {
            "source_files": sources,
            "builder_sha256": file_sha256(Path(__file__)),
            "formal_fit_executed": False,
            "centroid_projection2d": "stored centroid multiplied by frozen SVD basis",
            "centroid_projection3d": (
                "mean of existing linear projections after "
                "centroid/member-mean equality verification"
            ),
            "distance": "Euclidean distance in actual stored clustering input space",
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    target = cache_path / "inspection.json"
    temporary = cache_path / f"{uuid4().hex}.tmp"
    write_json(temporary, value)
    temporary.replace(target)
    write_json(cache_path / "inspection_manifest.json", {"inspection_sha256": file_sha256(target)})
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    save_inspection_cache(Settings(args.root), args.run_id)
