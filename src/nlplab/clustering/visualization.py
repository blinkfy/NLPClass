"""从已保存的实际聚类向量生成三维显示缓存，绝不拟合聚类模型。"""

import argparse
import json
import platform
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.course import file_sha256


def project_three_dimensions(vectors, ids, *, n_iter=7, random_state=42):
    if len(ids) != vectors.shape[0] or len(set(ids)) != len(ids):
        raise DataValidationError("投影要求唯一且与向量行数一致的文档 ID")
    if min(vectors.shape) < 3:
        raise DataValidationError("三维投影至少需要三个文档与三个特征")
    model = TruncatedSVD(n_components=3, n_iter=n_iter, random_state=random_state)
    with threadpool_limits(limits=1):
        coordinates = model.fit_transform(vectors)
    return {
        "method": "TruncatedSVD",
        "purpose": "visualization_only",
        "dimensions": 3,
        "config": {"n_components": 3, "n_iter": n_iter, "random_state": random_state},
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "points": [
            {"id": document_id, "x": float(row[0]), "y": float(row[1]), "z": float(row[2])}
            for document_id, row in zip(ids, coordinates)
        ],
    }


def save_projection_cache(settings, run_id):
    from nlplab.services.clustering_service import ClusteringService

    service = ClusteringService(settings)
    directory = service.store.resolve(run_id)
    service.verify_artifacts(directory.name)
    model_path = settings.models / "exp2" / directory.name
    source = model_path / "document_vectors.npy"
    if source.is_file():
        vectors = np.load(source, allow_pickle=False)
    else:
        source = model_path / "document_vectors.npz"
        vectors = load_npz(source)
    ids_path = directory / "input_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    config = service.store.read(directory.name, "config.json").get("projection", {})
    value = project_three_dimensions(
        vectors, ids, n_iter=config.get("n_iter", 7), random_state=config.get("random_state", 42)
    )
    value["run_id"] = directory.name
    value["source"] = {
        "vectors_sha256": file_sha256(source),
        "ids_sha256": file_sha256(ids_path),
        "input_dimension": vectors.shape[1],
        "clustering_fit_executed": False,
    }
    value["environment"] = {"python": sys.version, "platform": platform.platform()}
    cache = settings.outputs / "exp2" / "visualizations" / directory.name
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "projection3d.json"
    temporary = cache / f"{uuid4().hex}.tmp"
    write_json(temporary, value)
    temporary.replace(target)
    write_json(cache / "manifest.json", {"projection3d_sha256": file_sha256(target)})
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    save_projection_cache(Settings(args.root), args.run_id)
