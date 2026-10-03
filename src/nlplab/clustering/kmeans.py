"""L2 TF-IDF 上的 sklearn Lloyd K-means；显式配置和透明中心保存。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from threadpoolctl import threadpool_limits

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: int = 5
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    algorithm: str = "lloyd"


class KMeansClusterer:
    def __init__(self, config=KMeansConfig()):
        self.config = config
        self.model = KMeans(**asdict(config))

    def fit_predict(self, vectors):
        if not 2 <= self.config.n_clusters < vectors.shape[0]:
            raise DataValidationError("K 必须在 2 与文档数-1 之间")
        with threadpool_limits(limits=1):
            labels = self.model.fit_predict(vectors)
        if len(np.unique(labels)) != self.config.n_clusters:
            raise DataValidationError("有效簇数小于 K；拒绝保存伪完整结果")
        self.centers = self.model.cluster_centers_
        self.n_iter = int(self.model.n_iter_)
        self.inertia = float(self.model.inertia_)
        return labels

    def predict(self, vectors):
        labels, distances = pairwise_distances_argmin_min(vectors, self.centers)
        return labels, distances

    def save(self, path: Path, *, space="tfidf_l2"):
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "model.json",
            {
                "model": "sklearn-kmeans",
                "config": asdict(self.config),
                "n_iter": self.n_iter,
                "inertia": self.inertia,
                "distance": f"euclidean on {space}",
                "format_version": 1,
            },
        )
        np.save(path / "centers.npy", self.centers, allow_pickle=False)

    @classmethod
    def load(cls, path: Path):
        info = json.loads((path / "model.json").read_text(encoding="utf-8"))
        result = cls(KMeansConfig(**info["config"]))
        result.centers = np.load(path / "centers.npy", allow_pickle=False)
        result.n_iter, result.inertia = info["n_iter"], info["inertia"]
        return result
