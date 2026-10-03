"""真正用于聚类的 LSA 表示，保存 components 后支持无拟合 transform。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json


@dataclass(frozen=True)
class LSAConfig:
    n_components: int = 100
    n_iter: int = 7
    random_state: int = 42
    algorithm: str = "randomized"


class TextLSA:
    def __init__(self, config=LSAConfig()):
        self.config = config

    @staticmethod
    def _normalize(vectors):
        if np.any(np.linalg.norm(vectors, axis=1) <= np.finfo(np.float64).eps):
            raise DataValidationError("LSA 出现零向量，不静默删除文档")
        return normalize(vectors, norm="l2")

    def fit_transform(self, vectors):
        if not 2 <= self.config.n_components < min(vectors.shape):
            raise DataValidationError("LSA 维度须小于文档数和词空间维度；需显式新配置")
        model = TruncatedSVD(**asdict(self.config))
        with threadpool_limits(limits=1):
            latent = model.fit_transform(vectors)
        self.components = model.components_
        self.explained_variance_ratio = model.explained_variance_ratio_
        self.singular_values = model.singular_values_
        return self._normalize(latent)

    def transform(self, vectors):
        if vectors.shape[1] != self.components.shape[1]:
            raise DataValidationError("LSA transform 输入词空间维度不符")
        return self._normalize(np.asarray(vectors @ self.components.T))

    def metadata(self):
        return {
            **asdict(self.config),
            "method": "TruncatedSVD + L2",
            "purpose": "clustering_input",
            "input_dimension": self.components.shape[1],
            "output_dimension": self.components.shape[0],
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "explained_variance_sum": float(self.explained_variance_ratio.sum()),
        }

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "metadata.json", {"config": asdict(self.config), **self.metadata()})
        np.savez_compressed(
            path / "lsa.npz",
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            singular_values=self.singular_values,
        )

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        result = cls(LSAConfig(**meta["config"]))
        with np.load(path / "lsa.npz", allow_pickle=False) as saved:
            result.components = saved["components"]
            result.explained_variance_ratio = saved["explained_variance_ratio"]
            result.singular_values = saved["singular_values"]
        return result
