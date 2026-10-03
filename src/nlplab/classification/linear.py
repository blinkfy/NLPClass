"""C0/C1 在相同稀疏向量上的训练、无 pickle 保存及精确线性解释。"""

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.thucnews import CATEGORIES


class LinearTextClassifier:
    def __init__(self, model_id: str, parameter: float):
        if model_id not in {"C0", "C1"} or parameter <= 0:
            raise DataValidationError("仅支持 C0/C1 的正 alpha/C")
        self.model_id, self.parameter = model_id, parameter

    def fit(self, vectors, labels):
        if set(labels) != set(CATEGORIES):
            raise DataValidationError("训练数据必须覆盖固定五类")
        start = perf_counter()
        model = (
            MultinomialNB(alpha=self.parameter, force_alpha=True)
            if self.model_id == "C0"
            else LinearSVC(
                C=self.parameter,
                random_state=42,
                dual=True,
                tol=1e-4,
                max_iter=10000,
                penalty="l2",
                loss="squared_hinge",
                multi_class="ovr",
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(vectors, labels)
        order = [list(model.classes_).index(label) for label in CATEGORIES]
        if self.model_id == "C0":
            self.weights = model.feature_log_prob_[order].copy()
            self.intercept = model.class_log_prior_[order].copy()
            self.extra = {
                "class_prior": np.exp(self.intercept).tolist(),
                "class_count": model.class_count_[order].tolist(),
            }
        else:
            self.weights = model.coef_[order].copy()
            self.intercept = model.intercept_[order].copy()
            self.extra = {
                "n_iter": int(model.n_iter_),
                "support_vectors_available": False,
                "support_note": "LinearSVC/liblinear 不暴露 support_vectors_ 或支持向量数量",
                "solver": "liblinear",
                "dual": True,
                "random_state": 42,
                "max_iter": 10000,
                "tol": 1e-4,
                "penalty": "l2",
                "loss": "squared_hinge",
                "multi_class": "ovr",
            }
        self.fit_seconds = perf_counter() - start
        # 用 sklearn 预测验证数值保存表示，与固定类别顺序无关。
        expected = model.predict(vectors)
        if not np.array_equal(self.predict(vectors), expected):
            raise DataValidationError("线性参数导出与 sklearn 预测不一致")
        return self

    def scores(self, vectors):
        return np.asarray(vectors @ self.weights.T) + self.intercept

    def predict(self, vectors):
        # sklearn classes_ 按字典序；并列也必须保持其预测规则。
        order = np.asarray(sorted(range(len(CATEGORIES)), key=lambda i: CATEGORIES[i]))
        return np.asarray(CATEGORIES)[order[self.scores(vectors)[:, order].argmax(axis=1)]]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / "parameters.npz", weights=self.weights, intercept=self.intercept)
        write_json(
            path / "model.json",
            {
                "model_id": self.model_id,
                "parameter": self.parameter,
                "parameter_name": "alpha" if self.model_id == "C0" else "C",
                "label_order": list(CATEGORIES),
                "fit_seconds": self.fit_seconds,
                **self.extra,
            },
        )

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "model.json").read_text("utf-8"))
        if meta["label_order"] != list(CATEGORIES):
            raise DataValidationError("模型类别映射不符")
        result = cls(meta["model_id"], meta["parameter"])
        with np.load(path / "parameters.npz", allow_pickle=False) as arrays:
            result.weights, result.intercept = arrays["weights"], arrays["intercept"]
        result.fit_seconds, result.extra = meta["fit_seconds"], meta
        return result

    def explain(self, row, terms, top_n=12):
        scores = self.scores(row)[0]
        ranking = sorted(range(len(CATEGORIES)), key=lambda i: (-scores[i], CATEGORIES[i]))
        best, second = map(int, ranking[:2])
        indices, values = row.indices, row.data
        direct = values * self.weights[best, indices]
        contrast = values * (self.weights[best, indices] - self.weights[second, indices])
        # NB 的 log P(term|class) 为负；用与 runner-up 的 log-score 差解释判别证据。
        displayed = contrast if self.model_id == "C0" else direct
        positions = sorted(range(len(indices)), key=lambda p: (-displayed[p], terms[indices[p]]))
        posterior = np.exp(scores - logsumexp(scores)) if self.model_id == "C0" else None
        return {
            "prediction": CATEGORIES[best],
            "runner_up": CATEGORIES[second],
            "score_type": "posterior" if self.model_id == "C0" else "decision_score",
            "model_score": float(posterior[best] if posterior is not None else scores[best]),
            "scores": [
                {
                    "label": label,
                    "value": float(posterior[i] if posterior is not None else scores[i]),
                    "joint_log_score": float(scores[i]) if posterior is not None else None,
                }
                for i, label in enumerate(CATEGORIES)
            ],
            "score_gap": float(scores[best] - scores[second]),
            "zero_vector": row.nnz == 0,
            "contribution_method": "x_j * (log_theta_pred - log_theta_runner_up)"
            if self.model_id == "C0"
            else "x_j * w_pred,j",
            "intercept": float(self.intercept[best]),
            "feature_sum": float(direct.sum()),
            "reconstructed_score": float(self.intercept[best] + direct.sum()),
            "contrast_intercept": float(self.intercept[best] - self.intercept[second]),
            "contrast_feature_sum": float(contrast.sum()),
            "top_contributing_features": [
                {
                    "term": str(terms[indices[p]]),
                    "tfidf": float(values[p]),
                    "weight": float(self.weights[best, indices[p]]),
                    "contribution": float(displayed[p]),
                    "direct_contribution": float(direct[p]),
                    "runner_up_contrast": float(contrast[p]),
                }
                for p in positions[:top_n]
            ],
        }

    def global_features(self, terms, top_n=20):
        result = []
        for i, label in enumerate(CATEGORIES):
            discriminative = self.weights[i] - np.mean(np.delete(self.weights, i, axis=0), axis=0)
            order = np.argsort(-discriminative, kind="stable")[:top_n]
            high = np.argsort(-self.weights[i], kind="stable")[:top_n]
            result.append(
                {
                    "label": label,
                    "intercept": float(self.intercept[i]),
                    "class_prior": float(np.exp(self.intercept[i]))
                    if self.model_id == "C0"
                    else None,
                    "discriminative_features": [
                        {
                            "term": str(terms[j]),
                            "weight": float(self.weights[i, j]),
                            "contrast_weight": float(discriminative[j]),
                        }
                        for j in order
                    ],
                    "high_weight_features": [
                        {"term": str(terms[j]), "weight": float(self.weights[i, j])} for j in high
                    ],
                }
            )
        return result
