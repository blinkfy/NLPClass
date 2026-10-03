"""C2 使用冻结 C1 的分类器族和数值解释，不修改其实现。"""

import numpy as np

from nlplab.classification.linear import LinearTextClassifier
from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES


class C2LinearClassifier(LinearTextClassifier):
    def __init__(self, model_id="C2", parameter=1.0):
        if model_id != "C2":
            raise DataValidationError("C2 模型标识不符")
        super().__init__("C1", parameter)
        self.model_id = "C2"

    def explain(self, row, terms, top_n=12):
        result = super().explain(row, terms, top_n)
        best, second = CATEGORIES.index(result["prediction"]), CATEGORIES.index(result["runner_up"])
        contrast = row.data * (self.weights[best, row.indices] - self.weights[second, row.indices])
        order = sorted(range(row.nnz), key=lambda p: (-contrast[p], terms[row.indices[p]]))[:top_n]
        result["feature_kind"] = "character_bigram"
        result["top_relative_features"] = [
            {
                "term": str(terms[row.indices[p]]),
                "tfidf": float(row.data[p]),
                "contribution": float(contrast[p]),
                "coefficient_difference": float(
                    self.weights[best, row.indices[p]] - self.weights[second, row.indices[p]]
                ),
            }
            for p in order
        ]
        return result

    def global_features(self, terms, chi2_scores, top_n=20):
        result = super().global_features(terms, top_n)
        for i, entry in enumerate(result):
            order = np.argsort(-self.weights[i], kind="stable")
            entry["positive_features"] = [
                {
                    "term": str(terms[j]),
                    "svc_coefficient": float(self.weights[i, j]),
                    "chi2_score": float(chi2_scores[j]),
                }
                for j in order
                if self.weights[i, j] > 0
            ][:top_n]
            entry["feature_kind"] = "character_bigram"
        return result
