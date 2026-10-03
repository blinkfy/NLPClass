"""固定五类顺序的分类评价；所有指标使用相同 Test 预测。"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES


def classification_metrics(gold, predicted):
    if not gold or len(gold) != len(predicted) or not set(gold + predicted) <= set(CATEGORIES):
        raise DataValidationError("分类评价输入无效")
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=CATEGORIES, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "micro_f1": float(
            precision_recall_fscore_support(
                gold, predicted, labels=CATEGORIES, average="micro", zero_division=0
            )[2]
        ),
        "weighted_f1": float(np.average(f1, weights=support)),
        "label_order": list(CATEGORIES),
        "per_class": [
            {
                "label": label,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(CATEGORIES, precision, recall, f1, support)
        ],
        "confusion_matrix": confusion_matrix(gold, predicted, labels=CATEGORIES).tolist(),
        "confusion_matrix_axes": {"rows": "gold", "columns": "prediction"},
        "sample_count": len(gold),
    }
