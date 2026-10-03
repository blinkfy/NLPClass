"""复用统一分类评价；多 seed 选择及 Test 之后的配对，不导入 PyTorch。"""

from statistics import median

import numpy as np

from nlplab.classification.stage2_experiment import paired_outcomes
from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics


def annotate_history(result, val_docs):
    gold = [d.label for d in val_docs]
    history = []
    for row in result["history"]:
        prediction = [CATEGORIES[i] for i in row["validation_prediction_indices"]]
        metrics = classification_metrics(gold, prediction)
        if not np.isclose(metrics["macro_f1"], row["validation_macro_f1"], atol=1e-12):
            raise DataValidationError("early stopping F1 与统一指标不一致")
        history.append(
            {**row, "validation_metrics": metrics, "validation_accuracy": metrics["accuracy"]}
        )
    predicted = [CATEGORIES[i] for i in result["best_prediction_indices"]]
    best_metrics = classification_metrics(gold, predicted)
    return (
        history,
        best_metrics,
        [
            {
                "document_id": doc.id,
                "gold": doc.label,
                "prediction": prediction,
                "correct": doc.label == prediction,
            }
            for doc, prediction in zip(val_docs, predicted)
        ],
    )


def summarize_seeds(results):
    summary = []
    for lr in sorted({r["learning_rate"] for r in results}, reverse=True):
        rows = [r for r in results if r["learning_rate"] == lr]
        if sorted(r["seed"] for r in rows) != [42, 43, 44]:
            raise DataValidationError("每个 lr 必须有且仅有 seed42/43/44")
        scores = [r["best_validation_macro_f1"] for r in rows]
        summary.append(
            {
                "learning_rate": lr,
                "seeds": rows,
                "mean_macro_f1": float(np.mean(scores)),
                "std_macro_f1": float(np.std(scores, ddof=0)),
                "std_ddof": 0,
            }
        )
    return summary


def select_learning_rate(summary):
    if not summary:
        raise DataValidationError("空 lr summary")
    return max(
        summary, key=lambda r: (r["mean_macro_f1"], -r["std_macro_f1"], -r["learning_rate"])
    )["learning_rate"]


def median_final_epochs(best_epochs):
    if len(best_epochs) != 3 or any(e < 1 or int(e) != e for e in best_epochs):
        raise DataValidationError("final epoch 需要三个正整数 best epoch")
    return int(median(best_epochs))


def paired_stage3(old, c3, old_model_id):
    # 完整 384 通道在 C3 predictions 中保存；配对资产只重复 Top spans 并引用 document ID。
    compact = [{k: v for k, v in r.items() if k != "all_channels"} for r in c3]
    result = paired_outcomes(old, compact)
    names = {
        "both_correct": "both_correct",
        "c1_only_correct": "old_only_correct",
        "c2_only_correct": "c3_only_correct",
        "both_wrong": "both_wrong",
    }
    samples = {
        ("corrected_old" if key == "corrected_c1" else key): [
            {"document_id": p["document_id"], "gold": p["gold"], "old": p["c1"], "c3": p["c2"]}
            for p in rows
        ]
        for key, rows in result["samples"].items()
    }
    return {
        "phase": "post_test_analysis_only",
        "old_model_id": old_model_id,
        "new_model_id": "C3",
        "sample_count": result["sample_count"],
        "counts": {names[k]: v for k, v in result["counts"].items()},
        "document_ids": {names[k]: v for k, v in result["document_ids"].items()},
        "disagreement_ids": [p["document_id"] for p in samples["disagreements"]],
        "samples": samples,
    }
