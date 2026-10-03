"""Uncertainty for frozen predictions; no classifiers, fitting, or inference."""

from collections import Counter

import numpy as np
from scipy.stats import binomtest, norm

from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics

MODELS = ("C0", "C1", "C2", "C3")
PRIMARY_PAIRS = (("C0", "C1"), ("C1", "C2"), ("C2", "C3"))
BOOTSTRAP = {
    "replicates": 20000,
    "seed": 606,
    "method": "class-stratified paired percentile",
    "confidence": 0.95,
    "quantiles": [0.025, 0.975],
    "percentile_method": "linear",
    "per_class_draws": 18,
    "replacement": True,
    "bit_generator": "PCG64",
    "shared_indices": "one index matrix reused for all four models and three pairs",
}


def align_predictions(rows_by_model, frozen_metrics):
    if set(rows_by_model) != set(MODELS) or set(frozen_metrics) != set(MODELS):
        raise DataValidationError("Four frozen models are required")
    indexes, original_orders = {}, {}
    for model_id in MODELS:
        rows = rows_by_model[model_id]
        ids = [r["document_id"] for r in rows]
        if len(ids) != 90 or len(set(ids)) != 90:
            raise DataValidationError("Frozen Test must have 90 unique document IDs")
        original_orders[model_id] = ids
        indexes[model_id] = {r["document_id"]: r for r in rows}
    canonical = original_orders["C0"]
    if any(set(indexes[m]) != set(canonical) for m in MODELS):
        raise DataValidationError("Frozen Test ID sets differ; do not repair historical results")
    aligned = []
    for doc_id in canonical:
        records = [indexes[m][doc_id] for m in MODELS]
        gold = records[0]["gold"]
        if gold not in CATEGORIES or any(r["gold"] != gold for r in records):
            raise DataValidationError("Frozen Test gold labels differ")
        if any(r["prediction"] not in CATEGORIES for r in records):
            raise DataValidationError("Unknown frozen prediction label")
        aligned.append(
            {
                "document_id": doc_id,
                "gold": gold,
                **{m: indexes[m][doc_id]["prediction"] for m in MODELS},
            }
        )
    support = Counter(r["gold"] for r in aligned)
    if support != Counter({label: 18 for label in CATEGORIES}):
        raise DataValidationError("Frozen Test support must be 18 for each class")
    points = {}
    for model_id in MODELS:
        actual = classification_metrics(
            [r["gold"] for r in aligned], [r[model_id] for r in aligned]
        )
        if actual != frozen_metrics[model_id]:
            raise DataValidationError(f"{model_id} metrics differ from frozen formal results; stop")
        points[model_id] = actual
    return (
        aligned,
        points,
        {
            "sample_count": 90,
            "label_order": list(CATEGORIES),
            "support": dict(support),
            "join": "explicit document ID join; canonical C0 saved order",
            "same_id_sets": True,
            "gold_identical": True,
            "point_metrics_exactly_reproduced": True,
            "original_order_identical": {m: original_orders[m] == canonical for m in MODELS},
        },
    )


def wilson(correct, n, confidence=0.95):
    if not isinstance(correct, (int, np.integer)) or n <= 0 or not 0 <= correct <= n:
        raise ValueError("Wilson requires a valid integer binomial count")
    z = float(norm.ppf(0.5 + confidence / 2))
    point, z2 = correct / n, z * z
    denominator = 1 + z2 / n
    center = (point + z2 / (2 * n)) / denominator
    half = z * np.sqrt(point * (1 - point) / n + z2 / (4 * n * n)) / denominator
    return {
        "point": point,
        "low": float(max(0, center - half)),
        "high": float(min(1, center + half)),
        "correct": int(correct),
        "n": int(n),
    }


def stratified_indices(gold_indices, replicates=20000, seed=606):
    groups = [np.flatnonzero(gold_indices == c) for c in range(5)]
    if any(len(group) != 18 for group in groups):
        raise ValueError("Each bootstrap class must contain 18 frozen Test samples")
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.concatenate(
        [rng.choice(group, size=(replicates, 18), replace=True) for group in groups], axis=1
    )


def bootstrap_macro_f1(gold, predictions, indices):
    """One shared index matrix is supplied, never generated per model or pair."""
    values = np.empty((len(indices), predictions.shape[0]), dtype=np.float64)
    for start in range(0, len(indices), 500):
        sampled = indices[start : start + 500]
        sampled_gold = gold[sampled]
        offsets = np.arange(len(sampled))[:, None] * 25
        for model in range(predictions.shape[0]):
            counts = np.bincount(
                (offsets + 5 * sampled_gold + predictions[model][sampled]).ravel(),
                minlength=len(sampled) * 25,
            ).reshape(-1, 5, 5)
            diagonal = np.diagonal(counts, axis1=1, axis2=2)
            denominator = counts.sum(1) + counts.sum(2)
            f1 = np.divide(
                2 * diagonal,
                denominator,
                out=np.zeros_like(diagonal, dtype=float),
                where=denominator != 0,
            )
            values[start : start + len(sampled), model] = f1.mean(1)
    return values


def exact_mcnemar(a_only, b_only):
    if a_only < 0 or b_only < 0:
        raise ValueError("Discordant counts cannot be negative")
    discordant = a_only + b_only
    return (
        1.0
        if not discordant
        else float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue)
    )


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)) or not np.all(np.isfinite(values)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values, kind="stable")
    corrected = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(corrected, 1)
    return result.tolist()


def interval(values, point):
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return {"point": float(point), "low": float(low), "high": float(high)}


def analyze(aligned, points):
    labels = {label: i for i, label in enumerate(CATEGORIES)}
    gold = np.asarray([labels[r["gold"]] for r in aligned], dtype=np.int64)
    predicted = np.asarray([[labels[r[m]] for r in aligned] for m in MODELS], dtype=np.int64)
    indices = stratified_indices(gold, BOOTSTRAP["replicates"], BOOTSTRAP["seed"])
    boot = bootstrap_macro_f1(gold, predicted, indices)
    models, pairs = [], []
    for i, model_id in enumerate(MODELS):
        correct = predicted[i] == gold
        recalls = []
        for c, label in enumerate(CATEGORIES):
            ci = wilson(int(correct[gold == c].sum()), 18)
            recalls.append({"label": label, "support": 18, **ci})
        models.append(
            {
                "model_id": model_id,
                "accuracy": wilson(int(correct.sum()), 90),
                "macro_f1": interval(boot[:, i], points[model_id]["macro_f1"]),
                "recall": recalls,
            }
        )
    for a, b in PRIMARY_PAIRS:
        ia, ib = MODELS.index(a), MODELS.index(b)
        ca, cb = predicted[ia] == gold, predicted[ib] == gold
        counts = {
            "both_correct": int(np.sum(ca & cb)),
            "a_only_correct": int(np.sum(ca & ~cb)),
            "b_only_correct": int(np.sum(~ca & cb)),
            "both_wrong": int(np.sum(~ca & ~cb)),
        }
        ids = {
            key: [r["document_id"] for r, keep in zip(aligned, mask) if keep]
            for key, mask in {
                "both_correct": ca & cb,
                "a_only_correct": ca & ~cb,
                "b_only_correct": ~ca & cb,
                "both_wrong": ~ca & ~cb,
            }.items()
        }
        pairs.append(
            {
                "a": a,
                "b": b,
                **counts,
                "document_ids": ids,
                "accuracy_difference": points[b]["accuracy"] - points[a]["accuracy"],
                "raw_p": exact_mcnemar(counts["a_only_correct"], counts["b_only_correct"]),
                "macro_f1_difference": interval(
                    boot[:, ib] - boot[:, ia], points[b]["macro_f1"] - points[a]["macro_f1"]
                ),
            }
        )
    for pair, adjusted in zip(pairs, holm_adjust([p["raw_p"] for p in pairs])):
        pair["holm_p"] = adjusted
    return (
        {
            "models": models,
            "pairs": pairs,
            "bootstrap": BOOTSTRAP,
            "interpretation": [
                "Intervals describe Test90 sampling; exclude training randomness and shift.",
                "F1 difference CI containing zero means uncertainty, not model equivalence.",
                "McNemar p>0.05 means insufficient evidence, not proof of no difference.",
                "Holm correction applies only to the three preregistered primary McNemar tests.",
            ],
        },
        boot,
        indices,
    )
