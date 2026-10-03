"""Pure subset, aggregation and deterministic checks for development learning curves."""

import hashlib
import json

import numpy as np

from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES

SUBSET_SEEDS = (101, 202, 303)
PER_CLASS_SIZES = (16, 32, 48, 64, 84)
MODEL_IDS = ("C0", "C1", "C2", "C3")


def ids_hash(ids):
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def build_ladders(documents, seeds=SUBSET_SEEDS, sizes=PER_CLASS_SIZES):
    if len({d.id for d in documents}) != len(documents):
        raise DataValidationError("Duplicate training IDs")
    categories = {c: [d.id for d in documents if d.label == c] for c in CATEGORIES}
    if any(len(ids) < max(sizes) for ids in categories.values()):
        raise DataValidationError("Not enough balanced Train documents")
    canonical = [d.id for d in documents]
    ladders = []
    for seed in seeds:
        ordered = {
            c: sorted(ids, key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i))
            for c, ids in categories.items()
        }
        for n in sizes:
            by_class = {c: ordered[c][:n] for c in CATEGORIES}
            selected = {i for ids in by_class.values() for i in ids}
            ids = [i for i in canonical if i in selected]
            ladders.append(
                {
                    "subset_seed": seed,
                    "train_size": n * len(CATEGORIES),
                    "per_class": n,
                    "category_ids": by_class,
                    "document_ids": ids,
                    "ids_sha256": ids_hash(ids),
                    "category_counts": {c: n for c in CATEGORIES},
                }
            )
    return ladders


def effective_k(raw_dimension, target=10000):
    if raw_dimension <= 0 or target <= 0:
        raise DataValidationError("Empty character feature space")
    return min(raw_dimension, target)


def moments(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def verify_full_size(points):
    checks = {}
    for model_id in MODEL_IDS:
        full = [p for p in points if p["model_id"] == model_id and p["train_size"] == 420]
        if len(full) != 3 or {p["subset_seed"] for p in full} != set(SUBSET_SEEDS):
            raise DataValidationError("420 sanity check requires all three replicates")
        signatures = [p["deterministic_signature"] for p in full]
        if len(set(signatures)) != 1:
            raise DataValidationError(f"STOP: {model_id} 420 deterministic mismatch")
        checks[model_id] = {"identical": True, "signature": signatures[0], "replicates": 3}
    return checks


def aggregate_points(points):
    result = []
    for model in MODEL_IDS:
        for n in PER_CLASS_SIZES:
            group = sorted(
                [p for p in points if p["model_id"] == model and p["train_size"] == n * 5],
                key=lambda p: p["subset_seed"],
            )
            if len(group) != 3 or {p["subset_seed"] for p in group} != set(SUBSET_SEEDS):
                raise DataValidationError("Each curve point requires all three ladders")
            row = {
                "model_id": model,
                "train_size": n * 5,
                "replicates": group,
                "train_macro_f1": moments([p["train_metrics"]["macro_f1"] for p in group]),
                "validation_macro_f1": moments(
                    [p["validation_metrics"]["macro_f1"] for p in group]
                ),
                "validation_accuracy": moments(
                    [p["validation_metrics"]["accuracy"] for p in group]
                ),
                "gap": moments(
                    [
                        p["train_metrics"]["macro_f1"] - p["validation_metrics"]["macro_f1"]
                        for p in group
                    ]
                ),
                "per_class_f1": [
                    {
                        "label": c,
                        **moments([p["validation_metrics"]["per_class"][i]["f1"] for p in group]),
                    }
                    for i, c in enumerate(CATEGORIES)
                ],
            }
            numeric_keys = sorted(
                {
                    k
                    for p in group
                    for k, v in p["representation"].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            )
            row["representation"] = {
                k: moments([p["representation"][k] for p in group]) for k in numeric_keys
            }
            result.append(row)
    return result
