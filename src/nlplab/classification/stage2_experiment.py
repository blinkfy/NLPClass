"""C2 两阶段协议与完成 Test 后的只读配对分析。"""

from time import perf_counter

from nlplab.classification.c2 import C2LinearClassifier
from nlplab.core.exceptions import DataValidationError
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.char_chi2 import CharacterChi2TFIDF, CharacterCounts, sparse_stats


def selection_key(candidate):
    metrics = candidate["metrics"]
    return (
        metrics["macro_f1"],
        metrics["accuracy"],
        -candidate["selected_feature_count"],
        -candidate["C"],
    )


def select_stage2(train, val, config):
    if set(d.id for d in train) & set(d.id for d in val):
        raise DataValidationError("Train/Validation IDs 有交集")
    start = perf_counter()
    raw = CharacterCounts()
    train_counts = raw.fit_transform([d.text for d in train])
    val_counts = raw.transform([d.text for d in val])
    count_seconds = perf_counter() - start
    candidates, representations = [], {}
    available = train_counts.shape[1]
    for k in config["candidates"]["k"]:
        if k != "all" and k >= available:
            for C in config["candidates"]["C"]:
                candidates.append(
                    {
                        "k": k,
                        "C": C,
                        "status": "skipped",
                        "available_feature_count": available,
                        "selected_feature_count": None,
                        "reason": "k exceeds available features"
                        if k > available
                        else "equivalent to all; all candidate retained",
                    }
                )
            continue
        start = perf_counter()
        representation = CharacterChi2TFIDF(k)
        xtrain = representation.fit_from_counts(train_counts, [d.label for d in train], raw)
        xval, val_stats = representation.transform_counts(val_counts)
        representation_seconds = perf_counter() - start
        representations[k] = (representation, xtrain, xval)
        for C in config["candidates"]["C"]:
            model = C2LinearClassifier(parameter=C).fit(xtrain, [d.label for d in train])
            start = perf_counter()
            prediction = model.predict(xval).tolist()
            inference_seconds = perf_counter() - start
            candidates.append(
                {
                    "k": k,
                    "C": C,
                    "status": "evaluated",
                    "available_feature_count": available,
                    "selected_feature_count": xtrain.shape[1],
                    "metrics": classification_metrics([d.label for d in val], prediction),
                    "model_fit_seconds": model.fit_seconds,
                    "shared_representation_fit_seconds": representation_seconds,
                    "fit_seconds": representation_seconds + model.fit_seconds,
                    "inference_seconds": inference_seconds,
                    "validation_predictions": [
                        {"document_id": d.id, "gold": d.label, "prediction": p}
                        for d, p in zip(val, prediction)
                    ],
                }
            )
    valid = [candidate for candidate in candidates if candidate["status"] == "evaluated"]
    if not valid:
        raise DataValidationError("无可用 C2 候选")
    winner = max(valid, key=selection_key)
    representation, xtrain, xval = representations[winner["k"]]
    return (
        representation,
        train_counts,
        val_counts,
        xtrain,
        xval,
        {
            "phase": "stage2-validation",
            "test_opened": False,
            "fit_ids": [d.id for d in train],
            "validation_ids": [d.id for d in val],
            "tie_break": config["tie_break"],
            "count_fit_seconds": count_seconds,
            "registered_candidate_count": len(candidates),
            "executed_model_fit_count": len(valid),
            "candidates": candidates,
            "selected": {
                "k": winner["k"],
                "C": winner["C"],
                "validation_selected_dimension": winner["selected_feature_count"],
            },
            "train_stats": representation.fit_stats,
            "validation_stats": {
                "raw_count": sparse_stats(val_counts),
                "tfidf": sparse_stats(xval),
            },
        },
    )


def fit_stage2_final(train_val, selected):
    start = perf_counter()
    representation = CharacterChi2TFIDF(selected["k"])
    vectors = representation.fit_transform(
        [d.text for d in train_val], [d.label for d in train_val]
    )
    representation_seconds = perf_counter() - start
    model = C2LinearClassifier(parameter=selected["C"]).fit(vectors, [d.label for d in train_val])
    return representation, vectors, model, representation_seconds


def evaluate_stage2_test(test, representation, model):
    start = perf_counter()
    raw = representation.raw.transform([d.text for d in test])
    vectors, selection_stats = representation.transform_counts(raw)
    transform_seconds = perf_counter() - start
    start = perf_counter()
    predicted = model.predict(vectors).tolist()
    inference_seconds = perf_counter() - start
    metrics = classification_metrics([d.label for d in test], predicted)
    start = perf_counter()
    predictions = [
        {
            "document_id": doc.id,
            "gold": doc.label,
            "correct": doc.label == prediction,
            "text_snippet": " ".join(doc.text.split())[:240],
            **model.explain(vectors.getrow(i), representation.terms),
        }
        for i, (doc, prediction) in enumerate(zip(test, predicted))
    ]
    return (
        vectors,
        metrics,
        predictions,
        {
            **selection_stats,
            "tfidf": sparse_stats(vectors),
            "transform_seconds": transform_seconds,
            "inference_seconds": inference_seconds,
            "explanation_seconds": perf_counter() - start,
            "test_evaluation_count": 1,
        },
    )


def paired_outcomes(c1, c2):
    left, right = {}, {}
    for rows, target in ((c1, left), (c2, right)):
        for row in rows:
            if row["document_id"] in target:
                raise DataValidationError("配对 ID 重复")
            if row["correct"] != (row["gold"] == row["prediction"]):
                raise DataValidationError("配对正确标志不一致")
            target[row["document_id"]] = row
    if left.keys() != right.keys():
        raise DataValidationError("配对 ID 未一一覆盖")
    buckets = {
        name: [] for name in ("both_correct", "c1_only_correct", "c2_only_correct", "both_wrong")
    }
    samples = {
        name: [] for name in ("corrected_c1", "new_errors", "common_errors", "disagreements")
    }
    for doc_id, a in left.items():
        b = right[doc_id]
        if a["gold"] != b["gold"]:
            raise DataValidationError("配对 gold 不一致")
        key = (
            "both_correct"
            if a["correct"] and b["correct"]
            else "c1_only_correct"
            if a["correct"]
            else "c2_only_correct"
            if b["correct"]
            else "both_wrong"
        )
        buckets[key].append(doc_id)
        pair = {"document_id": doc_id, "gold": a["gold"], "c1": a, "c2": b}
        if key == "c2_only_correct":
            samples["corrected_c1"].append(pair)
        elif key == "c1_only_correct":
            samples["new_errors"].append(pair)
        elif key == "both_wrong":
            samples["common_errors"].append(pair)
        if a["prediction"] != b["prediction"]:
            samples["disagreements"].append(pair)
    return {
        "phase": "post_test_analysis_only",
        "sample_count": len(left),
        "counts": {k: len(v) for k, v in buckets.items()},
        "document_ids": buckets,
        "samples": samples,
    }
