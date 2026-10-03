"""阶段 A 无 Test 参数；阶段 B 接收已经冻结的选择，禁止重新选参。"""

from collections import Counter
from time import perf_counter

import numpy as np

from nlplab.classification.linear import LinearTextClassifier
from nlplab.core.exceptions import DataValidationError
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig


def matrix_stats(vectors):
    return {
        "shape": list(vectors.shape),
        "vocabulary_size": vectors.shape[1],
        "nnz": int(vectors.nnz),
        "sparsity": float(1 - vectors.nnz / np.prod(vectors.shape)),
        "zero_vector_count": int(np.sum(vectors.getnnz(axis=1) == 0)),
    }


def training_token_stats(tokenized, config):
    df = Counter(word for tokens in tokenized for word in set(tokens))
    threshold = config["max_df"] * len(tokenized)
    return {
        "documents": len(tokenized),
        "token_count": sum(map(len, tokenized)),
        "unique_unigrams_before_df": len(df),
        "max_unigram_df": max(df.values(), default=0),
        "unigrams_above_max_df": [
            {"term": word, "df": count} for word, count in sorted(df.items()) if count > threshold
        ],
        "uses_labels": False,
        "representation_search_executed": False,
    }


def select_candidates(train, val, train_tokens, val_tokens, config):
    if set(d.id for d in train) & set(d.id for d in val):
        raise DataValidationError("Train/Validation IDs 有交集")
    representation = TextTFIDF(TFIDFConfig(**config["representation"]))
    start = perf_counter()
    train_vectors = representation.fit_transform(train_tokens)
    val_vectors = representation.transform(val_tokens)
    representation_seconds = perf_counter() - start
    searches, selected = {}, {}
    for model_id in ("C0", "C1"):
        candidates = []
        for parameter in config["candidates"][model_id]:
            model = LinearTextClassifier(model_id, parameter).fit(
                train_vectors, [doc.label for doc in train]
            )
            start = perf_counter()
            predicted = model.predict(val_vectors).tolist()
            inference_seconds = perf_counter() - start
            candidates.append(
                {
                    "parameter": parameter,
                    "metrics": classification_metrics([d.label for d in val], predicted),
                    "fit_seconds": model.fit_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
        # max 保留第一个并列值；规则在看 Validation 之前已声明。
        best = max(candidates, key=lambda item: item["metrics"]["macro_f1"])
        searches[model_id], selected[model_id] = candidates, best["parameter"]
    return (
        representation,
        train_vectors,
        val_vectors,
        {
            "phase": "validation_selection",
            "fit_ids": [d.id for d in train],
            "validation_ids": [d.id for d in val],
            "test_opened": False,
            "selection_metric": "macro_f1",
            "tie_break": "first_in_declared_candidate_order",
            "candidates": searches,
            "selected": selected,
            "shared_representation": True,
            "matrix_stats": {
                "train": matrix_stats(train_vectors),
                "val": matrix_stats(val_vectors),
            },
            "training_token_stats": training_token_stats(train_tokens, config["representation"]),
            "representation_seconds": representation_seconds,
        },
    )


def fit_final(train_val, tokens, config, selected):
    start = perf_counter()
    representation = TextTFIDF(TFIDFConfig(**config["representation"]))
    vectors = representation.fit_transform(tokens)
    seconds = perf_counter() - start
    models = {
        model_id: LinearTextClassifier(model_id, selected[model_id]).fit(
            vectors, [d.label for d in train_val]
        )
        for model_id in ("C0", "C1")
    }
    return representation, vectors, models, seconds


def evaluate_final(test, test_tokens, representation, models):
    start = perf_counter()
    vectors = representation.transform(test_tokens)
    transform_seconds = perf_counter() - start
    metrics, predictions, runtime = {}, {}, {}
    for model_id, model in models.items():
        start = perf_counter()
        predicted = model.predict(vectors).tolist()
        inference_seconds = perf_counter() - start
        metrics[model_id] = classification_metrics([d.label for d in test], predicted)
        start = perf_counter()
        predictions[model_id] = [
            {
                "document_id": doc.id,
                "gold": doc.label,
                "correct": doc.label == prediction,
                "text_snippet": " ".join(doc.text.split())[:240],
                **model.explain(vectors.getrow(i), representation.terms),
            }
            for i, (doc, prediction) in enumerate(zip(test, predicted))
        ]
        runtime[model_id] = {
            "fit_seconds": model.fit_seconds,
            "test_inference_seconds": inference_seconds,
            "test_explanation_seconds": perf_counter() - start,
        }
    return (
        vectors,
        metrics,
        predictions,
        {
            "test_transform_seconds": transform_seconds,
            "models": runtime,
            "test_evaluation_count_per_model": 1,
        },
    )
