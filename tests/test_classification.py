"""Toy-only contract tests for experiment 3; never fit on frozen THUCNews."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from app.main import create_app
from nlplab.classification.__main__ import artifact_manifest, verify_artifacts
from nlplab.classification.experiment import fit_final, select_candidates
from nlplab.classification.linear import LinearTextClassifier
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.classification_input import ClassificationDocument, FrozenClassificationData
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig


def _toy_rows():
    train, train_tokens, val, val_tokens, test, test_tokens = [], [], [], [], [], []
    for label in CATEGORIES:
        for variant in ("a", "b"):
            train.append(
                ClassificationDocument(
                    f"train-{label}-{variant}", f"toy {label} {variant}", label
                )
            )
            train_tokens.append([f"topic_{label}", "新闻", f"variant_{variant}"])
        val.append(ClassificationDocument(f"val-{label}", f"toy val {label}", label))
        val_tokens.append([f"topic_{label}", f"val_only_{label}"])
        test.append(ClassificationDocument(f"test-{label}", f"toy test {label}", label))
        test_tokens.append([f"topic_{label}", f"test_only_{label}"])
    return train, train_tokens, val, val_tokens, test, test_tokens


def _toy_config():
    return {
        "representation": {
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
            "sublinear_tf": True,
            "smooth_idf": True,
            "norm": "l2",
        },
        "candidates": {"C0": [0.1], "C1": [1.0]},
    }


def test_frozen_classification_splits_match_pinned_manifest_without_fitting():
    root = Path(__file__).resolve().parents[1]
    data = FrozenClassificationData(root)
    provenance = data.provenance()

    assert provenance["dataset_fingerprint"] == (
        "43f2a6a8237dfe0d4cb6834dc22a126da8f7fc92272d58ee5433803e7e896922"
    )
    assert provenance["split_manifest_sha256"] == (
        "4dd233114f2192d155ff3eb76267a59d524992f42af189ae34fb8427ea77d200"
    )
    assert provenance["selection_manifest_sha256"] == (
        "23f5a4aa04f65eabe96f3fdd62f32e0aa29e87fe0526ed03cd81a97182992928"
    )
    assert provenance["file_sha256"] == {
        "train.jsonl": "ee9c7254bd851cda8b58f0888c98595e9aeeab4b04c68a10b3d611e613c302ab",
        "val.jsonl": "b98c1069b32fceb9c9bf0970e7d47ba12ea654381c529182d2d1b1587a7bb362",
        "test.jsonl": "4bda6f1f34fa2252b86a675f92e54906a5174fb0c21d69e4eb8a8de10b554ea5",
        "clustering_all.jsonl": "cf10145e55fd99c4bcce2b6208134bc0923deee59ad6809a432ef19710a1c378",
    }

    train, val, test = (data.load(split) for split in ("train", "val", "test"))
    assert [len(train), len(val), len(test)] == [420, 90, 90]
    assert Counter(doc.label for doc in train) == Counter({label: 84 for label in CATEGORIES})
    assert Counter(doc.label for doc in val) == Counter({label: 18 for label in CATEGORIES})
    assert Counter(doc.label for doc in test) == Counter({label: 18 for label in CATEGORIES})
    train_ids, val_ids, test_ids = ({doc.id for doc in split} for split in (train, val, test))
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    assert len(train_ids | val_ids | test_ids) == 600
    assert data.opened_splits == ["train", "val", "test"]


def test_candidate_and_final_tfidf_fit_only_their_allowed_toy_tokens(monkeypatch):
    train, train_tokens, val, val_tokens, _test, test_tokens = _toy_rows()
    fit_inputs, transform_inputs, fit_matrices = [], [], []
    original_fit_transform = TextTFIDF.fit_transform
    original_transform = TextTFIDF.transform
    original_model_fit = LinearTextClassifier.fit

    def record_fit_transform(self, tokens):
        fit_inputs.append([list(row) for row in tokens])
        return original_fit_transform(self, tokens)

    def record_transform(self, tokens):
        transform_inputs.append([list(row) for row in tokens])
        return original_transform(self, tokens)

    def record_model_fit(self, vectors, labels):
        fit_matrices.append(vectors)
        return original_model_fit(self, vectors, labels)

    monkeypatch.setattr(TextTFIDF, "fit_transform", record_fit_transform)
    monkeypatch.setattr(TextTFIDF, "transform", record_transform)
    monkeypatch.setattr(LinearTextClassifier, "fit", record_model_fit)

    config = _toy_config()
    representation, candidate_train, candidate_val, selection = select_candidates(
        train, val, train_tokens, val_tokens, config
    )
    assert fit_inputs == [train_tokens]
    assert transform_inputs == [val_tokens]
    assert "val_only_finance" not in representation.vectorizer.vocabulary_
    assert not any(
        token.startswith("test_only_")
        for tokens in fit_inputs
        for document_tokens in tokens
        for token in document_tokens
    )
    assert fit_matrices[0] is candidate_train is fit_matrices[1]
    assert selection["test_opened"] is False
    assert selection["fit_ids"] == [doc.id for doc in train]
    assert selection["validation_ids"] == [doc.id for doc in val]

    final_representation, final_vectors, models, _ = fit_final(
        train + val, train_tokens + val_tokens, config, selection["selected"]
    )
    assert fit_inputs == [train_tokens, train_tokens + val_tokens]
    assert "val_only_finance" in final_representation.vectorizer.vocabulary_
    assert not any(
        token.startswith("test_only_")
        for tokens in fit_inputs
        for document_tokens in tokens
        for token in document_tokens
    )
    assert fit_matrices[2] is final_vectors is fit_matrices[3]
    assert set(models) == {"C0", "C1"}


def test_both_linear_models_are_deterministic_on_a_five_class_toy_fixture():
    train, train_tokens, _val, _val_tokens, _test, _test_tokens = _toy_rows()
    vectors = TextTFIDF(TFIDFConfig(min_df=1, max_df=1.0)).fit_transform(train_tokens)
    labels = [doc.label for doc in train]

    for model_id, parameter in (("C0", 0.1), ("C1", 1.0)):
        first = LinearTextClassifier(model_id, parameter).fit(vectors, labels)
        second = LinearTextClassifier(model_id, parameter).fit(vectors, labels)
        np.testing.assert_array_equal(first.predict(vectors), second.predict(vectors))
        np.testing.assert_allclose(
            first.scores(vectors), second.scores(vectors), rtol=0, atol=1e-12
        )
        assert set(first.predict(vectors)) == set(CATEGORIES)


def test_zero_vector_tie_prediction_matches_sklearn_class_order():
    train, train_tokens, _val, _val_tokens, _test, _test_tokens = _toy_rows()
    vectors = TextTFIDF(TFIDFConfig(min_df=1, max_df=1.0)).fit_transform(train_tokens)
    labels = [doc.label for doc in train]
    zero = csr_matrix((1, vectors.shape[1]), dtype=vectors.dtype)
    estimators = {
        "C0": MultinomialNB(alpha=0.1, force_alpha=True),
        "C1": LinearSVC(
            C=1.0,
            random_state=42,
            dual=True,
            tol=1e-4,
            max_iter=10000,
            penalty="l2",
            loss="squared_hinge",
            multi_class="ovr",
        ),
    }

    for model_id, estimator in estimators.items():
        estimator.fit(vectors, labels)
        model = LinearTextClassifier(model_id, 0.1 if model_id == "C0" else 1.0).fit(
            vectors, labels
        )
        np.testing.assert_array_equal(model.predict(zero), estimator.predict(zero))

        # Force an exact all-class tie: sklearn resolves it by its sorted classes_ order.
        model.weights = np.zeros_like(model.weights)
        model.intercept = np.zeros_like(model.intercept)
        if model_id == "C0":
            estimator.class_log_prior_[:] = 0
        else:
            estimator.coef_[:] = 0
            estimator.intercept_[:] = 0
        assert len(set(model.scores(zero)[0])) == 1
        np.testing.assert_array_equal(model.predict(zero), estimator.predict(zero))
        assert model.predict(zero).tolist() == [sorted(CATEGORIES)[0]]


def test_classification_metrics_use_fixed_label_order_and_gold_rows():
    gold = ["finance", "education", "science", "sports", "entertainment"]
    predicted = ["finance", "science", "science", "entertainment", "entertainment"]

    metrics = classification_metrics(gold, predicted)

    assert metrics["label_order"] == list(CATEGORIES)
    assert metrics["confusion_matrix_axes"] == {"rows": "gold", "columns": "prediction"}
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    assert metrics["sample_count"] == 5
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["micro_f1"] == pytest.approx(0.6)
    assert metrics["macro_precision"] == pytest.approx(0.4)
    assert metrics["macro_recall"] == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(7 / 15)
    assert metrics["weighted_f1"] == pytest.approx(7 / 15)
    assert [row["label"] for row in metrics["per_class"]] == list(CATEGORIES)
    assert [row["support"] for row in metrics["per_class"]] == [1] * 5


@pytest.mark.parametrize(("model_id", "parameter"), [("C0", 0.1), ("C1", 1.0)])
def test_linear_model_numeric_save_load_and_explanation_contributions(
    tmp_path, model_id, parameter
):
    train, train_tokens, _val, _val_tokens, _test, _test_tokens = _toy_rows()
    representation = TextTFIDF(TFIDFConfig(min_df=1, max_df=1.0))
    vectors = representation.fit_transform(train_tokens)
    labels = [doc.label for doc in train]
    model = LinearTextClassifier(model_id, parameter).fit(vectors, labels)
    path = tmp_path / model_id
    model.save(path)
    restored = LinearTextClassifier.load(path)

    np.testing.assert_allclose(restored.scores(vectors), model.scores(vectors), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(restored.predict(vectors), model.predict(vectors))

    row = vectors.getrow(0)
    explanation = restored.explain(row, representation.terms, top_n=100)
    scores = restored.scores(row)[0]
    ranked = np.argsort(-scores, kind="stable")
    best, runner_up = map(int, ranked[:2])
    direct_sum = float(row.data @ restored.weights[best, row.indices])
    contrast_sum = float(
        row.data @ (restored.weights[best, row.indices] - restored.weights[runner_up, row.indices])
    )

    assert explanation["prediction"] == restored.predict(row)[0]
    assert explanation["feature_sum"] == pytest.approx(direct_sum)
    assert explanation["reconstructed_score"] == pytest.approx(scores[best])
    assert explanation["contrast_feature_sum"] == pytest.approx(contrast_sum)
    assert explanation["contrast_intercept"] + explanation["contrast_feature_sum"] == pytest.approx(
        scores[best] - scores[runner_up]
    )
    if model_id == "C0":
        assert explanation["contribution_method"] == "x_j * (log_theta_pred - log_theta_runner_up)"
        assert sum(score["value"] for score in explanation["scores"]) == pytest.approx(1.0)
        assert sum(
            feature["runner_up_contrast"] for feature in explanation["top_contributing_features"]
        ) == pytest.approx(contrast_sum)
    else:
        assert explanation["contribution_method"] == "x_j * w_pred,j"
        assert sum(
            feature["direct_contribution"] for feature in explanation["top_contributing_features"]
        ) == pytest.approx(direct_sum)


def test_classifier_artifact_manifest_detects_tampering(tmp_path):
    run = tmp_path / "toy-run"
    run.mkdir()
    (run / "result.json").write_text('{"ok":true}\n', encoding="utf-8")
    (run / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest(run)), encoding="utf-8"
    )

    assert verify_artifacts(run)["files"]["result.json"]
    (run / "result.json").write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="artifact 指纹不符"):
        verify_artifacts(run)


def test_classification_api_schema_and_mocked_service(tmp_path):
    application = create_app(Settings(root=tmp_path), service=object())
    fake = type(
        "FakeClassificationService",
        (),
        {
            "list_runs": lambda self: [{"run_id": "toy-run"}],
            "predict": lambda self, text, model_id, run_id: {
                "text": text,
                "model_id": model_id,
                "run_id": run_id,
                "prediction": "finance",
            },
        },
    )()
    application.state.classification = fake

    with TestClient(application) as client:
        assert client.get("/api/v1/experiments/3/runs").json() == [{"run_id": "toy-run"}]
        response = client.post(
            "/api/v1/experiments/3/predict",
            json={"text": "toy news", "model_id": "C1", "run_id": "toy-run"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "text": "toy news",
            "model_id": "C1",
            "run_id": "toy-run",
            "prediction": "finance",
        }
        assert client.post(
            "/api/v1/experiments/3/predict", json={"text": "toy", "model_id": "C2"}
        ).status_code == 422

        schema = client.get("/openapi.json").json()
        request_schema = schema["paths"]["/api/v1/experiments/3/predict"]["post"][
            "requestBody"
        ]["content"]["application/json"]["schema"]
        prediction_input = schema["components"]["schemas"][request_schema["$ref"].rsplit("/", 1)[1]]
        properties = prediction_input["properties"]
        assert properties["text"]["minLength"] == 1
        assert properties["text"]["maxLength"] == 5000
        assert properties["model_id"]["enum"] == ["C0", "C1"]
        assert properties["run_id"]["default"] == "latest"


def test_classification_api_reports_missing_run_as_503(tmp_path):
    application = create_app(Settings(root=tmp_path), service=object())

    with TestClient(application) as client:
        response = client.get("/api/v1/experiments/3/overview")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "resource_unavailable"
