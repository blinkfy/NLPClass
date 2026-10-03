"""Toy-only Stage2 contracts; the frozen THUCNews splits are never fit here."""

import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest

from app.main import create_app
from nlplab.classification.__main__ import verify_artifacts
from nlplab.classification.c2 import C2LinearClassifier
from nlplab.classification.stage2_experiment import (
    evaluate_stage2_test,
    fit_stage2_final,
    paired_outcomes,
    select_stage2,
    selection_key,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.char_chi2 import (
    CharacterChi2TFIDF,
    CharacterCounts,
    char_bigrams,
)
from nlplab.segmentation.hmm_segmenter import HMMSegmenter

STAGE1_RUN_ID = "20261001_123337_9c1e7341"
STAGE1_MANIFEST_SHA256 = "ddb329ec1bc5f9e0020ecff5fa35c6832091b7b8981fd9d3ac40fc2d3991aeee"
STAGE1_LATEST_SHA256 = "3988fe4fb98bedbb3ad8d70dc0300148ecefa0a143406323bb56a6dd7063f8dd"


def _toy_splits():
    train, val, test = [], [], []
    for label in CATEGORIES:
        train.extend(
            [
                ClassificationDocument(f"train-{label}-a", f"共同新闻主题{label}甲", label),
                ClassificationDocument(f"train-{label}-b", f"共同新闻主题{label}乙", label),
            ]
        )
        val.append(ClassificationDocument(f"val-{label}", f"共同新闻验证{label}", label))
        test.append(ClassificationDocument(f"test-{label}", f"共同新闻测试{label}", label))
    return train, val, test


def _candidate_config(k_values, C_values=(0.1, 0.5, 1.0, 2.0)):
    return {
        "candidates": {"k": list(k_values), "C": list(C_values)},
        "tie_break": [
            "macro_f1_desc",
            "accuracy_desc",
            "selected_feature_count_asc",
            "C_asc",
        ],
    }


def test_char_bigram_content_boundaries_nfkc_lowercase_and_overlap():
    assert char_bigrams("ＡＢＣ中文１２٣") == ["ab", "bc", "c中", "中文", "文1", "12", "2٣"]
    assert char_bigrams("漢字١٢") == ["漢字", "字١", "١٢"]
    assert unicodedata.category("١") == "Nd"
    assert char_bigrams("a-b a_b a🙂b 中!文") == []
    assert char_bigrams("aaaa") == ["aa", "aa", "aa"]
    assert char_bigrams("ÉΩa") == []


def test_character_counts_are_nonnegative_and_reject_negative_chi2_input():
    counts = CharacterCounts().fit_transform(["aaaa", "aaaa"])
    assert counts.data.tolist() == [3, 3]
    assert np.all(counts.data >= 0)

    with pytest.raises(DataValidationError, match="非负 Count"):
        CharacterChi2TFIDF().fit_from_counts(csr_matrix([[-1]]), ["x"], None)


def test_toy_candidate_grid_fit_boundaries_alignment_and_no_hmm(monkeypatch):
    train, val, test = _toy_splits()
    train_texts = [row.text for row in train]
    val_texts = [row.text for row in val]
    test_texts = [row.text for row in test]
    available = CharacterCounts().fit_transform(train_texts).shape[1]
    assert available > 1

    def fail_hmm(*args, **kwargs):
        raise AssertionError("Stage2 character analysis called HMM")

    monkeypatch.setattr(HMMSegmenter, "segment", fail_hmm)

    count_fits, count_transforms = [], []
    selector_fits, selector_transforms = [], []
    tfidf_fits, tfidf_transforms = [], []
    original_count_fit = CountVectorizer.fit_transform
    original_count_transform = CountVectorizer.transform
    original_selector_fit = SelectKBest.fit
    original_selector_transform = SelectKBest.transform
    original_tfidf_fit = TfidfTransformer.fit
    original_tfidf_transform = TfidfTransformer.transform

    def record_count_fit(self, texts, y=None):
        count_fits.append(list(texts))
        return original_count_fit(self, texts, y)

    def record_count_transform(self, texts):
        count_transforms.append(list(texts))
        return original_count_transform(self, texts)

    def record_selector_fit(self, matrix, labels=None, **params):
        selector_fits.append((matrix.shape, tuple(labels)))
        return original_selector_fit(self, matrix, labels, **params)

    def record_selector_transform(self, matrix):
        selector_transforms.append(matrix.shape)
        return original_selector_transform(self, matrix)

    def record_tfidf_fit(self, matrix, y=None):
        tfidf_fits.append(matrix.shape)
        return original_tfidf_fit(self, matrix, y)

    def record_tfidf_transform(self, matrix):
        tfidf_transforms.append(matrix.shape)
        return original_tfidf_transform(self, matrix)

    monkeypatch.setattr(CountVectorizer, "fit_transform", record_count_fit)
    monkeypatch.setattr(CountVectorizer, "transform", record_count_transform)
    monkeypatch.setattr(SelectKBest, "fit", record_selector_fit)
    monkeypatch.setattr(SelectKBest, "transform", record_selector_transform)
    monkeypatch.setattr(TfidfTransformer, "fit", record_tfidf_fit)
    monkeypatch.setattr(TfidfTransformer, "transform", record_tfidf_transform)

    config = _candidate_config([1, available, available + 1, "all"])
    selected_rep, train_counts, val_counts, train_vectors, val_vectors, selection = select_stage2(
        train, val, config
    )
    assert selection["test_opened"] is False
    assert selection["fit_ids"] == [row.id for row in train]
    assert selection["validation_ids"] == [row.id for row in val]
    assert selection["registered_candidate_count"] == 16
    assert selection["executed_model_fit_count"] == 8
    assert len(selection["candidates"]) == 16
    assert sum(row["status"] == "skipped" for row in selection["candidates"]) == 8
    assert {row["reason"] for row in selection["candidates"] if row["status"] == "skipped"} == {
        "k exceeds available features",
        "equivalent to all; all candidate retained",
    }
    assert np.all(train_counts.data >= 0)
    assert np.all(val_counts.data >= 0)
    assert train_vectors.shape == (len(train), len(selected_rep.terms))
    assert val_vectors.shape == (len(val), len(selected_rep.terms))
    np.testing.assert_array_equal(
        selected_rep.terms,
        selected_rep.raw.terms[selected_rep.selected_indices],
    )
    for column, term in enumerate(selected_rep.terms):
        raw_column = selected_rep.raw.vectorizer.vocabulary_[term]
        assert selected_rep.selected_indices[column] == raw_column

    # Stage A fits only Train; Validation is transformed through all three fitted stages.
    assert count_fits == [train_texts]
    assert count_transforms == [val_texts]
    assert len(selector_fits) == len(tfidf_fits) == 2
    assert all(shape[0] == len(train) for shape, _ in selector_fits)
    assert all(len(labels) == len(train) for _, labels in selector_fits)
    assert all(shape[0] == len(train) for shape in tfidf_fits)
    assert selector_transforms.count((len(val), available)) == 2
    assert (len(val), available) in tfidf_transforms

    final_rep, final_vectors, final_model, _ = fit_stage2_final(
        train + val, selection["selected"]
    )
    assert final_vectors.shape == (len(train) + len(val), len(final_rep.terms))
    assert count_fits[-1] == train_texts + val_texts
    assert len(count_fits) == 2
    assert selector_fits[-1][0][0] == len(train) + len(val)
    assert tfidf_fits[-1][0] == len(train) + len(val)
    assert final_model.model_id == "C2"

    test_vectors, _metrics, predictions, test_stats = evaluate_stage2_test(
        test, final_rep, final_model
    )
    assert test_vectors.shape == (len(test), len(final_rep.terms))
    assert count_transforms[-1] == test_texts
    assert len(count_transforms) == 2
    assert selector_transforms[-1][0] == len(test)
    assert tfidf_transforms[-1][0] == len(test)
    assert [row["document_id"] for row in predictions] == [row.id for row in test]
    assert test_stats["test_evaluation_count"] == 1


def test_selection_tie_break_prefers_macro_accuracy_fewer_features_then_lower_c():
    candidates = [
        {"metrics": {"macro_f1": 0.8, "accuracy": 0.89}, "selected_feature_count": 1, "C": 0.1},
        {"metrics": {"macro_f1": 0.8, "accuracy": 0.9}, "selected_feature_count": 9, "C": 0.5},
        {"metrics": {"macro_f1": 0.8, "accuracy": 0.9}, "selected_feature_count": 7, "C": 0.5},
        {"metrics": {"macro_f1": 0.8, "accuracy": 0.9}, "selected_feature_count": 7, "C": 0.1},
        {"metrics": {"macro_f1": 0.81, "accuracy": 0.1}, "selected_feature_count": 99, "C": 2.0},
    ]
    assert max(candidates, key=selection_key) is candidates[4]
    assert max(candidates[:4], key=selection_key) is candidates[3]
    assert max(candidates[:2], key=selection_key) is candidates[1]
    assert max(candidates[1:3], key=selection_key) is candidates[2]


def test_c2_representation_and_model_round_trip_exact_scores_and_score_gap(tmp_path, monkeypatch):
    train, _val, _test = _toy_splits()
    texts = [row.text for row in train]
    labels = [row.label for row in train]
    representation = CharacterChi2TFIDF(k="all")
    train_vectors = representation.fit_transform(texts, labels)
    model = C2LinearClassifier(parameter=0.5).fit(train_vectors, labels)
    representation_path, model_path = tmp_path / "representation", tmp_path / "C2"
    representation.save(representation_path)
    model.save(model_path)

    restored_rep = CharacterChi2TFIDF.load(representation_path)
    restored_model = C2LinearClassifier.load(model_path)
    raw_terms = restored_rep.raw.terms
    selected = set(restored_rep.selected_indices.tolist())
    expected_order = sorted(
        range(len(restored_rep.chi2_scores)),
        key=lambda i: (-restored_rep.chi2_scores[i], raw_terms[i]),
    )[:30]
    expected_top = [
        {
            "term": str(raw_terms[i]),
            "chi2_score": float(restored_rep.chi2_scores[i]),
            "df": int(restored_rep.raw.df[i]),
            "selected": i in selected,
        }
        for i in expected_order
    ]
    terms_reads = 0
    original_terms = CharacterCounts.terms.fget

    def count_terms_reads(raw):
        nonlocal terms_reads
        terms_reads += 1
        return original_terms(raw)

    monkeypatch.setattr(CharacterCounts, "terms", property(count_terms_reads))
    assert restored_rep.top_chi2() == expected_top
    assert terms_reads == 1

    probe = restored_rep.transform(["共同新闻主题finance甲"])
    np.testing.assert_allclose(
        restored_rep.transform(texts).toarray(), train_vectors.toarray(), rtol=0, atol=0
    )
    np.testing.assert_allclose(restored_model.scores(probe), model.scores(probe), rtol=0, atol=0)
    np.testing.assert_array_equal(restored_model.predict(probe), model.predict(probe))

    explanation = restored_model.explain(probe, restored_rep.terms, top_n=100)
    scores = restored_model.scores(probe)[0]
    ranking = sorted(range(len(CATEGORIES)), key=lambda i: (-scores[i], CATEGORIES[i]))
    best, runner_up = ranking[:2]
    direct_features = explanation["top_contributing_features"]
    relative_features = explanation["top_relative_features"]
    assert explanation["reconstructed_score"] == pytest.approx(scores[best])
    assert sum(row["direct_contribution"] for row in direct_features) == pytest.approx(
        explanation["feature_sum"]
    )
    assert explanation["intercept"] + explanation["feature_sum"] == pytest.approx(scores[best])
    assert explanation["contrast_intercept"] + explanation["contrast_feature_sum"] == pytest.approx(
        scores[best] - scores[runner_up]
    )
    assert sum(row["contribution"] for row in relative_features) == pytest.approx(
        explanation["contrast_feature_sum"]
    )


def test_paired_outcomes_cover_all_buckets_disagreements_and_reject_invalid_pairs():
    c1 = [
        {"document_id": "both", "gold": "finance", "prediction": "finance", "correct": True},
        {"document_id": "c1", "gold": "education", "prediction": "education", "correct": True},
        {"document_id": "c2", "gold": "science", "prediction": "sports", "correct": False},
        {"document_id": "wrong", "gold": "sports", "prediction": "science", "correct": False},
    ]
    c2 = [
        {"document_id": "both", "gold": "finance", "prediction": "finance", "correct": True},
        {"document_id": "c1", "gold": "education", "prediction": "science", "correct": False},
        {"document_id": "c2", "gold": "science", "prediction": "science", "correct": True},
        {"document_id": "wrong", "gold": "sports", "prediction": "entertainment", "correct": False},
    ]
    result = paired_outcomes(c1, c2)
    assert result["document_ids"] == {
        "both_correct": ["both"],
        "c1_only_correct": ["c1"],
        "c2_only_correct": ["c2"],
        "both_wrong": ["wrong"],
    }
    assert result["counts"] == {
        "both_correct": 1,
        "c1_only_correct": 1,
        "c2_only_correct": 1,
        "both_wrong": 1,
    }
    assert [row["document_id"] for row in result["samples"]["disagreements"]] == [
        "c1",
        "c2",
        "wrong",
    ]

    with pytest.raises(DataValidationError, match="ID 重复"):
        paired_outcomes(c1 + [dict(c1[0])], c2)
    changed_gold = [dict(row) for row in c2]
    changed_gold[0]["gold"] = "sports"
    changed_gold[0]["correct"] = False
    with pytest.raises(DataValidationError, match="gold"):
        paired_outcomes(c1, changed_gold)


def test_stage1_artifact_verification_is_read_only_and_hash_pinned():
    root = Path(__file__).resolve().parents[1]
    stage1 = root / "outputs/exp3" / STAGE1_RUN_ID
    manifest_path = stage1 / "artifact_manifest.json"
    latest_path = root / "outputs/exp3/latest.json"
    before = (file_sha256(manifest_path), file_sha256(latest_path))

    assert before == (STAGE1_MANIFEST_SHA256, STAGE1_LATEST_SHA256)
    assert json.loads(latest_path.read_text("utf-8")) == {"run_id": STAGE1_RUN_ID}
    manifest = verify_artifacts(stage1)
    assert manifest["algorithm"] == "sha256"
    assert len(manifest["files"]) == 32

    assert (file_sha256(manifest_path), file_sha256(latest_path)) == before


def test_stage2_api_keeps_legacy_schema_and_dispatches_c2_routes(tmp_path):
    class FakeClassificationService:
        def __init__(self):
            self.calls = []

        def list_runs(self):
            return [{"run_id": "toy-run"}]

        def predict(self, text, model_id, run_id):
            self.calls.append(("predict", text, model_id, run_id))
            return {"text": text, "model_id": model_id, "run_id": run_id}

        def predictions(self, run_id, model_id, errors_only):
            self.calls.append(("predictions", run_id, model_id, errors_only))
            return [{"model_id": model_id}]

        def features(self, run_id, model_id):
            self.calls.append(("features", run_id, model_id))
            return [{"model_id": model_id}]

        def paired(self, run_id):
            self.calls.append(("paired", run_id))
            return {"sample_count": 1}

    fake = FakeClassificationService()
    application = create_app(Settings(root=tmp_path), service=fake)
    application.state.classification = fake
    with TestClient(application) as client:
        legacy = client.post(
            "/api/v1/experiments/3/predict",
            json={"text": "toy news", "model_id": "C1", "run_id": "toy-run"},
        )
        assert legacy.status_code == 200
        assert client.post(
            "/api/v1/experiments/3/predict", json={"text": "toy", "model_id": "C2"}
        ).status_code == 422
        c2 = client.post(
            "/api/v1/experiments/3/stage2/predict",
            json={"text": "toy news", "run_id": "toy-run"},
        )
        assert c2.status_code == 200
        assert c2.json()["model_id"] == "C2"
        assert client.post(
            "/api/v1/experiments/3/stage2/predict",
            json={"text": "toy", "model_id": "C1"},
        ).status_code == 422

        assert client.get(
            "/api/v1/experiments/3/stage2/predictions?run_id=toy-run&errors_only=true"
        ).json() == [{"model_id": "C2"}]
        assert client.get("/api/v1/experiments/3/stage2/features?run_id=toy-run").json() == [
            {"model_id": "C2"}
        ]
        assert client.get("/api/v1/experiments/3/stage2/paired?run_id=toy-run").json() == {
            "sample_count": 1
        }
        schema = client.get("/openapi.json").json()

    paths = schema["paths"]
    legacy_ref = paths["/api/v1/experiments/3/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].rsplit("/", 1)[1]
    c2_ref = paths["/api/v1/experiments/3/stage2/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].rsplit("/", 1)[1]
    schemas = schema["components"]["schemas"]
    assert schemas[legacy_ref]["properties"]["model_id"]["enum"] == ["C0", "C1"]
    assert schemas[c2_ref]["properties"]["model_id"]["const"] == "C2"
    assert {call[0] for call in fake.calls} == {"predict", "predictions", "features", "paired"}
