"""玩具语料验证词级 N-gram 与 LSA 表示的拟合、对照和存档。"""

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, load_npz

from nlplab.clustering import experiment as clustering_experiment
from nlplab.clustering.kmeans import KMeansClusterer, KMeansConfig
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.clustering_input import ClusteringDocument
from nlplab.features.lsa import LSAConfig, TextLSA
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig
from nlplab.services.clustering_service import ClusteringService


def test_r1_tfidf_uses_real_bigrams_filters_by_df_and_round_trips(tmp_path):
    assert TFIDFConfig().ngram_range == (1, 1)
    tokenized = [
        ["shared", "market", "trend", "pair"],
        ["shared", "market", "growth", "pair"],
        ["other", "market", "risk", "pair"],
        ["other", "market", "gain", "pair"],
    ]
    config = TFIDFConfig(
        min_df=2,
        max_df=0.5,
        max_features=100,
        ngram_range=(1, 2),
    )
    first = TextTFIDF(config)
    first_vectors = first.fit_transform(tokenized)
    second = TextTFIDF(config)
    second_vectors = second.fit_transform(tokenized)

    terms = set(first.terms)
    assert "shared market" in terms
    assert "market" not in terms  # DF=4 exceeds the configured 0.5 cutoff.
    assert "trend" not in terms  # DF=1 is below min_df.
    assert "market trend" not in terms
    np.testing.assert_allclose(first_vectors.toarray(), second_vectors.toarray())

    path = tmp_path / "r1-tfidf"
    first.save(path)
    restored = TextTFIDF.load(path)
    restored_vectors = restored.transform(tokenized)
    np.testing.assert_allclose(restored_vectors.toarray(), first_vectors.toarray())
    assert set(restored.terms) == terms
    assert restored.config.ngram_range == (1, 2)


def test_lsa_is_deterministic_l2_normalized_and_round_trips(tmp_path):
    word_vectors = csr_matrix(
        [
            [1.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.8, 0.2, 0.0],
        ]
    )
    assert LSAConfig() == LSAConfig(100, 7, 42, "randomized")
    config = LSAConfig(n_components=3, n_iter=7, random_state=42)
    first, second = TextLSA(config), TextLSA(config)
    latent = first.fit_transform(word_vectors)
    other_latent = second.fit_transform(word_vectors)

    assert latent.shape == (word_vectors.shape[0], config.n_components)
    assert isinstance(latent, np.ndarray)
    np.testing.assert_allclose(np.linalg.norm(latent, axis=1), np.ones(len(latent)))
    np.testing.assert_allclose(latent, other_latent, atol=1e-10)
    np.testing.assert_allclose(first.transform(word_vectors), latent, atol=1e-8)

    path = tmp_path / "lsa"
    first.save(path)
    restored = TextLSA.load(path)
    np.testing.assert_allclose(restored.transform(word_vectors), latent, atol=1e-8)
    metadata = restored.metadata()
    assert metadata["input_dimension"] == word_vectors.shape[1]
    assert metadata["output_dimension"] == config.n_components
    assert metadata["purpose"] == "clustering_input"
    assert metadata["algorithm"] == "randomized"


@pytest.mark.parametrize("n_components", [0, 1, 6])
def test_lsa_rejects_invalid_dimensions(n_components):
    vectors = csr_matrix(np.eye(6, dtype=np.float64))
    with pytest.raises(DataValidationError, match="LSA 维度"):
        TextLSA(LSAConfig(n_components=n_components)).fit_transform(vectors)


def _comparison_fixture():
    groups = [
        ("finance", ["bank", "market", "stock", "profit"]),
        ("sports", ["football", "team", "score", "league"]),
        ("education", ["school", "student", "teacher", "lesson"]),
    ]
    documents, gold, tokenized = [], [], []
    for label, terms in groups:
        for index in range(4):
            document_id = f"{label}-{index}"
            tokens = [*terms, "report", "update", f"edition{index}"]
            text = " ".join(tokens)
            documents.append(ClusteringDocument(document_id, text))
            tokenized.append(tokens)
            gold.append({"id": document_id, "text": text, "primary_label": label})
    return documents, gold, tokenized


def _config(representation_id, *, lsa=False):
    return {
        "representation_id": representation_id,
        "baseline_run_id": "virtual-baseline",
        "preprocessing": {
            "unicode_normalization": "NFKC",
            "lowercase": True,
            "min_token_length": 2,
            "token_pattern": r"[\u3400-\u9fff]+|[a-zA-Z][a-zA-Z0-9]*",
            "stopwords": [],
        },
        "representation": asdict(
            TFIDFConfig(
                min_df=1,
                max_df=0.95,
                max_features=200,
                ngram_range=(1, 2),
            )
        ),
        "lsa": asdict(LSAConfig(n_components=3, n_iter=7, random_state=42)) if lsa else None,
        "kmeans": asdict(KMeansConfig(n_clusters=2, n_init=2, max_iter=80, random_state=42)),
        "k_values": [2],
        "stability_seeds": [42, 43],
        "reference_k": 2,
        "projection": {"n_components": 2, "n_iter": 5, "random_state": 42},
    }


def test_r1_r2_batch_keeps_spaces_separate_and_persists_without_moving_pointers(
    tmp_path, monkeypatch
):
    settings = Settings(tmp_path)
    service = ClusteringService(settings)

    exp1_latest = settings.outputs / "exp1" / "latest.json"
    exp1_latest.parent.mkdir(parents=True)
    exp1_latest.write_bytes(b'{"run_id":"virtual-exp1"}\n')
    exp1_latest_before = exp1_latest.read_bytes()

    exp2_dir = settings.outputs / "exp2"
    virtual_baseline = exp2_dir / "virtual-baseline"
    virtual_baseline.mkdir(parents=True)
    baseline_run = virtual_baseline / "run.json"
    baseline_run.write_bytes(b'{"run_id":"virtual-baseline"}\n')
    baseline_before = baseline_run.read_bytes()
    exp2_latest = exp2_dir / "latest.json"
    exp2_latest.write_bytes(b'{"run_id":"virtual-baseline"}\n')
    exp2_latest_before = exp2_latest.read_bytes()

    documents, gold, tokenized = _comparison_fixture()
    assert tuple(field.name for field in fields(ClusteringDocument)) == ("id", "text")
    events, fit_shapes, captured_executions, representation_inputs = [], [], [], []

    original_fit = KMeansClusterer.fit_predict

    def tracked_fit(clusterer, vectors):
        fit_shapes.append((clusterer.config.random_state, vectors.shape))
        events.append(("fit", clusterer.config.random_state))
        return original_fit(clusterer, vectors)

    original_execute = clustering_experiment.execute_unsupervised

    def capture_execution(*args, **kwargs):
        execution = original_execute(*args, **kwargs)
        captured_executions.append(execution)
        events.append(("representation-complete", len(captured_executions)))
        return execution

    from nlplab.features.text_tfidf import TextTFIDF

    original_tfidf_fit = TextTFIDF.fit_transform

    def inspect_representation_input(representation, values):
        assert all(isinstance(document_tokens, list) for document_tokens in values)
        assert all(
            isinstance(token, str) for document_tokens in values for token in document_tokens
        )
        representation_inputs.append(values)
        return original_tfidf_fit(representation, values)

    def gold_loader():
        assert len(fit_shapes) == 4
        assert len(captured_executions) == 2
        events.append(("gold",))
        return gold, {"phase": "posthoc after R1 and R2 toy fits"}

    monkeypatch.setattr(KMeansClusterer, "fit_predict", tracked_fit)
    monkeypatch.setattr(clustering_experiment, "execute_unsupervised", capture_execution)
    monkeypatch.setattr(TextTFIDF, "fit_transform", inspect_representation_input)

    configs = [_config("R1"), _config("R2", lsa=True)]
    run_infos = service.run_representation_batch(
        documents,
        lambda text: (_ for _ in ()).throw(AssertionError("cached tokens must be reused")),
        configs,
        gold_loader=gold_loader,
        dataset={"name": "synthetic comparison", "count": len(documents)},
        tokenizer_info={"name": "toy cached tokenizer", "training_executed": False},
        tokenized=tokenized,
        is_demo=True,
    )

    assert events == [
        ("fit", 42),
        ("fit", 43),
        ("representation-complete", 1),
        ("fit", 42),
        ("fit", 43),
        ("representation-complete", 2),
        ("gold",),
    ]
    assert len(representation_inputs) == 2
    assert len(run_infos) == 2
    assert run_infos[0]["run_id"] != run_infos[1]["run_id"]
    assert [info["representation_id"] for info in run_infos] == ["R1", "R2"]
    assert all(info["toy_test_fit_executed"] for info in run_infos)
    assert all(not info["formal_exp2_fit_executed"] for info in run_infos)
    assert all(not info["latest_pointer_updated"] for info in run_infos)
    assert all(info["lab3_started"] is False for info in run_infos)
    assert fit_shapes[0][1][1] == fit_shapes[1][1][1]  # R1: sparse word+bigram space.
    assert fit_shapes[2][1][1] == fit_shapes[3][1][1]  # R2: reduced dense LSA space.
    assert fit_shapes[0][1][1] > fit_shapes[2][1][1]

    assert exp1_latest.read_bytes() == exp1_latest_before
    assert baseline_run.read_bytes() == baseline_before
    assert exp2_latest.read_bytes() == exp2_latest_before

    for index, info in enumerate(run_infos):
        run_dir = service.store.resolve(info["run_id"])
        model_dir = Path(info["model_path"])
        ids = json.loads((run_dir / "input_ids.json").read_text(encoding="utf-8"))
        token_rows = [
            json.loads(line)
            for line in (run_dir / "tokens.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert ids == [document.id for document in documents]
        assert [row["id"] for row in token_rows] == ids
        saved_tokens = [row["tokens"] for row in token_rows]

        tfidf = TextTFIDF.load(model_dir / "tfidf")
        word_vectors = tfidf.transform(saved_tokens)
        if index == 0:
            vectors = load_npz(model_dir / "document_vectors.npz")
            np.testing.assert_allclose(vectors.toarray(), word_vectors.toarray())
            assert not (model_dir / "lsa").exists()
        else:
            saved_word_vectors = load_npz(model_dir / "word_vectors.npz")
            np.testing.assert_allclose(saved_word_vectors.toarray(), word_vectors.toarray())
            lsa = TextLSA.load(model_dir / "lsa")
            vectors = np.load(model_dir / "document_vectors.npy", allow_pickle=False)
            np.testing.assert_allclose(lsa.transform(word_vectors), vectors, atol=1e-8)
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(len(documents)))

        model = KMeansClusterer.load(model_dir / "k2")
        predicted, distances = model.predict(vectors)
        result = service.result(info["run_id"], k=2)
        assert [row["id"] for row in result["assignments"]] == ids
        np.testing.assert_array_equal(
            predicted, [row["cluster_id"] for row in result["assignments"]]
        )
        np.testing.assert_allclose(
            distances, [row["distance"] for row in result["assignments"]], atol=1e-8
        )
        assert service.verify_artifacts(info["run_id"])["status"] == "unchanged"

    # R2 labels come from word-space member means; representatives remain measured in LSA space.
    r2 = captured_executions[1]
    result = r2["results"][2]
    assert r2["lsa"] is not None
    assert r2["vectors"].shape[1] == r2["lsa"].config.n_components
    assert r2["word_vectors"].shape[1] > r2["vectors"].shape[1]
    terms = list(r2["representation"].terms)
    assignments = result["assignments"]
    labels = np.array([row["cluster_id"] for row in assignments])
    for cluster in result["clusters"]:
        cluster_id = cluster["cluster_id"]
        mean = np.asarray(r2["word_vectors"][labels == cluster_id].mean(axis=0)).ravel()
        ranked = sorted(range(len(terms)), key=lambda i: (-mean[i], terms[i]))[:12]
        expected_terms = [terms[i] for i in ranked if mean[i] > 0]
        assert [item["term"] for item in cluster["top_terms"]] == expected_terms

    primary_model = r2["models"][2]
    document_index = {document.id: index for index, document in enumerate(documents)}
    for assignment in assignments:
        row = document_index[assignment["id"]]
        expected_distance = np.linalg.norm(
            r2["vectors"][row] - primary_model.centers[assignment["cluster_id"]]
        )
        assert assignment["distance"] == pytest.approx(expected_distance)
    for cluster in result["clusters"]:
        members = [row for row in assignments if row["cluster_id"] == cluster["cluster_id"]]
        expected_representatives = [
            item["id"]
            for item in sorted(members, key=lambda item: (item["distance"], item["id"]))[:3]
        ]
        assert [item["id"] for item in cluster["representatives"]] == expected_representatives
