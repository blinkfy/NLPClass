"""实验二的玩具规模聚类契约测试；不读取或拟合冻结 600 篇数据。"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from app.main import create_app
from nlplab.clustering.analysis import summarize_clusters
from nlplab.clustering.kmeans import KMeansClusterer, KMeansConfig
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.clustering_input import ClusteringDocument, load_clustering_input
from nlplab.evaluation.clustering import (
    external_metrics,
    internal_metrics,
    join_gold_posthoc,
)
from nlplab.features.text_tfidf import (
    ChinesePreprocessor,
    PreprocessingConfig,
    TextTFIDF,
    TFIDFConfig,
)
from nlplab.services.clustering_service import ClusteringService


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_clustering_input_rejects_labels_extra_fields_and_duplicate_ids(tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [{"id": "a", "text": "甲乙"}, {"id": "b", "text": "丙丁"}])
    documents = load_clustering_input(path)
    assert documents == [ClusteringDocument("a", "甲乙"), ClusteringDocument("b", "丙丁")]

    write_jsonl(path, [{"id": "a", "text": "甲乙", "primary_label": "finance"}])
    with pytest.raises(DataValidationError, match="只有 id/text"):
        load_clustering_input(path)

    write_jsonl(path, [{"id": "a", "text": "甲乙"}, {"id": "a", "text": "丙丁"}])
    with pytest.raises(DataValidationError, match="重复"):
        load_clustering_input(path)


def test_chinese_preprocessing_is_deterministic_after_unicode_normalization():
    calls = []

    def segment(text):
        calls.append(text)
        return ["中文", "机器"]

    processor = ChinesePreprocessor(
        segment,
        PreprocessingConfig(min_token_length=2, stopwords=("机器",)),
    )
    assert processor.tokens("ＡBC 中文机器") == ["abc", "中文"]
    assert processor.tokens("ＡBC 中文机器") == ["abc", "中文"]
    assert calls == ["中文机器", "中文机器"]


def test_tfidf_fit_transform_round_trip_and_l2_norm(tmp_path):
    tokenized = [
        ["新闻", "财经", "市场"],
        ["新闻", "财经", "银行"],
        ["体育", "足球", "联赛"],
        ["体育", "足球", "比赛"],
    ]
    transformer = TextTFIDF(TFIDFConfig(min_df=1, max_df=1.0, max_features=50))
    vectors = transformer.fit_transform(tokenized)
    norms = np.sqrt(vectors.multiply(vectors).sum(axis=1)).A1
    np.testing.assert_allclose(norms, np.ones(len(tokenized)))

    path = tmp_path / "tfidf"
    transformer.save(path)
    restored = TextTFIDF.load(path)
    round_trip = restored.transform(tokenized)
    np.testing.assert_allclose(round_trip.toarray(), vectors.toarray())
    assert list(restored.terms) == list(transformer.terms)


def test_tfidf_rejects_non_l2_and_zero_document_vectors():
    with pytest.raises(DataValidationError, match="L2"):
        TextTFIDF(TFIDFConfig(norm="l1"))

    transformer = TextTFIDF(TFIDFConfig(min_df=2, max_df=1.0))
    with pytest.raises(DataValidationError, match="零向量"):
        transformer.fit_transform([["共有"], ["共有"], ["孤词"]])


def test_kmeans_is_deterministic_and_centers_round_trip(tmp_path):
    vectors = csr_matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
            [0.0, 0.2, 0.8, 0.0],
        ]
    )
    config = KMeansConfig(n_clusters=2, n_init=5, random_state=19)
    first, second = KMeansClusterer(config), KMeansClusterer(config)
    first_labels = first.fit_predict(vectors)
    second_labels = second.fit_predict(vectors)
    assert len(first_labels) == vectors.shape[0]
    assert len(set(first_labels)) == config.n_clusters
    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_allclose(first.centers, second.centers)

    predicted, distances = first.predict(vectors)
    assert len(predicted) == len(distances) == vectors.shape[0]
    path = tmp_path / "kmeans"
    first.save(path)
    restored = KMeansClusterer.load(path)
    restored_labels, restored_distances = restored.predict(vectors)
    np.testing.assert_array_equal(restored_labels, predicted)
    np.testing.assert_allclose(restored_distances, distances)
    np.testing.assert_allclose(restored.centers, first.centers)
    with pytest.raises(FileExistsError):
        first.save(path)


def test_internal_and_external_metrics_perfect_clustering_ignore_label_permutation():
    dense = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.85, 0.15, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.95, 0.05],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.85, 0.15],
        ]
    )
    vectors = csr_matrix(dense)
    clusterer = KMeansClusterer(KMeansConfig(n_clusters=2, n_init=5, random_state=3))
    labels = clusterer.fit_predict(vectors)
    gold = ["finance"] * 4 + ["sports"] * 4

    internal = internal_metrics(vectors, dense, labels, clusterer)
    assert internal["silhouette"] > 0.9
    assert sorted(internal["cluster_sizes"]) == [4, 4]
    for value in external_metrics(labels, gold).values():
        assert value == pytest.approx(1.0)
    permuted = [1 - int(label) for label in labels]
    for value in external_metrics(permuted, gold).values():
        assert value == pytest.approx(1.0)


def test_posthoc_gold_join_uses_ids_and_checks_text_and_assignment_order():
    documents = [ClusteringDocument("a", "甲乙文本"), ClusteringDocument("b", "丙丁文本")]
    assignments = [{"id": "a", "cluster_id": 1}, {"id": "b", "cluster_id": 0}]
    gold = [
        {"id": "b", "text": "丙丁文本", "primary_label": "sports"},
        {"id": "a", "text": "甲乙文本", "primary_label": "finance"},
    ]
    assert join_gold_posthoc(documents, assignments, gold) == ["finance", "sports"]

    wrong_text = [{**gold[0]}, {**gold[1], "text": "其他文本"}]
    with pytest.raises(DataValidationError, match="不匹配"):
        join_gold_posthoc(documents, assignments, wrong_text)
    with pytest.raises(DataValidationError, match="输入顺序"):
        join_gold_posthoc(documents, list(reversed(assignments)), gold)
    with pytest.raises(DataValidationError, match="唯一 id"):
        join_gold_posthoc(documents, assignments, [gold[0], gold[0]])


def test_cluster_representatives_use_distance_then_id_for_ties():
    documents = [
        ClusteringDocument("z", "较近的文档"),
        ClusteringDocument("a", "同距离文档"),
        ClusteringDocument("m", "另一个簇"),
    ]
    vectors = csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    centers = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, clusters = summarize_clusters(
        documents, vectors, labels, centers, np.array(["alpha", "beta"]), examples=1
    )
    assert clusters[0]["representatives"][0]["id"] == "a"
    assert clusters[0]["boundary_documents"][0]["id"] == "z"


def _toy_documents():
    groups = [
        ("finance", "market bank finance profit"),
        ("sports", "football team match goal"),
        ("education", "school student class lesson"),
    ]
    documents, gold = [], []
    for label, terms in groups:
        for index in range(3):
            document_id = f"{label}-{index}"
            text = f"{terms} report update edition{index}"
            documents.append(ClusteringDocument(document_id, text))
            gold.append({"id": document_id, "text": text, "primary_label": label})
    return documents, gold


def _toy_config():
    return {
        "preprocessing": asdict(PreprocessingConfig()),
        "representation": asdict(TFIDFConfig(min_df=1, max_df=1.0, max_features=100)),
        "kmeans": asdict(KMeansConfig(n_clusters=2, n_init=2, max_iter=100, random_state=42)),
        "k_values": [2, 3],
        "stability_seeds": [42, 43],
        "reference_k": 2,
        "projection": {"n_components": 2, "n_iter": 5, "random_state": 42},
    }


def test_toy_service_fits_all_seeds_before_gold_and_preserves_prior_runs(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    service = ClusteringService(settings)
    exp1_pointer = settings.outputs / "exp1" / "latest.json"
    exp1_pointer.parent.mkdir(parents=True)
    exp1_pointer.write_text('{"run_id":"exp1-sentinel"}\n', encoding="utf-8")
    exp1_bytes = exp1_pointer.read_bytes()

    exp2_dir = settings.outputs / "exp2"
    prior_run = exp2_dir / "prior-toy-run"
    prior_run.mkdir(parents=True)
    prior_run_path = prior_run / "run.json"
    prior_run_path.write_text('{"run_id":"prior-toy-run"}\n', encoding="utf-8")
    prior_run_bytes = prior_run_path.read_bytes()
    (exp2_dir / "latest.json").write_text('{"run_id":"prior-toy-run"}\n', encoding="utf-8")

    documents, gold = _toy_documents()
    events = []
    original_fit = KMeansClusterer.fit_predict

    def tracked_fit(clusterer, vectors):
        events.append(("fit", clusterer.config.n_clusters, clusterer.config.random_state))
        return original_fit(clusterer, vectors)

    def gold_loader():
        assert len([event for event in events if event[0] == "fit"]) == 4
        events.append(("gold",))
        return gold, {"phase": "posthoc toy test"}

    monkeypatch.setattr(KMeansClusterer, "fit_predict", tracked_fit)
    info = service.run(
        documents,
        lambda text: text.split(),
        _toy_config(),
        gold_loader=gold_loader,
        dataset={"name": "synthetic toy", "count": len(documents)},
        tokenizer_info={"name": "toy segmenter", "training_executed": False},
        is_demo=True,
    )

    assert events == [
        ("fit", 2, 42),
        ("fit", 2, 43),
        ("fit", 3, 42),
        ("fit", 3, 43),
        ("gold",),
    ]
    assert info["toy_test_fit_executed"] is True
    assert info["formal_exp2_fit_executed"] is False
    assert info["lab3_started"] is False
    assert info["fit_count"] == 4
    assert exp1_pointer.read_bytes() == exp1_bytes
    assert prior_run_path.read_bytes() == prior_run_bytes
    assert (
        json.loads((exp2_dir / "latest.json").read_text(encoding="utf-8"))["run_id"]
        == info["run_id"]
    )

    overview = service.overview(info["run_id"])
    assert overview["dataset"]["count"] == len(documents)
    result = service.result(info["run_id"], k=2)
    assert len(result["assignments"]) == len(documents)
    assert all("gold_category" in assignment for assignment in result["assignments"])
    assert service.verify_artifacts(info["run_id"])["status"] == "unchanged"
    assert info["run_id"] in {item["run_id"] for item in service.list_runs()["items"]}

    config_path = service.store.resolve(info["run_id"]) / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="指纹不符"):
        service.verify_artifacts(info["run_id"])


def test_clustering_page_and_empty_runs_api_are_read_only(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only page/API must not fit models")

    monkeypatch.setattr(KMeansClusterer, "fit_predict", forbidden)
    monkeypatch.setattr(ClusteringService, "run_formal", forbidden)
    client = TestClient(create_app(Settings(tmp_path)))

    runs = client.get("/api/v1/experiments/2/runs")
    assert runs.status_code == 200
    assert runs.json() == {"items": [], "latest": None}
    page = client.get("/clustering")
    assert page.status_code == 200
    assert "文本聚类" in page.text
    assert 'id="clustering-data-content"' in page.text
