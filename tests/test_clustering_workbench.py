import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nlplab.clustering.inspection import full_vector_values, inspect_centers
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.services.clustering_service import ClusteringService


@pytest.fixture
def frozen_vectors():
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 2.0],
            [2.0, 5.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 1.0]])
    components2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projection2d = vectors @ components2d.T
    projection3d = vectors * np.array([1.0, 1.0, 2.0])
    return vectors, centers, labels, projection2d, components2d, projection3d


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_inspect_centers_derives_real_distances_and_frozen_projections(frozen_vectors, sparse):
    vectors, centers, labels, projection2d, components2d, projection3d = frozen_vectors
    if sparse:
        vectors = csr_matrix(vectors)

    result = inspect_centers(vectors, centers, labels, projection2d, components2d, projection3d)

    expected_distances = np.array(
        [
            [1.0, np.sqrt(27.0)],
            [1.0, np.sqrt(27.0)],
            [np.sqrt(30.0), np.sqrt(2.0)],
            [np.sqrt(26.0), np.sqrt(2.0)],
        ]
    )
    assert np.allclose(result["distances"], expected_distances)
    assert result["nearest"] == [0, 0, 1, 1]
    assert [row["cluster_id"] for row in result["centroids"]] == [0, 1]
    assert [row["dimension"] for row in result["centroids"]] == [3, 3]
    assert np.allclose([row["norm"] for row in result["centroids"]], [1.0, np.sqrt(27.0)])
    assert result["centroids"][0]["vector_preview"] == [
        {"dimension": 1, "value": 1.0},
        {"dimension": 2, "value": 0.0},
        {"dimension": 3, "value": 0.0},
    ]
    assert np.allclose(
        [row["projection2d"] for row in result["centroids"]],
        centers @ components2d.T,
    )
    assert np.allclose(
        [row["projection3d"] for row in result["centroids"]],
        [[1.0, 0.0, 0.0], [1.0, 5.0, 2.0]],
    )
    assert all(row["member_mean_max_error"] < 1e-12 for row in result["centroids"])


def test_full_vector_values_only_exposes_dense_100d_coordinates():
    dense = np.linspace(-0.5, 0.5, 100)

    values = full_vector_values(dense)

    assert values == dense.tolist()
    assert full_vector_values(csr_matrix(dense.reshape(1, -1))) is None
    assert full_vector_values(dense[:99]) is None


def test_inspect_centers_rejects_projection_order_mismatch(frozen_vectors):
    vectors, centers, labels, projection2d, components2d, projection3d = frozen_vectors
    projection2d = projection2d.copy()
    projection2d[[0, 1]] = projection2d[[1, 0]]

    with pytest.raises(DataValidationError, match="二维坐标"):
        inspect_centers(vectors, centers, labels, projection2d, components2d, projection3d)


def test_inspect_centers_rejects_projection_dimension_mismatch(frozen_vectors):
    vectors, centers, labels, projection2d, _, projection3d = frozen_vectors
    wrong_components = np.eye(3)

    with pytest.raises(DataValidationError, match="投影基底"):
        inspect_centers(vectors, centers, labels, projection2d, wrong_components, projection3d)


def test_inspect_centers_rejects_labels_that_are_not_nearest(frozen_vectors):
    vectors, centers, _, projection2d, components2d, projection3d = frozen_vectors
    wrong_labels = np.array([1, 1, 0, 0])

    with pytest.raises(DataValidationError, match="最近存储质心"):
        inspect_centers(vectors, centers, wrong_labels, projection2d, components2d, projection3d)


def test_inspect_centers_rejects_centers_that_are_not_member_means(frozen_vectors):
    vectors, centers, labels, projection2d, components2d, projection3d = frozen_vectors
    wrong_centers = centers.copy()
    wrong_centers[0, 0] += 0.1

    with pytest.raises(DataValidationError, match="成员均值"):
        inspect_centers(vectors, wrong_centers, labels, projection2d, components2d, projection3d)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_cached_service(root):
    run_id = "toy-run"
    run_dir = root / "outputs" / "exp2" / run_id
    cache_dir = root / "outputs" / "exp2" / "visualizations" / run_id
    source_path = root / "frozen" / "source.txt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("frozen toy input\n", encoding="utf-8")

    write_json(run_dir / "run.json", {"run_id": run_id})
    write_json(run_dir / "config.json", {"k_values": [2], "reference_k": 2})
    write_json(
        run_dir / "reference" / "k2.json",
        {
            "assignments": [
                {"id": "doc-a", "cluster_id": 0, "distance": 0.25, "snippet": "sample A"},
                {"id": "doc-b", "cluster_id": 1, "distance": 0.5, "snippet": "sample B"},
            ]
        },
    )
    inspection = {
        "run_id": run_id,
        "space": "tfidf_l2",
        "dimension": 3,
        "document_count": 2,
        "documents": [
            {
                "id": "doc-a",
                "tokens": ["商品", "服务"],
                "token_count": 2,
                "top_terms": [{"term": "商品", "weight": 0.8}],
                "vector_preview": [{"dimension": 1, "value": 0.6}],
                "vector_norm": 1.0,
                "tfidf_dimension": 5,
                "input_dimension": 3,
            },
            {
                "id": "doc-b",
                "tokens": ["课程", "语言"],
                "token_count": 2,
                "top_terms": [{"term": "课程", "weight": 0.7}],
                "vector_preview": [{"dimension": 2, "value": 0.5}],
                "vector_norm": 1.0,
                "tfidf_dimension": 5,
                "input_dimension": 3,
            },
        ],
        "results": {
            "2": {
                "centroids": [
                    {"cluster_id": 0, "dimension": 3, "norm": 1.0},
                    {"cluster_id": 1, "dimension": 3, "norm": 2.0},
                ],
                "distances": [[0.25, 2.5], [3.5, 0.5]],
                "nearest": [0, 1],
            }
        },
        "provenance": {
            "source_files": {"frozen/source.txt": file_sha256(source_path)},
            "formal_fit_executed": False,
        },
    }
    inspection_path = cache_dir / "inspection.json"
    write_json(inspection_path, inspection)
    write_json(
        cache_dir / "inspection_manifest.json",
        {"inspection_sha256": file_sha256(inspection_path)},
    )
    return ClusteringService(Settings(root=root)), inspection_path, source_path


def test_service_returns_cached_inspection_and_document_details(tmp_path, monkeypatch):
    service, _, _ = make_cached_service(tmp_path)
    monkeypatch.setattr(
        "nlplab.services.clustering_service.subprocess.run",
        lambda *args, **kwargs: pytest.fail("prebuilt cache must avoid worker execution"),
    )

    inspected = service.inspection("toy-run", 2)
    document = service.document("toy-run", 2, "doc-a")

    assert inspected["run_id"] == "toy-run"
    assert inspected["k"] == 2
    assert inspected["centroids"][1]["cluster_id"] == 1
    assert inspected["provenance"]["formal_fit_executed"] is False
    assert document["tokens"] == ["商品", "服务"]
    assert document["top_terms"] == [{"term": "商品", "weight": 0.8}]
    assert document["snippet"] == "sample A"
    assert document["nearest_cluster"] == document["cluster_id"] == 0
    assert document["centroid_distances"] == [
        {"cluster_id": 0, "distance": 0.25},
        {"cluster_id": 1, "distance": 2.5},
    ]
    assert document["gold_role"] == "posthoc_only"
    assert document["formal_fit_executed"] is False
    with pytest.raises(DataValidationError, match="文档不在此冻结 Run 中"):
        service.document("toy-run", 2, "not-a-saved-document")


def test_service_rejects_tampered_inspection_cache_hash(tmp_path):
    service, inspection_path, _ = make_cached_service(tmp_path)
    inspection_path.write_text(inspection_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(DataValidationError, match="检视缓存指纹不符"):
        service.inspection("toy-run", 2)


def test_service_rejects_changed_frozen_source_hash(tmp_path):
    service, _, source_path = make_cached_service(tmp_path)
    source_path.write_text("changed after cache creation\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="冻结来源指纹不符"):
        service.inspection("toy-run", 2)


def test_service_cache_only_projection_missing_cache_does_not_spawn_worker(tmp_path, monkeypatch):
    service, _, _ = make_cached_service(tmp_path)
    monkeypatch.setattr(
        "nlplab.services.clustering_service.subprocess.run",
        lambda *args, **kwargs: pytest.fail("cache-only projection must not launch a worker"),
    )

    with pytest.raises(ResourceUnavailableError, match="不会重新拟合投影"):
        service.projection("toy-run", dimensions=3, cache_only=True)
