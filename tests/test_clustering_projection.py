"""三维显示投影与缓存的玩具数据契约；不拟合 K-Means。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix, save_npz

from app.main import create_app
from nlplab.clustering.visualization import project_three_dimensions, save_projection_cache
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.services.clustering_service import ClusteringService


def dense_fixture():
    return np.random.default_rng(17).normal(size=(8, 5))


def prepare_toy_run(root: Path, *, sparse=True):
    settings = Settings(root)
    run_id = "toy-projection"
    run_dir = settings.outputs / "exp2" / run_id
    model_dir = settings.models / "exp2" / run_id
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    ids = [f"doc-{index}" for index in range(8)]
    vectors = dense_fixture()
    if sparse:
        save_npz(model_dir / "document_vectors.npz", csr_matrix(vectors))
    else:
        np.save(model_dir / "document_vectors.npy", vectors, allow_pickle=False)
    write_json(run_dir / "run.json", {"run_id": run_id, "representation_id": "toy"})
    write_json(run_dir / "config.json", {"projection": {"n_iter": 6, "random_state": 11}})
    write_json(run_dir / "input_ids.json", ids)
    write_json(
        run_dir / "projection.json",
        {
            "dimensions": 2,
            "method": "fixture",
            "points": [{"id": value, "x": index, "y": -index} for index, value in enumerate(ids)],
        },
    )
    return settings, run_id, run_dir, model_dir, ids, vectors


@pytest.mark.parametrize("matrix_type", ["dense", "sparse"])
def test_three_dimension_projection_is_deterministic_with_unique_ids(matrix_type):
    dense = dense_fixture()
    vectors = dense if matrix_type == "dense" else csr_matrix(dense)
    ids = [f"doc-{index}" for index in range(vectors.shape[0])]
    first = project_three_dimensions(vectors, ids, n_iter=6, random_state=11)
    second = project_three_dimensions(vectors, ids, n_iter=6, random_state=11)

    assert first["dimensions"] == 3
    assert first["purpose"] == "visualization_only"
    assert first["config"] == {"n_components": 3, "n_iter": 6, "random_state": 11}
    assert [point["id"] for point in first["points"]] == ids
    assert all(set(point) == {"id", "x", "y", "z"} for point in first["points"])
    np.testing.assert_allclose(
        [[point[key] for key in ("x", "y", "z")] for point in first["points"]],
        [[point[key] for key in ("x", "y", "z")] for point in second["points"]],
        atol=1e-10,
    )


def test_three_dimension_projection_rejects_id_mismatch_and_small_shapes():
    vectors = csr_matrix(dense_fixture())
    with pytest.raises(DataValidationError):
        project_three_dimensions(vectors, ["duplicate"] * vectors.shape[0])
    with pytest.raises(DataValidationError):
        project_three_dimensions(vectors, ["only-one-id"])
    with pytest.raises(DataValidationError):
        project_three_dimensions(csr_matrix(np.ones((2, 5))), ["a", "b"])
    with pytest.raises(DataValidationError):
        project_three_dimensions(csr_matrix(np.ones((5, 2))), [f"doc-{i}" for i in range(5)])


def _tree_hashes(path: Path):
    return {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_projection_cache_round_trip_preserves_run_model_and_latest(tmp_path, monkeypatch):
    settings, run_id, run_dir, model_dir, ids, _ = prepare_toy_run(tmp_path, sparse=True)
    baseline_dir = settings.outputs / "exp2" / "virtual-baseline"
    baseline_dir.mkdir()
    baseline_run = baseline_dir / "run.json"
    baseline_run.write_bytes(b'{"run_id":"virtual-baseline"}\n')
    exp2_latest = settings.outputs / "exp2" / "latest.json"
    exp2_latest.write_bytes(b'{"run_id":"virtual-baseline"}\n')
    exp1_latest = settings.outputs / "exp1" / "latest.json"
    exp1_latest.parent.mkdir(parents=True)
    exp1_latest.write_bytes(b'{"run_id":"exp1-toy-sentinel"}\n')

    run_before = _tree_hashes(run_dir)
    model_before = _tree_hashes(model_dir)
    baseline_before = hashlib.sha256(baseline_run.read_bytes()).hexdigest()
    exp2_latest_before = exp2_latest.read_bytes()
    exp1_latest_before = exp1_latest.read_bytes()

    monkeypatch.setattr(
        ClusteringService,
        "verify_artifacts",
        lambda self, requested_run_id="latest": {"status": "unchanged", "file_count": 1},
    )
    target = save_projection_cache(settings, run_id)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["run_id"] == run_id
    assert saved["source"]["input_dimension"] == 5
    assert saved["source"]["clustering_fit_executed"] is False
    assert len(saved["points"]) == len(ids)

    loaded = ClusteringService(settings).projection(run_id, dimensions=3)
    assert loaded == saved
    assert _tree_hashes(run_dir) == run_before
    assert _tree_hashes(model_dir) == model_before
    assert hashlib.sha256(baseline_run.read_bytes()).hexdigest() == baseline_before
    assert exp2_latest.read_bytes() == exp2_latest_before
    assert exp1_latest.read_bytes() == exp1_latest_before
    assert target.parent == settings.outputs / "exp2" / "visualizations" / run_id


def test_projection_cache_rejects_modified_source_vectors(tmp_path, monkeypatch):
    settings, run_id, _, model_dir, _, _ = prepare_toy_run(tmp_path, sparse=True)
    monkeypatch.setattr(ClusteringService, "verify_artifacts", lambda self, requested="latest": {})
    save_projection_cache(settings, run_id)
    vectors_path = model_dir / "document_vectors.npz"
    vectors_path.write_bytes(vectors_path.read_bytes() + b"tampered")

    with pytest.raises(DataValidationError, match="来源向量"):
        ClusteringService(settings).projection(run_id, dimensions=3)


def test_projection_api_selects_dimensions_and_rejects_invalid_values(tmp_path, monkeypatch):
    settings, run_id, _, _, _, _ = prepare_toy_run(tmp_path, sparse=False)
    monkeypatch.setattr(ClusteringService, "verify_artifacts", lambda self, requested="latest": {})
    save_projection_cache(settings, run_id)
    client = TestClient(create_app(settings))

    two_d = client.get(
        "/api/v1/experiments/2/projection", params={"run_id": run_id, "dimensions": 2}
    )
    three_d = client.get(
        "/api/v1/experiments/2/projection", params={"run_id": run_id, "dimensions": 3}
    )
    assert two_d.status_code == three_d.status_code == 200
    assert two_d.json()["dimensions"] == 2
    assert three_d.json()["dimensions"] == 3
    assert [point["id"] for point in three_d.json()["points"]] == [
        f"doc-{index}" for index in range(8)
    ]

    for invalid_dimensions in (1, 4):
        response = client.get(
            "/api/v1/experiments/2/projection",
            params={"run_id": run_id, "dimensions": invalid_dimensions},
        )
        assert response.status_code == 422
