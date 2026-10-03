"""Final representation selection is read from frozen toy metadata only."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.services.clustering_service import ClusteringService

REPRESENTATIONS = {"R0": "raw TF-IDF", "R1": "filtered TF-IDF", "R2": "LSA"}
RUN_IDS = {"R0": "run-r0", "R1": "run-r1", "R2": "run-r2"}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def build_final_fixture(root: Path, *, mismatch_run=False):
    settings = Settings(root)
    comparison_path = settings.outputs / "exp2/comparisons/toy/comparison.json"
    representations = {}
    for index, (representation_id, representation) in enumerate(REPRESENTATIONS.items()):
        run_id = RUN_IDS[representation_id]
        representations[representation_id] = {
            "run_id": run_id,
            "representation": representation,
            "config": {"reference_k": 5, "marker": representation_id},
            "sweep": [
                {
                    "k": k,
                    "stability": {"mean_ari": index + k / 100},
                    "min_cluster_size": index + k,
                    "max_cluster_size": index + 2 * k,
                }
                for k in (5, 6)
            ],
        }
        run_dir = settings.outputs / "exp2" / run_id
        write_json(run_dir / "run.json", {"run_id": run_id, "representation_id": representation_id})
        write_json(run_dir / "config.json", {"k_values": [5, 6], "reference_k": 5})
        for k, relative_path in ((5, "reference/k5.json"), (6, "results/k6.json")):
            write_json(
                run_dir / relative_path,
                {
                    "internal": {"silhouette": index + k / 1000},
                    "external": {"ari": index + k / 100},
                    "clusters": [{"cluster_id": index, "marker": f"{representation_id}-k{k}"}],
                },
            )

    comparison = {"comparison_id": "toy-comparison", "representations": representations}
    write_json(comparison_path, comparison)
    comparison_sha256 = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    selected_runs = dict(RUN_IDS)
    if mismatch_run:
        selected_runs["R2"] = "different-run"
    write_json(
        settings.outputs / "exp2/final_selection.json",
        {
            "primary_run_id": RUN_IDS["R2"],
            "primary_representation": "R2",
            "comparison_path": "exp2/comparisons/toy/comparison.json",
            "comparison_sha256": comparison_sha256,
            "representations": selected_runs,
        },
    )
    write_json(settings.outputs / "exp2/latest.json", {"run_id": RUN_IDS["R0"]})
    return settings


def test_runs_primary_comes_from_selection_and_latest_remains_independent(tmp_path):
    settings = build_final_fixture(tmp_path)

    response = TestClient(create_app(settings)).get("/api/v1/experiments/2/runs")

    assert response.status_code == 200
    assert response.json()["primary"] == RUN_IDS["R2"]
    assert response.json()["latest"] == RUN_IDS["R0"]


@pytest.mark.parametrize("k", [5, 6])
def test_summary_uses_each_representation_and_matching_k_metrics(tmp_path, k):
    settings = build_final_fixture(tmp_path)

    response = TestClient(create_app(settings)).get(
        "/api/v1/experiments/2/summary", params={"k": k}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["k"] == k
    assert body["selection"]["primary_run_id"] == RUN_IDS["R2"]
    assert [item["representation_id"] for item in body["items"]] == list(REPRESENTATIONS)
    for index, item in enumerate(body["items"]):
        representation_id = list(REPRESENTATIONS)[index]
        assert item["run_id"] == RUN_IDS[representation_id]
        assert item["internal"]["silhouette"] == index + k / 1000
        assert item["external"]["ari"] == index + k / 100
        assert item["stability"]["mean_ari"] == index + k / 100
        assert item["clusters"][0]["marker"] == f"{representation_id}-k{k}"


def test_summary_without_final_selection_is_empty(tmp_path):
    settings = Settings(tmp_path)

    response = TestClient(create_app(settings)).get("/api/v1/experiments/2/summary")

    assert response.status_code == 200
    assert response.json() == {"selection": None, "items": []}
    assert ClusteringService(settings).final_summary() == {"selection": None, "items": []}


def test_summary_rejects_comparison_hash_mismatch(tmp_path):
    settings = build_final_fixture(tmp_path)
    selection_path = settings.outputs / "exp2/final_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["comparison_sha256"] = "0" * 64
    write_json(selection_path, selection)

    response = TestClient(create_app(settings)).get("/api/v1/experiments/2/summary")

    assert response.status_code == 422
    assert "对照来源指纹不符" in response.json()["error"]["message"]


def test_summary_rejects_selection_run_that_disagrees_with_comparison(tmp_path):
    settings = build_final_fixture(tmp_path, mismatch_run=True)

    response = TestClient(create_app(settings)).get("/api/v1/experiments/2/summary")

    assert response.status_code == 422
    assert "冻结对照 Run 不一致" in response.json()["error"]["message"]


def test_clustering_page_exposes_evolution_comparison_projection_and_details(tmp_path):
    response = TestClient(create_app(Settings(tmp_path))).get("/clustering")

    assert response.status_code == 200
    for expected in (
        "表示演进",
        'id="clustering-representation-evolution"',
        "表示对照",
        'id="clustering-comparison-table"',
        'id="clustering-projection-dimensions"',
        "三维",
        "详细处理配置",
        'id="clustering-config"',
    ):
        assert expected in response.text
