import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.schema import SequenceSample
from nlplab.services.experiment_service import ExperimentService


def test_missing_model_and_ner_are_explicit(tmp_path) -> None:
    client = TestClient(create_app(Settings(tmp_path)))
    assert client.get("/api/v1/experiments/1/runs").json()["items"] == []
    response = client.post("/api/v1/analyze", json={"text": "北京", "tasks": ["segmentation"]})
    assert response.status_code == 503
    response = client.post("/api/v1/analyze", json={"text": "北京", "tasks": ["ner"]})
    assert response.json()["ner"]["status"] == "unavailable"
    assert "entities" not in response.json()["ner"]
    assert client.post("/api/v1/analyze", json={"text": ""}).status_code == 422
    assert (
        client.post(
            "/api/v1/analyze", json={"text": "北京", "tasks": ["classification"]}
        ).status_code
        == 422
    )
    assert client.post("/api/v1/classify", json={"text": "北京"}).status_code == 404
    assert client.get("/api/v1/experiments/1/comparison").json()["rows"] == []


@pytest.mark.parametrize("page", ["/", "/sequence", "/clustering", "/classification", "/analysis"])
def test_web_pages(page, tmp_path) -> None:
    response = TestClient(create_app(Settings(tmp_path))).get(page)
    assert response.status_code == 200
    assert "NLPLab" in response.text


def test_real_run_api_consistency(toy_model, tmp_path) -> None:
    settings = Settings(tmp_path)
    run = ExperimentService(settings).run(
        toy_model,
        [SequenceSample("S1", "D1", "北京大学", ["北京大学"])],
        dataset="toy",
        is_demo=True,
    )
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/analyze", json={"text": "商品🙂和服务", "run_id": run["run_id"]}
    )
    assert response.status_code == 200
    value = response.json()
    assert "".join(value["segmentation"]["tokens"]) == value["text"]
    assert value["run"]["is_demo"]
    trace = client.get(
        "/api/v1/experiments/1/trace", params={"text": "商品🙂和服务", "run_id": run["run_id"]}
    ).json()
    assert "".join(trace["tokens"]) == "商品🙂和服务"
    assert len(trace["chunks"][0]["score"]) == len("商品🙂和服务")
    assert client.get("/api/v1/experiments/1/trace", params={"text": "中" * 31}).status_code == 422
    for endpoint in ("models/hmm/parameters", "experiments/1/metrics", "experiments/1/errors"):
        value = client.get(f"/api/v1/{endpoint}").json()
        assert value["run"]["run_id"] == run["run_id"]
    items = client.get("/api/v1/experiments/1/errors").json()["items"]
    assert items
    assert (
        client.patch("/api/v1/experiments/1/errors/S1", json={"category": "Ambiguity"}).status_code
        == 200
    )
    assert client.get("/api/v1/experiments/1/errors?category=Ambiguity").json()["total"] == 1
    assert client.get("/api/v1/models/hmm/parameters?run_id=..%2F").status_code == 422
    assert client.get("/api/v1/experiments/1/export/metrics").status_code == 200
