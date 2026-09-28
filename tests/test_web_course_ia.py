"""页面职责、兼容路由与只读 view；不训练或执行正式评价。"""

import json
from hashlib import sha256
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings

ROUTES = [
    "/",
    "/segmentation",
    "/sequence-labeling",
    "/tagging",
    "/information-extraction",
    "/clustering",
    "/classification",
    "/dependency-parsing",
    "/deep-learning",
    "/integrated-analysis",
]


class Navigation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.links = []
        self.active = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "nav" and attrs.get("aria-label") == "主导航":
            self.inside = True
        if self.inside and tag == "a":
            self.links.append(attrs["href"])
            if attrs.get("aria-current") == "page":
                self.active.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "nav":
            self.inside = False


@pytest.mark.parametrize("route", ROUTES)
def test_single_navigation_and_canonical_page(route, tmp_path):
    response = TestClient(create_app(Settings(tmp_path))).get(route)
    assert response.status_code == 200
    navigation = Navigation()
    navigation.feed(response.text)
    assert navigation.links == ROUTES
    assert navigation.active == [route]
    assert 'id="drawer-toggle"' in response.text


@pytest.mark.parametrize(
    "old,new", [("/sequence", "/segmentation"), ("/analysis", "/integrated-analysis")]
)
def test_compatibility_redirect(old, new, tmp_path):
    response = TestClient(create_app(Settings(tmp_path))).get(old, follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == new


def test_task_and_model_views_have_distinct_responsibilities(tmp_path):
    client = TestClient(create_app(Settings(tmp_path)))
    pages = {route: client.get(route).text for route in ROUTES}
    assert [r for r, text in pages.items() if 'id="comparison-table"' in text] == ["/segmentation"]
    assert 'id="ner-result"' not in pages["/segmentation"]
    assert 'id="ner-metrics"' in pages["/tagging"]
    assert 'id="run-select"' not in pages["/segmentation"]
    assert "NER / 课程样本 Run" in pages["/tagging"]
    for route in ["/sequence-labeling", "/deep-learning"]:
        assert 'id="model-specific-parameters"' in pages[route]
        assert "js/model-page.js" in pages[route]
    for route in ["/information-extraction", "/dependency-parsing"]:
        assert "待后续实验接入" in pages[route]
        assert 'id="analysis-form"' not in pages[route]
    assert client.get("/annotation/sequence").status_code == 200


def test_cws_view_reads_extension_errors_without_mutation(tmp_path, monkeypatch):
    import app.api.views as views

    directory = tmp_path / "outputs/exp1/extensions/final"
    directory.mkdir(parents=True)
    errors = directory / "errors.jsonl"
    errors.write_text(
        json.dumps({"sample_id": "E1", "text": "课程🙂", "category": "OOV"}) + "\n",
        encoding="utf-8",
    )
    before = sha256(errors.read_bytes()).hexdigest()
    monkeypatch.setattr(
        views, "extension_status", lambda _: {"items": [{"model": "bigram", "run_id": "final"}]}
    )
    client = TestClient(create_app(Settings(tmp_path)))
    response = client.get("/api/v1/views/cws/errors?model=bigram&category=OOV")
    assert response.status_code == 200
    data = response.json()
    assert data["read_only"] is True and data["evaluation_task"] == "cws"
    assert data["total"] == 1 and data["items"][0]["text"] == "课程🙂"
    assert client.get("/api/v1/views/cws/errors?model=bigram&category=invalid").status_code == 422
    assert client.get("/api/v1/views/cws/errors?model=bigram&page=0").status_code == 422
    assert client.patch("/api/v1/views/cws/errors", json={"category": "Other"}).status_code == 405
    assert sha256(errors.read_bytes()).hexdigest() == before


def test_ner_run_cannot_supply_cws_gold(tmp_path):
    directory = tmp_path / "outputs/exp1/course_samples/ner"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps({"run_id": "ner", "run_type": "course_samples"}), encoding="utf-8"
    )
    client = TestClient(create_app(Settings(tmp_path)))
    response = client.get("/api/v1/views/cws/errors?run_id=ner")
    assert response.status_code == 503
    assert "没有 CWS Gold" in response.json()["error"]["message"]
