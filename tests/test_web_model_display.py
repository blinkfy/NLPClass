"""只用用户式短 demo、加载冻结模型；不训练、不评测官方数据。"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.trace_display import display_trace
from app.main import create_app
from nlplab.extensions.integrity import verify_frozen
from nlplab.sequence_labeling.hmm import HMMTagger

ROOT = Path(__file__).resolve().parents[1]
TEXT = "商品🙂和服务"


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "outputs/exp1/extensions/20260926_185552_0ffff5d1/run.json").is_file():
        pytest.skip("无本地冻结模型，禁止为测试训练模型")
    with TestClient(create_app()) as value:
        yield value


@pytest.mark.parametrize(
    "model,kind",
    [
        ("longest", "longest_scan"),
        ("bigram", "bigram_lattice"),
        ("hmm1-smooth", "hmm1_viterbi"),
        ("hmm2-smooth", "hmm2_pair_viterbi"),
        ("bilstm-crf", "bilstm_crf_emission"),
    ],
)
def test_demo_model_trace_schema(client, monkeypatch, model, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("本测试禁止训练")

    monkeypatch.setattr(HMMTagger, "fit", forbidden)
    if model == "bilstm-crf":
        pytest.importorskip("torch")
    response = client.get("/api/v1/experiments/1/trace", params={"text": TEXT, "model": model})
    assert response.status_code == 200
    trace = response.json()
    assert trace["trace_type"] == kind
    assert trace["text"] == TEXT and trace["character_count"] == len(TEXT)
    assert "".join(trace["tokens"]) == TEXT and len(trace["tags"]) == len(TEXT)
    if model == "longest":
        assert trace["fallback_count"] >= 1
        assert any(
            c["kind"] == "single-char fallback" for s in trace["steps"] for c in s["candidate_info"]
        )
    if model == "bigram":
        assert [e["word"] for e in trace["optimal_chain"]] == trace["tokens"]
        assert sum(e["on_best_path"] for e in trace["edges"]) == len(trace["tokens"])
        for edge in trace["edges"]:
            assert np.isclose(-np.log(edge["probability"]), edge["negative_log_cost"])
    if model in {"hmm1-smooth", "hmm2-smooth"}:
        for frame in trace.get("chunks", [trace]):
            if "score" not in frame:
                continue
            expected = (
                (len(frame["text"]), 4) if model == "hmm1-smooth" else (len(frame["text"]), 4, 4)
            )
            assert np.shape(frame["score"]) == np.shape(frame["backpointer"]) == expected
    if model == "bilstm-crf":
        assert np.shape(trace["emissions"]) == (len(TEXT), 4)
        assert np.shape(trace["crf_transitions"]) == (4, 4)
        assert trace["bmes_source"] == "model decoded path"


def test_long_demo_segmentation_without_trace(client):
    text = "课程展示文本🙂" * 5
    assert client.get("/api/v1/experiments/1/trace", params={"text": text}).status_code == 422
    response = client.post("/api/v1/analyze", json={"text": text, "tasks": ["segmentation"]})
    assert response.status_code == 200
    assert "".join(response.json()["segmentation"]["tokens"]) == text


def test_adapter_does_not_mutate_original_trace():
    original = {
        "model": "longest",
        "steps": [{"offset": 0, "candidates": ["🙂"], "chosen": "🙂"}],
        "tokens": ["🙂"],
        "tags": ["S"],
    }
    value = display_trace(original, "🙂", "longest", set())
    assert value["fallback_count"] == 1
    assert "candidate_info" not in original["steps"][0]


def test_hmm2_whitespace_chunks_keep_code_point_offsets(client):
    text = "商品 🙂服务"
    response = client.get(
        "/api/v1/experiments/1/trace", params={"text": text, "model": "hmm2-smooth"}
    )
    assert response.status_code == 200
    value = response.json()
    assert "".join(value["tokens"]) == text
    assert [chunk["offset"] for chunk in value["chunks"]] == [0, 2, 3]
    assert np.shape(value["chunks"][2]["score"]) == (3, 4, 4)
    assert value["chunks"][1]["whitespace"] is True


def test_models_and_benchmark_files_unchanged():
    base = ROOT / "artifacts/cache/extensions_frozen_before_20260926_160328.json"
    if not base.is_file():
        pytest.skip("无本地 frozen manifest")
    result = verify_frozen(
        ROOT,
        [
            base,
            ROOT / "artifacts/cache/extensions_frozen_supplement.json",
            ROOT / "outputs/exp1/extensions/acceptance_20260926_190001/extension_freeze.json",
        ],
    )
    assert result["status"] == "unchanged" and result["file_count"] == 354
