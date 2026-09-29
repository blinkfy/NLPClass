from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.sequence_labeling.features import crf_features, sequence_features
from nlplab.sequence_labeling.linear_crf import CRFConfig, LinearChainCRF
from nlplab.sequence_labeling.perceptron import (
    AveragedStructuredPerceptron,
    PerceptronConfig,
)
from nlplab.sequence_labeling.tags import STATES, is_legal_sequence
from nlplab.sequence_labeling.viterbi import viterbi
from nlplab.tagging.hanlp_pos import HanLPPosTagger, normalize_pos


def _toy_training_data() -> tuple[list[list[str]], list[list[str]]]:
    return (
        [list("我爱"), list("自然语"), list("汉")],
        [["S", "S"], ["B", "M", "E"], ["S"]],
    )


def test_sequence_features_cover_unicode_context_and_crf_encoding():
    rows = sequence_features("A猫😊5")

    assert len(rows) == 4
    assert {"BOS", "c[-1]=<BOS>", "c[0]=A", "c[+1]=猫", "type=ascii-letter"} <= set(rows[0])
    assert {"c[-1]=A", "c[0]=猫", "c[+1]=😊", "type=han"} <= set(rows[1])
    assert {"c[-1]=猫", "c[0]=😊", "c[+1]=5", "type=other"} <= set(rows[2])
    assert {"EOS", "c[-1]=😊", "c[0]=5", "type=ascii-digit"} <= set(rows[3])

    encoded = crf_features("A猫")
    assert len(encoded) == 2
    assert all(value is True for row in encoded for value in row.values())
    assert set(encoded[0]) == set(sequence_features("A猫")[0])


def test_viterbi_returns_only_legal_bmes_even_when_illegal_emission_is_best():
    # M is overwhelmingly preferred at position zero, but cannot start a BMES path.
    emissions = np.array([[0.0, 50.0, 0.0, 0.0], [0.0, 50.0, 0.0, 0.0]])
    trace = viterbi(np.zeros(4), np.zeros((4, 4)), emissions)

    assert len(trace.tags) == 2
    assert is_legal_sequence(trace.tags)
    assert trace.tags[0] in {"B", "S"}
    assert trace.tags[-1] in {"E", "S"}
    assert trace.score.shape == (2, len(STATES))
    assert trace.backpointer.shape == (2, len(STATES))


def test_perceptron_training_is_deterministic_updates_and_averages(tmp_path):
    observations, tags = _toy_training_data()
    config = PerceptronConfig(epochs=4, seed=19)
    first = AveragedStructuredPerceptron(config).fit(observations, tags)
    second = AveragedStructuredPerceptron(config).fit(observations, tags)

    assert first.weights == second.weights
    assert first.history == second.history
    assert any(epoch["updates"] > 0 for epoch in first.history)
    assert first.update_example is not None
    assert first.update_example["positive_features"]
    assert first.update_example["negative_features"]
    assert all(is_legal_sequence(first.predict(chars)) for chars in observations)
    assert any(not float(value).is_integer() for value in first.weights.values())

    destination = tmp_path / "perceptron"
    first.save(destination)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model"] == first.name
    restored = AveragedStructuredPerceptron.load(destination)
    assert restored.predict(list("我爱")) == first.predict(list("我爱"))
    assert restored.explain("我爱")["update_example"] == json.loads(
        json.dumps(first.update_example, ensure_ascii=False)
    )


def test_linear_chain_crf_marginals_transitions_and_save_load(tmp_path):
    observations, tags = _toy_training_data()
    model = LinearChainCRF(CRFConfig(c1=0.01, c2=0.1, max_iterations=8)).fit(
        observations, tags, tmp_path / "crf-fit"
    )

    explanation = model.explain("自然语")
    assert is_legal_sequence(explanation["tags"])
    assert len(explanation["marginals"]) == len("自然语")
    assert all(len(row) == len(STATES) for row in explanation["marginals"])
    assert all(sum(row) == pytest.approx(1.0, abs=1e-5) for row in explanation["marginals"])
    assert len(explanation["transitions"]) == len(STATES)
    assert all(len(row) == len(STATES) for row in explanation["transitions"])
    assert explanation["features"]
    assert model.explain("自然语", details=False)["transitions"] == []

    destination = tmp_path / "crf-copy"
    model.save(destination)
    restored = LinearChainCRF.load(destination)
    assert restored.predict(list("自然语")) == model.predict(list("自然语"))
    assert is_legal_sequence(restored.predict(list("我爱")))


def test_pos_normalization_uses_unicode_codepoint_spans_and_rejects_mismatch():
    text = "A 🐈!"
    items = normalize_pos(text, ["A", " ", "🐈", "!"], ["NN", "NN", "PU"])

    assert [(item["word"], item["pos"], item["start"], item["end"]) for item in items] == [
        ("A", "NN", 0, 1),
        ("🐈", "NN", 2, 3),
        ("!", "PU", 3, 4),
    ]
    assert [text[item["start"] : item["end"]] for item in items] == ["A", "🐈", "!"]
    with pytest.raises(DataValidationError):
        normalize_pos(text, ["A", " ", "🐈", "!"], ["NN"])
    with pytest.raises(DataValidationError):
        normalize_pos(text, ["A", "🐈", "!"], ["NN", "NN", "PU"])


def test_pos_unavailable_and_course_api_schema(tmp_path):
    with pytest.raises(ResourceUnavailableError, match="尚未准备"):
        HanLPPosTagger().tag("汉", ["汉"])

    app = create_app(Settings(root=tmp_path))
    with TestClient(app) as client:
        resources = client.get("/api/v1/course/resources")
        assert resources.status_code == 200
        assert resources.json() == {"perceptron": False, "crf": False, "pos": False}

        unavailable = client.post("/api/v1/course/pos", json={"text": "汉"})
        assert unavailable.status_code == 503
        assert "POS 模型尚未准备" in unavailable.json()["error"]["message"]

        invalid_model = client.post("/api/v1/course/sequence", json={"text": "汉", "model": "hmm"})
        too_long = client.post("/api/v1/course/pos", json={"text": "汉" * 501})
        assert invalid_model.status_code == 422
        assert too_long.status_code == 422
