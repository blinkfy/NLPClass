"""Offline service, API-schema, resource-integrity, and CRF-trace tests."""

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.course_modules import DeepInput
from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.course_neural.crf_trace import crf_trace
from nlplab.course_neural.resources import frozen_bilstm, prepared_resource, verified_bert
from nlplab.services import course_neural_service as service_module
from nlplab.services.course_neural_service import CourseNeuralService


def _write_manifest(root, resources):
    directory = root / "artifacts" / "course_models"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"resources": resources}), encoding="utf-8"
    )
    return directory


def _record(path, content):
    return {
        "path": path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fingerprint(content):
    return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _seed_frozen_bilstm(root, model_path=None):
    run_id = "20260926_185552_0ffff5d1"
    model_path = model_path or root / "artifacts/models" / run_id
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "metadata.json").write_text("{}", encoding="utf-8")
    (model_path / "state_dict.pt").write_bytes(b"toy state dict")
    directory = root / "outputs/exp1/extensions" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    info = {"run_id": run_id, "model": "bilstm-crf", "phase": "final",
            "is_demo": False, "model_path": str(model_path)}
    (directory / "run.json").write_text(json.dumps(info), encoding="utf-8")
    return directory, info, model_path


@pytest.fixture
def service_with_stub_runtime(tmp_path):
    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)
    service.runtime.parent.mkdir(parents=True)
    service.runtime.write_bytes(b"test double; never executed")
    return service


def test_partial_resource_is_not_ready(tmp_path):
    _write_manifest(
        tmp_path,
        {
            "parser": {
                "files": [
                    _record("parser/config.json", b"{}"),
                    _record("parser/model.pt", b"weights"),
                ]
            }
        },
    )
    (tmp_path / "artifacts/course_models/parser").mkdir()
    (tmp_path / "artifacts/course_models/parser/config.json").write_bytes(b"{}")

    with pytest.raises(ResourceUnavailableError):
        prepared_resource(tmp_path, "parser")

    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)
    assert service.resources()["dependency_ready"] is False


def test_resource_manifest_hash_is_verified(tmp_path):
    payload = b"model-weights"
    directory = _write_manifest(tmp_path, {"parser": {"files": [_record("parser/model.pt", payload)]}})
    model = directory / "parser/model.pt"
    model.parent.mkdir()
    model.write_bytes(payload)

    base, resource = prepared_resource(tmp_path, "parser")
    assert base == directory.resolve()
    assert resource["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()

    model.write_bytes(b"model-weightX")  # Same byte length, different digest.
    with pytest.raises(ResourceUnavailableError):
        prepared_resource(tmp_path, "parser")


def test_resource_manifest_rejects_path_escape(tmp_path):
    _write_manifest(tmp_path, {"parser": {"files": [_record("../outside.bin", b"x")]}})

    with pytest.raises(ResourceUnavailableError):
        prepared_resource(tmp_path, "parser")


def test_verified_bert_uses_pinned_revision_after_home_relocation(monkeypatch, tmp_path):
    revision = "b" * 40
    alternate_revision = "a" * 40
    filenames = ("config.json", "model.safetensors", "tokenizer_config.json",
                 "tokenizer.json", "vocab.txt")
    contents = {name: f"toy:{name}".encode() for name in filenames}
    old_snapshot = tmp_path / "recorded-home" / "snapshots" / revision
    home = tmp_path / "relocated-home"
    snapshots = home / ".cache/huggingface/hub/models--bert-base-chinese/snapshots"
    pinned = snapshots / revision
    alternate = snapshots / alternate_revision
    for directory in (pinned, alternate):
        directory.mkdir(parents=True)
        for name in filenames:
            (directory / name).write_bytes(contents[name])

    reference = {
        "revision": revision,
        "path": str(old_snapshot),
        "files": {name: _fingerprint(contents[name]) for name in filenames},
    }
    resource_dir = _write_manifest(tmp_path, {})
    (resource_dir / "bert-reference.json").write_text(json.dumps(reference), encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    resolved, record = verified_bert(tmp_path)

    assert resolved == pinned
    assert resolved != alternate
    assert record["revision"] == revision


def test_verified_bert_rejects_same_size_hash_change(monkeypatch, tmp_path):
    revision = "c" * 40
    filenames = ("config.json", "model.safetensors", "tokenizer_config.json",
                 "tokenizer.json", "vocab.txt")
    contents = {name: f"toy:{name}".encode() for name in filenames}
    home = tmp_path / "home"
    snapshot = home / ".cache/huggingface/hub/models--bert-base-chinese/snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in filenames:
        (snapshot / name).write_bytes(contents[name])
    resource_dir = _write_manifest(tmp_path, {})
    reference = {"revision": revision,
                 "files": {name: _fingerprint(contents[name]) for name in filenames}}
    (resource_dir / "bert-reference.json").write_text(json.dumps(reference), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"x" * len(contents["model.safetensors"]))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(ResourceUnavailableError):
        verified_bert(tmp_path)


def test_bilstm_readiness_requires_the_frozen_run_record(tmp_path):
    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)
    assert service.resources()["bilstm_ready"] is False

    _seed_frozen_bilstm(tmp_path)

    resources = service.resources()
    assert resources["bilstm_ready"] is True
    assert resources["training"] is False


def test_frozen_bilstm_rejects_model_outside_artifacts(tmp_path):
    _seed_frozen_bilstm(tmp_path, tmp_path / "foreign-model")

    with pytest.raises(ResourceUnavailableError):
        frozen_bilstm(tmp_path)


def test_infer_rejects_unknown_kind_before_checking_runtime(tmp_path):
    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)

    with pytest.raises(DataValidationError, match="未知课程推理任务"):
        service.infer("fit", {})


def test_infer_subprocess_result_is_cached_without_a_second_worker(monkeypatch, tmp_path):
    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)
    service.runtime.parent.mkdir(parents=True)
    service.runtime.write_bytes(b"test double; never executed")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] == 120
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        request = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"kind": request["kind"], "text": request["text"]}),
            "",
        )

    monkeypatch.setattr(service_module.subprocess, "run", fake_run)
    first = service.infer("bert", {"text": "测试"})
    second = service.infer("bert", {"text": "测试"})

    assert len(calls) == 1
    assert first == {"kind": "bert", "text": "测试", "elapsed_ms": first["elapsed_ms"], "cached": False}
    assert second == {**first, "cached": True}


def test_infer_translates_worker_timeout(monkeypatch, service_with_stub_runtime):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(service_module.subprocess, "run", timeout)

    with pytest.raises(ResourceUnavailableError, match="未完成"):
        service_with_stub_runtime.infer("bert", {"text": "玩具输入"})


def test_infer_rejects_malformed_worker_response(monkeypatch, service_with_stub_runtime):
    def malformed(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "{not-json", "")

    monkeypatch.setattr(service_module.subprocess, "run", malformed)

    with pytest.raises(ResourceUnavailableError, match="无效结果"):
        service_with_stub_runtime.infer("bert", {"text": "玩具输入"})


def test_infer_reports_busy_without_dispatching_worker(monkeypatch, service_with_stub_runtime):
    def unexpected_worker(*args, **kwargs):
        pytest.fail("a busy request must not dispatch a worker")

    monkeypatch.setattr(service_module.subprocess, "run", unexpected_worker)
    assert service_with_stub_runtime.lock.acquire(blocking=False)
    try:
        with pytest.raises(ResourceUnavailableError, match="正在处理"):
            service_with_stub_runtime.infer("bert", {"text": "玩具输入"})
    finally:
        service_with_stub_runtime.lock.release()


def test_dependency_rejects_more_than_six_sentences_before_nlp_or_worker(tmp_path):
    class NoModelCalls:
        def analyze(self, *args, **kwargs):
            pytest.fail("sentence-count rejection must happen before tokenization")

    service = CourseNeuralService(Settings(root=tmp_path), nlp=NoModelCalls())
    service.infer = lambda *args, **kwargs: pytest.fail("must reject before worker dispatch")

    with pytest.raises(DataValidationError, match="1–6"):
        service.dependency("我爱你。" * 7)


def test_dependency_binds_frozen_tokenizer_not_latest(tmp_path):
    from types import SimpleNamespace
    calls = []
    info = {"run_type": "frozen-final-benchmark", "variant": "smooth", "is_demo": False}
    nlp = SimpleNamespace(store=SimpleNamespace(read=lambda run, name: info),
                          analyze=lambda text, tasks, run, model: calls.append(run) or {"segmentation": {"tokens": [text]}})
    service = CourseNeuralService(Settings(root=tmp_path), nlp)
    service.infer = lambda *args: {"sentences": [{"heads": [0], "relations": ["root"]}]}
    result = service.dependency("猫")
    assert calls == ["20260926_130612_32a5e168_smooth"]
    assert result["tokenizer_run"] == calls[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "task": "chat", "max_new_tokens": 48},
        {"text": "x" * 501, "task": "chat", "max_new_tokens": 48},
        {"text": "x", "task": "predict", "max_new_tokens": 48},
        {"text": "x", "task": "chat", "max_new_tokens": 0},
        {"text": "x", "task": "chat", "max_new_tokens": 65},
    ],
)
def test_deep_input_schema_rejects_out_of_contract_values(payload):
    with pytest.raises(ValidationError):
        DeepInput.model_validate(payload)


def test_deep_input_schema_accepts_documented_boundaries():
    minimum = DeepInput.model_validate({"text": "x", "max_new_tokens": 1})
    maximum = DeepInput.model_validate(
        {"text": "x" * 500, "task": "extract", "max_new_tokens": 64, "sampling": True}
    )

    assert minimum.max_new_tokens == 1
    assert maximum.max_new_tokens == 64
    assert maximum.text == "x" * 500


def test_course_resources_api_reports_no_partial_models(tmp_path):
    app = create_app(Settings(root=tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/v1/course/modules/resources")
        assert response.status_code == 200
        assert response.json()["dependency_ready"] is False
        assert response.json()["llm_ready"] is False
        assert response.json()["training"] is False


def test_invalid_deep_api_post_does_not_dispatch_models(tmp_path):
    app = create_app(Settings(root=tmp_path))
    app.state.course_neural.infer = lambda *args, **kwargs: pytest.fail(
        "Pydantic must reject invalid input before inference"
    )
    with TestClient(app) as client:
        invalid = client.post(
            "/api/v1/course/deep/llm",
            json={"text": "测试", "task": "chat", "max_new_tokens": 65},
        )
        assert invalid.status_code == 422


def _sequence_score(path, emissions, transitions, start, end):
    score = start[path[0]] + emissions[0][path[0]] + end[path[-1]]
    for position in range(1, len(path)):
        score += transitions[path[position - 1]][path[position]]
        score += emissions[position][path[position]]
    return score


def test_crf_trace_matches_exhaustive_partition_and_best_path():
    emissions = [[0.2, -0.4], [-0.1, 0.8], [0.6, 0.0]]
    transitions = [[0.1, 0.7], [-0.3, 0.2]]
    start = [0.3, -0.2]
    end = [-0.1, 0.4]
    paths = list(itertools.product(range(2), repeat=len(emissions)))
    scores = [_sequence_score(path, emissions, transitions, start, end) for path in paths]

    trace = crf_trace(emissions, transitions, start, end)

    assert tuple(trace["path"]) == paths[scores.index(max(scores))]
    assert trace["path_score"] == pytest.approx(max(scores))
    expected_partition = max(scores) + math.log(sum(math.exp(score - max(scores)) for score in scores))
    assert trace["log_partition"] == pytest.approx(expected_partition)
