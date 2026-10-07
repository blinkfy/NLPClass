"""Streaming worker/API contracts using fake processes and tokenizers only."""

import json
import subprocess
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.course_neural.token_stream import TokenStream
from nlplab.services import course_neural_service as service_module
from nlplab.services.course_neural_service import CourseNeuralService


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        decoded = {
            (1,): "你",
            (1, 2): "你好\ufffd",
            (1, 2, 3): "你好界",
        }
        return decoded[tuple(token_ids)]


def test_token_stream_skips_prompt_and_emits_cumulative_chinese_without_trailing_replacement():
    events = []
    stream = TokenStream(FakeTokenizer(), lambda event, data: events.append((event, data)))

    stream.put(FakeTensor([[90, 91]]))  # generate() sends the prompt first.
    stream.put(FakeTensor([[1]]))
    stream.put(FakeTensor([[2]]))
    stream.put(FakeTensor([[3]]))
    stream.end()

    assert events == [
        ("token", {"response": "你", "token_count": 1}),
        ("token", {"response": "你好", "token_count": 2}),
        ("token", {"response": "你好界", "token_count": 3}),
    ]


class FakeStdin:
    def __init__(self):
        self.value = ""
        self.closed = False

    def write(self, value):
        self.value += value

    def close(self):
        self.closed = True


class FakeStdout:
    def __init__(self, lines=(), *, block_after_first=False):
        self.lines = list(lines)
        self.block_after_first = block_after_first
        self.release = threading.Event()
        self.started_blocking = threading.Event()
        self.closed = False

    def __iter__(self):
        for index, line in enumerate(self.lines):
            if self.block_after_first and index == 1:
                self.started_blocking.set()
                self.release.wait(timeout=5)
            yield line

    def close(self):
        self.closed = True
        self.release.set()


class FakeProcess:
    def __init__(self, lines=(), *, returncode=0, block_after_first=False):
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(lines, block_after_first=block_after_first)
        self.returncode = returncode
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9


def _line(event, data):
    return json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n"


def _service(tmp_path):
    service = CourseNeuralService(Settings(root=tmp_path), nlp=None)
    service.runtime.parent.mkdir(parents=True)
    service.runtime.write_bytes(b"stub runtime; never executed")
    return service


def test_stream_llm_emits_token_then_caches_complete_done_result(monkeypatch, tmp_path):
    service = _service(tmp_path)
    result = {"text": "测试", "response": "你好", "tokens": [{"position": 0}],
              "source": {"model": "fake"}, "worker_elapsed_ms": 3.5}
    process = FakeProcess([
        _line("status", {"message": "ready"}),
        _line("token", {"response": "你好", "token_count": 2}),
        _line("done", result),
    ])
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(service_module.subprocess, "Popen", fake_popen)
    payload = {"text": "测试", "task": "chat", "max_new_tokens": 8, "sampling": False}

    events = list(service.stream_llm(payload, threading.Event()))

    assert [event["event"] for event in events] == ["status", "status", "token", "done"]
    assert events[2]["data"] == {"response": "你好", "token_count": 2}
    assert events[3]["data"]["response"] == "你好"
    assert events[3]["data"]["cached"] is False
    assert events[3]["data"]["elapsed_ms"] >= 0
    assert service.cache[json.dumps({"kind": "llm", **payload}, ensure_ascii=False, sort_keys=True)]["tokens"] == result["tokens"]
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-1] == "--stream"
    assert json.loads(process.stdin.value) == {"kind": "llm", **payload}
    assert process.stdin.closed
    assert process.wait_calls == [5]
    assert service.lock.locked() is False


def test_closing_stream_terminates_child_and_releases_lock(monkeypatch, tmp_path):
    service = _service(tmp_path)
    process = FakeProcess([
        _line("status", {"message": "ready"}),
        _line("token", {"response": "部分", "token_count": 1}),
    ], returncode=None, block_after_first=True)
    monkeypatch.setattr(service_module.subprocess, "Popen", lambda *args, **kwargs: process)
    iterator = service.stream_llm({"text": "测试"}, threading.Event())

    assert next(iterator)["event"] == "status"
    assert next(iterator)["data"] == {"message": "ready"}
    assert process.stdout.started_blocking.wait(timeout=2)
    iterator.close()

    assert process.terminate_called
    assert process.stdout.closed
    assert service.lock.locked() is False
    assert not service.cache


@pytest.mark.parametrize(
    ("lines", "returncode", "message"),
    [
        (["{bad json\n"], 0, "无效流式结果"),
        ([], 1, "生成中断"),
        ([_line("error", {"message": "fake worker error"})], 0, "fake worker error"),
    ],
    ids=["bad-frame", "early-eof", "worker-error"],
)
def test_stream_llm_errors_do_not_cache_and_release_lock(monkeypatch, tmp_path, lines, returncode, message):
    service = _service(tmp_path)
    process = FakeProcess(lines, returncode=returncode)
    monkeypatch.setattr(service_module.subprocess, "Popen", lambda *args, **kwargs: process)

    events = list(service.stream_llm({"text": "测试"}, threading.Event()))

    assert events[-1]["event"] == "error"
    assert message in events[-1]["data"]["message"]
    assert not service.cache
    assert service.lock.locked() is False


def test_stream_api_validates_input_and_returns_sse_headers_with_cleanup(monkeypatch, tmp_path):
    app = create_app(Settings(root=tmp_path))
    service = app.state.course_neural
    calls = []
    cleanup = []
    stop_events = []

    def fake_stream(payload, stop):
        calls.append(payload)
        stop_events.append(stop)

        def events():
            try:
                yield {"event": "token", "data": {"response": "你好"}}
                yield {"event": "done", "data": {"response": "你好", "tokens": []}}
            finally:
                cleanup.append(True)

        return events()

    monkeypatch.setattr(service, "stream_llm", fake_stream)
    with TestClient(app) as client:
        invalid = client.post("/api/v1/course/deep/llm/stream", json={"text": "", "task": "chat"})
        assert invalid.status_code == 422
        assert not calls

        response = client.post(
            "/api/v1/course/deep/llm/stream",
            json={"text": "测试", "task": "chat", "max_new_tokens": 8},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: token\ndata: {\"response\": \"你好\"}\n\n" in response.text
    assert "event: done\ndata: {\"response\": \"你好\", \"tokens\": []}\n\n" in response.text
    assert calls[0]["text"] == "测试"
    assert stop_events[0].is_set()
    assert cleanup == [True]
