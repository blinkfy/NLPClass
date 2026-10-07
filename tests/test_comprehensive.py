"""Toy-only coverage for the Experiment 4 orchestration boundary.

All model, classifier, clustering, and worker calls are local stubs. These
tests must not load frozen weights, fit a formal model, or inspect Test data.
"""

import json
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.comprehensive.contracts import document_id
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.services.comprehensive_service import ComprehensiveAnalysisService


class StubNLP:
    def __init__(self, tokenizer=None, error=None):
        self.tokenizer = tokenizer or (lambda text: [text])
        self.error = error
        self.calls = []

    def analyze(self, text, tasks, run_id="latest", segmentation_model="hmm1-smooth"):
        self.calls.append((text, list(tasks), run_id, segmentation_model))
        if self.error is not None:
            raise self.error
        tokens = self.tokenizer(text)
        return {
            "text": text,
            "segmentation": {"model": "hmm-cws", "tokens": tokens, "tags": []},
            "ner": None,
            "classification": None,
            "run": {"run_id": run_id, "model_path": "toy/frozen-smooth"},
        }


class StubClassifier:
    def __init__(self, unavailable_for=()):
        self.unavailable_for = set(unavailable_for)
        self.calls = []

    def predict(self, text, model_id, run_id):
        self.calls.append((text, model_id, run_id))
        if text in self.unavailable_for:
            raise ResourceUnavailableError("toy classifier unavailable")
        return {
            "model_id": model_id,
            "run_id": run_id,
            "prediction": CATEGORIES[0],
            "score_type": "decision_score",
            "decision_scores": {label: float(index) for index, label in enumerate(CATEGORIES)},
            "notice": "decision scores are not calibrated probabilities",
        }


class StubNeural:
    def __init__(self):
        self.calls = []

    def dependency(self, text):
        self.calls.append(("dependency", text))
        return {"task": "dependency", "text": text}

    def infer(self, kind, payload):
        self.calls.append((kind, payload))
        return {"task": kind, "text": payload["text"]}


class StubClustering:
    def __init__(self, assignments, points):
        self.assignments = assignments
        self.points = points
        self.calls = []

    def verify_artifacts(self, run_id):
        self.calls.append(("verify", run_id))

    def result(self, run_id, k):
        self.calls.append(("result", run_id, k))
        return {"assignments": self.assignments, "clusters": [{"cluster_id": 0}]}

    def projection(self, run_id, dimensions, *, cache_only=False):
        self.calls.append(("projection", run_id, dimensions, cache_only))
        assert cache_only is True
        return {"points": self.points, "method": "cached projection"}


def make_service(tmp_path, *, nlp=None, clustering=None, neural=None, classifier=None):
    service = ComprehensiveAnalysisService(
        Settings(tmp_path),
        nlp or StubNLP(),
        course=object(),
        clustering=clustering or StubClustering([], []),
        neural=neural or StubNeural(),
    )
    service.classifier = classifier or StubClassifier()
    return service


def lexical_result(text, tokens, entities=(), *, pos=()):
    return {
        "pos": list(pos),
        "entities": list(entities),
        "modules": {
            "pos": {"status": "ok", "elapsed_ms": 1.0},
            "ner": {"status": "ok", "elapsed_ms": 1.0},
        },
        "provenance": {
            "pos": {"model": "toy-pos", "input_protocol": "HMM token spans"},
            "ner": {"model": "toy-ner", "input_protocol": "original text"},
        },
    }


def install_lexical_worker(service, factory=None):
    calls = []

    def worker(kind, payload, **kwargs):
        calls.append((kind, payload))
        assert kind == "lexical"
        assert "".join(payload["tokens"]) == payload["text"]
        return (factory or (lambda text, tokens: lexical_result(text, tokens)))(
            payload["text"], payload["tokens"]
        )

    service._run_worker = worker
    return calls


def entity(text, entity_type, start, end):
    return {"text": text, "type": entity_type, "start": start, "end": end, "score": None}


def test_original_text_offsets_survive_emoji_whitespace_and_entity_crossing_tokens(tmp_path):
    text = "🙂 在北京大学"
    tokens = ["🙂", " ", "在", "北京", "大学"]
    nlp = StubNLP(tokenizer=lambda value: tokens)
    service = make_service(tmp_path, nlp=nlp)
    entity_value = entity("北京大学", "ORG", 3, 7)
    install_lexical_worker(
        service,
        lambda value, token_values: lexical_result(value, token_values, [entity_value]),
    )

    result = service.analyze(text)

    assert result["original_text"] == text
    assert [(row["form"], row["start"], row["end"], row["whitespace"])
            for row in result["lexical"]["tokens"]] == [
        ("🙂", 0, 1, False), (" ", 1, 2, True), ("在", 2, 3, False),
        ("北京", 3, 5, False), ("大学", 5, 7, False),
    ]
    assert result["lexical"]["entities"] == [entity_value]
    assert text[result["lexical"]["entities"][0]["start"]:
                result["lexical"]["entities"][0]["end"]] == "北京大学"
    assert result["provenance"]["segmentation"]["run_id"] == service.TOKENIZER_RUN
    assert nlp.calls == [(text, ["segmentation"], service.TOKENIZER_RUN, "hmm1-smooth")]


def test_analyze_pins_frozen_c2_decision_scores_and_never_calls_optional_neural(tmp_path):
    nlp = StubNLP()
    neural = StubNeural()
    service = make_service(tmp_path, nlp=nlp, neural=neural)
    install_lexical_worker(service)

    result = service.analyze("这是一条新闻正文")

    assert nlp.calls[0][2:] == (service.TOKENIZER_RUN, "hmm1-smooth")
    assert service.classifier.calls == [
        ("这是一条新闻正文", "C2", service.CLASSIFIER_RUN)
    ]
    assert result["classification"]["model_id"] == "C2"
    assert result["classification"]["score_type"] == "decision_score"
    assert "decision_scores" in result["classification"]
    assert "probability" not in result["classification"]
    assert "probabilities" not in result["classification"]
    assert result["provenance"]["classification"]["run_id"] == service.CLASSIFIER_RUN
    assert neural.calls == []


def test_only_success_is_cached_partial_results_retry_and_are_not_fabricated(tmp_path):
    service = make_service(tmp_path)
    worker_calls = []
    mode = {"fail": True}

    def worker(kind, payload, **kwargs):
        worker_calls.append(kind)
        if mode["fail"]:
            raise ResourceUnavailableError("toy lexical resource missing")
        return lexical_result(payload["text"], payload["tokens"])

    service._run_worker = worker
    first = service.analyze("新闻正文A")
    second = service.analyze("新闻正文A")
    assert first["status"] == second["status"] == "partial"
    assert first["cached"] is second["cached"] is False
    assert first["lexical"]["pos"] == first["lexical"]["entities"] == []
    assert first["modules"]["pos"]["status"] == first["modules"]["ner"]["status"] == "unavailable"
    assert first["classification"]["status"] == "ok"
    assert worker_calls == ["lexical", "lexical"]

    mode["fail"] = False
    complete = service.analyze("新闻正文A")
    cached = service.analyze("新闻正文A")
    assert complete["status"] == "ok" and complete["cached"] is False
    assert cached["status"] == "ok" and cached["cached"] is True
    assert worker_calls == ["lexical", "lexical", "lexical"]


def test_missing_segmentation_is_partial_and_does_not_invent_tokens_or_lexical_results(tmp_path):
    nlp = StubNLP(error=ResourceUnavailableError("toy frozen HMM missing"))
    service = make_service(tmp_path, nlp=nlp)
    worker_calls = []
    service._run_worker = lambda kind, payload, **kwargs: worker_calls.append(kind)

    result = service.analyze("新闻正文")

    assert result["status"] == "partial"
    assert result["lexical"] == {"tokens": [], "pos": [], "entities": []}
    assert result["modules"]["segmentation"]["status"] == "unavailable"
    assert result["modules"]["pos"]["status"] == result["modules"]["ner"]["status"] == "unavailable"
    assert result["classification"]["status"] == "ok"
    assert worker_calls == []


@pytest.mark.parametrize(
    ("text", "maximum"),
    [("   ", 5000), ("……！？", 5000), ("正文" * 2501, 5000)],
)
def test_analyze_rejects_blank_punctuation_only_and_oversize_before_any_model_call(
    tmp_path, text, maximum
):
    service = make_service(tmp_path)
    service._run_worker = lambda *args, **kwargs: pytest.fail("worker called for invalid input")
    with pytest.raises(DataValidationError):
        service.analyze(text)
    assert len(text) > maximum or not any(char.isalnum() for char in text)
    assert service.nlp.calls == []
    assert service.classifier.calls == []


def test_custom_collection_is_temporary_and_rejects_bad_k_counts_duplicates_and_empty_text(
    tmp_path
):
    service = make_service(tmp_path)
    calls = []

    def worker(kind, payload, **kwargs):
        calls.append((kind, payload))
        assert kind == "custom"
        rows = payload["documents"]
        return {
            "assignments": [
                {"id": row["id"], "cluster_id": index % payload["k"], "distance": 0.1}
                for index, row in enumerate(rows)
            ],
            "clusters": [],
            "projection": {"points": [
                {"id": row["id"], "x": float(index), "y": 0.0}
                for index, row in enumerate(rows)
            ]},
            "representation": {"kind": "toy"},
            "provenance": {"temporary_analysis": True, "formal_run_written": False},
        }

    service._run_worker = worker
    docs = [{"text": f"自定义正文{i}", "title": f"文档{i}"} for i in range(3)]
    collection = service.custom_collection(docs, 2)
    assert collection["mode"] == "custom"
    assert collection["provenance"]["formal_run_written"] is False
    assert len(calls) == 1 and calls[0][0] == "custom"
    assert not list((tmp_path / "outputs").rglob("latest.json"))
    assert not (tmp_path / "outputs/exp4").exists()

    invalid = [
        ([{"text": "甲"}, {"text": "乙"}], 2),  # too few documents / invalid K
        ([{"text": f"正文{i}"} for i in range(3)], 1),
        ([{"text": "相同正文"}] * 3, 2),  # not more unique texts than K
        ([{"text": "……"}, {"text": "正文二"}, {"text": "正文三"}], 2),
        ([{"text": f"正文{i}"} for i in range(31)], 2),
    ]
    for rows, k in invalid:
        with pytest.raises(DataValidationError):
            service.custom_collection(rows, k)
    assert len(calls) == 1, "all invalid custom requests must fail before worker dispatch"


def write_course_splits(root):
    base = root / "data/splits/thucnews_600"
    base.mkdir(parents=True)
    rows = {
        "train.jsonl": [{"id": "train-a", "text": "Train正文A", "label": "must-not-leak"}],
        "val.jsonl": [{"id": "val-b", "text": "Validation正文B", "label": "must-not-leak"}],
        "test.jsonl": [{"id": "test-c", "text": "Test正文不可分析", "label": "must-not-leak"}],
    }
    hashes = {}
    for name, values in rows.items():
        path = base / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
                        encoding="utf-8")
        hashes[name] = file_sha256(path)
    (base / "manifest.json").write_text(json.dumps({"file_sha256": hashes}), encoding="utf-8")
    return base


def course_clustering():
    assignments = [
        {"id": "train-a", "cluster_id": 0, "snippet": "train snippet", "distance": 0.1},
        {"id": "val-b", "cluster_id": 0, "snippet": "val snippet", "distance": 0.2},
        {"id": "test-c", "cluster_id": 0, "snippet": "test snippet", "distance": 0.3},
    ]
    points = [{"id": row["id"], "x": index, "y": -index}
              for index, row in enumerate(assignments)]
    return StubClustering(assignments, points)


def test_course_collection_uses_cached_projection_and_reads_only_train_validation_whitelist(
    tmp_path, monkeypatch
):
    base = write_course_splits(tmp_path)
    clustering = course_clustering()
    service = make_service(tmp_path, clustering=clustering)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == (base / "test.jsonl").resolve():
            raise AssertionError("course analysis must never open test.jsonl")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    collection = service.course_collection()

    assert collection["provenance"] == {
        "run_id": service.CLUSTER_RUN, "frozen": True, "fit_executed": False
    }
    assert [row["eligible"] for row in collection["documents"]] == [True, True, False]
    assert all("label" not in row for row in collection["documents"])
    assert ("projection", service.CLUSTER_RUN, 2, True) in clustering.calls
    assert service._safe_course_texts() == {
        "train-a": "Train正文A", "val-b": "Validation正文B"
    }


def test_links_empty_does_not_infer_or_count_and_only_analyzed_entity_types_count(tmp_path):
    write_course_splits(tmp_path)
    clustering = course_clustering()
    classifier = StubClassifier(unavailable_for={"Validation正文B"})
    nlp = StubNLP()
    service = make_service(tmp_path, clustering=clustering, classifier=classifier, nlp=nlp)
    collection = service.course_collection()
    worker_calls = []

    def worker(kind, payload, **kwargs):
        worker_calls.append((kind, payload["text"]))
        text = payload["text"]
        spans = {
            "Train正文A": [entity("Train", "PER", 0, 5)],
            "Validation正文B": [entity("Validation", "LOC", 0, 10),
                                entity("正文", "ORG", 10, 12)],
        }[text]
        return lexical_result(text, payload["tokens"], spans)

    service._run_worker = worker
    empty = service.link(collection["id"], [])
    assert empty["analyzed_count"] == 0
    assert empty["classification_count"] == empty["ner_document_count"] == 0
    assert empty["entity_matrix"] == [[None, None, None]]
    assert empty["classification_matrix"] == [[None] * len(CATEGORIES)]
    assert worker_calls == [] and classifier.calls == []

    summary = service.link(collection["id"], ["train-a", "val-b"])
    assert summary["analyzed_count"] == 2 and summary["total_count"] == 3
    assert summary["classification_count"] == 1
    assert summary["ner_document_count"] == 2
    assert summary["entity_types"] == ["PER", "LOC", "ORG"]
    assert summary["entity_matrix"] == [[1, 1, 1]]
    assert [row["entity_types"] for row in summary["documents"]] == [["PER"], ["LOC", "ORG"]]
    assert all(row["id"] != "test-c" for row in summary["documents"])
    assert worker_calls == [("lexical", "Train正文A"), ("lexical", "Validation正文B")]
    assert nlp.calls[0][0] == "Train正文A" and nlp.calls[1][0] == "Validation正文B"


def test_real_zero_entity_count_is_distinct_from_unavailable_and_unprocessed(tmp_path):
    service = make_service(
        tmp_path,
        classifier=StubClassifier(unavailable_for={"NER资源不可用的正文"}),
    )
    collection = service._remember({
        "id": "ner-observation-toy", "mode": "custom",
        "documents": [
            {"id": "zero", "cluster_id": 0, "title": "zero", "snippet": "zero",
             "distance": 0.0, "x": 0.0, "y": 0.0, "eligible": True},
            {"id": "unavailable", "cluster_id": 1, "title": "unavailable",
             "snippet": "unavailable", "distance": 0.0, "x": 1.0, "y": 0.0,
             "eligible": True},
            {"id": "unprocessed", "cluster_id": 2, "title": "unprocessed",
             "snippet": "unprocessed", "distance": 0.0, "x": 2.0, "y": 0.0,
             "eligible": True},
        ],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {},
        "notice": "toy", "texts": {
            "zero": "没有实体的正文", "unavailable": "NER资源不可用的正文",
            "unprocessed": "没有执行的正文",
        }, "analyses": {},
    })
    worker_calls = []

    def worker(kind, payload, **kwargs):
        worker_calls.append(payload["text"])
        if payload["text"] == "NER资源不可用的正文":
            raise ResourceUnavailableError("toy NER unavailable")
        return lexical_result(payload["text"], payload["tokens"], [])

    service._run_worker = worker
    service.profile(collection["id"], "zero", analyze=True)
    service.profile(collection["id"], "unavailable", analyze=True)
    visible = service._public_collection(service._collections[collection["id"]])
    assert [row["analyzed"] for row in visible["documents"]] == [True, False, False]

    aggregate = service.link(collection["id"], [])
    by_id = {row["id"]: row for row in aggregate["documents"]}
    assert by_id["zero"]["ner_status"] == "ok"
    assert by_id["zero"]["entity_count"] == 0
    assert by_id["zero"]["entity_types"] == []
    assert by_id["unavailable"]["ner_status"] == "unavailable"
    assert by_id["unavailable"]["entity_count"] is None
    assert by_id["unavailable"]["entity_types"] == []
    assert "unprocessed" not in by_id
    assert aggregate["ner_document_counts"] == [1, 0, 0]
    assert aggregate["entity_matrix"] == [[0, 0, 0], [None, None, None], [None, None, None]]
    assert worker_calls == ["没有实体的正文", "NER资源不可用的正文"]


def test_partial_profile_is_not_marked_analyzed_and_can_retry_to_success(tmp_path):
    service = make_service(tmp_path)
    collection = service._remember({
        "id": "retry-toy", "mode": "custom",
        "documents": [{"id": "retry", "cluster_id": 0, "title": "retry",
                       "snippet": "retry", "distance": 0.0, "x": 0.0, "y": 0.0,
                       "eligible": True}],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {},
        "notice": "toy", "texts": {"retry": "可重试的新闻正文"}, "analyses": {},
    })
    worker_calls = []

    def worker(kind, payload, **kwargs):
        worker_calls.append(kind)
        if len(worker_calls) == 1:
            raise ResourceUnavailableError("one-time lexical failure")
        return lexical_result(payload["text"], payload["tokens"], [])

    service._run_worker = worker
    partial = service.profile(collection["id"], "retry", analyze=True)
    current = service._public_collection(service._collections[collection["id"]])
    assert partial["analysis"]["status"] == "partial"
    assert current["documents"][0]["analyzed"] is False

    recovered = service.profile(collection["id"], "retry", analyze=True)
    current = service._public_collection(service._collections[collection["id"]])
    assert recovered["analysis"]["status"] == "ok"
    assert current["documents"][0]["analyzed"] is True
    assert worker_calls == ["lexical", "lexical"]
    assert len(service.classifier.calls) == 2


def test_link_aggregate_receives_an_independent_consistent_snapshot(tmp_path, monkeypatch):
    import nlplab.services.comprehensive_service as service_module

    service = make_service(tmp_path)
    live = {
        "id": "snapshot-toy", "mode": "custom",
        "documents": [{"id": "doc", "cluster_id": 0, "title": "doc",
                       "snippet": "doc", "distance": 0.0, "x": 0.0, "y": 0.0,
                       "eligible": True}],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {},
        "notice": "toy", "texts": {"doc": "news"},
        "analyses": {"doc": {"classification": {"prediction": "frozen-value"},
                               "modules": {"classification": {"status": "ok"}}}},
    }
    public = service._remember(live)
    captured = {}

    def inspect_snapshot(snapshot):
        captured["value"] = snapshot
        captured["prediction"] = snapshot["analyses"]["doc"]["classification"]["prediction"]
        live["analyses"]["doc"]["classification"]["prediction"] = "concurrent-mutation"
        return {"prediction": captured["prediction"]}

    monkeypatch.setattr(service_module, "aggregate", inspect_snapshot)
    result = service.link(public["id"], [])

    snapshot = captured["value"]
    assert snapshot is not live
    assert snapshot["analyses"] is not live["analyses"]
    assert snapshot["analyses"]["doc"] is not live["analyses"]["doc"]
    assert snapshot["analyses"]["doc"]["classification"]["prediction"] == "frozen-value"
    assert result == {"prediction": "frozen-value"}


def test_worker_lock_releases_after_success_and_exception(tmp_path):
    service = make_service(tmp_path)
    service._execute_worker = lambda kind, payload, **kwargs: {"kind": kind, "ok": True}
    assert service._run_worker("custom", {}) == {"kind": "custom", "ok": True}
    assert not service._worker_lock.locked()

    def fail(*args, **kwargs):
        raise RuntimeError("toy worker failure")

    service._execute_worker = fail
    with pytest.raises(RuntimeError, match="toy worker failure"):
        service._run_worker("custom", {})
    assert not service._worker_lock.locked()


def test_system_tests_reject_busy_state_and_singleflight_concurrent_calls(tmp_path, monkeypatch):
    import nlplab.comprehensive.system_tests as system_test_module

    service = make_service(tmp_path)
    calls = []
    entered, release = Event(), Event()

    def blocked_system_tests(value):
        calls.append(value)
        entered.set()
        assert release.wait(2)
        return {"status": "toy-ok"}

    monkeypatch.setattr(system_test_module, "run_system_tests", blocked_system_tests)

    service._worker_lock.acquire()
    try:
        with pytest.raises(ResourceUnavailableError):
            service.system_tests()
        assert calls == []
        assert not service._system_tests_lock.locked()
    finally:
        service._worker_lock.release()

    result, errors = [], []

    def invoke():
        try:
            result.append(service.system_tests())
        except Exception as error:  # surfaced in the main test thread below
            errors.append(error)

    thread = Thread(target=invoke)
    thread.start()
    try:
        assert entered.wait(2), "first system-test call did not start"
        with pytest.raises(ResourceUnavailableError):
            service.system_tests()
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert result == [{"status": "toy-ok"}]
    assert len(calls) == 1
    assert not service._system_tests_lock.locked()


def test_link_prevalidates_all_ids_before_any_profile_or_model_call(tmp_path):
    service = make_service(tmp_path)
    public = service._remember({
        "id": "preflight-toy", "mode": "custom",
        "documents": [
            {"id": "eligible", "cluster_id": 0, "title": "eligible", "snippet": "ok",
             "distance": 0.0, "x": 0.0, "y": 0.0, "eligible": True},
            {"id": "ineligible", "cluster_id": 1, "title": "ineligible", "snippet": "no",
             "distance": 0.0, "x": 1.0, "y": 0.0, "eligible": False},
        ],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {},
        "notice": "toy", "texts": {"eligible": "新闻正文", "ineligible": "超范围正文"},
        "analyses": {},
    })
    profile_calls = []
    service.profile = lambda *args, **kwargs: profile_calls.append(args)

    with pytest.raises(DataValidationError):
        service.link(public["id"], ["eligible", "ineligible"])
    with pytest.raises(DataValidationError):
        service.link(public["id"], ["eligible", "missing"])
    assert profile_calls == []
    assert service.nlp.calls == [] and service.classifier.calls == []


@pytest.mark.parametrize("invalid_module", ["pos", "ner"])
def test_bad_pos_or_ner_spans_degrade_only_the_invalid_module(tmp_path, invalid_module):
    text = "新闻正文"
    service = make_service(tmp_path, nlp=StubNLP(tokenizer=lambda value: [value]))
    pos = [{"word": "新闻", "pos": "NN", "category": "noun", "start": 0, "end": 2}]
    entities = [entity("正文", "ORG", 2, 4)]
    if invalid_module == "pos":
        pos = [{"word": "错位", "pos": "NN", "category": "noun", "start": 0, "end": 2}]
    else:
        entities = [entity("错位", "ORG", 2, 4)]
    install_lexical_worker(
        service,
        lambda value, tokens: lexical_result(value, tokens, entities, pos=pos),
    )

    result = service.analyze(text)

    assert result["status"] == "partial"
    if invalid_module == "pos":
        assert result["modules"]["pos"]["status"] == "unavailable"
        assert result["lexical"]["pos"] == []
        assert result["modules"]["ner"]["status"] == "ok"
        assert result["lexical"]["entities"] == [entity("正文", "ORG", 2, 4)]
    else:
        assert result["modules"]["pos"]["status"] == "ok"
        assert result["lexical"]["pos"] == pos
        assert result["modules"]["ner"]["status"] == "unavailable"
        assert result["lexical"]["entities"] == []


def test_system_test_operation_lock_is_reentrant_inside_but_blocks_external_work(
    tmp_path, monkeypatch
):
    import nlplab.comprehensive.system_tests as system_test_module

    nlp = StubNLP()
    neural = StubNeural()
    service = make_service(tmp_path, nlp=nlp, neural=neural)
    worker_calls = []

    def worker(kind, payload, **kwargs):
        worker_calls.append(kind)
        if kind != "lexical":
            pytest.fail(f"unexpected worker dispatch: {kind}")
        return lexical_result(payload["text"], payload["tokens"], [])

    service._run_worker = worker
    collection = service._remember({
        "id": "operation-lock-toy", "mode": "custom",
        "documents": [{"id": "doc", "cluster_id": 0, "title": "doc", "snippet": "doc",
                       "distance": 0.0, "x": 0.0, "y": 0.0, "eligible": True}],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {},
        "notice": "toy", "texts": {"doc": "新闻正文"}, "analyses": {},
    })
    entered, release = Event(), Event()
    inside = []

    def running_tests(value):
        # system_tests owns the operation gate; this same-thread call must re-enter.
        inside.append(value.analyze("内部系统测试新闻正文"))
        entered.set()
        assert release.wait(5), "test did not release the blocked system-test stub"
        return {"status": "stubbed"}

    monkeypatch.setattr(system_test_module, "run_system_tests", running_tests)
    system_result, system_errors = [], []

    def run_system_tests():
        try:
            system_result.append(service.system_tests())
        except Exception as error:  # report thread errors in the test thread
            system_errors.append(error)

    system_thread = Thread(target=run_system_tests)
    system_thread.start()
    try:
        assert entered.wait(2), "system-test stub did not reach its held section"
        assert inside[0]["status"] == "ok"
        assert len(nlp.calls) == len(worker_calls) == len(service.classifier.calls) == 1

        operations = {
            "analyze": lambda: service.analyze("外部分析新闻正文"),
            "custom": lambda: service.custom_collection(
                [{"text": "临时新闻甲"}, {"text": "临时新闻乙"}, {"text": "临时新闻丙"}], 2
            ),
            "optional": lambda: service.optional("显式摘要正文", "summary"),
            "link": lambda: service.link(collection["id"], []),
        }
        barrier = Barrier(len(operations) + 1)
        outcomes = {}

        def call_while_busy(name, operation):
            barrier.wait(2)
            try:
                operation()
                outcomes[name] = None
            except Exception as error:
                outcomes[name] = error

        external_threads = [
            Thread(target=call_while_busy, args=(name, operation))
            for name, operation in operations.items()
        ]
        for thread in external_threads:
            thread.start()
        barrier.wait(2)
        for thread in external_threads:
            thread.join(4)
        assert all(not thread.is_alive() for thread in external_threads)
        assert set(outcomes) == set(operations)
        assert all(isinstance(error, ResourceUnavailableError) for error in outcomes.values())
        # Rejected calls must not reach any of the stubbed model/native services.
        assert len(nlp.calls) == len(worker_calls) == len(service.classifier.calls) == 1
        assert neural.calls == []
    finally:
        release.set()
        system_thread.join(3)

    assert not system_thread.is_alive()
    assert system_errors == []
    assert system_result == [{"status": "stubbed"}]
    # The gate is released after the system test exits.
    assert service.optional("系统测试后摘要", "summary")["result"]["task"] == "llm"
    assert len(neural.calls) == 1


def test_link_rejects_unknown_duplicate_and_over_ten_document_ids_before_dispatch(tmp_path):
    service = make_service(tmp_path)
    collection = service._remember({
        "id": "toy", "mode": "custom", "documents": [], "clusters": [],
        "projection": {}, "representation": {}, "provenance": {}, "notice": "toy",
        "texts": {}, "analyses": {},
    })
    service._run_worker = lambda *args, **kwargs: pytest.fail("worker must not be called")
    for ids in (["missing"], ["x", "x"], [f"doc-{i}" for i in range(11)]):
        with pytest.raises(DataValidationError):
            service.link(collection["id"], ids)


def test_optional_modules_are_explicit_and_enforce_kind_specific_limits(tmp_path):
    neural = StubNeural()
    service = make_service(tmp_path, neural=neural)
    worker_calls = []
    service._run_worker = lambda kind, payload, **kwargs: worker_calls.append((kind, payload)) or {
        "status": "ok", "relations": []
    }

    dep = service.optional("依存正文", "dependency")
    summary = service.optional("摘要正文", "summary")
    extraction = service.optional("实体关系正文", "information_extraction")
    assert dep["result"]["task"] == "dependency"
    assert summary["result"]["task"] == "llm"
    assert extraction["result"]["status"] == "ok"
    assert [call[0] for call in neural.calls] == ["dependency", "llm"]
    assert neural.calls[1][1]["task"] == "summary"
    assert neural.calls[1][1]["max_new_tokens"] == 48
    assert worker_calls[0][0] == "information_extraction"
    assert not any(kind == "lexical" for kind, _ in worker_calls)

    # Exact limits are accepted; the next code point is rejected before dispatch.
    service.optional("依" * 300, "dependency")
    service.optional("摘" * 500, "summary")
    service.optional("文" * 5000, "information_extraction")

    for text, kind in (
        ("正文" * 151, "dependency"),
        ("摘要" * 251, "summary"),
        ("实体" * 2501, "information_extraction"),
    ):
        with pytest.raises(DataValidationError):
            service.optional(text, kind)
    with pytest.raises(DataValidationError):
        service.optional("正文", "not-a-real-task")
    assert len(neural.calls) == 4 and len(worker_calls) == 2


def test_comprehensive_api_schema_four_tabs_and_legacy_analyze_remain_available(tmp_path):
    nlp = StubNLP()
    app = create_app(settings=Settings(tmp_path), service=nlp)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "/api/v1/comprehensive/analyze" in paths
        assert "/api/v1/comprehensive/optional" in paths
        assert "/api/v1/comprehensive/custom" in paths
        custom_ref = paths["/api/v1/comprehensive/custom"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        custom_schema = schema["components"]["schemas"][custom_ref.rsplit("/", 1)[-1]]
        assert custom_schema["properties"]["documents"]["minItems"] == 3
        assert custom_schema["properties"]["documents"]["maxItems"] == 30
        assert custom_schema["properties"]["k"]["minimum"] == 2
        link_ref = paths["/api/v1/comprehensive/collections/{collection_id}/link"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        link_schema = schema["components"]["schemas"][link_ref.rsplit("/", 1)[-1]]
        assert link_schema["properties"]["document_ids"]["maxItems"] == 10

        page = client.get("/integrated-analysis")
        assert page.status_code == 200
        for label in ("文档工作台", "语料空间", "关联探索", "应用评测"):
            assert label in page.text
        assert "comprehensive.js" in page.text
        assert client.get("/analysis", follow_redirects=False).headers["location"] == "/integrated-analysis"

        old = client.post("/api/v1/analyze", json={"text": "旧接口正文", "tasks": ["segmentation"]})
        assert old.status_code == 200
        assert old.json()["text"] == "旧接口正文"
        assert nlp.calls[-1] == ("旧接口正文", ["segmentation"], "latest", "hmm1-smooth")
