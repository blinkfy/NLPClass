"""All data below are temporary synthetic fixtures, never real corpus or real Gold."""

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.cli import main
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Document, SequenceSample
from nlplab.data.corpus import load_annotations, load_documents, write_jsonl
from nlplab.data.course import DEFAULT_PLAN, audit_corpus, file_sha256, load_plan
from nlplab.data.sequence_annotations import (
    GOLD_VALIDATOR,
    AnnotationStore,
    DraftConflictError,
    atomic_json,
    boundary_tokens,
    validate_record,
)
from nlplab.data.sequence_sampling import (
    allocate,
    difficult_features,
    sample_candidates,
    save_candidates,
    sentence_spans,
)
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.tags import is_legal_sequence


def course_fixture(
    directory: Path, *, demo: bool = False, difficult: bool = True
) -> AnnotationStore:
    plan = load_plan()
    documents = []
    for label in plan["primary_labels"]:
        quantities = {"news": 30, "blog": 15, "comment": 10}
        if label == "culture_entertainment":
            quantities["fiction"] = 20
        for kind, count in quantities.items():
            for _ in range(count):
                n = len(documents)
                text = (
                    f"🙂编号{n:04d}的课程测试记录：张三访问北京大学并来到上海参加会议，"
                    "2026年9月26日增长15%。"
                )
                if not difficult:
                    text = (
                        chr(0x6000 + n) + "张三访问北京大学然后前往上海参加这次课程的人工标注工作"
                        "并记录完整来源信息也便于完整检验操作流程和原文追溯规则。"
                    )
                documents.append(
                    Document(
                        f"D{n:05d}",
                        text,
                        kind,
                        label,
                        "合成测试，非真实语料",
                        f"https://example.invalid/toy/{n}",
                        is_demo=demo,
                    )
                )
    corpus = directory / "data/raw/documents.jsonl"
    candidates = directory / "data/annotations/candidates.jsonl"
    config = directory / "configs/course_corpus.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(DEFAULT_PLAN.read_bytes())
    write_jsonl(corpus, [d.to_dict() for d in documents])
    samples, manifest = sample_candidates(corpus, plan, allow_demo=demo)
    assert manifest["success"]
    manifest["config_sha256"] = file_sha256(config)
    save_candidates(candidates, samples, manifest)
    return AnnotationStore(directory)


def one_fixture(
    directory: Path, text="🙂张三访问中国石油大学（华东），随后前往青岛参加课程实验。"
) -> AnnotationStore:
    doc = Document(
        "D1",
        "前🙂" + text + "后",
        "news",
        "education",
        "测试",
        "https://example.invalid/toy",
        is_demo=True,
    )
    corpus = directory / "data/raw/documents.jsonl"
    candidates = directory / "data/annotations/candidates.jsonl"
    write_jsonl(corpus, [doc.to_dict()])
    write_jsonl(
        candidates,
        [
            {
                "sample_id": "SEQ00001",
                "doc_id": doc.id,
                "text": text,
                "source_start": 2,
                "source_end": 2 + len(text),
                "primary_label": doc.primary_label,
                "source_type": doc.source_type,
                "is_demo": True,
                "status": "pending",
                "gold_tokens": None,
                "entities": None,
                "revision_count": 0,
                "notes": "",
                "selection_reasons": ["toy"],
            }
        ],
    )
    return AnnotationStore(directory)


def fixture_reviewed(store, base, *, status="reviewed", review=True):
    # Placeholder boundaries test structural gates only; this is not genuine Gold.
    text = base["text"]
    entities = []
    for word, kind in (("张三", "PER"), ("北京大学", "ORG"), ("上海", "LOC")):
        start = text.index(word)
        entities.append({"text": word, "type": kind, "start": start, "end": start + len(word)})
    value = {
        **base,
        "status": status,
        "gold_tokens": [text],
        "entities": entities,
        "draft_version": 2,
        "revision_count": int(review),
        "review_history": [],
    }
    if review:
        value["review_history"] = [
            {
                "action": "second_pass_review",
                "previous_status": "done",
                "confirmed": True,
                "draft_version": 2,
                "reviewed_at": "2026-09-26T00:00:00+00:00",
            }
        ]
    atomic_json(store.draft_path(base["sample_id"]), value)


def test_legacy_serialization_and_exact_source_offsets(tmp_path):
    old = SequenceSample("S1", "D1", "🙂北京", ["🙂", "北京"]).to_dict()
    assert set(old) == {"sample_id", "doc_id", "text", "gold_tokens", "entities", "is_demo"}
    doc = Document("D1", "🙂北京🙂北京", "news", "education", "测试", "https://example.invalid")
    path = tmp_path / "gold.jsonl"
    value = {**old, "source_start": 1, "source_end": 4}
    write_jsonl(path, [value])
    with pytest.raises(DataValidationError, match="原文"):
        load_annotations(path, [doc])
    value.update(source_start=3, source_end=6)
    write_jsonl(path, [value])
    assert load_annotations(path, [doc])[0].text == "🙂北京"


@pytest.mark.parametrize(
    "text",
    [
        "前言。🙂他说：“第一句。第二句！”随后走开。尾句",
        '他说"Hello! Bye!"然后离开。\r\n下一行。',
        "保留  空格\r\n🙂保留制表符\t和中文。",
        "他说：“她问‘你好？’”然后回答。",
        "He's testing 3.14 and 15%. Next sentence.",
    ],
)
def test_sentence_spans_preserve_every_character(text):
    spans = sentence_spans(text)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(a < b for a, b in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_quotes_are_not_cut_inside():
    text = "他说：“第一句。第二句！”随后离开。"
    assert [text[a:b] for a, b in sentence_spans(text)] == [
        "他说：“第一句。第二句！”",
        "随后离开。",
    ]


def test_surface_features_and_boundary_bmes():
    assert set(difficult_features("“中英AI２０２６年9月26日增长15%”")) == {
        "number",
        "ascii",
        "fullwidth",
        "quotation",
        "mixed_chinese_english",
        "date_or_percentage",
    }
    words = boundary_tokens("🙂张三在北京", [1, 3, 4])
    assert words == ["🙂", "张三", "在", "北京"]
    assert encode(words) == ["S", "B", "E", "S", "B", "E"]
    assert is_legal_sequence(encode(words))
    for cuts in ([2, 1], [1, 1], [0], [6], [True]):
        with pytest.raises(DataValidationError):
            boundary_tokens("🙂张三在北京", cuts)


def test_deterministic_sampling_and_no_doc_repetition(tmp_path):
    store = course_fixture(tmp_path)
    plan = load_plan(store.config)
    first, manifest = sample_candidates(store.corpus, plan)
    second, other_manifest = sample_candidates(store.corpus, plan)
    assert first == second and manifest == other_manifest
    assert len(first) == len({s["doc_id"] for s in first}) == 200
    assert (
        Counter(s["primary_label"] for s in first)
        == plan["sequence_gold_sampling"]["primary_label_targets"]
    )
    assert (
        Counter(s["source_type"] for s in first)
        == plan["sequence_gold_sampling"]["source_type_targets"]
    )
    doc_map = {d.id: d for d in load_documents(store.corpus)}
    for sample in first:
        assert 15 <= len(sample["text"]) <= 80
        assert (
            doc_map[sample["doc_id"]].text[sample["source_start"] : sample["source_end"]]
            == sample["text"]
        )
        assert sample["gold_tokens"] is None and sample["entities"] is None
    path1, path2 = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    save_candidates(path1, first, deepcopy(manifest))
    save_candidates(path2, second, deepcopy(other_manifest))
    assert path1.read_bytes() == path2.read_bytes()
    assert (
        path1.with_suffix(".manifest.json").read_bytes()
        == path2.with_suffix(".manifest.json").read_bytes()
    )


def test_joint_flow_and_only_declared_tolerance():
    cells = {("a", "news"): 3, ("b", "fiction"): 1}
    assert allocate(cells, {"a": 2, "b": 2}, {"news": 2, "fiction": 2}, 0) is None
    assert allocate(cells, {"a": 2, "b": 1}, {"news": 1, "fiction": 2}, 1) == {
        ("a", "news"): 2,
        ("a", "fiction"): 0,
        ("b", "news"): 0,
        ("b", "fiction"): 1,
    }


def test_sampling_shortfall_is_explicit_and_not_partial(tmp_path):
    store = course_fixture(tmp_path)
    docs = [d for d in load_documents(store.corpus) if d.source_type != "fiction"]
    missing = tmp_path / "missing.jsonl"
    write_jsonl(missing, [d.to_dict() for d in docs])
    output = tmp_path / "shortfall.jsonl"
    assert (
        main(
            [
                "sample-sequence-gold",
                "--corpus",
                str(missing),
                "--output",
                str(output),
                "--reports",
                str(tmp_path / "reports"),
            ]
        )
        == 2
    )
    assert not output.exists()
    report = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert not report["success"]
    assert report["shortfalls"]["source_types"]["fiction"]["shortfall"] == 12
    assert report["uses_model_predictions"] is False


def test_audit_counts_missing_metadata_duplicates_and_does_not_mutate(tmp_path):
    path = tmp_path / "raw.jsonl"
    a = {
        "id": "D1",
        "text": "甲🙂乙" * 20,
        "primary_label": "technology",
        "source_type": "news",
        "source_title": "测试",
        "source_url": "https://a.example.invalid/article",
    }
    records = [
        a,
        {**a, "id": "D2"},
        {**a, "id": "D3", "text": "不同原文" * 12, "source_type": "blog"},
        {**a, "id": "D4", "text": "第四份原文" * 12, "primary_label": "finance"},
    ]
    write_jsonl(path, records)
    before = path.read_bytes()
    report = audit_corpus(path, load_plan())
    assert path.read_bytes() == before
    assert report["total"] == 4 and report["primary_label_distribution"] == {
        "finance": 1,
        "technology": 3,
    }
    assert report["source_type_distribution"] == {"blog": 1, "news": 3}
    assert report["cross_table"]["technology"] == {"blog": 1, "news": 2}
    assert report["domains"] == {"a.example.invalid": 4}
    assert len(report["duplicates"]) == 1 and len(report["missing_metadata"]) == 4
    assert report["url_limit_violations"][a["source_url"]] == 4
    assert not report["quota_achieved"]
    assert (
        main(["audit-course-corpus", "--corpus", str(path), "--reports", str(tmp_path / "reports")])
        == 0
    )
    assert len(list((tmp_path / "reports").glob("*/report.md"))) == 1


def test_draft_manual_gold_workflow_reviews_and_versions(tmp_path):
    store = one_fixture(tmp_path)
    assert store.get("SEQ00001")["gold_tokens"] is None
    saved = store.save("SEQ00001", {"notes": "人工备注"}, 0)
    assert saved["gold_tokens"] is None and saved["revision_count"] == 0
    saved = store.save("SEQ00001", {"boundaries": [1, 3], "entities": []}, 1)
    assert saved["status"] == "in_progress" and "".join(saved["gold_tokens"]) == saved["text"]
    with pytest.raises(DraftConflictError):
        store.save("SEQ00001", {"notes": "旧版本"}, 1)
    with pytest.raises(DataValidationError, match="先保存 done"):
        store.save("SEQ00001", {"status": "reviewed", "review_confirmation": True}, 2)
    saved = store.save("SEQ00001", {"status": "done"}, 2)
    with pytest.raises(DataValidationError):
        store.save("SEQ00001", {"status": "reviewed"}, 3)
    saved = store.save("SEQ00001", {"status": "reviewed", "review_confirmation": True}, 3)
    assert saved["revision_count"] == len(saved["review_history"]) == 1
    saved = store.save("SEQ00001", {"boundaries": [1, 2, 3]}, 4)
    assert saved["status"] == "in_progress" and saved["revision_count"] == 1
    assert not (store.work.parent / "sequence_gold.jsonl").exists()
    assert not list(store.work.glob("*.tmp"))


def test_atomic_draft_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "SEQ00001.json"
    atomic_json(path, {"notes": "旧稿🙂"})
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("nlplab.data.sequence_annotations.os.replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"notes": "新稿"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_api_codepoints_emoji_entities_overlap_and_readonly_suggestions(tmp_path):
    store = one_fixture(tmp_path)
    app = create_app(Settings(tmp_path))
    app.state.annotations = store
    client = TestClient(app)
    assert client.get("/annotation/sequence").status_code == 200
    assert "Array.from" in client.get("/static/js/annotation.js").text
    url = "/api/v1/annotation/sequence/SEQ00001"
    correct = {"text": "张三", "type": "PER", "start": 1, "end": 3}
    result = client.patch(
        url, json={"expected_version": 0, "boundaries": [1, 3], "entities": [correct]}
    )
    assert result.status_code == 200 and result.json()["entities"][0]["start"] == 1
    wrong = {**correct, "start": 2, "end": 4}  # UTF-16 indices after emoji, rejected.
    assert client.patch(url, json={"expected_version": 1, "entities": [wrong]}).status_code == 422
    overlapping = {"text": "三访", "type": "LOC", "start": 2, "end": 4}
    assert (
        client.patch(
            url, json={"expected_version": 1, "entities": [correct, overlapping]}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            url, json={"expected_version": 1, "entities": [{**correct, "score": 0.9}]}
        ).status_code
        == 422
    )
    before = store.draft_path("SEQ00001").read_bytes()
    suggestions = client.get(url + "/suggestions").json()
    assert suggestions["notice"] == "参考建议，非 Gold" and not suggestions["writes_gold"]
    assert suggestions["ner"]["status"] == "unavailable"
    assert store.draft_path("SEQ00001").read_bytes() == before
    assert client.patch(url, json={"expected_version": 0, "notes": "conflict"}).status_code == 409
    assert client.get("/api/v1/annotation/sequence?label=education").json()["filtered_count"] == 1
    assert (
        client.get("/api/v1/annotation/sequence?source_type=fiction").json()["filtered_count"] == 0
    )


def test_source_span_and_entity_validation(tmp_path):
    store = one_fixture(tmp_path)
    records, documents = store.base_records()
    bad = {**records[0], "source_start": 3}
    with pytest.raises(DataValidationError, match="source_start"):
        validate_record(bad, documents)
    with pytest.raises(DataValidationError):
        store.save(
            "SEQ00001", {"entities": [{"text": "张三", "type": "TIME", "start": 1, "end": 3}]}, 0
        )
    with pytest.raises(DataValidationError):
        store.save("../escape", {}, 0)


def test_formal_export_requires_exactly_200_reviewed_and_exports_no_pending(tmp_path):
    store = course_fixture(tmp_path)
    bases, _ = store.base_records()
    for base in bases[:-1]:
        fixture_reviewed(store, base)
    output = tmp_path / "gold/sequence_gold.jsonl"
    with pytest.raises(DataValidationError, match="exactly_200_reviewed"):
        store.export(output, tmp_path / "reports")
    assert not output.exists()
    fixture_reviewed(store, bases[-1], status="done")
    assert store.validation()[1]["reviewed_count"] == 199
    fixture_reviewed(store, bases[-1])
    assert store.validation()[1]["valid"]
    result = store.export(output, tmp_path / "reports")
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 200 and all(r["status"] == "reviewed" for r in exported)
    assert all(not list(GOLD_VALIDATOR.iter_errors(r)) for r in exported)
    assert all(set(e) == {"text", "type", "start", "end"} for r in exported for e in r["entities"])
    assert len(load_annotations(output, load_documents(store.corpus))) == 200
    assert Path(result["manifest"]).name == "sequence_gold_manifest.json"
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert (
        manifest["gold_sha256"] == file_sha256(output)
        and not manifest["model_evaluation_performed"]
    )
    before = output.read_bytes()
    with pytest.raises(DataValidationError, match="拒绝覆盖"):
        store.export(output, tmp_path / "reports")
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "violation", ["entity_support", "with_entity", "second_pass", "demo", "difficult"]
)
def test_final_quality_gates_are_enforced(tmp_path, violation):
    store = course_fixture(tmp_path, demo=violation == "demo", difficult=violation != "difficult")
    bases, _ = store.base_records()
    for base in bases:
        fixture_reviewed(store, base, review=violation != "second_pass")
        path = store.draft_path(base["sample_id"])
        value = json.loads(path.read_text(encoding="utf-8"))
        if violation == "entity_support":
            value["entities"] = [e for e in value["entities"] if e["type"] != "ORG"]
        if violation == "with_entity":
            value["entities"] = []
        atomic_json(path, value)
    reviewed, report = store.validation()
    assert len(reviewed) == 200 and not report["valid"]
    check = {
        "entity_support": "ORG",
        "with_entity": "with_entity",
        "second_pass": "second_pass_samples",
        "demo": "demo_not_formal_gold",
        "difficult": "difficult_samples",
    }[violation]
    assert any(e.get("check") == check for e in report["errors"])
