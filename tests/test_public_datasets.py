"""合成 fixture 覆盖公开数据流程；不下载或包含公开全文/真实 Gold。"""

from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Entity, text_sha256
from nlplab.data.corpus import read_jsonl, write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.msra_ner import bio_entities, prepare_msra, read_bio, select_exp1
from nlplab.data.personal_selection import (
    read_object,
    selection_score,
    subset_manifest,
    verify_subset,
)
from nlplab.data.public_audit import audit_public
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import (
    CATEGORIES,
    category_label,
    clustering_view,
    import_thucnews,
    select_thucnews,
    split_thucnews,
)
from nlplab.ner.hanlp_adapter import model_chunks
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.services.course_samples import course_predictions, frozen_smooth
from nlplab.services.experiment_service import RunStore


def imported(path, rows, dataset, split="test"):
    write_jsonl(path, rows)
    manifest = {
        "dataset_name": dataset,
        "dataset_version": "synthetic-test-v1",
        "source_hashes": {"synthetic": "a" * 64},
        "split": split,
        "data_sha256": file_sha256(path),
        "dataset_variant": "thucnews-official-14",
        "provenance": {"license": "synthetic fixture"},
    }
    atomic_json(subset_manifest(path), manifest)


def fixture_row(record_id, text):
    return {
        "source_record_id": record_id,
        "text": text,
        "sha256": text_sha256(text),
        "dataset_version": "synthetic-test-v1",
        "source_hashes": {"synthetic": "a" * 64},
    }


@pytest.fixture
def msra_source(tmp_path):
    entities = [
        Entity("张三", "PER", 1, 3),
        Entity("北京", "LOC", 4, 6),
        Entity("公司", "ORG", 7, 9),
    ]
    rows = [
        {
            **fixture_row(f"test:{i:06d}", "🙂张三到北京的公司" + str(i)),
            "sample_id": f"TOY{i}",
            "source_dataset": "MSRA-NER",
            "source_split": "test",
            "ner_gold": [asdict(entity) for entity in entities],
        }
        for i in range(250)
    ]
    path = tmp_path / "test.jsonl"
    imported(path, rows, "MSRA-NER")
    return path


@pytest.fixture
def thuc_source(tmp_path):
    rows = []
    for label in CATEGORIES:
        for i in range(122):
            text = (
                "合成全文测试文档用于验证稳定选择以及数据划分，不是真实新闻语料。" * 4
                + label
                + str(i)
            )
            row = fixture_row(f"{label}/{i}.txt", text)
            rows.append(
                {
                    **row,
                    "id": f"TOY-{label}-{i}",
                    "primary_label": label,
                    "source_dataset": "THUCNews",
                    "source_variant": "thucnews-official-14",
                }
            )
    duplicate = {
        **rows[0],
        "id": "TOY-duplicate",
        "source_record_id": "finance/duplicate.txt",
        "text": "  " + rows[0]["text"] + "\n",
    }
    duplicate["sha256"] = text_sha256(duplicate["text"])
    rows.append(duplicate)
    rows.append(
        {
            **fixture_row("finance/short.txt", "短标题"),
            "id": "TOY-short",
            "primary_label": "finance",
        }
    )
    path = tmp_path / "thuc.jsonl"
    imported(path, rows, "THUCNews")
    return path


def test_bio_unicode_three_types():
    text = "🙂张三到北京的公司"
    values = bio_entities(
        list(text), ["O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "B-ORG", "I-ORG"]
    )
    assert [(e.type, e.start, e.end) for e in values] == [
        ("PER", 1, 3),
        ("LOC", 4, 6),
        ("ORG", 7, 9),
    ]
    for entity in values:
        entity.validate(text)


@pytest.mark.parametrize(
    "tags",
    [
        ["I-PER", "O"],
        ["B-PER", "I-ORG"],
        ["B-MISC", "O"],
        ["", "O"],
        ["O", "S-PER"],
        ["O-PER", "O"],
    ],
)
def test_malformed_bio_rejected(tags):
    with pytest.raises(DataValidationError):
        bio_entities(list("甲乙"), tags)


def test_bio_physical_lines_and_missing_tag(tmp_path):
    path = tmp_path / "bio.txt"
    path.write_text("🙂\tO\n\u2028\tO\n张\tB-PER\n三\tI-PER\n\n\n", "utf-8")
    rows = read_bio(path, "test", "v1", {})
    assert len(rows) == 1 and rows[0]["ner_gold"][0]["start"] == 2
    path.write_text("甲\n", "utf-8")
    with pytest.raises(DataValidationError, match="TAB"):
        read_bio(path, "test", "v1", {})


def test_personal_msra_selection_frozen_unique(msra_source, tmp_path, monkeypatch):
    monkeypatch.setattr("nlplab.data.personal_selection.secrets.token_hex", lambda n: "1" * 32)
    output, keys = tmp_path / "selected.jsonl", tmp_path / "keys.json"
    manifest = select_exp1(msra_source, output, keys)
    first = output.read_bytes(), subset_manifest(output).read_bytes(), keys.read_bytes()
    assert select_exp1(msra_source, output, keys) == manifest
    assert first == (output.read_bytes(), subset_manifest(output).read_bytes(), keys.read_bytes())
    rows, _ = verify_subset(output, dataset="MSRA-NER", count=200)
    assert len({r["source_record_id"] for r in rows}) == 200
    assert rows[0]["selection_score"] == selection_score(
        "synthetic-test-v1", "1" * 32, rows[0]["source_record_id"]
    )
    assert manifest["selection_uses_model_predictions"] is False


def test_selection_independent_of_model_fields(msra_source, tmp_path, monkeypatch):
    monkeypatch.setattr("nlplab.data.personal_selection.secrets.token_hex", lambda n: "2" * 32)
    a = select_exp1(msra_source, tmp_path / "a.jsonl", tmp_path / "ka.json")
    rows = read_jsonl(msra_source, lambda r: {**r, "model_f1": 100, "hmm_prediction": ["invented"]})
    changed = tmp_path / "changed.jsonl"
    imported(changed, list(reversed(rows)), "MSRA-NER")
    b = select_exp1(changed, tmp_path / "b.jsonl", tmp_path / "kb.json")
    assert a["selected_records"] == b["selected_records"]


@pytest.mark.parametrize("field", ["dataset_version", "source_hashes"])
def test_source_version_hash_rejection(msra_source, tmp_path, field):
    manifest = read_object(subset_manifest(msra_source))
    manifest[field] = "mismatch" if field == "dataset_version" else {"bad": "b" * 64}
    atomic_json(subset_manifest(msra_source), manifest)
    with pytest.raises(DataValidationError, match="source hash/version"):
        select_exp1(msra_source, tmp_path / "out.jsonl", tmp_path / "keys.json")


def test_train_not_used(msra_source, tmp_path):
    manifest = read_object(subset_manifest(msra_source))
    manifest["split"] = "train"
    atomic_json(subset_manifest(msra_source), manifest)
    with pytest.raises(DataValidationError, match="test"):
        select_exp1(msra_source, tmp_path / "out.jsonl", tmp_path / "keys.json")


def test_msra_invalid_train_is_reported_not_repaired(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    config = {
        "version": "toy-bio-v1",
        "script_url": "https://example.invalid/toy.py",
        "license": "synthetic",
        "usage_note": "fixture",
        "sources": {},
    }
    for split, content in (("train", "甲\t\n\n"), ("test", "张\tB-PER\n三\tI-PER\n\n")):
        path = source / f"msra_{split}_bio.txt"
        path.write_text(content, "utf-8")
        config["sources"][split] = {
            "url": "https://example.invalid/" + split,
            "sha256": file_sha256(path),
        }
    manifest = prepare_msra(tmp_path, config, source)
    assert manifest["splits"]["train"]["validation_status"] == "rejected"
    assert not (tmp_path / "data/external/msra_ner/train.jsonl").exists()
    assert (tmp_path / "data/external/msra_ner/bio_import_error.json").exists()


def test_hmm_roundtrip_and_exact_ner(msra_source, toy_model):
    rows = read_jsonl(msra_source, lambda r: r)[:2]

    class FakeNER:
        def recognize(self, text):
            return [Entity("张三", "PER", 1, 3)]

    predictions, metrics = course_predictions(rows, HMMSegmenter(toy_model), FakeNER())
    assert metrics["segmentation"] is None
    assert metrics["ner"]["PER"]["f1"] == 1
    assert metrics["ner"]["Overall"]["recall"] == pytest.approx(1 / 3)
    for row in predictions:
        assert "".join(row["segmentation"]["tokens"]) == row["text"]
        assert len(row["segmentation"]["bmes"]) == len(row["text"])
        assert "gold_tokens" not in row


def test_hanlp_actual_token_budget_and_source_offsets():
    from types import SimpleNamespace

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def tokenize(self, char):
            return [char, char] if char == "🙂" else [char]

    component = SimpleNamespace(
        config={"max_seq_length": 128}, transform=SimpleNamespace(tokenizer=Tokenizer())
    )
    text = "中" * 125 + "🙂" + "文" * 140
    chunks = list(model_chunks(text, component))
    assert "".join(chunk for _, chunk in chunks) == text
    assert chunks[1][0] == 125
    for start, chunk in chunks:
        assert text[start : start + len(chunk)] == chunk
        assert sum(len(Tokenizer().tokenize(char)) for char in chunk) <= 126


def test_thuc_categories():
    assert [category_label(label) for label in ("财经", "教育", "科技", "体育", "娱乐")] == list(
        CATEGORIES
    )
    assert category_label("股票") is None


def test_thuc_selection_and_splits(thuc_source, tmp_path, monkeypatch):
    monkeypatch.setattr("nlplab.data.personal_selection.secrets.token_hex", lambda n: "3" * 32)
    output, keys = tmp_path / "thuc600.jsonl", tmp_path / "keys.json"
    manifest = select_thucnews(thuc_source, output, keys)
    assert manifest["category_counts"] == {label: 120 for label in CATEGORIES}
    assert manifest["excluded"] == {"normalized_duplicate": 1, "short": 1}
    assert select_thucnews(thuc_source, output, keys) == manifest
    rows, _ = verify_subset(output, dataset="THUCNews", count=600)
    assert all(set(row) == {"id", "text"} for row in clustering_view(rows))
    split = split_thucnews(output, tmp_path / "splits", keys)
    assert split_thucnews(output, tmp_path / "splits", keys) == split
    ids = [set(value["ids"]) for value in split["splits"].values()]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
    assert len(set.union(*ids)) == 600
    for name, count in (("train", 84), ("val", 18), ("test", 18)):
        assert split["splits"][name]["category_counts"] == {label: count for label in CATEGORIES}
    changed = read_jsonl(thuc_source, lambda row: row)
    imported(tmp_path / "reversed.jsonl", list(reversed(changed)), "THUCNews")
    other = select_thucnews(
        tmp_path / "reversed.jsonl", tmp_path / "other.jsonl", tmp_path / "other_keys.json"
    )
    assert manifest["selected_records"] == other["selected_records"]


def test_thuc_shortage(thuc_source, tmp_path):
    rows = read_jsonl(thuc_source, lambda row: row)
    imported(thuc_source, [r for r in rows if r["primary_label"] != "education"], "THUCNews")
    with pytest.raises(DataValidationError, match="配额不足"):
        select_thucnews(thuc_source, tmp_path / "600.jsonl", tmp_path / "keys.json")
    assert not (tmp_path / "600.jsonl").exists()


def test_title_only_fulltext_rejected(tmp_path):
    source = tmp_path / "cnews.train.txt"
    source.write_text("体育\t这是一个只有标题的新闻文档\n" * 10, "utf-8")
    provenance = {
        "variant": "cnews-fulltext-10",
        "version": "toy",
        "source_url": "https://example.invalid/toy",
        "downloaded_at": "toy",
        "license": "synthetic",
        "usage_note": "fixture",
        "derivation_evidence": "synthetic fixture",
    }
    with pytest.raises(DataValidationError, match="标题版"):
        import_thucnews(source, tmp_path / "out.jsonl", provenance)
    assert not (tmp_path / "out.jsonl").exists()


def test_frozen_model_and_sources_unchanged():
    root = Path(__file__).resolve().parents[1]
    before = root / "artifacts/cache/public_datasets_frozen_before.json"
    if not before.exists():
        pytest.skip("本地冻结 benchmark 不提交 Git")
    for name, digest in read_object(before).items():
        assert file_sha256(root / name) == digest, name
    freeze = root / "outputs/exp1/benchmarks/20260926_130612_32a5e168/hmm_freeze.json"
    frozen_smooth(
        Settings(root), root / "artifacts/models/msr-final-20260926_130612_32a5e168_smooth", freeze
    )


def test_public_status_empty_and_course_run_read_only(tmp_path):
    path = tmp_path / "outputs/exp1/course_samples/toy_course"
    path.mkdir(parents=True)
    info = {
        "run_id": "toy_course",
        "run_type": "course_samples",
        "read_only": True,
        "is_demo": True,
        "dataset": "synthetic",
        "model": "toy",
    }
    atomic_json(path / "run.json", info)
    atomic_json(path / "metrics.json", {"segmentation": None, "ner": None})
    write_jsonl(path / "errors.jsonl", [{"sample_id": "toy", "category": "Other"}])
    store = RunStore(Settings(tmp_path))
    assert store.resolve().name == "toy_course" and store.list_runs() == [info]
    client = TestClient(create_app(Settings(tmp_path)))
    assert client.get("/api/v1/experiments/1/metrics").json()["metrics"]["segmentation"] is None
    assert (
        client.patch("/api/v1/experiments/1/errors/toy", json={"category": "Other"}).status_code
        == 422
    )
    result = client.get("/api/v1/datasets/public/status").json()
    assert result["THUCNews"]["status"] == "waiting for dataset"
    assert audit_public(tmp_path)["datasets"]["MSRA-NER"]["status"] == "waiting for dataset"


def test_frozen_subset_tampering_rejected(msra_source, tmp_path):
    out = tmp_path / "selected.jsonl"
    select_exp1(msra_source, out, tmp_path / "key.json")
    rows = read_jsonl(out, lambda row: row)
    rows[0]["text"] += "被改动"
    write_jsonl(out, rows)
    with pytest.raises(DataValidationError, match="SHA256"):
        verify_subset(out, dataset="MSRA-NER", count=200)


def test_public_audit_counts_and_manifest_schema(thuc_source, tmp_path):
    import jsonschema

    output = tmp_path / "data/course/thucnews_600.jsonl"
    keys = tmp_path / "data/manifests/keys.json"
    manifest = select_thucnews(thuc_source, output, keys)
    split_thucnews(output, tmp_path / "data/splits/thucnews_600", keys)
    audit = audit_public(tmp_path)["datasets"]["THUCNews"]
    assert audit["category_counts"] == {label: 120 for label in CATEGORIES}
    assert audit["lengths"]["min"] >= 80
    assert {name: value["count"] for name, value in audit["split_manifest"]["splits"].items()} == {
        "train": 420,
        "val": 90,
        "test": 90,
    }
    schema_path = Path(__file__).resolve().parents[1] / "docs/public_subset_manifest.schema.json"
    schema = read_object(schema_path)
    jsonschema.validate(manifest, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**manifest, "key": "must-not-leak"}, schema)
