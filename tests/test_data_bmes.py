import json

import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Document, Entity, SequenceSample
from nlplab.data.corpus import deduplicate, load_annotations, load_documents
from nlplab.segmentation.bmes import decode, encode
from nlplab.sequence_labeling.tags import is_legal_sequence


def test_toy_roundtrip() -> None:
    tokens = ["商品", "和", "服务"]
    tags = encode(tokens)
    assert tags == ["B", "E", "S", "B", "E"]
    assert len(tags) == len("".join(tokens))
    assert decode("".join(tokens), tags) == tokens
    assert is_legal_sequence(tags)


@pytest.mark.parametrize("tags", [["M", "B", "S"], ["B", "B", "M"], ["E", "M", "E"]])
def test_invalid_structure_preserves_text(tags) -> None:
    assert "".join(decode("甲乙丙", tags)) == "甲乙丙"


def test_gold_validation() -> None:
    with pytest.raises(DataValidationError):
        SequenceSample("S1", "D1", "商品", ["商"])
    with pytest.raises(DataValidationError):
        Entity("北京", "LOC", 1, 3).validate("北京")
    assert encode([]) == decode("", []) == []


def test_jsonl_provenance_and_dedup(tmp_path) -> None:
    raw = {
        "id": "D1",
        "text": "商品和服务",
        "source_type": "news",
        "primary_label": "economy",
        "source_title": "测试",
        "source_url": "https://example.org/a",
    }
    file = tmp_path / "raw.jsonl"
    file.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    documents = load_documents(file)
    assert documents[0].text == raw["text"]
    assert len(documents[0].sha256) == 64
    unique, duplicates = deduplicate([*documents, Document(**{**raw, "id": "D2"})])
    assert len(unique) == 1
    assert duplicates == [{"id": "D2", "duplicate_of": "D1"}]
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "sample_id": "S1",
                "doc_id": "D1",
                "text": "商品和服务",
                "gold_tokens": ["商品", "和", "服务"],
            }
        ),
        encoding="utf-8",
    )
    assert load_annotations(gold, documents)[0].doc_id == "D1"
    wrong = {"sample_id": "S2", "doc_id": "D1", "text": "北京", "gold_tokens": ["北京"]}
    gold.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(DataValidationError, match="原文"):
        load_annotations(gold, documents)
    file.write_text('{"id": 2}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="raw.jsonl:1"):
        load_documents(file)
