"""Toy contracts for the separate course dependency module."""

import json
from pathlib import Path

import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.course_neural.dependency import (
    align_words,
    attachment_metrics,
    sentence_spans,
    tree_record,
    validate_tree,
)


def test_validate_tree_accepts_one_rooted_acyclic_tree():
    assert validate_tree([2, 0, 2, 2], ["nsubj", "root", "dobj", "punct"]) == {
        "single_root": True,
        "acyclic": True,
        "root_reachable": True,
        "node_count": 4,
    }


@pytest.mark.parametrize(
    ("heads", "relations"),
    [
        ([1, 0], ["dep", "root"]),  # self loop
        ([0, 0], ["root", "root"]),  # multiple roots
        ([2, 1, 0], ["dep", "dep", "root"]),  # disconnected cycle
    ],
)
def test_validate_tree_rejects_self_loop_multiple_roots_and_cycles(heads, relations):
    with pytest.raises(DataValidationError):
        validate_tree(heads, relations)


@pytest.mark.parametrize(
    ("heads", "relations"),
    [
        ([0, 2], ["root"]),  # mismatched relation count
        ([0, True], ["root", "dep"]),  # bool is not a token ID
        ([0, 3], ["root", "dep"]),  # head outside 0..N
        ([0], [""]),  # missing relation
    ],
)
def test_validate_tree_rejects_malformed_nodes(heads, relations):
    with pytest.raises(DataValidationError):
        validate_tree(heads, relations)


def test_sentence_spans_split_multiple_sentences_and_keep_absolute_offsets():
    text = "我喜欢苹果。她喜欢香蕉！"
    spans = sentence_spans(text)

    assert spans == [
        {"text": "我喜欢苹果。", "start": 0, "end": 6},
        {"text": "她喜欢香蕉！", "start": 6, "end": 12},
    ]
    assert all(text[item["start"] : item["end"]] == item["text"] for item in spans)


def test_align_words_returns_original_text_offsets_after_sentence_prefix():
    text = "前文。我喜欢猫。"

    assert align_words(text, ["我", "喜欢", "猫", "。"], start=3) == [
        {"id": 1, "form": "我", "start": 3, "end": 4},
        {"id": 2, "form": "喜欢", "start": 4, "end": 6},
        {"id": 3, "form": "猫", "start": 6, "end": 7},
        {"id": 4, "form": "。", "start": 7, "end": 8},
    ]


def test_align_words_rejects_uncovered_non_whitespace_text():
    with pytest.raises(DataValidationError):
        align_words("我喜欢猫", ["我"])


def test_tree_record_exports_complete_conll_and_preserves_sentence_offsets():
    text = "前文。我喜欢猫。"
    record = tree_record(
        text,
        ["我", "喜欢", "猫", "。"],
        {"heads": [2, 0, 2, 2], "relations": ["nsubj", "root", "dobj", "punct"]},
        start=3,
    )

    assert [(item["start"], item["end"]) for item in record["items"]] == [
        (3, 4),
        (4, 6),
        (6, 7),
        (7, 8),
    ]
    assert record["items"][1]["explanation"] == "句子根节点"
    rows = record["conll"].splitlines()
    assert len(rows) == 4
    assert [row.split("\t")[0] for row in rows] == ["1", "2", "3", "4"]
    assert [len(row.split("\t")) for row in rows] == [10, 10, 10, 10]
    assert rows[0] == "1\t我\t_\t_\t_\t_\t2\tnsubj\t_\t_"
    assert rows[1] == "2\t喜欢\t_\t_\t_\t_\t0\troot\t_\t_"


def test_attachment_metrics_report_scores_with_and_without_punctuation():
    words = ["我", "喜欢", "鱼", "。"]
    gold_heads = [2, 0, 2, 2]
    gold_relations = ["nsubj", "root", "dobj", "punct"]
    predicted_heads = [2, 0, 2, 1]
    predicted_relations = ["nsubj", "root", "dobj", "punct"]

    without_punctuation = attachment_metrics(
        words, gold_heads, gold_relations, predicted_heads, predicted_relations
    )
    with_punctuation = attachment_metrics(
        words,
        gold_heads,
        gold_relations,
        predicted_heads,
        predicted_relations,
        exclude_punctuation=False,
    )

    assert without_punctuation == {
        "support": 3,
        "correct_heads": 3,
        "correct_labels": 3,
        "uas": 1.0,
        "las": 1.0,
        "exclude_punctuation": True,
    }
    assert with_punctuation == {
        "support": 4,
        "correct_heads": 3,
        "correct_labels": 3,
        "uas": 0.75,
        "las": 0.75,
        "exclude_punctuation": False,
    }


def test_teaching_gold_is_fixed_gold_token_diagnostic_not_ctb_benchmark():
    path = Path(__file__).resolve().parents[1] / "src/nlplab/course_neural/teaching_gold.json"
    gold = json.loads(path.read_text(encoding="utf-8"))

    assert gold["schema_version"] == 1
    assert "toy" in gold["purpose"]
    assert gold["evaluation_protocol"]["dataset_scope"] == "toy_teaching_gold"
    assert gold["evaluation_protocol"]["benchmark_scope"] == "not_ctb_benchmark"
    assert gold["evaluation_protocol"]["tokenization"] == "gold-token"
    assert len(gold["examples"]) in range(6, 9)
    for example in gold["examples"]:
        assert "".join(example["tokens"]) == example["text"]
        assert len(example["tokens"]) == len(example["heads"]) == len(example["relations"])
        validate_tree(example["heads"], example["relations"])
