import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Entity
from nlplab.evaluation.ner import evaluate_ner
from nlplab.evaluation.segmentation import evaluate_segmentation


def test_repeated_words_use_position() -> None:
    result = evaluate_segmentation([["北京", "北京"]], [["北京", "北", "京"]], {"北京"})
    assert result["tp"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["f1"] == pytest.approx(0.4)
    assert result["iv_recall"] == pytest.approx(0.5)
    assert result["oov_recall"] is None
    assert result["tag_accuracy"] == 0.5
    assert sum(map(sum, result["confusion_matrix"])) == 4


def test_oov_training_vocabulary_and_sample_scoping() -> None:
    result = evaluate_segmentation([["甲"], ["乙丙"]], [["甲"], ["乙", "丙"]], {"甲"})
    assert result["tp"] == 1
    assert result["oov_recall"] == 0
    assert result["iv_recall"] == 1
    with pytest.raises(DataValidationError):
        evaluate_segmentation([["甲"]], [["乙"]], set())


def test_entity_type_and_span_must_match() -> None:
    gold = [[Entity("北京", "LOC", 0, 2), Entity("北京", "LOC", 2, 4)]]
    predicted = [[Entity("北京", "LOC", 0, 2), Entity("北京", "ORG", 2, 4)]]
    result = evaluate_ner(["北京北京"], gold, predicted)
    assert result["LOC"]["recall"] == 0.5
    assert result["ORG"]["tp"] == 0
    assert result["Overall"]["f1"] == 0.5
