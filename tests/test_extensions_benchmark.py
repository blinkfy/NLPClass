import pytest

from nlplab.extensions.benchmark import project_metrics


def test_abstention_preserves_gold_denominator():
    metrics = project_metrics([["甲乙"], ["新", "词"]], [["甲乙"], None], {"甲乙", "词"})
    assert metrics["precision"] == 1
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["tag_accuracy"] == 0.5
    assert metrics["oov_recall"] == 0
    assert metrics["iv_recall"] == 0.5
    assert sum(row[4] for row in metrics["confusion_matrix"]) == 2
    assert metrics["successful_only"]["f1"] == 1
    assert project_metrics([["甲乙"]], [None], {"甲乙"})["recall"] == 0
