import numpy as np
import pytest
from scipy.sparse import csr_matrix

from nlplab.clustering.calculation import lsa_trace, term_counts, tfidf_trace
from nlplab.core.exceptions import DataValidationError


def test_ngram_counts_include_full_token_sequence():
    assert term_counts(["新闻", "文本", "新闻"], [1, 2]) == {
        "新闻": 2,
        "文本": 1,
        "新闻 文本": 1,
        "文本 新闻": 1,
    }


def test_tfidf_arithmetic_matches_normalized_frozen_values():
    raw = np.array([(1 + np.log(2)) * 1.2, 2.0])
    row = csr_matrix((raw / np.linalg.norm(raw)).reshape(1, -1))
    trace = tfidf_trace(
        {"新闻": 2, "文本": 1},
        row,
        ["新闻", "文本"],
        np.array([1.2, 2.0]),
        np.array([3, 1]),
        {"sublinear_tf": True},
    )
    assert trace["norm"] == pytest.approx(np.linalg.norm(raw))
    for item in trace["rows"]:
        assert item["raw_weight"] / trace["norm"] == pytest.approx(item["weight"])
    with pytest.raises(DataValidationError):
        tfidf_trace(
            {"新闻": 2, "文本": 1},
            row * 2,
            ["新闻", "文本"],
            np.array([1.2, 2.0]),
            np.array([3, 1]),
            {"sublinear_tf": True},
        )


def test_lsa_signed_contributions_and_norm_match_frozen_values():
    row = csr_matrix([[0.8, 0.6, 0.0]])
    components = np.array([[0.3, -0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.0]])
    latent = np.asarray(row @ components.T).ravel()
    actual = latent / np.linalg.norm(latent)
    trace = lsa_trace(row, components, actual, ["新闻", "文本", "边界"])
    assert sum(item["product"] for item in trace["rows"]) + trace["remaining_sum"] == pytest.approx(
        latent[0]
    )
    assert trace["raw_value"] / trace["norm"] == pytest.approx(actual[0])
    assert any(item["product"] < 0 for item in trace["rows"])
    with pytest.raises(DataValidationError):
        lsa_trace(row, components, actual + 0.1, ["新闻", "文本", "边界"])
