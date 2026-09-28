from itertools import product

import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.extensions.dictionary import EOS, BigramSegmenter, DictionarySegmenter


def test_longest_unknown_unicode_and_persistence(tmp_path):
    model = DictionarySegmenter([["中国", "石油", "大学"], ["中国石油", "大学"]])
    result = model.segment("中国石油大学😀 ")
    assert result.tokens == ["中国石油", "大学", "😀", " "]
    assert "".join(result.tokens) == "中国石油大学😀 "
    assert model.max_word_length == 4
    assert model.trace("中国石油大学")["steps"][0]["offset"] == 0
    model.save(tmp_path / "model")
    assert model.load(tmp_path / "model").segment("中国石油大学😀") == model.segment(
        "中国石油大学😀"
    )
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "model")
    with pytest.raises(DataValidationError):
        model.trace("中" * 31)


def test_bigram_lattice_matches_exhaustive_and_roundtrip(tmp_path):
    model = BigramSegmenter([["甲乙", "丙"], ["甲", "乙丙"], ["甲乙", "丙"]])
    import math

    possibilities = [["甲", "乙", "丙"], ["甲乙", "丙"], ["甲", "乙丙"]]

    def cost(words):
        return sum(
            -math.log(model.probability(a, b)) for a, b in zip(["<BOS>", *words], [*words, EOS])
        )

    trace = model.trace("甲乙丙")
    assert trace["tokens"] == min(possibilities, key=cost)
    assert trace["cost"] == pytest.approx(min(map(cost, possibilities)))
    assert trace["edges"] and trace["states"][-1]
    assert "".join(model.segment("😀甲乙 丙").tokens) == "😀甲乙 丙"
    assert model.segment("").tokens == []
    model.save(tmp_path / "bigram")
    assert model.load(tmp_path / "bigram").trace("甲乙丙") == trace
    for a, b in product(["甲", "未知", "<BOS>"], ["乙", "未知", EOS]):
        assert 0 < model.probability(a, b) <= 1
    for previous in ["甲", "未知", "<BOS>"]:
        assert sum(
            model.probability(previous, w) for w in [*sorted(model.words), EOS, "未登录"]
        ) == pytest.approx(1)
