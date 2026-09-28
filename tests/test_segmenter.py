import pytest

from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.sequence_labeling.tags import is_legal_sequence


@pytest.mark.parametrize("text", ["", "商品和服务", "甲🙂乙", " 北京\n大学\t ", "English2026年。"])
def test_segmenter_roundtrip(toy_model, text) -> None:
    result = HMMSegmenter(toy_model).segment(text)
    assert "".join(result.tokens) == text
    assert len(result.tags) == len(text)
    assert is_legal_sequence(result.tags)
