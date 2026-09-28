"""将字符级 HMM 解码结果恢复为词；保留全部原始输入字符。"""

import re

from nlplab.core.schema import SegmentationResult
from nlplab.segmentation.bmes import decode
from nlplab.sequence_labeling.hmm import HMMTagger


class HMMSegmenter:
    name = "hmm-cws"

    def __init__(self, tagger: HMMTagger) -> None:
        self.tagger = tagger

    def segment(self, text: str) -> SegmentationResult:
        """空白字符作为独立 S 保留，其余连续文本由 HMM 预测。"""
        self.tagger.require_fitted()
        tokens: list[str] = []
        tags: list[str] = []
        for chunk in re.split(r"(\s)", text):
            if not chunk:
                continue
            chunk_tags = ["S"] if chunk.isspace() else self.tagger.predict(list(chunk))
            tokens.extend(decode(chunk, chunk_tags))
            tags.extend(chunk_tags)
        assert "".join(tokens) == text and len(tags) == len(text)
        return SegmentationResult(model=self.name, tokens=tokens, tags=tags)
