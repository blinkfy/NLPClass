"""分词算法可替换接口。"""

from typing import Protocol

from nlplab.core.schema import SegmentationResult


class Segmenter(Protocol):
    def segment(self, text: str) -> SegmentationResult: ...
