"""无标签、确定性的中文词级 TF-IDF；算法层不依赖 Web。"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json


@dataclass(frozen=True)
class PreprocessingConfig:
    unicode_normalization: str = "NFKC"
    lowercase: bool = True
    min_token_length: int = 2
    token_pattern: str = r"[\u3400-\u9fff]+|[a-zA-Z][a-zA-Z0-9]*"
    stopwords: tuple[str, ...] = ()


@dataclass(frozen=True)
class TFIDFConfig:
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 20000
    sublinear_tf: bool = True
    smooth_idf: bool = True
    norm: str = "l2"
    ngram_range: tuple[int, int] = (1, 1)

    def __post_init__(self):
        object.__setattr__(self, "ngram_range", tuple(self.ngram_range))


class ChinesePreprocessor:
    def __init__(self, segment: Callable[[str], list[str]], config=PreprocessingConfig()):
        if config.min_token_length < 1 or config.unicode_normalization not in {"NFC", "NFKC"}:
            raise DataValidationError("预处理参数无效")
        self.segment = segment
        self.config = config
        self.pattern = re.compile(config.token_pattern)
        self.stopwords = frozenset(config.stopwords)

    def tokens(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise DataValidationError("预处理只接受文本，不接受含标签记录")
        text = unicodedata.normalize(self.config.unicode_normalization, text)
        if self.config.lowercase:
            text = text.lower()
        result = []
        for chunk in self.pattern.findall(text):
            words = [chunk] if chunk.isascii() else self.segment(chunk)
            result.extend(
                word
                for word in words
                if len(word) >= self.config.min_token_length and word not in self.stopwords
            )
        return result


def token_identity(tokens):
    return tokens


class TextTFIDF:
    def __init__(self, config=TFIDFConfig()):
        if config.norm != "l2":
            raise DataValidationError("主 baseline 要求 L2 文档向量")
        self.config = config
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=token_identity,
            preprocessor=token_identity,
            token_pattern=None,
            lowercase=False,
            dtype=np.float64,
            **asdict(config),
        )

    def fit_transform(self, tokenized: list[list[str]]):
        try:
            vectors = self.vectorizer.fit_transform(tokenized).tocsr()
        except ValueError as error:
            raise DataValidationError(f"TF-IDF 无可用词表: {error}") from error
        if np.any(vectors.getnnz(axis=1) == 0):
            raise DataValidationError("TF-IDF 出现零向量；请显式调整配置并保存新 Run")
        return vectors

    def transform(self, tokenized: list[list[str]]):
        return self.vectorizer.transform(tokenized).tocsr()

    @property
    def terms(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "vocabulary.json",
            {
                "config": asdict(self.config),
                "vocabulary": {
                    term: int(index) for term, index in self.vectorizer.vocabulary_.items()
                },
            },
        )
        np.save(path / "idf.npy", self.vectorizer.idf_, allow_pickle=False)

    @classmethod
    def load(cls, path: Path):
        data = json.loads((path / "vocabulary.json").read_text(encoding="utf-8"))
        result = cls(TFIDFConfig(**data["config"]))
        result.vectorizer.set_params(vocabulary=data["vocabulary"])
        result.vectorizer.idf_ = np.load(path / "idf.npy", allow_pickle=False)
        return result
