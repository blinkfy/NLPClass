"""标签无关的词序列词表、Train 长度规则与可追溯 head-tail 编码。"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json

PAD, UNK = 0, 1


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def length_statistics(sequences):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    if not len(lengths):
        raise DataValidationError("长度统计需要非空训练集")
    return {
        "count": len(lengths),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "percentile_method": "numpy linear",
        "lengths": lengths.tolist(),
    }


def max_len_from_train(stats):
    return max(128, min(768, math.ceil(stats["p95"] / 32) * 32))


def retain_head_tail(tokens, max_len):
    if max_len < 5:
        raise DataValidationError("max_len 必须容纳最大卷积核 5")
    if len(tokens) <= max_len:
        positions = list(range(len(tokens)))
    else:
        head, tail = math.ceil(max_len / 2), max_len // 2
        positions = list(range(head)) + list(range(len(tokens) - tail, len(tokens)))
    return [tokens[i] for i in positions], positions


class WordSequenceVocabulary:
    def __init__(self, max_len, min_freq=2):
        self.max_len, self.min_freq = int(max_len), int(min_freq)

    def fit(self, sequences):
        self.frequencies = Counter(token for sequence in sequences for token in sequence)
        if any(t in self.frequencies for t in ("<PAD>", "<UNK>")):
            raise DataValidationError("语料 token 与保留符号冲突")
        # 同频按字典序，不使用类别、Validation 或 Test。
        terms = sorted(
            (t for t, n in self.frequencies.items() if n >= self.min_freq),
            key=lambda t: (-self.frequencies[t], t),
        )
        self.terms = ["<PAD>", "<UNK>", *terms]
        self.vocabulary = {t: i for i, t in enumerate(self.terms)}
        return self

    def encode(self, sequences):
        matrix = np.zeros((len(sequences), self.max_len), dtype=np.int64)
        retained_sequences, original_positions, lengths = [], [], []
        original_count = retained_count = unk_count = retained_unk = truncated = 0
        per_document = []
        for i, tokens in enumerate(sequences):
            kept, positions = retain_head_tail(tokens, self.max_len)
            ids = [self.vocabulary.get(t, UNK) for t in kept]
            matrix[i, : len(ids)] = ids
            retained_sequences.append(kept)
            original_positions.append(positions)
            lengths.append(len(kept))
            unknown = sum(t not in self.vocabulary for t in tokens)
            kept_unknown = sum(t == UNK for t in ids)
            n, m = len(tokens), len(kept)
            original_count += n
            retained_count += m
            unk_count += unknown
            retained_unk += kept_unknown
            truncated += n > self.max_len
            per_document.append(
                {
                    "original_length": n,
                    "retained_length": m,
                    "unk_count": unknown,
                    "retained_unk_count": kept_unknown,
                    "truncated": n > self.max_len,
                }
            )
        return {
            "ids": matrix,
            "lengths": np.asarray(lengths, dtype=np.int64),
            "tokens": retained_sequences,
            "original_positions": original_positions,
            "stats": {
                "document_count": len(sequences),
                "vocabulary_size": len(self.terms),
                "max_len": self.max_len,
                "original_token_count": original_count,
                "retained_token_count": retained_count,
                "unk_count": unk_count,
                "unk_rate": unk_count / original_count if original_count else 0.0,
                "retained_unk_count": retained_unk,
                "retained_unk_rate": retained_unk / retained_count if retained_count else 0.0,
                "truncated_count": truncated,
                "truncation_rate": truncated / len(sequences) if sequences else 0.0,
                "retained_token_ratio": retained_count / original_count if original_count else 1.0,
                "per_document": per_document,
            },
        }

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        write_json(
            directory / "vocabulary.json",
            {
                "terms": self.terms,
                "pad": PAD,
                "unk": UNK,
                "min_freq": self.min_freq,
                "max_len": self.max_len,
                "order": "frequency desc, lexical asc",
            },
        )
        write_json(directory / "token_frequencies.json", dict(self.frequencies))

    @classmethod
    def load(cls, directory):
        meta = read_json(directory / "vocabulary.json")
        if meta["pad"] != PAD or meta["unk"] != UNK or meta["terms"][:2] != ["<PAD>", "<UNK>"]:
            raise DataValidationError("PAD/UNK mapping 不符")
        result = cls(meta["max_len"], meta["min_freq"])
        result.terms = meta["terms"]
        result.vocabulary = {t: i for i, t in enumerate(result.terms)}
        result.frequencies = Counter(read_json(directory / "token_frequencies.json"))
        return result
