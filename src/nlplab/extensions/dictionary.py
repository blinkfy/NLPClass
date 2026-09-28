"""训练词典索引、前向最长匹配与透明的二元词网动态规划。"""

import math
from collections import Counter
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.core.schema import SegmentationResult
from nlplab.segmentation.bmes import encode

BOS, EOS, UNKNOWN = "<BOS>", "<EOS>", "<UNK_WORD>"


class DictionarySegmenter:
    """使用 Python set 索引，不另写 Trie；当前位置扫描至训练最大词长。"""

    name = "longest"

    def __init__(self, sentences: list[list[str]]) -> None:
        if not sentences or any(not words or any(not w for w in words) for words in sentences):
            raise DataValidationError("训练词序列不能为空")
        self.unigram = Counter(word for words in sentences for word in words)
        if {BOS, EOS, UNKNOWN} & self.unigram.keys():
            raise DataValidationError("训练词与保留标记冲突")
        self.words = set(self.unigram)
        self.token_total = sum(self.unigram.values())
        self.max_word_length = max(map(len, self.words))
        self.sentence_count = len(sentences)
        self.bigram = Counter(
            (left, right)
            for words in sentences
            for left, right in zip([BOS, *words], [*words, EOS])
        )
        self.context_count = Counter()
        for (left, _), count in self.bigram.items():
            self.context_count[left] += count

    def candidates(self, text: str, start: int) -> list[str]:
        if text[start].isspace():
            return [text[start]]
        result = [text[start]]  # 未登录字符总有单字路径，已登录单字不重复。
        for end in range(start + 2, min(len(text), start + self.max_word_length) + 1):
            word = text[start:end]
            if any(c.isspace() for c in word):
                break
            if word in self.words:
                result.append(word)
        return result

    def segment(self, text: str) -> SegmentationResult:
        return self._longest(text)[0]

    def _longest(self, text: str) -> tuple[SegmentationResult, list[dict]]:
        offset, tokens, steps = 0, [], []
        while offset < len(text):
            candidates = self.candidates(text, offset)
            chosen = candidates[-1]
            steps.append({"offset": offset, "candidates": candidates, "chosen": chosen})
            tokens.append(chosen)
            offset += len(chosen)
        return SegmentationResult(self.name, tokens, encode(tokens)), steps

    def trace(self, text: str) -> dict:
        if len(text) > 30:
            raise DataValidationError("算法过程只支持 <=30 Unicode code point")
        result, steps = self._longest(text)
        return {"model": self.name, "steps": steps, "tokens": result.tokens, "tags": result.tags}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "model.json",
            {
                "model": self.name,
                "format_version": 1,
                "sentence_count": self.sentence_count,
                "unigram": dict(sorted(self.unigram.items())),
                "bigram": [[a, b, n] for (a, b), n in sorted(self.bigram.items())],
                "max_word_length": self.max_word_length,
                "lambda": getattr(self, "interpolation", None),
                "alpha": getattr(self, "alpha", None),
            },
        )

    @classmethod
    def load(cls, path: Path) -> "DictionarySegmenter":
        import json

        data = json.loads((path / "model.json").read_text(encoding="utf-8"))
        target = BigramSegmenter if data["model"] == "bigram" else DictionarySegmenter
        model = target.__new__(target)
        model.unigram = Counter(data["unigram"])
        model.words = set(model.unigram)
        model.token_total = sum(model.unigram.values())
        model.max_word_length = data["max_word_length"]
        model.sentence_count = data["sentence_count"]
        model.bigram = Counter({(a, b): n for a, b, n in data["bigram"]})
        model.context_count = Counter()
        for (a, _), n in model.bigram.items():
            model.context_count[a] += n
        if target is BigramSegmenter:
            model.interpolation, model.alpha = data["lambda"], data["alpha"]
        return model

    def parameters(self) -> dict:
        return {
            "model": self.name,
            "dictionary_size": len(self.words),
            "max_word_length": self.max_word_length,
            "word_count": sum(self.unigram.values()),
            "bigram_types": len(self.bigram),
            "sentence_count": self.sentence_count,
        }


class BigramSegmenter(DictionarySegmenter):
    """Jelinek–Mercer: .9*bigram MLE + .1*add-.1 unigram；未知单字共享桶。"""

    name = "bigram"

    def __init__(self, sentences: list[list[str]], interpolation: float = 0.9, alpha: float = 0.1):
        super().__init__(sentences)
        if not 0 < interpolation < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise DataValidationError("二元语法平滑参数非法")
        self.interpolation, self.alpha = interpolation, alpha

    def probability(self, previous: str, word: str) -> float:
        target = word if word in self.words or word == EOS else UNKNOWN
        context = previous if previous in self.words or previous == BOS else UNKNOWN
        frequency = self.sentence_count if target == EOS else self.unigram[target]
        total = self.token_total + self.sentence_count
        backoff = (frequency + self.alpha) / (total + self.alpha * (len(self.words) + 2))
        count = self.context_count[context]
        if not count:
            return backoff  # 未观测上下文采用完整一元分布，保证条件概率归一化。
        bigram = self.bigram[context, target] / count
        return self.interpolation * bigram + (1 - self.interpolation) * backoff

    def segment(self, text: str) -> SegmentationResult:
        return self._decode(text, False)[0]

    def _decode(self, text: str, tracing: bool) -> tuple[SegmentationResult, dict]:
        # states[end][last_word] = (cumulative cost, predecessor end, predecessor word).
        states: list[dict] = [{} for _ in range(len(text) + 1)]
        states[0][BOS] = (0.0, None, None)
        lattice, edges = [], []
        for start in range(len(text)):
            for word in self.candidates(text, start):
                end = start + len(word)
                if tracing:
                    lattice.append({"start": start, "end": end, "word": word})
                best = None
                for previous, (cost, _, _) in states[start].items():
                    edge_cost = -math.log(self.probability(previous, word))
                    candidate = (cost + edge_cost, start, previous)
                    if tracing:
                        edges.append(
                            {
                                "from": [start, previous],
                                "to": [end, word],
                                "negative_log_cost": edge_cost,
                                "cumulative_cost": candidate[0],
                            }
                        )
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                if best is not None:
                    states[end][word] = best
        final = min(states[-1], key=lambda w: states[-1][w][0] - math.log(self.probability(w, EOS)))
        total_cost = states[-1][final][0] - math.log(self.probability(final, EOS))
        tokens, end, word = [], len(text), final
        while end:
            tokens.append(word)
            _, end, word = states[end][word]
        tokens.reverse()
        result = SegmentationResult(self.name, tokens, encode(tokens))
        trace = {
            "model": self.name,
            "lattice": lattice,
            "edges": edges,
            "states": [
                {w: {"cost": v[0], "predecessor": [v[1], v[2]]} for w, v in row.items()}
                for row in states
            ],
            "tokens": tokens,
            "tags": result.tags,
            "cost": total_cost,
            "eos_cost": -math.log(self.probability(final, EOS)),
        }
        return result, trace

    def trace(self, text: str) -> dict:
        if len(text) > 30:
            raise DataValidationError("算法过程只支持 <=30 Unicode code point")
        return self._decode(text, True)[1]

    def parameters(self) -> dict:
        return {
            **super().parameters(),
            "interpolation": self.interpolation,
            "alpha": self.alpha,
            "smoothing": "Jelinek-Mercer + add-alpha unigram",
        }
