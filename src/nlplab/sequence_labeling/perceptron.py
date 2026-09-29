"""字符级 Averaged Structured Perceptron；合法路径复用项目 Viterbi。"""

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.segmentation.bmes import decode
from nlplab.sequence_labeling.features import sequence_features
from nlplab.sequence_labeling.tags import STATES, is_legal_sequence
from nlplab.sequence_labeling.viterbi import ViterbiTrace, viterbi


@dataclass(frozen=True)
class PerceptronConfig:
    epochs: int = 5
    seed: int = 42


class AveragedStructuredPerceptron:
    name = "course-perceptron-bmes"

    def __init__(self, config: PerceptronConfig | None = None) -> None:
        self.config = config or PerceptronConfig()
        if self.config.epochs < 1:
            raise DataValidationError("感知机 epochs 必须为正整数")
        self.weights: dict[str, float] = {}
        self.history: list[dict[str, int]] = []
        self.update_example: dict | None = None
        self.training_metadata: dict = {}
        self._fitted = False

    @staticmethod
    def _counts(text: str, tags: list[str]) -> Counter[str]:
        features = sequence_features(text)
        counts: Counter[str] = Counter()
        previous = "<BOS>"
        for row, tag in zip(features, tags):
            counts.update(f"E|{tag}|{feature}" for feature in row)
            counts[f"T|{previous}|{tag}"] += 1
            previous = tag
        return counts

    def _trace(self, text: str) -> ViterbiTrace:
        rows = sequence_features(text)
        start = np.array([self.weights.get(f"T|<BOS>|{tag}", 0.0) for tag in STATES])
        transition = np.array(
            [[self.weights.get(f"T|{left}|{right}", 0.0) for right in STATES] for left in STATES]
        )
        local = np.array(
            [
                [
                    sum(self.weights.get(f"E|{tag}|{feature}", 0.0) for feature in row)
                    for tag in STATES
                ]
                for row in rows
            ],
            dtype=float,
        ).reshape((-1, len(STATES)))
        return viterbi(start, transition, local)

    def predict(self, observations: list[str]) -> list[str]:
        if not self._fitted:
            raise ResourceUnavailableError("课程感知机模型尚未训练或加载")
        if any(not isinstance(char, str) or len(char) != 1 for char in observations):
            raise DataValidationError("感知机输入必须为 Unicode 字符序列")
        return self._trace("".join(observations)).tags

    def fit(
        self, observations: list[list[str]], tags: list[list[str]]
    ) -> "AveragedStructuredPerceptron":
        if len(observations) != len(tags) or not observations:
            raise DataValidationError("感知机训练样本与标签数量不一致或为空")
        for chars, gold in zip(observations, tags):
            if len(chars) != len(gold) or not is_legal_sequence(gold):
                raise DataValidationError("感知机训练 Gold 必须是合法且等长的 BMES")
        totals: dict[str, float] = {}
        timestamps: dict[str, int] = {}
        step = 0
        self._fitted = True
        for epoch in range(self.config.epochs):
            order = list(range(len(observations)))
            random.Random(self.config.seed + epoch).shuffle(order)
            updates = 0
            for index in order:
                step += 1
                text = "".join(observations[index])
                gold = tags[index]
                predicted = self._trace(text).tags
                if predicted == gold:
                    continue
                updates += 1
                difference = self._counts(text, gold)
                difference.subtract(self._counts(text, predicted))
                for key, delta in difference.items():
                    if delta == 0:
                        continue
                    old = self.weights.get(key, 0.0)
                    totals[key] = totals.get(key, 0.0) + (step - timestamps.get(key, 0)) * old
                    timestamps[key] = step
                    self.weights[key] = old + delta
                if self.update_example is None:
                    positive = sorted(
                        (key, value) for key, value in difference.items() if value > 0
                    )
                    negative = sorted(
                        (key, value) for key, value in difference.items() if value < 0
                    )
                    self.update_example = {
                        "text": text,
                        "gold_tags": gold,
                        "predicted_tags": predicted,
                        "positive_features": positive[:8],
                        "negative_features": negative[:8],
                    }
            self.history.append({"epoch": epoch + 1, "updates": updates})
        self.weights = {
            key: (totals.get(key, 0.0) + (step + 1 - timestamps.get(key, 0)) * value) / (step + 1)
            for key, value in self.weights.items()
        }
        return self

    def explain(self, text: str) -> dict:
        if not self._fitted:
            raise ResourceUnavailableError("课程感知机模型尚未训练或加载")
        trace = self._trace(text)
        rows = sequence_features(text)
        contributions = []
        for index, (row, tag) in enumerate(zip(rows, trace.tags)):
            previous = trace.tags[index - 1] if index else "<BOS>"
            weighted = [(feature, self.weights.get(f"E|{tag}|{feature}", 0.0)) for feature in row]
            weighted.append(
                (f"previous_tag={previous} → {tag}", self.weights.get(f"T|{previous}|{tag}", 0.0))
            )
            contributions.append(sorted(weighted, key=lambda item: (-abs(item[1]), item[0]))[:5])
        return {
            "text": text,
            "tokens": decode(text, trace.tags),
            "tags": trace.tags,
            "scores": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in trace.score
            ],
            "backpointer": trace.backpointer.tolist(),
            "contributions": contributions,
            "update_example": self.update_example,
            "metadata": self.training_metadata,
        }

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise ResourceUnavailableError("感知机模型尚未训练")
        path.mkdir(parents=True, exist_ok=False)
        weights = path / "weights.json"
        weights.write_text(
            json.dumps(self.weights, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        metadata = {
            "format_version": 1,
            "model": self.name,
            "states": list(STATES),
            "config": asdict(self.config),
            "history": self.history,
            "update_example": self.update_example,
            "training_metadata": self.training_metadata,
            "weights_sha256": sha256(weights.read_bytes()).hexdigest(),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "AveragedStructuredPerceptron":
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            weights = path / "weights.json"
            if (
                metadata["format_version"] != 1
                or metadata["states"] != list(STATES)
                or metadata["weights_sha256"] != sha256(weights.read_bytes()).hexdigest()
            ):
                raise DataValidationError("感知机模型格式或权重指纹不匹配")
            model = cls(PerceptronConfig(**metadata["config"]))
            model.weights = json.loads(weights.read_text(encoding="utf-8"))
            model.history = metadata["history"]
            model.update_example = metadata["update_example"]
            model.training_metadata = metadata["training_metadata"]
            model._fitted = True
            return model
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ResourceUnavailableError(f"感知机模型无法加载: {path}") from error
