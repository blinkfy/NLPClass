"""教学一阶 BMES HMM：显式频次统计、归一化与对数概率。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.schema import text_sha256
from nlplab.data.corpus import duplicate_key
from nlplab.segmentation.bmes import decode
from nlplab.sequence_labeling.tags import LEGAL_NEXT, LEGAL_START, STATES, is_legal_sequence

UNK = "<UNK>"


@dataclass(frozen=True)
class HMMConfig:
    """none 为纯 MLE；add_alpha 对合法初始/转移和整个发射词表平滑。"""

    smoothing: str = "add_alpha"
    alpha: float = 0.1
    unknown_strategy: str = "unk_bucket"

    def __post_init__(self) -> None:
        if self.smoothing not in {"none", "add_alpha"}:
            raise DataValidationError("smoothing 必须为 none 或 add_alpha")
        if type(self.alpha) not in (int, float) or not np.isfinite(self.alpha):
            raise DataValidationError("alpha 必须是有限数")
        if self.alpha <= 0:
            raise DataValidationError("alpha 必须大于 0")
        if self.unknown_strategy != "unk_bucket":
            raise DataValidationError("当前未知字符策略为 unk_bucket")


def log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """0 概率严格转换为 -inf，避免 log(0) 警告。"""
    result = np.full_like(probabilities, -np.inf, dtype=float)
    np.log(probabilities, out=result, where=probabilities > 0)
    return result


class HMMTagger:
    """自行训练 π (4)、A (4×4)、B (4×V，含 UNK 列)。"""

    states = STATES

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.config = config or HMMConfig()
        self.vocabulary: list[str] = []
        self.character_index: dict[str, int] = {}
        self.pi = np.zeros(4)
        self.A = np.zeros((4, 4))
        self.B = np.zeros((4, 0))
        self.log_pi = log_probabilities(self.pi)
        self.log_A = log_probabilities(self.A)
        self.log_B = self.B.copy()
        self.training_metadata: dict[str, Any] = {}
        self.training_words: set[str] = set()
        self.training_hashes: set[str] = set()

    def fit(self, observations: list[list[str]], tags: list[list[str]]) -> "HMMTagger":
        """从有监督字符/BMES 序列统计 MLE，不依赖 HMM 库。"""
        if not observations or len(observations) != len(tags):
            raise DataValidationError("训练序列为空或 observations/tags 数量不一致")
        for characters, states in zip(observations, tags):
            if not characters or len(characters) != len(states):
                raise DataValidationError("训练字符/标签长度不一致或为空")
            if any(not isinstance(char, str) or len(char) != 1 for char in characters):
                raise DataValidationError("HMM 观测必须是单个 Unicode 字符")
            if not is_legal_sequence(states):
                raise DataValidationError("训练 Gold 含非法 BMES 序列")

        vocabulary = [UNK, *sorted({char for sentence in observations for char in sentence})]
        character_index = {char: index for index, char in enumerate(vocabulary)}
        state_index = {state: index for index, state in enumerate(STATES)}
        start_counts = np.zeros(4)
        transition_counts = np.zeros((4, 4))
        emission_counts = np.zeros((4, len(vocabulary)))
        for characters, states in zip(observations, tags):
            start_counts[state_index[states[0]]] += 1
            for previous, current in zip(states, states[1:]):
                transition_counts[state_index[previous], state_index[current]] += 1
            for character, state in zip(characters, states):
                emission_counts[state_index[state], character_index[character]] += 1

        legal_start = np.array([state in LEGAL_START for state in STATES])
        legal_transition = np.array(
            [[target in LEGAL_NEXT[source] for target in STATES] for source in STATES]
        )
        alpha = self.config.alpha if self.config.smoothing == "add_alpha" else 0.0
        start_counts += alpha * legal_start
        transition_counts += alpha * legal_transition
        emission_counts += alpha  # 显式 UNK 与所有已见字符共同归一化。
        if np.any(transition_counts.sum(axis=1) == 0) or np.any(emission_counts.sum(axis=1) == 0):
            raise DataValidationError("不平滑 MLE 存在无观测状态/转移行，请增加语料或用 add_alpha")

        self.vocabulary = vocabulary
        self.character_index = character_index
        self.pi = start_counts / start_counts.sum()
        self.A = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        self.B = emission_counts / emission_counts.sum(axis=1, keepdims=True)
        self.log_pi = log_probabilities(self.pi)
        self.log_A = log_probabilities(self.A)
        self.log_B = log_probabilities(self.B)
        self.training_metadata = {
            "sentence_count": len(observations),
            "character_count": sum(map(len, observations)),
            "vocabulary_size": len(vocabulary) - 1,
        }
        self.training_words = {
            word
            for chars, states in zip(observations, tags)
            for word in decode("".join(chars), states)
        }
        self.training_hashes = {
            text_sha256(duplicate_key("".join(chars))) for chars in observations
        }
        return self

    def require_fitted(self) -> None:
        if not self.vocabulary:
            raise ResourceUnavailableError("HMM 尚未训练或加载")

    def predict(self, observations: list[str]) -> list[str]:
        """将字符映射到词表/UNK，再调用独立 Viterbi。"""
        from nlplab.sequence_labeling.viterbi import viterbi

        self.require_fitted()
        if any(not isinstance(char, str) or len(char) != 1 for char in observations):
            raise DataValidationError("预测观测必须是单个 Unicode 字符")
        indices = [self.character_index.get(character, 0) for character in observations]
        emissions = self.log_B[:, indices].T
        return viterbi(self.log_pi, self.log_A, emissions).tags

    def save(self, path: Path) -> None:
        """透明 JSON + NPZ 格式，不覆盖已有模型。"""
        import json
        from dataclasses import asdict
        from hashlib import sha256

        self.require_fitted()
        path.mkdir(parents=True, exist_ok=False)
        arrays = path / "probabilities.npz"
        np.savez_compressed(arrays, pi=self.pi, A=self.A, B=self.B)
        metadata = {
            "format_version": 1,
            "states": list(STATES),
            "vocabulary": self.vocabulary,
            "config": asdict(self.config),
            "training_metadata": self.training_metadata,
            "training_words": sorted(self.training_words),
            "training_hashes": sorted(self.training_hashes),
            "probabilities_sha256": sha256(arrays.read_bytes()).hexdigest(),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "HMMTagger":
        """检查形状、概率分布、BMES mask 和内容指纹，再重新生成 log 值。"""
        import json
        from hashlib import sha256

        if not (path / "metadata.json").is_file() or not (path / "probabilities.npz").is_file():
            raise ResourceUnavailableError(f"HMM 模型文件缺失: {path}")
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata["format_version"] != 1 or metadata["states"] != list(STATES):
                raise DataValidationError("HMM 模型格式版本或状态顺序错误")
            arrays = path / "probabilities.npz"
            if sha256(arrays.read_bytes()).hexdigest() != metadata["probabilities_sha256"]:
                raise DataValidationError("HMM 参数文件指纹不匹配")
            model = cls(HMMConfig(**metadata["config"]))
            vocabulary = metadata["vocabulary"]
            if (
                not isinstance(vocabulary, list)
                or not vocabulary
                or vocabulary[0] != UNK
                or len(set(vocabulary)) != len(vocabulary)
                or any(not isinstance(char, str) or len(char) != 1 for char in vocabulary[1:])
            ):
                raise DataValidationError("HMM 词表格式错误")
            with np.load(arrays, allow_pickle=False) as data:
                model.pi, model.A, model.B = data["pi"], data["A"], data["B"]
            if (
                model.pi.shape != (4,)
                or model.A.shape != (4, 4)
                or model.B.shape != (4, len(vocabulary))
            ):
                raise DataValidationError("HMM 参数形状错误")
            for array in (model.pi, model.A, model.B):
                if not np.isfinite(array).all() or (array < 0).any():
                    raise DataValidationError("HMM 参数必须是非负有限概率")
            if not (
                np.isclose(model.pi.sum(), 1)
                and np.allclose(model.A.sum(axis=1), 1)
                and np.allclose(model.B.sum(axis=1), 1)
            ):
                raise DataValidationError("HMM 概率分布未归一化")
            if (
                model.pi[1] != 0
                or model.pi[2] != 0
                or any(
                    model.A[row, column] != 0
                    for row, source in enumerate(STATES)
                    for column, target in enumerate(STATES)
                    if target not in LEGAL_NEXT[source]
                )
            ):
                raise DataValidationError("HMM 参数包含非法 BMES 起始/转移")
            model.vocabulary = vocabulary
            model.character_index = {char: index for index, char in enumerate(vocabulary)}
            model.training_metadata = metadata["training_metadata"]
            model.training_words = set(metadata["training_words"])
            model.training_hashes = set(metadata["training_hashes"])
            model.log_pi, model.log_A, model.log_B = map(
                log_probabilities, (model.pi, model.A, model.B)
            )
            return model
        except (ValueError, TypeError, KeyError, OSError) as error:
            raise DataValidationError(f"损坏的 HMM 模型: {path}: {error}") from error

    def parameters(self, topk: int = 20) -> dict[str, Any]:
        """完整 π/A 与每状态 Top-K 发射字符，避免输出完整 B。"""
        from dataclasses import asdict

        self.require_fitted()
        if type(topk) is not int or not 1 <= topk <= 100:
            raise DataValidationError("emission_topk 必须是 1–100 的整数")
        emissions = {}
        for state_index, state in enumerate(STATES):
            indices = np.argsort(-self.B[state_index, 1:], kind="stable")[:topk] + 1
            emissions[state] = [
                {
                    "character": self.vocabulary[index],
                    "probability": float(self.B[state_index, index]),
                }
                for index in indices
            ]
        return {
            "states": list(STATES),
            "start": self.pi.tolist(),
            "transition": self.A.tolist(),
            "emission_topk": emissions,
            "unknown_probabilities": self.B[:, 0].tolist(),
            "training": self.training_metadata,
            "config": asdict(self.config),
            "model": "hmm-cws",
        }
