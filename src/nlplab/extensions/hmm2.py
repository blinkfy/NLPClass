"""可读二阶 HMM：HanLP companion 的首对初始化与状态对 Viterbi，加合法 BMES。"""

import json
from dataclasses import asdict

import numpy as np

from nlplab.core.exceptions import DataValidationError, NoValidPathError
from nlplab.core.io import write_json
from nlplab.extensions.benchmark import fingerprint
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.sequence_labeling.hmm import HMMConfig, HMMTagger, log_probabilities
from nlplab.sequence_labeling.tags import LEGAL_END, LEGAL_NEXT, LEGAL_START, STATES

SOURCE_REVISION = "dadd5c76a53eb05cc84720628b89e4a334d5d753"
SOURCE_URL = (
    "https://github.com/hankcs/HanLP/blob/"
    + SOURCE_REVISION
    + "/src/main/java/com/hankcs/hanlp/model/hmm/SecondOrderHiddenMarkovModel.java"
)
PAIR_MASK = np.array([[b in LEGAL_NEXT[a] for b in STATES] for a in STATES])
TRIPLE_MASK = PAIR_MASK[:, :, None] & PAIR_MASK[None, :, :]


def viterbi2(log_pi, log_initial, log_tensor, emissions) -> dict:
    """D[t,s,u]=max_r(D[t-1,r,s]+log A2[r,s,u])+log B[u,x_t]。"""
    if log_pi.shape != (4,) or log_initial.shape != (4, 4) or log_tensor.shape != (4, 4, 4):
        raise DataValidationError("二阶参数形状错误")
    if emissions.ndim != 2 or emissions.shape[1] != 4:
        raise DataValidationError("发射分数必须为 T×4")
    for array in (log_pi, log_initial, log_tensor, emissions):
        if np.isnan(array).any() or np.isposinf(array).any():
            raise DataValidationError("概率分数不得含 NaN/+inf")
    length = len(emissions)
    score = np.full((length, 4, 4), -np.inf)
    back = np.full((length, 4, 4), -1, dtype=int)
    if not length:
        return {"tags": [], "score": score, "backpointer": back, "best_log_probability": 0.0}
    first = np.where([s in LEGAL_START for s in STATES], log_pi + emissions[0], -np.inf)
    np.fill_diagonal(score[0], first)
    if length == 1:
        # 同时满足 BMES 的 start/end，长度 1 只能为 S。
        final = np.where([s in LEGAL_END for s in STATES], first, -np.inf)
        state = int(np.argmax(final))
        best = float(final[state])
        tags = [STATES[state]]
    else:
        initial = np.where(PAIR_MASK, log_initial, -np.inf)
        tensor = np.where(TRIPLE_MASK, log_tensor, -np.inf)
        score[1] = first[:, None] + initial + emissions[1][None, :]
        for t in range(2, length):
            candidates = score[t - 1][:, :, None] + tensor
            back[t] = np.argmax(candidates, axis=0)
            score[t] = np.max(candidates, axis=0) + emissions[t][None, :]
        final = np.where(np.array([s in LEGAL_END for s in STATES])[None, :], score[-1], -np.inf)
        previous, current = np.unravel_index(np.argmax(final), (4, 4))
        best = float(final[previous, current])
        indices = [0] * length
        indices[-2:] = [int(previous), int(current)]
        for t in range(length - 1, 1, -1):
            indices[t - 2] = int(back[t, indices[t - 1], indices[t]])
        tags = [STATES[s] for s in indices]
    if not np.isfinite(best):
        raise NoValidPathError("二阶 Viterbi 无有效合法 BMES 路径")
    return {
        "tags": tags,
        "score": score,
        "backpointer": back,
        "first_score": first,
        "best_log_probability": best,
    }


class SecondOrderHMM:
    name = "hmm2"

    def __init__(self, config: HMMConfig | None = None):
        self.config = config or HMMConfig()
        self.base = HMMTagger(self.config)
        self.initial = np.zeros((4, 4))
        self.tensor = np.zeros((4, 4, 4))

    @property
    def training_words(self):
        return self.base.training_words

    @property
    def character_index(self):
        return self.base.character_index

    def fit(self, observations, tags):
        self.base.fit(observations, tags)  # 只复用词表、π、发射与 codec；不改 HMM1。
        initial, tensor = np.zeros((4, 4)), np.zeros((4, 4, 4))
        for row in tags:
            ids = [STATES.index(s) for s in row]
            if len(ids) >= 2:
                initial[ids[0], ids[1]] += 1
            for a, b, c in zip(ids, ids[1:], ids[2:]):
                tensor[a, b, c] += 1
        alpha = self.config.alpha if self.config.smoothing == "add_alpha" else 0
        initial += alpha * (PAIR_MASK & np.array([s in LEGAL_START for s in STATES])[:, None])
        tensor += alpha * TRIPLE_MASK

        def normalize(counts):
            sums = counts.sum(axis=-1, keepdims=True)
            return np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 0)

        self.initial, self.tensor = normalize(initial), normalize(tensor)
        return self

    def require_fitted(self):
        self.base.require_fitted()

    def predict(self, observations):
        return self.decode_trace(observations)["tags"]

    def decode_trace(self, observations):
        self.require_fitted()
        if any(not isinstance(c, str) or len(c) != 1 for c in observations):
            raise DataValidationError("观测必须是单个 Unicode code point")
        indices = [self.character_index.get(c, 0) for c in observations]
        return viterbi2(
            self.base.log_pi,
            log_probabilities(self.initial),
            log_probabilities(self.tensor),
            self.base.log_B[:, indices].T,
        )

    def segment(self, text):
        result = HMMSegmenter(self).segment(text)
        result.model = self.name
        return result

    def trace(self, text):
        if len(text) > 30:
            raise DataValidationError("算法过程只支持 <=30 Unicode code point")
        if any(c.isspace() for c in text):
            import re

            chunks, offset = [], 0
            for chunk in re.split(r"(\s)", text):
                if not chunk:
                    continue
                chunks.append(
                    {
                        "offset": offset,
                        "text": chunk,
                        "trace": {"tags": ["S"], "whitespace": True}
                        if chunk.isspace()
                        else self.trace(chunk),
                    }
                )
                offset += len(chunk)
            result = self.segment(text)
            return {
                "model": self.name,
                "tokens": result.tokens,
                "tags": result.tags,
                "chunks": chunks,
                "states": list(STATES),
            }
        trace = self.decode_trace(list(text))
        result = self.segment(text)
        return {
            "model": self.name,
            "tokens": result.tokens,
            "tags": trace["tags"],
            "best_log_probability": trace["best_log_probability"],
            "score": [
                [[float(x) if np.isfinite(x) else None for x in row] for row in frame]
                for frame in trace["score"]
            ],
            "backpointer": trace["backpointer"].tolist(),
            "states": list(STATES),
        }

    def save(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.base.save(path / "emission_model")
        np.savez_compressed(path / "second_order.npz", initial=self.initial, tensor=self.tensor)
        write_json(
            path / "metadata.json",
            {
                "format_version": 1,
                "model": self.name,
                "config": asdict(self.config),
                "source_url": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "parameters_sha256": fingerprint(path / "second_order.npz"),
            },
        )

    @classmethod
    def load(cls, path):
        data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            data["format_version"] != 1
            or fingerprint(path / "second_order.npz") != data["parameters_sha256"]
        ):
            raise DataValidationError("二阶模型格式/指纹错误")
        model = cls(HMMConfig(**data["config"]))
        model.base = HMMTagger.load(path / "emission_model")
        if model.base.config != model.config:
            raise DataValidationError("二阶模型配置不一致")
        with np.load(path / "second_order.npz", allow_pickle=False) as arrays:
            model.initial, model.tensor = arrays["initial"], arrays["tensor"]
        for array, mask, shape in [
            (model.initial, PAIR_MASK, (4, 4)),
            (model.tensor, TRIPLE_MASK, (4, 4, 4)),
        ]:
            if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
                raise DataValidationError("二阶参数形状/数值错误")
            if array[~mask].any() or not np.all(
                np.isclose(array.sum(-1), 0) | np.isclose(array.sum(-1), 1)
            ):
                raise DataValidationError("二阶参数非法转移/未归一化")
        return model

    def parameters(self, topk: int = 20):
        return {
            **self.base.parameters(topk),
            "model": self.name,
            "initial_transition": self.initial.tolist(),
            "transition_tensor": self.tensor.tolist(),
            "legal_triples": TRIPLE_MASK.tolist(),
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "initialization": "pi*B at t=0; initial-pair A from first pairs at t=1",
        }
