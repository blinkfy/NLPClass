"""PyTorch BiLSTM-CRF 应用：字符序列输入，BMES 序列输出。

CRF 的 forward、partition 与 Viterbi 来自 pytorch-crf；本模块只负责
中文数据编码、BiLSTM 发射层、BMES 约束适配和透明持久化。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from nlplab.core.exceptions import DataValidationError, NoValidPathError, ResourceUnavailableError
from nlplab.core.io import write_json
from nlplab.core.schema import SegmentationResult
from nlplab.evaluation.segmentation import evaluate_segmentation
from nlplab.extensions.benchmark import fingerprint
from nlplab.segmentation.bmes import decode, encode
from nlplab.sequence_labeling.tags import (
    LEGAL_END,
    LEGAL_NEXT,
    LEGAL_START,
    STATES,
    is_legal_sequence,
)

TAG_TO_ID = {tag: index for index, tag in enumerate(STATES)}
PAD = "<PAD>"
UNK = "<UNK>"


@dataclass(frozen=True)
class BiLSTMCRFConfig:
    embedding_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    optimizer: str = "AdamW"
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 10
    patience: int = 2
    gradient_clip: float = 5.0
    seed: int = 42
    device: str = "cpu"
    cpu_threads: int = 2
    legal_bmes: bool = True
    crf_package: str = "pytorch-crf"
    crf_version: str = "0.7.2"

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW" or self.crf_package != "pytorch-crf":
            raise DataValidationError("BiLSTM-CRF 只允许固定的 AdamW/pytorch-crf 配置")
        if self.num_layers < 1 or self.embedding_dim < 1 or self.hidden_size < 1:
            raise DataValidationError("BiLSTM-CRF 网络维度非法")
        if not 0 < self.lr or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 0:
            raise DataValidationError("BiLSTM-CRF 训练配置非法")


def _torch_import():
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from torchcrf import CRF
    except ImportError as error:  # pragma: no cover - exercised in unavailable envs
        raise ResourceUnavailableError(
            "BiLSTM-CRF 需要 sequence-neural 环境：torch 2.4–2.14 与 pytorch-crf==0.7.2"
        ) from error
    return torch, nn, pack_padded_sequence, pad_packed_sequence, CRF


def _set_deterministic(seed: int, threads: int) -> None:
    torch, _, _, _, _ = _torch_import()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        # PyTorch 只允许在首次并行计算前设置 interop threads。
        pass
    torch.use_deterministic_algorithms(True)


class _MaskedParameter:
    """register_parametrization 所需的有限负值 mask，避免 -inf*0 产生 NaN。"""

    @staticmethod
    def install(module, name: str, mask):
        torch, nn, _, _, _ = _torch_import()
        from torch.nn.utils.parametrize import register_parametrization

        class Mask(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))

            def forward(self, value):
                return torch.where(self.mask, value, torch.full_like(value, -10000.0))

        register_parametrization(module, name, Mask())


class ConstrainedCRF:
    """将库 CRF 的三组参数映射到合法 BMES mask。"""

    def __init__(self, crf):
        self.crf = crf
        starts = [state in LEGAL_START for state in STATES]
        ends = [state in LEGAL_END for state in STATES]
        transitions = [[target in LEGAL_NEXT[source] for target in STATES] for source in STATES]
        _MaskedParameter.install(crf, "start_transitions", starts)
        _MaskedParameter.install(crf, "end_transitions", ends)
        _MaskedParameter.install(crf, "transitions", transitions)

    def __getattr__(self, name):
        return getattr(self.crf, name)


class _BiLSTMCRFNetwork:
    """延迟导入 torch 的网络对象，方便主环境继续运行旧 HMM 测试。"""

    def __init__(self, vocab_size: int, config: BiLSTMCRFConfig):
        torch, nn, _, _, CRF = _torch_import()

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    config.embedding_dim,
                    config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0.0,
                    bidirectional=True,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(config.dropout)
                self.emission = nn.Linear(config.hidden_size * 2, len(STATES))
                self.crf = CRF(len(STATES), batch_first=True)
                self.constrained_crf = ConstrainedCRF(self.crf)

            def emissions(self, characters, lengths):
                _, _, pack, unpack, _ = _torch_import()
                embedded = self.embedding(characters)
                packed = pack(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
                encoded, _ = self.lstm(packed)
                padded, _ = unpack(encoded, batch_first=True, total_length=characters.shape[1])
                return self.emission(self.dropout(padded))

        self.module = Network()
        self.torch = torch

    def __getattr__(self, name):
        return getattr(self.module, name)


class BiLSTMCRFSegmenter:
    """训练完成后的 Segmenter，接口可被独立 benchmark runner 复用。"""

    name = "bilstm-crf"

    def __init__(self, vocabulary: list[str], config: BiLSTMCRFConfig | None = None):
        if (
            not vocabulary
            or vocabulary[:2] != [PAD, UNK]
            or len(set(vocabulary)) != len(vocabulary)
        ):
            raise DataValidationError("BiLSTM-CRF 词表必须以 PAD/UNK 开头且无重复")
        self.config = config or BiLSTMCRFConfig()
        self.vocabulary = vocabulary
        self.character_index = {char: index for index, char in enumerate(vocabulary)}
        _set_deterministic(self.config.seed, self.config.cpu_threads)
        self.network = _BiLSTMCRFNetwork(len(vocabulary), self.config)
        self.training_words: set[str] = set()
        self.training_metadata: dict[str, Any] = {}
        self._device = self.network.torch.device(self.config.device)
        self.network.module.to(self._device)

    @classmethod
    def from_samples(cls, samples: list, config: BiLSTMCRFConfig | None = None):
        chars = sorted({char for sample in samples for char in sample.text})
        return cls([PAD, UNK, *chars], config)

    def require_fitted(self):
        if not self.training_metadata:
            raise ResourceUnavailableError("BiLSTM-CRF 尚未训练或加载")

    def _encode(self, text: str) -> list[int]:
        return [self.character_index.get(char, self.character_index[UNK]) for char in text]

    def _batch(self, samples: list, indices: list[int]):
        torch = self.network.torch
        ordered = sorted(indices, key=lambda i: len(samples[i].text), reverse=True)
        lengths = torch.tensor([len(samples[i].text) for i in ordered], dtype=torch.long)
        width = int(lengths.max())
        characters = torch.zeros((len(ordered), width), dtype=torch.long)
        tags = torch.zeros((len(ordered), width), dtype=torch.long)
        mask = torch.zeros((len(ordered), width), dtype=torch.bool)
        for row, index in enumerate(ordered):
            text = samples[index].text
            tag_ids = [TAG_TO_ID[tag] for tag in encode(samples[index].gold_tokens)]
            characters[row, : len(text)] = torch.tensor(self._encode(text), dtype=torch.long)
            tags[row, : len(text)] = torch.tensor(tag_ids, dtype=torch.long)
            mask[row, : len(text)] = True
        return (
            ordered,
            characters.to(self._device),
            tags.to(self._device),
            mask.to(self._device),
            lengths.to(self._device),
        )

    def _decode_batch(self, samples: list, indices: list[int]) -> list[list[str]]:
        self.network.module.eval()
        with self.network.torch.no_grad():
            ordered, chars, _, mask, lengths = self._batch(samples, indices)
            emissions = self.network.emissions(chars, lengths)
            decoded = self.network.crf.decode(emissions, mask=mask)
        result: dict[int, list[str]] = {}
        for row, index in enumerate(ordered):
            tags = [STATES[tag] for tag in decoded[row]]
            if len(tags) != len(samples[index].text) or not is_legal_sequence(tags):
                raise NoValidPathError("BiLSTM-CRF 解码产生非法 BMES 序列")
            result[index] = tags
        return [result[index] for index in indices]

    def evaluate(self, samples: list, training_words: set[str] | None = None) -> dict[str, Any]:
        predictions = []
        for start in range(0, len(samples), self.config.batch_size):
            predictions.extend(
                self._decode_batch(
                    samples, list(range(start, min(start + self.config.batch_size, len(samples))))
                )
            )
        return evaluate_segmentation(
            [sample.gold_tokens for sample in samples],
            [decode(sample.text, tags) for sample, tags in zip(samples, predictions)],
            training_words if training_words is not None else self.training_words,
        )

    def loss(self, samples: list) -> float | None:
        """在给定 Gold 上计算平均 token NLL；不执行参数更新。"""
        if not samples:
            return None
        total, count = 0.0, 0
        self.network.module.eval()
        with self.network.torch.no_grad():
            for start in range(0, len(samples), self.config.batch_size):
                _, chars, tags, mask, lengths = self._batch(
                    samples, list(range(start, min(start + self.config.batch_size, len(samples))))
                )
                emissions = self.network.emissions(chars, lengths)
                value = -self.network.crf(emissions, tags, mask=mask, reduction="token_mean")
                tokens = int(mask.sum())
                total += float(value) * tokens
                count += tokens
        return total / count

    def fit(
        self,
        train_samples: list,
        dev_samples: list,
        *,
        epochs: int | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        """固定 seed 训练；dev 只用于 early stopping/best epoch，不读取 official test。"""
        _set_deterministic(self.config.seed, self.config.cpu_threads)
        torch = self.network.torch
        optimizer = torch.optim.AdamW(self.network.module.parameters(), lr=self.config.lr)
        limit = epochs or self.config.max_epochs
        history, best_epoch, best_f1, stale = [], 0, -1.0, 0
        self.training_words = {word for sample in train_samples for word in sample.gold_tokens}
        started = perf_counter()
        order_rng = random.Random(self.config.seed)
        for epoch in range(1, limit + 1):
            self.network.module.train()
            indices = list(range(len(train_samples)))
            order_rng.shuffle(indices)
            # 先按长度分桶，再在桶内按 epoch seed 打乱；完整保留每条句子。
            indices.sort(key=lambda i: len(train_samples[i].text))
            buckets = [
                indices[i : i + self.config.batch_size * 8]
                for i in range(0, len(indices), self.config.batch_size * 8)
            ]
            random.Random(self.config.seed + epoch).shuffle(buckets)
            for bucket in buckets:
                random.Random(self.config.seed + epoch + len(bucket)).shuffle(bucket)
            ordered_indices = [index for bucket in buckets for index in bucket]
            total_loss, count = 0.0, 0
            for start in range(0, len(ordered_indices), self.config.batch_size):
                batch_indices = ordered_indices[start : start + self.config.batch_size]
                _, chars, tags, mask, lengths = self._batch(train_samples, batch_indices)
                optimizer.zero_grad(set_to_none=True)
                emissions = self.network.emissions(chars, lengths)
                loss = -self.network.crf(emissions, tags, mask=mask, reduction="token_mean")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.module.parameters(), self.config.gradient_clip
                )
                optimizer.step()
                total_loss += float(loss.detach()) * int(mask.sum())
                count += int(mask.sum())
            dev_metrics = self.evaluate(dev_samples, self.training_words) if dev_samples else None
            dev_loss = self.loss(dev_samples)
            item = {
                "epoch": epoch,
                "train_loss": total_loss / count,
                "dev_loss": dev_loss,
                "dev_f1": dev_metrics["f1"] if dev_metrics else None,
                "dev_precision": dev_metrics["precision"] if dev_metrics else None,
                "dev_recall": dev_metrics["recall"] if dev_metrics else None,
                "dev_tag_accuracy": dev_metrics["tag_accuracy"] if dev_metrics else None,
            }
            history.append(item)
            print(json.dumps({"phase": "training", **item}), flush=True)
            if epochs is not None or item["dev_f1"] is None:
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.module.state_dict().items()
                }
                if item["dev_f1"] is not None:
                    best_f1 = item["dev_f1"]
            elif item["dev_f1"] > best_f1:
                best_f1, best_epoch, stale = item["dev_f1"], epoch, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.module.state_dict().items()
                }
            else:
                stale += 1
            if checkpoint_path is not None:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint_path / "checkpoint.tmp"
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": self.network.module.state_dict(),
                        "best_state_dict": best_state,
                        "optimizer": optimizer.state_dict(),
                    },
                    temporary,
                )
                temporary.replace(checkpoint_path / "checkpoint.pt")
                write_json(
                    checkpoint_path / "progress.json",
                    {
                        "history": history,
                        "best_epoch": best_epoch,
                        "best_dev_f1": best_f1,
                        "config": asdict(self.config),
                        "vocabulary": self.vocabulary,
                    },
                )
            if epochs is None and stale and stale >= self.config.patience:
                break
        self.network.module.load_state_dict(best_state)
        self.training_metadata = {
            "sentence_count": len(train_samples),
            "character_count": sum(len(s.text) for s in train_samples),
            "vocabulary_size": len(self.vocabulary) - 2,
            "parameter_count": sum(
                parameter.numel() for parameter in self.network.module.parameters()
            ),
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "training_seconds": perf_counter() - started,
            "device": str(self._device),
            "cpu_threads": self.config.cpu_threads,
            "torch_version": torch.__version__,
            "crf_version": "0.7.2",
            "seed": self.config.seed,
        }
        self.training_metadata["training_words_count"] = len(self.training_words)
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_dev_f1": best_f1,
            "metadata": self.training_metadata,
        }

    def predict(self, observations: list[str]) -> list[str]:
        text = "".join(observations)
        if len(text) != len(observations) or any(
            not isinstance(c, str) or len(c) != 1 for c in observations
        ):
            raise DataValidationError("观测必须是单个 Unicode code point")

        class Sample:
            pass

        sample = Sample()
        sample.text, sample.gold_tokens = text, [text]
        return self._decode_batch([sample], [0])[0]

    def segment(self, text: str) -> SegmentationResult:
        self.require_fitted()
        tokens, tags = [], []
        for part in __import__("re").split(r"(\s)", text):
            if not part:
                continue
            part_tags = ["S"] if part.isspace() else self.predict(list(part))
            tokens.extend(decode(part, part_tags))
            tags.extend(part_tags)
        if "".join(tokens) != text:
            raise DataValidationError("BiLSTM-CRF 分词未 roundtrip")
        return SegmentationResult(self.name, tokens, tags)

    def trace(self, text: str) -> dict[str, Any]:
        if len(text) > 30:
            raise DataValidationError("算法过程只支持 <=30 Unicode code point")
        if not text or any(char.isspace() for char in text):
            result = self.segment(text)
            chunks, offset = [], 0
            for chunk in re.split(r"(\s)", text):
                if not chunk:
                    continue
                chunks.append(
                    {
                        "offset": offset,
                        "text": chunk,
                        **(
                            {"tags": ["S"], "whitespace": True}
                            if chunk.isspace()
                            else self.trace(chunk)
                        ),
                    }
                )
                offset += len(chunk)
            return {
                "model": self.name,
                "tokens": result.tokens,
                "tags": result.tags,
                "chunks": chunks,
                "architecture": self.parameters()["architecture"],
                "crf_transitions": self.network.crf.transitions.detach().cpu().tolist(),
            }
        tags = self.predict(list(text))
        self.network.module.eval()
        torch = self.network.torch
        with torch.no_grad():
            chars = torch.tensor([self._encode(text)], dtype=torch.long, device=self._device)
            lengths = torch.tensor([len(text)], dtype=torch.long, device=self._device)
            emissions = self.network.emissions(chars, lengths)[0].cpu().tolist()
        return {
            "model": self.name,
            "architecture": self.parameters()["architecture"],
            "emissions": emissions,
            "tags": tags,
            "tokens": decode(text, tags),
            "crf_transitions": self.network.crf.transitions.detach().cpu().tolist(),
        }

    def parameters(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "architecture": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_size_per_direction": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "output": "Linear(2*hidden, 4 BMES) -> CRF",
            },
            "config": asdict(self.config),
            "vocabulary_size": len(self.vocabulary) - 2,
            "parameter_count": sum(
                parameter.numel() for parameter in self.network.module.parameters()
            ),
            "training": self.training_metadata,
            "crf_transition_matrix": self.network.crf.transitions.detach().cpu().tolist(),
        }

    def save(self, path: Path) -> None:
        self.require_fitted()
        path.mkdir(parents=True, exist_ok=False)
        torch = self.network.torch
        torch.save(self.network.module.state_dict(), path / "state_dict.pt")
        write_json(
            path / "metadata.json",
            {
                "format_version": 1,
                "model": self.name,
                "vocabulary": self.vocabulary,
                "config": asdict(self.config),
                "training_metadata": self.training_metadata,
                "training_words": sorted(self.training_words),
                "state_dict_sha256": fingerprint(path / "state_dict.pt"),
            },
        )

    @classmethod
    def load(cls, path: Path):
        torch, _, _, _, _ = _torch_import()
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["format_version"] != 1
            or fingerprint(path / "state_dict.pt") != metadata["state_dict_sha256"]
        ):
            raise DataValidationError("BiLSTM-CRF 模型格式或 state_dict 指纹错误")
        model = cls(metadata["vocabulary"], BiLSTMCRFConfig(**metadata["config"]))
        model.network.module.load_state_dict(
            torch.load(path / "state_dict.pt", map_location=model._device, weights_only=True)
        )
        model.training_metadata = metadata["training_metadata"]
        model.training_words = set(metadata["training_words"])
        return model
