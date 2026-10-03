"""固定 word TextCNN；无 PAD 响应污染；NPZ state_dict 无 pickle 保存。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nlplab.classification.stage3_config import ARCHITECTURE
from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.word_sequence import read_json


def set_seed(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 128, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(128, 128, k) for k in (3, 4, 5)])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(384, 5)

    def forward(self, token_ids, lengths, return_responses=False):
        # Padded inputs always right-padded, at least width 5, full valid windows only.
        if token_ids.ndim != 2 or token_ids.shape[1] < 5:
            raise DataValidationError("TextCNN 输入须为 batch × sequence，宽度至少 5")
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled, positions = [], []
        for convolution in self.convs:
            k = convolution.kernel_size[0]
            response = torch.relu(convolution(embedded))
            n = (lengths - k + 1).clamp_min(0)
            mask = torch.arange(response.shape[2], device=response.device)[None, :] < n[:, None]
            masked = response.masked_fill(~mask[:, None, :], float("-inf"))
            values, argmax = masked.max(dim=2)
            valid = n > 0
            values = torch.where(valid[:, None], values, torch.zeros_like(values))
            argmax = torch.where(valid[:, None], argmax, -torch.ones_like(argmax))
            pooled.append(values)
            positions.append(argmax)
        features = torch.cat(pooled, dim=1)
        logits = self.classifier(self.dropout(features))
        if return_responses:
            return logits, features, torch.cat(positions, dim=1)
        return logits

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, directory: Path, metadata=None):
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            directory / "state.npz",
            **{name: tensor.detach().cpu().numpy() for name, tensor in self.state_dict().items()},
        )
        write_json(
            directory / "model.json",
            {
                "model_id": "C3",
                "format": "numpy-npz-state-dict-no-pickle-v1",
                "vocabulary_size": self.embedding.num_embeddings,
                "architecture": ARCHITECTURE,
                "parameter_count": self.parameter_count,
                "label_order": list(CATEGORIES),
                "training": metadata or {},
            },
        )

    @classmethod
    def load(cls, directory, device="cpu"):
        meta = read_json(directory / "model.json")
        if meta["architecture"] != ARCHITECTURE or meta["label_order"] != list(CATEGORIES):
            raise DataValidationError("TextCNN architecture 或类别映射不符")
        model = cls(meta["vocabulary_size"])
        expected = model.state_dict()
        with np.load(directory / "state.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise DataValidationError("TextCNN state 参数键不符")
            state = {}
            for name, ref in expected.items():
                array = arrays[name]
                if array.shape != tuple(ref.shape) or array.dtype != np.float32:
                    raise DataValidationError("TextCNN state shape/dtype 不符")
                state[name] = torch.from_numpy(array.copy())
        model.load_state_dict(state, strict=True)
        if model.parameter_count != meta["parameter_count"]:
            raise DataValidationError("TextCNN 参数数目不符")
        return model.to(device).eval()


def explain_batch(model, encoded, vocabulary, top_n=12):
    """eval 模式的真实池化通道加法分解；不是 attention 或因果解释。"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, activations, positions = model(
            torch.as_tensor(encoded["ids"], device=device),
            torch.as_tensor(encoded["lengths"], device=device),
            True,
        )
        outputs = torch.softmax(logits, dim=1).cpu().numpy()
    logits, activations = logits.cpu().numpy(), activations.cpu().numpy()
    positions = positions.cpu().numpy()
    weights = model.classifier.weight.detach().cpu().numpy()
    bias = model.classifier.bias.detach().cpu().numpy()
    results = []
    for i, raw in enumerate(logits):
        order = sorted(range(5), key=lambda c: (-float(raw[c]), c))
        pred, runner = order[:2]
        channels = []
        for j, activation in enumerate(activations[i]):
            k, filter_id = (3, 4, 5)[j // 128], j % 128
            position = int(positions[i, j])
            span = encoded["tokens"][i][position : position + k] if position >= 0 else []
            original = (
                encoded["original_positions"][i][position : position + k] if position >= 0 else []
            )
            ids = encoded["ids"][i, position : position + k] if position >= 0 else []
            channels.append(
                {
                    "channel": j,
                    "kernel_size": k,
                    "filter_id": filter_id,
                    "max_activation": float(activation),
                    "argmax_position": position,
                    "token_span": span,
                    "model_token_span": [vocabulary.terms[int(t)] for t in ids],
                    "original_token_positions": original,
                    "crosses_truncation_join": any(
                        b != a + 1 for a, b in zip(original, original[1:])
                    ),
                    "predicted_class_weight": float(weights[pred, j]),
                    "contribution": float(activation) * float(weights[pred, j]),
                }
            )
        top = sorted(
            (c for c in channels if c["contribution"] > 0),
            key=lambda c: (-c["contribution"], c["channel"]),
        )[:top_n]
        contribution_sum = sum(c["contribution"] for c in channels)
        reconstructed = float(bias[pred]) + contribution_sum
        if not np.isclose(reconstructed, raw[pred], atol=2e-5, rtol=2e-5):
            raise DataValidationError("卷积贡献无法还原 raw logit")
        results.append(
            {
                "model_id": "C3",
                "feature_kind": "convolution_spans",
                "score_type": "softmax",
                "prediction": CATEGORIES[pred],
                "runner_up": CATEGORIES[runner],
                "model_score": float(outputs[i, pred]),
                "score_gap": float(raw[pred] - raw[runner]),
                "raw_logits": raw.tolist(),
                "softmax": outputs[i].tolist(),
                "scores": [
                    {"label": c, "value": float(outputs[i, j]), "raw_logit": float(raw[j])}
                    for j, c in enumerate(CATEGORIES)
                ],
                "all_channels": channels,
                "top_convolution_spans": top,
                "bias": float(bias[pred]),
                "channel_contribution_sum": contribution_sum,
                "reconstructed_logit": reconstructed,
                "raw_predicted_logit": float(raw[pred]),
                "explanation_note": (
                    "卷积最大响应片段；加法分解不是 attention、因果解释或已校准概率"
                ),
            }
        )
    return results
