"""Frozen TextCNN parameters, with deletion-boundary masks only for ablated inputs."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from nlplab.classification.stage3_config import TRAINING
from nlplab.classification.stage3_neural import load_encoding
from nlplab.classification.textcnn import TextCNN, explain_batch, set_seed
from nlplab.core.io import write_json
from nlplab.data.classification_stage5 import CleaningFreeze, install_stage5_guard
from nlplab.features.shortcut_cleaning import overlap
from nlplab.features.word_sequence import WordSequenceVocabulary, read_json


class BoundaryTextCNN(TextCNN):
    """No new layers/parameters. Windows across removed spans cannot enter pooling."""

    def forward(self, token_ids, lengths, return_responses=False, segment_ids=None):
        segments = segment_ids if segment_ids is not None else getattr(self, "input_segments", None)
        if segments is None:
            return super().forward(token_ids, lengths, return_responses)
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled, positions = [], []
        for convolution in self.convs:
            k = convolution.kernel_size[0]
            response = torch.relu(convolution(embedded))
            windows = segments.unfold(1, k, 1)
            valid = (windows >= 0).all(-1) & (windows == windows[:, :, :1]).all(-1)
            valid &= torch.arange(response.shape[2])[None, :] < (lengths - k + 1)[:, None]
            values, argmax = response.masked_fill(~valid[:, None, :], float("-inf")).max(2)
            exists = valid.any(1)[:, None]
            pooled.append(torch.where(exists, values, torch.zeros_like(values)))
            positions.append(torch.where(exists, argmax, -torch.ones_like(argmax)))
        features = torch.cat(pooled, 1)
        logits = self.classifier(self.dropout(features))
        return (logits, features, torch.cat(positions, 1)) if return_responses else logits


def train_clean(encoded, vocabulary_size):
    set_seed(42, TRAINING["threads"])
    model = BoundaryTextCNN(vocabulary_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    x, lengths, y, segments = (
        torch.as_tensor(encoded[k]) for k in ("ids", "lengths", "labels", "segment_ids")
    )
    generator = torch.Generator().manual_seed(42)
    history, start = [], perf_counter()
    for epoch in range(1, 13):
        model.train()
        loss_sum = correct = 0
        for indices in torch.randperm(len(y), generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices], lengths[indices], segment_ids=segments[indices])
            loss = torch.nn.functional.cross_entropy(logits, y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == y[indices]).sum())
        history.append(
            {"epoch": epoch, "train_loss": loss_sum / len(y), "train_accuracy": correct / len(y)}
        )
        print(f"AA C3 epoch={epoch} loss={loss_sum / len(y):.5f}", flush=True)
    model.eval()
    return model, {
        "seed": 42,
        "learning_rate": 0.001,
        "epochs": 12,
        "early_stopping": False,
        "runtime_seconds": perf_counter() - start,
        "history": history,
        "training_config_reference": "Stage3 TRAINING; exact Stage4 frozen duration",
    }


def infer(model, encoded, vocabulary, removed):
    results, start = [], perf_counter()
    for offset in range(0, len(encoded["ids"]), 32):
        stop = offset + 32
        batch = {
            k: encoded[k][offset:stop] for k in ("ids", "lengths", "tokens", "original_positions")
        }
        model.input_segments = torch.as_tensor(encoded["segment_ids"][offset:stop])
        rows = explain_batch(model, batch, vocabulary)
        for i, row in enumerate(rows, offset):
            for channel in row["all_channels"]:
                p, k = channel["argmax_position"], channel["kernel_size"]
                offsets = encoded["character_offsets"][i][p : p + k] if p >= 0 else []
                channel["character_offsets"] = offsets
                for kind in ("source", "template"):
                    channel[kind + "_overlap"] = overlap(
                        offsets, [s for s in removed[i] if s["kind"] == kind]
                    )
            row["document_id"] = encoded["document_ids"][i]
        results.extend(rows)
    model.input_segments = None
    return {"predictions": results, "inference_seconds": perf_counter() - start}


def run_job(job):
    if set(job) - {
        "root",
        "freeze",
        "mode",
        "train",
        "vocabulary",
        "model",
        "output",
        "evaluations",
    }:
        raise ValueError("Unknown Stage5 fields; no Test or tuning inputs")
    freeze = CleaningFreeze(job["freeze"])
    freeze.verify()
    install_stage5_guard(job["root"], freeze)
    vocabulary = WordSequenceVocabulary.load(Path(job["vocabulary"]))
    if vocabulary.max_len != 768 or vocabulary.min_freq != 2:
        raise ValueError("Stage5 frozen vocabulary protocol changed")
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    if job["mode"] == "AA":
        train = load_encoding(Path(job["train"]))
        model, training = train_clean(train, len(vocabulary.terms))
        model.save(output / "model", training)
        write_json(output / "training.json", training)
    elif job["mode"] == "original":
        if "train" in job:
            raise ValueError("Original models must be loaded, not retrained")
        model = BoundaryTextCNN.load(Path(job["model"]))
    else:
        raise ValueError("Only original and AA modes are supported")
    for condition, spec in job["evaluations"].items():
        if condition not in {"OO", "OS", "OT", "OA", "AA"}:
            raise ValueError("Unknown ablation condition")
        encoded = load_encoding(Path(spec["encoding"]))
        result = infer(model, encoded, vocabulary, spec["removed_spans"])
        result["model_reference"] = str(output / "model") if job["mode"] == "AA" else job["model"]
        write_json(output / f"{condition}.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    run_job(read_json(parser.parse_args().job))
