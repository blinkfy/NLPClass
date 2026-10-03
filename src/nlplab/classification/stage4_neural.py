"""C3 Stage4 worker: fixed-duration Train-only fit, one post-fit Train/Val forward pass."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from nlplab.classification.stage3_config import TRAINING
from nlplab.classification.stage3_neural import load_encoding
from nlplab.classification.textcnn_training import train_textcnn
from nlplab.core.io import write_json
from nlplab.data.classification_development import install_development_io_guard
from nlplab.features.word_sequence import read_json


def evaluate(model, encoded, batch_size=32):
    model.eval()
    logits = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, len(encoded["ids"]), batch_size):
            x = torch.as_tensor(encoded["ids"][i : i + batch_size])
            lengths = torch.as_tensor(encoded["lengths"][i : i + batch_size])
            logits.extend(model(x, lengths).tolist())
    return {
        "prediction_indices": np.asarray(logits).argmax(axis=1).tolist(),
        "logits": logits,
        "inference_seconds": perf_counter() - start,
    }


def run_job(job):
    if set(job) - {
        "root",
        "train",
        "evaluation_val",
        "output",
        "vocabulary_size",
        "max_len",
        "learning_rate",
        "seed",
        "epochs",
        "toy",
    }:
        raise ValueError("Unexpected job fields; Test/early stopping prohibited")
    if (job["max_len"], job["learning_rate"], job["seed"], job["epochs"]) != (768, 0.001, 42, 12):
        raise ValueError("Stage4 C3 configuration must remain frozen")
    install_development_io_guard(Path(job["root"]))
    train = load_encoding(Path(job["train"]))
    if train["ids"].shape[1] != 768:
        raise ValueError("Stage4 sequence width must be 768")
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    model, training = train_textcnn(
        train, job["vocabulary_size"], 0.001, 42, config=TRAINING, validation=None, final_epochs=12
    )
    if training["early_stopping_used"] or training["epochs_executed"] != 12:
        raise ValueError("Stage4 must execute exactly 12 epochs")
    model.save(
        output / "model",
        {"stage": "development-learning-curve", "epochs": 12, "learning_rate": 0.001, "seed": 42},
    )
    # Validation encoding is opened only after the fixed training has completed.
    val = load_encoding(Path(job["evaluation_val"]))
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    write_json(
        output / "result.json",
        {
            "training": training,
            "train": evaluate(model, train),
            "val": evaluate(model, val),
            "state_sha256": digest.hexdigest(),
            "validation_prediction_count": 1,
            "test_evaluation_count": 0,
            "toy": job.get("toy", False),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run_job(read_json(args.job))


if __name__ == "__main__":
    main()
