"""Toy-only Stage4 neural worker contracts; no frozen split or formal fitting."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import nlplab.classification.stage4_neural as stage4_neural
from nlplab.features.word_sequence import UNK, WordSequenceVocabulary


def test_subset_vocabulary_is_refit_without_validation_only_terms():
    subset_tokens = [["shared", "subset_term"], ["shared", "subset_term"]]
    vocabulary = WordSequenceVocabulary(max_len=8).fit(subset_tokens)

    assert "subset_term" in vocabulary.vocabulary
    assert "validation_only_term" not in vocabulary.vocabulary
    encoded = vocabulary.encode([["validation_only_term", "shared"]])
    assert encoded["ids"][0, :2].tolist() == [UNK, vocabulary.vocabulary["shared"]]
    assert encoded["stats"]["unk_count"] == 1


def _write_encoding(path, rows=2, width=768):
    ids = np.zeros((rows, width), dtype=np.int64)
    ids[:, :5] = np.arange(2, 7, dtype=np.int64)
    lengths = np.full(rows, 5, dtype=np.int64)
    labels = np.arange(rows, dtype=np.int64) % 5
    np.savez_compressed(path.with_suffix(".npz"), ids=ids, lengths=lengths, labels=labels)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "tokens": [["a", "b", "c", "d", "e"]] * rows,
                "original_positions": [list(range(5))] * rows,
                "stats": {"document_count": rows},
            }
        ),
        encoding="utf-8",
    )


def _valid_job(root):
    return {
        "root": str(root),
        "train": str(root / "train"),
        "evaluation_val": str(root / "val"),
        "output": str(root / "worker-output"),
        "vocabulary_size": 8,
        "max_len": 768,
        "learning_rate": 0.001,
        "seed": 42,
        "epochs": 12,
        "toy": True,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [("max_len", 512), ("learning_rate", 0.0005), ("seed", 43), ("epochs", 30)],
)
def test_worker_rejects_changes_to_frozen_c3_fit_configuration(tmp_path, key, value):
    job = _valid_job(tmp_path)
    job[key] = value

    with pytest.raises(ValueError, match="configuration must remain frozen"):
        stage4_neural.run_job(job)
    assert not Path(job["output"]).exists()


def test_worker_rejects_test_job_fields_before_any_file_access(tmp_path):
    job = {**_valid_job(tmp_path), "test": str(tmp_path / "test.jsonl")}

    with pytest.raises(ValueError, match="Test/early stopping prohibited"):
        stage4_neural.run_job(job)
    assert not Path(job["output"]).exists()


def test_worker_uses_train_only_for_fixed_twelve_epochs_then_evaluates_train_and_val(
    tmp_path, monkeypatch
):
    train_path, val_path = tmp_path / "train", tmp_path / "val"
    _write_encoding(train_path, rows=2)
    _write_encoding(val_path, rows=3)
    events = []
    original_load = stage4_neural.load_encoding

    def tracked_load(path):
        events.append(f"load:{Path(path).name}")
        return original_load(path)

    class ToyModel:
        def __init__(self):
            self.batch_sizes = []

        def eval(self):
            return self

        def __call__(self, token_ids, lengths):
            self.batch_sizes.append(len(token_ids))
            events.append(f"forward:{len(token_ids)}")
            return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]).repeat(len(token_ids), 1)

        def state_dict(self):
            return {"toy-weight": torch.tensor([12.0])}

        def save(self, directory, metadata):
            directory.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(directory / "state.npz", **self.state_dict())
            (directory / "model.json").write_text(json.dumps(metadata), encoding="utf-8")

    model = ToyModel()
    training = {"early_stopping_used": False, "epochs_executed": 12, "best_epoch": 12}

    def fixed_train(
        train, vocabulary_size, learning_rate, seed, *, config, validation, final_epochs
    ):
        assert train["ids"].shape == (2, 768)
        assert train["labels"].tolist() == [0, 1]
        assert vocabulary_size == 8
        assert learning_rate == 0.001
        assert seed == 42
        assert config is stage4_neural.TRAINING
        assert validation is None
        assert final_epochs == 12
        assert events == ["guard", "load:train"]
        events.append("fit")
        return model, training

    monkeypatch.setattr(
        stage4_neural, "install_development_io_guard", lambda root: events.append("guard")
    )
    monkeypatch.setattr(stage4_neural, "load_encoding", tracked_load)
    monkeypatch.setattr(stage4_neural, "train_textcnn", fixed_train)
    job = _valid_job(tmp_path)
    stage4_neural.run_job(job)

    assert events == [
        "guard",
        "load:train",
        "fit",
        "load:val",
        "forward:2",
        "forward:3",
    ]
    assert model.batch_sizes == [2, 3]
    result = json.loads((Path(job["output"]) / "result.json").read_text("utf-8"))
    assert result["training"]["early_stopping_used"] is False
    assert result["training"]["epochs_executed"] == 12
    assert result["validation_prediction_count"] == 1
    assert result["test_evaluation_count"] == 0
    for split in ("train", "val"):
        logits = np.asarray(result[split]["logits"])
        predictions = np.asarray(result[split]["prediction_indices"])
        np.testing.assert_array_equal(logits.argmax(axis=1), predictions)
        np.testing.assert_array_equal(predictions, np.full(len(predictions), 4))
    assert len(result["state_sha256"]) == 64


def test_worker_rejects_non_768_encoded_width_before_creating_output(tmp_path, monkeypatch):
    train_path = tmp_path / "train"
    _write_encoding(train_path, rows=1, width=128)
    monkeypatch.setattr(stage4_neural, "install_development_io_guard", lambda root: None)
    job = _valid_job(tmp_path)

    with pytest.raises(ValueError, match="sequence width must be 768"):
        stage4_neural.run_job(job)
    assert not Path(job["output"]).exists()
