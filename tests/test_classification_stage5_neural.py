"""Toy-only Stage5 TextCNN boundary and fixed-training contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nlplab.classification.stage5_neural as stage5_neural
from nlplab.classification.textcnn import TextCNN, explain_batch


def test_boundary_adapter_adds_no_parameters_and_preserves_unsegmented_outputs():
    torch.manual_seed(42)
    original = TextCNN(12).eval()
    boundary = stage5_neural.BoundaryTextCNN(12).eval()
    boundary.load_state_dict(original.state_dict())
    assert list(boundary.state_dict()) == list(original.state_dict())
    assert boundary.parameter_count == original.parameter_count

    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 0], [3, 4, 5, 6, 0, 0, 0]])
    lengths = torch.tensor([6, 4])
    with torch.inference_mode():
        expected = original(ids, lengths, return_responses=True)
        no_segments = boundary(ids, lengths, return_responses=True)
        one_segment = boundary(
            ids,
            lengths,
            return_responses=True,
            segment_ids=torch.zeros_like(ids),
        )
    for actual in (no_segments, one_segment):
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_boundary_mask_selects_only_windows_within_one_surviving_segment():
    model = stage5_neural.BoundaryTextCNN(12).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    lengths = torch.tensor([8])
    segments = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        _, _, positions = model(ids, lengths, return_responses=True, segment_ids=segments)

    allowed_starts = {
        3: {0, 2, 3, 4, 5},
        4: {2, 3, 4},
        5: {2, 3},
    }
    for kernel_index, kernel_size in enumerate((3, 4, 5)):
        selected = positions[0, kernel_index * 128 : (kernel_index + 1) * 128].tolist()
        assert set(selected) <= allowed_starts[kernel_size]


def test_cnn_channel_contributions_reconstruct_logits_without_boundaries():
    torch.manual_seed(7)
    model = stage5_neural.BoundaryTextCNN(10).eval()
    encoded = {
        "ids": np.asarray([[2, 3, 4, 5, 6, 0], [3, 4, 5, 0, 0, 0]], dtype=np.int64),
        "lengths": np.asarray([5, 3], dtype=np.int64),
        "tokens": [["a", "b", "c", "d", "e"], ["b", "c", "d"]],
        "original_positions": [[0, 1, 2, 3, 4], [0, 1, 2]],
    }
    vocabulary = SimpleNamespace(terms=["<PAD>", "<UNK>", *list("abcdefgh")])

    rows = explain_batch(model, encoded, vocabulary)
    assert len(rows) == 2
    for row in rows:
        assert row["reconstructed_logit"] == pytest.approx(row["raw_predicted_logit"], abs=2e-5)
        assert row["bias"] + row["channel_contribution_sum"] == pytest.approx(
            row["raw_predicted_logit"], abs=2e-5
        )


def test_clean_cnn_uses_exact_twelve_epoch_fixed_configuration(monkeypatch):
    monkeypatch.setattr(stage5_neural, "set_seed", lambda seed, threads=2: None)
    encoded = {
        "ids": np.asarray([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], dtype=np.int64),
        "lengths": np.asarray([5, 5], dtype=np.int64),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "segment_ids": np.zeros((2, 5), dtype=np.int64),
    }

    model, training = stage5_neural.train_clean(encoded, vocabulary_size=8)

    assert model.training is False
    assert training["seed"] == 42
    assert training["learning_rate"] == 0.001
    assert training["epochs"] == 12
    assert training["early_stopping"] is False
    assert [row["epoch"] for row in training["history"]] == list(range(1, 13))


def test_original_worker_reuses_the_requested_model_path_without_training(tmp_path, monkeypatch):
    root = tmp_path
    freeze_dir = root / "outputs/exp3/stage5/toy/freeze"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "cleaning_freeze.json").write_text(
        json.dumps({"files": {}, "source_hashes": {}}), encoding="utf-8"
    )
    model_path = root / "outputs/exp3/stage4/toy/points/ladder101_n420/C3/model"
    output = root / "outputs/exp3/stage5/toy/original_C3"
    vocabulary_path = freeze_dir / "vocabulary"
    encoding_path = freeze_dir / "OO"
    loaded_paths = []
    fake_model = object()
    vocabulary = SimpleNamespace(max_len=768, min_freq=2, terms=["<PAD>", "<UNK>"])

    monkeypatch.setattr(stage5_neural, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        stage5_neural.WordSequenceVocabulary,
        "load",
        classmethod(lambda cls, path: vocabulary),
    )

    def fake_load(cls, path):
        loaded_paths.append(Path(path))
        return fake_model

    monkeypatch.setattr(stage5_neural.BoundaryTextCNN, "load", classmethod(fake_load))
    monkeypatch.setattr(stage5_neural, "load_encoding", lambda path: {"toy": True})
    monkeypatch.setattr(
        stage5_neural,
        "infer",
        lambda model, encoded, vocabulary, removed: {"predictions": [], "inference_seconds": 0},
    )
    job = {
        "root": str(root),
        "freeze": str(freeze_dir),
        "mode": "original",
        "vocabulary": str(vocabulary_path),
        "model": str(model_path),
        "output": str(output),
        "evaluations": {"OO": {"encoding": str(encoding_path), "removed_spans": []}},
    }

    stage5_neural.run_job(job)
    result = json.loads((output / "OO.json").read_text(encoding="utf-8"))
    assert loaded_paths == [model_path]
    assert result["model_reference"] == str(model_path)

    retrain_job = {**job, "output": str(output.parent / "retrain_attempt"), "train": "forbidden"}
    with pytest.raises(ValueError, match="Original models must be loaded, not retrained"):
        stage5_neural.run_job(retrain_job)
