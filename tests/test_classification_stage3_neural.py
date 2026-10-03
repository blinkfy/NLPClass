"""Toy-only Stage3 neural contracts; never fit or evaluate formal splits."""

import json

import numpy as np
import pytest
import torch

import nlplab.classification.stage3_neural as neural_job
from nlplab.classification.stage3_config import TRAINING
from nlplab.classification.textcnn import TextCNN, explain_batch
from nlplab.classification.textcnn_training import EarlyStopping, train_textcnn
from nlplab.core.exceptions import DataValidationError
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.word_sequence import (
    PAD,
    UNK,
    WordSequenceVocabulary,
    length_statistics,
    max_len_from_train,
    retain_head_tail,
)


def test_train_only_frequency_vocabulary_unknowns_and_right_padding(tmp_path):
    train_tokens = [["shared", "beta", "alpha"], ["shared", "alpha", "beta"]]
    vocabulary = WordSequenceVocabulary(max_len=5).fit(train_tokens)

    assert vocabulary.min_freq == 2
    assert vocabulary.terms == ["<PAD>", "<UNK>", "alpha", "beta", "shared"]
    assert vocabulary.vocabulary["<PAD>"] == PAD == 0
    assert vocabulary.vocabulary["<UNK>"] == UNK == 1

    encoded = vocabulary.encode([["shared", "validation_only"], ["alpha"]])
    np.testing.assert_array_equal(encoded["ids"], [[4, 1, 0, 0, 0], [2, 0, 0, 0, 0]])
    np.testing.assert_array_equal(encoded["lengths"], [2, 1])
    assert encoded["stats"]["unk_count"] == 1
    assert encoded["stats"]["retained_unk_count"] == 1

    vocabulary.save(tmp_path / "vocabulary")
    restored = WordSequenceVocabulary.load(tmp_path / "vocabulary")
    assert restored.terms == vocabulary.terms
    assert restored.vocabulary == vocabulary.vocabulary


def test_train_p95_length_rule_and_head_tail_source_positions():
    stats = length_statistics([["a"] * 8, ["b"] * 200])
    assert stats["percentile_method"] == "numpy linear"
    assert stats["count"] == 2
    assert stats["p95"] == pytest.approx(190.4)
    assert max_len_from_train(stats) == 192
    assert max_len_from_train(length_statistics([["x"] * 10])) == 128
    assert max_len_from_train(length_statistics([["x"] * 1000])) == 768

    kept, positions = retain_head_tail(list(range(9)), 5)
    assert kept == [0, 1, 2, 7, 8]
    assert positions == [0, 1, 2, 7, 8]
    with pytest.raises(DataValidationError, match="最大卷积核"):
        retain_head_tail(list(range(8)), 4)


def test_textcnn_shapes_and_valid_window_mask_ignores_large_padded_responses():
    torch.manual_seed(3)
    model = TextCNN(vocabulary_size=12).eval()
    assert model.embedding.embedding_dim == 128
    assert model.embedding.padding_idx == PAD
    assert [conv.kernel_size[0] for conv in model.convs] == [3, 4, 5]
    assert [conv.out_channels for conv in model.convs] == [128, 128, 128]
    assert model.dropout.p == 0.5
    assert model.classifier.in_features == 384
    assert model.classifier.out_features == len(CATEGORIES) == 5

    with torch.no_grad():
        model.embedding.weight[2:7].fill_(-1.0)
        for conv in model.convs:
            conv.weight.fill_(1.0)
            conv.bias.fill_(400.0)
    tokens = torch.tensor([[2, 3, 4, 5, 6]])
    padded = torch.tensor([[2, 3, 4, 5, 6, PAD, PAD, PAD]])
    lengths = torch.tensor([5])
    with torch.no_grad():
        logits, features, positions = model(tokens, lengths, return_responses=True)
        padded_logits, padded_features, padded_positions = model(
            padded, lengths, return_responses=True
        )
        # Confirm padded windows would win without the length mask.
        embedded = model.embedding(padded).transpose(1, 2)
        invalid_tail = torch.relu(model.convs[0](embedded))[0, :, 3:]
    assert logits.shape == padded_logits.shape == (1, 5)
    assert features.shape == padded_features.shape == (1, 384)
    assert positions.shape == padded_positions.shape == (1, 384)
    assert torch.all(invalid_tail > features[0, :128, None])
    torch.testing.assert_close(features, padded_features)
    torch.testing.assert_close(logits, padded_logits)
    torch.testing.assert_close(positions, padded_positions)
    assert torch.equal(model.embedding.weight[PAD], torch.zeros(128))

    short_logits, short_features, short_positions = model(
        torch.tensor([[2, 3, 4, PAD, PAD]]), torch.tensor([3]), return_responses=True
    )
    assert short_logits.shape == (1, 5)
    assert torch.all(short_features[:, :128] >= 0)
    assert torch.all(short_features[:, 128:] == 0)
    assert torch.all(short_positions[:, 128:] == -1)


def test_explanation_maps_argmax_spans_and_reconstructs_raw_logit():
    vocabulary = WordSequenceVocabulary(max_len=8).fit(
        [["alpha", "beta", "gamma", "delta", "epsilon", "alpha"]]
    )
    encoded = vocabulary.encode([["alpha", "beta", "gamma", "delta", "epsilon", "alpha"]])
    model = TextCNN(len(vocabulary.terms)).eval()

    explanation = explain_batch(model, encoded, vocabulary)[0]
    assert explanation["model_id"] == "C3"
    assert explanation["feature_kind"] == "convolution_spans"
    assert explanation["score_type"] == "softmax"
    assert explanation["prediction"] == CATEGORIES[int(np.argmax(explanation["raw_logits"]))]
    assert len(explanation["all_channels"]) == 384
    assert sum(explanation["softmax"]) == pytest.approx(1.0)
    assert explanation["model_score"] == pytest.approx(
        explanation["softmax"][CATEGORIES.index(explanation["prediction"])]
    )
    assert explanation["reconstructed_logit"] == pytest.approx(
        explanation["raw_predicted_logit"], abs=2e-5
    )
    assert explanation["bias"] + explanation["channel_contribution_sum"] == pytest.approx(
        explanation["raw_predicted_logit"], abs=2e-5
    )
    for channel in explanation["all_channels"]:
        position, width = channel["argmax_position"], channel["kernel_size"]
        if position < 0:
            assert channel["token_span"] == []
            continue
        assert channel["token_span"] == encoded["tokens"][0][position : position + width]
        assert channel["original_token_positions"] == encoded["original_positions"][0][
            position : position + width
        ]


def test_early_stopping_uses_strict_improvement_and_five_stale_epochs():
    stopping = EarlyStopping(patience=5)
    assert stopping.update(0.5, 1) == (True, False)
    assert stopping.update(0.5, 2) == (False, False)
    assert stopping.update(0.6, 3) == (True, False)
    assert [stopping.update(0.6, epoch)[1] for epoch in range(4, 9)] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert stopping.best_epoch == 3


def _toy_encoded_training_data():
    tokens = [
        [f"class{label}", "shared", f"word{label}", "context", "news"]
        for label in range(5)
        for _ in range(2)
    ]
    vocabulary = WordSequenceVocabulary(max_len=8).fit(tokens)
    encoded = vocabulary.encode(tokens)
    encoded["labels"] = np.repeat(np.arange(5, dtype=np.int64), 2)
    return vocabulary, encoded


def test_toy_training_is_deterministic_and_npz_round_trip_rejects_tampering(tmp_path):
    vocabulary, encoded = _toy_encoded_training_data()
    config = {**TRAINING, "batch_size": 10, "max_epochs": 2, "threads": 1}
    first, first_result = train_textcnn(
        encoded, len(vocabulary.terms), 0.001, 42, config=config, final_epochs=2
    )
    second, second_result = train_textcnn(
        encoded, len(vocabulary.terms), 0.001, 42, config=config, final_epochs=2
    )
    assert first_result["epochs_executed"] == second_result["epochs_executed"] == 2
    assert first_result["best_epoch"] == second_result["best_epoch"] == 2
    with torch.no_grad():
        x = torch.as_tensor(encoded["ids"])
        lengths = torch.as_tensor(encoded["lengths"])
        torch.testing.assert_close(first(x, lengths), second(x, lengths), rtol=0, atol=0)

    model_dir = tmp_path / "model"
    first.save(model_dir, {"seed": 42})
    arrays = np.load(model_dir / "state.npz", allow_pickle=False)
    assert arrays.files
    arrays.close()
    restored = TextCNN.load(model_dir)
    with torch.no_grad():
        torch.testing.assert_close(first(x, lengths), restored(x, lengths))

    with np.load(model_dir / "state.npz", allow_pickle=False) as source:
        state = {name: source[name].copy() for name in source.files}
    state[next(iter(state))] = state[next(iter(state))].reshape(-1)[:1]
    np.savez_compressed(model_dir / "state.npz", **state)
    with pytest.raises(DataValidationError, match="shape/dtype"):
        TextCNN.load(model_dir)


@pytest.mark.parametrize(
    ("mode", "extra"),
    [
        ("final", {"validation": "unused-val.npz"}),
        ("final", {"test": "unused-test.npz"}),
        ("validation", {"test": "unused-test.npz"}),
    ],
)
def test_training_jobs_reject_forbidden_split_inputs_before_writing(tmp_path, mode, extra):
    output = tmp_path / "must-not-be-created"
    job = {
        "mode": mode,
        "train": "unused-train.npz",
        "output": str(output),
        "learning_rate": 0.001,
        "seed": 42,
        "vocabulary_size": 8,
        **extra,
    }
    with pytest.raises(ValueError, match="禁止"):
        neural_job.run_job(job)
    assert not output.exists()


def _write_worker_encoding(path, encoded):
    arrays = {name: encoded[name] for name in ("ids", "lengths", "labels") if name in encoded}
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    metadata = {
        name: encoded[name]
        for name in ("tokens", "original_positions", "stats")
        if name in encoded
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_toy_worker_validation_final_and_infer_job_file_contracts(tmp_path, monkeypatch):
    vocabulary, encoded = _toy_encoded_training_data()
    fast_training = {**TRAINING, "batch_size": 10, "max_epochs": 1, "threads": 1}
    monkeypatch.setattr(neural_job, "TRAINING", fast_training)
    train_path = tmp_path / "train"
    val_path = tmp_path / "validation"
    test_path = tmp_path / "test"
    _write_worker_encoding(train_path, encoded)
    _write_worker_encoding(val_path, encoded)
    _write_worker_encoding(test_path, encoded)

    validation_output = tmp_path / "validation-output"
    neural_job.run_job(
        {
            "mode": "validation",
            "train": str(train_path),
            "validation": str(val_path),
            "output": str(validation_output),
            "vocabulary_size": len(vocabulary.terms),
            "learning_rate": 0.001,
            "seed": 42,
        }
    )
    validation_result = json.loads(
        (validation_output / "training_result.json").read_text("utf-8")
    )
    assert validation_result["early_stopping_used"] is True
    assert validation_result["epochs_executed"] == 1
    assert (validation_output / "model/state.npz").is_file()

    final_output = tmp_path / "final-output"
    neural_job.run_job(
        {
            "mode": "final",
            "train": str(train_path),
            "output": str(final_output),
            "vocabulary_size": len(vocabulary.terms),
            "learning_rate": 0.001,
            "seed": 42,
            "final_epochs": 1,
        }
    )
    final_result = json.loads((final_output / "training_result.json").read_text("utf-8"))
    assert final_result["early_stopping_used"] is False
    assert final_result["best_epoch"] == 1
    vocabulary.save(tmp_path / "vocabulary")
    inference_path = tmp_path / "predictions.json"
    neural_job.run_job(
        {
            "mode": "infer",
            "vocabulary": str(tmp_path / "vocabulary"),
            "encoding": str(test_path),
            "model": str(final_output / "model"),
            "output": str(inference_path),
        }
    )
    predictions = json.loads(inference_path.read_text("utf-8"))
    assert len(predictions) == len(encoded["ids"])
    assert all(row["model_id"] == "C3" for row in predictions)
    assert all(len(row["raw_logits"]) == 5 for row in predictions)
