import pytest

from nlplab.core.exceptions import ResourceUnavailableError
from nlplab.core.schema import SequenceSample
from nlplab.extensions.bilstm_crf import BiLSTMCRFConfig, BiLSTMCRFSegmenter
from nlplab.sequence_labeling.tags import is_legal_sequence

pytest.importorskip("torch")
pytest.importorskip("torchcrf")


def toy_samples():
    return [
        SequenceSample("a", "a", "中国", ["中国"]),
        SequenceSample("b", "b", "北京", ["北京"]),
        SequenceSample("c", "c", "大学", ["大学"]),
        SequenceSample("d", "d", "自然语言", ["自然语言"]),
    ]


def test_bilstm_crf_smoke_roundtrip_constraints_and_save_load(tmp_path):
    samples = toy_samples()
    config = BiLSTMCRFConfig(batch_size=2, max_epochs=1, cpu_threads=2)
    model = BiLSTMCRFSegmenter.from_samples(samples, config)
    result = model.fit(samples, samples, epochs=1)
    assert result["best_epoch"] == 1
    segmentation = model.segment("中国😀大学")
    assert "".join(segmentation.tokens) == "中国😀大学"
    assert is_legal_sequence(segmentation.tags)
    spaced = "中国 😀大学"
    assert model.trace(spaced)["tags"] == model.segment(spaced).tags
    assert model.trace("")["tokens"] == []
    assert model.parameters()["parameter_count"] > 0
    model.save(tmp_path / "model")
    restored = BiLSTMCRFSegmenter.load(tmp_path / "model")
    assert "".join(restored.segment("中国😀大学").tokens) == "中国😀大学"


def test_bilstm_crf_rejects_unfitted_model():
    model = BiLSTMCRFSegmenter(["<PAD>", "<UNK>", "中"])
    with pytest.raises(ResourceUnavailableError):
        model.segment("中")


def test_initialization_and_training_are_reproducible_and_dev_is_batched(tmp_path):
    import torch

    samples = toy_samples()
    config = BiLSTMCRFConfig(batch_size=2, max_epochs=1, cpu_threads=2)
    first = BiLSTMCRFSegmenter.from_samples(samples, config)
    second = BiLSTMCRFSegmenter.from_samples(samples, config)
    assert all(
        torch.equal(value, second.network.module.state_dict()[key])
        for key, value in first.network.module.state_dict().items()
    )
    first_result = first.fit(samples, samples, epochs=1, checkpoint_path=tmp_path / "checkpoint")
    second_result = second.fit(samples, samples, epochs=1)
    assert first_result["history"] == second_result["history"]
    assert first_result["history"][0]["dev_loss"] is not None
    assert (tmp_path / "checkpoint/progress.json").is_file()
    batches = []
    decode = first._decode_batch

    def observed_decode(items, indices):
        batches.append(len(indices))
        return decode(items, indices)

    first._decode_batch = observed_decode
    first.evaluate(samples * 3)
    assert max(batches) <= config.batch_size
    assert sum(batches) == 12
