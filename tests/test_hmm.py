import numpy as np
import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.hmm import HMMConfig, HMMTagger
from nlplab.sequence_labeling.tags import LEGAL_NEXT, STATES


def test_probabilities_and_mask(toy_model) -> None:
    model = toy_model
    assert np.isclose(model.pi.sum(), 1)
    assert np.allclose(model.A.sum(axis=1), 1)
    assert np.allclose(model.B.sum(axis=1), 1)
    assert model.pi[1] == model.pi[2] == 0
    for row, source in enumerate(STATES):
        for column, target in enumerate(STATES):
            if target not in LEGAL_NEXT[source]:
                assert model.A[row, column] == 0
                assert model.log_A[row, column] == -np.inf
    assert np.all(model.B[:, 0] > 0)  # UNK 的概率来自平滑统计。


def test_mle_counts(toy_tokens) -> None:
    model = HMMTagger(HMMConfig(smoothing="none")).fit(
        [list("".join(words)) for words in toy_tokens], [encode(words) for words in toy_tokens]
    )
    assert np.isclose(model.pi[0], 4 / 5)
    assert np.isclose(model.pi[3], 1 / 5)
    assert np.all(model.B[:, 0] == 0)
    assert np.allclose(model.B.sum(axis=1), 1)


def test_training_rejects_illegal_gold() -> None:
    with pytest.raises(DataValidationError):
        HMMTagger().fit([list("商品")], [["M", "E"]])
    with pytest.raises(DataValidationError):
        HMMTagger(HMMConfig(smoothing="none")).fit([["商"]], [["S"]])


@pytest.mark.parametrize("config_name", ["hmm_mle.yaml", "hmm_smooth.yaml"])
def test_course_configs_preserve_masks_and_probability_estimates(config_name, toy_tokens):
    from pathlib import Path

    from nlplab.cli import read_config
    from nlplab.sequence_labeling.tags import is_legal_sequence

    config = read_config(Path("configs") / config_name)
    model = HMMTagger(
        HMMConfig(
            **{
                key: config[key]
                for key in ("smoothing", "alpha", "unknown_strategy")
                if key in config
            }
        )
    ).fit(
        [list("".join(tokens)) for tokens in toy_tokens], [encode(tokens) for tokens in toy_tokens]
    )
    alpha = 0 if config_name == "hmm_mle.yaml" else 0.1
    assert model.config.smoothing == ("none" if alpha == 0 else "add_alpha")
    assert model.config.alpha == 0.1
    assert model.pi[0] == pytest.approx((4 + alpha) / (5 + 2 * alpha))
    assert model.pi[1] == model.pi[2] == 0
    for row, source in enumerate(STATES):
        for column, target in enumerate(STATES):
            if target not in LEGAL_NEXT[source]:
                assert model.A[row, column] == 0
                assert np.isneginf(model.log_A[row, column])
    assert np.allclose(model.A.sum(axis=1), 1)
    assert np.allclose(model.B.sum(axis=1), 1)
    assert np.all(model.B[:, 0] == 0) if alpha == 0 else np.all(model.B[:, 0] > 0)
    assert is_legal_sequence(model.predict(list("商品和服务")))
