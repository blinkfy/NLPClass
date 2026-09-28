from itertools import product

import numpy as np
import pytest

from nlplab.core.exceptions import NoValidPathError
from nlplab.sequence_labeling.hmm import HMMConfig, HMMTagger
from nlplab.sequence_labeling.tags import STATES, is_legal_sequence
from nlplab.sequence_labeling.viterbi import viterbi


@pytest.mark.parametrize("text", ["", "商", "商品和服务", "未见🙂字", "北京大学"])
def test_prediction_constraints(toy_model, text) -> None:
    tags = toy_model.predict(list(text))
    assert len(tags) == len(text)
    assert is_legal_sequence(tags)
    if tags:
        assert tags[0] in {"B", "S"}
        assert tags[-1] in {"E", "S"}


def test_dynamic_programming_matches_exhaustive_search(toy_model) -> None:
    model = toy_model
    text = "商品和"
    emissions = model.log_B[:, [model.character_index[char] for char in text]].T
    trace = viterbi(model.log_pi, model.log_A, emissions)
    candidates = []
    for tags in product(STATES, repeat=len(text)):
        if not is_legal_sequence(list(tags)):
            continue
        indices = [STATES.index(tag) for tag in tags]
        value = model.log_pi[indices[0]] + emissions[0, indices[0]]
        for time in range(1, len(text)):
            value += model.log_A[indices[time - 1], indices[time]] + emissions[time, indices[time]]
        candidates.append((float(value), list(tags)))
    best = max(candidates, key=lambda item: item[0])
    assert np.isclose(trace.best_log_probability, best[0])
    assert trace.tags == best[1]
    assert trace.score.shape == trace.backpointer.shape == (3, 4)


def test_unsmoothed_unknown_does_not_fabricate(toy_tokens) -> None:
    from nlplab.segmentation.bmes import encode

    model = HMMTagger(HMMConfig(smoothing="none")).fit(
        [list("".join(tokens)) for tokens in toy_tokens], [encode(tokens) for tokens in toy_tokens]
    )
    with pytest.raises(NoValidPathError):
        model.predict(["🙂"])
