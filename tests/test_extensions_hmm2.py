from itertools import product

import numpy as np
import pytest

from nlplab.core.exceptions import DataValidationError, NoValidPathError
from nlplab.extensions.hmm2 import TRIPLE_MASK, SecondOrderHMM, viterbi2
from nlplab.sequence_labeling.hmm import HMMConfig, log_probabilities
from nlplab.sequence_labeling.tags import STATES, is_legal_sequence


@pytest.fixture
def model():
    return SecondOrderHMM().fit(
        [list("甲乙丙丁"), list("甲乙丙"), list("甲乙丙丁")],
        [list("BMES"), list("SBE"), list("BESS")],
    )


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_hmm2_viterbi_exhaustive(model, length):
    observations = list("甲乙丙丁甲")[:length]
    ids = [model.character_index[c] for c in observations]
    emissions = model.base.log_B[:, ids].T
    pi, a, tensor = map(log_probabilities, [model.base.pi, model.initial, model.tensor])
    scored = []
    for path in product(range(4), repeat=length):
        if not is_legal_sequence([STATES[i] for i in path]):
            continue
        score = pi[path[0]] + emissions[0, path[0]]
        if length > 1:
            score += a[path[0], path[1]] + emissions[1, path[1]]
        for t in range(2, length):
            score += tensor[path[t - 2], path[t - 1], path[t]] + emissions[t, path[t]]
        scored.append((float(score), [STATES[i] for i in path]))
    best_score, best_path = max(scored, key=lambda p: p[0])
    trace = viterbi2(pi, a, tensor, emissions)
    assert trace["best_log_probability"] == pytest.approx(best_score)
    assert trace["tags"] == best_path


def test_hmm2_masks_unk_persistence(model, tmp_path):
    assert model.tensor.shape == (4, 4, 4)
    assert len(model.parameters(1)["emission_topk"]["B"]) == 1
    assert not model.tensor[~TRIPLE_MASK].any()
    assert "".join(model.segment("甲乙😀 丙").tokens) == "甲乙😀 丙"
    model.save(tmp_path / "hmm2")
    restored = SecondOrderHMM.load(tmp_path / "hmm2")
    assert restored.trace("甲乙丙") == model.trace("甲乙丙")
    with pytest.raises(DataValidationError):
        model.predict(["两个"])
    model.config = HMMConfig(smoothing="none")
    model.base.B[:, 0] = 0
    model.base.log_B = log_probabilities(model.base.B)
    with pytest.raises(NoValidPathError):
        model.predict(["😀"])


def test_hmm2_decoder_enforces_masks_and_invalid_input():
    trace = viterbi2(np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros((3, 4)))
    assert is_legal_sequence(trace["tags"])
    assert viterbi2(np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros((1, 4)))[
        "tags"
    ] == ["S"]
    with pytest.raises(DataValidationError):
        viterbi2(np.zeros(3), np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros((1, 4)))
