"""Focused contracts for relation-score teaching traces; no model loads."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nlplab.course_neural.dependency import tree_record
from nlplab.course_neural.relation_trace import relation_trace


class _Tensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        value = self.value
        for part in key if isinstance(key, tuple) else (key,):
            value = value[part]
        return _Tensor(value) if isinstance(value, list) else value

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self

    def tolist(self):
        return self.value


def test_worker_uses_decoded_heads_and_parser_relation_vocabulary(tmp_path, monkeypatch):
    from nlplab.course_neural import worker

    labels = ["root", "nsubj", "obj"]
    relation_tensor = [[[[0.0, 0.0, 0.0] for _ in range(4)] for _ in range(4)]]
    relation_tensor[0][1][2] = [0.0, 2.0, 5.0]
    relation_tensor[0][1][0] = [0.0, 8.0, 1.0]  # Wrong if head 0 is used.
    relation_tensor[0][2][0] = [7.0, 0.0, 1.0]
    relation_tensor[0][3][2] = [0.0, 6.0, 3.0]
    relation_tensor[0][3][1] = [0.0, 1.0, 9.0]  # Wrong if head 1 is used.
    arc_tensor = [[[0.0] * 4 for _ in range(4)]]

    class _Hook:
        def remove(self):
            pass

    class _Model:
        def eval(self):
            pass

        def register_forward_hook(self, hook):
            self.hook = hook
            return _Hook()

    class _Parser:
        model = _Model()
        config = SimpleNamespace(tree=True, proj=False)
        vocabs = {"rel": SimpleNamespace(idx_to_token=labels)}

        def __call__(self, _words):
            self.model.hook(
                self.model,
                (),
                (_Tensor(arc_tensor), _Tensor(relation_tensor)),
            )
            return [
                SimpleNamespace(head=2, deprel="obj"),
                SimpleNamespace(head=0, deprel="root"),
                SimpleNamespace(head=2, deprel="nsubj"),
            ]

    parser = _Parser()
    monkeypatch.setitem(sys.modules, "hanlp", SimpleNamespace(load=lambda *_args, **_kwargs: parser))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    model_dir = tmp_path / "parser"
    model_dir.mkdir()
    (model_dir / "model.pt").touch()
    resource = {"files": [{"path": "parser/config.json"}], "archive": {"sha256": "fixture"}}
    monkeypatch.setattr(worker, "prepared_resource", lambda *_args: (Path(tmp_path), resource))

    result = worker.dependency(tmp_path, {"sentences": [["甲", "乙", "丙"]]})
    traces = result["sentences"][0]["relation_scores"]

    assert [(row["head"], row["relation"]) for row in traces] == [
        (2, "obj"),
        (0, "root"),
        (2, "nsubj"),
    ]
    assert [candidate["label"] for candidate in traces[0]["candidates"]] == [
        "obj",
        "nsubj",
        "root",
    ]
    assert traces[0]["candidates"][0]["score"] == 5.0
    assert traces[2]["candidates"][0]["label"] == "nsubj"


def test_relation_trace_orders_scores_and_breaks_ties_by_vocab_index():
    rows = relation_trace(
        [2], ["first"], [[3.0, 1.0, 3.0]], ["first", "second", "third"]
    )

    assert [candidate["index"] for candidate in rows[0]["candidates"]] == [0, 2, 1]
    assert [candidate["label"] for candidate in rows[0]["candidates"]] == [
        "first",
        "third",
        "second",
    ]
    assert rows[0]["verified"] is True
    assert rows[0]["score_type"] == "biaffine relation logit, not probability"


def test_relation_trace_rejects_relation_label_that_disagrees_with_argmax():
    with pytest.raises(ValueError, match="argmax"):
        relation_trace([0], ["second"], [[2.0, 1.0]], ["first", "second"])


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_relation_trace_rejects_nonfinite_scores(bad_score):
    with pytest.raises(ValueError, match="数值无效"):
        relation_trace([0], ["root"], [[bad_score, 0.0]], ["root", "dep"])


@pytest.mark.parametrize(
    ("heads", "relations", "vectors", "labels"),
    [
        ([0], ["root", "dep"], [[1.0]], ["root"]),
        ([0], ["root"], [], ["root"]),
        ([0], ["root"], [[1.0, 0.0]], ["root"]),
        ([0], ["root"], [[1.0]], []),
    ],
)
def test_relation_trace_rejects_misaligned_or_wrong_shape_inputs(
    heads, relations, vectors, labels
):
    with pytest.raises(ValueError):
        relation_trace(heads, relations, vectors, labels)


def test_tree_record_preserves_relation_scores():
    relation_scores = relation_trace([0], ["root"], [[2.0, 1.0]], ["root", "dep"])
    record = tree_record(
        "甲",
        ["甲"],
        {"heads": [0], "relations": ["root"], "relation_scores": relation_scores},
    )

    assert record["relation_scores"] == relation_scores
