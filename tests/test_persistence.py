import json

import pytest

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import SequenceSample
from nlplab.sequence_labeling.hmm import HMMTagger
from nlplab.services.experiment_service import ExperimentService


def test_model_save_load_and_tampering(toy_model, tmp_path) -> None:
    path = tmp_path / "model"
    toy_model.save(path)
    loaded = HMMTagger.load(path)
    assert loaded.predict(list("北京🙂大学")) == toy_model.predict(list("北京🙂大学"))
    assert loaded.training_words == toy_model.training_words
    metadata = path / "metadata.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["vocabulary"].append("甲")
    metadata.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError):
        HMMTagger.load(path)


def test_runs_are_independent_and_guard_leakage(toy_model, tmp_path) -> None:
    service = ExperimentService(Settings(tmp_path))
    sample = SequenceSample("S1", "D1", "商品和北京", ["商品", "和", "北京"])
    first = service.run(toy_model, [sample], dataset="toy", is_demo=True)
    second = service.run(toy_model, [sample], dataset="toy", is_demo=True)
    assert first["run_id"] != second["run_id"]
    assert service.store.resolve(first["run_id"]).exists()
    assert service.store.resolve().name == second["run_id"]
    leaked = SequenceSample("S2", "D2", "商品和服务", ["商品", "和", "服务"])
    with pytest.raises(DataValidationError, match="重叠"):
        service.run(toy_model, [leaked], dataset="invalid")
    assert service.store.resolve().name == second["run_id"]
