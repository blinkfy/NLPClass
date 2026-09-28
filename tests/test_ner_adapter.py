import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.ner.hanlp_adapter import (
    HanLPRecognizer,
    model_entity_tagset,
    prepare_ner,
)


def test_missing_ner_never_fakes_entities() -> None:
    with pytest.raises(ResourceUnavailableError, match="setup-ner"):
        HanLPRecognizer().recognize("北京")


def test_adapter_maps_types_and_character_offsets() -> None:
    adapter = HanLPRecognizer(label_map={"NR": "PER", "NS": "LOC", "NT": "ORG"})
    # 固定 mock 仅验证 adapter 结构，不作为真实模型或网页预测。
    adapter._component = lambda chars: [("北京", "NS", len(chars) - 2, len(chars))]
    result = adapter.recognize("🙂北京")
    assert (result[0].text, result[0].type, result[0].start, result[0].end) == ("北京", "LOC", 1, 3)


def test_adapter_preserves_sentence_offsets() -> None:
    adapter = HanLPRecognizer(label_map={"NR": "PER", "NS": "LOC", "NT": "ORG"})
    adapter._component = lambda chars: [("张三", "NR", 0, 2)] if chars[0] == "张" else []
    result = adapter.recognize("你好。张三在此。")
    assert result[0].start == 3 and result[0].end == 5
    adapter._component = lambda chars: [("错误", "NS", 0, 2)]
    with pytest.raises(DataValidationError):
        adapter.recognize("北京")


@pytest.mark.parametrize("label", ["NS", "LOC", "LOCATION"])
def test_adapter_uses_only_explicit_model_mapping(label):
    adapter = HanLPRecognizer(label_map={label: "LOC"})
    adapter._component = lambda chars: [("北京", label, 1, 3)]
    assert adapter.recognize("🙂北京")[0].type == "LOC"
    adapter._component = lambda chars: [("北京", "UNCONFIGURED", 1, 3)]
    with pytest.raises(DataValidationError, match="显式映射"):
        adapter.recognize("🙂北京")


def test_missing_mapping_does_not_assume_msra_labels():
    adapter = HanLPRecognizer()
    adapter._component = lambda chars: []
    with pytest.raises(ResourceUnavailableError, match="标签映射"):
        adapter.recognize("北京")


def fake_component(tags, backend="tf"):
    vocab = SimpleNamespace(idx_to_token=tags)
    if backend == "tf":
        return SimpleNamespace(transform=SimpleNamespace(tag_vocab=vocab))
    return SimpleNamespace(vocabs={"tag": vocab})


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_model_tagset_is_read_from_loaded_vocabulary(backend):
    component = fake_component(["<pad>", "O", "B-PER", "I-PER", "E-LOC", "S-ORG"], backend)
    assert model_entity_tagset(component) == ["LOC", "ORG", "PER"]


@pytest.mark.parametrize("tags", [None, ["O"], ["O", "X-NS"]])
def test_unreadable_or_unsupported_model_tagset_fails(tags):
    with pytest.raises(DataValidationError):
        model_entity_tagset(fake_component(tags))


def mock_hanlp(monkeypatch, component, resource):
    hanlp = ModuleType("hanlp")
    hanlp.pretrained = SimpleNamespace(ner=SimpleNamespace(MSRA_NER_BERT_BASE_ZH="fixture://msra"))
    hanlp.load = lambda path, devices: component
    io_util = ModuleType("hanlp.utils.io_util")
    io_util.get_resource = lambda value: str(resource)
    monkeypatch.setitem(sys.modules, "hanlp", hanlp)
    monkeypatch.setitem(sys.modules, "hanlp.utils.io_util", io_util)


def test_msra_configuration_matches_audited_official_vocabulary():
    config = yaml.safe_load(Path("configs/hanlp_msra_bert.yaml").read_text(encoding="utf-8"))
    # 官方归档 vocabs.json 的标签；不加载权重，不声称真实 NER 性能。
    official_tags = [
        "<pad>",
        "O",
        "S-NS",
        "B-NS",
        "E-NS",
        "B-NT",
        "M-NT",
        "E-NT",
        "M-NS",
        "B-NR",
        "M-NR",
        "E-NR",
        "S-NR",
        "S-NT",
    ]
    assert config["model"] == "MSRA_NER_BERT_BASE_ZH"
    assert config["expected_entity_tags"] == model_entity_tagset(fake_component(official_tags))
    assert config["label_map"] == {"NR": "PER", "NS": "LOC", "NT": "ORG"}


def test_prepare_records_verified_mapping_and_runtime_rechecks_it(tmp_path, monkeypatch):
    resource = tmp_path / "local-model"
    resource.mkdir()
    component = fake_component(["O", "B-NR", "E-NS", "S-NT"])
    mock_hanlp(monkeypatch, component, resource)
    value = prepare_ner(Settings(tmp_path), Path("configs/hanlp_msra_bert.yaml"))
    setup_path = tmp_path / "artifacts/cache/ner_setup.json"
    assert json.loads(setup_path.read_text(encoding="utf-8")) == value
    assert value["entity_tagset"] == ["NR", "NS", "NT"]
    adapter = HanLPRecognizer.from_settings(Settings(tmp_path))
    assert adapter.label_map == value["label_map"]
    assert adapter.name == "hanlp-msra-bert"
    adapter._load()
    assert adapter._component is component
    # 替换本地模型后，哪怕其预测没有实体也必须先检查全部标签词表。
    incompatible = fake_component(["O", "B-PER", "E-LOC", "S-ORG"], "torch")
    mock_hanlp(monkeypatch, incompatible, resource)
    adapter = HanLPRecognizer.from_settings(Settings(tmp_path))
    with pytest.raises(DataValidationError, match="tagset"):
        adapter._load()
    assert adapter._component is None
    before = setup_path.read_bytes()
    with pytest.raises(DataValidationError, match="tagset"):
        prepare_ner(Settings(tmp_path), Path("configs/hanlp_msra_bert.yaml"))
    assert setup_path.read_bytes() == before


def test_mismatched_setup_never_writes_success(tmp_path, monkeypatch):
    mock_hanlp(monkeypatch, fake_component(["O", "B-PER", "E-LOC", "S-ORG"]), tmp_path)
    with pytest.raises(DataValidationError, match="tagset"):
        prepare_ner(Settings(tmp_path), Path("configs/hanlp_msra_bert.yaml"))
    assert not (tmp_path / "artifacts/cache/ner_setup.json").exists()


def test_legacy_setup_is_explicitly_unavailable(tmp_path):
    setup_path = tmp_path / "artifacts/cache/ner_setup.json"
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text(json.dumps({"model_path": str(tmp_path)}), encoding="utf-8")
    with pytest.raises(ResourceUnavailableError, match="setup-ner"):
        HanLPRecognizer.from_settings(Settings(tmp_path)).recognize("北京")


def test_invalid_target_mapping_is_rejected():
    with pytest.raises(DataValidationError, match="PER/LOC/ORG"):
        HanLPRecognizer(label_map={"NR": "PERSON"})
