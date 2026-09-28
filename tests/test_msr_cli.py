import json

import pytest

from nlplab.cli import main, read_config, train_model
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.dataset import load_samples
from nlplab.data.msr import import_msr


def test_msr_import_keeps_test_and_separates_splits(tmp_path) -> None:
    source = tmp_path / "external"
    (source / "training").mkdir(parents=True)
    (source / "gold").mkdir()
    (source / "training/msr_training.utf8").write_text(
        "商品  和  服务\n我 爱 北京\n自然语言 处理\n我 爱 北京\n测试 句\n", encoding="utf-8"
    )
    (source / "gold/msr_test_gold.utf8").write_text("测试 句\n留出 样本\n", encoding="utf-8")
    destination = tmp_path / "splits"
    manifest = import_msr(source, destination, protocol="clean")
    train, dev, test = [
        load_samples(destination / f"{split}.jsonl") for split in ("train", "dev", "test")
    ]
    assert manifest["protocol"] == "clean"
    assert len(test) == 2
    assert {item.text for item in train}.isdisjoint({item.text for item in dev})
    assert {item.text for item in train + dev}.isdisjoint({item.text for item in test})
    assert manifest["removed_training_duplicates"] == 1
    assert manifest["removed_training_test_overlap"] == 1
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["seed"] == 42


def test_missing_resource_and_cli_error(tmp_path, capsys) -> None:
    with pytest.raises(ResourceUnavailableError):
        import_msr(tmp_path / "missing", tmp_path / "splits")
    assert main(["import-msr", "--source", str(tmp_path / "missing")]) == 2
    assert "MSR" in capsys.readouterr().err


def test_demo_training_retains_provenance() -> None:
    from pathlib import Path

    model = train_model(Path("data/demo/toy_train.jsonl"), {})
    assert model.training_metadata["is_demo"] is True


def test_config_typos_are_not_silently_ignored(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("smoothng: add_alpha\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="未知配置"):
        read_config(config)
