import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from nlplab.cli import main, read_config, train_model
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.data.dataset import load_samples, read_msr
from nlplab.data.msr import import_msr, load_msr_provenance
from nlplab.sequence_labeling.hmm import HMMTagger
from nlplab.services.experiment_service import ExperimentService


@pytest.fixture
def msr_source(tmp_path):
    # 临时小型夹具只验证 protocol，不能作为正式 MSR 数据。
    source = tmp_path / "source"
    (source / "training").mkdir(parents=True)
    (source / "gold").mkdir()
    (source / "training/msr_training.utf8").write_text(
        "商品 和 服务\n我 爱 北京\n自然语言 处理\n我 爱 北京\n测试 句\n",
        encoding="utf-8",
    )
    (source / "gold/msr_test_gold.utf8").write_text(
        "测试 句\n留出 样本\n测试 句\n", encoding="utf-8"
    )
    return source


def test_standard_preserves_every_official_record(msr_source, tmp_path):
    train_path = msr_source / "training/msr_training.utf8"
    test_path = msr_source / "gold/msr_test_gold.utf8"
    original = [path.read_bytes() for path in (train_path, test_path)]
    destination = tmp_path / "standard"
    manifest = import_msr(msr_source, destination)
    assert manifest["protocol"] == "standard"
    assert load_samples(destination / "train.jsonl") == read_msr(train_path, "MSR_TRAIN")
    assert load_samples(destination / "test.jsonl") == read_msr(test_path, "MSR_TEST")
    assert (destination / "dev.jsonl").read_bytes() == b""
    assert manifest["original_train_count"] == manifest["train_count"] == 5
    assert manifest["test_count"] == 3
    assert manifest["dev_ratio"] == manifest["dev_count"] == 0
    assert manifest["potential_training_duplicates"] == 1
    assert manifest["potential_test_duplicates"] == 1
    assert manifest["potential_training_test_overlap"] == 1
    assert manifest["removed_training_duplicates"] == 0
    assert manifest["removed_training_test_overlap"] == 0
    assert original == [path.read_bytes() for path in (train_path, test_path)]
    assert [value["sha256"] for value in manifest["sources"]] == [
        sha256(data).hexdigest() for data in original
    ]


def test_clean_is_reproducible_and_retains_full_test(msr_source, tmp_path):
    destinations = [tmp_path / "clean1", tmp_path / "clean2"]
    manifests = [import_msr(msr_source, path, protocol="clean") for path in destinations]
    assert manifests[0] == manifests[1]
    train, dev, test = [
        load_samples(destinations[0] / f"{name}.jsonl") for name in ("train", "dev", "test")
    ]
    assert len(train) + len(dev) == 3
    assert {sample.text for sample in train}.isdisjoint(sample.text for sample in dev)
    assert {sample.text for sample in train + dev}.isdisjoint(sample.text for sample in test)
    assert len(test) == 3
    assert manifests[0]["removed_training_duplicates"] == 1
    assert manifests[0]["removed_training_test_overlap"] == 1


def test_conflicting_gold_is_preserved_only_in_standard(msr_source, tmp_path):
    path = msr_source / "training/msr_training.utf8"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("我爱 北京\n")
    manifest = import_msr(msr_source, tmp_path / "standard")
    assert manifest["train_count"] == 6
    assert manifest["potential_training_duplicates"] == 2
    with pytest.raises(DataValidationError, match="同文不同 Gold"):
        import_msr(msr_source, tmp_path / "clean", protocol="clean")
    assert not (tmp_path / "clean").exists()


@pytest.mark.parametrize("protocol", [None, "standard", "clean"])
def test_cli_protocol_default_and_selection(msr_source, tmp_path, capsys, protocol):
    destination = tmp_path / "cli"
    argv = ["import-msr", "--source", str(msr_source), "--output", str(destination)]
    if protocol is not None:
        argv += ["--protocol", protocol]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == (protocol or "standard")
    with pytest.raises(DataValidationError, match="拒绝覆盖"):
        import_msr(msr_source, destination)
    with pytest.raises(DataValidationError, match="protocol"):
        import_msr(msr_source, tmp_path / "bad", protocol="invalid")


def test_standard_overlap_is_allowed_only_with_matching_provenance(msr_source, tmp_path):
    destination = tmp_path / "standard"
    import_msr(msr_source, destination)
    model = train_model(destination / "train.jsonl", read_config(Path("configs/hmm_smooth.yaml")))
    model.save(tmp_path / "trained")
    model = HMMTagger.load(tmp_path / "trained")
    samples = load_samples(destination / "test.jsonl")
    service = ExperimentService(Settings(tmp_path))
    # 改 dataset 名称或只有模型 protocol 不能绕过普通重叠拦截。
    with pytest.raises(DataValidationError, match="重叠"):
        service.run(model, samples, dataset="msr-standard")
    run = service.run(
        model, samples, dataset="msr-standard", evaluation_path=destination / "test.jsonl"
    )
    assert run["msr"]["protocol"] == "standard"
    assert run["evaluation_count"] == 3
    assert run["training_evaluation_overlap_count"] == 2
    config = yaml.safe_load((service.store.resolve() / "config.yaml").read_text(encoding="utf-8"))
    assert config["msr"] == run["msr"]
    with pytest.raises(DataValidationError, match="指纹"):
        service.run(
            model, samples[:1], dataset="invalid", evaluation_path=destination / "test.jsonl"
        )
    with pytest.raises(DataValidationError, match="正确划分"):
        service.run(
            model,
            load_samples(destination / "train.jsonl"),
            dataset="invalid",
            evaluation_path=destination / "train.jsonl",
        )
    assert service.store.resolve().name == run["run_id"]


def test_protocol_mixing_and_tampered_splits_are_rejected(msr_source, tmp_path):
    standard, clean = tmp_path / "standard", tmp_path / "clean"
    import_msr(msr_source, standard)
    import_msr(msr_source, clean, protocol="clean")
    model = train_model(clean / "train.jsonl", read_config(Path("configs/hmm_smooth.yaml")))
    with pytest.raises(DataValidationError, match="同一 protocol"):
        ExperimentService(Settings(tmp_path)).run(
            model,
            load_samples(standard / "test.jsonl"),
            dataset="invalid",
            evaluation_path=standard / "test.jsonl",
        )
    with pytest.raises(DataValidationError, match="只能使用 train"):
        train_model(standard / "test.jsonl", {})
    path = standard / "train.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(path.read_text(encoding="utf-8").splitlines()[0] + "\n")
    with pytest.raises(DataValidationError, match="指纹"):
        train_model(path, {})


def test_legacy_msr_manifest_cannot_claim_standard(msr_source, tmp_path):
    destination = tmp_path / "legacy"
    import_msr(msr_source, destination)
    path = destination / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    del value["protocol"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError, match="重新导入"):
        load_msr_provenance(destination / "train.jsonl", load_samples(destination / "train.jsonl"))


def test_cli_training_and_evaluation_keep_standard_protocol(msr_source, tmp_path, capsys):
    destination = tmp_path / "standard"
    import_msr(msr_source, destination)
    model_path = tmp_path / "model"
    assert (
        main(["train", "--train", str(destination / "train.jsonl"), "--model", str(model_path)])
        == 0
    )
    assert HMMTagger.load(model_path).config.smoothing == "none"
    # 临时夹具测试含未见字，使用明确的增强模型验证端到端 benchmark 流程。
    smooth_path = tmp_path / "smooth-model"
    assert (
        main(
            [
                "train",
                "--train",
                str(destination / "train.jsonl"),
                "--model",
                str(smooth_path),
                "--config",
                "configs/hmm_smooth.yaml",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "evaluate",
                "--model",
                str(smooth_path),
                "--test",
                str(destination / "test.jsonl"),
                "--dataset",
                "fixture-standard",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["msr"]["protocol"] == "standard"
    assert result["evaluation_count"] == 3
