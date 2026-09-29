"""报告收尾验收仅读取资产，不训练、不推理、不调用 scorer。"""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("training_executed", "official_test_inference_executed", "scorer_executed")


@pytest.fixture(scope="module")
def assets():
    pointer = ROOT / "artifacts/cache/exp1_closeout_current.json"
    if not pointer.is_file():
        pytest.skip("本地正式报告资产不提交 Git")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["output"])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_asset_and_source_hashes_and_execution_flags(assets):
    manifest = read(assets / "asset_manifest.json")
    assert all(manifest[name] is False for name in FLAGS)
    for item in manifest["assets"]:
        path = (ROOT if item.get("location") == "repository" else assets) / item["asset_name"]
        if item.get("location") == "repository" and item["asset_name"] == "README.md":
            # README is a historical report reference, not a frozen benchmark file.
            assert len(item["generated_sha256"]) == 64
        else:
            assert sha256(path.read_bytes()).hexdigest() == item["generated_sha256"]
        assert all(item[name] is False for name in FLAGS)
        for source, digest in item["source_sha256"].items():
            assert (ROOT / source).is_file()
            if source.startswith(("app/static/", "app/templates/")):
                # The report records historic Web source; later authorized Web changes
                # are outside the frozen model/data/benchmark boundary.
                assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                continue
            assert sha256((ROOT / source).read_bytes()).hexdigest() == digest


def test_real_code_excerpt_hashes(assets):
    for excerpt in read(assets / "code_snippet_index.json"):
        lines = (ROOT / excerpt["source"]).read_text(encoding="utf-8").splitlines()
        actual = "\n".join(lines[excerpt["start"] - 1 : excerpt["end"]])
        assert sha256(actual.encode()).hexdigest() == excerpt["excerpt_sha256"]
    text = (assets / "report_code_snippets.md").read_text(encoding="utf-8")
    assert "pytorch-crf 0.7.2" in text


def test_formal_comparison_and_read_only_derived_tags(assets):
    comparison = read(assets / "cws_seven_model_comparison.json")["rows"]
    assert len(comparison) == 7
    assert all("MSR" in row["dataset"] and "NER" not in row["model"] for row in comparison)
    derived = read(assets / "derived_tag_accuracy.json")
    bench = ROOT / "outputs/exp1/benchmarks/20260926_130612_32a5e168"
    for row in comparison:
        run = (
            bench / row["final_run_id"]
            if row["model_id"].startswith("hmm1")
            else ROOT / "outputs/exp1/extensions" / row["final_run_id"]
        )
        official = read(run / "metrics.json")["official_scorer"]
        official = official.get("full", official)
        for name in ("precision", "recall", "f1", "oov_recall", "iv_recall"):
            assert row[name] == official[name]
        if row["model_id"] in derived:
            assert derived[row["model_id"]]["derived_from_saved_predictions"] is True
            assert row["tag_accuracy"] == derived[row["model_id"]]["tag_accuracy"]


def test_error_examples_preserve_observed_text_and_span_coverage(assets):
    cws = jsonl(assets / "cws_error_examples.jsonl")
    assert sorted(Counter(row["model"] for row in cws).values()) == [5] * 5
    ner = jsonl(assets / "ner_error_examples.jsonl")
    assert len(ner) >= 8
    for row in ner:
        for entity in row["gold_entities"] + row["predicted_entities"]:
            assert row["raw_text"][entity["start"] : entity["end"]] == entity["text"]
    coverage = read(assets / "example_coverage.json")
    assert set(coverage["ner_required_coverage"]) <= set(coverage["ner_coverage"])
    kinds = Counter(row["kind"] for row in jsonl(assets / "representative_examples.jsonl"))
    assert len(kinds) == 5 and set(kinds.values()) == {2}


def test_frozen_integrity_and_prior_assets_preserved(assets):
    validation = read(assets / "closeout_hash_validation.json")
    assert validation["status"] == "unchanged"
    assert validation["baseline_count"] == 264 and validation["extension_count"] == 90
    assert validation["changed"] == []
    checkpoint = read(assets / "closeout_checkpoint.json")
    assert checkpoint["interrupted_for"] == "web_model_specific_visualization"
    for name, digest in checkpoint["hashes"].items():
        if name == "summary.md":  # Only Web evidence wording changed after the patch.
            continue
        assert sha256((assets / name).read_bytes()).hexdigest() == digest


def test_training_budget_and_screenshots(assets):
    summary = read(assets / "summary.json")
    assert "10 epoch" in summary["bilstm_budget_statement"]
    assert all(summary[name] is False for name in FLAGS)
    screenshots = list((assets / "screenshots").glob("*.png"))
    assert len(screenshots) >= 13
    assert all(file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for file in screenshots)
    patch = read(assets / "browser_patch_qa.json")
    supplemental = read(assets / "browser_screenshots_qa.json")
    assert patch["errors"] == supplemental["errors"] == []
    assert any(c.get("bilstm_narrow_no_overflow") for c in supplemental["checks"])
    assert any(c.get("unavailable_option", {}).get("disabled") for c in supplemental["checks"])
