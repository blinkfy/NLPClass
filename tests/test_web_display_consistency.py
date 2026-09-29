"""展示修订只读取正式 Run 与已派生报告指标。"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.comparison_display import with_derived_tags
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_homepage_and_benchmark_labels():
    with TestClient(create_app()) as client:
        home = client.get("/").text
        assert "实验一已完成并冻结" not in home
        assert "Completed / Frozen" not in home
        assert "标准评测已接入" in home
        assert home.count("数据已准备 · 模型待开发") == 2
        assert "NER 通过可选 HanLP" not in home
        page = client.get("/tagging").text
        assert "NER / 课程样本 Run" in page
        assert "NER benchmark: MSRA-200" in page
        assert "CWS benchmark：SIGHAN MSR" in client.get("/segmentation").text
        sequence = client.get("/sequence-labeling").text
        assert "method-overview" in sequence
        assert "02 · 已接入" in sequence
        assert "线性链 CRF" in sequence


def test_saved_derived_tags_match_run_and_do_not_mutate_input(tmp_path):
    directory = tmp_path / "outputs/exp1/final_report_assets/report"
    directory.mkdir(parents=True)
    pointer = tmp_path / "artifacts/cache/exp1_closeout_current.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"output": str(directory)}), encoding="utf-8")
    derived = {
        "longest": {
            "tag_accuracy": 0.94132,
            "derived_from_saved_predictions": True,
            "training_executed": False,
            "official_test_inference_executed": False,
            "scorer_executed": False,
        }
    }
    (directory / "derived_tag_accuracy.json").write_text(json.dumps(derived), encoding="utf-8")
    (directory / "cws_seven_model_comparison.json").write_text(
        json.dumps(
            {"rows": [{"model_id": "longest", "final_run_id": "final", "tag_accuracy": 0.94132}]}
        ),
        encoding="utf-8",
    )
    original = {"rows": [{"model_id": "longest", "run_id": "final", "tag_accuracy": None}]}
    result = with_derived_tags(original, tmp_path)
    assert result["rows"][0]["tag_accuracy"] == 0.94132
    assert result["rows"][0]["tag_accuracy_derived_from_saved_predictions"] is True
    assert original["rows"][0]["tag_accuracy"] is None
    original["rows"][0]["run_id"] = "different"
    assert with_derived_tags(original, tmp_path)["rows"][0]["tag_accuracy"] is None


def test_formal_web_comparison_uses_saved_derived_tags():
    pointer = ROOT / "artifacts/cache/exp1_closeout_current.json"
    if not pointer.is_file():
        import pytest

        pytest.skip("无本地正式报告资产")
    directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["output"])
    expected = json.loads((directory / "derived_tag_accuracy.json").read_text(encoding="utf-8"))
    with TestClient(create_app()) as client:
        result = client.get("/api/v1/experiments/1/comparison").json()
    for row in result["rows"]:
        if row["model_id"] in expected:
            assert row["tag_accuracy"] == expected[row["model_id"]]["tag_accuracy"]
            assert row["tag_accuracy_derived_from_saved_predictions"] is True
