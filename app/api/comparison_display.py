"""读取已完成报告的派生指标，仅用于 Web 展示，不修改 frozen Run。"""

import json
from pathlib import Path


def with_derived_tags(value: dict, root: Path) -> dict:
    pointer = root / "artifacts/cache/exp1_closeout_current.json"
    if not pointer.is_file():
        return value
    directory = Path(json.loads(pointer.read_text(encoding="utf-8"))["output"]).resolve()
    if not directory.is_relative_to((root / "outputs/exp1/final_report_assets").resolve()):
        return value
    derived_path = directory / "derived_tag_accuracy.json"
    comparison_path = directory / "cws_seven_model_comparison.json"
    if not derived_path.is_file() or not comparison_path.is_file():
        return value
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    report = {
        row["model_id"]: row
        for row in json.loads(comparison_path.read_text(encoding="utf-8"))["rows"]
    }
    rows = []
    for row in value["rows"]:
        model = row["model_id"]
        item, source = derived.get(model, {}), report.get(model, {})
        if (
            model in {"longest", "bigram"}
            and item.get("derived_from_saved_predictions") is True
            and source.get("final_run_id") == row["run_id"]
            and source.get("tag_accuracy") == item.get("tag_accuracy")
            and all(
                item.get(flag) is False
                for flag in (
                    "training_executed",
                    "official_test_inference_executed",
                    "scorer_executed",
                )
            )
        ):
            row = {
                **row,
                "tag_accuracy": item["tag_accuracy"],
                "tag_accuracy_derived_from_saved_predictions": True,
                "tag_accuracy_source": str(derived_path.relative_to(root)),
            }
        rows.append(row)
    return {**value, "rows": rows}
