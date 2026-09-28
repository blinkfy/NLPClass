from nlplab.core.io import write_json
from nlplab.extensions.comparison import collect, save


def test_course_reference_units_and_cws_only_comparison(tmp_path):
    config = tmp_path / "configs/extension_course_references.yaml"
    config.parent.mkdir()
    config.write_text("longest: [89.41, 94.64, 91.95, 2.58, 97.14]\n", encoding="utf-8")
    run = tmp_path / "outputs/exp1/extensions/toy"
    run.mkdir(parents=True)
    write_json(run / "run.json", {"model": "longest", "phase": "final", "run_id": "toy"})
    write_json(
        run / "metrics.json",
        {
            "official_scorer": {
                "precision": 0.92,
                "recall": 0.96,
                "f1": 0.9365,
                "oov_recall": 0.0247,
                "iv_recall": 0.9821,
            },
            "project_evaluator": {"tag_accuracy": 0.9},
            "coverage": 1,
            "inference_seconds": 2,
        },
    )
    row = collect(tmp_path)["rows"][0]
    assert abs(row["course_reference_delta"]["f1"] - 1.7) < 1e-8
    assert row["tag_accuracy"] is None
    assert row["runtime_metadata"]["inference_seconds"] == 2
    assert save(tmp_path).is_file()
