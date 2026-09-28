"""汇总七种 MSR final 方法的结果和课件 sanity comparison。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nlplab.core.io import write_json

METRICS = ("precision", "recall", "f1", "oov_recall", "iv_recall")
DISPLAY = {
    "longest": "Longest Matching",
    "bigram": "Bigram",
    "hmm1-mle": "HMM-1 MLE",
    "hmm1-smooth": "HMM-1 Smooth",
    "hmm2-mle": "HMM-2 MLE",
    "hmm2-smooth": "HMM-2 Smooth",
    "bilstm-crf": "BiLSTM-CRF",
}


def _score(value: dict) -> dict:
    official = value.get("official_scorer")
    if official:
        return official.get("full", official)
    project = value["project_evaluator"]
    return project.get("full", project)


def collect(root: Path) -> dict:
    rows = []
    frozen = root / "outputs/exp1/benchmarks/20260926_130612_32a5e168"
    for model_id, filename in [
        ("hmm1-mle", "20260926_130612_32a5e168_mle"),
        ("hmm1-smooth", "20260926_130612_32a5e168_smooth"),
    ]:
        path = frozen / filename / "metrics.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        score = value["official_scorer"]["full"]
        project = value["project_evaluator"]["full"]
        rows.append(
            {
                "model_id": model_id,
                "model": DISPLAY[model_id],
                "run_id": path.parent.name,
                "source": "frozen MSR benchmark",
                **{key: score.get(key) for key in METRICS},
                "tag_accuracy": project.get("tag_accuracy"),
                "coverage": value.get("inference", {}).get("coverage"),
                "runtime_metadata": value.get("inference", {}),
            }
        )
    for path in sorted((root / "outputs/exp1/extensions").glob("*/run.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("phase") != "final" or info.get("model") not in DISPLAY:
            continue
        metrics_path = path.parent / "metrics.json"
        if not metrics_path.is_file():
            continue
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = _score(value)
        project = value["project_evaluator"]
        model_id = info["model"]
        rows.append(
            {
                "model_id": model_id,
                "model": DISPLAY[model_id],
                "run_id": info["run_id"],
                "source": "MSR official raw test",
                **{key: score.get(key) for key in METRICS},
                "tag_accuracy": None
                if model_id in {"longest", "bigram"}
                else project.get("tag_accuracy"),
                "coverage": value.get("coverage"),
                "runtime_metadata": {
                    "inference_seconds": value.get("inference_seconds"),
                    "training": info.get("config", {}).get("training"),
                    "model_config": info.get("config", {}),
                },
            }
        )
    rows.sort(key=lambda row: list(DISPLAY).index(row["model_id"]))
    reference_path = root / "configs/extension_course_references.yaml"
    reference = (
        yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        if reference_path.is_file()
        else {}
    )
    ref_rows = {
        "longest": reference.get("longest"),
        "bigram": reference.get("bigram"),
        "hmm1-mle": reference.get("hmm1"),
        "hmm1-smooth": reference.get("hmm1"),
        "hmm2-mle": reference.get("hmm2"),
        "hmm2-smooth": reference.get("hmm2"),
    }
    for row in rows:
        expected = ref_rows.get(row["model_id"])
        row["course_reference_delta"] = (
            {key: row[key] * 100 - expected[i] for i, key in enumerate(METRICS)}
            if expected is not None and all(row[key] is not None for key in METRICS)
            else None
        )
    return {
        "dataset": "SIGHAN Bakeoff 2005 MSR official raw test",
        "metric_order": list(METRICS),
        "rows": rows,
        "course_reference": ref_rows,
        "reference_config": str(reference_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save(root: Path) -> Path:
    result = collect(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = root / "outputs/exp1/extensions"
    json_path = directory / f"comparison_{stamp}.json"
    md_path = directory / f"comparison_{stamp}.md"
    write_json(json_path, result)
    lines = [
        "# MSR final 课程方法比较",
        "",
        "| 方法 | P | R | F1 | OOV-R | IV-R | Tag Accuracy | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        cells = [row["model"]] + [
            "—" if row[key] is None else f"{row[key] * 100:.2f}%"
            for key in (*METRICS, "tag_accuracy", "coverage")
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "课件参考值来自 `configs/extension_course_references.yaml`，只作 sanity "
            "comparison，不用于 test 调参。",
            "",
            "| 方法 | 课件 P/R/F1 | 课件 OOV-R/IV-R | 本项目 F1 差（百分点） |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in result["rows"]:
        expected = result["course_reference"].get(row["model_id"])
        if expected is not None:
            lines.append(
                f"| {row['model']} | {' / '.join(f'{v:.2f}' for v in expected[:3])} | "
                f"{' / '.join(f'{v:.2f}' for v in expected[3:])} | "
                f"{row['course_reference_delta']['f1']:+.2f} |"
            )
    lines.extend(
        [
            "",
            "最长匹配与 Bigram 使用训练词典、单字回退，OOV 召回有限；"
            "Bigram 的统计消歧改善 IV 召回。",
            "二阶 HMM 的状态组合更稀疏；阶数增加不保证 F1 提升，平滑改善 coverage 与 OOV 召回。",
            "词典构造、平滑/UNK、初始化、BMES 约束与课件 HanLP 版本可能解释合理差异。",
            "所有 final 模型的配置在 official raw test 推理之前确定，未按 test 分数调整。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path
