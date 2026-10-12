from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "build/github-pages"
PYTHON = ROOT / "artifacts/runtime/exp2-venv/Scripts/python.exe"
STAGE1_RUN = "20261001_123337_9c1e7341"
STAGE2_RUN = "20261001_133829_68ee50d7"

SAMPLES = [
    "央行宣布下调贷款利率，市场预计经济增长。",
    "中国队在足球比赛中攻入两球，球员庆祝胜利。",
    "大学发布新课程，学生报名参加考试。",
    "教育🙂 运动！财经\n科技",
    "ABC １２３ ＡＢＣ xyz",
    "\ue000\U000f0000\ue001",
    "𠀀中文扩展字\U0010ffff未分配码点",
]

PYTHON_BATCH = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from nlplab.classification.__main__ import predict_saved as predict_stage1
from nlplab.classification.stage2 import predict_saved as predict_stage2
requests = json.loads(sys.stdin.read())
results = []
for item in requests:
    if item["model_id"] == "C2":
        results.append(predict_stage2(Path.cwd(), item["run_id"], item["text"]))
    else:
        results.append(predict_stage1(Path.cwd(), item["run_id"], item["model_id"], item["text"]))
print(json.dumps(results, ensure_ascii=False, allow_nan=False))
"""

NODE_BATCH = r"""
import {readFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
const site = path.resolve(process.argv.at(-1));
globalThis.document = {baseURI:pathToFileURL(site + path.sep).href};
globalThis.fetch = async input => {
  const filename = fileURLToPath(input instanceof URL ? input : new URL(input));
  return {ok:true, json:async()=>JSON.parse(await readFile(filename,'utf8'))};
};
const manifest = JSON.parse(await readFile(path.join(site,'static-manifest.json'),'utf8'));
const moduleUrl = pathToFileURL(path.join(site,'static/js/static-classification.js')).href;
const {classify} = await import(moduleUrl);
const requests = JSON.parse(await new Promise(resolve => {
  let data=''; process.stdin.setEncoding('utf8'); process.stdin.on('data',chunk=>data+=chunk);
  process.stdin.on('end',()=>resolve(data));
}));
const results=[];
for (const request of requests) results.push(await classify(request.path,request.body,manifest));
process.stdout.write(JSON.stringify(results));
"""


def _close(actual, expected, where):
    assert abs(actual - expected) <= 1e-10, f"{where}: {actual!r} != {expected!r}"


def _compare_feature_rows(actual, expected, keys, where):
    assert len(actual) == len(expected), f"{where}: feature count differs"
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert left["term"] == right["term"], f"{where}[{index}].term"
        for key in keys:
            _close(left[key], right[key], f"{where}[{index}].{key}")


def test_browser_classifiers_match_frozen_python_predictions():
    assert PYTHON.is_file(), f"missing isolated runtime: {PYTHON}"
    assert (SITE / "static-manifest.json").is_file(), f"missing exported site: {SITE}"
    sample_requests = []
    for text in SAMPLES:
        for model_id in ("C0", "C1"):
            sample_requests.append(
                {
                    "path": "/api/v1/experiments/3/predict",
                    "body": {"run_id": STAGE1_RUN, "model_id": model_id, "text": text},
                    "run_id": STAGE1_RUN,
                    "model_id": model_id,
                    "text": text,
                }
            )
        sample_requests.append(
            {
                "path": "/api/v1/experiments/3/stage2/predict",
                "body": {"run_id": STAGE2_RUN, "text": text},
                "run_id": STAGE2_RUN,
                "model_id": "C2",
                "text": text,
            }
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONIOENCODING"] = "utf-8"
    python = subprocess.run(
        [str(PYTHON), "-c", PYTHON_BATCH],
        input=json.dumps(sample_requests, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=environment,
        timeout=180,
    )
    node = subprocess.run(
        ["node", "--input-type=module", "-e", NODE_BATCH, str(SITE)],
        input=json.dumps(sample_requests, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        cwd=ROOT,
        timeout=180,
    )
    expected_results = json.loads(python.stdout)
    actual_results = json.loads(node.stdout)
    assert len(actual_results) == len(expected_results) == len(sample_requests)

    for request, actual, expected in zip(sample_requests, actual_results, expected_results):
        label = f"{request['model_id']} {request['text']!r}"
        for key in (
            "prediction",
            "runner_up",
            "score_type",
            "zero_vector",
            "token_count",
            "known_feature_count",
        ):
            assert actual[key] == expected[key], f"{label}: {key} differs"
        if request["text"] == SAMPLES[5]:
            assert actual["zero_vector"] is True, f"{label}: no-hit sample was not empty"
        for key in (
            "model_score",
            "score_gap",
            "intercept",
            "feature_sum",
            "reconstructed_score",
            "contrast_intercept",
            "contrast_feature_sum",
        ):
            _close(actual[key], expected[key], f"{label}.{key}")
        assert len(actual["scores"]) == len(expected["scores"])
        for index, (left, right) in enumerate(zip(actual["scores"], expected["scores"])):
            assert left["label"] == right["label"], f"{label}.scores[{index}].label"
            _close(left["value"], right["value"], f"{label}.scores[{index}].value")
            if right["joint_log_score"] is None:
                assert left["joint_log_score"] is None
            else:
                _close(
                    left["joint_log_score"],
                    right["joint_log_score"],
                    f"{label}.scores[{index}].joint_log_score",
                )
        _compare_feature_rows(
            actual["top_contributing_features"],
            expected["top_contributing_features"],
            (
                "tfidf",
                "weight",
                "contribution",
                "direct_contribution",
                "runner_up_contrast",
            ),
            f"{label}.top_contributing_features",
        )
        if request["model_id"] == "C2":
            assert actual["coverage"] == expected["coverage"], f"{label}.coverage"
            if request["text"] == SAMPLES[5]:
                assert actual["coverage"]["candidate_occurrences"] == 0
            _compare_feature_rows(
                actual["top_relative_features"],
                expected["top_relative_features"],
                ("tfidf", "contribution", "coefficient_difference"),
                f"{label}.top_relative_features",
            )
