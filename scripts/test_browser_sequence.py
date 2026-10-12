"""Parity-check exported static sequence inference against frozen Python models."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.api.trace_display import display_trace  # noqa: E402
from nlplab.extensions.dictionary import BigramSegmenter  # noqa: E402
from nlplab.extensions.hmm2 import SecondOrderHMM  # noqa: E402
from nlplab.sequence_labeling.linear_crf import LinearChainCRF  # noqa: E402
from nlplab.sequence_labeling.perceptron import AveragedStructuredPerceptron  # noqa: E402

CASES = ["汉", "  ", "商品🙂和服务", "南京市长江大桥", "学校开设人工智能课程，学生参加篮球训练。"]
TOL = 1e-6
MAX_ABS_DIFF = 0.0

NODE_RUNNER = r"""
import fs from 'node:fs/promises';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
const [root, moduleFile, casesFile] = process.argv.slice(1);
const rootUrl = pathToFileURL(path.join(root, 'build/github-pages/index.html')).href;
globalThis.document = {baseURI: rootUrl};
globalThis.fetch = async input => {
  const file = fileURLToPath(input instanceof URL ? input : new URL(input, rootUrl));
  try {
    const bytes = await fs.readFile(file);
    return {ok:true, status:200, json:async()=>JSON.parse(bytes.toString('utf8')),
      arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
  } catch (error) {
    return {ok:false,status:404,json:async()=>({}),
      arrayBuffer:async()=>new ArrayBuffer(0)};
  }
};
const {sequence} = await import(pathToFileURL(moduleFile).href);
const manifestPath = path.join(root,'build/github-pages/static-manifest.json');
const manifest = JSON.parse(await fs.readFile(manifestPath,'utf8'));
let stdin = ''; for await (const chunk of process.stdin) stdin += chunk;
const cases = JSON.parse(stdin);
const results = {};
for (const model of ['bigram','hmm2-smooth','perceptron','crf']) {
  results[model] = {};
  for (const text of cases) {
    const isSequenceTagger = model === 'perceptron' || model === 'crf';
    const body = isSequenceTagger ? {model,text} : {
      text,segmentation_model:model,tasks:['segmentation']
    };
    const apiPath = isSequenceTagger ? '/api/v1/course/sequence' : '/api/v1/analyze';
    const ordinary = await sequence(apiPath,new URLSearchParams(),body,manifest);
    if (!isSequenceTagger) {
      const params = new URLSearchParams({model,text});
      const trace = await sequence('/api/v1/experiments/1/trace',params,{},manifest);
      results[model][text] = {ordinary,trace};
    } else results[model][text] = {ordinary};
  }
}
process.stdout.write(JSON.stringify(results));
"""


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def make_expected():
    artifacts = ROOT / "artifacts/models"
    models = {
        "bigram": BigramSegmenter.load(artifacts / "msr-extension-20260926_161208_5bfe382f"),
        "hmm2-smooth": SecondOrderHMM.load(artifacts / "msr-extension-20260926_161715_5a7e4a3c"),
        "perceptron": AveragedStructuredPerceptron.load(
            artifacts / "course_sequence/perceptron_v1"
        ),
        "crf": LinearChainCRF.load(artifacts / "course_sequence/crf_v1"),
    }
    expected = {name: {} for name in models}
    for text in CASES:
        for name in ("bigram", "hmm2-smooth"):
            model = models[name]
            segmentation = model.segment(text)
            trace = display_trace(model.trace(text), text, name)
            expected[name][text] = {
                "ordinary": {
                    "text": text,
                    "segmentation": clean(
                        {"model": name, "tokens": segmentation.tokens, "tags": segmentation.tags}
                    ),
                },
                "trace": clean(trace),
            }
        for name in ("perceptron", "crf"):
            model = models[name]
            value = model.explain(text)
            value["model"] = model.name
            value["course_extension"] = True
            value["source_split"] = "MSR official training internal subset"
            expected[name][text] = {"ordinary": clean(value)}
    return expected


def equal(actual, expected, path="$"):
    global MAX_ABS_DIFF
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        difference = abs(actual - expected) if isinstance(actual, (int, float)) else float("inf")
        MAX_ABS_DIFF = max(MAX_ABS_DIFF, difference)
        if (
            actual is None
            or not isinstance(actual, (int, float))
            or difference > TOL * max(1.0, abs(expected))
        ):
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected object, got {type(actual).__name__}")
        for key, value in expected.items():
            if key == "corpus_source":  # exporter deliberately redacts local absolute paths
                continue
            if key not in actual:
                raise AssertionError(f"{path}.{key}: missing from browser result")
            equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: expected list length {len(expected)}, got {actual!r}")
        for index, value in enumerate(expected):
            equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    expected = make_expected()
    module = ROOT / "app/static/js/static-sequence.js"
    env = os.environ.copy()
    env.setdefault("NODE_NO_WARNINGS", "1")
    process = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            NODE_RUNNER,
            str(ROOT),
            str(module),
            "",
        ],
        cwd=ROOT,
        env=env,
        input=json.dumps(CASES, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if process.returncode:
        raise RuntimeError(f"Node static sequence call failed:\n{process.stderr}")
    actual = json.loads(process.stdout)

    checked = 0
    for model in expected:
        for text in CASES:
            browser_value, python_value = actual[model][text], expected[model][text]
            equal(
                browser_value["ordinary"], python_value["ordinary"], f"{model}[{text!r}].ordinary"
            )
            if "trace" in python_value:
                browser_trace = browser_value["trace"]
                source_trace = python_value["trace"]
                equal(
                    browser_trace["tokens"],
                    source_trace["tokens"],
                    f"{model}[{text!r}].trace.tokens",
                )
                equal(browser_trace["tags"], source_trace["tags"], f"{model}[{text!r}].trace.tags")
                if model == "bigram":
                    for key in ("cost", "eos_cost", "lattice", "edges", "states"):
                        equal(
                            browser_trace[key], source_trace[key], f"{model}[{text!r}].trace.{key}"
                        )
                else:

                    def hmm2_chunks(value):
                        if value.get("chunks"):
                            return [
                                [
                                    chunk.get("offset"),
                                    chunk.get("text"),
                                    chunk.get("tags"),
                                    chunk.get("score"),
                                    chunk.get("backpointer"),
                                    chunk.get("best_log_probability"),
                                ]
                                for chunk in value["chunks"]
                            ]
                        return [
                            [
                                0,
                                text,
                                value.get("tags"),
                                value.get("score"),
                                value.get("backpointer"),
                                value.get("best_log_probability"),
                            ]
                        ]

                    equal(
                        hmm2_chunks(browser_trace),
                        hmm2_chunks(source_trace),
                        f"{model}[{text!r}].trace.chunks",
                    )
            checked += 1
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": checked,
                "texts": CASES,
                "models": list(expected),
                "tolerance": TOL,
                "max_absolute_numeric_difference": MAX_ABS_DIFF,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
