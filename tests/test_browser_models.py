"""Parity checks for exported frozen models in the dependency-free browser runtime."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.api.trace_display import display_trace
from app.main import app
from nlplab.core.exceptions import NoValidPathError
from nlplab.extensions.traces import algorithm_trace
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.sequence_labeling.tags import STATES
from nlplab.sequence_labeling.viterbi import viterbi
from scripts.export_browser_models import export_models

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["商品和服务", "商品🙂和服务", "中文 🙂 English", "  Hello\tworld\n", "OpenAI NLP 2026"]
NODE_SCRIPT = r"""
import fs from 'node:fs/promises';
import path from 'node:path';
globalThis.document = {baseURI: 'http://localhost/segmentation'};
globalThis.fetch = async (url) => ({
  ok: true,
  json: async () => {
    const name = path.basename(new URL(url).pathname);
    const file = path.join(process.env.NLPLAB_BROWSER_TEST_MODELS, name);
    return JSON.parse(await fs.readFile(file, 'utf8'));
  },
});
const {compute} = await import(process.env.NLPLAB_BROWSER_TEST_MODULE);
const specs = JSON.parse(await new Promise(resolve => {
  let value = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => value += chunk);
  process.stdin.on('end', () => resolve(value));
}));
const output = [];
for (const spec of specs) {
  try {
    const analysis = await compute('/api/v1/analyze', {}, {
      text: spec.text, tasks: ['segmentation'], run_id: 'latest', segmentation_model: spec.model,
    });
    const trace = spec.withTrace ? await compute('/api/v1/experiments/1/trace',
      new URLSearchParams({text: spec.text, model: spec.model, run_id: 'latest'})) : null;
    output.push({analysis, trace});
  } catch (error) { output.push({error: error.message}); }
}
process.stdout.write(JSON.stringify(output));
"""


def _contains_absolute_path(value):
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return bool(len(value) > 2 and value[1:3] in {":\\", ":/"}) or value.startswith("/")
    return False


@pytest.fixture(scope="module")
def browser_results(tmp_path_factory):
    destination = tmp_path_factory.mktemp("browser-export")
    manifest = export_models(app, destination)
    assert set(manifest["models"]) >= {"hmm1-smooth", "hmm1-mle", "longest"}
    model_directory = destination / "models"
    for model_file in model_directory.glob("*.json"):
        payload = json.loads(model_file.read_text(encoding="utf-8"))
        assert "destination" not in payload
        assert not _contains_absolute_path(payload)

    specs = [{"model": "hmm1-smooth", "text": text, "withTrace": True} for text in TEXTS]
    specs += [
        {"model": "hmm1-mle", "text": "商品和服务", "withTrace": True},
        {"model": "hmm1-mle", "text": "商品🙂和服务", "withTrace": False},
    ]
    specs += [{"model": "longest", "text": text, "withTrace": True} for text in TEXTS]
    environment = {
        **os.environ,
        "NLPLAB_BROWSER_TEST_MODELS": str(model_directory),
        "NLPLAB_BROWSER_TEST_MODULE": (ROOT / "app/static/js/static-compute.js").as_uri(),
    }
    completed = subprocess.run(
        ["node", "--experimental-default-type=module", "--input-type=module", "-e", NODE_SCRIPT],
        input=json.dumps(specs, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=environment,
    )
    return specs, json.loads(completed.stdout.decode("utf-8"))


def test_static_browser_matches_frozen_hmm_and_longest(browser_results):
    specs, outputs = browser_results
    store = app.state.nlp.store
    run_ids = {
        "hmm1-smooth": "20260926_130612_32a5e168_smooth",
        "hmm1-mle": "20260926_130612_32a5e168_mle",
    }
    longest = app.state.nlp._load_extension_segmenter("longest")

    for spec, output in zip(specs, outputs):
        text, model_name = spec["text"], spec["model"]
        if model_name == "longest":
            expected = longest.segment(text)
            assert output["analysis"]["segmentation"]["tokens"] == expected.tokens
            assert output["analysis"]["segmentation"]["tags"] == expected.tags
            trace = output["trace"]
            expected_trace = display_trace(longest.trace(text), text, "longest", longest.words)
            for key in ("steps", "tokens", "tags", "fallback_count", "character_count"):
                assert trace[key] == expected_trace[key]
            continue

        segmenter = HMMSegmenter(store.model(run_ids[model_name]))
        try:
            expected = segmenter.segment(text)
        except NoValidPathError:
            assert "error" in output
            assert "BMES" in output["error"]
            continue
        actual = output["analysis"]["segmentation"]
        assert actual["tokens"] == expected.tokens
        assert actual["tags"] == expected.tags
        assert "error" not in output
        if spec["withTrace"]:
            if model_name == "hmm1-smooth":
                expected_trace = display_trace(
                    algorithm_trace(app.state.nlp, text, "hmm1-smooth"), text, "hmm1-smooth"
                )
            else:
                tagger = segmenter.tagger
                chunks, offset = [], 0
                for chunk in re.split(r"(\s)", text):
                    if not chunk:
                        continue
                    if chunk.isspace():
                        chunks.append(
                            {"offset": offset, "text": chunk, "tags": ["S"], "whitespace": True}
                        )
                    else:
                        indices = [tagger.character_index.get(char, 0) for char in chunk]
                        decoded = viterbi(tagger.log_pi, tagger.log_A, tagger.log_B[:, indices].T)
                        chunks.append(
                            {
                                "offset": offset,
                                "text": chunk,
                                "tags": decoded.tags,
                                "score": [
                                    [float(value) if np.isfinite(value) else None for value in row]
                                    for row in decoded.score
                                ],
                                "backpointer": decoded.backpointer.tolist(),
                                "best_log_probability": decoded.best_log_probability,
                            }
                        )
                    offset += len(chunk)
                expected_trace = {
                    "model": model_name,
                    "states": list(STATES),
                    "chunks": chunks,
                    "tokens": expected.tokens,
                    "tags": expected.tags,
                }
            trace = output["trace"]
            assert trace["tags"] == expected_trace["tags"]
            assert trace["tokens"] == expected_trace["tokens"]
            for actual_chunk, expected_chunk in zip(trace["chunks"], expected_trace["chunks"]):
                assert actual_chunk["tags"] == expected_chunk["tags"]
                if expected_chunk.get("whitespace"):
                    continue
                assert actual_chunk["backpointer"] == expected_chunk["backpointer"]
                actual_scores = [
                    [np.nan if value is None else value for value in row]
                    for row in actual_chunk["score"]
                ]
                expected_scores = [
                    [np.nan if value is None else value for value in row]
                    for row in expected_chunk["score"]
                ]
                np.testing.assert_allclose(
                    actual_scores, expected_scores, rtol=1e-12, atol=1e-12, equal_nan=True
                )


def test_browser_rejects_invalid_or_unmatched_run_ids():
    script = r"""
const {compute} = await import(process.env.NLPLAB_BROWSER_TEST_MODULE);
globalThis.document = {baseURI: 'http://localhost/segmentation'};
globalThis.fetch = async () => ({ok:true,json:async()=>({})});
try {
  await compute('/api/v1/analyze', {}, {
    text:'商品', tasks:['segmentation'], run_id:'../bad', segmentation_model:'hmm1-smooth'
  });
  process.exitCode=1;
}
catch (error) { if (error.message !== 'run_id 无效') process.exitCode=2; }
const unmatched = await compute('/api/v1/analyze', {}, {
  text:'商品', tasks:['segmentation'], run_id:'unrelated-run', segmentation_model:'hmm1-smooth'
});
if (unmatched !== undefined) process.exitCode=3;
"""
    result = subprocess.run(
        ["node", "--experimental-default-type=module", "--input-type=module", "-e", script],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env={
            **os.environ,
            "NLPLAB_BROWSER_TEST_MODULE": (ROOT / "app/static/js/static-compute.js").as_uri(),
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
