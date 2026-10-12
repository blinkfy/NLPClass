"""Compare the frozen browser kernels with their saved PyTorch models."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "build/github-pages/models/neural"
RUNTIME = ROOT / "artifacts/runtime/hanlp-venv/Scripts/python.exe"


PYTHON_PROBE = r"""
import json, sys
from pathlib import Path
import torch
from nlplab.core.config import Settings
from nlplab.services.experiment_service import RunStore
from nlplab.classification.textcnn import TextCNN
from nlplab.features.word_sequence import WordSequenceVocabulary
from nlplab.extensions.bilstm_crf import BiLSTMCRFSegmenter
from nlplab.course_neural.crf_trace import crf_trace

root = Path(sys.argv[1])
request = json.load(sys.stdin)
torch.set_num_threads(2)
cnn_run = request['cnn_run']
cnn_dir = RunStore(Settings(root), 'exp3').resolve(cnn_run)
cnn = TextCNN.load(cnn_dir / 'final/model')
vocab = WordSequenceVocabulary.load(cnn_dir / 'final_vocabulary')
cnn_rows = []
with torch.no_grad():
    for tokens in request['cnn_tokens']:
        encoded = vocab.encode([tokens])
        logits, pooled, positions = cnn(
            torch.as_tensor(encoded['ids']), torch.as_tensor(encoded['lengths']), True
        )
        cnn_rows.append({
            'ids': encoded['ids'][0, :int(encoded['lengths'][0])].tolist(),
            'positions': encoded['original_positions'][0],
            'diagnostic': encoded['stats']['per_document'][0],
            'logits': logits[0].tolist(),
            'pooled': pooled[0].tolist(),
            'argmax': positions[0].tolist(),
            'softmax': torch.softmax(logits, dim=1)[0].tolist(),
        })
bi_store = RunStore(Settings(root), 'exp1')
matches = []
for path in (root / 'outputs/exp1/extensions').glob('*/run.json'):
    info = json.loads(path.read_text(encoding='utf-8'))
    if info.get('model') == 'bilstm-crf' and info.get('phase') == 'final':
        matches.append(info)
if not matches:
    raise RuntimeError('No frozen final BiLSTM-CRF run found')
bi_info = sorted(matches, key=lambda item: item['run_id'])[-1]
bi = BiLSTMCRFSegmenter.load(Path(bi_info['model_path']))
bi.network.module.eval()
bi_rows = []
with torch.no_grad():
    for text in request['bi_texts']:
        ids = bi._encode(text)
        chars = torch.tensor([ids], dtype=torch.long, device=bi._device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=bi._device)
        emissions = bi.network.emissions(chars, lengths)[0].cpu().tolist()
        tags = bi.predict(list(text))
        transitions = bi.network.crf.transitions.detach().cpu().tolist()
        start = bi.network.crf.start_transitions.detach().cpu().tolist()
        end = bi.network.crf.end_transitions.detach().cpu().tolist()
        numeric = crf_trace(emissions, transitions, start, end)
        bi_rows.append({'ids': ids, 'emissions': emissions, 'tags': tags,
                        'path': numeric['path'], 'log_partition': numeric['log_partition']})
print(json.dumps({'cnn': cnn_rows, 'bi': bi_rows}, ensure_ascii=False))
"""


NODE_PROBE = r"""
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const root = process.cwd();
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const dir = `${root}/build/github-pages/models/neural`;
const load = async name => {
  const model = JSON.parse(fs.readFileSync(`${dir}/${name}/model.json`, 'utf8'));
  const tensors = {};
  for (const [key, spec] of Object.entries(model.tensors)) {
    const buffer = fs.readFileSync(`${dir}/${name}/${spec.file}`);
    tensors[key] = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }
  return {model, tensors};
};
const kernels = await import(pathToFileURL(`${root}/app/static/js/static-neural-kernels.js`));
const browser = await import(pathToFileURL(`${root}/app/static/js/static-neural.js`));
const cnn = await load('textcnn'), bi = await load('bilstm-crf');
const cnnRows = request.cnn_tokens.map(tokens => {
  const encoded = browser.encodeSequence(cnn.model, tokens);
  const raw = kernels.textcnn(cnn.model, cnn.tensors, encoded.ids);
  return {encoded, raw, prediction:browser.cnnPrediction(cnn.model, raw, encoded)};
});
const biRows = request.bi_ids.map(ids => kernels.bilstm(bi.model, bi.tensors, ids));
process.stdout.write(JSON.stringify({cnn:cnnRows, bi:biRows}));
"""


def _run_json(command, *, input_text, env=None):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=180,
        env=env,
    )
    return json.loads(completed.stdout)


def _assert_close(actual, expected, *, atol, rtol, context):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}"
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{context}: max_abs_error={error:.9g}"
    )


@pytest.mark.skipif(not RUNTIME.is_file(), reason="HanLP PyTorch runtime is unavailable")
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_browser_neural_kernels_match_saved_pytorch_models():
    cnn_model = json.loads((MODEL_DIR / "textcnn/model.json").read_text(encoding="utf-8"))
    bi_model = json.loads((MODEL_DIR / "bilstm-crf/model.json").read_text(encoding="utf-8"))
    known = cnn_model["vocabulary"]["terms"]
    # Include each short convolution boundary, ordinary Chinese words, UNK, and a
    # long head-tail sequence. The synthetic filler is not a corpus document.
    cnn_tokens = [
        [],
        [known[2]],
        [known[2], known[3]],
        [known[2], known[3], known[4]],
        [known[2], known[3], known[4], known[5]],
        [known[2], known[3], known[4], known[5], known[6]],
        ["基金", "市场", "收益", "未知合成词"],
        [known[2 + i % (len(known) - 2)] for i in range(820)],
    ]
    bi_texts = ["中", "中文", "😀", "English", "中文😀abc", "中国市场经济发展研究成果" * 3 + "中英"]
    request = {"cnn_run": cnn_model["run_id"], "cnn_tokens": cnn_tokens, "bi_texts": bi_texts}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    reference = _run_json(
        [str(RUNTIME), "-c", PYTHON_PROBE, str(ROOT)],
        input_text=json.dumps(request, ensure_ascii=False),
        env=env,
    )
    request["bi_ids"] = [row["ids"] for row in reference["bi"]]
    actual = _run_json(
        [shutil.which("node"), "--input-type=module", "-e", NODE_PROBE],
        input_text=json.dumps(request, ensure_ascii=False),
    )

    assert bi_model["kind"] == "bilstm-crf"
    for index, (got, expected) in enumerate(zip(actual["cnn"], reference["cnn"], strict=True)):
        encoded = got["encoded"]
        assert encoded["ids"] == expected["ids"], f"CNN[{index}] encoded IDs mismatch"
        assert encoded["positions"] == expected["positions"], (
            f"CNN[{index}] retained positions mismatch"
        )
        assert encoded["diagnostic"] == expected["diagnostic"], (
            f"CNN[{index}] sequence diagnostic mismatch"
        )
        raw = got["raw"]
        _assert_close(
            raw["logits"], expected["logits"], atol=2e-5, rtol=2e-5,
            context=f"CNN[{index}] logits",
        )
        _assert_close(
            raw["softmax"], expected["softmax"], atol=2e-5, rtol=2e-5,
            context=f"CNN[{index}] softmax",
        )
        _assert_close(
            raw["pooled"], expected["pooled"], atol=2e-5, rtol=2e-5,
            context=f"CNN[{index}] pooled",
        )
        assert raw["argmax"] == expected["argmax"], f"CNN[{index}] argmax mismatch"
        assert int(np.argmax(raw["logits"])) == int(np.argmax(expected["logits"])), (
            f"CNN[{index}] class mismatch"
        )
        expected_class = cnn_model["labels"][int(np.argmax(expected["logits"]))]
        assert got["prediction"]["prediction"] == expected_class

    for index, (got, expected) in enumerate(zip(actual["bi"], reference["bi"], strict=True)):
        _assert_close(
            got["emissions"], expected["emissions"], atol=1e-5, rtol=1e-5,
            context=f"BiLSTM[{index}] emissions",
        )
        assert got["tags"] == expected["tags"], f"BiLSTM[{index}] predicted tags mismatch"
        trace = got["numeric_trace"]
        assert trace["path"] == expected["path"], f"BiLSTM[{index}] CRF path mismatch"
        assert trace["verified_path"] is True
        assert abs(trace["log_partition"] - expected["log_partition"]) <= 1e-5
