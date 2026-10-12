"""Convert frozen count/HMM/linear-chain models to static browser assets."""

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
from export_browser_classifiers import unicode_ranges

STATES = ["B", "M", "E", "S"]


def emit_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


def export_sequence(app, destination):
    from export_static_data import _scrub

    destination = Path(destination)
    output = destination / "models/sequence"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"models": {}, "unavailable": {}}
    for name in ("bigram", "hmm2-smooth"):
        run_path, info = app.state.nlp.extension_run(name)
        model = app.state.nlp._load_extension_segmenter(name)
        if name.startswith("hmm2"):
            payload = {
                "model": name,
                "vocabulary": model.base.vocabulary,
                "start": model.base.pi.tolist(),
                "emission": model.base.B.tolist(),
                "initial": model.initial.tolist(),
                "tensor": model.tensor.tolist(),
            }
        else:
            words = sorted(model.words)
            vocabulary = [*words, "<BOS>", "<EOS>", "<UNK_WORD>"]
            indexes = {word: i for i, word in enumerate(vocabulary)}
            rows = sorted((indexes[a], indexes[b], count) for (a, b), count in model.bigram.items())
            offsets = np.zeros(len(vocabulary) + 1, dtype="<u4")
            targets = np.array([b for _, b, _ in rows], dtype="<u4")
            counts = np.array([n for _, _, n in rows], dtype="<u4")
            for a, _, _ in rows:
                offsets[a + 1] += 1
            offsets = np.cumsum(offsets, dtype=np.uint32).astype("<u4")
            files = {}
            for key, array in (("offsets", offsets), ("targets", targets), ("counts", counts)):
                file = output / f"bigram-{key}.u32"
                file.write_bytes(array.tobytes())
                files[key] = {
                    "file": file.name,
                    "bytes": file.stat().st_size,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            payload = {
                "model": "bigram",
                "vocabulary": vocabulary,
                "unigram": [model.unigram.get(word, 0) for word in vocabulary],
                "token_total": model.token_total,
                "sentence_count": model.sentence_count,
                "max_word_length": model.max_word_length,
                "alpha": model.alpha,
                "interpolation": model.interpolation,
                "arrays": files,
            }
        payload.update(run=_scrub(info, True, False))
        file = output / f"{name}.json"
        emit_json(file, payload)
        manifest["models"][name] = {
            "path": f"models/sequence/{file.name}",
            "run_id": info["run_id"],
        }
    path = app.state.course._model_path("perceptron")
    model = app.state.course._model("perceptron")
    weights = {}
    for key, value in model.weights.items():
        if key.startswith("E|"):
            _, state, feature = key.split("|", 2)
            weights.setdefault(feature, [0.0] * 4)[STATES.index(state)] = value
    payload = {
        "model": "perceptron",
        "state_weights": weights,
        "start": [model.weights.get(f"T|<BOS>|{tag}", 0) for tag in STATES],
        "transition": [[model.weights.get(f"T|{a}|{b}", 0) for b in STATES] for a in STATES],
        "metadata": model.training_metadata,
        "update_example": model.update_example,
        "source_sha256": hashlib.sha256((path / "weights.json").read_bytes()).hexdigest(),
    }
    emit_json(output / "perceptron.json", _scrub(payload, True, False))
    manifest["models"]["perceptron"] = {"path": "models/sequence/perceptron.json"}
    path = app.state.course._model_path("crf")
    runtime = app.state.store.settings.root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
    job = subprocess.run(
        [str(runtime), str(Path(__file__).resolve()), "--crf", str(path), str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(app.state.store.settings.root / "src")},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if job.returncode:
        raise ValueError(job.stderr[-600:])
    manifest["models"]["crf"] = {"path": "models/sequence/crf.json"}
    emit_json(
        output / "unicode.json",
        {
            "whitespace": unicode_ranges(str.isspace),
            "punctuation": unicode_ranges(lambda c: unicodedata.category(c).startswith("P")),
        },
    )
    manifest["unicode_path"] = "models/sequence/unicode.json"
    return manifest


if __name__ == "__main__" and sys.argv[1] == "--crf":
    from export_static_data import _scrub

    from nlplab.sequence_labeling.linear_crf import LinearChainCRF

    model = LinearChainCRF.load(Path(sys.argv[2]))
    weights = {}
    for (feature, state), value in model._info.state_features.items():
        weights.setdefault(feature, [0.0] * 4)[STATES.index(state)] = value
    emit_json(
        Path(sys.argv[3]) / "crf.json",
        {
            "model": "crf",
            "state_weights": weights,
            "labels": model._tagger.labels(),
            "transition": [
                [model._info.transitions.get((a, b), 0) for b in STATES] for a in STATES
            ],
            "display_transition": [
                [model._info.transitions.get((a, b)) for b in STATES] for a in STATES
            ],
            "metadata": _scrub(model.training_metadata, True, False),
            "source_sha256": hashlib.sha256(
                (Path(sys.argv[2]) / "model.crfsuite").read_bytes()
            ).hexdigest(),
        },
    )
