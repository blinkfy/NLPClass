"""Convert frozen TextCNN/BiLSTM-CRF weights to browser float32 tensors."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from export_browser_classifiers import unicode_ranges


def write_tensors(output, arrays):
    tensors = {}
    for name, value in arrays.items():
        data = np.asarray(value, dtype="<f4").tobytes(order="C")
        filename = name.replace(".", "-") + ".f32"
        (output / filename).write_bytes(data)
        tensors[name] = {
            "file": filename,
            "shape": list(value.shape),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return tensors


def export_neural(app, destination):
    from export_static_data import _scrub

    destination = Path(destination)
    base = destination / "models/neural"
    base.mkdir(parents=True, exist_ok=True)
    result = {"unavailable": {}}
    store = app.state.classification.store
    pointer = store.directory / "stage3_latest.json"
    if pointer.is_file():
        run = json.loads(pointer.read_text("utf-8"))["run_id"]
        directory = store.resolve(run)
        artifacts = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))["files"]

        def verified(name):
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[name]:
                raise ValueError(f"Frozen TextCNN hash mismatch: {name}")
            return path

        meta = json.loads(verified("final/model/model.json").read_text("utf-8"))
        architecture = meta["architecture"]
        if (
            architecture["embedding_dim"] != 128
            or architecture["kernels"] != [3, 4, 5]
            or architecture["channels_per_kernel"] != 128
            or architecture["output_classes"] != 5
        ):
            raise ValueError("Unsupported TextCNN architecture for browser conversion")
        vocab = json.loads(verified("final_vocabulary/vocabulary.json").read_text("utf-8"))
        config = json.loads(verified("config.json").read_text("utf-8"))
        output = base / "textcnn"
        output.mkdir(exist_ok=True)
        with np.load(verified("final/model/state.npz"), allow_pickle=False) as arrays:
            if any(arrays[name].dtype != np.float32 for name in arrays.files):
                raise ValueError("TextCNN browser export requires original float32 weights")
            tensors = write_tensors(output, {name: arrays[name] for name in arrays.files})
        payload = {
            "kind": "textcnn",
            "model_id": "C3",
            "run_id": run,
            "architecture": meta["architecture"],
            "labels": meta["label_order"],
            "vocabulary": vocab,
            "preprocessing": config["preprocessing"],
            "tokenizer_run_id": config["tokenizer_run_id"],
            "tensors": tensors,
            "source_sha256": artifacts["final/model/state.npz"],
        }
        (output / "model.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        result["textcnn"] = {
            "path": "models/neural/textcnn/model.json",
            "run_id": run,
            "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
        }
    try:
        run_path, info = app.state.nlp.extension_run("bilstm-crf")
        model_path = Path(info["model_path"])
        metadata = json.loads((model_path / "metadata.json").read_text("utf-8"))
        state = model_path / "state_dict.pt"
        if hashlib.sha256(state.read_bytes()).hexdigest() != metadata["state_dict_sha256"]:
            raise ValueError("Frozen BiLSTM hash mismatch")
        if (
            metadata["config"]["num_layers"] != 1
            or metadata["config"]["embedding_dim"] != 128
            or metadata["config"]["hidden_size"] != 128
        ):
            raise ValueError("Browser BiLSTM converter requires the frozen single-layer network")
        output = base / "bilstm-crf"
        output.mkdir(exist_ok=True)
        runtime = app.state.store.settings.root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
        job = subprocess.run(
            [
                str(runtime),
                str(Path(__file__).resolve()),
                "--extract-bilstm",
                str(state),
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(app.state.store.settings.root / "src"),
            },
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if job.returncode:
            raise ValueError(job.stderr[-600:])
        tensors = json.loads((output / "tensors.json").read_text("utf-8"))
        history_path = run_path / "training_history.json"
        payload = {
            "kind": "bilstm-crf",
            "model_id": "bilstm-crf",
            "run_id": info["run_id"],
            "run": _scrub(info, True, False),
            "config": metadata["config"],
            "vocabulary": metadata["vocabulary"],
            "tensors": tensors,
            "training": metadata["training_metadata"],
            "history": json.loads(history_path.read_text("utf-8"))
            if history_path.is_file()
            else None,
            "whitespace": unicode_ranges(str.isspace),
            "source_sha256": metadata["state_dict_sha256"],
        }
        (output / "model.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (output / "tensors.json").unlink()
        result["bilstm"] = {
            "path": "models/neural/bilstm-crf/model.json",
            "run_id": info["run_id"],
            "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
        }
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        result["unavailable"]["bilstm-crf"] = str(error)
    return result


if __name__ == "__main__" and sys.argv[1] == "--extract-bilstm":
    import torch

    state = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
    arrays = {
        name: value.numpy()
        for name, value in state.items()
        if not name.startswith("crf.parametrizations.")
    }
    for name in ("start_transitions", "end_transitions", "transitions"):
        original = state[f"crf.parametrizations.{name}.original"].numpy()
        mask = state[f"crf.parametrizations.{name}.0.mask"].numpy()
        arrays[f"crf.{name}"] = np.where(mask, original, np.float32(-10000))
    output = Path(sys.argv[3])
    tensors = write_tensors(output, arrays)
    (output / "tensors.json").write_text(json.dumps(tensors), encoding="utf-8")
