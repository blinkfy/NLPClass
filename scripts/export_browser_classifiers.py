"""Export frozen C0/C1/C2 numeric inference parameters; never fit or evaluate."""

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np


def unicode_ranges(predicate):
    ranges = []
    start = previous = None
    for codepoint in range(0x110000):
        if predicate(chr(codepoint)):
            if start is None:
                start = codepoint
            previous = codepoint
        elif start is not None:
            ranges.append([start, previous])
            start = previous = None
    if start is not None:
        ranges.append([start, previous])
    return ranges


def export_classifiers(app, destination):
    output = Path(destination) / "models"
    output.mkdir(parents=True, exist_ok=True)
    store = app.state.classification.store
    manifest = {"runs": {}, "unavailable": {}}
    loaded = {}

    def emit(directory, model_id):
        key = (directory.name, model_id)
        if key in loaded:
            return loaded[key]
        artifacts = json.loads((directory / "artifact_manifest.json").read_text("utf-8"))["files"]

        def verified(relative):
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Invalid classifier artifact path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[relative]:
                raise ValueError(f"Frozen classifier hash mismatch: {relative}")
            return path

        def read(relative):
            return json.loads(verified(relative).read_text("utf-8"))

        meta = read(f"final/{model_id}/model.json")
        with np.load(verified(f"final/{model_id}/parameters.npz"), allow_pickle=False) as arrays:
            weights = arrays["weights"].tolist()
            intercept = arrays["intercept"].tolist()
        payload = {
            "model_id": model_id,
            "run_id": directory.name,
            "labels": meta["label_order"],
            "weights": weights,
            "intercept": intercept,
        }
        if model_id == "C2":
            features = read("final/representation/selected_features.json")
            payload["terms"] = [
                item["term"] for item in sorted(features, key=lambda f: f["column"])
            ]
            with np.load(
                verified("final/representation/parameters.npz"), allow_pickle=False
            ) as arrays:
                payload["idf"] = arrays["idf"].tolist()
            payload["tfidf_config"] = read("final/representation/representation.json")[
                "tfidf_config"
            ]
        else:
            vocabulary = read("final/tfidf/vocabulary.json")
            payload["terms"] = [
                term
                for term, _ in sorted(vocabulary["vocabulary"].items(), key=lambda item: item[1])
            ]
            payload["idf"] = np.load(verified("final/tfidf/idf.npy"), allow_pickle=False).tolist()
            payload["tfidf_config"] = vocabulary["config"]
            config = read("config.json")
            payload["preprocessing"] = config["preprocessing"]
            payload["tokenizer_run_id"] = config["tokenizer_run_id"]
        assert len(payload["terms"]) == len(payload["idf"]) == len(weights[0])
        assert payload["tfidf_config"]["sublinear_tf"] and payload["tfidf_config"]["norm"] == "l2"
        file = output / f"{directory.name}-{model_id}.json"
        file.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        loaded[key] = {
            "path": f"models/{file.name}",
            "run_id": directory.name,
            "features": len(payload["terms"]),
        }
        return loaded[key]

    for run in app.state.classification.list_runs():
        try:
            directory = store.resolve(run["run_id"])
            info = store.read(directory.name, "run.json")
            parent = store.resolve(info.get("stage1_run_id", directory.name))
            available = {model: emit(parent, model) for model in ("C0", "C1")}
            if info["run_type"] in {"formal-exp3-stage2", "formal-exp3-stage3"}:
                c2 = store.resolve(info.get("stage2_run_id", directory.name))
                available["C2"] = emit(c2, "C2")
            manifest["runs"][directory.name] = available
        except (OSError, KeyError, ValueError, AssertionError) as error:
            manifest["unavailable"][run["run_id"]] = str(error)
    latest = store.resolve("latest").name
    manifest["runs"]["latest"] = dict(manifest["runs"].get(latest, {}))
    pointer = store.directory / "stage2_latest.json"
    if pointer.is_file():
        c2 = store.resolve(json.loads(pointer.read_text("utf-8"))["run_id"])
        manifest["runs"]["latest"]["C2"] = emit(c2, "C2")
    character_rules = {
        "unicode_version": unicodedata.unidata_version,
        "assigned": unicode_ranges(lambda c: unicodedata.category(c) != "Cn"),
        "content": unicode_ranges(
            lambda c: (
                "a" <= c <= "z"
                or unicodedata.category(c) == "Nd"
                or unicodedata.name(c, "").startswith(
                    ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
                )
            )
        ),
    }
    file = output / "classifier-unicode.json"
    file.write_text(json.dumps(character_rules, separators=(",", ":")), encoding="utf-8")
    manifest["unicode_path"] = "models/classifier-unicode.json"
    return manifest
