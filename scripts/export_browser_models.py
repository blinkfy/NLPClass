"""Export frozen HMM-1 parameters for the static browser implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def export_models(app: Any, destination: Path) -> dict[str, Any]:
    """Write available frozen HMM-1 variants and return a resource manifest.

    This reads completed RunStore entries and their already-fitted models. It never
    trains or evaluates a model. Missing variants are reported in ``unavailable``.
    """
    nlp = getattr(app.state, "nlp", None)
    store = getattr(app.state, "store", None) or getattr(nlp, "store", None)
    if store is None:
        return {
            "models": {},
            "unavailable": {"service": "app.state.store / app.state.nlp.store unavailable"},
        }
    output = Path(destination) / "models"
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"models": {}, "unavailable": {}}

    def portable(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: portable(item)
                for key, item in value.items()
                if key.lower() not in {"destination", "settings"}
                and not key.lower().endswith("_path")
            }
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and re.match(r"^(?:[A-Za-z]:[\\/]|/)", value):
            return None
        return value

    for variant in ("smooth", "mle"):
        model_name = f"hmm1-{variant}"
        try:
            runs = []
            for run_file in store.directory.glob("benchmarks/*/*/run.json"):
                try:
                    info = json.loads(run_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    info.get("run_type") == "frozen-final-benchmark"
                    and info.get("variant") == variant
                ):
                    runs.append((run_file.parent.name, info))
            if not runs:
                raise FileNotFoundError(f"No frozen-final-benchmark HMM-1 {variant} run")
            run_id, info = sorted(runs, key=lambda item: item[0])[-1]
            model = store.model(run_id)
            model.require_fitted()
            payload = {
                "format_version": 1,
                "model": model_name,
                "run": portable(info),
                "states": list(model.states),
                "vocabulary": model.vocabulary,
                "start": model.pi.tolist(),
                "transition": model.A.tolist(),
                "emission": model.B.tolist(),
                "config": {
                    "smoothing": model.config.smoothing,
                    "alpha": model.config.alpha,
                    "unknown_strategy": model.config.unknown_strategy,
                },
            }
            filename = f"{model_name}.json"
            (output / filename).write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            manifest["models"][model_name] = {
                "path": f"models/{filename}",
                "run_id": info.get("run_id", run_id),
                "variant": variant,
                "vocabulary_size": len(model.vocabulary) - 1,
            }
        except Exception as error:  # optional model assets must not block other variants
            manifest["unavailable"][model_name] = str(error) or error.__class__.__name__

    if nlp is None:
        manifest["unavailable"]["service"] = "app.state.nlp is unavailable"

    if nlp is not None:
        try:
            _, info = nlp.extension_run("longest")
            dictionary = nlp._load_extension_segmenter("longest")
            payload = {
                "format_version": 1,
                "model": "longest",
                "run": portable(info),
                "words": sorted(dictionary.words),
                "max_word_length": dictionary.max_word_length,
            }
            filename = "longest.json"
            (output / filename).write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            manifest["models"]["longest"] = {
                "path": f"models/{filename}",
                "run_id": info.get("run_id"),
                "dictionary_size": len(dictionary.words),
            }
        except Exception as error:  # optional extension resources must be reported, not fatal
            manifest["unavailable"]["longest"] = str(error) or error.__class__.__name__
    return manifest
