"""Compare frozen browser NER fixture values with the original HanLP model."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "HANLP_HOME", str(Path(__file__).resolve().parents[1] / "artifacts/cache/hanlp")
)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hanlp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/cache/hanlp/ner/ner_bert_base_msra_20211227_114712"
FIXTURE = ROOT / "build/ner-parity.json"
OUTPUT = ROOT / "build/ner-oracle.json"
MAX_LENGTH = 128


def _numpy(value):
    if isinstance(value, (list, tuple)):
        return _numpy(value[0])
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def main() -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    component = hanlp.load(str(MODEL), devices=-1)
    vocabs = getattr(component, "vocabs", None)
    tag_vocab = vocabs.get("tag") if isinstance(vocabs, dict) else None
    if tag_vocab is None:
        tag_vocab = component.transform.tag_vocab
    tag_names = list(tag_vocab.idx_to_token)
    if len(tag_names) != 14:
        raise AssertionError(f"Expected 14 tags including pad, got {len(tag_names)}")
    pad_index = tag_names.index("<pad>")
    max_abs = 0.0
    outputs = []

    for case in cases:
        ids = list(case["ids"])
        if len(ids) > MAX_LENGTH:
            raise AssertionError(f"Fixture exceeds input length: {case['text']!r}")
        padded = ids + [0] * (MAX_LENGTH - len(ids))
        mask = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
        inputs = (
            np.asarray([padded], dtype=np.int32),
            np.asarray([mask], dtype=np.int32),
            np.zeros((1, MAX_LENGTH), dtype=np.int32),
        )
        result = component.model(inputs, training=False)
        logits = _numpy(result)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != 14:
            raise AssertionError(f"Unexpected HanLP model output shape: {logits.shape}")
        expected = np.asarray(case["float_logits"], dtype=np.float32)
        actual = logits[0, : len(ids), :]
        # The browser fixture removes the pad class while preserving HanLP's tag order.
        actual_without_pad = np.delete(actual, pad_index, axis=1)
        if actual_without_pad.shape != expected.shape:
            raise AssertionError(
                f"Logit shape mismatch: {actual_without_pad.shape} != {expected.shape}"
            )
        difference = float(np.max(np.abs(actual_without_pad - expected)))
        max_abs = max(max_abs, difference)
        if not np.allclose(actual_without_pad, expected, atol=1e-5, rtol=1e-6):
            raise AssertionError(f"Float-logit mismatch for {case['text']!r}: {difference}")

        raw_entities = component(list(case["text"]))
        decoded = ["O"] * len(case["positions"])
        for entity_text, label, start, end in raw_entities:
            span_length = end - start
            if span_length == 1:
                decoded[start] = f"S-{label}"
            else:
                decoded[start] = f"B-{label}"
                for offset in range(start + 1, end - 1):
                    decoded[offset] = f"M-{label}"
                decoded[end - 1] = f"E-{label}"
            if entity_text != case["text"][start:end]:
                raise AssertionError(f"Entity span mismatch: {entity_text!r} at {start}:{end}")
        # HanLP tags use BMES names; the static model's fixture exposes the same tag sequence.
        if decoded != case["tags"]:
            raise AssertionError(
                f"Entity/tag mismatch for {case['text']!r}: {decoded!r} != {case['tags']!r}"
            )
        outputs.append(
            {
                "text": case["text"],
                "ids": case["ids"],
                "positions": case["positions"],
                "tags": decoded,
                "entities": [list(entity) for entity in raw_entities],
                "max_abs_logit_error": difference,
            }
        )

    OUTPUT.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "max_abs_logit_error": max_abs, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
