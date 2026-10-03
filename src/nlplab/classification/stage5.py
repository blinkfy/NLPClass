"""Stage5 Train-only rule freeze, then frozen-model Validation shortcut ablations."""

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from nlplab.classification.__main__ import artifact_manifest, processor_for
from nlplab.classification.c2 import C2LinearClassifier
from nlplab.classification.linear import LinearTextClassifier
from nlplab.classification.stage3 import runtime_information, subprocess_environment
from nlplab.classification.stage4 import frozen_references
from nlplab.core.io import write_json
from nlplab.data.classification_stage5 import CleaningFreeze, Stage5Data, install_stage5_guard
from nlplab.data.course import file_sha256
from nlplab.data.sequence_annotations import atomic_json
from nlplab.data.thucnews import CATEGORIES
from nlplab.evaluation.classification import classification_metrics
from nlplab.features.char_chi2 import CharacterChi2TFIDF, content_character, sparse_stats
from nlplab.features.shortcut_cleaning import (
    clean_view,
    encode_sections,
    overlap,
    prevalence,
    segmented_word_features,
    tokenize_view,
)
from nlplab.features.text_tfidf import TextTFIDF, TFIDFConfig
from nlplab.features.word_sequence import WordSequenceVocabulary, read_json

STAGE4 = "20261001_164602_48e6c6b6"
CONDITIONS = {"OO": [], "OS": ["source"], "OT": ["template"], "OA": ["source", "template"]}
PROTOCOL = {
    "analysis_type": "exploratory development-set shortcut ablation; Validation only",
    "allowed_splits": ["train", "val"],
    "formal_test_evaluation_count": 0,
    "test_access_permitted": False,
    "rules_design_split": "train",
    "rules_freeze_before_validation": True,
    "hyperparameter_selection": False,
    "conditions": {**CONDITIONS, "AA": ["source", "template"]},
    "original_models": "reuse Stage4 ladder101_n420; verify OO against saved 420 results",
    "AA": "clean Train420 refit; same fixed hyperparameters; C3 exact12 epochs seed42",
    "boundaries": "word per segment; char newline; CNN same-segment windows only",
    "interpretation": "ablation sensitivity; not causal proof or independent benchmark",
}


def sources(root):
    names = [
        "classification/stage5.py",
        "classification/stage5_neural.py",
        "features/shortcut_cleaning.py",
        "data/classification_stage5.py",
    ]
    return {"src/nlplab/" + name: file_sha256(root / "src/nlplab" / name) for name in names}


def preregister(root, rules_path, audit_path):
    install_stage5_guard(root)
    base = root / "outputs/exp3"
    if (base / "stage5_preregistration.json").exists():
        raise ValueError("Stage5 already frozen; do not overwrite")
    train = Stage5Data(root).load("train")
    rules, audit = read_json(rules_path), read_json(audit_path)
    if rules["design_split"] != "train" or audit["train_count"] != 420:
        raise ValueError("Rules require Train-only evidence")
    if audit["train_sha256"] != file_sha256(root / "data/splits/thucnews_600/train.jsonl"):
        raise ValueError("Train-only cleaning evidence fingerprint changed")
    references, _ = frozen_references(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    directory = base / "stage5" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "cleaning_rules.json", rules)
    write_json(directory / "train_audit.json", audit)
    write_json(directory / "protocol.json", PROTOCOL)
    write_json(directory / "frozen_config_references.json", references)
    stage4 = base / "stage4" / STAGE4
    write_json(
        directory / "original_model_reference.json",
        {
            "run_id": STAGE4,
            "point": "ladder101_n420",
            "manifest_sha256": file_sha256(stage4 / "artifact_manifest.json"),
        },
    )
    write_json(
        directory / "cleaning_freeze.json",
        {
            "root": str(root),
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "validation_opened": False,
            "design_ids": [d.id for d in train],
            "train_sha256": file_sha256(root / "data/splits/thucnews_600/train.jsonl"),
            "files": {
                name: file_sha256(directory / name)
                for name in (
                    "cleaning_rules.json",
                    "train_audit.json",
                    "protocol.json",
                    "frozen_config_references.json",
                )
            },
            "source_hashes": sources(root),
        },
    )
    # Per-class statistics use labels only after label-free matching rules are finalized.
    views = [clean_view(d.text, rules, {"source", "template"}) for d in train]
    write_json(directory / "train_prevalence.json", prevalence(train, views))
    write_json(directory / "environment.json", runtime_information(root))
    atomic_json(
        base / "stage5_preregistration.json",
        {"run_id": run_id, "freeze_sha256": file_sha256(directory / "cleaning_freeze.json")},
    )
    print(
        json.dumps({"run_id": run_id, "rules_frozen": True, "validation_opened": False}), flush=True
    )


def word_representation(config, path=None):
    rep = TextTFIDF.load(path) if path else TextTFIDF(TFIDFConfig(**config))
    rep.vectorizer.set_params(analyzer=segmented_word_features)
    return rep


def save_encoding(path, encoded, docs):
    np.savez_compressed(
        path.with_suffix(".npz"),
        **{k: encoded[k] for k in ("ids", "lengths", "segment_ids")},
        labels=np.asarray([CATEGORIES.index(d.label) for d in docs], dtype=np.int64),
    )
    write_json(
        path.with_suffix(".json"),
        {
            **{
                k: encoded[k]
                for k in ("tokens", "original_positions", "character_offsets", "stats")
            },
            "document_ids": [d.id for d in docs],
        },
    )


def feature_positions(token_row, text, model_id):
    positions = {}

    def add(term, offsets):
        positions.setdefault(term, []).append(offsets)

    if model_id == "C2":
        for i in range(len(text) - 1):
            if content_character(text[i]) and content_character(text[i + 1]):
                add(text[i : i + 2], [[i, i + 2]])
    else:
        tokens, offsets = token_row["tokens"], token_row["offsets"]
        for i, term in enumerate(tokens):
            add(term, [offsets[i]])
            if i + 1 < len(tokens):
                add(term + " " + tokens[i + 1], [offsets[i], offsets[i + 1]])
    return positions


def evaluate_linear(model, rep, matrix, docs, token_rows, views, all_spans, condition, reference):
    start = perf_counter()
    predicted = model.predict(matrix).tolist()
    explanations = []
    for i, (doc, row) in enumerate(zip(docs, matrix)):
        explained = model.explain(row, rep.terms)
        if condition == "OO":
            positions = feature_positions(token_rows[i], views[i]["text"], model.model_id)
            for feature in explained["top_contributing_features"]:
                occurrences = positions.get(feature["term"], [])
                feature["original_occurrences"] = len(occurrences)
                for kind in ("source", "template"):
                    spans = [s for s in all_spans[i] if s["kind"] == kind]
                    feature[kind + "_occurrences"] = sum(
                        overlap(offsets, spans) for offsets in occurrences
                    )
        explanations.append(
            {
                "document_id": doc.id,
                "gold": doc.label,
                "prediction": predicted[i],
                "correct": predicted[i] == doc.label,
                "text_snippet": doc.text[:600],
                **explained,
            }
        )
    return {
        "model_id": model.model_id,
        "condition": condition,
        "metrics": classification_metrics([d.label for d in docs], predicted),
        "predictions": explanations,
        "model_reference": reference,
        "inference_seconds": perf_counter() - start,
        "matrix": sparse_stats(matrix),
    }


def neural_job(root, directory, job):
    job_path = directory / (job["mode"] + "_neural_job.json")
    write_json(job_path, job)
    completed = subprocess.run(
        [
            str(root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"),
            "-m",
            "nlplab.classification.stage5_neural",
            "--job",
            str(job_path),
        ],
        cwd=root,
        env=subprocess_environment(root),
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode:
        raise ValueError(
            f"Stage5 neural worker failed with exit {completed.returncode}; no automatic rerun"
        )


def build_flips(evaluations, docs, spans):
    flips = []
    for model_id in ("C0", "C1", "C2", "C3"):
        original = evaluations[(model_id, "OO")]["predictions"]
        for condition in ("OS", "OT", "OA", "AA"):
            changed = evaluations[(model_id, condition)]["predictions"]
            for i, (old, new, doc) in enumerate(zip(original, changed, docs)):
                if old["prediction"] == new["prediction"]:
                    continue
                before, after = old["prediction"] == doc.label, new["prediction"] == doc.label
                direction = (
                    "correct_to_wrong"
                    if before and not after
                    else ("wrong_to_correct" if after and not before else "wrong_to_wrong")
                )
                flips.append(
                    {
                        "model_id": model_id,
                        "condition": condition,
                        "document_id": doc.id,
                        "gold": doc.label,
                        "original_prediction": old["prediction"],
                        "prediction": new["prediction"],
                        "direction": direction,
                        "text_snippet": doc.text[:900],
                        "removed_spans": spans[i],
                        "original_explanation": old,
                        "ablated_explanation": new,
                    }
                )
    return flips


def shortcut_summary(evaluations):
    result = {}
    for model_id in ("C0", "C1", "C2", "C3"):
        rows = evaluations[(model_id, "OO")]["predictions"]
        summary = {}
        for kind in ("source", "template"):
            if model_id == "C3":
                affected = [
                    r for r in rows if any(c[kind + "_overlap"] for c in r["top_convolution_spans"])
                ]
                channels = [c for r in rows for c in r["all_channels"] if c["contribution"] > 0]
                overlapping = [c for c in channels if c[kind + "_overlap"]]
                summary[kind] = {
                    "top_span_overlap_documents": len(affected),
                    "positive_channel_overlap_count": len(overlapping),
                    "positive_channel_count": len(channels),
                    "overlap_positive_contribution_share": sum(
                        c["contribution"] for c in overlapping
                    )
                    / sum(c["contribution"] for c in channels)
                    if channels
                    else 0,
                    "document_ids": [r["document_id"] for r in affected],
                }
            else:
                features = [
                    (r, f)
                    for r in rows
                    for f in r["top_contributing_features"]
                    if f.get(kind + "_occurrences", 0)
                ]
                ids = sorted({r["document_id"] for r, _ in features})
                summary[kind] = {
                    "top_feature_overlap_documents": len(ids),
                    "document_ids": ids,
                    "features": dict(Counter(f["term"] for _, f in features)),
                    "note": (
                        "Removed-region features may also be in body; "
                        "TF-IDF is not region-additive"
                    ),
                }
        result[model_id] = summary
    return result


def run(root):
    base = root / "outputs/exp3"
    pointer = read_json(base / "stage5_preregistration.json")
    directory = base / "stage5" / pointer["run_id"]
    if file_sha256(directory / "cleaning_freeze.json") != pointer["freeze_sha256"]:
        raise ValueError("Stage5 freeze seal changed")
    freeze = CleaningFreeze(directory)
    freeze.verify()
    install_stage5_guard(root, freeze)
    execution = directory / "execution.json"
    with execution.open("x", encoding="utf-8") as stream:
        json.dump(
            {"status": "started", "formal_test_evaluation_count": 0, "rerun_allowed": False}, stream
        )
    start = perf_counter()
    rules = read_json(directory / "cleaning_rules.json")
    refs, configs = frozen_references(root)
    if refs != read_json(directory / "frozen_config_references.json"):
        raise ValueError("Frozen parent references changed")
    data = Stage5Data(root, freeze)
    train = data.load("train")
    # This is the first Validation read in the Stage5 process, after verified freeze.
    val = data.load("val")
    write_json(
        directory / "validation_open_record.json",
        {
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": pointer["freeze_sha256"],
            "validation_sha256": file_sha256(root / "data/splits/thucnews_600/val.jsonl"),
            "document_ids": [d.id for d in val],
            "train_ids": [d.id for d in train],
        },
    )
    processor, tokenizer = processor_for(root, configs["stage1"])
    write_json(directory / "tokenizer_reference.json", tokenizer)
    train_views = [clean_view(d.text, rules, {"source", "template"}) for d in train]
    views = {
        c: [clean_view(d.text, rules, set(kinds)) for d in val] for c, kinds in CONDITIONS.items()
    }
    all_spans = [v["removed_spans"] for v in views["OA"]]
    tokens = {c: [tokenize_view(processor, v) for v in rows] for c, rows in views.items()}
    train_tokens = [tokenize_view(processor, v) for v in train_views]
    write_json(
        directory / "prevalence.json",
        {"train": prevalence(train, train_views), "validation": prevalence(val, views["OA"])},
    )
    write_json(
        directory / "validation_cleaning.json",
        [{"document_id": d.id, "gold": d.label, **v} for d, v in zip(val, views["OA"])],
    )
    original = base / "stage4" / STAGE4 / "points/ladder101_n420"
    manifest = read_json(base / "stage4" / STAGE4 / "artifact_manifest.json")["files"]
    original_files = [
        p
        for p in original.rglob("*")
        if p.is_file()
        and (
            p.suffix in {".npz", ".npy"}
            or p.name
            in {"model.json", "vocabulary.json", "representation.json", "observed_df.json"}
        )
    ]
    original_hashes = {}
    for path in original_files:
        relative = path.relative_to(base / "stage4" / STAGE4).as_posix()
        digest = file_sha256(path)
        if digest != manifest[relative]:
            raise ValueError("Original Stage4 model/representation hash changed")
        original_hashes[relative] = digest
    write_json(directory / "original_model_hashes.json", original_hashes)
    word = word_representation(
        configs["stage1"]["representation"], original / "word_representation"
    )
    char = CharacterChi2TFIDF.load(original / "char_representation")
    evaluations = {}
    original_models = {
        model_id: (C2LinearClassifier if model_id == "C2" else LinearTextClassifier).load(
            original / model_id
        )
        for model_id in ("C0", "C1", "C2")
    }
    for condition in CONDITIONS:
        word_matrix = word.transform([r["sections"] for r in tokens[condition]])
        char_matrix = char.transform([v["char_text"] for v in views[condition]])
        for model_id in ("C0", "C1", "C2"):
            model = original_models[model_id]
            rep, matrix = (char, char_matrix) if model_id == "C2" else (word, word_matrix)
            reference = str(original / model_id)
            evaluations[(model_id, condition)] = evaluate_linear(
                model,
                rep,
                matrix,
                val,
                tokens[condition],
                views[condition],
                all_spans,
                condition,
                reference,
            )
    vocab = WordSequenceVocabulary.load(original / "sequence_vocabulary")
    encoded_directory = directory / "encodings"
    encoded_directory.mkdir()
    for condition in CONDITIONS:
        save_encoding(encoded_directory / condition, encode_sections(vocab, tokens[condition]), val)
    neural_job(
        root,
        directory,
        {
            "root": str(root),
            "freeze": str(directory),
            "mode": "original",
            "vocabulary": str(original / "sequence_vocabulary"),
            "model": str(original / "C3/model"),
            "output": str(directory / "original_C3"),
            "evaluations": {
                c: {"encoding": str(encoded_directory / c), "removed_spans": all_spans}
                for c in CONDITIONS
            },
        },
    )

    def adopt_neural(condition, path):
        payload = read_json(path)
        rows = payload["predictions"]
        for row, doc in zip(rows, val):
            row.update(
                gold=doc.label, correct=row["prediction"] == doc.label, text_snippet=doc.text[:600]
            )
        evaluations[("C3", condition)] = {
            "model_id": "C3",
            "condition": condition,
            "predictions": rows,
            "metrics": classification_metrics(
                [d.label for d in val], [r["prediction"] for r in rows]
            ),
            "model_reference": payload["model_reference"],
            "inference_seconds": payload["inference_seconds"],
        }

    for condition in CONDITIONS:
        adopt_neural(condition, directory / f"original_C3/{condition}.json")
    # Abort before AA if loading or boundary adapters changed the original predictions.
    sanity = {}
    for model_id in ("C0", "C1", "C2", "C3"):
        expected = read_json(original / f"{model_id}_result.json")
        actual = evaluations[(model_id, "OO")]
        sanity[model_id] = [r["prediction"] for r in actual["predictions"]] == expected[
            "validation_predictions"
        ] and actual["metrics"] == expected["validation_metrics"]
    write_json(directory / "OO_sanity.json", sanity)
    if not all(sanity.values()):
        raise ValueError("OO differs from Stage4 420; stop before clean retraining")
    print("OO predictions and metrics match Stage4 for all four models", flush=True)
    clean = directory / "AA"
    clean.mkdir()
    fit_start = perf_counter()
    clean_word = word_representation(configs["stage1"]["representation"])
    train_matrix = clean_word.fit_transform([r["sections"] for r in train_tokens])
    val_matrix = clean_word.transform([r["sections"] for r in tokens["OA"]])
    clean_word.save(clean / "word_representation")
    for model_id in ("C0", "C1"):
        model = LinearTextClassifier(model_id, 0.5 if model_id == "C0" else 2.0)
        model.fit(train_matrix, [d.label for d in train])
        model.save(clean / model_id)
        evaluations[(model_id, "AA")] = evaluate_linear(
            model,
            clean_word,
            val_matrix,
            val,
            tokens["OA"],
            views["OA"],
            all_spans,
            "AA",
            str(clean / model_id),
        )
        evaluations[(model_id, "AA")]["fit_seconds"] = model.fit_seconds
    clean_char = CharacterChi2TFIDF(10000)
    char_train = clean_char.fit_transform(
        [v["char_text"] for v in train_views], [d.label for d in train]
    )
    char_val = clean_char.transform([v["char_text"] for v in views["OA"]])
    clean_char.save(clean / "char_representation")
    c2 = C2LinearClassifier(parameter=2.0).fit(char_train, [d.label for d in train])
    c2.save(clean / "C2")
    evaluations[("C2", "AA")] = evaluate_linear(
        c2, clean_char, char_val, val, tokens["OA"], views["OA"], all_spans, "AA", str(clean / "C2")
    )
    evaluations[("C2", "AA")]["fit_seconds"] = c2.fit_seconds
    sparse_fit = perf_counter() - fit_start
    clean_vocab = WordSequenceVocabulary(768, 2).fit([r["tokens"] for r in train_tokens])
    clean_vocab.save(clean / "sequence_vocabulary")
    save_encoding(clean / "train", encode_sections(clean_vocab, train_tokens), train)
    save_encoding(clean / "val", encode_sections(clean_vocab, tokens["OA"]), val)
    neural_job(
        root,
        directory,
        {
            "root": str(root),
            "freeze": str(directory),
            "mode": "AA",
            "train": str(clean / "train"),
            "vocabulary": str(clean / "sequence_vocabulary"),
            "output": str(clean / "C3"),
            "evaluations": {"AA": {"encoding": str(clean / "val"), "removed_spans": all_spans}},
        },
    )
    adopt_neural("AA", clean / "C3/AA.json")
    result_rows = []
    for model_id in ("C0", "C1", "C2", "C3"):
        for condition in (*CONDITIONS, "AA"):
            payload = evaluations[(model_id, condition)]
            payload["delta_macro_f1"] = (
                payload["metrics"]["macro_f1"]
                - evaluations[(model_id, "OO")]["metrics"]["macro_f1"]
            )
            write_json(
                directory / f"{model_id}_{condition}_predictions.json", payload["predictions"]
            )
            result_rows.append({k: v for k, v in payload.items() if k != "predictions"})
    flips = build_flips(evaluations, val, all_spans)
    write_json(directory / "results.json", result_rows)
    write_json(directory / "flips.json", flips)
    write_json(directory / "shortcut_summary.json", shortcut_summary(evaluations))
    freeze.verify()
    write_json(
        directory / "runtime.json",
        {"wall_seconds": perf_counter() - start, "AA_sparse_fit_seconds": sparse_fit},
    )
    write_json(
        directory / "run.json",
        {
            "run_id": directory.name,
            "analysis_type": PROTOCOL["analysis_type"],
            "conditions": list(CONDITIONS) + ["AA"],
            "formal_test_evaluation_count": 0,
            "test_accessed": False,
            "model_selection": False,
            "completed_evaluations": 20,
            "AA_fits": 4,
            "OO_sanity": sanity,
            "parent_stage4_run": STAGE4,
            "cleaning_freeze_sha256": pointer["freeze_sha256"],
        },
    )
    write_json(
        execution,
        {
            "status": "complete",
            "AA_fits": 4,
            "formal_test_evaluation_count": 0,
            "test_accessed": False,
            "rerun_allowed": False,
        },
    )
    write_json(directory / "artifact_manifest.json", artifact_manifest(directory))
    atomic_json(
        base / "stage5_latest.json",
        {"run_id": directory.name, "analysis_type": PROTOCOL["analysis_type"]},
    )
    print(
        json.dumps({"run_id": directory.name, "completed_evaluations": 20, "flips": len(flips)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.preregister:
        preregister(args.root.resolve(), args.rules, args.audit)
    elif args.formal:
        run(args.root.resolve())
    else:
        parser.error("Explicit preregistration or formal mode required")
