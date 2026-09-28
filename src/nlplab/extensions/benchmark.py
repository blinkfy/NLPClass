"""独立拓展 Run：原始 raw 输入、既定官方 scorer 与跨度 evaluator 交叉验证。"""

import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from nlplab.core.exceptions import DataValidationError, NoValidPathError
from nlplab.core.io import write_json
from nlplab.core.schema import SequenceSample
from nlplab.data.corpus import read_jsonl, write_jsonl
from nlplab.data.dataset import load_samples
from nlplab.evaluation.errors import error_record
from nlplab.evaluation.segmentation import evaluate_segmentation, prf
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.tags import STATES


def fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def context(root: Path) -> dict:
    paths = list((root / "outputs/exp1/benchmarks").glob("*/benchmark_context.json"))
    if len(paths) != 1:
        raise DataValidationError("需要唯一的已冻结 MSR benchmark_context.json")
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    data["benchmark"] = paths[0].parent
    data["development"] = root / "data/splits/msr_development_20260926_124541_c0c7fa3b_attempt2"
    return data


def project_metrics(gold: list[list[str]], predictions: list[list[str] | None], words: set) -> dict:
    """沿用冻结 benchmark 的 abstention：无预测、零正确、Gold 保留在分母。"""
    success = [i for i, p in enumerate(predictions) if p is not None]
    baseline = (
        evaluate_segmentation([gold[i] for i in success], [predictions[i] for i in success], words)
        if success
        else None
    )
    gold_count = sum(map(len, gold))
    chars = sum(len(w) for row in gold for w in row)
    oov = sum(w not in words for row in gold for w in row)
    iv = gold_count - oov
    matrix = [[0] * 5 for _ in STATES]
    if baseline:
        for i, row in enumerate(baseline["confusion_matrix"]):
            matrix[i][:4] = row
    for row, prediction in zip(gold, predictions):
        if prediction is None:
            for tag in encode(row):
                matrix[STATES.index(tag)][4] += 1
    return {
        **prf(
            baseline["tp"] if baseline else 0, baseline["predicted"] if baseline else 0, gold_count
        ),
        "tag_accuracy": sum(matrix[i][i] for i in range(4)) / chars,
        "oov_recall": (baseline["oov_recall"] or 0) * baseline["oov_support"] / oov
        if baseline and oov
        else (0.0 if oov else None),
        "iv_recall": (baseline["iv_recall"] or 0) * baseline["iv_support"] / iv
        if baseline and iv
        else (0.0 if iv else None),
        "oov_support": oov,
        "iv_support": iv,
        "sentence_count": len(gold),
        "character_count": chars,
        "gold_labels": list(STATES),
        "prediction_labels": [*STATES, "ABSTAIN"],
        "confusion_matrix": matrix,
        "successful_only": baseline,
    }


def official_score(source: Path, protocol: Path, run: Path, perl: str, failed: bool) -> dict:
    """原 scorer（零预测时采用已核验的除零 guard）；保存原始 diff 与调用指纹。"""
    scorer = protocol / ("score_abstention_guard.pl" if failed else "score")
    dictionary = source / "gold/msr_training_words.utf8"
    truth = source / "gold/msr_test_gold.utf8"
    prediction = run / "prediction.utf8"
    command = [perl, str(scorer), str(dictionary), str(truth), str(prediction)]
    started = perf_counter()
    with (
        (run / "official.stdout.txt").open("xb") as stdout,
        (run / "official.stderr.txt").open("xb") as stderr,
    ):
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    write_json(
        run / "official.invocation.json",
        {
            "command": command,
            "exit_code": process.returncode,
            "seconds": perf_counter() - started,
            "scorer_sha256": fingerprint(scorer),
            "gold_sha256": fingerprint(truth),
            "prediction_sha256": fingerprint(prediction),
            "abstention_guard": failed,
        },
    )
    if process.returncode:
        raise DataValidationError(f"官方 scorer 失败，见 {run / 'official.stderr.txt'}")
    # 与冻结 scorer 相同的 diff 计数规则，避免仅保留三位小数造成比较损失。
    dictionary_bytes = set(dictionary.read_bytes().split())
    summary, in_body = {}, False
    oov = iv = oov_missed = iv_missed = recall_missed = precision_missed = 0
    with (run / "official.stdout.txt").open("rb") as stream:
        for row in stream:
            if row.startswith(b"--") and re.search(rb"----(\d+)\s*$", row):
                in_body = True
                continue
            if row.startswith(b"INSERTIONS:"):
                in_body = False
            if in_body:
                left = re.match(rb"^([^\s]+)\s", row)
                if left:
                    unseen = left.group(1) not in dictionary_bytes
                    missed = bool(re.search(rb"^[^\s]+\s.*\s[\|><]\s", row))
                    oov += int(unseen)
                    iv += int(not unseen)
                    oov_missed += int(unseen and missed)
                    iv_missed += int(not unseen and missed)
                    recall_missed += int(missed)
                precision_missed += int(bool(re.search(rb"\s[\|><]\s.*[^\s]$", row)))
            if row.startswith(b"=== ") and b":\t" in row:
                key, value = row[4:].split(b":\t", 1)
                summary[key.decode()] = value.strip().decode()
    total_gold = int(summary["TOTAL TRUE WORD COUNT"])
    total_prediction = int(summary["TOTAL TEST WORD COUNT"])
    assert oov + iv == total_gold
    p = (total_prediction - precision_missed) / total_prediction if total_prediction else 0
    r = (total_gold - recall_missed) / total_gold
    metrics = {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0,
        "oov_recall": 1 - oov_missed / oov if oov else None,
        "iv_recall": 1 - iv_missed / iv if iv else None,
        "support": total_gold,
        "predicted": total_prediction,
        "printed_summary": summary,
    }
    for key, printed in [
        ("precision", "TOTAL TEST WORDS PRECISION"),
        ("recall", "TOTAL TRUE WORDS RECALL"),
        ("f1", "F MEASURE"),
        ("oov_recall", "OOV Recall Rate"),
        ("iv_recall", "IV Recall Rate"),
    ]:
        assert abs(metrics[key] - float(summary[printed])) <= 0.000501
    return metrics


def run_model(
    root: Path,
    name: str,
    model,
    config: dict,
    *,
    phase: str = "final",
    training_result: dict | None = None,
) -> dict:
    """模型训练完且配置确定后调用；先一次推理并落盘，然后才载入 test Gold。"""
    ctx = context(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run = root / "outputs/exp1/extensions" / run_id
    run.mkdir(parents=True, exist_ok=False)
    model_path = root / "artifacts/models" / ("msr-extension-" + run_id)
    model.save(model_path)
    write_json(run / "config.json", config)
    if training_result is not None:
        write_json(run / "training_history.json", training_result)
    source = Path(ctx["source_path"])
    train_path = (
        Path(ctx["standard_split_path"]) / "train.jsonl"
        if phase == "final"
        else (ctx["development"] / "train.jsonl")
    )
    if phase == "final":
        raw_path = source / "testing/msr_test.utf8"
        raw = raw_path.read_bytes().decode("utf-8").split("\r\n")
        assert raw.pop() == "" and len(raw) == 3985
    else:
        raw_path = ctx["development"] / "dev.jsonl"
        raw = [s.text for s in load_samples(raw_path)]
    fingerprints = {
        "training_sha256": fingerprint(train_path),
        "input_sha256": fingerprint(raw_path),
        "protocol": "standard",
        "raw_input_not_derived_from_gold": phase == "final",
    }
    write_json(run / "data_fingerprints.json", fingerprints)
    write_json(
        run / "implementation_fingerprints.json",
        {
            p.relative_to(root).as_posix(): fingerprint(p)
            for p in (root / "src/nlplab/extensions").glob("*.py")
        },
    )
    records, submissions, predictions = [], [], []
    started = perf_counter()
    for i, text in enumerate(raw, 1):
        record = {
            "sample_id": f"MSR_{phase}_{i:06}",
            "text": text,
            "input_sha256": sha256(text.encode()).hexdigest(),
        }
        record["unseen_char_positions"] = (
            [
                {"position": j, "character": c}
                for j, c in enumerate(text)
                if not c.isspace() and c not in model.character_index
            ]
            if hasattr(model, "character_index")
            else []
        )
        try:
            result = model.segment(text)
            assert "".join(result.tokens) == text and len(result.tags) == len(text)
        except NoValidPathError as error:
            record.update(status="NoValidViterbiPath", error=str(error))
            predictions.append(None)
            submissions.append("")
        else:
            record.update(status="decoded", segmentation=asdict(result))
            predictions.append([w for w in result.tokens if not w.isspace()])
            submissions.append(" ".join(result.tokens))
        records.append(record)
    seconds = perf_counter() - started
    write_jsonl(run / "predictions.jsonl", records)
    (run / "prediction.utf8").write_bytes(("\r\n".join(submissions) + "\r\n").encode("utf-8"))
    # Gold 只在预测文件冻结之后读取。
    gold = (
        [s.gold_tokens for s in load_samples(raw_path)]
        if phase != "final"
        else [
            r["gold_tokens"]
            for r in read_jsonl(
                ctx["benchmark"] / "project_evaluator_gold_alignment.jsonl", lambda r: r
            )
        ]
    )
    assert len(gold) == len(records)
    words = model.words if hasattr(model, "words") else model.training_words
    project = project_metrics(gold, predictions, words)
    failed = sum(p is None for p in predictions)
    official = (
        official_score(source, ctx["benchmark"] / "protocol", run, ctx["perl_path"], bool(failed))
        if phase == "final"
        else None
    )
    metrics = {
        "official_scorer": official,
        "project_evaluator": project,
        "coverage": 1 - failed / len(records),
        "successful_decodes": len(records) - failed,
        "NoValidViterbiPath": failed,
        "unseen_character_related_failures": sum(
            r["status"] != "decoded" and bool(r["unseen_char_positions"]) for r in records
        ),
        "inference_seconds": seconds,
    }
    if official is not None:
        write_json(
            run / "scorer_crosscheck.json",
            {
                "primary": "SIGHAN official scorer",
                "secondary": "project exact-span evaluator with frozen raw-coordinate Gold",
                "project_minus_official": {
                    key: project[key] - official[key]
                    for key in ["precision", "recall", "f1", "oov_recall", "iv_recall"]
                    if project[key] is not None and official[key] is not None
                },
                "protocol_note": "Official raw/Gold have 16 boundary differences; "
                "reuse frozen protocol and save both outputs without changing algorithms.",
            },
        )
    errors = []
    for row, tokens, record in zip(gold, predictions, records):
        sample = SequenceSample(record["sample_id"], record["sample_id"], "".join(row), row)
        if tokens is None:
            errors.append({**record, "gold_tokens": row, "category": "NoValidViterbiPath"})
        else:
            error = error_record(sample, tokens, words)
            if error:
                errors.append(error)
    write_json(run / "metrics.json", metrics)
    write_jsonl(run / "errors.jsonl", errors)
    write_json(run / "parameters.json", model.parameters())
    if hasattr(model, "trace"):
        try:
            trace = model.trace("中国石油大学（华东）")
        except NoValidPathError as error:
            trace = {"status": "NoValidViterbiPath", "error": str(error)}
        write_json(run / "example_trace.json", trace)
    info = {
        "run_id": run_id,
        "model": name,
        "phase": phase,
        "run_type": "extension-cws",
        "dataset": "SIGHAN Bakeoff 2005 MSR " + phase,
        "model_path": str(model_path.resolve()),
        "evaluation_count": len(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_demo": False,
        "test_inference_passes": 1 if phase == "final" else 0,
        "read_only": True,
        "config": config,
    }
    write_json(run / "run.json", info)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "model": name,
                "phase": phase,
                "metrics": official
                or {
                    k: project[k]
                    for k in [
                        "precision",
                        "recall",
                        "f1",
                        "oov_recall",
                        "iv_recall",
                        "tag_accuracy",
                    ]
                },
                "coverage": metrics["coverage"],
                "NoValidViterbiPath": failed,
                "run_path": str(run),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return info
