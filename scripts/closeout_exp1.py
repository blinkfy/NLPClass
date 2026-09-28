"""实验一报告资产：只读 frozen files，禁止模型调用及子进程。

先创建 closeout_before.json；此脚本仅构建报告，不创建 Run，不载入模型。
"""

# ruff: noqa: E501 -- 报告 Markdown/CSV 字段原文与真实代码摘录保持完整。

from __future__ import annotations

import ast
import csv
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(json.loads((ROOT / "artifacts/cache/exp1_closeout_current.json").read_text())["output"])
FLAGS = dict(training_executed=False, official_test_inference_executed=False, scorer_executed=False)
ASSETS: dict[str, dict] = {}
READS: dict[str, str] = {}


def guard(event, args):
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.spawn"}:
        raise RuntimeError(f"closeout 禁止执行: {event}")
    if event == "open":
        path, mode, flags = args
        writing = isinstance(mode, str) and any(c in mode for c in "wa+")
        writing |= bool(flags & 3) if isinstance(flags, int) else False
        if writing and isinstance(path, (str, bytes)):
            target = Path(path).resolve()
            if not (
                target.is_relative_to(OUT)
                or target.is_relative_to(ROOT / "docs")
                or target == ROOT / "README.md"
            ):
                raise RuntimeError(f"closeout 禁止写入: {target}")


def profile(frame, event, arg):
    if event == "call" and "/src/nlplab/" in frame.f_code.co_filename.replace("\\", "/"):
        if frame.f_code.co_name in {
            "fit",
            "predict",
            "segment",
            "recognize",
            "viterbi",
            "viterbi2",
            "_decode",
            "_decode_batch",
            "emissions",
            "official_score",
            "run_model",
        }:
            raise RuntimeError(f"closeout 禁止调用模型/评分: {frame.f_code.co_name}")


sys.addaudithook(guard)
sys.setprofile(profile)

from nlplab.evaluation.segmentation import evaluate_segmentation, word_spans  # noqa: E402


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    READS[path.relative_to(ROOT).as_posix()] = sha256(data).hexdigest()
    return data.decode("utf-8")


def load(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line]


def emit(name, value, sources, runs=(), derived=True):
    text = (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / name).write_text(text, encoding="utf-8")
    source_hashes = {}
    for source in sources:
        source = Path(source)
        if not source.is_absolute():
            source = ROOT / source
        key = source.relative_to(ROOT).as_posix()
        if key not in READS:
            read(source)
        source_hashes[key] = READS[key]
    ASSETS[name] = {
        "asset_name": name,
        "source_run_id": list(runs),
        "source_files": list(source_hashes),
        "source_sha256": source_hashes,
        "generated_sha256": sha256(text.encode()).hexdigest(),
        "derived": derived,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **FLAGS,
    }


def jsonl(name, value, sources, runs=()):
    emit(
        name, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in value), sources, runs
    )


def table_csv(name, values, sources, runs=()):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
    emit(name, stream.getvalue(), sources, runs)


def update_document(name, content):
    path = ROOT / name
    old = path.read_bytes().decode("utf-8") if path.exists() else ""
    newline = "\r\n" if "\r\n" in old else "\n"
    path.write_bytes(content.replace("\r\n", "\n").replace("\n", newline).encode("utf-8"))


def verify_before():
    snapshot = load(OUT / "closeout_before.json")
    supplemental = load(OUT / "closeout_before_supplement.json")
    expected = snapshot["files"] | supplemental["files"]
    changed = [
        name
        for name, digest in expected.items()
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError(f"冻结文件发生变化: {changed}")
    return snapshot, expected


BEGIN, PROTECTED = verify_before()
BENCH = "outputs/exp1/benchmarks/20260926_130612_32a5e168"
NER = "outputs/exp1/course_samples/20260926_152118_34a953fb"
NER_ID = Path(NER).name
ALIGN = f"{BENCH}/project_evaluator_gold_alignment.jsonl"
GOLD = [r["gold_tokens"] for r in rows(ALIGN)]
CTX = load(f"{BENCH}/benchmark_context.json")
MSRA = load("data/course/exp1_msra_200_manifest.json")
THU = load("data/course/thucnews_600_manifest.json")
SPLIT = load("data/splits/thucnews_600/manifest.json")
REFERENCE = load("outputs/exp1/extensions/comparison_20260926_190001.json")["course_reference"]
METHODS = [
    ("longest", "Longest Matching", "20260926_160759_676a087e", "dictionary/rule"),
    ("bigram", "Bigram", "20260926_161208_5bfe382f", "statistical language model"),
    (
        "hmm1-mle",
        "HMM-1 MLE",
        "20260926_130612_32a5e168_mle",
        "first-order generative sequence model",
    ),
    (
        "hmm1-smooth",
        "HMM-1 Smooth",
        "20260926_130612_32a5e168_smooth",
        "first-order generative sequence model",
    ),
    ("hmm2-mle", "HMM-2 MLE", "20260926_161715_d8132887", "second-order generative sequence model"),
    (
        "hmm2-smooth",
        "HMM-2 Smooth",
        "20260926_161715_5a7e4a3c",
        "second-order generative sequence model",
    ),
    (
        "bilstm-crf",
        "BiLSTM-CRF",
        "20260926_185552_0ffff5d1",
        "neural discriminative sequence model",
    ),
]
DATA, COMPARISON, metric_sources = {}, [], []
for model_id, model, run_id, implementation in METHODS:
    run = (
        f"{BENCH}/{run_id}" if model_id.startswith("hmm1") else f"outputs/exp1/extensions/{run_id}"
    )
    info, metrics, predictions = (
        load(f"{run}/run.json"),
        load(f"{run}/metrics.json"),
        rows(f"{run}/predictions.jsonl"),
    )
    assert len(predictions) == 3985 and not info["is_demo"] and info["test_inference_passes"] == 1
    official = metrics["official_scorer"]
    official = official.get("full", official)
    project = metrics["project_evaluator"]
    project = project.get("full", project)
    config = load(f"{run}/config.json") if (ROOT / run / "config.json").exists() else None
    metadata = (
        load(Path(info["model_path"]) / "metadata.json")
        if (Path(info["model_path"]) / "metadata.json").exists()
        else None
    )
    parameters = load(f"{run}/parameters.json")
    scored_tokens = [
        r.get(
            "score_tokens",
            [t for t in r.get("segmentation", {}).get("tokens", []) if not t.isspace()],
        )
        for r in predictions
    ]
    DATA[model_id] = {
        "run": run,
        "info": info,
        "metrics": metrics,
        "project": project,
        "predictions": predictions,
        "tokens": scored_tokens,
        "config": config,
        "metadata": metadata,
        "parameters": parameters,
    }
    metric_sources += [f"{run}/metrics.json", f"{run}/run.json"]
    train_seconds = config.get("training_seconds") if config else None
    inference_seconds = metrics.get("inference_seconds")
    if model_id == "bilstm-crf":
        train_seconds = metadata["training_metadata"]["training_seconds"]
    COMPARISON.append(
        {
            "model": model,
            "model_id": model_id,
            **{k: official[k] for k in ("precision", "recall", "f1", "oov_recall", "iv_recall")},
            "tag_accuracy": project["tag_accuracy"],
            "coverage": metrics.get("coverage", metrics.get("inference", {}).get("coverage")),
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "implementation_type": implementation,
            "final_run_id": run_id,
            "dataset": "SIGHAN Bakeoff 2005 MSR official raw test",
            "tag_accuracy_derived_from_saved_predictions": model_id in {"longest", "bigram"},
            "training_and_inference_seconds": info.get("training_and_inference_seconds"),
            "timing_note": "Only combined duration was saved; separate times unavailable"
            if model_id.startswith("hmm1")
            else "saved training/inference durations",
        }
    )

TRAIN_WORDS = set(DATA["hmm1-smooth"]["metadata"]["training_words"])
derived = {}
for model_id in ("longest", "bigram"):
    result = evaluate_segmentation(GOLD, DATA[model_id]["tokens"], TRAIN_WORDS)
    assert result["tag_accuracy"] == DATA[model_id]["project"]["tag_accuracy"]
    derived[model_id] = {
        "tag_accuracy": result["tag_accuracy"],
        "character_count": result["character_count"],
        "confusion_matrix": result["confusion_matrix"],
        "derived_from_saved_predictions": True,
        **FLAGS,
    }
emit(
    "derived_tag_accuracy.json",
    derived,
    [ALIGN, *[f"{DATA[m]['run']}/predictions.jsonl" for m in derived]],
    [DATA[m]["info"]["run_id"] for m in derived],
)
all_runs = [item[2] for item in METHODS]
comparison = {
    "metric_units": "ratios in [0,1]",
    "primary_word_metrics": "frozen official scorer",
    "tag_accuracy_source": "project raw-coordinate Gold; Longest/Bigram additionally verified from saved predictions",
    "rows": COMPARISON,
    "derived_from_saved_predictions": True,
    **FLAGS,
}
emit(
    "cws_seven_model_comparison.json",
    comparison,
    metric_sources + [ALIGN, *[f"{DATA[m]['run']}/predictions.jsonl" for m in derived]],
    all_runs,
)
emit(
    "metrics_cws.json",
    {
        "official_and_project": {m: d["metrics"] for m, d in DATA.items()},
        "derived_tag_accuracy": derived,
        **FLAGS,
    },
    metric_sources,
    all_runs,
    False,
)
table_csv("cws_seven_model_comparison.csv", COMPARISON, metric_sources, all_runs)
reference_rows = []
for item in COMPARISON:
    ref = REFERENCE.get(item["model_id"])
    if ref:
        for index, metric in enumerate(("precision", "recall", "f1", "oov_recall", "iv_recall")):
            reference_rows.append(
                {
                    "model": item["model"],
                    "metric": metric,
                    "course_percent": ref[index],
                    "project_percent": item[metric] * 100,
                    "delta_pp": item[metric] * 100 - ref[index],
                    "reference_source": "user-provided Chapter 3 Table 3-3 / Chapter 4 4.7.3",
                    "benchmark_source": "frozen MSR official scorer",
                    "final_run_id": item["final_run_id"],
                }
            )
table_csv(
    "course_reference_comparison.csv",
    reference_rows,
    ["configs/extension_course_references.yaml", *metric_sources],
    all_runs,
)

for name, prefix in (("hmm1_parameters.json", "hmm1"), ("hmm2_parameters.json", "hmm2")):
    variants = {}
    for variant in ("mle", "smooth"):
        data = DATA[f"{prefix}-{variant}"]
        p = data["parameters"]
        variants[variant] = {
            "states": p["states"],
            "pi": p["start"],
            "A": p["transition"] if prefix == "hmm1" else None,
            "emission_topk": p["emission_topk"],
            "top_k": 20,
            "vocabulary_size": p["training"]["vocabulary_size"],
            "training_sentence_count": p["training"]["sentence_count"],
            "smoothing": p["config"],
            "unknown_probabilities": p["unknown_probabilities"],
            "unk_policy": "explicit shared UNK emission bucket; MLE zero / add-alpha=0.1 smoothed",
            "source_run_id": data["info"]["run_id"],
        }
        if prefix == "hmm2":
            variants[variant].update(
                initial_pair_transition=p["initial_transition"],
                tensor_shape=[4, 4, 4],
                legal_triple_count=sum(
                    int(v) for plane in p["legal_triples"] for row in plane for v in row
                ),
                selected_tensor_slices={
                    "previous_previous=B": p["transition_tensor"][0],
                    "previous_previous=E": p["transition_tensor"][2],
                },
                source_url=p["source_url"],
                source_revision=p["source_revision"],
            )
    emit(
        name,
        variants,
        [f"{DATA[prefix + '-' + v]['run']}/parameters.json" for v in ("mle", "smooth")],
        [DATA[prefix + "-" + v]["info"]["run_id"] for v in ("mle", "smooth")],
        False,
    )
emit(
    "longest_statistics.json",
    {
        **DATA["longest"]["parameters"],
        "fallback_strategy": "unseen character -> one-character token; original whitespace preserved",
    },
    [f"{DATA['longest']['run']}/parameters.json"],
    [DATA["longest"]["info"]["run_id"]],
    False,
)
emit(
    "bigram_statistics.json",
    {
        **DATA["bigram"]["parameters"],
        "vocabulary_size": DATA["bigram"]["parameters"]["dictionary_size"],
        "unigram_token_count": DATA["bigram"]["parameters"]["word_count"],
        "unigram_types": DATA["bigram"]["parameters"]["dictionary_size"],
        "bigram_types": DATA["bigram"]["parameters"]["bigram_types"],
        "markers": ["<BOS>", "<EOS>"],
        "formula": "0.9 * P_MLE(w|previous) + 0.1 * P_add-0.1(w)",
        "unobserved_context": "full smoothed unigram backoff",
        "report_wording": "依据课件的二元语法与低阶平滑思想，本项目采用固定插值平滑实现。",
    },
    [f"{DATA['bigram']['run']}/parameters.json", "src/nlplab/extensions/dictionary.py"],
    [DATA["bigram"]["info"]["run_id"]],
)
NEURAL = DATA["bilstm-crf"]
HISTORY = load(f"{NEURAL['run']}/training_history.json")
DEV = HISTORY["selected_from_development"]
BUDGET = "在预先固定的 10 epoch 训练预算下，development 最优结果出现在第 10 轮，说明模型仍可能具有进一步训练空间。但 official test 已在配置冻结后完成。为避免 test 后继续调整训练预算，本实验保留原 10 epoch 设置，不再增加训练轮数或重新评价。"
emit(
    "bilstm_architecture.json",
    {
        **NEURAL["parameters"],
        "library_application": "PyTorch Embedding / bidirectional LSTM / Linear; CRF forward, partition and Viterbi from pytorch-crf 0.7.2 (MIT)",
        "hidden_size_per_direction": 128,
        "bidirectional": True,
        "linear_input": 256,
        "linear_output": 4,
    },
    [f"{NEURAL['run']}/parameters.json", "src/nlplab/extensions/bilstm_crf.py"],
    [NEURAL["info"]["run_id"]],
    False,
)
emit(
    "bilstm_training_summary.json",
    {
        "configuration": NEURAL["metadata"]["config"],
        "final_metadata": NEURAL["metadata"]["training_metadata"],
        "development_best_epoch": DEV["best_epoch"],
        "development_best_f1": DEV["best_dev_f1"],
        "final_fixed_epoch": HISTORY["best_epoch"],
        "max_epochs": 10,
        "early_stopping_triggered": False,
        "stopping_reason": "max_epochs budget reached",
        "budget_statement": BUDGET,
        "final_history": HISTORY["history"],
        "development_history": DEV["history"],
        "inference_seconds": NEURAL["metrics"]["inference_seconds"],
        **FLAGS,
    },
    [
        f"{NEURAL['run']}/training_history.json",
        Path(NEURAL["info"]["model_path"]) / "metadata.json",
        f"{NEURAL['run']}/metrics.json",
    ],
    [NEURAL["info"]["run_id"]],
)


def span_words(tokens):
    position, result = 0, []
    for token in tokens:
        result.append(
            {
                "start": position,
                "end": position + len(token),
                "text": token,
                "oov": token not in TRAIN_WORDS,
            }
        )
        position += len(token)
    return result


cws_cases, case_sources = [], [ALIGN]
for model_id in ("longest", "bigram", "hmm1-smooth", "hmm2-smooth", "bilstm-crf"):
    data = DATA[model_id]
    saved_errors = rows(f"{data['run']}/errors.jsonl")
    by_id = {e["sample_id"]: e for e in saved_errors}
    candidates = []
    for i, record in enumerate(data["predictions"]):
        tokens = data["tokens"][i]
        if (
            record["status"] != "decoded"
            or tokens == GOLD[i]
            or any(c.isspace() for c in record["text"])
        ):
            continue
        missing = word_spans(GOLD[i]) - word_spans(tokens)
        extra = word_spans(tokens) - word_spans(GOLD[i])
        affected = [w for w in span_words(GOLD[i]) if (w["start"], w["end"]) in missing]
        category = by_id.get(record["sample_id"], {}).get("category", "Other")
        category = {
            "Number/date": "number/date",
            "English mixed": "English mixed",
            "Proper noun": "proper noun",
            "Ambiguity": "ambiguity",
            "Other": "boundary error",
        }.get(category, category)
        candidates.append(
            {
                "sample_id": record["sample_id"],
                "raw_line": i + 1,
                "raw_text": record["text"],
                "gold_segmentation": GOLD[i],
                "prediction": tokens,
                "error_spans": {
                    "missing_gold_spans": sorted(missing),
                    "extra_predicted_spans": sorted(extra),
                },
                "oov_iv": affected,
                "model": model_id,
                "source_run_id": data["info"]["run_id"],
                "heuristic_category": category,
                "category_source": "saved heuristic; Other normalized to observable boundary error",
                "factual_error_description": f"缺失 {len(missing)} 个 Gold 词跨度，新增 {len(extra)} 个预测词跨度；错误 Gold 词中 {sum(w['oov'] for w in affected)} 个 OOV、{sum(not w['oov'] for w in affected)} 个 IV。",
                "derived_from_saved_predictions": True,
            }
        )
    chosen = []
    categories = set()
    for case in sorted(candidates, key=lambda c: (len(c["raw_text"]), c["raw_line"])):
        if case["heuristic_category"] not in categories:
            chosen.append(case)
            categories.add(case["heuristic_category"])
        if len(chosen) == 5:
            break
    for case in sorted(candidates, key=lambda c: (len(c["raw_text"]), c["raw_line"])):
        if len(chosen) >= 5:
            break
        if case not in chosen:
            chosen.append(case)
    assert len(chosen) == 5
    cws_cases.extend(chosen)
    case_sources += [f"{data['run']}/errors.jsonl", f"{data['run']}/predictions.jsonl"]
jsonl(
    "cws_error_examples.jsonl",
    cws_cases,
    case_sources,
    [
        DATA[m]["info"]["run_id"]
        for m in ("longest", "bigram", "hmm1-smooth", "hmm2-smooth", "bilstm-crf")
    ],
)

NER_METRICS, NER_ROWS = load(f"{NER}/metrics.json"), rows(f"{NER}/predictions.jsonl")
assert len(NER_ROWS) == 200 and NER_METRICS["segmentation"] is None
emit("metrics_ner.json", NER_METRICS, [f"{NER}/metrics.json"], [NER_ID], False)


def entity_key(entity):
    return entity["start"], entity["end"], entity["type"]


def ner_case(record):
    gs, ps = (
        {entity_key(e) for e in record["ner_gold"]},
        {entity_key(e) for e in record["ner_prediction"]},
    )
    gold = [e for e in record["ner_gold"] if entity_key(e) not in ps]
    pred = [e for e in record["ner_prediction"] if entity_key(e) not in gs]
    categories = set()
    if gold:
        categories.add("false negative")
    if pred:
        categories.add("false positive")
    for g in gold:
        for p in pred:
            if g["start"] == p["start"] and g["end"] == p["end"] and g["type"] != p["type"]:
                categories.add("type mismatch")
            if g["type"] == p["type"] and g["start"] <= p["start"] and p["end"] <= g["end"]:
                categories.add("boundary too short")
            if g["type"] == p["type"] and p["start"] <= g["start"] and g["end"] <= p["end"]:
                categories.add("boundary too long")
    for entity in record["ner_gold"] + record["ner_prediction"]:
        assert record["text"][entity["start"] : entity["end"]] == entity["text"]
    return {
        "sample_id": record["sample_id"],
        "raw_text": record["text"],
        "gold_entities": record["ner_gold"],
        "predicted_entities": record["ner_prediction"],
        "differing_spans": {"false_negative": gold, "false_positive": pred},
        "error_categories": sorted(categories),
        "error_entity_types": sorted({e["type"] for e in gold + pred}),
        "source_run_id": NER_ID,
        "factual_error_description": f"exact-span Gold 未匹配 {len(gold)} 个，预测未匹配 {len(pred)} 个。",
        "derived_from_saved_predictions": True,
    }


ner_candidates = [ner_case(r) for r in NER_ROWS]
ner_candidates = [r for r in ner_candidates if r["error_categories"]]
wanted = {
    "PER",
    "LOC",
    "ORG",
    "false negative",
    "false positive",
    "boundary too short",
    "boundary too long",
    "type mismatch",
}
covered, ner_chosen = set(), []
while wanted - covered:
    best = max(
        (r for r in ner_candidates if r not in ner_chosen),
        key=lambda r: (
            len((set(r["error_categories"]) | set(r["error_entity_types"])) - covered),
            -len(r["raw_text"]),
        ),
    )
    gain = (set(best["error_categories"]) | set(best["error_entity_types"])) - covered
    assert gain, f"无真实案例覆盖: {wanted - covered}"
    ner_chosen.append(best)
    covered |= gain
for case in sorted(ner_candidates, key=lambda r: (len(r["raw_text"]), r["sample_id"])):
    if len(ner_chosen) >= 8:
        break
    if case not in ner_chosen:
        ner_chosen.append(case)
jsonl(
    "ner_error_examples.jsonl",
    ner_chosen,
    [f"{NER}/predictions.jsonl", f"{NER}/errors.jsonl"],
    [NER_ID],
)
emit(
    "example_coverage.json",
    {
        "cws_by_model": dict(Counter(c["model"] for c in cws_cases)),
        "ner_count": len(ner_chosen),
        "ner_coverage": sorted(covered),
        "ner_required_coverage": sorted(wanted),
        "selection_note": "post-benchmark report examples only; frozen data subsets unaffected",
    },
    case_sources + [f"{NER}/predictions.jsonl"],
    all_runs + [NER_ID],
)

representative = []
for category, a, b in (
    ("Longest vs Bigram", "longest", "bigram"),
    ("HMM1 vs HMM2", "hmm1-smooth", "hmm2-smooth"),
):
    options = [
        i
        for i in range(3985)
        if DATA[a]["tokens"][i] != DATA[b]["tokens"][i]
        and not any(c.isspace() for c in DATA[a]["predictions"][i]["text"])
    ]
    for i in sorted(options, key=lambda i: (len(DATA[a]["predictions"][i]["text"]), i))[:2]:
        representative.append(
            {
                "kind": category,
                "raw_line": i + 1,
                "raw_text": DATA[a]["predictions"][i]["text"],
                "gold": GOLD[i],
                "predictions": {a: DATA[a]["tokens"][i], b: DATA[b]["tokens"][i]},
                "source_run_ids": [DATA[a]["info"]["run_id"], DATA[b]["info"]["run_id"]],
            }
        )
oov_examples = []
for i in range(3985):
    recovered = [
        w
        for w in span_words(GOLD[i])
        if w["oov"] and (w["start"], w["end"]) in word_spans(NEURAL["tokens"][i])
    ]
    if recovered and not any(c.isspace() for c in NEURAL["predictions"][i]["text"]):
        oov_examples.append(
            {
                "kind": "BiLSTM-CRF correct OOV recall",
                "raw_line": i + 1,
                "raw_text": NEURAL["predictions"][i]["text"],
                "gold": GOLD[i],
                "prediction": NEURAL["tokens"][i],
                "correct_oov_spans": recovered,
                "source_run_ids": [NEURAL["info"]["run_id"]],
            }
        )
representative += sorted(oov_examples, key=lambda r: (len(r["raw_text"]), r["raw_line"]))[:2]
correct = [
    r
    for r in NER_ROWS
    if r["ner_gold"]
    and {entity_key(e) for e in r["ner_gold"]} == {entity_key(e) for e in r["ner_prediction"]}
]
for r in sorted(correct, key=lambda r: (len(r["text"]), r["sample_id"]))[:2]:
    representative.append({"kind": "NER completely correct", **ner_case(r)})
representative += [{"kind": "NER error", **r} for r in ner_chosen[:2]]
assert len(representative) == 10
jsonl(
    "representative_examples.jsonl",
    representative,
    case_sources + [f"{NEURAL['run']}/predictions.jsonl", f"{NER}/predictions.jsonl"],
    all_runs + [NER_ID],
)

ideas = [
    (
        "longest",
        "第二章",
        "从训练词典选择当前位置最长词",
        "简单、IV 召回强",
        "训练词典与单字回退限制多字 OOV 召回",
    ),
    (
        "bigram",
        "第三章",
        "词网中利用相邻词概率寻找最小负对数代价路径",
        "上下文统计消歧提高 IV 召回",
        "词典候选约束，OOV 召回仍弱",
    ),
    (
        "hmm1-smooth",
        "第四章",
        "字符发射与一阶状态转移概率",
        "字符级组合可召回未登录词",
        "条件独立假设限制上下文表示",
    ),
    (
        "hmm2-smooth",
        "第四章 4.7",
        "状态依赖前两个状态，维护状态对 DP",
        "本次 OOV-R 略高于一阶",
        "状态组合更稀疏，不保证总体 F1 提高",
    ),
    (
        "bilstm-crf",
        "实验指导书选做",
        "双向上下文 emission 加线性链 CRF",
        "明显增强 OOV 泛化",
        "训练成本较高，IV-R 低于 Bigram，总体 F1 仅略高",
    ),
]
evolution = []
for model, chapter, idea, strength, limitation in ideas:
    metrics = next(r for r in COMPARISON if r["model_id"] == model)
    evolution.append(
        {
            "name": metrics["model"],
            "chapter": chapter,
            "main_idea": idea,
            "f1": metrics["f1"],
            "oov_recall": metrics["oov_recall"],
            "strength": strength,
            "limitation": limitation,
        }
    )
emit(
    "method_evolution.json",
    {
        "nodes": evolution,
        "edge_ideas": [
            "加入上下文统计",
            "转为字符序列概率建模",
            "加入更高阶状态依赖",
            "学习双向上下文表示",
        ],
        "performance_monotonic": False,
    },
    metric_sources,
    all_runs,
)

# 每段代码均按当前文件的真实行号摘录；没有补写算法。
snippets = [
    ("HMM-1 / Gold tokens → BMES", "src/nlplab/segmentation/bmes.py", 7, 14, "encode"),
    ("HMM-1 / π/A/B 频次", "src/nlplab/sequence_labeling/hmm.py", 75, 89, "HMMTagger.fit"),
    (
        "HMM-1 / 合法 mask、平滑、归一化",
        "src/nlplab/sequence_labeling/hmm.py",
        91,
        108,
        "HMMTagger.fit",
    ),
    (
        "HMM-1 / UNK 与 log probability",
        "src/nlplab/sequence_labeling/hmm.py",
        37,
        41,
        "log_probabilities",
    ),
    ("HMM-1 / 未见字符映射", "src/nlplab/sequence_labeling/hmm.py", 126, 135, "HMMTagger.predict"),
    ("HMM-1 / Viterbi 初始化", "src/nlplab/sequence_labeling/viterbi.py", 34, 44, "viterbi"),
    ("HMM-1 / Viterbi 递推", "src/nlplab/sequence_labeling/viterbi.py", 46, 64, "viterbi"),
    ("HMM-1 / 终止与回溯", "src/nlplab/sequence_labeling/viterbi.py", 66, 82, "viterbi"),
    ("HMM-1 / BMES → words", "src/nlplab/segmentation/bmes.py", 17, 42, "decode"),
    (
        "Longest / 词典与最大词长",
        "src/nlplab/extensions/dictionary.py",
        20,
        37,
        "DictionarySegmenter.__init__",
    ),
    (
        "Longest / 扫描、单字回退与最长选择",
        "src/nlplab/extensions/dictionary.py",
        39,
        62,
        "DictionarySegmenter.candidates / _longest",
    ),
    (
        "Bigram / unigram、bigram 与 BOS/EOS",
        "src/nlplab/extensions/dictionary.py",
        13,
        37,
        "DictionarySegmenter.__init__",
    ),
    (
        "Bigram / 固定插值平滑",
        "src/nlplab/extensions/dictionary.py",
        122,
        138,
        "BigramSegmenter.probability",
    ),
    (
        "Bigram / 词网与负对数边权",
        "src/nlplab/extensions/dictionary.py",
        143,
        166,
        "BigramSegmenter._decode",
    ),
    (
        "Bigram / predecessor、终止与最优回溯",
        "src/nlplab/extensions/dictionary.py",
        167,
        181,
        "BigramSegmenter._decode",
    ),
    ("HMM-2 / first state 与长度 1/2 初始化", "src/nlplab/extensions/hmm2.py", 35, 49, "viterbi2"),
    ("HMM-2 / 递推、backpointer、终止、回溯", "src/nlplab/extensions/hmm2.py", 49, 62, "viterbi2"),
    (
        "HMM-2 / A2 tensor 统计与归一化",
        "src/nlplab/extensions/hmm2.py",
        91,
        109,
        "SecondOrderHMM.fit",
    ),
    (
        "BiLSTM-CRF / Embedding、BiLSTM、Linear、CRF",
        "src/nlplab/extensions/bilstm_crf.py",
        131,
        151,
        "_BiLSTMCRFNetwork.__init__",
    ),
    (
        "BiLSTM-CRF / 双向 emission",
        "src/nlplab/extensions/bilstm_crf.py",
        152,
        159,
        "Network.emissions",
    ),
    (
        "BiLSTM-CRF / input IDs 与 padding mask",
        "src/nlplab/extensions/bilstm_crf.py",
        197,
        220,
        "BiLSTMCRFSegmenter._encode / _batch",
    ),
    (
        "BiLSTM-CRF / 第三方 CRF adapter",
        "src/nlplab/extensions/bilstm_crf.py",
        112,
        125,
        "ConstrainedCRF",
    ),
    (
        "BiLSTM-CRF / loss call 与梯度更新",
        "src/nlplab/extensions/bilstm_crf.py",
        298,
        311,
        "BiLSTMCRFSegmenter.fit",
    ),
    (
        "BiLSTM-CRF / CRF decode",
        "src/nlplab/extensions/bilstm_crf.py",
        222,
        234,
        "BiLSTMCRFSegmenter._decode_batch",
    ),
    (
        "BiLSTM-CRF / 保存 state_dict",
        "src/nlplab/extensions/bilstm_crf.py",
        483,
        500,
        "BiLSTMCRFSegmenter.save",
    ),
    (
        "BiLSTM-CRF / 加载与 SHA 校验",
        "src/nlplab/extensions/bilstm_crf.py",
        501,
        515,
        "BiLSTMCRFSegmenter.load",
    ),
]
code_lines = [
    "# 当前真实实现代码摘录",
    "",
    "只摘录，不执行训练或解码。CRF forward / partition / Viterbi 来自 pytorch-crf 0.7.2（MIT），本项目编写 adapter，不宣称自行实现 CRF。",
    "",
]
snippet_index = []
for title, source, start, end, symbol in snippets:
    text = read(source)
    source_lines = text.splitlines()
    end = min(end, len(source_lines))
    excerpt = "\n".join(source_lines[start - 1 : end])
    code_lines += [
        f"## {title}",
        "",
        f"来源：`{source}:{start}–{end}`；符号：`{symbol}`；文件 SHA256：`{READS[source]}`。",
        "",
        "```python",
        excerpt,
        "```",
        "",
    ]
    snippet_index.append(
        {
            "title": title,
            "source": source,
            "start": start,
            "end": end,
            "symbol": symbol,
            "excerpt_sha256": sha256(excerpt.encode()).hexdigest(),
        }
    )
emit("report_code_snippets.md", "\n".join(code_lines), [s[1] for s in snippets])
emit("code_snippet_index.json", snippet_index, [s[1] for s in snippets])

protocol = """# 评分协议（报告附录）

official raw test = 3985 physical lines；只移除 CRLF，ASCII spaces 保留。模型输入从未使用 Gold 分词。
raw/Gold 的 8 个左引号跨 CRLF 所处侧不同，影响 16 行；没有空行、BOM 或 Unicode 行分隔。
官方 scorer 使用逐行 GNU diff 词字符串对齐；项目 evaluator 使用冻结 raw-coordinate exact spans。
MLE 采用既有 abstention 与冻结的除零保护副本，完整 Gold 分母保留；没有伪造预测。

BiLSTM-CRF official F1=94.4074%，project F1=94.4140%，差约 0.0066 pp。
16 个边界差异行使 official TP 相对 project 少 8；另一个相同行边界 diff 对齐案例多 1，净差 −7。
这是 evaluator alignment protocol 差异，不是模型结果不稳定。OOV-R 相同。
两种输出分别保留，主表沿用官方计数恢复的未舍入比值。详细逐行证据见冻结拓展验收目录。

本阶段仅读取保存的 stdout、metrics 和 validation，没有运行 scorer、模型推理或训练。
"""
emit(
    "scoring_protocol.md",
    protocol,
    [
        f"{BENCH}/report.md",
        f"{BENCH}/protocol/boundary_validation.json",
        "outputs/exp1/extensions/acceptance_20260926_190001/artifact_validation.json",
        f"{NEURAL['run']}/scorer_crosscheck.json",
    ],
    all_runs,
)
lookup = {r["model_id"]: r for r in COMPARISON}


def delta(a, b, metric):
    return (lookup[a][metric] - lookup[b][metric]) * 100


summary = {
    "status": "Completed / Frozen",
    "cws": comparison,
    "ner": NER_METRICS["ner"],
    "changes_pp": {
        "Bigram_minus_Longest": {k: delta("bigram", "longest", k) for k in ("f1", "iv_recall")},
        "HMM2Smooth_minus_HMM1Smooth": {
            k: delta("hmm2-smooth", "hmm1-smooth", k) for k in ("f1", "oov_recall", "iv_recall")
        },
        "BiLSTM_minus_Bigram": {
            k: delta("bilstm-crf", "bigram", k) for k in ("f1", "oov_recall", "iv_recall")
        },
    },
    "bilstm_budget_statement": BUDGET,
    "optional_nlp4j": "not selected",
    "later_experiments": "THUCNews data/split ready; clustering/classification models not started",
    **FLAGS,
}
emit(
    "summary.json",
    summary,
    metric_sources + [f"{NER}/metrics.json", f"{NEURAL['run']}/training_history.json"],
    all_runs + [NER_ID],
)
md = [
    "# 实验一最终收尾",
    "",
    "状态：Completed / Frozen。所有数字来自正式 frozen Run；toy 不进入报告。",
    "",
    "## MSR CWS（百分数）",
    "",
    "| 方法 | P | R | F1 | OOV-R | IV-R | Tag Accuracy | Coverage |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in COMPARISON:
    md.append(
        "| "
        + r["model"]
        + " | "
        + " | ".join(
            f"{r[k] * 100:.4f}"
            for k in (
                "precision",
                "recall",
                "f1",
                "oov_recall",
                "iv_recall",
                "tag_accuracy",
                "coverage",
            )
        )
        + " |"
    )
md += [
    "",
    "Longest/Bigram Tag Accuracy 为 derived from saved predictions，新资产指标，未写回冻结 Run。",
    "HMM1 只保存训练+推理合计时长；CSV 中独立 train/inference seconds 为 null，未拆造。",
    "",
    "## NER：MSRA test personal-200",
    "",
    "| 类别 | P | R | F1 | Gold support |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for label in ("PER", "LOC", "ORG", "Overall"):
    r = NER_METRICS["ner"][label]
    md.append(
        f"| {label} | {r['precision'] * 100:.4f} | {r['recall'] * 100:.4f} | {r['f1'] * 100:.4f} | {r['support']} |"
    )
md += [
    "",
    "ORG F1=85.33%，相对 PER=96.89%、LOC=93.25% 更困难；精确跨度错误包括漏报、误报、边界与类型错误。",
    "MSRA-200 无与 MSR 规范一致的 CWS Gold，只展示 HMM BMES/分词，不报告该集合 CWS P/R/F1。",
    "",
    "## 方法差异",
    "",
    "Longest → Bigram：F1 +0.67 pp，IV-R +1.04 pp；两者 OOV-R≈2.47%。训练词典+单字 fallback 限制多字 OOV 召回，Bigram 收益主要来自上下文统计消歧。",
    "HMM1 Smooth → HMM2 Smooth：F1 −0.08 pp、OOV-R +0.99 pp、IV-R −0.33 pp；更高阶状态组合稀疏，不保证总体性能提高。",
    "BiLSTM-CRF 显著改善未登录词召回，但在该实验中总体 F1 仅略高于 Bigram（+0.09 pp），并且 IV-R 低于 Bigram（94.95% vs 99.25%）。不能宣称所有维度优于传统方法。",
    "",
    "## BiLSTM 训练预算",
    "",
    BUDGET,
    "",
    "max_epochs=10，best_epoch=10，dev F1=95.0573%；patience=2 没有触发提前停止。",
    "",
    "## 实现与评分",
    "",
    "依据课件的二元语法与低阶平滑思想，本项目采用固定插值平滑实现：0.9*P_MLE+0.1*P_add-0.1；并非完全复现课件公式。",
    "CRF forward/partition/Viterbi 来自 pytorch-crf 0.7.2；PyTorch 提供 Embedding/LSTM/Linear，本项目编写 adapter。",
    "主表使用官方 scorer，项目 evaluator 交叉验证；BiLSTM F1 相差约 0.0066 pp，完整原因见 scoring_protocol.md。",
    "",
    "## 只读执行声明",
    "",
    "training_executed=false",
    "official_test_inference_executed=false",
    "scorer_executed=false",
    "",
    "截图采用当前 Web 的冻结输出回放；所有模型/trace/analysis 请求在 Chromium 中以已保存资产响应，未调用推理接口。HMM1 未保存逐步 score/backpointer，仅展示真实源码与保存路径，不重算或伪造 DP 数字。",
    "",
    "所有报告资产写在本目录；原 Run、预测、模型、MSRA selection、MSR protocol 和 split 均不修改。",
    "",
    "实验一已经结束；实验二模型尚未开始。",
    "",
]
emit(
    "summary.md",
    "\n".join(md),
    metric_sources + [f"{NER}/metrics.json", f"{NEURAL['run']}/training_history.json"],
    all_runs + [NER_ID],
)

# Chromium 请求回放包：只保存冻结 JSON，绝不调用模型 trace/segment。
replay = {
    "parameters": {},
    "traces": {},
    "comparison": load("outputs/exp1/extensions/comparison_20260926_190001.json"),
    "analysis": {},
    "source_note": "frozen output replay; no inference",
    **FLAGS,
}
for m in ("longest", "bigram", "hmm2-smooth", "bilstm-crf"):
    d = DATA[m]
    replay["parameters"][m] = {
        "run": d["info"],
        "parameters": d["parameters"],
        "training_history": HISTORY if m == "bilstm-crf" else None,
    }
    replay["traces"][m] = load(f"{d['run']}/example_trace.json")
h = DATA["hmm1-smooth"]
replay["parameters"]["hmm1-smooth"] = {"run": h["info"], "parameters": h["parameters"]}
saved = h["predictions"][0]
replay["traces"]["hmm1-smooth"] = {
    "model": "hmm1-smooth",
    "tokens": saved["segmentation"]["tokens"],
    "tags": saved["segmentation"]["tags"],
    "text": saved["text"],
    "source_sample_id": saved["sample_id"],
    "trace_note": "No per-step scores saved for this frozen input; show source and saved decoded path only",
    "derived_from_saved_predictions": True,
}
replay["analysis"]["hmm1-smooth"] = {
    "text": saved["text"],
    "segmentation": saved["segmentation"],
    "ner": None,
    "elapsed_ms": None,
    "derived_from_saved_predictions": True,
}
replay["hmm1_source_excerpt"] = read("src/nlplab/sequence_labeling/viterbi.py")
emit(
    "browser_replay.json",
    replay,
    [
        f"{h['run']}/predictions.jsonl",
        "src/nlplab/sequence_labeling/viterbi.py",
        *[
            f"{DATA[m]['run']}/example_trace.json"
            for m in ("longest", "bigram", "hmm2-smooth", "bilstm-crf")
        ],
        *[
            f"{DATA[m]['run']}/parameters.json"
            for m in ("hmm1-smooth", "longest", "bigram", "hmm2-smooth", "bilstm-crf")
        ],
    ],
    all_runs,
)

methodology = f"""# 实验一数据与评价口径

状态：Completed / Frozen。CWS 和 NER 分开报告；toy 不作为正式样本或指标。

## CWS

SIGHAN Bakeoff 2005 MSR，standard protocol。七种方法共享完整 official training（86918 句、4050469 字符、2368391 词）及 official raw test（3985 physical lines）。
Longest Matching、Bigram、HMM1 MLE/Smooth、HMM2 MLE/Smooth、BiLSTM-CRF 的正式 word P/R/F1、OOV-R、IV-R 来自已保存官方 scorer。
BMES Tag Accuracy 来自 raw-coordinate 项目 evaluator；Longest/Bigram 新报告值只读派生于保存 prediction tokens 与冻结 Gold alignment，旧 Run 不改。
Coverage 是成功解码句数/3985；MLE 无路径采用既定 abstention，完整 Gold 分母与 successful-only 均保留。
原始 CRLF/ASCII space/16 行边界差异、diff/span 对齐口径见报告资产 scoring_protocol.md；不因差异调整模型。

## NER

MSRA NER test personal subset 200；BIO2 PER/LOC/ORG 转为 Python Unicode code-point [start,end) 实体，并验证 text[start:end]==entity.text。
固定版本 `{MSRA["dataset_version"]}`；个人 key fingerprint 和 selected IDs 保存于原 selection manifest。
抽样在模型推理前冻结，不根据预测换样本。正式实体评价要求 sample/start/end/type 全匹配，报告 PER/LOC/ORG/Overall P/R/F1。
MSRA-200 没有与当前 MSR CWS 标注规范一致的分词 Gold：不报告 MSRA-200 CWS P/R/F1；HMM 只输出 BMES/segmentation。
CWS 正式指标来自 MSR，NER 正式指标来自 MSRA-200，两者不能混成同一个 benchmark。

## 后续实验数据

THUCNews 官方全文个人平衡子集 600：财经/finance、教育/education、科技/science、体育/sports、娱乐/entertainment，各120。
固定分层 train/val/test=420/90/90，各类84/18/18，ID互斥。实验二使用全部600的 id/text 视图，标签仅供事后评价。
实验二、三模型未实现；本收尾不启动它们。custom corpus / 620-slot / manual annotation 保留为 Optional Workflow。

## BiLSTM 训练预算

{BUDGET}

max_epochs=10，development best_epoch=10、F1=95.0573%；达到预算而非 patience=2 提前停止。final 固定10轮。
CRF forward/partition/Viterbi 来自 pytorch-crf 0.7.2，不宣称自行实现 CRF。
Bigram：依据课件的二元语法与低阶平滑思想，本项目采用固定插值平滑实现。

## 本次收尾

资产目录：`{OUT.relative_to(ROOT).as_posix()}`。本阶段 training_executed=false、official_test_inference_executed=false、scorer_executed=false。
截图显示冻结输出回放；HMM1 未保存 DP 数值时展示真实源码和保存 BMES path，不生成缺失 score/backpointer。
"""
update_document("docs/exp1_data_methodology.md", methodology)
requirements = [
    (
        "HMM 原理",
        "HMMTagger",
        "MSR training",
        "π/A/B 概率与说明",
        "src/nlplab/sequence_labeling/hmm.py",
        "HMMTagger",
        "hmm1_parameters.json; HMM π/A/B 截图",
    ),
    (
        "HMM 序列预测",
        "log-space Viterbi",
        "MSR raw test",
        "BMES / 保存路径",
        "src/nlplab/sequence_labeling/viterbi.py",
        "viterbi",
        "report_code_snippets.md; HMM1 路径截图",
    ),
    (
        ">=200 中文样本",
        "test-only personal hash selection",
        "MSRA test personal-200",
        "200 unique IDs / selection fingerprint",
        "src/nlplab/data/msra_ner.py",
        "select_exp1",
        "data/course/exp1_msra_200_manifest.json",
    ),
    (
        "π 构建",
        "起始计数与合法 B/S 平滑",
        "MSR training",
        "4 维 π",
        "src/nlplab/sequence_labeling/hmm.py",
        "HMMTagger.fit",
        "hmm1_parameters.json",
    ),
    (
        "A 构建",
        "相邻状态计数与行归一化",
        "MSR training",
        "4×4 A",
        "src/nlplab/sequence_labeling/hmm.py",
        "HMMTagger.fit",
        "hmm1_parameters.json",
    ),
    (
        "B 构建",
        "状态字符频次、UNK、行归一化",
        "MSR training",
        "各状态 Top-K emission",
        "src/nlplab/sequence_labeling/hmm.py",
        "HMMTagger.fit",
        "hmm1_parameters.json",
    ),
    (
        "BMES 标签",
        "词边界 Gold codec",
        "MSR Gold tokens",
        "B/M/E/S",
        "src/nlplab/segmentation/bmes.py",
        "encode",
        "report_code_snippets.md",
    ),
    (
        "中文分词结果",
        "codec roundtrip",
        "MSR raw + MSRA200 input",
        "分词与 BMES，MSRA 不计 CWS F1",
        "src/nlplab/segmentation/bmes.py",
        "decode",
        "interactive segmentation 截图; representative_examples.jsonl",
    ),
    (
        "PER",
        "HanLP NR→PER 显式映射",
        "MSRA200",
        "P/R/F1; F1≈96.89%",
        "src/nlplab/ner/hanlp_adapter.py",
        "HanLPRecognizer.recognize",
        "metrics_ner.json; NER metrics 截图",
    ),
    (
        "LOC",
        "HanLP NS→LOC 显式映射",
        "MSRA200",
        "P/R/F1; F1≈93.25%",
        "src/nlplab/ner/hanlp_adapter.py",
        "HanLPRecognizer.recognize",
        "metrics_ner.json",
    ),
    (
        "ORG",
        "HanLP NT→ORG 显式映射",
        "MSRA200",
        "P/R/F1; F1≈85.33%",
        "src/nlplab/ner/hanlp_adapter.py",
        "HanLPRecognizer.recognize",
        "metrics_ner.json; ner_error_examples.jsonl",
    ),
    (
        "准确率/P/R/F1",
        "词跨度/实体跨度评价",
        "MSR 与 MSRA 分开",
        "Tag Accuracy、word PRF、NER exact-span PRF",
        "src/nlplab/evaluation/segmentation.py",
        "evaluate_segmentation",
        "cws_seven_model_comparison.csv; metrics_ner.json",
    ),
    (
        "误差分析",
        "保存差异与透明 heuristic",
        "frozen predictions/errors",
        "25 CWS / >=8 NER 错例",
        "src/nlplab/evaluation/errors.py",
        "error_record",
        "cws_error_examples.jsonl; ner_error_examples.jsonl",
    ),
    (
        "可视化",
        "原生 JS + ECharts",
        "saved parameters/trace/metrics",
        "概率、DP、曲线、comparison",
        "app/static/js/charts.js",
        "loadParameters",
        "screenshots/; browser_replay.json",
    ),
    (
        "BiLSTM-CRF 选做",
        "PyTorch + pytorch-crf 0.7.2 adapter",
        "MSR",
        "F1≈94.41%; architecture/history",
        "src/nlplab/extensions/bilstm_crf.py",
        "BiLSTMCRFSegmenter",
        "bilstm_architecture.json; bilstm_training_summary.json",
    ),
    (
        "二阶 HMM 课件拓展",
        "状态三元组 tensor + pair DP",
        "MSR",
        "MLE/Smooth metrics; A2 slices",
        "src/nlplab/extensions/hmm2.py",
        "viterbi2",
        "hmm2_parameters.json; second-order trace 截图",
    ),
    (
        "Longest Matching",
        "训练词典前向最长匹配",
        "MSR",
        "F1≈93.65%; dictionary/max-length",
        "src/nlplab/extensions/dictionary.py",
        "DictionarySegmenter._longest",
        "longest_statistics.json; longest trace 截图",
    ),
    (
        "Bigram",
        "词网、固定插值、负对数代价 DP",
        "MSR",
        "F1≈94.32%; predecessor/path",
        "src/nlplab/extensions/dictionary.py",
        "BigramSegmenter._decode",
        "bigram_statistics.json; lattice 截图",
    ),
]
traceability = [
    "# 实验指导书要求—证据矩阵",
    "",
    "Completed / Frozen；CWS 与 NER 数据及指标分开，报告资产仅从真实实现与正式 Run 提取。",
    "",
    "| 实验要求 | 实现 | 数据 | 指标/输出 | 代码证据 | Web/报告证据 |",
    "| --- | --- | --- | --- | --- | --- |",
]
for requirement, implementation, dataset, output, file, symbol, evidence in requirements:
    tree = ast.parse(read(file))
    parts = symbol.split(".")
    nodes = tree.body
    for part in parts:
        node = next(
            n
            for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == part
        )
        nodes = node.body
    traceability.append(
        f"| {requirement} | {implementation} | {dataset} | {output} | [{symbol}](../{file}#L{node.lineno}) | {evidence} |"
    )
traceability += [
    "",
    f"报告资产根目录：`{OUT.relative_to(ROOT).as_posix()}`。代码路径及符号已通过 AST 校验。",
    "NER evaluator：`src/nlplab/evaluation/ner.py::evaluate_ner`；实体要求 sample/start/end/type 全匹配。",
    "CRF forward/partition/Viterbi 为第三方库；NLP4J 未选做。",
    "截图为已保存输出回放；HMM1 的缺失 DP 数值没有重算。",
    "",
]
update_document("docs/exp1_requirement_traceability.md", "\n".join(traceability))

deps = {}
for line in read(f"{NER}/runtime_requirements.txt").splitlines():
    if "==" in line:
        package, version = line.split("==", 1)
        if package.lower() in {"numpy", "hanlp", "tensorflow"}:
            deps[package] = version
neural_env = load("outputs/exp1/extensions/acceptance_20260926_190001/environment_check.json")
runtime = load(f"{BENCH}/protocol/runtime_provenance.json")
freeze = [
    "# 实验一正式冻结",
    "",
    "## Dataset",
    "",
    f"MSR：SIGHAN Bakeoff 2005 MSR / standard。URL `{CTX['source_download_provenance']['download_url']}`；archive SHA256 `{CTX['source_download_provenance']['archive_sha256']}`。",
    f"MSRA：`{MSRA['dataset_version']}`；test URL `{MSRA['provenance']['sources']['test']['url']}`；test source SHA256 `{MSRA['source_hashes']['test']}`。",
    f"MSRA-200 selection key fingerprint：`{MSRA['selection_key_fingerprint']}`；manifest SHA256 `{sha256((ROOT / 'data/course/exp1_msra_200_manifest.json').read_bytes()).hexdigest()}`。",
    f"THUCNews：{THU['dataset_variant']} / {THU['dataset_version']}，600 manifest SHA256 `{sha256((ROOT / 'data/course/thucnews_600_manifest.json').read_bytes()).hexdigest()}`。",
    f"THUCNews split manifest SHA256 `{sha256((ROOT / 'data/splits/thucnews_600/manifest.json').read_bytes()).hexdigest()}`；split key fingerprint `{SPLIT['split_key_fingerprint']}`；420/90/90。",
    "原始公开数据、selection 原始键及权重不提交 Git；许可未明确者沿用 provenance 的 unknown。",
    "",
    "## Final Runs",
    "",
    "| 方法 | Final Run ID |",
    "| --- | --- |",
]
freeze += [f"| {m} | `{rid}` |" for _, m, rid, _ in METHODS]
freeze += [
    f"| MSRA NER personal-200 | `{NER_ID}` |",
    "",
    "## Dependency versions",
    "",
    f"Python `{neural_env['python']}`，NumPy `{deps.get('numpy')}`，PyTorch `{neural_env['torch']}`（CPU），pytorch-crf `{neural_env['pytorch-crf']}`，HanLP `{deps.get('hanlp')}`。",
    "scorer 环境来自已保存 runtime_provenance：Perl 5.38.2，GNU diffutils 3.12；本阶段未执行 scorer。",
    f"Perl executable SHA256 `{runtime['runtime']['perl']['sha256']}`；diff executable SHA256 `{runtime['runtime']['diff']['sha256']}`。",
    "",
    "## Freeze statement",
    "",
    "实验一正式冻结。后续实验二、实验三开发不得根据实验一 official-test 结果重新调整实验一模型、超参数、数据子集或评价协议。",
    "",
    "原有264文件与拓展90文件在收尾开始前均 unchanged；本次 begin/end snapshot 保护全部模型、Run、预测、metrics、scorer输出和 selected200。结束报告见 closeout_hash_validation.json。",
    f"只读报告资产：`{OUT.relative_to(ROOT).as_posix()}`；资产清单：asset_manifest.json。",
    "training_executed=false；official_test_inference_executed=false；scorer_executed=false。",
    "",
    BUDGET,
    "",
]
update_document("docs/exp1_freeze.md", "\n".join(freeze))
readme = read("README.md")
readme = readme.replace(
    "| 一：序列标注 | 冻结 MSR CWS + MSRA NER test 个人 200 子集 | MSR 保持冻结；200 条只保存分词/BMES 预测并评价 NER |",
    "| 一：序列标注 | 七方法 MSR CWS + MSRA NER test personal-200 | Completed / Frozen：标准评价、错误分析、算法过程与 Web 均完成 |",
)
readme = readme.replace(
    "# 词典最长匹配、二元语法（直接进行一次 final raw-test 评测）",
    "# 历史训练/评测命令，仅供了解已完成协议；冻结后不得执行或据 test 调整",
)
readme = readme.replace(
    "# 二阶 HMM：先 development，再用固定配置做 final",
    "# 历史二阶 HMM 协议：development → 固定配置 final（已完成冻结）",
)
readme = readme.replace(
    "# BiLSTM-CRF 使用独立 PyTorch 环境；先 development 确定 best epoch",
    "# 历史 BiLSTM-CRF 协议：development best epoch=10 → final 10轮（已完成冻结）",
)
readme = readme.replace("## 当前状态", "## 当前状态：实验一 Completed / Frozen", 1)
block = f"""## 实验一最终报告资产（只读收尾）

已完成 Longest Matching、Bigram、HMM-1 MLE/Smooth、HMM-2 MLE/Smooth、BiLSTM-CRF、MSRA NER、标准评价、错误分析、算法过程可视化及 Web。
资产目录：`{OUT.relative_to(ROOT).as_posix()}`，含 summary、七模型 CSV/JSON、NER metrics、真实代码摘录、参数/训练预算、错误/代表案例、截图、来源 SHA256 和执行声明。
Longest/Bigram Tag Accuracy 为保存预测的只读派生值，不修改旧 Run。MSRA-200 不报告不存在的 CWS Gold F1。
BiLSTM development 达到 max_epochs=10，best_epoch=10，patience=2 未提前停止；不增加轮数、不重新评价。
当前数据/模型/权重与正式 benchmark 已准备并冻结。本收尾仅提取保存资产；训练、official-test 推理、scorer 执行均为 false。

- [数据与评价口径](docs/exp1_data_methodology.md)
- [实验要求—证据矩阵](docs/exp1_requirement_traceability.md)
- [最终冻结声明](docs/exp1_freeze.md)

实验二 THUCNews 600 全文数据已准备，模型未实现；实验三固定 420/90/90 split 已准备，模型未实现。
custom corpus / 620-slot / manual annotation 保留为 Optional Workflow，不作为当前主线条件。

"""
if "## 实验一最终报告资产（只读收尾）" not in readme:
    readme = readme.replace("## 环境安装", block + "## 环境安装", 1)
update_document("README.md", readme)
emit(
    "execution_audit.json",
    {
        "scope": "asset generation only",
        "subprocess_guard": "deny subprocess/os.system/exec/spawn",
        "model_call_guard": "deny fit/predict/segment/recognize/Viterbi/CRF emission/scorer calls",
        "write_guard": "only new report directory and authorized README/docs",
        "prohibited_calls_executed": 0,
        "source_reads": READS,
        **FLAGS,
    },
    ["scripts/closeout_exp1.py", OUT / "closeout_before.json"],
)
_, protected = verify_before()
emit(
    "closeout_hash_validation.json",
    {
        "status": "unchanged",
        "baseline_count": 264,
        "extension_count": 90,
        "protected_count": len(protected),
        "changed": [],
        "model_hashes_unchanged": True,
        "prediction_hashes_unchanged": True,
        "metrics_hashes_unchanged": True,
        "scorer_outputs_unchanged": True,
        "selected_200_manifest_unchanged": True,
        **FLAGS,
    },
    [OUT / "closeout_before.json", OUT / "closeout_before_supplement.json"],
)

# Add snapshots/docs to the inventory. Chromium screenshot entries are finalized separately.
for file in ("closeout_before.json", "closeout_before_supplement.json"):
    path = OUT / file
    value = load(path)
    emit(
        file,
        value,
        [
            "outputs/exp1/extensions/acceptance_20260926_190001/extension_freeze.json",
            "artifacts/cache/extensions_frozen_before_20260926_160328.json",
        ],
        all_runs + [NER_ID],
        False,
    )
(OUT / "asset_manifest.json").write_text(
    json.dumps(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "self_hash_excluded": True,
            "assets": list(ASSETS.values()),
            **FLAGS,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "output": str(OUT),
            "assets": len(ASSETS),
            "derived_tag_accuracy": derived,
            "ner_coverage": sorted(covered),
            **FLAGS,
        },
        ensure_ascii=False,
    )
)
