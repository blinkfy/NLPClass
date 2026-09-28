"""恢复已保存检查点，只补文档/清单；不重建已完成的报告资产。"""

# ruff: noqa: E501 -- 报告 Markdown 的表格和说明保留为完整文本。

from __future__ import annotations

import ast
import json
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(json.loads((ROOT / "artifacts/cache/exp1_closeout_current.json").read_text())["output"])
FLAGS = dict(training_executed=False, official_test_inference_executed=False, scorer_executed=False)
READS = {}
SOURCE = (ROOT / "scripts/closeout_exp1.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
# Reuse the original guards and file helpers without executing its asset pipeline.
for definition in TREE.body:
    if isinstance(definition, ast.FunctionDef) and definition.name in {
        "guard",
        "profile",
        "read",
        "load",
        "update_document",
    }:
        exec(
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                str(ROOT / "scripts/closeout_exp1.py"),
                "exec",
            )
        )
guard, profile, read, load, update_document = (
    globals()[name] for name in ("guard", "profile", "read", "load", "update_document")
)
sys.addaudithook(guard)
sys.setprofile(profile)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def literal(name):
    node = next(
        n
        for n in TREE.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node.value)


def verify():
    before = load(OUT / "closeout_before.json")
    supplement = load(OUT / "closeout_before_supplement.json")
    expected = before["files"] | supplement["files"]
    changed = [name for name, value in expected.items() if digest(ROOT / name) != value]
    allowed = [name for name in changed if name.startswith("app/")]
    forbidden = [name for name in changed if name not in allowed]
    if forbidden:
        raise RuntimeError(f"冻结资产变化：{forbidden}")
    checkpoint = load(OUT / "closeout_checkpoint.json")
    preserved = {
        name: value
        for name, value in checkpoint["hashes"].items()
        if name not in {"summary.md", "browser_replay.json"}
    }
    if any(digest(OUT / name) != value for name, value in preserved.items()):
        raise RuntimeError("已完成的非 Web 报告资产被覆盖")
    return before, allowed, len(expected), len(preserved)


BEGIN, WEB_CHANGES, PROTECTED_COUNT, PRESERVED_COUNT = verify()
BENCH = "outputs/exp1/benchmarks/20260926_130612_32a5e168"
NER = "outputs/exp1/course_samples/20260926_152118_34a953fb"
NER_ID = Path(NER).name
CTX = load(f"{BENCH}/benchmark_context.json")
MSRA = load("data/course/exp1_msra_200_manifest.json")
THU = load("data/course/thucnews_600_manifest.json")
SPLIT = load("data/splits/thucnews_600/manifest.json")
METHODS, BUDGET = literal("METHODS"), literal("BUDGET")
RUNS = {
    model: f"{BENCH}/{rid}" if model.startswith("hmm1") else f"outputs/exp1/extensions/{rid}"
    for model, _, rid, _ in METHODS
}

if not any(flag in sys.argv for flag in ("--finalize", "--inventory")):
    # Validate real symbols. JavaScript exports are checked as JS, not Python AST.
    matrix = [
        "# 实验指导书要求—证据矩阵",
        "",
        "Completed / Frozen；CWS 与 NER 数据及指标分开。",
        "",
        "| 实验要求 | 实现 | 数据 | 指标/输出 | 代码证据 | Web/报告证据 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for requirement, implementation, dataset, output, file, symbol, evidence in literal(
        "requirements"
    ):
        code = read(file)
        if file.endswith(".js"):
            match = re.search(r"(?:async\s+)?function\s+" + re.escape(symbol) + r"\s*\(", code)
            assert match, (file, symbol)
            line = code[: match.start()].count("\n") + 1
        else:
            nodes = ast.parse(code).body
            for part in symbol.split("."):
                node = next(
                    n
                    for n in nodes
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and n.name == part
                )
                nodes = node.body
            line = node.lineno
        matrix.append(
            f"| {requirement} | {implementation} | {dataset} | {output} | [{symbol}](../{file}#L{line}) | {evidence} |"
        )
    matrix += [
        "",
        "## 当前 Web 证据",
        "",
        "`app/static/js/algorithm-view.js`：scan（最长候选）、lattice（词网/边代价/前驱）、hmm1（state DP）、hmm2（pair DP/backpointer）、neural（emission/CRF path）、renderModelInfo（A2 slice/架构/CRF heatmap/训练曲线）。",
        "API 使用 `app/api/trace_display.py::display_trace` 展示适配，调用原有 Segmenter；没有新增算法。",
        "≤30 code point 展示完整过程，长文本只展示完整分词与模型参数；NER 独立。",
        "截图的交互内容仅来自短 demo，不含 official-test 推理；正式指标读取 frozen Run。",
        f"报告根目录：`{OUT.relative_to(ROOT).as_posix()}`；截图索引：screenshots/index.md。",
        "CRF forward/partition/Viterbi 来自 pytorch-crf 0.7.2，未宣称自行实现 CRF。NLP4J 未选做。",
        "",
    ]
    update_document("docs/exp1_requirement_traceability.md", "\n".join(matrix))
    # Resume only the unexecuted docs/README block from the original closeout script.
    tail = SOURCE[SOURCE.index("deps = {}", SOURCE.index("requirements = [")) :]
    tail = tail[: tail.index('emit(\n    "execution_audit.json"')]
    exec(compile(tail, "closeout pending docs", "exec"))
    web_description = (
        "本次 Web 补丁已验收：Longest 扫描；Bigram 词网、代价与前驱链；HMM1 state-score；"
        "HMM2 state-pair/backpointer/A2 slice；BiLSTM 架构、emission、CRF transition 与 development 曲线。"
        "截图使用短 demo 的正常交互推理及已保存正式指标；未执行 official-test inference、训练或 scorer。"
        "browser_demo_responses.json 保存本次 demo 响应，browser_replay.json 是插入补丁前的只读回放包，未用于本次截图。"
    )
    for file in (OUT / "summary.md", ROOT / "docs/exp1_data_methodology.md"):
        text = file.read_text(encoding="utf-8")
        text = re.sub(r"截图采用当前 Web.*?不重算或伪造 DP 数字。", web_description, text)
        text = text.replace(
            "截图显示冻结输出回放；HMM1 未保存 DP 数值时展示真实源码和保存 BMES path，不生成缺失 score/backpointer。",
            web_description,
        )
        if file.is_relative_to(OUT):
            file.write_text(text, encoding="utf-8")
        else:
            update_document(file.relative_to(ROOT).as_posix(), text)
    readme = read("README.md")
    readme = readme.replace(
        "本收尾仅提取保存资产；训练、official-test 推理、scorer 执行均为 false。",
        "本收尾提取保存资产，并验收短 demo 的模型特异化 Web 展示；训练、official-test 推理、scorer 执行均为 false。",
    )
    readme += "\n### 模型特异化交互过程（已验收）\n\n公共分词结果下分别展示最长匹配扫描、Bigram 词网/边代价/前驱链、一阶 Viterbi state-score、二阶 state-pair/backpointer，以及 BiLSTM-CRF emission/解码路径。参数区分别显示词典统计、固定插值、π/A/B、A2 切片或神经网络架构/CRF heatmap/development 曲线。≤30 Unicode code point 自动展示；长文本仍完整分词。NER 继续独立，默认仍为 HMM1 Smooth。\n\n冻结收尾验证使用 `pytest --verify-only`，在训练/正式 benchmark/scorer 调用前跳过相关历史测试；无需重训 toy 模型。\n"
    update_document("README.md", readme)
    checkpoint = load(OUT / "closeout_checkpoint.json")
    checkpoint["resumed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint["status"] = "resumed from checkpoint; final verification pending"
    checkpoint["completed_steps"] += [
        "model-specific Web patch + Chromium QA",
        "new screenshots",
        "requirement traceability",
        "final freeze document",
        "README status",
    ]
    checkpoint["pending_steps"] = [
        "verify-only full pytest + lint/dependencies",
        "end hash validation and asset_manifest",
    ]
    write("closeout_checkpoint.json", checkpoint)

# Final inventory can be refreshed without changing any metric/parameter/example asset.
screenshots = sorted((OUT / "screenshots").glob("*.png"))
index = (
    "# 当前 Web 截图\n\n仅使用短 demo 交互推理；正式指标读取冻结 Run。\n\n"
    + "\n".join(f"- [{p.name}]({p.name})" for p in screenshots)
    + "\n"
)
(OUT / "screenshots/index.md").write_text(index, encoding="utf-8")
write(
    "execution_audit.json",
    {
        "scope": "read-only closeout + authorized short-demo Web QA",
        "asset_generator_guards": "deny model fit/predict/segment/recognize/Viterbi/CRF emission/scorer and subprocess",
        "demo_inference_executed": True,
        "official_test_records_used_as_demo": False,
        "verify_only": "training/benchmark/scorer blocked before execution",
        "previous_non_web_assets_preserved": PRESERVED_COUNT,
        **FLAGS,
    },
)
verify()
write(
    "closeout_hash_validation.json",
    {
        "status": "unchanged",
        "baseline_count": 264,
        "extension_count": 90,
        "protected_count": PROTECTED_COUNT,
        "changed": [],
        "authorized_web_source_changes": WEB_CHANGES,
        "model_hashes_unchanged": True,
        "prediction_hashes_unchanged": True,
        "metrics_hashes_unchanged": True,
        "scorer_outputs_unchanged": True,
        "selected_200_manifest_unchanged": True,
        "thucnews_split_unchanged": True,
        **FLAGS,
    },
)
if "--finalize" in sys.argv:
    verification = load(OUT / "verification.json")
    assert verification["status"] == "passed"
    checkpoint = load(OUT / "closeout_checkpoint.json")
    checkpoint.update(
        status="completed after checkpoint resume",
        completed_at=datetime.now(timezone.utc).isoformat(),
        pending_steps=[],
    )
    for step in (
        "verify-only full pytest + lint/dependencies",
        "end hash validation and asset_manifest",
    ):
        if step not in checkpoint["completed_steps"]:
            checkpoint["completed_steps"].append(step)
    write("closeout_checkpoint.json", checkpoint)

source_files = [
    f"{p}/{name}" for p in RUNS.values() for name in ("run.json", "metrics.json", "parameters.json")
]
source_files += [
    f"{NER}/run.json",
    f"{NER}/metrics.json",
    "data/course/exp1_msra_200_manifest.json",
    "data/course/thucnews_600_manifest.json",
    "data/splits/thucnews_600/manifest.json",
    "scripts/closeout_exp1.py",
    "scripts/finish_exp1_closeout.py",
]
all_ids = [rid for _, _, rid, _ in METHODS] + [NER_ID]
web_sources = [p.relative_to(ROOT).as_posix() for p in (ROOT / "app/static/js").glob("*.js")]
web_sources += [
    "app/templates/sequence.html",
    "app/templates/index.html",
    "app/static/css/lab.css",
    "app/api/experiment.py",
    "app/api/trace_display.py",
    "app/api/comparison_display.py",
]
replayed_screenshots = set()
display_fix = OUT / "display_fix_validation.json"
if display_fix.is_file():
    replayed_screenshots = set(load(display_fix)["screenshots"])
artifacts = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file() or path.name == "asset_manifest.json":
        continue
    name = path.relative_to(OUT).as_posix()
    sources = source_files.copy()
    if path.suffix == ".png" or name.startswith("browser_"):
        sources += web_sources + [
            "scripts/web_closeout_qa.js",
            "scripts/web_closeout_screenshots.js",
            "scripts/web_closeout_unicode_qa.js",
            "scripts/web_closeout_display_fix_qa.js",
        ]
    if name.startswith("report_code") or name.startswith("code_snippet"):
        sources += sorted({row["source"] for row in load(OUT / "code_snippet_index.json")})
    if "error_examples" in name or "representative_examples" in name or "derived_tag" in name:
        sources += [f"{p}/predictions.jsonl" for p in RUNS.values()] + [
            f"{NER}/predictions.jsonl",
            f"{BENCH}/project_evaluator_gold_alignment.jsonl",
        ]
    if "bilstm" in name:
        sources += [f"{RUNS['bilstm-crf']}/training_history.json"]
    hashes = {source: digest(ROOT / source) for source in dict.fromkeys(sources)}
    artifacts.append(
        {
            "asset_name": name,
            "source_run_id": all_ids,
            "source_files": list(hashes),
            "source_sha256": hashes,
            "generated_sha256": digest(path),
            "derived": name
            not in {
                "closeout_before.json",
                "closeout_before_supplement.json",
                "browser_demo_responses.json",
                "browser_replay.json",
                "metrics_ner.json",
            },
            "generated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "demo_inference_executed": (path.suffix == ".png" and name not in replayed_screenshots)
            or name in {"browser_demo_responses.json", "browser_unicode_qa.json"},
            "inference_executed": (path.suffix == ".png" and name not in replayed_screenshots)
            or name in {"browser_demo_responses.json", "browser_unicode_qa.json"},
            **FLAGS,
        }
    )
for filename in (
    "README.md",
    "docs/exp1_data_methodology.md",
    "docs/exp1_requirement_traceability.md",
    "docs/exp1_freeze.md",
):
    artifacts.append(
        {
            "asset_name": filename,
            "location": "repository",
            "source_run_id": all_ids,
            "source_files": source_files,
            "source_sha256": {s: digest(ROOT / s) for s in source_files},
            "generated_sha256": digest(ROOT / filename),
            "derived": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **FLAGS,
        }
    )
write(
    "asset_manifest.json",
    {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "self_hash_excluded": True,
        "assets": artifacts,
        **FLAGS,
    },
)
print(
    json.dumps(
        {
            "output": str(OUT),
            "assets": len(artifacts),
            "protected_count": PROTECTED_COUNT,
            "preserved": PRESERVED_COUNT,
            **FLAGS,
        }
    )
)
