"""课程数据准备、训练、留出评价和演示的可复现命令行。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, NLPLabError
from nlplab.core.io import write_json
from nlplab.data.corpus import deduplicate, load_annotations, load_documents, write_jsonl
from nlplab.data.course import file_sha256, load_plan, save_quality_report
from nlplab.data.dataset import load_samples
from nlplab.data.msr import import_msr, load_msr_provenance
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.hmm import HMMConfig, HMMTagger
from nlplab.services.experiment_service import ExperimentService


def read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"配置文件不存在: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataValidationError("配置必须是 YAML mapping")
    unknown = set(value) - {
        "smoothing",
        "alpha",
        "unknown_strategy",
        "seed",
        "dev_ratio",
        "emission_topk",
    }
    if unknown:
        raise DataValidationError(f"未知配置项: {sorted(unknown)}")
    if type(value.get("seed", 42)) is not int:
        raise DataValidationError("seed 必须为整数")
    topk = value.get("emission_topk", 20)
    if type(topk) is not int or not 1 <= topk <= 100:
        raise DataValidationError("emission_topk 必须是 1–100 的整数")
    return value


def train_model(path: Path, config: dict[str, Any], *, is_demo: bool = False) -> HMMTagger:
    samples = load_samples(path)
    msr_provenance = load_msr_provenance(path, samples)
    if msr_provenance is not None and msr_provenance["split"] != "train":
        raise DataValidationError("MSR 模型只能使用 train.jsonl 训练")
    model = HMMTagger(
        HMMConfig(
            **{
                key: config[key]
                for key in ("smoothing", "alpha", "unknown_strategy")
                if key in config
            }
        )
    )
    model.fit(
        [list(sample.text) for sample in samples],
        [encode(sample.gold_tokens) for sample in samples],
    )
    model.training_metadata.update(
        {
            "source": str(path.resolve()),
            "is_demo": is_demo or any(sample.is_demo for sample in samples),
        }
    )
    if msr_provenance is not None:
        model.training_metadata["msr"] = msr_provenance
    return model


def main(argv: list[str] | None = None) -> int:
    """错误返回非零退出码，不静默改用 toy 数据。"""
    parser = argparse.ArgumentParser(prog="nlplab", description="NLPLab 课程实验工具")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="模型与 Run 的工程目录")
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("import-corpus", help="验证主语料并输出唯一记录及重复报告")
    corpus.add_argument("--input", type=Path, required=True)
    corpus.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate-course", help="验收真实主语料和 200 条关联 Gold")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--gold", type=Path, required=True)
    msr = sub.add_parser("import-msr", help="导入解压后的 icwb2-data")
    msr.add_argument("--source", type=Path, required=True)
    msr.add_argument("--output", type=Path, default=Path("data/splits/msr"))
    msr.add_argument("--seed", type=int)
    msr.add_argument("--protocol", choices=("standard", "clean"), default="standard")
    msr.add_argument("--config", type=Path, default=Path("configs/hmm_mle.yaml"))
    training = sub.add_parser("train", help="训练 HMM；模型目录必须尚不存在")
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--config", type=Path, default=Path("configs/hmm_mle.yaml"))
    training.add_argument("--demo", action="store_true")
    evaluation = sub.add_parser("evaluate", help="使用留出集生成独立 Run")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--test", type=Path, required=True)
    evaluation.add_argument("--corpus", type=Path, help="自建 Gold 必须提供统一主语料")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--ner", action="store_true")
    demo = sub.add_parser("demo", help="显式 toy 演示，非课程实验结果")
    demo.add_argument("--config", type=Path, default=Path("configs/hmm.yaml"))
    setup_ner = sub.add_parser("setup-ner", help="显式准备 HanLP NER 并核对模型标签词表")
    setup_ner.add_argument(
        "--config", type=Path, help="模型及映射 YAML，默认 configs/hanlp_msra_bert.yaml"
    )
    public_commands = {
        "prepare-msra-ner": "下载/严格导入固定 MSRA BIO 版本",
        "select-exp1-samples": "冻结 MSRA test 个人 200 子集",
        "run-exp1-course-samples": "冻结 Smooth 预测及 HanLP exact-span NER 正式 Run",
        "import-thucnews": "导入官方全文或核验过 provenance 的 cnews 全文",
        "select-thucnews-600": "去重后每类稳定哈希选择 120 条全文",
        "split-thucnews-600": "冻结每类 84/18/18 划分",
        "audit-public-datasets": "只读审计公开数据来源、指纹、子集与划分",
    }
    for name, help_text in public_commands.items():
        command = sub.add_parser(name, help=help_text)
        if name == "prepare-msra-ner":
            command.add_argument("--source", type=Path, help="可选：本地两个原始 BIO 文件目录")
            command.add_argument(
                "--config", type=Path, default=Path("configs/public_datasets.yaml")
            )
        elif name == "import-thucnews":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument(
                "--provenance",
                type=Path,
                required=True,
                help="来源 JSON；下载时间、版本、URL、SHA256、许可说明",
            )
            command.add_argument(
                "--output", type=Path, default=Path("data/external/thucnews_candidates.jsonl")
            )
        elif name.startswith("select-"):
            msra_selection = name == "select-exp1-samples"
            command.add_argument(
                "--input",
                type=Path,
                default=Path(
                    "data/external/msra_ner/test.jsonl"
                    if msra_selection
                    else "data/external/thucnews_candidates.jsonl"
                ),
            )
            command.add_argument(
                "--output",
                type=Path,
                default=Path(
                    "data/course/exp1_msra_200.jsonl"
                    if msra_selection
                    else "data/course/thucnews_600.jsonl"
                ),
            )
            command.add_argument(
                "--keys", type=Path, default=Path("data/manifests/personal_selection.json")
            )
        elif name == "split-thucnews-600":
            command.add_argument(
                "--input", type=Path, default=Path("data/course/thucnews_600.jsonl")
            )
            command.add_argument("--output", type=Path, default=Path("data/splits/thucnews_600"))
            command.add_argument(
                "--keys", type=Path, default=Path("data/manifests/personal_selection.json")
            )
        elif name == "run-exp1-course-samples":
            command.add_argument(
                "--input", type=Path, default=Path("data/course/exp1_msra_200.jsonl")
            )
            command.add_argument(
                "--model", type=Path, help="冻结 Smooth 模型目录；默认从最后一个冻结清单定位"
            )
            command.add_argument("--freeze", type=Path, help="最终 benchmark 的 hmm_freeze.json")
    audit = sub.add_parser("audit-course-corpus", help="只读审计课程语料及 600 条配额")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--config", type=Path, default=Path("configs/course_corpus.yaml"))
    audit.add_argument("--reports", type=Path)
    sampling = sub.add_parser(
        "sample-sequence-gold", help="固定 seed=42 分层抽取 200 条无 Gold 候选"
    )
    sampling.add_argument("--corpus", type=Path, required=True)
    sampling.add_argument("--config", type=Path, default=Path("configs/course_corpus.yaml"))
    sampling.add_argument("--output", type=Path, default=Path("data/annotations/candidates.jsonl"))
    sampling.add_argument("--allow-demo", action="store_true", help="仅测试；正式导出拒绝 demo")
    sampling.add_argument("--reports", type=Path)
    for command in ("validate-sequence-annotations", "export-sequence-gold"):
        annotation = sub.add_parser(
            command,
            help="严格校验人工草稿"
            if command.startswith("validate")
            else "正式导出 exactly 200 reviewed",
        )
        annotation.add_argument("--corpus", type=Path, required=True)
        annotation.add_argument(
            "--candidates", type=Path, default=Path("data/annotations/candidates.jsonl")
        )
        annotation.add_argument("--work-dir", type=Path, default=Path("data/annotations/work"))
        annotation.add_argument("--config", type=Path, default=Path("configs/course_corpus.yaml"))
        annotation.add_argument("--reports", type=Path)
        if command == "export-sequence-gold":
            annotation.add_argument(
                "--output", type=Path, default=Path("data/annotations/sequence_gold.jsonl")
            )
    args = parser.parse_args(argv)
    try:
        reports = getattr(args, "reports", None) or args.root / "outputs/data_quality"
        if args.command in public_commands:
            from nlplab.data.personal_selection import read_object

            if args.command == "prepare-msra-ner":
                from nlplab.data.msra_ner import prepare_msra

                result = prepare_msra(
                    args.root, yaml.safe_load(args.config.read_text("utf-8"))["msra"], args.source
                )
            elif args.command == "select-exp1-samples":
                from nlplab.data.msra_ner import select_exp1

                result = select_exp1(args.input, args.output, args.keys)
            elif args.command == "import-thucnews":
                from nlplab.data.thucnews import import_thucnews

                result = import_thucnews(args.source, args.output, read_object(args.provenance))
            elif args.command == "select-thucnews-600":
                from nlplab.data.thucnews import select_thucnews

                result = select_thucnews(args.input, args.output, args.keys)
            elif args.command == "split-thucnews-600":
                from nlplab.data.thucnews import split_thucnews

                result = split_thucnews(args.input, args.output, args.keys)
            elif args.command == "audit-public-datasets":
                from nlplab.data.public_audit import save_public_audit

                result = save_public_audit(args.root)
            else:
                from nlplab.services.course_samples import run_course_samples

                freezes = sorted((args.root / "outputs/exp1/benchmarks").glob("*/hmm_freeze.json"))
                if not args.freeze and not freezes:
                    raise DataValidationError("缺少冻结 benchmark 清单；请提供 --freeze")
                freeze = args.freeze or freezes[-1]
                model_path = (
                    args.model
                    or args.root / f"artifacts/models/msr-final-{freeze.parent.name}_smooth"
                )
                result = run_course_samples(
                    Settings(args.root.resolve()), args.input, model_path, freeze
                )
            # 不将 600 个 selected IDs 打到终端；完整记录保留在清单。
            if "selected_records" in result:
                result = {
                    key: value
                    for key, value in result.items()
                    if key not in {"selected_records", "provenance"}
                }
        elif args.command == "audit-course-corpus":
            from nlplab.data.course import audit_corpus

            result = audit_corpus(args.corpus, load_plan(args.config))
            result["config_sha256"] = file_sha256(args.config)
            report = save_quality_report(result, reports, "课程主语料只读审计")
            result = {
                "total": result["total"],
                "quota_achieved": result["quota_achieved"],
                "report": str(report),
            }
        elif args.command == "sample-sequence-gold":
            from nlplab.data.sequence_sampling import sample_candidates, save_candidates

            samples, manifest = sample_candidates(
                args.corpus, load_plan(args.config), allow_demo=args.allow_demo
            )
            manifest["config_sha256"] = file_sha256(args.config)
            manifest_path = save_candidates(args.output, samples, manifest)
            report = save_quality_report(manifest, reports, "Sequence Gold 候选抽样")
            if not manifest["success"]:
                raise DataValidationError(
                    f"抽样不足/联合配额不可满足；manifest={manifest_path}；"
                    f"报告={report / 'report.md'}"
                )
            result = {
                "samples": len(samples),
                "output": str(args.output),
                "manifest": str(manifest_path),
                "report": str(report),
            }
        elif args.command in {"validate-sequence-annotations", "export-sequence-gold"}:
            from nlplab.data.sequence_annotations import AnnotationStore

            store = AnnotationStore(
                args.root,
                corpus=args.corpus,
                candidates=args.candidates,
                work=args.work_dir,
                config=args.config,
            )
            if args.command == "export-sequence-gold":
                result = store.export(args.output, reports)
            else:
                _, result = store.validation()
                report = save_quality_report(result, reports, "Sequence Gold 草稿校验")
                if not result["valid"]:
                    raise DataValidationError(
                        f"人工 Gold 未满足正式要求；"
                        f"错误={json.dumps(result['errors'], ensure_ascii=False)}；"
                        f"报告={report / 'report.md'}"
                    )
                result = {
                    "status": "valid",
                    "reviewed": result["reviewed_count"],
                    "report": str(report),
                }
        elif args.command == "import-corpus":
            documents = load_documents(args.input)
            unique, duplicates = deduplicate(documents)
            report = args.output.with_suffix(".duplicates.json")
            if args.output.exists() or report.exists():
                raise DataValidationError("导入输出或重复报告已存在，拒绝覆盖")
            write_jsonl(args.output, [document.to_dict() for document in unique])
            write_json(report, duplicates)
            result = {
                "unique": len(unique),
                "duplicates": len(duplicates),
                "output": str(args.output),
            }
        elif args.command == "validate-course":
            documents = load_documents(args.corpus)
            unique, duplicates = deduplicate(documents)
            samples = load_annotations(args.gold, documents)
            if (
                duplicates
                or len(unique) < 500
                or len(samples) < 200
                or any(doc.is_demo for doc in documents)
                or any(sample.is_demo for sample in samples)
            ):
                raise DataValidationError(
                    "需 >=500 篇无重复真实文档、>=200 条 Gold；不允许 demo 语料"
                )
            if len({sample.text for sample in samples}) != len(samples):
                raise DataValidationError("序列 Gold 文本有重复")
            result = {"documents": len(unique), "sequence_gold": len(samples), "status": "valid"}
        elif args.command == "import-msr":
            config = read_config(args.config)
            result = import_msr(
                args.source,
                args.output,
                protocol=args.protocol,
                seed=args.seed if args.seed is not None else config.get("seed", 42),
                dev_ratio=config.get("dev_ratio", 0.1),
            )
        elif args.command == "train":
            model = train_model(args.train, read_config(args.config), is_demo=args.demo)
            model.save(args.model)
            result = {"model": str(args.model), "training": model.training_metadata}
        elif args.command == "setup-ner":
            from nlplab.ner.hanlp_adapter import prepare_ner

            result = prepare_ner(Settings(args.root.resolve()), args.config)
        else:
            service = ExperimentService(Settings(args.root.resolve()))
            if args.command == "demo":
                config = read_config(args.config)
                model = train_model(Path("data/demo/toy_train.jsonl"), config, is_demo=True)
                result = service.run(
                    model,
                    load_samples(Path("data/demo/toy_test.jsonl")),
                    dataset="toy-demo",
                    is_demo=True,
                    config=config,
                )
            else:
                model = HMMTagger.load(args.model)
                samples = (
                    load_annotations(args.test, load_documents(args.corpus))
                    if args.corpus
                    else load_samples(args.test)
                )
                ner = None
                if args.ner:
                    from nlplab.ner.hanlp_adapter import HanLPRecognizer

                    ner = HanLPRecognizer.from_settings(Settings(args.root.resolve()))
                result = service.run(
                    model,
                    samples,
                    dataset=args.dataset,
                    evaluation_path=args.test,
                    ner=ner,
                    is_demo=model.training_metadata.get("is_demo", False),
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (NLPLabError, OSError, UnicodeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
