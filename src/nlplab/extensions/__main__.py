"""阶段化运行课程拓展，任何模型与结果都使用新目录。"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml

from nlplab.data.dataset import load_samples
from nlplab.extensions.benchmark import context, run_model
from nlplab.extensions.dictionary import BigramSegmenter, DictionarySegmenter


def main():
    parser = argparse.ArgumentParser(description="实验一独立分词拓展")
    parser.add_argument(
        "method", choices=["longest", "bigram", "hmm2-mle", "hmm2-smooth", "bilstm-crf", "compare"]
    )
    parser.add_argument("--phase", choices=["development", "final"], default="final")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    ctx = context(args.root)
    if args.method == "compare":
        from nlplab.extensions.comparison import save

        print(save(args.root))
        return
    if args.method == "bilstm-crf":
        from nlplab.core.exceptions import DataValidationError
        from nlplab.extensions.benchmark import fingerprint
        from nlplab.extensions.bilstm_crf import BiLSTMCRFConfig, BiLSTMCRFSegmenter

        config_path = args.root / "configs" / "bilstm_crf.yaml"
        inputs = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = BiLSTMCRFConfig(**inputs)
        checkpoint = args.root / "artifacts/cache" / ("bilstm-training-" + uuid4().hex[:12])
        if args.phase == "final":
            candidates = []
            for run_path in (args.root / "outputs/exp1/extensions").glob("*/run.json"):
                info = json.loads(run_path.read_text(encoding="utf-8"))
                history = run_path.parent / "training_history.json"
                if (
                    info.get("model") == "bilstm-crf"
                    and info.get("phase") == "development"
                    and history.is_file()
                    and info.get("config", {}).get("input_config_sha256")
                    == fingerprint(config_path)
                ):
                    candidates.append(history)
            if not candidates:
                raise DataValidationError("BiLSTM-CRF final 需要先完成 development Run")
            training_result = json.loads(sorted(candidates)[-1].read_text(encoding="utf-8"))
            best_epoch = int(training_result["best_epoch"])
            if best_epoch < 1:
                raise DataValidationError("development 未产生合法 best epoch")
            train_samples = load_samples(Path(ctx["standard_split_path"]) / "train.jsonl")
            model = BiLSTMCRFSegmenter.from_samples(train_samples, config)
            final_result = model.fit(
                train_samples, [], epochs=best_epoch, checkpoint_path=checkpoint
            )
            phase_result = {**final_result, "selected_from_development": training_result}
            samples = train_samples
        else:
            train_samples = load_samples(ctx["development"] / "train.jsonl")
            dev_samples = load_samples(ctx["development"] / "dev.jsonl")
            model = BiLSTMCRFSegmenter.from_samples(train_samples, config)
            phase_result = model.fit(train_samples, dev_samples, checkpoint_path=checkpoint)
            samples = train_samples
        run_model(
            args.root,
            args.method,
            model,
            {
                **model.parameters(),
                "training": "full official training"
                if args.phase == "final"
                else "training-derived",
                "test_tuning": False,
                "input_config": inputs,
                "input_config_sha256": fingerprint(config_path),
                "best_epoch": phase_result["best_epoch"],
            },
            phase=args.phase,
            training_result=phase_result,
        )
        return
    samples = load_samples(
        (Path(ctx["standard_split_path"]) if args.phase == "final" else ctx["development"])
        / "train.jsonl"
    )
    started = perf_counter()
    if args.method.startswith("hmm2"):
        from nlplab.core.exceptions import DataValidationError
        from nlplab.extensions.benchmark import fingerprint
        from nlplab.extensions.hmm2 import SecondOrderHMM
        from nlplab.segmentation.bmes import encode
        from nlplab.sequence_labeling.hmm import HMMConfig

        config_path = args.root / "configs" / (args.method.replace("-", "_") + ".yaml")
        inputs = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = HMMConfig(**{k: inputs[k] for k in ["smoothing", "alpha", "unknown_strategy"]})
        if args.phase == "final":
            development = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in (args.root / "outputs/exp1/extensions").glob("*/run.json")
            ]
            if not any(
                r["model"] == args.method
                and r["phase"] == "development"
                and r["config"].get("input_config_sha256") == fingerprint(config_path)
                for r in development
            ):
                raise DataValidationError("二阶 final 需要同配置完成的 development Run")
        model = SecondOrderHMM(config).fit(
            [list(s.text) for s in samples], [encode(s.gold_tokens) for s in samples]
        )
    else:
        cls = DictionarySegmenter if args.method == "longest" else BigramSegmenter
        model = cls([s.gold_tokens for s in samples])
    run_model(
        args.root,
        args.method,
        model,
        {
            **model.parameters(),
            "training": "full official training" if args.phase == "final" else "training-derived",
            "training_seconds": perf_counter() - started,
            "test_tuning": False,
            **(
                {"input_config": inputs, "input_config_sha256": fingerprint(config_path)}
                if args.method.startswith("hmm2")
                else {}
            ),
        },
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
