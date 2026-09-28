"""独立 Run 存储和严格的留出集实验执行。"""

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.io import write_json
from nlplab.core.schema import SequenceSample, text_sha256
from nlplab.data.corpus import duplicate_key, write_jsonl
from nlplab.data.msr import load_msr_provenance
from nlplab.evaluation.errors import error_record
from nlplab.evaluation.segmentation import evaluate_segmentation
from nlplab.ner.base import NERRecognizer
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.sequence_labeling.hmm import HMMTagger


class RunStore:
    """latest 仅在全部结果成功写出后原子更新。"""

    def __init__(self, settings: Settings, experiment: str = "exp1") -> None:
        if not re.fullmatch(r"exp[1-4]", experiment):
            raise DataValidationError("实验标识无效")
        self.settings = settings
        self.directory = settings.outputs / experiment

    def resolve(self, run_id: str = "latest") -> Path:
        if run_id == "latest":
            course_runs = sorted(self.directory.glob("course_samples/*/run.json"))
            frozen_smooth = sorted(self.directory.glob("benchmarks/*/*_smooth/run.json"))
            if course_runs or frozen_smooth:
                return (course_runs or frozen_smooth)[-1].parent
            latest = self.directory / "latest.json"
            if not latest.is_file():
                raise ResourceUnavailableError("尚无实验 Run；请先训练并在留出集上评测")
            run_id = json.loads(latest.read_text(encoding="utf-8"))["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise DataValidationError("run_id 无效")
        directory = self.directory / run_id
        if not (directory / "run.json").is_file():
            matches = list(self.directory.glob(f"course_samples/{run_id}/run.json")) + list(
                self.directory.glob(f"benchmarks/*/{run_id}/run.json")
            )
            if len(matches) != 1:
                raise ResourceUnavailableError(f"实验 Run 不存在或未完成: {run_id}")
            directory = matches[0].parent
        return directory

    def read(self, run_id: str, name: str) -> Any:
        path = self.resolve(run_id) / name
        if not path.is_file():
            raise ResourceUnavailableError(f"Run 结果文件不存在: {name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if name == "metrics.json" and "official_scorer" in value:
            # 只读展示已冻结的评分结果，不重评、不改写任何旧文件。
            project = value["project_evaluator"]["full"]
            return {
                "segmentation": {
                    **project,
                    **value["official_scorer"]["full"],
                    "states": project["gold_labels"],
                },
                "ner": None,
                "frozen_original": value,
            }
        return value

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                [
                    *self.directory.glob("*/run.json"),
                    *self.directory.glob("course_samples/*/run.json"),
                    *self.directory.glob("benchmarks/*/*/run.json"),
                ],
                reverse=True,
            )
        ]

    def model(self, run_id: str = "latest") -> HMMTagger:
        info = self.read(run_id, "run.json")
        if info.get("run_type") in {"course_samples", "frozen-final-benchmark"}:
            path = Path(info["model_path"]).resolve()
            if not path.is_relative_to(self.settings.models.resolve()):
                raise DataValidationError("Run 模型路径必须位于本地 models 目录")
            return HMMTagger.load(path)
        return HMMTagger.load(self.settings.models / info["run_id"])


class ExperimentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RunStore(settings)

    def run(
        self,
        model: HMMTagger,
        samples: list[SequenceSample],
        *,
        dataset: str,
        evaluation_path: Path | None = None,
        is_demo: bool = False,
        config: dict[str, Any] | None = None,
        ner: NERRecognizer | None = None,
    ) -> dict[str, Any]:
        """仅经清单校验的官方 standard test 允许原有 overlap；失败不更新 latest。"""
        if not samples:
            raise DataValidationError("评价集为空")
        hashes = [text_sha256(duplicate_key(sample.text)) for sample in samples]
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise DataValidationError("评价集含重复 sample_id")
        msr_evaluation = (
            load_msr_provenance(evaluation_path, samples) if evaluation_path is not None else None
        )
        if msr_evaluation is not None:
            training = model.training_metadata.get("msr", {})
            if (
                training.get("manifest_sha256") != msr_evaluation["manifest_sha256"]
                or training.get("protocol") != msr_evaluation["protocol"]
                or training.get("split") != "train"
                or msr_evaluation["split"] not in {"dev", "test"}
            ):
                raise DataValidationError("MSR 训练/评价必须来自同一 protocol 清单的正确划分")
        overlap_count = sum(value in model.training_hashes for value in hashes)
        official_standard_test = (
            msr_evaluation is not None
            and msr_evaluation["protocol"] == "standard"
            and msr_evaluation["split"] == "test"
        )
        if overlap_count and not official_standard_test:
            raise DataValidationError("训练/评价文本有重叠，拒绝生成测试指标")
        segmenter = HMMSegmenter(model)
        predictions, errors, predicted_tokens = [], [], []
        for sample in samples:
            result = segmenter.segment(sample.text)
            predicted_tokens.append(result.tokens)
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "doc_id": sample.doc_id,
                    "text": sample.text,
                    "gold_tokens": sample.gold_tokens,
                    "segmentation": asdict(result),
                }
            )
            error = error_record(sample, result.tokens, model.training_words)
            if error:
                errors.append(error)
        metrics = {
            "segmentation": evaluate_segmentation(
                [sample.gold_tokens for sample in samples], predicted_tokens, model.training_words
            ),
            "ner": None,
        }
        if ner is not None:
            from nlplab.evaluation.ner import evaluate_ner

            annotated = [sample for sample in samples if sample.entities is not None]
            if annotated:
                entity_predictions = [ner.recognize(sample.text) for sample in annotated]
                metrics["ner"] = evaluate_ner(
                    [sample.text for sample in annotated],
                    [sample.entities for sample in annotated],
                    entity_predictions,
                )
                entity_map = {
                    sample.sample_id: entities
                    for sample, entities in zip(annotated, entity_predictions)
                }
                for record in predictions:
                    if record["sample_id"] in entity_map:
                        record["ner"] = {
                            "model": ner.name,
                            "entities": [
                                asdict(entity) for entity in entity_map[record["sample_id"]]
                            ],
                        }

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        directory = self.store.directory / run_id
        directory.mkdir(parents=True, exist_ok=False)
        model.save(self.settings.models / run_id)
        parameters = model.parameters((config or {}).get("emission_topk", 20))
        parameter_dir = directory / "parameters"
        parameter_dir.mkdir()
        for key in ("start", "transition", "emission_topk"):
            write_json(parameter_dir / f"{key}.json", parameters[key])
        write_json(directory / "parameters.json", parameters)
        write_json(directory / "metrics.json", metrics)
        write_jsonl(directory / "predictions.jsonl", predictions)
        write_jsonl(directory / "errors.jsonl", errors)
        (directory / "figures").mkdir()
        full_config = {"hmm": asdict(model.config), **(config or {})}
        if msr_evaluation is not None:
            full_config["msr"] = msr_evaluation
        (directory / "config.yaml").write_text(
            yaml.safe_dump(full_config, allow_unicode=True), encoding="utf-8"
        )
        info = {
            "run_id": run_id,
            "dataset": dataset,
            "is_demo": is_demo
            or model.training_metadata.get("is_demo", False)
            or any(sample.is_demo for sample in samples),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "hmm-cws",
            "training": model.training_metadata,
            "evaluation_count": len(samples),
            "error_count": len(errors),
            "evaluation_hashes": hashes,
            "training_evaluation_overlap_count": overlap_count,
            "msr": msr_evaluation,
        }
        write_json(directory / "run.json", info)
        pointer = self.store.directory / (run_id + ".latest.tmp")
        write_json(pointer, {"run_id": run_id})
        pointer.replace(self.store.directory / "latest.json")
        return info
