"""实验三只读 Run 服务；沿用独立 sklearn runtime，不污染实验一环境。"""

import json
import os
import subprocess
import sys

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.corpus import iter_jsonl
from nlplab.data.course import file_sha256
from nlplab.services.experiment_service import RunStore


class ClassificationService:
    def __init__(self, settings):
        self.settings = settings
        self.store = RunStore(settings, "exp3")

    def verify_artifacts(self, run_id="latest"):
        directory = self.store.resolve(run_id)
        manifest = self.store.read(run_id, "artifact_manifest.json")
        for name, digest in manifest["files"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
                raise DataValidationError(f"分类 artifact 指纹不符: {name}")
        return {"status": "unchanged", "file_count": len(manifest["files"])}

    def list_runs(self):
        return self.store.list_runs()

    def overview(self, run_id="latest"):
        self.verify_artifacts(run_id)
        return {
            name: self.store.read(run_id, file)
            for name, file in {
                "run": "run.json",
                "dataset": "dataset.json",
                "config": "config.json",
                "representation": "representation_stats.json",
                "validation": "validation_selection.json",
                "test": "test_metrics.json",
                "runtime": "runtime.json",
                "environment": "environment.json",
                "selection_freeze": "selection_freeze.json",
                "label_mapping": "label_mapping.json",
            }.items()
        }

    def predictions(self, run_id="latest", model_id="C0", errors_only=False):
        self.validate_model(model_id)
        self.verify_artifacts(run_id)
        return [
            row
            for row in iter_jsonl(
                self.store.resolve(run_id) / f"{model_id}_test_predictions.jsonl", lambda row: row
            )
            if not errors_only or not row["correct"]
        ]

    def features(self, run_id="latest", model_id="C0"):
        self.validate_model(model_id)
        self.verify_artifacts(run_id)
        return self.store.read(run_id, f"final/{model_id}_global_features.json")

    @staticmethod
    def validate_model(model_id):
        if model_id not in {"C0", "C1"}:
            raise DataValidationError("第一阶段仅支持 C0/C1")

    def predict(self, text, model_id="C0", run_id="latest"):
        self.validate_model(model_id)
        if not text.strip() or len(text) > self.settings.max_text_length:
            raise DataValidationError(f"请输入 1–{self.settings.max_text_length} 字新闻文本")
        directory = self.store.resolve(run_id)
        runtime = self.settings.root / "artifacts/runtime/exp2-venv/Scripts/python.exe"
        executable = str(runtime) if runtime.is_file() else sys.executable
        try:
            result = subprocess.run(
                [
                    executable,
                    "-m",
                    "nlplab.classification",
                    "--root",
                    str(self.settings.root),
                    "--predict",
                    "--run-id",
                    directory.name,
                    "--model-id",
                    model_id,
                ],
                input=json.dumps({"text": text}, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=45,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.settings.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                },
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as error:
            raise ResourceUnavailableError("分类推理超过 45 秒，请稍后重试") from error
        except OSError as error:
            raise ResourceUnavailableError(f"分类推理环境不可用: {error}") from error
        if result.returncode:
            detail = (
                result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无诊断信息"
            )
            raise ResourceUnavailableError(
                f"分类推理进程退出 ({result.returncode}): {detail[:400]}"
            )
        return json.loads(result.stdout)
