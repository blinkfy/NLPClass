"""扩展既有分类服务；Stage1 返回值来源仍为冻结 Run，Stage2 使用独立指针。"""

import json
import os
import subprocess
import sys

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.services.classification_service import ClassificationService


class ClassificationStage2Service(ClassificationService):
    def stage2_directory(self, run_id="latest"):
        if run_id == "latest":
            pointer = self.store.directory / "stage2_latest.json"
            if not pointer.is_file():
                raise ResourceUnavailableError("尚无 Stage2 正式 Run")
            run_id = json.loads(pointer.read_text("utf-8"))["run_id"]
        directory = self.store.resolve(run_id)
        info = self.store.read(directory.name, "run.json")
        if info["run_type"] != "formal-exp3-stage2":
            raise DataValidationError("C2 需要选择 Stage2 Run")
        return directory

    def parent_run(self, run_id):
        info = self.store.read(run_id, "run.json")
        return info["stage1_run_id"] if info["run_type"] == "formal-exp3-stage2" else info["run_id"]

    def overview(self, run_id="latest"):
        info = self.store.read(run_id, "run.json")
        if info["run_type"] != "formal-exp3-stage2":
            return {
                **super().overview(run_id),
                "available_models": ["C0", "C1"],
                "view_run_id": info["run_id"],
            }
        self.verify_artifacts(run_id)
        parent = info["stage1_run_id"]
        base = super().overview(parent)
        stage2 = {
            name: self.store.read(run_id, filename)
            for name, filename in {
                "run": "run.json",
                "config": "config.json",
                "representation": "representation_stats.json",
                "validation": "validation_selection.json",
                "paired": "paired_c1_c2.json",
                "coverage": "coverage.json",
                "top_chi2": "top_chi2_features.json",
                "environment": "environment.json",
            }.items()
        }
        return {
            **base,
            "test": {**base["test"], **self.store.read(run_id, "test_metrics.json")},
            "stage2": stage2,
            "available_models": ["C0", "C1", "C2"],
            "view_run_id": info["run_id"],
        }

    def predictions(self, run_id="latest", model_id="C0", errors_only=False):
        if model_id == "C2":
            directory = self.stage2_directory(run_id)
            self.verify_artifacts(directory.name)
            from nlplab.data.corpus import iter_jsonl

            return [
                row
                for row in iter_jsonl(directory / "C2_test_predictions.jsonl", lambda r: r)
                if not errors_only or not row["correct"]
            ]
        return super().predictions(self.parent_run(run_id), model_id, errors_only)

    def features(self, run_id="latest", model_id="C0"):
        if model_id == "C2":
            directory = self.stage2_directory(run_id)
            self.verify_artifacts(directory.name)
            return self.store.read(directory.name, "final/C2_global_features.json")
        return super().features(self.parent_run(run_id), model_id)

    def paired(self, run_id="latest"):
        directory = self.stage2_directory(run_id)
        self.verify_artifacts(directory.name)
        return self.store.read(directory.name, "paired_c1_c2.json")

    def predict(self, text, model_id="C0", run_id="latest"):
        if model_id != "C2":
            return super().predict(text, model_id, self.parent_run(run_id))
        if not text.strip() or len(text) > self.settings.max_text_length:
            raise DataValidationError(f"请输入 1–{self.settings.max_text_length} 字新闻文本")
        directory = self.stage2_directory(run_id)
        runtime = self.settings.root / "artifacts/runtime/exp2-venv/Scripts/python.exe"
        try:
            result = subprocess.run(
                [
                    str(runtime) if runtime.is_file() else sys.executable,
                    "-m",
                    "nlplab.classification.stage2",
                    "--root",
                    str(self.settings.root),
                    "--predict",
                    "--run-id",
                    directory.name,
                ],
                input=json.dumps({"text": text}, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=45,
                cwd=self.settings.root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.settings.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                },
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ResourceUnavailableError(f"C2 推理环境失败或超时: {error}") from error
        if result.returncode:
            detail = (
                result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无诊断信息"
            )
            raise ResourceUnavailableError(f"C2 推理进程退出 ({result.returncode}): {detail[:400]}")
        return json.loads(result.stdout)
