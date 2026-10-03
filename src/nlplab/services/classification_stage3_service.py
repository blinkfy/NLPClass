"""C3 扩展服务；冻结 Stage1/2 由原服务读取，Web 不触发训练。"""

import json
import os
import subprocess

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.corpus import iter_jsonl
from nlplab.features.word_sequence import read_json
from nlplab.services.classification_stage2_service import ClassificationStage2Service


class ClassificationStage3Service(ClassificationStage2Service):
    def stage3_directory(self, run_id="latest"):
        if run_id == "latest":
            pointer = self.store.directory / "stage3_latest.json"
            if not pointer.is_file():
                raise ResourceUnavailableError("尚无 Stage3 正式 Run")
            run_id = read_json(pointer)["run_id"]
        directory = self.store.resolve(run_id)
        if self.store.read(directory.name, "run.json")["run_type"] != "formal-exp3-stage3":
            raise DataValidationError("C3 需要 Stage3 Run")
        return directory

    def stage2_directory(self, run_id="latest"):
        if run_id != "latest":
            info = self.store.read(run_id, "run.json")
            if info["run_type"] == "formal-exp3-stage3":
                run_id = info["stage2_run_id"]
        return super().stage2_directory(run_id)

    def parent_run(self, run_id):
        info = self.store.read(run_id, "run.json")
        if info["run_type"] == "formal-exp3-stage3":
            return info["stage1_run_id"]
        return super().parent_run(run_id)

    def overview(self, run_id="latest"):
        info = self.store.read(run_id, "run.json")
        if info["run_type"] != "formal-exp3-stage3":
            return super().overview(run_id)
        self.verify_artifacts(run_id)
        base = super().overview(info["stage2_run_id"])
        stage3 = {
            name: self.store.read(run_id, file)
            for name, file in {
                "run": "run.json",
                "config": "config.json",
                "preregistration": "preregistration.json",
                "representation": "representation_stats.json",
                "validation": "validation_selection.json",
                "selection_freeze": "selection_freeze.json",
                "environment": "environment.json",
                "runtime": "runtime.json",
                "final_metadata": "final_fit_metadata.json",
                "paired_c1": "paired_c1_c3.json",
                "paired_c2": "paired_c2_c3.json",
            }.items()
        }
        return {
            **base,
            "stage3": stage3,
            "test": {**base["test"], **self.store.read(run_id, "test_metrics.json")},
            "available_models": ["C0", "C1", "C2", "C3"],
            "view_run_id": info["run_id"],
            "default_model": "C3",
        }

    def predictions(self, run_id="latest", model_id="C0", errors_only=False):
        if model_id != "C3":
            return super().predictions(run_id, model_id, errors_only)
        directory = self.stage3_directory(run_id)
        self.verify_artifacts(directory.name)
        # 384 通道完整记录留在 artifact；Web 列表只传 Top spans。
        return [
            {k: v for k, v in r.items() if k != "all_channels"}
            for r in iter_jsonl(directory / "C3_test_predictions.jsonl", lambda r: r)
            if not errors_only or not r["correct"]
        ]

    def features(self, run_id="latest", model_id="C0"):
        if model_id != "C3":
            return super().features(run_id, model_id)
        directory = self.stage3_directory(run_id)
        self.verify_artifacts(directory.name)
        return self.store.read(directory.name, "final/model/model.json")

    def paired_stage3(self, run_id="latest", old_model="C2"):
        if old_model not in ("C1", "C2"):
            raise DataValidationError("配对仅支持 C1/C3 或 C2/C3")
        directory = self.stage3_directory(run_id)
        self.verify_artifacts(directory.name)
        return self.store.read(directory.name, f"paired_{old_model.lower()}_c3.json")

    def predict(self, text, model_id="C0", run_id="latest"):
        if model_id != "C3":
            return super().predict(text, model_id, run_id)
        if not text.strip() or len(text) > self.settings.max_text_length:
            raise DataValidationError("请输入 1–5000 字新闻文本")
        directory = self.stage3_directory(run_id)
        runtime = self.settings.root / "artifacts/runtime/exp2-venv/Scripts/python.exe"
        try:
            result = subprocess.run(
                [
                    str(runtime),
                    "-m",
                    "nlplab.classification.stage3",
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
                timeout=90,
                cwd=self.settings.root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(self.settings.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                },
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ResourceUnavailableError(f"C3 推理环境失败或超时: {error}") from error
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无诊断"
            raise ResourceUnavailableError(f"C3 推理进程退出: {detail[:400]}")
        return json.loads(result.stdout)
