"""Read-only access to Stage5 source/template ablation artifacts."""

import re

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.features.word_sequence import read_json


class ClassificationStage5Service:
    _verified = set()

    def __init__(self, settings):
        self.directory = settings.outputs / "exp3" / "stage5"

    def resolve(self, run_id="latest"):
        if run_id == "latest":
            pointer = self.directory.parent / "stage5_latest.json"
            if not pointer.is_file():
                raise ResourceUnavailableError("尚无已完成的 Stage5 消融分析")
            pointer_data = read_json(pointer)
            if not isinstance(pointer_data, dict):
                raise DataValidationError("Stage5 latest pointer 格式无效")
            run_id = pointer_data.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise DataValidationError("Stage5 run_id 无效")

        directory = self.directory / run_id
        root = self.directory.resolve()
        if not directory.resolve().is_relative_to(root):
            raise DataValidationError("Stage5 run 路径无效")
        if not (directory / "run.json").is_file():
            raise ResourceUnavailableError("Stage5 分析不存在或未完成")
        manifest_path = directory / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise ResourceUnavailableError("Stage5 artifact manifest 不存在")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise DataValidationError("Stage5 artifact manifest 格式无效")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise DataValidationError("Stage5 artifact manifest 格式无效")

        run_root = directory.resolve()
        file_state = []
        for name, digest in sorted(files.items()):
            if not isinstance(name, str) or name in {"", ".", "..", "artifact_manifest.json"}:
                raise DataValidationError("Stage5 artifact manifest 文件名无效")
            path = (directory / name).resolve()
            if not path.is_relative_to(run_root):
                raise DataValidationError("Stage5 artifact 路径无效")
            if not path.is_file():
                raise ResourceUnavailableError(f"Stage5 artifact 不存在: {name}")
            stat = path.stat()
            file_state.append((name, stat.st_mtime_ns, stat.st_size))

        key = (
            str(directory),
            manifest_path.stat().st_mtime_ns,
            manifest_path.stat().st_size,
            tuple(file_state),
        )
        if key not in self._verified:
            for (name, _, _), (_, digest) in zip(file_state, sorted(files.items())):
                if not isinstance(digest, str) or file_sha256(directory / name) != digest:
                    raise DataValidationError(f"Stage5 artifact hash 不符: {name}")
            self._verified.add(key)
        return directory

    def overview(self, run_id="latest"):
        directory = self.resolve(run_id)
        results = read_json(directory / "results.json")
        prevalence = read_json(directory / "prevalence.json")
        shortcut_summary = read_json(directory / "shortcut_summary.json")
        if not isinstance(results, list):
            raise DataValidationError("Stage5 results 格式无效")
        if not isinstance(prevalence, dict) or not isinstance(shortcut_summary, dict):
            raise DataValidationError("Stage5 分析摘要格式无效")
        return {
            "run": read_json(directory / "run.json"),
            "results": results,
            "prevalence": prevalence,
            "shortcut_summary": shortcut_summary,
            "formal_test_evaluation_count": 0,
            "test_access_permitted": False,
        }

    def flips(self, run_id="latest", model_id=None, condition=None):
        directory = self.resolve(run_id)
        rows = read_json(directory / "flips.json")
        if not isinstance(rows, list):
            raise DataValidationError("Stage5 flips 格式无效")
        return [
            row
            for row in rows
            if (model_id is None or row.get("model_id") == model_id)
            and (condition is None or row.get("condition") == condition)
        ]

    def export_path(self, run_id, name):
        allowed = {
            "results": "results.json",
            "flips": "flips.json",
            "rules": "cleaning_rules.json",
            "prevalence": "prevalence.json",
            "manifest": "artifact_manifest.json",
            "protocol": "protocol.json",
        }
        if name not in allowed:
            raise DataValidationError("不支持的 Stage5 分析导出")
        path = self.resolve(run_id) / allowed[name]
        if not path.is_file():
            raise ResourceUnavailableError(f"Stage5 artifact 不存在: {allowed[name]}")
        return path
