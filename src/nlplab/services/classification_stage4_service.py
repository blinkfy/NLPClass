"""Read-only development analysis assets; no historical Test service dispatch."""

import re

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.features.word_sequence import read_json
from nlplab.services.experiment_service import RunStore


class ClassificationStage4Service:
    _verified = set()

    def __init__(self, settings):
        self.store = RunStore(settings, "exp3")
        self.directory = self.store.directory / "stage4"

    def resolve(self, run_id="latest"):
        if run_id == "latest":
            pointer = self.store.directory / "stage4_latest.json"
            if not pointer.is_file():
                raise ResourceUnavailableError("尚无已完成的 Stage4 开发集学习曲线")
            run_id = read_json(pointer)["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise DataValidationError("Analysis run_id 无效")
        directory = self.directory / run_id
        if not (directory / "run.json").is_file():
            raise ResourceUnavailableError("Stage4 分析不存在或未完成")
        manifest_path = directory / "artifact_manifest.json"
        key = (str(directory), manifest_path.stat().st_mtime_ns)
        if key not in self._verified:
            manifest = read_json(manifest_path)
            for name, digest in manifest["files"].items():
                path = (directory / name).resolve()
                if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
                    raise DataValidationError("Stage4 artifact hash 不符")
            self._verified.add(key)
        return directory

    def list_runs(self):
        return [read_json(path) for path in sorted(self.directory.glob("*/run.json"), reverse=True)]

    def overview(self, run_id="latest", model_id=None):
        directory = self.resolve(run_id)
        curves = read_json(directory / "learning_curves.json")
        compact = []
        for row in curves:
            if model_id is not None and row["model_id"] != model_id:
                continue
            compact.append(
                {
                    **{k: v for k, v in row.items() if k != "replicates"},
                    "replicates": [
                        {
                            "subset_seed": p["subset_seed"],
                            "train_macro_f1": p["train_metrics"]["macro_f1"],
                            "validation_macro_f1": p["validation_metrics"]["macro_f1"],
                            "validation_accuracy": p["validation_metrics"]["accuracy"],
                            "representation": {
                                k: v
                                for k, v in p["representation"].items()
                                if isinstance(v, (int, float))
                            },
                        }
                        for p in row["replicates"]
                    ],
                }
            )
        return {
            "run": read_json(directory / "run.json"),
            "protocol": read_json(directory / "protocol.json"),
            "curves": compact,
            "subsets": read_json(directory / "subset_ladders.json"),
            "sanity_420": read_json(directory / "sanity_420.json"),
            "environment": read_json(directory / "environment.json"),
            "runtimes": {
                k: v for k, v in read_json(directory / "runtimes.json").items() if k != "points"
            },
            "formal_test_evaluation_count": 0,
            "test_access_permitted": False,
        }

    def export_path(self, run_id, name):
        allowed = {
            "protocol": "protocol.json",
            "subsets": "subset_ladders.json",
            "curves": "learning_curves.json",
            "raw": "raw_results.json",
            "per-class": "per_class_curves.json",
            "diagnostics": "representation_diagnostics.json",
            "runtimes": "runtimes.json",
            "manifest": "artifact_manifest.json",
        }
        if name not in allowed:
            raise DataValidationError("不支持的 Stage4 分析导出")
        return self.resolve(run_id) / allowed[name]
