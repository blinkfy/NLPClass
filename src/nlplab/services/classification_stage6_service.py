"""Read-only Stage6 uncertainty analysis artifacts."""

import re

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.features.word_sequence import read_json


class ClassificationStage6Service:
    """Expose frozen Stage6 summaries without fitting or inference routes."""

    _verified = set()
    _run_id_pattern = re.compile(r"[A-Za-z0-9_-]+")
    _aligned_fields = ("document_id", "gold", "C0", "C1", "C2", "C3")

    def __init__(self, settings):
        self.directory = settings.outputs / "exp3" / "stage6"

    def resolve(self, run_id="latest"):
        if run_id == "latest":
            pointer = self.directory.parent / "stage6_latest.json"
            if not pointer.is_file():
                raise ResourceUnavailableError("尚无已完成的 Stage6 不确定性分析")
            pointer_data = read_json(pointer)
            if not isinstance(pointer_data, dict):
                raise DataValidationError("Stage6 latest pointer 格式无效")
            run_id = pointer_data.get("run_id")

        if not isinstance(run_id, str) or not self._run_id_pattern.fullmatch(run_id):
            raise DataValidationError("Stage6 run_id 无效")

        directory = self.directory / run_id
        root = self.directory.resolve()
        if not directory.resolve().is_relative_to(root):
            raise DataValidationError("Stage6 run 路径无效")
        run_root = directory.resolve()
        run_path = (directory / "run.json").resolve()
        if not run_path.is_relative_to(run_root):
            raise DataValidationError("Stage6 run metadata 路径无效")
        if not run_path.is_file():
            raise ResourceUnavailableError("Stage6 分析不存在或未完成")

        manifest_path = (directory / "artifact_manifest.json").resolve()
        if not manifest_path.is_relative_to(run_root):
            raise DataValidationError("Stage6 artifact manifest 路径无效")
        if not manifest_path.is_file():
            raise ResourceUnavailableError("Stage6 artifact manifest 不存在")
        manifest = read_json(manifest_path)
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            raise DataValidationError("Stage6 artifact manifest 格式无效")

        file_state = []
        for name, digest in sorted(files.items()):
            if not isinstance(name, str) or name in {"", ".", "..", "artifact_manifest.json"}:
                raise DataValidationError("Stage6 artifact manifest 文件名无效")
            if not isinstance(digest, str):
                raise DataValidationError("Stage6 artifact manifest SHA256 格式无效")
            path = (directory / name).resolve()
            if not path.is_relative_to(run_root):
                raise DataValidationError("Stage6 artifact 路径无效")
            if not path.is_file():
                raise ResourceUnavailableError(f"Stage6 artifact 不存在: {name}")
            stat = path.stat()
            file_state.append((name, stat.st_mtime_ns, stat.st_size, digest))

        key = (
            str(directory),
            manifest_path.stat().st_mtime_ns,
            manifest_path.stat().st_size,
            tuple(file_state),
        )
        if key not in self._verified:
            for name, _, _, digest in file_state:
                if file_sha256(directory / name) != digest:
                    raise DataValidationError(f"Stage6 artifact hash 不符: {name}")
            self._verified.add(key)
        return directory

    def overview(self, run_id="latest"):
        directory = self.resolve(run_id)
        self._require_manifested(
            directory, ("run.json", "statistics.json", "alignment_audit.json", "protocol.json")
        )
        statistics = read_json(directory / "statistics.json")
        alignment = read_json(directory / "alignment_audit.json")
        protocol = read_json(directory / "protocol.json")
        if not isinstance(statistics, dict):
            raise DataValidationError("Stage6 statistics 格式无效")
        if not isinstance(alignment, dict):
            raise DataValidationError("Stage6 alignment audit 格式无效")
        if not isinstance(protocol, dict):
            raise DataValidationError("Stage6 protocol 格式无效")
        return {
            "run": read_json(directory / "run.json"),
            "statistics": statistics,
            "alignment": alignment,
            "protocol": protocol,
        }

    def aligned_labels(self, run_id="latest"):
        """Return only the 90 document IDs and gold/predicted labels."""
        directory = self.resolve(run_id)
        self._require_manifested(directory, ("aligned_predictions.json",))
        rows = read_json(directory / "aligned_predictions.json")
        if not isinstance(rows, list) or len(rows) != 90:
            raise DataValidationError("Stage6 aligned predictions 必须包含 90 篇记录")
        result = []
        seen_ids = set()
        for row in rows:
            if not isinstance(row, dict):
                raise DataValidationError("Stage6 aligned prediction 格式无效")
            compact = {field: row.get(field) for field in self._aligned_fields}
            if any(not isinstance(value, str) or not value for value in compact.values()):
                raise DataValidationError("Stage6 aligned prediction 缺少类别标签或 document_id")
            if compact["document_id"] in seen_ids:
                raise DataValidationError("Stage6 aligned predictions 含重复 document_id")
            seen_ids.add(compact["document_id"])
            result.append(compact)
        return result

    @staticmethod
    def _require_manifested(directory, names):
        manifest = read_json(directory / "artifact_manifest.json")
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            raise DataValidationError("Stage6 artifact manifest 格式无效")
        for name in names:
            if name not in files:
                raise DataValidationError(f"Stage6 artifact 未纳入 manifest: {name}")

    def export_path(self, run_id, name):
        allowed = {
            "protocol": "protocol.json",
            "statistics": "statistics.json",
            "alignment": "alignment_audit.json",
            "manifest": "artifact_manifest.json",
            "bootstrap": "bootstrap_configuration.json",
            "aligned": "aligned_predictions.json",
        }
        if name not in allowed:
            raise DataValidationError("不支持的 Stage6 分析导出")
        directory = self.resolve(run_id)
        filename = allowed[name]
        if filename != "artifact_manifest.json":
            self._require_manifested(directory, (filename,))
        path = directory / filename
        if not path.is_file():
            raise ResourceUnavailableError(f"Stage6 artifact 不存在: {allowed[name]}")
        return path
