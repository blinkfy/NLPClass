"""严格的无标签聚类输入；标签关联仅属于事后评价层。"""

import json
from dataclasses import dataclass
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.data.course import file_sha256


@dataclass(frozen=True, slots=True)
class ClusteringDocument:
    id: str
    text: str


def load_clustering_input(path: Path, *, manifest_path: Path | None = None):
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["file_sha256"].get(path.name)
        if expected is None or file_sha256(path) != expected:
            raise DataValidationError("聚类输入与冻结 split manifest 不符")
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"id", "text"}:
            raise DataValidationError("聚类输入必须严格只有 id/text；拒绝 label 或其他字段")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in row):
            raise DataValidationError("聚类 id/text 必须是非空字符串")
        documents.append(ClusteringDocument(**row))
    if not documents or len({doc.id for doc in documents}) != len(documents):
        raise DataValidationError("聚类输入为空或 id 重复")
    return documents
