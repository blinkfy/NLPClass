"""检查启动前保存的只读内容指纹；新文件不影响既有内容冻结。"""

import json
from pathlib import Path

from nlplab.core.exceptions import DataValidationError
from nlplab.extensions.benchmark import fingerprint


def verify_frozen(root: Path, snapshots: list[Path]) -> dict:
    expected = {}
    for snapshot in snapshots:
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        # 初始快照为 {files: {...}}；补充快照直接使用路径→SHA256。
        files = data.get("files", data)
        for name, value in files.items():
            if name in expected and expected[name] != value:
                raise DataValidationError(f"冻结快照互相冲突: {name}")
            expected[name] = value
    changed = [
        name
        for name, value in expected.items()
        if not (root / name).is_file() or fingerprint(root / name) != value
    ]
    if changed:
        raise DataValidationError(f"冻结文件被更改: {changed}")
    return {
        "status": "unchanged",
        "file_count": len(expected),
        "snapshots": [str(p) for p in snapshots],
        "changed": [],
    }
