import json
from pathlib import Path

import pytest

from nlplab.core.exceptions import DataValidationError
from nlplab.extensions.benchmark import fingerprint
from nlplab.extensions.integrity import verify_frozen


def test_freeze_detects_mutation(tmp_path):
    path = tmp_path / "frozen.txt"
    path.write_text("冻结", encoding="utf-8")
    snapshot = tmp_path / "before.json"
    snapshot.write_text(json.dumps({"frozen.txt": fingerprint(path)}), encoding="utf-8")
    assert verify_frozen(tmp_path, [snapshot])["file_count"] == 1
    (tmp_path / "new.txt").write_text("新增", encoding="utf-8")
    assert verify_frozen(tmp_path, [snapshot])["status"] == "unchanged"
    path.write_text("改变", encoding="utf-8")
    with pytest.raises(DataValidationError):
        verify_frozen(tmp_path, [snapshot])


def test_local_frozen_hmm_msr_msra_unchanged():
    root = Path(__file__).resolve().parents[1]
    snapshots = sorted((root / "artifacts/cache").glob("extensions_frozen_before_*.json"))
    if not snapshots:
        pytest.skip("无本地正式数据与启动快照")
    snapshots += [root / "artifacts/cache/extensions_frozen_supplement.json"]
    assert verify_frozen(root, snapshots)["file_count"] >= 249
