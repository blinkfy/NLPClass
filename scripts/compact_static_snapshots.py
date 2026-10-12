"""Share repeated frozen model metadata without changing response values."""

import hashlib
import json
from pathlib import Path

SHARED_FIELDS = {"raw_vocabulary", "selected_features", "chi2_metadata"}


def compact_snapshots(destination, manifest):
    destination = Path(destination)
    shared = destination / "data/shared"
    shared.mkdir(exist_ok=True)
    before = after = 0
    shared_files = set()

    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def compact(value):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key in SHARED_FIELDS and isinstance(child, (dict, list)):
                    data = encode(child).encode("utf-8")
                    if len(data) >= 32768:
                        digest = hashlib.sha256(data).hexdigest()
                        target = shared / f"{digest}.json"
                        if not target.exists():
                            target.write_bytes(data)
                        shared_files.add(target)
                        result[key] = {"__nlplab_static_ref__": f"data/shared/{digest}.json"}
                        continue
                result[key] = compact(child)
            return result
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    for file in {entry["file"] for entry in manifest["responses"].values()}:
        target = destination / file
        if target.suffix != ".json":
            continue
        data = target.read_bytes()
        before += len(data)
        value = json.loads(data)
        transformed = compact(value)
        content = encode(transformed).encode("utf-8")
        if content != data:
            target.write_bytes(content)
        after += len(content)
    after += sum(file.stat().st_size for file in shared_files)
    return {"before_bytes": before, "after_bytes": after, "shared_files": len(shared_files)}
