"""Publish captured real API diagnostics without another model forward."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nlplab.course_neural.resources import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S") + "_diagnostics"
    folder = root / "outputs/course_deep" / run_id
    records = {}
    for name in ("bilstm", "bert", "llm", "extensions"):
        path = args.responses / f"{name}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if "environment" not in record or record.get("source", {}).get("training") is not False:
            raise ValueError(f"{name}: missing actual runtime/source contract")
        if name == "bert":
            ref = json.loads((root / "artifacts/course_models/bert-reference.json").read_text(encoding="utf-8"))
            if record["source"]["weights_sha256"] != ref["files"]["model.safetensors"]["sha256"]:
                raise ValueError("Captured BERT weights do not match source reference")
            if any(c["max_softmax_error"] > 1e-5 for c in record["checks"]):
                raise ValueError("BERT numerical check failed")
        if name == "bilstm" and not record["trace"]["numeric_trace"]["verified_path"]:
            raise ValueError("CRF path not verified")
        records[name] = record
    folder.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, record in records.items():
        path = folder / f"{name}.json"
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        hashes[path.name] = digest(path)
    sources = ["src/nlplab/course_neural/worker.py", "src/nlplab/course_neural/crf_trace.py",
               "src/nlplab/course_neural/resources.py", "app/api/course_modules.py",
               "src/nlplab/services/course_neural_service.py", "docs/course_modules_07_08_plan.md"]
    manifest = {"schema_version": 1, "run_id": run_id, "created_at": now.isoformat(),
                "scope": "Captured single-text API teaching diagnostics, not a formal benchmark",
                "training": False, "old_test_access": False, "files": hashes,
                "evidence_directory": str(args.responses.resolve()),
                "source_hashes_at_publication": {name: digest(root/name) for name in sources},
                "resource_references": {name: digest(root/"artifacts/course_models"/name)
                                        for name in ("manifest.json", "bert-reference.json")}}
    (folder / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"run_id":run_id,"path":str(folder),"forward_count":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
