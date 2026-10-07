"""Record an existing local BERT snapshot; no download, copy or inference."""
import json
from pathlib import Path
from nlplab.course_neural.resources import bert_snapshot, digest

root = Path(__file__).resolve().parents[1]
path = bert_snapshot()
target = root / "artifacts/course_models/bert-reference.json"
if target.exists():
    raise SystemExit("BERT reference already exists; do not overwrite its baseline")
record = {"model": "bert-base-chinese", "revision": path.name, "path": str(path),
          "source": "https://huggingface.co/google-bert/bert-base-chinese", "training": False,
          "note": "Existing local snapshot recorded during course implementation; local fingerprint, not an upstream checksum claim",
          "files": {name: {"size_bytes": (path/name).stat().st_size, "sha256": digest(path/name)}
                    for name in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt")}}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(record["revision"], record["files"]["model.safetensors"]["sha256"])
