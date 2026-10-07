"""Read-only resource contracts; a partial download is never a ready model."""

import hashlib
import json
from pathlib import Path

from nlplab.core.exceptions import ResourceUnavailableError


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepared_resource(root, name, verify=True):
    base = (root / "artifacts/course_models").resolve()
    try:
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        resource = manifest["resources"][name]
        records = resource["files"]
        if not records:
            raise ValueError("empty manifest")
        for record in records:
            path = (base / record["path"]).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise ValueError("missing or unsafe resource path")
            if path.stat().st_size != record["size_bytes"]:
                raise ValueError("resource size changed")
            if verify and digest(path) != record["sha256"]:
                raise ValueError("resource hash changed")
        return base, resource
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ResourceUnavailableError(f"{name} 尚未完整准备或未通过校验") from error


def bert_snapshot():
    hub = Path.home() / ".cache/huggingface/hub/models--bert-base-chinese/snapshots"
    for path in sorted(hub.glob("*")):
        if all((path / name).is_file() for name in ("model.safetensors", "config.json", "vocab.txt")):
            return path
    raise ResourceUnavailableError("本地 bert-base-chinese safetensors 不可用")


def verified_bert(root, verify=True):
    try:
        reference = json.loads((root / "artifacts/course_models/bert-reference.json").read_text(encoding="utf-8"))
        # Pin revision and content, not the recording machine's user directory.
        revision = reference["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("invalid BERT revision")
        path = Path.home() / ".cache/huggingface/hub/models--bert-base-chinese/snapshots" / revision
        if set(reference["files"]) != {"config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt"}:
            raise ValueError("BERT reference incomplete")
        for name, record in reference["files"].items():
            file = path / name
            if file.stat().st_size != record["size_bytes"] or (verify and digest(file) != record["sha256"]):
                raise ValueError("BERT resource changed")
        return path, reference
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ResourceUnavailableError("本地 BERT 未建立来源记录或未通过哈希校验") from error


def frozen_bilstm(root):
    """Resolve the existing final Run without loading Torch or following latest."""
    run_id = "20260926_185552_0ffff5d1"
    directory = root / "outputs/exp1/extensions" / run_id
    try:
        info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if info.get("run_id") != run_id or info.get("model") != "bilstm-crf" or info.get("phase") != "final" or info.get("is_demo"):
            raise ValueError("not the frozen final BiLSTM")
        model = Path(info["model_path"]).resolve()
        if not model.is_relative_to((root / "artifacts/models").resolve()):
            raise ValueError("model outside frozen model directory")
        if not all((model / name).is_file() for name in ("metadata.json", "state_dict.pt")):
            raise ValueError("incomplete frozen BiLSTM")
        return directory, info, model
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ResourceUnavailableError("冻结 BiLSTM-CRF 资源未准备") from error
