"""Download the pinned local models used by course modules 07 and 08.

This is an explicit preparation command. It does not install packages or load
models. Runtime code should consume only the files recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT / "artifacts" / "course_models"
MANIFEST_PATH = RESOURCE_ROOT / "manifest.json"

QWEN_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION = "c89bee90d9f811437d9735454613c35b4a3c4dc8"
QWEN_EXPECTED_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
QWEN_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)
HANLP_DEP_URL = (
    "https://file.hankcs.com/hanlp/dep/"
    "ctb9_dep_electra_small_20220216_100306.zip"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NLPClass-course-model-preparation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return result


def download_file(url: str, target: Path, expected_sha256: str | None = None) -> str:
    """Stream one file to a sibling partial file, resuming an interrupted transfer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        existing_hash = sha256_file(target)
        if expected_sha256 is None or existing_hash == expected_sha256:
            print(f"reuse {target.relative_to(ROOT)} sha256={existing_hash}", flush=True)
            return existing_hash
        print(f"existing hash mismatch; replacing after verified download: {target}", flush=True)

    partial = target.with_name(target.name + ".part")
    curl_partial = target.with_name(target.name + ".partial")
    if not partial.exists() and curl_partial.exists():
        partial = curl_partial
    print(f"download {target.relative_to(ROOT)} from {url}", flush=True)

    # A bounded number of ranged continuations handles servers/proxies that end
    # a response early without an HTTP error. It never loops indefinitely.
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "NLPClass-course-model-preparation/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = False
                total: int | None = None
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)$", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise RuntimeError(f"Unexpected Content-Range while resuming {target}")
                    append = offset > 0
                    total = int(match.group(3)) if match.group(3) != "*" else None
                elif offset:
                    # Do not let a server that ignored Range overwrite the
                    # preserved partial; fail with it intact.
                    raise RuntimeError(f"Server did not honor Range; partial kept at {partial}")
                else:
                    content_length = response.headers.get("Content-Length")
                    total = int(content_length) if content_length and content_length.isdigit() else None

                received = offset
                last_report = time.monotonic()
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            size = f"{received / 1024**2:.1f} MiB"
                            if total is not None:
                                size += f" / {total / 1024**2:.1f} MiB ({received / total * 100:.1f}%)"
                            print(f"progress {target.name}: {size}", flush=True)
                            last_report = now
                    output.flush()
                    os.fsync(output.fileno())
        except urllib.error.HTTPError as error:
            if error.code == 416 and partial.is_file():
                content_range = error.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes \*/(\d+)", content_range)
                total = int(match.group(1)) if match else None
                actual_size = partial.stat().st_size
                actual_hash = sha256_file(partial)
                if total == actual_size and (
                    expected_sha256 is None or actual_hash == expected_sha256
                ):
                    os.replace(partial, target)
                    print(f"saved complete ranged file {target.relative_to(ROOT)} sha256={actual_hash}", flush=True)
                    return actual_hash
            raise RuntimeError(f"Download failed for {url}: HTTP {error.code}; partial kept") from error
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            print(f"transfer attempt {attempt}/3 ended early; partial kept for Range resume", flush=True)
            if attempt == 3:
                raise RuntimeError(f"Download failed for {url}: {error}") from error
            continue

        actual_size = partial.stat().st_size
        if total is not None and actual_size != total:
            print(
                f"incomplete {target.name}: {actual_size} / {total} bytes; "
                f"resuming ({attempt}/3)",
                flush=True,
            )
            if actual_size > total:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Received more than Content-Length for {target.name}")
            if attempt < 3:
                continue
            raise RuntimeError(f"Incomplete download after 3 attempts; partial kept at {partial}")

        actual_hash = sha256_file(partial)
        if expected_sha256 and actual_hash != expected_sha256:
            if total is None and attempt < 3:
                print("unknown total length with hash mismatch; trying one ranged continuation", flush=True)
                continue
            if total is not None:
                partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual_hash}"
            )
        os.replace(partial, target)
        print(f"saved {target.relative_to(ROOT)} sha256={actual_hash}", flush=True)
        return actual_hash

    raise RuntimeError(f"Download failed after bounded attempts: {url}")


def qwen_resource() -> dict[str, Any]:
    print(f"checking pinned Hugging Face revision {QWEN_REVISION}", flush=True)
    api = f"https://huggingface.co/api/models/{QWEN_REPO}/revision/{QWEN_REVISION}"
    metadata = get_json(api)
    resolved_revision = metadata.get("sha")
    if resolved_revision != QWEN_REVISION:
        raise RuntimeError(
            f"Hugging Face resolved unexpected revision: {resolved_revision!r}"
        )
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise RuntimeError("Hugging Face revision metadata has no siblings list")
    available = {
        item.get("rfilename") for item in siblings if isinstance(item, dict)
    }
    missing = [name for name in QWEN_REQUIRED_FILES if name not in available]
    if missing:
        raise RuntimeError(f"Pinned Qwen revision is missing required files: {missing}")

    license_name = next(
        (
            name
            for name in sorted(available)
            if PurePosixPath(name).parent == PurePosixPath(".")
            and PurePosixPath(name).name.lower() in {"license", "license.txt", "license.md"}
        ),
        None,
    )
    folder = RESOURCE_ROOT / "qwen2.5-0.5b-instruct"
    selected_files = [*QWEN_REQUIRED_FILES]
    if license_name:
        selected_files.append(license_name)
    records = []
    for name in selected_files:
        url = f"https://huggingface.co/{QWEN_REPO}/resolve/{QWEN_REVISION}/{name}"
        expected = QWEN_EXPECTED_SHA256 if name == "model.safetensors" else None
        target = folder / name
        digest = download_file(url, target, expected)
        records.append(
            {
                "path": target.relative_to(RESOURCE_ROOT).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
                "source_url": url,
            }
        )

    card = metadata.get("cardData") or {}
    license_id = card.get("license") if isinstance(card, dict) else None
    return {
        "source": f"https://huggingface.co/{QWEN_REPO}",
        "revision": QWEN_REVISION,
        "license": {
            "spdx_id": license_id,
            "file": license_name,
            "note": "License identifier from the pinned Hugging Face model metadata.",
        },
        "format": "safetensors; remote code disabled",
        "files": records,
    }


def safe_extract(zip_path: Path, destination: Path) -> list[dict[str, Any]]:
    resource_root = RESOURCE_ROOT.resolve()
    if not destination.resolve().is_relative_to(resource_root):
        raise RuntimeError(f"Extraction target escapes the new resource directory: {destination}")
    staging = destination.with_name(destination.name + ".partial")
    if destination.exists():
        raise RuntimeError(f"Extraction target already exists: {destination}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    destination_root = staging.resolve()

    try:
        with zipfile.ZipFile(zip_path) as archive:
            entries = archive.infolist()
            if not entries:
                raise RuntimeError("HanLP archive is empty")
            for entry in entries:
                raw_name = entry.filename.replace("\\", "/")
                member = PurePosixPath(raw_name)
                if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
                    raise RuntimeError(f"Unsafe path in HanLP archive: {entry.filename!r}")
                unix_mode = (entry.external_attr >> 16) & 0xFFFF
                if unix_mode and (unix_mode & 0o170000) == 0o120000:
                    raise RuntimeError(f"Symlinks are not accepted in HanLP archive: {entry.filename!r}")
                target = (destination_root / Path(*member.parts)).resolve()
                if not target.is_relative_to(destination_root):
                    raise RuntimeError(f"HanLP archive path escapes extraction target: {entry.filename!r}")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        basenames = {path.name for path in files}
        if "config.json" not in basenames or not any(
            name in basenames
            for name in ("model.pt", "model.h5", "pytorch_model.bin", "model.safetensors")
        ):
            raise RuntimeError("HanLP archive lacks a model config or recognizable weight file")
        os.replace(staging, destination)
        records = []
        for path in sorted(path for path in destination.rglob("*") if path.is_file()):
            resolved = path.resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise RuntimeError(f"Extracted file escaped its target: {path}")
            records.append(
                {
                    "path": path.relative_to(RESOURCE_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return records
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def hanlp_resource(previous: dict[str, Any] | None) -> dict[str, Any]:
    folder = RESOURCE_ROOT / "hanlp-ctb9-dep-electra-small"
    archive_path = folder / "ctb9_dep_electra_small_20220216_100306.zip"
    archive_hash = download_file(HANLP_DEP_URL, archive_path)
    extracted = folder / f"extracted-{archive_hash[:12]}"
    previous_matches = (
        previous is not None
        and previous.get("archive", {}).get("sha256") == archive_hash
        and previous.get("extracted_root") == extracted.relative_to(RESOURCE_ROOT).as_posix()
        and isinstance(previous.get("files"), list)
    )
    if previous_matches:
        extracted_files = previous["files"]
        expected_paths = set()
        for record in extracted_files:
            relative = Path(record["path"])
            path = (RESOURCE_ROOT / relative).resolve()
            if not path.is_relative_to(extracted.resolve()) or not path.is_file():
                raise RuntimeError(f"Manifested HanLP file is missing or escapes target: {record['path']}")
            if sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"Manifested HanLP file hash changed: {record['path']}")
            expected_paths.add(path)
        actual_paths = {path.resolve() for path in extracted.rglob("*") if path.is_file()}
        if actual_paths != expected_paths:
            raise RuntimeError("HanLP extraction differs from the saved file manifest")
        print(f"reuse verified HanLP extraction {extracted.relative_to(ROOT)}", flush=True)
    else:
        if extracted.exists():
            raise RuntimeError(
                f"Extraction target exists without a matching manifest: {extracted}"
            )
        print(f"extracting archive under {extracted.relative_to(ROOT)}", flush=True)
        extracted_files = safe_extract(archive_path, extracted)
    bundled_license = next(
        (record for record in extracted_files if Path(record["path"]).name.lower().startswith("license")),
        None,
    )
    return {
        "source": HANLP_DEP_URL,
        "version": "20220216_100306",
        "license": {
            "file": bundled_license["path"] if bundled_license else None,
            "note": (
                "See the bundled license file."
                if bundled_license
                else "Archive contains no license file; consult HanLP upstream model terms."
            ),
            "reference": "https://hanlp.hankcs.com/docs/install.html",
        },
        "archive": {
            "path": archive_path.relative_to(RESOURCE_ROOT).as_posix(),
            "size_bytes": archive_path.stat().st_size,
            "sha256": archive_hash,
        },
        "extracted_root": extracted.relative_to(RESOURCE_ROOT).as_posix(),
        "files": extracted_files,
    }


def save_manifest(resources: dict[str, Any]) -> None:
    atomic_json(
        MANIFEST_PATH,
        {
            "schema_version": 1,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "runtime_note": "Prepared files only; no model was loaded or executed.",
            "resources": resources,
        },
    )


def prepare() -> dict[str, Any]:
    RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    prior_resources: dict[str, Any] = {}
    if MANIFEST_PATH.is_file():
        try:
            value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("resources"), dict):
                prior_resources = value["resources"]
        except (OSError, json.JSONDecodeError):
            print("existing manifest is unreadable; rebuilding after hash checks", flush=True)

    # Complete the smaller 07 parser resource first and commit its manifest
    # independently, so a later large Qwen transfer cannot hide it.
    resources = {}
    hanlp = hanlp_resource(prior_resources.get("hanlp-ctb9-dep-electra-small"))
    resources["hanlp-ctb9-dep-electra-small"] = hanlp
    save_manifest(resources)
    qwen = qwen_resource()
    resources["qwen2.5-0.5b-instruct"] = qwen
    save_manifest(resources)
    return {
        "schema_version": 1,
        "resources": resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        value = prepare()
    except Exception as error:
        print(f"course model preparation failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(f"manifest saved: {MANIFEST_PATH.relative_to(ROOT)}", flush=True)
    print(f"prepared resources: {', '.join(value['resources'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
