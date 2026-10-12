"""Incrementally publish an exported static site into GitHub Pages /docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = "nlplab-github-pages"
INTEGRITY_NAME = "export-integrity.json"


def _safe_child(root: Path, relative: str, *, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError(f"invalid {context} path: {relative!r}")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or any(":" in part for part in pure.parts)
    ):
        raise ValueError(f"path escapes {context} root: {relative!r}")
    resolved_root = root.resolve()
    candidate = root.joinpath(*pure.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path escapes {context} root: {relative!r}") from error
    if resolved == resolved_root:
        raise ValueError(f"{context} path cannot name its root: {relative!r}")
    if context == "destination":
        cursor = root
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"symlink destination path is not allowed: {relative!r}")
    return resolved


def _read_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} integrity manifest: {path}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError(f"invalid {label} integrity manifest: expected a files list")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_workflow(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return len(parts) >= 2 and parts[0] == ".github" and parts[1] == "workflows"


def publish_site(source: Path, destination: Path) -> dict[str, Any]:
    """Copy only integrity-listed site files, preserving unrelated /docs content."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_manifest_path = _safe_child(source, INTEGRITY_NAME, context="source")
    source_manifest = _read_manifest(source_manifest_path, "source")

    previous_path = _safe_child(destination, INTEGRITY_NAME, context="destination")
    previous: dict[str, Any] | None = None
    if previous_path.is_file():
        previous = _read_manifest(previous_path, "destination")
    previous_owned = set()
    if previous and previous.get("publisher") == PUBLISHER:
        previous_owned = {
            entry["path"]
            for entry in previous["files"]
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and not _is_workflow(entry["path"])
            and entry.get("owned", True)
        }

    # Validate the full inventory and hash every source before changing /docs.
    if source == destination:
        raise ValueError("source and destination must be different directories")
    source_files: dict[str, Path] = {}
    for entry in source_manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid source integrity entry: expected a path")
        relative = entry["path"]
        source_file = _safe_child(source, relative, context="source")
        if _is_workflow(relative):
            continue
        if relative == INTEGRITY_NAME:
            raise ValueError("source integrity must not list itself")
        if relative in source_files:
            raise ValueError(f"duplicate source integrity path: {relative}")
        if not source_file.is_file():
            raise ValueError(f"listed source file is missing: {relative}")
        size = source_file.stat().st_size
        digest = _sha256(source_file)
        if entry.get("bytes") != size or entry.get("sha256") != digest:
            raise ValueError(f"source integrity check failed: {relative}")
        target = _safe_child(destination, relative, context="destination")
        existed = target.exists()
        if existed and relative not in previous_owned:
            if not target.is_file() or target.stat().st_size != size or _sha256(target) != digest:
                raise FileExistsError(
                    f"destination collision with non-publisher file: {relative}"
                )
        source_files[relative] = source_file

    stale = previous_owned - set(source_files)
    stale_targets = {
        relative: _safe_child(destination, relative, context="destination")
        for relative in stale
    }
    # Check all paths from the previous manifest, including non-owned/reused
    # files, before cleanup so a tampered manifest cannot direct filesystem ops.
    if previous and previous.get("publisher") == PUBLISHER:
        for entry in previous["files"]:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                _safe_child(destination, entry["path"], context="destination")

    records = []
    copied = reused = bytes_copied = 0
    for relative, source_file in source_files.items():
        target = _safe_child(destination, relative, context="destination")
        size = source_file.stat().st_size
        owned = relative in previous_owned or not target.exists()
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": _sha256(source_file),
                "owned": owned,
            }
        )

    new_manifest = {
        "publisher": PUBLISHER,
        "files": sorted(records, key=lambda item: item["path"]),
    }
    manifest_bytes = (json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    if previous is None and previous_path.exists():
        if previous_path.read_bytes() != manifest_bytes:
            raise FileExistsError(
                f"destination collision with non-publisher file: {INTEGRITY_NAME}"
            )
    if previous is not None and previous.get("publisher") != PUBLISHER:
        if previous_path.read_bytes() != manifest_bytes:
            raise FileExistsError(
                f"destination collision with non-publisher file: {INTEGRITY_NAME}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    removed = 0
    for relative, target in stale_targets.items():
        if target.is_file() or target.is_symlink():
            target.unlink()
            removed += 1

    for relative, source_file in source_files.items():
        target = _safe_child(destination, relative, context="destination")
        if (
            target.is_file()
            and target.stat().st_size == source_file.stat().st_size
            and _sha256(target) == _sha256(source_file)
        ):
            reused += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        copied += 1
        bytes_copied += source_file.stat().st_size
    integrity_target = destination / INTEGRITY_NAME
    integrity_target.write_bytes(manifest_bytes)
    return {
        "source_files": len(source_files),
        "copied": copied,
        "reused": reused,
        "removed": removed,
        "bytes_copied": bytes_copied,
        "destination": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build/github-pages")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    print(json.dumps(publish_site(args.source, args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
