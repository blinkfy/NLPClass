"""JSONL 验证、稳定去重和主语料与标注关联校验。"""

import json
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.schema import Document, SequenceSample

T = TypeVar("T")


def read_jsonl(path: Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    return list(iter_jsonl(path, factory))


def iter_jsonl(path: Path, factory: Callable[[dict[str, Any]], T]) -> Iterator[T]:
    """逐行严格读取，错误保留路径与行号，不跳过错误记录。"""
    if not path.is_file():
        raise ResourceUnavailableError(f"数据文件不存在: {path}")
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataValidationError("每行必须是 JSON object")
                record = factory(value)
                count += 1
                yield record
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise DataValidationError(f"{path}:{number}: {error}") from error
            except DataValidationError as error:
                raise DataValidationError(f"{path}:{number}: {error}") from error
    if not count:
        raise DataValidationError(f"数据文件为空: {path}")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """UTF-8 无损输出；调用方负责保护旧实验与原始文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def duplicate_key(text: str) -> str:
    """仅生成去重键，原文与用于字符索引的文本保持不变。"""
    return " ".join(unicodedata.normalize("NFC", text).split())


def deduplicate(documents: list[Document]) -> tuple[list[Document], list[dict[str, str]]]:
    """返回唯一文档与可检查的重复报告；重复 ID 直接报错。"""
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    unique, duplicates = [], []
    for document in documents:
        if document.id in seen_ids:
            raise DataValidationError(f"重复 document id: {document.id}")
        seen_ids.add(document.id)
        key = duplicate_key(document.text)
        if key in seen_text:
            duplicates.append({"id": document.id, "duplicate_of": seen_text[key]})
        else:
            unique.append(document)
            seen_text[key] = document.id
    return unique, duplicates


def load_documents(path: Path) -> list[Document]:
    return read_jsonl(path, lambda value: Document(**value))


def load_annotations(path: Path, documents: list[Document]) -> list[SequenceSample]:
    """保证自建 Gold 可追溯至主语料中的原始句子。"""
    samples = read_jsonl(path, SequenceSample.from_dict)
    document_map = {document.id: document for document in documents}
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise DataValidationError(f"重复 sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        if sample.doc_id not in document_map:
            raise DataValidationError(f"{sample.sample_id}: doc_id 不存在于主语料")
        document_text = document_map[sample.doc_id].text
        if sample.source_start is not None:
            matches = document_text[sample.source_start : sample.source_end] == sample.text
        else:
            matches = sample.text in document_text
        if not matches:
            raise DataValidationError(f"{sample.sample_id}: 标注文本不是文档原文中的句子")
    return samples
