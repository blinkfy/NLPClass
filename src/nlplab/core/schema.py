"""跨算法与 Web 的透明数据结构；跨度使用 Python 字符索引 [start, end)。"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

from nlplab.core.exceptions import DataValidationError


def text_sha256(text: str) -> str:
    """计算原始文本 UTF-8 内容的稳定指纹，不改写文本。"""
    return sha256(text.encode("utf-8")).hexdigest()


def require_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} 必须是非空字符串")


@dataclass
class Document:
    """统一主语料；体裁 source_type 与主题 primary_label 独立。"""

    id: str
    text: str
    source_type: str
    primary_label: str
    source_title: str
    source_url: str
    sha256: str = ""
    secondary_labels: list[str] = field(default_factory=list)
    is_demo: bool = False

    def __post_init__(self) -> None:
        for name in ("id", "text", "source_type", "primary_label", "source_title", "source_url"):
            require_text(getattr(self, name), name)
        if type(self.is_demo) is not bool:
            raise DataValidationError("is_demo 必须为布尔值")
        if not isinstance(self.sha256, str):
            raise DataValidationError("sha256 必须为字符串")
        if self.source_type == "synthetic" and not self.is_demo:
            raise DataValidationError("synthetic 语料必须明确标记 is_demo=true")
        if not isinstance(self.secondary_labels, list) or any(
            not isinstance(label, str) or not label.strip() for label in self.secondary_labels
        ):
            raise DataValidationError("secondary_labels 必须是字符串列表")
        url = urlparse(self.source_url)
        if not (url.scheme in {"http", "https"} and url.netloc) and not (
            self.is_demo and url.scheme == "demo"
        ):
            raise DataValidationError("真实语料必须记录 http/https 来源 URL")
        fingerprint = text_sha256(self.text)
        if self.sha256 and self.sha256 != fingerprint:
            raise DataValidationError("sha256 与原始文本不一致")
        self.sha256 = fingerprint

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    """统一命名实体；end 为排他边界，score 不可用时为 None。"""

    text: str
    type: str
    start: int
    end: int
    score: float | None = None

    def validate(self, text: str) -> None:
        if self.type not in {"PER", "LOC", "ORG"}:
            raise DataValidationError(f"不支持的实体类型: {self.type}")
        if type(self.start) is not int or type(self.end) is not int:
            raise DataValidationError("实体边界必须是整数")
        if not 0 <= self.start < self.end <= len(text):
            raise DataValidationError("实体跨度越界")
        if text[self.start : self.end] != self.text:
            raise DataValidationError("实体文本与跨度不一致")
        if self.score is not None and not 0 <= self.score <= 1:
            raise DataValidationError("实体置信度必须在 [0,1] 内")


@dataclass
class SequenceSample:
    """BMES 不单独保存；entities=None 表示尚未进行 NER Gold 标注。"""

    sample_id: str
    doc_id: str
    text: str
    gold_tokens: list[str]
    entities: list[Entity] | None = None
    is_demo: bool = False
    source_start: int | None = None
    source_end: int | None = None
    status: str | None = None
    revision_count: int | None = None
    notes: str | None = None
    selection_reasons: list[str] | None = None
    primary_label: str | None = None
    source_type: str | None = None
    review_history: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "doc_id", "text"):
            require_text(getattr(self, name), name)
        if type(self.is_demo) is not bool:
            raise DataValidationError("is_demo 必须为布尔值")
        if not isinstance(self.gold_tokens, list) or not self.gold_tokens:
            raise DataValidationError("gold_tokens 必须是非空列表")
        if any(not isinstance(token, str) or not token for token in self.gold_tokens):
            raise DataValidationError("Gold 分词不得包含空词或非字符串")
        if "".join(self.gold_tokens) != self.text:
            raise DataValidationError("''.join(gold_tokens) 必须等于 text")
        if self.source_start is not None or self.source_end is not None:
            if (
                type(self.source_start) is not int
                or type(self.source_end) is not int
                or self.source_start < 0
                or self.source_end - self.source_start != len(self.text)
            ):
                raise DataValidationError("source span 必须是与 text 长度一致的 code-point 边界")
        if self.status is not None and self.status not in {
            "pending",
            "in_progress",
            "done",
            "reviewed",
        }:
            raise DataValidationError("未知标注状态")
        if self.revision_count is not None and (
            type(self.revision_count) is not int or self.revision_count < 0
        ):
            raise DataValidationError("revision_count 必须为非负整数")
        if self.notes is not None and not isinstance(self.notes, str):
            raise DataValidationError("notes 必须为字符串")
        if self.selection_reasons is not None and (
            not isinstance(self.selection_reasons, list)
            or any(not isinstance(reason, str) for reason in self.selection_reasons)
        ):
            raise DataValidationError("selection_reasons 必须为字符串列表")
        if self.entities is not None:
            if not isinstance(self.entities, list):
                raise DataValidationError("entities 必须为列表或 null")
            for entity in self.entities:
                entity.validate(self.text)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # 旧 demo/MSR 记录的 JSON 形状与记录指纹保持完全一致。
        for name in (
            "source_start",
            "source_end",
            "status",
            "revision_count",
            "notes",
            "selection_reasons",
            "primary_label",
            "source_type",
            "review_history",
        ):
            if value[name] is None:
                del value[name]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SequenceSample":
        value = dict(value)
        if value.get("entities") is not None:
            value["entities"] = [Entity(**entity) for entity in value["entities"]]
        return cls(**value)


@dataclass
class SegmentationResult:
    model: str
    tokens: list[str]
    tags: list[str]
