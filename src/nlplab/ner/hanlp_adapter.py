"""HanLP 字符级 NER adapter；标签映射须由显式配置和实际模型词表共同确认。"""

import json
import os
import re
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.io import write_json
from nlplab.core.schema import Entity


def validate_label_map(label_map: Any, entity_tags: Any) -> None:
    """显式映射必须完整覆盖当前模型，目标限定为课程的 PER/LOC/ORG。"""
    if (
        not isinstance(label_map, dict)
        or not label_map
        or any(not isinstance(key, str) or not key for key in label_map)
        or any(value not in ("PER", "LOC", "ORG") for value in label_map.values())
        or not isinstance(entity_tags, list)
        or any(not isinstance(tag, str) or not tag for tag in entity_tags)
        or len(set(entity_tags)) != len(entity_tags)
        or set(label_map) != set(entity_tags)
    ):
        raise DataValidationError("NER label_map 必须完整匹配模型实体 tagset，并映射至 PER/LOC/ORG")


def model_entity_tagset(component: Any) -> list[str]:
    """读取加载后的 TF/PyTorch 标签词表，移除 BIO/IOBES/BMES 前缀及非实体标签。"""
    vocabs = getattr(component, "vocabs", None)
    vocab = vocabs.get("tag") if isinstance(vocabs, dict) else None
    if vocab is None:
        vocab = getattr(getattr(component, "transform", None), "tag_vocab", None)
    tags = getattr(vocab, "idx_to_token", None)
    if not isinstance(tags, (list, tuple)) or not tags:
        raise DataValidationError("无法读取 HanLP 模型实际标签词表；不能假设其 tagset")
    entity_tags = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise DataValidationError("HanLP tag vocabulary must contain strings")
        if tag in {"O", "<pad>", "<unk>"}:
            continue
        if not isinstance(tag, str) or not re.fullmatch(r"[BIMES]-.+", tag):
            raise DataValidationError(f"不支持的 HanLP 序列标签: {tag!r}")
        entity_tags.add(tag.split("-", 1)[1])
    if not entity_tags:
        raise DataValidationError("HanLP 模型词表不含实体标签")
    return sorted(entity_tags)


class HanLPRecognizer:
    """默认只加载准备好的本地模型，交互请求不会隐式下载权重。"""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        name: str = "hanlp-ner",
        label_map: dict[str, str] | None = None,
        entity_tagset: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.name = name
        self.label_map = dict(label_map) if label_map is not None else {}
        self.entity_tagset = (
            list(entity_tagset) if entity_tagset is not None else sorted(self.label_map)
        )
        if label_map is not None:
            validate_label_map(self.label_map, self.entity_tagset)
        self._component: Any = None
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "HanLPRecognizer":
        os.environ.setdefault(
            "HANLP_HOME", str((settings.root / "artifacts/cache/hanlp").resolve())
        )
        config = settings.root / "artifacts/cache/ner_setup.json"
        if not config.is_file():
            return cls()
        try:
            value = json.loads(config.read_text(encoding="utf-8"))
            # 旧配置没有映射时保持不可用；需重新 setup-ner，不恢复写死映射。
            return cls(
                Path(value["model_path"]),
                name=value["name"],
                label_map=value["label_map"],
                entity_tagset=value["entity_tagset"],
            )
        except KeyError:
            return cls()
        except (ValueError, TypeError) as error:
            raise DataValidationError(f"NER 配置损坏: {config}") from error

    def _load(self) -> None:
        if not self.label_map:
            raise ResourceUnavailableError(
                "HanLP NER 尚未准备或配置缺少模型标签映射：安装 .[ner] 后运行 "
                "python -m nlplab.cli setup-ner"
            )
        if self._component is not None:
            return
        if self.model_path is None or not self.model_path.is_dir():
            raise ResourceUnavailableError(
                "HanLP NER 尚未准备：安装 .[ner] 后运行 python -m nlplab.cli setup-ner"
            )
        try:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import hanlp

            component = hanlp.load(str(self.model_path), devices=-1)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ResourceUnavailableError(f"HanLP NER 加载失败: {error}") from error
        actual_tags = model_entity_tagset(component)
        validate_label_map(self.label_map, actual_tags)
        if actual_tags != sorted(self.entity_tagset):
            raise DataValidationError("HanLP 实际 tagset 与准备记录不一致，请重新 setup-ner")
        self._component = component

    def recognize(self, text: str) -> list[Entity]:
        """分句及长句分块后推理，输出统一原文字符坐标；不依赖 HMM 分词。"""
        if not text:
            return []
        entities: list[Entity] = []
        with self._lock:
            self._load()
            # 保留旧 200 字上限，同时尊重模型真实 WordPiece 预算，避免库静默截断。
            for sentence in re.finditer(r"[^。！？!?\n]+[。！？!?\n]?|[。！？!?\n]", text):
                value = sentence.group()
                for offset, chunk in model_chunks(value, self._component):
                    start_offset = sentence.start() + offset
                    try:
                        raw_entities = self._component(list(chunk))
                    except (OSError, RuntimeError) as error:
                        raise ResourceUnavailableError(f"HanLP NER 推理失败: {error}") from error
                    for raw in raw_entities:
                        if (
                            not isinstance(raw, (tuple, list))
                            or len(raw) != 4
                            or not isinstance(raw[0], str)
                            or not isinstance(raw[1], str)
                            or raw[1] not in self.label_map
                            or type(raw[2]) is not int
                            or type(raw[3]) is not int
                        ):
                            raise DataValidationError("HanLP 返回格式/标签不符合当前模型显式映射")
                        entity = Entity(
                            raw[0],
                            self.label_map[raw[1]],
                            raw[2] + start_offset,
                            raw[3] + start_offset,
                        )
                        entity.validate(text)
                        entities.append(entity)
        return entities


def model_chunks(text: str, component: Any):
    """字符级输入的 WordPiece 分块；所有字符恰好覆盖一次，不截断/改写。"""
    config = getattr(component, "config", {})
    maximum = config.get("max_seq_length") if isinstance(config, dict) else None
    tokenizer = getattr(getattr(component, "transform", None), "tokenizer", None)
    budget = (
        maximum - tokenizer.num_special_tokens_to_add(pair=False) if maximum and tokenizer else None
    )
    start, used = 0, 0
    for index, char in enumerate(text):
        cost = max(1, len(tokenizer.tokenize(char))) if budget is not None else 1
        if budget is not None and cost > budget:
            raise DataValidationError("单字符 WordPiece 数超过模型输入预算")
        if index > start and (
            index - start >= 200 or (budget is not None and used + cost > budget)
        ):
            yield start, text[start:index]
            start, used = index, 0
        used += cost
    if start < len(text):
        yield start, text[start:]


def prepare_ner(settings: Settings, config_path: Path | None = None) -> dict[str, Any]:
    """显式准备指定官方模型，核对真实词表后才保存模型及映射。"""
    config_path = config_path or settings.root / "configs/hanlp_msra_bert.yaml"
    if not config_path.is_file():
        raise DataValidationError(f"NER 模型配置不存在: {config_path}")
    try:
        config_bytes = config_path.read_bytes()
        model_config = yaml.safe_load(config_bytes.decode("utf-8"))
    except (ValueError, yaml.YAMLError) as error:
        raise DataValidationError(f"NER 模型配置损坏: {config_path}") from error
    required = {"model", "name", "expected_entity_tags", "label_map", "tagset_source"}
    if not isinstance(model_config, dict) or set(model_config) != required:
        raise DataValidationError(f"NER 模型配置必须恰好包含 {sorted(required)}")
    for key in ("model", "name", "tagset_source"):
        if not isinstance(model_config[key], str) or not model_config[key]:
            raise DataValidationError(f"NER 配置 {key} 必须为非空字符串")
    validate_label_map(model_config["label_map"], model_config["expected_entity_tags"])
    cache = settings.root / "artifacts/cache/hanlp"
    os.environ.setdefault("HANLP_HOME", str(cache.resolve()))
    try:
        import hanlp
        from hanlp.utils.io_util import get_resource
    except ImportError as error:
        raise ResourceUnavailableError("请先执行 python -m pip install -e '.[ner]'") from error
    model_resource = getattr(hanlp.pretrained.ner, model_config["model"], None)
    if not isinstance(model_resource, str):
        raise DataValidationError(f"未知的 HanLP 预训练 NER 模型: {model_config['model']}")
    try:
        resource = get_resource(model_resource)
        component = hanlp.load(resource, devices=-1)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        raise ResourceUnavailableError(f"NER 模型准备失败，未保存成功状态: {error}") from error
    actual_tags = model_entity_tagset(component)
    validate_label_map(model_config["label_map"], actual_tags)
    if actual_tags != sorted(model_config["expected_entity_tags"]):
        raise DataValidationError("实际 HanLP tagset 与模型配置不一致，未保存成功状态")
    value = {
        "model": model_config["model"],
        "name": model_config["name"],
        "model_path": str(Path(resource).resolve()),
        "label_map": model_config["label_map"],
        "entity_tagset": actual_tags,
        "tagset_source": model_config["tagset_source"],
        "model_config_sha256": sha256(config_bytes).hexdigest(),
    }
    setup_path = settings.root / "artifacts/cache/ner_setup.json"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(setup_path, value)
    return value
