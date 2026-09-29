"""HanLP CTB9 POS 模型适配；保留原始标签与原文 code-point 跨度。"""

import json
import os
from collections import Counter
from pathlib import Path
from threading import Lock

from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError

POS_MODEL_URL = "https://file.hankcs.com/hanlp/pos/pos_ctb_electra_small_20220215_111944.zip"
POS_TAG_EXPLANATIONS = {
    "NN": "普通名词",
    "NR": "专有名词",
    "NT": "时间名词",
    "VV": "普通动词",
    "VA": "形容词性谓词",
    "VC": "系动词",
    "VE": "存在动词",
    "JJ": "名词前修饰语",
    "PN": "代词",
    "AD": "副词",
    "PU": "标点",
}


def coarse_category(tag: str) -> str:
    """中文粗类仅供配色；HanLP 原始 CTB 标签保持不变。"""
    if tag in {"NN", "NR", "NT"}:
        return "noun"
    if tag in {"VV", "VC", "VE"}:
        return "verb"
    if tag in {"VA", "JJ"}:
        return "adjective"
    if tag == "PN":
        return "pronoun"
    if tag == "AD":
        return "adverb"
    return "other"


def normalize_pos(text: str, tokens: list[str], labels: list[str]) -> list[dict]:
    if "".join(tokens) != text:
        raise DataValidationError("POS 分词必须无损拼接回原文")
    expected = [token for token in tokens if not token.isspace()]
    if len(labels) != len(expected) or any(
        not isinstance(label, str) or not label for label in labels
    ):
        raise DataValidationError("HanLP POS 标签与非空白词数量不一致")
    result, position, tag_index = [], 0, 0
    for token in tokens:
        end = position + len(token)
        if not token.isspace():
            tag = labels[tag_index]
            result.append(
                {
                    "word": token,
                    "pos": tag,
                    "category": coarse_category(tag),
                    "start": position,
                    "end": end,
                }
            )
            tag_index += 1
        position = end
    if any(text[item["start"] : item["end"]] != item["word"] for item in result):
        raise DataValidationError("POS 字符跨度与原文不一致")
    return result


class HanLPPosTagger:
    name = "hanlp-ctb9-pos-electra-small"

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self._component = None
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "HanLPPosTagger":
        config = settings.root / "artifacts/cache/pos_setup.json"
        if not config.is_file():
            return cls()
        try:
            value = json.loads(config.read_text(encoding="utf-8"))
            if value["source_url"] != POS_MODEL_URL:
                raise DataValidationError("POS 配置来源与固定模型不一致")
            return cls(Path(value["model_path"]))
        except (KeyError, ValueError, TypeError) as error:
            raise DataValidationError("POS 模型配置损坏") from error

    def _load(self) -> None:
        if self._component is not None:
            return
        if self.model_path is None or not self.model_path.is_dir():
            raise ResourceUnavailableError("HanLP POS 模型尚未准备；请运行课程 POS setup")
        try:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import hanlp

            self._component = hanlp.load(str(self.model_path), devices=-1)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ResourceUnavailableError(f"HanLP POS 模型无法加载: {error}") from error

    def tag(self, text: str, tokens: list[str]) -> dict:
        words = [token for token in tokens if not token.isspace()]
        if not words:
            return {"model": self.name, "items": [], "counts": {}, "labels": {}}
        with self._lock:
            self._load()
            try:
                labels = self._component(words)
            except (RuntimeError, ValueError, TypeError) as error:
                raise ResourceUnavailableError(f"HanLP POS 推理失败: {error}") from error
        if isinstance(labels, list) and len(labels) == 1 and isinstance(labels[0], list):
            labels = labels[0]
        if not isinstance(labels, list):
            raise DataValidationError("HanLP POS 返回格式不是标签列表")
        items = normalize_pos(text, tokens, labels)
        counts = dict(Counter(item["category"] for item in items))
        used = sorted({item["pos"] for item in items})
        return {
            "model": self.name,
            "items": items,
            "counts": counts,
            "labels": {tag: POS_TAG_EXPLANATIONS.get(tag, "保留 HanLP 原始标签") for tag in used},
        }
