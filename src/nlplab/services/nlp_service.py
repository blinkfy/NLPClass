"""交互任务协调，依赖稳定接口；不包含 Web 或外部模型数据类型。"""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from nlplab.core.config import Settings
from nlplab.core.exceptions import ResourceUnavailableError
from nlplab.ner.base import NERRecognizer
from nlplab.segmentation.base import Segmenter
from nlplab.segmentation.hmm_segmenter import HMMSegmenter
from nlplab.services.experiment_service import RunStore


class NLPService:
    def __init__(
        self,
        settings: Settings,
        ner: NERRecognizer | None = None,
        segmenter_loader: Callable[[str], Segmenter] | None = None,
    ) -> None:
        self.store = RunStore(settings)
        self.ner = ner
        self.segmenter_loader = segmenter_loader or self._load_segmenter

    @lru_cache(maxsize=4)
    def _load_segmenter(self, run_id: str) -> Segmenter:
        return HMMSegmenter(self.store.model(run_id))

    def extension_run(self, model_name: str) -> tuple[Path, dict[str, Any]]:
        """查找已完成的拓展 final Run；不把 dev 资源用于正式交互。"""
        if model_name not in {"longest", "bigram", "hmm2-smooth", "bilstm-crf"}:
            raise ResourceUnavailableError(f"未知拓展分词模型: {model_name}")
        import json

        candidates = []
        for path in (self.store.settings.outputs / "exp1/extensions").glob("*/run.json"):
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("model") == model_name and info.get("phase") == "final":
                candidates.append((path.parent, info))
        if not candidates:
            raise ResourceUnavailableError(f"拓展模型尚未准备: {model_name}")
        return sorted(candidates, key=lambda item: item[0].name)[-1]

    @lru_cache(maxsize=4)
    def _load_extension_segmenter(self, model_name: str) -> Segmenter:
        _, info = self.extension_run(model_name)
        model_path = Path(info["model_path"]).resolve()
        if not model_path.is_relative_to(self.store.settings.models.resolve()):
            raise ResourceUnavailableError("拓展模型路径不在本地 models 目录")
        if model_name in {"longest", "bigram"}:
            from nlplab.extensions.dictionary import DictionarySegmenter

            return DictionarySegmenter.load(model_path)
        if model_name == "hmm2-smooth":
            from nlplab.extensions.hmm2 import SecondOrderHMM

            return SecondOrderHMM.load(model_path)
        from nlplab.extensions.bilstm_crf import BiLSTMCRFSegmenter

        return BiLSTMCRFSegmenter.load(model_path)

    def analyze(
        self,
        text: str,
        tasks: list[str],
        run_id: str = "latest",
        segmentation_model: str = "hmm1-smooth",
    ) -> dict[str, Any]:
        started = perf_counter()
        response: dict[str, Any] = {
            "text": text,
            "segmentation": None,
            "ner": None,
            "classification": None,
            "run": None,
        }
        if "segmentation" in tasks:
            if segmentation_model == "hmm1-smooth":
                info = self.store.read(run_id, "run.json")
                if info.get("run_type") == "frozen-final-benchmark":
                    info = self.store.read(info["run_id"].rsplit("_", 1)[0] + "_smooth", "run.json")
                segmenter = self.segmenter_loader(info["run_id"])
            else:
                _, info = self.extension_run(segmentation_model)
                segmenter = self._load_extension_segmenter(segmentation_model)
            response["run"] = info
            result = segmenter.segment(text)
            response["segmentation"] = asdict(result)
        if "ner" in tasks:
            try:
                if self.ner is None:
                    raise ResourceUnavailableError("HanLP NER 尚未准备；请按 README 安装并下载模型")
                entities = self.ner.recognize(text)
                response["ner"] = {
                    "status": "ok",
                    "model": self.ner.name,
                    "entities": [asdict(entity) for entity in entities],
                }
            except ResourceUnavailableError as error:
                response["ner"] = {
                    "status": "unavailable",
                    "model": getattr(self.ner, "name", "hanlp-msra-bert"),
                    "error": str(error),
                }
        response["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        return response
