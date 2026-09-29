"""课程补充模型的只读交互服务；与实验一正式 Run 完全隔离。"""

from functools import lru_cache
from pathlib import Path
from time import perf_counter

from nlplab.core.config import Settings
from nlplab.core.exceptions import ResourceUnavailableError
from nlplab.sequence_labeling.linear_crf import LinearChainCRF
from nlplab.sequence_labeling.perceptron import AveragedStructuredPerceptron
from nlplab.services.nlp_service import NLPService
from nlplab.tagging.hanlp_pos import HanLPPosTagger


class CourseExtensionService:
    def __init__(self, settings: Settings, nlp: NLPService) -> None:
        self.settings = settings
        self.nlp = nlp
        self.pos = HanLPPosTagger.from_settings(settings)

    def _model_path(self, model: str) -> Path:
        if model not in {"perceptron", "crf"}:
            raise ResourceUnavailableError(f"未知课程序列标注模型: {model}")
        return self.settings.models / "course_sequence" / f"{model}_v1"

    @lru_cache(maxsize=2)
    def _model(self, name: str):
        path = self._model_path(name)
        return (
            AveragedStructuredPerceptron.load(path)
            if name == "perceptron"
            else LinearChainCRF.load(path)
        )

    def resources(self) -> dict:
        return {
            "perceptron": (self._model_path("perceptron") / "metadata.json").is_file(),
            "crf": (self._model_path("crf") / "metadata.json").is_file(),
            "pos": self.pos.model_path is not None and self.pos.model_path.is_dir(),
        }

    def sequence(self, text: str, name: str) -> dict:
        started = perf_counter()
        model = self._model(name)
        result = model.explain(text)
        result["model"] = model.name
        result["course_extension"] = True
        result["source_split"] = "MSR official training internal subset"
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        return result

    def pos_tag(self, text: str) -> dict:
        if self.pos.model_path is None:
            raise ResourceUnavailableError("HanLP POS 模型尚未准备")
        started = perf_counter()
        segmentation = self.nlp.analyze(text, ["segmentation"], "latest", "hmm1-smooth")
        tokens = segmentation["segmentation"]["tokens"]
        result = self.pos.tag(text, tokens)
        result.update(
            {
                "text": text,
                "tokens": tokens,
                "tokenizer": "frozen HMM-1 Smooth",
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }
        )
        return result
