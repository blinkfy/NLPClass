"""独立线性链 CRF 适配层；训练、边缘概率和解码由 python-crfsuite 提供。"""

import json
import shutil
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.segmentation.bmes import decode
from nlplab.sequence_labeling.features import crf_features
from nlplab.sequence_labeling.tags import STATES, is_legal_sequence


def _library():
    try:
        import pycrfsuite
    except ImportError as error:
        raise ResourceUnavailableError("课程 CRF 需要 python-crfsuite==0.9.12") from error
    return pycrfsuite


@dataclass(frozen=True)
class CRFConfig:
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 80


class LinearChainCRF:
    name = "course-linear-chain-crf-bmes"

    def __init__(self, config: CRFConfig | None = None) -> None:
        self.config = config or CRFConfig()
        self.path: Path | None = None
        self.training_metadata: dict = {}
        self._tagger = None
        self._info = None
        self._lock = Lock()

    def fit(
        self, observations: list[list[str]], tags: list[list[str]], path: Path
    ) -> "LinearChainCRF":
        if len(observations) != len(tags) or not observations:
            raise DataValidationError("CRF 训练样本与标签数量不一致或为空")
        library = _library()
        trainer = library.Trainer(verbose=False, algorithm="lbfgs")
        for characters, gold in zip(observations, tags):
            if len(characters) != len(gold) or not is_legal_sequence(gold):
                raise DataValidationError("CRF 训练 Gold 必须是合法且等长的 BMES")
            trainer.append(crf_features("".join(characters)), gold)
        trainer.set_params(
            {
                "c1": self.config.c1,
                "c2": self.config.c2,
                "max_iterations": self.config.max_iterations,
                "feature.possible_states": True,
                "feature.possible_transitions": False,
            }
        )
        path.mkdir(parents=True, exist_ok=False)
        trainer.train(str(path / "model.crfsuite"))
        self.path = path
        self._open()
        self._write_metadata()
        return self

    def _open(self) -> None:
        if self.path is None:
            raise ResourceUnavailableError("独立 CRF 模型尚未训练或加载")
        library = _library()
        tagger = library.Tagger()
        tagger.open(str(self.path / "model.crfsuite"))
        self._tagger = tagger
        self._info = tagger.info()

    def _write_metadata(self) -> None:
        if self.path is None:
            raise ResourceUnavailableError("独立 CRF 模型尚未训练")
        from importlib.metadata import version

        weights = self.path / "model.crfsuite"
        metadata = {
            "format_version": 1,
            "model": self.name,
            "states": list(STATES),
            "library": "python-crfsuite",
            "library_version": version("python-crfsuite"),
            "config": asdict(self.config),
            "training_metadata": self.training_metadata,
            "model_sha256": sha256(weights.read_bytes()).hexdigest(),
        }
        (self.path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save(self, path: Path) -> None:
        if self.path is None or self._tagger is None:
            raise ResourceUnavailableError("独立 CRF 模型尚未训练")
        path.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(self.path / "model.crfsuite", path / "model.crfsuite")
        self.path = path
        self._write_metadata()

    @classmethod
    def load(cls, path: Path) -> "LinearChainCRF":
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            model_path = path / "model.crfsuite"
            if (
                metadata["format_version"] != 1
                or metadata["states"] != list(STATES)
                or metadata["model_sha256"] != sha256(model_path.read_bytes()).hexdigest()
            ):
                raise DataValidationError("CRF 模型格式或权重指纹不匹配")
            model = cls(CRFConfig(**metadata["config"]))
            model.path = path
            model.training_metadata = metadata["training_metadata"]
            model._open()
            return model
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ResourceUnavailableError(f"独立 CRF 模型无法加载: {path}") from error

    def predict(self, observations: list[str]) -> list[str]:
        return self.explain("".join(observations), details=False)["tags"]

    def explain(self, text: str, *, details: bool = True) -> dict:
        if self._tagger is None or self._info is None:
            raise ResourceUnavailableError("独立 CRF 模型尚未训练或加载")
        if not text:
            return {"text": "", "tokens": [], "tags": [], "marginals": [], "features": []}
        features = crf_features(text)
        with self._lock:
            self._tagger.set(features)
            tags = list(self._tagger.tag())
            marginals = (
                [
                    [
                        self._tagger.marginal(tag, index) if tag in self._info.labels else 0.0
                        for tag in STATES
                    ]
                    for index in range(len(text))
                ]
                if details
                else []
            )
        if not is_legal_sequence(tags):
            raise DataValidationError("CRF 返回非法 BMES 序列；拒绝把修复序列当作模型预测")
        top_features = []
        if details:
            for row, tag in zip(features, tags):
                weighted = [
                    (feature, self._info.state_features.get((feature, tag), 0.0)) for feature in row
                ]
                top_features.append(sorted(weighted, key=lambda item: (-abs(item[1]), item[0]))[:5])
        return {
            "text": text,
            "tokens": decode(text, tags),
            "tags": tags,
            "marginals": marginals,
            "features": top_features,
            "transitions": [
                [self._info.transitions.get((left, right)) for right in STATES] for left in STATES
            ]
            if details
            else [],
            "metadata": self.training_metadata if details else {},
        }
