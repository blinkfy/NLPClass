"""集中配置，所有路径相对于工程根目录解析。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """运行目录和交互输入限制。"""

    root: Path
    max_text_length: int = 5000
    ner_model: str = "MSRA_NER_BERT_BASE_ZH"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"


def default_settings() -> Settings:
    """支持通过 NLPLAB_ROOT 指向课程数据与运行结果所在目录。"""
    import os

    root = Path(os.environ.get("NLPLAB_ROOT", Path.cwd())).resolve()
    return Settings(root=root)
