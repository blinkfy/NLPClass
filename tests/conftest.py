import pytest

from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.hmm import HMMTagger


def pytest_addoption(parser):
    parser.addoption(
        "--verify-only", action="store_true", help="冻结验收：跳过训练及正式 benchmark/scorer 调用"
    )


@pytest.fixture(autouse=True)
def frozen_verify_only(request, monkeypatch):
    if not request.config.getoption("--verify-only"):
        return
    from nlplab.extensions import benchmark
    from nlplab.extensions.bilstm_crf import BiLSTMCRFSegmenter
    from nlplab.extensions.dictionary import BigramSegmenter, DictionarySegmenter
    from nlplab.extensions.hmm2 import SecondOrderHMM

    def forbidden(*args, **kwargs):
        pytest.skip("verify-only: 禁止训练或正式 benchmark/scorer，未执行该调用")

    for cls in (HMMTagger, SecondOrderHMM, BiLSTMCRFSegmenter):
        monkeypatch.setattr(cls, "fit", forbidden)
    for cls in (DictionarySegmenter, BigramSegmenter):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(benchmark, "official_score", forbidden)
    monkeypatch.setattr(benchmark, "run_model", forbidden)


@pytest.fixture
def toy_tokens() -> list[list[str]]:
    return [
        ["商品", "和", "服务"],
        ["自然语言", "处理", "课程"],
        ["我", "爱", "北京"],
        ["北京", "大学", "的", "学生"],
        ["服务中心", "欢迎", "你"],
    ]


@pytest.fixture
def toy_model(toy_tokens) -> HMMTagger:
    return HMMTagger().fit(
        [list("".join(words)) for words in toy_tokens], [encode(words) for words in toy_tokens]
    )
