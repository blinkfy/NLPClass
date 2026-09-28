"""可由命令行与 Web 层明确报告的领域异常。"""


class NLPLabError(Exception):
    """课程实验的可预期失败。"""


class DataValidationError(NLPLabError):
    """输入数据违反 schema 或标注约束。"""


class ResourceUnavailableError(NLPLabError):
    """数据集或模型尚未准备。"""


class NoValidPathError(NLPLabError):
    """零概率或约束导致 Viterbi 无有效路径。"""
