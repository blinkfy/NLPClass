"""Gold 分词编码与预测标签解码，始终保护字符顺序。"""

from nlplab.core.exceptions import DataValidationError
from nlplab.sequence_labeling.tags import STATES


def encode(tokens: list[str]) -> list[str]:
    """BMES Gold 只从词边界生成。"""
    result: list[str] = []
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise DataValidationError("分词列表不得包含空词或非字符串")
        result.extend(["S"] if len(token) == 1 else ["B", *(["M"] * (len(token) - 2)), "E"])
    return result


def decode(characters: str, tags: list[str]) -> list[str]:
    """非法 BMES 结构通过冲刷 buffer 恢复，未知标签或长度错误拒绝。"""
    if len(characters) != len(tags) or any(tag not in STATES for tag in tags):
        raise DataValidationError("字符/标签长度不一致或标签不属于 BMES")
    words: list[str] = []
    buffer = ""
    for character, tag in zip(characters, tags):
        if tag == "S":
            if buffer:
                words.append(buffer)
                buffer = ""
            words.append(character)
        elif tag == "B":
            if buffer:
                words.append(buffer)
            buffer = character
        elif tag == "M":
            buffer += character
        else:  # E: 追加并结束，孤立 E 也不会丢失字符。
            words.append(buffer + character)
            buffer = ""
    if buffer:
        words.append(buffer)
    assert "".join(words) == characters
    return words
