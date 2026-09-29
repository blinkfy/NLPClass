"""感知机与独立 CRF 共用的字符级 BMES 特征模板。"""

import unicodedata


def character_type(char: str) -> str:
    code = ord(char)
    if char.isascii() and char.isdigit():
        return "ascii-digit"
    if char.isascii() and char.isalpha():
        return "ascii-letter"
    if 0xFF10 <= code <= 0xFF19:
        return "fullwidth-digit"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "han"
    if char.isspace():
        return "space"
    if unicodedata.category(char).startswith("P"):
        return "punctuation"
    return "other"


def character_features(characters: list[str], index: int) -> tuple[str, ...]:
    """特征仅依赖当前输入及位置，不读取 Gold 或模型预测。"""
    current = characters[index]
    previous = characters[index - 1] if index else "<BOS>"
    following = characters[index + 1] if index + 1 < len(characters) else "<EOS>"
    features = [
        "bias",
        f"c[-1]={previous}",
        f"c[0]={current}",
        f"c[+1]={following}",
        f"c[-1]+c[0]={previous}{current}",
        f"c[0]+c[+1]={current}{following}",
        f"type={character_type(current)}",
    ]
    if index == 0:
        features.append("BOS")
    if index + 1 == len(characters):
        features.append("EOS")
    return tuple(features)


def sequence_features(text: str) -> list[tuple[str, ...]]:
    characters = list(text)
    return [character_features(characters, index) for index in range(len(characters))]


def crf_features(text: str) -> list[dict[str, bool]]:
    return [{feature: True for feature in row} for row in sequence_features(text)]
