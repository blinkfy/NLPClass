"""按字符跨度计算分词 P/R/F1、OOV/IV Recall 与 BMES confusion。"""

from typing import Any

from nlplab.core.exceptions import DataValidationError
from nlplab.segmentation.bmes import encode
from nlplab.sequence_labeling.tags import STATES


def word_spans(tokens: list[str]) -> set[tuple[int, int]]:
    """重复字符串仍有独立跨度，不使用字符串 membership 匹配词。"""
    spans = set()
    position = 0
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise DataValidationError("评价分词不得包含空词")
        spans.add((position, position + len(token)))
        position += len(token)
    return spans


def prf(true_positive: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "predicted": predicted,
        "support": gold,
    }


def evaluate_segmentation(
    gold: list[list[str]], predicted: list[list[str]], training_words: set[str]
) -> dict[str, Any]:
    """逐样本累计，防止不同句子相同坐标互相匹配。"""
    if not gold or len(gold) != len(predicted):
        raise DataValidationError("Gold/预测样本数量不一致或为空")
    true_positive = predicted_count = gold_count = tag_correct = character_count = 0
    oov_count = iv_count = oov_correct = iv_correct = 0
    confusion = [[0] * 4 for _ in STATES]
    for gold_words, predicted_words in zip(gold, predicted):
        if "".join(gold_words) != "".join(predicted_words):
            raise DataValidationError("Gold/预测未对应同一原文")
        gold_spans, predicted_spans = word_spans(gold_words), word_spans(predicted_words)
        matches = gold_spans & predicted_spans
        true_positive += len(matches)
        predicted_count += len(predicted_spans)
        gold_count += len(gold_spans)
        position = 0
        for word in gold_words:
            correct = (position, position + len(word)) in predicted_spans
            if word in training_words:
                iv_count += 1
                iv_correct += int(correct)
            else:
                oov_count += 1
                oov_correct += int(correct)
            position += len(word)
        for gold_tag, predicted_tag in zip(encode(gold_words), encode(predicted_words)):
            tag_correct += int(gold_tag == predicted_tag)
            character_count += 1
            confusion[STATES.index(gold_tag)][STATES.index(predicted_tag)] += 1
    if character_count == 0:
        raise DataValidationError("无法评价零字符数据集")
    return {
        **prf(true_positive, predicted_count, gold_count),
        "tag_accuracy": tag_correct / character_count,
        "oov_recall": oov_correct / oov_count if oov_count else None,
        "iv_recall": iv_correct / iv_count if iv_count else None,
        "oov_support": oov_count,
        "iv_support": iv_count,
        "sentence_count": len(gold),
        "character_count": character_count,
        "states": list(STATES),
        "confusion_matrix": confusion,
    }
