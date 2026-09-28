"""NER 必须同时匹配字符 start/end/type，逐样本精确评价。"""

from typing import Any

from nlplab.core.exceptions import DataValidationError
from nlplab.core.schema import Entity
from nlplab.evaluation.segmentation import prf


def evaluate_ner(
    texts: list[str], gold: list[list[Entity]], predictions: list[list[Entity]]
) -> dict[str, Any]:
    if not texts or len(texts) != len(gold) or len(gold) != len(predictions):
        raise DataValidationError("NER Gold/预测数量不一致或为空")
    gold_set, predicted_set = set(), set()
    for sample_index, (text, gold_entities, predicted_entities) in enumerate(
        zip(texts, gold, predictions)
    ):
        for target, entities in ((gold_set, gold_entities), (predicted_set, predicted_entities)):
            for entity in entities:
                entity.validate(text)
                target.add((sample_index, entity.start, entity.end, entity.type))
    result: dict[str, Any] = {}
    for label in ("PER", "LOC", "ORG", "Overall"):
        expected = {item for item in gold_set if label == "Overall" or item[-1] == label}
        predicted = {item for item in predicted_set if label == "Overall" or item[-1] == label}
        result[label] = prf(len(expected & predicted), len(predicted), len(expected))
    result["sentence_count"] = len(texts)
    return result
