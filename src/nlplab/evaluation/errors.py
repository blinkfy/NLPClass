"""分词错误案例与透明的启发式分类，保留人工类别入口。"""

import re
from typing import Any

from nlplab.core.schema import SequenceSample
from nlplab.evaluation.segmentation import word_spans
from nlplab.segmentation.bmes import encode

CATEGORIES = ("OOV", "Proper noun", "Ambiguity", "Number/date", "English mixed", "Other")


def error_record(
    sample: SequenceSample, tokens: list[str], training_words: set[str]
) -> dict[str, Any] | None:
    if sample.gold_tokens == tokens:
        return None
    disagreement = sorted(word_spans(sample.gold_tokens) ^ word_spans(tokens))
    spans: list[list[int]] = []
    for start, end in disagreement:
        if spans and start < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    affected = "".join(sample.text[start:end] for start, end in spans)
    position = 0
    has_oov = False
    predicted_spans = word_spans(tokens)
    for word in sample.gold_tokens:
        if (position, position + len(word)) not in predicted_spans and word not in training_words:
            has_oov = True
        position += len(word)
    has_proper_noun = any(
        start < entity.end and entity.start < end
        for start, end in spans
        for entity in (sample.entities or [])
    )
    candidates = []
    if has_oov:
        candidates.append("OOV")
    if has_proper_noun:
        candidates.append("Proper noun")
    if re.search(r"\d", affected):
        candidates.append("Number/date")
    if re.search(r"[A-Za-z]", affected):
        candidates.append("English mixed")
    return {
        "sample_id": sample.sample_id,
        "doc_id": sample.doc_id,
        "text": sample.text,
        "gold_tokens": sample.gold_tokens,
        "predicted_tokens": tokens,
        "gold_tags": encode(sample.gold_tokens),
        "predicted_tags": encode(tokens),
        "error_spans": spans,
        "category": candidates[0] if candidates else "Other",
        "category_candidates": candidates,
        "category_source": "heuristic",
        "manual_category": None,
    }
