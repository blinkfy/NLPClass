"""Validate forward relation logits conditioned on the decoded tree heads."""

import math


def relation_trace(heads, relations, vectors, labels):
    if len(heads) != len(relations) or len(vectors) != len(heads) or not labels:
        raise ValueError("关系 trace 未对齐")
    rows = []
    for position, (head, relation, scores) in enumerate(zip(heads, relations, vectors), 1):
        if len(scores) != len(labels) or not all(math.isfinite(v) for v in scores):
            raise ValueError("关系 logits 的维度或数值无效")
        # Match torch.argmax's first-index tie rule and the parser's label vocab.
        order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
        if labels[order[0]] != relation:
            raise ValueError("真实 relation logit argmax 与 parser 最终标签不一致")
        rows.append({"dependent": position, "head": head, "relation": relation,
                     "verified": True, "score_type": "biaffine relation logit, not probability",
                     "candidates": [{"index": index, "label": labels[index], "score": scores[index]}
                                    for index in order]})
    return rows
