"""Dependency tree contracts, original offsets, CoNLL and teaching metrics."""

import re
import unicodedata

from nlplab.core.exceptions import DataValidationError

RELATIONS = {
    "root": "句子根节点", "nsubj": "名词性主语", "nsubjpass": "被动主语",
    "dobj": "直接宾语", "iobj": "间接宾语", "nn": "名词修饰",
    "amod": "形容词修饰", "advmod": "状语修饰", "assmod": "关联修饰",
    "assm": "关联标记", "det": "限定词", "clf": "量词", "nummod": "数量修饰",
    "prep": "介词", "pobj": "介词宾语", "tmod": "时间修饰", "loc": "方位关系",
    "lobj": "方位宾语", "asp": "体标记", "aux": "助动词", "neg": "否定修饰",
    "conj": "并列关系", "cc": "并列连接词", "ccomp": "从句补语",
    "xcomp": "开放从句补语", "rcmod": "关系从句修饰", "cop": "系词",
    "punct": "标点", "dep": "其他依存关系", "attr": "属性关系",
}


def validate_tree(heads, relations):
    n = len(heads)
    if not n or len(relations) != n or any(type(h) is not int or not 0 <= h <= n or h == i + 1 for i, h in enumerate(heads)):
        raise DataValidationError("依存 head 必须在 0..N 且不能指向自身")
    if sum(h == 0 for h in heads) != 1:
        raise DataValidationError("依存树必须恰有一个 ROOT")
    if any(not isinstance(r, str) or not r for r in relations):
        raise DataValidationError("依存关系标签缺失")
    for node in range(1, n + 1):
        seen = set()
        while node:
            if node in seen:
                raise DataValidationError("依存树出现循环")
            seen.add(node)
            node = heads[node - 1]
    return {"single_root": True, "acyclic": True, "root_reachable": True, "node_count": n}


def sentence_spans(text):
    return [{"text": m.group(), "start": m.start(), "end": m.end()}
            for m in re.finditer(r"[^。！？!?\n]+[。！？!?]*|[。！？!?]+", text) if m.group().strip()]


def align_words(text, words, start=0, end=None):
    end = len(text) if end is None else end
    if not 0 <= start <= end <= len(text):
        raise DataValidationError("词跨度范围无效")
    items, cursor = [], start
    for i, word in enumerate(words):
        if not isinstance(word, str) or not word or word.isspace():
            raise DataValidationError("依存词项不能为空或空白")
        at = text.find(word, cursor, end)
        if at < 0 or text[cursor:at].strip():
            raise DataValidationError("依存词项与原始文本不能无损对齐")
        items.append({"id": i + 1, "form": word, "start": at, "end": at + len(word)})
        cursor = at + len(word)
    if text[cursor:end].strip():
        raise DataValidationError("依存词项未覆盖句子尾部正文")
    return items


def tree_record(text, words, prediction, start=0, end=None):
    heads, relations = prediction["heads"], prediction["relations"]
    if len(words) != len(heads):
        raise DataValidationError("解析器节点数与分词结果不一致")
    checks = validate_tree(heads, relations)
    items = align_words(text, words, start, end)
    for item, head, relation in zip(items, heads, relations):
        item.update(head=head, relation=relation, explanation=RELATIONS.get(relation, "保留模型原始标签"))
    conll = "\n".join(f"{r['id']}\t{r['form']}\t_\t_\t_\t_\t{r['head']}\t{r['relation']}\t_\t_" for r in items)
    return {"items": items, "checks": checks, "conll": conll, "arc_scores": prediction.get("arc_scores"),
            "relation_scores": prediction.get("relation_scores"),
            "decoder": prediction.get("decoder"), "score_type": "biaffine arc logit, not probability"}


def punctuation(word):
    return bool(word) and all(unicodedata.category(c).startswith("P") for c in word)


def attachment_metrics(words, gold_heads, gold_relations, heads, relations, exclude_punctuation=True):
    validate_tree(gold_heads, gold_relations)
    validate_tree(heads, relations)
    if not (len(words) == len(gold_heads) == len(heads)):
        raise DataValidationError("评测节点未对齐")
    included = [i for i, word in enumerate(words) if not exclude_punctuation or not punctuation(word)]
    correct_heads = sum(heads[i] == gold_heads[i] for i in included)
    correct_labels = sum(heads[i] == gold_heads[i] and relations[i] == gold_relations[i] for i in included)
    return {"support": len(included), "correct_heads": correct_heads, "correct_labels": correct_labels,
            "uas": correct_heads / len(included) if included else None,
            "las": correct_labels / len(included) if included else None,
            "exclude_punctuation": exclude_punctuation}
