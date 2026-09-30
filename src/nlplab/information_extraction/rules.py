"""显式触发词与实体类型约束的关系抽取课程 baseline。"""

from dataclasses import asdict, dataclass

from nlplab.core.schema import Entity

_BOUNDARIES = "。！？!?；;\n\r，,"
_NEGATION = ("未", "没有", "并未", "不曾", "否认", "未曾", "不")
_FOUNDING_TRIGGERS = ("创立", "创建", "创办")


@dataclass(frozen=True)
class Rule:
    name: str
    subject_type: str
    object_type: str
    triggers: tuple[str, ...]


RULES = (
    Rule("创立", "PER", "ORG", _FOUNDING_TRIGGERS),
    Rule("任职于", "PER", "ORG", ("任职于", "就职于", "供职于", "就业于", "加入")),
    Rule("毕业于", "PER", "ORG", ("毕业于",)),
    Rule("就读于", "PER", "ORG", ("就读于",)),
    Rule("出生于", "PER", "LOC", ("出生于", "生于")),
    Rule("位于", "ORG", "LOC", ("总部位于", "总部设在", "坐落于", "位于")),
)


def _coordinated_prefix(
    text: str, subject: Entity, trigger_start: int, rule: Rule, between: list[Entity]
) -> bool:
    """只展开连续、同类型且由“和/、”连接的主语。"""
    cursor = subject.end
    remaining = [entity for entity in between if entity.end <= trigger_start]
    for entity in remaining:
        connector = text[cursor : entity.start].strip()
        if entity.type == subject.type and connector in {"和", "、"}:
            cursor = entity.end
            continue
        if rule.name == "创立" and entity.type == "LOC" and connector == "在":
            cursor = entity.end
            continue
        return False
    trailing = text[cursor:trigger_start].strip()
    if rule.name == "创立":
        return trailing in {"", "曾", "曾经"}
    if rule.name == "位于":
        return trailing in {"", "的", "总部", "的总部"}
    return trailing == ""


def _founder_modifier(text: str, start: int, end: int, between: list[Entity]) -> bool:
    """允许受限的“PER [在 LOC] 创立/创建/创办 的 ORG”宾语修饰语。"""
    if end - start > 22 or not between or between[0].type != "PER":
        return False
    founder = between[0]
    if text[start : founder.start].strip() or len(between) > 2:
        return False
    location = between[1] if len(between) == 2 else None
    if location is not None and location.type != "LOC":
        return False
    for word in _FOUNDING_TRIGGERS:
        trigger_start = text.find(word, founder.end, end)
        if trigger_start < 0:
            continue
        if location is None:
            prefix_ok = text[founder.end : trigger_start].strip() in {"", "曾", "曾经"}
        else:
            prefix_ok = (
                text[founder.end : location.start].strip() == "在"
                and not text[location.end : trigger_start].strip()
            )
        if (
            prefix_ok
            and not any(neg in text[founder.end : trigger_start] for neg in _NEGATION)
            and text[trigger_start + len(word) : end].strip() == "的"
        ):
            return True
    return False


def _trigger(text: str, subject: Entity, object_: Entity, rule: Rule, between: list[Entity]):
    """同时检查主语连接、有限宾语修饰语、类型与原文触发跨度。"""
    gap = text[subject.end : object_.start]
    if len(gap) > 30 or any(mark in gap for mark in _BOUNDARIES):
        return None
    matches = []
    for word in rule.triggers:
        offset = gap.find(word)
        if offset < 0:
            continue
        start = subject.end + offset
        end = start + len(word)
        if any(entity.start < end and entity.end > start for entity in between):
            continue
        if any(neg in text[subject.end : start] for neg in _NEGATION):
            continue
        if not _coordinated_prefix(text, subject, start, rule, between):
            continue
        tail = text[end : object_.start]
        after = [entity for entity in between if entity.start >= end]
        repeated_trigger = rule.name == "创立" and tail in {f"并{other}" for other in rule.triggers}
        simple_tail = not after and (
            repeated_trigger
            or (
                len(tail) <= 5
                and not any(char in tail for char in "说称表示认为访问参观创立创建创办")
            )
        )
        if not simple_tail and not (
            rule.name == "任职于" and _founder_modifier(text, end, object_.start, after)
        ):
            continue
        matches.append((offset, -len(word), word))
    if not matches:
        return None
    offset, _, word = min(matches)
    start = subject.end + offset
    return {"text": word, "start": start, "end": start + len(word)}


def extract_relations(text: str, entities: list[Entity]) -> list[dict]:
    """从原文和 NER 实体抽取有证据的 SPO；不补全或推测隐含关系。"""
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end, entity.type))
    for entity in ordered:
        entity.validate(text)
    if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
        raise ValueError("NER entities overlap")
    relations = []
    seen = set()
    for left_index, subject in enumerate(ordered):
        for object_ in ordered[left_index + 1 :]:
            for rule in RULES:
                if (subject.type, object_.type) != (rule.subject_type, rule.object_type):
                    continue
                between = [
                    entity
                    for entity in ordered
                    if subject.end <= entity.start
                    and entity.end <= object_.start
                    and entity not in (subject, object_)
                ]
                trigger = _trigger(text, subject, object_, rule, between)
                if trigger is None:
                    continue
                identity = (subject.start, subject.end, rule.name, object_.start, object_.end)
                if identity in seen:
                    continue
                seen.add(identity)
                relations.append(
                    {
                        "subject": asdict(subject),
                        "predicate": rule.name,
                        "object": asdict(object_),
                        "evidence": {
                            "text": text[subject.start : object_.end],
                            "start": subject.start,
                            "end": object_.end,
                        },
                        "trigger": trigger,
                        "rule_id": f"{rule.subject_type}-{rule.name}-{rule.object_type}",
                    }
                )
    return relations
