"""BMES 状态顺序及一阶合法转移。"""

STATES = ("B", "M", "E", "S")
LEGAL_START = {"B", "S"}
LEGAL_END = {"E", "S"}
LEGAL_NEXT = {"B": {"M", "E"}, "M": {"M", "E"}, "E": {"B", "S"}, "S": {"B", "S"}}


def is_legal_sequence(tags: list[str]) -> bool:
    """空序列合法；非空序列检查起始、相邻与结束约束。"""
    if not tags:
        return True
    return (
        tags[0] in LEGAL_START
        and tags[-1] in LEGAL_END
        and all(right in LEGAL_NEXT.get(left, set()) for left, right in zip(tags, tags[1:]))
    )
