"""Output diagnostics without rewriting or repairing a model's text."""

import json


def extraction_format(text):
    expected = {"人物", "地点", "事件"}
    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")
    try:
        value = json.loads(text, parse_constant=reject_constant)
    except ValueError:
        return {"valid_json": False, "schema_valid": False,
                "note": "真实输出未满足 JSON 语法；没有自动补写结果"}
    keys = set(value) if isinstance(value, dict) else set()
    valid_types = isinstance(value, dict) and all(
        isinstance(item, str) or (isinstance(item, list) and all(isinstance(v, str) for v in item))
        for item in value.values()
    )
    return {"valid_json": True, "schema_valid": keys == expected and valid_types,
            "value": value, "expected_fields": sorted(expected),
            "missing_fields": sorted(expected-keys), "extra_fields": sorted(keys-expected),
            "note": "只校验语法与字段协议，不评价抽取事实正确性"}
