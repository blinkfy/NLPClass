import pytest

from nlplab.course_neural.contracts import extraction_format
from nlplab.course_neural.resources import verified_bert


def test_extraction_syntax_does_not_imply_schema_or_rewrite_output():
    text = '{"人物":"王敏","地点":"北京大学","机构":""}'
    result = extraction_format(text)
    assert result["valid_json"] is True
    assert result["schema_valid"] is False
    assert result["missing_fields"] == ["事件"]
    assert result["extra_fields"] == ["机构"]
    assert result["value"]["机构"] == ""


@pytest.mark.parametrize("text", ['["王敏"]', '{"人物":1,"地点":[],"事件":[]}', '```json\n{}\n```', '{"人物":NaN,"地点":[],"事件":[]}'])
def test_invalid_extraction_structure_is_reported(text):
    assert extraction_format(text)["schema_valid"] is False


def test_extraction_valid_fields_can_still_be_factually_wrong():
    result = extraction_format('{"人物":["虚构人物"],"地点":[],"事件":"无依据事件"}')
    assert result["schema_valid"] is True
    assert "事实" in result["note"]


def test_non_json_nan_does_not_break_worker_result_serialization():
    import json
    result = extraction_format('{"人物":NaN}')
    assert result["valid_json"] is False
    json.dumps(result, allow_nan=False)


def test_bert_reference_is_required_even_if_external_snapshot_exists(tmp_path):
    from nlplab.core.exceptions import ResourceUnavailableError
    with pytest.raises(ResourceUnavailableError, match="来源记录"):
        verified_bert(tmp_path)
