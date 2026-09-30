"""信息抽取规则的确定性功能验证；不涉及正式关系 benchmark。"""

from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.core.config import Settings
from nlplab.core.schema import Entity
from nlplab.information_extraction import extract_relations
from nlplab.services.information_extraction_service import InformationExtractionService


def entities(text, *items):
    result = []
    for word, kind in items:
        start = text.index(word)
        result.append(Entity(word, kind, start, start + len(word)))
    return result


@pytest.mark.parametrize(
    "text,items,predicate,trigger",
    [
        (
            "马云在杭州创立了阿里巴巴。",
            (("马云", "PER"), ("杭州", "LOC"), ("阿里巴巴", "ORG")),
            "创立",
            "创立",
        ),
        ("李明任职于腾讯。", (("李明", "PER"), ("腾讯", "ORG")), "任职于", "任职于"),
        ("李明就职于腾讯。", (("李明", "PER"), ("腾讯", "ORG")), "任职于", "就职于"),
        ("李明供职于腾讯。", (("李明", "PER"), ("腾讯", "ORG")), "任职于", "供职于"),
        ("李明就业于腾讯。", (("李明", "PER"), ("腾讯", "ORG")), "任职于", "就业于"),
        ("李明加入腾讯。", (("李明", "PER"), ("腾讯", "ORG")), "任职于", "加入"),
        (
            "李明毕业于中国石油大学。",
            (("李明", "PER"), ("中国石油大学", "ORG")),
            "毕业于",
            "毕业于",
        ),
        (
            "李明就读于中国石油大学。",
            (("李明", "PER"), ("中国石油大学", "ORG")),
            "就读于",
            "就读于",
        ),
        ("李明出生于北京。", (("李明", "PER"), ("北京", "LOC")), "出生于", "出生于"),
        ("腾讯总部位于深圳。", (("腾讯", "ORG"), ("深圳", "LOC")), "位于", "总部位于"),
    ],
)
def test_explicit_relations_have_source_evidence(text, items, predicate, trigger):
    relations = extract_relations(text, entities(text, *items))
    assert len(relations) == 1
    relation = relations[0]
    assert relation["predicate"] == predicate
    assert relation["trigger"]["text"] == trigger
    for field in ("subject", "object", "evidence", "trigger"):
        value = relation[field]
        assert text[value["start"] : value["end"]] == value["text"]
    assert "confidence" not in relation and "probability" not in relation


@pytest.mark.parametrize(
    "text,items",
    [
        ("李明参观了中国石油大学。", (("李明", "PER"), ("中国石油大学", "ORG"))),
        ("李明和腾讯都在北京。", (("李明", "PER"), ("腾讯", "ORG"), ("北京", "LOC"))),
        ("李明没有创立腾讯。", (("李明", "PER"), ("腾讯", "ORG"))),
        ("腾讯表示，位于深圳。", (("腾讯", "ORG"), ("深圳", "LOC"))),
        ("杭州创立了腾讯。", (("杭州", "LOC"), ("腾讯", "ORG"))),
    ],
)
def test_no_relation_from_cooccurrence_or_wrong_type(text, items):
    assert extract_relations(text, entities(text, *items)) == []


def test_founder_location_is_not_company_location():
    text = "马云在杭州创立了阿里巴巴。"
    result = extract_relations(
        text, entities(text, ("马云", "PER"), ("杭州", "LOC"), ("阿里巴巴", "ORG"))
    )
    assert [
        (item["subject"]["text"], item["predicate"], item["object"]["text"]) for item in result
    ] == [("马云", "创立", "阿里巴巴")]


@pytest.mark.parametrize(
    "text,names",
    [
        ("张三和李四任职于阿里巴巴。", ("张三", "李四")),
        ("张三、李四任职于阿里巴巴。", ("张三", "李四")),
        ("张三、李四和王五任职于阿里巴巴。", ("张三", "李四", "王五")),
    ],
)
def test_coordinated_subjects_expand_only_with_explicit_trigger(text, names):
    result = extract_relations(
        text, entities(text, *((name, "PER") for name in names), ("阿里巴巴", "ORG"))
    )
    assert [
        (item["subject"]["text"], item["predicate"], item["object"]["text"]) for item in result
    ] == [(name, "任职于", "阿里巴巴") for name in names]
    assert all(item["trigger"]["text"] == "任职于" for item in result)


def test_coordinated_employment_and_modified_object_regression():
    text = "张三和李四就业于马云在杭州创立的阿里巴巴。"
    result = extract_relations(
        text,
        entities(
            text,
            ("张三", "PER"),
            ("李四", "PER"),
            ("马云", "PER"),
            ("杭州", "LOC"),
            ("阿里巴巴", "ORG"),
        ),
    )
    assert [
        (item["subject"]["text"], item["predicate"], item["object"]["text"]) for item in result
    ] == [
        ("张三", "任职于", "阿里巴巴"),
        ("李四", "任职于", "阿里巴巴"),
        ("马云", "创立", "阿里巴巴"),
    ]
    assert [item["trigger"]["text"] for item in result] == ["就业于", "就业于", "创立"]
    for relation in result:
        for field in ("subject", "object", "evidence", "trigger"):
            value = relation[field]
            assert text[value["start"] : value["end"]] == value["text"]


@pytest.mark.parametrize(
    "text,items",
    [
        (
            "张三和李四参观了阿里巴巴。",
            (("张三", "PER"), ("李四", "PER"), ("阿里巴巴", "ORG")),
        ),
        (
            "张三和李四就业于马云参观的阿里巴巴。",
            (("张三", "PER"), ("李四", "PER"), ("马云", "PER"), ("阿里巴巴", "ORG")),
        ),
        (
            "张三和李四就业于马云在杭州创立的，阿里巴巴。",
            (
                ("张三", "PER"),
                ("李四", "PER"),
                ("马云", "PER"),
                ("杭州", "LOC"),
                ("阿里巴巴", "ORG"),
            ),
        ),
        (
            "张三和李四没有就业于阿里巴巴。",
            (("张三", "PER"), ("李四", "PER"), ("阿里巴巴", "ORG")),
        ),
    ],
)
def test_coordinated_or_modified_object_negative_cases(text, items):
    assert extract_relations(text, entities(text, *items)) == []


def test_unicode_offsets_and_stable_deduplication():
    text = "🙂李明创立腾讯。"
    expected = extract_relations(text, entities(text, ("李明", "PER"), ("腾讯", "ORG")))
    assert expected == extract_relations(
        text, list(reversed(entities(text, ("李明", "PER"), ("腾讯", "ORG"))))
    )
    assert expected[0]["evidence"] == {"text": "李明创立腾讯", "start": 1, "end": 7}
    assert extract_relations("无实体", []) == []
    repeated = "李明创建并创办腾讯。"
    assert (
        len(extract_relations(repeated, entities(repeated, ("李明", "PER"), ("腾讯", "ORG")))) == 1
    )
    with pytest.raises(Exception, match="实体|text|跨度|span"):
        extract_relations(text, [Entity("错误", "PER", 1, 3)])


class FakeNLP:
    def __init__(self, ner):
        self.ner = ner

    def analyze(self, text, tasks):
        assert tasks == ["ner"]
        return {"ner": self.ner(text)}


def test_api_returns_real_entities_relations_and_empty_state(tmp_path):
    def fake_ner(text):
        if "创立" in text:
            found = entities(text, ("李明", "PER"), ("腾讯", "ORG"))
        else:
            found = entities(text, ("李明", "PER"), ("腾讯", "ORG"))
        return {
            "status": "ok",
            "model": "fixture-ner",
            "entities": [asdict(item) for item in found],
        }

    app = create_app(Settings(tmp_path))
    app.state.information_extraction = InformationExtractionService(FakeNLP(fake_ner))
    client = TestClient(app)
    positive = client.post(
        "/api/v1/information-extraction/analyze", json={"text": "李明创立腾讯。"}
    )
    assert positive.status_code == 200
    assert positive.json()["relations"][0]["evidence"]["text"] == "李明创立腾讯"
    negative = client.post(
        "/api/v1/information-extraction/analyze", json={"text": "李明参观腾讯。"}
    )
    assert negative.status_code == 200 and negative.json()["relations"] == []
    assert (
        client.post("/api/v1/information-extraction/analyze", json={"text": ""}).status_code == 422
    )


def test_unavailable_ner_does_not_create_relation(tmp_path):
    app = create_app(Settings(tmp_path))
    app.state.information_extraction = InformationExtractionService(
        FakeNLP(lambda _: {"status": "unavailable", "model": "fixture", "error": "missing"})
    )
    result = (
        TestClient(app)
        .post("/api/v1/information-extraction/analyze", json={"text": "李明创立腾讯。"})
        .json()
    )
    assert result["status"] == "unavailable" and result["entities"] == result["relations"] == []
