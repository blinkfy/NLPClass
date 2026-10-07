"""Pure workspace-contract tests; all model and clustering work is stubbed."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from nlplab.comprehensive.workspace import (
    SAMPLING_SEED,
    contextual_result,
    replay_cases,
    sampling_plan,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError
from nlplab.services.comprehensive_service import ComprehensiveAnalysisService


class StubNLP:
    def analyze(self, *args, **kwargs):
        raise AssertionError("workspace tests must not run NLP")


class StubNeural:
    def __init__(self):
        self.calls = []

    def dependency(self, text):
        self.calls.append(("dependency", text))
        return {
            "text": text,
            "sentences": [{
                "text": text,
                "start": 0,
                "end": len(text),
                "items": [{"id": 1, "form": text, "start": 0, "end": len(text),
                           "head": 0, "relation": "root"}],
                "arc_scores": [[0.0, 1.0]],
                "relation_scores": [[0.2, 0.8]],
                "decoder": "stub-parser-decoder",
                "score_type": "stub logit, not probability",
            }],
        }

    def infer(self, kind, payload):
        self.calls.append((kind, deepcopy(payload)))
        return {"task": kind, "text": payload["text"]}


class StubClustering:
    pass


class StubClassifier:
    pass


def make_service(tmp_path):
    neural = StubNeural()
    service = ComprehensiveAnalysisService(
        Settings(tmp_path), StubNLP(), course=object(), clustering=StubClustering(), neural=neural
    )
    service.classifier = StubClassifier()
    return service, neural


def sampling_collection():
    docs = []
    analyses = {}
    for cluster_id, completed in ((0, 3), (1, 0), (2, 1)):
        for index in range(completed):
            doc_id = f"c{cluster_id}-done{index}"
            docs.append({"id": doc_id, "cluster_id": cluster_id, "eligible": True,
                         "gold_label": "must not be inspected"})
            analyses[doc_id] = {"status": "ok"}
        for index in range(4):
            doc_id = f"c{cluster_id}-todo{index}"
            docs.append({"id": doc_id, "cluster_id": cluster_id, "eligible": True,
                         "gold_label": "must not be inspected"})
    docs.append({"id": "excluded", "cluster_id": 1, "eligible": False,
                 "gold_label": "must not be inspected"})
    return {"documents": docs, "analyses": analyses}


def test_sampling_plan_is_deterministic_cluster_aware_and_eligible_only():
    collection = sampling_collection()
    plan = sampling_plan(collection, limit=3)
    repeated = sampling_plan(collection, limit=3)
    reversed_plan = sampling_plan({**collection, "documents": list(reversed(collection["documents"]))}, limit=3)

    assert plan == repeated == reversed_plan
    assert plan["seed"] == SAMPLING_SEED
    assert plan["gold_labels_used"] is False
    assert plan["inference_executed"] is False
    assert "excluded" not in plan["document_ids"]
    assert all(collection["documents"][next(
        i for i, row in enumerate(collection["documents"]) if row["id"] == doc_id
    )]["eligible"] for doc_id in plan["document_ids"])
    assert [int(doc_id[1]) for doc_id in plan["document_ids"]] == [1, 1, 2]
    assert not any(key in {"label", "gold_label"} for row in plan["clusters"] for key in row)


def test_sampling_null_scope_means_all_but_empty_scope_means_none():
    collection = sampling_collection()
    all_plan = sampling_plan(collection, scope_ids=None, limit=10)
    empty_plan = sampling_plan(collection, scope_ids=[], limit=10)

    assert all_plan["scope_count"] == len(collection["documents"])
    assert all_plan["document_ids"]
    assert empty_plan["scope_count"] == 0
    assert empty_plan["document_ids"] == []
    assert all_plan["inference_executed"] is empty_plan["inference_executed"] is False


@pytest.mark.parametrize("limit", [0, 11, True, 1.5])
def test_sampling_rejects_invalid_limits(limit):
    with pytest.raises(DataValidationError):
        sampling_plan(sampling_collection(), limit=limit)


def test_sampling_rejects_scope_ids_outside_collection():
    with pytest.raises(DataValidationError, match="集合之外"):
        sampling_plan(sampling_collection(), scope_ids=["missing"])


def test_contextual_dependency_translates_unicode_offsets_without_changing_parser_fields():
    original = "前🙂微软在北京后"
    start, end = 2, 7
    result = {
        "text": original[start:end],
        "source": {"model": "frozen parser", "run_id": "fixed-run"},
        "sentences": [{
            "text": original[start:end], "start": 0, "end": end - start,
            "items": [
                {"id": 1, "form": "微软", "start": 0, "end": 2, "head": 2,
                 "relation": "nn", "explanation": "noun modifier"},
                {"id": 2, "form": "在", "start": 2, "end": 3, "head": 0,
                 "relation": "root", "explanation": "root"},
                {"id": 3, "form": "北京", "start": 3, "end": 5, "head": 2,
                 "relation": "pobj", "explanation": "object"},
            ],
            "arc_scores": [[0.1, 0.9]], "relation_scores": [[0.2, 0.8]],
            "decoder": "global-tree-decoder", "score_type": "biaffine arc logit, not probability",
        }],
    }
    before = deepcopy(result)

    contextual = contextual_result(original, result, "dependency", start, end)

    sentence = contextual["sentences"][0]
    assert result == before
    assert (sentence["start"], sentence["end"]) == (start, end)
    assert [(item["form"], item["start"], item["end"], item["head"], item["relation"])
            for item in sentence["items"]] == [
        ("微软", 2, 4, 2, "nn"), ("在", 4, 5, 0, "root"), ("北京", 5, 7, 2, "pobj")
    ]
    assert sentence["arc_scores"] == before["sentences"][0]["arc_scores"]
    assert sentence["relation_scores"] == before["sentences"][0]["relation_scores"]
    assert sentence["decoder"] == "global-tree-decoder"
    assert sentence["score_type"] == "biaffine arc logit, not probability"
    assert all(original[row["start"]:row["end"]] == row["form"] for row in sentence["items"])


def test_contextual_information_extraction_translates_all_entity_and_relation_ranges():
    original = "前🙂微软在北京后"
    result = {
        "status": "ok", "ner_model": "stub-ner",
        "entities": [
            {"text": "微软", "type": "ORG", "start": 0, "end": 2, "score": 0.7},
            {"text": "北京", "type": "LOC", "start": 3, "end": 5, "score": None},
        ],
        "relations": [{
            "subject": {"text": "微软", "type": "ORG", "start": 0, "end": 2},
            "predicate": "位于", "object": {"text": "北京", "type": "LOC", "start": 3, "end": 5},
            "evidence": {"text": "微软在北京", "start": 0, "end": 5},
            "trigger": {"text": "在", "start": 2, "end": 3}, "rule_id": "ORG-位于-LOC",
        }],
        "method": "NER + explicit typed trigger rules",
    }
    before = deepcopy(result)

    contextual = contextual_result(original, result, "information_extraction", 2, 7)

    assert result == before
    assert [(row["start"], row["end"], original[row["start"]:row["end"]])
            for row in contextual["entities"]] == [(2, 4, "微软"), (5, 7, "北京")]
    relation = contextual["relations"][0]
    assert relation["subject"]["start"] == 2 and relation["subject"]["end"] == 4
    assert relation["object"]["start"] == 5 and relation["object"]["end"] == 7
    assert relation["evidence"]["start"] == 2 and relation["evidence"]["end"] == 7
    assert original[relation["trigger"]["start"]:relation["trigger"]["end"]] == "在"
    assert relation["rule_id"] == "ORG-位于-LOC"


def test_replay_cases_contain_prompts_not_precomputed_model_outputs():
    cases = replay_cases()

    assert cases
    assert all("expected" in case and "result" not in case and "model_output" not in case
               for case in cases)


def test_custom_selection_terms_are_exact_means_and_missing_vectors_stay_unknown(tmp_path):
    service, _ = make_service(tmp_path)
    collection = {
        "id": "custom-terms", "mode": "custom",
        "documents": [
            {"id": "a", "cluster_id": 0, "title": "a", "snippet": "a", "distance": 0,
             "x": 0, "y": 0, "eligible": True},
            {"id": "b", "cluster_id": 0, "title": "b", "snippet": "b", "distance": 0,
             "x": 0, "y": 0, "eligible": True},
        ],
        "clusters": [], "projection": {}, "representation": {}, "provenance": {}, "notice": "toy",
        "texts": {"a": "private a", "b": "private b"}, "analyses": {},
        "term_vectors": {
            "a": [{"term": "alpha", "weight": 0.8}, {"term": "beta", "weight": 0.4}],
            "b": [{"term": "alpha", "weight": 0.2}, {"term": "gamma", "weight": 0.6}],
        },
    }
    service._remember(collection)

    result = service.selection_terms(collection["id"], ["a", "b"])
    empty = service.selection_terms(collection["id"], [])
    collection["id"] = "custom-no-vectors"
    collection.pop("term_vectors")
    service._remember(collection)
    unknown = service.selection_terms(collection["id"], ["a"])

    assert {row["term"]: row["weight"] for row in result["terms"]} == {
        "alpha": pytest.approx(0.5), "beta": pytest.approx(0.2), "gamma": pytest.approx(0.3)
    }
    assert result["fit_executed"] is False
    assert empty["terms"] == [] and empty["fit_executed"] is False
    assert unknown["terms"] is None
    assert "did not save term vectors" in unknown["source"]
    with pytest.raises(DataValidationError, match="无效文档ID"):
        service.selection_terms("custom-no-vectors", ["unknown"])


def test_optional_api_keeps_legacy_request_without_start_or_end(tmp_path):
    service, neural = make_service(tmp_path)
    application = create_app(settings=Settings(tmp_path), service=StubNLP())
    application.state.comprehensive = service

    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()
        optional_ref = schema["paths"]["/api/v1/comprehensive/optional"]["post"]["requestBody"][
            "content"]["application/json"]["schema"]["$ref"]
        optional_schema = schema["components"]["schemas"][optional_ref.rsplit("/", 1)[-1]]
        assert "start" not in optional_schema.get("required", [])
        assert "end" not in optional_schema.get("required", [])
        response = client.post("/api/v1/comprehensive/optional",
                               json={"text": "北京大学", "kind": "dependency"})

    assert response.status_code == 200
    body = response.json()
    assert neural.calls == [("dependency", "北京大学")]
    assert body["source_span"] == {"start": 0, "end": 4, "text": "北京大学"}
    assert body["offset_space"] == "original_text Unicode code-point"
    assert body["result"]["sentences"][0]["items"][0]["start"] == 0
