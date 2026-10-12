"""Compare browser dependency and relation ports with the native HanLP/Python references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HANLP_HOME", str(ROOT / "artifacts/cache/hanlp"))

sys.path.insert(0, str(ROOT / "src"))
from hanlp.components.parsers.alg import eisner as native_eisner  # noqa: E402

from nlplab.core.schema import Entity  # noqa: E402
from nlplab.information_extraction.rules import extract_relations  # noqa: E402


def _relation_cases():
    specifications = [
        ("李明创立腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("马云在杭州创立了阿里巴巴。", [("马云", "PER"), ("杭州", "LOC"), ("阿里巴巴", "ORG")]),
        ("张三和李四任职于阿里巴巴。", [("张三", "PER"), ("李四", "PER"), ("阿里巴巴", "ORG")]),
        (
            "张三、李四和王五就业于阿里巴巴。",
            [("张三", "PER"), ("李四", "PER"), ("王五", "PER"), ("阿里巴巴", "ORG")],
        ),
        (
            "张三和李四就业于马云在杭州创立的阿里巴巴。",
            [
                ("张三", "PER"),
                ("李四", "PER"),
                ("马云", "PER"),
                ("杭州", "LOC"),
                ("阿里巴巴", "ORG"),
            ],
        ),
        ("李明没有创立腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("李明毕业于中国石油大学。", [("李明", "PER"), ("中国石油大学", "ORG")]),
        ("李明就读于北京大学。", [("李明", "PER"), ("北京大学", "ORG")]),
        ("李明出生于北京。", [("李明", "PER"), ("北京", "LOC")]),
        ("腾讯总部位于深圳。", [("腾讯", "ORG"), ("深圳", "LOC")]),
        ("腾讯坐落于深圳。", [("腾讯", "ORG"), ("深圳", "LOC")]),
        ("腾讯在深圳。", [("腾讯", "ORG"), ("深圳", "LOC")]),
        ("李明参观了腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("李明创立腾讯，腾讯位于深圳。", [("李明", "PER"), ("腾讯", "ORG"), ("深圳", "LOC")]),
        ("🙂李明创立腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("😊马云在杭州创立阿里巴巴。", [("马云", "PER"), ("杭州", "LOC"), ("阿里巴巴", "ORG")]),
        ("李明在北京出生。", [("李明", "PER"), ("北京", "LOC")]),
        ("李明未曾就职于腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("腾讯表示，位于深圳。", [("腾讯", "ORG"), ("深圳", "LOC")]),
        ("杭州创立了腾讯。", [("杭州", "LOC"), ("腾讯", "ORG")]),
        (
            "李明毕业于北京大学，并加入腾讯。",
            [("李明", "PER"), ("北京大学", "ORG"), ("腾讯", "ORG")],
        ),
        ("李明在腾讯工作。", [("李明", "PER"), ("腾讯", "ORG")]),
        ("🙂李明创立并创建腾讯。", [("李明", "PER"), ("腾讯", "ORG")]),
    ]
    cases = []
    for text, mentions in specifications:
        cursor = 0
        entities = []
        for value, kind in mentions:
            start = text.index(value, cursor)
            end = start + len(value)
            cursor = end
            entities.append(Entity(value, kind, start, end))
        cases.append({"text": text, "entities": entities})
    return cases


def _random_arc_cases():
    rng = np.random.default_rng(127)
    cases = []
    for length in range(1, 13):
        random_scores = rng.normal(size=(length + 1, length + 1)).astype(np.float32)
        cases.append({"name": f"random-{length}", "scores": random_scores.tolist()})
        cases.append({"name": f"zeros-{length}", "scores": np.zeros_like(random_scores).tolist()})
    scores = np.zeros((13, 13), dtype=np.float32)
    scores[:, 0] = 10.0
    scores[0, :] = 0.0
    cases.append({"name": "single-root", "scores": scores.tolist()})
    return cases


def _native_heads(scores):
    tensor = torch.tensor([scores], dtype=torch.float32)
    mask = torch.tensor([[False] + [True] * (len(scores) - 1)])
    return native_eisner(tensor, mask)[0, 1:].tolist()


def main():
    parity = json.loads((ROOT / "build/dependency-parity.json").read_text(encoding="utf-8"))
    model = json.loads(
        (ROOT / "build/github-pages/models/transformers/dependency.json").read_text(
            encoding="utf-8"
        )
    )
    arc_cases = _random_arc_cases()
    relation_cases = _relation_cases()
    expected_relations = [
        extract_relations(case["text"], case["entities"]) for case in relation_cases
    ]
    relation_request = [
        {
            "text": case["text"],
            "entities": [
                {"text": e.text, "type": e.type, "start": e.start, "end": e.end, "score": e.score}
                for e in case["entities"]
            ],
        }
        for case in relation_cases
    ]
    fixture_request = [
        {
            "arc": item["arc"],
            "relation": item["relation"],
            "heads": item["heads"],
            "relations": item["relations"],
        }
        for item in parity
    ]
    node_request = {
        "labels": model["labels"],
        "fixture": fixture_request,
        "arcs": arc_cases,
        "relations": relation_request,
    }
    node_code = r"""
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = process.cwd();
const {labels, fixture, arcs, relations} = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const dep = await import(pathToFileURL(path.join(root, 'app/static/js/static-dependency.js')));
const rel = await import(pathToFileURL(path.join(root, 'app/static/js/static-relations.js')));
const fixtureResults = fixture.map(row => dep.decodeRelations(row.arc, row.relation, labels));
const arcResults = arcs.map(row => ({name: row.name, heads: dep.eisner(row.scores)}));
const relationResults = relations.map(row => rel.extractRelations(row.text, row.entities));
process.stdout.write(JSON.stringify({fixtureResults, arcResults, relationResults}));
"""
    with tempfile.TemporaryDirectory(prefix="nlplab-dependency-", dir=ROOT / "build") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(json.dumps(node_request, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            ["node", "--input-type=module", "-e", node_code, str(request_path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(f"Node parity helper failed: {result.stderr.strip()}")
    actual = json.loads(result.stdout)

    for index, (source, decoded) in enumerate(zip(parity, actual["fixtureResults"], strict=True)):
        if decoded["heads"] != source["heads"] or decoded["relations"] != source["relations"]:
            raise AssertionError(f"Dependency fixture {index} differs: {decoded}")
    for case, actual_case in zip(arc_cases, actual["arcResults"], strict=True):
        expected = _native_heads(case["scores"])
        if actual_case["heads"] != expected:
            raise AssertionError(
                f"Eisner mismatch {case['name']}: JS={actual_case['heads']} native={expected}"
            )
        if actual_case["heads"].count(0) != 1:
            raise AssertionError(f"Eisner did not produce one root: {case['name']}")
    for index, (expected, actual_case) in enumerate(
        zip(expected_relations, actual["relationResults"], strict=True)
    ):
        if expected != actual_case:
            raise AssertionError(
                f"Relation case {index} differs for {relation_cases[index]['text']!r}:\n"
                f"Python={expected!r}\nJS={actual_case!r}"
            )
    print(
        json.dumps(
            {
                "native_dependency_fixtures": len(parity),
                "random_and_tie_arcs": len(arc_cases),
                "relation_cases": len(relation_cases),
                "relation_positive": sum(bool(x) for x in expected_relations),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
