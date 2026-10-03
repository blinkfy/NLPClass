"""Toy-only Stage4 analysis contracts; never access held-out or historical Test assets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.classification import router as classification_router
from nlplab.classification.linear import LinearTextClassifier
from nlplab.classification.stage4 import fit_sparse_points
from nlplab.classification.stage4_analysis import (
    MODEL_IDS,
    PER_CLASS_SIZES,
    SUBSET_SEEDS,
    aggregate_points,
    build_ladders,
    effective_k,
    ids_hash,
    moments,
    verify_full_size,
)
from nlplab.core.config import Settings
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.schema import text_sha256
from nlplab.data.classification_development import (
    DevelopmentClassificationData,
    forbidden_development_path,
)
from nlplab.data.classification_input import ClassificationDocument
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.data.thucnews import CATEGORIES
from nlplab.features.text_tfidf import TFIDFConfig
from nlplab.services.classification_stage4_service import ClassificationStage4Service


def _toy_documents(per_class=84):
    return [
        ClassificationDocument(f"{label}-{index:03d}", f"toy {label} item {index}", label)
        for index in range(per_class)
        for label in CATEGORIES
    ]


def test_ladders_are_balanced_nested_and_keep_canonical_train_id_order():
    documents = _toy_documents()
    canonical = [document.id for document in documents]
    ladders = build_ladders(documents)

    assert len(ladders) == len(SUBSET_SEEDS) * len(PER_CLASS_SIZES) == 15
    for seed in SUBSET_SEEDS:
        seed_ladders = [row for row in ladders if row["subset_seed"] == seed]
        assert [row["per_class"] for row in seed_ladders] == list(PER_CLASS_SIZES)
        previous = {label: set() for label in CATEGORIES}
        for row in seed_ladders:
            assert row["train_size"] == row["per_class"] * len(CATEGORIES)
            assert row["category_counts"] == {label: row["per_class"] for label in CATEGORIES}
            assert set(row["category_ids"]) == set(CATEGORIES)
            selected = set()
            for label in CATEGORIES:
                ids = row["category_ids"][label]
                assert len(ids) == row["per_class"]
                assert previous[label] <= set(ids)
                previous[label] = set(ids)
                selected.update(ids)
            assert row["document_ids"] == [doc_id for doc_id in canonical if doc_id in selected]
            assert row["ids_sha256"] == ids_hash(row["document_ids"])
        assert seed_ladders[-1]["document_ids"] == canonical
        assert seed_ladders[-1]["train_size"] == 420

    with pytest.raises(DataValidationError, match="Duplicate training IDs"):
        build_ladders(documents + [documents[0]])


def test_effective_k_and_population_moments_have_fixed_boundaries():
    assert effective_k(20000) == 10000
    assert effective_k(500) == 500
    assert moments([1, 2, 3]) == pytest.approx({"mean": 2.0, "std": np.sqrt(2 / 3)})
    with pytest.raises(DataValidationError, match="Empty character feature space"):
        effective_k(0)
    with pytest.raises(DataValidationError, match="Empty character feature space"):
        effective_k(100, target=0)


def _toy_curve_points():
    points = []
    for model_id in MODEL_IDS:
        for per_class in PER_CLASS_SIZES:
            train_size = per_class * len(CATEGORIES)
            for seed_index, seed in enumerate(SUBSET_SEEDS):
                validation_f1 = 0.25 + seed_index * 0.05 + per_class / 1000
                points.append(
                    {
                        "model_id": model_id,
                        "train_size": train_size,
                        "subset_seed": seed,
                        "train_metrics": {"macro_f1": validation_f1 + 0.1},
                        "validation_metrics": {
                            "macro_f1": validation_f1,
                            "accuracy": 0.6 + seed_index * 0.02,
                            "per_class": [
                                {"label": label, "f1": i * 0.1 + 0.1 + seed_index * 0.02}
                                for i, label in enumerate(CATEGORIES)
                            ],
                        },
                        "representation": {
                            "nnz": per_class + seed_index,
                            "density": per_class / 1000 + seed_index * 0.01,
                            "included": True,
                        },
                        "deterministic_signature": (
                            f"{model_id}-420" if train_size == 420 else f"{seed}-{train_size}"
                        ),
                    }
                )
    return points


def test_curve_aggregates_population_std_and_per_class_f1_without_boolean_fields():
    aggregate = aggregate_points(_toy_curve_points())
    assert len(aggregate) == len(MODEL_IDS) * len(PER_CLASS_SIZES)
    row = next(
        point for point in aggregate if point["model_id"] == "C0" and point["train_size"] == 80
    )
    assert row["replicates"] and len(row["replicates"]) == 3
    assert row["validation_macro_f1"]["mean"] == pytest.approx(0.316)
    assert row["validation_macro_f1"]["std"] == pytest.approx(np.std([0.266, 0.316, 0.366], ddof=0))
    assert row["gap"]["mean"] == pytest.approx(0.1)
    assert row["gap"]["std"] == pytest.approx(0.0)
    assert row["per_class_f1"][0]["label"] == CATEGORIES[0]
    assert row["per_class_f1"][0]["mean"] == pytest.approx(0.12)
    assert row["per_class_f1"][0]["std"] == pytest.approx(np.std([0.1, 0.12, 0.14], ddof=0))
    assert set(row["representation"]) == {"density", "nnz"}
    assert row["representation"]["nnz"]["mean"] == pytest.approx(17.0)


def test_420_replicates_require_all_models_all_seeds_and_identical_signatures():
    points = _toy_curve_points()
    checks = verify_full_size(points)
    assert set(checks) == set(MODEL_IDS)
    assert all(check["identical"] and check["replicates"] == 3 for check in checks.values())

    mismatch = [dict(point) for point in points]
    row = next(
        point for point in mismatch if point["model_id"] == "C2" and point["train_size"] == 420
    )
    row["deterministic_signature"] = "different-full-fit"
    with pytest.raises(DataValidationError, match="STOP: C2 420 deterministic mismatch"):
        verify_full_size(mismatch)

    missing = [
        point
        for point in points
        if not (
            point["model_id"] == "C1" and point["train_size"] == 420 and point["subset_seed"] == 303
        )
    ]
    with pytest.raises(DataValidationError, match="all three replicates"):
        verify_full_size(missing)


def test_development_loader_reads_only_mock_train_and_val_records(tmp_path):
    root = tmp_path
    split_dir = root / "data/splits/thucnews_600"
    split_dir.mkdir(parents=True)
    split_info = {}
    file_hashes = {}
    for split, per_class in (("train", 84), ("val", 18)):
        rows = []
        for index in range(per_class):
            for label in CATEGORIES:
                document_id = f"{split}-{label}-{index:03d}"
                text = f"toy only {split} {label} record {index}"
                rows.append(
                    {
                        "id": document_id,
                        "primary_label": label,
                        "text": text,
                        "sha256": text_sha256(text),
                    }
                )
        path = split_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_info[split] = {
            "count": len(rows),
            "ids": [row["id"] for row in rows],
            "category_counts": {label: per_class for label in CATEGORIES},
        }
        file_hashes[path.name] = file_sha256(path)
    (split_dir / "manifest.json").write_text(
        json.dumps({"splits": split_info, "file_sha256": file_hashes}, ensure_ascii=False),
        encoding="utf-8",
    )

    data = DevelopmentClassificationData(root)
    assert len(data.load("train")) == 420
    assert len(data.load("val")) == 90
    assert data.opened_splits == ["train", "val"]
    provenance = data.provenance()
    assert provenance["test_access_permitted"] is False
    assert provenance["full_corpus_opened"] is False
    with pytest.raises(DataValidationError, match="only permits train/val"):
        data.load("test")


def test_development_path_guard_blocks_test_full_corpus_and_historical_assets(tmp_path):
    blocked = [
        tmp_path / "data/splits/thucnews_600/test.jsonl",
        tmp_path / "data/course/thucnews_600/train.jsonl",
        tmp_path / "outputs/exp3/old-run/C2_test_predictions.jsonl",
        tmp_path / "outputs/exp3/old-run/comparison_metrics.json",
        tmp_path / "outputs/exp3/old-run/paired_c1_c2.json",
    ]
    assert all(forbidden_development_path(tmp_path, path) for path in blocked)
    allowed = [
        tmp_path / "data/splits/thucnews_600/train.jsonl",
        tmp_path / "data/splits/thucnews_600/val.jsonl",
        tmp_path / "outputs/exp3/stage4/toy/test_predictions.json",
        tmp_path / "outputs/exp3/stage4_latest.json",
        tmp_path / "outputs/exp3/stage4_preregistration.json",
        tmp_path.parent / "outside/test.jsonl",
    ]
    assert not any(forbidden_development_path(tmp_path, path) for path in allowed)


def test_sparse_subset_fit_shares_c0_c1_matrix_and_refits_c2_k_on_subset(tmp_path, monkeypatch):
    docs, tokens, val, val_tokens = [], [], [], []
    for class_index, label in enumerate(CATEGORIES):
        for item_index in range(3):
            docs.append(
                ClassificationDocument(
                    f"fit-{class_index}-{item_index}",
                    f"新闻类别{label}主题{class_index}故事{item_index}新闻",
                    label,
                )
            )
            tokens.append(
                [
                    f"label_{class_index}",
                    f"theme_{class_index}",
                    f"story_{item_index}",
                    f"unique_{class_index}_{item_index}",
                ]
            )
        val.append(
            ClassificationDocument(
                f"val-{class_index}", f"新闻类别{label}验证主题{class_index}新闻", label
            )
        )
        val_tokens.append([f"label_{class_index}", f"val_only_{class_index}"])

    observed_matrices = {}
    original_fit = LinearTextClassifier.fit

    def record_fit(model, matrix, labels):
        observed_matrices[model.model_id] = matrix
        return original_fit(model, matrix, labels)

    monkeypatch.setattr(LinearTextClassifier, "fit", record_fit)
    output = tmp_path / "sparse-point"
    output.mkdir()
    subset = {
        "subset_seed": SUBSET_SEEDS[0],
        "per_class": 3,
        "ids_sha256": "toy-subset-hash",
    }
    points = fit_sparse_points(
        output,
        docs,
        val,
        tokens,
        val_tokens,
        subset,
        {"representation": TFIDFConfig().__dict__},
    )

    assert [point["model_id"] for point in points] == ["C0", "C1", "C2"]
    assert observed_matrices["C0"] is observed_matrices["C1"]
    assert observed_matrices["C2"] is not observed_matrices["C0"]
    assert all(point["representation_fit_ids"] == [doc.id for doc in docs] for point in points)
    assert all(
        not set(point["representation_fit_ids"]) & {doc.id for doc in val} for point in points
    )
    c2_stats = points[2]["representation"]
    assert c2_stats["effective_k"] == min(10000, c2_stats["raw_dimension"])
    word_vocabulary = json.loads(
        (output / "word_representation/vocabulary.json").read_text("utf-8")
    )["vocabulary"]
    assert not any(term.startswith("val_only_") for term in word_vocabulary)


def _write_stage4_service_fixture(root):
    output_root = root / "outputs/exp3"
    directory = output_root / "stage4" / "toy-run"
    directory.mkdir(parents=True)
    rows = []
    for model_id in MODEL_IDS:
        rows.append(
            {
                "model_id": model_id,
                "train_size": 80,
                "validation_macro_f1": {"mean": 0.5, "std": 0.1},
                "replicates": [
                    {
                        "subset_seed": seed,
                        "train_metrics": {"macro_f1": 0.6},
                        "validation_metrics": {"macro_f1": 0.5, "accuracy": 0.7},
                        "representation": {"dimension": 10},
                    }
                    for seed in SUBSET_SEEDS
                ],
            }
        )
    payloads = {
        "run.json": {"run_id": "toy-run", "run_type": "stage4-toy"},
        "protocol.json": {"allowed_splits": ["train", "val"], "test_access_permitted": False},
        "learning_curves.json": rows,
        "subset_ladders.json": [{"train_size": 80, "document_ids": ["toy-id"]}],
        "sanity_420.json": {model_id: {"identical": True} for model_id in MODEL_IDS},
        "environment.json": {"device": "cpu", "toy": True},
        "runtimes.json": {"points": [], "model_fits": 12},
        "raw_results.json": [{"toy": True}],
        "per_class_curves.json": [{"toy": True}],
        "representation_diagnostics.json": [{"toy": True}],
    }
    for name, value in payloads.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    manifest = {
        "algorithm": "sha256",
        "files": {name: file_sha256(directory / name) for name in payloads},
    }
    (directory / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stage4_latest.json").write_text(
        json.dumps({"run_id": "toy-run"}), encoding="utf-8"
    )
    # A parent experiment run is deliberately outside the Stage4 namespace.
    legacy = output_root / "legacy-run"
    legacy.mkdir()
    (legacy / "run.json").write_text(json.dumps({"run_id": "legacy-run"}), encoding="utf-8")
    return directory


def _stage4_router_app(root):
    application = FastAPI()
    application.include_router(classification_router)
    application.state.classification = SimpleNamespace(settings=Settings(root=root))

    @application.exception_handler(DataValidationError)
    async def invalid(request, error):
        return JSONResponse({"error": {"message": str(error)}}, status_code=422)

    return application


def test_stage4_service_and_router_use_nested_run_and_model_filter(tmp_path):
    directory = _write_stage4_service_fixture(tmp_path)
    service = ClassificationStage4Service(Settings(root=tmp_path))
    assert service.directory == tmp_path / "outputs/exp3/stage4"
    assert service.resolve("latest") == directory
    assert [row["run_id"] for row in service.list_runs()] == ["toy-run"]
    with pytest.raises(ResourceUnavailableError, match="不存在或未完成"):
        service.resolve("legacy-run")

    with TestClient(_stage4_router_app(tmp_path)) as client:
        runs = client.get("/api/v1/experiments/3/stage4/runs")
        assert runs.status_code == 200
        assert [row["run_id"] for row in runs.json()] == ["toy-run"]

        for model_id in MODEL_IDS:
            filtered = client.get(
                "/api/v1/experiments/3/stage4/overview",
                params={"run_id": "toy-run", "model_id": model_id},
            )
            assert filtered.status_code == 200
            assert [row["model_id"] for row in filtered.json()["curves"]] == [model_id]
        unfiltered = client.get("/api/v1/experiments/3/stage4/overview?run_id=toy-run")
        assert unfiltered.status_code == 200
        assert [row["model_id"] for row in unfiltered.json()["curves"]] == list(MODEL_IDS)
        invalid_model = client.get(
            "/api/v1/experiments/3/stage4/overview",
            params={"run_id": "toy-run", "model_id": "C4"},
        )
        assert invalid_model.status_code == 422
        assert client.post("/api/v1/experiments/3/stage4/run").status_code == 404


def test_stage4_export_whitelist_serves_only_toy_nested_artifacts(tmp_path):
    directory = _write_stage4_service_fixture(tmp_path)
    service = ClassificationStage4Service(Settings(root=tmp_path))
    allowed = {
        "protocol": "protocol.json",
        "subsets": "subset_ladders.json",
        "curves": "learning_curves.json",
        "raw": "raw_results.json",
        "per-class": "per_class_curves.json",
        "diagnostics": "representation_diagnostics.json",
        "runtimes": "runtimes.json",
        "manifest": "artifact_manifest.json",
    }
    for name, filename in allowed.items():
        assert service.export_path("toy-run", name) == directory / filename

    with pytest.raises(DataValidationError, match="不支持的 Stage4"):
        service.export_path("toy-run", "dataset")
    with TestClient(_stage4_router_app(tmp_path)) as client:
        for name, filename in allowed.items():
            response = client.get(
                f"/api/v1/experiments/3/stage4/export/{name}", params={"run_id": "toy-run"}
            )
            assert response.status_code == 200
            assert json.loads(response.content) == json.loads(
                (directory / filename).read_text("utf-8")
            )
        denied = client.get(
            "/api/v1/experiments/3/stage4/export/dataset", params={"run_id": "toy-run"}
        )
        assert denied.status_code == 422


def test_preexisting_model_selector_routes_keep_their_literals():
    class FakeClassificationService:
        def __init__(self):
            self.calls = []

        def predict(self, text, model_id, run_id):
            self.calls.append((model_id, run_id))
            return {"model_id": model_id, "run_id": run_id}

    fake = FakeClassificationService()
    application = FastAPI()
    application.include_router(classification_router)
    application.state.classification = fake
    with TestClient(application) as client:
        assert (
            client.post(
                "/api/v1/experiments/3/predict", json={"text": "toy", "model_id": "C1"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/experiments/3/predict", json={"text": "toy", "model_id": "C2"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/experiments/3/stage2/predict", json={"text": "toy", "model_id": "C2"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/experiments/3/stage2/predict", json={"text": "toy", "model_id": "C1"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/experiments/3/stage3/predict", json={"text": "toy", "model_id": "C3"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/experiments/3/stage3/predict", json={"text": "toy", "model_id": "C4"}
            ).status_code
            == 422
        )
    assert fake.calls == [("C1", "latest"), ("C2", "latest"), ("C3", "latest")]
