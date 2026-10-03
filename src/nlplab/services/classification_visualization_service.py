"""Read-only numerical decomposition of frozen rows; never fit/evaluate a model."""

import json
import os
import subprocess
from collections import OrderedDict
from threading import Lock

import numpy as np

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.thucnews import CATEGORIES
from nlplab.services.classification_workbench_service import ClassificationWorkbenchService


class ClassificationVisualizationService(ClassificationWorkbenchService):
    _traces = OrderedDict()
    _trace_lock = Lock()

    def decision(self, run_id, document_id, model_id):
        if model_id not in ("C1", "C2"):
            raise DataValidationError("Decision Explorer 只支持 C1/C2")
        view = self.formal.store.resolve(run_id)
        source = (self.formal.store.resolve(self.formal.parent_run(view.name))
                  if model_id == "C1" else self.formal.stage2_directory(view.name))
        record = self._record(source, f"{model_id}_test_predictions.jsonl", document_id)
        dataset = self._json(source, "dataset.json")
        ids = dataset["splits"]["test"]["ids"]
        if document_id not in ids:
            raise DataValidationError("Explorer 仅支持当前冻结 Test ID")
        row_index = ids.index(document_id)
        if model_id == "C1":
            vocabulary = self._json(source, "final/tfidf/vocabulary.json")["vocabulary"]
            terms = [t for t, _ in sorted(vocabulary.items(), key=lambda r: r[1])]
            idf = np.load(self._path(source, "final/tfidf/idf.npy"), allow_pickle=False)
            raw_vocabulary, chi2, df, rank = {}, {}, {}, {}
        else:
            meta = self._json(source, "final/representation/representation.json")
            raw_vocabulary = meta["vocabulary"]
            selected = self._json(source, "final/representation/selected_features.json")
            terms = [r["term"] for r in sorted(selected, key=lambda r: r["column"])]
            with np.load(self._path(source, "final/representation/parameters.npz"), allow_pickle=False) as arrays:
                idf = arrays["idf"].copy()
                scores, frequencies = arrays["chi2_scores"], arrays["df"]
                chi2 = {t: float(scores[j]) for t, j in raw_vocabulary.items()}
                df = {t: int(frequencies[j]) for t, j in raw_vocabulary.items()}
                # Display rank uses descending saved score, raw-column tie break.
                order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))
                ranks = {j: i + 1 for i, j in enumerate(order)}
                rank = {t: ranks[j] for t, j in raw_vocabulary.items()}
        model_meta = self._json(source, f"final/{model_id}/model.json")
        if model_meta["label_order"] != list(CATEGORIES):
            raise DataValidationError("冻结线性类别顺序不符")
        with np.load(self._path(source, f"final/{model_id}/parameters.npz"), allow_pickle=False) as arrays:
            weights, bias = arrays["weights"].copy(), arrays["intercept"].copy()
        with np.load(self._path(source, "final/test.npz"), allow_pickle=False) as matrix:
            if matrix["format"].item() != b"csr" or tuple(matrix["shape"]) != (len(ids), len(terms)):
                raise DataValidationError("保存的 CSR shape/ID 不符")
            start, end = matrix["indptr"][row_index:row_index + 2]
            columns, values = matrix["indices"][start:end].copy(), matrix["data"][start:end].copy()
        if weights.shape != (5, len(terms)) or len(idf) != len(terms):
            raise DataValidationError("冻结词表/IDF/系数维度不符")
        pred, runner = CATEGORIES.index(record["prediction"]), CATEGORIES.index(record["runner_up"])
        scores = {r["label"]: r["value"] for r in record["scores"]}
        reconstructed = bias + (weights[:, columns] * values).sum(axis=1)
        if not np.allclose(reconstructed, [scores[c] for c in CATEGORIES], atol=2e-8, rtol=2e-8):
            raise DataValidationError("冻结 CSR × coefficients 无法还原保存的决策分数")
        features = []
        for j, x in zip(columns, values):
            j = int(j); term = terms[j]
            features.append({"term": term, "column": j, "tfidf": float(x), "idf": float(idf[j]),
                             "predicted_weight": float(weights[pred, j]), "runner_up_weight": float(weights[runner, j]),
                             "direct_contribution": float(x * weights[pred, j]),
                             "relative_contribution": float(x * (weights[pred, j] - weights[runner, j]))})
        features.sort(key=lambda r: (-abs(r["relative_contribution"]), r["column"]))
        return {"model_id": model_id, "document_id": document_id, "view_run_id": view.name,
                "source_run_id": source.name, "prediction": record["prediction"], "runner_up": record["runner_up"],
                "predicted_score": scores[record["prediction"]], "runner_up_score": scores[record["runner_up"]],
                "margin": scores[record["prediction"]] - scores[record["runner_up"]],
                "intercept_difference": float(bias[pred] - bias[runner]), "features": features,
                "dimension": len(terms), "chi2_metadata": {t: {"raw_column": raw_vocabulary[t], "chi2_score": chi2[t], "df": df[t], "display_rank": rank[t]} for t in raw_vocabulary} if model_id == "C2" else {},
                "rank_definition": "descending frozen chi2_score; ties by raw column; display rank only",
                "tf_available": False, "tokens_available": False,
                "provenance": {"matrix": "final/test.npz", "weights": f"final/{model_id}/parameters.npz", "idf": "frozen IDF", "fit_executed": False, "classifier_inference_executed": False}}

    def trace(self, run_id, document_id, kernel=3, filter_id=0):
        if kernel not in (3, 4, 5) or not 0 <= filter_id < 128:
            raise DataValidationError("kernel/filter 越界")
        # Concurrent linked views must share the same single-document forward.
        with self._trace_lock:
            return self._cached_trace(run_id, document_id, kernel, filter_id)

    def _cached_trace(self, run_id, document_id, kernel, filter_id):
        directory = self.formal.stage3_directory(run_id)
        names = ("final/model/state.npz", "final/model/model.json", "test_encoding.npz", "test_encoding.json", "C3_test_predictions.jsonl")
        paths = [self._path(directory, name) for name in names]
        key = (directory.name, document_id, tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
        created = key not in self._traces
        if created:
            runtime = self.settings.root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
            try:
                result = subprocess.run([str(runtime), "-m", "nlplab.classification.visualization_trace", "--directory", str(directory), "--document-id", document_id], capture_output=True, text=True, encoding="utf-8", timeout=90, cwd=self.settings.root,
                                        env={**os.environ, "PYTHONPATH": str(self.settings.root / "src"), "PYTHONIOENCODING": "utf-8"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ResourceUnavailableError(f"冻结 diagnostic trace 环境失败: {error}") from error
            if result.returncode:
                raise DataValidationError("冻结 diagnostic trace 校验失败: " + result.stderr.strip()[-500:])
            self._traces[key] = json.loads(result.stdout)
            while len(self._traces) > 3:
                self._traces.popitem(last=False)
        self._traces.move_to_end(key)
        trace = self._traces[key]
        channel = (kernel - 3) * 128 + filter_id
        return {k: v for k, v in trace.items() if k != "activations"} | {
            "kernel_size": kernel, "filter_id": filter_id, "channel": channel,
            "activations": trace["activations"][str(kernel)][filter_id],
            "trace_created": created, "view_run_id": run_id, "source_run_id": directory.name}
