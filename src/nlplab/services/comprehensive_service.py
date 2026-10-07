"""Thin application orchestration over frozen services; no new NLP algorithms."""

import copy
import json
import os
import subprocess
import uuid
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter

from nlplab.comprehensive.contracts import (LABEL_NAMES, aggregate, document_id,
                                           token_spans, validate_annotations, validate_text)
from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.data.course import file_sha256
from nlplab.services.classification_stage2_service import ClassificationStage2Service


def application_operation(function):
    """Reentrant scheduling gate: system checks cannot race ordinary NLP jobs."""
    @wraps(function)
    def guarded(self, *args, **kwargs):
        if not self._operation_lock.acquire(timeout=2):
            raise ResourceUnavailableError("综合应用正在运行分析或系统测试，请等待当前任务完成")
        try:
            return function(self, *args, **kwargs)
        finally:
            self._operation_lock.release()
    return guarded


class ComprehensiveAnalysisService:
    TOKENIZER_RUN = "20260926_130612_32a5e168_smooth"
    CLASSIFIER_RUN = "20261001_133829_68ee50d7"
    CLUSTER_RUN = "20260930_102533_9c28e666"
    MAX_TEXT = 5000

    def __init__(self, settings, nlp, course, clustering, neural):
        self.settings, self.nlp, self.course = settings, nlp, course
        self.clustering, self.neural = clustering, neural
        self.classifier = ClassificationStage2Service(settings)
        self._lock, self._analysis_lock = RLock(), Lock()
        self._worker_lock, self._system_tests_lock = Lock(), Lock()
        self._operation_lock = RLock()
        self._cache, self._collections = OrderedDict(), OrderedDict()
        self._course_texts = None
        self._application_index = None

    def _run_worker(self, kind, payload, *, runtime=None):
        if not self._worker_lock.acquire(timeout=2):
            raise ResourceUnavailableError("本地分析正在处理其他任务，请稍后运行")
        try:
            return self._execute_worker(kind, payload, runtime=runtime)
        finally:
            self._worker_lock.release()

    def _execute_worker(self, kind, payload, *, runtime=None):
        environment = "exp2-venv" if kind in ("custom", "terms", "classify") else "hanlp-venv"
        runtime = runtime or self.settings.root / f"artifacts/runtime/{environment}/Scripts/python.exe"
        if not runtime.is_file():
            raise ResourceUnavailableError("本地分析资源不可用；不会下载或返回替代模型结果")
        try:
            job = subprocess.run([str(runtime), "-m", "nlplab.comprehensive.worker", "--root", str(self.settings.root)],
                                 input=json.dumps({"kind": kind, **payload}, ensure_ascii=False),
                                 capture_output=True, text=True, encoding="utf-8", timeout=150,
                                 cwd=self.settings.root,
                                 env={**os.environ, "PYTHONPATH": str(self.settings.root / "src"),
                                      "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ResourceUnavailableError("本地分析未完成，请稍后重试") from error
        if job.returncode:
            raise ResourceUnavailableError("本地模型无法加载或推理，不使用示例值替代")
        try:
            result = json.loads(job.stdout)
        except ValueError as error:
            raise ResourceUnavailableError("本地分析返回无效结果") from error
        if "error" in result:
            raise DataValidationError(result["error"])
        return result

    def status(self):
        root = self.settings.root
        model = root / f"outputs/exp3/{self.CLASSIFIER_RUN}/final/C2/model.json"
        return {"resources": [
            {"key": "segmentation", "label": "冻结中文分词", "ready": (root / "artifacts/models/msr-final-20260926_130612_32a5e168_smooth/metadata.json").is_file()},
            {"key": "pos", "label": "词性标注", "ready": (root / "artifacts/cache/pos_setup.json").is_file()},
            {"key": "ner", "label": "命名实体识别", "ready": (root / "artifacts/cache/ner_setup.json").is_file()},
            {"key": "classification", "label": "五类新闻分类", "ready": model.is_file()},
        ], "busy": self._analysis_lock.locked() or self._worker_lock.locked() or self._system_tests_lock.locked(),
            "limits": {"text": self.MAX_TEXT, "documents": 30, "batch_link": 10,
                       "dependency": 300, "summary": 500},
            "classification": {"model_id": "C2", "domain": "五类中文新闻", "labels": LABEL_NAMES},
            "notice": "资源存在不等于模型已执行；实际状态与耗时在分析后显示。"}

    @application_operation
    def analyze(self, text):
        validate_text(text, self.MAX_TEXT)
        key = document_id(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return {**copy.deepcopy(self._cache[key]), "cached": True}
        if not self._analysis_lock.acquire(timeout=2):
            raise ResourceUnavailableError("综合分析正在处理前一篇文本，请稍后运行")
        started = perf_counter()
        try:
            result = {"id": key, "original_text": text,
                      "lexical": {"tokens": [], "pos": [], "entities": []},
                      "classification": {}, "modules": {}, "provenance": {}, "cached": False}
            tick = perf_counter()
            try:
                segmented = self.nlp.analyze(text, ["segmentation"], self.TOKENIZER_RUN, "hmm1-smooth")
                tokens = segmented["segmentation"]["tokens"]
                result["lexical"]["tokens"] = token_spans(text, tokens)
                result["modules"]["segmentation"] = {"status": "ok", "elapsed_ms": round((perf_counter()-tick)*1000, 2)}
                result["provenance"]["segmentation"] = {"model": "HMM-1 Smooth", "run_id": self.TOKENIZER_RUN,
                                                           "model_path": segmented["run"].get("model_path"), "input_protocol": "original text"}
            except (DataValidationError, ResourceUnavailableError) as error:
                tokens = None
                result["modules"]["segmentation"] = {"status": "unavailable", "error": str(error), "elapsed_ms": round((perf_counter()-tick)*1000, 2)}
            if tokens is not None:
                try:
                    lexical = self._run_worker("lexical", {"text": text, "tokens": tokens})
                    result["modules"].update(lexical["modules"])
                    result["provenance"].update(lexical["provenance"])
                    for module, lexical_key, field in (("pos", "pos", "word"), ("ner", "entities", "text")):
                        if result["modules"][module]["status"] != "ok":
                            continue
                        try:
                            validate_annotations(text, lexical[lexical_key], field)
                            result["lexical"][lexical_key] = lexical[lexical_key]
                        except DataValidationError as error:
                            result["modules"][module] = {**result["modules"][module],
                                                         "status": "unavailable", "error": str(error)}
                except (DataValidationError, ResourceUnavailableError) as error:
                    for kind in ("pos", "ner"):
                        result["modules"][kind] = {"status": "unavailable", "error": str(error)}
            else:
                for kind in ("pos", "ner"):
                    result["modules"][kind] = {"status": "unavailable", "error": "基础分词未完成，词性/实体分析未执行"}
            tick = perf_counter()
            try:
                classified = next((value for doc_id, value in (self._application_index or {}).items()
                                   if value.get("application_text_id") == key), None)
                classified = copy.deepcopy(classified) if classified else self.classifier.predict(text, "C2", self.CLASSIFIER_RUN)
                if classified.get("model_id") != "C2" or classified.get("score_type") != "decision_score":
                    raise DataValidationError("09 仅接入冻结 C2 决策分数")
                from nlplab.comprehensive.classification_application import attach_evidence
                classified = classified if "evidence" in classified else attach_evidence(text, classified)
                result["classification"] = {**classified, "status": "ok", "domain": "五类中文新闻",
                                            "label_names": LABEL_NAMES,
                                            "notice": classified.get("notice") or "SVC decision score / margin 不是校准概率；非新闻文本仍会被分到这五类之一。"}
                result["modules"]["classification"] = {"status": "ok", "elapsed_ms": round((perf_counter()-tick)*1000, 2)}
                result["provenance"]["classification"] = {"model": "C2 · char χ² LinearSVC", "run_id": self.CLASSIFIER_RUN,
                                                             "input_protocol": "original full text → native C2 normalization/char bigram"}
            except (DataValidationError, ResourceUnavailableError) as error:
                result["modules"]["classification"] = {"status": "unavailable", "error": str(error), "elapsed_ms": round((perf_counter()-tick)*1000, 2)}
            result["status"] = "ok" if all(row["status"] == "ok" for row in result["modules"].values()) else "partial"
            result["elapsed_ms"] = round((perf_counter()-started)*1000, 2)
            if result["status"] == "ok":
                with self._lock:
                    self._cache[key] = copy.deepcopy(result)
                    while len(self._cache) > 64:
                        self._cache.popitem(last=False)
            return result
        finally:
            self._analysis_lock.release()

    @application_operation
    def optional(self, text, kind, start=None, end=None):
        from nlplab.comprehensive.workspace import contextual_result
        validate_text(text, self.MAX_TEXT)
        if (start is None) != (end is None):
            raise DataValidationError("原文选区必须同时提供start和end")
        if start is None:
            start, end = 0, len(text)
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise DataValidationError("原文选区范围无效")
        fragment = text[start:end]
        validate_text(fragment, {"dependency": 300, "summary": 500, "information_extraction": self.MAX_TEXT}.get(kind, 0))
        started = perf_counter()
        if kind == "dependency":
            result = self.neural.dependency(fragment)
        elif kind == "summary":
            result = self.neural.infer("llm", {"text": fragment, "task": "summary", "max_new_tokens": 48, "sampling": False})
        elif kind == "information_extraction":
            result = self._run_worker(kind, {"text": fragment})
        else:
            raise DataValidationError("未知更多分析任务")
        result = contextual_result(text, result, kind, start, end)
        return {"kind": kind, "document_id": document_id(text), "result": result,
                "source_span": {"start": start, "end": end, "text": fragment},
                "offset_space": "original_text Unicode code-point",
                "elapsed_ms": round((perf_counter()-started)*1000, 2), "scope": "explicit contextual optional analysis"}

    def _safe_course_texts(self):
        if self._course_texts is None:
            base = self.settings.root / "data/splits/thucnews_600"
            manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
            records = {}
            for split in ("train", "val"):
                path = base / f"{split}.jsonl"
                if file_sha256(path) != manifest["file_sha256"][path.name]:
                    raise DataValidationError("课程 Train/Validation 文件指纹不符")
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        records[row["id"]] = row["text"]  # No labels enter analysis.
            self._course_texts = records
        return self._course_texts

    def _remember(self, collection):
        with self._lock:
            self._collections[collection["id"]] = collection
            self._collections.move_to_end(collection["id"])
            while len(self._collections) > 8:
                self._collections.popitem(last=False)
        return self._public_collection(collection)

    @staticmethod
    def _public_collection(collection):
        value = {key: copy.deepcopy(item) for key, item in collection.items() if key not in ("texts", "analyses", "term_vectors", "application_predictions")}
        for doc in value["documents"]:
            doc["analyzed"] = collection["analyses"].get(doc["id"], {}).get("status") == "ok"
        return value

    def _collection(self, collection_id):
        with self._lock:
            if collection_id not in self._collections:
                raise ResourceUnavailableError("临时文档集合已失效，请重新读取或聚类")
            self._collections.move_to_end(collection_id)
            return self._collections[collection_id]

    def course_collection(self):
        with self._lock:
            saved = self._collections.get("course-r2-k5")
            if saved is not None:
                return self._public_collection(saved)
        self.clustering.verify_artifacts(self.CLUSTER_RUN)
        result = self.clustering.result(self.CLUSTER_RUN, 5)
        projection = self.clustering.projection(self.CLUSTER_RUN, 2, cache_only=True)
        coords = {row["id"]: row for row in projection["points"]}
        texts = self._safe_course_texts()
        documents = [{**{key: row[key] for key in ("id", "cluster_id", "snippet", "distance")},
                      "title": row["snippet"][:32], "x": coords[row["id"]]["x"], "y": coords[row["id"]]["y"],
                      "eligible": row["id"] in texts and len(texts[row["id"]]) <= self.MAX_TEXT}
                     for row in result["assignments"]]
        collection = {"id": "course-r2-k5", "mode": "course", "documents": documents,
                      "clusters": result["clusters"], "projection": {key: value for key, value in projection.items() if key != "points"},
                      "representation": {"space": "lsa_l2", "dimension": 100, "k": 5},
                      "provenance": {"run_id": self.CLUSTER_RUN, "frozen": True, "fit_executed": False},
                      "notice": "历史600篇冻结聚类；新增联动只分析Train/Validation白名单中≤5000字的全文。分类Test不新增推理或评价。", "texts": {}, "analyses": {}, "application_predictions": self._load_application_index(texts)}
        return self._remember(collection)

    @property
    def classification_index_path(self):
        return self.settings.root / "outputs/applications/lab4/classification-index.json"

    def _classification_signature(self):
        base = self.settings.root / "outputs/exp3" / self.CLASSIFIER_RUN
        names = ("artifact_manifest.json", "final/C2/model.json", "final/C2/parameters.npz",
                 "final/representation/representation.json", "final/representation/parameters.npz",
                 "final/representation/observed_df.json")
        return {"model_run_id": self.CLASSIFIER_RUN, "model_id": "C2",
                "source_sha256": {name: file_sha256(base / name) for name in names},
                "split_sha256": {split: file_sha256(self.settings.root / f"data/splits/thucnews_600/{split}.jsonl")
                                 for split in ("train", "val")}, "max_text": self.MAX_TEXT}

    def _load_application_index(self, texts):
        if self._application_index is not None:
            return self._application_index
        path = self.classification_index_path
        if not path.is_file():
            self._application_index = {}
            return self._application_index
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise DataValidationError("09应用分类缓存无法读取") from error
        if value.get("schema_version") != 1 or value.get("signature") != self._classification_signature():
            raise DataValidationError("09应用分类缓存来源指纹不符，拒绝使用")
        from hashlib import sha256
        digest = sha256(json.dumps(value["records"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if digest != value.get("records_sha256"):
            raise DataValidationError("09应用分类缓存内容指纹不符")
        records = {}
        for row in value["records"]:
            doc_id, prediction = row["id"], row["classification"]
            if doc_id in records or doc_id not in texts or len(texts[doc_id]) > self.MAX_TEXT or row["text_id"] != document_id(texts[doc_id]):
                raise DataValidationError("09分类缓存含白名单外或正文不符的文档")
            if prediction.get("model_id") != "C2" or prediction.get("run_id") != self.CLASSIFIER_RUN or prediction.get("score_type") != "decision_score" or prediction.get("prediction") not in LABEL_NAMES:
                raise DataValidationError("09应用分类缓存模型协议不符")
            records[doc_id] = {**prediction, "application_text_id": row["text_id"]}
        self._application_index = records
        return records

    @application_operation
    def build_classification_index(self):
        """Explicit application inference only; never formal results or NER."""
        from hashlib import sha256
        from nlplab.data.sequence_annotations import atomic_json
        self.course_collection()
        collection = self._collection("course-r2-k5")
        texts = self._safe_course_texts()
        existing = self._load_application_index(texts)
        eligible = [d for d in collection["documents"] if d["eligible"]]
        missing = [{"id": d["id"], "text": texts[d["id"]]} for d in eligible if d["id"] not in existing]
        started = perf_counter()
        new_records = []
        if missing:
            output = self._run_worker("classify", {"run_id": self.CLASSIFIER_RUN, "documents": missing})
            if output.get("fit_executed") is not False or output.get("ner_executed") is not False or output.get("gold_labels_used") is not False or output.get("formal_evaluation_executed") is not False:
                raise DataValidationError("分类应用批次返回了不允许的执行语义")
            new_records = output["records"]
            if {r["id"] for r in new_records} != {d["id"] for d in missing} or len(new_records) != len(missing):
                raise DataValidationError("应用分类批次文档未完全对齐")
            for row in new_records:
                result = row["classification"]
                if row["text_id"] != document_id(texts[row["id"]]) or result.get("model_id") != "C2" or result.get("run_id") != self.CLASSIFIER_RUN or result.get("score_type") != "decision_score" or result.get("prediction") not in LABEL_NAMES:
                    raise DataValidationError("应用分类结果不符合冻结C2协议")
        merged = {**existing, **{r["id"]: {**r["classification"], "application_text_id": r["text_id"]} for r in new_records}}
        records = [{"id": doc_id, "text_id": c["application_text_id"], "classification": {k:v for k,v in c.items() if k != "application_text_id"}}
                   for doc_id, c in sorted(merged.items())]
        if new_records or not self.classification_index_path.is_file():
            value = {"schema_version": 1, "scope": "09 frozen-model application inference cache; Train/Validation only",
                     "signature": self._classification_signature(), "records": records,
                     "records_sha256": sha256(json.dumps(records, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                     "fit_executed": False, "ner_executed": False, "gold_labels_used": False,
                     "test_text_accessed": False, "formal_evaluation_executed": False,
                     "formal_run_written": False, "elapsed_ms": round((perf_counter()-started)*1000, 2)}
            self.classification_index_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.classification_index_path, value)
        with self._lock:
            self._application_index = merged
            collection["application_predictions"] = merged
        return {"eligible": len(eligible), "classification_known": len(merged), "new_inferences": len(new_records),
                "ner_executed": False, "test_text_accessed": False, "gold_labels_used": False,
                "formal_run_written": False, "formal_evaluation_executed": False,
                "elapsed_ms": round((perf_counter()-started)*1000, 2)}

    @application_operation
    def custom_collection(self, documents, k):
        if not 3 <= len(documents) <= 30 or not 2 <= k < len(documents):
            raise DataValidationError("临时聚类需3–30篇文档，K在2与文档数-1之间")
        if sum(len(row["text"]) for row in documents) > 50000:
            raise DataValidationError("自定义文档总长度最多50000字")
        rows = []
        for index, doc in enumerate(documents):
            validate_text(doc["text"], self.MAX_TEXT)
            rows.append({"id": f"custom-{index+1}", "text": doc["text"], "title": doc.get("title") or f"文档 {index+1}"})
        if len({row["text"] for row in rows}) <= k:
            raise DataValidationError("可区分文档不足，请提供超过K篇不同正文")
        result = self._run_worker("custom", {"documents": rows, "k": k})
        coords = {row["id"]: row for row in result["projection"]["points"]}
        titles = {row["id"]: row["title"] for row in rows}
        docs = [{**row, "title": titles[row["id"]], "x": coords[row["id"]]["x"], "y": coords[row["id"]]["y"], "eligible": True}
                for row in result["assignments"]]
        return self._remember({"id": "custom-" + uuid.uuid4().hex[:16], "mode": "custom", "documents": docs,
                               "clusters": result["clusters"], "projection": {key: value for key, value in result["projection"].items() if key != "points"},
                               "representation": result["representation"], "provenance": result["provenance"],
                               "notice": "临时分析：当前集合独立fit，二维坐标仅为显示投影；不写正式结果、不更新latest。",
                               "texts": {row["id"]: row["text"] for row in rows}, "term_vectors": result.get("term_vectors"), "analyses": {}})

    def profile(self, collection_id, doc_id, *, analyze=False):
        collection = self._collection(collection_id)
        doc = next((row for row in collection["documents"] if row["id"] == doc_id), None)
        if doc is None:
            raise DataValidationError("文档不在当前集合中")
        text = collection["texts"].get(doc_id)
        if collection["mode"] == "course":
            text = self._safe_course_texts().get(doc_id)
        value = {"id": doc_id, "title": doc["title"], "snippet": doc["snippet"], "eligible": doc["eligible"],
                 "original_text": text if text is not None else None,
                 "clustering": {key: doc[key] for key in ("cluster_id", "x", "y", "distance")},
                 "analysis": copy.deepcopy(collection["analyses"].get(doc_id)),
                 "notice": None if doc["eligible"] else "仅查看历史聚类摘要；该文档不在新增联动的白名单/长度范围内。"}
        if not doc["eligible"]:
            value["original_text"] = None
        classified = collection.get("application_predictions", {}).get(doc_id)
        if classified and value["analysis"] is None:
            value["analysis"] = {"id": document_id(text), "original_text": text, "status": "classification_only",
                                 "lexical": {"tokens": [], "pos": [], "entities": []},
                                 "modules": {"classification": {"status": "ok", "cached": True}},
                                 "classification": copy.deepcopy(classified),
                                 "provenance": {"classification": {"run_id": self.CLASSIFIER_RUN, "scope": "09 application inference cache"}}}
        elif classified:
            analysis = value["analysis"]
            value["analysis"] = {**analysis, "classification": copy.deepcopy(classified),
                                 "modules": {**analysis["modules"], "classification": {"status": "ok", "cached": True}},
                                 "provenance": {**analysis.get("provenance", {}),
                                                "classification": {"run_id": self.CLASSIFIER_RUN, "scope": "09 application inference cache"}}}
            if all(m["status"] == "ok" for m in value["analysis"]["modules"].values()):
                value["analysis"]["status"] = "ok"
        if analyze:
            if not doc["eligible"]:
                raise DataValidationError(value["notice"])
            record = self.analyze(text)
            with self._lock:
                collection["analyses"][doc_id] = record
            value["analysis"] = record
        return value

    @application_operation
    def link(self, collection_id, document_ids):
        if len(document_ids) > 10 or len(set(document_ids)) != len(document_ids):
            raise DataValidationError("每次联动最多10个不同文档ID；空列表仅读取已有统计")
        collection = self._collection(collection_id)
        eligible = {row["id"] for row in collection["documents"] if row["eligible"]}
        if any(doc_id not in eligible for doc_id in document_ids):
            raise DataValidationError("联动包含集合外或不可分析文档，未启动任何模型")
        for doc_id in document_ids:
            self.profile(collection_id, doc_id, analyze=True)
        with self._lock:
            snapshot = copy.deepcopy(collection)
        return aggregate(snapshot)

    def workspace_snapshot(self, collection_id):
        from nlplab.comprehensive.workspace import sampling_plan
        collection = self._collection(collection_id)
        with self._lock:
            snapshot = copy.deepcopy(collection)
        return {"collection": self._public_collection(snapshot), "link": aggregate(snapshot),
                "sampling": sampling_plan(snapshot)}

    def sampling_plan(self, collection_id, scope_ids=None, limit=10):
        from nlplab.comprehensive.workspace import sampling_plan
        collection = self._collection(collection_id)
        with self._lock:
            snapshot = copy.deepcopy(collection)
        return sampling_plan(snapshot, scope_ids, limit)

    def selection_terms(self, collection_id, document_ids):
        collection = self._collection(collection_id)
        known = {row["id"] for row in collection["documents"]}
        if len(document_ids) > len(known) or len(set(document_ids)) != len(document_ids) or not set(document_ids) <= known:
            raise DataValidationError("关键词选区包含无效文档ID")
        if not document_ids:
            return {"document_ids": [], "terms": [], "source": "empty selection", "fit_executed": False}
        if collection["mode"] == "course":
            # Numeric frozen vectors only; no raw text or classifier Test access.
            return self._execute_worker("terms", {"run_id": self.CLUSTER_RUN, "document_ids": document_ids})
        vectors = collection.get("term_vectors")
        if vectors is None:
            return {"document_ids": document_ids, "terms": None, "source": "this temporary collection did not save term vectors", "fit_executed": False}
        from collections import Counter
        weights = Counter()
        for doc_id in document_ids:
            for row in vectors[doc_id]:
                weights[row["term"]] += row["weight"] / len(document_ids)
        return {"document_ids": document_ids, "terms": [{"term": term, "weight": weight}
                for term, weight in weights.most_common(12)], "source": "mean of selected documents' fitted word TF-IDF vectors", "fit_executed": False}

    def system_tests(self):
        from nlplab.comprehensive.system_tests import run_system_tests
        if not self._system_tests_lock.acquire(blocking=False):
            raise ResourceUnavailableError("系统测试已在运行，请等待本次记录完成")
        acquired = False
        try:
            acquired = self._operation_lock.acquire(timeout=2)
            if not acquired or self._analysis_lock.locked() or self._worker_lock.locked():
                raise ResourceUnavailableError("先前分析仍在运行，请结束后再执行系统测试")
            return run_system_tests(self)
        finally:
            if acquired:
                self._operation_lock.release()
            self._system_tests_lock.release()
