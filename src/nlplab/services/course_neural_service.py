"""Bounded offline CPU inference in the existing, unmodified neural runtime."""

import json
import logging
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from threading import Lock, Thread
from queue import Queue, Empty
from tempfile import TemporaryFile
from time import perf_counter

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.course_neural.dependency import attachment_metrics, sentence_spans, tree_record
from nlplab.course_neural.resources import digest, frozen_bilstm, prepared_resource, verified_bert


class CourseNeuralService:
    TOKENIZER_RUN = "20260926_130612_32a5e168_smooth"
    def __init__(self, settings, nlp):
        self.settings, self.nlp = settings, nlp
        self.runtime = settings.root / "artifacts/runtime/hanlp-venv/Scripts/python.exe"
        self.cache, self.lock = OrderedDict(), Lock()

    def resources(self):
        def ready(name):
            try:
                prepared_resource(self.settings.root, name, verify=False)
                return True
            except ResourceUnavailableError:
                return False
        try:
            bert, _ = verified_bert(self.settings.root, verify=False)
        except ResourceUnavailableError:
            bert = None
        try:
            frozen_bilstm(self.settings.root)
            bilstm_ready = True
        except ResourceUnavailableError:
            bilstm_ready = False
        return {"runtime_ready": self.runtime.is_file(), "device": "CPU", "training": False,
                "busy": self.lock.locked(),
                "bilstm_ready": bilstm_ready,
                "bert_ready": bert is not None, "bert_source": "bert-base-chinese · existing local snapshot",
                "dependency_ready": ready("hanlp-ctb9-dep-electra-small"),
                "llm_ready": ready("qwen2.5-0.5b-instruct"),
                "integrity_status": "Manifest and file sizes checked here; SHA-256 verified before each uncached inference",
                "policy": "Explicit preparation only; inference is offline; frozen experiments remain read-only"}

    def infer(self, kind, payload):
        if kind not in ("dependency", "bert", "llm", "bilstm"):
            raise DataValidationError("未知课程推理任务")
        if not self.runtime.is_file():
            raise ResourceUnavailableError("现有 PyTorch CPU runtime 不可用")
        value = {"kind": kind, **payload}
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if not self.lock.acquire(timeout=2):
            raise ResourceUnavailableError("课程模型正在处理其他请求；请稍后重试")
        try:
            if key in self.cache:
                self.cache.move_to_end(key)
                return {**self.cache[key], "cached": True}
            env = {**os.environ, "PYTHONPATH": str(self.settings.root / "src"),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
                   "HANLP_HOME": str(self.settings.root / "artifacts/cache/hanlp"),
                   "PYTHONIOENCODING": "utf-8"}
            started = perf_counter()
            try:
                result = subprocess.run([str(self.runtime), "-m", "nlplab.course_neural.worker", "--root", str(self.settings.root)],
                                        input=key, capture_output=True, text=True, encoding="utf-8", env=env,
                                        cwd=self.settings.root, timeout=120,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ResourceUnavailableError("本地课程模型推理未完成；请稍后重试") from error
            if result.returncode:
                logging.getLogger(__name__).error("Course worker %s failed: %s", kind, result.stderr[-1200:])
                raise ResourceUnavailableError("本地课程模型无法加载或推理；资源准备记录可用于定位")
            try:
                data = json.loads(result.stdout)
            except ValueError as error:
                raise ResourceUnavailableError("课程模型返回无效结果") from error
            if "error" in data:
                raise DataValidationError(data["error"])
            data.update(elapsed_ms=round((perf_counter() - started) * 1000, 2), cached=False)
            self.cache[key] = data
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
            return data
        finally:
            self.lock.release()

    def stream_llm(self, payload, stop):
        """Live token events only; the old JSON endpoint/cache remains compatible."""
        if not self.runtime.is_file():
            raise ResourceUnavailableError("现有 PyTorch CPU runtime 不可用")
        return self._stream_llm(payload, stop)

    def _stream_llm(self, payload, stop):
        acquired, process = False, None
        started = perf_counter()
        try:
            acquired = self.lock.acquire(timeout=2)
            if not acquired:
                raise ResourceUnavailableError("课程模型正在处理其他请求；请稍后重试")
            if stop.is_set():
                return
            yield {"event": "status", "data": {"message": "正在校验并加载本地模型…"}}
            value = {"kind": "llm", **payload}
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            env = {**os.environ, "PYTHONPATH": str(self.settings.root / "src"),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
                   "PYTHONIOENCODING": "utf-8"}
            with TemporaryFile() as errors:
                process = subprocess.Popen(
                    [str(self.runtime), "-u", "-m", "nlplab.course_neural.worker",
                     "--root", str(self.settings.root), "--stream"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                    text=True, encoding="utf-8", env=env, cwd=self.settings.root,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                process.stdin.write(key)
                process.stdin.close()
                messages = Queue()
                def read_lines():
                    try:
                        for line in process.stdout:
                            messages.put(line)
                    finally:
                        messages.put(None)
                reader = Thread(target=read_lines, daemon=True)
                reader.start()
                complete = False
                while not stop.is_set():
                    if perf_counter() - started > 120:
                        raise ResourceUnavailableError("本地课程模型推理超过时间限制")
                    try:
                        line = messages.get(timeout=0.5)
                    except Empty:
                        yield {"event": "heartbeat", "data": {}}
                        continue
                    if line is None:
                        break
                    try:
                        event = json.loads(line)
                        if event["event"] not in ("status", "token", "done", "error") or not isinstance(event["data"], dict):
                            raise ValueError("invalid event")
                    except (ValueError, KeyError, TypeError) as error:
                        raise ResourceUnavailableError("课程模型返回无效流式结果") from error
                    if event["event"] == "done":
                        process.wait(timeout=5)
                        if process.returncode:
                            raise ResourceUnavailableError("本地课程模型流式生成失败")
                        data = event["data"]
                        data.update(elapsed_ms=round((perf_counter()-started)*1000, 2), cached=False)
                        self.cache[key] = data
                        while len(self.cache) > 8:
                            self.cache.popitem(last=False)
                        complete = True
                    yield event
                    if event["event"] in ("done", "error"):
                        return
                if not complete and not stop.is_set():
                    errors.seek(0)
                    logging.getLogger(__name__).error("Streaming worker ended: %s", errors.read()[-1200:].decode("utf-8", errors="replace"))
                    raise ResourceUnavailableError("本地课程模型生成中断；未返回完整结果")
        except (ResourceUnavailableError, OSError, subprocess.TimeoutExpired) as error:
            yield {"event": "error", "data": {"message": str(error) if isinstance(error, ResourceUnavailableError) else "本地课程模型流式生成失败"}}
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                    if process.stdout is not None:
                        process.stdout.close()
            finally:
                if acquired:
                    self.lock.release()

    def dependency(self, text):
        if not text.strip() or len(text) > 300:
            raise DataValidationError("请输入 1–300 字中文短句")
        spans, words = sentence_spans(text), []
        if not spans or len(spans) > 6:
            raise DataValidationError("依存分析只支持 1–6 个句子")
        info = self.nlp.store.read(self.TOKENIZER_RUN, "run.json")
        if info.get("is_demo") or info.get("run_type") != "frozen-final-benchmark" or info.get("variant") != "smooth":
            raise ResourceUnavailableError("课程解析需要正式冻结 HMM-1 Smooth，不能使用 demo/latest")
        for span in spans:
            tokens = self.nlp.analyze(span["text"], ["segmentation"], self.TOKENIZER_RUN, "hmm1-smooth")["segmentation"]["tokens"]
            tokens = [word for word in tokens if not word.isspace()]
            if not 1 <= len(tokens) <= 60:
                raise DataValidationError("每个句子必须包含 1–60 个非空词")
            words.append(tokens)
        raw = self.infer("dependency", {"sentences": words})
        if len(raw.get("sentences", [])) != len(words):
            raise DataValidationError("解析器句子数量不一致")
        return {**raw, "text": text, "tokenizer": "frozen HMM-1 Smooth",
                "tokenizer_run": self.TOKENIZER_RUN,
                "sentences": [{**span, **tree_record(text, tokens, prediction, span["start"], span["end"])}
                              for span, tokens, prediction in zip(spans, words, raw["sentences"])]}

    def dependency_evaluation(self, use_saved=True):
        directory = self.settings.root / "outputs/course_dependency"
        latest = directory / "evaluation_latest.json"
        if use_saved and latest.is_file():
            reference = json.loads(latest.read_text(encoding="utf-8"))
            path = (directory / reference["run_id"] / "evaluation.json").resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != reference["sha256"]:
                raise DataValidationError("教学评价记录哈希校验失败")
            record = json.loads(path.read_text(encoding="utf-8"))
            gold_path = self.settings.root / "src/nlplab/course_neural/teaching_gold.json"
            if digest(gold_path) != record["gold_reference"]["sha256"]:
                raise DataValidationError("教学 Gold 与已保存评价记录不一致")
            return record
        gold_path = self.settings.root / "src/nlplab/course_neural/teaching_gold.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        examples = gold["examples"]
        raw = self.infer("dependency", {"sentences": [sample["tokens"] for sample in examples]})
        if len(raw["sentences"]) != len(examples):
            raise DataValidationError("教学评测句子数量不一致")
        samples, totals = [], {False: [0, 0, 0], True: [0, 0, 0]}
        for sample, prediction in zip(examples, raw["sentences"]):
            record = tree_record(sample["text"], sample["tokens"], prediction)
            metrics = {}
            for exclude in (False, True):
                item = attachment_metrics(sample["tokens"], sample["heads"], sample["relations"],
                                          prediction["heads"], prediction["relations"], exclude)
                metrics["exclude_punctuation" if exclude else "include_punctuation"] = item
                for i, key in enumerate(("support", "correct_heads", "correct_labels")):
                    totals[exclude][i] += item[key]
            samples.append({"id": sample["id"], "text": sample["text"], "words": sample["tokens"],
                            "gold_heads": sample["heads"], "gold_relations": sample["relations"],
                            "prediction": record, "metrics": metrics["exclude_punctuation"], "protocol_metrics": metrics})
        def total(values):
            n, heads, labels = values
            return {"support": n, "correct_heads": heads, "correct_labels": labels,
                    "uas": heads/n if n else None, "las": labels/n if n else None}
        return {"protocol": gold["evaluation_protocol"], "title": gold["title"], "samples": samples,
                "aggregate": total(totals[True]), "including_punctuation": total(totals[False]),
                "source": raw["source"], "environment": raw["environment"], "elapsed_ms": raw["elapsed_ms"]}
