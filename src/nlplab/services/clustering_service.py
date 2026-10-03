"""实验二执行与只读结果服务；复用 RunStore 的 exp2 命名空间。"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import yaml

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.core.io import write_json
from nlplab.core.schema import text_sha256
from nlplab.data.clustering_input import load_clustering_input
from nlplab.data.corpus import write_jsonl
from nlplab.data.course import file_sha256
from nlplab.services.experiment_service import RunStore


class ClusteringService:
    def __init__(self, settings):
        self.settings = settings
        self.store = RunStore(settings, "exp2")
        self._projection_lock = Lock()

    def _inspection_cache(self, run_id):
        directory = self.store.resolve(run_id)
        cache = self.settings.outputs / "exp2/visualizations" / directory.name
        target = cache / "inspection.json"
        with self._projection_lock:
            if not target.is_file():
                runtime = self.settings.root / "artifacts/runtime/exp2-venv/Scripts/python.exe"
                executable = str(runtime) if runtime.is_file() else sys.executable
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(self.settings.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                }
                try:
                    job = subprocess.run(
                        [
                            executable,
                            "-m",
                            "nlplab.clustering.inspection",
                            "--root",
                            str(self.settings.root),
                            "--run-id",
                            directory.name,
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=60,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise ResourceUnavailableError("只读文档检视缓存生成失败") from error
                if job.returncode:
                    raise ResourceUnavailableError(
                        "无法验证冻结向量、中心与显示坐标，请检查实验二运行环境"
                    )
            manifest_path = cache / "inspection_manifest.json"
            if not manifest_path.is_file():
                raise DataValidationError("文档检视缓存缺失校验清单")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if file_sha256(target) != manifest["inspection_sha256"]:
                raise DataValidationError("文档检视缓存指纹不符")
            value = json.loads(target.read_text(encoding="utf-8"))
            for name, digest in value["provenance"]["source_files"].items():
                path = (self.settings.root / name).resolve()
                if (
                    not path.is_relative_to(self.settings.root.resolve())
                    or not path.is_file()
                    or file_sha256(path) != digest
                ):
                    raise DataValidationError("文档检视的冻结来源指纹不符")
            if value["run_id"] != directory.name:
                raise DataValidationError("文档检视 Run 不一致")
            return value

    def inspection(self, run_id="latest", k=5):
        value = self._inspection_cache(run_id)
        if str(k) not in value["results"]:
            raise DataValidationError("此 Run 未保存请求的 K")
        return {
            "run_id": value["run_id"],
            "k": k,
            "space": value["space"],
            "dimension": value["dimension"],
            "document_count": value["document_count"],
            "centroids": value["results"][str(k)]["centroids"],
            "provenance": value["provenance"],
        }

    def document(self, run_id="latest", k=5, document_id=""):
        value = self._inspection_cache(run_id)
        if str(k) not in value["results"]:
            raise DataValidationError("此 Run 未保存请求的 K")
        index = next(
            (i for i, item in enumerate(value["documents"]) if item["id"] == document_id), None
        )
        if index is None:
            raise DataValidationError("文档不在此冻结 Run 中")
        result = self.result(value["run_id"], k)
        assignment = result["assignments"][index]
        if assignment["id"] != document_id:
            raise DataValidationError("文档检视与冻结分配顺序不一致")
        derived = value["results"][str(k)]
        return {
            **value["documents"][index],
            **assignment,
            "run_id": value["run_id"],
            "k": k,
            "space": value["space"],
            "nearest_cluster": derived["nearest"][index],
            "centroid_distances": [
                {"cluster_id": i, "distance": distance}
                for i, distance in enumerate(derived["distances"][index])
            ],
            "gold_role": "posthoc_only",
            "formal_fit_executed": False,
            "calculation": self._document_calculation(value["run_id"], document_id),
        }

    def _document_calculation(self, run_id, document_id):
        """Read a separately verified arithmetic cache; never build or fit in Web."""
        directory = self.settings.outputs / "exp2/visualizations" / run_id
        target = directory / "calculation.json"
        if not target.is_file():
            return None
        manifest = json.loads((directory / "calculation_manifest.json").read_text(encoding="utf-8"))
        if file_sha256(target) != manifest["sha256"]:
            raise DataValidationError("计算演示缓存指纹不符")
        value = json.loads(target.read_text(encoding="utf-8"))
        if value["run_id"] != run_id:
            raise DataValidationError("计算演示 Run 不一致")
        for name, digest in value["provenance"]["source_files"].items():
            source = (self.settings.root / name).resolve()
            if (
                not source.is_relative_to(self.settings.root.resolve())
                or file_sha256(source) != digest
            ):
                raise DataValidationError("计算演示的冻结来源指纹不符")
        return {
            **value["documents"][document_id],
            "document_count": value["document_count"],
            "config": value["config"],
            "provenance": value["provenance"],
        }

    def projection(self, run_id="latest", dimensions=2, *, cache_only=False):
        if dimensions not in (2, 3):
            raise DataValidationError("投影维数仅支持 2 或 3")
        directory = self.store.resolve(run_id)
        if dimensions == 2:
            return self.store.read(directory.name, "projection.json")
        cache = self.settings.outputs / "exp2/visualizations" / directory.name
        target = cache / "projection3d.json"
        with self._projection_lock:
            if not target.is_file():
                if cache_only:
                    raise ResourceUnavailableError(
                        "冻结 Run 缺少既有三维显示坐标；工作台不会重新拟合投影"
                    )
                runtime = self.settings.root / "artifacts/runtime/exp2-venv/Scripts/python.exe"
                executable = str(runtime) if runtime.is_file() else sys.executable
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(self.settings.root / "src"),
                    "PYTHONIOENCODING": "utf-8",
                }
                try:
                    job = subprocess.run(
                        [
                            executable,
                            "-m",
                            "nlplab.clustering.visualization",
                            "--root",
                            str(self.settings.root),
                            "--run-id",
                            directory.name,
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=60,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise ResourceUnavailableError(
                        "三维可视化缓存生成失败，请检查实验二运行环境"
                    ) from error
                if job.returncode:
                    raise ResourceUnavailableError("三维可视化需要可用的实验二 sklearn 运行环境")
            if not (cache / "manifest.json").is_file():
                raise DataValidationError("三维投影缓存缺失校验清单")
            manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
            if file_sha256(target) != manifest["projection3d_sha256"]:
                raise DataValidationError("三维投影缓存指纹不符")
            value = json.loads(target.read_text(encoding="utf-8"))
            model_directory = self.settings.models / "exp2" / directory.name
            vectors_path = model_directory / "document_vectors.npy"
            if not vectors_path.is_file():
                vectors_path = model_directory / "document_vectors.npz"
            if (
                file_sha256(vectors_path) != value["source"]["vectors_sha256"]
                or file_sha256(directory / "input_ids.json") != value["source"]["ids_sha256"]
            ):
                raise DataValidationError("三维投影来源向量或文档 ID 指纹不符")
            return value

    def overview(self, run_id="latest"):
        value = self.store.read(run_id, "overview.json")
        return {**value, "run": self.store.read(run_id, "run.json")}

    def result(self, run_id="latest", k=5):
        config = self.store.read(run_id, "config.json")
        if k not in config["k_values"]:
            raise DataValidationError("此 Run 未执行请求的 K")
        name = f"reference/k{k}.json" if k == config["reference_k"] else f"results/k{k}.json"
        return self.store.read(run_id, name)

    def list_runs(self):
        items = self.store.list_runs()
        try:
            latest = self.store.read("latest", "run.json")["run_id"]
        except ResourceUnavailableError:
            latest = None
        selection_path = self.settings.outputs / "exp2/final_selection.json"
        selection = (
            json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_path.is_file()
            else {}
        )
        return {"items": items, "latest": latest, "primary": selection.get("primary_run_id")}

    def final_summary(self, k=5):
        path = self.settings.outputs / "exp2/final_selection.json"
        if not path.is_file():
            return {"selection": None, "items": []}
        selection = json.loads(path.read_text(encoding="utf-8"))
        comparison_path = (self.settings.outputs / selection["comparison_path"]).resolve()
        if not comparison_path.is_relative_to(self.settings.outputs.resolve()):
            raise DataValidationError("对照路径超出 outputs 目录")
        if file_sha256(comparison_path) != selection["comparison_sha256"]:
            raise DataValidationError("最终方案的对照来源指纹不符")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        items = []
        for name, run_id in selection["representations"].items():
            saved = comparison["representations"][name]
            if saved["run_id"] != run_id:
                raise DataValidationError("最终方案与冻结对照 Run 不一致")
            result = self.result(run_id, k)
            sweep = next(row for row in saved["sweep"] if row["k"] == k)
            items.append(
                {
                    "representation_id": name,
                    "run_id": run_id,
                    "representation": saved["representation"],
                    "config": saved["config"],
                    "internal": result["internal"],
                    "external": result["external"],
                    "stability": sweep["stability"],
                    "min_cluster_size": sweep["min_cluster_size"],
                    "max_cluster_size": sweep["max_cluster_size"],
                    "clusters": result["clusters"],
                }
            )
        return {"selection": selection, "k": k, "items": items}

    def comparison_view(self, k=5):
        value = self.final_summary(k)
        for item in value["items"]:
            result = self.result(item["run_id"], k)
            assignments = {row["id"]: row for row in result["assignments"]}
            projection = self.store.read(item["run_id"], "projection.json")
            item["projection"] = {
                "purpose": "visualization_only",
                "points": [
                    {**point, "cluster_id": assignments[point["id"]]["cluster_id"]}
                    for point in projection["points"]
                ],
                "centroids": self.inspection(item["run_id"], k)["centroids"],
            }
            item["sweep"] = self.store.read(item["run_id"], "overview.json")["sweep"]
        return value

    def _prepare_formal(self, config_path=None):
        # 此处仅加载严格无标签文件；冻结 gold 全文在所有 fit 完成后才打开。
        root = self.settings.root
        input_path = root / "data/splits/thucnews_600/clustering_all.jsonl"
        split_path = input_path.parent / "manifest.json"
        documents = load_clustering_input(input_path, manifest_path=split_path)
        if len(documents) != 600:
            raise DataValidationError("正式实验必须使用冻结的全部 600 篇")
        config_path = config_path or root / "configs/clustering.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["k_values"] != list(range(2, 11)) or config["reference_k"] != 5:
            raise DataValidationError("正式协议要求 K=2…10 与 K=5 reference")
        from nlplab.segmentation.hmm_segmenter import HMMSegmenter

        tokenizer_store = RunStore(self.settings)
        model = tokenizer_store.model(config["tokenizer_run_id"])
        segmenter = HMMSegmenter(model)
        info = tokenizer_store.read(config["tokenizer_run_id"], "run.json")
        model_path = Path(info["model_path"])
        tokenizer_info = {
            "name": "frozen-hmm1-smooth",
            "run_id": config["tokenizer_run_id"],
            "model_files": {
                str(p.relative_to(root)): file_sha256(p)
                for p in sorted(model_path.glob("*"))
                if p.is_file()
            },
            "training_executed": False,
        }
        dataset = {
            "name": "THUCNews personal-600 fulltext",
            "count": len(documents),
            "input_fields": ["id", "text"],
            "input_path": str(input_path.relative_to(root)),
            "input_sha256": file_sha256(input_path),
            "split_manifest_sha256": file_sha256(split_path),
            "ids_sha256": text_sha256(
                json.dumps([doc.id for doc in documents], ensure_ascii=False)
            ),
            "config_file_sha256": file_sha256(config_path),
        }

        def gold_loader():
            from nlplab.data.personal_selection import verify_subset

            path = root / "data/course/thucnews_600.jsonl"
            records, manifest = verify_subset(path, dataset="THUCNews", count=600)
            provenance = {
                "phase": "posthoc_after_all_fits",
                "path": str(path.relative_to(root)),
                "data_sha256": file_sha256(path),
                "source_hashes": manifest["source_hashes"],
                "selection_manifest_sha256": file_sha256(
                    path.with_name("thucnews_600_manifest.json")
                ),
            }
            return records, provenance

        return documents, segmenter, config, gold_loader, dataset, tokenizer_info

    def run_formal(self, config_path=None, progress=print):
        documents, segmenter, config, gold_loader, dataset, tokenizer_info = self._prepare_formal(
            config_path
        )
        return self.run(
            documents,
            lambda text: segmenter.segment(text).tokens,
            config,
            gold_loader=gold_loader,
            dataset=dataset,
            tokenizer_info=tokenizer_info,
            is_demo=False,
            progress=progress,
        )

    def run(
        self,
        documents,
        segment,
        config,
        *,
        gold_loader,
        dataset,
        tokenizer_info,
        is_demo=True,
        progress=lambda message: None,
    ):
        from nlplab.clustering.experiment import execute_unsupervised

        started = perf_counter()
        execution = execute_unsupervised(documents, segment, config, progress)
        # 标签只能在所有 K/seed 的 fit、指标、代表文档、投影完成之后关联。
        gold_records, gold_provenance = gold_loader()
        return self._persist_execution(
            documents,
            config,
            execution,
            gold_records,
            gold_provenance,
            dataset=dataset,
            tokenizer_info=tokenizer_info,
            is_demo=is_demo,
            elapsed_seconds=perf_counter() - started,
        )

    def run_representation_batch(
        self,
        documents,
        segment,
        configs,
        *,
        gold_loader,
        dataset,
        tokenizer_info,
        tokenized=None,
        is_demo=True,
        progress=lambda message: None,
    ):
        """两种表示的全部 fit 完成后才打开 gold；不改已有 latest 指针。"""
        from nlplab.clustering.experiment import execute_unsupervised

        executions = []
        for config in configs:
            started = perf_counter()
            execution = execute_unsupervised(
                documents, segment, config, progress, tokenized=tokenized
            )
            executions.append((config, execution, perf_counter() - started))
        gold_records, gold_provenance = gold_loader()
        return [
            self._persist_execution(
                documents,
                config,
                execution,
                gold_records,
                gold_provenance,
                dataset=dataset,
                tokenizer_info=tokenizer_info,
                is_demo=is_demo,
                elapsed_seconds=elapsed,
                publish_latest=False,
            )
            for config, execution, elapsed in executions
        ]

    def run_representation_comparison(self, protocol_path, progress=print):
        """显式 R1/R2 批次；R0 只读，协议必须与冻结 baseline 一致。"""
        from copy import deepcopy

        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        baseline_id = protocol["baseline_run_id"]
        self.verify_artifacts(baseline_id)
        baseline = self.overview(baseline_id)
        original_config = self.store.read(baseline_id, "config.json")
        if (
            original_config["k_values"] != list(range(2, 11))
            or original_config["stability_seeds"] != [42, 43, 44]
            or original_config["reference_k"] != 5
            or original_config["kmeans"]["random_state"] != 42
        ):
            raise DataValidationError("baseline 不符合固定 K/seeds/reference 协议")
        documents, _, _, gold_loader, dataset, _ = self._prepare_formal()
        for key in ("input_sha256", "ids_sha256", "split_manifest_sha256"):
            if dataset[key] != baseline["dataset"][key]:
                raise DataValidationError("表示对照输入必须与 baseline 完全相同")
        token_path = self.store.resolve(baseline_id) / "tokens.jsonl"
        token_rows = [
            json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()
        ]
        if any(set(row) != {"id", "tokens"} for row in token_rows) or [
            row["id"] for row in token_rows
        ] != [doc.id for doc in documents]:
            raise DataValidationError("baseline 无标签 token 缓存的 ID/字段不符")
        configs = []
        for representation_id in ("R1", "R2"):
            config = deepcopy(original_config)
            config.update(
                {
                    "representation_id": representation_id,
                    "baseline_run_id": baseline_id,
                    "representation": protocol["enhanced_tfidf"],
                    "lsa": protocol["lsa"] if representation_id == "R2" else None,
                    "comparison_protocol_sha256": file_sha256(protocol_path),
                    "selection_evidence": protocol.get("selection_evidence", {}),
                }
            )
            configs.append(config)
            for field in (
                "k_values",
                "stability_seeds",
                "kmeans",
                "preprocessing",
                "tokenizer_run_id",
                "projection",
            ):
                if config[field] != original_config[field]:
                    raise DataValidationError(f"表示实验不得改变 baseline 的 {field}")
        dataset = {
            **baseline["dataset"],
            "token_cache_run_id": baseline_id,
            "token_cache_sha256": file_sha256(token_path),
            "comparison_protocol_sha256": file_sha256(protocol_path),
        }
        return self.run_representation_batch(
            documents,
            lambda text: (_ for _ in ()).throw(AssertionError("已验证 token 缓存禁止重新分词")),
            configs,
            gold_loader=gold_loader,
            dataset=dataset,
            tokenizer_info=baseline["representation"]["tokenizer"],
            tokenized=[row["tokens"] for row in token_rows],
            is_demo=False,
            progress=progress,
        )

    def _persist_execution(
        self,
        documents,
        config,
        execution,
        gold_records,
        gold_provenance,
        *,
        dataset,
        tokenizer_info,
        is_demo,
        elapsed_seconds,
        publish_latest=True,
    ):
        from importlib.metadata import version

        import numpy as np
        from scipy.sparse import save_npz

        from nlplab.evaluation.clustering import add_posthoc_composition, join_gold_posthoc

        for result in execution["results"].values():
            labels = join_gold_posthoc(documents, result["assignments"], gold_records)
            add_posthoc_composition(result, labels)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        directory = self.store.directory / run_id
        model_directory = self.settings.models / "exp2" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        model_directory.mkdir(parents=True, exist_ok=False)
        for name in ("results", "reference", "figures"):
            (directory / name).mkdir()
        execution["representation"].save(model_directory / "tfidf")
        if execution["lsa"] is None:
            save_npz(model_directory / "document_vectors.npz", execution["vectors"])
        else:
            execution["lsa"].save(model_directory / "lsa")
            np.save(
                model_directory / "document_vectors.npy", execution["vectors"], allow_pickle=False
            )
            save_npz(model_directory / "word_vectors.npz", execution["word_vectors"])
        np.save(
            model_directory / "svd_components.npy", execution["svd"].components_, allow_pickle=False
        )
        for k, model in execution["models"].items():
            model.save(model_directory / f"k{k}", space=execution["stats"]["space"])
        write_json(directory / "input_ids.json", [doc.id for doc in documents])
        write_jsonl(
            directory / "tokens.jsonl",
            [
                {"id": doc.id, "tokens": tokens}
                for doc, tokens in zip(documents, execution["tokenized"])
            ],
        )
        write_json(directory / "config.json", config)
        (directory / "config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )
        write_json(directory / "sweep.json", execution["sweep"])
        write_json(directory / "stability.json", execution["stability_records"])
        write_json(directory / "gold_provenance.json", gold_provenance)
        write_json(directory / "projection.json", execution["projection"])
        for k, result in execution["results"].items():
            write_json(directory / f"results/k{k}.json", result)
        reference_k = config["reference_k"]
        reference = {
            **execution["results"][reference_k],
            "role": "known-topic-reference",
            "reuses_sweep_fit": True,
            "selection": "known 5-topic sampling design; not best K",
        }
        write_json(directory / f"reference/k{reference_k}.json", reference)
        stats = {**execution["stats"], "tokenizer": tokenizer_info}
        overview = {
            "dataset": dataset,
            "config": config,
            "representation": stats,
            "sweep": execution["sweep"],
            "projection": execution["projection"],
        }
        write_json(directory / "overview.json", overview)
        environment = {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "PyYAML")
            },
            "threadpool_limit": 1,
        }
        write_json(directory / "environment.json", environment)
        source_files = [
            "clustering/experiment.py",
            "clustering/kmeans.py",
            "clustering/analysis.py",
            "features/text_tfidf.py",
            "features/lsa.py",
            "evaluation/clustering.py",
            "data/clustering_input.py",
            "services/clustering_service.py",
        ]
        write_json(
            directory / "source_fingerprints.json",
            {name: file_sha256(Path(__file__).parents[1] / name) for name in source_files},
        )
        # 清单包括模型和结果文件；run.json 最后写出，失败目录不进入 RunStore 列表。
        write_json(
            directory / "artifact_manifest.json",
            {
                "files": {
                    str(path.relative_to(self.settings.root)): file_sha256(path)
                    for base in (directory, model_directory)
                    for path in sorted(base.rglob("*"))
                    if path.is_file()
                },
            },
        )
        info = {
            "run_id": run_id,
            "dataset": dataset["name"],
            "is_demo": is_demo,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "word-tfidf-lsa-kmeans"
            if execution["lsa"] is not None
            else "word-tfidf-kmeans",
            "representation_id": config.get("representation_id", "R0"),
            "baseline_run_id": config.get("baseline_run_id"),
            "run_type": "exp2-kmeans-sweep",
            "document_count": len(documents),
            "reference_k": reference_k,
            "best_k": None,
            "fit_count": len(config["k_values"]) * len(config["stability_seeds"]),
            "primary_seed": config["kmeans"]["random_state"],
            "model_path": str(model_directory),
            "elapsed_seconds": elapsed_seconds,
            "formal_exp2_fit_executed": not is_demo,
            "toy_test_fit_executed": is_demo,
            "lab3_started": False,
            "label_join_phase": "after_all_unsupervised_fits",
            "reference_reuses_sweep_fit": True,
            "latest_pointer_updated": publish_latest,
        }
        write_json(directory / "run.json", info)
        if publish_latest:
            pointer = self.store.directory / (run_id + ".latest.tmp")
            write_json(pointer, {"run_id": run_id})
            pointer.replace(self.store.directory / "latest.json")
        return info

    def verify_artifacts(self, run_id="latest"):
        manifest = self.store.read(run_id, "artifact_manifest.json")
        for name, digest in manifest["files"].items():
            path = (self.settings.root / name).resolve()
            if not path.is_relative_to(self.settings.root.resolve()):
                raise DataValidationError("产物路径超出项目目录")
            if not path.is_file() or file_sha256(path) != digest:
                raise DataValidationError(f"实验二产物指纹不符: {name}")
        return {"status": "unchanged", "file_count": len(manifest["files"])}
