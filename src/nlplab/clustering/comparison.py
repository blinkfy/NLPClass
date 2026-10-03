"""只读比较三种已完成表示；不排序选优、不 fit、不生成综合分数。"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from nlplab.core.exceptions import DataValidationError
from nlplab.core.io import write_json
from nlplab.data.course import file_sha256
from nlplab.services.clustering_service import ClusteringService


def save_comparison(settings, baseline_run_id, new_runs, protocol_path: Path):
    service = ClusteringService(settings)
    run_ids = {
        "R0": baseline_run_id,
        **{info["representation_id"]: info["run_id"] for info in new_runs},
    }
    representations = {}
    common_input = None
    for name, run_id in run_ids.items():
        service.verify_artifacts(run_id)
        overview = service.overview(run_id)
        ids = service.store.read(run_id, "input_ids.json")
        input_key = (overview["dataset"]["input_sha256"], ids)
        if common_input is None:
            common_input = input_key
        elif common_input != input_key:
            raise DataValidationError("表示对照数据指纹或文档顺序不一致")
        reference = service.result(run_id, 5)
        stats = overview["representation"]
        primary = next(row for row in overview["sweep"] if row["k"] == 5)
        representations[name] = {
            "run_id": run_id,
            "config": overview["config"],
            "dataset": overview["dataset"],
            "representation": stats,
            "k5": {
                "internal": reference["internal"],
                "external": reference["external"],
                "stability": primary["stability"],
                "cluster_sizes": primary["cluster_sizes"],
                "imbalance_ratio": primary["imbalance_ratio"],
                "clusters": reference["clusters"],
                "clustering_space": reference.get("clustering_space", "tfidf_l2"),
            },
            "sweep": overview["sweep"],
            "environment": service.store.read(run_id, "environment.json"),
        }
    comparison_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    directory = settings.outputs / "exp2/comparisons" / comparison_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.yaml").write_bytes(protocol_path.read_bytes())
    write_json(
        directory / "comparison.json",
        {
            "comparison_id": comparison_id,
            "baseline_run_id": baseline_run_id,
            "baseline_refitted": False,
            "winner": None,
            "representations": representations,
            "comparison_notes": {
                "inertia": "SSE is space-specific; cross-space values are not a winner criterion",
                "internal": "LSA geometry changes; consider stability, balance and interpretation",
                "external": "post-hoc evidence only; no feature/K/parameter selection",
                "top_terms": "R0/R1 TF-IDF centers; R2 member mean in original R1 word space",
                "representatives": "Actual-space Euclidean distances; not cross-space comparable",
                "latest_pointer": "Exp2 latest stays baseline; no comparison pointer introduced",
            },
        },
    )
    write_json(
        directory / "manifest.json",
        {
            "files": {
                path.name: file_sha256(path)
                for path in sorted(directory.glob("*"))
                if path.is_file()
            },
            "input_run_ids": run_ids,
        },
    )
    return directory
