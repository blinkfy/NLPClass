"""显式启动实验二；Web/API 始终只读取已有 Run。"""

import argparse
import json
from pathlib import Path

from nlplab.core.config import default_settings
from nlplab.services.clustering_service import ClusteringService

parser = argparse.ArgumentParser(description="THUCNews600 TF-IDF K-means sweep")
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("--representation-comparison", type=Path, default=None)
args = parser.parse_args()
service = ClusteringService(default_settings())
if args.representation_comparison:
    info = service.run_representation_comparison(args.representation_comparison)
    from nlplab.clustering.comparison import save_comparison

    baseline_id = info[0]["baseline_run_id"]
    comparison = save_comparison(
        default_settings(), baseline_id, info, args.representation_comparison
    )
    print(f"comparison: {comparison}")
else:
    info = service.run_formal(args.config)
print(json.dumps(info, ensure_ascii=False, indent=2))
