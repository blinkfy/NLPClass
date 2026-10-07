"""Create an independent teaching diagnostic; never opens an old Test split."""

import json
from datetime import datetime, timezone
from pathlib import Path

from nlplab.core.config import Settings
from nlplab.course_neural.resources import digest
from nlplab.services.course_neural_service import CourseNeuralService
from nlplab.services.nlp_service import NLPService


def main():
    root = Path(__file__).resolve().parents[1]
    settings = Settings(root=root)
    service = CourseNeuralService(settings, NLPService(settings))
    result = service.dependency_evaluation(use_saved=False)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S") + "_teaching"
    folder = root / "outputs/course_dependency" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    gold = root / "src/nlplab/course_neural/teaching_gold.json"
    result.update(run_id=run_id, created_at=now.isoformat(), evaluation_scope="manual teaching Gold, not CTB benchmark",
                  gold_reference={"path": str(gold.relative_to(root)), "sha256": digest(gold)})
    path = folder / "evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    sources = ["src/nlplab/course_neural/dependency.py", "src/nlplab/course_neural/worker.py",
               "src/nlplab/course_neural/resources.py", "src/nlplab/services/course_neural_service.py",
               "docs/course_modules_07_08_plan.md", "scripts/run_course_dependency_evaluation.py"]
    manifest = {"schema_version": 1, "run_id": run_id, "files": {"evaluation.json": digest(path)},
                "source_hashes": {name: digest(root/name) for name in sources}, "training": False,
                "old_test_access": False, "source_model": result["source"]}
    (folder / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    reference = {"run_id": run_id, "sha256": digest(path)}
    latest = folder.parent / "evaluation_latest.json"
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(json.dumps(reference,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(latest)
    print(json.dumps({"run_id":run_id,"aggregate":result["aggregate"],"path":str(folder)},ensure_ascii=False))


if __name__ == "__main__":
    main()
