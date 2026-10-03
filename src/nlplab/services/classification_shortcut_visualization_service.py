"""Single-document projections of saved Stage5 inputs and explanations only."""

from nlplab.core.exceptions import DataValidationError, ResourceUnavailableError
from nlplab.services.classification_stage5_service import ClassificationStage5Service
from nlplab.services.classification_workbench_service import ClassificationWorkbenchService


def retained_segments(text, spans):
    """Reconstruct boundaries from frozen offsets, never apply a cleaning rule."""
    intervals = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        a, b = span["start"], span["end"]
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text) or text[a:b] != span["text"]:
            raise DataValidationError("Stage5 保存的删除区间与文本不一致")
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(b, intervals[-1][1])
        else:
            intervals.append([a, b])
    segments, cursor = [], 0
    for a, b in intervals:
        if a > cursor:
            segments.append({"start": cursor, "end": a, "text": text[cursor:a]})
        cursor = b
    if cursor < len(text):
        segments.append({"start": cursor, "end": len(text), "text": text[cursor:]})
    return segments


class ClassificationShortcutVisualizationService(ClassificationWorkbenchService):
    def document(self, run_id, document_id, model_id):
        if model_id not in ("C0", "C1", "C2", "C3"):
            raise DataValidationError("未知模型")
        directory = ClassificationStage5Service(self.settings).resolve(run_id)
        flips = self._json(directory, "flips.json")
        if not any(row["document_id"] == document_id for row in flips):
            raise ResourceUnavailableError("仅支持冻结 Stage5 flip 文档")
        clean = next((row for row in self._json(directory, "validation_cleaning.json") if row["document_id"] == document_id), None)
        opened = self._json(directory, "validation_open_record.json")
        if clean is None or document_id not in opened["document_ids"]:
            raise DataValidationError("Stage5 Validation ID 不一致")
        text, spans = clean["text"], clean["removed_spans"]
        all_segments = retained_segments(text, spans)
        if all_segments != clean["segments"] or "\n".join(s["text"] for s in all_segments) != clean["char_text"]:
            raise DataValidationError("Stage5 保存的清理边界不一致")
        results = self._json(directory, "results.json")
        conditions = {}
        for condition in ("OO", "OS", "OT", "OA", "AA"):
            # Arrays contain complete saved channel explanations. Keep one row only.
            record = next((r for r in self._json(directory, f"{model_id}_{condition}_predictions.json") if r["document_id"] == document_id), None)
            if record is None or record["gold"] != clean["gold"]:
                raise DataValidationError("Stage5 保存预测/Gold 缺失或不一致")
            selected = [] if condition == "OO" else [s for s in spans if condition in ("OA", "AA") or s["kind"] == ("source" if condition == "OS" else "template")]
            segments = retained_segments(text, selected)
            reference = next((r.get("model_reference") for r in results if r["model_id"] == model_id and r["condition"] == condition), None)
            conditions[condition] = {"record": record, "removed_spans": selected, "segments": segments,
                                     "input_text": "\n".join(s["text"] for s in segments), "model_reference": reference,
                                     "model_identity": "clean-trained" if condition == "AA" else "original",
                                     "input_identity": "original" if condition == "OO" else "cleaned"}
        return {"document": {"document_id": document_id, "gold": clean["gold"], "text": text, "split": "Validation · Stage5", "source_run": directory.name},
                "model_id": model_id, "conditions": conditions, "all_removed_spans": spans,
                "original_model_run": self._json(directory, "run.json")["parent_stage4_run"],
                "coordinate_system": "saved NFKC/lowercase text; Python Unicode-codepoint offsets",
                "provenance": {"source_run_id": directory.name, "saved_explanations_only": True,
                               "cleaning_rules_executed": False, "inference_executed": False, "fit_executed": False}}
