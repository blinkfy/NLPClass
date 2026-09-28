"""只读课程算法过程；复用冻结 HMM 的 Viterbi，不改模型。"""

import re

import numpy as np

from nlplab.core.exceptions import DataValidationError
from nlplab.sequence_labeling.tags import STATES
from nlplab.sequence_labeling.viterbi import viterbi


def algorithm_trace(service, text: str, model_name: str, run_id: str = "latest") -> dict:
    if not text or len(text) > 30:
        raise DataValidationError("算法过程只支持 1–30 Unicode code point")
    if model_name != "hmm1-smooth":
        return service._load_extension_segmenter(model_name).trace(text)
    result = service.analyze(text, ["segmentation"], run_id, model_name)
    segmenter = service.segmenter_loader(result["run"]["run_id"])
    tagger = segmenter.tagger
    chunks, offset = [], 0
    for chunk in re.split(r"(\s)", text):
        if not chunk:
            continue
        if chunk.isspace():
            chunks.append({"offset": offset, "text": chunk, "tags": ["S"], "whitespace": True})
        else:
            indices = [tagger.character_index.get(c, 0) for c in chunk]
            trace = viterbi(tagger.log_pi, tagger.log_A, tagger.log_B[:, indices].T)
            chunks.append(
                {
                    "offset": offset,
                    "text": chunk,
                    "tags": trace.tags,
                    "score": [
                        [float(x) if np.isfinite(x) else None for x in row] for row in trace.score
                    ],
                    "backpointer": trace.backpointer.tolist(),
                    "best_log_probability": trace.best_log_probability,
                }
            )
        offset += len(chunk)
    return {
        "model": model_name,
        "states": list(STATES),
        "chunks": chunks,
        "tokens": result["segmentation"]["tokens"],
        "tags": result["segmentation"]["tags"],
    }
