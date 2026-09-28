"""只读 trace 展示字段：派生说明，不执行或重新实现算法。"""

import math

from nlplab.sequence_labeling.tags import LEGAL_NEXT, STATES

KINDS = {
    "longest": "longest_scan",
    "bigram": "bigram_lattice",
    "hmm1-smooth": "hmm1_viterbi",
    "hmm2-smooth": "hmm2_pair_viterbi",
    "bilstm-crf": "bilstm_crf_emission",
}


def display_trace(trace: dict, text: str, model: str, dictionary: set | None = None) -> dict:
    """保留原始 score/backpointer/token；只补充类型、词网标记和展示计数。"""
    value = {**trace, "trace_type": KINDS[model], "text": text, "character_count": len(text)}
    value["bmes_source"] = (
        "derived from tokens" if model in {"longest", "bigram"} else "model decoded path"
    )
    value["pair_mask"] = [[b in LEGAL_NEXT[a] for b in STATES] for a in STATES]
    if model == "hmm2-smooth" and "chunks" in trace:
        # Frozen HMM2 nests each existing trace; only flatten its display envelope.
        value["chunks"] = [
            {**chunk["trace"], "offset": chunk["offset"], "text": chunk["text"]}
            for chunk in trace["chunks"]
        ]
    if model == "longest":
        dictionary = dictionary or set()
        steps = []
        for step in trace["steps"]:
            candidates = [
                {
                    "word": word,
                    "kind": "whitespace"
                    if word.isspace()
                    else "dictionary match"
                    if word in dictionary
                    else "single-char fallback",
                    "selected": word == step["chosen"],
                }
                for word in step["candidates"]
            ]
            steps.append(
                {
                    **step,
                    "candidate_info": candidates,
                    "next_offset": step["offset"] + len(step["chosen"]),
                }
            )
        value["steps"] = steps
        value["fallback_count"] = sum(
            any(c["selected"] and c["kind"] == "single-char fallback" for c in s["candidate_info"])
            for s in steps
        )
    if model == "bigram":
        previous, offset, chain = "<BOS>", 0, []
        for word in trace["tokens"]:
            end = offset + len(word)
            chain.append({"start": offset, "end": end, "previous": previous, "word": word})
            previous, offset = word, end
        chosen = {(item["start"], item["end"], item["previous"], item["word"]) for item in chain}
        edges = []
        for edge in trace["edges"]:
            state = trace["states"][edge["to"][0]][edge["to"][1]]
            best = state["predecessor"] == edge["from"] and math.isclose(
                state["cost"], edge["cumulative_cost"], abs_tol=1e-10
            )
            edges.append(
                {
                    **edge,
                    "probability": math.exp(-edge["negative_log_cost"]),
                    "best_to_state": best,
                    "on_best_path": (edge["from"][0], edge["to"][0], edge["from"][1], edge["to"][1])
                    in chosen,
                }
            )
        value["edges"] = edges
        value["optimal_chain"] = chain
        value["solver"] = "词图上的 Viterbi / 最短路径动态规划"
    return value
