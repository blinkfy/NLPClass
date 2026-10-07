"""Numerical reconstruction for visualization, not a replacement CRF model."""

import math


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v-maximum) for v in values))


def crf_trace(emissions, transitions, start, end):
    """Scores use transitions[from][to]. Include actual start/end parameters."""
    if not emissions or any(len(row) != len(start) for row in emissions):
        raise ValueError("emission tensor shape mismatch")
    n = len(start)
    delta = [[start[j] + emissions[0][j] for j in range(n)]]
    alpha = [delta[0][:]]
    pointers = [[None] * n]
    for emission in emissions[1:]:
        candidates = [[delta[-1][i] + transitions[i][j] for i in range(n)] for j in range(n)]
        previous = [max(range(n), key=lambda i: candidates[j][i]) for j in range(n)]
        delta.append([candidates[j][previous[j]] + emission[j] for j in range(n)])
        pointers.append(previous)
        alpha.append([logsumexp([alpha[-1][i] + transitions[i][j] for i in range(n)]) + emission[j] for j in range(n)])
    last = max(range(n), key=lambda j: delta[-1][j] + end[j])
    path = [last]
    for pointer in reversed(pointers[1:]):
        path.append(pointer[path[-1]])
    path.reverse()
    return {"delta": delta, "alpha": alpha, "backpointers": pointers, "path": path,
            "path_score": delta[-1][last] + end[last],
            "log_partition": logsumexp([alpha[-1][j] + end[j] for j in range(n)])}
