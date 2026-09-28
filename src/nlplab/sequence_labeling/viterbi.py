"""层状有向图上的动态规划；教学实现保留 score 与 backpointer。"""

from dataclasses import dataclass

import numpy as np

from nlplab.core.exceptions import DataValidationError, NoValidPathError
from nlplab.sequence_labeling.tags import LEGAL_END, LEGAL_NEXT, LEGAL_START, STATES


@dataclass
class ViterbiTrace:
    tags: list[str]
    score: np.ndarray
    backpointer: np.ndarray
    best_log_probability: float


def viterbi(
    log_start: np.ndarray, log_transition: np.ndarray, log_emissions: np.ndarray
) -> ViterbiTrace:
    """log_emissions[t,s] 为时刻 t 的观测在状态 s 下的 log probability。"""
    if log_start.shape != (4,) or log_transition.shape != (4, 4):
        raise DataValidationError("BMES 初始/转移矩阵形状错误")
    if log_emissions.ndim != 2 or log_emissions.shape[1] != 4:
        raise DataValidationError("发射观测矩阵应为 T×4")
    if any(
        np.isnan(array).any() or np.isposinf(array).any()
        for array in (log_start, log_transition, log_emissions)
    ):
        raise DataValidationError("log probability 不允许 NaN 或 +inf")
    length = log_emissions.shape[0]
    score = np.full((length, 4), -np.inf)
    backpointer = np.full((length, 4), -1, dtype=int)
    if length == 0:
        return ViterbiTrace([], score, backpointer, 0.0)

    # 1. 初始化：首状态只允许 B/S。
    for state_index, state in enumerate(STATES):
        if state in LEGAL_START:
            score[0, state_index] = log_start[state_index] + log_emissions[0, state_index]

    # 2. 递推：每一层维护以各状态结束的最佳局部路径及其前驱。
    for time in range(1, length):
        for current_index, current_state in enumerate(STATES):
            best_score = -np.inf
            best_previous = -1
            for previous_index, previous_state in enumerate(STATES):
                if current_state not in LEGAL_NEXT[previous_state]:
                    continue
                candidate = (
                    score[time - 1, previous_index]
                    + log_transition[previous_index, current_index]
                    + log_emissions[time, current_index]
                )
                if candidate > best_score:
                    best_score = candidate
                    best_previous = previous_index
            score[time, current_index] = best_score
            backpointer[time, current_index] = best_previous

    # 3. 终止：最后一层只能从 E/S 中选择全局最优终点。
    final_states = [index for index, state in enumerate(STATES) if state in LEGAL_END]
    final_state = max(final_states, key=lambda index: score[-1, index])
    best_log_probability = float(score[-1, final_state])
    if not np.isfinite(best_log_probability):
        raise NoValidPathError("HMM 无有效 BMES 路径；检查语料覆盖或使用 add_alpha 平滑")

    # 4. 回溯：从终点沿前驱指针反向恢复最佳状态序列。
    indices = [final_state]
    for time in range(length - 1, 0, -1):
        indices.append(int(backpointer[time, indices[-1]]))
    indices.reverse()
    return ViterbiTrace(
        [STATES[index] for index in indices], score, backpointer, best_log_probability
    )
