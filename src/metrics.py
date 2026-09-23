"""Métricas de evaluación para experimentos P1."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from discretizer import ANGLE_LABELS, DIST_LABELS, N_ACTIONS, N_ANGLE_BINS, N_DIST_BINS


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def success_rate(success_flags: Sequence[int], window: int = 50) -> np.ndarray:
    """Tasa de éxito temporal en ventanas móviles."""
    return moving_average(success_flags, window=window)


def summarize_run(result: Dict, last_n: int = 100) -> Dict[str, float]:
    rewards = result["rewards"]
    lengths = result["lengths"]
    success = result["success"]
    n = min(last_n, len(rewards))
    return {
        "mean_return_last": float(np.mean(rewards[-n:])),
        "mean_steps_last": float(np.mean(lengths[-n:])),
        "success_rate_last": float(np.mean(success[-n:])),
        "success_rate_all": float(np.mean(success)),
    }


def q_to_value_policy(Q: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Matrices V*(d,a) y π*(d,a) sobre la grilla de discretización."""
    V = np.full((N_DIST_BINS, N_ANGLE_BINS), np.nan)
    Pi = np.full((N_DIST_BINS, N_ANGLE_BINS), -1, dtype=int)
    for (d_bin, a_bin), q_vals in Q.items():
        V[d_bin, a_bin] = float(np.max(q_vals))
        Pi[d_bin, a_bin] = int(np.argmax(q_vals))
    return V, Pi


def format_policy_grid(Pi: np.ndarray, action_names: Sequence[str] | None = None) -> List[List[str]]:
    names = action_names or [f"A{i}" for i in range(N_ACTIONS)]
    grid = []
    for d in range(N_DIST_BINS):
        row = []
        for a in range(N_ANGLE_BINS):
            idx = int(Pi[d, a])
            row.append(names[idx] if idx >= 0 else "—")
        grid.append(row)
    return grid


def label_axes():
    return DIST_LABELS, ANGLE_LABELS
