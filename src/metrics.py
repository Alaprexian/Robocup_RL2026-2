"""Métricas de evaluación para experimentos P1."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
    out: Dict[str, float] = {
        "mean_return_last": float(np.mean(rewards[-n:])),
        "mean_steps_last": float(np.mean(lengths[-n:])),
        "success_rate_last": float(np.mean(success[-n:])),
        "success_rate_all": float(np.mean(success)),
    }
    extras = result.get("extra") or []
    if extras:
        last = extras[-n:]
        if any("advance" in e for e in last):
            out["mean_advance_last"] = float(np.mean([e.get("advance", 0.0) for e in last]))
        if any("completed_passes" in e for e in last):
            out["mean_passes_last"] = float(np.mean([e.get("completed_passes", 0) for e in last]))
            out["mean_possession_last"] = float(np.mean([e.get("possession_steps", 0) for e in last]))
        if any("has_gk" in e for e in last):
            open_eps = [1 if e.get("success") else 0 for e in last if not e.get("has_gk")]
            gk_eps = [1 if e.get("success") else 0 for e in last if e.get("has_gk")]
            if open_eps:
                out["success_open_goal_last"] = float(np.mean(open_eps))
            if gk_eps:
                out["success_with_gk_last"] = float(np.mean(gk_eps))
    return out


def evaluate_shooting_split(Q: Dict, seed: int = 99, n_episodes: int = 200) -> Dict[str, float]:
    """Eval greedy open-goal vs with-GK for shooting task."""
    from envs.shooting import GoalShootingSimEnv

    def _run(with_gk: bool) -> float:
        env = GoalShootingSimEnv(seed=seed, with_gk=with_gk)
        wins = 0
        for i in range(n_episodes):
            state = env.reset()
            # Single-shot episodes; take greedy action once (or until done)
            for _ in range(env.max_steps):
                q = Q.get(state)
                action = int(np.argmax(q)) if q is not None else 0
                state, _, done, info = env.step(action)
                if done:
                    wins += int(bool(info.get("success")))
                    break
        return wins / n_episodes

    return {
        "success_open_goal": _run(False),
        "success_with_gk": _run(True),
    }


def q_to_value_policy(Q: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Matrices V*(d,a) y π*(d,a) sobre la grilla polar 4×5 (pursuit)."""
    V = np.full((N_DIST_BINS, N_ANGLE_BINS), np.nan)
    Pi = np.full((N_DIST_BINS, N_ANGLE_BINS), -1, dtype=int)
    for state, q_vals in Q.items():
        if not (isinstance(state, tuple) and len(state) == 2):
            continue
        d_bin, a_bin = state
        if 0 <= d_bin < N_DIST_BINS and 0 <= a_bin < N_ANGLE_BINS:
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
            row.append(names[idx] if 0 <= idx < len(names) else "—")
        grid.append(row)
    return grid


def label_axes():
    return DIST_LABELS, ANGLE_LABELS
