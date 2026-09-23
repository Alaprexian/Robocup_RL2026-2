"""Políticas ε-greedy y esquemas de exploración."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from discretizer import N_ACTIONS

State = Tuple[int, int]
QTable = Dict[State, np.ndarray]


def epsilon_schedule(
    mode: str,
    episode: int,
    *,
    eps_fixed: float = 0.1,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.998,
) -> float:
    """
    mode='fixed'  → ε constante
    mode='decay'  → ε_k = max(ε_min, ε_0 · λ^k)
    """
    if mode == "fixed":
        return float(eps_fixed)
    if mode == "decay":
        return float(max(eps_min, eps_start * (eps_decay ** episode)))
    raise ValueError(f"modo de exploración desconocido: {mode}")


def select_action(Q: QTable, state: State, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    q_vals = Q[state]
    if np.allclose(q_vals, q_vals[0]):
        return int(rng.integers(0, N_ACTIONS))
    return int(np.argmax(q_vals))


def make_empty_q(n_actions: int = N_ACTIONS) -> Callable[[], np.ndarray]:
    return lambda: np.zeros(n_actions, dtype=np.float64)
