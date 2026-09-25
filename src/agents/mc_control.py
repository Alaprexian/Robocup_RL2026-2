"""Monte Carlo Control On-Policy (First-Visit)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, List

import numpy as np

from agents.policies import epsilon_schedule, get_n_actions, make_empty_q, select_action
from envs.pursuit import BallPursuitSimEnv

QTable = Dict[Hashable, np.ndarray]


def train_mc_control(
    env: Any | None = None,
    n_episodes: int = 3500,
    gamma: float = 0.99,
    exploration: str = "decay",
    eps_fixed: float = 0.1,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.998,
    seed: int = 42,
) -> Dict:
    """Entrena First-Visit MC Control y retorna Q + historiales."""
    rng = np.random.default_rng(seed)
    env = env or BallPursuitSimEnv(seed=seed)
    n_actions = get_n_actions(env)
    Q: QTable = defaultdict(make_empty_q(n_actions))
    returns_sum = defaultdict(make_empty_q(n_actions))
    returns_count: Dict[Hashable, np.ndarray] = defaultdict(
        lambda: np.zeros(n_actions, dtype=np.int32)
    )

    history_rewards: List[float] = []
    history_lengths: List[int] = []
    history_success: List[int] = []
    history_eps: List[float] = []
    history_extra: List[Dict] = []

    for ep in range(n_episodes):
        eps = epsilon_schedule(
            exploration,
            ep,
            eps_fixed=eps_fixed,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_decay=eps_decay,
        )
        history_eps.append(eps)
        state = env.reset()
        episode = []

        while True:
            action = select_action(Q, state, eps, rng, n_actions=n_actions)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                history_success.append(1 if info.get("success") else 0)
                history_extra.append(dict(info))
                break

        G = 0.0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]

        history_rewards.append(float(sum(r for _, _, r in episode)))
        history_lengths.append(len(episode))

    return {
        "algorithm": "mc_control",
        "exploration": exploration,
        "task": getattr(env, "task_name", "unknown"),
        "Q": dict(Q),
        "rewards": history_rewards,
        "lengths": history_lengths,
        "success": history_success,
        "eps": history_eps,
        "extra": history_extra,
        "n_actions": n_actions,
    }
