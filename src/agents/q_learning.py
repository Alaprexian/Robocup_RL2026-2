"""Q-Learning tabular off-policy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, List

import numpy as np

from agents.policies import epsilon_schedule, get_n_actions, make_empty_q, select_action
from envs.pursuit import BallPursuitSimEnv

QTable = Dict[Hashable, np.ndarray]


def train_q_learning(
    env: Any | None = None,
    n_episodes: int = 3500,
    gamma: float = 0.99,
    alpha: float = 0.1,
    exploration: str = "decay",
    eps_fixed: float = 0.1,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.998,
    seed: int = 42,
) -> Dict:
    """
    Actualización: Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
    (terminal: target = r).
    """
    rng = np.random.default_rng(seed)
    env = env or BallPursuitSimEnv(seed=seed)
    n_actions = get_n_actions(env)
    Q: QTable = defaultdict(make_empty_q(n_actions))

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
        ep_return = 0.0
        steps = 0
        last_info: Dict = {}

        while True:
            action = select_action(Q, state, eps, rng, n_actions=n_actions)
            next_state, reward, done, info = env.step(action)
            ep_return += reward
            steps += 1
            last_info = info

            if done:
                target = reward
            else:
                target = reward + gamma * float(np.max(Q[next_state]))

            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state

            if done:
                history_success.append(1 if info.get("success") else 0)
                history_extra.append(dict(info))
                break

        history_rewards.append(float(ep_return))
        history_lengths.append(steps)

    return {
        "algorithm": "q_learning",
        "exploration": exploration,
        "task": getattr(env, "task_name", "unknown"),
        "Q": dict(Q),
        "rewards": history_rewards,
        "lengths": history_lengths,
        "success": history_success,
        "eps": history_eps,
        "extra": history_extra,
        "alpha": alpha,
        "n_actions": n_actions,
    }
