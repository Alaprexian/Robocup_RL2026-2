"""Persecución e intercepción de balón (Ball Pursuit)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from discretizer import ACTION_NAMES, discretize

State = Tuple[int, int]


class BallPursuitSimEnv:
    """
    Proxy cinemático: observación polar (d_b, θ_b), macros dash/turn.
    Reset con d_b ∈ [5, 40] m (enunciado). Éxito: d_b < 0.8 m.
    """

    n_actions = 4
    action_names = ACTION_NAMES
    task_name = "pursuit"

    def __init__(self, max_steps: int = 120, seed: int | None = None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> State:
        self.player_x = float(self.rng.uniform(-20.0, -8.0))
        self.player_y = float(self.rng.uniform(-8.0, 8.0))
        self.player_theta = float(self.rng.uniform(-180.0, 180.0))

        # Ball at stochastic distance in [5, 40] m (enunciado); dash fuerte 1.2 m/paso
        dist = float(self.rng.uniform(5.0, 40.0))
        bearing = float(self.rng.uniform(-180.0, 180.0))
        rad = math.radians(bearing)
        self.ball_x = self.player_x + dist * math.cos(rad)
        self.ball_y = self.player_y + dist * math.sin(rad)
        self.ball_x = float(np.clip(self.ball_x, -45.0, 45.0))
        self.ball_y = float(np.clip(self.ball_y, -30.0, 30.0))

        self.steps = 0
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        self.success = False
        return self._get_state()

    def _get_obs(self) -> Tuple[float, float]:
        dx = self.ball_x - self.player_x
        dy = self.ball_y - self.player_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _get_state(self) -> State:
        return discretize(*self._get_obs())

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        self.steps += 1
        dist_prev, _ = self._get_obs()

        if action == 0:
            rad = math.radians(self.player_theta)
            self.player_x += 1.2 * math.cos(rad)
            self.player_y += 1.2 * math.sin(rad)
        elif action == 1:
            rad = math.radians(self.player_theta)
            self.player_x += 0.6 * math.cos(rad)
            self.player_y += 0.6 * math.sin(rad)
        elif action == 2:
            self.player_theta = (self.player_theta + 35.0 + 180.0) % 360.0 - 180.0
        elif action == 3:
            self.player_theta = (self.player_theta - 35.0 + 180.0) % 360.0 - 180.0

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)

        dist_curr, angle_curr = self._get_obs()
        done = False
        self.success = False

        if dist_curr < 0.8:
            reward = 100.0
            done = True
            self.success = True
        elif self.steps >= self.max_steps:
            reward = -20.0
            done = True
        else:
            reward = -1.0 + 5.0 * (dist_prev - dist_curr)

        return discretize(dist_curr, angle_curr), reward, done, {
            "dist": dist_curr,
            "angle": angle_curr,
            "success": self.success,
            "steps": self.steps,
        }
