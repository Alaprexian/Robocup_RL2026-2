"""Entorno cinemático de persecución de balón para RL tabular (proxy de RoboCup 2D)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from discretizer import discretize

State = Tuple[int, int]


class BallPursuitSimEnv:
    """
    Simulador cinemático reproducible para entrenamiento tabular rápido.

    Observación polar: (d_b, θ_b). Acciones macro: dash fuerte/suave, turn ±35°.
    Éxito: d_b < 0.8 m (kickable area).
    """

    def __init__(self, max_steps: int = 80, seed: int | None = None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> State:
        self.player_x = -15.0 + float(self.rng.uniform(-2.0, 2.0))
        self.player_y = float(self.rng.uniform(-4.0, 4.0))
        self.player_theta = float(self.rng.uniform(-180.0, 180.0))
        self.ball_x = float(self.rng.uniform(-2.0, 2.0))
        self.ball_y = float(self.rng.uniform(-3.0, 3.0))
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
        dist, angle_rel = self._get_obs()
        return discretize(dist, angle_rel)

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        self.steps += 1
        dist_prev, _ = self._get_obs()

        if action == 0:
            rad = math.radians(self.player_theta)
            self.player_x += 1.0 * math.cos(rad)
            self.player_y += 1.0 * math.sin(rad)
        elif action == 1:
            rad = math.radians(self.player_theta)
            self.player_x += 0.5 * math.cos(rad)
            self.player_y += 0.5 * math.sin(rad)
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
        }
