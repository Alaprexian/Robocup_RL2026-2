"""Conducción y drible de balón (Ball Dribbling)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from discretizer import discretize

State = Tuple[int, int, int, int]

ACTION_NAMES = ("KICK_25", "DASH_80", "GIRAR_IZQ", "GIRAR_DER")


def _bin_goal_dist(dg: float) -> int:
    if dg < 10.0:
        return 0
    if dg < 25.0:
        return 1
    if dg < 40.0:
        return 2
    return 3


def _bin_goal_angle(ag: float) -> int:
    if abs(ag) <= 20.0:
        return 0
    if -90.0 <= ag < -20.0:
        return 1
    if 20.0 < ag <= 90.0:
        return 2
    return 3


class BallDribblingSimEnv:
    """
    Agente con balón avanza hacia meta rival (x=+52.5) con micro-pateos y dash.
    Éxito: avance neto > 30 m manteniendo posesión (d_b < 1.2 m) al terminar.
    """

    n_actions = 4
    action_names = ACTION_NAMES
    task_name = "dribbling"
    GOAL_X = 52.5
    GOAL_Y = 0.0
    POSSESSION_RADIUS = 2.0
    SUCCESS_ADVANCE = 30.0

    def __init__(self, max_steps: int = 120, seed: int | None = None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> State:
        self.player_x = float(self.rng.uniform(-10.0, 5.0))
        self.player_y = float(self.rng.uniform(-8.0, 8.0))
        self.player_theta = float(self.rng.uniform(-30.0, 30.0))
        # Ball near feet
        self.ball_x = self.player_x + float(self.rng.uniform(0.3, 0.7))
        self.ball_y = self.player_y + float(self.rng.uniform(-0.3, 0.3))
        self.start_x = self.player_x
        self.steps = 0
        self.possession_steps = 0
        self.lost_streak = 0
        self.lost_ball = False
        self.out_of_bounds = False
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        self.success = False
        return self._get_state()

    def _ball_obs(self) -> Tuple[float, float]:
        dx = self.ball_x - self.player_x
        dy = self.ball_y - self.player_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _goal_obs(self) -> Tuple[float, float]:
        dx = self.GOAL_X - self.player_x
        dy = self.GOAL_Y - self.player_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _get_state(self) -> State:
        db, tb = self._ball_obs()
        dg, tg = self._goal_obs()
        d_bin, a_bin = discretize(db, tb)
        return (d_bin, a_bin, _bin_goal_dist(dg), _bin_goal_angle(tg))

    def _in_possession(self) -> bool:
        return self._ball_obs()[0] < self.POSSESSION_RADIUS

    def _on_pitch(self) -> bool:
        return -52.5 <= self.player_x <= 52.5 and -34.0 <= self.player_y <= 34.0

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        self.steps += 1
        x_prev = self.player_x
        dg_prev, _ = self._goal_obs()

        if action == 0:  # micro-kick forward (short)
            if self._in_possession():
                rad = math.radians(self.player_theta)
                self.ball_x += 1.2 * math.cos(rad)
                self.ball_y += 1.2 * math.sin(rad)
        elif action == 1:  # dash — keep ball close if possessed
            rad = math.radians(self.player_theta)
            self.player_x += 0.9 * math.cos(rad)
            self.player_y += 0.9 * math.sin(rad)
            if self._ball_obs()[0] < self.POSSESSION_RADIUS + 0.5:
                self.ball_x = self.player_x + 0.55 * math.cos(rad)
                self.ball_y = self.player_y + 0.55 * math.sin(rad)
        elif action == 2:
            self.player_theta = (self.player_theta + 35.0 + 180.0) % 360.0 - 180.0
        elif action == 3:
            self.player_theta = (self.player_theta - 35.0 + 180.0) % 360.0 - 180.0

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)

        if self._in_possession():
            self.possession_steps += 1
            self.lost_streak = 0
        else:
            self.lost_streak += 1
            if self.lost_streak >= 3:
                self.lost_ball = True

        if not self._on_pitch():
            self.out_of_bounds = True

        dg_curr, _ = self._goal_obs()
        advance = self.player_x - self.start_x
        done = False
        self.success = False

        if self.out_of_bounds or self.lost_ball:
            reward = -30.0
            done = True
        elif advance >= self.SUCCESS_ADVANCE and self._in_possession():
            reward = 100.0
            done = True
            self.success = True
        elif self.steps >= self.max_steps:
            reward = -20.0
            done = True
            if advance >= self.SUCCESS_ADVANCE and (self.possession_steps / max(1, self.steps)) >= 0.8:
                self.success = True
                reward = 80.0
        else:
            reward = -0.2 + 2.0 * (dg_prev - dg_curr) + 0.5 * (self.player_x - x_prev)

        return self._get_state(), float(reward), done, {
            "success": self.success,
            "advance": advance,
            "possession_ratio": self.possession_steps / max(1, self.steps),
            "dist_ball": self._ball_obs()[0],
            "steps": self.steps,
        }
