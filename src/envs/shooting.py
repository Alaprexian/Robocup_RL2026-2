"""Definición y tiro a puerta / penales (Goal Shooting)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Multi-step: align then shoot (turns do not end the episode).
ACTION_NAMES = (
    "GIRAR_IZQ",
    "GIRAR_DER",
    "KICK_POSTE_IZQ",
    "KICK_ANGULO_IZQ",
    "KICK_CENTRO",
    "KICK_ANGULO_DER",
    "KICK_POSTE_DER",
)

# Kick actions 2..6 → (power, lateral aim at goal line, relative to goal center)
_KICK_AIM = {
    2: (65.0, -5.5),   # near left post
    3: (70.0, -3.0),   # left angle
    4: (75.0, 0.0),    # center
    5: (70.0, 3.0),    # right angle
    6: (65.0, 5.5),    # near right post
}

State = Tuple[int, int, int, int]  # dist, body_sector, gk_bin, facing_ok


def _bin_goal_dist(dg: float) -> int:
    if dg < 10.0:
        return 0
    if dg < 18.0:
        return 1
    if dg < 28.0:
        return 2
    return 3


def _bin_body_sector(angle_rel: float) -> int:
    if abs(angle_rel) <= 12.0:
        return 0  # facing goal
    if -45.0 <= angle_rel < -12.0:
        return 1
    if 12.0 < angle_rel <= 45.0:
        return 2
    return 3


def _bin_gk(gk_y: Optional[float], has_gk: bool) -> int:
    if not has_gk or gk_y is None:
        return 0  # open
    if gk_y < -1.5:
        return 1  # GK low (prefer shoot high/right)
    if gk_y > 1.5:
        return 2  # GK high (prefer shoot low/left)
    return 3  # centered


class GoalShootingSimEnv:
    """
    Tiro a arco rival con fase de alineación (giros) + disparo.
    Entrenamiento mixto 50% con portero. Éxito: gol.
    """

    n_actions = 7
    action_names = ACTION_NAMES
    task_name = "shooting"
    GOAL_X = 52.5
    POST_Y = 7.01

    def __init__(
        self,
        max_steps: int = 8,
        seed: int | None = None,
        with_gk: Optional[bool] = None,
        mix_gk_prob: float = 0.5,
    ):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.with_gk_fixed = with_gk
        self.mix_gk_prob = mix_gk_prob
        self.reset()

    def reset(self) -> State:
        if self.with_gk_fixed is None:
            self.has_gk = bool(self.rng.random() < self.mix_gk_prob)
        else:
            self.has_gk = bool(self.with_gk_fixed)

        self.player_x = float(self.rng.uniform(32.0, 44.0))
        self.player_y = float(self.rng.uniform(-10.0, 10.0))
        self.player_theta = float(self.rng.uniform(-50.0, 50.0))
        self.ball_x = self.player_x + 0.45
        self.ball_y = self.player_y
        self.gk_y = float(self.rng.uniform(-3.5, 3.5)) if self.has_gk else None
        self.gk_x = 50.5
        self.steps = 0
        self.shot_taken = False
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        self.success = False
        self.last_outcome = "none"
        return self._get_state()

    def _goal_bearing(self) -> Tuple[float, float]:
        dx = self.GOAL_X - self.ball_x
        dy = 0.0 - self.ball_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _get_state(self) -> State:
        dg, ag = self._goal_bearing()
        facing_ok = 1 if abs(ag) <= 20.0 else 0
        return (_bin_goal_dist(dg), _bin_body_sector(ag), _bin_gk(self.gk_y, self.has_gk), facing_ok)

    def _simulate_shot(self, power: float, aim_y: float) -> str:
        """
        Disparo dirigido a una altura aim_y en la línea de gol.
        Si el cuerpo no mira al arco, el tiro se degrada.
        """
        _, body_err = self._goal_bearing()
        # Misalignment adds bias + noise
        bias = 0.04 * body_err
        noise_std = 0.35 + 0.04 * abs(body_err)
        if abs(body_err) > 35.0:
            noise_std += 1.2
        impact_y = aim_y + bias + float(self.rng.normal(0.0, noise_std))
        # Slight power effect: too strong center kicks wobble more near posts
        if abs(aim_y) < 0.5 and power > 85:
            impact_y += float(self.rng.normal(0.0, 0.4))

        self.ball_x = self.GOAL_X
        self.ball_y = float(np.clip(impact_y, -18.0, 18.0))

        if abs(impact_y) > self.POST_Y:
            return "miss"

        if self.has_gk and self.gk_y is not None:
            # Reach shrinks for shots far from GK (opposite corner)
            base_reach = 1.35
            dist_to_gk = abs(impact_y - self.gk_y)
            # Dive bonus if shot is central-ish
            reach = base_reach + (0.35 if abs(aim_y) < 2.0 else 0.0)
            if dist_to_gk < reach:
                return "saved"
        return "goal"

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        self.steps += 1
        action = int(np.clip(action, 0, self.n_actions - 1))
        done = False
        self.success = False
        reward = -0.5

        if action in (0, 1):
            delta = 35.0 if action == 0 else -35.0
            self.player_theta = (self.player_theta + delta + 180.0) % 360.0 - 180.0
            # Shaping: reward facing the goal
            _, ag = self._goal_bearing()
            reward = 2.0 if abs(ag) <= 15.0 else (-0.3 if abs(ag) < abs(ag + delta) else 0.8)
            if self.steps >= self.max_steps:
                # Forced weak miss if never shot
                reward = -25.0
                done = True
                self.last_outcome = "timeout"
        else:
            power, aim_y = _KICK_AIM[action]
            outcome = self._simulate_shot(power, aim_y)
            self.shot_taken = True
            self.last_outcome = outcome
            done = True
            self.success = outcome == "goal"
            if outcome == "goal":
                # Extra bonus for beating GK with an angled shot
                bonus = 10.0 if self.has_gk and abs(aim_y) >= 3.0 else 0.0
                reward = 100.0 + bonus
            elif outcome == "saved":
                reward = -40.0
            else:
                reward = -25.0

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)

        return self._get_state(), float(reward), done, {
            "success": self.success,
            "outcome": self.last_outcome,
            "has_gk": self.has_gk,
            "steps": self.steps,
            "shot_taken": self.shot_taken,
        }
