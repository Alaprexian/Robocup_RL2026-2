"""Cooperación 2 vs 1 y pase (Passing - Possession)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from discretizer import discretize

State = Tuple[int, int, int, int, int]

ACTION_NAMES = ("PASE", "DRIBLE", "GIRAR", "ALEJARSE")


class PassingPossessionSimEnv:
    """
    Atacante + compañero vs defensor estocástico.
    Éxito: posesión ≥ 50 pasos y ≥ 3 pases completados.
    Recompensa alineada al criterio (posesión densa, pases moderados).

    Libertad del defensor (diseño del proxy, no del enunciado):
    - ``defender_stun_on_pass``: pasos inmóvil tras pase limpio (0 = sin stun).
    - ``defender_speed_base`` / ``defender_speed_jitter``: velocidad de persecución.
    Ablación en ``scripts/ablate_passing_defender.py``.
    """

    n_actions = 4
    action_names = ACTION_NAMES
    task_name = "passing"
    PASS_RANGE = 20.0
    INTERCEPT_RADIUS = 1.35

    def __init__(
        self,
        max_steps: int = 100,
        seed: int | None = None,
        defender_mode: str = "stochastic",
        defender_stun_on_pass: int = 0,
        defender_speed_base: float = 0.18,
        defender_speed_jitter: float = 0.10,
    ):
        assert defender_mode in ("stochastic", "passive")
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.defender_mode = defender_mode
        self.defender_stun_on_pass = int(max(0, defender_stun_on_pass))
        self.defender_speed_base = float(defender_speed_base)
        self.defender_speed_jitter = float(defender_speed_jitter)
        self.reset()

    def reset(self) -> State:
        self.player_x = float(self.rng.uniform(-10.0, -2.0))
        self.player_y = float(self.rng.uniform(-7.0, 7.0))
        self.player_theta = float(self.rng.uniform(-30.0, 30.0))
        self.mate_x = self.player_x + float(self.rng.uniform(7.0, 14.0))
        self.mate_y = self.player_y + float(self.rng.uniform(-10.0, 10.0))
        # Defender starts farther away
        self.def_x = float(self.rng.uniform(18.0, 28.0))
        self.def_y = float(self.rng.uniform(-10.0, 10.0))
        self.ball_x = self.player_x + 0.4
        self.ball_y = self.player_y
        self.has_ball = True  # True → player; False → mate
        self.steps = 0
        self.possession_steps = 0
        self.completed_passes = 0
        self.defender_stun = 0
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        self.success = False
        return self._get_state()

    def _rel(self, tx: float, ty: float) -> Tuple[float, float]:
        dx = tx - self.player_x
        dy = ty - self.player_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _get_state(self) -> State:
        dc, tc = self._rel(self.mate_x, self.mate_y)
        dd, td = self._rel(self.def_x, self.def_y)
        c_bin, ct_bin = discretize(dc, tc)
        d_bin, dt_bin = discretize(dd, td)
        return (c_bin, ct_bin, d_bin, dt_bin, 1 if self.has_ball else 0)

    def _move_defender(self):
        if self.defender_mode == "passive":
            return
        if self.defender_stun > 0:
            self.defender_stun -= 1
            return
        bx, by = self.ball_x, self.ball_y
        dx, dy = bx - self.def_x, by - self.def_y
        dist = math.hypot(dx, dy) + 1e-6
        speed = self.defender_speed_base + self.defender_speed_jitter * float(self.rng.random())
        self.def_x += speed * dx / dist
        self.def_y += speed * dy / dist

    def _ball_intercepted(self) -> bool:
        return math.hypot(self.ball_x - self.def_x, self.ball_y - self.def_y) < self.INTERCEPT_RADIUS

    def _ball_out(self) -> bool:
        return not (-52.5 <= self.ball_x <= 52.5 and -34.0 <= self.ball_y <= 34.0)

    def _team_has_ball(self) -> bool:
        if self._ball_intercepted() or self._ball_out():
            return False
        if self.has_ball:
            return True
        return math.hypot(self.ball_x - self.mate_x, self.ball_y - self.mate_y) < 2.5

    def _pass_lane_clear(self) -> bool:
        mid_x = 0.5 * (self.player_x + self.mate_x)
        mid_y = 0.5 * (self.player_y + self.mate_y)
        return math.hypot(mid_x - self.def_x, mid_y - self.def_y) >= 2.0

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        self.steps += 1
        done = False
        self.success = False
        reward = 0.0

        if action == 0:  # PASS
            d_mate, _ = self._rel(self.mate_x, self.mate_y)
            if d_mate > self.PASS_RANGE:
                reward = -3.0
            elif not self._pass_lane_clear():
                # Intercepted in lane
                mid_x = 0.5 * (self.player_x + self.mate_x)
                mid_y = 0.5 * (self.player_y + self.mate_y)
                self.ball_x, self.ball_y = mid_x, mid_y
                reward = -35.0
                done = True
            else:
                if self.has_ball:
                    self.ball_x, self.ball_y = self.mate_x, self.mate_y
                    self.has_ball = False
                else:
                    self.ball_x = self.player_x + 0.4
                    self.ball_y = self.player_y
                    self.has_ball = True
                self.completed_passes += 1
                self.defender_stun = self.defender_stun_on_pass
                # Moderate pass reward — avoid spam dominating G0
                reward = 8.0 + (5.0 if self.completed_passes <= 3 else 0.0)
        elif action == 1:  # DRIBBLE (keep ball, move)
            rad = math.radians(self.player_theta)
            self.player_x += 0.65 * math.cos(rad)
            self.player_y += 0.65 * math.sin(rad)
            if self.has_ball:
                self.ball_x = self.player_x + 0.4 * math.cos(rad)
                self.ball_y = self.player_y + 0.4 * math.sin(rad)
            # Mate drifts supportively
            self.mate_x += float(self.rng.uniform(-0.15, 0.35))
            self.mate_y += float(self.rng.uniform(-0.25, 0.25))
            # Reward moving away from defender
            d_before = math.hypot(self.def_x - self.player_x, self.def_y - self.player_y)
            reward = 0.5
            if d_before > 8.0:
                reward += 0.8
        elif action == 2:  # TURN
            self.player_theta = (self.player_theta + 35.0 + 180.0) % 360.0 - 180.0
            reward = -0.1
        elif action == 3:  # ALEJARSE — step away from defender while keeping ball
            dx = self.player_x - self.def_x
            dy = self.player_y - self.def_y
            dist = math.hypot(dx, dy) + 1e-6
            self.player_x += 0.7 * dx / dist
            self.player_y += 0.7 * dy / dist
            self.player_theta = math.degrees(math.atan2(dy, dx))
            if self.has_ball:
                self.ball_x = self.player_x + 0.35
                self.ball_y = self.player_y
            reward = 1.2

        if not done:
            self._move_defender()
            # Soft mate movement when holding ball
            if not self.has_ball:
                self.mate_x += float(self.rng.uniform(-0.1, 0.2))
                self.mate_y += float(self.rng.uniform(-0.2, 0.2))
                self.ball_x, self.ball_y = self.mate_x, self.mate_y

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)

        if not done and self._team_has_ball():
            self.possession_steps += 1
            # Dense reward aligned with ≥50 possession criterion
            reward += 1.2
            if self.possession_steps >= 40:
                reward += 0.5  # late-episode push

        if not done and self._ball_out():
            reward = -12.0
            done = True
        elif not done and self._ball_intercepted():
            reward = -35.0
            done = True
        elif not done and self.possession_steps >= 50 and self.completed_passes >= 3:
            done = True
            self.success = True
            reward = 120.0
        elif not done and self.steps >= self.max_steps:
            done = True
            if self.possession_steps >= 50 and self.completed_passes >= 3:
                self.success = True
                reward = 120.0
            else:
                # Partial credit toward criterion (helps learning signal)
                reward = (
                    -15.0
                    + 0.4 * min(self.possession_steps, 50)
                    + 4.0 * min(self.completed_passes, 3)
                )

        return self._get_state(), float(reward), done, {
            "success": self.success,
            "possession_steps": self.possession_steps,
            "completed_passes": self.completed_passes,
            "has_ball": self.has_ball,
            "steps": self.steps,
        }
