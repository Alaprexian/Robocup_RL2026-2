"""Entornos cinemáticos tabulares del catálogo P1."""

from envs.pursuit import BallPursuitSimEnv
from envs.dribbling import BallDribblingSimEnv
from envs.shooting import GoalShootingSimEnv
from envs.passing import PassingPossessionSimEnv

TASK_ENVS = {
    "pursuit": BallPursuitSimEnv,
    "dribbling": BallDribblingSimEnv,
    "shooting": GoalShootingSimEnv,
    "passing": PassingPossessionSimEnv,
}

__all__ = [
    "BallPursuitSimEnv",
    "BallDribblingSimEnv",
    "GoalShootingSimEnv",
    "PassingPossessionSimEnv",
    "TASK_ENVS",
]
