#!/usr/bin/env python3
"""Entrena MC Control y Q-Learning (ε fijo vs decay) y exporta figuras/métricas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agents.mc_control import train_mc_control  # noqa: E402
from agents.q_learning import train_q_learning  # noqa: E402
from env import BallPursuitSimEnv  # noqa: E402
from metrics import summarize_run  # noqa: E402
from plotting import plot_learning_curves, plot_trajectory, plot_value_and_policy, rollout_greedy  # noqa: E402

N_EPISODES = 3500
SEED = 42
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def main():
    configs = [
        ("MC | ε decay", train_mc_control, {"exploration": "decay"}),
        ("MC | ε fijo=0.1", train_mc_control, {"exploration": "fixed", "eps_fixed": 0.1}),
        ("QL | ε decay", train_q_learning, {"exploration": "decay", "alpha": 0.1}),
        ("QL | ε fijo=0.1", train_q_learning, {"exploration": "fixed", "eps_fixed": 0.1, "alpha": 0.1}),
    ]

    results = {}
    summaries = {}
    for name, trainer, kwargs in configs:
        print(f"Entrenando: {name} ...")
        env = BallPursuitSimEnv(seed=SEED)
        res = trainer(env=env, n_episodes=N_EPISODES, seed=SEED, **kwargs)
        results[name] = res
        summaries[name] = summarize_run(res, last_n=100)
        print(f"  → {summaries[name]}")

    plot_learning_curves(results, window=50, save_path=FIG / "curvas_aprendizaje.png")
    plt_close = True

    # Mejor corrida por tasa de éxito final para V*/π* y trayectoria
    best_name = max(summaries, key=lambda k: summaries[k]["success_rate_last"])
    best = results[best_name]
    plot_value_and_policy(
        best["Q"],
        title=f"Mejor política: {best_name}",
        save_path=FIG / "valor_politica.png",
    )
    env_vis = BallPursuitSimEnv(seed=7)
    rollout_greedy(env_vis, best["Q"])
    plot_trajectory(
        env_vis,
        title=f"Trayectoria greedy — {best_name}",
        save_path=FIG / "trayectoria.png",
    )

    out = {
        "n_episodes": N_EPISODES,
        "seed": SEED,
        "best": best_name,
        "summaries": summaries,
    }
    (FIG / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nResumen guardado en figures/metrics.json")
    print(json.dumps(out, indent=2))

    if plt_close:
        import matplotlib.pyplot as plt
        plt.close("all")


if __name__ == "__main__":
    main()
