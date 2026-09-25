#!/usr/bin/env python3
"""Entrena MC Control y Q-Learning (ε fijo vs decay) en las 4 tareas P1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agents.mc_control import train_mc_control  # noqa: E402
from agents.q_learning import train_q_learning  # noqa: E402
from envs import TASK_ENVS  # noqa: E402
from metrics import evaluate_shooting_split, summarize_run  # noqa: E402
from plotting import plot_learning_curves, plot_trajectory, plot_value_and_policy, rollout_greedy  # noqa: E402

N_EPISODES = 3500
SEED = 42
FIG_ROOT = ROOT / "figures" / "tasks"


def _configs():
    return [
        ("MC | ε decay", train_mc_control, {"exploration": "decay"}),
        ("MC | ε fijo=0.1", train_mc_control, {"exploration": "fixed", "eps_fixed": 0.1}),
        ("QL | ε decay", train_q_learning, {"exploration": "decay", "alpha": 0.1}),
        ("QL | ε fijo=0.1", train_q_learning, {"exploration": "fixed", "eps_fixed": 0.1, "alpha": 0.1}),
    ]


def run_task(task_name: str) -> dict:
    print(f"\n===== Tarea: {task_name} =====")
    out_dir = FIG_ROOT / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    EnvCls = TASK_ENVS[task_name]

    results = {}
    summaries = {}
    for name, trainer, kwargs in _configs():
        print(f"  Entrenando: {name} ...")
        env = EnvCls(seed=SEED)
        res = trainer(env=env, n_episodes=N_EPISODES, seed=SEED, **kwargs)
        results[name] = res
        summaries[name] = summarize_run(res, last_n=100)
        print(f"    → {summaries[name]}")

    plot_learning_curves(
        results,
        window=50,
        title=f"Curvas — {task_name}",
        save_path=out_dir / "curvas_aprendizaje.png",
    )

    best_name = max(summaries, key=lambda k: summaries[k]["success_rate_last"])
    best = results[best_name]
    action_names = getattr(EnvCls(seed=0), "action_names", None)

    if task_name == "pursuit":
        plot_value_and_policy(
            best["Q"],
            title=f"Mejor política: {best_name}",
            save_path=out_dir / "valor_politica.png",
            action_names=action_names,
        )

    env_vis = EnvCls(seed=7)
    # For shooting evaluation, prefer open-goal visualization
    if task_name == "shooting":
        env_vis = EnvCls(seed=7, with_gk=False)
    rollout_greedy(env_vis, best["Q"])
    plot_trajectory(
        env_vis,
        title=f"Trayectoria greedy — {task_name} ({best_name})",
        save_path=out_dir / "trayectoria.png",
        info=getattr(env_vis, "_rollout_info", None),
        history=getattr(env_vis, "_rollout_history", None),
    )

    # Shooting: also save a with-GK tray for the informe
    if task_name == "shooting":
        env_gk = EnvCls(seed=11, with_gk=True)
        rollout_greedy(env_gk, best["Q"])
        plot_trajectory(
            env_gk,
            title=f"Trayectoria greedy — shooting con portero ({best_name})",
            save_path=out_dir / "trayectoria_gk.png",
            info=getattr(env_gk, "_rollout_info", None),
            history=getattr(env_gk, "_rollout_history", None),
        )

    task_metrics = {
        "best": best_name,
        "summaries": summaries,
    }
    if task_name == "shooting":
        split = evaluate_shooting_split(best["Q"], seed=99, n_episodes=200)
        task_metrics["eval_split"] = split
        print(f"  Eval tiro abierto/GK: {split}")

    (out_dir / "metrics.json").write_text(json.dumps(task_metrics, indent=2), encoding="utf-8")
    return task_metrics


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = {"n_episodes": N_EPISODES, "seed": SEED, "tasks": {}}
    for task in ("pursuit", "dribbling", "shooting", "passing"):
        all_metrics["tasks"][task] = run_task(task)

    out = ROOT / "figures" / "metrics_all_tasks.json"
    out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nResumen global: {out}")

    # Also refresh legacy pursuit figures for informe compatibility
    pursuit_dir = FIG_ROOT / "pursuit"
    legacy = ROOT / "figures"
    for name in ("curvas_aprendizaje.png", "valor_politica.png", "trayectoria.png"):
        src = pursuit_dir / name
        if src.exists():
            (legacy / name).write_bytes(src.read_bytes())
    pursuit_metrics = all_metrics["tasks"]["pursuit"]
    (legacy / "metrics.json").write_text(
        json.dumps(
            {
                "n_episodes": N_EPISODES,
                "seed": SEED,
                "best": pursuit_metrics["best"],
                "summaries": pursuit_metrics["summaries"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    main()
