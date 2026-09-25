#!/usr/bin/env python3
"""Genera varios sets de trayectorias por tarea (éxitos con seeds distintas)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_control import train_mc_control  # noqa: E402
from agents.q_learning import train_q_learning  # noqa: E402
from envs import TASK_ENVS  # noqa: E402
from plotting import plot_trajectory, rollout_greedy  # noqa: E402

N_EPISODES = 3500
TRAIN_SEED = 42
N_PER_TASK = 5
OUT = ROOT / "figures" / "tasks"

PREFERRED = {
    "pursuit": ("QL | ε decay", train_q_learning, {"exploration": "decay", "alpha": 0.1}),
    "dribbling": ("QL | ε decay", train_q_learning, {"exploration": "decay", "alpha": 0.1}),
    "shooting": ("MC | ε fijo=0.1", train_mc_control, {"exploration": "fixed", "eps_fixed": 0.1}),
    "passing": ("MC | ε fijo=0.1", train_mc_control, {"exploration": "fixed", "eps_fixed": 0.1}),
}


def _score_episode(task: str, env, info: dict) -> float:
    """Mayor = más interesante visualmente (y exitoso)."""
    if not getattr(env, "success", False):
        return -1e9
    xs = list(env.trajectory_x)
    ys = list(env.trajectory_y)
    path_len = 0.0
    for i in range(1, len(xs)):
        path_len += ((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5

    if task == "pursuit":
        return path_len + 0.5 * info.get("steps", 0)
    if task == "dribbling":
        return float(info.get("advance", 0.0)) + 0.2 * path_len
    if task == "shooting":
        sep = 0.0
        if getattr(env, "has_gk", False) and env.gk_y is not None:
            sep = abs(float(env.ball_y) - float(env.gk_y))
        # Prefer goals inside posts and, if GK, away from keeper
        inside = 3.0 if abs(float(env.ball_y)) <= 6.8 else -5.0
        return sep + inside + 0.5 * info.get("steps", 1)
    if task == "passing":
        return (
            float(info.get("possession_steps", 0))
            + 2.0 * float(info.get("completed_passes", 0))
            + 0.15 * path_len
        )
    return path_len


def _collect_candidates(task: str, EnvCls, Q, with_gk=None, n_want: int = 5, scan: int = 120):
    scored = []
    seen_sig = set()
    for s in range(scan):
        kwargs = {}
        if task == "shooting":
            kwargs["with_gk"] = False if with_gk is None else with_gk
        env = EnvCls(seed=s, **kwargs)
        rollout_greedy(env, Q)
        info = getattr(env, "_rollout_info", {}) or {}
        if not getattr(env, "success", False):
            continue
        # Diversity signature: coarse end pose + ball
        sig = (
            round(env.trajectory_x[-1] / 3.0),
            round(env.trajectory_y[-1] / 3.0),
            round(getattr(env, "ball_x", 0.0) / 3.0),
            round(getattr(env, "ball_y", 0.0) / 3.0),
            int(getattr(env, "has_gk", False)),
        )
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        scored.append((_score_episode(task, env, info), s, env, info))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:n_want]


def main():
    for task, (name, trainer, kwargs) in PREFERRED.items():
        print(f"\n=== {task}: entrenando {name} ===")
        EnvCls = TASK_ENVS[task]
        res = trainer(env=EnvCls(seed=TRAIN_SEED), n_episodes=N_EPISODES, seed=TRAIN_SEED, **kwargs)
        Q = res["Q"]

        sets_dir = OUT / task / "sets"
        sets_dir.mkdir(parents=True, exist_ok=True)

        if task == "shooting":
            # 3 open + 2 with GK (or as many as found)
            open_cands = _collect_candidates(task, EnvCls, Q, with_gk=False, n_want=3, scan=150)
            gk_cands = _collect_candidates(task, EnvCls, Q, with_gk=True, n_want=2, scan=150)
            cands = [("open", i + 1, *c) for i, c in enumerate(open_cands)]
            cands += [("gk", i + 1, *c) for i, c in enumerate(gk_cands)]
        else:
            raw = _collect_candidates(task, EnvCls, Q, n_want=N_PER_TASK, scan=150)
            cands = [("run", i + 1, *c) for i, c in enumerate(raw)]

        if not cands:
            print(f"  ! no se encontraron episodios exitosos diversos para {task}")
            continue

        for kind, idx, score, seed, env, info in cands:
            fname = f"tray_{kind}_{idx:02d}_seed{seed}.png"
            path = sets_dir / fname
            subtitle = {
                "open": "arco abierto",
                "gk": "con portero",
                "run": f"variante {idx}",
            }[kind]
            plot_trajectory(
                env,
                title=f"{task.capitalize()} — {subtitle} ({name}, seed={seed})",
                save_path=path,
                info=getattr(env, "_rollout_info", None),
                history=getattr(env, "_rollout_history", None),
            )
            print(f"  → {path.relative_to(ROOT)}  score={score:.1f} success={env.success}")

        # Also refresh the canonical single tray with the best of the set
        best = max(cands, key=lambda t: t[2])
        _, _, _, seed, env, _ = best
        canon = OUT / task / "trayectoria.png"
        plot_trajectory(
            env,
            title=f"{task.capitalize()} — trayectoria greedy ({name})",
            save_path=canon,
            info=getattr(env, "_rollout_info", None),
            history=getattr(env, "_rollout_history", None),
        )
        # Sync latex tray
        latex = ROOT / "plantilla_informe_latex" / f"fig_{task}_tray.png"
        latex.write_bytes(canon.read_bytes())
        if task == "shooting":
            gk_only = [c for c in cands if c[0] == "gk"]
            if gk_only:
                _, _, _, _, env_gk, _ = max(gk_only, key=lambda t: t[2])
                gk_path = OUT / task / "trayectoria_gk.png"
                plot_trajectory(
                    env_gk,
                    title=f"Shooting — trayectoria greedy con portero ({name})",
                    save_path=gk_path,
                    info=getattr(env_gk, "_rollout_info", None),
                    history=getattr(env_gk, "_rollout_history", None),
                )
                (ROOT / "plantilla_informe_latex" / "fig_shooting_tray_gk.png").write_bytes(gk_path.read_bytes())

    import matplotlib.pyplot as plt

    plt.close("all")
    print("\nListo. Sets en figures/tasks/<tarea>/sets/")


if __name__ == "__main__":
    main()
