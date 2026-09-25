#!/usr/bin/env python3
"""Ablación de libertad del defensor en passing (stun / velocidad).

Guarda métricas y una figura de barras en figures/tasks/passing/.
También exporta trays comparativas (stun=3 vs stun=0) para el informe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_control import train_mc_control  # noqa: E402
from agents.q_learning import train_q_learning  # noqa: E402
from envs.passing import PassingPossessionSimEnv  # noqa: E402
from metrics import summarize_run  # noqa: E402
from plotting import plot_trajectory, rollout_greedy  # noqa: E402

OUT = ROOT / "figures" / "tasks" / "passing"
N_EPISODES = 3000
SEED = 42

# (label, stun, speed_base, speed_jitter)
CONFIGS = [
    ("stun=3 (histórico)", 3, 0.18, 0.10),
    ("stun=1", 1, 0.18, 0.10),
    ("stun=0 (adoptado)", 0, 0.18, 0.10),
    ("stun=0, v+25%", 0, 0.225, 0.125),
    ("stun=0, v+50%", 0, 0.27, 0.15),
]


def _train(stun: int, speed_base: float, speed_jitter: float, algo: str) -> dict:
    env = PassingPossessionSimEnv(
        seed=SEED,
        defender_mode="stochastic",
        defender_stun_on_pass=stun,
        defender_speed_base=speed_base,
        defender_speed_jitter=speed_jitter,
    )
    if algo == "MC":
        res = train_mc_control(
            env=env, n_episodes=N_EPISODES, seed=SEED,
            exploration="fixed", eps_fixed=0.1,
        )
    else:
        res = train_q_learning(
            env=env, n_episodes=N_EPISODES, seed=SEED,
            exploration="fixed", eps_fixed=0.1, alpha=0.1,
        )
    summary = summarize_run(res, last_n=100)
    summary["algo"] = algo
    summary["stun"] = stun
    summary["speed_base"] = speed_base
    summary["speed_jitter"] = speed_jitter
    return {"summary": summary, "Q": res["Q"]}


def _plot_bars(rows: list[dict], path: Path) -> None:
    labels = [r["label"] for r in rows]
    mc = [r["MC"]["success_rate_last"] * 100 for r in rows]
    ql = [r["QL"]["success_rate_last"] * 100 for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w / 2, mc, w, label="MC ε fijo", color="#1565c0")
    ax.bar(x + w / 2, ql, w, label="QL ε fijo", color="#2e7d32")
    ax.axhline(90, color="#ef6c00", linestyle="--", linewidth=1.2, label="ref. 90%")
    ax.set_ylabel("Éxito últimos 100 eps (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("Passing — ablación de libertad del defensor (3000 eps, seed 42)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _tray_pair(Q_old, Q_new) -> None:
    """Trays lado a lado: stun=3 vs stun=0 (misma semilla visual)."""
    sets = OUT / "sets"
    sets.mkdir(parents=True, exist_ok=True)
    seed = 40
    for tag, Q, stun in (("stun3", Q_old, 3), ("stun0", Q_new, 0)):
        env = PassingPossessionSimEnv(
            seed=seed, defender_mode="stochastic", defender_stun_on_pass=stun,
        )
        rollout_greedy(env, Q)
        h = env._rollout_history
        dx = np.asarray(h["def_x"], float)
        dy = np.asarray(h["def_y"], float)
        disp = float(np.hypot(dx[-1] - dx[0], dy[-1] - dy[0]))
        path = sets / f"tray_ablation_{tag}_seed{seed}.png"
        plot_trajectory(
            env,
            title=f"Passing — defensor stun={stun} (disp≈{disp:.1f} m, seed={seed})",
            save_path=path,
            info=env._rollout_info,
            history=h,
        )
        plt.close("all")
        print(f"  tray {tag}: success={env.success} def_disp={disp:.1f}m → {path.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    Q_stun3 = Q_stun0 = None
    for label, stun, sb, sj in CONFIGS:
        print(f"=== {label} ===")
        entry = {"label": label, "stun": stun, "speed_base": sb, "speed_jitter": sj}
        for algo in ("MC", "QL"):
            print(f"  training {algo} ...")
            packed = _train(stun, sb, sj, algo)
            entry[algo] = packed["summary"]
            print(
                f"    success={packed['summary']['success_rate_last']:.0%}  "
                f"passes={packed['summary'].get('mean_passes_last', 0):.1f}  "
                f"poss={packed['summary'].get('mean_possession_last', 0):.1f}"
            )
            if label.startswith("stun=3") and algo == "MC":
                Q_stun3 = packed["Q"]
            if label.startswith("stun=0 (adoptado)") and algo == "MC":
                Q_stun0 = packed["Q"]
        rows.append(entry)

    # Persist without Q tables
    serializable = []
    for r in rows:
        serializable.append({
            "label": r["label"],
            "stun": r["stun"],
            "speed_base": r["speed_base"],
            "speed_jitter": r["speed_jitter"],
            "MC": {k: v for k, v in r["MC"].items() if k != "Q"},
            "QL": {k: v for k, v in r["QL"].items() if k != "Q"},
        })
    (OUT / "defender_ablation.json").write_text(
        json.dumps({"n_episodes": N_EPISODES, "seed": SEED, "rows": serializable}, indent=2),
        encoding="utf-8",
    )
    fig_path = OUT / "defender_ablation.png"
    _plot_bars(rows, fig_path)
    latex = ROOT / "plantilla_informe_latex" / "fig_passing_ablation.png"
    latex.write_bytes(fig_path.read_bytes())
    print(f"Figura → {fig_path}")

    if Q_stun3 is not None and Q_stun0 is not None:
        print("Trays comparativas stun=3 vs stun=0 ...")
        _tray_pair(Q_stun3, Q_stun0)
        # Sync ablation trays to LaTeX
        for tag in ("stun3", "stun0"):
            src = OUT / "sets" / f"tray_ablation_{tag}_seed40.png"
            if src.exists():
                (ROOT / "plantilla_informe_latex" / f"fig_passing_ablation_{tag}.png").write_bytes(
                    src.read_bytes()
                )

    # Refresh canonical tray with adopted policy (stun=0, MC)
    if Q_stun0 is not None:
        best_seed, best_env, best_disp = None, None, -1.0
        for seed in range(0, 80):
            env = PassingPossessionSimEnv(seed=seed, defender_stun_on_pass=0)
            rollout_greedy(env, Q_stun0)
            if not env.success:
                continue
            h = env._rollout_history
            dx = np.asarray(h["def_x"], float)
            dy = np.asarray(h["def_y"], float)
            disp = float(np.hypot(dx[-1] - dx[0], dy[-1] - dy[0]))
            if disp > best_disp:
                best_disp, best_seed, best_env = disp, seed, env
        if best_env is not None:
            canon = OUT / "trayectoria.png"
            plot_trajectory(
                best_env,
                title="Passing — trayectoria greedy (MC | ε fijo, stun=0)",
                save_path=canon,
                info=best_env._rollout_info,
                history=best_env._rollout_history,
            )
            plt.close("all")
            (ROOT / "plantilla_informe_latex" / "fig_passing_tray.png").write_bytes(canon.read_bytes())
            print(f"Canónica actualizada (seed={best_seed}, def_disp={best_disp:.1f}m)")

            # Refresh set variants for informe (3 trays)
            picked = []
            for seed in range(0, 120):
                env = PassingPossessionSimEnv(seed=seed, defender_stun_on_pass=0)
                rollout_greedy(env, Q_stun0)
                if not env.success:
                    continue
                h = env._rollout_history
                dx = np.asarray(h["def_x"], float)
                dy = np.asarray(h["def_y"], float)
                disp = float(np.hypot(dx[-1] - dx[0], dy[-1] - dy[0]))
                picked.append((disp, seed, env))
                if len(picked) >= 30:
                    break
            picked.sort(reverse=True)
            chosen = []
            for disp, seed, env in picked:
                if any(abs(seed - s) < 3 for s, _ in chosen):
                    continue
                chosen.append((seed, env))
                if len(chosen) >= 5:
                    break
            sets = OUT / "sets"
            for i, (seed, env) in enumerate(chosen, 1):
                path = sets / f"tray_run_{i:02d}_seed{seed}.png"
                plot_trajectory(
                    env,
                    title=f"Passing — variante {i} (stun=0, seed={seed})",
                    save_path=path,
                    info=env._rollout_info,
                    history=env._rollout_history,
                )
                plt.close("all")
                if i <= 3:
                    (ROOT / "plantilla_informe_latex" / f"set_passing_{i:02d}.png").write_bytes(
                        path.read_bytes()
                    )
            print(f"Sets regenerados: {[s for s, _ in chosen]}")

    print("Listo.")


if __name__ == "__main__":
    main()
