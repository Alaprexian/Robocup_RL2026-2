"""Utilidades de visualización para P1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from discretizer import ACTION_NAMES
from metrics import label_axes, moving_average, q_to_value_policy, success_rate


def plot_learning_curves(
    results: Dict[str, Dict],
    window: int = 50,
    save_path: Optional[str | Path] = None,
    title: str = "",
):
    """Curvas de G0, tasa de éxito y pasos para múltiples corridas."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, res in results.items():
        g = moving_average(res["rewards"], window)
        s = success_rate(res["success"], window)
        L = moving_average(res["lengths"], window)
        x_g = np.arange(window - 1, window - 1 + len(g))
        x_s = np.arange(window - 1, window - 1 + len(s))
        x_L = np.arange(window - 1, window - 1 + len(L))
        axes[0].plot(x_g, g, label=name, linewidth=1.6)
        axes[1].plot(x_s, s, label=name, linewidth=1.6)
        axes[2].plot(x_L, L, label=name, linewidth=1.6)

    axes[0].set_title(f"Retorno $G_0$ (MA-{window})")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Retorno")
    axes[1].set_title(f"Tasa de éxito (ventana {window})")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Éxito")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title(f"Pasos / episodio (MA-{window})")
    axes[2].set_xlabel("Episodio")
    axes[2].set_ylabel("Pasos")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_value_and_policy(
    Q: Dict,
    title: str = "",
    save_path: Optional[str | Path] = None,
    action_names: Optional[Sequence[str]] = None,
):
    V, Pi = q_to_value_policy(Q)
    names = list(action_names) if action_names is not None else list(ACTION_NAMES)
    dist_labels, angle_labels = label_axes()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    im = axes[0].imshow(V, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(angle_labels)))
    axes[0].set_xticklabels(angle_labels, rotation=30, ha="right")
    axes[0].set_yticks(range(len(dist_labels)))
    axes[0].set_yticklabels(dist_labels)
    axes[0].set_title(r"$V^*(s)=\max_a Q(s,a)$")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].imshow(Pi, cmap="tab10", vmin=0, vmax=max(3, len(names) - 1), aspect="auto")
    axes[1].set_xticks(range(len(angle_labels)))
    axes[1].set_xticklabels(angle_labels, rotation=30, ha="right")
    axes[1].set_yticks(range(len(dist_labels)))
    axes[1].set_yticklabels(dist_labels)
    axes[1].set_title(r"$\pi^*(s)=\arg\max_a Q(s,a)$")
    for d in range(Pi.shape[0]):
        for a in range(Pi.shape[1]):
            if Pi[d, a] >= 0:
                label = names[Pi[d, a]][:4] if Pi[d, a] < len(names) else "?"
                axes[1].text(
                    a, d, label,
                    ha="center", va="center", color="white", fontsize=7, fontweight="bold",
                )

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def draw_pitch(
    ax,
    title: str = "",
    xlim: Tuple[float, float] = (-52.5, 52.5),
    ylim: Tuple[float, float] = (-34, 34),
    show_goals: bool = True,
):
    """Cancha reglamentaria con franjas y áreas."""
    # Grass stripes
    x0, x1 = -52.5, 52.5
    stripe_w = 5.25
    for i, x in enumerate(np.arange(x0, x1, stripe_w)):
        color = "#1b5e20" if i % 2 == 0 else "#2e7d32"
        ax.add_patch(patches.Rectangle((x, -34), stripe_w, 68, facecolor=color, edgecolor="none", zorder=0))

    ax.add_patch(patches.Rectangle((-52.5, -34), 105, 68, linewidth=2.0, edgecolor="white", facecolor="none", zorder=1))
    ax.add_line(plt.Line2D([0, 0], [-34, 34], color="white", linewidth=1.6, zorder=1))
    ax.add_patch(patches.Circle((0, 0), 9.15, linewidth=1.6, edgecolor="white", facecolor="none", zorder=1))
    # Center spot — small and muted so it is not confused with the ball
    ax.add_patch(patches.Circle((0, 0), 0.25, facecolor="#e8f5e9", edgecolor="white", linewidth=0.6, alpha=0.7, zorder=1))
    ax.add_patch(patches.Rectangle((-52.5, -20.16), 16.5, 40.32, linewidth=1.3, edgecolor="white", facecolor="none", zorder=1))
    ax.add_patch(patches.Rectangle((36.0, -20.16), 16.5, 40.32, linewidth=1.3, edgecolor="white", facecolor="none", zorder=1))
    ax.add_patch(patches.Rectangle((-52.5, -9.16), 5.5, 18.32, linewidth=1.1, edgecolor="white", facecolor="none", zorder=1))
    ax.add_patch(patches.Rectangle((47.0, -9.16), 5.5, 18.32, linewidth=1.1, edgecolor="white", facecolor="none", zorder=1))
    if show_goals:
        # RoboCup 2D goal mouth ≈ ±7.01 m (not FIFA indoor ±3.66)
        ax.add_patch(patches.Rectangle((-54.5, -7.01), 2.0, 14.02, linewidth=1.5, edgecolor="#ffeb3b", facecolor="#fffde7", alpha=0.85, zorder=2))
        ax.add_patch(patches.Rectangle((52.5, -7.01), 2.0, 14.02, linewidth=1.5, edgecolor="#ffeb3b", facecolor="#fffde7", alpha=0.85, zorder=2))
        ax.plot([52.5, 52.5], [-7.01, 7.01], color="#ffee58", linewidth=2.6, zorder=2, solid_capstyle="round")
        ax.plot([-52.5, -52.5], [-7.01, 7.01], color="#ffee58", linewidth=2.6, zorder=2, solid_capstyle="round")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)


def _gradient_path(ax, xs: Sequence[float], ys: Sequence[float], cmap_name: str = "YlGn", lw: float = 2.6):
    """Trayectoria con color por progreso (inicio→fin)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2:
        ax.plot(x, y, "o", color="#00e676", markersize=6, zorder=4)
        return
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap_name, linewidths=lw, zorder=4, alpha=0.95)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    # Soft underlay
    ax.plot(x, y, "-", color="white", linewidth=lw + 1.8, alpha=0.25, zorder=3)


def _orientation_arrow(ax, x: float, y: float, theta_deg: float, color: str = "#ffeb3b"):
    rad = np.radians(theta_deg)
    ax.annotate(
        "",
        xy=(x + 2.2 * np.cos(rad), y + 2.2 * np.sin(rad)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14),
        zorder=7,
    )


def _frame_limits(
    env,
    pad: float = 4.0,
    history: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = list(getattr(env, "trajectory_x", []) or [0.0])
    ys: List[float] = list(getattr(env, "trajectory_y", []) or [0.0])
    for attr in ("ball_x", "mate_x", "def_x", "gk_x", "start_x"):
        if hasattr(env, attr) and getattr(env, attr) is not None:
            xs.append(float(getattr(env, attr)))
    for attr in ("ball_y", "mate_y", "def_y", "gk_y"):
        if hasattr(env, attr) and getattr(env, attr) is not None:
            ys.append(float(getattr(env, attr)))
    if history:
        for key in ("ball_x", "mate_x", "def_x"):
            xs.extend(float(v) for v in history.get(key, []) or [])
        for key in ("ball_y", "mate_y", "def_y"):
            ys.extend(float(v) for v in history.get(key, []) or [])
    # Always include goal mouth for shooting/dribbling context
    task = getattr(env, "task_name", "")
    if task in ("shooting", "dribbling"):
        xs.extend([36.0, 54.0])
        ys.extend([-12.0, 12.0])
    if task == "pursuit":
        xs.extend([getattr(env, "ball_x", 0.0)])
        ys.extend([getattr(env, "ball_y", 0.0)])

    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    # Keep aspect reasonable
    if xmax - xmin < 16:
        mid = 0.5 * (xmin + xmax)
        xmin, xmax = mid - 8, mid + 8
    if ymax - ymin < 12:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - 6, mid + 6
    xmin = max(-54.0, xmin)
    xmax = min(55.0, xmax)
    ymin = max(-34.0, ymin)
    ymax = min(34.0, ymax)
    return (xmin, xmax), (ymin, ymax)


def _entity_trail(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    label: str,
    start_marker: str = "o",
    end_marker: str = "X",
    lw: float = 2.0,
):
    """Rastro visible de una entidad (inicio → fin)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) == 0:
        return
    if len(x) == 1 or float(np.ptp(x) + np.ptp(y)) < 0.15:
        ax.scatter(
            x[-1], y[-1], s=110, c=color, marker=end_marker,
            linewidths=1.4, zorder=6, label=f"{label} (sin desplazamiento)",
        )
        return
    ax.plot(x, y, "-", color=color, linewidth=lw, alpha=0.9, zorder=4, label=f"Rastro {label.lower()}")
    ax.scatter(
        x[0], y[0], s=70, facecolors="none", edgecolors=color,
        marker=start_marker, linewidths=1.6, zorder=6, label=f"{label} (inicio)",
    )
    ax.scatter(
        x[-1], y[-1], s=130, c=color, marker=end_marker,
        linewidths=1.5, edgecolors="white", zorder=7, label=f"{label} (fin)",
    )


def plot_trajectory(
    env,
    title: str = "",
    save_path: Optional[str | Path] = None,
    info: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, List[float]]] = None,
):
    """
    Trayectoria enriquecida: cancha con franjas, path en degradé,
    orientación, entidades y recorte adaptativo.
    """
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="#0d1f12")
    ax.set_facecolor("#0d1f12")
    xlim, ylim = _frame_limits(env, history=history)
    draw_pitch(ax, title="", xlim=xlim, ylim=ylim)

    xs = list(env.trajectory_x)
    ys = list(env.trajectory_y)
    _gradient_path(ax, xs, ys, cmap_name="winter" if getattr(env, "success", False) else "Wistia")

    # Start / end
    ax.scatter(xs[0], ys[0], s=120, c="#29b6f6", marker="s", edgecolors="white", linewidths=1.2, zorder=6, label="Inicio")
    ax.scatter(xs[-1], ys[-1], s=140, c="#1565c0", marker="o", edgecolors="white", linewidths=1.3, zorder=6, label="Jugador (fin)")
    if hasattr(env, "player_theta"):
        _orientation_arrow(ax, xs[-1], ys[-1], float(env.player_theta))

    # Ball trail if recorded and the ball actually moved
    if history and history.get("ball_x") and len(history["ball_x"]) > 1:
        bx = np.asarray(history["ball_x"], dtype=float)
        by = np.asarray(history["ball_y"], dtype=float)
        if float(np.ptp(bx) + np.ptp(by)) > 0.8:
            ax.plot(bx, by, "--", color="#eceff1", linewidth=1.4, alpha=0.75, zorder=3, label="Rastro del balón")
            step = max(1, len(bx) // 10)
            ax.scatter(bx[::step], by[::step], s=16, c="#cfd8dc", zorder=3, alpha=0.85)

    if hasattr(env, "ball_x"):
        ax.scatter(
            env.ball_x, env.ball_y, s=160, c="white", edgecolors="black", linewidths=1.6, zorder=6, label="Balón",
        )

    # Mate / defender: show full trails (critical for passing — otherwise look "static")
    if history and history.get("mate_x"):
        _entity_trail(
            ax, history["mate_x"], history["mate_y"],
            color="#ffeb3b", label="Compañero", start_marker="^", end_marker="^", lw=1.8,
        )
    elif hasattr(env, "mate_x"):
        ax.scatter(
            env.mate_x, env.mate_y, s=130, c="#ffeb3b", marker="^",
            edgecolors="#333", linewidths=0.8, zorder=6, label="Compañero",
        )

    if history and history.get("def_x"):
        _entity_trail(
            ax, history["def_x"], history["def_y"],
            color="#ef5350", label="Defensor", start_marker="o", end_marker="X", lw=2.4,
        )
    elif hasattr(env, "def_x"):
        ax.scatter(
            env.def_x, env.def_y, s=120, c="#ef5350", marker="X",
            linewidths=1.5, zorder=6, label="Defensor",
        )

    if getattr(env, "has_gk", False) and getattr(env, "gk_y", None) is not None:
        ax.scatter(
            getattr(env, "gk_x", 50.5), env.gk_y,
            s=140, c="#ab47bc", marker="D", edgecolors="white", linewidths=1.0, zorder=6, label="Portero",
        )

    # Shot line for shooting (from last player pose to ball impact)
    task = getattr(env, "task_name", "")
    if task == "shooting" and len(xs) >= 1 and hasattr(env, "ball_x"):
        ax.annotate(
            "",
            xy=(env.ball_x, env.ball_y),
            xytext=(xs[-1], ys[-1]),
            arrowprops=dict(arrowstyle="->", color="#ffe082", lw=1.8, linestyle="--"),
            zorder=5,
        )
        ax.text(
            0.5 * (xs[-1] + env.ball_x),
            0.5 * (ys[-1] + env.ball_y) + 1.0,
            "disparo",
            color="#ffe082",
            fontsize=8,
            ha="center",
            zorder=5,
        )

    success = bool(getattr(env, "success", False))
    if info is not None:
        success = bool(info.get("success", success))
    badge = "ÉXITO" if success else "SIN ÉXITO"
    badge_color = "#00c853" if success else "#ff6d00"
    ax.text(
        0.02, 0.98, badge,
        transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_color, edgecolor="white", alpha=0.92),
        zorder=10,
    )

    # Compact metrics strip
    meta_bits = []
    if info:
        if "steps" in info:
            meta_bits.append(f"pasos={info['steps']}")
        if "dist" in info:
            meta_bits.append(f"dist={info['dist']:.1f}m")
        if "advance" in info:
            meta_bits.append(f"avance={info['advance']:.1f}m")
        if "possession_steps" in info:
            meta_bits.append(f"posesión={info['possession_steps']}")
        if "completed_passes" in info:
            meta_bits.append(f"pases={info['completed_passes']}")
        if "outcome" in info:
            meta_bits.append(f"tiro={info['outcome']}")
    if meta_bits:
        ax.text(
            0.98, 0.02, " · ".join(meta_bits),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa", edgecolor="none"),
            zorder=10,
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, facecolor="#1b5e20", labelcolor="white", edgecolor="white")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="#e8f5e9", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def rollout_greedy(env, Q: Dict, max_steps: Optional[int] = None):
    """Ejecuta política greedy; guarda historial de entidades y último info."""
    state = env.reset()
    max_steps = max_steps or env.max_steps
    history = {
        "ball_x": [float(getattr(env, "ball_x", env.player_x))],
        "ball_y": [float(getattr(env, "ball_y", env.player_y))],
        "mate_x": [],
        "mate_y": [],
        "def_x": [],
        "def_y": [],
    }
    if hasattr(env, "mate_x"):
        history["mate_x"].append(float(env.mate_x))
        history["mate_y"].append(float(env.mate_y))
    if hasattr(env, "def_x"):
        history["def_x"].append(float(env.def_x))
        history["def_y"].append(float(env.def_y))

    last_info: Dict[str, Any] = {}
    for _ in range(max_steps):
        q_vals = Q.get(state)
        if q_vals is None or np.allclose(q_vals, q_vals[0]):
            action = 0
        else:
            action = int(np.argmax(q_vals))
        state, _, done, info = env.step(action)
        last_info = info
        history["ball_x"].append(float(getattr(env, "ball_x", env.player_x)))
        history["ball_y"].append(float(getattr(env, "ball_y", env.player_y)))
        if hasattr(env, "mate_x"):
            history["mate_x"].append(float(env.mate_x))
            history["mate_y"].append(float(env.mate_y))
        if hasattr(env, "def_x"):
            history["def_x"].append(float(env.def_x))
            history["def_y"].append(float(env.def_y))
        if done:
            break
    env._rollout_history = history
    env._rollout_info = last_info
    return env
