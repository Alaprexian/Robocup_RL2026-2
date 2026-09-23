"""Utilidades de visualización para P1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from discretizer import ACTION_NAMES
from metrics import label_axes, moving_average, q_to_value_policy, success_rate


def plot_learning_curves(
    results: Dict[str, Dict],
    window: int = 50,
    save_path: Optional[str | Path] = None,
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
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_value_and_policy(Q: Dict, title: str = "", save_path: Optional[str | Path] = None):
    V, Pi = q_to_value_policy(Q)
    dist_labels, angle_labels = label_axes()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    im = axes[0].imshow(V, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(angle_labels)))
    axes[0].set_xticklabels(angle_labels, rotation=30, ha="right")
    axes[0].set_yticks(range(len(dist_labels)))
    axes[0].set_yticklabels(dist_labels)
    axes[0].set_title(r"$V^*(s)=\max_a Q(s,a)$")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].imshow(Pi, cmap="tab10", vmin=0, vmax=3, aspect="auto")
    axes[1].set_xticks(range(len(angle_labels)))
    axes[1].set_xticklabels(angle_labels, rotation=30, ha="right")
    axes[1].set_yticks(range(len(dist_labels)))
    axes[1].set_yticklabels(dist_labels)
    axes[1].set_title(r"$\pi^*(s)=\arg\max_a Q(s,a)$")
    for d in range(Pi.shape[0]):
        for a in range(Pi.shape[1]):
            if Pi[d, a] >= 0:
                axes[1].text(
                    a, d, ACTION_NAMES[Pi[d, a]][:4],
                    ha="center", va="center", color="white", fontsize=7, fontweight="bold",
                )

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def draw_pitch(ax, title: str = ""):
    ax.set_facecolor("#2e7d32")
    ax.add_patch(patches.Rectangle((-52.5, -34), 105, 68, linewidth=1.8, edgecolor="white", facecolor="none"))
    ax.add_line(plt.Line2D([0, 0], [-34, 34], color="white", linewidth=1.8))
    ax.add_patch(patches.Circle((0, 0), 9.15, linewidth=1.8, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Circle((0, 0), 0.5, color="white"))
    ax.add_patch(patches.Rectangle((-52.5, -20.16), 16.5, 40.32, linewidth=1.2, edgecolor="white", facecolor="none"))
    ax.add_patch(patches.Rectangle((36.0, -20.16), 16.5, 40.32, linewidth=1.2, edgecolor="white", facecolor="none"))
    ax.set_xlim(-30, 15)
    ax.set_ylim(-18, 18)
    ax.set_title(title)
    ax.set_aspect("equal")


def plot_trajectory(env, title: str = "", save_path: Optional[str | Path] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_pitch(ax, title)
    ax.plot(env.trajectory_x, env.trajectory_y, "-o", color="#00e676", markersize=3, linewidth=1.8)
    ax.plot(env.trajectory_x[0], env.trajectory_y[0], "s", color="#29b6f6", markersize=8, label="Inicio")
    ax.plot(env.ball_x, env.ball_y, "o", color="white", markeredgecolor="black", markersize=10, label="Balón")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def rollout_greedy(env, Q: Dict, max_steps: Optional[int] = None):
    """Ejecuta política greedy y deja trayectoria en env."""
    state = env.reset()
    max_steps = max_steps or env.max_steps
    for _ in range(max_steps):
        q_vals = Q.get(state)
        if q_vals is None or np.allclose(q_vals, q_vals[0]):
            action = 0
        else:
            action = int(np.argmax(q_vals))
        state, _, done, _ = env.step(action)
        if done:
            break
    return env
