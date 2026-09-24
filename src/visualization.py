"""
Módulo de Visualización Gráfica y Renderizado 2D para RoboCup 2D
DS5345 - Aprendizaje por Refuerzo (UTEC 2026-II)

Incluye:
- Terreno de juego oficial 2D (105m x 68m con áreas reglamentarias)
- Curvas de aprendizaje (Retorno G_0, pasos por episodio, tasa de éxito en ventana móvil)
- Mapa de calor de la función de valor óptima V*(s) y matriz de decisiones pi*(s)
- Comparación de trayectorias (Agente inicial vs. Agente entrenado)
- Animación interactiva paso a paso
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output


def draw_soccer_pitch(
    ax: plt.Axes,
    title: str = "RoboCup 2D Soccer Simulation League (105m x 68m)",
    xlim: Tuple[float, float] = (-58.0, 58.0),
    ylim: Tuple[float, float] = (-38.0, 38.0)
) -> None:
    """
    Dibuja el campo oficial de fútbol de RoboCup 2D (105m x 68m)
    con césped verde, líneas blancas, círculo central, áreas grandes y porterías.
    """
    ax.set_facecolor("#2e7d32")

    # Borde exterior del campo (105m x 68m)
    pitch_border = patches.Rectangle((-52.5, -34.0), 105.0, 68.0, linewidth=2.0, edgecolor="white", facecolor="none")
    half_line = plt.Line2D([0, 0], [-34.0, 34.0], color="white", linewidth=2.0)
    center_circle = patches.Circle((0, 0), 9.15, linewidth=2.0, edgecolor="white", facecolor="none")
    center_spot = patches.Circle((0, 0), 0.5, color="white")

    # Áreas de penal (16.5m x 40.32m)
    left_box = patches.Rectangle((-52.5, -20.16), 16.5, 40.32, linewidth=1.6, edgecolor="white", facecolor="none")
    right_box = patches.Rectangle((36.0, -20.16), 16.5, 40.32, linewidth=1.6, edgecolor="white", facecolor="none")

    # Porterías (2m de fondo x 14.02m de ancho)
    left_goal = patches.Rectangle((-54.5, -7.01), 2.0, 14.02, linewidth=2.0, edgecolor="#ffeb3b", facecolor="#ffffff", alpha=0.35)
    right_goal = patches.Rectangle((52.5, -7.01), 2.0, 14.02, linewidth=2.0, edgecolor="#ffeb3b", facecolor="#ffffff", alpha=0.35)

    ax.add_patch(pitch_border)
    ax.add_line(half_line)
    ax.add_patch(center_circle)
    ax.add_patch(center_spot)
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Coordenada X (metros)", fontsize=10, color="white")
    ax.set_ylabel("Coordenada Y (metros)", fontsize=10, color="white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.grid(False)


def plot_learning_curves(
    rewards: List[float],
    lengths: List[int],
    success: Optional[List[bool]] = None,
    title: str = "Curvas de Aprendizaje - Q-Learning Tabular",
    window: int = 50,
    target_success: float = 90.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la evolución del retorno acumulado G_0, los pasos por episodio
    y la tasa de éxito temporal en ventana móvil de episodios.
    """
    n_plots = 3 if success is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.8), dpi=100)

    # 1. Retorno Acumulado G_0
    ax1 = axes[0]
    ax1.plot(rewards, color="#90caf9", alpha=0.3, label="G_0 por episodio")
    if len(rewards) >= window:
        ma_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(rewards)), ma_r, color="#1565c0", linewidth=2.2, label=f"Media móvil ({window} ep)")
    ax1.set_title("Retorno Acumulado $G_0$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episodio", fontsize=10)
    ax1.set_ylabel("Retorno", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Pasos por Episodio
    ax2 = axes[1]
    ax2.plot(lengths, color="#ffab91", alpha=0.3, label="Pasos por episodio")
    if len(lengths) >= window:
        ma_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(lengths)), ma_l, color="#d84315", linewidth=2.2, label=f"Media móvil ({window} ep)")
    ax2.axhline(40, color="#2e7d32", linestyle="--", linewidth=1.5, label="Umbral máx (40 pasos)")
    ax2.set_title("Pasos por Episodio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episodio", fontsize=10)
    ax2.set_ylabel("Número de pasos", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    # 3. Tasa de Éxito Temporal (%)
    if success is not None:
        ax3 = axes[2]
        succ_numeric = [100.0 if s else 0.0 for s in success]
        if len(succ_numeric) >= window:
            ma_s = np.convolve(succ_numeric, np.ones(window) / window, mode="valid")
            ax3.plot(range(window - 1, len(succ_numeric)), ma_s, color="#2e7d32", linewidth=2.5, label=f"Tasa éxito ({window} ep)")
        else:
            ax3.plot(succ_numeric, color="#2e7d32", linewidth=2.0)
        ax3.axhline(target_success, color="#c62828", linestyle="--", linewidth=1.8, label=f"Meta oficial ({target_success:.0f}%)")
        ax3.set_title("Tasa de Éxito Temporal", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Episodio", fontsize=10)
        ax3.set_ylabel("Éxito (%)", fontsize=10)
        ax3.set_ylim(-5, 105)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="lower right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_policy_heatmap(
    Q: Dict[Tuple[int, int], np.ndarray],
    n_dist: int = 4,
    n_angle: int = 5,
    action_names: Optional[List[str]] = None,
    title: str = "Matriz de Decisiones de la Política Óptima pi*(s)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera el mapa de calor de la función valor V*(s) = max_a Q(s, a)
    con las macro-acciones óptimas inscritas en cada celda discreta.
    """
    if action_names is None:
        action_names = ["DASH 100", "DASH 50", "TURN +35", "TURN -35"]

    dist_labels = ["Zona 0 (<0.8m)", "Zona 1 (0.8-3m)", "Zona 2 (3-8m)", "Zona 3 (>=8m)"]
    angle_labels = ["Frontal (<=15°)", "Der (-60 a -15°)", "Izq (15 a 60°)", "Post Der (<-60°)", "Post Izq (>60°)"]

    V_matrix = np.zeros((n_dist, n_angle))
    best_action_matrix = np.zeros((n_dist, n_angle), dtype=int)

    for d in range(n_dist):
        for a in range(n_angle):
            state = (d, a)
            if state in Q:
                V_matrix[d, a] = np.max(Q[state])
                best_action_matrix[d, a] = int(np.argmax(Q[state]))
            else:
                V_matrix[d, a] = 0.0
                best_action_matrix[d, a] = 0

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=110)
    im = ax.imshow(V_matrix, cmap="YlGnBu", aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Valor Óptimo $V^*(s)$", rotation=-90, va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(n_angle))
    ax.set_yticks(np.arange(n_dist))
    ax.set_xticklabels(angle_labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(dist_labels, fontsize=10, fontweight="bold")

    # Inscribir nombre de acción y valor numérico
    for i in range(n_dist):
        for j in range(n_angle):
            act_idx = best_action_matrix[i, j]
            act_str = action_names[act_idx]
            val = V_matrix[i, j]
            text_color = "white" if val < (np.max(V_matrix) + np.min(V_matrix)) / 2 else "black"
            ax.text(j, i, f"{act_str}\n(V={val:.1f})", ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sector Angular Relativo al Balón (theta_b)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Zona de Distancia Relativa (d_b)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_trajectory_comparison(
    untrained_data: Tuple[List[float], List[float], Tuple[float, float]],
    trained_data: Tuple[List[float], List[float], Tuple[float, float]],
    title_untrained: str = "Agente No Entrenado (Paseo Aleatorio)",
    title_trained: str = "Agente Entrenado con Q-Learning (Óptimo)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compara lado a lado en dos canchas 2D la trayectoria de un agente aleatorio
    versus la trayectoria óptima directa del agente entrenado con Q-Learning.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=110)

    # 1. Agente No Entrenado
    draw_soccer_pitch(ax1, title=title_untrained, xlim=(-32, 16), ylim=(-18, 18))
    u_px, u_py, u_ball = untrained_data
    ax1.plot(u_px, u_py, "-o", color="#ff9100", markersize=3.5, linewidth=1.8, alpha=0.85, label="Trayectoria Aleatoria")
    ax1.plot(u_px[0], u_py[0], "s", markersize=9, color="#29b6f6", label="Inicio Jugador")
    ax1.plot(u_px[-1], u_py[-1], "^", markersize=10, color="#d50000", label="Posición Final")
    ax1.plot(u_ball[0], u_ball[1], "o", markersize=10, color="white", markeredgecolor="black", markeredgewidth=1.8, label="Balón")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # 2. Agente Entrenado con Q-Learning
    draw_soccer_pitch(ax2, title=title_trained, xlim=(-32, 16), ylim=(-18, 18))
    t_px, t_py, t_ball = trained_data
    ax2.plot(t_px, t_py, "-o", color="#00e676", markersize=3.5, linewidth=2.2, alpha=0.9, label="Trayectoria Óptima")
    ax2.plot(t_px[0], t_py[0], "s", markersize=9, color="#29b6f6", label="Inicio Jugador")
    ax2.plot(t_px[-1], t_py[-1], "^", markersize=10, color="#1565c0", label="Balón Interceptado")
    ax2.plot(t_ball[0], t_ball[1], "o", markersize=10, color="white", markeredgecolor="black", markeredgewidth=1.8, label="Balón")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def animate_pursuit_step_by_step(
    env: Any,
    Q: Dict[Tuple[int, int], np.ndarray],
    max_steps: int = 40,
    delay: float = 0.05
) -> None:
    """
    Ejecuta y anima paso a paso la simulación del agente persiguiendo el balón
    utilizando la política greedy aprendida.
    """
    state = env.reset()
    action_names = ["DASH 100", "DASH 50", "GIRAR +35°", "GIRAR -35°"]

    for step in range(max_steps):
        best_a = int(np.argmax(Q[state]))
        next_state, reward, done, info = env.step(best_a)

        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
        draw_soccer_pitch(ax, title=f"Simulación en Vivo | Paso {step+1} | Acción: {action_names[best_a]} | Dist Balón: {info['dist']:.2f}m", xlim=(-32, 18), ylim=(-18, 18))

        # Trayectoria histórica
        ax.plot(env.trajectory_x, env.trajectory_y, "-o", color="#00e676", markersize=3.5, linewidth=1.8, alpha=0.7)
        # Posición actual
        ax.plot(env.player_x, env.player_y, "o", markersize=13, color="#1565c0", label="Jugador UTEC")

        # Flecha de orientación del cuerpo
        rad = math.radians(env.player_theta)
        ax.arrow(env.player_x, env.player_y, 1.8 * math.cos(rad), 1.8 * math.sin(rad),
                 head_width=1.0, head_length=0.8, fc="#ffeb3b", ec="#ffeb3b", label="Orientación")

        # Balón
        ax.plot(env.ball_x, env.ball_y, "o", markersize=11, color="white", markeredgecolor="black", markeredgewidth=2.0, label=f"Balón (d={info['dist']:.1f}m)")

        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        plt.tight_layout()
        display(fig)
        plt.close(fig)

        state = next_state
        time.sleep(delay)

        if done:
            status = "BALÓN INTERCEPTADO CON ÉXITO" if info.get("captured", True) else "EPISODIO FINALIZADO"
            print(f"Resultado: {status} en {step+1} pasos.")
            break
