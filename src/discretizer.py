"""
Discretización polar (distancia, ángulo) → estados tabulares.

Justificación de bins (ver informe P1):
- Distancia 0.8 m: radio de kickable area de rcssserver.
- 3 m / 8 m: zonas de reacción próxima vs. media vs. lejana.
- Ángulo ±15°: cono frontal para dash efectivo; ±60° giro parcial; resto giro completo.
"""

from typing import Tuple

N_DIST_BINS = 4
N_ANGLE_BINS = 5
N_STATES = N_DIST_BINS * N_ANGLE_BINS  # 20
N_ACTIONS = 4

ACTION_NAMES = (
    "DASH_FUERTE",  # potencia 100
    "DASH_SUAVE",   # potencia 50
    "GIRAR_IZQ",    # +35°
    "GIRAR_DER",    # -35°
)

DIST_LABELS = ("<0.8m", "0.8-3m", "3-8m", ">=8m")
ANGLE_LABELS = ("Frontal", "Der.fr.", "Izq.fr.", "Post.der.", "Post.izq.")


def discretize(dist: float, angle_rel: float) -> Tuple[int, int]:
    """Mapea observación continua a (d_bin, a_bin)."""
    if dist < 0.8:
        d_bin = 0
    elif dist < 3.0:
        d_bin = 1
    elif dist < 8.0:
        d_bin = 2
    else:
        d_bin = 3

    if abs(angle_rel) <= 15.0:
        a_bin = 0
    elif -60.0 <= angle_rel < -15.0:
        a_bin = 1
    elif 15.0 < angle_rel <= 60.0:
        a_bin = 2
    elif -180.0 <= angle_rel < -60.0:
        a_bin = 3
    else:
        a_bin = 4

    return (d_bin, a_bin)


def state_index(state: Tuple[int, int]) -> int:
    d_bin, a_bin = state
    return d_bin * N_ANGLE_BINS + a_bin
