"""
Módulo de Entornos Cinemáticos de Simulación para los 4 Escenarios de la Fase 1
RoboCup 2D Soccer Simulation League — DS5345 (UTEC 2026-II)

Catálogo de Tareas Acotadas:
1. Persecución e Intercepción de Balón (Ball Pursuit)
2. Conducción y Drible de Balón (Ball Dribbling)
3. Definición y Tiro a Puerta / Penales (Goal Shooting)
4. Cooperación 2 vs 1 y Pase (Passing - Possession)
"""

import math
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


# ==============================================================================
# 1. ESCENARIO 1: Persecución e Intercepción de Balón (Ball Pursuit)
# ==============================================================================

class BallPursuitEnv:
    """
    Escenario 1: Persecución e Intercepción de Balón (Ball Pursuit).
    El agente inicia en posición estocástica y debe orientarse y acelerar
    para interceptar el balón dentro de su radio de captura (d_b <= 0.8m)
    en el menor tiempo posible (< 40 pasos, tasa de éxito > 90%).
    """

    def __init__(self, max_steps: int = 60, seed: Optional[int] = None):
        self.max_steps = max_steps
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self) -> Tuple[int, int]:
        # El jugador inicia en su campo (-15m aprox) con orientación aleatoria
        self.player_x = -15.0 + np.random.uniform(-3.0, 3.0)
        self.player_y = np.random.uniform(-5.0, 5.0)
        self.player_theta = np.random.uniform(-180.0, 180.0)

        # Balón a distancia estocástica d_b in [5, 35] metros
        angle_ball = np.random.uniform(-math.pi, math.pi)
        dist_init = np.random.uniform(8.0, 28.0)
        self.ball_x = self.player_x + dist_init * math.cos(angle_ball)
        self.ball_y = self.player_y + dist_init * math.sin(angle_ball)

        # Delimitar dentro del campo (-52.5 a 52.5, -34 a 34)
        self.ball_x = np.clip(self.ball_x, -45.0, 45.0)
        self.ball_y = np.clip(self.ball_y, -28.0, 28.0)

        self.steps = 0
        self.captured = False
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]

        return self._get_state()

    def _get_obs(self) -> Tuple[float, float]:
        dx = self.ball_x - self.player_x
        dy = self.ball_y - self.player_y
        dist = math.hypot(dx, dy)
        angle_global = math.degrees(math.atan2(dy, dx))
        angle_rel = (angle_global - self.player_theta + 180.0) % 360.0 - 180.0
        return dist, angle_rel

    def _discretize(self, dist: float, angle_rel: float) -> Tuple[int, int]:
        # 4 zonas de distancia
        if dist < 0.8:
            d_bin = 0  # Captura / control
        elif dist < 3.0:
            d_bin = 1  # Próxima
        elif dist < 8.0:
            d_bin = 2  # Media
        else:
            d_bin = 3  # Lejana

        # 5 cuadrantes angulares
        if abs(angle_rel) <= 15.0:
            a_bin = 0  # Frontal
        elif -60.0 <= angle_rel < -15.0:
            a_bin = 1  # Derecha frontal
        elif 15.0 < angle_rel <= 60.0:
            a_bin = 2  # Izquierda frontal
        elif -180.0 <= angle_rel < -60.0:
            a_bin = 3  # Posterior derecha
        else:
            a_bin = 4  # Posterior izquierda

        return (d_bin, a_bin)

    def _get_state(self) -> Tuple[int, int]:
        dist, angle_rel = self._get_obs()
        return self._discretize(dist, angle_rel)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        dist_prev, _ = self._get_obs()

        # Acciones: 0: DASH 100, 1: DASH 50, 2: TURN +35, 3: TURN -35
        if action == 0:
            rad = math.radians(self.player_theta)
            self.player_x += 1.0 * math.cos(rad)
            self.player_y += 1.0 * math.sin(rad)
        elif action == 1:
            rad = math.radians(self.player_theta)
            self.player_x += 0.5 * math.cos(rad)
            self.player_y += 0.5 * math.sin(rad)
        elif action == 2:
            self.player_theta = (self.player_theta + 35.0 + 180.0) % 360.0 - 180.0
        elif action == 3:
            self.player_theta = (self.player_theta - 35.0 + 180.0) % 360.0 - 180.0

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)

        dist_curr, angle_curr = self._get_obs()
        done = False
        success = False

        # Recompensa según enunciado: Delta d_b - 0.2 + 100 * I_{d_b <= 0.8}
        delta_d = dist_prev - dist_curr
        reward = 3.0 * delta_d - 0.2

        if dist_curr <= 0.8:
            reward += 100.0
            done = True
            self.captured = True
            success = (self.steps <= 40)
        elif self.steps >= self.max_steps:
            reward -= 20.0
            done = True

        info = {
            "dist": dist_curr,
            "angle": angle_curr,
            "captured": self.captured,
            "success": success,
            "steps": self.steps
        }

        return self._discretize(dist_curr, angle_curr), reward, done, info


# ==============================================================================
# 2. ESCENARIO 2: Conducción y Drible de Balón (Ball Dribbling)
# ==============================================================================

class BallDribblingEnv:
    """
    Escenario 2: Conducción y Drible de Balón (Ball Dribbling).
    El agente en control del balón debe avanzar hacia la portería rival (x = 52.5)
    mediante micro-pateos (KICK 25) y carreras cortas (DASH 80) sin perder posesión ni salir de la cancha.
    Criterio de Éxito: Conducción continua durante más de 30 metros manteniendo el balón en > 80% de los episodios.
    """

    def __init__(self, max_steps: int = 70, target_dist: float = 30.0, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.target_dist = target_dist
        self.rival_goal_x = 52.5
        self.rival_goal_y = 0.0
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self) -> Tuple[int, int, int, int]:
        # Agente inicia en su propio campo con el balón controlado frente a él
        self.player_x = -20.0 + np.random.uniform(-3.0, 3.0)
        self.player_y = np.random.uniform(-4.0, 4.0)
        self.player_theta = np.random.uniform(-20.0, 20.0)  # Orientado en general hacia el frente

        # El balón inicia a 0.5m frente al jugador
        rad = math.radians(self.player_theta)
        self.ball_x = self.player_x + 0.5 * math.cos(rad)
        self.ball_y = self.player_y + 0.5 * math.sin(rad)
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.start_x = self.player_x
        self.distance_dribbled = 0.0
        self.steps = 0
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        self.ball_trajectory_x = [self.ball_x]
        self.ball_trajectory_y = [self.ball_y]

        return self._get_state()

    def _get_obs(self) -> Tuple[float, float, float, float]:
        # Vector al balón
        dx_b = self.ball_x - self.player_x
        dy_b = self.ball_y - self.player_y
        dist_b = math.hypot(dx_b, dy_b)
        ang_b_global = math.degrees(math.atan2(dy_b, dx_b))
        ang_b_rel = (ang_b_global - self.player_theta + 180.0) % 360.0 - 180.0

        # Vector a la portería rival (52.5, 0.0)
        dx_g = self.rival_goal_x - self.player_x
        dy_g = self.rival_goal_y - self.player_y
        dist_g = math.hypot(dx_g, dy_g)
        ang_g_global = math.degrees(math.atan2(dy_g, dx_g))
        ang_g_rel = (ang_g_global - self.player_theta + 180.0) % 360.0 - 180.0

        return dist_b, ang_b_rel, dist_g, ang_g_rel

    def _discretize(self, dist_b: float, ang_b: float, dist_g: float, ang_g: float) -> Tuple[int, int, int, int]:
        # 3 zonas de distancia al balón
        if dist_b < 0.8:
            db_bin = 0  # En control de pateo
        elif dist_b < 1.8:
            db_bin = 1  # Próximo (alcanzable)
        else:
            db_bin = 2  # En riesgo / perdido

        # 3 sectores angulares al balón
        if abs(ang_b) <= 20.0:
            ab_bin = 0  # Frontal
        elif ang_b < -20.0:
            ab_bin = 1  # Derecha
        else:
            ab_bin = 2  # Izquierda

        # 3 zonas de distancia a la portería rival
        if dist_g < 25.0:
            dg_bin = 0  # Cerca del arco rival
        elif dist_g < 45.0:
            dg_bin = 1  # Media cancha
        else:
            dg_bin = 2  # Propio campo

        # 3 sectores angulares a la portería rival
        if abs(ang_g) <= 20.0:
            ag_bin = 0  # Orientado directo al arco
        elif ang_g < -20.0:
            ag_bin = 1  # Arco a la derecha
        else:
            ag_bin = 2  # Arco a la izquierda

        return (db_bin, ab_bin, dg_bin, ag_bin)

    def _get_state(self) -> Tuple[int, int, int, int]:
        dist_b, ang_b, dist_g, ang_g = self._get_obs()
        return self._discretize(dist_b, ang_b, dist_g, ang_g)

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        dist_b_prev, _, dist_g_prev, _ = self._get_obs()
        x_prev = self.player_x

        # Ejecución de macro-acciones
        # 0: KICK 25 (micro-pateo en dirección hacia adelante)
        # 1: DASH 80 (carrera hacia adelante)
        # 2: TURN +30 (giro antihorario)
        # 3: TURN -30 (giro horario)
        if action == 0:
            if dist_b_prev <= 0.85:
                # Micro-pateo suave hacia el frente
                rad_p = math.radians(self.player_theta)
                kick_power = 1.6
                self.ball_vx = kick_power * math.cos(rad_p)
                self.ball_vy = kick_power * math.sin(rad_p)
        elif action == 1:
            rad = math.radians(self.player_theta)
            self.player_x += 0.85 * math.cos(rad)
            self.player_y += 0.85 * math.sin(rad)
        elif action == 2:
            self.player_theta = (self.player_theta + 30.0 + 180.0) % 360.0 - 180.0
        elif action == 3:
            self.player_theta = (self.player_theta - 30.0 + 180.0) % 360.0 - 180.0

        # Dinámica del balón con rozamiento
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.ball_vx *= 0.75
        self.ball_vy *= 0.75

        self.trajectory_x.append(self.player_x)
        self.trajectory_y.append(self.player_y)
        self.ball_trajectory_x.append(self.ball_x)
        self.ball_trajectory_y.append(self.ball_y)

        dist_b_curr, ang_b_curr, dist_g_curr, ang_g_curr = self._get_obs()
        net_advance = self.player_x - x_prev
        self.distance_dribbled = max(0.0, self.player_x - self.start_x)

        done = False
        success = False
        reward = 2.0 * net_advance - 0.2  # Recompensa proporcional al avance neto

        # Penalización por perder balón o salirse de la cancha (105m x 68m)
        if dist_b_curr > 2.5:
            reward -= 30.0
            done = True
        elif abs(self.player_y) > 34.0 or self.player_x > 52.5 or self.player_x < -52.5:
            reward -= 40.0
            done = True
        elif self.distance_dribbled >= self.target_dist and dist_b_curr <= 1.2:
            reward += 100.0
            done = True
            success = True
        elif self.steps >= self.max_steps:
            done = True

        info = {
            "distance_dribbled": self.distance_dribbled,
            "dist_ball": dist_b_curr,
            "success": success,
            "steps": self.steps
        }

        return self._discretize(dist_b_curr, ang_b_curr, dist_g_curr, ang_g_curr), reward, done, info


# ==============================================================================
# 3. ESCENARIO 3: Definición y Tiro a Puerta / Penales (Goal Shooting)
# ==============================================================================

class GoalShootingEnv:
    """
    Escenario 3: Definición y Tiro a Puerta / Penales (Goal Shooting).
    El agente frente al arco rival (con o sin portero) debe aprender el perfil
    de disparo óptimo (potencia y dirección) hacia los ángulos para anotar gol.
    Criterio de Éxito: Tasa de conversión de gol > 75% en arco abierto y > 50% con portero activo.
    """

    def __init__(self, with_goalkeeper: bool = True, max_steps: int = 15, seed: Optional[int] = None):
        self.with_goalkeeper = with_goalkeeper
        self.max_steps = max_steps
        self.goal_x = 52.5
        self.post_left_y = 7.01
        self.post_right_y = -7.01
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self) -> Tuple[int, int, int]:
        # Jugador en el área o bordes del área rival
        self.player_x = 36.0 + np.random.uniform(0.0, 7.0)
        self.player_y = np.random.uniform(-10.0, 10.0)
        self.player_theta = 0.0  # mirando hacia el arco rival

        # Balón a sus pies
        self.ball_x = self.player_x + 0.4
        self.ball_y = self.player_y

        # Portero en la línea de gol o cerca (x = 50.0m a 52.0m)
        if self.with_goalkeeper:
            self.gk_x = 50.5 + np.random.uniform(-0.5, 0.5)
            self.gk_y = np.random.uniform(-3.5, 3.5)
        else:
            self.gk_x = 52.5
            self.gk_y = 99.0  # Sin portero

        self.steps = 0
        self.trajectory_x = [self.player_x]
        self.trajectory_y = [self.player_y]
        return self._get_state()

    def _get_obs(self) -> Tuple[float, float, float, float]:
        dx_g = self.goal_x - self.player_x
        dy_g = 0.0 - self.player_y
        dist_g = math.hypot(dx_g, dy_g)

        # Ángulos a los postes
        ang_left = math.degrees(math.atan2(self.post_left_y - self.player_y, self.goal_x - self.player_x))
        ang_right = math.degrees(math.atan2(self.post_right_y - self.player_y, self.goal_x - self.player_x))

        # Posición lateral del portero
        gk_pos_y = self.gk_y if self.with_goalkeeper else 0.0
        return dist_g, ang_left, ang_right, gk_pos_y

    def _discretize(self, dist_g: float, ang_l: float, ang_r: float, gk_y: float) -> Tuple[int, int, int]:
        # 3 zonas de distancia al arco
        if dist_g < 13.0:
            d_bin = 0  # Muy cerca
        elif dist_g < 17.0:
            d_bin = 1  # Media
        else:
            d_bin = 2  # Lejana

        # 3 cuadrantes de ángulo de tiro del jugador (centro, perfil izq, perfil der)
        if abs(self.player_y) <= 4.0:
            angle_bin = 0  # Centrado
        elif self.player_y > 4.0:
            angle_bin = 1  # Perfil izquierdo
        else:
            angle_bin = 2  # Perfil derecho

        # 4 estados del portero (cubriendo izq, centro, der, o sin portero)
        if not self.with_goalkeeper:
            gk_bin = 0  # Arco abierto
        else:
            if gk_y > 1.8:
                gk_bin = 1  # Portero cubriendo palo izquierdo
            elif gk_y < -1.8:
                gk_bin = 2  # Portero cubriendo palo derecho
            else:
                gk_bin = 3  # Portero centrado

        return (d_bin, angle_bin, gk_bin)

    def _get_state(self) -> Tuple[int, int, int]:
        dist_g, ang_l, ang_r, gk_y = self._get_obs()
        return self._discretize(dist_g, ang_l, ang_r, gk_y)

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        done = True  # El tiro es decisivo en pocos pasos
        goal = False
        blocked = False
        out = False

        # Acciones:
        # 0: KICK_POSTE_IZQ (dirigido a y = 5.5m con potencia 100)
        # 1: KICK_POSTE_DER (dirigido a y = -5.5m con potencia 100)
        # 2: KICK_CENTRO (dirigido a y = 0.0m con potencia 100)
        # 3: KICK_COLOCADO (dirigido a poste desguarnecido con potencia 75)
        target_y = 0.0
        if action == 0:
            target_y = 5.5
        elif action == 1:
            target_y = -5.5
        elif action == 2:
            target_y = 0.0
        elif action == 3:
            # Apunta automáticamente lejos del arquero
            target_y = -5.5 if (self.with_goalkeeper and self.gk_y > 0) else 5.5

        # Añadir pequeña dispersión estocástica al disparo (+/- 0.6m)
        shot_y = target_y + np.random.uniform(-0.6, 0.6)

        # Movimiento reactivo del portero hacia el tiro con probabilidad 0.8
        if self.with_goalkeeper:
            gk_reaction = (shot_y - self.gk_y) * np.random.uniform(0.5, 0.85)
            final_gk_y = np.clip(self.gk_y + gk_reaction, -6.0, 6.0)
        else:
            final_gk_y = 99.0

        # Verificación de gol
        if abs(shot_y) <= 7.0:
            # Dentro de los postes
            if self.with_goalkeeper and abs(shot_y - final_gk_y) < 1.4:
                # Arquero bloquea el disparo
                blocked = True
                reward = -50.0
            else:
                # GOL ANOTADO
                goal = True
                reward = 100.0
        else:
            # Disparo desviado fuera de los postes
            out = True
            reward = -30.0

        info = {
            "goal": goal,
            "blocked": blocked,
            "out": out,
            "success": goal,
            "target_y": target_y,
            "shot_y": shot_y,
            "steps": self.steps
        }

        dist_g, ang_l, ang_r, gk_y = self._get_obs()
        return self._discretize(dist_g, ang_l, ang_r, gk_y), reward, done, info


# ==============================================================================
# 4. ESCENARIO 4: Cooperación 2 vs 1 y Pase (Passing - Possession)
# ==============================================================================

class PassingPossessionEnv:
    """
    Escenario 4: Cooperación 2 vs 1 y Pase (Passing - Possession).
    Dos atacantes cooperativos (P1 con balón, P2 libre) deben mantener
    la posesión frente a un defensor rival (D), decidiendo cuándo conducir y cuándo pasar.
    Criterio de Éxito: Mantenimiento de posesión continuada > 50 pasos y al menos 3 pases efectivos.
    """

    def __init__(self, max_steps: int = 70, seed: Optional[int] = None):
        self.max_steps = max_steps
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()

    def reset(self) -> Tuple[int, int, int, int]:
        # P1 (atacante con balón)
        self.p1_x = 0.0 + np.random.uniform(-2.0, 2.0)
        self.p1_y = 5.0 + np.random.uniform(-2.0, 2.0)

        # P2 (compañero libre desmarcándose al otro lado)
        self.p2_x = 10.0 + np.random.uniform(-2.0, 2.0)
        self.p2_y = -8.0 + np.random.uniform(-2.0, 2.0)

        # Balón en posesión de P1
        self.ball_x = self.p1_x
        self.ball_y = self.p1_y
        self.possession_holder = 1  # 1: P1, 2: P2

        # Defensor rival D ubicado entre P1 y P2
        self.def_x = 5.0 + np.random.uniform(-1.0, 1.0)
        self.def_y = 0.0 + np.random.uniform(-1.0, 1.0)

        self.steps = 0
        self.passes_completed = 0
        self.intercepted = False
        self.out_of_bounds = False

        self.p1_traj_x = [self.p1_x]
        self.p1_traj_y = [self.p1_y]
        self.p2_traj_x = [self.p2_x]
        self.p2_traj_y = [self.p2_y]

        return self._get_state()

    def _get_obs(self) -> Tuple[float, float, float, float, int]:
        curr_holder_x = self.p1_x if self.possession_holder == 1 else self.p2_x
        curr_holder_y = self.p1_y if self.possession_holder == 1 else self.p2_y
        teammate_x = self.p2_x if self.possession_holder == 1 else self.p1_x
        teammate_y = self.p2_y if self.possession_holder == 1 else self.p1_y

        # Vector al compañero
        dx_comp = teammate_x - curr_holder_x
        dy_comp = teammate_y - curr_holder_y
        dist_comp = math.hypot(dx_comp, dy_comp)
        ang_comp = math.degrees(math.atan2(dy_comp, dx_comp))

        # Vector al defensor
        dx_def = self.def_x - curr_holder_x
        dy_def = self.def_y - curr_holder_y
        dist_def = math.hypot(dx_def, dy_def)
        ang_def = math.degrees(math.atan2(dy_def, dx_def))

        return dist_comp, ang_comp, dist_def, ang_def, self.possession_holder

    def _discretize(self, dist_comp: float, ang_comp: float, dist_def: float, ang_def: float, holder: int) -> Tuple[int, int, int, int]:
        # 3 zonas de distancia al compañero
        if dist_comp < 10.0:
            c_bin = 0  # Cercano
        elif dist_comp < 18.0:
            c_bin = 1  # Distancia óptima de pase
        else:
            c_bin = 2  # Muy lejano

        # 3 zonas de presión del defensor
        if dist_def < 2.5:
            d_bin = 0  # Presión alta (riesgo inminente)
        elif dist_def < 5.5:
            d_bin = 1  # Presión media
        else:
            d_bin = 2  # Libre

        # 3 estados del cono de pase (bloqueado por defensor vs línea abierta)
        angle_diff = abs((ang_comp - ang_def + 180.0) % 360.0 - 180.0)
        if angle_diff < 22.0 and dist_def < dist_comp:
            line_bin = 0  # Línea de pase bloqueada por el defensor
        elif angle_diff < 45.0 and dist_def < dist_comp:
            line_bin = 1  # Línea disputada
        else:
            line_bin = 2  # Línea de pase completamente abierta

        h_bin = 0 if holder == 1 else 1
        return (c_bin, d_bin, line_bin, h_bin)

    def _get_state(self) -> Tuple[int, int, int, int]:
        dist_comp, ang_comp, dist_def, ang_def, holder = self._get_obs()
        return self._discretize(dist_comp, ang_comp, dist_def, ang_def, holder)

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        done = False
        reward = 0.5  # Recompensa constante por mantener posesión viva

        dist_comp, ang_comp, dist_def, ang_def, holder = self._get_obs()
        angle_diff = abs((ang_comp - ang_def + 180.0) % 360.0 - 180.0)

        # Acciones:
        # 0: PASE (dar un pase al compañero desmarcado)
        # 1: DRIBLE (conducción corta protegiendo el balón)
        # 2: GIRAR (orientarse para mejorar ángulo de pase)
        # 3: DESPEJE (salvar el balón a zona libre)
        if action == 0:
            # Intento de pase
            if angle_diff < 20.0 and dist_def < dist_comp and dist_def < 4.0:
                # Interceptado por el defensor
                self.intercepted = True
                reward = -30.0
                done = True
            else:
                # Pase exitoso al compañero
                self.passes_completed += 1
                self.possession_holder = 2 if self.possession_holder == 1 else 1
                reward += 30.0
                # El receptor recibe el balón
                if self.possession_holder == 1:
                    self.ball_x, self.ball_y = self.p1_x, self.p1_y
                else:
                    self.ball_x, self.ball_y = self.p2_x, self.p2_y
        elif action == 1:
            # Drible: alejarse del defensor
            if holder == 1:
                self.p1_x += 0.8
                self.p1_y += 0.5 if self.p1_y > self.def_y else -0.5
                self.ball_x, self.ball_y = self.p1_x, self.p1_y
            else:
                self.p2_x += 0.8
                self.p2_y += 0.5 if self.p2_y > self.def_y else -0.5
                self.ball_x, self.ball_y = self.p2_x, self.p2_y
            reward += 1.5
        elif action == 2:
            # Maniobra de cambio de ángulo
            reward += 0.2
        elif action == 3:
            # Despeje largo de seguridad
            reward += 2.0

        # Movimiento del defensor (presiona al poseedor del balón)
        target_holder_x = self.p1_x if self.possession_holder == 1 else self.p2_x
        target_holder_y = self.p1_y if self.possession_holder == 1 else self.p2_y
        dx_d = target_holder_x - self.def_x
        dy_d = target_holder_y - self.def_y
        dist_d = math.hypot(dx_d, dy_d)
        if dist_d > 0.1:
            self.def_x += 0.55 * (dx_d / dist_d)
            self.def_y += 0.55 * (dy_d / dist_d)

        # Si el defensor le quita el balón al jugador en control
        if dist_d < 0.9:
            self.intercepted = True
            reward -= 30.0
            done = True

        # Desmarque continuo del compañero sin balón
        if self.possession_holder == 1:
            self.p2_x += 0.4 * np.random.uniform(0.5, 1.2)
            self.p2_y += 0.3 * (-1.0 if self.def_y > 0 else 1.0)
        else:
            self.p1_x += 0.4 * np.random.uniform(0.5, 1.2)
            self.p1_y += 0.3 * (-1.0 if self.def_y > 0 else 1.0)

        # Balón fuera de límites (-52.5 a 52.5, -34 a 34)
        if abs(self.ball_y) > 30.0 or abs(self.ball_x) > 50.0:
            self.out_of_bounds = True
            reward -= 10.0
            done = True

        success = (self.steps >= 50 and self.passes_completed >= 3 and not self.intercepted)
        if self.steps >= self.max_steps:
            done = True

        self.p1_traj_x.append(self.p1_x)
        self.p1_traj_y.append(self.p1_y)
        self.p2_traj_x.append(self.p2_x)
        self.p2_traj_y.append(self.p2_y)

        info = {
            "passes_completed": self.passes_completed,
            "intercepted": self.intercepted,
            "success": success,
            "steps": self.steps
        }

        dist_comp, ang_comp, dist_def, ang_def, holder = self._get_obs()
        return self._discretize(dist_comp, ang_comp, dist_def, ang_def, holder), reward, done, info
