"""
Módulo de Aprendizaje por Refuerzo Tabular: Q-Learning y Algoritmos Basales
Curso: DS5345 - Aprendizaje por Refuerzo (UTEC 2026-II)

Implementa:
- Q-Learning Tabular (Off-Policy TD Control)
- Políticas de exploración: epsilon constante vs. epsilon decreciente geométrico
- Monte Carlo Control On-Policy (First-Visit) para comparación y ablación
- Funciones de evaluación de políticas y registro estadístico detallado
"""

import math
import time
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Callable, Optional
import numpy as np


class QLearningAgent:
    """
    Agente de Aprendizaje por Refuerzo Tabular basado en Q-Learning (Watkins, 1989).
    Aprende de manera off-policy la función óptima de acción-valor Q*(s, a)
    utilizando la ecuación de optimalidad de Bellman:
        Q(S, A) <- Q(S, A) + alpha * [ R + gamma * max_a' Q(S', a') - Q(S, A) ]
    """

    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.15,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.998,
        exploration_strategy: str = "decay",
        constant_eps: float = 0.10,
        seed: Optional[int] = 42
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.exploration_strategy = exploration_strategy
        self.constant_eps = constant_eps
        
        if seed is not None:
            np.random.seed(seed)

        # Tabla Q(s, a): diccionario con vector de acciones por estado
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

        # Registros de entrenamiento
        self.history_rewards: List[float] = []
        self.history_lengths: List[int] = []
        self.history_success: List[bool] = []
        self.history_td_errors: List[float] = []
        self.history_epsilons: List[float] = []

    def get_epsilon(self, episode_idx: int) -> float:
        """Calcula el valor de epsilon para el episodio dado según la estrategia."""
        if self.exploration_strategy == "constant":
            return self.constant_eps
        elif self.exploration_strategy == "decay":
            return max(self.eps_min, self.eps_start * (self.eps_decay ** episode_idx))
        else:
            raise ValueError(f"Estrategia de exploración desconocida: {self.exploration_strategy}")

    def select_action(self, state: Any, epsilon: float = 0.0) -> int:
        """
        Selecciona una acción siguiendo la política epsilon-greedy.
        Rompe empates de manera aleatoria uniforme para asegurar exploración adecuada.
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_actions))

        q_values = self.Q[state]
        max_val = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_val)
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """
        Aplica la regla de actualización off-policy de Q-Learning:
            delta = R + gamma * max_a' Q(S', a') - Q(S, A)
            Q(S, A) <- Q(S, A) + alpha * delta
        Retorna el error temporal (TD error).
        """
        best_next_q = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return float(td_error)

    def train(
        self,
        env: Any,
        n_episodes: int = 3500,
        success_fn: Optional[Callable[[int, float, Dict[str, Any]], bool]] = None,
        verbose_every: int = 500,
        render_callback: Optional[Callable[[int, Any, List[float], List[int]], None]] = None,
        render_every: int = 100
    ) -> Dict[str, Any]:
        """
        Entrena el agente en el entorno provisto durante n_episodes.
        Retorna un diccionario con estadísticas completas del aprendizaje.
        """
        self.history_rewards = []
        self.history_lengths = []
        self.history_success = []
        self.history_td_errors = []
        self.history_epsilons = []

        t_start = time.time()

        for ep in range(n_episodes):
            eps = self.get_epsilon(ep)
            self.history_epsilons.append(eps)
            state = env.reset()
            episode_reward = 0.0
            steps = 0
            ep_td_errors = []

            while True:
                action = self.select_action(state, epsilon=eps)
                next_state, reward, done, info = env.step(action)

                td_err = self.update(state, action, reward, next_state, done)
                ep_td_errors.append(abs(td_err))

                episode_reward += reward
                steps += 1
                state = next_state

                if done:
                    # Evaluación de criterio de éxito
                    if success_fn is not None:
                        is_success = success_fn(steps, reward, info)
                    else:
                        is_success = bool(info.get("success", reward >= 50.0))

                    self.history_success.append(is_success)
                    self.history_rewards.append(episode_reward)
                    self.history_lengths.append(steps)
                    if ep_td_errors:
                        self.history_td_errors.append(float(np.mean(ep_td_errors)))
                    break

            if render_callback is not None and ((ep + 1) % render_every == 0 or ep == 0):
                render_callback(ep, env, self.history_rewards, self.history_lengths)

            if verbose_every > 0 and (ep + 1) % verbose_every == 0:
                win = min(100, len(self.history_rewards))
                mean_r = np.mean(self.history_rewards[-win:])
                mean_l = np.mean(self.history_lengths[-win:])
                succ_rate = np.mean(self.history_success[-win:]) * 100.0
                print(
                    f"Episodio {ep+1:5d}/{n_episodes} | "
                    f"Retorno Medio (u100): {mean_r:7.2f} | "
                    f"Pasos: {mean_l:5.1f} | "
                    f"Tasa Éxito: {succ_rate:5.1f}% | "
                    f"Epsilon: {eps:.4f}"
                )

        duration = time.time() - t_start
        win = min(100, len(self.history_rewards))
        final_succ = np.mean(self.history_success[-win:]) * 100.0
        final_steps = np.mean(self.history_lengths[-win:])
        final_ret = np.mean(self.history_rewards[-win:])

        return {
            "duration_sec": duration,
            "rewards": self.history_rewards,
            "lengths": self.history_lengths,
            "success": self.history_success,
            "epsilons": self.history_epsilons,
            "td_errors": self.history_td_errors,
            "final_success_rate": final_succ,
            "final_mean_steps": final_steps,
            "final_mean_reward": final_ret,
            "q_table": self.Q
        }

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 100,
        success_fn: Optional[Callable[[int, float, Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Evalúa el desempeño de la política aprendida en modo determinista (greedy, epsilon=0).
        """
        eval_rewards = []
        eval_lengths = []
        eval_success = []

        for _ in range(n_episodes):
            state = env.reset()
            ep_ret = 0.0
            steps = 0
            while True:
                action = self.select_action(state, epsilon=0.0)
                next_state, reward, done, info = env.step(action)
                ep_ret += reward
                steps += 1
                state = next_state
                if done:
                    if success_fn is not None:
                        is_success = success_fn(steps, reward, info)
                    else:
                        is_success = bool(info.get("success", reward >= 50.0))
                    eval_success.append(is_success)
                    eval_rewards.append(ep_ret)
                    eval_lengths.append(steps)
                    break

        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_steps": float(np.mean(eval_lengths)),
            "std_steps": float(np.std(eval_lengths)),
            "success_rate": float(np.mean(eval_success)) * 100.0,
            "n_episodes": n_episodes
        }

    def get_optimal_policy_and_value(self) -> Tuple[Dict[Any, int], Dict[Any, float]]:
        """
        Extrae la política óptima pi*(s) = argmax_a Q(s, a)
        y la función de valor óptima V*(s) = max_a Q(s, a).
        """
        policy = {}
        V = {}
        for state, q_vals in self.Q.items():
            policy[state] = int(np.argmax(q_vals))
            V[state] = float(np.max(q_vals))
        return policy, V


def train_mc_control(
    env: Any,
    n_episodes: int = 3500,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.998,
    n_actions: int = 4,
    success_fn: Optional[Callable[[int, float, Dict[str, Any]], bool]] = None,
    seed: Optional[int] = 42
) -> Tuple[Dict[Any, np.ndarray], Dict[str, Any]]:
    """
    Ejecuta el entrenamiento del baseline clásico: Monte Carlo Control On-Policy (First-Visit).
    Permite la comparación empírica y análisis de ablación contra Q-Learning.
    """
    if seed is not None:
        np.random.seed(seed)

    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    returns_sum = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
    returns_count = defaultdict(lambda: np.zeros(n_actions, dtype=np.int32))

    history_rewards = []
    history_lengths = []
    history_success = []
    history_eps = []

    t_start = time.time()

    for ep in range(n_episodes):
        eps = max(eps_min, eps_start * (eps_decay ** ep))
        history_eps.append(eps)
        state = env.reset()
        episode = []

        # 1. Generar trayectoria con política epsilon-greedy
        while True:
            if np.random.random() < eps:
                action = int(np.random.randint(n_actions))
            else:
                q_vals = Q[state]
                max_val = np.max(q_vals)
                best_actions = np.flatnonzero(q_vals == max_val)
                action = int(np.random.choice(best_actions))

            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

            if done:
                if success_fn is not None:
                    is_success = success_fn(len(episode), reward, info)
                else:
                    is_success = bool(info.get("success", reward >= 50.0))
                history_success.append(is_success)
                break

        # 2. Actualización retrospectiva First-Visit
        G = 0.0
        visited = set()
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]

        history_rewards.append(sum(r for _, _, r in episode))
        history_lengths.append(len(episode))

    duration = time.time() - t_start
    win = min(100, len(history_rewards))

    stats = {
        "duration_sec": duration,
        "rewards": history_rewards,
        "lengths": history_lengths,
        "success": history_success,
        "epsilons": history_eps,
        "final_success_rate": float(np.mean(history_success[-win:])) * 100.0,
        "final_mean_steps": float(np.mean(history_lengths[-win:])),
        "final_mean_reward": float(np.mean(history_rewards[-win:]))
    }

    return Q, stats


def compare_exploration_strategies(
    env_class: Any,
    n_episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    success_fn: Optional[Callable[[int, float, Dict[str, Any]], bool]] = None,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta el experimento comparativo requerido por la rúbrica de evaluación:
    1. Q-Learning con epsilon decreciente geométrico (eps_0=1.0, eps_min=0.05, decay=0.998)
    2. Q-Learning con epsilon constante (eps=0.10)
    3. Monte Carlo Control On-Policy (First-Visit) como baseline
    """
    print("=" * 75)
    print("INICIANDO COMPARACIÓN EMPÍRICA DE POLÍTICAS DE EXPLORACIÓN Y BASELINES")
    print("=" * 75)

    # 1. Q-Learning con Epsilon Decreciente
    print("\n[1/3] Entrenando Q-Learning (Epsilon Decreciente Geométrico)...")
    env1 = env_class()
    agent_decay = QLearningAgent(
        n_actions=4,
        alpha=alpha,
        gamma=gamma,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.998,
        exploration_strategy="decay",
        seed=seed
    )
    stats_decay = agent_decay.train(env1, n_episodes=n_episodes, success_fn=success_fn, verbose_every=1000)

    # 2. Q-Learning con Epsilon Constante
    print("\n[2/3] Entrenando Q-Learning (Epsilon Constante eps=0.10)...")
    env2 = env_class()
    agent_const = QLearningAgent(
        n_actions=4,
        alpha=alpha,
        gamma=gamma,
        constant_eps=0.10,
        exploration_strategy="constant",
        seed=seed
    )
    stats_const = agent_const.train(env2, n_episodes=n_episodes, success_fn=success_fn, verbose_every=1000)

    # 3. Monte Carlo Control (Baseline)
    print("\n[3/3] Entrenando Monte Carlo Control On-Policy (First-Visit)...")
    env3 = env_class()
    _, stats_mc = train_mc_control(
        env3,
        n_episodes=n_episodes,
        gamma=gamma,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.998,
        success_fn=success_fn,
        seed=seed
    )

    print("\n" + "=" * 75)
    print("RESUMEN DE COMPARACIÓN (ÚLTIMOS 100 EPISODIOS):")
    print("-" * 75)
    print(f"| {'Algoritmo':<32} | {'Retorno':<9} | {'Pasos':<8} | {'Éxito (%)':<10} | {'Tiempo (s)':<10} |")
    print("-" * 75)
    print(f"| {'Q-Learning (eps decreciente)':<32} | {stats_decay['final_mean_reward']:9.2f} | {stats_decay['final_mean_steps']:8.1f} | {stats_decay['final_success_rate']:9.1f}% | {stats_decay['duration_sec']:10.2f} |")
    print(f"| {'Q-Learning (eps constante)':<32} | {stats_const['final_mean_reward']:9.2f} | {stats_const['final_mean_steps']:8.1f} | {stats_const['final_success_rate']:9.1f}% | {stats_const['duration_sec']:10.2f} |")
    print(f"| {'Monte Carlo Control (First-Visit)':<32} | {stats_mc['final_mean_reward']:9.2f} | {stats_mc['final_mean_steps']:8.1f} | {stats_mc['final_success_rate']:9.1f}% | {stats_mc['duration_sec']:10.2f} |")
    print("=" * 75)

    return {
        "q_learning_decay": {
            "agent": agent_decay,
            "stats": stats_decay
        },
        "q_learning_constant": {
            "agent": agent_const,
            "stats": stats_const
        },
        "monte_carlo": {
            "stats": stats_mc
        }
    }
