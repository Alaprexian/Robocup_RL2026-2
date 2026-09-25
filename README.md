# Starter Kit + Entrega P1 — Proyecto de Aprendizaje por Refuerzo (DS5345)

**Curso:** DS5345 · Aprendizaje por Refuerzo (2026-II)  
**Institución:** UTEC  
**Docente:** Percy W. Lovon Ramos  
**Dominio:** Fútbol autónomo multi-agente (RoboCup 2D Soccer Simulation)

---

## Estructura del repositorio

```text
.
├── docker/                  # Dockerfile (+ compose de referencia)
├── src/                     # Código modular
│   ├── agents/              # MC Control + Q-Learning (agnósticos al env)
│   ├── envs/                # 4 tareas P1 (pursuit, dribbling, shooting, passing)
│   ├── env.py               # Reexport BallPursuitSimEnv
│   ├── discretizer.py
│   ├── metrics.py
│   ├── plotting.py
│   └── robocup_client.py
├── notebooks/
│   ├── p1_four_tasks.ipynb               # Catálogo completo (4 tareas)
│   ├── p1_baseline_tabular.ipynb         # Baseline pursuit
│   ├── tutorial_primer_paso.ipynb
│   └── agente_cero_mc_control_persecucion.ipynb
├── scripts/
│   ├── run_all_tasks.py     # Entrena las 4 tareas + exporta figuras
│   ├── run_p1_experiments.py
│   └── test_random_agent.py
├── figures/                 # Curvas globales + figures/tasks/<tarea>/
├── plantilla_informe_latex/ # Informe LaTeX P1
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Inicio rápido (Docker)

```bash
docker compose up -d
```

Abrir Jupyter Lab: **http://localhost:8888**

Smoke test de conexión:

```bash
docker compose exec rl-agent python /workspace/scripts/test_random_agent.py
```

Experimentos P1 — las 4 tareas:

```bash
docker compose exec rl-agent python /workspace/scripts/run_all_tasks.py
```

Solo persecución (legacy):

```bash
docker compose exec rl-agent python /workspace/scripts/run_p1_experiments.py
```

Cuaderno principal del catálogo: `notebooks/p1_four_tasks.ipynb`.

---

## Fase 1 — Qué implementa este repo

| Requisito P1 | Estado |
| :--- | :--- |
| MDP formal ⟨S,A,P,R,γ⟩ + discretización | Sí (4 tareas en `src/envs/`) |
| Monte Carlo Control First-Visit | Sí |
| Segundo algoritmo tabular (Q-Learning) | Sí |
| Comparación ε fijo vs ε decreciente | Sí (4 configs × tarea) |
| Curvas G₀, tasa de éxito, pasos, trayectorias | Sí (`figures/tasks/`) |
| Cliente UDP + Docker | Sí |
| Informe LaTeX | `plantilla_informe_latex/` |

### Tareas del catálogo

1. **Pursuit** — intercepción con \(d_b \in [5,40]\) m  
2. **Dribbling** — avance >30 m con posesión  
3. **Shooting** — tiro con/sin portero (eval separada)  
4. **Passing** — 2v1, ≥3 pases y posesión ≥50 pasos  

---

## Fase 2 (futuro)

Deep RL (DQN/PPO), reward shaping avanzado y vía A (partido) o B (transferencia).

---

## Comandos útiles

```bash
docker compose ps
docker compose logs -f rcssserver
docker compose restart rcssserver
docker compose down
```
