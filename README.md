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
├── src/                     # Código modular (env, discretizer, agentes, cliente)
│   ├── agents/              # MC Control + Q-Learning
│   ├── env.py
│   ├── discretizer.py
│   ├── metrics.py
│   ├── plotting.py
│   └── robocup_client.py
├── notebooks/
│   ├── p1_baseline_tabular.ipynb          # Entrega P1 (experimentos)
│   ├── tutorial_primer_paso.ipynb
│   └── agente_cero_mc_control_persecucion.ipynb  # Baseline original
├── scripts/
│   ├── run_p1_experiments.py
│   └── test_random_agent.py
├── figures/                 # Curvas y métricas exportadas
├── plantilla_informe_latex/ # Informe LaTeX P1
├── docker-compose.yml       # Orquestación (usar desde la raíz)
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

Experimentos P1 (sin Jupyter):

```bash
docker compose exec rl-agent python /workspace/scripts/run_p1_experiments.py
```

Cuaderno principal: `notebooks/p1_baseline_tabular.ipynb`.

---

## Fase 1 — Qué implementa este repo

| Requisito P1 | Estado |
| :--- | :--- |
| MDP formal ⟨S,A,P,R,γ⟩ + discretización justificada | Sí (`src/`, informe) |
| Monte Carlo Control First-Visit | Sí |
| Segundo algoritmo tabular (Q-Learning) | Sí |
| Comparación ε fijo vs ε decreciente | Sí (4 configs) |
| Curvas G₀, tasa de éxito, pasos, V*/π* | Sí |
| Cliente UDP + Docker | Sí |
| Informe LaTeX | `plantilla_informe_latex/` → compilar a `informe_p1.pdf` |

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
