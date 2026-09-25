# Guía de diseño e implementación — RoboCup RL 2026-2 (P1)

Este documento no es el informe del curso. Es una guía interna para entender **qué hace el repo, por qué está armado así y cómo se implementó**. Si llegas frío al código, empieza por aquí.

> **Nota tipográfica:** este archivo **no usa** `$...$` / `$$...$$`. La vista previa de Markdown de Cursor a menudo no renderiza MathJax; todo va en Unicode o bloques `text`.

---

## 1. Qué problema estamos resolviendo

El dominio del proyecto es **RoboCup 2D Soccer Simulation**: un partido multi-agente ruidoso (POMDP) sobre el servidor `rcssserver`. Aprender “jugar fútbol completo” de cero con métodos tabulares no es viable: el espacio de observación/acción explota.

La Fase 1 (P1) del curso pide otra cosa: **aislar habilidades** del catálogo (persecución, drible, tiro, pase 2v1), formalizarlas como MDPs, entrenar baselines tabulares y mostrar que cumplen umbrales de éxito.

Nuestra decisión de diseño central:

> Entrenamos sobre **proxies cinemáticos** en Python (`src/envs/`), no sobre el servidor completo en cada update de Q.

El servidor sigue existiendo (Docker + `robocup_client.py`) para smoke tests y como destino de despliegue futuro. El proxy no modela visión ruidosa ni estamina; las políticas hay que revalidarlas en `rcssserver` si se pasa a Fase 2.

---

## 2. Mapa del repositorio

```text
.
├── src/
│   ├── agents/          # MC Control + Q-Learning (agnósticos al env)
│   ├── envs/            # Las 4 tareas P1
│   ├── env.py           # Compat: reexporta BallPursuitSimEnv
│   ├── discretizer.py   # Bins polares compartidos (+ ACTION_NAMES de pursuit)
│   ├── metrics.py       # Resúmenes, eval split de tiro, V*/π*
│   ├── plotting.py      # Curvas + trayectorias “bonitas” para el informe
│   └── robocup_client.py
├── scripts/
│   ├── run_all_tasks.py      # Pipeline canónico: 4 tareas × 4 configs
│   ├── run_p1_experiments.py # Legacy: solo pursuit → figures/
│   ├── generate_tray_sets.py # Sets de trayectorias exitosas (3–5)
│   ├── regenerate_trays.py   # Refresca tray canónicas + figs LaTeX
│   └── test_random_agent.py  # Smoke UDP vs rcssserver
├── notebooks/
│   ├── p1_four_tasks.ipynb   # Demo del catálogo (suele usar menos episodios)
│   └── …
├── figures/
│   ├── metrics_all_tasks.json
│   └── tasks/<tarea>/{curvas,trayectoria,metrics,sets/…}
└── plantilla_informe_latex/  # informe_p1.tex + PDF
```

**Convención de imports:** los scripts meten `src/` en `sys.path`. En Docker, `PYTHONPATH=/workspace/src`. Por eso ves `from envs.pursuit import …` y no `from src.envs…`.

Hay **dos stacks** conviviendo:

| Stack | Para qué |
| :--- | :--- |
| Proxies `src/envs/*.py` | Todo el entrenamiento tabular P1 |
| `RoboCup2DClient` + `rcssserver` | Conexión real / smoke / futuro deploy |

Si solo quieres reproducir métricas del informe, **no necesitas el servidor**.

---

## 3. Contrato común de los entornos

Todas las tareas siguen la misma interfaz mínima (estilo Gym ligero):

```text
reset() → state
step(a) → (state, reward, done, info)

Atributos esperados:
  n_actions, action_names, task_name, max_steps
  trajectory_x / trajectory_y   # para plotting
  success                       # flag booleano al terminar
```

**Estados:** tuplas hashables (enteros de bins). Así la tabla Q es un `dict[state] → np.ndarray[n_actions]` sin asumir un tamaño fijo de |S|.

**Acciones:** los agentes leen `env.n_actions` (`get_n_actions`). Eso permite que shooting tenga 7 acciones y el resto 4 sin forking del trainer.

**γ (gamma)** no vive en el env: lo inyectan los trainers (`gamma=0.99`).

---

## 4. Discretización polar compartida

Archivo: `src/discretizer.py`.

La idea es mapear distancia y ángulo relativo al balón (u otro objeto) a una grilla pequeña. Un **bin** es un intervalo donde agrupas valores continuos para indexar la tabla Q.

| Bin distancia | Rango | Por qué |
| :--- | :--- | :--- |
| 0 | d < 0.8 m | Radio **kickable** de rcssserver |
| 1 | [0.8, 3) m | Reacción cercana |
| 2 | [3, 8) m | Media |
| 3 | d ≥ 8 m | Lejana |

| Bin ángulo | Significado |
| :--- | :--- |
| 0 | Frontal abs(θ) ≤ 15° (cono de dash útil) |
| 1 / 2 | Frontal derecha / izquierda (hasta 60°) |
| 3 / 4 | Posterior derecha / izquierda |

Para pursuit puro: |S| = 4 × 5 = 20.  
Las otras tareas **extienden** el estado con bins extra (meta, portero, compañero, posesión, etc.).

---

## 5. Las cuatro tareas (detalle de diseño)

Registro: `TASK_ENVS` en `src/envs/__init__.py`.

### 5.1 Pursuit — `BallPursuitSimEnv`

**Intuición:** “corre hacia el balón desde lejos”.

| Pieza | Diseño |
| :--- | :--- |
| Estado | `(d_bin, a_bin)` — 20 estados |
| Acciones | DASH fuerte (+1.2 m), suave (+0.6 m), giro ±35° |
| Horizonte | 120 pasos |
| Reset | Distancia inicial al balón d_b ∈ [5, 40] m (como pide el enunciado) |
| Éxito | d_b < 0.8 → R = +100 |
| Timeout | R = −20 |
| Shaping por paso | R = −1 + 5·(d_prev − d_curr) |

**Por qué este shaping:** sin el término de acercamiento, first-visit MC promedia retornos muy distintos (timeout vs captura) y tarda más en descubrir “alinear y empujar”. El costo −1 por paso empuja a capturas cortas.

**Dinámica:** determinista una vez fijado el reset; la aleatoriedad está en pose/balón inicial.

**Mejor config observada:** Q-Learning + ε decay → ~100% éxito, ~31 pasos (últimos 100 eps).

**Lección de exploración:** MC + ε=0.1 fijo se queda corto (~53%) en el rango largo del enunciado. First-visit “ensucia” pares (s, a) con retornos de fallos si la exploración no se apaga. QL, al hacer bootstrap, es más robusto aquí.

Compat: `src/env.py` reexporta esta clase para notebooks viejos.

---

### 5.2 Dribbling — `BallDribblingSimEnv`

**Intuición:** “el balón ya está en los pies; avanza >30 m sin perderlo”.

| Pieza | Diseño |
| :--- | :--- |
| Estado | `(d_ball, a_ball, d_goal, a_goal)` |
| Acciones | micro-kick, dash, giros |
| Éxito | avance neto ≥ 30 m **y** posesión vigente |
| Fallos | fuera de cancha, o balón perdido 3 pasos seguidos |
| Shaping | premio por reducir distancia a la meta + avance en x |

**Truco de implementación:** en dash cerca del balón, el env **pega** el balón ~0.55 m delante del jugador. Eso aproxima “llevar posesión” sin física completa de kick/bounce.

Aquí el shaping alcanza: las cuatro configs terminan altas; QL + decay es la más ágil.

---

### 5.3 Shooting — `GoalShootingSimEnv`

Archivo: `src/envs/shooting.py`.

**Intuición:** no es un MDP largo de movimiento; es casi un **bandit contextual**: el agente elige un perfil de disparo (y, si hace falta, gira antes). El resultado es estocástico (ruido + portero). El enunciado pide tasas distintas con arco abierto (>75%) y con portero (>50%).

---

#### Por qué no bastaba “un solo kick”

Versión temprana: un episodio = un disparo al primer tick.

Problemas:

1. El cuerpo nace con orientación aleatoria (±50°). Si no mira al arco, el modelo de tiro mete **sesgo + mucho ruido** → casi siempre miss.
2. El estado no podía expresar bien “ya estoy alineado”.
3. Contra portero, sin alinear ni elegir esquina, la política colapsaba a patear al centro.

**Rediseño:** episodio corto multi-paso (hasta 8 ticks):

```text
[giro]*  →  kick a un sector del arco  →  fin
```

Los giros **no** terminan el episodio; solo un kick (o el timeout) lo hace.

---

#### Espacio de estados

Tupla de 4 enteros:

```text
s = (dist_bin, body_sector, gk_bin, facing_ok)
```

| Componente | Bins | Significado |
| :--- | :--- | :--- |
| `dist_bin` | 0–3 | Distancia al centro del arco: <10 / <18 / <28 / más lejos |
| `body_sector` | 0–3 | Orientación del cuerpo vs arco: de frente (≤12°), izq., der., u “muy torcido” |
| `gk_bin` | 0–3 | 0 = sin portero; 1 = GK abajo (y<−1.5); 2 = GK arriba; 3 = centrado |
| `facing_ok` | 0/1 | 1 si abs(ángulo al arco) ≤ 20° |

Eso permite políticas del tipo: “si no facing_ok → girar; si GK está abajo → patear al poste de arriba”.

Reset típico: jugador en x∈[32, 44], y∈[−10, 10], θ∈[−50°, 50°]; balón a los pies; GK (si hay) en x=50.5, y∈[−3.5, 3.5].

---

#### Acciones (7)

| id | Nombre | Efecto |
| :--- | :--- | :--- |
| 0 / 1 | `GIRAR_IZQ` / `GIRAR_DER` | ±35°; episodio **sigue** |
| 2 | `KICK_POSTE_IZQ` | potencia 65, aim_y = −5.5 |
| 3 | `KICK_ANGULO_IZQ` | 70, −3.0 |
| 4 | `KICK_CENTRO` | 75, 0.0 |
| 5 | `KICK_ANGULO_DER` | 70, +3.0 |
| 6 | `KICK_POSTE_DER` | 65, +5.5 |

`aim_y` es el punto objetivo sobre la línea de gol (y=0 = centro). Los postes oficiales del proxy usan **POST_Y = ±7.01 m** (RoboCup 2D; no el arco FIFA indoor ±3.66).

---

#### Modelo del disparo (`_simulate_shot`)

Dado potencia y `aim_y`:

1. Se mide el error de orientación del cuerpo respecto al arco (`body_err`).
2. Impacto en la línea de gol:

```text
bias      = 0.04 · body_err
noise_std = 0.35 + 0.04 · |body_err|
            (+ 1.2 extra si |body_err| > 35°)
impact_y  = aim_y + bias + N(0, noise_std)
```

3. Si `|impact_y| > 7.01` → **miss**.
4. Si hay GK y `|impact_y − gk_y| < reach` → **saved**  
   (`reach ≈ 1.35`, un poco mayor si el tiro va al centro).
5. Si no → **goal**.

Consecuencia de diseño: **alinear reduce ruido**; patear torcido es casi suicidio. Contra GK, los tiros a esquina (aim lejos del portero) tienen más chance, y además hay bonus de reward (ver abajo).

---

#### Recompensas

| Situación | R |
| :--- | :--- |
| Giro y quedas mirando (≤15°) | +2 |
| Giro que mejora / empeora orientación | +0.8 / −0.3 (aprox.) |
| Cada paso base (kick path) | −0.5 |
| Gol | +100 (+10 extra si hay GK y tiro angulado \|aim_y\|≥3) |
| Atajada | −40 |
| Miss (fuera de postes) | −25 |
| Timeout sin haber pateado | −25 |

El shaping de giro enseña “mira al arco antes de chutar”. El bonus angulado vs GK empuja a no spamear el centro cuando hay portero.

---

#### Entrenamiento vs evaluación (importante)

**Train:** por defecto `mix_gk_prob=0.5` → cada episodio elige al azar si hay portero. La tasa agregada de éxito en train (~87% en la mejor config) **mezcla** ambos mundos.

**Eval del enunciado:** `metrics.evaluate_shooting_split(Q, seed=99, n_episodes=200)`:

- 200 episodios greedy **solo open**
- 200 episodios greedy **solo con GK**

Números del informe (MC + ε fijo): **98% open / 83% GK**, ambos por encima de 75% / 50%.

Por eso no hay que mirar solo la curva de train mixto: el criterio oficial es el **split**.

---

#### Qué suele aprender la política

En rollouts greedy típicos:

1. Uno o dos giros hasta `facing_ok`.
2. Kick a un sector (a menudo esquina/ángulo si `gk_bin` ≠ 0; centro o ángulo suave si arco abierto).
3. Episodio termina en 1–3 pasos (horizonte máx. 8).

Las trays (`fig_shooting_tray*.png`, sets open/gk) muestran la flecha de disparo hacia el punto de impacto; con GK aparece el diamante del portero.

---

#### Hiperparámetros / knobs útiles

| Knob | Default | Si lo tocas… |
| :--- | :--- | :--- |
| `max_steps` | 8 | Más giros posibles antes del timeout |
| `mix_gk_prob` | 0.5 | Más/menos episodios con portero en train |
| `with_gk` | None | Forzar open/`True` para trays o eval |
| `POST_Y` | 7.01 | Ancho del arco (no cambiar sin reentrenar) |
| ruido en `_simulate_shot` | ver arriba | Más ruido = tarea más dura |

Mejor config observada: **MC Control + ε fijo 0.1**.

---

### 5.4 Passing — `PassingPossessionSimEnv`

**Intuición:** 2v1 (tú + compañero vs defensor estocástico). El criterio del enunciado es **conjunto**:

```text
posesión ≥ 50 pasos   Y   pases ≥ 3
```

#### Decisión clave: alinear R al criterio (no solo a G₀)

Fallo inicial: premiar cada pase con ≈ +30. El agente spameaba pases cortos, el retorno subía, pero la posesión no llegaba a 50 → **éxito estricto bajo**.

Rediseño:

| Elemento | Qué hace |
| :--- | :--- |
| Posesión densa | +1.2 por paso con balón (+ bonus tras 40 pasos) |
| Pase moderado | 8 + 5 si aún no llegas a 3 pases (no +30) |
| Acción `ALEJARSE` | Escapar del marcador |
| Timeout parcial | crédito por posesión/pases incompletos |
| Stun post-pase | **Diseño nuestro** (no enunciado); por defecto **0** tras ablación |

Parámetros del defensor (`PassingPossessionSimEnv`):

```text
defender_mode            = "stochastic" | "passive"
defender_stun_on_pass    = 0   # default adoptado (antes 3)
defender_speed_base      = 0.18
defender_speed_jitter    = 0.10   # speed = base + jitter·U
```

#### Ablación de libertad del defensor

Script: `scripts/ablate_passing_defender.py`  
Salidas: `figures/tasks/passing/defender_ablation.{json,png}` y trays `sets/tray_ablation_stun{0,3}_*.png`.

Resultado (3000 eps, seed 42, últimos 100):

| Config | MC éxito | QL éxito |
| :--- | ---: | ---: |
| stun=3 (histórico) | 97% | 85% |
| stun=1 | 97% | 90% |
| stun=0 (adoptado) | 93% | 81% |
| stun=0, v+25% | 77% | 65% |
| stun=0, v+50% | 76% | 72% |

Conclusión: se puede **quitar el stun** sin romper el criterio; subir mucho la velocidad sí degrada. En trays, stun=0 deja un rastro del defensor mucho más largo (misma seed: ~1 m vs ~16 m de desplazamiento).

**Efecto observable:** QL todavía puede **inflar** el retorno con muchos pases; MC + ε fijo sigue siendo el más fiable en éxito estricto.

Esta tarea es el mejor ejemplo del repo de “maximizar G₀ ≠ cumplir el enunciado” si R está mal calibrado.

---

## 6. Agentes y exploración

### 6.1 Políticas (`src/agents/policies.py`)

```text
ε fijo:     ε = 0.1
ε decay:    ε_k = max(0.05, 1.0 · 0.998^k)
```

Selección ε-greedy; empates en Q → acción uniforme (evita sesgo al índice 0).

### 6.2 Monte Carlo Control First-Visit (`mc_control.py`)

Tras cada episodio, recorre el historial hacia atrás, acumula G, y solo actualiza la **primera visita** de cada par (s, a):

```text
Q(s,a) ← promedio de los G observados en visitas a (s,a)
```

Sin α: es promedio empírico (N cuentas).

### 6.3 Q-Learning (`q_learning.py`)

Update online por transición, α = 0.1:

```text
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

En terminal, el target es solo r.

### 6.4 Matriz de experimentos

Por **cada** tarea se corren 4 configs:

1. MC + ε decay  
2. MC + ε fijo 0.1  
3. QL + ε decay  
4. QL + ε fijo 0.1  

Hiperparámetros canónicos (`run_all_tasks.py`):

| Parámetro | Valor |
| :--- | :--- |
| Episodios | 3500 |
| Semilla train | 42 |
| gamma (γ) | 0.99 |
| alpha α (QL) | 0.1 |
| Ventana MA (plots) | 50 |
| Resumen “final” | últimos 100 episodios |

El notebook `p1_four_tasks.ipynb` a menudo usa **menos** episodios (p. ej. 1500) para demo interactiva; las métricas del informe vienen de los scripts.

---

## 7. Pipeline de métricas y figuras

### Métricas (`src/metrics.py`)

- `summarize_run`: retorno, pasos, éxito (y extras por tarea: avance, pases, posesión, open/GK).
- `evaluate_shooting_split`: tasas greedy open vs GK.
- `q_to_value_policy`: solo tiene sentido limpio en pursuit (grilla 4×5 polar).

### Plotting (`src/plotting.py`)

- Curvas: G₀, tasa de éxito, longitud de episodio.
- Trayectorias: cancha con postes ±7.01, trails (jugador, balón, compañero, **defensor**), GK, flecha de tiro, badge de éxito.
- `rollout_greedy(env, Q)`: ejecuta política greedy y deja historial en el env para dibujar.

**Nota:** las trayectorias “bonitas” son artefactos de comunicación (informe), no parte del aprendizaje.

---

## 8. Scripts: qué correr y cuándo

| Script | Rol |
| :--- | :--- |
| `run_all_tasks.py` | **Fuente de verdad** de entrenamiento + `figures/tasks/` + `metrics_all_tasks.json` |
| `run_p1_experiments.py` | Solo pursuit → `figures/` raíz (legacy / compat informe viejo) |
| `regenerate_trays.py` | Reentrena la **config preferida** por tarea y refresca trays canónicas + figs LaTeX |
| `generate_tray_sets.py` | Escanea semillas, guarda 5 trays exitosas (tiro: 3 open + 2 GK) en `sets/` |
| `ablate_passing_defender.py` | Ablación stun/velocidad del defensor + figura/trays |
| `test_random_agent.py` | ¿Vive el `rcssserver`? |

Configs preferidas para trays (las del mejor rendimiento):

- pursuit / dribbling → QL + ε decay  
- shooting / passing → MC + ε fijo  

---

## 9. Decisiones de diseño que más importan (resumen)

1. **Proxy primero, servidor después** — iteración rápida y semilla reproducible.
2. **Contrato uniforme de envs** — un solo pipeline de agentes/métricas/plots para 4 MDPs.
3. **Agentes agnósticos a |A|** — shooting no rompe el trainer.
4. **Kickable 0.8 m** — alineado con rcssserver, no un número arbitrario.
5. **Postes ±7.01** — escala visual y de gol coherente con RoboCup 2D.
6. **Tiro multi-paso** — el estado debe poder expresar “ya miro al arco”.
7. **Pase: R amarrada al criterio conjunto** — posesión densa > spam de pases.
8. **Comparar ε fijo vs decay** — no es un detalle: en pursuit cambia el veredicto de MC.
9. **Eval de tiro separada** — el mix 50/50 de train no es el umbral del enunciado.
10. **Legacy paths** (`env.py`, `figures/` raíz) — no borrar sin migrar notebooks/informe.

---

## 10. Cómo reproducir desde cero

### Con Docker (recomendado por el starter kit)

```bash
docker compose up -d --build
# Jupyter: http://localhost:8888

docker compose exec rl-agent python /workspace/scripts/run_all_tasks.py
docker compose exec rl-agent python /workspace/scripts/generate_tray_sets.py
```

### Local (solo proxies)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # mínimo útil: numpy matplotlib
export PYTHONPATH="$(pwd)/src"
python scripts/run_all_tasks.py
```

### Informe PDF

```bash
cd plantilla_informe_latex
tectonic informe_p1.tex
```

### Snippet mínimo de API

```python
from envs import TASK_ENVS
from agents.q_learning import train_q_learning
from metrics import summarize_run
from plotting import rollout_greedy, plot_trajectory

env = TASK_ENVS["pursuit"](seed=42)
res = train_q_learning(
    env=env, n_episodes=3500, seed=42,
    exploration="decay", alpha=0.1, gamma=0.99,
)
print(summarize_run(res, last_n=100))

vis = TASK_ENVS["pursuit"](seed=7)
rollout_greedy(vis, res["Q"])
plot_trajectory(vis, info=vis._rollout_info, history=vis._rollout_history)
```

---

## 11. Resultados de referencia (corrida canónica)

Fuente: `figures/metrics_all_tasks.json` (3500 eps, seed 42).

| Tarea | Mejor config | Qué mirar |
| :--- | :--- | :--- |
| Pursuit | QL + ε decay | 100% éxito, ~31 pasos |
| Dribbling | QL + ε decay | 100%, avance ~30.4 m |
| Shooting | MC + ε fijo | Eval greedy: open 98%, GK 83% |
| Passing | MC + ε fijo | 100% criterio estricto; ~18 pases, posesión ~50 |

Si reentrenas y los números cambian un poco: hay estocasticidad en tiro (ruido), pase (defensor) y mezcla GK. El orden cualitativo (qué config gana) suele mantenerse.

---

## 12. Qué queda fuera (a propósito)

- Aprendizaje profundo (DQN/PPO) → Fase 2.
- Partido 11v11 / jerarquías de skills → vía A del enunciado.
- Transferencia del proxy a `rcssserver` con las mismas Q → no garantizada; haría falta re-entrenamiento o fine-tuning.
- Ruido de visión, estamina, sincronización de ciclos del servidor → no están en el proxy.

---

## 13. Dónde mirar si quieres cambiar algo concreto

| Quieres… | Empieza en… |
| :--- | :--- |
| Cambiar bins de distancia/ángulo | `src/discretizer.py` |
| Criterio de éxito de una tarea | `src/envs/<tarea>.py` (flags `success` + rewards terminales) |
| Shaping / recompensas | mismo archivo, cuerpo de `step` |
| ε o α | `scripts/run_all_tasks.py` o defaults de `policies.py` / trainers |
| Figuras del informe | `scripts/generate_tray_sets.py` + `plantilla_informe_latex/informe_p1.tex` |
| Conectar al servidor real | `src/robocup_client.py`, `scripts/test_random_agent.py` |

---

*Documento vivo del repo. Si cambias rewards o horizontes, actualiza la sección de la tarea correspondiente y vuelve a correr `run_all_tasks.py` antes de confiar en las tablas del informe.*
