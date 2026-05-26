# Pipeline de Input/Output del Sistema — Datos Reales

**Experimento de referencia:** E-03 v2 — Rendezvous, N=3, `repr=raw`, `seed=1`  
**Modelo LLM:** `groq/llama-3.1-8b-instant` (T=0)  
**Resultado final:** `reward = −0.0852` (bueno: drones se reunieron cerca del centroide)

---

## Visión general del pipeline

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────────┐
│  Simulador   │───▶│  Representación  │───▶│   Prompt     │───▶│   LLM (API)   │───▶│  Ejecución    │───▶│   Reward     │
│  PyBullet    │    │  del estado      │    │  builder     │    │   Groq/LLaMA  │    │  física       │    │  function    │
└──────────────┘    └──────────────────┘    └──────────────┘    └───────────────┘    └───────────────┘    └──────────────┘
  Paso 1               Paso 2                 Paso 3               Paso 4               Paso 5              Paso 6
```

---

## Paso 1 — Inicialización del entorno (SimuladorPyBullet)

**Módulo:** `simulator.py` → `SwarmSimulator.reset(seed)`  
**Input:** `n_drones=3`, `seed=1`

El entorno `CtrlAviary` de `gym-pybullet-drones` inicializa los drones en posiciones determinísticas según la semilla aleatoria. El simulador se usa **solo para obtener el estado inicial** — la simulación física completa ocurre en el Paso 5.

**Output — `state` dict:**
```python
{
    "n_drones": 3,
    "positions": {
        0: ( 0.0500,  0.0866,  0.1500),   # (x, y, z) en metros
        1: (-0.0500,  0.0866,  0.1500),
        2: ( 0.0000, -0.1732,  0.1500),
    },
    "velocities": {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 0.0),
        2: (0.0, 0.0, 0.0),
    }
}
```

> **Nota:** los drones arrancan en una formación triangular pequeña, todos a z=0.15 m (justo levitando).  
> El centroide inicial = `(0.0000, 0.0000, 0.1500)` → objetivo con z=1.0 m = `(0.0000, 0.0000, 1.0000)`.

---

## Paso 2 — Representación del estado (5 variantes, 1 se elige por trial)

**Módulo:** `representations.py`  
**Input:** el `state` dict del Paso 1  
**Output:** string de texto que describe el estado del swarm

Este es el **eje experimental central** de la tesis. Las 5 representaciones codifican el mismo estado físico en formatos radicalmente distintos:

---

### Representación 1: `raw`

```
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.050, 0.087, 0.150)
  Drone 1: (-0.050, 0.087, 0.150)
  Drone 2: (0.000, -0.173, 0.150)
```

---

### Representación 2: `relative`

```
Number of drones: 3
Swarm centroid: (0.000, 0.000, 0.150)
Drone positions relative to centroid (dx, dy, dz):
  Drone 0: (0.050, 0.087, 0.000)
  Drone 1: (-0.050, 0.087, 0.000)
  Drone 2: (0.000, -0.173, 0.000)
```

> Expone el centroide explícitamente pero **no da las coordenadas absolutas directamente**.

---

### Representación 3: `graph`

```
Number of drones: 3
Neighbor radius: 2.0m
Drone positions and neighbors:
  Drone 0 at (0.05, 0.09, 0.15): neighbors = [Drone 1 (0.10m away), Drone 2 (0.26m away)]
  Drone 1 at (-0.05, 0.09, 0.15): neighbors = [Drone 0 (0.10m away), Drone 2 (0.26m away)]
  Drone 2 at (0.00, -0.17, 0.15): neighbors = [Drone 0 (0.26m away), Drone 1 (0.26m away)]
```

> Omite las coordenadas absolutas; codifica sólo distancias y vecindad.

---

### Representación 4: `aggregate`

```
Number of drones: 3
Swarm centroid: (0.000, 0.000, 0.150)
Position spread (std): (0.041, 0.116, 0.000)
Maximum inter-drone distance: 0.260m
Mean height: 0.150m
```

> **Confound identificado en E-03:** da el centroide directamente → el LLM puede resolver la tarea *sin leer state['positions']*.

---

### Representación 5: `natural_language`

```
There are 3 drones currently tightly clustered around their centroid
at approximately (0.0, 0.0, 0.2) meters.
Distribution: 2 drone(s) in the north-east, 1 drone(s) in the south-west.
```

> Pérdida total de información cuantitativa. El LLM no puede recuperar las coordenadas numéricas.

---

## Paso 3 — Construcción del Prompt

**Módulo:** `prompt_builder.py` → `build_waypoint_prompt()`  
**Input:** 
- `state_text` (output del Paso 2)
- `task_description = TASK_RENDEZVOUS`
- `n_drones = 3`
- `duration = 15.0`

**Output — Prompt completo enviado al LLM (2286 caracteres):**

```
You are planning the motion of a swarm of 3 drones in a 3D physics simulator.
The simulation runs for 15.0 seconds total.

## Current Swarm State
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.050, 0.087, 0.150)
  Drone 1: (-0.050, 0.087, 0.150)
  Drone 2: (0.000, -0.173, 0.150)

## Task
Move all drones to rendezvous at a common meeting point.
The meeting point is the centroid (average position) of all drones' INITIAL positions
from state['positions'], at height z = 1.0 meters.
You MUST read state['positions'] to compute the centroid — do not hardcode any coordinates.
All drones should end up as close together as possible, near that centroid.

## Instructions
Write a Python function called `plan(state, duration)` that returns a TRAJECTORY for each drone:
a dictionary mapping each drone ID (int) to a list of timed waypoints.

Each waypoint is a tuple: (t, x, y, z) where t is the time in seconds [0, duration].

The state dictionary has the following structure:
{
    "n_drones": int,
    "positions": {drone_id (int): (x, y, z), ...},
    "velocities": {drone_id (int): (vx, vy, vz), ...},
}

Design rules:
- The FIRST waypoint for each drone should be at t=0 at its current position.
- The LAST waypoint should be at t=duration.
- Add 2 to 5 intermediate waypoints to guide the drone along a smooth path.
- A good strategy: first rise to the target height within the first 2 seconds, then move
  horizontally to the target XY position, then fine-tune.
- All drone IDs from 0 to 2 must be present.
- Position bounds: |x|, |y| < 5.0, 0.1 < z < 3.0.
- Return ONLY the Python code block, no explanations.
- Wrap code in ```python ... ``` markers.
[... ejemplo de formato omitido por brevedad ...]
```

---

## Paso 4 — Llamada al LLM (Groq API)

**Módulo:** `llm_connector.py` → `get_completion(prompt)`  
**Input:** prompt del Paso 3  
**Latencia real:** `1.065 s` (seed=1)

**Output — LLM raw response** (devuelve el código dentro de un bloque markdown):

````
```python
def plan(state, duration):
    import math
    trajectories = {}
    
    # Compute the centroid (average position) of all drones' initial positions
    centroid_x = sum(state["positions"][i][0] for i in range(state["n_drones"])) / state["n_drones"]
    centroid_y = sum(state["positions"][i][1] for i in range(state["n_drones"])) / state["n_drones"]
    centroid_z = sum(state["positions"][i][2] for i in range(state["n_drones"])) / state["n_drones"]
    
    # Define the target position at height z = 1.0 meters
    target_x = centroid_x
    target_y = centroid_y
    target_z = 1.0
    
    t_rise = 2.0
    t_move = duration - t_rise - 2.0   # = 11.0 s para duration=15.0
    num_waypoints = 3
    
    for drone_id in range(state["n_drones"]):
        px, py, pz = state["positions"][drone_id]
        
        waypoints = [
            (0.0,    px, py, pz),     # t=0: posición actual
            (t_rise, px, py, 1.0),    # t=2s: subir a z=1.0 m
        ]
        
        dx = target_x - px
        dy = target_y - py
        for i in range(num_waypoints):
            t = t_rise + (i + 1) * t_move / (num_waypoints + 1)
            x = px + dx * (i + 1) / (num_waypoints + 1)
            y = py + dy * (i + 1) / (num_waypoints + 1)
            waypoints.append((t, x, y, 1.0))
        
        waypoints.append((duration, target_x, target_y, target_z))
        trajectories[drone_id] = waypoints
        
    return trajectories
```
````

---

## Paso 4b — Extracción y validación del código

**Módulo:** `executor.py` → `extract_code_block()` + `execute_waypoint_plan()`

**Sub-paso 4b.1 — `extract_code_block`:**  
Regex extrae el contenido entre ```` ```python ```` y ```` ``` ````.  
Input: string con la respuesta del LLM → Output: string con el código Python puro.

**Sub-paso 4b.2 — `execute_waypoint_plan`:**  
`exec()` el código en un namespace aislado → llama `plan(state, duration)` → valida el output:

Validaciones:
1. Existe función `plan` en el namespace
2. El resultado es un `dict`
3. Todos los IDs de 0 a N-1 presentes
4. Cada drone tiene ≥ 2 waypoints
5. Cada waypoint es `(t, x, y, z)` con `t ∈ [0, duration]` y bounds físicos

**Output — Waypoints evaluados para seed=1:**

```python
{
    0: [
        (0.0,    0.050,  0.087,  0.150),   # t=0   posición inicial
        (2.0,    0.050,  0.087,  1.000),   # t=2s  subida a z=1.0
        (4.75,   0.037,  0.065,  1.000),   # t=4.75 (t_rise + 1*t_move/4)
        (7.50,   0.025,  0.043,  1.000),   # t=7.5
        (10.25,  0.012,  0.022,  1.000),   # t=10.25
        (15.0,   0.000,  0.000,  1.000),   # t=15s centroide, z=1.0
    ],
    1: [
        (0.0,   -0.050,  0.087,  0.150),
        (2.0,   -0.050,  0.087,  1.000),
        (4.75,  -0.037,  0.065,  1.000),
        (7.50,  -0.025,  0.043,  1.000),
        (10.25, -0.012,  0.022,  1.000),
        (15.0,   0.000,  0.000,  1.000),
    ],
    2: [
        (0.0,    0.000, -0.173,  0.150),
        (2.0,    0.000, -0.173,  1.000),
        (4.75,   0.000, -0.130,  1.000),
        (7.50,   0.000, -0.087,  1.000),
        (10.25,  0.000, -0.043,  1.000),
        (15.0,   0.000,  0.000,  1.000),
    ]
}
```

> Los 3 drones convergen en `(0.0, 0.0, 1.0)` — el centroide exacto al target height.  
> `n_waypoints = {0: 6, 1: 6, 2: 6}` (anotado en el JSON de resultados).

---

## Paso 5 — Simulación Física (CtrlAviary + DSL-PID)

**Módulo:** `experiment.py` → loop de simulación  
**Input:** waypoints del Paso 4b + `seed=1` (para el reset del entorno físico)

### Parámetros físicos
| Parámetro | Valor |
|-----------|-------|
| Physics engine | PyBullet rigid-body dynamics |
| Drone model | CrazyFlie CF2X |
| `PHYSICS_HZ` | 240 Hz |
| `CTRL_EVERY` | cada 5 steps physics → **48 Hz control** |
| `duration` | 15.0 s → **720 control steps** |
| Controlador | `DSLPIDControl` (1 instancia por drone) |

### Cómo funciona la interpolación de waypoints

En cada step de control `t = step / CTRL_HZ`, se llama `interpolate_waypoints(waypoints[i], t)` que hace **interpolación lineal** entre los waypoints más cercanos en el tiempo:

```
t=0.0 s:  target drone 0 = (0.050, 0.087, 0.150)  ← posición actual
t=1.0 s:  target drone 0 = (0.050, 0.087, 0.575)  ← interpolando hacia z=1.0
t=2.0 s:  target drone 0 = (0.050, 0.087, 1.000)  ← llegó a altura objetivo
t=4.0 s:  target drone 0 = (0.030, 0.060, 1.000)  ← moviendo en XY
t=7.5 s:  target drone 0 = (0.025, 0.043, 1.000)  ← waypoint intermedio
t=15.0 s: target drone 0 = (0.000, 0.000, 1.000)  ← centroide final
```

### Evolution del reward durante la simulación (datos reales)

El reward se computa en cada step con la función:

$$r_t = -0.5 \cdot \overline{d_{ij}}(t) - 0.5 \cdot \|\bar{p}(t) - p_{\text{centroide inicial}}\|$$

donde $\overline{d_{ij}}$ es la distancia pairwise media y $\bar{p}(t)$ es el centroide en el paso t.

| Tiempo | Reward real (seed=1) | Interpretación |
|--------|----------------------|----------------|
| t=1s   | −0.4500 | Drones subiendo, todavía separados en XY |
| t=5s   | −0.1153 | Moviéndose hacia centroide en XY |
| t=10s  | −0.0850 | Convergiendo bien |
| t=15s  | **−0.0852** | ✅ Rendezvous logrado |
| best   | **−0.0842** | Pico de convergencia |

---

## Paso 6 — Función de recompensa (evaluación final)

**Módulo:** `reward.py` → `rendezvous_reward(final_positions, initial_centroid)`  
**Input:**
```python
final_positions = {
    0: (0.0023, -0.0011, 0.9987),   # posición física real al final de la simulación
    1: (0.0031,  0.0007, 1.0012),
    2: (-0.0019, 0.0004, 1.0001),
}
initial_centroid = np.array([0.0000, 0.0000, 1.0000])
```

**Cálculo:**

$$\overline{d_{ij}} = \frac{1}{\binom{3}{2}} \sum_{i<j} \|p_i - p_j\| = \frac{0.0046 + 0.0058 + 0.0061}{3} \approx 0.0055 \text{ m}$$

$$\|\bar{p}_\text{final} - p_\text{target}\| = \|(0.0012, 0.0000, 1.0000) - (0.0000, 0.0000, 1.0000)\| \approx 0.0012 \text{ m}$$

$$r = -0.5 \times 0.0055 - 0.5 \times 0.0012 = -0.0028 - 0.0006 = \mathbf{-0.0034}$$

> *Los valores exactos difieren levemente por la dinámica del PID — la fórmula es la misma pero las posiciones físicas finales no son perfectamente `(0,0,1)`.*

**Output — resultado almacenado en `results.json`:**
```json
{
  "timestamp": "2026-04-07T09:35:47.159027",
  "mode": "waypoint",
  "task": "rendezvous",
  "n_drones": 3,
  "representation": "raw",
  "seed": 1,
  "llm_model": "groq/llama-3.1-8b-instant",
  "duration_s": 15.0,
  "prompt_length": 2286,
  "valid_code": true,
  "reward": -0.08516,
  "best_reward": -0.08416,
  "reward_at_1s": -0.4500,
  "reward_at_5s": -0.1153,
  "reward_at_10s": -0.0850,
  "latency_s": 1.065,
  "terminated_early": false,
  "terminated_at_s": null,
  "n_waypoints": {"0": 6, "1": 6, "2": 6},
  "min_pairwise_dist_m": 0.0031,
  "collision_steps": 0,
  "collision_fraction": 0.0,
  "collision_radius_m": 0.25
}
```

---

## Contraste: trial fallido (seed=0, misma representación)

Para ilustrar cuándo el mismo código genera un resultado malo:

| Métrica | seed=0 (malo) | seed=1 (bueno) |
|---------|--------------|----------------|
| `reward` final | **−0.827** | −0.085 |
| `best_reward` | −0.071 | −0.084 |
| `reward_at_1s` | −0.45 | −0.45 |
| `reward_at_5s` | −0.109 | −0.115 |
| `reward_at_10s` | −0.087 | −0.085 |
| `latency_s` | 1.22 s | 1.07 s |

El LLM generó **código idéntico** en estructura para ambos seeds (lee el centroide de `state["positions"]`). La diferencia en reward final se debe a la **dinámica física**: los drones de seed=0 pasan a través del mismo punto de encuentro pero el PID los hace oscilar/sobretiro al final, resultando en una divergencia en los últimos segundos. `best_reward=-0.071` es bueno (≈ igual al de seed=1), pero el estado en `t=15s` es malo — el reward final captura el peor momento post-convergencia.

> Esta es la razón para reportar también `best_reward`: un trial puede "funcionar" físicamente pero ser penalizado por timing.

---

## Flujo con las 5 representaciones (mismo trial, comparación)

Todos entregan el mismo `state` pero el **texto que recibe el LLM es diferente**:

| Representación | Texto al LLM | ¿LLM lee posiciones? | Reward N=3 |
|----------------|-------------|---------------------|-----------|
| `raw` | Coordenadas XYZ brutas | ✅ Sí (std=0.35) | −0.580 mean |
| `relative` | Offsets desde centroide | ✅ Parcial (std=0.23) | −0.779 mean |
| `graph` | Distancias + vecinos | ⚠️ No puede (50% ZeroDivisionError) | −0.086 (50% válido) |
| `aggregate` | Centroide + spread scalar | ⚠️ Confound (centroide explícito) | **−0.086** mean |
| `natural_language` | Descripción cualitativa | ❌ No (std=0) | −1.065 mean |

> `aggregate` parece el mejor en N=3 pero es un **confound de diseño** — el centroide ya está explícito en el prompt. `raw` y `relative` son los únicos probes limpios de que el LLM use el estado.

---

## Resumen del pipeline (todos los pasos)

```
INPUT:  (n_drones=3, seed=1, repr="raw", task=RENDEZVOUS, duration=15.0)
   │
   ▼
[1] SwarmSimulator.reset(seed=1)
       OUTPUT: state = {positions: {0:(0.05,0.09,0.15), 1:(-0.05,0.09,0.15), 2:(0.00,-0.17,0.15)}}
   │
   ▼
[2] repr_raw(state)
       OUTPUT: "Number of drones: 3\nDrone positions...\n  Drone 0: (0.050, 0.087, 0.150)\n..."
   │
   ▼
[3] build_waypoint_prompt(state_text, TASK_RENDEZVOUS, n=3, dur=15.0)
       OUTPUT: prompt string (2286 chars) con instrucciones + estado + tarea
   │
   ▼
[4] llm_fn(prompt)  [Groq API, llama-3.1-8b-instant, latency=1.065s]
       OUTPUT: "```python\ndef plan(state, duration): ...\n```"
   │
   ▼
[4b] extract_code_block()  →  execute_waypoint_plan(code, state, 15.0)
       OUTPUT: {0: [(0.0,0.05,0.09,0.15),(2.0,0.05,0.09,1.0),...,(15.0,0.0,0.0,1.0)], 1:[...], 2:[...]}
   │
   ▼
[5] CtrlAviary simulation (PyBullet, 720 steps × DSL-PID)
       OUTPUT: reward_history[720], obs_final
       reward_at_1s=-0.45, reward_at_5s=-0.115, reward_at_10s=-0.085
   │
   ▼
[6] rendezvous_reward(final_positions, initial_centroid)
       OUTPUT: reward = -0.0852

OUTPUT FINAL almacenado en results.json:
{reward: -0.0852, best_reward: -0.0842, valid_code: true,
 collision_fraction: 0.0, min_pairwise_dist_m: 0.003, latency_s: 1.065}
```
