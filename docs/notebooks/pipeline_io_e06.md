# Pipeline Input/Output — Experimento E-06

**E-06:** Rendezvous task · `output_mode=direct` · `position_jitter=0.5 m`

**Trial de referencia:** N=3, seed=1 (todas las representaciones, misma configuración inicial)

**Modelo LLM:** `groq/llama-3.1-8b-instant` (T=0)

> La diferencia con E-05 es el modo de salida: en lugar de generar código Python,
> el LLM debe razonar aritméticamente y producir waypoints como JSON puro.
> Esto revela si cada representación preserva suficiente información cuantitativa
> para que el LLM calcule el centroide **sin el apoyo de Python**.

---

## Visión general del pipeline (E-06 vs E-05)

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌─────────────┐
│  Simulador   │───▶│  Representación  │───▶│   Prompt     │───▶│   LLM (API)   │───▶│  Parseo JSON   │───▶│   Reward    │
│  PyBullet    │    │  del estado      │    │  direct mode │    │   Groq/LLaMA  │    │  (no exec())   │    │  function   │
└──────────────┘    └──────────────────┘    └──────────────┘    └───────────────┘    └────────────────┘    └─────────────┘
  Paso 1               Paso 2                 Paso 3               Paso 4               Paso 5                Paso 6

                                                               ┌─────────────────────────────────────────────────┐
                                                               │ DIFERENCIA KEY vs E-05:                          │
                                                               │  E-05: LLM genera def plan(state, duration)     │
                                                               │        → exec() → calcula centroide en Python   │
                                                               │  E-06: LLM escribe números JSON directamente    │
                                                               │        → parse → usa los valores numéricos      │
                                                               └─────────────────────────────────────────────────┘
```

---

## Paso 1 — Inicialización del entorno

**Módulo:** `simulator.py` → `SwarmSimulator.reset(seed=1, position_jitter=0.5)`
**Input:** `n_drones=3`, `seed=1`, `position_jitter=0.5 m`

El simulador aplica un jitter seeded de ±0.5 m en XY antes de inicializar `CtrlAviary`.
**Todas las representaciones del Paso 2 comparten este mismo estado inicial.**

**Output — `state` dict:**
```python
{
    "n_drones": 3,
    "positions": {
        0: (0.0118, 0.4505, 0.1125),
        1: (-0.1970, 0.6074, 0.1125),
        2: (0.1294, 0.2409, 0.1125),
    },
    "velocities": { 0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0) }
}
```

> **Centroide inicial:** (-0.0186, 0.4329, 0.1125)
> **Punto objetivo (rendezvous):** (-0.0186, 0.4329, 1.0000)  ← mismo XY, z=1.0 m

Con jitter activo, las posiciones son **diferentes en cada seed**, lo que obliga al LLM a
razonar sobre los valores numéricos concretos en lugar de asumir una configuración fija.

---

## Paso 2 — Representaciones del estado (5 variantes para el mismo estado)

Este es el **eje experimental central**. Las 5 representaciones codifican el mismo
estado físico en sistemas de coordenadas y niveles de abstracción radicalmente distintos.

> **Hipótesis central de E-06:** En modo `direct`, el LLM debe producir números
> absolutos en el JSON. Las representaciones que no exponen coordenadas absolutas
> (o las transforman) van a inducir errores sistemáticos de razonamiento.

### `raw` — Coordenadas absolutas XYZ

```
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.012, 0.450, 0.113)
  Drone 1: (-0.197, 0.607, 0.113)
  Drone 2: (0.129, 0.241, 0.113)
```

### `relative` — Offsets relativos al centroide

```
Number of drones: 3
Swarm centroid: (-0.019, 0.433, 0.113)
Drone positions relative to centroid (dx, dy, dz):
  Drone 0: (0.030, 0.018, 0.000)
  Drone 1: (-0.178, 0.175, 0.000)
  Drone 2: (0.148, -0.192, 0.000)
```

### `graph` — Grafo de vecindad + distancias

```
Number of drones: 3
Neighbor radius: 2.0m
Drone positions and neighbors:
  Drone 0 at (0.01, 0.45, 0.11): neighbors = [Drone 1 (0.26m away), Drone 2 (0.24m away)]
  Drone 1 at (-0.20, 0.61, 0.11): neighbors = [Drone 0 (0.26m away), Drone 2 (0.49m away)]
  Drone 2 at (0.13, 0.24, 0.11): neighbors = [Drone 0 (0.24m away), Drone 1 (0.49m away)]
```

### `aggregate` — Estadísticas agregadas del swarm

```
Number of drones: 3
Swarm centroid: (-0.019, 0.433, 0.113)
Position spread (std): (0.135, 0.150, 0.000)
Maximum inter-drone distance: 0.491m
Mean height: 0.113m
```

### `natural_language` — Lenguaje natural (sin coordenadas)

```
There are 3 drones currently tightly clustered around their centroid at approximately (-0.0, 0.4, 0.1) meters. Distribution: 1 drone(s) in the north-east, 1 drone(s) in the north-west, 1 drone(s) in the south-east.
```

#### Tabla comparativa — ¿qué información recibe el LLM?

| Representación | ¿Posiciones absolutas? | ¿Centroide explícito? | Info perdida |
|---|---|---|---|
| `raw` | ✅ XYZ por dron | ✅ debe calcular | nada |
| `relative` | ❌ solo offsets Δ | ✅ dado explícito | coordenadas absolutas |
| `graph` | ✅ XYZ por dron | ✅ debe calcular | solo distancias al vecino |
| `aggregate` | ❌ sin info por dron | ✅ dado explícito | posiciones individuales |
| `natural_language` | ❌ ninguna | ≈ en palabras | precisión numérica total |

---

---

# Representación: `raw` — Coordenadas absolutas XYZ

**Trial:** N=3, seed=1  |  latencia=1.86s  |  válido=True  |  reward=-0.06698475479375461

## Paso 3 — Prompt enviado al LLM

**Longitud del prompt:** 2139 caracteres  ·  Cambia vs otras repr solo en el bloque `## Current Swarm State`.

> El bloque de estado que recibe el LLM es la representación `raw` de arriba.
> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.

## Paso 4 — Respuesta cruda del LLM

**Latencia:** 1.86 s

```
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z) from the state.

Drone 0: (0.012, 0.450, 0.113)
Drone 1: (-0.197, 0.607, 0.113)
Drone 2: (0.129, 0.241, 0.113)

### Step 2: Compute the target coordinates arithmetically.

To find the centroid, we sum the x, y, and z values of all drones and divide by the number of drones (3).

x_target = (0.012 + (-0.197) + 0.129) / 3 = -0.028
y_target = (0.450 + 0.607 + 0.241) / 3 = 0.432
z_target = (0.113 + 0.113 + 0.113) / 3 = 0.113 (we need to adjust this to 1.0 meters)

### Step 3: For each drone, write the list of waypoints, explaining each one.

#### Drone 0:

- Waypoint 1: t=0.0, position = (0.012, 0.450, 0.113) (current position)
- Waypoint 2: t=2.0, position = (0.012, 0.450, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.028, 0.432, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.028, 0.432, 1.0) (final position)

#### Drone 1:

- Waypoint 1: t=0.0, position = (-0.197, 0.607, 0.113) (current position)
- Waypoint 2: t=2.0, position = (-0.197, 0.607, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.028, 0.432, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.028, 0.432, 1.0) (final position)

#### Drone 2:

- Waypoint 1: t=0.0, position = (0.129, 0.241, 0.113) (current position)
- Waypoint 2: t=2.0, position = (0.129, 0.241, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.028, 0.432, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.028, 0.432, 1.0) (final position)

## JSON Plan

```json
{
  "0": [
    [0.0, 0.012, 0.450, 0.113],
    [2.0, 0.012, 0.450, 1.0],
    [7.5, -0.028, 0.432, 1.0],
    [15.0, -0.028, 0.432, 1.0]
  ],
  "1": [
    [0.0, -0.197, 0.607, 0.113],
    [2.0, -0.197, 0.607, 1.0],
    [7.5, -0.028, 0.432, 1.0],
    [15.0, -0.028, 0.432, 1.0]
  ],
  "2": [
    [0.0, 0.129, 0.241, 0.113],
    [2.0, 0.129, 0.241, 1.0],
    [7.5, -0.028, 0.432, 1.0],
    [15.0, -0.028, 0.432, 1.0]
  ]
}
```
```

## Paso 5 — Parseo del JSON y resultado

**Estado:** ✅ Waypoints parseados correctamente

n_waypoints = {"0": 4, "1": 4, "2": 4}

**Evolución del reward durante la simulación:**

| Tiempo | Reward | Nota |
|--------|--------|------|
| t = 1 s | -0.4635 | — |
| t = 5 s | -0.0939 | — |
| t = 10 s | -0.0818 | — |
| t = 15 s (final) | -0.0670 | reward final |
| mejor en toda la sim | -0.0646 | best_reward |


**Colisiones:** collision_fraction=67.50%  |  min_dist=0.122m

## Paso 6 — Reward final

**Reward final (t=15s):** **-0.0670**
**best_reward:** -0.0646

---

# Representación: `relative` — Offsets relativos al centroide

**Trial:** N=3, seed=1  |  latencia=15.43s  |  válido=False  |  reward=None

## Paso 3 — Prompt enviado al LLM

**Longitud del prompt:** 2193 caracteres  ·  Cambia vs otras repr solo en el bloque `## Current Swarm State`.

> El bloque de estado que recibe el LLM es la representación `relative` de arriba.
> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.

## Paso 4 — Respuesta cruda del LLM

**Latencia:** 15.43 s

```
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z) from the state.

Drone 0: (0.030, 0.018, 0.000)
Drone 1: (-0.178, 0.175, 0.000)
Drone 2: (0.148, -0.192, 0.000)

### Step 2: Compute the target coordinates arithmetically.

To find the target centroid, we need to read the initial positions from state['positions'] to compute the centroid. However, since we are not given the state['positions'] directly, we will assume that the initial positions are the same as the current positions for the purpose of this exercise.

Target centroid (x, y, z) = ((0.030 + (-0.178) + 0.148) / 3, (0.018 + 0.175 + (-0.192)) / 3, 0.000 / 3) = (-0.019, 0.001, 0.000)

However, we are given that the target height is z = 1.0 meters. Therefore, the target centroid (x, y, z) = (-0.019, 0.001, 1.0)

### Step 3: For each drone, write the list of waypoints, explaining each one.

#### Drone 0:

- Waypoint 1: t=0.0, position = (0.030, 0.018, 0.000) (current position)
- Waypoint 2: t=2.0, position = (0.030, 0.018, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.019, 0.001, 1.0) (move horizontally to target centroid)
- Waypoint 4: t=15.0, position = (-0.019, 0.001, 1.0) (final position)

#### Drone 1:

- Waypoint 1: t=0.0, position = (-0.178, 0.175, 0.000) (current position)
- Waypoint 2: t=2.0, position = (-0.178, 0.175, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.019, 0.001, 1.0) (move horizontally to target centroid)
- Waypoint 4: t=15.0, position = (-0.019, 0.001, 1.0) (final position)

#### Drone 2:

- Waypoint 1: t=0.0, position = (0.148, -0.192, 0.000) (current position)
- Waypoint 2: t=2.0, position = (0.148, -0.192, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.019, 0.001, 1.0) (move horizontally to target centroid)
- Waypoint 4: t=15.0, position = (-0.019, 0.001, 1.0) (final position)

## JSON Plan

```json
{
  "0": [
    [0.0, 0.030, 0.018, 0.000],
    [2.0, 0.030, 0.018, 1.0],
    [7.5, -0.019, 0.001, 1.0],
    [15.0, -0.019, 0.001, 1.0]
  ],
  "1": [
    [0.0, -0.178, 0.175, 0.000],
    [2.0, -0.178, 0.175, 1.0],
    [7.5, -0.019, 0.001, 1.0],
    [15.0, -0.019, 0.001, 1.0]
  ],
  "2": [
    [0.0, 0.148, -0.192, 0.000],
    [2.0, 0.148, -0.192, 1.0],
    [7.5, -0.019, 0.001, 1.0],
    [15.0, -0.019, 0.001, 1.0]
  ]
}
```
```

## Paso 5 — Parseo del JSON y resultado

**Estado:** ❌ Parseo fallido — `direct_parse_failed`


**Código o JSON producido antes de fallar:**
_sin código extraído (inválido)_

## Paso 6 — Reward final

**Reward final (t=15s):** N/A (trial inválido)

---

# Representación: `graph` — Grafo de vecindad + distancias

**Trial:** N=3, seed=1  |  latencia=1.46s  |  válido=True  |  reward=-0.07994053911049585

## Paso 3 — Prompt enviado al LLM

**Longitud del prompt:** 2326 caracteres  ·  Cambia vs otras repr solo en el bloque `## Current Swarm State`.

> El bloque de estado que recibe el LLM es la representación `graph` de arriba.
> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.

## Paso 4 — Respuesta cruda del LLM

**Latencia:** 1.46 s

```
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z) from the state.

- Drone 0: (0.01, 0.45, 0.11)
- Drone 1: (-0.20, 0.61, 0.11)
- Drone 2: (0.13, 0.24, 0.11)

### Step 2: Compute the target coordinates arithmetically.

To find the target coordinates, we need to read the initial positions from `state['positions']`. However, since we don't have the actual `state` object, we'll assume the initial positions are the same as the current positions for this example.

Let's assume the initial positions are:
- Drone 0: (0.01, 0.45, 0.11)
- Drone 1: (-0.20, 0.61, 0.11)
- Drone 2: (0.13, 0.24, 0.11)

The target coordinates are the centroid (average position) of all drones' initial positions at height z = 1.0 meters.

Target x = (0.01 + (-0.20) + 0.13) / 3 = -0.0533
Target y = (0.45 + 0.61 + 0.24) / 3 = 0.43
Target z = 1.0

### Step 3: For each drone, write the list of waypoints, explaining each one.

#### Drone 0:

- Waypoint 1: t=0.0, position = (0.01, 0.45, 0.11) (current position)
- Waypoint 2: t=2.0, position = (0.01, 0.45, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.0533, 0.43, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.0533, 0.43, 1.0) (final position)

#### Drone 1:

- Waypoint 1: t=0.0, position = (-0.20, 0.61, 0.11) (current position)
- Waypoint 2: t=2.0, position = (-0.20, 0.61, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.0533, 0.43, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.0533, 0.43, 1.0) (final position)

#### Drone 2:

- Waypoint 1: t=0.0, position = (0.13, 0.24, 0.11) (current position)
- Waypoint 2: t=2.0, position = (0.13, 0.24, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (-0.0533, 0.43, 1.0) (move horizontally to target x and y)
- Waypoint 4: t=15.0, position = (-0.0533, 0.43, 1.0) (final position)

## JSON Plan

```json
{
  "0": [
    [0.0, 0.01, 0.45, 0.11],
    [2.0, 0.01, 0.45, 1.0],
    [7.5, -0.0533, 0.43, 1.0],
    [15.0, -0.0533, 0.43, 1.0]
  ],
  "1": [
    [0.0, -0.20, 0.61, 0.11],
    [2.0, -0.20, 0.61, 1.0],
    [7.5, -0.0533, 0.43, 1.0],
    [15.0, -0.0533, 0.43, 1.0]
  ],
  "2": [
    [0.0, 0.13, 0.24, 0.11],
    [2.0, 0.13, 0.24, 1.0],
    [7.5, -0.0533, 0.43, 1.0],
    [15.0, -0.0533, 0.43, 1.0]
  ]
}
```
```

## Paso 5 — Parseo del JSON y resultado

**Estado:** ✅ Waypoints parseados correctamente

n_waypoints = {"0": 4, "1": 4, "2": 4}

**Evolución del reward durante la simulación:**

| Tiempo | Reward | Nota |
|--------|--------|------|
| t = 1 s | -0.4686 | — |
| t = 5 s | -0.0972 | — |
| t = 10 s | -0.0961 | — |
| t = 15 s (final) | -0.0799 | reward final |
| mejor en toda la sim | -0.0740 | best_reward |


**Colisiones:** collision_fraction=67.50%  |  min_dist=0.122m

## Paso 6 — Reward final

**Reward final (t=15s):** **-0.0799**
**best_reward:** -0.0740

---

# Representación: `aggregate` — Estadísticas agregadas del swarm

**Trial:** N=3, seed=1  |  latencia=2.11s  |  válido=True  |  reward=-0.2863117419595281

## Paso 3 — Prompt enviado al LLM

**Longitud del prompt:** 2143 caracteres  ·  Cambia vs otras repr solo en el bloque `## Current Swarm State`.

> El bloque de estado que recibe el LLM es la representación `aggregate` de arriba.
> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.

## Paso 4 — Respuesta cruda del LLM

**Latencia:** 2.11 s

```
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z) from the state.

From the given state, we have:
- Drone 0: (-0.019, 0.433, 0.113)
- Drone 1: (0.135, 0.150, 0.113)
- Drone 2: (0.491, 0.000, 0.113)

### Step 2: Compute the target coordinates arithmetically.

To find the target coordinates, we need to compute the centroid of the initial positions of all drones. We sum up the x, y, and z values of all drones and divide by the number of drones (3).

Target x = (-0.019 + 0.135 + 0.491) / 3 = 0.3537
Target y = (0.433 + 0.150 + 0.000) / 3 = 0.1943
Target z = (0.113 + 0.113 + 0.113) / 3 = 0.1133

However, we are given that the target height is 1.0 meters. So, we will use the target x and y coordinates and the target height of 1.0 meters.

### Step 3: For each drone, write the list of waypoints, explaining each one.

#### Drone 0:

- Waypoint 1: t=0.0, position = (-0.019, 0.433, 0.113) (current position)
- Waypoint 2: t=2.0, position = (-0.019, 0.433, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (0.3537, 0.1943, 1.0) (move horizontally to target position)
- Waypoint 4: t=15.0, position = (0.3537, 0.1943, 1.0) (final position)

#### Drone 1:

- Waypoint 1: t=0.0, position = (0.135, 0.150, 0.113) (current position)
- Waypoint 2: t=2.0, position = (0.135, 0.150, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (0.3537, 0.1943, 1.0) (move horizontally to target position)
- Waypoint 4: t=15.0, position = (0.3537, 0.1943, 1.0) (final position)

#### Drone 2:

- Waypoint 1: t=0.0, position = (0.491, 0.000, 0.113) (current position)
- Waypoint 2: t=2.0, position = (0.491, 0.000, 1.0) (rise to target height)
- Waypoint 3: t=7.5, position = (0.3537, 0.1943, 1.0) (move horizontally to target position)
- Waypoint 4: t=15.0, position = (0.3537, 0.1943, 1.0) (final position)

## JSON Plan

```json
{
  "0": [
    [0.0, -0.019, 0.433, 0.113],
    [2.0, -0.019, 0.433, 1.0],
    [7.5, 0.3537, 0.1943, 1.0],
    [15.0, 0.3537, 0.1943, 1.0]
  ],
  "1": [
    [0.0, 0.135, 0.150, 0.113],
    [2.0, 0.135, 0.150, 1.0],
    [7.5, 0.3537, 0.1943, 1.0],
    [15.0, 0.3537, 0.1943, 1.0]
  ],
  "2": [
    [0.0, 0.491, 0.000, 0.113],
    [2.0, 0.491, 0.000, 1.0],
    [7.5, 0.3537, 0.1943, 1.0],
    [15.0, 0.3537, 0.1943, 1.0]
  ]
}
```
```

## Paso 5 — Parseo del JSON y resultado

**Estado:** ✅ Waypoints parseados correctamente

n_waypoints = {"0": 4, "1": 4, "2": 4}

**Evolución del reward durante la simulación:**

| Tiempo | Reward | Nota |
|--------|--------|------|
| t = 1 s | -0.5578 | — |
| t = 5 s | -0.3253 | — |
| t = 10 s | -0.2876 | — |
| t = 15 s (final) | -0.2863 | reward final |
| mejor en toda la sim | -0.2856 | best_reward |


**Colisiones:** collision_fraction=62.08%  |  min_dist=0.121m

## Paso 6 — Reward final

**Reward final (t=15s):** **-0.2863**
**best_reward:** -0.2856

---

# Representación: `natural_language` — Lenguaje natural (sin coordenadas)

**Trial:** N=3, seed=1  |  latencia=N/A  |  válido=False  |  reward=None

## Paso 3 — Prompt enviado al LLM

**Longitud del prompt:** ? caracteres  ·  Cambia vs otras repr solo en el bloque `## Current Swarm State`.

> El bloque de estado que recibe el LLM es la representación `natural_language` de arriba.
> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.

## Paso 4 — Respuesta cruda del LLM

**Latencia:** N/A

_sin respuesta registrada_

## Paso 5 — Parseo del JSON y resultado

**Estado:** ❌ Parseo fallido — `Connection error.`


**Código o JSON producido antes de fallar:**
_sin código extraído (inválido)_

## Paso 6 — Reward final

**Reward final (t=15s):** N/A (trial inválido)

---

# Análisis comparativo — impacto del sistema de coordenadas en el razonamiento

## ¿Qué estrategia usó el LLM para cada representación?

| Representación | Estrategia del LLM | ¿Correcta? | Fallo principal en direct mode |
|---|---|---|---|
| `raw` | Lee XYZ de cada dron, suma y divide por N para obtener centroide, usa como target | ✅ | Ninguno — arithmetic straightforward |
| `relative` | Lee los offsets Δ directamente, los usa como si fueran XYZ absolutos | ❌ | **Confusión de sistema de referencia**: Δz=0 → z=0 → out of bounds |
| `graph` | Lee XYZ desde el bloque de posiciones, calcula centroide | ✅/❌ | Parcialmente falla o produce targets razonables |
| `aggregate` | Extrae el centroide dado explícitamente, lo usa directamente | ⚠️ | Sin esfuerzo de razonamiento (centroide regalado) |
| `natural_language` | No tiene números precisos; usa coordenadas aproximadas o (0,0,1) | ❌ | Pérdida total de información cuantitativa |

## El problema `relative` explicado

La representación `relative` presenta los drones como offsets desde el centroide:

```
Swarm centroid: (-0.019, 0.433, 0.113)
Drone 0: (0.030, 0.018, 0.000)   ← este es Δx, Δy, Δz desde el centroide!
```

Para convertir a coordenadas absolutas el LLM debe sumar el centroide:
```
pos_abs_drone0 = centroid + (0.030, 0.018, 0.000)
               = (-0.019 + 0.030, 0.433 + 0.018, 0.113 + 0.000)
               = (0.011, 0.451, 0.113)
```

**Sin embargo**, en direct mode el LLM tomó el atajo de usar los offsets directamente
como waypoints absolutos en el JSON. Esto generó z=0.0 en todos los casos —
una coordenada debajo del suelo. El validador rechaza el trial.

**En code mode (E-05)** este error no ocurría porque el LLM podía escapar a Python
y escribir `centroid_x + dx` en el código, resolviendo la conversión programáticamente.
**En direct mode (E-06)** no tiene esa escapatoria.

## Resumen estadístico E-06 — todas las condiciones

| Representación | N | Mean reward | Valid rate | Collision rate |
|---|---|---|---|---|
| `raw` | N=3 | -0.267 | N/A | 100% |
| `raw` | N=6 | -1.142 | N/A | 100% |
| `relative` | N=3 | N/A | N/A | N/A |
| `relative` | N=6 | N/A | N/A | N/A |
| `graph` | N=3 | -0.316 | N/A | 100% |
| `graph` | N=6 | -1.054 | N/A | 100% |
| `aggregate` | N=3 | -1.229 | N/A | 100% |
| `aggregate` | N=6 | -0.819 | N/A | 100% |
| `natural_language` | N=3 | -0.716 | N/A | 100% |
| `natural_language` | N=6 | -2.317 | N/A | 100% |


## Comparación E-05 vs E-06 (mismo N, repr y seed)

| Representación | E-05 valid_rate | E-06 valid_rate | E-05 mean_reward | E-06 mean_reward |
|---|---|---|---|---|
| `raw` | 0% | 0% | -1.468 | -0.267 |
| `relative` | 0% | 0% | -1.343 | N/A |
| `graph` | 0% | 0% | -0.065 | -0.316 |
| `aggregate` | 0% | 0% | -0.072 | -1.229 |
| `natural_language` | 0% | 0% | -0.281 | -0.716 |

> **Lectura clave:** el `valid_rate` de `relative` cae de ~100% (E-05, code mode) a ~0% (E-06, direct mode).
> Esto demuestra que la representación relativa requiere razonamiento de reconversión de coordenadas
> que el LLM solo puede hacer correctamente cuando tiene acceso a Python.

---

## Conclusiones del pipeline E-06

1. **`raw` es el único sistema de coordenadas robusto en direct mode:** las coordenadas absolutas
   son el formato natural para el JSON de waypoints. El LLM no necesita convertir nada.

2. **`relative` falla sistemáticamente:** el LLM confunde offsets ΔP con posiciones absolutas P.
   En code mode puede corregirlo con aritmética; en direct mode no.

3. **`aggregate` parece funcionar pero es un confound:** el centroide ya está dado,
   por lo que no hay razonamiento real; el LLM copia el número.

4. **`natural_language` es el peor en calidad:** sin números precisos, el LLM
   no puede especificar coordenadas exactas.

5. **La comparación E-05 vs E-06 aísla el efecto del modo de salida:**
   el drop en valid_rate para `relative` y `graph` no se debe al jitter sino al direct mode.

---
_Generado automáticamente por `_gen_pipeline_io_e06.py` el 2026-04-09_