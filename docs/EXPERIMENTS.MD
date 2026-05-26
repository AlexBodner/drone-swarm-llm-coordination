# EXPERIMENTS — LLM-Coordinated Drone Swarm

This file is a raw results log and master index. For the hypothesis, design rationale, and analysis of each experiment, see the **per-experiment notebooks** in [`experiments/`](experiments/).

## Quick Index

| ID | Task | N | Output Mode | Status | Notebook |
|----|------|---|-------------|--------|----------|
| **Current** | | | | | |
| E-04 | Rendezvous | 3, 6 | **Direct JSON** | ✅ | [E-04.md](notebooks/E-04-rendezvous-direct.md) |
| E-06 | Rendezvous (Jitter) | 3, 6 | **Direct JSON** | ✅ | — |
| E-07 | Rendezvous (Fixed) | 3, 6 | **Direct JSON** | ✅ | — |
| E-08 | Multi-Task Sweep | 3, 6 | **Direct JSON** | ✅ | — |
| **Archived (Legacy)** | | | | | |
| E-01 | Circle (endpoint) | 3 | Python Code | 📦 Archived | [E-01.md](notebooks/E-01-pilot.md) |
| E-02 | Circle (waypoints) | 3, 6 | Python Code | 📦 Archived | [E-02.md](notebooks/E-02-circle-scale.md) |
| E-03 | Rendezvous | 3, 6 | Python Code | 📦 Archived | [E-03.md](notebooks/E-03-rendezvous.md) |
| E-05 | Code Jitter Test | 3 | Python Code | 📦 Archived | — |

---

---

## Experiment E-01 — Pilot (N=3, All Representations, 5 Seeds)

**Date:** 2026-04-06  
**Run time:** ~21:09–21:10 UTC-3  
**Status:** ✅ Complete

### Configuration

| Parameter          | Value                              |
|-------------------|------------------------------------|
| Task              | Circle formation, radius=2.0m, z=1.0m |
| N_drones          | 3                                  |
| Representations   | raw, relative, graph, aggregate, natural_language |
| Seeds             | 0, 1, 2, 3, 4                      |v
| LLM               | llama-3.3-70b-versatile (Groq)     |
| Temperature       | 0                                  |
| Simulator         | gym-pybullet-drones 2.0.0          |
| Physics           | PyBullet (PYB)                     |
| Drone model       | CF2X (Crazyflie 2.x)               |
| Total trials      | 25                                 |
| API sleep         | 1.0s between trials                |

### Raw Results

| Trial | Representation    | Seed | Valid | Reward    | Latency (s) | Timestamp                       |
|-------|-------------------|------|-------|-----------|-------------|---------------------------------|
| 1     | raw               | 0    | ✅    | -0.0000   | 0.969       | 2026-04-06T21:09:27.156623      |
| 2     | raw               | 1    | ✅    | -0.0000   | 0.758       | 2026-04-06T21:09:29.006522      |
| 3     | raw               | 2    | ✅    | -0.0000   | 0.763       | 2026-04-06T21:09:30.852412      |
| 4     | raw               | 3    | ✅    | -0.0000   | 0.861       | 2026-04-06T21:09:32.796978      |
| 5     | raw               | 4    | ✅    | -0.0000   | 1.047       | 2026-04-06T21:09:34.931496      |
| 6     | relative          | 0    | ✅    | -0.0000   | 0.760       | 2026-04-06T21:09:36.774367      |
| 7     | relative          | 1    | ✅    | -0.0000   | 0.706       | 2026-04-06T21:09:38.562583      |
| 8     | relative          | 2    | ✅    | -0.0000   | 0.714       | 2026-04-06T21:09:40.374981      |
| 9     | relative          | 3    | ✅    | -0.0000   | 0.623       | 2026-04-06T21:09:42.076719      |
| 10    | relative          | 4    | ✅    | -0.0000   | 0.798       | 2026-04-06T21:09:43.956808      |
| 11    | graph             | 0    | ✅    | -0.0000   | 0.527       | 2026-04-06T21:09:45.576269      |
| 12    | graph             | 1    | ✅    | -0.0000   | 0.973       | 2026-04-06T21:09:47.638022      |
| 13    | graph             | 2    | ✅    | -0.0000   | 1.000       | 2026-04-06T21:09:49.723275      |
| 14    | graph             | 3    | ✅    | -0.0000   | 0.720       | 2026-04-06T21:09:51.519790      |
| 15    | graph             | 4    | ✅    | -0.0000   | 0.944       | 2026-04-06T21:09:53.555090      |
| 16    | aggregate         | 0    | ✅    | -0.0000   | 0.576       | 2026-04-06T21:09:55.214949      |
| 17    | aggregate         | 1    | ✅    | -0.0000   | 0.753       | 2026-04-06T21:09:57.052479      |
| 18    | aggregate         | 2    | ✅    | -0.0000   | 0.671       | 2026-04-06T21:09:58.803051      |
| 19    | aggregate         | 3    | ✅    | -0.0000   | 0.857       | 2026-04-06T21:10:00.752131      |
| 20    | aggregate         | 4    | ✅    | -0.0000   | 1.607       | 2026-04-06T21:10:03.485268      |
| 21    | natural_language  | 0    | ✅    | -0.0000   | 0.708       | 2026-04-06T21:10:05.280290      |
| 22    | natural_language  | 1    | ✅    | -0.0000   | 1.141       | 2026-04-06T21:10:07.509300      |
| 23    | natural_language  | 2    | ✅    | -0.0000   | 0.859       | 2026-04-06T21:10:09.455911      |
| 24    | natural_language  | 3    | ✅    | -0.0000   | 0.734       | 2026-04-06T21:10:11.268761      |
| 25    | natural_language  | 4    | ✅    | -0.0000   | 8.956       | 2026-04-06T21:10:21.317124      |

### Aggregate by Representation

| Representation   | Reward (mean±std) | Validity | Prompt Chars | Response Chars | Latency (mean) |
|-----------------|-------------------|----------|--------------|----------------|----------------|
| raw             | 0.0000 ± 0.0000   | 100%     | 1,277        | 359            | 0.88s          |
| relative        | 0.0000 ± 0.0000   | 100%     | 1,331        | 441            | 0.72s          |
| graph           | 0.0000 ± 0.0000   | 100%     | 1,464        | 336            | 0.83s          |
| aggregate       | 0.0000 ± 0.0000   | 100%     | 1,281        | 336            | 0.89s          |
| natural_language| 0.0000 ± 0.0000   | 100%     | 1,305        | 510            | 2.48s          |

### Initial Drone Positions (All Trials — Same per Seed)

All seeds produce nearly identical initial positions (drones spawn in a diagonal line near the origin):

| Drone | x      | y      | z      |
|-------|--------|--------|--------|
| 0     | 0.0000 | 0.0000 | 0.1125 |
| 1     | 0.1588 | 0.1588 | 0.1125 |
| 2     | 0.3176 | 0.3176 | 0.1125 |

Note: `gym-pybullet-drones` seeds did not produce random initial positions at N=3 — all seeds shared the same starting layout. This likely contributes to zero variance in rewards.

---

## Sample LLM Responses (Seed=0 per Representation)

### `raw` Representation

**Prompt state section:**
```
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
```

**LLM Response:**
```python
import math

def plan(state):
    targets = {}
    radius = 2.0
    height = 1.0
    num_drones = state["n_drones"]
    
    for i in range(num_drones):
        angle = 2 * math.pi * i / num_drones
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        targets[i] = (x, y, z)
    
    return targets
```

---

### `relative` Representation

**Prompt state section:**
```
Number of drones: 3
Swarm centroid: (0.159, 0.159, 0.113)
Drone positions relative to centroid (dx, dy, dz):
  Drone 0: (-0.159, -0.159, 0.000)
  Drone 1: (0.000, 0.000, 0.000)
  Drone 2: (0.159, 0.159, 0.000)
```

**LLM Response:**
```python
import math

def plan(state):
    targets = {}
    n_drones = state["n_drones"]
    if n_drones != 3:                           # ⚠️ hardcoded N-guard — will fail at N>3
        raise ValueError("Exactly 3 drones are expected")
    
    radius = 2.0
    height = 1.0
    angle_step = 2 * math.pi / n_drones
    
    for i in range(n_drones):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        targets[i] = (x, y, z)
    
    return targets
```

---

### `graph` Representation

**Prompt state section:**
```
Number of drones: 3
Neighbor radius: 2.0m
Drone positions and neighbors:
  Drone 0 at (0.00, 0.00, 0.11): neighbors = [Drone 1 (0.22m away), Drone 2 (0.45m away)]
  Drone 1 at (0.16, 0.16, 0.11): neighbors = [Drone 0 (0.22m away), Drone 2 (0.22m away)]
  Drone 2 at (0.32, 0.32, 0.11): neighbors = [Drone 0 (0.45m away), Drone 1 (0.22m away)]
```

**LLM Response:**
```python
import math

def plan(state):
    targets = {}
    n_drones = state["n_drones"]
    radius = 2.0
    height = 1.0
    for i in range(n_drones):
        angle = 2 * math.pi * i / n_drones
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        targets[i] = (x, y, height)
    return targets
```

---

### `aggregate` Representation

**Prompt state section:**
```
Number of drones: 3
Swarm centroid: (0.159, 0.159, 0.113)
Position spread (std): (0.130, 0.130, 0.000)
Maximum inter-drone distance: 0.449m
Mean height: 0.113m
```

**LLM Response:**
```python
import math

def plan(state):
    targets = {}
    n_drones = state["n_drones"]
    radius = 2.0
    height = 1.0
    for i in range(n_drones):
        angle = 2 * math.pi * i / n_drones
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        targets[i] = (x, y, height)
    return targets
```

---

### `natural_language` Representation

**Prompt state section:**
```
There are 3 drones currently tightly clustered around their centroid at 
approximately (0.2, 0.2, 0.1) meters. Distribution: 1 drone(s) in the 
north-east, 2 drone(s) in the south-west.
```

**LLM Response:**
```python
import math

def plan(state):
    targets = {}
    n_drones = state["n_drones"]
    if n_drones != 3:                           # ⚠️ hardcoded N-guard
        raise ValueError("Exactly 3 drones are expected")
    
    # Calculate angles for even spacing around the circle
    angles = [2 * math.pi * i / n_drones for i in range(n_drones)]
    
    # Calculate target positions
    for i in range(n_drones):
        x = 2.0 * math.cos(angles[i])
        y = 2.0 * math.sin(angles[i])
        z = 1.0
        targets[i] = (x, y, z)
    
    return targets
```

---

## Experiment E-DEMO-02 — Waypoint Tracking Demo (Option A)

**Date:** 2026-04-06  
**Status:** ✅ Complete  
**Script:** `swarm-llm-experiment/waypoint_demo.py`

### Motivation

E-DEMO-01 revealed a fundamental planning–execution gap: the LLM produced a geometrically perfect plan (reward = 0.0) but the PID controller could not physically reach the targets (execution reward = −1.73). The controller overshoots 2m horizontal moves and loses altitude in the process. This experiment tests **Option A**: instead of a single endpoint per drone, the LLM generates a timed waypoint trajectory `{drone: [(t, x, y, z), ...]}` and the PID tracks an interpolated moving target.

### Configuration

| Parameter | Value |
|-----------|-------|
| Task | Circle formation, radius=2.0m, z=1.0m |
| N_drones | 3 |
| Seed | 42 |
| LLM | llama-3.3-70b-versatile (Groq) |
| Temperature | 0 |
| Duration | 15 s |
| Control rate | 48 Hz |
| LLM output format | `plan(state, duration)` → `{int: [(t,x,y,z), ...]}` |
| Interpolation | Linear between waypoints at each control step |
| Prompt length | 2,089 chars |
| LLM latency | 2.96 s |

### LLM-Generated Waypoints

The LLM wrote code that computes 5 timed waypoints per drone using the rise-first strategy prompted in the system message:

| Drone | t=0s (start) | t=2s (rise) | t=6s (mid) | t=10.5s (approach) | t=15s (target) |
|-------|-------------|-------------|------------|-------------------|----------------|
| 0 | (0.00, 0.00, 0.11) | (0.00, 0.00, 1.00) | (1.00, 0.00, 1.00) | (1.80, 0.00, 1.00) | (2.00, 0.00, 1.00) |
| 1 | (0.16, 0.16, 0.11) | (0.16, 0.16, 1.00) | (−0.42, 0.95, 1.00) | (−0.88, 1.57, 1.00) | (−1.00, 1.73, 1.00) |
| 2 | (0.32, 0.32, 0.11) | (0.32, 0.32, 1.00) | (−0.34, −0.71, 1.00) | (−0.87, −1.53, 1.00) | (−1.00, −1.73, 1.00) |

### Physics Simulation Results

| t (s) | Reward | Notes |
|--------|--------|-------|
| 0.0 | −2.0875 | Start |
| 3.0 | −1.8916 | All drones at z=1.0m, starting horizontal travel |
| 6.0 | −1.1496 | Midpoint waypoints reached |
| 9.0 | −0.5253 | Fine-approach phase |
| 12.0 | −0.0952 | Near formation |
| **15.0** | **−0.0364** | **Final — excellent convergence** |

### Result

| Metric | E-DEMO-01 (single target) | E-DEMO-02 (waypoints) | Improvement |
|--------|--------------------------|----------------------|-------------|
| Best reward | −0.980 | **−0.007** | 132× |
| Final reward | −1.732 | **−0.036** | 47× |
| Reward at 10s | −1.732 (plateau) | **−0.333** | — |
| Formation quality | POOR | **EXCELLENT (<10cm)** | — |
| Drones at target z | ✗ (z≈0.01m) | ✓ (z≈1.0m) | — |

**Video:** `swarm-llm-experiment/videos/waypoint_demo.mp4` (360 frames, 15s, 0.2 MB)

---

## Experiment E-02 — Waypoint Experiment: N × Representation Sweep

**Status:** ✅ Complete  
**Date:** 2026-04-06  
**Prerequisites:** E-DEMO-02 validated ✅

**Configuration:**
- N ∈ {3, 6}
- 5 representations: raw, relative, graph, aggregate, natural_language
- 5 seeds per (N, repr) combination
- LLM output: waypoint trajectories (Option A)
- LLM: llama-3.3-70b-versatile (Groq), temp=0
- Total trials: 50
- Wall time: 3.3 min (196s)
- Physics simulation: 15s per trial, 240Hz physics, 48Hz control

**Results Table:**

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% | Crash% |
|---|-------------------|-------------|--------|-------------|--------|--------|
| 3 | aggregate          |   −0.0418   | 0.0065 |   −0.0076   |  100%  |    0%  |
| 3 | graph              |   −0.0364   | 0.0000 |   −0.0074   |  100%  |    0%  |
| 3 | natural_language   |   −0.0259   | 0.0167 |   −0.0181   |  100%  |    0%  |
| 3 | raw                |   −0.0364   | 0.0000 |   −0.0074   |  100%  |    0%  |
| 3 | relative           |   −0.0255   | 0.0134 |   −0.0081   |  100%  |    0%  |
| 6 | aggregate          |   −0.0264   | 0.0096 |   −0.0113   |  100%  |    0%  |
| 6 | graph              |   −0.0310   | 0.0094 |   −0.0088   |  100%  |    0%  |
| 6 | natural_language   |   −0.0384   | 0.0107 |   −0.0273   |  100%  |    0%  |
| 6 | raw                |   −0.0357   | 0.0000 |   −0.0079   |  100%  |    0%  |
| 6 | relative           |   −0.0269   | 0.0111 |   −0.0086   |  100%  |    0%  |

**Key Findings:**
- 100% code validity across all 50 trials — waypoint format reliably parseable by LLM
- 0% crash rate — PID waypoint tracking is stable across all conditions
- Mean reward −0.025 to −0.042 vs −1.73 baseline (E-DEMO-01): **~47× improvement confirmed at scale**
- `relative` and `natural_language` best at N=3; `aggregate` and `relative` best at N=6
- `graph` and `raw` produce zero variance (std=0.000) — LLM generates identical plans regardless of seed, suggesting deterministic prompt → output mapping
- N=6 slightly outperforms N=3 in most representations (counter-intuitive; likely reward normalization artifact)

**Output files:**
- `results_e02/results.json` — full trial records
- `results_e02/reward_vs_n.png` — mean reward per representation vs N
- `results_e02/validity_rate.png` — validity bar chart
- `results_e02/reward_vs_repr.png` — reward per representation heatmap

---

## Module Test Log

All modules tested on 2026-04-06 before E-01:

| Module             | Test Method                         | Result | Notes                                    |
|-------------------|-------------------------------------|--------|------------------------------------------|
| simulator.py       | `__main__` block, N=3               | ✅ PASS | Positions correct, near-origin           |
| reward.py          | `__main__` block, 5 random drones  | ✅ PASS | Reward -1.74 (random) → 0.00 (perfect)  |
| executor.py        | `__main__` block, 3 test cases     | ✅ PASS | Bug found & fixed (exec scope)           |
| representations.py | `__main__` block, 4-drone state    | ✅ PASS | All 5 repr non-empty, distinct           |
| llm_connector.py   | `__main__` block, Groq              | ✅ PASS | Code block returned correctly            |
| prompt_builder.py  | `__main__` block                    | ✅ PASS | Circle task prompt correctly formatted   |
| experiment.py      | Full pilot run (E-01)               | ✅ PASS | 25/25 trials completed                   |
| analysis.py        | `results_pilot/results.json`       | ✅ PASS | Table + 2 plots generated                |

---

---

## Experiment E-03 v2 — Rendezvous Task (Corrected Reward)

**Status:** 🔄 Running  
**Date:** 2026-04-07  
**Prerequisites:** E-02 ✅, reward bug fix ✅  
**Full notebook:** [experiments/E-03-rendezvous.md](experiments/E-03-rendezvous.md)

### What Changed from v1

| | v1 (stale) | v2 (current) |
|--|------------|---------------|
| Reward | `−mean(dist_to_centroid)` | `−0.5×pairwise_dist − 0.5×centroid_drift` |
| Problem | All drones target same point → collision | Rewards tight cluster near centroid |
| LLM model | llama-3.3-70b-versatile (100K TPD) | llama-3.1-8b-instant (500K TPD) |
| Token limit | Hit at trial 83 | No hit expected |

### Configuration

| Parameter         | Value                                               |
|-------------------|-----------------------------------------------------|
| Task              | Rendezvous at initial centroid, z=1.0m              |
| Planner           | Waypoint trajectories                               |
| N_drones          | 3, 6                                               |
| Representations   | raw, relative, graph, aggregate, natural_language   |
| Seeds             | 0–9 (10 seeds)                                      |
| LLM               | llama-3.1-8b-instant (Groq), T=0                   |
| Duration          | 15s per trial                                       |
| Total trials      | 100                                                 |
| Videos            | 1 per (N, repr) condition                           |

### Results (v2 — Final)

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% | Crash% |
|---|--------------------|-------------|--------|-------------|--------|--------|
| 3 | aggregate          |   −0.0860   | 0.0000 |   −0.0840   |  100%  |    0%  |
| 3 | graph              |   −0.0856   | 0.0000 |   −0.0846   |   50%  |    0%  |
| 3 | natural_language   |   −1.0652   | 0.0000 |   −0.0700   |  100%  |    0%  |
| 3 | raw                |   −0.5798   | 0.3498 |   −0.0756   |   90%  |    0%  |
| 3 | relative           |   −0.7794   | 0.2314 |   −0.0716   |  100%  |    0%  |
| 6 | graph              |   −1.0148   | 0.0000 |   −0.1305   |  100%  |    0%  |
| 6 | relative           |   −1.0491   | 0.1551 |   −0.1424   |  100%  |    0%  |
| 6 | raw                |   −1.1249   | 0.5075 |   −0.1361   |  100%  |    0%  |
| 6 | aggregate          |   −1.1439   | 0.0000 |   −0.1378   |  100%  |    0%  |
| 6 | natural_language   |   −1.3527   | 0.0000 |   −0.1282   |  100%  |    0%  |

**Wall time:** 18.3 min (1099s)  
**Full analysis:** [experiments/E-03-rendezvous.md](experiments/E-03-rendezvous.md)

---

## Experiment E-04 — Rendezvous via Direct JSON Waypoints

**Date:** 2026-04-07  
**Status:** ✅ Complete  
**Full notebook:** [experiments/E-04-rendezvous-direct.md](experiments/E-04-rendezvous-direct.md)

### Motivation

E-03 revealed that LLMs solve rendezvous by offloading centroid computation to the Python interpreter (`sum([...]) / n`). E-04 removes code execution entirely — the LLM must output a raw JSON waypoint dict after reasoning through the arithmetic in plain language.

### Configuration

| Parameter | Value |
|-----------|-------|
| Task | Rendezvous (`reward_mode="rendezvous"`) |
| `output_mode` | `"direct"` (JSON, no code exec) |
| N_drones | 3, 6 |
| Representations | raw, relative, graph, aggregate, natural_language |
| Seeds | 0–9 (10 per condition) |
| Total trials | 100 |
| LLM | llama-3.1-8b-instant (Groq) |
| Temperature | 0 |
| Duration | 15 s |
| Collision radius | 0.25 m |

### Raw Results

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% | Crash% | Col% | MinDist |
|---|--------------------|-------------|--------|-------------|--------|--------|------|---------|
| 3 | aggregate          |   −0.8780   | 0.0000 |   −0.0702   |  100%  |    0%  | 100% | 0.073m  |
| 3 | graph              |   −0.7372   | 0.0000 |   −0.0681   |  100%  |    0%  | 100% | 0.058m  |
| 3 | natural_language   |   −0.9757   | 0.1301 |   −0.2299   |  100%  |    0%  | 100% | 0.067m  |
| 3 | raw                |   −0.9600   | 0.0000 |   −0.0695   |  100%  |    0%  | 100% | 0.075m  |
| 3 | relative           |      N/A    |  N/A   |     N/A     |    0%  |    0%  | N/A  |   N/A   |
| 6 | aggregate          |      N/A    |  N/A   |     N/A     |    0%  |    0%  | N/A  |   N/A   |
| 6 | graph              |   −0.7762   | 0.0000 |   −0.1751   |  100%  |    0%  | 100% | 0.044m  |
| 6 | natural_language   |   −0.7158   | 0.0000 |   −0.4986   |  100%  |    0%  | 100% | 0.033m  |
| 6 | raw                |   −2.0096   | 0.0000 |   −0.1710   |  100%  |    0%  | 100% | 0.074m  |
| 6 | relative           |      N/A    |  N/A   |     N/A     |    0%  |    0%  | N/A  |   N/A   |

**Overall validity:** 70/100 = **70.0%**  
**Wall time:** 34.5 min (2072s)  
**Results file:** `swarm-llm-experiment/results_e04/results.json`

### Key Findings

- Direct JSON output exposes representation-dependent arithmetic failures that were masked in code mode.
- `relative` collapses completely (0% valid at N=3 and N=6), indicating coordinate-frame conversion errors in language reasoning.
- `aggregate` is brittle at N=6 (0% valid), likely due to malformed/invalid waypoint JSON plans.
- `raw`, `graph`, and `natural_language` remain 100% valid where parse succeeds, but still collide heavily (100% collision-trial rate).
- `raw` at N=6 is the worst-performing valid condition (mean reward −2.0096), showing severe convergence failure despite valid JSON.

---

## Known Issues / Observations

| ID  | Issue                                              | Severity | Status      |
|-----|----------------------------------------------------|----------|-------------|
| I-01 | `relative`+`natural_language` add hardcoded N-guard | Medium  | Monitored — will cause failures at N>3 |
| I-02 | All seeds produce identical initial positions (N=3) | Low      | Expected for gym-pybullet-drones default init |
| I-03 | `natural_language` latency spike (8.9s, trial 25) | Low      | Rate-limit or network jitter; retry logic can help |
| I-04 | Reward always 0.0000 at N=3                       | Informational | Task too easy; use N≥6 for differentiation |
| I-05 | `exec` globals scoping bug in executor.py          | High     | Fixed: changed `exec(code,{},ns)` → `exec(code,ns)` |
| I-06 | PID overshoot + altitude loss at single 2m targets | High     | Resolved by Option A waypoint trajectories (E-DEMO-02) |
| I-07 | `env.reset()` returns 2 values not 5 in CtrlAviary | Medium   | Fixed: use `obs, _ = env.reset(seed=...)` pattern |
| I-09 | Groq daily token limit (100K TPD) exhausted mid-experiment | High | **Fixed:** switched to `llama-3.1-8b-instant` (500K TPD) |
| I-10 | `natural_language` repr: 10% valid on rendezvous task (v1) | High | Under investigation; may improve with corrected prompt in v2 |
| I-11 | LLM generates waypoint at t=15.75 > duration=15.0 | Low | Observed in E-03 v2 trial 6; executor raises AssertionError; consider clamping |
| I-12 | Bimodal reward pattern in E-03 v2 (−0.83 vs −0.085) | Info | Two plan classes observed — LLM either finds centroid or falls back to fixed point |
| I-13 | E-04 post-run plotting error in `plot_collision_rate` (`TypeError: list + float`) | Medium | Results saved; summary tables valid; fix pending in `analysis.py` |
