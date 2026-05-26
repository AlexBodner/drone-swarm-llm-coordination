# Simulator Reference Guide
## gym-pybullet-drones for LLM-Drone Swarm Experiments

**Audience:** Research supervisors and thesis committees planning or evaluating experiments  
**Scope:** Architecture, I/O specification, scene generation, goal definition, task completion metrics, LLM integration, and experimental design considerations

---

## Table of Contents

1. [What is gym-pybullet-drones?](#1-what-is-gym-pybullet-drones)
2. [Architecture Overview](#2-architecture-overview)
3. [Simulator Inputs — What You Control](#3-simulator-inputs--what-you-control)
4. [Simulator Outputs — What You Observe](#4-simulator-outputs--what-you-observe)
5. [Generating Scenes (Initial Conditions)](#5-generating-scenes-initial-conditions)
6. [Defining Tasks and Goals](#6-defining-tasks-and-goals)
7. [Measuring Task Completion](#7-measuring-task-completion)
8. [How the LLM Interacts with the Simulator](#8-how-the-llm-interacts-with-the-simulator)
9. [Physics Options and Fidelity Levels](#9-physics-options-and-fidelity-levels)
10. [Scaling Considerations](#10-scaling-considerations)
11. [Experiment Design Recommendations](#11-experiment-design-recommendations)
12. [Quick Reference: API Cheatsheet](#12-quick-reference-api-cheatsheet)

---

## 1. What is gym-pybullet-drones?

[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) is an open-source Python library that wraps the **PyBullet** physics engine in a **Gymnasium**-compatible interface for multi-drone simulation. It was developed at the University of Toronto Autonomous Systems Laboratory and is purpose-built for robotics and reinforcement learning research.

**Key properties relevant to this thesis:**
- **Rigid-body physics** via PyBullet: gravity, drag, propeller thrust, ground effects
- **Drone model:** Bitcraze Crazyflie 2.x (CF2X) — a real quadrotor with verified dynamics
- **Multi-agent:** any number N of simultaneous drones with independent dynamics  
- **Headless-capable:** runs without a GUI (`gui=False`), enabling server/cluster execution
- **Deterministic:** same seed → same trajectory (essential for reproducibility)
- **Python-native:** all interactions through standard NumPy arrays and Python dicts

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT PIPELINE                                 │
│                                                                             │
│  ┌──────────────┐     ┌────────────────────┐     ┌──────────────────────┐  │
│  │   Simulator  │────▶│  State Extractor   │────▶│  Representation Fn  │  │
│  │  (PyBullet)  │     │  (parse obs dict)  │     │  raw/relative/graph │  │
│  └──────────────┘     └────────────────────┘     └──────────┬───────────┘  │
│         ▲                                                    │              │
│         │ RPM actions                                        ▼              │
│  ┌──────┴───────┐     ┌────────────────────┐     ┌──────────────────────┐  │
│  │  PID Control │◀────│  Plan Executor     │◀────│   Prompt Builder    │  │
│  │  (per drone) │     │  (exec() sandbox)  │     │   (state + task)    │  │
│  └──────────────┘     └────────────────────┘     └──────────┬───────────┘  │
│                                                              │              │
│                                                              ▼              │
│                                              ┌───────────────────────────┐  │
│                                              │         LLM API           │  │
│                                              │  (Groq/Gemini/Claude)     │  │
│                                              └───────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Reward Calculator (Hungarian algorithm + formation_reward)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Execution flow for one trial:**
1. `SwarmSimulator.reset(seed)` → spawns N drones, returns initial state
2. `repr_fn(state)` → converts state to a text string for the LLM
3. `build_prompt(state_text, task)` → wraps state + task description in structured prompt
4. `get_completion(prompt)` → LLM generates a Python `plan(state)` function
5. `execute_plan_code(code, state)` → executes the plan, returns `{drone_id: (x,y,z)}`
6. `formation_reward(final_positions, targets)` → computes reward (negative mean distance)

This is an **open-loop** system: the LLM generates one plan, the plan executes, the reward is measured. There is no feedback during execution in Phase 1.

---

## 3. Simulator Inputs — What You Control

### 3.1 At construction time (`CtrlAviary.__init__`)

| Parameter         | Type               | Default  | Description |
|------------------|--------------------|----------|-------------|
| `drone_model`    | `DroneModel` enum  | CF2X     | Physical drone model (affects mass, thrust, inertia) |
| `num_drones`     | `int`              | 1        | N, the swarm size |
| `physics`        | `Physics` enum     | PYB      | Physics fidelity level (see §9) |
| `initial_xyzs`   | `ndarray (N,3)`    | diagonal | Starting positions for all drones |
| `initial_rpys`   | `ndarray (N,3)`    | zeros    | Starting roll/pitch/yaw for all drones |
| `gui`            | `bool`             | False    | Open PyBullet GUI window |
| `record`         | `bool`             | False    | Save onboard camera PNGs |
| `obstacles`      | `bool`             | False    | Add static obstacles to the scene |
| `output_folder`  | `str`              | `results`| Where to save logs/frames |

### 3.2 At each step (`env.step(action)`)

The action is an `ndarray` of shape `(N, 4)`:

```
action[drone_i] = [rpm_motor_0, rpm_motor_1, rpm_motor_2, rpm_motor_3]
```

**Motor RPM ranges:**
- Hover RPM: ≈ 14,468 RPM (computed from `sqrt(mg / 4kf)`)
- Maximum RPM: ≈ 21,702 RPM
- Minimum RPM: 0 (motor off)

You rarely set RPMs directly. Instead, you use a **PID controller** (`DSLPIDControl`) that converts a desired 3D target position into the correct RPM commands:

```python
rpm, pos_err, yaw_err = controller.computeControl(
    control_timestep = 1/48,        # seconds
    cur_pos          = obs[i, 0:3], # current (x, y, z)
    cur_quat         = obs[i, 3:7], # orientation quaternion
    cur_vel          = obs[i, 10:13], # velocity
    cur_ang_vel      = obs[i, 13:16], # angular velocity
    target_pos       = np.array([x_target, y_target, z_target]),
)
```

**This is the bridge between the LLM's high-level target positions and the low-level motor commands the physics engine requires.**

---

## 4. Simulator Outputs — What You Observe

### 4.1 Observation vector layout

`env.step(action)` and `env.reset()` both return `obs` of shape `(N, 20)`.

Each drone gets a 20-element state vector:

```
obs[i] = [x, y, z,  qx, qy, qz, qw,  r, p, ya,  vx, vy, vz,  wx, wy, wz,  rpm0, rpm1, rpm2, rpm3]
          └──┬──┘   └────────┬────┘   └───┬───┘   └──────┬───┘  └──────┬──┘   └──────────┬──────┘
          position        quaternion    Euler          linear vel   angular vel      last motor RPMs
          [0:3]           [3:7]         [7:10]         [10:13]      [13:16]          [16:20]
```

| Index range | Variable | Unit   | Notes |
|------------|----------|--------|-------|
| `[0:3]`    | x, y, z  | meters | World-frame position |
| `[3:7]`    | qx,qy,qz,qw | —   | Orientation quaternion (PyBullet convention) |
| `[7:10]`   | roll, pitch, yaw | radians | Euler angles, derived from quaternion |
| `[10:13]`  | vx, vy, vz | m/s  | Linear velocity in world frame |
| `[13:16]`  | wx, wy, wz | rad/s | Angular velocity in body frame |
| `[16:20]`  | rpm₀…rpm₃  | RPM  | Last commanded motor speeds |

### 4.2 Additional return values from `env.step()`

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| Variable     | Type              | Notes |
|-------------|-------------------|-------|
| `obs`       | `ndarray (N,20)`  | Full drone state (see above) |
| `reward`    | `float`           | Built-in reward (0.0 by default in CtrlAviary — we use our own) |
| `terminated`| `bool`            | True if a drone crashes (z < 0.01m) |
| `truncated` | `bool`            | True if episode time limit reached |
| `info`      | `dict`            | Currently empty in CtrlAviary |

### 4.3 Camera output

The PyBullet client exposes a camera API that can be called at any step:

```python
width, height, rgb, depth, segmentation = p.getCameraImage(
    width=1280, height=720,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_TINY_RENDERER,       # headless-compatible
    physicsClientId=env.CLIENT,
)
```

This is used in `record_simulation.py` to generate the videos in `swarm-llm-experiment/videos/`.

---

## 5. Generating Scenes (Initial Conditions)

### 5.1 Default initial layout

By default, drones spawn in a **diagonal line** along the X=Y axis, spaced 4× the arm length apart (≈0.159m), all at the minimum collision-safe height (~0.112m):

```
Drone 0: (0.000, 0.000, 0.112)
Drone 1: (0.159, 0.159, 0.112)
Drone 2: (0.318, 0.318, 0.112)
...
```

This tight clustering at the origin means all seeds produce **similar initial states** for small N, which explains the zero variance observed in the pilot experiment.

### 5.2 Custom initial positions

Pass `initial_xyzs` as an `(N,3)` NumPy array:

```python
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Randomized positions in a 4m² area
rng = np.random.default_rng(seed=42)
initial_xyzs = np.column_stack([
    rng.uniform(-2.0, 2.0, size=6),   # x ∈ [-2, 2]
    rng.uniform(-2.0, 2.0, size=6),   # y ∈ [-2, 2]
    np.full(6, 0.5),                   # all at z = 0.5m
])

env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=6,
    initial_xyzs=initial_xyzs,
    physics=Physics.PYB,
    gui=False,
)
```

### 5.3 Initial orientations

All drones face the same direction by default (`rpy = [0,0,0]`). To randomize yaw:

```python
initial_rpys = np.column_stack([
    np.zeros(N),                             # roll = 0
    np.zeros(N),                             # pitch = 0
    rng.uniform(-np.pi, np.pi, size=N),      # random yaw
])
env = CtrlAviary(..., initial_rpys=initial_rpys)
```

### 5.4 Obstacles

Setting `obstacles=True` adds a few static geometric obstacles to the PyBullet world. Custom obstacles can be added directly via the PyBullet API after environment creation:

```python
env = CtrlAviary(..., obstacles=False)
obs, _ = env.reset()

# Add a box obstacle at (1.0, 0.0, 0.5) with dimensions 0.5×0.5×1.0m
import pybullet as p
box_id = p.createMultiBody(
    baseMass=0,   # static
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5]),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5], rgbaColor=[1,0,0,0.8]),
    basePosition=[1.0, 0.0, 0.5],
    physicsClientId=env.CLIENT,
)
```

### 5.5 Recommended scene variations for experiments

| Variation | `initial_xyzs` strategy | Purpose |
|-----------|------------------------|---------|
| Default diagonal | None (gym default) | Baseline, reproducible |
| **Random spread** | Uniform in `[-R, R]² × {h}` | Tests if representation captures spatial spread |
| Ring start | N drones on a large circle | Tests convergence from known geometry |
| Clustered random | N drones in a small sphere | Tests whether LLM can decompose from crowded state |
| Grid | Regular grid layout | Structured starting point |

**Recommendation:** For the main experiment, use random spread (`R=3.0m`) so the LLM actually has to reason about each drone's specific position. The default diagonal causes all drones to start almost at the same point, hiding differences between representations.

---

## 6. Defining Tasks and Goals

### 6.1 What does "task" mean in this system?

A **task** is a natural language description given to the LLM. The LLM must generate a `plan(state)` function that returns target positions `{drone_id: (x, y, z)}` that, when achieved, constitute completing the task.

The task description is entirely in the **prompt** — the simulator itself has no concept of objectives. The objectives are operationalized in `reward.py`.

### 6.2 Task 1: Circle Formation

```
Move the drones to form a circle of radius 2.0 meters centered at the origin (0,0),
at height z=1.0 meters. The drones should be evenly spaced around the circle.
```

**Ground-truth targets:** Generated analytically —
```python
angles = np.linspace(0, 2π, N, endpoint=False)
targets[i] = (R·cos(angles[i]), R·sin(angles[i]), z)
```

**What the LLM must infer:** The formula for N evenly-spaced points on a circle.

### 6.3 Task 2: Rendezvous (Convergence)

```
Move all drones to converge to the swarm centroid position.
All drones should end up at approximately the same location, hovering at z=1.0m.
```

**Ground-truth targets:** All drones assigned `(centroid_x, centroid_y, 1.0)`.

**What the LLM must infer:** How to compute the centroid from the state.

### 6.4 Task 3: Line Formation (suggested for next phase)

```
Arrange the drones in a straight line along the X-axis,
evenly spaced 1.0 meter apart, at height z=1.0m.
```

### 6.5 Task 4: Grid Formation (suggested for next phase)

```
Arrange N drones in a square grid. Each drone occupies one cell of the grid,
with 1.5m spacing between adjacent drones, at height z=1.0m.
The grid should be centered at the origin.
```

**This is a good stress test for representations:** The LLM must compute a 2D grid layout, which requires reasoning about `int(sqrt(N))` rows and columns. At N=9 or N=16, the LLM may fail if the representation doesn't clearly state individual positions.

### 6.6 Adding a new task: step-by-step

1. Add the text description to `prompt_builder.py`:
   ```python
   TASK_LINE = "Arrange drones in a straight line along the X-axis, 1.0m apart, at z=1.0m."
   ```

2. Add a target generator to `reward.py`:
   ```python
   def line_formation_targets(n: int, spacing: float = 1.0, height: float = 1.0) -> list:
       return [(i * spacing - (n-1) * spacing / 2, 0.0, height) for i in range(n)]
   ```

3. Pass the new task to `run_trial()`:
   ```python
   result = run_trial(n_drones=6, representation_name="raw", seed=0,
                      task_description=TASK_LINE)
   ```

---

## 7. Measuring Task Completion

### 7.1 Formation reward (current metric)

The reward is the **negative mean Euclidean distance** after **optimal assignment** via the Hungarian algorithm:

```
reward = − (1/N) · Σᵢ ‖final_pos[assignment[i]] − target[i]‖₂
```

- **Range:** (−∞, 0]
- **Perfect formation:** reward = 0.0
- **Higher is always better**
- The Hungarian algorithm ensures drones are assigned to the nearest target slots, removing the ambiguity of which drone goes where.

Example: `reward = -0.15` means the average drone is 15 cm from its optimal target slot.

### 7.2 Additional completion metrics

These are **not yet implemented** but recommended for the main experiment:

#### Binary success rate
```python
def success_rate(results: list, threshold: float = 0.3) -> float:
    """Fraction of trials where mean distance < threshold meters."""
    successes = [r for r in results if r["reward"] is not None and abs(r["reward"]) < threshold]
    return len(successes) / len(results)
```

#### Code validity rate
```python
validity_rate = sum(1 for r in results if r["valid_code"]) / len(results)
```
This is already tracked in the current `experiment.py`.

#### Normalized reward
```python
def normalized_reward(reward: float, n_drones: int, task_radius: float = 2.0) -> float:
    """Reward normalized by task radius, so it's comparable across tasks and N."""
    return reward / task_radius   # -1.0 = average offset equals the formation radius
```

#### Variance across seeds
```python
import numpy as np
rewards = [r["reward"] for r in trials if r["reward"] is not None]
variance = np.var(rewards)
```
High variance = representation produces inconsistent plans; low variance = consistent.

### 7.3 Success criterion thresholds

| Threshold | Interpretation |
|-----------|----------------|
| reward > −0.1 | Excellent: drones within 10cm of targets |
| reward > −0.3 | Good: within 30cm (practical formation) |
| reward > −1.0 | Partial: formation recognizable but imprecise |
| reward ≤ −1.0 | Poor: formation failed |

For the **circle formation** (radius=2.0m), a reward of −1.0 means the average drone is half the radius away from its target — a clearly failed formation.

### 7.4 Completion vs. planning quality

**Important distinction:**
- The *current* experiment measures **planning quality** (how good is the LLM's target prescription?), not physical execution.
- In Phase 1, `execute_plan` simply reads back the targets without simulating flight.
- In Phase 2 (with PID control), the **execution reward** will differ from the **planning reward** because the drone dynamics may prevent reaching targets precisely.

The gap between planning reward and execution reward is itself a research finding: "How much does physics degrade the LLM's ideal plan?"

---

## 8. How the LLM Interacts with the Simulator

### 8.1 The full interaction model

```
State (N×20 floats)
    │
    ▼  repr_fn(state) → text
Representation
    │
    ▼  build_prompt(state_text, task, N) → prompt string
Prompt
    │
    ▼  LLM API call (1 call per trial, temperature=0)
Code string (Python)
    │
    ▼  extract_code_block() → clean Python
    │  execute_plan_code(code, state) → {0: (x,y,z), 1: (x,y,z), ...}
Target positions
    │
    ▼  formation_reward(targets, ground_truth)
Reward (scalar ∈ [−∞, 0])
```

The LLM has **exactly one chance** per trial, with **no feedback**. It reads the state, writes a plan, and that's it.

### 8.2 Abstraction levels

The LLM operates at the **geometric planning** level, not the control level:

| Level | Handled by | Example |
|-------|-----------|---------|
| **Geometric planning** | LLM | "Drone 0 should go to (2.0, 0.0, 1.0)" |
| **Trajectory planning** | (Future) LLM or planner | "Follow path from current pos to target" |
| **Control** | DSLPIDControl | Converts waypoint to RPM commands |
| **Physics** | PyBullet | Integrates RPMs → forces → motion |

This separation is deliberate: it isolates the LLM's planning ability from drone control complexity.

### 8.3 What information the LLM can see

Depending on the representation, the LLM has access to:

| Representation | Position info | Velocity | Relative structure | Abstract stats |
|---------------|--------------|----------|--------------------|----------------|
| `raw`         | ✅ Full XYZ per drone | ❌ | ❌ | ❌ |
| `relative`    | ✅ Delta from centroid | ❌ | Partial | Centroid |
| `graph`       | ✅ XYZ + neighbor distances | ❌ | ✅ | ❌ |
| `aggregate`   | ❌ No per-drone | ❌ | ❌ | ✅ mean, std, spread |
| `natural_language` | ❌ No precise coords | ❌ | Quadrant counts | ✅ Verbal |

**Key design insight:** At N=3, the LLM doesn't need any state information to form a circle — the formula is purely geometric. The representations become discriminating only when N is large enough that the LLM must reason about *which* drone goes *where* (assignment problem).

### 8.4 What information is hidden from the LLM

- **Velocities:** Not included in any representation (by design)
- **Quaternions / RPY angles:** Not included (drones assumed to be level)
- **Motor RPMs:** Not included
- **Time elapsed / step count:** Not included
- **Physics parameters** (mass, thrust coefficient): Not included

These could be added in future representations to study whether richer context helps.

### 8.5 Closing the loop (Phase 2 — future work)

In Phase 1, the LLM gets one shot. A closed-loop variant would:

```python
for iteration in range(max_iterations):
    state = sim.get_state()
    state_text = repr_fn(state)
    reward = measure_current_reward(state, targets)
    feedback_text = f"Current reward: {reward:.3f}. Current state: {state_text}"
    prompt = build_refinement_prompt(feedback_text, task)
    new_plan = llm(prompt)
    sim.execute_plan(new_plan)
```

This turns the experiment into an **LLM-in-the-loop controller**, analogous to MPC with an LLM as the prediction model. This is Phase 2 of the thesis.

---

## 9. Physics Options and Fidelity Levels

`gym-pybullet-drones` supports 5 physics backends, controlled by the `Physics` enum:

| Mode | Code | Ground Effect | Motor Drag | Downwash | Notes |
|------|------|:---:|:---:|:---:|-------|
| `PYB` | `Physics.PYB` | ❌ | ❌ | ❌ | **Default. Fastest.** Pure rigid-body, no aerodynamics |
| `DYN` | `Physics.DYN` | ❌ | ❌ | ❌ | Analytical dynamics instead of PyBullet solver |
| `PYB_GND` | `Physics.PYB_GND` | ✅ | ❌ | ❌ | Adds ground effect near the floor |
| `PYB_DRAG` | `Physics.PYB_DRAG` | ❌ | ✅ | ❌ | Adds motor drag model |
| `PYB_DW` | `Physics.PYB_DW` | ❌ | ❌ | ✅ | Adds downwash between drones |
| `PYB_GND_DRAG_DW` | `Physics.PYB_GND_DRAG_DW` | ✅ | ✅ | ✅ | Slowest, most realistic |

For thesis experiments, `Physics.PYB` (default) is appropriate. The LLM's planning quality doesn't depend on aerodynamics accuracy — it depends on the state representation, and `PYB` gives deterministic, fast simulations.

**Simulation frequencies:**
- Physics: 240 Hz (default)
- Control: 48 Hz (every 5 physics steps)  
- Recording: 30 fps (every 8 physics steps)

---

## 10. Scaling Considerations

### 10.1 Computational cost

| N drones | Time per trial | Memory (approx) |
|----------|---------------|-----------------|
| 3        | ~2s per sim   | ~100 MB         |
| 6        | ~3s per sim   | ~150 MB         |
| 10       | ~5s per sim   | ~200 MB         |
| 15       | ~8s per sim   | ~280 MB         |

The bottleneck is PyBullet (C++ physics) — negligible at all scales in this study. The real bottleneck is **LLM API latency**: 0.7–9s per call.

### 10.2 Prompt token growth by representation

| Representation | Token growth with N | Concern |
|---------------|---------------------|---------|
| `raw`         | O(N) — linear       | None until N≈100 |
| `relative`    | O(N)                | None |
| `graph`       | O(N²) — quadratic   | Hits limits around N=50 |
| `aggregate`   | O(1) — constant     | Too compressed at N≥6? |
| `natural_language` | O(1)           | Loses precision early |

For N=15 with `graph` representation, each drone lists up to 14 neighbors → the prompt could exceed 2,000 tokens. For a 128K-context model this is trivial, but it counts toward inference cost.

### 10.3 LLM failure modes at scale (expected)

| N | Expected failure mode |
|---|----------------------|
| 3 | None — trivial formula |
| 6 | LLM hardcodes N=6; edge case |
| 10 | Incorrect variable indexing; off-by-one |
| 15 | LLM outputs formation for wrong N; fails validation |

These failures are the **signal** of the experiment — they reveal at which N each representation breaks down.

---

## 11. Experiment Design Recommendations

### 11.1 Minimum viable experiment

For a thesis experiment to be statistically meaningful:

| Factor | Recommendation | Rationale |
|--------|---------------|-----------|
| Seeds per cell | ≥ 10 | Enough to compute stable mean and std |
| N values | {3, 6, 10, 15} | Covers easy → hard transition |
| Representations | All 5 | Core independent variable |
| Tasks | ≥ 2 (circle + rendezvous) | Cross-validate results |
| LLMs | ≥ 2 (Groq + Claude) | Check for LLM-specific effects |

**Total trials (minimum):** 4 N × 5 repr × 10 seeds × 2 tasks = **400 trials**

### 11.2 Statistical analysis plan

For comparing representations at each N:
- **Test:** Welch's t-test (unequal variances are expected)
- **Effect size:** Cohen's d
- **Threshold:** `|reward_repr_A - reward_repr_B| > noise_std`

For measuring the effect of N:
- **Test:** Spearman correlation between N and reward (monotone decay expected)
- **Expected finding:** reward degrades faster for `aggregate` and `natural_language` than for `raw`

### 11.3 Controlling confounders

| Confounder | How to control |
|-----------|---------------|
| LLM non-determinism | `temperature=0` (fully greedy) |
| Different initial positions | Same seeds across all representations |
| Task difficulty variation | Same target positions for same seed |
| Order effects | Randomize trial order in `run_experiment()` |
| Rate limits | `time.sleep(1.0)` between API calls |

### 11.4 Pilot experiment validation criteria (from execution plan)

Before scaling up:
1. Code validity rate > 80% for all representations → prompts are working
2. Reward values vary across seeds → task is non-trivial
3. Reward values differ across representations → representations have signal

In E-01 (N=3, 5 seeds), all 3 criteria are satisfied but criteria 2 and 3 are not yet meaningful (all rewards are 0.0). **This is expected** — increase N.

### 11.5 What to look for in the data

**Primary finding candidates:**

1. **Representation × N interaction:** Does `raw` maintain higher reward than `aggregate` as N grows? This would confirm the hypothesis that richer state descriptions enable better planning.

2. **Code validity crossover:** At what N does each representation start producing invalid code? `relative` and `natural_language` are expected to fail first due to N-hardcoding.

3. **Latency divergence:** Does `natural_language` become significantly slower at large N? (Longer LLM output → higher latency → rate limit risk)

4. **Response strategy audit:** What algorithm does the LLM use for each representation? (Circle formula? Sorted assignment? Random assignment?) This is qualitative analysis of the raw LLM responses in `results.json`.

---

## 12. Quick Reference: API Cheatsheet

### Complete single-trial example with PID execution

```python
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

N = 6
SIM_HZ = 240
CTRL_HZ = 48
DURATION_S = 10.0

# 1. Create environment with random initial positions
rng = np.random.default_rng(seed=42)
init_xyzs = np.column_stack([
    rng.uniform(-2.0, 2.0, N),
    rng.uniform(-2.0, 2.0, N),
    np.full(N, 0.5),
])
env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=N,
    initial_xyzs=init_xyzs, physics=Physics.PYB, gui=False,
)
obs, _ = env.reset(seed=42)

# 2. Define targets (circle formation)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
targets = np.column_stack([2.0*np.cos(angles), 2.0*np.sin(angles), np.ones(N)])

# 3. PID controllers
controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N)]
actions = np.zeros((N, 4))
CTRL_EVERY = int(SIM_HZ / CTRL_HZ)

# 4. Simulation loop
for step in range(int(DURATION_S * SIM_HZ)):
    if step % CTRL_EVERY == 0:
        for i in range(N):
            rpm, _, _ = controllers[i].computeControl(
                control_timestep=1/CTRL_HZ,
                cur_pos=obs[i, 0:3], cur_quat=obs[i, 3:7],
                cur_vel=obs[i, 10:13], cur_ang_vel=obs[i, 13:16],
                target_pos=targets[i],
            )
            actions[i] = rpm
    obs, _, terminated, _, _ = env.step(actions)
    if terminated:
        print("Drone crashed!")
        break

# 5. Measure final reward
final_positions = {i: tuple(obs[i, 0:3]) for i in range(N)}
from reward import formation_reward
reward = formation_reward(final_positions, [tuple(t) for t in targets])
print(f"Final reward: {reward:.4f}")
env.close()
```

### Observation space at a glance

```python
obs[i, 0]    # x position           obs[i, 10]  # vx velocity
obs[i, 1]    # y position           obs[i, 11]  # vy velocity
obs[i, 2]    # z position (height)  obs[i, 12]  # vz velocity
obs[i, 3:7]  # quaternion          obs[i, 13:16]# angular velocity
obs[i, 7]    # roll                obs[i, 16:20]# motor RPMs
obs[i, 8]    # pitch
obs[i, 9]    # yaw
```

### Useful constants (CF2X Crazyflie drone)

| Property | Value | Meaning |
|----------|-------|---------|
| Mass | 27g | 0.027 kg |
| Arm length | 39.7mm | 0.0397 m |
| Hover RPM | ≈14,468 | RPM at which drone hovers |
| Max RPM | ≈21,702 | Maximum motor speed |
| Thrust-to-weight | 2.25 | Can lift 2.25× its own weight |
| Max speed | 30 km/h | Physical limit |

---

## Videos

Three simulation recordings are available in `swarm-llm-experiment/videos/`:

| File | Scenario | N | Duration | Description |
|------|----------|---|----------|-------------|
| `circle_n3.mp4` | Circle Formation | 3 | 8s | 3 drones moving from diagonal start to circle |
| `circle_n6.mp4` | Circle Formation | 6 | 8s | 6 drones forming a larger circle |
| `rendezvous.mp4` | Rendezvous | 5 | 8s | 5 drones converging to their centroid |

**Note on the videos:** The PID controller requires ~12–15 seconds to fully converge from the default starting positions (drones start near z=0.1m and need to reach z=1.0m over 2–3m horizontal distances). The 8-second videos show the motion trajectory but may not show full convergence. The relevant quantity for the experiments is the **planning quality** (do the LLM-generated target positions match the formation geometry?), not the physical convergence time.

To generate longer or differently-configured videos:
```bash
# Edit DURATION_S and CAMERA_* in record_simulation.py, then:
conda activate swarm-llm
python record_simulation.py --scenario circle_n6
```
