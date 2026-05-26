# E-08 Circle Task — Model Procedure (N=3, Seed=0)

> **Task:** place 3 drones on a circle of radius 2 m centred at the world origin at height z = 1 m, evenly spaced.
> **Model:** llama-3.1-8b-instant via Groq · output mode: `direct` (raw JSON waypoints)

---

## Initial Swarm State (shared by both representations)

```
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
```

---

## Representation: RAW (absolute coordinates)

**Reward:** `-1.1931`  · **Valid parse:** `True`
**Waypoints parsed:** `{'0': 5, '1': 5, '2': 5}` · **Latency:** `7.14s`
**Reward timeline:** @1s `-1.9015` / @5s `-1.5873` / @10s `-1.1517`

### State Representation fed to LLM

```
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
```

### Full Prompt sent to LLM

```markdown
You are planning the motion of a swarm of 3 drones in a 3D physics simulator.
The simulation runs for 15.0 seconds total.

## Current Swarm State
Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)

## Task
Move the drones to form a circle of radius 2.0 meters centered at the origin (0, 0), at height z = 1.0 meters. The drones should be evenly spaced around the circle.

## Instructions
You must output a flight plan as a JSON object — NO Python code, NO functions, NO programming.
You must reason step by step in plain language first, then output the numbers.

The JSON must map each drone ID (as a string) to a list of timed waypoints.
Each waypoint is a list of four numbers: [t, x, y, z]
  - t : time in seconds, must be in [0, 15.0]
  - x, y : horizontal position in meters (|x| < 4.5, |y| < 4.5)
  - z : height in meters (0.05 < z < 3.0)

Requirements:
- First waypoint for every drone: t=0.0, position = that drone's CURRENT position exactly (copy it from the state).
- Last waypoint for every drone: t=15.0, at the computed target position.
- Include 3 to 5 intermediate waypoints per drone.
- Recommended flight strategy: rise to z=1.0 m in the first 2 seconds, then move horizontally to target XY.
- All drone IDs [0, 1, 2] must be present as string keys.

REASONING STEPS (work through these explicitly before writing the JSON):
1. Write down each drone's current (x, y, z) from the state above.
2. Compute the target (x, y, z) for each drone using the task instructions (show all arithmetic).
3. For each drone, write the list of timed waypoints following the recommended flight strategy.

Then output the JSON plan inside a ```json ... ``` block. Nothing else after the block.

Example format (for 1 drone, duration=15.0s, target at (1.5, 0.5, 1.0)):
```json
{
  "0": [
    [0.0,  <current_x>,  <current_y>,  <current_z>],
    [2.0,  <current_x>,  <current_y>,  1.0],
    [7.5,  1.5,  0.5,  1.0],
    [15.0,  1.5,  0.5,  1.0]
  ]
}
```
```

### LLM Response

```markdown
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z)

Drone 0: (0.137, -0.230, 0.113)
Drone 1: (-0.300, -0.325, 0.113)
Drone 2: (0.631, 0.730, 0.113)

### Step 2: Compute the target (x, y, z) for each drone

To form a circle of radius 2.0 meters centered at the origin (0, 0) at height z = 1.0 meters, we need to calculate the target positions for each drone.

Let's assume the drones are evenly spaced around the circle. We can use the following formula to calculate the angle for each drone:

angle = (drone ID * 360) / 3

We'll convert this angle to radians and use it to calculate the target x and y coordinates:

target x = radius * cos(angle)
target y = radius * sin(angle)

Here are the calculations for each drone:

Drone 0:
angle = (0 * 360) / 3 = 0° = 0 rad
target x = 2 * cos(0) = 2
target y = 2 * sin(0) = 0
target z = 1.0

Drone 1:
angle = (1 * 360) / 3 = 120° = 2.0944 rad
target x = 2 * cos(2.0944) = -1.732
target y = 2 * sin(2.0944) = 1.732
target z = 1.0

Drone 2:
angle = (2 * 360) / 3 = 240° = 4.1888 rad
target x = 2 * cos(4.1888) = 1.732
target y = 2 * sin(4.1888) = -1.732
target z = 1.0

### Step 3: Write the list of timed waypoints for each drone

We'll follow the recommended flight strategy: rise to z=1.0 m in the first 2 seconds, then move horizontally to target XY.

For each drone, we'll create 3 to 5 intermediate waypoints. We'll space them out evenly between the current position and the target position.

Here are the waypoints for each drone:

Drone 0:
- Waypoint 1: t=0.0, position = (0.137, -0.230, 0.113)
- Waypoint 2: t=1.0, position = (0.137, -0.230, 0.5)
- Waypoint 3: t=2.0, position = (0.137, -0.230, 1.0)
- Waypoint 4: t=10.0, position = (2, 0, 1.0)
- Waypoint 5: t=15.0, position = (2, 0, 1.0)

Drone 1:
- Waypoint 1: t=0.0, position = (-0.300, -0.325, 0.113)
- Waypoint 2: t=1.0, position = (-0.300, -0.325, 0.5)
- Waypoint 3: t=2.0, position = (-0.300, -0.325, 1.0)
- Waypoint 4: t=10.0, position = (-1.732, 1.732, 1.0)
- Waypoint 5: t=15.0, position = (-1.732, 1.732, 1.0)

Drone 2:
- Waypoint 1: t=0.0, position = (0.631, 0.730, 0.113)
- Waypoint 2: t=1.0, position = (0.631, 0.730, 0.5)
- Waypoint 3: t=2.0, position = (0.631, 0.730, 1.0)
- Waypoint 4: t=10.0, position = (1.732, -1.732, 1.0)
- Waypoint 5: t=15.0, position = (1.732, -1.732, 1.0)

## JSON Plan

```json
{
  "0": [
    [0.0, 0.137, -0.230, 0.113],
    [1.0, 0.137, -0.230, 0.5],
    [2.0, 0.137, -0.230, 1.0],
    [10.0, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, -0.300, -0.325, 0.113],
    [1.0, -0.300, -0.325, 0.5],
    [2.0, -0.300, -0.325, 1.0],
    [10.0, -1.732, 1.732, 1.0],
    [15.0, -1.732, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.631, 0.730, 0.113],
    [1.0, 0.631, 0.730, 0.5],
    [2.0, 0.631, 0.730, 1.0],
    [10.0, 1.732, -1.732, 1.0],
    [15.0, 1.732, -1.732, 1.0]
  ]
}
```
```

### Parsed Waypoints (JSON extracted from response)

```json
{
  "0": [
    [0.0, 0.137, -0.230, 0.113],
    [1.0, 0.137, -0.230, 0.5],
    [2.0, 0.137, -0.230, 1.0],
    [10.0, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, -0.300, -0.325, 0.113],
    [1.0, -0.300, -0.325, 0.5],
    [2.0, -0.300, -0.325, 1.0],
    [10.0, -1.732, 1.732, 1.0],
    [15.0, -1.732, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.631, 0.730, 0.113],
    [1.0, 0.631, 0.730, 0.5],
    [2.0, 0.631, 0.730, 1.0],
    [10.0, 1.732, -1.732, 1.0],
    [15.0, 1.732, -1.732, 1.0]
  ]
}
```

---

## Representation: RELATIVE (centroid-offset)

**Reward:** `-0.0484`  · **Valid parse:** `True`
**Waypoints parsed:** `{'0': 5, '1': 5, '2': 5}` · **Latency:** `18.55s`
**Reward timeline:** @1s `-1.8260` / @5s `-1.2123` / @10s `-0.0934`

### State Representation fed to LLM

```
Number of drones: 3
Swarm centroid (absolute world-frame): (0.159, 0.159, 0.113)
Per-drone offset from centroid (dx, dy, dz):
  Drone 0: (-0.159, -0.159, 0.000)
  Drone 1: (0.000, 0.000, 0.000)
  Drone 2: (0.159, 0.159, 0.000)

COORDINATE SYSTEM NOTE: the offsets above are relative to the centroid — they are NOT absolute positions.
Your response MUST use absolute world-frame coordinates.
XY conversion:  absolute_x = 0.159 + dx,   absolute_y = 0.159 + dy
Z WARNING: the centroid z shown above (0.113 m) is the initial SPAWN height, NOT your target height.
For z: use the target height stated in the task description (e.g. z = 1.0 m) — do NOT add dz to centroid z.
```

### Full Prompt sent to LLM

```markdown
You are planning the motion of a swarm of 3 drones in a 3D physics simulator.
The simulation runs for 15.0 seconds total.

## Current Swarm State
Number of drones: 3
Swarm centroid (absolute world-frame): (0.159, 0.159, 0.113)
Per-drone offset from centroid (dx, dy, dz):
  Drone 0: (-0.159, -0.159, 0.000)
  Drone 1: (0.000, 0.000, 0.000)
  Drone 2: (0.159, 0.159, 0.000)

COORDINATE SYSTEM NOTE: the offsets above are relative to the centroid — they are NOT absolute positions.
Your response MUST use absolute world-frame coordinates.
XY conversion:  absolute_x = 0.159 + dx,   absolute_y = 0.159 + dy
Z WARNING: the centroid z shown above (0.113 m) is the initial SPAWN height, NOT your target height.
For z: use the target height stated in the task description (e.g. z = 1.0 m) — do NOT add dz to centroid z.

## Task
Move the drones to form a circle of radius 2.0 meters centered at the origin (0, 0), at height z = 1.0 meters. The drones should be evenly spaced around the circle.

## Instructions
You must output a flight plan as a JSON object — NO Python code, NO functions, NO programming.
You must reason step by step in plain language first, then output the numbers.

The JSON must map each drone ID (as a string) to a list of timed waypoints.
Each waypoint is a list of four numbers: [t, x, y, z]
  - t : time in seconds, must be in [0, 15.0]
  - x, y : horizontal position in meters (|x| < 4.5, |y| < 4.5)
  - z : height in meters (0.05 < z < 3.0)

Requirements:
- First waypoint for every drone: t=0.0, position = that drone's CURRENT position exactly (copy it from the state).
- Last waypoint for every drone: t=15.0, at the computed target position.
- Include 3 to 5 intermediate waypoints per drone.
- Recommended flight strategy: rise to z=1.0 m in the first 2 seconds, then move horizontally to target XY.
- All drone IDs [0, 1, 2] must be present as string keys.

REASONING STEPS (work through these explicitly before writing the JSON):
1. Write down each drone's current (x, y, z) from the state above.
2. Compute the target (x, y, z) for each drone using the task instructions (show all arithmetic).
3. For each drone, write the list of timed waypoints following the recommended flight strategy.

Then output the JSON plan inside a ```json ... ``` block. Nothing else after the block.

Example format (for 1 drone, duration=15.0s, target at (1.5, 0.5, 1.0)):
```json
{
  "0": [
    [0.0,  <current_x>,  <current_y>,  <current_z>],
    [2.0,  <current_x>,  <current_y>,  1.0],
    [7.5,  1.5,  0.5,  1.0],
    [15.0,  1.5,  0.5,  1.0]
  ]
}
```
```

### LLM Response

```markdown
## Reasoning Steps

### Step 1: Write down each drone's current (x, y, z) from the state above.

- Drone 0: (0.156 + (-0.019), 0.058 + (-0.289), 1.0) = (0.137, -0.231, 1.0)
- Drone 1: (0.156 + (-0.456), 0.058 + (-0.383), 1.0) = (-0.300, -0.325, 1.0)
- Drone 2: (0.156 + 0.475, 0.058 + 0.672, 1.0) = (0.631, 0.730, 1.0)

### Step 2: Compute the target (x, y, z) for each drone using the task instructions.

- The target is a circle of radius 2.0 meters centered at the origin (0, 0), at height z = 1.0 meters.
- To evenly space the drones around the circle, we need to calculate the angle for each drone.
- There are 3 drones, so we divide the circle into 3 equal parts: 360° / 3 = 120°.
- Drone 0 is at an angle of 0° (or 0 radians), Drone 1 is at an angle of 120° (or 2π/3 radians), and Drone 2 is at an angle of 240° (or 4π/3 radians).
- We can use the following formulas to calculate the target (x, y) for each drone:
  - x = r * cos(θ)
  - y = r * sin(θ)
  - where r is the radius of the circle (2.0 meters) and θ is the angle for each drone.

- Drone 0: θ = 0° (or 0 radians)
  - x = 2.0 * cos(0) = 2.0
  - y = 2.0 * sin(0) = 0
- Drone 1: θ = 120° (or 2π/3 radians)
  - x = 2.0 * cos(2π/3) = -1.0
  - y = 2.0 * sin(2π/3) = 1.732
- Drone 2: θ = 240° (or 4π/3 radians)
  - x = 2.0 * cos(4π/3) = -1.0
  - y = 2.0 * sin(4π/3) = -1.732

- The target (x, y, z) for each drone is:
  - Drone 0: (2.0, 0, 1.0)
  - Drone 1: (-1.0, 1.732, 1.0)
  - Drone 2: (-1.0, -1.732, 1.0)

### Step 3: For each drone, write the list of timed waypoints following the recommended flight strategy.

- The recommended flight strategy is to rise to z=1.0 m in the first 2 seconds, then move horizontally to target XY.
- We will add 3 to 5 intermediate waypoints per drone to ensure smooth movement.

- Drone 0:
  - Waypoint 1: t=0.0, (0.137, -0.231, 1.0)
  - Waypoint 2: t=1.0, (0.137, -0.231, 0.5) (rise to z=0.5 m)
  - Waypoint 3: t=2.0, (0.137, -0.231, 1.0) (reach target z=1.0 m)
  - Waypoint 4: t=7.5, (2.0, 0, 1.0) (reach target XY)
  - Waypoint 5: t=15.0, (2.0, 0, 1.0) (final target position)

- Drone 1:
  - Waypoint 1: t=0.0, (-0.300, -0.325, 1.0)
  - Waypoint 2: t=1.0, (-0.300, -0.325, 0.5) (rise to z=0.5 m)
  - Waypoint 3: t=2.0, (-0.300, -0.325, 1.0) (reach target z=1.0 m)
  - Waypoint 4: t=7.5, (-1.0, 1.732, 1.0) (reach target XY)
  - Waypoint 5: t=15.0, (-1.0, 1.732, 1.0) (final target position)

- Drone 2:
  - Waypoint 1: t=0.0, (0.631, 0.730, 1.0)
  - Waypoint 2: t=1.0, (0.631, 0.730, 0.5) (rise to z=0.5 m)
  - Waypoint 3: t=2.0, (0.631, 0.730, 1.0) (reach target z=1.0 m)
  - Waypoint 4: t=7.5, (-1.0, -1.732, 1.0) (reach target XY)
  - Waypoint 5: t=15.0, (-1.0, -1.732, 1.0) (final target position)

## JSON Plan

```json
{
  "0": [
    [0.0, 0.137, -0.231, 1.0],
    [1.0, 0.137, -0.231, 0.5],
    [2.0, 0.137, -0.231, 1.0],
    [7.5, 2.0, 0, 1.0],
    [15.0, 2.0, 0, 1.0]
  ],
  "1": [
    [0.0, -0.300, -0.325, 1.0],
    [1.0, -0.300, -0.325, 0.5],
    [2.0, -0.300, -0.325, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.631, 0.730, 1.0],
    [1.0, 0.631, 0.730, 0.5],
    [2.0, 0.631, 0.730, 1.0],
    [7.5, -1.0, -1.732, 1.0],
    [15.0, -1.0, -1.732, 1.0]
  ]
}
```
```

### Parsed Waypoints (JSON extracted from response)

```json
{
  "0": [
    [0.0, 0.137, -0.231, 1.0],
    [1.0, 0.137, -0.231, 0.5],
    [2.0, 0.137, -0.231, 1.0],
    [7.5, 2.0, 0, 1.0],
    [15.0, 2.0, 0, 1.0]
  ],
  "1": [
    [0.0, -0.300, -0.325, 1.0],
    [1.0, -0.300, -0.325, 0.5],
    [2.0, -0.300, -0.325, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.631, 0.730, 1.0],
    [1.0, 0.631, 0.730, 0.5],
    [2.0, 0.631, 0.730, 1.0],
    [7.5, -1.0, -1.732, 1.0],
    [15.0, -1.0, -1.732, 1.0]
  ]
}
```

---

## Comparison & Observations

| Metric | RAW | RELATIVE |
|--------|-----|----------|
| Final reward | `-1.1931` | `-0.0484` |
| Best reward | `-1.1462` | `-0.0075` |
| @1 s | `-1.9015` | `-1.8260` |
| @5 s | `-1.5873` | `-1.2123` |
| @10 s | `-1.1517` | `-0.0934` |
| N waypoints | `{'0': 5, '1': 5, '2': 5}` | `{'0': 5, '1': 5, '2': 5}` |
| Latency (s) | `7.14` | `18.55` |
| Min pairwise dist (m) | `0.203` | `0.441` |
| Collision fraction | `0.000` | `0.000` |


**Key observation:** **RELATIVE** wins by `1.1447`. With N=3 the RELATIVE representation provides the swarm centroid directly, allowing the model to place the three drones at clean 120° intervals. The RAW model failed to correctly infer the circle centre from absolute coordinates alone, producing a poorly aligned formation.



