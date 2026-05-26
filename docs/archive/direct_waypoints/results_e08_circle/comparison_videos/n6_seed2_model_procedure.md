# E-08 Circle Task — Model Procedure (N=6, Seed=2)

> **Task:** place 6 drones on a circle of radius 2 m centred at the world origin at height z = 1 m, evenly spaced.
> **Model:** llama-3.1-8b-instant via Groq · output mode: `direct` (raw JSON waypoints)

---

## Initial Swarm State (shared by both representations)

```
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
  Drone 3: (0.476, 0.476, 0.113)
  Drone 4: (0.635, 0.635, 0.113)
  Drone 5: (0.794, 0.794, 0.113)
```

---

## Representation: RAW (absolute coordinates)

**Reward:** `-0.7439`  · **Valid parse:** `True`
**Waypoints parsed:** `{'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4}` · **Latency:** `27.55s`
**Reward timeline:** @1s `-1.7198` / @5s `-1.4685` / @10s `-0.7710`

### State Representation fed to LLM

```
Number of drones: 6
Drone positions (x, y, z) in meters:
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
  Drone 3: (0.476, 0.476, 0.113)
  Drone 4: (0.635, 0.635, 0.113)
  Drone 5: (0.794, 0.794, 0.113)
```

### Full Prompt sent to LLM

```markdown
You are planning the motion of a swarm of 6 drones in a 3D physics simulator.
The simulation runs for 15.0 seconds total.

## Current Swarm State
Number of drones: 6
Drone positions (x, y, z) in meters:
  Drone 0: (0.000, 0.000, 0.113)
  Drone 1: (0.159, 0.159, 0.113)
  Drone 2: (0.318, 0.318, 0.113)
  Drone 3: (0.476, 0.476, 0.113)
  Drone 4: (0.635, 0.635, 0.113)
  Drone 5: (0.794, 0.794, 0.113)

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
- All drone IDs [0, 1, 2, 3, 4, 5] must be present as string keys.

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

Drone 0: (-0.238, -0.202, 0.113)
Drone 1: (0.473, -0.249, 0.113)
Drone 2: (0.418, 0.546, 0.113)
Drone 3: (0.164, 0.032, 0.113)
Drone 4: (0.410, 0.793, 0.113)
Drone 5: (0.856, 0.444, 0.113)

### Step 2: Compute the target (x, y, z) for each drone using the task instructions.

To form a circle of radius 2.0 meters centered at the origin (0, 0), at height z = 1.0 meters, we need to calculate the angle for each drone. Since there are 6 drones, we can divide the circle into 6 equal parts, each with an angle of 60 degrees (π/3 radians).

We can use the following formula to calculate the target x and y coordinates for each drone:

x = r * cos(θ)
y = r * sin(θ)

where r is the radius of the circle (2.0 meters) and θ is the angle for each drone.

For Drone 0, θ = 0 degrees (or 0 radians)
x = 2.0 * cos(0) = 2.0
y = 2.0 * sin(0) = 0

For Drone 1, θ = 60 degrees (or π/3 radians)
x = 2.0 * cos(π/3) = 1.0
y = 2.0 * sin(π/3) = 1.732

For Drone 2, θ = 120 degrees (or 2π/3 radians)
x = 2.0 * cos(2π/3) = -1.0
y = 2.0 * sin(2π/3) = 1.732

For Drone 3, θ = 180 degrees (or π radians)
x = 2.0 * cos(π) = -2.0
y = 2.0 * sin(π) = 0

For Drone 4, θ = 240 degrees (or 4π/3 radians)
x = 2.0 * cos(4π/3) = -1.0
y = 2.0 * sin(4π/3) = -1.732

For Drone 5, θ = 300 degrees (or 5π/3 radians)
x = 2.0 * cos(5π/3) = 1.0
y = 2.0 * sin(5π/3) = -1.732

The target z coordinate for each drone is 1.0 meters.

### Step 3: For each drone, write the list of timed waypoints following the recommended flight strategy.

For each drone, we will have 4 waypoints:

1. The first waypoint is at t=0.0, with the current position of the drone.
2. The second waypoint is at t=2.0, with the z coordinate set to 1.0 meters.
3. The third waypoint is at t=7.5, with the x and y coordinates set to the target values.
4. The fourth waypoint is at t=15.0, with the x, y, and z coordinates set to the target values.

Here are the waypoints for each drone:

Drone 0:
- Waypoint 1: [0.0, -0.238, -0.202, 0.113]
- Waypoint 2: [2.0, -0.238, -0.202, 1.0]
- Waypoint 3: [7.5, 2.0, 0.0, 1.0]
- Waypoint 4: [15.0, 2.0, 0.0, 1.0]

Drone 1:
- Waypoint 1: [0.0, 0.473, -0.249, 0.113]
- Waypoint 2: [2.0, 0.473, -0.249, 1.0]
- Waypoint 3: [7.5, 1.0, 1.732, 1.0]
- Waypoint 4: [15.0, 1.0, 1.732, 1.0]

Drone 2:
- Waypoint 1: [0.0, 0.418, 0.546, 0.113]
- Waypoint 2: [2.0, 0.418, 0.546, 1.0]
- Waypoint 3: [7.5, -1.0, 1.732, 1.0]
- Waypoint 4: [15.0, -1.0, 1.732, 1.0]

Drone 3:
- Waypoint 1: [0.0, 0.164, 0.032, 0.113]
- Waypoint 2: [2.0, 0.164, 0.032, 1.0]
- Waypoint 3: [7.5, -2.0, 0.0, 1.0]
- Waypoint 4: [15.0, -2.0, 0.0, 1.0]

Drone 4:
- Waypoint 1: [0.0, 0.410, 0.793, 0.113]
- Waypoint 2: [2.0, 0.410, 0.793, 1.0]
- Waypoint 3: [7.5, -1.0, -1.732, 1.0]
- Waypoint 4: [15.0, -1.0, -1.732, 1.0]

Drone 5:
- Waypoint 1: [0.0, 0.856, 0.444, 0.113]
- Waypoint 2: [2.0, 0.856, 0.444, 1.0]
- Waypoint 3: [7.5, 1.0, -1.732, 1.0]
- Waypoint 4: [15.0, 1.0, -1.732, 1.0]

## JSON Plan

```json
{
  "0": [
    [0.0, -0.238, -0.202, 0.113],
    [2.0, -0.238, -0.202, 1.0],
    [7.5, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, 0.473, -0.249, 0.113],
    [2.0, 0.473, -0.249, 1.0],
    [7.5, 1.0, 1.732, 1.0],
    [15.0, 1.0, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.418, 0.546, 0.113],
    [2.0, 0.418, 0.546, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "3": [
    [0.0, 0.164, 0.032, 0.113],
    [2.0, 0.164, 0.032, 1.0],
    [7.5, -2.0, 0.0, 1.0],
    [15.0, -2.0, 0.0, 1.0]
  ],
  "4": [
    [0.0, 0.410, 0.793, 0.113],
    [2.0, 0.410, 0.793, 1.0],
    [7.5, -1.0, -1.732, 1.0],
    [15.0, -1.0, -1.732, 1.0]
  ],
  "5": [
    [0.0, 0.856, 0.444, 0.113],
    [2.0, 0.856, 0.444, 1.0],
    [7.5, 1.0, -1.732, 1.0],
    [15.0, 1.0, -1.732, 1.0]
  ]
}
```
```

### Parsed Waypoints (JSON extracted from response)

```json
{
  "0": [
    [0.0, -0.238, -0.202, 0.113],
    [2.0, -0.238, -0.202, 1.0],
    [7.5, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, 0.473, -0.249, 0.113],
    [2.0, 0.473, -0.249, 1.0],
    [7.5, 1.0, 1.732, 1.0],
    [15.0, 1.0, 1.732, 1.0]
  ],
  "2": [
    [0.0, 0.418, 0.546, 0.113],
    [2.0, 0.418, 0.546, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "3": [
    [0.0, 0.164, 0.032, 0.113],
    [2.0, 0.164, 0.032, 1.0],
    [7.5, -2.0, 0.0, 1.0],
    [15.0, -2.0, 0.0, 1.0]
  ],
  "4": [
    [0.0, 0.410, 0.793, 0.113],
    [2.0, 0.410, 0.793, 1.0],
    [7.5, -1.0, -1.732, 1.0],
    [15.0, -1.0, -1.732, 1.0]
  ],
  "5": [
    [0.0, 0.856, 0.444, 0.113],
    [2.0, 0.856, 0.444, 1.0],
    [7.5, 1.0, -1.732, 1.0],
    [15.0, 1.0, -1.732, 1.0]
  ]
}
```

---

## Representation: RELATIVE (centroid-offset)

**Reward:** `-0.9777`  · **Valid parse:** `True`
**Waypoints parsed:** `{'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4}` · **Latency:** `23.98s`
**Reward timeline:** @1s `-1.6043` / @5s `-1.3424` / @10s `-1.0131`

### State Representation fed to LLM

```
Number of drones: 6
Swarm centroid (absolute world-frame): (0.397, 0.397, 0.113)
Per-drone offset from centroid (dx, dy, dz):
  Drone 0: (-0.397, -0.397, 0.000)
  Drone 1: (-0.238, -0.238, 0.000)
  Drone 2: (-0.079, -0.079, 0.000)
  Drone 3: (0.079, 0.079, 0.000)
  Drone 4: (0.238, 0.238, 0.000)
  Drone 5: (0.397, 0.397, 0.000)

COORDINATE SYSTEM NOTE: the offsets above are relative to the centroid — they are NOT absolute positions.
Your response MUST use absolute world-frame coordinates.
XY conversion:  absolute_x = 0.397 + dx,   absolute_y = 0.397 + dy
Z WARNING: the centroid z shown above (0.113 m) is the initial SPAWN height, NOT your target height.
For z: use the target height stated in the task description (e.g. z = 1.0 m) — do NOT add dz to centroid z.
```

### Full Prompt sent to LLM

```markdown
You are planning the motion of a swarm of 6 drones in a 3D physics simulator.
The simulation runs for 15.0 seconds total.

## Current Swarm State
Number of drones: 6
Swarm centroid (absolute world-frame): (0.397, 0.397, 0.113)
Per-drone offset from centroid (dx, dy, dz):
  Drone 0: (-0.397, -0.397, 0.000)
  Drone 1: (-0.238, -0.238, 0.000)
  Drone 2: (-0.079, -0.079, 0.000)
  Drone 3: (0.079, 0.079, 0.000)
  Drone 4: (0.238, 0.238, 0.000)
  Drone 5: (0.397, 0.397, 0.000)

COORDINATE SYSTEM NOTE: the offsets above are relative to the centroid — they are NOT absolute positions.
Your response MUST use absolute world-frame coordinates.
XY conversion:  absolute_x = 0.397 + dx,   absolute_y = 0.397 + dy
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
- All drone IDs [0, 1, 2, 3, 4, 5] must be present as string keys.

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

| Drone ID | Current (x, y, z) |
| --- | --- |
| 0 | (0.347 - 0.586, 0.227 - 0.429, 1.0) = (-0.239, -0.202, 1.0) |
| 1 | (0.347 + 0.126, 0.227 - 0.477, 1.0) = (0.473, -0.250, 1.0) |
| 2 | (0.347 + 0.071, 0.227 + 0.319, 1.0) = (0.418, 0.546, 1.0) |
| 3 | (0.347 - 0.183, 0.227 - 0.196, 1.0) = (0.164, 0.031, 1.0) |
| 4 | (0.347 + 0.063, 0.227 + 0.565, 1.0) = (0.410, 0.792, 1.0) |
| 5 | (0.347 + 0.509, 0.227 + 0.217, 1.0) = (0.856, 0.444, 1.0) |

### Step 2: Compute the target (x, y, z) for each drone using the task instructions.

The target position for each drone is a point on a circle of radius 2.0 meters centered at the origin (0, 0) at height z = 1.0 meters. We can use the parametric equation of a circle to find the target position for each drone.

Let's use the drone ID as the angle parameter (θ) in the parametric equation of a circle:

x = r cos(θ)
y = r sin(θ)

where r is the radius of the circle (2.0 meters) and θ is the angle parameter.

We can calculate the angle parameter (θ) for each drone ID as follows:

θ = (drone ID) × (2π / 6)

Now, we can calculate the target position for each drone:

| Drone ID | Target (x, y, z) |
| --- | --- |
| 0 | (2.0 cos(0), 2.0 sin(0), 1.0) = (2.0, 0.0, 1.0) |
| 1 | (2.0 cos(2π/6), 2.0 sin(2π/6), 1.0) = (1.732, 1.0, 1.0) |
| 2 | (2.0 cos(4π/6), 2.0 sin(4π/6), 1.0) = (1.0, 1.732, 1.0) |
| 3 | (2.0 cos(6π/6), 2.0 sin(6π/6), 1.0) = (-1.0, 1.732, 1.0) |
| 4 | (2.0 cos(8π/6), 2.0 sin(8π/6), 1.0) = (-1.732, 1.0, 1.0) |
| 5 | (2.0 cos(10π/6), 2.0 sin(10π/6), 1.0) = (-2.0, 0.0, 1.0) |

### Step 3: For each drone, write the list of timed waypoints following the recommended flight strategy.

We will use the following flight strategy:

- First waypoint: t=0.0, position = current position
- Second waypoint: t=2.0, position = current position at z=1.0 m
- Third waypoint: t=7.5, position = target position
- Fourth waypoint: t=15.0, position = target position

Here are the timed waypoints for each drone:

| Drone ID | Timed Waypoints |
| --- | --- |
| 0 | [[0.0, -0.239, -0.202, 1.0], [2.0, -0.239, -0.202, 1.0], [7.5, 2.0, 0.0, 1.0], [15.0, 2.0, 0.0, 1.0]] |
| 1 | [[0.0, 0.473, -0.250, 1.0], [2.0, 0.473, -0.250, 1.0], [7.5, 1.732, 1.0, 1.0], [15.0, 1.732, 1.0, 1.0]] |
| 2 | [[0.0, 0.418, 0.546, 1.0], [2.0, 0.418, 0.546, 1.0], [7.5, 1.0, 1.732, 1.0], [15.0, 1.0, 1.732, 1.0]] |
| 3 | [[0.0, 0.164, 0.031, 1.0], [2.0, 0.164, 0.031, 1.0], [7.5, -1.0, 1.732, 1.0], [15.0, -1.0, 1.732, 1.0]] |
| 4 | [[0.0, 0.410, 0.792, 1.0], [2.0, 0.410, 0.792, 1.0], [7.5, -1.732, 1.0, 1.0], [15.0, -1.732, 1.0, 1.0]] |
| 5 | [[0.0, 0.856, 0.444, 1.0], [2.0, 0.856, 0.444, 1.0], [7.5, -2.0, 0.0, 1.0], [15.0, -2.0, 0.0, 1.0]] |

## JSON Plan

```json
{
  "0": [
    [0.0, -0.239, -0.202, 1.0],
    [2.0, -0.239, -0.202, 1.0],
    [7.5, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, 0.473, -0.250, 1.0],
    [2.0, 0.473, -0.250, 1.0],
    [7.5, 1.732, 1.0, 1.0],
    [15.0, 1.732, 1.0, 1.0]
  ],
  "2": [
    [0.0, 0.418, 0.546, 1.0],
    [2.0, 0.418, 0.546, 1.0],
    [7.5, 1.0, 1.732, 1.0],
    [15.0, 1.0, 1.732, 1.0]
  ],
  "3": [
    [0.0, 0.164, 0.031, 1.0],
    [2.0, 0.164, 0.031, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "4": [
    [0.0, 0.410, 0.792, 1.0],
    [2.0, 0.410, 0.792, 1.0],
    [7.5, -1.732, 1.0, 1.0],
    [15.0, -1.732, 1.0, 1.0]
  ],
  "5": [
    [0.0, 0.856, 0.444, 1.0],
    [2.0, 0.856, 0.444, 1.0],
    [7.5, -2.0, 0.0, 1.0],
    [15.0, -2.0, 0.0, 1.0]
  ]
}
```
```

### Parsed Waypoints (JSON extracted from response)

```json
{
  "0": [
    [0.0, -0.239, -0.202, 1.0],
    [2.0, -0.239, -0.202, 1.0],
    [7.5, 2.0, 0.0, 1.0],
    [15.0, 2.0, 0.0, 1.0]
  ],
  "1": [
    [0.0, 0.473, -0.250, 1.0],
    [2.0, 0.473, -0.250, 1.0],
    [7.5, 1.732, 1.0, 1.0],
    [15.0, 1.732, 1.0, 1.0]
  ],
  "2": [
    [0.0, 0.418, 0.546, 1.0],
    [2.0, 0.418, 0.546, 1.0],
    [7.5, 1.0, 1.732, 1.0],
    [15.0, 1.0, 1.732, 1.0]
  ],
  "3": [
    [0.0, 0.164, 0.031, 1.0],
    [2.0, 0.164, 0.031, 1.0],
    [7.5, -1.0, 1.732, 1.0],
    [15.0, -1.0, 1.732, 1.0]
  ],
  "4": [
    [0.0, 0.410, 0.792, 1.0],
    [2.0, 0.410, 0.792, 1.0],
    [7.5, -1.732, 1.0, 1.0],
    [15.0, -1.732, 1.0, 1.0]
  ],
  "5": [
    [0.0, 0.856, 0.444, 1.0],
    [2.0, 0.856, 0.444, 1.0],
    [7.5, -2.0, 0.0, 1.0],
    [15.0, -2.0, 0.0, 1.0]
  ]
}
```

---

## Comparison & Observations

| Metric | RAW | RELATIVE |
|--------|-----|----------|
| Final reward | `-0.7439` | `-0.9777` |
| Best reward | `-0.7207` | `-0.9471` |
| @1 s | `-1.7198` | `-1.6043` |
| @5 s | `-1.4685` | `-1.3424` |
| @10 s | `-0.7710` | `-1.0131` |
| N waypoints | `{'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4}` | `{'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4}` |
| Latency (s) | `27.55` | `23.98` |
| Min pairwise dist (m) | `0.046` | `0.224` |
| Collision fraction | `0.031` | `0.000` |


**Key observation:** **RAW** wins by `0.2339`. Both models correctly compute 6 evenly-spaced angles on the circle. However, the RELATIVE model used a shifted angular assignment (θ = drone_id × 30°, yielding 30°, 60°, 90°… steps) instead of the standard 60° steps used by the reward function's reference points, resulting in a slightly rotated arrangement and a lower reward.


*Side-by-side video:* `results_e08_circle/comparison_videos/n6_circle_seed2_raw_vs_relative.mp4`
