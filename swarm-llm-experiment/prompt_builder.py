# prompt_builder.py

TASK_CIRCLE = (
    "Move the drones to form a circle of radius 2.0 meters centered "
    "at the origin (0, 0), at height z = 1.0 meters. The drones should be evenly "
    "spaced around the circle."
)

TASK_RENDEZVOUS = (
    "Move all drones to rendezvous at a common meeting point. "
    "The meeting point is the centroid (average position) of all drones' INITIAL positions, "
    "at height z = 1.0 meters. "
    "You MUST compute the centroid from the positions listed in the Current Swarm State above — "
    "sum each coordinate and divide by the number of drones. Do not hardcode any positions. "
    "All drones should end up as close together as possible, near that centroid."
)

TASK_SWAP_POSITIONS = (
    "Each drone must travel to the INITIAL position of the NEXT drone (by ID), at height z = 1.0 meters. "
    "Specifically: drone 0 moves to where drone 1 started, drone 1 moves to where drone 2 started, "
    "..., drone N-1 moves to where drone 0 started (cyclic shift). "
    "You MUST read every drone's initial position from the Current Swarm State above. "
    "Do not hardcode coordinates — each target is a different drone's initial XY position at z = 1.0."
)

TASK_EXPAND_FORMATION = (
    "Expand the swarm formation by doubling each drone's distance from the swarm centroid. "
    "For each drone i: target = centroid + 2.0 * (initial_position_i - centroid), at height z = 1.0 meters. "
    "Step-by-step: (1) compute the XY centroid of all initial positions; "
    "(2) for each drone, compute its XY offset from the centroid; "
    "(3) multiply that XY offset by 2.0; "
    "(4) add the scaled offset back to the centroid to get the absolute target XY; "
    "(5) use z = 1.0 m for all drones. "
    "Do not hardcode coordinates — derive everything from the state above."
)

TASK_SCATTER_CIRCLE = (
    "Move the drones to form a circle of radius 2.0 meters centred at the CENTROID of their "
    "initial positions (NOT at the world origin), evenly spaced, at height z = 1.0 meters. "
    "Step-by-step: (1) compute the XY centroid (cx, cy) from the state above; "
    "(2) assign angle_i = i * (360 / N) degrees for drone i (i = 0, 1, ..., N-1); "
    "(3) target_x = cx + 2.0 * cos(angle_i),  target_y = cy + 2.0 * sin(angle_i),  z = 1.0; "
    "(4) output those absolute XY coordinates. "
    "Do NOT place the circle at the world origin (0, 0) — it must be centred at (cx, cy)."
)

TASK_LINE_FORMATION = (
    "Arrange all drones in a straight line parallel to the X-axis, centred at the CENTROID of "
    "their initial positions, with drones spaced exactly 1.0 meter apart, all at height z = 1.0 meters. "
    "Step-by-step: (1) compute the XY centroid (cx, cy) from the state above; "
    "(2) sort drone IDs 0 ... N-1; "
    "(3) drone i gets: x = cx + (i - (N-1)/2) * 1.0,  y = cy,  z = 1.0; "
    "(4) verify the middle drone(s) land at or near cx. "
    "Do not hardcode cx or cy — compute them from the state above."
)


def build_prompt(state_text: str, task_description: str, n_drones: int) -> str:
    return f"""You are planning the motion of a swarm of {n_drones} drones in a 3D physics simulator.

## Current Swarm State
{state_text}

## Task
{task_description}

## Instructions
Write a Python function called `plan(state)` that takes the current state dictionary and returns a dictionary mapping each drone ID (integer) to its target position as a tuple (x, y, z).

The state dictionary has the following structure:
{{
    "n_drones": int,
    "positions": {{drone_id (int): (x, y, z), ...}},
    "velocities": {{drone_id (int): (vx, vy, vz), ...}},
}}

Rules:
- Return ONLY the Python code block, no explanations.
- The function must handle exactly {n_drones} drones.
- All drone IDs from 0 to {n_drones - 1} must be present in the output.
- Target positions should be physically reasonable (|x|, |y| < 5.0, 0.1 < z < 3.0).
- Always wrap your code in ```python ... ``` markers.

Example output format:
```python
def plan(state):
    targets = {{}}
    # your logic here
    targets[0] = (1.0, 0.0, 1.0)
    # ...
    return targets
```"""


def _wp_range(n_drones: int) -> str:
    """Return the recommended intermediate-waypoint count string scaled with swarm size.
    More drones = more complex manoeuvres = more intermediate waypoints needed."""
    if n_drones <= 3:
        return "3 to 5"
    elif n_drones <= 6:
        return "3 to 10"
    else:
        return "3 to 15"


def build_direct_waypoint_prompt(state_text: str, task_description: str, n_drones: int, duration: float = 15.0) -> str:
    """
    Prompt variant that asks the LLM to produce waypoints as raw JSON numbers,
    NOT as Python code. The LLM must reason arithmetically in language to derive
    the coordinates — no programming allowed.

    This tests whether the model can do explicit spatial reasoning from the state
    description alone, without the crutch of executable math functions.
    """
    drone_ids = list(range(n_drones))
    example_id = drone_ids[0]
    return f"""You are planning the motion of a swarm of {n_drones} drones in a 3D physics simulator.
The simulation runs for {duration:.1f} seconds total.

## Current Swarm State
{state_text}

## Task
{task_description}

## Instructions
You must output a flight plan as a JSON object — NO Python code, NO functions, NO programming.
You must reason step by step in plain language first, then output the numbers.

The JSON must map each drone ID (as a string) to a list of timed waypoints.
Each waypoint is a list of four numbers: [t, x, y, z]
  - t : time in seconds, must be in [0, {duration:.1f}]
  - x, y : horizontal position in meters (|x| < 4.5, |y| < 4.5)
  - z : height in meters (0.05 < z < 3.0)

Requirements:
- First waypoint for every drone: t=0.0, position = that drone's CURRENT position exactly (copy it from the state).
- Last waypoint for every drone: t={duration:.1f}, at the computed target position.
- Include {_wp_range(n_drones)} intermediate waypoints per drone.
- Recommended flight strategy: rise to z=1.0 m in the first 2 seconds, then move horizontally to target XY.
- All drone IDs {drone_ids} must be present as string keys.
- Keep the JSON compact: no extra blank lines inside the block.

REASONING STEPS (work through these explicitly before writing the JSON):
1. Write down each drone's current (x, y, z) from the state above.
2. Compute the target (x, y, z) for each drone using the task instructions (show all arithmetic).
3. For each drone, write the list of timed waypoints following the recommended flight strategy.

Then output the JSON plan inside a ```json ... ``` block. Nothing else after the block.

Example format (for 1 drone, duration={duration:.1f}s, target at (1.5, 0.5, 1.0)):
```json
{{
  "{example_id}": [
    [0.0,  <current_x>,  <current_y>,  <current_z>],
    [2.0,  <current_x>,  <current_y>,  1.0],
    [{duration*0.5:.1f},  1.5,  0.5,  1.0],
    [{duration:.1f},  1.5,  0.5,  1.0]
  ]
}}
```"""


def build_waypoint_prompt(state_text: str, task_description: str, n_drones: int, duration: float = 15.0) -> str:
    """
    Prompt variant that asks the LLM to produce a timed waypoint trajectory
    instead of a single endpoint. The LLM returns a sequence of (time, x, y, z)
    checkpoints per drone so the PID controller can track a smooth path.
    """
    return f"""You are planning the motion of a swarm of {n_drones} drones in a 3D physics simulator.
The simulation runs for {duration:.1f} seconds total.

## Current Swarm State
{state_text}

## Task
{task_description}

## Instructions
Write a Python function called `plan(state, duration)` that returns a TRAJECTORY for each drone:
a dictionary mapping each drone ID (int) to a list of timed waypoints.

Each waypoint is a tuple: (t, x, y, z) where t is the time in seconds [0, duration].

The state dictionary has the following structure:
{{
    "n_drones": int,
    "positions": {{drone_id (int): (x, y, z), ...}},
    "velocities": {{drone_id (int): (vx, vy, vz), ...}},
}}

Design rules:
- The FIRST waypoint for each drone should be at t=0 at its current position (from state["positions"]).
- The LAST waypoint should be at t=duration.
- Add 2 to 5 intermediate waypoints to guide the drone along a smooth path.
- A good strategy: first rise to the target height within the first 2 seconds, then move
  horizontally to the target XY position, then fine-tune. This avoids altitude loss during high-speed moves.
- All drone IDs from 0 to {n_drones - 1} must be present.
- Position bounds: |x|, |y| < 5.0, 0.1 < z < 3.0.
- Return ONLY the Python code block, no explanations.
- Wrap code in ```python ... ``` markers.

Example output format (for 1 drone, duration=10.0):
```python
def plan(state, duration):
    import math
    trajectories = {{}}
    # Drone 0: rise first, then move out
    px, py, pz = state["positions"][0]
    trajectories[0] = [
        (0.0,          px,   py,   pz),    # start: current position
        (2.0,          px,   py,   1.0),   # rise to target height
        (duration*0.7, 1.0,  0.0,  1.0),   # move toward target
        (duration,     1.0,  0.0,  1.0),   # hold at final target
    ]
    return trajectories
```"""


if __name__ == "__main__":
    print("Testing prompt_builder.py...")

    sample_state_text = """Number of drones: 3
Drone positions (x, y, z) in meters:
  Drone 0: (0.050, 0.030, 0.800)
  Drone 1: (-0.040, 0.020, 0.900)
  Drone 2: (0.010, -0.060, 1.000)"""

    prompt = build_prompt(sample_state_text, TASK_CIRCLE, n_drones=3)
    print("Generated prompt (first 500 chars):")
    print(prompt[:500])
    print("...")
    assert "plan(state)" in prompt
    assert "3 drones" in prompt
    assert "circle" in prompt.lower()
    print("\nCheckpoint PASSED: prompt_builder.py works correctly.")
