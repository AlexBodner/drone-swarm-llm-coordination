# executor.py
import re
import json
import traceback


def extract_code_block(llm_response: str) -> str:
    """Extract Python code from LLM response (handles ```python ... ``` blocks)."""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block markers, assume entire response is code
    return llm_response.strip()


def extract_json_block(llm_response: str) -> str | None:
    """Extract JSON from LLM response (handles ```json ... ``` blocks).
    Falls back to searching for a bare { ... } object if no block markers found."""
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: find the first { ... } spanning multiple lines
    brace_match = re.search(r"\{[\s\S]*\}", llm_response)
    if brace_match:
        return brace_match.group(0).strip()
    return None


def parse_direct_waypoints(llm_response: str, state: dict, duration: float) -> dict | None:
    """
    Parse a direct JSON waypoint plan from an LLM response.

    Expected LLM output format (inside ```json ... ``` markers):
        {
          "0": [[t, x, y, z], [t, x, y, z], ...],
          "1": [[t, x, y, z], ...],
          ...
        }

    Returns the same {drone_id (int): [(t, x, y, z), ...]} format as
    execute_waypoint_plan() so the simulation loop is identical.
    Returns None on any parse or validation error.
    """
    json_str = extract_json_block(llm_response)
    if json_str is None:
        print("ERROR [direct]: no JSON block found in LLM response.")
        return None

    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"ERROR [direct]: JSON parse failed: {e}")
        return None

    try:
        n = state["n_drones"]
        result = {}
        for drone_id in range(n):
            # Accept both string keys ("0") and int keys (0)
            key = str(drone_id) if str(drone_id) in raw else drone_id
            assert key in raw, f"Missing drone {drone_id} in JSON plan (keys present: {list(raw.keys())})"
            wps_raw = raw[key]
            assert isinstance(wps_raw, list), f"Drone {drone_id} value must be a list, got {type(wps_raw)}"
            assert len(wps_raw) >= 2, f"Drone {drone_id} needs at least 2 waypoints, got {len(wps_raw)}"

            wps = []
            for wp in wps_raw:
                assert len(wp) == 4, f"Each waypoint must be [t, x, y, z], got {wp}"
                t, x, y, z = float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])
                assert 0 <= t <= duration + 0.5, f"Waypoint time {t} out of range [0, {duration}]"
                assert abs(x) < 10 and abs(y) < 10 and 0 < z < 5, \
                    f"Waypoint position out of bounds: {[t, x, y, z]}"
                wps.append((t, x, y, z))

            result[drone_id] = sorted(wps, key=lambda w: w[0])
        return result
    except (AssertionError, TypeError, ValueError, KeyError) as e:
        print(f"ERROR [direct]: waypoint validation failed: {e}")
        return None


def execute_plan_code(code: str, state: dict) -> dict | None:
    """
    Execute LLM-generated code and return target positions.
    Returns None if execution fails.

    The code is expected to define a function called `plan(state)`
    that returns a dict {drone_id (int): (x, y, z)}.
    """
    local_namespace = {}
    try:
        exec(code, local_namespace)   # use same dict for globals+locals so imports are visible inside functions
        if "plan" not in local_namespace:
            print("ERROR: No 'plan' function found in generated code.")
            return None
        result = local_namespace["plan"](state)
        # Validate output format
        assert isinstance(result, dict), "plan() must return a dict"
        for k, v in result.items():
            assert isinstance(k, int), f"Keys must be int drone IDs, got {type(k)}"
            assert len(v) == 3, f"Values must be (x, y, z) tuples, got {v}"
        return result
    except Exception as e:
        print(f"ERROR executing plan code: {e}")
        print(traceback.format_exc())
        return None


def execute_waypoint_plan(code: str, state: dict, duration: float) -> dict | None:
    """
    Execute LLM waypoint-trajectory code and return validated trajectories.
    Returns None on failure.

    Expected return format:
        {drone_id (int): [(t, x, y, z), ...]}  sorted by t, t in [0, duration]
    """
    local_namespace = {}
    try:
        exec(code, local_namespace)
        if "plan" not in local_namespace:
            print("ERROR: No 'plan' function found in waypoint code.")
            return None
        result = local_namespace["plan"](state, duration)
        assert isinstance(result, dict), "plan() must return a dict"
        n = state["n_drones"]
        for drone_id in range(n):
            assert drone_id in result, f"Missing drone {drone_id} in plan output"
            wps = result[drone_id]
            assert len(wps) >= 2, f"Drone {drone_id} needs at least 2 waypoints"
            for wp in wps:
                assert len(wp) == 4, f"Each waypoint must be (t, x, y, z), got {wp}"
                t, x, y, z = wp
                assert 0 <= t <= duration + 0.1, f"Waypoint time {t} out of range [0, {duration}]"
                assert abs(x) < 10 and abs(y) < 10 and 0 < z < 5, f"Waypoint position out of bounds: {wp}"
            # Sort by time
            result[drone_id] = sorted(wps, key=lambda w: w[0])
        return result
    except Exception as e:
        print(f"ERROR executing waypoint plan: {e}")
        print(traceback.format_exc())
        return None


def interpolate_waypoints(waypoints: list, t: float) -> tuple:
    """
    Linear interpolation between timed waypoints at time t.
    waypoints: [(t0, x0, y0, z0), (t1, x1, y1, z1), ...] sorted by time.
    Returns (x, y, z) at the given time.
    """
    if t <= waypoints[0][0]:
        return waypoints[0][1:]
    if t >= waypoints[-1][0]:
        return waypoints[-1][1:]
    for i in range(len(waypoints) - 1):
        t0, x0, y0, z0 = waypoints[i]
        t1, x1, y1, z1 = waypoints[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return (
                x0 + alpha * (x1 - x0),
                y0 + alpha * (y1 - y0),
                z0 + alpha * (z1 - z0),
            )
    return waypoints[-1][1:]


if __name__ == "__main__":
    import numpy as np

    print("Testing executor.py...")

    # Test 1: valid code block with markers
    test_response = '''
Here is the plan:

```python
import numpy as np

def plan(state):
    n = state["n_drones"]
    targets = {}
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i in range(n):
        targets[i] = (2.0 * np.cos(angles[i]), 2.0 * np.sin(angles[i]), 1.0)
    return targets
```
'''

    code = extract_code_block(test_response)
    print(f"Extracted code:\n{code}\n")

    state = {
        "n_drones": 3,
        "positions": {0: (0.0, 0.0, 0.5), 1: (0.1, 0.0, 0.5), 2: (-0.1, 0.0, 0.5)},
        "velocities": {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)},
    }
    result = execute_plan_code(code, state)
    print(f"Plan result: {result}")
    assert result is not None, "Should not return None for valid code"
    assert len(result) == 3, "Should have 3 entries"
    for k, v in result.items():
        assert isinstance(k, int), f"Keys should be int, got {type(k)}"
        assert len(v) == 3, f"Values should be tuples of length 3"

    # Test 2: invalid code
    bad_code = "def plan(state): return 'not a dict'"
    result_bad = execute_plan_code(bad_code, state)
    assert result_bad is None, "Invalid code should return None"
    print("Bad code correctly returned None")

    # Test 3: code without markers (fallback)
    no_markers_code = """
def plan(state):
    return {i: (float(i), 0.0, 1.0) for i in range(state['n_drones'])}
"""
    extracted = extract_code_block(no_markers_code)
    result_no_markers = execute_plan_code(extracted, state)
    assert result_no_markers is not None
    print(f"No-marker code result: {result_no_markers}")

    print("\nCheckpoint PASSED: executor.py works correctly.")
