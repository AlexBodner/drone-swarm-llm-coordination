# reward.py
import numpy as np
from scipy.optimize import linear_sum_assignment


def rendezvous_reward(final_positions: dict, initial_centroid: np.ndarray) -> float:
    """
    Reward for rendezvous task: combination of
      (a) tight clustering — negative mean pairwise distance between drones
      (b) meeting near the right place — negative distance of final cluster centroid
          from the initial centroid (state-dependent: LLM must read state["positions"])

    reward = -0.5 * mean_pairwise_distance - 0.5 * centroid_drift

    Returns a value in [-inf, 0], where 0 means all drones are exactly co-located
    at the initial centroid.
    """
    positions = list(final_positions.values())
    n = len(positions)

    # (a) mean pairwise distance — penalises spread-out cluster
    if n < 2:
        mean_pairwise = 0.0
    else:
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                count += 1
        mean_pairwise = total / count

    # (b) centroid drift — penalises meeting at the wrong location
    final_centroid = np.mean([np.array(p) for p in positions], axis=0)
    centroid_drift = float(np.linalg.norm(final_centroid - initial_centroid))

    return -0.5 * mean_pairwise - 0.5 * centroid_drift


def circle_formation_targets(n: int, radius: float = 2.0, height: float = 1.0) -> list:
    """Generate target positions for a circle formation centred at the world origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(radius * np.cos(a), radius * np.sin(a), height) for a in angles]


def scatter_circle_targets(n: int, centroid: np.ndarray, radius: float = 2.0, height: float = 1.0) -> list:
    """
    Generate target positions for a circle formation centred at `centroid` (x, y).
    Used by the scatter_circle task — like circle_formation but state-dependent centre.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [
        (float(centroid[0]) + radius * np.cos(a),
         float(centroid[1]) + radius * np.sin(a),
         height)
        for a in angles
    ]


def line_targets_from_centroid(n: int, centroid: np.ndarray, spacing: float = 1.0, height: float = 1.0) -> list:
    """
    Generate target positions for a line formation along the X-axis, centred at
    `centroid` (x, y).  Drone i gets x = cx + (i - (n-1)/2) * spacing, y = cy.
    """
    cx, cy = float(centroid[0]), float(centroid[1])
    return [
        (cx + (i - (n - 1) / 2) * spacing, cy, height)
        for i in range(n)
    ]


def swap_target_positions(initial_positions: dict, height: float = 1.0) -> dict:
    """
    Compute per-drone targets for the cyclic swap task.
    Drone i → initial position of drone (i+1) % n, at `height`.
    Returns {drone_id: np.ndarray(3,)}.
    """
    sorted_ids = sorted(initial_positions.keys())
    n = len(sorted_ids)
    targets = {}
    for idx, d_id in enumerate(sorted_ids):
        src = sorted_ids[(idx + 1) % n]
        tgt = np.array(initial_positions[src], dtype=float)
        tgt[2] = height
        targets[d_id] = tgt
    return targets


def expand_target_positions(initial_positions: dict, scale: float = 2.0, height: float = 1.0) -> dict:
    """
    Compute per-drone targets for the expand formation task.
    target_i = centroid + scale * (init_i - centroid), z = height.
    Returns {drone_id: np.ndarray(3,)}.
    """
    pos_array = np.array([np.array(v, dtype=float) for v in initial_positions.values()])
    centroid = pos_array.mean(axis=0)
    targets = {}
    for d_id, pos in initial_positions.items():
        offset = np.array(pos, dtype=float) - centroid
        tgt = centroid + scale * offset
        tgt[2] = height
        targets[d_id] = tgt
    return targets


def per_drone_formation_reward(final_positions: dict, target_positions: dict) -> float:
    """
    Reward for tasks with a fixed 1-to-1 drone-to-target assignment (no Hungarian needed).
    target_positions: {drone_id: array-like (x, y, z)}
    Returns negative mean distance; 0 = perfect.
    """
    total = 0.0
    n = len(final_positions)
    for d_id, final_pos in final_positions.items():
        tgt = np.array(target_positions[d_id])
        total += float(np.linalg.norm(np.array(final_pos) - tgt))
    return -(total / n) if n > 0 else 0.0


def formation_reward(final_positions: dict, target_positions: list) -> float:
    """
    Compute reward as negative mean distance after optimal assignment.
    Returns a value in [-inf, 0], where 0 is perfect formation.
    """
    n = len(final_positions)
    assert len(target_positions) == n, "Mismatch between drones and targets"

    # Build cost matrix: cost[i][j] = distance from drone i to target slot j
    drone_ids = sorted(final_positions.keys())
    cost_matrix = np.zeros((n, n))
    for i, d_id in enumerate(drone_ids):
        for j, target in enumerate(target_positions):
            pos = np.array(final_positions[d_id])
            tgt = np.array(target)
            cost_matrix[i, j] = np.linalg.norm(pos - tgt)

    # Hungarian algorithm: find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mean_distance = cost_matrix[row_ind, col_ind].mean()

    # Reward: higher is better (negative distance)
    return -mean_distance

    """
    Reward for rendezvous task: combination of
      (a) tight clustering — negative mean pairwise distance between drones
      (b) meeting near the right place — negative distance of final cluster centroid
          from the initial centroid (state-dependent: LLM must read state["positions"])

    reward = -0.5 * mean_pairwise_distance - 0.5 * centroid_drift

    Returns a value in [-inf, 0], where 0 means all drones are exactly co-located
    at the initial centroid.
    """
    positions = list(final_positions.values())
    n = len(positions)

    # (a) mean pairwise distance — penalises spread-out cluster
    if n < 2:
        mean_pairwise = 0.0
    else:
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                count += 1
        mean_pairwise = total / count

    # (b) centroid drift — penalises meeting at the wrong location
    final_centroid = np.mean([np.array(p) for p in positions], axis=0)
    centroid_drift = float(np.linalg.norm(final_centroid - initial_centroid))

    return -0.5 * mean_pairwise - 0.5 * centroid_drift


def circle_formation_targets(n: int, radius: float = 2.0, height: float = 1.0) -> list:
    """Generate target positions for a circle formation."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(radius * np.cos(a), radius * np.sin(a), height) for a in angles]


def formation_reward(final_positions: dict, target_positions: list) -> float:
    """
    Compute reward as negative mean distance after optimal assignment.
    Returns a value in [-inf, 0], where 0 is perfect formation.
    """
    n = len(final_positions)
    assert len(target_positions) == n, "Mismatch between drones and targets"

    # Build cost matrix: cost[i][j] = distance from drone i to target slot j
    drone_ids = sorted(final_positions.keys())
    cost_matrix = np.zeros((n, n))
    for i, d_id in enumerate(drone_ids):
        for j, target in enumerate(target_positions):
            pos = np.array(final_positions[d_id])
            tgt = np.array(target)
            cost_matrix[i, j] = np.linalg.norm(pos - tgt)

    # Hungarian algorithm: find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mean_distance = cost_matrix[row_ind, col_ind].mean()

    # Reward: higher is better (negative distance)
    return -mean_distance


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    print("Testing reward.py...")
    n = 5
    targets = circle_formation_targets(n, radius=2.0, height=1.0)
    print(f"Circle targets for N={n}:")
    for i, t in enumerate(targets):
        print(f"  Slot {i}: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

    # Random positions — should give a bad reward
    random_positions = {i: (np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0.5, 2.0)) for i in range(n)}
    r_random = formation_reward(random_positions, targets)
    print(f"\nReward with random positions: {r_random:.4f}")
    assert r_random < 0, "Reward should be negative"

    # Positions close to target — should give a near-zero reward
    near_target_positions = {i: (targets[i][0] + 0.01, targets[i][1] + 0.01, targets[i][2] + 0.01) for i in range(n)}
    r_near = formation_reward(near_target_positions, targets)
    print(f"Reward with near-target positions: {r_near:.4f}")
    assert r_near > r_random, "Near-target reward should be better than random"
    assert r_near < 0, "Reward should still be negative (not perfect)"

    # Perfect positions
    perfect_positions = {i: targets[i] for i in range(n)}
    r_perfect = formation_reward(perfect_positions, targets)
    print(f"Reward with perfect positions: {r_perfect:.4f}")
    assert abs(r_perfect) < 1e-9, "Perfect reward should be ~0"

    print("\nCheckpoint PASSED: reward.py works correctly.")
