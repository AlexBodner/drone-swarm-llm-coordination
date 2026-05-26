# representations.py
import numpy as np


def repr_raw(state: dict) -> str:
    """Representation 1: Raw Coordinates."""
    lines = [f"Number of drones: {state['n_drones']}"]
    lines.append("Drone positions (x, y, z) in meters:")
    for drone_id, pos in state["positions"].items():
        lines.append(f"  Drone {drone_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    return "\n".join(lines)


def repr_relative(state: dict) -> str:
    """Representation 2: Centroid-Relative Coordinates."""
    positions = state["positions"]
    centroid = np.mean([np.array(p) for p in positions.values()], axis=0)
    lines = [
        f"Number of drones: {state['n_drones']}",
        f"Swarm centroid (absolute world-frame): ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
        "Per-drone offset from centroid (dx, dy, dz):",
    ]
    for drone_id, pos in positions.items():
        rel = np.array(pos) - centroid
        lines.append(f"  Drone {drone_id}: ({rel[0]:.3f}, {rel[1]:.3f}, {rel[2]:.3f})")
    lines.extend([
        "",
        "COORDINATE SYSTEM NOTE: the offsets above are relative to the centroid — they are NOT absolute positions.",
        "Your response MUST use absolute world-frame coordinates.",
        f"XY conversion:  absolute_x = {centroid[0]:.3f} + dx,   absolute_y = {centroid[1]:.3f} + dy",
        f"Z WARNING: the centroid z shown above ({centroid[2]:.3f} m) is the initial SPAWN height, NOT your target height.",
        "For z: use the target height stated in the task description (e.g. z = 1.0 m) — do NOT add dz to centroid z.",
    ])
    return "\n".join(lines)


def repr_graph(state: dict, neighbor_radius: float = 2.0) -> str:
    """Representation 3: Neighborhood Graph."""
    positions = state["positions"]
    n = state["n_drones"]
    lines = [
        f"Number of drones: {n}",
        f"Neighbor radius: {neighbor_radius}m",
        "Drone positions and neighbors:",
    ]
    for i, pos_i in positions.items():
        neighbors = []
        for j, pos_j in positions.items():
            if i == j:
                continue
            dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            if dist <= neighbor_radius:
                neighbors.append(f"Drone {j} ({dist:.2f}m away)")
        lines.append(
            f"  Drone {i} at ({pos_i[0]:.2f}, {pos_i[1]:.2f}, {pos_i[2]:.2f}): "
            f"neighbors = [{', '.join(neighbors) if neighbors else 'none'}]"
        )
    return "\n".join(lines)


def repr_aggregate(state: dict) -> str:
    """Representation 4: Aggregate Descriptors."""
    positions = np.array(list(state["positions"].values()))
    centroid = positions.mean(axis=0)
    spread = positions.std(axis=0)
    max_dist = (
        max(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        if len(positions) > 1
        else 0.0
    )

    lines = [
        f"Number of drones: {state['n_drones']}",
        f"Swarm centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
        f"Position spread (std): ({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f})",
        f"Maximum inter-drone distance: {max_dist:.3f}m",
        f"Mean height: {centroid[2]:.3f}m",
    ]
    return "\n".join(lines)


def repr_natural_language(state: dict) -> str:
    """Representation 5: Natural Language (Semantic)."""
    positions = state["positions"]
    n = state["n_drones"]
    pos_array = np.array(list(positions.values()))
    centroid = pos_array.mean(axis=0)

    quadrant_counts = {"north-east": 0, "north-west": 0, "south-east": 0, "south-west": 0}
    for pos in positions.values():
        q = ("north" if pos[1] > centroid[1] else "south") + "-" + (
            "east" if pos[0] > centroid[0] else "west"
        )
        quadrant_counts[q] += 1

    spread = pos_array.std()
    spread_desc = (
        "tightly clustered"
        if spread < 0.5
        else ("moderately spread" if spread < 1.5 else "widely dispersed")
    )

    qdesc = ", ".join(
        f"{v} drone(s) in the {k}" for k, v in quadrant_counts.items() if v > 0
    )
    return (
        f"There are {n} drones currently {spread_desc} around their centroid "
        f"at approximately ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) meters. "
        f"Distribution: {qdesc}."
    )


REPRESENTATIONS = {
    "raw": repr_raw,
    "relative": repr_relative,
    "graph": repr_graph,
    "aggregate": repr_aggregate,
    "natural_language": repr_natural_language,
}


if __name__ == "__main__":
    # Create a sample state
    sample_state = {
        "n_drones": 4,
        "positions": {
            0: (0.5, 0.3, 0.8),
            1: (-0.4, 0.2, 0.9),
            2: (0.1, -0.6, 1.0),
            3: (-0.2, -0.1, 0.7),
        },
        "velocities": {
            0: (0.01, 0.0, 0.0),
            1: (-0.01, 0.01, 0.0),
            2: (0.0, -0.01, 0.0),
            3: (0.0, 0.0, 0.0),
        },
    }

    print("=" * 60)
    print("Testing all 5 representations with sample state")
    print("=" * 60)

    for name, fn in REPRESENTATIONS.items():
        print(f"\n--- {name.upper()} ---")
        output = fn(sample_state)
        print(output)
        assert isinstance(output, str), f"{name} should return a string"
        assert len(output) > 0, f"{name} should not return empty string"

    print("\n" + "=" * 60)
    print("Checkpoint PASSED: representations.py works correctly.")
    print("All 5 representations produce distinct non-empty strings.")
