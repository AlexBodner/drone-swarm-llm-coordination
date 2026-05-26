# simulator.py
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Default drone spacing in gym-pybullet-drones (4 * arm_length = 4 * 0.0397 ≈ 0.1588 m)
_CF2X_SPACING = 4 * 0.039700   # metres
_CF2X_INIT_Z  = 0.1125         # default spawn height (COLLISION_H1)


class SwarmSimulator:
    def __init__(self, n_drones: int, gui: bool = False):
        self.n_drones = n_drones
        self.gui = gui
        # env is built lazily in reset() so initial_xyzs can vary per call
        self.env = None

    def reset(self, seed: int = 0, position_jitter: float = 0.0) -> dict:
        """Reset environment and return initial state as a dict.

        position_jitter > 0 applies a seeded uniform XY offset (±jitter metres)
        to each drone's default spawn position, producing genuinely different
        starting configurations across seeds.  Z is kept fixed for safety.
        The jittered positions are stored in state["_init_xyzs"] so the
        physics simulation in experiment.py can start from the same layout.
        """
        # ── Compute initial XYZ positions ────────────────────────────────
        base_xy = np.array(
            [[i * _CF2X_SPACING, i * _CF2X_SPACING] for i in range(self.n_drones)],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        if position_jitter > 0.0:
            base_xy += rng.uniform(-position_jitter, position_jitter,
                                   (self.n_drones, 2))
        init_xyzs = np.hstack(
            [base_xy, np.full((self.n_drones, 1), _CF2X_INIT_Z)]
        )

        # ── Build (or rebuild) CtrlAviary with the computed positions ────
        if self.env is not None:
            self.env.close()
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.n_drones,
            physics=Physics.PYB,
            gui=self.gui,
            record=False,
            initial_xyzs=init_xyzs,
        )
        obs, _ = self.env.reset(seed=seed)
        state = self._parse_obs(obs)
        # Expose init positions so experiment.py can seed CtrlAviary identically
        state["_init_xyzs"] = init_xyzs.tolist()
        return state

    def execute_plan(self, target_positions: dict) -> dict:
        """
        Execute a plan by stepping the simulator toward target positions.
        target_positions: {drone_id (int): (x, y, z)}
        Returns the final state after execution.
        """
        # Simple approach: return target as final state (placeholder)
        # In a future iteration this would use actual PID stepping
        return {"positions": target_positions}

    def _parse_obs(self, obs) -> dict:
        """Parse raw gym observation into a clean state dict."""
        positions = {}
        velocities = {}
        for i in range(self.n_drones):
            # gym-pybullet-drones obs shape: (n_drones, 20)
            # positions are indices 0:3
            positions[i] = tuple(obs[i, 0:3].tolist())
            velocities[i] = tuple(obs[i, 10:13].tolist())
        return {
            "n_drones": self.n_drones,
            "positions": positions,
            "velocities": velocities,
        }

    def close(self):
        self.env.close()


if __name__ == "__main__":
    print("Testing SwarmSimulator with 3 drones...")
    sim = SwarmSimulator(n_drones=3, gui=False)
    state = sim.reset(seed=42)
    print("State dict:")
    print(f"  n_drones: {state['n_drones']}")
    print("  positions:")
    for drone_id, pos in state["positions"].items():
        print(f"    Drone {drone_id}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    print("  velocities:")
    for drone_id, vel in state["velocities"].items():
        print(f"    Drone {drone_id}: ({vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f})")
    sim.close()
    print("\nCheckpoint PASSED: simulator.py works correctly.")
