"""Quick smoke test for the E-05 jitter fix."""
import sys
import numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
from simulator import SwarmSimulator

sim = SwarmSimulator(n_drones=3, gui=False)

# No jitter → default diagonal positions
s0 = sim.reset(seed=0, position_jitter=0.0)
print("seed=0 no_jitter:", [list(map(lambda x: round(x, 4), row)) for row in s0["_init_xyzs"]])

# Jitter=0.5, seed=0
s0j = sim.reset(seed=0, position_jitter=0.5)
print("seed=0 jitter=0.5:", [list(map(lambda x: round(x, 4), row)) for row in s0j["_init_xyzs"]])

# Jitter=0.5, seed=1
s1j = sim.reset(seed=1, position_jitter=0.5)
print("seed=1 jitter=0.5:", [list(map(lambda x: round(x, 4), row)) for row in s1j["_init_xyzs"]])

# Reproducibility
s0j2 = sim.reset(seed=0, position_jitter=0.5)
same = np.allclose(s0j["_init_xyzs"], s0j2["_init_xyzs"])
print(f"Reproducible (seed=0 twice): {same}")

diff = not np.allclose(s0j["_init_xyzs"], s1j["_init_xyzs"])
print(f"Different across seeds: {diff}")

sim.close()
assert same, "FAIL: jitter not reproducible"
assert diff, "FAIL: seeds produce same positions"
print("Smoke test PASSED")
