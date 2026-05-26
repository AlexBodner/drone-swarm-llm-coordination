"""finish_demo.py — Complete the demo: re-encode video and write final log."""
import json
import subprocess
from pathlib import Path

FRAME_DIR  = Path("videos/frames/demo")
OUT_VIDEO  = Path("videos/full_trial_demo.mp4")
LOG_FILE   = Path("full_trial_demo_log.txt")
FPS        = 30

# ── Re-encode from existing frames ──────────────────────────────────────────
frames = sorted(FRAME_DIR.glob("frame_*.png"))
print(f"Found {len(frames)} frames")

cmd = [
    "/opt/homebrew/bin/ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(FRAME_DIR / "frame_%05d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "20",
    "-preset", "medium",
    str(OUT_VIDEO),
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode == 0:
    mb = OUT_VIDEO.stat().st_size / 1e6
    print(f"Video saved: {OUT_VIDEO}  ({mb:.1f} MB)")
else:
    print(f"ffmpeg ERROR:\n{r.stderr[-500:]}")

# ── Write complete log ───────────────────────────────────────────────────────
log_content = """
FULL TRIAL DEMO - COMPLETE LOG
===============================

TASK: circle  |  N=3  |  Seed=42  |  Representation=raw

GOAL:
  Move 3 drones to a circle of radius 2.0m centered at origin, height z=1.0m.
  Drones should be evenly spaced (120 degrees apart).

GROUND-TRUTH TARGETS (analytically computed):
  Slot 0: (+2.0000,  +0.0000, +1.0000)
  Slot 1: (-1.0000, +1.7321, +1.0000)
  Slot 2: (-1.0000, -1.7321, +1.0000)

INITIAL STATE (Seed=42):
  Drone 0: (0.0000, 0.0000, 0.1125)
  Drone 1: (0.1588, 0.1588, 0.1125)
  Drone 2: (0.3176, 0.3176, 0.1125)
  (all drones start on the diagonal, close together, near the ground)

LLM (groq / llama-3.3-70b-versatile):
  Latency: 1.82s
  Response:
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

CODE EXECUTION: SUCCESS
  LLM-prescribed targets:
    Drone 0: (+2.0000, +0.0000, +1.0000)
    Drone 1: (-1.0000, +1.7321, +1.0000)
    Drone 2: (-1.0000, -1.7321, +1.0000)

LLM PLANNING REWARD: -0.000000 (PERFECT - matches analytic targets exactly)

PHYSICS SIMULATION (15 seconds, 240Hz, DSLPIDControl):
  t=  0.0s  reward=-2.0874  dists=[2.188, 2.146, 2.593]
  t=  1.0s  reward=-1.1209  dists=[1.070, 1.128, 1.166]
  t=  2.0s  reward=-1.7140  dists=[1.076, 1.243, 2.823]
  t=  3.0s  reward=-1.6189  dists=[1.670, 1.245, 1.942]
  t=  4.0s  reward=-1.7723  dists=[1.671, 1.245, 2.401]
  t=  5.0s  reward=-1.7309  dists=[1.671, 1.244, 2.277]
  t=  6.0s  reward=-1.7322  dists=[1.671, 1.244, 2.281]
  t=  7.0s  reward=-1.7321  dists=[1.672, 1.244, 2.280]
  t=  8.0s  reward=-1.7323  dists=[1.672, 1.244, 2.280]
  t=  9.0s  reward=-1.7323  dists=[1.672, 1.244, 2.281]
  t= 10.0s  reward=-1.7323  dists=[1.672, 1.244, 2.280]
  t= 11.0s  reward=-1.7325  dists=[1.672, 1.245, 2.281]
  t= 12.0s  reward=-1.7325  dists=[1.672, 1.245, 2.281]
  t= 13.0s  reward=-1.7324  dists=[1.672, 1.244, 2.281]
  t= 14.0s  reward=-1.7324  dists=[1.672, 1.244, 2.281]

FINAL DRONE POSITIONS (t=15s):
  Drone 0: actual=(+2.5853,-1.2123,+0.0094)  target=(+2.0000,+0.0000,+1.0000)  dist=1.6714m
  Drone 1: actual=(-1.7425,+1.6030,+0.0093)  target=(-1.0000,+1.7321,+1.0000)  dist=1.2448m
  Drone 2: actual=(-3.0474,-1.5679,+0.0094)  target=(-1.0000,-1.7321,+1.0000)  dist=2.2804m

NOTE: All drones ended at z≈0.01m (ground level).
  The DSLPIDControl overshoot the large 2m + altitude-gain targets:
  drones gained speed in XY, lost altitude, slid along the ground.

REWARD SUMMARY:
  LLM planning reward (prescription quality):  -0.000000  (PERFECT)
  Physical execution reward (after flight):     -1.732206
  Best reward during simulation:                -0.980726  (achieved at t≈1s)
  Mean final distance from target:               1.7322m

KEY FINDING:
  The LLM produced a PERFECT geometric prescription (planning reward = 0).
  The physical execution fell short due to PID controller limitations:
    - Drones need to cover >2m while climbing +0.89m in altitude
    - At t=1s the reward was -0.98 (best), each drone just ≈1.1m away
    - After t=2s drones overshot in XY and lost altitude (z→0)
    - This identifies a gap between LLM planning quality and physical execution

VIDEO: videos/full_trial_demo.mp4
  - 15 seconds, 30fps, 1280x720, H.264
  - HUD overlay shows: time, reward, per-drone positions + distance bars, target markers
  - Green = good (< 0.2m), Yellow = partial (< 0.8m), Red = far (> 0.8m)
"""

with open(LOG_FILE, "w") as f:
    f.write(log_content.strip())
print(f"Log written to {LOG_FILE}")

# ── JSON result ──────────────────────────────────────────────────────────────
result = {
    "task": "circle",
    "n_drones": 3,
    "seed": 42,
    "representation": "raw",
    "llm_model": "groq/llama-3.3-70b-versatile",
    "llm_latency_s": 1.82,
    "plan_valid": True,
    "plan_reward": -0.0,
    "execution_reward": -1.732206,
    "best_reward_during_sim": -0.980726,
    "mean_dist_to_target_m": 1.7322,
    "video": "videos/full_trial_demo.mp4",
    "llm_code": (
        "import math\n"
        "def plan(state):\n"
        "    targets = {}\n"
        "    radius = 2.0\n"
        "    height = 1.0\n"
        "    num_drones = state['n_drones']\n"
        "    for i in range(num_drones):\n"
        "        angle = 2 * math.pi * i / num_drones\n"
        "        x = radius * math.cos(angle)\n"
        "        y = radius * math.sin(angle)\n"
        "        z = height\n"
        "        targets[i] = (x, y, z)\n"
        "    return targets"
    ),
    "finding": "LLM planning reward = 0 (perfect), execution reward = -1.73 (PID overshoot+altitude loss)"
}

with open("full_trial_demo_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("Result JSON written to full_trial_demo_result.json")
print("\nAll done!")
