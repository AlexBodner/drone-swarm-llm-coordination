# record_simulation.py
"""
Records three simulation scenarios and saves them as MP4 videos.

## Scenarios
  1. circle_n3  — N=3 drones form a circle (radius 2.0m)
  2. circle_n6  — N=6 drones form a circle (radius 2.5m)
  3. rendezvous — N=5 drones converge to their swarm centroid

## How it works
  - Runs fully in DIRECT (headless) mode — no GUI needed
  - Uses DSLPIDControl to move each drone toward its target position
  - Captures frames with pybullet.getCameraImage (isometric 3rd-person view)
  - Assembles frames into MP4 with ffmpeg

## Requirements
  - conda activate swarm-llm
  - ffmpeg must be on PATH (brew install ffmpeg)

## Usage
  python record_simulation.py
  python record_simulation.py --scenario circle_n6
  python record_simulation.py --all
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ── Video settings ─────────────────────────────────────────────────────────────
VIDEO_WIDTH   = 1280
VIDEO_HEIGHT  = 720
FPS           = 30
CAMERA_DIST   = 5.0
CAMERA_YAW    = 45
CAMERA_PITCH  = -40
CAMERA_TARGET = [0.0, 0.0, 0.5]

# ── Simulation settings ────────────────────────────────────────────────────────
SIM_HZ     = 240        # physics steps per second (gym-pybullet-drones default)
CTRL_HZ    = 48         # control updates per second
DURATION_S = 8.0        # seconds of simulation per scenario
CTRL_EVERY = int(SIM_HZ / CTRL_HZ)   # physics steps between control updates
FRAME_EVERY = max(1, int(SIM_HZ / FPS))  # physics steps between captured frames

OUTPUT_DIR = Path("videos")


# ── Target generators ──────────────────────────────────────────────────────────

def circle_targets(n: int, radius: float = 2.0, height: float = 1.0) -> np.ndarray:
    """N evenly-spaced points on a circle in the XY plane at given height."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    targets = np.zeros((n, 3))
    targets[:, 0] = radius * np.cos(angles)
    targets[:, 1] = radius * np.sin(angles)
    targets[:, 2] = height
    return targets


def rendezvous_targets(initial_positions: np.ndarray, height: float = 1.0) -> np.ndarray:
    """All drones converge to the centroid of their initial positions at given height."""
    centroid = initial_positions.mean(axis=0)
    targets = np.tile(centroid, (len(initial_positions), 1))
    targets[:, 2] = height
    return targets


# ── Scenario definitions ───────────────────────────────────────────────────────

SCENARIOS = {
    "circle_n3": {
        "label": "Circle Formation — N=3",
        "n_drones": 3,
        "get_targets": lambda init_pos: circle_targets(3, radius=2.0, height=1.0),
        "camera_dist": 5.0,
        "camera_pitch": -35,
    },
    "circle_n6": {
        "label": "Circle Formation — N=6",
        "n_drones": 6,
        "get_targets": lambda init_pos: circle_targets(6, radius=2.5, height=1.5),
        "camera_dist": 7.0,
        "camera_pitch": -40,
    },
    "rendezvous": {
        "label": "Rendezvous (Converge to Centroid) — N=5",
        "n_drones": 5,
        "get_targets": lambda init_pos: rendezvous_targets(init_pos, height=1.0),
        "camera_dist": 5.0,
        "camera_pitch": -30,
    },
}


# ── Core simulation + recording ────────────────────────────────────────────────

def run_and_record(scenario_name: str) -> Path:
    """
    Run a scenario, capture frames, assemble MP4. Returns path to output video.
    """
    cfg = SCENARIOS[scenario_name]
    n = cfg["n_drones"]
    print(f"\n{'='*60}")
    print(f"  Recording: {cfg['label']}")
    print(f"  Duration : {DURATION_S}s  |  Physics: {SIM_HZ}Hz  |  Video: {FPS}fps")
    print(f"{'='*60}")

    # ── 1. Create environment ──────────────────────────────────────────────────
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=n,
        physics=Physics.PYB,
        gui=False,
        record=False,        # we capture frames manually for better camera control
    )
    obs, _ = env.reset(seed=0)

    # Grab initial positions
    init_positions = np.array([obs[i, 0:3] for i in range(n)])
    targets = cfg["get_targets"](init_positions)
    print(f"  Initial positions:\n{init_positions.round(3)}")
    print(f"  Target  positions:\n{targets.round(3)}")

    # ── 2. PID controllers (one per drone) ───────────────────────────────────
    controllers = [
        DSLPIDControl(drone_model=DroneModel.CF2X)
        for _ in range(n)
    ]
    ctrl_timestep = 1.0 / CTRL_HZ

    # ── 3. Frame capture setup ────────────────────────────────────────────────
    frame_dir = OUTPUT_DIR / "frames" / scenario_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    # Clear old frames
    for f in frame_dir.glob("frame_*.png"):
        f.unlink()

    # Camera parameters
    cam_dist   = cfg.get("camera_dist", CAMERA_DIST)
    cam_pitch  = cfg.get("camera_pitch", CAMERA_PITCH)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET,
        distance=cam_dist,
        yaw=CAMERA_YAW,
        pitch=cam_pitch,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.CLIENT,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=env.CLIENT,
    )

    # ── 4. Draw target markers in simulation ─────────────────────────────────
    for i, tgt in enumerate(targets):
        # Draw a small cross at each target position
        p.addUserDebugLine(
            [tgt[0] - 0.15, tgt[1], tgt[2]],
            [tgt[0] + 0.15, tgt[1], tgt[2]],
            lineColorRGB=[0, 1, 0], lineWidth=2,
            physicsClientId=env.CLIENT,
        )
        p.addUserDebugLine(
            [tgt[0], tgt[1] - 0.15, tgt[2]],
            [tgt[0], tgt[1] + 0.15, tgt[2]],
            lineColorRGB=[0, 1, 0], lineWidth=2,
            physicsClientId=env.CLIENT,
        )
        p.addUserDebugText(
            f"T{i}",
            [tgt[0], tgt[1], tgt[2] + 0.2],
            textColorRGB=[0, 1, 0],
            textSize=1.2,
            physicsClientId=env.CLIENT,
        )

    # ── 5. Simulation loop ────────────────────────────────────────────────────
    total_steps = int(DURATION_S * SIM_HZ)
    frame_count = 0
    actions = np.zeros((n, 4))
    t0 = time.time()

    for step in range(total_steps):
        # Update PID every CTRL_EVERY physics steps
        if step % CTRL_EVERY == 0:
            cur_obs = obs  # shape (n, 20)
            for i in range(n):
                pos  = cur_obs[i, 0:3]
                quat = cur_obs[i, 3:7]
                vel  = cur_obs[i, 10:13]
                rpm, _, _ = controllers[i].computeControl(
                    control_timestep=ctrl_timestep,
                    cur_pos=pos,
                    cur_quat=quat,
                    cur_vel=vel,
                    cur_ang_vel=cur_obs[i, 13:16],
                    target_pos=targets[i],
                )
                actions[i] = rpm

        # Step physics
        obs, _, _, _, _ = env.step(actions)

        # Capture frame
        if step % FRAME_EVERY == 0:
            _, _, rgb, _, _ = p.getCameraImage(
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.CLIENT,
            )
            img = np.reshape(rgb, (VIDEO_HEIGHT, VIDEO_WIDTH, 4))[:, :, :3]  # drop alpha
            # Save with PIL
            from PIL import Image as PILImage
            PILImage.fromarray(img.astype(np.uint8)).save(
                frame_dir / f"frame_{frame_count:05d}.png"
            )
            frame_count += 1

    elapsed = time.time() - t0
    print(f"  Simulation done: {total_steps} steps in {elapsed:.1f}s, {frame_count} frames captured")

    # Print final distances to targets
    final_positions = np.array([obs[i, 0:3] for i in range(n)])
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(final_positions[:, None, :] - targets[None, :, :], axis=-1)
    r_idx, c_idx = linear_sum_assignment(cost)
    mean_dist = cost[r_idx, c_idx].mean()
    print(f"  Final mean distance to target: {mean_dist:.4f}m")

    env.close()

    # ── 6. Assemble MP4 with ffmpeg ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    video_path = OUTPUT_DIR / f"{scenario_name}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        str(video_path),
    ]
    print(f"  Assembling video: {video_path}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error:\n{result.stderr}")
    else:
        size_mb = video_path.stat().st_size / 1e6
        print(f"  Video saved: {video_path}  ({size_mb:.1f} MB)")

    return video_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record drone swarm simulation videos")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default=None,
        help="Record a single scenario by name",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Record all scenarios",
    )
    args = parser.parse_args()

    if args.all or (not args.scenario):
        for name in SCENARIOS:
            run_and_record(name)
    else:
        run_and_record(args.scenario)

    print("\nDone. Videos saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
