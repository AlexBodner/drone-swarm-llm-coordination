"""
waypoint_demo.py
================
Option A demonstration: LLM generates timed waypoint trajectories per drone.
The PID controller tracks interpolated targets from the trajectories,
giving a smooth path instead of a fixed single endpoint.

Key changes vs full_trial_demo.py:
  - build_waypoint_prompt()  → asks LLM for {drone: [(t, x, y, z), ...]}
  - execute_waypoint_plan()  → validates the trajectory dict
  - interpolate_waypoints()  → moving PID target at each sim step

Output: videos/waypoint_demo.mp4, waypoint_demo_result.json, waypoint_demo_log.txt
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import json
import time
import traceback
import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Project dir on sys.path ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from simulator      import SwarmSimulator
from reward         import circle_formation_targets, formation_reward
from representations import repr_raw
from llm_connector  import get_available_connector
from prompt_builder import build_waypoint_prompt, TASK_CIRCLE
from executor       import extract_code_block, execute_waypoint_plan, interpolate_waypoints

# ── Constants ─────────────────────────────────────────────────────────────
N_DRONES     = 3
DURATION_SEC = 15.0
VID_DIR      = SCRIPT_DIR / "videos"
VID_DIR.mkdir(exist_ok=True)

PHYSICS_HZ   = 240
CTRL_EVERY   = 5
CTRL_HZ      = PHYSICS_HZ // CTRL_EVERY   # 48 Hz
VIDEO_FPS    = 24
VIDEO_W      = 640
VIDEO_H      = 480
FRAME_EVERY  = CTRL_HZ // VIDEO_FPS       # capture every 2 control steps

# ── Logging ───────────────────────────────────────────────────────────────
_log_lines = []


def _log(msg: str = ""):
    print(msg, flush=True)
    _log_lines.append(msg)


# ── HUD drawing ───────────────────────────────────────────────────────────

def _draw_hud(rgb: np.ndarray, info: dict) -> np.ndarray:
    """Annotate a frame with simulation info using PIL."""
    img = Image.fromarray(rgb).convert("RGBA")

    # Semi-transparent top bar
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([(0, 0), (VIDEO_W, 95)], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img, overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    t    = info["sim_time"]
    r    = info["reward_now"]
    best = info["best_reward"]
    seg  = info.get("current_segment", "")

    draw.text((8,  5), f"WAYPOINT DEMO  t={t:.1f}s / {DURATION_SEC:.0f}s",
              fill=(255, 230, 60))
    draw.text((8, 25), f"Reward: {r:+.4f}   Best: {best:+.4f}",
              fill=(100, 255, 100))
    draw.text((8, 45), f"LLM: {info['llm_model']}   N={N_DRONES} drones",
              fill=(200, 200, 200))
    draw.text((8, 65), f"Seg: {seg}",
              fill=(150, 200, 255))

    # Mini top-down map (bottom-right corner)
    cx, cy, scale = VIDEO_W - 65, VIDEO_H - 65, 28
    draw.rectangle([(cx - 55, cy - 55), (cx + 55, cy + 55)],
                   fill=(20, 20, 20), outline=(80, 80, 80))

    # Target positions (yellow)
    for tgt in info["targets"]:
        px = int(cx + tgt[0] * scale)
        py = int(cy - tgt[1] * scale)
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=(220, 220, 0))

    # Drone positions + waypoint targets
    colors = [(0, 255, 100), (0, 180, 255), (255, 100, 0)]
    for i, pos in enumerate(info["positions"]):
        c = colors[i % len(colors)]
        px = int(cx + pos[0] * scale)
        py = int(cy - pos[1] * scale)
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=c)
        draw.text((px + 5, py - 5), str(i), fill=c)

    for i, wt in enumerate(info["waypoint_targets"]):
        c = colors[i % len(colors)]
        px = int(cx + wt[0] * scale)
        py = int(cy - wt[1] * scale)
        draw.line([(px - 5, py), (px + 5, py)], fill=c, width=1)
        draw.line([(px, py - 5), (px, py + 5)], fill=c, width=1)

    return np.array(img)


# ── Main demo ─────────────────────────────────────────────────────────────

def run_waypoint_demo():
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_drones": N_DRONES,
        "duration_sec": DURATION_SEC,
    }

    # 1. LLM connector
    _log()
    _log("=" * 70)
    _log("  WAYPOINT DEMO — Option A: LLM-generated timed waypoints")
    _log("=" * 70)
    _log()
    _log("[1/6] Connecting to LLM …")
    conn_name, conn_fn = get_available_connector()
    _log(f"      Using: {conn_name}")
    result["llm_connector"] = conn_name

    # 2. Simulator state
    _log()
    _log("[2/6] Resetting simulator …")
    sim = SwarmSimulator(n_drones=N_DRONES, gui=False)
    state = sim.reset(seed=42)
    targets = circle_formation_targets(N_DRONES, radius=2.0, height=1.0)
    _log("      Initial positions:")
    for i, pos in state["positions"].items():
        _log(f"        Drone {i}: {tuple(round(v, 3) for v in pos)}")
    _log("      Target circle positions (r=2m, h=1m):")
    for i, tgt in enumerate(targets):
        _log(f"        Target {i}: {tuple(round(v, 3) for v in tgt)}")

    # 3. Build prompt + query LLM
    _log()
    _log("[3/6] Building waypoint prompt …")
    state_text = repr_raw(state)
    prompt = build_waypoint_prompt(state_text, TASK_CIRCLE, N_DRONES, DURATION_SEC)
    result["prompt_length"] = len(prompt)
    _log(f"      Prompt length: {len(prompt)} chars")
    _log()
    _log("─── PROMPT " + "─" * 59)
    print(prompt, flush=True)
    _log("─" * 70)

    _log()
    _log("[4/6] Querying LLM …")
    t0 = time.time()
    llm_response = conn_fn(prompt)
    llm_time = time.time() - t0
    _log(f"      Response in {llm_time:.2f}s  ({len(llm_response)} chars)")
    result["llm_response_time_s"] = round(llm_time, 3)
    result["llm_model"] = conn_name
    _log()
    _log("─── LLM RESPONSE " + "─" * 53)
    print(llm_response, flush=True)
    _log("─" * 70)

    code = extract_code_block(llm_response)
    if not code:
        _log("ERROR: No code block in LLM response. Aborting.")
        result["error"] = "no_code_block"
        return result
    _log()
    _log("─── EXTRACTED CODE " + "─" * 51)
    print(code, flush=True)
    _log("─" * 70)

    waypoints = execute_waypoint_plan(code, state, DURATION_SEC)
    if waypoints is None:
        _log("ERROR: Waypoint plan execution failed. Aborting.")
        result["error"] = "plan_execution_failed"
        return result

    _log()
    _log("      Waypoints per drone:")
    for did, wps in waypoints.items():
        _log(f"        Drone {did} ({len(wps)} waypoints):")
        for wp in wps:
            _log(f"          t={wp[0]:5.2f}s  ({wp[1]:.3f}, {wp[2]:.3f}, {wp[3]:.3f})")
    result["waypoints"] = {str(k): [list(wp) for wp in v] for k, v in waypoints.items()}

    # 5. Physics simulation
    _log()
    _log("[5/6] Running physics simulation …")
    _log(f"      Duration: {DURATION_SEC:.0f}s  "
         f"|  Control @ {CTRL_HZ}Hz  |  Video @ {VIDEO_FPS}fps")

    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    import pybullet as p

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=N_DRONES,
        physics=Physics.PYB,
        gui=False,
        record=False,
    )
    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N_DRONES)]

    view_mat = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=6, yaw=0, pitch=-70, roll=0,
        upAxisIndex=2, physicsClientId=env.CLIENT,
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60, aspect=VIDEO_W / VIDEO_H,
        nearVal=0.1, farVal=20, physicsClientId=env.CLIENT,
    )

    obs, _ = env.reset(seed=42)
    actions = np.tile([[env.HOVER_RPM] * 4], (N_DRONES, 1))

    total_steps   = int(DURATION_SEC * CTRL_HZ)
    reward_history = []
    best_reward    = float("-inf")
    frame_dir      = Path(tempfile.mkdtemp(prefix="wpdemo_frames_"))
    frame_count    = 0

    for step in range(total_steps):
        current_t = step / CTRL_HZ
        waypoint_targets = []

        for i in range(N_DRONES):
            target_pos = np.array(interpolate_waypoints(waypoints[i], current_t))
            waypoint_targets.append(target_pos.tolist())

            rpm, _, _ = controllers[i].computeControl(
                control_timestep=CTRL_EVERY / PHYSICS_HZ,
                cur_pos=obs[i, 0:3],
                cur_quat=obs[i, 3:7],
                cur_vel=obs[i, 10:13],
                cur_ang_vel=obs[i, 13:16],
                target_pos=target_pos,
                target_vel=np.zeros(3),
            )
            actions[i] = rpm

        for phys_step in range(CTRL_EVERY):
            obs, _, terminated, _, _ = env.step(actions)

        pos_dict = {i: tuple(obs[i, 0:3]) for i in range(N_DRONES)}
        r = formation_reward(pos_dict, targets)
        reward_history.append(float(r))
        if float(r) > best_reward:
            best_reward = float(r)

        if step % FRAME_EVERY == 0:
            _, _, rgb_raw, _, _ = p.getCameraImage(
                width=VIDEO_W, height=VIDEO_H,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.CLIENT,
            )
            rgb = np.reshape(rgb_raw, (VIDEO_H, VIDEO_W, 4))[:, :, :3].astype(np.uint8)

            # Build waypoint segment labels
            seg_parts = []
            for did in range(N_DRONES):
                wps = waypoints[did]
                seg = "end"
                for wi in range(len(wps) - 1):
                    if wps[wi][0] <= current_t < wps[wi + 1][0]:
                        seg = f"D{did}:{wi}→{wi+1}"
                        break
                seg_parts.append(seg)

            frame = _draw_hud(rgb, {
                "sim_time": current_t,
                "reward_now": r,
                "best_reward": best_reward,
                "llm_model": conn_name,
                "positions": [obs[i, 0:3].tolist() for i in range(N_DRONES)],
                "targets": [list(tgt) for tgt in targets],
                "waypoint_targets": waypoint_targets,
                "current_segment": " | ".join(seg_parts),
            })
            Image.fromarray(frame).save(frame_dir / f"frame_{frame_count:05d}.png")
            frame_count += 1

        if step % (CTRL_HZ * 3) == 0:
            _log(f"      t={current_t:5.1f}s  reward={r:+.4f}  best={best_reward:+.4f}  "
                 + "  ".join(
                     f"D{i}=({obs[i,0]:.2f},{obs[i,1]:.2f},{obs[i,2]:.2f})"
                     for i in range(N_DRONES)
                 ))

        if terminated:
            _log(f"      ** DRONE CRASH at t={current_t:.2f}s **")
            break

    env.close()

    # 6. Encode video
    _log()
    _log("[6/6] Encoding video …")
    video_path  = VID_DIR / "waypoint_demo.mp4"
    ffmpeg_bin  = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    enc_result  = subprocess.run(
        [ffmpeg_bin, "-y",
         "-framerate", str(VIDEO_FPS),
         "-i", str(frame_dir / "frame_%05d.png"),
         "-c:v", "libx264",
         "-pix_fmt", "yuv420p",
         "-crf", "20",
         "-preset", "medium",
         str(video_path)],
        capture_output=True, text=True,
    )
    if enc_result.returncode == 0:
        size_mb = video_path.stat().st_size / 1e6
        _log(f"      Saved {frame_count} frames → {video_path}  ({size_mb:.1f} MB)")
    else:
        _log(f"      ffmpeg ERROR: {enc_result.stderr[-500:]}")
    shutil.rmtree(frame_dir, ignore_errors=True)

    # Results summary
    final_reward = reward_history[-1] if reward_history else float("nan")
    result["final_reward"] = round(final_reward, 4)
    result["best_reward"]  = round(best_reward, 4)
    for sec in [1, 5, 10]:
        idx = min(sec * CTRL_HZ, len(reward_history) - 1)
        result[f"reward_at_{sec}s"] = round(reward_history[idx], 4)
    result["video_frames"] = frame_count
    result["video_path"]   = str(video_path)

    _log()
    _log("=" * 70)
    _log("  RESULTS")
    _log("=" * 70)
    for sec in [1, 5, 10]:
        _log(f"  Reward at {sec:2d}s  : {result[f'reward_at_{sec}s']:.4f}")
    _log(f"  Final reward  : {final_reward:.4f}")
    _log(f"  Best reward   : {best_reward:.4f}")
    _log(f"  Video frames  : {frame_count}")
    _log(f"  Video path    : {video_path}")
    if abs(best_reward) < 0.1:
        _log("  Interpretation: EXCELLENT — formation within 10cm")
    elif abs(best_reward) < 0.5:
        _log("  Interpretation: GOOD — formation within 50cm")
    elif abs(best_reward) < 1.0:
        _log("  Interpretation: PARTIAL — formation recognizable")
    else:
        _log("  Interpretation: POOR — waypoints did not converge well")
    _log("=" * 70)

    return result


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in environment.")

    result = run_waypoint_demo()

    result_path = SCRIPT_DIR / "waypoint_demo_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved → {result_path}", flush=True)

    log_path = SCRIPT_DIR / "waypoint_demo_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(_log_lines))
    print(f"Log saved    → {log_path}", flush=True)
