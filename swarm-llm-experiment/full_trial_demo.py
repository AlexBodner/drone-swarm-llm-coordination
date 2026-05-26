"""
full_trial_demo.py
------------------
A single, fully observable trial that shows EVERYTHING:
  - The task goal
  - The LLM prompt and response
  - The generated plan code
  - The simulation execution with PID control
  - Reward tracked at every second
  - A final MP4 video with on-screen annotations

Run:
    conda activate swarm-llm
    GROQ_API_KEY=... python full_trial_demo.py

Output:
    videos/full_trial_demo.mp4   — annotated video
    full_trial_demo_log.txt      — complete human-readable log
"""

import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import traceback
from pathlib import Path

import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
N_DRONES       = 3
SEED           = 42
REPRESENTATION = "raw"          # which state repr to use (change to test others)
TASK_NAME      = "circle"       # matches a key in TASKS below
SIM_HZ         = 240
CTRL_HZ        = 48
DURATION_S     = 15.0           # enough for PID to fully converge
VIDEO_W        = 1280
VIDEO_H        = 720
FPS            = 30
CAMERA_DIST    = 6.0
CAMERA_YAW     = 50
CAMERA_PITCH   = -38
CAMERA_TARGET  = [0.5, 0.5, 0.5]
OUTPUT_DIR     = Path("videos")
FRAME_DIR      = OUTPUT_DIR / "frames" / "demo"
LOG_FILE       = Path("full_trial_demo_log.txt")

TASKS = {
    "circle": {
        "description": (
            "Move the drones to form a circle of radius 2.0 meters centered "
            "at the origin (0, 0), at height z = 1.0 meters. "
            "The drones should be evenly spaced around the circle."
        ),
        "compute_targets": lambda n: [
            (2.0 * np.cos(a), 2.0 * np.sin(a), 1.0)
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
        ],
    },
}


# ─── Imports that require the conda env ────────────────────────────────────────
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

from simulator import SwarmSimulator
from representations import REPRESENTATIONS
from reward import formation_reward
from executor import extract_code_block, execute_plan_code
from prompt_builder import build_prompt
from llm_connector import get_completion


# ─── Helpers ───────────────────────────────────────────────────────────────────

def hungarian_reward(positions_dict: dict, targets: list) -> float:
    return formation_reward(positions_dict, targets)


def draw_hud(frame_rgb: np.ndarray, info: dict) -> np.ndarray:
    """Overlay a HUD (text panel) on a frame using PIL."""
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    # Semi-transparent dark panel top-left
    panel_w, panel_h = 520, 290
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    panel = ImageDraw.Draw(overlay)
    panel.rectangle([(10, 10), (panel_w, panel_h)], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Text
    font_size = 17
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    WHITE  = (255, 255, 255)
    YELLOW = (255, 230, 50)
    GREEN  = (80, 255, 100)
    RED    = (255, 100, 80)
    CYAN   = (80, 220, 255)

    y = 18
    def line(text, color=WHITE, f=None):
        nonlocal y
        draw.text((18, y), text, fill=color, font=f or font)
        y += font_size + 4

    line(f"TASK: {info['task_name'].upper()}", YELLOW)
    line(f"Representation: {info['repr_name']}  |  N={info['n']}  |  Seed={info['seed']}", CYAN)
    line(f"LLM: {info['llm_model']}", CYAN)
    line("─" * 50, (100, 100, 100))
    line(f"Time: {info['sim_time']:.1f}s / {info['total_time']:.1f}s")
    line(f"Reward now:  {info['reward_now']:+.4f}  (perfect = 0.0000)")
    reward_color = GREEN if info["reward_now"] > -0.3 else (YELLOW if info["reward_now"] > -1.0 else RED)
    line(f"Best reward: {info['best_reward']:+.4f}", reward_color)
    line("─" * 50, (100, 100, 100))

    for i, (pos, tgt, dist) in enumerate(zip(info["positions"], info["targets"], info["distances"])):
        bar_filled = max(0, int(20 * (1 - dist / 3.0)))
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        c = GREEN if dist < 0.2 else (YELLOW if dist < 0.8 else RED)
        line(f"  D{i} ({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) → {dist:.3f}m {bar}", c, font_small)

    # Bottom banner: plan quality
    if info.get("plan_valid"):
        status = "✓ LLM PLAN VALID"
        sc = GREEN
    else:
        status = "✗ LLM PLAN INVALID"
        sc = RED
    draw.rectangle([(0, VIDEO_H - 36), (VIDEO_W, VIDEO_H)], fill=(0, 0, 0, 200))
    draw.text((18, VIDEO_H - 26), status, fill=sc, font=font)
    draw.text((VIDEO_W // 2 - 80, VIDEO_H - 26),
              f"LLM planning reward: {info['plan_reward']:+.4f}", fill=YELLOW, font=font)

    return np.array(img)


# ─── Main demo ─────────────────────────────────────────────────────────────────

def run_demo():
    log_lines = []

    def log(msg, also_print=True):
        log_lines.append(msg)
        if also_print:
            print(msg)
            sys.stdout.flush()

    CTRL_EVERY  = int(SIM_HZ / CTRL_HZ)
    FRAME_EVERY = max(1, int(SIM_HZ / FPS))
    task_cfg    = TASKS[TASK_NAME]

    # ── SECTION 0: Setup ───────────────────────────────────────────────────────
    log("=" * 70)
    log("  FULL TRIAL DEMO — LLM-COORDINATED DRONE SWARM")
    log("=" * 70)
    log(f"  Task         : {TASK_NAME}")
    log(f"  N drones     : {N_DRONES}")
    log(f"  Seed         : {SEED}")
    log(f"  Representation: {REPRESENTATION}")
    log(f"  Sim duration : {DURATION_S}s at {SIM_HZ}Hz physics / {CTRL_HZ}Hz control")
    log(f"  Video        : {VIDEO_W}x{VIDEO_H} @ {FPS}fps")
    log("")

    # ── SECTION 1: State extraction ───────────────────────────────────────────
    log("─" * 70)
    log("STEP 1 — GET INITIAL STATE FROM SIMULATOR")
    sim_wrapper = SwarmSimulator(n_drones=N_DRONES, gui=False)
    state = sim_wrapper.reset(seed=SEED)
    sim_wrapper.close()

    log(f"  n_drones : {state['n_drones']}")
    log("  positions (x, y, z):")
    for did, pos in state["positions"].items():
        log(f"    Drone {did}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    log("  velocities: all near zero at reset")
    log("")

    # ── SECTION 2: Representations ────────────────────────────────────────────
    log("─" * 70)
    log("STEP 2 — STATE REPRESENTATION (what the LLM will see)")
    repr_fn = REPRESENTATIONS[REPRESENTATION]
    state_text = repr_fn(state)
    log(f"  Representation type: '{REPRESENTATION}'")
    log("")
    log(state_text)
    log("")

    # ── SECTION 3: Goal / targets ─────────────────────────────────────────────
    log("─" * 70)
    log("STEP 3 — TASK DEFINITION & GROUND-TRUTH TARGETS")
    log(f"  Task description:\n    \"{task_cfg['description']}\"")
    log("")
    targets_list = task_cfg["compute_targets"](N_DRONES)
    log(f"  Ground-truth target positions (computed analytically):")
    for i, t in enumerate(targets_list):
        log(f"    Slot {i}: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
    targets_np = np.array(targets_list)
    log("")

    # ── SECTION 4: Prompt ─────────────────────────────────────────────────────
    log("─" * 70)
    log("STEP 4 — FULL PROMPT SENT TO LLM")
    prompt = build_prompt(state_text, task_cfg["description"], N_DRONES)
    log("")
    log(prompt)
    log("")

    # ── SECTION 5: LLM call ───────────────────────────────────────────────────
    log("─" * 70)
    log("STEP 5 — LLM API CALL")

    from llm_connector import get_available_connector
    llm_name, _ = get_available_connector()
    log(f"  LLM backend  : {llm_name}")
    log(f"  Temperature  : 0 (deterministic / greedy)")
    log("  Calling LLM...")

    import time
    t0 = time.time()
    llm_response = get_completion(prompt)
    latency = time.time() - t0

    log(f"  Latency      : {latency:.2f}s")
    log(f"  Response length: {len(llm_response)} chars")
    log("")
    log("  ── RAW LLM RESPONSE ──────────────────────────────────────────")
    log(llm_response)
    log("  ──────────────────────────────────────────────────────────────")
    log("")

    # ── SECTION 6: Code extraction & execution ────────────────────────────────
    log("─" * 70)
    log("STEP 6 — CODE EXTRACTION & PLAN EXECUTION")
    code = extract_code_block(llm_response)
    log(f"  Extracted code ({len(code)} chars):")
    log("")
    for ln in code.split("\n"):
        log(f"    {ln}")
    log("")

    llm_targets_dict = execute_plan_code(code, state)
    plan_valid = llm_targets_dict is not None

    log(f"  Code execution: {'SUCCESS ✓' if plan_valid else 'FAILED ✗'}")
    if plan_valid:
        log("  LLM-prescribed target positions:")
        for did, pos in llm_targets_dict.items():
            log(f"    Drone {did}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")

        # Planning reward: how good is the LLM's prescription?
        plan_reward = hungarian_reward(llm_targets_dict, targets_list)
        log("")
        log(f"  ┌────────────────────────────────────────────────────────┐")
        log(f"  │  LLM PLANNING REWARD (position quality, no physics)    │")
        log(f"  │  reward = {plan_reward:+.6f}                               │")
        log(f"  │  (= negative mean dist from LLM targets to true slots) │")
        log(f"  └────────────────────────────────────────────────────────┘")
        log("")

        # Use LLM targets as PID targets (they should match the circle)
        pid_targets = np.array([llm_targets_dict[i] for i in range(N_DRONES)])
    else:
        plan_reward = float("-inf")
        log("  Falling back to analytic circle targets for PID execution.")
        pid_targets = targets_np.copy()
    log("")

    # ── SECTION 7: Physical simulation ───────────────────────────────────────
    log("─" * 70)
    log("STEP 7 — PHYSICS SIMULATION (PID control toward LLM targets)")
    log(f"  Duration: {DURATION_S}s | Physics: {SIM_HZ}Hz | Control: {CTRL_HZ}Hz")
    log("")

    # Create fresh env
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=N_DRONES,
        physics=Physics.PYB,
        gui=False,
        record=False,
    )
    obs, _ = env.reset(seed=SEED)

    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N_DRONES)]
    # Start at hover RPM so drones don't drop before first PID update
    actions = np.tile([[env.HOVER_RPM] * 4], (N_DRONES, 1)).astype(float)

    # Camera setup for recording
    import pybullet as p

    # Draw target markers in the scene
    for i, tgt in enumerate(pid_targets):
        for dx, dy in [(-0.2, 0), (0.2, 0), (0, -0.2), (0, 0.2)]:
            p.addUserDebugLine(
                [tgt[0] + dx/2, tgt[1] + dy/2, tgt[2]],
                [tgt[0] + dx, tgt[1] + dy, tgt[2]],
                lineColorRGB=[0, 1, 0], lineWidth=3,
                physicsClientId=env.CLIENT,
            )
        p.addUserDebugText(
            f"Target {i}", [tgt[0], tgt[1], tgt[2] + 0.25],
            textColorRGB=[0.2, 1, 0.2], textSize=1.3,
            physicsClientId=env.CLIENT,
        )

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET,
        distance=CAMERA_DIST, yaw=CAMERA_YAW, pitch=CAMERA_PITCH,
        roll=0, upAxisIndex=2, physicsClientId=env.CLIENT,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=VIDEO_W / VIDEO_H, nearVal=0.1, farVal=100.0,
        physicsClientId=env.CLIENT,
    )

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAME_DIR.glob("frame_*.png"):
        f.unlink()

    total_steps = int(DURATION_S * SIM_HZ)
    frame_count = 0
    best_reward = float("-inf")
    reward_history = []
    reward_log_seconds = []

    for step in range(total_steps):
        # PID update
        if step % CTRL_EVERY == 0:
            for i in range(N_DRONES):
                rpm, _, _ = controllers[i].computeControl(
                    control_timestep=1.0 / CTRL_HZ,
                    cur_pos=obs[i, 0:3],
                    cur_quat=obs[i, 3:7],
                    cur_vel=obs[i, 10:13],
                    cur_ang_vel=obs[i, 13:16],
                    target_pos=pid_targets[i],
                )
                actions[i] = np.clip(rpm, 0, env.MAX_RPM)

        obs, _, terminated, _, _ = env.step(actions)

        # Reward at this instant
        pos_dict = {i: tuple(obs[i, 0:3].tolist()) for i in range(N_DRONES)}
        reward_now = hungarian_reward(pos_dict, targets_list)
        reward_history.append(reward_now)
        if reward_now > best_reward:
            best_reward = reward_now

        # Log every second
        if step % SIM_HZ == 0:
            t_sec = step / SIM_HZ
            dists = [np.linalg.norm(obs[i, 0:3] - pid_targets[i]) for i in range(N_DRONES)]
            reward_log_seconds.append((t_sec, reward_now, dists[:]))
            log(f"  t={t_sec:5.1f}s  reward={reward_now:+.4f}  "
                f"dists=[{', '.join(f'{d:.3f}' for d in dists)}]")

        # Capture frame with HUD
        if step % FRAME_EVERY == 0:
            _, _, rgb_raw, _, _ = p.getCameraImage(
                width=VIDEO_W, height=VIDEO_H,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT,
            )
            rgb = np.reshape(rgb_raw, (VIDEO_H, VIDEO_W, 4))[:, :, :3].astype(np.uint8)
            dists_now = [np.linalg.norm(obs[i, 0:3] - pid_targets[i]) for i in range(N_DRONES)]
            hud_info = {
                "task_name": TASK_NAME,
                "repr_name": REPRESENTATION,
                "n": N_DRONES,
                "seed": SEED,
                "llm_model": llm_name,
                "sim_time": step / SIM_HZ,
                "total_time": DURATION_S,
                "reward_now": reward_now,
                "best_reward": best_reward,
                "positions": [obs[i, 0:3].tolist() for i in range(N_DRONES)],
                "targets": [pid_targets[i].tolist() for i in range(N_DRONES)],
                "distances": dists_now,
                "plan_valid": plan_valid,
                "plan_reward": plan_reward,
            }
            annotated = draw_hud(rgb, hud_info)
            Image.fromarray(annotated).save(FRAME_DIR / f"frame_{frame_count:05d}.png")
            frame_count += 1

        if terminated:
            log(f"  ** DRONE CRASH at t={step/SIM_HZ:.2f}s **")
            # Pad remaining frames with last frame
            for pad_i in range(total_steps - step - 1):
                if (step + pad_i) % FRAME_EVERY == 0:
                    Image.fromarray(annotated).save(
                        FRAME_DIR / f"frame_{frame_count:05d}.png"
                    )
                    frame_count += 1
            break

    env.close()
    log("")

    # ── SECTION 8: Final reward ────────────────────────────────────────────────
    try:
        final_pos_dict = {i: tuple(obs[i, 0:3].tolist()) for i in range(N_DRONES)}
        final_reward = hungarian_reward(final_pos_dict, targets_list)
        final_dists = [np.linalg.norm(obs[i, 0:3] - targets_np[i]) for i in range(N_DRONES)]

        log("─" * 70)
        log("STEP 8 — FINAL RESULTS SUMMARY")
        log("")
        log(f"  Final drone positions:")
        for i in range(N_DRONES):
            pos = obs[i, 0:3]
            tgt = pid_targets[i]
            log(f"    Drone {i}: actual=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})  "
                f"target=({tgt[0]:+.4f},{tgt[1]:+.4f},{tgt[2]:+.4f})  "
                f"dist={final_dists[i]:.4f}m")
        log("")
        log(f"  REWARD SUMMARY")
        log(f"  LLM planning reward (pure position quality):   {plan_reward:+.6f}  (0 = perfect)")
        log(f"  Physical execution reward (after PID flight):  {final_reward:+.6f}")
        log(f"  Best reward during simulation:                 {best_reward:+.6f}")
        log(f"  Mean dist to target: {np.mean(final_dists):.4f}m")
        log("")
        log(f"  Interpretation:")
        if abs(final_reward) < 0.1:
            log(f"    EXCELLENT -- drones reached formation within 10cm")
        elif abs(final_reward) < 0.3:
            log(f"    GOOD -- drones reached formation within 30cm")
        elif abs(final_reward) < 1.0:
            log(f"    PARTIAL -- formation recognizable but imprecise")
        else:
            log(f"    POOR -- formation did not fully converge (PID limitation at 2m range)")
        log("")
    except Exception as e:
        log(f"  [STEP 8 ERROR] {e}")
        log(traceback.format_exc())
        final_reward = float("nan")
        final_dists = [float("nan")] * N_DRONES

    # ── SECTION 9: Encode video ───────────────────────────────────────────────
    log("─" * 70)
    log("STEP 9 — ENCODING VIDEO")
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "full_trial_demo.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(FRAME_DIR / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        size_mb = out_path.stat().st_size / 1e6
        log(f"  Video saved: {out_path}  ({size_mb:.1f} MB)")
    else:
        log(f"  ffmpeg ERROR: {result.stderr[-300:]}")
    log("")

    # ── SECTION 10: Save log ──────────────────────────────────────────────────
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Complete log saved to: {LOG_FILE}")
    log("")

    # ── SECTION 11: Print JSON result dict ───────────────────────────────────
    result_dict = {
        "task": TASK_NAME,
        "n_drones": N_DRONES,
        "seed": SEED,
        "representation": REPRESENTATION,
        "llm_model": llm_name,
        "llm_latency_s": round(latency, 3),
        "plan_valid": plan_valid,
        "plan_reward": round(plan_reward, 6) if plan_valid else None,
        "execution_reward": round(final_reward, 6),
        "best_reward_during_sim": round(best_reward, 6),
        "mean_dist_to_target_m": round(float(np.mean(final_dists)), 4),
        "video": str(out_path),
        "llm_response": llm_response,
        "code": code,
    }
    log("─" * 70)
    log("RESULT DICT (JSON):")
    log(json.dumps({k: v for k, v in result_dict.items() if k not in ("llm_response", "code")}, indent=2))
    log("=" * 70)


if __name__ == "__main__":
    run_demo()
