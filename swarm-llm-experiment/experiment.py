# experiment.py
import json
import time
import os
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from simulator import SwarmSimulator
from representations import REPRESENTATIONS
from reward import (
    circle_formation_targets, scatter_circle_targets, line_targets_from_centroid,
    swap_target_positions, expand_target_positions,
    formation_reward, per_drone_formation_reward, rendezvous_reward,
)
from executor import (extract_code_block, execute_plan_code, execute_waypoint_plan,
                      parse_direct_waypoints, interpolate_waypoints)
from prompt_builder import (build_prompt, build_waypoint_prompt,
                            build_direct_waypoint_prompt,
                            TASK_CIRCLE, TASK_RENDEZVOUS,
                            TASK_SWAP_POSITIONS, TASK_EXPAND_FORMATION,
                            TASK_SCATTER_CIRCLE, TASK_LINE_FORMATION)
from llm_connector import get_completion, get_available_connector

# ── Physics constants ────────────────────────────────────────────────────
PHYSICS_HZ = 240
CTRL_EVERY = 5
CTRL_HZ    = PHYSICS_HZ // CTRL_EVERY   # 48 Hz

# ── Collision detection threshold ────────────────────────────────────────
# CF2X physical size: arm_length L=0.0397m, propeller_radius=0.0231m
# Tip-to-tip diameter = 2*(L + r_prop) ≈ 0.126m — two drones touch when
# centre-to-centre distance falls below this value.
COLLISION_RADIUS = 0.13  # metres — drone pair is "colliding" if closer than this
                         # (physical propeller tip-to-tip diameter + ~3% margin)


def run_trial(
    n_drones: int,
    representation_name: str,
    seed: int,
    task_description: str = TASK_CIRCLE,
) -> dict:
    """
    Original single-endpoint trial (planning reward only, no physics).
    Kept for backwards compatibility and comparison.
    """
    sim = SwarmSimulator(n_drones=n_drones, gui=False)
    state = sim.reset(seed=seed)

    repr_fn = REPRESENTATIONS[representation_name]
    state_text = repr_fn(state)
    prompt = build_prompt(state_text, task_description, n_drones)

    t0 = time.time()
    llm_response = get_completion(prompt)
    latency = time.time() - t0

    code = extract_code_block(llm_response)
    target_positions = execute_plan_code(code, state)

    if target_positions is None:
        reward = None
        valid = False
    else:
        targets = circle_formation_targets(n_drones)
        reward = formation_reward(target_positions, targets)
        valid = True

    sim.close()

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "single_endpoint",
        "n_drones": n_drones,
        "representation": representation_name,
        "seed": seed,
        "prompt": prompt,
        "llm_response": llm_response,
        "code": code,
        "valid_code": valid,
        "reward": reward,
        "latency_s": latency,
    }


def run_trial_waypoint(
    n_drones: int,
    representation_name: str,
    seed: int,
    task_description: str = TASK_CIRCLE,
    duration: float = 15.0,
    llm_fn=None,
    llm_name: str = "",
    reward_mode: str = "circle",    # "circle" or "rendezvous"
    output_mode: str = "direct",    # "code" (exec Python) or "direct" (parse JSON waypoints)
    record_video: bool = False,
    video_path: str = None,
    position_jitter: float = 0.0,   # metres; 0 = fixed default positions
) -> dict:
    """
    Waypoint trial: LLM generates timed trajectories, physics simulation runs
    and returns execution reward (how well drones actually formed the shape).

    Returns a result dict with both planning metadata and physical execution metrics.
    """
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

    if llm_fn is None:
        _, base_fn = get_available_connector()
        # Scale output token budget with swarm size to avoid truncated JSON.
        # More drones → more waypoints per drone → larger output.
        # - N=3  → ~2500 tokens  (3 drones × up to 6 waypoints)
        # - N=6  → ~4500 tokens  (6 drones × up to 8 waypoints)
        # - N=12 → ~8192 tokens  (capped at llama-3.1-8b-instant max)
        _max_tokens = min(8192, 1500 + n_drones * 550)
        llm_fn = lambda p: base_fn(p, max_tokens=_max_tokens)

    # ── State from simulator ─────────────────────────────────────────────
    sim = SwarmSimulator(n_drones=n_drones, gui=False)
    state = sim.reset(seed=seed, position_jitter=position_jitter)
    sim.close()

    if reward_mode == "rendezvous":
        # Target = initial centroid; LLM must read state["positions"] to find it
        positions_array = np.array(list(state["positions"].values()))
        initial_centroid = positions_array.mean(axis=0)
        initial_centroid[2] = 1.0   # enforce target height
        targets = None              # not used for rendezvous
        per_drone_targets = None
    elif reward_mode == "circle":
        initial_centroid = None
        targets = circle_formation_targets(n_drones)
        per_drone_targets = None
    elif reward_mode == "swap":
        positions_array = np.array(list(state["positions"].values()))
        _centroid = positions_array.mean(axis=0)
        _centroid[2] = 1.0
        initial_centroid = _centroid          # shown in HUD/minimap
        targets = None
        per_drone_targets = swap_target_positions(state["positions"])
    elif reward_mode == "expand":
        positions_array = np.array(list(state["positions"].values()))
        _centroid = positions_array.mean(axis=0)
        _centroid[2] = 1.0
        initial_centroid = _centroid
        targets = None
        per_drone_targets = expand_target_positions(state["positions"])
    elif reward_mode == "scatter_circle":
        positions_array = np.array(list(state["positions"].values()))
        _centroid = positions_array.mean(axis=0)
        _centroid[2] = 1.0
        initial_centroid = _centroid
        targets = scatter_circle_targets(n_drones, _centroid)
        per_drone_targets = None
    elif reward_mode == "line":
        positions_array = np.array(list(state["positions"].values()))
        _centroid = positions_array.mean(axis=0)
        _centroid[2] = 1.0
        initial_centroid = _centroid
        targets = line_targets_from_centroid(n_drones, _centroid)
        per_drone_targets = None
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode!r}. "
                         f"Valid modes: rendezvous, circle, swap, expand, scatter_circle, line")

    repr_fn = REPRESENTATIONS[representation_name]
    state_text = repr_fn(state)
    if output_mode == "direct":
        prompt = build_direct_waypoint_prompt(state_text, task_description, n_drones, duration)
    else:
        prompt = build_waypoint_prompt(state_text, task_description, n_drones, duration)

    # ── LLM call ─────────────────────────────────────────────────────────
    t0 = time.time()
    llm_response = llm_fn(prompt)
    latency = time.time() - t0

    if output_mode == "direct":
        waypoints = parse_direct_waypoints(llm_response, state, duration)
        if waypoints is None:
            return _failed_result(n_drones, representation_name, seed, latency, llm_name,
                                  "direct_parse_failed", mode="waypoint", output_mode=output_mode,
                                  llm_response=llm_response, prompt_length=len(prompt))
        code = None  # no code generated in direct mode
    else:
        code = extract_code_block(llm_response)
        if not code:
            return _failed_result(n_drones, representation_name, seed, latency, llm_name,
                                  "no_code_block", mode="waypoint", output_mode=output_mode,
                                  llm_response=llm_response, prompt_length=len(prompt))
        waypoints = execute_waypoint_plan(code, state, duration)
        if waypoints is None:
            return _failed_result(n_drones, representation_name, seed, latency, llm_name,
                                  "plan_execution_failed", mode="waypoint", output_mode=output_mode,
                                  llm_response=llm_response, prompt_length=len(prompt), code=code)

    # ── Physics simulation ────────────────────────────────────────────────
    _init_xyzs_arg = {}
    if "_init_xyzs" in state:
        _init_xyzs_arg["initial_xyzs"] = np.array(state["_init_xyzs"])
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=n_drones,
        physics=Physics.PYB,
        gui=False,
        record=False,
        **_init_xyzs_arg,
    )
    controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(n_drones)]
    obs, _ = env.reset(seed=seed)
    actions = np.tile([[env.HOVER_RPM] * 4], (n_drones, 1))
    # ── Video setup ───────────────────────────────────────────────
    VID_W, VID_H, VID_FPS = 640, 480, 24
    FRAME_EVERY = CTRL_HZ // VID_FPS  # capture every Nth control step
    frame_dir = None
    frame_count = 0
    view_mat = proj_mat = None

    if record_video:
        import pybullet as p
        frame_dir = Path(tempfile.mkdtemp(prefix="exp_frames_"))
        view_mat = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=5.5, yaw=0, pitch=-65, roll=0,
            upAxisIndex=2, physicsClientId=env.CLIENT,
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=VID_W / VID_H,
            nearVal=0.1, farVal=20, physicsClientId=env.CLIENT,
        )
        # Drone colours for top-down map
        _DRONE_COLORS = [
            (0, 255, 100), (0, 180, 255), (255, 100, 0),
            (255, 0, 200), (255, 220, 0), (100, 255, 255),
        ]

    def _draw_frame(rgb_arr, current_t, r, best_r, wpt_targets):
        """Annotate a raw RGB frame with HUD and mini-map."""
        img = Image.fromarray(rgb_arr).convert("RGBA")
        ov  = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).rectangle([(0, 0), (VID_W, 80)], fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img, ov).convert("RGB")
        draw = ImageDraw.Draw(img)
        model_short = llm_name.split("/")[-1] if llm_name else "?"
        task_short  = reward_mode.upper()
        draw.text((8,  4), f"{task_short}  N={n_drones}  repr={representation_name}  seed={seed}", fill=(255, 230, 60))
        draw.text((8, 20), f"t={current_t:.1f}s / {duration:.0f}s    reward={r:+.4f}   best={best_r:+.4f}", fill=(100, 255, 100))
        draw.text((8, 36), f"LLM: {model_short}", fill=(200, 200, 200))
        if initial_centroid is not None:
            _centroid_label = "Target:" if reward_mode == "rendezvous" else "Centroid:"
            draw.text(
                (8, 52),
                f"{_centroid_label} ({initial_centroid[0]:+.2f}, {initial_centroid[1]:+.2f}, {initial_centroid[2]:.2f})",
                fill=(255, 255, 255),
            )
        # Mini top-down map (bottom-right)
        cx, cy, scale = VID_W - 65, VID_H - 65, 26
        draw.rectangle([(cx-58, cy-58), (cx+58, cy+58)], fill=(15, 15, 15), outline=(70, 70, 70))
        # Draw targets on minimap
        if per_drone_targets is not None:
            # Per-drone coloured crosshairs (swap / expand tasks)
            for i in range(n_drones):
                tgt = per_drone_targets.get(i)
                if tgt is not None:
                    c = _DRONE_COLORS[i % len(_DRONE_COLORS)]
                    px, py = int(cx + tgt[0]*scale), int(cy - tgt[1]*scale)
                    draw.line([(px-6, py), (px+6, py)], fill=c, width=2)
                    draw.line([(px, py-6), (px, py+6)], fill=c, width=2)
        elif targets is not None:
            # List of targets with Hungarian assignment (circle, scatter_circle, line)
            for tgt in targets:
                px, py = int(cx + tgt[0]*scale), int(cy - tgt[1]*scale)
                draw.ellipse([(px-3, py-3), (px+3, py+3)], fill=(220, 220, 0))
        elif initial_centroid is not None:
            # Rendezvous: single meeting-point crosshair
            px, py = int(cx + initial_centroid[0]*scale), int(cy - initial_centroid[1]*scale)
            draw.line([(px-6, py), (px+6, py)], fill=(255, 255, 255), width=2)
            draw.line([(px, py-6), (px, py+6)], fill=(255, 255, 255), width=2)
            draw.text((px + 8, py - 16), "C", fill=(255, 255, 255))
        # For tasks with list targets, also show initial_centroid as a reference dot
        if initial_centroid is not None and targets is not None:
            px, py = int(cx + initial_centroid[0]*scale), int(cy - initial_centroid[1]*scale)
            draw.ellipse([(px-3, py-3), (px+3, py+3)], outline=(200, 200, 200))
            draw.text((px + 5, py - 12), "C", fill=(200, 200, 200))
        # Draw drones
        for i in range(n_drones):
            pos = obs[i, 0:3]
            c = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            px, py = int(cx + pos[0]*scale), int(cy - pos[1]*scale)
            draw.ellipse([(px-4, py-4), (px+4, py+4)], fill=c)
            draw.text((px+5, py-5), str(i), fill=c)
        # Draw active waypoint targets as crosshairs
        for i, wpt in enumerate(wpt_targets):
            c = _DRONE_COLORS[i % len(_DRONE_COLORS)]
            px, py = int(cx + wpt[0]*scale), int(cy - wpt[1]*scale)
            draw.line([(px-4, py), (px+4, py)], fill=c, width=1)
            draw.line([(px, py-4), (px, py+4)], fill=c, width=1)
        return np.array(img)
    total_steps = int(duration * CTRL_HZ)
    reward_history = []
    best_reward = float("-inf")
    terminated_at = None
    min_pairwise_dist = float("inf")   # closest any two drones ever got
    collision_steps = 0                # steps where at least one pair < COLLISION_RADIUS

    for step in range(total_steps):
        current_t = step / CTRL_HZ
        wpt_targets = []

        for i in range(n_drones):
            target_pos = np.array(interpolate_waypoints(waypoints[i], current_t))
            wpt_targets.append(target_pos.tolist())
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

        for _ in range(CTRL_EVERY):
            obs, _, terminated, _, _ = env.step(actions)

        pos_dict = {i: tuple(obs[i, 0:3]) for i in range(n_drones)}
        if reward_mode == "rendezvous":
            r = float(rendezvous_reward(pos_dict, initial_centroid))
        elif reward_mode in ("swap", "expand"):
            r = float(per_drone_formation_reward(pos_dict, per_drone_targets))
        else:  # circle, scatter_circle, line
            r = float(formation_reward(pos_dict, targets))
        reward_history.append(r)
        if r > best_reward:
            best_reward = r

        # ── Collision tracking ──────────────────────────────────────────
        step_min_dist = float("inf")
        any_collision_this_step = False
        for _i in range(n_drones):
            for _j in range(_i + 1, n_drones):
                _d = float(np.linalg.norm(obs[_i, 0:3] - obs[_j, 0:3]))
                if _d < step_min_dist:
                    step_min_dist = _d
                if _d < COLLISION_RADIUS:
                    any_collision_this_step = True
        if step_min_dist < min_pairwise_dist:
            min_pairwise_dist = step_min_dist
        if any_collision_this_step:
            collision_steps += 1

        # ── Video frame capture ─────────────────────────────────────
        if record_video and (step % FRAME_EVERY == 0):
            _, _, rgb_raw, _, _ = p.getCameraImage(
                width=VID_W, height=VID_H,
                viewMatrix=view_mat, projectionMatrix=proj_mat,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.CLIENT,
            )
            rgb = np.reshape(rgb_raw, (VID_H, VID_W, 4))[:, :, :3].astype(np.uint8)
            frame = _draw_frame(rgb, current_t, r, best_reward, wpt_targets)
            Image.fromarray(frame).save(frame_dir / f"frame_{frame_count:05d}.png")
            frame_count += 1

        if terminated:
            terminated_at = current_t
            break

    env.close()

    # ── Encode video if recorded ─────────────────────────────────────────
    saved_video = None
    if record_video and frame_dir and frame_count > 0:
        out = Path(video_path) if video_path else Path(tempfile.mktemp(suffix=".mp4"))
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        res = subprocess.run(
            [ffmpeg, "-y", "-framerate", str(VID_FPS),
             "-i", str(frame_dir / "frame_%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-preset", "fast", str(out)],
            capture_output=True, text=True,
        )
        shutil.rmtree(frame_dir, ignore_errors=True)
        if res.returncode == 0:
            saved_video = str(out)
        else:
            print(f"[video] ffmpeg error: {res.stderr[-300:]}", flush=True)

    final_reward = reward_history[-1] if reward_history else float("nan")
    steps_run = len(reward_history)
    collision_fraction = collision_steps / steps_run if steps_run > 0 else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "waypoint",
        "task": reward_mode,
        "n_drones": n_drones,
        "representation": representation_name,
        "seed": seed,
        "llm_model": llm_name,
        "duration_s": duration,
        "prompt_length": len(prompt),
        "llm_response": llm_response,
        "code": code,
        "valid_code": True,
        "reward": final_reward,          # execution reward (what counts)
        "best_reward": best_reward,
        "reward_at_1s": _reward_at(reward_history, 1),
        "reward_at_5s": _reward_at(reward_history, 5),
        "reward_at_10s": _reward_at(reward_history, 10),
        "latency_s": latency,
        "terminated_early": terminated_at is not None,
        "terminated_at_s": terminated_at,        "output_mode": output_mode,        # ── Collision metrics ──────────────────────────────────────
        "min_pairwise_dist_m": round(min_pairwise_dist, 4),   # closest any two drones got
        "collision_steps": collision_steps,                    # steps with any pair < COLLISION_RADIUS
        "collision_fraction": round(collision_fraction, 4),    # fraction of trial in collision
        "collision_radius_m": COLLISION_RADIUS,                # threshold used
        # ──────────────────────────────────────────────────────────
        "n_waypoints": {str(k): len(v) for k, v in waypoints.items()},
        "video": saved_video,
    }


def _reward_at(history: list, sec: int) -> float:
    idx = min(sec * CTRL_HZ, len(history) - 1)
    return round(history[idx], 4) if history else float("nan")


def _failed_result(n, repr_name, seed, latency, llm_name, error,
                   mode="waypoint", output_mode="code",
                   llm_response=None, prompt_length=None, code=None) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "output_mode": output_mode,
        "n_drones": n,
        "representation": repr_name,
        "seed": seed,
        "llm_model": llm_name,
        "prompt_length": prompt_length,
        "llm_response": llm_response,
        "code": code,
        "valid_code": False,
        "reward": None,
        "best_reward": None,
        "latency_s": latency,
        "error": error,
    }


def run_experiment(
    n_drones_list: list = [3, 6, 10, 15],
    representations: list = list(REPRESENTATIONS.keys()),
    seeds: list = list(range(20)),
    output_dir: str = "results",
    sleep_between_trials: float = 0.0,
    mode: str = "waypoint",       # "waypoint" or "single_endpoint"
    duration: float = 15.0,
    task_description: str = TASK_CIRCLE,
    reward_mode: str = "circle",  # "circle" or "rendezvous"
    output_mode: str = "direct",  # "code" (exec Python) or "direct" (parse JSON waypoints)
    n_videos: int = 1,            # record this many videos per (N, repr) condition (first seeds)
    position_jitter: float = 0.0, # metres; 0 = fixed default spawn positions
):
    """
    Run a full sweep of (N, representation, seed) trials.

    mode="waypoint"          → run_trial_waypoint() with physics simulation
    mode="single_endpoint"   → run_trial() with analytical planning reward only
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    # Use a single LLM connector for the whole experiment
    llm_name, llm_fn = get_available_connector()
    print(f"LLM: {llm_name}  |  mode: {mode}  |  output: {output_dir}")

    total = len(n_drones_list) * len(representations) * len(seeds)
    done = 0
    vid_count: dict = {}   # (n, repr_name) → number of videos recorded so far

    for n in n_drones_list:
        for repr_name in representations:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] N={n}, repr={repr_name}, seed={seed}", end=" ... ", flush=True)

                # Record video for exactly the first n_videos seeds (by index, not by success).
                # Using a fixed seed index ensures raw and relative always capture the same
                # initial configuration for direct visual comparison.
                cond_key = (n, repr_name)
                vid_count.setdefault(cond_key, 0)
                seed_index = seeds.index(seed)
                should_record = (mode == "waypoint") and (n_videos > 0) and (seed_index < n_videos)
                vpath = (f"{output_dir}/videos/n{n}_{repr_name}_seed{seed}.mp4"
                         if should_record else None)

                try:
                    if mode == "waypoint":
                        result = run_trial_waypoint(
                            n_drones=n,
                            representation_name=repr_name,
                            seed=seed,
                            task_description=task_description,
                            duration=duration,
                            llm_fn=llm_fn,
                            llm_name=llm_name,
                            reward_mode=reward_mode,
                            output_mode=output_mode,
                            record_video=should_record,
                            video_path=vpath,
                            position_jitter=position_jitter,
                        )
                        if should_record and result.get("video"):
                            vid_count[cond_key] += 1
                    else:
                        result = run_trial(n, repr_name, seed, task_description)

                    status = (f"reward={result['reward']:.4f}" if result["reward"] is not None
                              else f"INVALID ({result.get('error', '?')})")
                    vid_tag = f"  📹 {result['video']}" if result.get("video") else ""
                    print(f"{status}  (latency={result['latency_s']:.1f}s){vid_tag}")

                except Exception as e:
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "mode": mode,
                        "n_drones": n,
                        "representation": repr_name,
                        "seed": seed,
                        "valid_code": False,
                        "reward": None,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"ERROR: {e}")

                all_results.append(result)

                # Save incrementally (safe against crashes/interruptions)
                with open(f"{output_dir}/results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

                if sleep_between_trials > 0:
                    time.sleep(sleep_between_trials)

    print(f"\nDone. {done} trials saved to {output_dir}/results.json")
    return all_results


if __name__ == "__main__":
    # Quick smoke test: N=3, raw repr, 1 seed, waypoint mode
    # Ensure GROQ_API_KEY is set in your environment or .env file
    result = run_experiment(
        n_drones_list=[3],
        representations=["raw"],
        seeds=[42],
        output_dir="results_smoke_test",
        mode="waypoint",
    )
    print(result[0]["reward"])
