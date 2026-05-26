import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiment import run_trial_waypoint
from prompt_builder import TASK_CIRCLE

RESULTS_PATH = Path("results_e08_circle_n3_5seeds/results.json")
OUT_DIR = Path("results_e08_circle_n3_5seeds/comparison_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_mock_llm(response_text: str):
    def _mock(_prompt: str) -> str:
        return response_text
    return _mock


def replay_seed(seed: int, repr_name: str, trial: dict) -> Path:
    out_mp4 = OUT_DIR / f"_tmp_{repr_name}_n3_seed{seed}.mp4"
    res = run_trial_waypoint(
        n_drones=3,
        representation_name=repr_name,
        seed=seed,
        task_description=TASK_CIRCLE,
        duration=15.0,
        llm_fn=make_mock_llm(trial["llm_response"]),
        llm_name=f"mock/{repr_name}",
        reward_mode="circle",
        output_mode="direct",
        record_video=True,
        video_path=str(out_mp4),
        position_jitter=0.5,
    )
    if not out_mp4.exists():
        raise RuntimeError(f"Replay video not generated for seed={seed}, repr={repr_name}. Result: {res}")
    return out_mp4


def make_sbs(seed: int, raw_trial: dict, rel_trial: dict):
    print(f"\nReplaying seed={seed} ...")
    left = replay_seed(seed, "raw", raw_trial)
    right = replay_seed(seed, "relative", rel_trial)

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        raise RuntimeError("ffmpeg not found")

    out = OUT_DIR / f"n3_circle_seed{seed}_raw_vs_relative.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(left),
        "-i", str(right),
        "-filter_complex", "hstack=inputs=2",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    try:
        left.unlink()
        right.unlink()
    except Exception:
        pass

    print(f"Done -> {out}")


if __name__ == "__main__":
    results = json.loads(RESULTS_PATH.read_text())

    by_seed = {}
    for r in results:
        if r.get("n_drones") != 3:
            continue
        by_seed.setdefault(r["seed"], {})[r["representation"]] = r

    comparable = []
    for seed, pair in by_seed.items():
        if "raw" in pair and "relative" in pair:
            raw = pair["raw"]
            rel = pair["relative"]
            if raw.get("reward") is not None and rel.get("reward") is not None:
                avg = (raw["reward"] + rel["reward"]) / 2.0
                comparable.append((seed, avg, raw, rel))

    if not comparable:
        raise RuntimeError("No comparable valid seeds found.")

    best_seed, _, best_raw, best_rel = max(comparable, key=lambda x: x[1])
    worst_seed, _, worst_raw, worst_rel = min(comparable, key=lambda x: x[1])

    print(f"Best comparable seed:  {best_seed}")
    print(f"Worst comparable seed: {worst_seed}")

    make_sbs(best_seed, best_raw, best_rel)
    if worst_seed != best_seed:
        make_sbs(worst_seed, worst_raw, worst_rel)

    summary = {
        "best_seed": best_seed,
        "worst_seed": worst_seed,
        "criterion": "max/min average reward across raw+relative for seeds where both are valid",
        "videos": [
            str(OUT_DIR / f"n3_circle_seed{best_seed}_raw_vs_relative.mp4"),
            str(OUT_DIR / f"n3_circle_seed{worst_seed}_raw_vs_relative.mp4"),
        ],
    }
    (OUT_DIR / "best_worst_selection.json").write_text(json.dumps(summary, indent=2))
    print("Wrote selection summary ->", OUT_DIR / "best_worst_selection.json")
