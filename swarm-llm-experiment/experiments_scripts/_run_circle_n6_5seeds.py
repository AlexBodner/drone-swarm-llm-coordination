import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

os.environ.setdefault(
    "GROQ_API_KEY",
    "YOUR_API_KEY_HERE",
)

from experiment import run_experiment
from prompt_builder import TASK_CIRCLE
from analysis import summarize, print_summary_table, plot_reward_vs_repr, plot_collision_rate

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e08_circle_n6_5seeds"
SEEDS = [0, 1, 2, 3, 4]
REPRESENTATIONS = ["raw", "relative"]

print("=" * 68)
print("  Circle experiment: N=6, seeds=0..4, raw vs relative")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 68)

start = time.time()
results = run_experiment(
    n_drones_list=[6],
    representations=REPRESENTATIONS,
    seeds=SEEDS,
    output_dir=str(OUTPUT_DIR),
    mode="waypoint",
    duration=15.0,
    task_description=TASK_CIRCLE,
    reward_mode="circle",
    output_mode="direct",
    sleep_between_trials=0.5,
    n_videos=0,
    position_jitter=0.5,
)
elapsed = time.time() - start
print(f"\nWall time: {elapsed/60:.1f} min")

summary = summarize(results)
print("\nSummary table:\n")
print_summary_table(summary)

plot_reward_vs_repr(summary, REPRESENTATIONS, n_list=[6],
                    save_path=str(OUTPUT_DIR / "reward_vs_repr.png"))
plot_collision_rate(summary, REPRESENTATIONS, n_list=[6],
                    save_path=str(OUTPUT_DIR / "collision_rate.png"))

by_seed = {}
for seed in SEEDS:
    pair = [r for r in results if r["seed"] == seed]
    pair.sort(key=lambda r: r["representation"])
    raw = next(r for r in pair if r["representation"] == "raw")
    rel = next(r for r in pair if r["representation"] == "relative")
    raw_reward = raw.get("reward")
    rel_reward = rel.get("reward")
    if raw_reward is None and rel_reward is None:
        winner = "none"
    elif rel_reward is None:
        winner = "raw"
    elif raw_reward is None:
        winner = "relative"
    else:
        winner = "raw" if raw_reward >= rel_reward else "relative"
    by_seed[seed] = {
        "raw": {
            "valid": raw.get("valid_code"),
            "reward": raw_reward,
            "latency_s": raw.get("latency_s"),
            "n_waypoints": raw.get("n_waypoints"),
        },
        "relative": {
            "valid": rel.get("valid_code"),
            "reward": rel_reward,
            "latency_s": rel.get("latency_s"),
            "n_waypoints": rel.get("n_waypoints"),
        },
        "winner": winner,
    }

comparison = {
    "config": {
        "task": "circle",
        "n_drones": 6,
        "seeds": SEEDS,
        "output_mode": "direct",
        "position_jitter": 0.5,
    },
    "summary": {
        repr_name: summary[(6, repr_name)]
        for repr_name in REPRESENTATIONS
    },
    "per_seed": by_seed,
}

with open(OUTPUT_DIR / "comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("\nPer-seed head-to-head:")
for seed in SEEDS:
    row = by_seed[seed]
    print(
        f"  seed={seed} | "
        f"raw={row['raw']['reward']} ({'valid' if row['raw']['valid'] else 'invalid'}) | "
        f"relative={row['relative']['reward']} ({'valid' if row['relative']['valid'] else 'invalid'}) | "
        f"winner={row['winner']}"
    )

print(f"\nWrote {OUTPUT_DIR / 'comparison.json'}")
