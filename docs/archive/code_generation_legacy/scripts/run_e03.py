"""
run_e03.py
==========
Experiment E-03: Rendezvous task — waypoint pipeline, N∈{3,6}, all 5 representations, 10 seeds.

The rendezvous task requires the LLM to read state["positions"] to compute the
initial centroid (meeting point), making it state-dependent unlike the circle task.
This lets us properly test whether state representation affects LLM planning quality.

Layout:
  results_e03/
    results.json          (all trials, saved incrementally)
    summary_table.txt     (printed at end)
    reward_vs_n.png
    reward_vs_repr.png
    validity_rate.png

Run:
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e03.py
"""

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
from prompt_builder import TASK_RENDEZVOUS
from analysis import load_results, summarize, print_summary_table, \
                     plot_reward_vs_n, plot_validity_rate, plot_reward_vs_repr, \
                     plot_collision_rate

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e03"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative", "graph", "aggregate", "natural_language"]
SEEDS           = list(range(10))   # 10 seeds × 2 N × 5 repr = 100 trials
DURATION        = 15.0              # seconds per simulation

print("=" * 65)
print("  EXPERIMENT E-03 — Rendezvous task, N∈{3,6}")
print(f"  {len(N_DRONES_LIST)} N values × {len(REPRESENTATIONS)} repr "
      f"× {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  Task: {TASK_RENDEZVOUS[:80]}...")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 65)

t_start = time.time()

results = run_experiment(
    n_drones_list=N_DRONES_LIST,
    representations=REPRESENTATIONS,
    seeds=SEEDS,
    output_dir=str(OUTPUT_DIR),
    mode="waypoint",
    duration=DURATION,
    task_description=TASK_RENDEZVOUS,
    reward_mode="rendezvous",
    output_mode="code",
    sleep_between_trials=0.5,
    n_videos=1,
)

elapsed = time.time() - t_start
print(f"\nTotal wall time: {elapsed/60:.1f} min  ({elapsed:.0f}s)")

# ── Analysis ───────────────────────────────────────────────────────────────
print("\nRunning analysis …")
results = load_results(str(OUTPUT_DIR / "results.json"))
summary = summarize(results)
print_summary_table(summary)

repr_list = REPRESENTATIONS
n_list = N_DRONES_LIST

plot_reward_vs_n(
    summary, repr_list,
    save_path=str(OUTPUT_DIR / "reward_vs_n.png"),
)
plot_validity_rate(
    summary, repr_list,
    save_path=str(OUTPUT_DIR / "validity_rate.png"),
)
plot_reward_vs_repr(
    summary, repr_list, n_list=n_list,
    save_path=str(OUTPUT_DIR / "reward_vs_repr.png"),
)
plot_collision_rate(
    summary, repr_list, n_list=n_list,
    save_path=str(OUTPUT_DIR / "collision_rate.png"),
)

# ── Per-condition collision summary ───────────────────────────────────────
print("\nCollision summary (physical threshold = 0.13 m):")
for n in n_list:
    for r in repr_list:
        s = summary.get((n, r), {})
        col_pct  = (s.get("collision_trial_rate") or 0.0) * 100
        min_dist = s.get("mean_min_dist_m")
        dist_str = f"{min_dist:.3f}m" if min_dist is not None else "N/A"
        print(f"  N={n} {r:<22} collisions={col_pct:5.0f}%  mean_min_dist={dist_str}")

print(f"\nE-03 complete. Results in {OUTPUT_DIR}/")
