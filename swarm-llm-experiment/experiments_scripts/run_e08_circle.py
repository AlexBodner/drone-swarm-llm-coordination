"""
run_e08_circle.py
=================
E-08-A: Circle formation at the world ORIGIN  (raw vs relative, focused comparison)

Task
----
Form a circle of radius 2 m centred at (0, 0), z = 1 m.
Target is ABSOLUTE and FIXED — the LLM does NOT need to read the state at all
(the answer is the same regardless of initial positions).

Hypothesis
----------
  raw     → ADVANTAGE: absolute target (2*cos θ, 2*sin θ, 1.0) can be written
             directly without any state-reading.  The representation gives exactly
             the information the LLM needs and nothing more.
  relative→ DISADVANTAGE: the centroid offset frame adds an unnecessary conversion
             step.  LLM must remember that origin targets are absolute, not
             centroid-relative.  Risk of confusing the two frames.

Expected outcome: raw ≥ relative in both validity rate and mean reward.

Setup
-----
  N = [3, 6]  ×  reprs = [raw, relative]  ×  seeds 0-9  = 40 trials
  output_mode: direct (JSON waypoints)
  position_jitter: ±0.5 m  (positions vary per seed; task target does NOT)
  reward: formation_reward with Hungarian assignment to circle slots
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault(
    "GROQ_API_KEY",
    "YOUR_API_KEY_HERE",
)

from experiment import run_experiment
from prompt_builder import TASK_CIRCLE
from analysis import (load_results, summarize, print_summary_table,
                      plot_reward_vs_n, plot_validity_rate,
                      plot_reward_vs_repr, plot_collision_rate)

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e08_circle"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative"]
SEEDS           = list(range(10))
DURATION        = 15.0
POSITION_JITTER = 0.5

print("=" * 65)
print("  E-08-A  Circle at origin  (raw vs relative)")
print(f"  {len(N_DRONES_LIST)} N × {len(REPRESENTATIONS)} repr × {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  Hypothesis: raw ≥ relative  (absolute fixed target)")
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
    task_description=TASK_CIRCLE,
    reward_mode="circle",
    output_mode="direct",
    sleep_between_trials=0.5,
    n_videos=1,
    position_jitter=POSITION_JITTER,
)

elapsed = time.time() - t_start
print(f"\nTotal wall time: {elapsed/60:.1f} min")

# ── Analysis ───────────────────────────────────────────────────────────────
print("\nRunning analysis …")
results = load_results(str(OUTPUT_DIR / "results.json"))
summary = summarize(results)
print_summary_table(summary)

plot_reward_vs_n(summary, REPRESENTATIONS, save_path=str(OUTPUT_DIR / "reward_vs_n.png"))
plot_validity_rate(summary, REPRESENTATIONS, save_path=str(OUTPUT_DIR / "validity_rate.png"))
plot_reward_vs_repr(summary, REPRESENTATIONS, n_list=N_DRONES_LIST,
                    save_path=str(OUTPUT_DIR / "reward_vs_repr.png"))
plot_collision_rate(summary, REPRESENTATIONS, n_list=N_DRONES_LIST,
                    save_path=str(OUTPUT_DIR / "collision_rate.png"))

# ── Head-to-head raw vs relative ──────────────────────────────────────────
print("\n--- Raw vs Relative head-to-head ---")
for n in N_DRONES_LIST:
    raw_s = summary.get((n, "raw"), {})
    rel_s = summary.get((n, "relative"), {})
    raw_r = raw_s.get("mean_reward")
    rel_r = rel_s.get("mean_reward")
    raw_v = raw_s.get("validity_rate", 0) * 100
    rel_v = rel_s.get("validity_rate", 0) * 100
    winner = "raw" if (raw_r or -999) >= (rel_r or -999) else "relative"
    print(f"  N={n}  raw: {raw_r:.4f if raw_r else 'N/A':>8}  ({raw_v:.0f}% valid) |"
          f"  relative: {rel_r:.4f if rel_r else 'N/A':>8}  ({rel_v:.0f}% valid)"
          f"  → {winner} wins")
