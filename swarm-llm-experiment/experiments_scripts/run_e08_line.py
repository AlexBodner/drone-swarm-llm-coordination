"""
run_e08_line.py
===============
E-08-E: Line formation centred at initial centroid  (raw vs relative)

Task
----
Arrange all drones in a line parallel to the X-axis, centred at the centroid of
their initial positions, spaced 1 m apart, all at z = 1 m.
drone i → x = cx + (i - (N-1)/2) * 1.0,  y = cy,  z = 1.0

Hypothesis
----------
  relative→ ADVANTAGE: (cx, cy) is given explicitly.  The LLM only needs to apply
             the spacing formula without computing a centroid from scratch.
  raw     → DISADVANTAGE: must sum N x-values and N y-values, divide by N to get
             (cx, cy), then apply the spacing formula.  Any error in the centroid
             propagates to all targets.

This task also tests whether the model can handle fractional offsets for even N
(e.g. N=4 → offsets -1.5, -0.5, +0.5, +1.5).

Expected outcome: relative ≥ raw.

Setup
-----
  N = [3, 6]  ×  reprs = [raw, relative]  ×  seeds 0-9  = 40 trials
  reward: formation_reward with Hungarian assignment (line slot assignment)
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
from prompt_builder import TASK_LINE_FORMATION
from analysis import (load_results, summarize, print_summary_table,
                      plot_reward_vs_n, plot_validity_rate,
                      plot_reward_vs_repr, plot_collision_rate)

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e08_line"

N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative"]
SEEDS           = list(range(10))
DURATION        = 15.0
POSITION_JITTER = 0.5

print("=" * 65)
print("  E-08-E  Line formation at centroid  (raw vs relative)")
print(f"  {len(N_DRONES_LIST)} N × {len(REPRESENTATIONS)} repr × {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  Hypothesis: relative ≥ raw  (centroid given directly in repr)")
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
    task_description=TASK_LINE_FORMATION,
    reward_mode="line",
    output_mode="direct",
    sleep_between_trials=0.5,
    n_videos=1,
    position_jitter=POSITION_JITTER,
)

elapsed = time.time() - t_start
print(f"\nTotal wall time: {elapsed/60:.1f} min")

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

print("\n--- Raw vs Relative head-to-head ---")
for n in N_DRONES_LIST:
    raw_s = summary.get((n, "raw"), {})
    rel_s = summary.get((n, "relative"), {})
    raw_r  = raw_s.get("mean_reward")
    rel_r  = rel_s.get("mean_reward")
    raw_v  = raw_s.get("validity_rate", 0) * 100
    rel_v  = rel_s.get("validity_rate", 0) * 100
    winner = "raw" if (raw_r or -999) >= (rel_r or -999) else "relative"
    print(f"  N={n}  raw: {raw_r:.4f if raw_r else 'N/A':>8}  ({raw_v:.0f}% valid) |"
          f"  relative: {rel_r:.4f if rel_r else 'N/A':>8}  ({rel_v:.0f}% valid)"
          f"  → {winner} wins")
