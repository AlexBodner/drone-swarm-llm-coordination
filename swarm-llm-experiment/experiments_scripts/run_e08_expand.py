"""
run_e08_expand.py
=================
E-08-C: Expand formation (double distance from centroid)  (raw vs relative)

Task
----
For each drone i:  target = centroid + 2 * (pos_i - centroid),  z = 1 m.
The drone keeps its relative direction from the centroid but doubles its distance.

Hypothesis
----------
  relative→ ADVANTAGE: the representation directly gives each drone's offset from
             the centroid.  The LLM only needs to multiply each (dx, dy) by 2 and
             convert back to absolute — a one-step operation.
  raw     → DISADVANTAGE: the LLM must first compute the centroid (sum / N),
             then compute each drone's offset, scale by 2, and add back to centroid.
             Each step accumulates rounding error; more chain = more potential failure.

Expected outcome: relative ≥ raw  (relative repr exposes the key computation).

Setup
-----
  N = [3, 6]  ×  reprs = [raw, relative]  ×  seeds 0-9  = 40 trials
  reward: per_drone_formation_reward (direct 1:1, no Hungarian)
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
from prompt_builder import TASK_EXPAND_FORMATION
from analysis import (load_results, summarize, print_summary_table,
                      plot_reward_vs_n, plot_validity_rate,
                      plot_reward_vs_repr, plot_collision_rate)

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e08_expand"

N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative"]
SEEDS           = list(range(10))
DURATION        = 15.0
POSITION_JITTER = 0.5

print("=" * 65)
print("  E-08-C  Expand formation ×2  (raw vs relative)")
print(f"  {len(N_DRONES_LIST)} N × {len(REPRESENTATIONS)} repr × {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  Hypothesis: relative ≥ raw  (offsets given directly → just multiply by 2)")
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
    task_description=TASK_EXPAND_FORMATION,
    reward_mode="expand",
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
