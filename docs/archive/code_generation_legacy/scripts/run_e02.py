"""
run_e02.py
==========
Experiment E-02: Waypoint pipeline, N∈{3,6}, all 5 representations, 5 seeds.

Layout:
  results_e02/
    results.json          (all trials, saved incrementally)
    summary_table.txt     (printed at end)
    reward_vs_n.png
    reward_vs_repr.png
    validity_rate.png

Run:
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e02.py
"""

import os
import sys
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

os.environ.setdefault(
    "GROQ_API_KEY",
    "YOUR_API_KEY_HERE",
)

from experiment import run_experiment
from analysis import load_results, summarize, print_summary_table, \
                     plot_reward_vs_n, plot_validity_rate, plot_reward_vs_repr

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e02"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative", "graph", "aggregate", "natural_language"]
SEEDS           = list(range(5))    # 5 seeds × 2 N × 5 repr = 50 trials
DURATION        = 15.0              # seconds per simulation
# Each trial takes ~15s physics + LLM call ≈ 18s wall time → ~15 min total

print("=" * 65)
print("  EXPERIMENT E-02 — Waypoint pipeline, N∈{3,6}")
print(f"  {len(N_DRONES_LIST)} N values × {len(REPRESENTATIONS)} repr "
      f"× {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
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
    sleep_between_trials=0.5,
)

elapsed = time.time() - t_start
print(f"\nTotal wall time: {elapsed/60:.1f} min  ({elapsed:.0f}s)")

# ── Analysis ───────────────────────────────────────────────────────────────
print("\nRunning analysis …")
results = load_results(str(OUTPUT_DIR / "results.json"))
summary = summarize(results)
print_summary_table(summary)

repr_list = REPRESENTATIONS

plot_reward_vs_n(
    summary, repr_list,
    save_path=str(OUTPUT_DIR / "reward_vs_n.png"),
)
plot_validity_rate(
    summary, repr_list,
    save_path=str(OUTPUT_DIR / "validity_rate.png"),
)
plot_reward_vs_repr(
    summary, repr_list,
    n_list=N_DRONES_LIST,
    save_path=str(OUTPUT_DIR / "reward_vs_repr.png"),
)

print(f"\nE-02 complete. Results in {OUTPUT_DIR}/")
