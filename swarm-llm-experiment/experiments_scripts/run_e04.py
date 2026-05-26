"""
run_e04.py
==========
Experiment E-04: Rendezvous task — DIRECT WAYPOINT OUTPUT (no code execution).

Same task and conditions as E-03 (N∈{3,6}, 5 representations, 10 seeds = 100 trials),
but the LLM is asked to output waypoints as raw JSON numbers instead of Python code.

Motivation
----------
E-03 revealed that the LLM solves the rendezvous task by writing Python code that
computes the centroid programmatically (sum/divide loops). This does not test
spatial reasoning — it tests whether the LLM can write correct arithmetic code.

In E-04 (output_mode="direct"), the LLM must:
  1. Read the numeric positions from the state description.
  2. Compute the centroid arithmetically in plain language (show the steps).
  3. Output the waypoints as a JSON list of [t, x, y, z] tuples — no code allowed.

This isolates "in-context arithmetic reasoning" from "ability to write Python math".

Hypotheses
----------
H1: Valid rate will DROP vs E-03 (harder to produce well-formed JSON than runnable code).
H2: `raw` reward will be comparable to E-03 if the LLM actually computes the centroid.
H3: `aggregate` advantage will DISAPPEAR or SHRINK because the LLM can no longer
    just forward the centroid string into a variable — it must parse and write numbers.
H4: `natural_language` will remain the worst — still no quantitative info in state.

Layout
------
  results_e04/
    results.json
    reward_vs_n.png
    reward_vs_repr.png
    validity_rate.png
    collision_rate.png

Run
---
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e04.py
"""

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault(
    "GROQ_API_KEY",
    "YOUR_API_KEY_HERE",
)

from experiment import run_experiment
from prompt_builder import TASK_RENDEZVOUS
from analysis import (load_results, summarize, print_summary_table,
                      plot_reward_vs_n, plot_validity_rate,
                      plot_reward_vs_repr, plot_collision_rate)

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e04"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative", "graph", "aggregate", "natural_language"]
SEEDS           = list(range(10))   # 10 seeds × 2 N × 5 repr = 100 trials
DURATION        = 15.0              # seconds per simulation

print("=" * 65)
print("  EXPERIMENT E-04 — Rendezvous, DIRECT JSON waypoints")
print(f"  {len(N_DRONES_LIST)} N values × {len(REPRESENTATIONS)} repr "
      f"× {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  output_mode: direct (JSON, no Python code)")
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
    output_mode="direct",            # ← key difference from E-03
    sleep_between_trials=0.3,
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

# ── Per-condition collision summary ────────────────────────────────────────
print("\nCollision summary (physical threshold = 0.13 m):")
for n in n_list:
    for r in repr_list:
        s = summary.get((n, r), {})
        col_pct  = (s.get("collision_trial_rate") or 0.0) * 100
        min_dist = s.get("mean_min_dist_m")
        dist_str = f"{min_dist:.3f}m" if min_dist is not None else "N/A"
        print(f"  N={n} {r:<22} collisions={col_pct:5.0f}%  mean_min_dist={dist_str}")

# ── E-03 vs E-04 comparison ────────────────────────────────────────────────
e03_path = SCRIPT_DIR / "results_e03" / "results.json"
if e03_path.exists():
    print("\n" + "=" * 80)
    print("  E-03 (code) vs E-04 (direct) comparison — mean reward")
    print("=" * 80)
    e03_results = load_results(str(e03_path))
    e03_summary = summarize(e03_results)
    print(f"{'N':>4} | {'Representation':<22} | {'E-03 (code)':>12} | {'E-04 (direct)':>13} | {'Δ':>8}")
    print("-" * 70)
    for n in n_list:
        for r in repr_list:
            r03 = e03_summary.get((n, r), {}).get("mean_reward")
            r04 = summary.get((n, r), {}).get("mean_reward")
            r03_s = f"{r03:+.4f}" if r03 is not None else "    N/A"
            r04_s = f"{r04:+.4f}" if r04 is not None else "     N/A"
            if r03 is not None and r04 is not None:
                delta_s = f"{r04-r03:+.4f}"
            else:
                delta_s = "    N/A"
            print(f"{n:>4} | {r:<22} | {r03_s:>12} | {r04_s:>13} | {delta_s:>8}")
    print("=" * 80)
else:
    print("\n(E-03 results not found; skipping comparison)")

print(f"\nE-04 complete. Results in {OUTPUT_DIR}/")
