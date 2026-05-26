"""
run_e06.py
==========
Experiment E-06: Rendezvous task — varied initial positions + DIRECT waypoint output.

Combines the fixes from E-05 (position_jitter=0.5 m so seeds are genuinely
independent) with the direct output mode from E-04 (LLM outputs raw JSON
waypoints instead of Python code).

Research question
-----------------
Does representation quality matter for rendezvous performance when:
  (a) initial positions truly vary across seeds, AND
  (b) the LLM must reason arithmetically (no code execution)?

This is the cleanest isolation of spatial reasoning from code-writing ability.

Comparison matrix
-----------------
  E-03: code mode,   fixed positions   (baseline)
  E-04: direct mode, fixed positions   (isolates code vs arithmetic)
  E-05: code mode,   jittered positions (isolates seed variance)
  E-06: direct mode, jittered positions (full factorial: both fixes)

Hypotheses
----------
H1: Valid rate lower than E-05 (direct JSON harder than Python code).
H2: Reward comparable to E-04 for `raw` (same arithmetic task, now with varied starts).
H3: `aggregate`/`natural_language` will degrade more than `raw`/`relative` —
    with no precise positions in the state, the LLM cannot compute the centroid.
H4: Seed variance in reward will be higher than E-05 (harder to get exact target).

Layout
------
  results_e06/
    results.json
    summary_table.txt
    reward_vs_n.png
    reward_vs_repr.png
    validity_rate.png
    collision_rate.png

Run
---
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e06.py
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
from analysis import (load_results, summarize, print_summary_table,
                      plot_reward_vs_n, plot_validity_rate,
                      plot_reward_vs_repr, plot_collision_rate)

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e06"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["raw", "relative", "graph", "aggregate", "natural_language"]
SEEDS           = list(range(10))   # 10 seeds × 2 N × 5 repr = 100 trials
DURATION        = 15.0              # seconds per simulation
POSITION_JITTER = 0.5               # metres — ±0.5 m uniform XY per drone (same as E-05)

print("=" * 65)
print("  EXPERIMENT E-06 — Rendezvous, varied positions + DIRECT JSON")
print(f"  {len(N_DRONES_LIST)} N values × {len(REPRESENTATIONS)} repr "
      f"× {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  output_mode: direct (JSON, no Python code)")
print(f"  position_jitter = ±{POSITION_JITTER} m  (seeded, reproducible)")
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
    output_mode="direct",
    sleep_between_trials=0.5,
    n_videos=1,
    position_jitter=POSITION_JITTER,
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

# ── Seed variance check ────────────────────────────────────────────────────
print("\nSeed variance check (reward std across seeds per condition):")
import json
import numpy as np
raw = json.loads((OUTPUT_DIR / "results.json").read_text())
for n in n_list:
    for r in repr_list:
        vals = [x["reward"] for x in raw
                if x.get("n_drones") == n and x.get("representation") == r
                and x.get("reward") is not None]
        if vals:
            arr = np.array(vals)
            print(f"  N={n} {r:<22} mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}")
        else:
            print(f"  N={n} {r:<22} no valid trials")

print(f"\nE-06 complete. Results in {OUTPUT_DIR}/")
