"""
run_e07.py
==========
Experiment E-07: Rendezvous — same config as E-06, with fixed `relative` representation.

What changed vs E-06
--------------------
The `relative` repr in representations.py now explicitly tells the LLM:
  - The offsets shown are relative to the centroid, NOT absolute positions.
  - Conversion formula is shown with the actual centroid values:
      absolute_pos = centroid + offset
      x = cx + dx,  y = cy + dy,  z = cz + dz

In E-06, `relative` achieved 0% valid rate because the LLM copied the Δ-offsets
directly as absolute coordinates, yielding z=0.0 (below ground) for all trials.
This fix tests whether a clear coordinate-system note resolves that failure.

Config (identical to E-06)
--------------------------
  output_mode:      direct (JSON — no Python code)
  position_jitter:  ±0.5 m  (seeded XY jitter per drone)
  N:                {3, 6}
  representations:  all 5
  seeds:            0–9  (10 seeds)
  total trials:     100

Comparison matrix
-----------------
  E-03: code mode,   fixed positions   (baseline)
  E-04: direct mode, fixed positions
  E-05: code mode,   jittered positions
  E-06: direct mode, jittered positions          ← `relative` broken
  E-07: direct mode, jittered positions          ← `relative` FIXED (this run)

Output
------
  results_e07/
    results.json
    summary_table.txt
    reward_vs_n.png
    reward_vs_repr.png
    validity_rate.png
    collision_rate.png

Run
---
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e07.py
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

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e07"

# ── Experiment configuration ───────────────────────────────────────────────
N_DRONES_LIST   = [3, 6]
REPRESENTATIONS = ["relative"]  # only re-running the repr that was broken in E-06
SEEDS           = list(range(10))   # 10 seeds × 2 N × 1 repr = 20 trials
DURATION        = 15.0              # seconds per simulation
POSITION_JITTER = 0.5               # metres — ±0.5 m uniform XY per drone (same as E-06)

print("=" * 70)
print("  EXPERIMENT E-07 — `relative` repr fix validation")
print("  Runs only the `relative` representation (was 0% valid in E-06).")
print("  All other repr results are taken from E-06 (unchanged).")
print(f"  {len(N_DRONES_LIST)} N values × {len(REPRESENTATIONS)} repr "
      f"× {len(SEEDS)} seeds = "
      f"{len(N_DRONES_LIST)*len(REPRESENTATIONS)*len(SEEDS)} trials")
print(f"  output_mode: direct (JSON, no Python code)")
print(f"  position_jitter = ±{POSITION_JITTER} m  (seeded, reproducible)")
print(f"  Task: {TASK_RENDEZVOUS[:80]}...")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 70)

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

# ── Per-condition proximity summary ───────────────────────────────────────
print("\nProximity summary (physical threshold = 0.13 m):")
print("(Prox%=100% is expected for rendezvous — all drones converge to same point)")
for n in n_list:
    for r in repr_list:
        s = summary.get((n, r), {})
        col_pct  = (s.get("collision_trial_rate") or 0.0) * 100
        min_dist = s.get("mean_min_dist_m")
        crash_pct = s.get("crash_rate", 0.0) * 100
        dist_str = f"{min_dist:.3f}m" if min_dist is not None else "N/A"
        print(f"  N={n} {r:<22} prox={col_pct:5.0f}%  crash={crash_pct:.0f}%  mean_min_dist={dist_str}")

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

# ── E-06 vs E-07 comparison for `relative` ───────────────────────────────
e06_path = SCRIPT_DIR / "results_e06" / "results.json"
if e06_path.exists():
    print("\n── `relative` repr: E-06 vs E-07 comparison ──")
    e06_raw = json.loads(e06_path.read_text())
    for n in n_list:
        e06_vals = [x["reward"] for x in e06_raw
                    if x.get("n_drones") == n and x.get("representation") == "relative"
                    and x.get("reward") is not None]
        e06_valid = sum(1 for x in e06_raw
                        if x.get("n_drones") == n and x.get("representation") == "relative"
                        and x.get("valid_code"))
        e07_vals = [x["reward"] for x in raw
                    if x.get("n_drones") == n and x.get("representation") == "relative"
                    and x.get("reward") is not None]
        e07_valid = sum(1 for x in raw
                        if x.get("n_drones") == n and x.get("representation") == "relative"
                        and x.get("valid_code"))
        e06_total = sum(1 for x in e06_raw
                        if x.get("n_drones") == n and x.get("representation") == "relative")
        e07_total = sum(1 for x in raw
                        if x.get("n_drones") == n and x.get("representation") == "relative")
        e06_mean = np.mean(e06_vals) if e06_vals else None
        e07_mean = np.mean(e07_vals) if e07_vals else None
        e06_reward_str = f"{e06_mean:.4f}" if e06_mean is not None else "N/A"
        e07_reward_str = f"{e07_mean:.4f}" if e07_mean is not None else "N/A"
        print(f"  N={n}  E-06: valid={e06_valid}/{e06_total}  reward={e06_reward_str}")
        print(f"         E-07: valid={e07_valid}/{e07_total}  reward={e07_reward_str}")

print(f"\nE-07 complete. Results in {OUTPUT_DIR}/")
