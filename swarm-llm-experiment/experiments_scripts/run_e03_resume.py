"""
run_e03_resume.py
=================
Resume E-03 by re-running only the failed trials (429 rate-limit errors).
Reads existing results_e03/results.json, identifies failures, re-runs them,
and replaces the error records with valid results.

Run after the Groq daily token limit resets (midnight UTC):
    GROQ_API_KEY=<key> /opt/anaconda3/envs/swarm-llm/bin/python run_e03_resume.py
"""

import os
import sys
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

os.environ.setdefault(
    "GROQ_API_KEY",
    "YOUR_API_KEY_HERE",
)

from experiment import run_trial_waypoint
from prompt_builder import TASK_RENDEZVOUS
from llm_connector import get_available_connector
from analysis import load_results, summarize, print_summary_table, \
                     plot_reward_vs_n, plot_validity_rate, plot_reward_vs_repr

OUTPUT_DIR = SCRIPT_DIR.parent / "results_archive" / "results_e03"
RESULTS_FILE = OUTPUT_DIR / "results.json"
REPRESENTATIONS = ["raw", "relative", "graph", "aggregate", "natural_language"]
DURATION = 15.0

# ── Load existing results ──────────────────────────────────────────────────
print(f"Loading {RESULTS_FILE}...")
with open(RESULTS_FILE) as f:
    all_results = json.load(f)

failed = [(i, r) for i, r in enumerate(all_results) if not r.get("valid_code", False)]
print(f"Found {len(failed)} failed trials to re-run out of {len(all_results)} total.")
if not failed:
    print("Nothing to re-run!")
    sys.exit(0)

for idx, r in failed:
    print(f"  [{idx}] N={r['n_drones']}, repr={r['representation']}, seed={r['seed']}  error={r.get('error','?')[:60]}")

print()

# ── Re-run failed trials ───────────────────────────────────────────────────
llm_name, llm_fn = get_available_connector()
print(f"LLM: {llm_name}  |  {len(failed)} trials to retry")
print("=" * 65)

t_start = time.time()
n_ok = 0

# Create videos dir for resume recordings
VIDEOS_DIR = RESULTS_FILE.parent / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# Track which conditions already have a video
vid_recorded: set = set()

for idx, r in failed:
    n = r["n_drones"]
    repr_name = r["representation"]
    seed = r["seed"]
    print(f"  Retrying N={n}, repr={repr_name}, seed={seed}...", end=" ", flush=True)

    cond_key = (n, repr_name)
    should_record = cond_key not in vid_recorded
    vpath = str(VIDEOS_DIR / f"n{n}_{repr_name}_seed{seed}.mp4") if should_record else None

    try:
        result = run_trial_waypoint(
            n_drones=n,
            representation_name=repr_name,
            seed=seed,
            task_description=TASK_RENDEZVOUS,
            duration=DURATION,
            llm_fn=llm_fn,
            llm_name=llm_name,
            reward_mode="rendezvous",
            record_video=should_record,
            video_path=vpath,
        )
        if should_record and result.get("video"):
            vid_recorded.add(cond_key)
        status = (f"reward={result['reward']:.4f}"
                  if result["reward"] is not None
                  else f"INVALID ({result.get('error', '?')})")
        print(status)
        if result.get("valid_code"):
            n_ok += 1
    except Exception as e:
        result = {**r, "error": str(e), "valid_code": False, "reward": None}
        print(f"ERROR: {e}")

    # Replace error record in-place
    all_results[idx] = result

    # Save after every trial
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    time.sleep(0.5)

elapsed = time.time() - t_start
print(f"\nRetried {len(failed)} trials in {elapsed/60:.1f} min. "
      f"{n_ok}/{len(failed)} now valid.")

# ── Re-run analysis ────────────────────────────────────────────────────────
print("\nRunning analysis …")
results = load_results(str(RESULTS_FILE))
summary = summarize(results)
print_summary_table(summary)

repr_list = REPRESENTATIONS
n_list = [3, 6]
plot_reward_vs_n(summary, repr_list, save_path=str(OUTPUT_DIR / "reward_vs_n.png"))
plot_validity_rate(summary, repr_list, save_path=str(OUTPUT_DIR / "validity_rate.png"))
plot_reward_vs_repr(summary, repr_list, n_list=n_list,
                    save_path=str(OUTPUT_DIR / "reward_vs_repr.png"))
print(f"\nE-03 resume complete. Full results in {OUTPUT_DIR}/")
