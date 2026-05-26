# analysis.py
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
from collections import defaultdict

# Physical collision threshold: CF2X tip-to-tip diameter ≈ 0.126m.
# Using this constant means even results from old experiments (where
# experiment.py used 0.25m) are re-evaluated with the correct value.
PHYS_COLLISION_M = 0.13  # metres


def load_results(path: str = "results/results.json") -> list:
    with open(path) as f:
        return json.load(f)


def summarize(results: list) -> dict:
    """Compute mean reward and validity rate per (n_drones, representation)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["n_drones"], r["representation"])
        groups[key].append(r)

    summary = {}
    for (n, repr_name), trials in groups.items():
        rewards = [t["reward"] for t in trials if t.get("reward") is not None]
        best_rewards = [t["best_reward"] for t in trials
                        if t.get("best_reward") is not None]
        valid_count = sum(1 for t in trials if t.get("valid_code"))
        crashed = sum(1 for t in trials if t.get("terminated_early"))
        # Collision metrics (only available in waypoint trials with the new schema)
        min_dists = [t["min_pairwise_dist_m"] for t in trials
                     if t.get("min_pairwise_dist_m") is not None]
        col_fracs = [t["collision_fraction"] for t in trials
                     if t.get("collision_fraction") is not None]
        # Use stored min_pairwise_dist_m against the physical threshold so that
        # old experiments (recorded with a looser 0.25m threshold) are also
        # evaluated correctly: a trial is a collision if drones actually touched.
        any_col   = [t["min_pairwise_dist_m"] < PHYS_COLLISION_M for t in trials
                     if t.get("min_pairwise_dist_m") is not None]
        summary[(n, repr_name)] = {
            "mean_reward":           np.mean(rewards) if rewards else None,
            "std_reward":            np.std(rewards) if rewards else None,
            "mean_best_reward":      np.mean(best_rewards) if best_rewards else None,
            "validity_rate":         valid_count / len(trials),
            "crash_rate":            crashed / len(trials),
            "n_trials":              len(trials),
            "n_valid":               len(rewards),
            "mean_latency_s":        np.mean([t["latency_s"] for t in trials
                                              if t.get("latency_s") is not None]),
            # ── Collision metrics ──────────────────────────────────────
            "mean_min_dist_m":        np.mean(min_dists) if min_dists else None,
            "min_min_dist_m":         min(min_dists) if min_dists else None,
            "mean_collision_fraction": np.mean(col_fracs) if col_fracs else None,
            "collision_trial_rate":    np.mean(any_col) if any_col else None,
        }
    return summary


def plot_reward_vs_n(summary: dict, representations: list, save_path: str = None):
    """Plot mean final reward vs N for each representation."""
    n_values = sorted(set(n for (n, _) in summary.keys()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for repr_name in representations:
        means, stds = [], []
        for n in n_values:
            key = (n, repr_name)
            if key in summary and summary[key]["mean_reward"] is not None:
                means.append(summary[key]["mean_reward"])
                stds.append(summary[key]["std_reward"])
            else:
                means.append(None)
                stds.append(None)

        valid_n     = [n for n, m in zip(n_values, means) if m is not None]
        valid_means = [m for m in means if m is not None]
        valid_stds  = [s for s in stds if s is not None]

        if valid_means:
            ax.plot(valid_n, valid_means, marker="o", label=repr_name)
            ax.fill_between(
                valid_n,
                [m - s for m, s in zip(valid_means, valid_stds)],
                [m + s for m, s in zip(valid_means, valid_stds)],
                alpha=0.15,
            )

    ax.set_xlabel("Number of drones (N)")
    ax.set_ylabel("Mean execution reward (higher is better, 0=perfect)")
    ax.set_title("LLM Waypoint Execution Quality vs Swarm Size\nby State Representation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(ax.figure, save_path)


def plot_reward_vs_repr(summary: dict, representations: list,
                        n_list: list = None, save_path: str = None):
    """Grouped bar chart: mean reward per representation, grouped by N."""
    if n_list is None:
        n_list = sorted(set(n for (n, _) in summary.keys()))

    x = np.arange(len(representations))
    width = 0.8 / len(n_list)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(n_list)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, n in enumerate(n_list):
        means = []
        errs  = []
        for repr_name in representations:
            key = (n, repr_name)
            if key in summary and summary[key]["mean_reward"] is not None:
                means.append(summary[key]["mean_reward"])
                errs.append(summary[key]["std_reward"])
            else:
                means.append(0.0)
                errs.append(0.0)
        offset = (i - len(n_list) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=f"N={n}",
               color=colors[i], edgecolor="black", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=errs, fmt="none",
                    color="black", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(representations, rotation=15, ha="right")
    ax.set_ylabel("Mean execution reward (higher is better, 0=perfect)")
    ax.set_title("Execution Reward by Representation and Swarm Size")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="green", linestyle="--", linewidth=0.8, label="Perfect")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_validity_rate(summary: dict, representations: list, save_path: str = None):
    """Bar chart of validity rate per representation (averaged across N)."""
    repr_valid = defaultdict(list)
    for (n, repr_name), stats in summary.items():
        repr_valid[repr_name].append(stats["validity_rate"])

    avg_valid = [np.mean(repr_valid[r]) if repr_valid[r] else 0.0
                 for r in representations]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(representations, [v * 100 for v in avg_valid],
                  color="steelblue", edgecolor="black")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Code Validity Rate (%)")
    ax.set_title("LLM Code Validity Rate by State Representation")
    ax.axhline(80, color="red", linestyle="--", label="80% threshold")
    ax.legend()
    for bar, val in zip(bars, avg_valid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val*100:.0f}%", ha="center", va="bottom")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_crash_rate(summary: dict, representations: list,
                    n_list: list = None, save_path: str = None):
    """Bar chart of crash rate per (N, representation)."""
    if n_list is None:
        n_list = sorted(set(n for (n, _) in summary.keys()))

    x = np.arange(len(representations))
    width = 0.8 / len(n_list)
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(n_list)))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, n in enumerate(n_list):
        rates = [summary.get((n, r), {}).get("crash_rate", 0.0) * 100
                 for r in representations]
        offset = (i - len(n_list) / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=f"N={n}",
               color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(representations, rotation=15, ha="right")
    ax.set_ylabel("Drone crash rate (%)")
    ax.set_title("Drone Crash Rate by Representation and Swarm Size")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_collision_rate(summary: dict, representations: list,
                        n_list: list = None, save_path: str = None):
    """Bar chart of inter-drone collision rate (% of trials with any collision)
    and a line for mean minimum pairwise distance."""
    if n_list is None:
        n_list = sorted(set(n for (n, _) in summary.keys()))

    x = np.arange(len(representations))
    width = 0.8 / len(n_list)
    colors = plt.cm.Oranges(np.linspace(0.4, 0.85, len(n_list)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for i, n in enumerate(n_list):
        col_rates = [summary.get((n, r), {}).get("collision_trial_rate", 0.0) or 0.0
                     for r in representations]
        col_rates_pct = [v * 100 for v in col_rates]
        offset = (i - len(n_list) / 2 + 0.5) * width
        ax1.bar(x + offset, col_rates_pct, width, label=f"N={n} (collision %)",
                color=colors[i], edgecolor="black", linewidth=0.5, alpha=0.85)

        min_dists = [summary.get((n, r), {}).get("mean_min_dist_m") for r in representations]
        valid_x   = [xi for xi, d in zip(x, min_dists) if d is not None]
        valid_d   = [d for d in min_dists if d is not None]
        if valid_d:
            ax2.plot(np.array(valid_x) + offset, valid_d, marker="x", linestyle="--",
                     color=colors[i], label=f"N={n} (mean min dist)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(representations, rotation=15, ha="right")
    ax1.set_ylabel("Trials with collision (%)")
    ax1.set_ylim(0, 115)
    ax2.set_ylabel("Mean minimum pairwise distance (m)")
    ax2.axhline(PHYS_COLLISION_M, color="red", linestyle=":", linewidth=1,
                label=f"Physical collision threshold ({PHYS_COLLISION_M} m)")
    ax1.set_title("Inter-Drone Collision Rate and Minimum Pairwise Distance\nby Representation and Swarm Size")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Plot saved: {save_path}")
    else:
        fig.show()
    plt.close(fig)


def print_summary_table(summary: dict):
    """Print a formatted summary table."""
    has_collision = any(s.get("collision_trial_rate") is not None for s in summary.values())
    if has_collision:
        print("\n" + "=" * 110)
        print(f"{'N':>4} | {'Representation':<22} | {'Mean Reward':>12} | {'Std':>8} | "
              f"{'Best':>8} | {'Valid%':>7} | {'Crash%':>7} | {'Prox%':>7} | {'MinDist':>8} | {'Trials':>6}")
        print("=" * 110)
        for (n, repr_name), s in sorted(summary.items()):
            mean_r  = f"{s['mean_reward']:+.4f}"       if s["mean_reward"]          is not None else "     N/A"
            std_r   = f"{s['std_reward']:.4f}"         if s["std_reward"]           is not None else "    N/A"
            best_r  = f"{s['mean_best_reward']:+.4f}"  if s["mean_best_reward"]     is not None else "     N/A"
            col_r   = f"{s['collision_trial_rate']*100:.0f}%" if s.get("collision_trial_rate") is not None else "   N/A"
            min_d   = f"{s['mean_min_dist_m']:.3f}m"  if s.get("mean_min_dist_m")  is not None else "   N/A"
            print(
                f"{n:>4} | {repr_name:<22} | {mean_r:>12} | {std_r:>8} | "
                f"{best_r:>8} | {s['validity_rate']*100:>6.0f}% | "
                f"{s['crash_rate']*100:>6.0f}% | {col_r:>7} | {min_d:>8} | {s['n_trials']:>6}"
            )
        print("=" * 110)
        print("  Crash% = trials where a drone hit the ground (terminated_early).")
        print("  Prox%  = trials where any pair of drones came within 13 cm of each other.")
        print("           For the rendezvous task, Prox%=100% is EXPECTED (all drones converge to same point).")
        print("           Use MinDist and Crash% to assess actual failures.")
    else:
        print("\n" + "=" * 90)
        print(f"{'N':>4} | {'Representation':<22} | {'Mean Reward':>12} | {'Std':>8} | "
              f"{'Best':>8} | {'Valid%':>7} | {'Crash%':>7} | {'Trials':>6}")
        print("=" * 90)
        for (n, repr_name), s in sorted(summary.items()):
            mean_r  = f"{s['mean_reward']:+.4f}"      if s["mean_reward"]      is not None else "    N/A"
            std_r   = f"{s['std_reward']:.4f}"        if s["std_reward"]       is not None else "     N/A"
            best_r  = f"{s['mean_best_reward']:+.4f}" if s["mean_best_reward"] is not None else "    N/A"
            print(
                f"{n:>4} | {repr_name:<22} | {mean_r:>12} | {std_r:>8} | "
                f"{best_r:>8} | {s['validity_rate']*100:>6.0f}% | "
                f"{s['crash_rate']*100:>6.0f}% | {s['n_trials']:>6}"
            )
        print("=" * 90)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results_pilot/results.json"
    print(f"Loading results from {path}...")
    results = load_results(path)
    print(f"Loaded {len(results)} trials.")

    summary = summarize(results)
    print_summary_table(summary)

    output_base = str(path).rsplit("/", 1)[0]
    repr_list = ["raw", "relative", "graph", "aggregate", "natural_language"]
    repr_list = [r for r in repr_list if any(r == rr["representation"] for rr in results)]
    n_list    = sorted({r["n_drones"] for r in results})

    plot_reward_vs_n(summary, repr_list,
                     save_path=f"{output_base}/reward_vs_n.png")
    plot_validity_rate(summary, repr_list,
                       save_path=f"{output_base}/validity_rate.png")
    plot_reward_vs_repr(summary, repr_list, n_list=n_list,
                        save_path=f"{output_base}/reward_vs_repr.png")
    plot_crash_rate(summary, repr_list, n_list=n_list,
                    save_path=f"{output_base}/crash_rate.png")
    plot_collision_rate(summary, repr_list, n_list=n_list,
                        save_path=f"{output_base}/collision_rate.png")
    print("\nAnalysis complete.")
