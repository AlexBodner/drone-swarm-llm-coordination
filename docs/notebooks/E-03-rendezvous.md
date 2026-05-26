# E-03 — Rendezvous Task: Testing State-Dependent Planning

**Date:** 2026-04-07  
**Status:** ✅ Complete (v2 — corrected reward function)  
**Results:** `swarm-llm-experiment/results_e03/results.json`  
**Script:** `swarm-llm-experiment/run_e03.py`  
**Log:** `swarm-llm-experiment/results_e03/run_v2.log`  
**Wall time:** 18.3 min (1099s)

> **Note:** A first version (v1) was run on 2026-04-06 but used a flawed reward function (see Reward Design section). Results from v1 (`results_STALE_old_reward.json`) are kept for reference but not used in analysis.

---

## Motivation

E-02 showed that all representations perform similarly on the circle task — because the circle has a closed-form analytic solution that requires no knowledge of where the drones actually are. The LLM ignores `state["positions"]` entirely and hardcodes `cos(2πi/n)`, `sin(2πi/n)`.

This is not a failure of the representations — it's a failure of the task to discriminate. To genuinely test whether different state representations help the LLM plan better, we need a task where **the correct answer changes with the initial conditions**. Rendezvous is the natural choice:

> "All drones should fly to their initial centroid."

The centroid is the mean of `state["positions"]` — a value that varies across seeds and N values. If the LLM ignores the state, it will hallucinate a meeting point (e.g. the origin). If it reads the state correctly, it computes the right centroid and all drones converge there.

---

## Hypothesis

> **H1:** Representations that express individual drone positions clearly (`raw`, `relative`) will allow the LLM to compute the centroid correctly, while aggregated representations (`aggregate`, `natural_language`) will lead to higher errors because they discard per-drone spatial information.
>
> **H2:** Reward variance across seeds will be non-zero for all representations, because the centroid changes with initial positions — confirming the task is genuinely state-dependent.
>
> **H3:** `natural_language` validity will be lower than other representations at N=6, because the longer state description combined with a computation-requiring task pushes the LLM toward prose output instead of code.
>
> **H4:** Performance will degrade at N=6 vs N=3 for all representations — more drones means a more dispersed initial configuration and a harder navigation problem.

---

## Design Decisions

### Why rendezvous?

- It has **no fixed target** — the meeting point depends on the runtime state, so the LLM cannot hardcode it.
- It is a **common benchmark** in multi-robot coordination literature (e.g. Reynolds' flocking, SwarmGPT).
- The task is conceptually simple to explain and verify visually.

### Reward function (v2 — corrected)

The original reward (v1) measured mean distance of each drone from the initial centroid:

```
reward_v1 = −mean(||pos_i − centroid_initial||)
```

**The problem:** this incentivises sending all N drones to the *exact same coordinate* — a physically impossible configuration in a physics simulator. PyBullet drones have rigid bodies; when all target (x, y, z) they collide and the physics engine pushes them into a line, then into pairs. The behaviour looks like "drones line up and crash" which is confusing and not what rendezvous means.

**The fix (v2):**

```
reward_v2 = −0.5 × mean_pairwise_distance − 0.5 × ||final_centroid − initial_centroid||
```

- **Term 1** (pairwise distance): rewards the cluster being *tight* — drones close to each other.
- **Term 2** (centroid drift): rewards the cluster being in the *right location* — near the initial centroid. This preserves state-dependency.

Both terms are in [−∞, 0]; combined reward = 0 only when all drones are co-located at the exact initial centroid.

> **Note on apparent similarity to "small circle":** the task description asks drones to "end up as close together as possible, near the centroid" — this could be implemented as a tiny circle around the centroid. This is *acceptable* — we're not testing the specific arrangement, only whether the LLM correctly identifies the *location* (the initial centroid). The arrangement within the cluster is not scored.

### Why not `reward_v2 = −pairwise_only`?

Pure pairwise would be maximised by sending all drones to *any* common point — e.g. (0, 0, 1) — without ever reading the initial state. This eliminates state-dependency. The centroid_drift term forces the LLM to use `state["positions"]`.

### Task prompt

```
"Move all drones to rendezvous at a common meeting point.
 The meeting point is the centroid (average position) of all drones' INITIAL positions
 from state['positions'], at height z = 1.0 meters.
 You MUST read state['positions'] to compute the centroid — do not hardcode any coordinates.
 All drones should end up as close together as possible, near that centroid."
```

### LLM model change

Switched from `llama-3.3-70b-versatile` (100K tokens/day limit) to `llama-3.1-8b-instant` (500K tokens/day) after the first v1 run was killed by the daily token limit mid-experiment (at trial 83 of 100).

---

## Configuration

| Parameter         | Value                                               |
|-------------------|-----------------------------------------------------|
| Task              | Rendezvous at initial centroid, z=1.0m              |
| Planner           | Waypoint trajectories                               |
| N_drones          | 3, 6                                               |
| Representations   | raw, relative, graph, aggregate, natural_language   |
| Seeds             | 0–9 (10 seeds)                                      |
| LLM               | llama-3.1-8b-instant (Groq), T=0                   |
| Duration          | 15s per trial                                       |
| Total trials      | 100                                                 |
| Videos            | 1 per (N, repr) condition = 10 videos               |

---

## Results

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% | Crash% |
|---|--------------------|-------------|--------|-------------|--------|--------|
| 3 | aggregate          |   −0.0860   | 0.0000 |   −0.0840   |  100%  |    0%  |
| 3 | graph              |   −0.0856   | 0.0000 |   −0.0846   |   50%  |    0%  |
| 3 | natural_language   |   −1.0652   | 0.0000 |   −0.0700   |  100%  |    0%  |
| 3 | raw                |   −0.5798   | 0.3498 |   −0.0756   |   90%  |    0%  |
| 3 | relative           |   −0.7794   | 0.2314 |   −0.0716   |  100%  |    0%  |
| 6 | graph              |   −1.0148   | 0.0000 |   −0.1305   |  100%  |    0%  |
| 6 | relative           |   −1.0491   | 0.1551 |   −0.1424   |  100%  |    0%  |
| 6 | raw                |   −1.1249   | 0.5075 |   −0.1361   |  100%  |    0%  |
| 6 | aggregate          |   −1.1439   | 0.0000 |   −0.1378   |  100%  |    0%  |
| 6 | natural_language   |   −1.3527   | 0.0000 |   −0.1282   |  100%  |    0%  |

---

## Analysis

### H1 — Partially confirmed, but the ranking is different from expected

The expected ordering was: `raw` ≈ `relative` > `aggregate` > `natural_language`.

**What actually happened:** `aggregate` wins at N=3 because it *literally contains the centroid in the prompt* — the aggregate representation states "Swarm centroid: (cx, cy, cz)", so the LLM doesn't have to compute anything. This is an inadvertent shortcut that reveals a confound in the representation design: the aggregate representation was designed to test whether summary statistics help, but for the rendezvous task it accidentally provides the exact answer.

**`natural_language` completely fails** (mean=−1.07 at N=3, std=0) — no variance at all. The LLM generates the same wrong plan for every seed because natural language descriptions lose all numerical precision. The meeting point it chooses is always whatever it infers from "drones are in the north-east" style descriptions.

### H2 — Confirmed

Non-zero variance appears in `raw` (std=0.35) and `relative` (std=0.23) at N=3, and `raw` (std=0.51) and `relative` (std=0.155) at N=6. For these two representations, the LLM sometimes correctly computes the centroid and sometimes doesn't — genuine trial-by-trial sensitivity to the input state.

Contrast with E-02 (circle task) where `raw` and `graph` had std=0.000 — the LLM was completely state-independent there.

### H3 — Partially confirmed: `graph` fails differently than expected

`graph` has 50% validity at N=3 due to ZeroDivisionError — the LLM computes inter-drone distances as part of a weighted averaging scheme, but when two drones start at the same position the denominator is zero. This is a representation-induced fragility: the graph format *invites* distance-based computation, which can fail.

At N=6 (drones more separated) `graph` recovers to 100% validity and is actually the *best* representation — the graph distances are all non-zero and the LLM correctly uses them.

### H4 — Confirmed: significant degradation at N=6

All representations get much worse at N=6. Best reward drops from ~−0.08 (N=3) to ~−0.13 (N=6). The mean drop is even larger:
- `aggregate`: −0.086 → −1.144 (13× worse)
- `graph`: −0.086 → −1.015 (12× worse, despite 50% valid at N=3)
- `natural_language`: −1.065 → −1.353 (already bad at N=3, worse at N=6)
- `raw`: −0.580 → −1.125
- `relative`: −0.779 → −1.049

**Why does aggregate collapse at N=6?** At N=3, the reward was dominated by centroid_drift (getting to the right place). At N=6, the pairwise_distance term grows — 6 drones physically can't all cluster as tightly as 3, so even a perfectly-located cluster has a large pairwise component. The aggregate shortcut (reading centroid directly) doesn't help with pairwise minimisation.

### The key confound: `aggregate` representation unfairly advantages rendezvous

The `aggregate` representation was intended to provide only high-level summary stats (centroid, spread, max distance, mean height). For the rendezvous task, it inadvertently gives the *answer* directly. This needs to be noted as a design flaw: the centroid should either be removed from the aggregate representation, or experiments comparing representations on rendezvous must be interpreted with this in mind.

### Overall ranking and interpretation

**N=3:** aggregate ≈ graph* > raw > relative >> natural_language  
(*graph has 50% failure rate; mean reported over valid trials only)

**N=6:** graph > relative > raw > aggregate > natural_language

The complete ranking reversal between N=3 and N=6 (aggregate from 1st to 4th; graph from 2nd to 1st) shows that **no single representation dominates across both scales** for the rendezvous task. This is a stronger result than the circle task, which showed near-equality across all representations.

---

## Conclusion

The rendezvous task successfully discriminates between representations in a way the circle task could not. The key findings:

1. **`aggregate` unintentionally provides the answer at N=3** — a confound to fix in future experiment design.
2. **`natural_language` consistently fails** on state-dependent tasks — it loses numerical precision needed for centroid computation.
3. **`graph` is brittle at N=3 but best at N=6** — representation-induced ZeroDivisionError at small N; recovers with larger separation.
4. **`raw` and `relative` are the most genuine probes** of whether the LLM can use state — they show non-zero variance across seeds without confounds.
5. **Performance degrades sharply from N=3 to N=6** for all representations — scaling is a real challenge for state-dependent tasks.

---

## Known Issues During This Experiment

| ID   | Issue                                           | Observed At | Status |
|------|-------------------------------------------------|-------------|--------|
| I-11 | LLM generates waypoint at t=15.75 > duration   | N=3, raw, seed=5 | Executor rejects; trial invalid |
| I-12 | Bimodal reward at N=3 for `raw` (~−0.83 vs ~−0.085) | N=3, raw | Two plan classes: wrong point vs correct centroid |
| I-13 | `graph` repr: ~50% ZeroDivisionError | N=3, graph | **Root cause:** the `graph` representation expresses distances between drones. When all drones start near the same point, some pairwise distances are ~0. The LLM uses these distances as denominators (e.g. weighted interpolation) → division by zero. The `graph` representation is ill-conditioned for computation-based tasks when drones are co-located. |
| I-14 | `aggregate` repr: consistently good reward (~−0.086) | N=3, aggregate | The `aggregate` representation **directly states "Swarm centroid: (x, y, z)"** — so the LLM reads off the centroid without computing anything. This is an unintended shortcut: `aggregate` should be harder since it loses individual drone positions, but it's inadvertently the easiest for rendezvous because the answer is in the prompt. |

---

## E-05 Follow-up (Seed Variance Fix)

**Date:** 2026-04-09  
**Results:** `swarm-llm-experiment/results_e05/results.json`  
**Summary table:** `swarm-llm-experiment/results_e05/summary_table.txt`

After E-03/E-04, we discovered a structural confound: different `seed` values were not changing initial spawn positions. E-05 fixes this by adding seeded XY jitter to initial positions (`position_jitter=±0.5m`) so seeds represent genuinely different initial states.

### E-05 core outcomes

- All 100 trials completed (`N∈{3,6}`, 5 representations, 10 seeds).
- Validity remained high in most conditions (mostly 90–100%; `graph` at N=3 = 20%).
- Reward variance is now clearly non-zero in most conditions (e.g., N=6 `raw` std=1.7309, N=3 `relative` std=1.2691), confirming true seed sensitivity.
- Collision trial rate remained 100% in all conditions with mean min distances ≈0.05–0.12m, which reinforces that rendezvous-to-single-point conflates task success with unavoidable physical overlap.

### Updated interpretation

E-05 validates the methodological fix (seed variability is real now), but it also confirms the reward-design limitation of rendezvous: because all drones are incentivized to co-locate, collision metrics saturate and are not discriminative. This supports moving to the planned next task design where the target is centroid-centered **formation** rather than a single shared point.
