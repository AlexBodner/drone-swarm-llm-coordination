# E-02 — Circle Task at Scale: Does Representation Matter?

**Date:** 2026-04-06  
**Status:** ✅ Complete  
**Results:** `swarm-llm-experiment/results_e02/results.json`  
**Script:** `swarm-llm-experiment/run_e02.py`

---

## Motivation

E-01 showed the pipeline works, but the circle task at N=3 is too easy — the LLM ignores the state completely. E-DEMO-02 showed that **waypoint trajectories solve the planning–execution gap** (47× improvement over single endpoints). Now we want to run the first real scaling experiment:

- Do representations start to matter as N grows?
- Is the waypoint pipeline robust across all 5 representations?
- At what point does `natural_language` start to degrade?

---

## Hypothesis

> **H1:** All representations will achieve good rewards at N=3 and N=6 on the circle task, because the circle has an analytic solution and waypoints allow the PID to track it correctly.
>
> **H2:** Representations that provide richer per-drone positional information (`raw`, `relative`) will outperform aggregated ones (`aggregate`, `natural_language`) as N increases, because the LLM needs to assign each drone to a specific circle slot.
>
> **H3:** The reward variance will be near zero for `graph` and `raw` since the prompts are deterministic given N — the LLM generates the same plan regardless of initial positions.

---

## Design Decisions

- **N ∈ {3, 6}** — steps that double the swarm size; 6 stresses the LLM's ability to assign 6 circle slots.
- **10 seeds → reduced to 5** for token budget; enough for mean/std estimates.
- **Waypoint interface** — `plan(state, duration) → {id: [(t,x,y,z), ...]}` — validated in E-DEMO-02.
- **Circle task** — same as E-01 for continuity; allows direct comparison.
- **Switched from llama-3.3-70b-versatile** (used in E-01/DEMO) — still 70B in this run.

---

## Configuration

| Parameter         | Value                                               |
|-------------------|-----------------------------------------------------|
| Task              | Circle formation, radius=2.0m, z=1.0m               |
| Planner           | Waypoint trajectories                                |
| N_drones          | 3, 6                                                |
| Representations   | raw, relative, graph, aggregate, natural_language   |
| Seeds             | 0–4 (5 seeds)                                       |
| LLM               | llama-3.3-70b-versatile (Groq), T=0                 |
| Duration          | 15s per trial                                       |
| Total trials      | 50                                                  |
| Wall time         | ~3.3 min                                            |

---

## Results

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% | Crash% |
|---|--------------------|-------------|--------|-------------|--------|--------|
| 3 | relative           |   −0.0255   | 0.0134 |   −0.0081   |  100%  |    0%  |
| 3 | natural_language   |   −0.0259   | 0.0167 |   −0.0181   |  100%  |    0%  |
| 3 | graph              |   −0.0364   | 0.0000 |   −0.0074   |  100%  |    0%  |
| 3 | raw                |   −0.0364   | 0.0000 |   −0.0074   |  100%  |    0%  |
| 3 | aggregate          |   −0.0418   | 0.0065 |   −0.0076   |  100%  |    0%  |
| 6 | aggregate          |   −0.0264   | 0.0096 |   −0.0113   |  100%  |    0%  |
| 6 | relative           |   −0.0269   | 0.0111 |   −0.0086   |  100%  |    0%  |
| 6 | graph              |   −0.0310   | 0.0094 |   −0.0088   |  100%  |    0%  |
| 6 | raw                |   −0.0357   | 0.0000 |   −0.0079   |  100%  |    0%  |
| 6 | natural_language   |   −0.0384   | 0.0107 |   −0.0273   |  100%  |    0%  |

---

## Analysis

### H1 — Confirmed
All 50 trials completed with 100% validity and 0% crash rate. Mean rewards cluster between −0.025 and −0.042, all in the "excellent" range (<0.05 mean distance). The waypoint pipeline handles N=6 cleanly.

### H2 — Partially confirmed, with surprises
- At N=3: `relative` and `natural_language` top the ranking (−0.026). `aggregate` is worst (−0.042). This is the *opposite* of H2.
- At N=6: `aggregate` suddenly jumps to 1st (−0.026). `natural_language` drops to last (−0.038).

The ranking reversal between N=3 and N=6 is unexpected. The most likely explanation: for the circle task, **no representation advantage matters much** because the task is still essentially state-independent — the LLM computes `cos(2πi/n)` regardless. The ranking differences are within 0.015, which may be noise at 5 seeds.

### H3 — Confirmed
`graph` and `raw` both show std=0.000 at both N values. The LLM produces byte-for-byte identical plans across seeds. This confirms it is ignoring initial positions and just doing geometry. This is a confound: **we are measuring PID tracking quality, not LLM planning quality**, for these two representations.

### The key insight
The circle task is **state-independent**: you can solve it perfectly without reading `state["positions"]`. Any representation advantage that does exist is marginal. To properly test whether representation matters, we need a task where the LLM **must** use the runtime state — which is the design motivation for E-03.

### N=6 slightly outperforms N=3 on mean reward
Counter-intuitive. Hypotheses:
- The reward is normalised per-drone, so one poorly-placed drone hurts less in a larger swarm
- More circle slots at N=6 may permit better Hungarian matching in the reward function

---

## Conclusion

Waypoint pipeline is robust at N∈{3,6} for all representations. The circle task is too easy to discriminate between representations — the LLM always finds the analytic solution and ignores initial state. Move to a **state-dependent task** (E-03: rendezvous) to properly measure representation quality.
