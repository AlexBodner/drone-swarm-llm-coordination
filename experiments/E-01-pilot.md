# E-01 — Pilot: Can the pipeline run end-to-end?

**Date:** 2026-04-06  
**Status:** ✅ Complete  
**Results:** `swarm-llm-experiment/results_pilot/results.json`  
**Script:** `swarm-llm-experiment/run_e01_pilot.py`

---

## Motivation

Before running large-scale experiments, we need to validate that the full pipeline works:
state extraction → representation → prompt → LLM → code execution → physics simulation → reward.

The circle task at N=3 is deliberately chosen to be trivially solvable — if anything fails, it's a bug in the pipeline, not a hard task. All representations should achieve near-perfect reward since the LLM can hardcode the analytic solution (`cos(2πi/n)`, `sin(2πi/n)`).

---

## Hypothesis

> All 5 representations will achieve ~0 reward at N=3 on the circle task, because the task has an analytic closed-form solution that requires no state information.

If any representation scores significantly worse, it indicates a pipeline bug in that representation's code path.

---

## Design Decisions

- **N=3** — smallest meaningful swarm; 3 drones on a circle have a unique, unambiguous solution.
- **5 seeds** — enough to catch randomness in prompt formatting, not enough to draw statistical conclusions at this stage.
- **Circle task** — analytic solution exists; success means the pipeline correctly parses and executes LLM code.
- **Single-step endpoint planning** (not waypoints) — simplest possible interface to validate first.

---

## Configuration

| Parameter         | Value                                               |
|-------------------|-----------------------------------------------------|
| Task              | Circle formation, radius=2.0m, z=1.0m               |
| Planner           | Single endpoint (`plan(state) → {id: (x,y,z)}`)    |
| N_drones          | 3                                                   |
| Representations   | raw, relative, graph, aggregate, natural_language   |
| Seeds             | 0–4 (5 seeds)                                       |
| LLM               | llama-3.3-70b-versatile (Groq), T=0                 |
| Total trials      | 25                                                  |

---

## Results

| Representation   | Mean Reward | Std   | Valid% | Mean Latency |
|-----------------|-------------|-------|--------|--------------|
| raw             | 0.0000      | 0.000 | 100%   | 0.88s        |
| relative        | 0.0000      | 0.000 | 100%   | 0.72s        |
| graph           | 0.0000      | 0.000 | 100%   | 0.83s        |
| aggregate       | 0.0000      | 0.000 | 100%   | 0.89s        |
| natural_language| 0.0000      | 0.000 | 100%   | 2.48s        |

---

## Analysis

**Hypothesis confirmed.** All representations reach reward=0.0 (perfect formation) on every trial. The pipeline works end-to-end with 100% code validity.

**What also became clear:**

1. **The LLM ignores `state["positions"]` entirely.** In all 25 trials, the generated `plan()` only uses `state["n_drones"]`. The task is solvable without reading actual drone positions, confirming N=3 circle is a bad discriminator.

2. **`natural_language` causes verbose responses and latency spikes** (max 8.9s one trial) — the LLM adds comments and reasoning around the code. At rate limits this will accumulate.

3. **`relative` representation caused the LLM to add hardcoded guards** like `if n_drones != 3: raise ValueError(...)`. This will break at N>3. Noted as I-01.

4. **The single-endpoint planner has a fundamental execution gap.** When we tested at larger N (see E-DEMO-01), the PID controller couldn't physically reach targets 2m away in free-fall style — it overshoots and loses altitude. The validated pipeline was then upgraded to waypoint trajectories (E-DEMO-02, E-02).

---

## Bugs Found & Fixed During This Experiment

| Bug | Description | Fix |
|-----|-------------|-----|
| `exec` scoping | `exec(code, {}, ns)` creates empty globals; imports inside the function closure are invisible when called | Changed to `exec(code, ns)` |
| `env.reset()` return values | `CtrlAviary.reset()` returns `(obs, info)` not 5 values | `obs, _ = env.reset(seed=...)` |

---

## Conclusion

Pipeline validated. Move to waypoint planner (E-DEMO-02) and then scale experiments (E-02, E-03).
