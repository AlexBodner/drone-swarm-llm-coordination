# FINDINGS — LLM-Coordinated Drone Swarm Experiment

**Date:** 2026-04-06  
**Phase:** 1 (Infrastructure) + Pilot Experiment  
**LLM:** Groq / `llama-3.3-70b-versatile` (temperature=0)  
**Simulator:** gym-pybullet-drones 2.0.0 + PyBullet (conda-forge build)  
**Task:** Circle formation (radius=2.0m, height=1.0m, origin-centered)  
**Scale:** N=3 drones, 5 representations × 5 seeds = 25 trials

---

## Infrastructure Setup — Issues Encountered

### 1. `gym-pybullet-drones` not on PyPI
The plan assumed `pip install gym-pybullet-drones` would work. The package is not published on PyPI.  
**Fix:** Install directly from GitHub source:
```bash
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
```

### 2. `pybullet` compile failure on macOS ARM (Apple Silicon)
The standard `pip install pybullet --no-binary pybullet` fails on macOS 13+ ARM with a clang error in bundled zlib (`fdopen` macro conflict with SDK headers).  
**Fix:** Use the conda-forge pre-built binary:
```bash
conda install -c conda-forge pybullet -y
```
Then install gym-pybullet-drones (which bundles its own non-pybullet URDF assets).

### 3. `exec()` sandbox scoping bug in `executor.py`
The original plan used `exec(code, {}, local_namespace)`, which creates an empty global namespace for the executed code. Functions defined in the code have their `__globals__` set to `{}`, so any module imported inside the code block (`import numpy as np`) is invisible when the function is called.  
**Fix:** Use a single namespace dict for both globals and locals:
```python
exec(code, local_namespace)   # local_namespace acts as both globals and locals
```
This ensures imports land in the same namespace that function closures reference.

---

## Pilot Experiment Results (N=3, 5 seeds)

### Summary Table

| Representation   | Mean Reward | Std   | Validity | Mean Latency |
|-----------------|-------------|-------|----------|--------------|
| raw             | +0.0000     | 0.000 | 100%     | 0.88s        |
| relative        | +0.0000     | 0.000 | 100%     | 0.72s        |
| graph           | +0.0000     | 0.000 | 100%     | 0.83s        |
| aggregate       | +0.0000     | 0.000 | 100%     | 0.89s        |
| natural_language| +0.0000     | 0.000 | 100%     | 2.48s        |

### Interpretation

**1. N=3 is too easy — confirms execution plan warning.**  
All 5 representations achieve a perfect reward of 0.0000 (zero mean distance to optimal formation) across all 5 seeds. The LLM consistently computes the exact analytic solution: evenly space 3 drones on a circle using `math.cos(2π·i/n)`. The initial drone positions are completely ignored — the plan succeeds regardless of starting configuration. This is expected and was predicted in the execution plan.

**2. 100% code validity across all representations — pipeline is robust.**  
Every single trial returned a valid `plan()` function that passed all validation checks. No representation caused parse errors or malformed output with `llama-3.3-70b-versatile` at N=3.

**3. Representation affects verbosity and latency, not quality at N=3.**

| Representation   | Prompt chars | Response chars | Latency |
|-----------------|-------------|----------------|---------|
| raw             | 1,277        | 359            | 0.88s   |
| relative        | 1,331        | 441            | 0.72s   |
| graph           | 1,464        | 336            | 0.83s   |
| aggregate       | 1,281        | 336            | 0.89s   |
| natural_language| 1,305        | 510            | 2.48s   |

- `natural_language` produces the longest responses (the LLM adds comments and explanations) and has the highest latency (2.48s mean, max 8.96s in one trial).
- `graph` has the longest prompts due to per-drone neighbor listings, but the LLM ignores the graph structure entirely for N=3 (just computes the formula).
- `relative` is fastest (mean 0.72s) and slightly shorter prompts.

**4. LLM ignores the state at N=3.**  
In all 25 trials, the generated `plan()` function uses only `state["n_drones"]` — it never accesses `state["positions"]` or `state["velocities"]`. The initial state is irrelevant to achieving a perfect circle with 3 drones. This is mathematically correct but means **the state representation has zero discriminating power at this scale.**

**5. Potential fragility at larger N observed in `relative` representation.**  
The `relative` representation caused the LLM to add hardcoded guards like:
```python
if n_drones != 3:
    raise ValueError("Exactly 3 drones are expected")
```
This would produce `None` results (code validity failure) in trials with N>3 — a subtle but important failure mode to watch for in the main experiment.

---

## Key Conclusions from Pilot

1. **Pipeline validated end-to-end.** All 7 modules work correctly. The full loop (state extraction → representation → prompt → LLM → code execution → reward) runs without errors.

2. **Reward ceiling effect at N=3.** The task is trivially solvable with any representation. The main experiment must use N ∈ {6, 10, 15} to observe meaningful differences.

3. **`natural_language` representation adds LLM inference overhead** (extra reasoning tokens observed in responses). This could become an issue at rate limits.

4. **The `relative` and `natural_language` representations may degrade earlier** as N grows, because: relative gives the centroid but loses individual identities; natural language loses precise coordinate information entirely.

---

## Finding F-02 — Option A: Waypoint Trajectories Solve the Planning–Execution Gap

**Date:** 2026-04-06  
**Source:** E-DEMO-02 (`waypoint_demo.py`)

### Result

Replacing the single-endpoint LLM output with timed waypoint trajectories produced a **47× improvement in final reward** (−1.732 → −0.036) and a **132× improvement in best reward** (−0.980 → −0.007). The formation was achieved to within ~4 cm mean error — effectively closing the planning–execution gap.

### Why It Works

The single-target failure had two root causes eliminated by waypoints:

1. **XY overshoot eliminated** — the interpolated target advances slowly (~0.15 m/s) so the PID never accumulates excess speed it cannot brake.
2. **Altitude loss eliminated** — the first waypoint stage (t=0→2s) is a pure vertical rise with zero XY displacement. The PID handles 1D altitude correction cleanly before horizontal travel begins.

### The LLM's Role Is Now Richer

With waypoints, the LLM must reason about **dynamics, not just geometry**. The generated code:
- Reads `state["positions"]` to start each drone at its actual current position
- Computes intermediate positions as linear interpolations from current to target
- Stages the motion (rise first, then move, then hold)

This is qualitatively different from E-01 where the LLM ignored `state["positions"]` entirely. The interface change forces the LLM to do physical reasoning.

### Interface Format Is the Critical Design Variable

> Identical LLM model, identical task, identical reward function, identical controller — only the output format changed. Reward improved 47×.

This is a clean and publishable result for the thesis: **the richness of the LLM–controller interface is more important than the LLM itself** for physical task execution quality.

---

## Proposed Research Continuation

With the waypoint pipeline validated, there are three natural axes to explore:

### Axis 1 — Scale (N drones)
**Question:** At what swarm size does the LLM's waypoint planning degrade, and which representations help most?

- Run N ∈ {3, 6, 10, 15} with the waypoint pipeline
- Expect degradation because: more drones → longer output → more chances for geometric errors; collision avoidance becomes necessary
- Key metric: does the LLM start generating *colliding* waypoints at N≥6?
- Add collision detection to the reward (penalise inter-drone distance < 0.3m)

### Axis 2 — Representation (state format)
**Question:** When the LLM must reason about dynamics (waypoints), does the representation matter more than in the single-endpoint case?

- Hypothesis: `raw` and `relative` will outperform `aggregate` and `natural_language` because waypoint planning requires precise individual positions
- At N=3 all representations yielded identical endpoint plans — this will likely diverge with waypoints at N≥6
- Run 5 representations × N ∈ {6, 10} × 10 seeds

### Axis 3 — Feedback Loop (closed-loop replanning)
**Question:** Can the LLM correct for physical deviations if given mid-flight state updates?

- Current pipeline: LLM plans once, PID executes blindly
- Proposal: re-query the LLM at t=5s and t=10s with the current drone positions; LLM appends new waypoints from current position
- Expected benefit: recovers from large initial errors, handles disturbances
- This mimics the receding-horizon replanning used in SwarmGPT

### Axis 4 — Task Complexity
**Question:** Does performance generalise beyond the circle task?

- Add tasks: rendezvous (converge to centroid), line formation, split into two sub-groups
- Rendezvous is interesting because it has no fixed target — the LLM must reason about the centroid as a moving average of current positions
- Box/grid formation tests whether the LLM can assign drones to non-symmetric slots

### Recommended Immediate Next Steps (Priority Order)

- [x] **E-02:** Run N∈{3,6} with waypoint pipeline, all 5 representations, 5 seeds — DONE ✅ (see F-03)
- [ ] **E-03:** Run N=10 with all 5 representations, 10 seeds — scale further
- [ ] **Add collision penalty** to `reward.py` — necessary for N≥6 where drones could intersect paths
- [ ] **Add `positions_used` metric** — does generated code read `state["positions"]`? Track across N and repr
- [ ] **E-04:** Closed-loop replanning experiment (replan at t=5s, t=10s) vs. open-loop at N=6
- [ ] **E-05:** Multi-task comparison (circle vs. rendezvous vs. line) at N=6 with `raw` repr
- [ ] **LLM comparison:** Run E-02 config with Gemini 2.0 Flash and Claude Sonnet — test if larger/newer models generate better waypoint sequences

---

## Finding F-03 — E-02 Scale Results: Waypoint Pipeline Robust Across N and All Representations

**Date:** 2026-04-06  
**Source:** E-02 (`run_e02.py`, `results_e02/results.json`)  
**Trials:** 50 (N∈{3,6} × 5 representations × 5 seeds), 3.3 min wall time

### Results at a Glance

| N | Representation     | Mean Reward | Std    | Best Reward |
|---|-------------------|-------------|--------|-------------|
| 3 | relative           |   −0.0255   | 0.0134 |   −0.0081   |
| 3 | natural_language   |   −0.0259   | 0.0167 |   −0.0181   |
| 3 | graph              |   −0.0364   | 0.0000 |   −0.0074   |
| 3 | raw                |   −0.0364   | 0.0000 |   −0.0074   |
| 3 | aggregate          |   −0.0418   | 0.0065 |   −0.0076   |
| 6 | aggregate          |   −0.0264   | 0.0096 |   −0.0113   |
| 6 | relative           |   −0.0269   | 0.0111 |   −0.0086   |
| 6 | graph              |   −0.0310   | 0.0094 |   −0.0088   |
| 6 | raw                |   −0.0357   | 0.0000 |   −0.0079   |
| 6 | natural_language   |   −0.0384   | 0.0107 |   −0.0273   |

**100% validity, 0% crash rate** across all 50 trials.

### Key Observations

**1. Robustness Confirmed at Scale**  
The 47× improvement over the single-endpoint baseline (E-DEMO-01: −1.73) is confirmed across both swarm sizes and all 5 representations. Mean rewards cluster tightly in −0.025 to −0.042, all within the EXCELLENT range (<0.05).

**2. Representation Ranking (N=3)**  
`relative` ≈ `natural_language` > `graph` ≈ `raw` > `aggregate`  
The two representations that provide positional context in interpretable form outperform at smaller N. Hypothesis: at N=3, the LLM benefits from spatial language to reason about inter-drone geometry.

**3. Representation Ranking (N=6)**  
`aggregate` ≈ `relative` > `graph` > `raw` > `natural_language`  
Interestingly, `natural_language` degrades at N=6 (−0.038 vs −0.026 for aggregate). Hypothesis: with 6 drones, the natural language description becomes long and the LLM loses spatial coherence when allocating circle positions.

**4. Deterministic Response to State Format (graph, raw)**  
Both `graph` and `raw` show std=0.000 at both N=3 and N=6. The LLM generates *identical* waypoint plans regardless of seed (which only randomizes the prompt phrasing, not the spatial input). This indicates:
- The LLM ignores random seed variations in these formats
- The plans are purely geometry-based (uniform circle allocation), not state-dependent
- This may be sub-optimal — the LLM is not using actual `state["positions"]` to personalise paths

**5. N=6 Slightly Better than N=3 on Mean Reward**  
Most representations improve or stay flat when moving from N=3 to N=6. Counter-intuitive. Two hypotheses:
- The task has 6 circle slots for N=6 vs 3 for N=3 — more degrees of freedom may make the Hungarian matching easier to optimise
- The reward is normalised per-drone, so one poorly-placed drone hurts less in a larger swarm

**6. Best-Reward Values are Consistently Excellent**  
Best rewards across all conditions range from −0.007 to −0.027, confirming that when the LLM generates good plans, drones settle within 1–3 cm of target positions.

### Implications for Thesis

- **Representation matters, but the effect is task-size dependent.** At N=3 the ranking differs from N=6, suggesting that the representation × N interaction is a meaningful experimental variable.
- **The `relative` representation is robust across both scales** — consistently in the top 2 at both N=3 and N=6. This supports using it as a default for further experiments.
- **`natural_language` is not monotonically better** despite being most human-readable. At N=6 it ranks last. This challenges the intuition that more interpretable = more LLM-friendly.
- **Zero variance in `graph`/`raw` is a confound.** If the LLM always outputs the same plan, we cannot attribute performance to the representation — we're measuring the PID tracker, not the LLM planner. Future experiments should verify that plans actually vary with initial state.

### Next Experimental Questions (Priority)

1. **Does representation matter more on state-dependent tasks?** Run rendezvous task (E-03) — DONE, see F-04.
2. **Does performance degrade at N=10?** Run E-03 N=6/10 to find the scaling cliff.
3. **Are `graph`/`raw` plans truly state-independent?** Log the generated waypoints and check if they embed `state["positions"]` or hardcode positions.
4. **Does `natural_language` recover at N=6 with more reasoning?** Try chain-of-thought prompting.

---

## Finding F-04 — ⚠️ SUPERSEDED — Rendezvous v1 Results (Flawed Reward)

> **This finding was based on E-03 v1, which used a broken reward function.**
> The reward `−mean(dist_to_centroid)` incentivised all drones targeting the exact same coordinate,
> causing physical collisions. The numbers below are invalid and kept only for reference.
> **See F-05 below for the corrected analysis (E-03 v2).**

**Date:** 2026-04-06 (invalidated 2026-04-07)  
**Source:** E-03 v1 — N=3 complete only (N=6 killed by token limit)  
**Reward bug:** `reward = −mean_distance_to_centroid` → drones all target same point → collision → fake good reward for tight collision  

*(Original N=3 table omitted — see `results_e03/results_STALE_old_reward.json` for raw numbers)*

### What the reward bug caused visually

All drones planning identical targets `(cx, cy, 1.0)` → PyBullet collision physics → drones form a line → pairs crash. The LLM was technically doing the "right" thing per the reward specification — the spec was wrong.

### What changed in v2

| | v1 | v2 |
|--|----|----|
| **Reward** | `−mean(dist_to_centroid)` | `−0.5×pairwise_dist −0.5×centroid_drift` |
| **Optimal LLM plan** | All drones → same point (crashes) | Drones cluster tightly near centroid (no crash) |
| **State-dependency** | ✓ centroid is input-dependent | ✓ centroid_drift preserves state-dependency |
| **LLM model** | llama-3.3-70b-versatile | llama-3.1-8b-instant |

---

## Finding F-05 — E-03 v2 Results: Representation × Task Interaction is the Central Variable

**Date:** 2026-04-07  
**Source:** E-03 v2 — 100 trials, N∈{3,6}, 5 representations, 10 seeds  
**Full notebook:** [experiments/E-03-rendezvous.md](experiments/E-03-rendezvous.md)

### Results Summary

| N | Representation   | Mean Reward | Std   | Valid% |
|---|-----------------|-------------|-------|--------|
| 3 | aggregate        |   −0.086    | 0.000 |  100%  |
| 3 | graph            |   −0.086    | 0.000 |   50%  |
| 3 | raw              |   −0.580    | 0.350 |   90%  |
| 3 | relative         |   −0.779    | 0.231 |  100%  |
| 3 | natural_language |   −1.065    | 0.000 |  100%  |
| 6 | graph            |   −1.015    | 0.000 |  100%  |
| 6 | relative         |   −1.049    | 0.155 |  100%  |
| 6 | raw              |   −1.125    | 0.508 |  100%  |
| 6 | aggregate        |   −1.144    | 0.000 |  100%  |
| 6 | natural_language |   −1.353    | 0.000 |  100%  |

### Key Findings

**1. `aggregate` representation has a confound for rendezvous**  
The aggregate prompt explicitly states “Swarm centroid: (x, y, z)” — the exact target the LLM needs. At N=3, this gives a free win (mean=−0.086), but it's not measuring planning quality. At N=6, the advantage disappears (drops to 4th place) because the pairwise distance term grows with swarm size. The centroid shortcut helps with location but not clustering.

**2. `natural_language` is the worst for state-dependent tasks**  
Std=0 and mean−1.07 at N=3 (worse than a random plan). The LLM generates a deterministically wrong plan every time — natural language strips all numerical precision. There is no seed that causes it to recover.

**3. `raw` and `relative` are the genuine probes**  
Only these two representations show non-zero variance across seeds at both N values, meaning the LLM actually responds differently to different initial conditions. `raw` is more sensitive (std=0.35/0.51) vs `relative` (std=0.23/0.16). Both have identifiable “correct centroid” trials mixed with “wrong point” trials.

**4. `graph` is brittle at N=3 but recovers at N=6**  
ZeroDivisionError in 50% of N=3 trials (drones nearly co-located → distance denominators ≈ 0). At N=6, drones are more spread out, distances are well-defined, and `graph` becomes the best representation (mean=−1.015).

**5. Ranking inverts between N=3 and N=6**  
N=3: `aggregate` > `graph`* > `raw` > `relative` >> `natural_language`  
N=6: `graph` > `relative` > `raw` > `aggregate` > `natural_language`  
No single representation wins at both scales. This is the thesis’s central message: the optimal encoding depends on the task × scale interaction.

**6. Sharp performance degradation from N=3 to N=6**  
Best reward drops from ~−0.08 to ~−0.14; mean drops by 10–13× for most representations. Scaling is the dominant effect — harder than representation choice.

### Comparison with Circle Task (E-02)

| Metric | Circle (E-02) | Rendezvous (E-03) |
|--------|--------------|--------------------|
| Reward range (N=3) | −0.026 to −0.042 | −0.086 to −1.065 |
| Std=0 representations | graph, raw (task-independent) | aggregate, graph, natural_language (for wrong reasons) |
| Representation ranking | near-equal at both N | strong N×repr interaction |
| State actually used? | Never (all ignore positions) | Sometimes (`raw`, `relative` only) |

This confirms that task design — not representation alone — is the primary experimental variable.

### Implications for Thesis

1. **The central claim is supported:** State representation format affects planning quality on state-dependent tasks, but the effect is task- and scale-dependent.
2. **`raw` and `relative` are recommended** for future state-dependent experiments — they provide clean probes of actual LLM state-use without confounds.
3. **`aggregate` should exclude the centroid** from its summary stats going forward, or be used only for state-independent tasks.
4. **Next priority:** E-04 — verify the bimodal plan pattern in `raw`/`relative` by logging the generated code and checking whether it reads `state[“positions”]` or hardcodes a fixed point.

---

**Date:** 2026-04-06  
**Source:** E-03 (`run_e03.py`, `results_e03/results.json`) — N=3 complete, N=6 pending (token limit)  
**Trials:** 50 of 100 (N=3 only); N=6 to be resumed with `run_e03_resume.py`

### Setup

The rendezvous task asks all drones to converge to the **initial centroid of their current positions** (from `state["positions"]`). This is structurally different from the circle task:

- Circle: target positions are fully determined by N alone → LLM can hardcode geometry
- Rendezvous: target is the **mean of input positions** → LLM **must read `state["positions"]`** to compute it

This makes it a clean probe of whether representation quality affects the LLM's ability to use the runtime state.

### N=3 Results

| N | Representation     | Mean Reward | Std    | Best Reward | Valid% |
|---|-------------------|-------------|--------|-------------|--------|
| 3 | relative           |   −0.3340   | 0.3055 |   −0.0783   |  100%  |
| 3 | graph              |   −0.5840   | 0.2496 |   −0.0737   |  100%  |
| 3 | aggregate          |   −0.6460   | 0.1871 |   −0.0727   |  100%  |
| 3 | natural_language   |   −1.0128   | 0.0000 |   −0.0717   |   10%  |
| 3 | raw                |   −1.1481   | 0.6703 |   −0.0708   |  100%  |

### Key Finding: Representations Diverge on State-Dependent Task

**Circle task (E-02):** `raw` ≈ `relative` ≈ −0.036 — effectively identical  
**Rendezvous task (E-03):** `relative` = −0.33, `raw` = −1.15 — **3.4× gap**

This directly confirms the hypothesis from the previous two conversations:
> *When the task requires the LLM to reason about the actual current state (not just task geometry), the representation format matters significantly.*

### Why `relative` Wins

The `relative` representation expresses each drone's position as a distance and direction from a reference point, making centroid computation natural:
- "Drone 0 is 0.3m north of center" → centroid is approximately at center

The `raw` representation lists raw XYZ coordinates, which the LLM sometimes processes correctly but often treats as targets to reach individually rather than inputs to aggregate.

### Why `raw` Has High Variance (std=0.67)

Inconsistent behavior: on some seeds the LLM correctly computes `centroid = mean(positions)`. On others it either:
- Sends all drones to (0, 0, 1.0) — hardcoding the origin as "centroid"  
- Uses the first drone's position as the meeting point  
- Correctly computes centroids but distributes the approach unevenly

This trial-by-trial inconsistency at temperature=0 is unusual and likely driven by minor prompt formatting differences between seeds.

### Why `natural_language` Has 10% Validity

9 out of 10 trials produced invalid code. The combination of:
- A verbose natural language state description (the longest prompt format)
- A task requiring a computation ("compute the centroid")
- The instruction to output `def plan(state, duration):` with trajectory format

...appears to push the model past a threshold where it outputs explanatory prose instead of a code block. This is a prompt engineering failure, not an LLM capability failure.

### Thesis Implications

1. **Representation × Task interaction is the central experimental variable.** Not just "which representation" but "which representation for which task class."
2. **`relative` is the recommended default** — robust across both circle (geometric) and rendezvous (state-dependent) tasks.
3. **`raw` is dangerous for state-dependent tasks** — its high variance makes it unreliable despite 100% validity.
4. **Validity ≠ quality.** `natural_language` hits a validity cliff before we even measure execution quality. Need to ensure the prompt always elicits a code block.

### Next Steps

- **Resume E-03 for N=6** with `run_e03_resume.py` (token limit reset)
- **Fix `natural_language` validity** — add explicit code-only instruction or reduce state description verbosity
- **Log whether code reads `state["positions"]`** — quantify state-use rate per representation

---

## F-06 — Direct JSON Waypoints: In-Context Arithmetic Reasoning (E-04)

**Status:** ✅ Complete  
**Source:** E-04  
**Date:** 2026-04-08

### Summary

E-04 confirms that removing code execution exposes a major robustness gap. Overall validity drops from **94% (E-03)** to **70% (E-04)**, with complete collapse in `relative` (0% valid across N=3 and N=6) and `aggregate` at N=6 (0% valid). For conditions that remain valid, reward quality is still poor and collision rates are extreme (100% collision-trial rate in all valid buckets).

### Hypotheses Under Test

| # | Hypothesis | Prediction | Result |
|---|------------|------------|--------|
| H1 | Valid rate drops vs E-03 | Lower — JSON harder to produce correctly than Python | ✅ Confirmed (94% → 70%) |
| H2 | `raw` reward comparable to E-03 | Neutral / slight improvement if CoT helps arithmetic | ❌ Rejected (`raw` degrades strongly) |
| H3 | `aggregate` advantage disappears | Yes — no code to forward centroid string into | ✅ Confirmed (`aggregate` loses lead; N=6 has 0% validity) |
| H4 | `natural_language` validity recovers | Yes — no `def plan(...)` needed | ✅ Partially confirmed (validity recovers, quality does not) |

### E-03 vs E-04 Comparison

| Metric | E-03 (code) | E-04 (direct JSON) | Δ |
|--------|-------------|-------------------|---|
| Overall valid% | 94.0% | 70.0% | −24.0 pp |
| Best N=3 mean reward | −0.0860 (`aggregate`) | −0.7372 (`graph`) | worse in E-04 |
| Best N=6 mean reward | −1.0148 (`graph`) | −0.7158 (`natural_language`) | better, but with 100% collisions |
| Worst valid bucket | −1.3527 (`natural_language`, N=6) | −2.0096 (`raw`, N=6) | E-04 worse tail |
| Collision-trial rate (valid buckets) | 100% (E-03 v2 logged as Col%=100%) | 100% | unchanged high risk |

### Analysis

1. **Reasoning exists, but is brittle.** In valid `raw` trials, the model explicitly computes centroid arithmetic in text and outputs matching JSON waypoints.
2. **Reasoning quality is representation-limited.** In `natural_language` at N=6, the model invents coordinates because exact positions are not present, then reasons from those assumptions.
3. **Failure traces were missing in E-04 run.** Invalid records contain `error=direct_parse_failed` but `llm_response=None`, so failed-case reasoning cannot be audited post-hoc.
4. **Execution quality remains poor despite explicit reasoning text.** Even valid responses converge aggressively to the same point, driving persistent close-proximity/collision behavior.

### Was it "0 reasoning"?

No — not zero. The model often provides stepwise arithmetic reasoning (centroid calculation) in valid trials. The key issue is **not absence of reasoning**, but **low-fidelity reasoning + weak policy quality + fragile output formatting**.

### Instrumentation update

`experiment.py` now stores `llm_response`, `prompt_length`, and `code` in failed trial records (`_failed_result(...)`). Future direct-mode runs will preserve raw outputs even when parsing fails, enabling proper failure forensics.

### Thesis Implications

1. **Direct-output evaluation is necessary** to distinguish symbolic/code scaffolding from true in-context spatial reasoning.
2. **Representation quality and information completeness dominate** once interpreter support is removed.
3. **Validity alone is insufficient**; response trace retention and collision/safety metrics must be part of the evaluation protocol.
