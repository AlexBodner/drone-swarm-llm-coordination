# E-04 — Rendezvous via Direct JSON Waypoints

**Date:** 2026-04-07  
**Status:** ✅ Complete  
**Script:** `run_e04.py`  
**Results dir:** `results_e04/`  
**Run time:** 34.5 min (2072s)

---

## Motivation

E-03 v2 showed that the rendezvous task is solved poorly when the LLM uses code to compute the meeting point. Inspection of LLM outputs revealed a systematic pattern: the model writes `centroid_x = sum([...]) / n` — effectively offloading the arithmetic to the Python interpreter rather than reasoning about the geometry in language. This means E-03 measured "can the LLM write correct Python to compute a centroid?" not "can the LLM reason spatially in context?"

E-04 removes the code-execution pathway entirely. The LLM must:
1. Write down the current positions (in its response).
2. Compute the meeting point using natural language arithmetic steps.
3. Output the result as a plain JSON waypoint dict.

This isolates **in-context arithmetic spatial reasoning** from **code-generation ability**.

---

## Research Question

*Can LLMs compute a spatial centroid and assign convergent waypoints through in-context arithmetic reasoning, without relying on a Python interpreter?*

---

## Hypotheses

| # | Hypothesis | Expected direction |
|---|------------|--------------------|
| H1 | Valid rate drops vs E-03 | Harder to produce well-formed JSON than syntactically valid Python |
| H2 | `raw` reward comparable to E-03 if LLM actually computes centroid in CoT | Neutral / slight improvement |
| H3 | `aggregate` advantage disappears | Aggregate pre-computes centroid in state text → no longer an advantage when no code to forward it into |
| H4 | `natural_language` validity recovers | Longest prompts no longer need to produce a valid `def plan(...)` block |

---

## Design Differences vs E-03

| Dimension | E-03 | E-04 |
|-----------|------|------|
| LLM output format | Python `def plan(state, duration):` | Raw JSON `{"0": [[t,x,y,z],...], ...}` |
| Code execution | `exec()` sandbox | None — JSON parsed directly |
| Prompt | `build_waypoint_prompt()` | `build_direct_waypoint_prompt()` |
| Reasoning required | Optional (can hardcode) | Mandatory REASONING STEPS section |
| `output_mode` param | `"code"` | `"direct"` |
| All other params | Same | Same |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Task | Rendezvous (meet at centroid, hold formation) |
| `reward_mode` | `"rendezvous"` |
| `output_mode` | `"direct"` |
| N_drones | 3, 6 |
| Representations | raw, relative, graph, aggregate, natural_language |
| Seeds | 0–9 (10 per condition) |
| Total trials | 100 |
| LLM | llama-3.1-8b-instant (Groq) |
| Temperature | 0 |
| Simulator | gym-pybullet-drones 2.0.0 + PyBullet |
| Physics | PyBullet (PYB) |
| Drone model | CF2X (Crazyflie 2.x) |
| Duration | 15 s |
| Waypoints / drone | ≥3 required |
| API sleep | 0.3 s between trials |
| Collision radius | 0.25 m |

### Reward Formula (Rendezvous)

$$R = -0.5 \cdot \overline{d_{pairwise}} - 0.5 \cdot \overline{d_{centroid\_drift}}$$

- $\overline{d_{pairwise}}$: mean pairwise inter-drone distance at each step (drives convergence)
- $\overline{d_{centroid\_drift}}$: centroid displacement from its time-averaged position (penalises the group drifting)

---

## Prompt Design

`build_direct_waypoint_prompt()` key elements:

```
IMPORTANT: You must output ONLY a JSON block — NO Python code, NO functions, NO programming.

REASONING STEPS (required in your response):
1. Write down each drone's current position.
2. Compute the meeting point (arithmetic, show your work).
3. For each drone, explain what waypoints it needs to reach the meeting point.

Output format:
```json
{
  "0": [[t1, x1, y1, z1], [t2, x2, y2, z2], ...],
  "1": [...],
  ...
}
```
```

### Parsing pipeline

```
LLM response
    → extract_json_block()   # regex ```json ... ``` with fallback
    → json.loads()           # parse
    → validate bounds        # 0 ≤ t ≤ duration+0.5, z > 0
    → key normalisation      # "0" → 0 (string → int)
    → {int: [(t,x,y,z),...]} # same format as execute_waypoint_plan()
```

---

## Results

Final run completed: 100/100 trials saved in `results_e04/results.json`.

### Summary Table (per condition)

| N | Representation | Valid% | Mean Reward | Std Reward | Collision% | Mean Min Dist |
|---|----------------|--------|-------------|------------|------------|---------------|
| 3 | aggregate | 100% | -0.8780 | 0.0000 | 100% | 0.073m |
| 3 | graph | 100% | -0.7372 | 0.0000 | 100% | 0.058m |
| 3 | natural_language | 100% | -0.9757 | 0.1301 | 100% | 0.067m |
| 3 | raw | 100% | -0.9600 | 0.0000 | 100% | 0.075m |
| 3 | relative | 0% | N/A | N/A | N/A | N/A |
| 6 | aggregate | 0% | N/A | N/A | N/A | N/A |
| 6 | graph | 100% | -0.7762 | 0.0000 | 100% | 0.044m |
| 6 | natural_language | 100% | -0.7158 | 0.0000 | 100% | 0.033m |
| 6 | raw | 100% | -2.0096 | 0.0000 | 100% | 0.074m |
| 6 | relative | 0% | N/A | N/A | N/A | N/A |

**Overall validity:** 70/100 = **70.0%**

### E-03 vs E-04 Comparison

| Metric | E-03 (code) | E-04 (direct JSON) |
|--------|-------------|-------------------|
| Overall valid% | 94.0% | 70.0% |
| Mean reward (N=3, best repr) | -0.0860 (`aggregate`) | -0.7372 (`graph`) |
| Mean reward (N=6, best repr) | -1.0148 (`graph`) | -0.7158 (`natural_language`) |
| Clear failure modes | `natural_language` formatting issues | `relative` frame-conversion collapse; `aggregate` N=6 collapse |

---

## Analysis

Direct JSON mode degrades reliability and quality compared to code mode. The model often writes apparently coherent reasoning text, but execution quality remains poor (high collision fractions and weak rewards), and parsing validity drops sharply in two representations (`relative`, `aggregate@N=6`).

### Hypothesis Evaluation

| Hypothesis | Result | Notes |
|------------|--------|-------|
| H1: Valid rate drops | ✅ Confirmed | E-03: 94% → E-04: 70% |
| H2: `raw` reward comparable | ❌ Rejected | `raw` degrades strongly (N=3: -0.9600; N=6: -2.0096) |
| H3: `aggregate` advantage disappears | ✅ Confirmed | `aggregate` no longer best; N=6 has 0% validity |
| H4: `natural_language` validity recovers | ✅ Partially confirmed | Validity is 100%, but reward quality still mediocre |

### Did the LLM reason, or 0 reasoning?

Short answer: **not zero reasoning**. In valid cases, the model explicitly writes arithmetic steps. But reasoning quality is brittle and sometimes disconnected from true state information.

#### Example A — `raw`, N=3, seed=0 (valid)

**Input state excerpt (from `raw` representation):**
- Drone 0: `(0.000, 0.000, 0.113)`
- Drone 1: `(0.159, 0.159, 0.113)`
- Drone 2: `(0.318, 0.318, 0.113)`

**LLM reasoning excerpt:**
```
x-target = (0.000 + 0.159 + 0.318) / 3 = 0.159
y-target = (0.000 + 0.159 + 0.318) / 3 = 0.159
z-target = (0.113 + 0.113 + 0.113) / 3 = 0.113 (we will use z=1.0 as specified)
```

**LLM JSON output excerpt:**
```json
{
  "0": [[0.0,0.000,0.000,0.113],[2.0,0.000,0.000,1.0],[7.5,0.159,0.159,1.0],[15.0,0.159,0.159,1.0]],
  "1": [[0.0,0.159,0.159,0.113],[2.0,0.159,0.159,1.0],[7.5,0.159,0.159,1.0],[15.0,0.159,0.159,1.0]],
  "2": [[0.0,0.318,0.318,0.113],[2.0,0.318,0.318,1.0],[7.5,0.159,0.159,1.0],[15.0,0.159,0.159,1.0]]
}
```

Interpretation: arithmetic is present and correct, but policy quality is poor (all drones converge to same point too early, causing sustained close spacing/collisions).

#### Example B — `natural_language`, N=6, seed=0 (valid but flawed reasoning)

**LLM reasoning excerpt:**
```
However, the exact positions are not provided, so we will assume the following positions ...
```

Then it includes a Python-style centroid computation block inside reasoning and outputs JSON waypoints to `(0,0,1.0)`.

Interpretation: there is reasoning text, but it relies on fabricated assumptions because natural-language state omits exact coordinates.

#### Example C — invalid cases (`relative`, `aggregate@N=6`)

All 30 invalid trials fail with `error = direct_parse_failed` and store `llm_response = None` in `results.json`, so we cannot inspect those outputs post-hoc.

This means evidence is **not** “0 reasoning”; rather, for failed parses we currently have **0 retained trace**. Future runs should log raw response text even on parse failure.

---

## Key Outputs

- `results_e04/results.json` — full trial-level data
- `results_e04/reward_vs_n.png`
- `results_e04/reward_vs_repr.png`
- `results_e04/validity_rate.png`
- `results_e04/collision_rate.png` *(not generated due post-run plotting TypeError)*
- `results_e04/run.log` — full stdout log

---

## Notes / Observations

### Early run observations (mid-run, first 20 trials)

**`raw` representation (trials 1–10):** All **valid**, all producing `reward = -0.9600`. This is a suspiciously constant value — suggests the LLM outputs the same waypoint JSON regardless of seed. Likely hardcoding a fixed meeting point (e.g., `(0, 0, 1.0)`) rather than computing the state-dependent centroid.

**`relative` representation (trials 11–20):** **100% invalid** with error:
```
Waypoint position out of bounds: [0.0, -0.159, -0.159, 0.0]
```
The `relative` format expresses each drone's position as an offset from a reference. The LLM misinterprets these relative offsets as absolute world coordinates, producing waypoints with `z=0.0` (ground level), which fails the altitude validity check (`z > 0`). This is a fundamental representation mismatch — the LLM does not understand that it must add the offsets back to the reference position to get absolute coordinates.

**Implication:** H4 may be wrong — `relative` validity, not just `natural_language` validity, collapses in the direct JSON setting. The intermediate translation step (offset → absolute) that was implicit in the Python code version is now exposed as a reasoning gap.

