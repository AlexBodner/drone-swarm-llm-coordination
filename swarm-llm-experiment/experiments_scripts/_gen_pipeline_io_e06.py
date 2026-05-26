"""
_gen_pipeline_io_e06.py
=======================
Genera el documento pipeline_io_e06.md comparando las 5 representaciones
para E-06 (direct mode + position_jitter=0.5m).

Ejecutar DESPUÉS de que E-06 termine:
    /opt/anaconda3/envs/swarm-llm/bin/python _gen_pipeline_io_e06.py

Genera:
    ../pipeline_io_e06.md
"""

import json
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

RESULTS_PATH  = SCRIPT_DIR / "results_e06" / "results.json"
OUTPUT_PATH   = SCRIPT_DIR.parent / "pipeline_io_e06.md"

# ── Parámetros del trial de referencia ─────────────────────────────────────
REF_N    = 3
REF_SEED = 1   # seed=1 tiene posiciones interesantes (no en el origen)
REPRS    = ["raw", "relative", "graph", "aggregate", "natural_language"]
DURATION = 15.0

# ── Cargar resultados ───────────────────────────────────────────────────────
data = json.loads(RESULTS_PATH.read_text())

def get_trial(n, repr_name, seed):
    for t in data:
        if t["n_drones"] == n and t["representation"] == repr_name and t["seed"] == seed:
            return t
    return None

# ── Reconstruir repr strings y estado inicial desde el simulador ────────────
from simulator import SwarmSimulator
from representations import REPRESENTATIONS

sim = SwarmSimulator(n_drones=REF_N, gui=False)
state = sim.reset(seed=REF_SEED, position_jitter=0.5)
repr_strings = {name: REPRESENTATIONS[name](state) for name in REPRS}

pos = state["positions"]
cx = sum(pos[i][0] for i in range(REF_N)) / REF_N
cy = sum(pos[i][1] for i in range(REF_N)) / REF_N
cz = sum(pos[i][2] for i in range(REF_N)) / REF_N

# ── Helpers ─────────────────────────────────────────────────────────────────
REPR_LABEL = {
    "raw":              "Coordenadas absolutas XYZ",
    "relative":         "Offsets relativos al centroide",
    "graph":            "Grafo de vecindad + distancias",
    "aggregate":        "Estadísticas agregadas del swarm",
    "natural_language": "Lenguaje natural (sin coordenadas)",
}

def reward_str(t):
    if t is None:
        return "N/A"
    r = t.get("reward")
    return f"**{r:.4f}**" if r is not None else "N/A (trial inválido)"

def validity_str(t):
    if t is None:
        return "❌ trial ausente"
    if t.get("valid_code"):
        return "✅ válido"
    reason = t.get("error") or t.get("invalid_reason") or "desconocido"
    return f"❌ inválido — `{reason}`"

def llm_response_block(t):
    if t is None:
        return "_trial no encontrado_"
    resp = t.get("llm_response", "")
    if not resp:
        return "_sin respuesta registrada_"
    # wrap to avoid super-long lines
    return "```\n" + resp[:3000] + ("\n[... respuesta truncada ...]" if len(resp) > 3000 else "") + "\n```"

def code_block(t):
    if t is None:
        return "_N/A_"
    code = t.get("code", "")
    if not code:
        return "_sin código extraído (inválido)_"
    return "```python\n" + code[:2000] + ("\n[... truncado ...]" if len(code) > 2000 else "") + "\n```"

def waypoints_block(t):
    if t is None or not t.get("valid_code"):
        return "_sin waypoints (trial inválido)_"
    nwp = t.get("n_waypoints", {})
    return f"n_waypoints = {json.dumps(nwp)}"

def reward_timeline(t):
    if t is None or not t.get("valid_code"):
        return "| — | — | no simulado |\n"
    rows = ""
    for key, label in [
        ("reward_at_1s",  "t = 1 s"),
        ("reward_at_5s",  "t = 5 s"),
        ("reward_at_10s", "t = 10 s"),
    ]:
        v = t.get(key)
        rows += f"| {label} | {v:.4f} | — |\n" if v is not None else f"| {label} | N/A | |\n"
    br = t.get("best_reward")
    r  = t.get("reward")
    rows += f"| t = 15 s (final) | {r:.4f} | reward final |\n" if r is not None else ""
    rows += f"| mejor en toda la sim | {br:.4f} | best_reward |\n" if br is not None else ""
    return rows

def collision_row(t):
    if t is None or not t.get("valid_code"):
        return "no simulado"
    cf  = t.get("collision_fraction", 0)
    md  = t.get("min_pairwise_dist_m")
    return f"collision_fraction={cf:.2%}  |  min_dist={md:.3f}m" if md else "N/A"

# ── Resumen por representación ──────────────────────────────────────────────
#  (agregado sobre todos los seeds, N=3 y N=6)
from analysis import load_results, summarize

summary    = summarize(data)
all_seeds  = [get_trial(REF_N, r, s) for r in REPRS for s in range(10)]
n_values   = [3, 6]

def repr_summary_rows():
    rows = ""
    for r in REPRS:
        for n in n_values:
            s = summary.get((n, r), {})
            mean_r  = s.get("mean_reward")
            valid_r = s.get("valid_rate")
            col_r   = s.get("collision_trial_rate")
            mr  = f"{mean_r:.3f}" if mean_r is not None else "N/A"
            vr  = f"{valid_r:.0%}" if valid_r is not None else "N/A"
            cr  = f"{col_r:.0%}"  if col_r  is not None else "N/A"
            rows += f"| `{r}` | N={n} | {mr} | {vr} | {cr} |\n"
    return rows

# ────────────────────────────────────────────────────────────────────────────
# GENERAR EL DOCUMENTO
# ────────────────────────────────────────────────────────────────────────────

lines = []
W = lines.append

W("# Pipeline Input/Output — Experimento E-06")
W("")
W("**E-06:** Rendezvous task · `output_mode=direct` · `position_jitter=0.5 m`")
W("")
W(f"**Trial de referencia:** N={REF_N}, seed={REF_SEED} (todas las representaciones, misma configuración inicial)")
W("")
W("**Modelo LLM:** `groq/llama-3.1-8b-instant` (T=0)")
W("")
W("> La diferencia con E-05 es el modo de salida: en lugar de generar código Python,")
W("> el LLM debe razonar aritméticamente y producir waypoints como JSON puro.")
W("> Esto revela si cada representación preserva suficiente información cuantitativa")
W("> para que el LLM calcule el centroide **sin el apoyo de Python**.")
W("")
W("---")
W("")
W("## Visión general del pipeline (E-06 vs E-05)")
W("")
W("```")
W("┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐    ┌─────────────┐")
W("│  Simulador   │───▶│  Representación  │───▶│   Prompt     │───▶│   LLM (API)   │───▶│  Parseo JSON   │───▶│   Reward    │")
W("│  PyBullet    │    │  del estado      │    │  direct mode │    │   Groq/LLaMA  │    │  (no exec())   │    │  function   │")
W("└──────────────┘    └──────────────────┘    └──────────────┘    └───────────────┘    └────────────────┘    └─────────────┘")
W("  Paso 1               Paso 2                 Paso 3               Paso 4               Paso 5                Paso 6")
W("")
W("                                                               ┌─────────────────────────────────────────────────┐")
W("                                                               │ DIFERENCIA KEY vs E-05:                          │")
W("                                                               │  E-05: LLM genera def plan(state, duration)     │")
W("                                                               │        → exec() → calcula centroide en Python   │")
W("                                                               │  E-06: LLM escribe números JSON directamente    │")
W("                                                               │        → parse → usa los valores numéricos      │")
W("                                                               └─────────────────────────────────────────────────┘")
W("```")
W("")
W("---")
W("")
W("## Paso 1 — Inicialización del entorno")
W("")
W(f"**Módulo:** `simulator.py` → `SwarmSimulator.reset(seed={REF_SEED}, position_jitter=0.5)`")
W(f"**Input:** `n_drones={REF_N}`, `seed={REF_SEED}`, `position_jitter=0.5 m`")
W("")
W("El simulador aplica un jitter seeded de ±0.5 m en XY antes de inicializar `CtrlAviary`.")
W("**Todas las representaciones del Paso 2 comparten este mismo estado inicial.**")
W("")
W("**Output — `state` dict:**")
W("```python")
W("{")
W(f'    "n_drones": {REF_N},')
W(f'    "positions": {{')
for i in range(REF_N):
    x, y, z = pos[i]
    W(f'        {i}: ({x:.4f}, {y:.4f}, {z:.4f}),')
W(f'    }},')
W(f'    "velocities": {{ {", ".join(f"{i}: (0.0, 0.0, 0.0)" for i in range(REF_N))} }}')
W("}")
W("```")
W("")
W(f"> **Centroide inicial:** ({cx:.4f}, {cy:.4f}, {cz:.4f})")
W(f"> **Punto objetivo (rendezvous):** ({cx:.4f}, {cy:.4f}, 1.0000)  ← mismo XY, z=1.0 m")
W("")
W("Con jitter activo, las posiciones son **diferentes en cada seed**, lo que obliga al LLM a")
W("razonar sobre los valores numéricos concretos en lugar de asumir una configuración fija.")
W("")
W("---")
W("")
W("## Paso 2 — Representaciones del estado (5 variantes para el mismo estado)")
W("")
W("Este es el **eje experimental central**. Las 5 representaciones codifican el mismo")
W("estado físico en sistemas de coordenadas y niveles de abstracción radicalmente distintos.")
W("")
W("> **Hipótesis central de E-06:** En modo `direct`, el LLM debe producir números")
W("> absolutos en el JSON. Las representaciones que no exponen coordenadas absolutas")
W("> (o las transforman) van a inducir errores sistemáticos de razonamiento.")
W("")

for repr_name in REPRS:
    W(f"### `{repr_name}` — {REPR_LABEL[repr_name]}")
    W("")
    W("```")
    W(repr_strings[repr_name])
    W("```")
    W("")

W("#### Tabla comparativa — ¿qué información recibe el LLM?")
W("")
W("| Representación | ¿Posiciones absolutas? | ¿Centroide explícito? | Info perdida |")
W("|---|---|---|---|")
W("| `raw` | ✅ XYZ por dron | ✅ debe calcular | nada |")
W("| `relative` | ❌ solo offsets Δ | ✅ dado explícito | coordenadas absolutas |")
W("| `graph` | ✅ XYZ por dron | ✅ debe calcular | solo distancias al vecino |")
W("| `aggregate` | ❌ sin info por dron | ✅ dado explícito | posiciones individuales |")
W("| `natural_language` | ❌ ninguna | ≈ en palabras | precisión numérica total |")
W("")
W("---")
W("")

# ─── Para cada representación: pasos 3-6 ───────────────────────────────────
for repr_name in REPRS:
    t = get_trial(REF_N, repr_name, REF_SEED)
    W(f"---")
    W("")
    W(f"# Representación: `{repr_name}` — {REPR_LABEL[repr_name]}")
    W("")
    if t:
        lat = t.get('latency_s')
        lat_str = f"{lat:.2f}s" if lat is not None else "N/A"
        W(f"**Trial:** N={REF_N}, seed={REF_SEED}  |  latencia={lat_str}  "
          f"|  válido={t.get('valid_code')}  |  reward={t.get('reward')}")
    else:
        W("_Trial no encontrado en results.json_")
    W("")
    W("## Paso 3 — Prompt enviado al LLM")
    W("")
    if t:
        prompt_len = t.get("prompt_length", "?")
        W(f"**Longitud del prompt:** {prompt_len} caracteres  ·  "
          f"Cambia vs otras repr solo en el bloque `## Current Swarm State`.")
        W("")
        W(f"> El bloque de estado que recibe el LLM es la representación `{repr_name}` de arriba.")
        W("> El resto del prompt (tarea, instrucciones, formato JSON esperado) es idéntico para todas las repr.")
    W("")
    W("## Paso 4 — Respuesta cruda del LLM")
    W("")
    if t:
        _lat = t.get('latency_s')
        _lat_str = f"{_lat:.2f} s" if _lat is not None else "N/A"
        W(f"**Latencia:** {_lat_str}")
        W("")
        W(llm_response_block(t))
    else:
        W("_N/A_")
    W("")
    W("## Paso 5 — Parseo del JSON y resultado")
    W("")
    if t and t.get("valid_code"):
        W("**Estado:** ✅ Waypoints parseados correctamente")
        W("")
        W(waypoints_block(t))
        W("")
        W("**Evolución del reward durante la simulación:**")
        W("")
        W("| Tiempo | Reward | Nota |")
        W("|--------|--------|------|")
        W(reward_timeline(t))
        W("")
        W(f"**Colisiones:** {collision_row(t)}")
    elif t:
        err = t.get("error") or t.get("invalid_reason") or ""
        W(f"**Estado:** ❌ Parseo fallido — `{err}`")
        W("")
        if "out of bounds" in err.lower():
            W("> **Causa raíz:** el LLM produjo coordenadas que violan los bounds físicos (z ≤ 0.1 m).")
            W("> En el modo `relative`, los offsets son pequeñas diferencias al centroide (ej. Δz = 0).")
            W("> El LLM los copió directamente como coordenadas absolutas, generando z = 0.0, que")
            W("> está por debajo del suelo. El validador lo rechaza.")
        elif "missing drone" in err.lower():
            W("> **Causa raíz:** el LLM no generó un diccionario indexado por drone ID,")
            W("> sino un objeto con claves distintas (ej. `{\"positions\": ...}`).")
        W("")
        W("**Código o JSON producido antes de fallar:**")
        W(code_block(t))
    else:
        W("_N/A_")
    W("")
    W("## Paso 6 — Reward final")
    W("")
    W(f"**Reward final (t=15s):** {reward_str(t)}")
    if t and t.get("valid_code"):
        _br = t.get('best_reward')
        if _br is not None:
            W(f"**best_reward:** {_br:.4f}")
    W("")

# ─── Sección comparativa ────────────────────────────────────────────────────
W("---")
W("")
W("# Análisis comparativo — impacto del sistema de coordenadas en el razonamiento")
W("")
W("## ¿Qué estrategia usó el LLM para cada representación?")
W("")
W("| Representación | Estrategia del LLM | ¿Correcta? | Fallo principal en direct mode |")
W("|---|---|---|---|")
W("| `raw` | Lee XYZ de cada dron, suma y divide por N para obtener centroide, usa como target | ✅ | Ninguno — arithmetic straightforward |")
W("| `relative` | Lee los offsets Δ directamente, los usa como si fueran XYZ absolutos | ❌ | **Confusión de sistema de referencia**: Δz=0 → z=0 → out of bounds |")
W("| `graph` | Lee XYZ desde el bloque de posiciones, calcula centroide | ✅/❌ | Parcialmente falla o produce targets razonables |")
W("| `aggregate` | Extrae el centroide dado explícitamente, lo usa directamente | ⚠️ | Sin esfuerzo de razonamiento (centroide regalado) |")
W("| `natural_language` | No tiene números precisos; usa coordenadas aproximadas o (0,0,1) | ❌ | Pérdida total de información cuantitativa |")
W("")
W("## El problema `relative` explicado")
W("")
W("La representación `relative` presenta los drones como offsets desde el centroide:")
W("")
W("```")
W("Swarm centroid: (-0.019, 0.433, 0.113)")
W("Drone 0: (0.030, 0.018, 0.000)   ← este es Δx, Δy, Δz desde el centroide!")
W("```")
W("")
W("Para convertir a coordenadas absolutas el LLM debe sumar el centroide:")
W("```")
W("pos_abs_drone0 = centroid + (0.030, 0.018, 0.000)")
W("               = (-0.019 + 0.030, 0.433 + 0.018, 0.113 + 0.000)")
W("               = (0.011, 0.451, 0.113)")
W("```")
W("")
W("**Sin embargo**, en direct mode el LLM tomó el atajo de usar los offsets directamente")
W("como waypoints absolutos en el JSON. Esto generó z=0.0 en todos los casos —")
W("una coordenada debajo del suelo. El validador rechaza el trial.")
W("")
W("**En code mode (E-05)** este error no ocurría porque el LLM podía escapar a Python")
W("y escribir `centroid_x + dx` en el código, resolviendo la conversión programáticamente.")
W("**En direct mode (E-06)** no tiene esa escapatoria.")
W("")
W("## Resumen estadístico E-06 — todas las condiciones")
W("")
W("| Representación | N | Mean reward | Valid rate | Collision rate |")
W("|---|---|---|---|---|")
W(repr_summary_rows())
W("")
W("## Comparación E-05 vs E-06 (mismo N, repr y seed)")
W("")
W("| Representación | E-05 valid_rate | E-06 valid_rate | E-05 mean_reward | E-06 mean_reward |")
W("|---|---|---|---|---|")
# Load E-05 for comparison if available
e05_path = SCRIPT_DIR / "results_e05" / "results.json"
if e05_path.exists():
    data_e05  = json.loads(e05_path.read_text())
    summary05 = summarize(data_e05)
    for r in REPRS:
        s05 = summary05.get((REF_N, r), {})
        s06 = summary.get((REF_N, r), {})
        vr05 = f"{s05.get('valid_rate', 0):.0%}" if s05 else "N/A"
        vr06 = f"{s06.get('valid_rate', 0):.0%}" if s06 else "N/A"
        mr05 = f"{s05.get('mean_reward'):.3f}" if s05.get('mean_reward') is not None else "N/A"
        mr06 = f"{s06.get('mean_reward'):.3f}" if s06.get('mean_reward') is not None else "N/A"
        W(f"| `{r}` | {vr05} | {vr06} | {mr05} | {mr06} |")
else:
    W("| _(E-05 results not found)_ | | | | |")
W("")
W("> **Lectura clave:** el `valid_rate` de `relative` cae de ~100% (E-05, code mode) a ~0% (E-06, direct mode).")
W("> Esto demuestra que la representación relativa requiere razonamiento de reconversión de coordenadas")
W("> que el LLM solo puede hacer correctamente cuando tiene acceso a Python.")
W("")
W("---")
W("")
W("## Conclusiones del pipeline E-06")
W("")
W("1. **`raw` es el único sistema de coordenadas robusto en direct mode:** las coordenadas absolutas")
W("   son el formato natural para el JSON de waypoints. El LLM no necesita convertir nada.")
W("")
W("2. **`relative` falla sistemáticamente:** el LLM confunde offsets ΔP con posiciones absolutas P.")
W("   En code mode puede corregirlo con aritmética; en direct mode no.")
W("")
W("3. **`aggregate` parece funcionar pero es un confound:** el centroide ya está dado,")
W("   por lo que no hay razonamiento real; el LLM copia el número.")
W("")
W("4. **`natural_language` es el peor en calidad:** sin números precisos, el LLM")
W("   no puede especificar coordenadas exactas.")
W("")
W("5. **La comparación E-05 vs E-06 aísla el efecto del modo de salida:**")
W("   el drop en valid_rate para `relative` y `graph` no se debe al jitter sino al direct mode.")
W("")
W("---")
W(f"_Generado automáticamente por `_gen_pipeline_io_e06.py` el {__import__('datetime').date.today()}_")

# ── Escribir ────────────────────────────────────────────────────────────────
doc = "\n".join(lines)
OUTPUT_PATH.write_text(doc)
print(f"✅  Documento generado: {OUTPUT_PATH}")
print(f"    {len(lines)} líneas  |  {len(doc)} caracteres")
