"""Audit: every number in README prose must match a results file (or a whitelisted, sourced constant).

Generated tables (<!-- GEN --> blocks) are skipped: they are produced from results files by make_readme_tables.py.
Code blocks are skipped. Each prose number is matched against every float/int in results/*.json (and
app/assets/summary.json) rendered in the formats used in the text (percent with 0-3 decimals, scientific, rounded
plain values, ratios). Writes results/readme_audit.json and exits 1 if anything is unmatched.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "README.md").read_text(encoding="utf-8")
text = re.sub(r"<!-- GEN:(\w+) -->.*?<!-- /GEN:\1 -->", "", text, flags=re.S)
text = re.sub(r"```.*?```", "", text, flags=re.S)
text = re.sub(r"`[^`]*`", "", text)                  # inline code: paths, commands, file names
text = re.sub(r"\]\([^)]*\)", "]", text)             # link targets

# constants with a named source (not measurements)
WHITELIST = {
    "5e-4": "generator config D", "0.29": "generator config por", "2880": "generator config rho_s",
    "3.5e-4": "generator config k_f", "0.874": "generator config n_f", "1e-6": "generator regulariser in R(u)",
    "1024": "grid cells (data shape)", "101": "stored times (data shape)", "501": "config tdim", "10,000": "samples (data)",
    "0000": "first key", "9999": "last key", "3600": "first test key", "3999": "last test key", "5": "dt = 5 s / seed count",
    "1e-3": "solve_ivp default rtol", "1e-8": "tight BDF rtol", "0.5": "solver dt / front level", "1": "BC value / dt",
    "0.2": "training IC range", "0": "range bound", "1.0": "BC value", "1.1": "BC perturbation", "10%": "1.0 -> 1.1",
    "3": "seeds (padding)", "6": "served checkpoints", "8": "max OOD samples per family", "17": "checkpoints", "46": "OOD set size",
    "400": "test samples", "20": "validation samples", "38": "UI checks (results/ui_check.json total)", "390": "phone width px",
    "402": "HTTP status (results/deploy_attempt.json)", "404": "HTTP status of the broken CDN URL", "512": "typical free-tier MB", "7860": "Space port", "8765": "local port",
    "1D": "dimension", "2.35.2": "Plotly version", "256": "points / table levels", "2.6.0": "torch version",
    "6000": "sweep optimiser steps", "6": "OOD per family (min)", "10": "latency calls", "x": "",
}
nums = []


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        nums.append(float(o))
    elif isinstance(o, str):
        for m in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", o):
            try:
                nums.append(float(m))
            except ValueError:
                pass


for f in [*(ROOT / "results").rglob("*.json"), ROOT / "app" / "assets" / "summary.json"]:
    walk(json.loads(f.read_text(encoding="utf-8")))
forms = set()
for v in nums:
    for d in range(0, 4):
        forms.add(f"{100 * v:.{d}f}%")
        forms.add(f"{v:.{d}f}")
    for d in range(0, 3):
        forms.add(f"{v:.{d}e}".replace("e-0", "e-").replace("e+0", "e"))
    forms.add(f"{v:,.0f}")
    forms.add(str(int(v)) if v == int(v) else str(v))
# ratios quoted in the text ("about 9x", "2-3x" etc.) are checked by hand below
tokens = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?%?(?![\w])", text)
res = {"matched": [], "whitelisted": [], "unmatched": []}
for t in sorted(set(tokens), key=tokens.index):
    tt = t.rstrip(",")
    if tt in WHITELIST:
        res["whitelisted"].append({"token": tt, "source": WHITELIST[tt]})
    elif tt in forms or tt.replace("%", "") in forms and tt.endswith("%") is False:
        res["matched"].append(tt)
    else:
        ctx = text[max(0, text.find(tt) - 60): text.find(tt) + 40].replace("\n", " ")
        res["unmatched"].append({"token": tt, "context": ctx})
# ratio claims ("8.8×", "1.54–4.06×") must equal ratios computed into app/assets/summary.json
summ = json.loads((ROOT / "app" / "assets" / "summary.json").read_text(encoding="utf-8"))
ratios = {summ["left_over_interior_mae"], summ["timing_slowdown"]["min"], summ["timing_slowdown"]["max"],
          *summ["timing_slowdown"]["factors_loaded_over_quiet"].values()}
rforms = {f"{r:.{d}f}" for r in ratios for d in (1, 2)}
res["ratio_claims"] = []
rtext = re.sub(r'"[^"\n]*"', "", text)  # quoted wrong values in the mistake log are deliberate
for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?\s*×(?!\s*\d)", rtext):  # skip dimensions "N × M"
    for g in [x for x in m.groups() if x]:
        ok = g in rforms
        res["ratio_claims"].append({"value": g, "ok": ok})
        if not ok:
            ctx = rtext[max(0, m.start() - 60): m.end() + 20].replace("\n", " ")
            res["unmatched"].append({"token": g + "×", "context": ctx})
(ROOT / "results" / "readme_audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"matched {len(res['matched'])}, whitelisted {len(res['whitelisted'])}, ratio claims {len(res['ratio_claims'])}, unmatched {len(res['unmatched'])}")
for u in res["unmatched"]:
    print(f"  UNMATCHED {u['token']!r}: ...{u['context']}...")
sys.exit(1 if res["unmatched"] else 0)
