"""Browser verification of the UI (Playwright driving the locally installed Microsoft Edge, headless).

Checks: console errors / page errors / failed requests; displayed card numbers equal an independent /api/compare call;
OOD banner for c > 0.2 and for stress-lab ICs; OOD table equals /api/meta; physics shift changes only the solver;
CSV upload; no horizontal scroll at 390 px; dark mode renders. Screenshots -> figures/ui/, summary -> results/ui_check.json.
Usage: python scripts/ui_check.py http://127.0.0.1:8765
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
OUT = ROOT / "figures" / "ui"
OUT.mkdir(parents=True, exist_ok=True)
checks, console, errors, failed = [], [], [], []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def api(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def pct(v):  # same formatting as app.js
    return f"{100 * v:.{3 if v < 0.01 else 2}f}%"


def wait_result(page, prev_id=None, timeout=30):
    """Wait until a new /api/compare result has been rendered."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok = page.evaluate("(p) => !!(window.__app && window.__app.last && window.__app.last !== p && "
                           "document.getElementById('health').textContent.startsWith('ready'))",
                           prev_id) if prev_id is None else page.evaluate(
            "() => window.__app.lastStamp !== window.__prevStamp && document.getElementById('health').textContent.startsWith('ready')")
        if ok:
            page.wait_for_timeout(300)
            return True
        page.wait_for_timeout(150)
    return False


def set_range(page, el_id, value):
    page.evaluate("([id, v]) => { window.__prevStamp = window.__app.lastStamp; const e = document.getElementById(id); "
                  "e.value = v; e.dispatchEvent(new Event('input')); }", [el_id, value])


def cards(page):
    return page.evaluate("() => Array.from(document.querySelectorAll('.card')).map(c => "
                         "[c.querySelector('.k').textContent, c.querySelector('.v').textContent])")


with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = ctx.new_page()
    page.on("console", lambda m: console.append({"type": m.type, "text": m.text}))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("requestfailed", lambda r: failed.append(f"{r.url} {r.failure}"))
    page.add_init_script("""
      const _f = window.fetch;
      window.fetch = async (...a) => { const r = await _f(...a);
        if (String(a[0]).includes('/api/compare')) { r.clone().json().then(() => { window.__app.lastStamp = Date.now(); }); }
        return r; };
    """)
    page.goto(URL + "/", wait_until="networkidle")
    check("initial render", wait_result(page))
    page.screenshot(path=str(OUT / "01_level_c0.1.png"), full_page=True)

    # --- displayed numbers vs an independent API call with the same request
    req = page.evaluate("() => window.__app.lastReq")
    ref = api("/api/compare", req)
    shown = dict(cards(page))
    m = ref["metrics"]
    check("card: surrogate vs solver", shown.get("Surrogate vs accurate solver") == pct(m["rel_l2_surrogate_vs_solver"]),
          f"{shown.get('Surrogate vs accurate solver')} vs api {pct(m['rel_l2_surrogate_vs_solver'])}")
    check("card: surrogate vs dataset-style", shown.get("Surrogate vs dataset-style") == pct(m["rel_l2_surrogate_vs_dataset_style"]),
          f"{shown.get('Surrogate vs dataset-style')} vs api {pct(m['rel_l2_surrogate_vs_dataset_style'])}")
    check("card: dataset-style vs accurate", shown.get("Dataset-style vs accurate") == pct(m["rel_l2_dataset_style_vs_solver"]),
          f"{shown.get('Dataset-style vs accurate')} vs api {pct(m['rel_l2_dataset_style_vs_solver'])}")
    check("no OOD banner at c=0.1", page.evaluate("() => document.getElementById('banner').classList.contains('hidden')"))
    n_traces = page.evaluate("() => document.getElementById('profile').data.length")
    check("profile shows three curves in distribution", n_traces == 3, f"{n_traces} traces")
    meta = api("/api/meta")
    expl = page.inner_text("#explainer")
    check("explainer shows dataset tolerance error from results",
          f"{100 * json.loads((ROOT / 'results' / 'solver_validation.json').read_text())['summary']['tight_bdf_vs_data']['mean']:.2f}%" in expl)
    heat = page.evaluate("() => ['hm-sur','hm-sol','hm-err'].map(i => document.getElementById(i).data[0].z.length)")
    check("heatmaps rendered with 101 time rows", heat == [101, 101, 101], str(heat))

    # --- profile time slider
    page.evaluate("() => { const e = document.getElementById('ts'); e.value = 60; e.dispatchEvent(new Event('input')); }")
    y = page.evaluate("() => document.getElementById('profile').data.find(d => d.name.startsWith('accurate')).y.slice(0,3)")
    check("time slider updates profile to t index 60", all(abs(a - b) < 1e-6 for a, b in zip(y, ref["solver"]["u"][60][:3])))

    # --- OOD level
    set_range(page, "c", 0.35)
    check("render after c=0.35", wait_result(page, prev_id=1))
    banner = page.inner_text("#banner")
    check("OOD banner for c=0.35", "Out of distribution" in banner, banner[:80])
    check("no dataset-style curve when OOD", page.evaluate("() => document.getElementById('profile').data.length") == 2)
    page.screenshot(path=str(OUT / "02_level_c0.35_ood.png"), full_page=True)

    # --- stress lab
    page.evaluate("() => window.__prevStamp = window.__app.lastStamp")
    page.click("button.tab[data-mode=stress]")
    check("render stress lab (pulse)", wait_result(page, prev_id=1))
    check("OOD banner for pulse IC", "not spatially constant" in page.inner_text("#banner"))
    rows = page.evaluate("() => Array.from(document.querySelectorAll('#ood-table tr')).slice(1).map(r => "
                         "Array.from(r.cells).map(c => c.textContent))")
    want = [[r["category"], pct(r["mean"]), pct(r["worst"]), str(r["n"])] for r in meta["evaluation"]["ood_table"]]
    check("OOD degradation table equals /api/meta", rows == want, f"{len(rows)} rows")
    page.screenshot(path=str(OUT / "03_stress_pulse.png"), full_page=True)

    page.evaluate("() => window.__prevStamp = window.__app.lastStamp")
    page.select_option("#ic-kind", "custom")
    csv = OUT / "_ic.csv"
    csv.write_text("\n".join(str(0.1 + 0.05 * ((i // 16) % 2)) for i in range(128)))
    page.set_input_files("#csv", str(csv))
    check("render CSV upload IC", wait_result(page, prev_id=1))
    check("CSV IC flagged OOD", page.evaluate("() => window.__app.last.ood.ood"))
    csv.unlink()

    # --- physics-shift lab
    page.evaluate("() => window.__prevStamp = window.__app.lastStamp")
    page.click("button.tab[data-mode=physics]")
    check("render physics lab (training values)", wait_result(page, prev_id=1))
    base = page.evaluate("() => ({s: window.__app.last.surrogate.u[50][40], e: window.__app.last.metrics.rel_l2_surrogate_vs_solver, ood: window.__app.last.ood.ood})")
    check("physics lab at training values is in distribution", not base["ood"])
    set_range(page, "ph-D", -3.0)  # log10(D) = -3 -> D = 1e-3 (2x the training value)
    check("render physics lab (D=1e-3)", wait_result(page, prev_id=1))
    sh = page.evaluate("() => ({s: window.__app.last.surrogate.u[50][40], e: window.__app.last.metrics.rel_l2_surrogate_vs_solver, ood: window.__app.last.ood.ood, req: window.__app.lastReq.params.D})")
    check("D slider sends D=1e-3", abs(sh["req"] - 1e-3) < 1e-9, str(sh["req"]))
    check("surrogate unchanged by physics shift", abs(sh["s"] - base["s"]) < 1e-9)
    check("error grows with physics shift", sh["e"] > base["e"], f"{base['e']:.4f} -> {sh['e']:.4f}")
    page.click("#run-sweep")
    page.wait_for_function("() => window.__app.lastSweep && document.getElementById('run-sweep').textContent === 'Run sweep'", timeout=60000)
    sw = page.evaluate("() => window.__app.lastSweep.rows")
    check("sweep plotted", page.evaluate("() => document.getElementById('sweep').data[0].x.length") == len(sw))
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / "04_physics_shift.png"), full_page=True)
    page.click("#reset-phys")
    page.wait_for_timeout(1500)

    # --- reload level mode at 1024 resolution
    page.evaluate("() => window.__prevStamp = window.__app.lastStamp")
    page.click("button.tab[data-mode=level]")
    wait_result(page, prev_id=1)
    page.evaluate("() => window.__prevStamp = window.__app.lastStamp")
    page.select_option("#nx", "1024")
    check("render at 1024 points", wait_result(page, prev_id=1) and len(page.evaluate("() => window.__app.last.x")) == 1024)
    page.select_option("#nx", "256")
    page.wait_for_timeout(1500)

    # --- phone width and dark mode
    m_ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
    mp = m_ctx.new_page()
    mp.on("pageerror", lambda e: errors.append("mobile: " + str(e)))
    mp.goto(URL + "/", wait_until="networkidle")
    mp.wait_for_function("() => window.__app && window.__app.last", timeout=30000)
    mp.wait_for_timeout(800)
    sw_ = mp.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]")
    check("no horizontal scroll at 390 px", sw_[0] <= sw_[1], f"scrollWidth {sw_[0]} vs {sw_[1]}")
    mp.screenshot(path=str(OUT / "05_mobile_390.png"), full_page=True)
    d_ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="dark")
    dp = d_ctx.new_page()
    dp.goto(URL + "/", wait_until="networkidle")
    dp.wait_for_function("() => window.__app && window.__app.last", timeout=30000)
    dp.wait_for_timeout(800)
    bg = dp.evaluate("() => getComputedStyle(document.body).backgroundColor")
    check("dark mode background applied", bg == "rgb(21, 23, 26)", bg)
    dp.screenshot(path=str(OUT / "06_dark.png"), full_page=False)
    browser.close()

err_console = [c for c in console if c["type"] in ("error", "warning")]
# the app aborts superseded /api/compare calls on purpose (AbortController); count those separately
aborted = [f for f in failed if "/api/compare" in f and "ERR_ABORTED" in f]
real_failed = [f for f in failed if f not in aborted]
check("no page errors", not errors, "; ".join(errors)[:300])
check("no console errors/warnings", not err_console, json.dumps(err_console)[:300])
check("no failed requests (excluding deliberately aborted, superseded calls)", not real_failed, "; ".join(real_failed)[:300])
res = {"url": URL, "passed": sum(c["ok"] for c in checks), "total": len(checks), "checks": checks,
       "console_errors": err_console, "page_errors": errors, "failed_requests": real_failed,
       "aborted_superseded_requests": len(aborted)}
(ROOT / "results" / "ui_check.json").write_text(json.dumps(res, indent=1))
print(f"{res['passed']}/{res['total']} checks passed")
