// Sorption Surrogate Lab: vanilla JS + Plotly. Every evaluation number shown comes from /api/meta (result files)
// or from the live /api/compare response.
"use strict";
const $ = (id) => document.getElementById(id);
const S = { mode: "level", meta: null, last: null, ctrl: null, csv: null, timer: null };
window.__app = S; // for verification in the browser

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  (Math.abs(v) >= 1e-2 || v === 0 ? v.toFixed(d) : v.toExponential(2));
const pct = (v) => (v === null || v === undefined) ? "–" : (100 * v).toFixed(v < 0.01 ? 3 : 2) + "%";

const STRESS = {
  pulse: [["base", 0, 0.3, 0.05], ["position", 0.05, 0.95, 0.5], ["width", 0.01, 0.3, 0.05], ["height", 0, 1, 0.5]],
  smooth: [["seed", 0, 99, 1, 1], ["base", 0, 0.3, 0.1], ["amplitude", 0, 0.3, 0.05], ["corr_len", 0.02, 0.5, 0.1]],
  noise: [["seed", 0, 99, 1, 1], ["base", 0, 0.3, 0.1], ["noise_level", 0, 0.3, 0.05]],
  step: [["base", 0, 0.3, 0.05], ["position", 0.05, 0.95, 0.5], ["height", 0, 1, 0.3]],
  custom: [],
};
const PHYS_LABEL = { D: "diffusion coefficient D", por: "porosity φ", rho_s: "bulk density ρ_s",
  k_f: "Freundlich constant k_f", n_f: "Freundlich exponent n_f", sol: "left boundary value (solubility)" };
const LOG_PARAMS = new Set(["D", "k_f"]);

async function api(path, body, signal) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body), signal } : { signal });
  if (!r.ok) { let m = r.statusText; try { const j = await r.json(); m = JSON.stringify(j.detail); } catch (e) {} throw new Error(m); }
  return r.json();
}

// ------------------------------------------------------------------ controls
function bindRange(id, outId, f) {
  const el = $(id), out = $(outId);
  const upd = () => { out.textContent = f ? f(+el.value) : el.value; };
  el.addEventListener("input", () => { upd(); schedule(); });
  upd();
}

function buildStress() {
  const kind = $("ic-kind").value, box = $("ic-params");
  box.innerHTML = "";
  $("csv-box").classList.toggle("hidden", kind !== "custom");
  for (const [name, lo, hi, val, step] of STRESS[kind]) {
    const id = "p-" + name;
    box.insertAdjacentHTML("beforeend",
      `<label>${name} <output id="${id}-o"></output></label>
       <input type="range" id="${id}" min="${lo}" max="${hi}" step="${step || (hi - lo) / 200}" value="${val}">`);
    bindRange(id, id + "-o");
  }
}

function buildPhysics(meta) {
  const box = $("phys-sliders"), sel = $("sweep-param");
  for (const [k, [lo, hi]] of Object.entries(meta.param_ranges)) {
    const d = meta.params_default[k], log = LOG_PARAMS.has(k);
    const toS = (v) => log ? Math.log10(v) : v, fromS = (s) => log ? 10 ** s : s;
    box.insertAdjacentHTML("beforeend",
      `<label>${PHYS_LABEL[k]} <output id="ph-${k}-o"></output></label>
       <input type="range" id="ph-${k}" min="${toS(lo)}" max="${toS(hi)}" step="${(toS(hi) - toS(lo)) / 200}"
        value="${toS(d)}" data-log="${log}">`);
    // range inputs snap to their step grid, so remember the snapped position of the training value
    const el = $("ph-" + k);
    el.dataset.default = el.value;
    bindRange("ph-" + k, `ph-${k}-o`, (s) => {
      const atDefault = s === +el.dataset.default;
      return `${fmt(atDefault ? d : fromS(s), 4)}${atDefault ? " (training)" : ""}`;
    });
    sel.insertAdjacentHTML("beforeend", `<option value="${k}">${PHYS_LABEL[k]}</option>`);
  }
}

function physValues() {
  const p = {};
  for (const k of Object.keys(S.meta.param_ranges)) {
    const el = $("ph-" + k), s = +el.value;
    let v = el.dataset.log === "true" ? 10 ** s : s;
    if (s === +el.dataset.default) v = S.meta.params_default[k]; // snapped training position -> exact training value
    p[k] = v;
  }
  return p;
}

function request() {
  const base = { model: $("model").value, nx: +$("nx").value, x_query: +$("xq").value };
  if (S.mode === "level") return { ...base, ic: { kind: "constant", level: +$("c").value } };
  if (S.mode === "physics") return { ...base, ic: { kind: "constant", level: +$("pc").value }, params: physValues() };
  const kind = $("ic-kind").value;
  if (kind === "custom") return S.csv ? { ...base, ic: { kind, values: S.csv } } : null;
  const ic = { kind };
  for (const [name] of STRESS[kind]) ic[name] = +$("p-" + name).value;
  return { ...base, ic };
}

function schedule() { clearTimeout(S.timer); S.timer = setTimeout(run, 220); }

async function run() {
  const req = request();
  if (!req) return;
  if (S.ctrl) S.ctrl.abort();
  S.ctrl = new AbortController();
  $("health").textContent = "computing…";
  try {
    const r = await api("/api/compare", req, S.ctrl.signal);
    S.last = r; S.lastReq = req;
    render(r);
    $("health").textContent = `ready · ${r.total_ms} ms`;
  } catch (e) {
    if (e.name === "AbortError") return;
    $("health").textContent = "error";
    banner("bad", "Request failed: " + e.message);
  }
}

// ------------------------------------------------------------------ rendering
function banner(kind, html) {
  const b = $("banner");
  if (!kind) { b.className = "banner hidden"; return; }
  b.className = "banner " + kind; b.innerHTML = html;
}

function card(k, v, n) { return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n || ""}</div></div>`; }

function render(r) {
  const ev = S.meta.evaluation || {};
  if (r.ood.ood) {
    banner("warn", "<b>Out of distribution.</b> " + r.ood.reasons.join("; ") +
      ". The surrogate output below is an extrapolation; trust the solver curve.");
  } else banner(null);

  const m = r.metrics, cards = [];
  const tk = { ensemble: "ensemble_mean", fno: "fno_pad0_s0", cnn: "cnn_s0", interp: "interp_u0" }[r.model];
  const held = (ev.test_two_errors || {})[tk] || {};
  const g = (ev.model_guide || {}).in_range || {};
  const heldData = r.model === "interp" ? g.interp_app_table_vs_data : held.vs_data;
  const errCard = (k, v, n) => `<div class="err-card"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`;
  $("errors").innerHTML =
    (r.model === "interp"
      ? errCard("Error vs dataset", pct(heldData), `held-out test mean (${g.app_table_levels}-level table); for this input the ` +
          "dataset-style curve is this interpolation itself")
      : errCard("Error vs dataset", m.rel_l2_surrogate_vs_dataset_style !== undefined ? pct(m.rel_l2_surrogate_vs_dataset_style) : "n/a",
          m.rel_l2_surrogate_vs_dataset_style !== undefined ? `this input · held-out test mean ${pct(heldData)}`
            : "no stored PDEBench trajectory exists for this input")) +
    errCard("Error vs accurate solver", pct(m.rel_l2_surrogate_vs_solver),
      `this input · held-out test mean ${pct(held.vs_solver)}`);
  $("two-errors").innerHTML = ev.two_errors_html || "";
  if (m.rel_l2_dataset_style_vs_solver !== undefined)
    cards.push(card("Dataset vs accurate", pct(m.rel_l2_dataset_style_vs_solver), "PDEBench generator tolerance"));
  cards.push(card("Max abs error", fmt(m.max_abs_surrogate_vs_solver), "model vs solver"));
  const cm = r.component_ms || {};
  if (r.model !== "interp") cards.push(card("Surrogate", (cm.surrogate ?? r.surrogate.timing_ms).toFixed(1) + " ms", r.model + ", CPU, this call"));
  cards.push(card("Interpolation", (cm.interp ?? 0).toFixed(2) + " ms", "CPU, this call"));
  cards.push(card("Solver", (cm.solver ?? r.solver.timing_ms).toFixed(0) + " ms", "implicit, 1024 cells, this call"));
  if (r.surrogate.spread_mean !== undefined) {
    const tr = ev.trust || {};
    cards.push(card("Ensemble spread", fmt(r.surrogate.spread_mean), tr.card_note || ""));
  }
  $("timing-live").innerHTML = `This call: model ${r.model === "interp" ? "–" : (cm.surrogate ?? 0).toFixed(1) + " ms"}, ` +
    `interpolation ${(cm.interp ?? 0).toFixed(2)} ms, solver ${(cm.solver ?? 0).toFixed(0)} ms; server total ${r.total_ms} ms ` +
    "(network and JSON excluded). The page always runs the solver too, so it updates at solver speed.";
  $("cards").innerHTML = cards.join("");

  const th = +$("th").value, nt = r.t.findIndex((v) => v >= th) + 1 || r.t.length;
  const t = r.t.slice(0, nt), cut = (u) => u.slice(0, nt);
  const font = { color: css("--ink"), size: 11 };
  const lay = (title, extra) => ({ title: { text: title, font: { size: 13 } }, margin: { l: 44, r: 10, t: 30, b: 36 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font,
    xaxis: { title: "x", gridcolor: css("--line") }, yaxis: { title: "t", gridcolor: css("--line") }, ...extra });
  const hm = (z, title, scale, zmax) => Plotly.react("hm-" + title.id, [{ z, x: r.x, y: t, type: "heatmap",
    colorscale: scale, zmin: 0, zmax, colorbar: { thickness: 10 } }], lay(title.text), { displayModeBar: false, responsive: true });
  hm(cut(r.surrogate.u), { id: "sur", text: "Surrogate u(x,t)" }, "Viridis", Math.max(1, ...r.solver.u[0]));
  hm(cut(r.solver.u), { id: "sol", text: "Accurate solver u(x,t)" }, "Viridis", Math.max(1, ...r.solver.u[0]));
  hm(cut(r.abs_error), { id: "err", text: "|surrogate − solver|" }, "Reds", undefined);

  $("ts").max = nt - 1;
  if (+$("ts").value > nt - 1) $("ts").value = nt - 1;
  drawProfile();
  drawObs(nt);
  explain(r);
}

function drawProfile() {
  const r = S.last; if (!r) return;
  const k = +$("ts").value;
  $("ts-out").textContent = `t = ${r.t[k]} s`;
  const tr = [
    { x: r.x, y: r.solver.u[k], name: "accurate solver", line: { color: css("--c-sol"), width: 3 } },
    { x: r.x, y: r.surrogate.u[k], name: (r.model === "interp" ? "model: " : "surrogate: ") + r.model, line: { color: css("--c-sur"), dash: "dash", width: 2 } },
  ];
  if (r.dataset_style) tr.splice(1, 0, { x: r.x, y: r.dataset_style.u[k], name: "dataset-style (PDEBench RK45)",
    line: { color: css("--c-ds"), dash: "dot", width: 2 } });
  Plotly.react("profile", tr, { margin: { l: 44, r: 10, t: 10, b: 36 }, paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)", font: { color: css("--ink"), size: 11 }, legend: { orientation: "h", y: 1.12 },
    xaxis: { title: "x", gridcolor: css("--line") }, yaxis: { title: "u", gridcolor: css("--line") } },
    { displayModeBar: false, responsive: true });
}

function drawObs(nt) {
  const r = S.last, t = r.t.slice(0, nt);
  const src = [["solver", "accurate solver", css("--c-sol"), "solid"], ["dataset_style", "dataset-style", css("--c-ds"), "dot"],
    ["surrogate", "surrogate", css("--c-sur"), "dash"]];
  const one = (div, key, title, ytitle) => {
    const tr = src.filter(([s]) => r[s]).map(([s, name, color, dash]) => ({ x: t, y: r[s].outputs[key].slice(0, nt), name,
      line: { color, dash, width: s === "solver" ? 3 : 2 } }));
    Plotly.react(div, tr, { title: { text: title, font: { size: 13 } }, margin: { l: 50, r: 10, t: 30, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: css("--ink"), size: 11 },
      showlegend: false, // same line styles as the profile legend above
      xaxis: { title: "t [s]", gridcolor: css("--line") }, yaxis: { title: ytitle, gridcolor: css("--line") } },
      { displayModeBar: false, responsive: true });
  };
  one("obs-conc", "conc_at_x", `u at x = ${(+$("xq").value).toFixed(2)}`, "u");
  one("obs-front", "front_position", "front position (u = 0.5)", "x");
  one("obs-uptake", "uptake", "uptake: total amount entered", "∫ C dx");
}

function explain(r) {
  const ev = S.meta.evaluation || {};
  $("explainer").innerHTML = (ev.explainer_html || "") +
    ((ev.trust || {}).spread_html ? "<p>" + ev.trust.spread_html + "</p>" : "");
  $("obs-explain").innerHTML = ev.observables_html || "";
}

function oodTable() {
  const o = (S.meta.evaluation || {}).ood_table;
  if (!o) return;
  $("ood-table").innerHTML = `<label>Measured degradation (held-out OOD set, ensemble vs solver)</label>
    <table><tr><th>IC family</th><th>mean rel L2</th><th>worst</th><th>n</th></tr>` +
    o.map((r) => `<tr><td>${r.category}</td><td>${pct(r.mean)}</td><td>${pct(r.worst)}</td><td>${r.n}</td></tr>`).join("") +
    `</table><div class="range-note">${(S.meta.evaluation || {}).ood_note || ""}</div>`;
}

function guideTable() {
  const ev = S.meta.evaluation || {}, g = ev.model_guide;
  $("guide").innerHTML = ev.model_guide_html || "";
  if (!g) return;
  const ir = g.in_range;
  $("guide-table").innerHTML = `<table><tr><th>input</th><th class="num">FNO ensemble</th><th class="num">interpolation</th>
    <th class="num">CNN</th><th>reference</th></tr>
    <tr><td>in range (held-out test)</td><td class="num">${pct(ir.ensemble_vs_data)}</td>
      <td class="num">${pct(ir.interp_full_vs_data)} (${pct(ir.interp_app_table_vs_data)} with ${ir.app_table_levels} levels)</td>
      <td class="num">${pct(ir.cnn_vs_data)}</td><td>dataset</td></tr>` +
    g.ood_rows.map((r) => `<tr><td>OOD: ${r.family}</td><td class="num">${pct(r.ensemble)}</td><td class="num">${pct(r.interp)}</td>
      <td class="num">${pct(r.cnn)}</td><td>accurate solver</td></tr>`).join("") +
    `</table><div class="range-note">Mean relative L2 from results/eval_metrics.json. OOD rows include the ~1.4% dataset-vs-solver floor
     (see const_in, an in-range control).</div>`;
}

function pivot(rows, unit) {
  const methods = [...new Set(rows.map((r) => r.method))], batches = [...new Set(rows.map((r) => r.batch))].sort((a, b) => a - b);
  return `<table><tr><th>method</th>${batches.map((b) => `<th class="num">batch ${b}</th>`).join("")}</tr>` +
    methods.map((m) => `<tr><td>${m}</td>${batches.map((b) => { const r = rows.find((x) => x.method === m && x.batch === b);
      return `<td class="num">${r ? (r.ms_per_sample < 0.1 ? r.ms_per_sample.toFixed(4) : r.ms_per_sample.toFixed(2)) : "–"}</td>`; }).join("")}</tr>`).join("") +
    `</table><div class="range-note">${unit}</div>`;
}

function benchTable() {
  const bb = (S.meta.evaluation || {}).batch_benchmark;
  if (!bb) return;
  $("bench-table").innerHTML = pivot(bb.rows, `ms per sample. Measured ${bb.measured} on the development laptop (${bb.threads} CPU threads); ` +
    "the solver runs one sample at a time, so its cost does not fall with batch size.");
}

async function runBench() {
  $("run-bench").textContent = "running…";
  try {
    const r = await api("/api/benchmark", {});
    S.lastBench = r;
    $("bench-live").innerHTML = pivot(r.rows, `ms per sample, measured just now on this server (${r.torch_threads} threads).`);
  } catch (e) { banner("bad", "Benchmark failed: " + e.message); }
  $("run-bench").textContent = "Run live benchmark on this server";
}

async function runSweep() {
  $("sweep-panel").classList.remove("hidden");
  $("run-sweep").textContent = "running…";
  try {
    const r = await api("/api/physics_sweep", { param: $("sweep-param").value, level: +$("pc").value, n: 7 });
    S.lastSweep = r;
    Plotly.react("sweep", [{ x: r.rows.map((d) => d.value), y: r.rows.map((d) => 100 * d.rel_l2_surrogate_vs_solver),
      mode: "lines+markers", name: "surrogate vs solver", line: { color: css("--c-sur") } }],
      { margin: { l: 54, r: 10, t: 10, b: 44 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: css("--ink"), size: 11 },
        shapes: [{ type: "line", x0: r.default, x1: r.default, yref: "paper", y0: 0, y1: 1, line: { dash: "dot", color: css("--muted") } }],
        xaxis: { title: PHYS_LABEL[r.param] + " (dotted = training value)", type: LOG_PARAMS.has(r.param) ? "log" : "linear",
                 tickformat: LOG_PARAMS.has(r.param) ? ".1e" : "", dtick: LOG_PARAMS.has(r.param) ? "D2" : undefined,
                 gridcolor: css("--line") },
        yaxis: { title: "relative L2 [%]", type: "log", gridcolor: css("--line") } }, { displayModeBar: false, responsive: true });
  } catch (e) { banner("bad", "Sweep failed: " + e.message); }
  $("run-sweep").textContent = "Run sweep";
}

// ------------------------------------------------------------------ init
async function init() {
  try {
    const [h, meta] = await Promise.all([api("/api/health"), api("/api/meta")]);
    S.meta = meta;
    $("health").textContent = `ready · ${h.ensemble_members}-member ensemble`;
    $("footer").innerHTML = ((meta.evaluation || {}).footer_html || "") + " " + ((meta.evaluation || {}).padding_note || "");
  } catch (e) { $("health").textContent = "API unavailable"; banner("bad", "Cannot reach the API: " + e.message); return; }

  bindRange("c", "c-out", (v) => {
    const out = v > S.meta.c_range[1] || v < S.meta.c_range[0];
    $("c-note").innerHTML = out ? `<b>Outside the training range (0, ${S.meta.c_range[1]}):</b> out of distribution.`
      : `Training data: spatially constant levels c ∈ (0, ${S.meta.c_range[1]}).`;
    $("c-note").style.color = out ? css("--warn-ink") : "";
    return v.toFixed(3);
  });
  bindRange("pc", "pc-out", (v) => v.toFixed(3));
  bindRange("th", "th-out", (v) => v + " s");
  bindRange("xq", "xq-out", (v) => v.toFixed(2));
  $("ts").addEventListener("input", drawProfile);
  for (const id of ["model", "nx"]) $(id).addEventListener("change", run);
  $("ic-kind").addEventListener("change", () => { buildStress(); schedule(); });
  $("csv").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const nums = (await f.text()).split(/[\s,;]+/).map(Number).filter((v) => Number.isFinite(v));
    if (nums.length < 2) { banner("bad", "CSV needs at least 2 numeric values."); return; }
    S.csv = nums.slice(0, 4096); run();
  });
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    S.mode = b.dataset.mode;
    for (const m of ["level", "stress", "physics"]) $("mode-" + m).classList.toggle("hidden", m !== S.mode);
    $("sweep-panel").classList.toggle("hidden", S.mode !== "physics" || !S.lastSweep);
    run();
  }));
  buildStress();
  buildPhysics(S.meta);
  $("reset-phys").addEventListener("click", () => {
    for (const k of Object.keys(S.meta.param_ranges)) { const el = $("ph-" + k); el.value = el.dataset.default; el.dispatchEvent(new Event("input")); }
  });
  $("run-sweep").addEventListener("click", runSweep);
  $("run-bench").addEventListener("click", runBench);
  oodTable();
  guideTable();
  benchTable();
  run();
}
init();
