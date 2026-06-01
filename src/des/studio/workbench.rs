//! Self-contained HTML workbench for authoring and running studio specs.
//!
//! This is intentionally dependency-free so `main_build_site` can emit a useful
//! UI artifact without requiring a web framework. The workbench mirrors the
//! JSON spec parser in [`super::spec`]: edit the spec, run it in-browser, drag
//! blocks to adjust layout, and export the updated model document.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::spec::example_spec;

/// Render the default Studio workbench as a complete HTML page.
pub fn workbench_html() -> String {
    workbench_html_for_spec(&example_spec())
}

/// Render the workbench with a custom initial spec.
pub fn workbench_html_for_spec(spec: &Value) -> String {
    TEMPLATE.replace("__SPEC__", &json_for_script(spec))
}

/// Write the default workbench to `path`, creating parent directories.
pub fn write_workbench(path: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, workbench_html())?;
    Ok(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

fn json_for_script(value: &Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|_| "{}".to_string())
        .replace("</", "<\\/")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

const TEMPLATE: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DES Studio Workbench</title>
<style>
  :root {
    color-scheme: light;
    --ink: #111827;
    --muted: #5b6472;
    --line: #d5dbe5;
    --paper: #ffffff;
    --surface: #f4f7fb;
    --source: #e8f8ed;
    --transform: #e9eefc;
    --sink: #fdecec;
    --accent: #1d4ed8;
    --good: #15803d;
    --warn: #a16207;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--ink);
    background: var(--surface);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  header {
    min-height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 18px;
    color: #f8fafc;
    background: #0f172a;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 680; }
  header p { margin: 2px 0 0; color: #b8c2d4; font-size: 13px; }
  button {
    min-height: 34px;
    border: 1px solid #b9c2d0;
    border-radius: 7px;
    padding: 6px 10px;
    color: #0f172a;
    background: #ffffff;
    cursor: pointer;
    font-weight: 600;
  }
  button.primary { border-color: #1d4ed8; color: #ffffff; background: #1d4ed8; }
  button.icon { min-width: 34px; padding: 6px 8px; }
  button:hover { filter: brightness(0.97); }
  .shell {
    display: grid;
    grid-template-columns: 238px minmax(360px, 1fr) minmax(330px, 430px);
    gap: 12px;
    height: calc(100vh - 64px);
    min-height: 620px;
    padding: 12px;
  }
  .panel {
    min-width: 0;
    min-height: 0;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--paper);
    overflow: hidden;
  }
  .panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: 44px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--line);
    background: #f8fafc;
  }
  .panel-head h2 {
    margin: 0;
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .panel-body { padding: 10px; overflow: auto; }
  .palette { display: grid; gap: 8px; }
  .palette button {
    display: grid;
    grid-template-columns: 28px 1fr;
    align-items: center;
    gap: 8px;
    width: 100%;
    text-align: left;
  }
  .workflow {
    display: grid;
    gap: 6px;
    margin-top: 12px;
  }
  .workflow-step {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: 32px;
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 5px 8px;
    background: #fbfdff;
  }
  .workflow-step.active { border-color: var(--accent); background: #eff6ff; }
  .workflow-step b { font-size: 12px; font-weight: 700; }
  .workflow-step span { color: var(--muted); font-size: 11px; }
  .glyph {
    display: inline-grid;
    width: 28px;
    height: 24px;
    place-items: center;
    border: 1px solid var(--line);
    border-radius: 5px;
    color: #334155;
    background: #eef2f7;
    font-weight: 700;
  }
  .stage-panel { display: grid; grid-template-rows: 44px minmax(360px, 1fr) 210px; }
  .stage-wrap { position: relative; min-height: 360px; background: #fbfdff; }
  svg#diagram { width: 100%; height: 100%; display: block; touch-action: none; }
  .block rect { stroke: #64748b; stroke-width: 1.4; }
  .block.source rect { fill: var(--source); }
  .block.transform rect { fill: var(--transform); }
  .block.sink rect { fill: var(--sink); }
  .block.selected rect { stroke: var(--accent); stroke-width: 2.5; }
  .wire { stroke: #94a3b8; stroke-width: 1.7; fill: none; }
  .wire.live { stroke: var(--accent); stroke-width: 2.7; }
  .port { fill: #475569; stroke: #ffffff; stroke-width: 1; }
  .block-title { font-weight: 700; fill: #0f172a; font-size: 12px; }
  .block-op { fill: #475569; font-size: 10px; }
  .block-value { fill: #1d4ed8; font-size: 11px; font-variant-numeric: tabular-nums; }
  .chart { border-top: 1px solid var(--line); padding: 10px; background: #ffffff; }
  svg#chart { width: 100%; height: 150px; display: block; }
  .status {
    min-height: 30px;
    padding: 7px 10px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    background: #f8fafc;
    font-size: 12px;
  }
  .status.ok { color: var(--good); }
  .status.warn { color: var(--warn); }
  textarea {
    width: 100%;
    min-height: 360px;
    resize: vertical;
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 10px;
    color: #0f172a;
    background: #fbfdff;
    font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  }
  textarea.short { min-height: 190px; }
  .inspector { display: grid; gap: 10px; }
  .kv { width: 100%; border-collapse: collapse; font-size: 12px; }
  .kv td { border-bottom: 1px solid #eef2f7; padding: 5px 2px; vertical-align: top; }
  .kv td:first-child { width: 35%; color: var(--muted); }
  .subhead {
    margin: 4px 0 -4px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .n2 { overflow: auto; border: 1px solid var(--line); border-radius: 7px; }
  .n2 table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .n2 th, .n2 td {
    min-width: 28px;
    height: 24px;
    border: 1px solid #e5eaf2;
    text-align: center;
  }
  .n2 th { color: var(--muted); background: #f8fafc; font-weight: 650; }
  .n2 td.self { background: #eef2f7; }
  .n2 td.conn { color: #ffffff; background: var(--accent); }
  .equation-tools {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .equation-chart {
    width: 100%;
    height: 150px;
    display: block;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: #ffffff;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 2px 8px;
    background: #f8fafc;
    color: #334155;
    font-size: 12px;
  }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  @media (max-width: 1080px) {
    .shell { grid-template-columns: 210px minmax(340px, 1fr); height: auto; }
    .right { grid-column: 1 / -1; }
  }
  @media (max-width: 760px) {
    header { align-items: flex-start; flex-direction: column; }
    .shell { grid-template-columns: 1fr; padding: 8px; }
    .stage-panel { grid-template-rows: 44px 420px 210px; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>DES Studio Workbench</h1>
    <p>Flat visual blocks, typed wires, nested runtime cells, and a JSON model contract.</p>
  </div>
  <div class="toolbar">
    <button class="primary" id="runBtn">Run</button>
    <button id="optimizeBtn">Optimize</button>
    <button id="applyBtn">Update From JSON</button>
    <button id="resetBtn">Reset</button>
    <button id="exportBtn">Export JSON</button>
  </div>
</header>
<main class="shell">
  <section class="panel">
    <div class="panel-head"><h2>Block Library</h2></div>
    <div class="panel-body palette">
      <button data-add="source"><span class="glyph">S</span><span>Source</span></button>
      <button data-add="gain"><span class="glyph">K</span><span>Gain</span></button>
      <button data-add="sum"><span class="glyph">+</span><span>Sum</span></button>
      <button data-add="saturation"><span class="glyph">[]</span><span>Saturation</span></button>
      <button data-add="integrator"><span class="glyph">1/s</span><span>Integrator</span></button>
      <button data-add="queue"><span class="glyph">Q</span><span>Queue</span></button>
      <button data-add="delay"><span class="glyph">z</span><span>Delay</span></button>
      <button data-add="sink"><span class="glyph">O</span><span>Scope</span></button>
    </div>
    <div class="panel-body workflow" id="workflow">
      <div class="workflow-step active"><b>Compose</b><span>ready</span></div>
      <div class="workflow-step active"><b>Simulate</b><span>ready</span></div>
      <div class="workflow-step"><b>Simplify</b><span>partial</span></div>
      <div class="workflow-step"><b>Calibrate</b><span>next</span></div>
      <div class="workflow-step"><b>UQ</b><span>next</span></div>
      <div class="workflow-step"><b>Surrogate</b><span>next</span></div>
      <div class="workflow-step active"><b>Control</b><span>ready</span></div>
    </div>
    <div class="status" id="libraryNote">Ready.</div>
  </section>

  <section class="panel stage-panel">
    <div class="panel-head">
      <h2>Diagram</h2>
      <div class="toolbar">
        <button class="icon" id="stepBackBtn" title="Previous frame">&lt;</button>
        <button class="icon" id="stepNextBtn" title="Next frame">&gt;</button>
      </div>
    </div>
    <div class="stage-wrap"><svg id="diagram" viewBox="0 0 920 360" aria-label="Block diagram"></svg></div>
    <div class="chart">
      <svg id="chart" viewBox="0 0 920 150" aria-label="Signal chart"></svg>
      <div class="chips" id="legend"></div>
    </div>
    <div class="status ok" id="status">Ready.</div>
  </section>

  <section class="panel right">
    <div class="panel-head"><h2>Inspector And Spec</h2></div>
    <div class="panel-body inspector">
      <div class="subhead">Selected Block</div>
      <table class="kv" id="inspector"></table>
      <div class="subhead">Design Driver</div>
      <table class="kv" id="designInfo"></table>
      <div class="subhead">Connection Matrix</div>
      <div class="n2" id="n2Matrix"></div>
      <div class="subhead">Equation Spec</div>
      <div class="equation-tools">
        <button class="primary" id="runEquationBtn">Run Eq</button>
        <button id="loadDecayBtn">Decay</button>
      </div>
      <textarea id="equationEditor" class="short" spellcheck="false"></textarea>
      <svg id="equationChart" class="equation-chart" viewBox="0 0 430 150" aria-label="Equation trace"></svg>
      <table class="kv" id="equationInspector"></table>
      <div class="subhead">Block Spec</div>
      <textarea id="specEditor" spellcheck="false"></textarea>
    </div>
  </section>
</main>

<script type="application/json" id="initial-spec">__SPEC__</script>
<script>
(function () {
  "use strict";
  var SVG = "http://www.w3.org/2000/svg";
  var initial = JSON.parse(document.getElementById("initial-spec").textContent);
  var initialEquation = {
    "$schema": "des/equation/v1",
    title: "Equation Spec: Exponential Decay",
    format: "json",
    kind: "ode",
    simulation: { t0: 0, t1: 4, dt: 0.05, method: "trapezoid" },
    constants: { k: 0.7 },
    states: [{ name: "x", initial: 1, derivative: "-k*x" }]
  };
  var spec = clone(initial);
  var equationSpec = clone(initialEquation);
  var selectedId = spec.blocks[0] && spec.blocks[0].id;
  var frames = [];
  var equationFrames = [];
  var frameIndex = 0;
  var seriesKeys = [];
  var equationSeriesKeys = [];
  var designTrace = [];
  var drag = null;
  var MAX_EQUATION_STATES = 16;
  var MAX_EQUATION_STEPS = 5000;
  var MAX_EQUATION_EXPR = 240;
  var EQUATION_FUNCTIONS = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    asin: Math.asin, acos: Math.acos, atan: Math.atan,
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    exp: Math.exp, log: Math.log, ln: Math.log,
    sqrt: Math.sqrt, abs: Math.abs, min: Math.min, max: Math.max,
    pow: Math.pow, floor: Math.floor, ceil: Math.ceil, round: Math.round
  };
  var RESERVED_EQUATION_NAMES = { t: true, tick: true, pi: true, e: true };
  Object.keys(EQUATION_FUNCTIONS).forEach(function (name) { RESERVED_EQUATION_NAMES[name] = true; });

  var diagram = document.getElementById("diagram");
  var chart = document.getElementById("chart");
  var legend = document.getElementById("legend");
  var editor = document.getElementById("specEditor");
  var equationEditor = document.getElementById("equationEditor");
  var equationChart = document.getElementById("equationChart");
  var equationInspector = document.getElementById("equationInspector");
  var status = document.getElementById("status");
  var inspector = document.getElementById("inspector");
  var designInfo = document.getElementById("designInfo");
  var n2Matrix = document.getElementById("n2Matrix");

  document.getElementById("runBtn").addEventListener("click", run);
  document.getElementById("optimizeBtn").addEventListener("click", optimize);
  document.getElementById("applyBtn").addEventListener("click", applyJson);
  document.getElementById("runEquationBtn").addEventListener("click", runEquation);
  document.getElementById("loadDecayBtn").addEventListener("click", function () {
    equationSpec = clone(initialEquation);
    syncEquationEditor();
    runEquation();
  });
  document.getElementById("resetBtn").addEventListener("click", function () {
    spec = clone(initial);
    selectedId = spec.blocks[0] && spec.blocks[0].id;
    syncEditor();
    run();
  });
  document.getElementById("exportBtn").addEventListener("click", function () {
    syncEditor();
    editor.focus();
    editor.select();
    try { document.execCommand("copy"); setStatus("Spec copied to clipboard.", "ok"); }
    catch (e) { setStatus("Spec selected for export.", "ok"); }
  });
  document.getElementById("stepBackBtn").addEventListener("click", function () {
    frameIndex = Math.max(0, frameIndex - 1);
    draw();
  });
  document.getElementById("stepNextBtn").addEventListener("click", function () {
    frameIndex = Math.min(Math.max(0, frames.length - 1), frameIndex + 1);
    draw();
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-add]"), function (btn) {
    btn.addEventListener("click", function () { addBlock(btn.getAttribute("data-add")); });
  });

  syncEditor();
  syncEquationEditor();
  run();
  runEquation();

  function clone(x) { return JSON.parse(JSON.stringify(x)); }
  function fmt(v) {
    if (typeof v !== "number" || !isFinite(v)) return String(v == null ? "" : v);
    if (Math.abs(v) >= 10000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) return v.toExponential(2);
    return (Math.round(v * 1000) / 1000).toString();
  }
  function setStatus(text, kind) {
    status.textContent = text;
    status.className = "status " + (kind || "");
  }
  function syncEditor() { editor.value = JSON.stringify(spec, null, 2); }
  function syncEquationEditor() { equationEditor.value = JSON.stringify(equationSpec, null, 2); }
  function applyJson() {
    try {
      spec = JSON.parse(editor.value);
      if (!Array.isArray(spec.blocks)) throw new Error("spec.blocks must be an array");
      if (!Array.isArray(spec.wires)) spec.wires = [];
      selectedId = selectedId || (spec.blocks[0] && spec.blocks[0].id);
      run();
    } catch (e) {
      setStatus("JSON error: " + e.message, "warn");
    }
  }

  function runEquation() {
    try {
      equationSpec = JSON.parse(equationEditor.value);
      var result = simulateEquation(equationSpec);
      equationFrames = result.frames;
      equationSeriesKeys = result.keys;
      drawEquationChart();
      drawEquationInspector(result);
      setStatus("Equation ran " + result.steps + " steps.", "ok");
    } catch (e) {
      equationFrames = [];
      equationSeriesKeys = [];
      drawEquationChart();
      equationInspector.innerHTML = "";
      setStatus("Equation error: " + e.message, "warn");
    }
  }

  function addBlock(kind) {
    var n = spec.blocks.length + 1;
    var id = kind + n;
    var block = { id: id, label: kind, role: "transform", x: 80 + (n % 4) * 185, y: 70 + Math.floor(n / 4) * 110, cell: [] };
    if (kind === "source") {
      block.role = "source";
      block.cell = [{ op: "source", signal: "const", value: 1.0 }];
    } else if (kind === "gain") {
      block.cell = [{ op: "gain", k: 1.0 }];
    } else if (kind === "sum") {
      block.cell = [{ op: "sum", weights: [1.0, 1.0] }];
    } else if (kind === "saturation") {
      block.cell = [{ op: "saturation", lo: -1.0, hi: 1.0 }];
    } else if (kind === "integrator") {
      block.cell = [{ op: "integrator", initial: 0.0 }];
    } else if (kind === "queue") {
      block.cell = [{ op: "queue", serviceRate: 1.0 }];
    } else if (kind === "delay") {
      block.cell = [{ op: "delay", ticks: 2 }];
    } else {
      block.role = "sink";
      block.label = "scope";
      block.cell = [{ op: "gain", k: 1.0, name: "probe" }];
    }
    spec.blocks.push(block);
    selectedId = id;
    syncEditor();
    draw();
    setStatus("Added `" + id + "`. Connect its inputs in JSON before running.", "warn");
  }

  function run() {
    try {
      var compiled = compile(spec);
      frames = simulate(compiled, spec.simulation || {});
      seriesKeys = (spec.blocks || []).map(function (b) { return b.id; });
      frameIndex = Math.min(frameIndex, Math.max(0, frames.length - 1));
      draw();
      var score = objectiveScore(spec, frames);
      setStatus("Ran " + spec.blocks.length + " blocks for " + frames.length + " steps." + (score == null ? "" : " Objective " + fmt(score) + "."), "ok");
    } catch (e) {
      frames = [];
      draw();
      setStatus("Model error: " + e.message, "warn");
    }
  }

  function simulateEquation(model) {
    if (!model || typeof model !== "object" || Array.isArray(model)) throw new Error("equation spec must be an object");
    if (model.kind && model.kind !== "ode") throw new Error("browser preview currently supports kind=ode");
    if (model.format && model.format !== "json") throw new Error("browser preview currently supports format=json");
    var sim = model.simulation || {};
    if (!sim || typeof sim !== "object" || Array.isArray(sim)) throw new Error("simulation must be an object");
    var states = model.states || [];
    if (!states.length) throw new Error("states must contain at least one ODE state");
    if (states.length > MAX_EQUATION_STATES) throw new Error("states is limited to " + MAX_EQUATION_STATES + " entries in the browser preview");
    var seen = {};
    var names = states.map(function (s, i) {
      if (!s || typeof s !== "object" || Array.isArray(s)) throw new Error("states[" + i + "] must be an object");
      if (!s.name) throw new Error("states[" + i + "].name is required");
      assertEquationName(s.name, "states[" + i + "].name");
      if (seen[s.name]) throw new Error("duplicate state name " + s.name);
      if (RESERVED_EQUATION_NAMES[s.name]) throw new Error("state name " + s.name + " is reserved");
      seen[s.name] = true;
      return s.name;
    });
    var constants = readEquationConstants(model.constants || {}, seen);
    var t0 = readFiniteNumber(sim.t0 == null ? 0 : sim.t0, "simulation.t0");
    var t1 = readFiniteNumber(sim.t1 == null ? 1 : sim.t1, "simulation.t1");
    var dt = readPositiveNumber(sim.dt == null ? 0.1 : sim.dt, "simulation.dt");
    if (!(t1 > t0)) throw new Error("simulation.t1 must be greater than simulation.t0");
    var method = sim.method || "euler";
    if (method !== "euler" && method !== "trapezoid") throw new Error("simulation.method must be euler or trapezoid");
    var y = states.map(function (s, i) { return readFiniteNumber(s.initial == null ? 0 : s.initial, "states[" + i + "].initial"); });
    var funcs = states.map(function (s, i) {
      return compileExpression(s.derivative || s.rhs || "0", "states[" + i + "].derivative");
    });
    var steps = Math.floor((t1 - t0) / dt) + 1;
    if (!isFinite(steps) || steps < 2) throw new Error("simulation horizon produced no integration steps");
    if (steps > MAX_EQUATION_STEPS) throw new Error("browser preview is limited to " + MAX_EQUATION_STEPS + " equation steps");
    var framesOut = [];
    function derivAt(t, yy) {
      var env = Object.create(null);
      Object.keys(constants).forEach(function (k) { env[k] = Number(constants[k]); });
      env.t = t;
      names.forEach(function (name, i) { env[name] = yy[i]; });
      return funcs.map(function (fn, i) {
        var value = Number(fn(env));
        if (!isFinite(value)) throw new Error("derivative for " + names[i] + " produced a non-finite value");
        return value;
      });
    }
    for (var k = 0; k < steps; k++) {
      var t = t0 + k * dt;
      var dy = derivAt(t, y);
      var row = { t: t, tick: k, caption: "t=" + fmt(t) };
      names.forEach(function (name, i) {
        row[name] = y[i];
        row["d_" + name] = dy[i];
      });
      framesOut.push(row);
      if (k < steps - 1) {
        if (method === "trapezoid") {
          var predictor = y.map(function (v, i) { return v + dt * dy[i]; });
          var dy2 = derivAt(t + dt, predictor);
          y = y.map(function (v, i) { return v + 0.5 * dt * (dy[i] + dy2[i]); });
        } else {
          y = y.map(function (v, i) { return v + dt * dy[i]; });
        }
        y.forEach(function (v, i) {
          if (!isFinite(v)) throw new Error("state " + names[i] + " produced a non-finite value");
        });
      }
    }
    return { frames: framesOut, keys: names, steps: steps, finalState: y, method: method };
  }

  function compileExpression(src, label) {
    var ast = parseEquationExpression(src, label);
    return function (env) { return evaluateEquationAst(ast, env, label); };
  }
  function readFiniteNumber(value, label) {
    var n = Number(value);
    if (!isFinite(n)) throw new Error(label + " must be a finite number");
    return n;
  }
  function readPositiveNumber(value, label) {
    var n = readFiniteNumber(value, label);
    if (!(n > 0)) throw new Error(label + " must be positive");
    return n;
  }
  function assertEquationName(name, label) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(label + " must be an identifier");
    if (name === "__proto__" || name === "constructor" || name === "prototype") throw new Error(label + " is reserved");
  }
  function readEquationConstants(raw, stateNames) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("constants must be an object");
    var out = Object.create(null);
    Object.keys(raw).forEach(function (name) {
      assertEquationName(name, "constants." + name);
      if (stateNames[name]) throw new Error("constant " + name + " collides with a state name");
      if (RESERVED_EQUATION_NAMES[name]) throw new Error("constant " + name + " is reserved");
      out[name] = readFiniteNumber(raw[name], "constants." + name);
    });
    return out;
  }
  function parseEquationExpression(src, label) {
    var text = String(src == null ? "0" : src).trim();
    if (!text) throw new Error(label + " must be a non-empty expression");
    if (text.length > MAX_EQUATION_EXPR) throw new Error(label + " is too long for browser preview");
    var tokens = tokenizeEquation(text, label);
    var pos = 0;
    function peek() { return tokens[pos]; }
    function take(type, value) {
      var tok = peek();
      if (tok && tok.type === type && (value == null || tok.value === value)) { pos++; return tok; }
      return null;
    }
    function expect(type, value) {
      var tok = take(type, value);
      if (!tok) throw new Error(label + " has invalid syntax near " + (peek() ? peek().value : "end"));
      return tok;
    }
    function parseAdd() {
      var node = parseMul();
      while (peek() && peek().type === "op" && (peek().value === "+" || peek().value === "-")) {
        var op = take("op").value;
        node = { type: "binary", op: op, left: node, right: parseMul() };
      }
      return node;
    }
    function parseMul() {
      var node = parsePow();
      while (peek() && peek().type === "op" && (peek().value === "*" || peek().value === "/" || peek().value === "%")) {
        var op = take("op").value;
        node = { type: "binary", op: op, left: node, right: parsePow() };
      }
      return node;
    }
    function parsePow() {
      var node = parseUnary();
      if (peek() && peek().type === "op" && peek().value === "^") {
        take("op", "^");
        node = { type: "binary", op: "^", left: node, right: parsePow() };
      }
      return node;
    }
    function parseUnary() {
      if (peek() && peek().type === "op" && (peek().value === "+" || peek().value === "-")) {
        return { type: "unary", op: take("op").value, expr: parseUnary() };
      }
      return parsePrimary();
    }
    function parsePrimary() {
      var tok = peek();
      if (!tok) throw new Error(label + " ended unexpectedly");
      if (take("number")) return { type: "number", value: tok.value };
      if (take("ident")) {
        if (take("paren", "(")) {
          var args = [];
          if (!take("paren", ")")) {
            do { args.push(parseAdd()); } while (take("comma", ","));
            expect("paren", ")");
          }
          if (!EQUATION_FUNCTIONS[tok.value]) throw new Error(label + " uses unknown function " + tok.value);
          return { type: "call", name: tok.value, args: args };
        }
        return { type: "var", name: tok.value };
      }
      if (take("paren", "(")) {
        var inner = parseAdd();
        expect("paren", ")");
        return inner;
      }
      throw new Error(label + " has invalid token " + tok.value);
    }
    var ast = parseAdd();
    if (pos !== tokens.length) throw new Error(label + " has trailing input near " + tokens[pos].value);
    return ast;
  }
  function tokenizeEquation(text, label) {
    var tokens = [];
    var i = 0;
    while (i < text.length) {
      var ch = text.charAt(i);
      if (/\s/.test(ch)) { i++; continue; }
      var number = text.slice(i).match(/^(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?/);
      if (number) {
        tokens.push({ type: "number", value: Number(number[0]) });
        i += number[0].length;
        continue;
      }
      var ident = text.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (ident) {
        tokens.push({ type: "ident", value: ident[0] });
        i += ident[0].length;
        continue;
      }
      if ("+-*/%^".indexOf(ch) >= 0) { tokens.push({ type: "op", value: ch }); i++; continue; }
      if (ch === "(" || ch === ")") { tokens.push({ type: "paren", value: ch }); i++; continue; }
      if (ch === ",") { tokens.push({ type: "comma", value: ch }); i++; continue; }
      throw new Error(label + " contains unsupported character " + ch);
    }
    return tokens;
  }
  function evaluateEquationAst(node, env, label) {
    var value;
    if (node.type === "number") return node.value;
    if (node.type === "var") {
      if (node.name === "pi") return Math.PI;
      if (node.name === "e") return Math.E;
      if (Object.prototype.hasOwnProperty.call(env, node.name)) return env[node.name];
      throw new Error(label + " uses unknown symbol " + node.name);
    }
    if (node.type === "unary") {
      value = evaluateEquationAst(node.expr, env, label);
      return node.op === "-" ? -value : value;
    }
    if (node.type === "binary") {
      var a = evaluateEquationAst(node.left, env, label);
      var b = evaluateEquationAst(node.right, env, label);
      if (node.op === "+") value = a + b;
      else if (node.op === "-") value = a - b;
      else if (node.op === "*") value = a * b;
      else if (node.op === "/") value = a / b;
      else if (node.op === "%") value = a % b;
      else value = Math.pow(a, b);
      if (!isFinite(value)) throw new Error(label + " produced a non-finite value");
      return value;
    }
    if (node.type === "call") {
      var args = node.args.map(function (arg) { return evaluateEquationAst(arg, env, label); });
      value = EQUATION_FUNCTIONS[node.name].apply(null, args);
      if (!isFinite(value)) throw new Error(label + " function " + node.name + " produced a non-finite value");
      return value;
    }
    throw new Error(label + " could not be evaluated");
  }

  function optimize() {
    try {
      var current = JSON.parse(editor.value);
      if (!Array.isArray(current.blocks)) throw new Error("spec.blocks must be an array");
      if (!Array.isArray(current.wires)) current.wires = [];
      var design = current.design || {};
      var vars = design.variables || [];
      var objectives = design.objectives || [];
      if (!vars.length || !objectives.length) throw new Error("design.variables and design.objectives are required");
      var driver = design.driver || {};
      var iterations = Math.max(1, Number(driver.iterations || 24));
      var step = Math.max(0, Number(driver.step || 0.2));
      var eps = Math.max(0.000001, Math.abs(Number(driver.eps || 0.0001)));
      designTrace = [];

      vars.forEach(function (v) {
        if (v.initial != null) writeDesignValue(current, v, clamp(Number(v.initial), lower(v), upper(v)));
      });

      var score = scoreSpec(current);
      designTrace.push(tracePoint(0, score, current, vars));
      for (var iter = 1; iter <= iterations; iter++) {
        var gradients = vars.map(function (v) {
          var x = readDesignValue(current, v);
          var xp = clamp(x + eps, lower(v), upper(v));
          var xm = clamp(x - eps, lower(v), upper(v));
          if (Math.abs(xp - xm) <= 1e-12) return 0;
          var plus = clone(current);
          var minus = clone(current);
          writeDesignValue(plus, v, xp);
          writeDesignValue(minus, v, xm);
          return (scoreSpec(plus) - scoreSpec(minus)) / (xp - xm);
        });
        vars.forEach(function (v, i) {
          var x = readDesignValue(current, v);
          writeDesignValue(current, v, clamp(x - step * gradients[i], lower(v), upper(v)));
        });
        score = scoreSpec(current);
        designTrace.push(tracePoint(iter, score, current, vars));
      }

      spec = current;
      selectedId = selectedId || (spec.blocks[0] && spec.blocks[0].id);
      syncEditor();
      run();
      var first = designTrace[0] && designTrace[0].objective;
      var last = designTrace[designTrace.length - 1] && designTrace[designTrace.length - 1].objective;
      setStatus("Optimized " + vars.length + " variable(s): objective " + fmt(first) + " -> " + fmt(last) + ".", "ok");
    } catch (e) {
      setStatus("Optimize error: " + e.message, "warn");
    }
  }

  function scoreSpec(model) {
    var compiled = compile(clone(model));
    var simFrames = simulate(compiled, model.simulation || {});
    var score = objectiveScore(model, simFrames);
    if (score == null) throw new Error("design objective could not be evaluated");
    return score;
  }

  function objectiveScore(model, simFrames) {
    var objectives = ((model.design || {}).objectives || []);
    if (!objectives.length || !simFrames.length) return null;
    var last = simFrames[simFrames.length - 1];
    return objectives.reduce(function (acc, o) {
      var value = Number(last[o.block] || 0);
      var target = Number(o.target || 0);
      var weight = Number(o.weight == null ? 1 : o.weight);
      var err = value - target;
      return acc + weight * err * err;
    }, 0);
  }

  function tracePoint(iteration, objective, model, vars) {
    var values = {};
    vars.forEach(function (v) { values[v.id || (v.block + "." + v.field)] = readDesignValue(model, v); });
    return { iteration: iteration, objective: objective, variables: values };
  }

  function readDesignValue(model, v) {
    var block = (model.blocks || []).find(function (b) { return b.id === v.block; });
    if (!block) throw new Error("unknown design block " + v.block);
    var op = (block.cell || [])[Number(v.op || 0)];
    if (!op) throw new Error("missing op " + Number(v.op || 0) + " on block " + v.block);
    var value = Number(op[v.field]);
    if (!isFinite(value)) throw new Error("design field " + v.field + " on " + v.block + " is not numeric");
    return value;
  }

  function writeDesignValue(model, v, value) {
    var block = (model.blocks || []).find(function (b) { return b.id === v.block; });
    if (!block) throw new Error("unknown design block " + v.block);
    var op = (block.cell || [])[Number(v.op || 0)];
    if (!op) throw new Error("missing op " + Number(v.op || 0) + " on block " + v.block);
    if (op[v.field] == null) throw new Error("missing design field " + v.field + " on " + v.block);
    op[v.field] = value;
  }

  function lower(v) { return v.lower == null && v.lo == null ? -Infinity : Number(v.lower == null ? v.lo : v.lower); }
  function upper(v) { return v.upper == null && v.hi == null ? Infinity : Number(v.upper == null ? v.hi : v.upper); }
  function clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

  function compile(model) {
    var blocks = model.blocks || [];
    var byId = {};
    blocks.forEach(function (b, i) {
      if (!b.id) throw new Error("blocks[" + i + "] is missing id");
      if (byId[b.id]) throw new Error("duplicate block id " + b.id);
      byId[b.id] = b;
      b._index = i;
      b._w = Math.max(56, Number(b.w || 132));
      b._h = Math.max(44, Number(b.h || 64));
      b._state = makeCellState(b.cell || []);
    });
    (model.wires || []).forEach(function (w, i) {
      if (!byId[w.from]) throw new Error("wires[" + i + "] unknown from block " + w.from);
      if (!byId[w.to]) throw new Error("wires[" + i + "] unknown to block " + w.to);
    });
    validateInputs(blocks, model.wires || []);
    return { blocks: blocks, byId: byId, wires: model.wires || [], order: topo(blocks, model.wires || []) };
  }
  function validateInputs(blocks, wires) {
    var drivers = {};
    wires.forEach(function (w, i) {
      var key = w.to + ":" + Number(w.in || 0);
      if (drivers[key] != null) throw new Error("input " + key + " is driven by more than one wire");
      drivers[key] = i;
    });
    blocks.forEach(function (b) {
      for (var p = 0; p < inputCount(b.cell || []); p++) {
        if (drivers[b.id + ":" + p] == null) throw new Error("block " + b.id + " input " + p + " is not connected");
      }
    });
  }
  function topo(blocks, wires) {
    var indeg = {}, adj = {};
    blocks.forEach(function (b) { indeg[b.id] = 0; adj[b.id] = []; });
    wires.forEach(function (w) { adj[w.from].push(w.to); indeg[w.to] += 1; });
    var q = blocks.filter(function (b) { return indeg[b.id] === 0; }).map(function (b) { return b.id; });
    var order = [];
    while (q.length) {
      var id = q.shift();
      order.push(id);
      adj[id].forEach(function (to) {
        indeg[to] -= 1;
        if (indeg[to] === 0) q.push(to);
      });
    }
    if (order.length !== blocks.length) throw new Error("cycle detected; add a delay or integrator to break feedback");
    return order;
  }
  function makeCellState(cell) {
    return cell.map(function (op) {
      if (op.op === "integrator") return { value: Number(op.initial || 0), lastT: 0 };
      if (op.op === "queue") return { backlog: 0 };
      if (op.op === "delay") return { buf: new Array(Math.max(1, Number(op.ticks || 1))).fill(0) };
      if (op.op === "composite") return { inner: makeCellState(op.cell || []) };
      return {};
    });
  }
  function resetStates(compiled) {
    compiled.blocks.forEach(function (b) { b._state = makeCellState(b.cell || []); });
  }
  function simulate(compiled, sim) {
    resetStates(compiled);
    var steps = Math.max(1, Number(sim.steps || 80));
    var dt = Math.max(0.000001, Number(sim.dt || 0.1));
    var out = [];
    for (var k = 0; k < steps; k++) {
      var t = k * dt;
      var outputs = {};
      compiled.order.forEach(function (id) {
        var b = compiled.byId[id];
        var nIn = inputCount(b.cell || []);
        var inputs = new Array(nIn).fill(0);
        compiled.wires.forEach(function (w) {
          if (w.to === id) inputs[Number(w.in || 0)] = ((outputs[w.from] || [])[Number(w.out || 0)] || 0);
        });
        outputs[id] = stepCell(b.cell || [], b._state, t, inputs);
      });
      var row = { t: t, outputs: outputs };
      compiled.blocks.forEach(function (b) {
        var val = primaryValue(b, compiled, outputs);
        row[b.id] = val;
      });
      out.push(row);
    }
    return out;
  }
  function primaryValue(b, compiled, outputs) {
    if (outputs[b.id] && outputs[b.id].length) return outputs[b.id][0];
    var incoming = compiled.wires.find(function (w) { return w.to === b.id; });
    if (!incoming) return 0;
    return ((outputs[incoming.from] || [])[Number(incoming.out || 0)] || 0);
  }
  function inputCount(cell) { return cell.length ? opInputCount(cell[0]) : 0; }
  function outputCount(cell) { return cell.length ? opOutputCount(cell[cell.length - 1]) : 0; }
  function opInputCount(op) {
    if (op.op === "source") return 0;
    if (op.op === "sum") return (op.weights || []).length;
    if (op.op === "composite") return inputCount(op.cell || []);
    return 1;
  }
  function opOutputCount(op) {
    if (op.op === "composite") return outputCount(op.cell || []);
    return 1;
  }
  function stepCell(cell, states, t, inputs) {
    var signal = inputs.slice();
    cell.forEach(function (op, i) { signal = stepOp(op, states[i] || {}, t, signal); });
    return signal;
  }
  function stepOp(op, st, t, inputs) {
    var x = Number(inputs[0] || 0);
    if (op.op === "source") {
      if (op.signal === "step") return [t >= Number(op.t0 || 0) ? Number(op.after || 1) : Number(op.before || 0)];
      if (op.signal === "ramp") return [Number(op.slope || 1) * t + Number(op.intercept || 0)];
      if (op.signal === "sine") return [Number(op.amp || 1) * Math.sin(2 * Math.PI * Number(op.freq || 1) * t) + Number(op.bias || 0)];
      return [Number(op.value || 0)];
    }
    if (op.op === "gain") return [Number(op.k || 0) * x];
    if (op.op === "saturation") return [Math.min(Math.max(x, Number(op.lo || 0)), Number(op.hi || 0))];
    if (op.op === "affine") return [Number(op.m == null ? 1 : op.m) * x + Number(op.b || 0)];
    if (op.op === "sum") return [(op.weights || []).reduce(function (acc, w, i) { return acc + Number(w || 0) * Number(inputs[i] || 0); }, 0)];
    if (op.op === "integrator") {
      var d = Math.max(0, t - Number(st.lastT || 0));
      st.value = Number(st.value || 0) + d * x;
      st.lastT = t;
      return [st.value];
    }
    if (op.op === "queue") {
      st.backlog = Math.max(0, Number(st.backlog || 0) + Math.max(0, x));
      var served = Math.min(st.backlog, Math.max(0, Number(op.serviceRate || 0)));
      st.backlog -= served;
      return [served];
    }
    if (op.op === "delay") {
      st.buf = st.buf || new Array(Math.max(1, Number(op.ticks || 1))).fill(0);
      st.buf.push(x);
      return [st.buf.shift() || 0];
    }
    if (op.op === "composite") return stepCell(op.cell || [], st.inner || [], t, inputs);
    return [x];
  }

  function draw() {
    drawDiagram();
    drawChart();
    drawInspector();
    drawDesignPanel();
    drawMatrix();
  }
  function currentFrame() { return frames[Math.max(0, Math.min(frameIndex, frames.length - 1))] || { outputs: {} }; }
  function blockValue(id) {
    var f = currentFrame();
    return typeof f[id] === "number" ? f[id] : 0;
  }
  function drawDiagram() {
    diagram.innerHTML = "";
    var frame = currentFrame();
    (spec.wires || []).forEach(function (w) {
      var src = findBlock(w.from), dst = findBlock(w.to);
      if (!src || !dst) return;
      var a = portXY(src, Number(w.out || 0), false);
      var b = portXY(dst, Number(w.in || 0), true);
      var live = Math.abs((((frame.outputs || {})[w.from] || [])[Number(w.out || 0)] || 0)) > 1e-9;
      diagram.appendChild(svg("path", { d: "M " + a.x + " " + a.y + " C " + (a.x + 70) + " " + a.y + ", " + (b.x - 70) + " " + b.y + ", " + b.x + " " + b.y, class: "wire" + (live ? " live" : "") }));
    });
    (spec.blocks || []).forEach(function (b) {
      var g = svg("g", { class: "block " + (b.role || "transform") + (selectedId === b.id ? " selected" : ""), "data-id": b.id });
      g.appendChild(svg("rect", { x: Number(b.x || 0), y: Number(b.y || 0), width: Number(b.w || 132), height: Number(b.h || 64), rx: 8 }));
      g.appendChild(text(Number(b.x || 0) + Number(b.w || 132) / 2, Number(b.y || 0) + 17, b.label || b.id, "block-title"));
      g.appendChild(text(Number(b.x || 0) + Number(b.w || 132) / 2, Number(b.y || 0) + 34, opNames(b.cell || []).join(" -> "), "block-op"));
      g.appendChild(text(Number(b.x || 0) + Number(b.w || 132) / 2, Number(b.y || 0) + Number(b.h || 64) - 9, fmt(blockValue(b.id)), "block-value"));
      for (var i = 0; i < inputCount(b.cell || []); i++) {
        var pi = portXY(b, i, true);
        g.appendChild(svg("circle", { cx: pi.x, cy: pi.y, r: 4, class: "port" }));
      }
      for (var o = 0; o < outputCount(b.cell || []); o++) {
        var po = portXY(b, o, false);
        g.appendChild(svg("circle", { cx: po.x, cy: po.y, r: 4, class: "port" }));
      }
      g.addEventListener("pointerdown", startDrag);
      diagram.appendChild(g);
    });
  }
  function findBlock(id) { return (spec.blocks || []).find(function (b) { return b.id === id; }); }
  function portXY(b, port, input) {
    var n = input ? inputCount(b.cell || []) : outputCount(b.cell || []);
    var x = Number(b.x || 0) + (input ? 0 : Number(b.w || 132));
    var y = Number(b.y || 0) + Number(b.h || 64) * (port + 1) / (n + 1);
    return { x: x, y: y };
  }
  function opNames(cell) { return cell.map(function (op) { return op.name || op.op; }); }
  function startDrag(e) {
    var id = this.getAttribute("data-id");
    var b = findBlock(id);
    if (!b) return;
    selectedId = id;
    var pt = pointerPoint(e);
    drag = { id: id, dx: pt.x - Number(b.x || 0), dy: pt.y - Number(b.y || 0) };
    diagram.setPointerCapture(e.pointerId);
    diagram.addEventListener("pointermove", moveDrag);
    diagram.addEventListener("pointerup", endDrag);
    draw();
  }
  function moveDrag(e) {
    if (!drag) return;
    var b = findBlock(drag.id), pt = pointerPoint(e);
    b.x = Math.max(0, Math.min(820, pt.x - drag.dx));
    b.y = Math.max(0, Math.min(290, pt.y - drag.dy));
    drawDiagram();
  }
  function endDrag() {
    drag = null;
    diagram.removeEventListener("pointermove", moveDrag);
    diagram.removeEventListener("pointerup", endDrag);
    syncEditor();
    drawInspector();
  }
  function pointerPoint(e) {
    var ctm = diagram.getScreenCTM();
    var p = diagram.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(ctm.inverse());
  }
  function drawChart() {
    chart.innerHTML = "";
    legend.innerHTML = "";
    if (!frames.length) return;
    var W = 920, H = 150, L = 44, R = 12, T = 10, B = 22;
    var ymin = Infinity, ymax = -Infinity;
    frames.forEach(function (f) { seriesKeys.forEach(function (k) {
      var v = f[k];
      if (typeof v === "number" && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    }); });
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymin === ymax) ymax = ymin + 1;
    function sx(i) { return L + i / Math.max(1, frames.length - 1) * (W - L - R); }
    function sy(v) { return H - B - (v - ymin) / (ymax - ymin) * (H - T - B); }
    chart.appendChild(svg("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#cbd5e1" }));
    chart.appendChild(svg("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#cbd5e1" }));
    seriesKeys.forEach(function (key, si) {
      var pts = frames.map(function (f, i) { return sx(i) + "," + sy(Number(f[key] || 0)); }).join(" ");
      chart.appendChild(svg("polyline", { points: pts, fill: "none", stroke: palette(si), "stroke-width": 1.7 }));
      var chip = document.createElement("span");
      chip.className = "chip";
      chip.style.borderColor = palette(si);
      chip.textContent = key;
      legend.appendChild(chip);
    });
    chart.appendChild(svg("line", { x1: sx(frameIndex), y1: T, x2: sx(frameIndex), y2: H - B, stroke: "#ef4444", "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
  }
  function drawEquationChart() {
    equationChart.innerHTML = "";
    if (!equationFrames.length) return;
    var W = 430, H = 150, L = 42, R = 12, T = 12, B = 24;
    var ymin = Infinity, ymax = -Infinity;
    equationFrames.forEach(function (f) { equationSeriesKeys.forEach(function (k) {
      var v = f[k];
      if (typeof v === "number" && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    }); });
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymin === ymax) ymax = ymin + 1;
    function sx(i) { return L + i / Math.max(1, equationFrames.length - 1) * (W - L - R); }
    function sy(v) { return H - B - (v - ymin) / (ymax - ymin) * (H - T - B); }
    equationChart.appendChild(svg("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#cbd5e1" }));
    equationChart.appendChild(svg("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#cbd5e1" }));
    equationSeriesKeys.forEach(function (key, si) {
      var pts = equationFrames.map(function (f, i) { return sx(i) + "," + sy(Number(f[key] || 0)); }).join(" ");
      equationChart.appendChild(svg("polyline", { points: pts, fill: "none", stroke: palette(si), "stroke-width": 1.8 }));
      var label = text(W - R - 48, T + 15 + 14 * si, key, "block-op");
      label.setAttribute("text-anchor", "start");
      equationChart.appendChild(label);
    });
  }
  function drawEquationInspector(result) {
    equationInspector.innerHTML = "";
    var last = equationFrames[equationFrames.length - 1] || {};
    var rows = [["method", result.method], ["steps", result.steps]];
    equationSeriesKeys.forEach(function (key) { rows.push([key + "(final)", fmt(last[key])]); });
    rows.forEach(function (row) { addKv(equationInspector, row[0], row[1]); });
  }
  function palette(i) {
    return ["#1d4ed8", "#15803d", "#b91c1c", "#0f766e", "#a16207", "#be185d", "#475569"][i % 7];
  }
  function drawInspector() {
    var b = findBlock(selectedId);
    inspector.innerHTML = "";
    if (!b) return;
    [
      ["id", b.id],
      ["role", b.role || "transform"],
      ["ops", opNames(b.cell || []).join(" -> ")],
      ["inputs", inputCount(b.cell || [])],
      ["outputs", outputCount(b.cell || [])],
      ["value", fmt(blockValue(b.id))]
    ].forEach(function (row) {
      var tr = document.createElement("tr");
      var a = document.createElement("td");
      var c = document.createElement("td");
      a.textContent = row[0];
      c.textContent = row[1];
      tr.appendChild(a);
      tr.appendChild(c);
      inspector.appendChild(tr);
    });
  }
  function drawDesignPanel() {
    designInfo.innerHTML = "";
    var design = spec.design || {};
    var vars = design.variables || [];
    var objectives = design.objectives || [];
    if (!vars.length && !objectives.length) {
      addKv(designInfo, "status", "No design study declared.");
      return;
    }
    vars.forEach(function (v) {
      var value;
      try { value = fmt(readDesignValue(spec, v)); }
      catch (e) { value = e.message; }
      addKv(designInfo, v.id || "variable", value + " [" + fmt(lower(v)) + ", " + fmt(upper(v)) + "]");
    });
    objectives.forEach(function (o) {
      var last = frames[frames.length - 1] || {};
      var value = Number(last[o.block] || 0);
      addKv(designInfo, o.id || "objective", o.block + " " + fmt(value) + " -> " + fmt(Number(o.target || 0)));
    });
    var score = objectiveScore(spec, frames);
    if (score != null) addKv(designInfo, "score", fmt(score));
    if (designTrace.length) {
      var first = designTrace[0].objective;
      var lastTrace = designTrace[designTrace.length - 1].objective;
      addKv(designInfo, "trace", fmt(first) + " -> " + fmt(lastTrace) + " over " + (designTrace.length - 1) + " iterations");
    }
  }
  function drawMatrix() {
    n2Matrix.innerHTML = "";
    var blocks = spec.blocks || [];
    if (!blocks.length) return;
    var table = document.createElement("table");
    var head = document.createElement("tr");
    head.appendChild(document.createElement("th"));
    blocks.forEach(function (b) {
      var th = document.createElement("th");
      th.textContent = shortId(b.id);
      th.title = b.id;
      head.appendChild(th);
    });
    table.appendChild(head);
    blocks.forEach(function (row) {
      var tr = document.createElement("tr");
      var th = document.createElement("th");
      th.textContent = shortId(row.id);
      th.title = row.id;
      tr.appendChild(th);
      blocks.forEach(function (col) {
        var td = document.createElement("td");
        var hit = (spec.wires || []).some(function (w) { return w.from === col.id && w.to === row.id; });
        td.className = row.id === col.id ? "self" : (hit ? "conn" : "");
        td.textContent = hit ? "x" : "";
        td.title = hit ? col.id + " -> " + row.id : "";
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    n2Matrix.appendChild(table);
  }
  function addKv(table, key, value) {
    var tr = document.createElement("tr");
    var a = document.createElement("td");
    var c = document.createElement("td");
    a.textContent = key;
    c.textContent = value;
    tr.appendChild(a);
    tr.appendChild(c);
    table.appendChild(tr);
  }
  function shortId(id) { return String(id || "").slice(0, 4); }
  function svg(tag, attrs) {
    var el = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function text(x, y, value, klass) {
    var el = svg("text", { x: x, y: y, class: klass, "text-anchor": "middle" });
    el.textContent = value;
    return el;
  }
})();
</script>
</body>
</html>
"####;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workbench_embeds_initial_spec_and_runner() {
        let html = workbench_html();
        assert!(html.contains("DES Studio Workbench"));
        assert!(html.contains("id=\"initial-spec\""));
        assert!(html.contains("\"$schema\": \"des/studio/v1\""));
        assert!(html.contains("function simulate"));
        assert!(html.contains("id=\"equationEditor\""));
        assert!(html.contains("function simulateEquation"));
        assert!(html.contains("function parseEquationExpression"));
        assert!(!html.contains("new Function"));
        assert!(html.contains("id=\"optimizeBtn\""));
        assert!(html.contains("function optimize"));
        assert!(html.contains("id=\"n2Matrix\""));
    }
}
