//! Self-contained Studio workbench HTML.
//!
//! The generated page is intentionally framework-free: embedders can write one
//! file, serve it statically, and still get block authoring, inspection, N2
//! structure, local simulation, and a small driver sweep.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::Value;

use super::analysis::analyze_model_spec;
use super::spec::{starter_model_spec, studio_palette, StudioModelSpec};
use super::sweep::run_first_design_sweep;

fn script_json(value: &Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|_| "null".to_string())
        .replace("</script", "<\\/script")
}

/// Render the Studio workbench as a single HTML document.
pub fn render_workbench_html(spec: &StudioModelSpec) -> Result<String, serde_json::Error> {
    let palette = serde_json::to_value(studio_palette())?;
    let model = serde_json::to_value(spec)?;
    let analysis = serde_json::to_value(analyze_model_spec(spec))?;
    let sweep = run_first_design_sweep(spec)
        .ok()
        .flatten()
        .and_then(|s| serde_json::to_value(s).ok())
        .unwrap_or(Value::Null);

    Ok(WORKBENCH_HTML
        .replace("__PALETTE_JSON__", &script_json(&palette))
        .replace("__MODEL_JSON__", &script_json(&model))
        .replace("__ANALYSIS_JSON__", &script_json(&analysis))
        .replace("__SWEEP_JSON__", &script_json(&sweep)))
}

/// Starter workbench used by binaries and site generation.
pub fn render_starter_workbench_html() -> Result<String, serde_json::Error> {
    render_workbench_html(&starter_model_spec())
}

/// Write the workbench to disk, creating parent directories as needed.
pub fn write_workbench_html(path: impl AsRef<Path>, spec: &StudioModelSpec) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let html = render_workbench_html(spec).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, html)
}

const WORKBENCH_HTML: &str = r###"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DES Modeling Studio</title>
<style>
:root {
  color-scheme: light;
  --ink: #16202a;
  --muted: #5b6b7c;
  --line: #cad4df;
  --panel: #f8fafc;
  --canvas: #ffffff;
  --blue: #1f6feb;
  --green: #17803d;
  --amber: #a86403;
  --red: #c2413b;
  --violet: #6d5bd0;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font: 14px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #e9eef4;
}
button, input, textarea, select {
  font: inherit;
}
button {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 7px 9px;
  cursor: pointer;
}
button:hover { border-color: var(--blue); }
button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
button.icon {
  width: 32px;
  height: 32px;
  padding: 0;
  display: inline-grid;
  place-items: center;
}
.shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}
.topbar {
  min-height: 48px;
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto auto;
  gap: 12px;
  align-items: center;
  padding: 8px 12px;
  background: #fbfdff;
  border-bottom: 1px solid var(--line);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.mark {
  width: 24px;
  height: 24px;
  border: 2px solid var(--blue);
  border-radius: 6px;
  position: relative;
}
.mark:before, .mark:after {
  content: "";
  position: absolute;
  background: var(--blue);
}
.mark:before { left: 4px; right: 4px; top: 10px; height: 2px; }
.mark:after { top: 4px; bottom: 4px; left: 10px; width: 2px; }
.brand strong { display: block; font-size: 14px; }
.brand span { display: block; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tabs { display: inline-flex; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #fff; }
.tabs button {
  border: 0;
  border-radius: 0;
  border-right: 1px solid var(--line);
  min-width: 74px;
}
.tabs button:last-child { border-right: 0; }
.tabs button.active { background: #e8f0ff; color: #0f4fba; }
.main {
  min-height: 0;
  display: grid;
  grid-template-columns: 246px minmax(420px, 1fr) 320px;
}
.left, .right {
  min-height: 0;
  overflow: auto;
  background: var(--panel);
}
.left { border-right: 1px solid var(--line); }
.right { border-left: 1px solid var(--line); }
.panel-head {
  min-height: 42px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
}
.panel-head h2 {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0;
}
.palette { padding: 10px; display: grid; gap: 12px; }
.category { display: grid; gap: 6px; }
.category-title {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.palette-item {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) 28px;
  gap: 8px;
  align-items: center;
  width: 100%;
  text-align: left;
}
.glyph {
  width: 24px;
  height: 24px;
  border: 1px solid var(--line);
  border-radius: 5px;
  background: #fff;
  position: relative;
}
.glyph.source { border-color: #75b984; background: #eef9f0; }
.glyph.math { border-color: #7ca7e8; background: #eff5ff; }
.glyph.events { border-color: #d59a37; background: #fff8ea; }
.glyph.sink { border-color: #df8d88; background: #fff0ef; }
.glyph:before { content: ""; position: absolute; inset: 6px; border: 2px solid currentColor; border-radius: 3px; color: #607080; }
.palette-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.palette-meta { color: var(--muted); font-size: 11px; }
.workspace {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) 202px;
  background: var(--canvas);
}
.stage-wrap, .view {
  grid-row: 1;
  grid-column: 1;
}
.stage-wrap { min-height: 0; overflow: auto; position: relative; }
.stage {
  min-width: 780px;
  min-height: 420px;
  width: 100%;
  height: 100%;
  display: block;
  background:
    linear-gradient(#edf2f7 1px, transparent 1px),
    linear-gradient(90deg, #edf2f7 1px, transparent 1px);
  background-size: 24px 24px;
}
.block rect { filter: drop-shadow(0 1px 1px rgba(22, 32, 42, .14)); }
.block.selected rect.main-rect { stroke: var(--blue); stroke-width: 2.5; }
.wire { stroke: #7891aa; stroke-width: 2; fill: none; }
.wire.hot { stroke: var(--blue); stroke-width: 3; }
.port { fill: #516273; }
.bottom {
  border-top: 1px solid var(--line);
  background: #fbfdff;
  min-height: 0;
  overflow: auto;
  padding: 10px 12px;
}
.results-grid {
  display: grid;
  grid-template-columns: minmax(180px, 260px) minmax(320px, 1fr);
  gap: 12px;
  align-items: start;
}
table {
  border-collapse: collapse;
  width: 100%;
  background: #fff;
}
th, td {
  border: 1px solid var(--line);
  padding: 6px 7px;
  text-align: left;
  vertical-align: middle;
}
th { background: #f1f5f9; font-size: 12px; }
.metric { display: grid; gap: 8px; }
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--green);
}
.dot.bad { background: var(--red); }
.form { padding: 12px; display: grid; gap: 12px; }
.field { display: grid; gap: 5px; }
.field label { color: var(--muted); font-size: 12px; }
.field input, .field textarea, .field select {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
  color: var(--ink);
}
.field textarea {
  min-height: 170px;
  resize: vertical;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
.actions { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 7px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #fff;
  color: var(--muted);
  font-size: 12px;
}
.view { min-height: 0; overflow: auto; padding: 14px; background: #fff; }
.hidden { display: none !important; }
.n2-grid {
  display: grid;
  gap: 0;
  width: max-content;
  min-width: 520px;
}
.n2-cell {
  width: 92px;
  height: 54px;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  display: grid;
  place-items: center;
  font-size: 11px;
  background: #fff;
  overflow: hidden;
  text-align: center;
  padding: 4px;
}
.n2-cell.header { background: #f1f5f9; font-weight: 700; }
.n2-cell.diag { background: #eef7f1; color: #145c2d; }
.n2-cell.link { background: #eaf2ff; color: #0f4fba; font-weight: 700; }
.driver-layout {
  display: grid;
  grid-template-columns: minmax(260px, 360px) minmax(360px, 1fr);
  gap: 14px;
}
.chart {
  width: 100%;
  min-height: 180px;
  border: 1px solid var(--line);
  background: #fff;
}
.json-actions { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
.error { color: var(--red); }
@media (max-width: 980px) {
  .main { grid-template-columns: 1fr; }
  .left, .right { max-height: 260px; border: 0; border-bottom: 1px solid var(--line); }
  .workspace { min-height: 620px; }
  .topbar { grid-template-columns: 1fr; }
  .driver-layout, .results-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="mark" aria-hidden="true"></div>
      <div><strong>DES Modeling Studio</strong><span id="model-subtitle"></span></div>
    </div>
    <nav class="tabs" id="view-tabs" aria-label="Workbench views"></nav>
    <div class="actions" id="top-actions"></div>
  </header>
  <main class="main">
    <aside class="left">
      <div class="panel-head"><h2>Palette</h2><span class="pill" id="palette-count"></span></div>
      <div class="palette" id="palette"></div>
    </aside>
    <section class="workspace">
      <div id="canvas-view" class="stage-wrap">
        <svg class="stage" id="stage" viewBox="0 0 900 460" role="img" aria-label="Studio model canvas"></svg>
      </div>
      <div id="n2-view" class="view hidden"></div>
      <div id="driver-view" class="view hidden"></div>
      <div id="json-view" class="view hidden"></div>
      <div class="bottom" id="bottom"></div>
    </section>
    <aside class="right">
      <div class="panel-head"><h2>Inspector</h2><span class="status" id="validation"></span></div>
      <div class="form" id="inspector"></div>
    </aside>
  </main>
</div>
<script>
const PALETTE = __PALETTE_JSON__;
const INITIAL_MODEL = __MODEL_JSON__;
const INITIAL_ANALYSIS = __ANALYSIS_JSON__;
const INITIAL_SWEEP = __SWEEP_JSON__;

let model = clone(INITIAL_MODEL);
let analysis = clone(INITIAL_ANALYSIS);
let selectedId = model.blocks[0] ? model.blocks[0].id : "";
let activeView = "canvas";
let lastRun = null;
let lastSweep = INITIAL_SWEEP && INITIAL_SWEEP.cases ? INITIAL_SWEEP : null;

const byKind = new Map(PALETTE.map(item => [item.kind, item]));
const stage = document.getElementById("stage");
const bottom = document.getElementById("bottom");

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}
function el(name, attrs = {}, children = []) {
  const node = document.createElement(name);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}
function svg(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}
function fmt(n) {
  return Number.isFinite(n) ? Number(n).toFixed(Math.abs(n) >= 10 ? 2 : 3) : "n/a";
}
function blockById(id) {
  return model.blocks.find(block => block.id === id);
}
function blockLabel(block) {
  return block.label || block.id;
}
function itemFor(block) {
  return byKind.get(block.kind) || PALETTE[0];
}
function blockIo(block) {
  const item = itemFor(block);
  let inputs = item ? item.inputs : 1;
  let outputs = item ? item.outputs : 1;
  if (block.kind === "sum") {
    const weights = block.params && Array.isArray(block.params.weights) ? block.params.weights : [1, 1];
    inputs = Math.max(1, weights.length);
  }
  return { inputs, outputs };
}
function defaultParams(item) {
  const params = {};
  for (const param of item.params || []) params[param.name] = clone(param.defaultValue);
  return params;
}
function sortedGroups() {
  const groups = new Map();
  for (const item of PALETTE) {
    if (!groups.has(item.category)) groups.set(item.category, []);
    groups.get(item.category).push(item);
  }
  return [...groups.entries()];
}
function glyphClass(category) {
  if (category === "Sources") return "source";
  if (category === "Math" || category === "Continuous/Discrete") return "math";
  if (category === "Discrete Events") return "events";
  return "sink";
}
function nextId(kind) {
  const stem = kind.replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "block";
  let i = 1;
  while (blockById(`${stem}_${i}`)) i += 1;
  return `${stem}_${i}`;
}
function addBlock(kind) {
  const item = byKind.get(kind);
  if (!item) return;
  const id = nextId(kind);
  model.blocks.push({
    id,
    kind,
    label: item.label,
    params: defaultParams(item),
    x: 80 + (model.blocks.length % 4) * 170,
    y: 70 + Math.floor(model.blocks.length / 4) * 110
  });
  selectedId = id;
  refresh();
}
function selectedBlock() {
  return blockById(selectedId) || model.blocks[0] || null;
}
function portXY(block, port, count, side) {
  const x = Number(block.x || 0);
  const y = Number(block.y || 0);
  const w = 132;
  const h = 64;
  return {
    x: side === "out" ? x + w : x,
    y: y + h * (port + 1) / (count + 1)
  };
}
function analyzeLocal() {
  const warnings = [];
  const ids = new Map(model.blocks.map((block, idx) => [block.id, idx]));
  const connections = model.wires.map(wire => ({
    from: wire.from,
    fromPort: wire.fromPort || 0,
    to: wire.to,
    toPort: wire.toPort || 0
  }));
  for (const wire of connections) {
    const src = blockById(wire.from);
    const dst = blockById(wire.to);
    if (!src || !dst) {
      warnings.push(`Unknown block in wire ${wire.from} -> ${wire.to}`);
      continue;
    }
    const srcIo = blockIo(src);
    const dstIo = blockIo(dst);
    if (wire.fromPort >= srcIo.outputs) warnings.push(`${wire.from} has no output ${wire.fromPort}`);
    if (wire.toPort >= dstIo.inputs) warnings.push(`${wire.to} has no input ${wire.toPort}`);
  }
  let validation = { ok: warnings.length === 0, message: warnings[0] || null, executionOrder: [], executive: "studio" };
  try {
    const compiled = compileModel();
    validation.executionOrder = compiled.order.map(idx => model.blocks[idx].id);
  } catch (err) {
    validation = { ok: false, message: err.message, executionOrder: [], executive: null };
  }
  return {
    name: model.name,
    components: model.blocks.map(block => {
      const io = blockIo(block);
      return {
        id: block.id,
        label: blockLabel(block),
        kind: block.kind,
        role: itemFor(block).category === "Sources" ? "source" : itemFor(block).category === "Sinks" ? "sink" : "transform",
        inputs: io.inputs,
        outputs: io.outputs,
        stateful: !!itemFor(block).stateful,
        elements: [block.kind],
        x: block.x,
        y: block.y
      };
    }),
    connections,
    n2: connections.map(conn => ({ row: ids.get(conn.to), col: ids.get(conn.from), connections: [conn] }))
      .filter(cell => Number.isInteger(cell.row) && Number.isInteger(cell.col)),
    designVariables: model.designVariables || [],
    objectives: model.objectives || [],
    constraints: model.constraints || [],
    validation,
    warnings
  };
}
function renderPalette() {
  const root = document.getElementById("palette");
  root.replaceChildren();
  document.getElementById("palette-count").textContent = `${PALETTE.length} blocks`;
  for (const [category, items] of sortedGroups()) {
    const group = el("div", { class: "category" }, [el("div", { class: "category-title", text: category })]);
    for (const item of items) {
      const button = el("button", { type: "button", class: "palette-item", onclick: () => addBlock(item.kind) }, [
        el("span", { class: `glyph ${glyphClass(category)}`, "aria-hidden": "true" }),
        el("span", {}, [
          el("span", { class: "palette-name", text: item.label }),
          el("span", { class: "palette-meta", text: `${item.inputs} in / ${item.outputs} out` })
        ]),
        el("span", { class: "pill", text: "+" })
      ]);
      group.append(button);
    }
    root.append(group);
  }
}
function renderTopbarControls() {
  const tabs = document.getElementById("view-tabs");
  tabs.replaceChildren();
  for (const [view, label] of [["canvas", "Canvas"], ["n2", "N2"], ["driver", "Driver"], ["json", "JSON"]]) {
    tabs.append(el("button", {
      type: "button",
      "data-view": view,
      class: view === activeView ? "active" : "",
      text: label,
      onclick: () => showView(view)
    }));
  }
  const actions = document.getElementById("top-actions");
  actions.replaceChildren(
    el("button", { type: "button", class: "primary", text: "Run", onclick: runAndRender }),
    el("button", { type: "button", text: "Sweep", onclick: sweepAndRender })
  );
}
function renderCanvas() {
  stage.replaceChildren();
  const values = lastRun ? lastRun.finalSignals : {};
  for (const wire of model.wires) {
    const src = blockById(wire.from);
    const dst = blockById(wire.to);
    if (!src || !dst) continue;
    const srcIo = blockIo(src);
    const dstIo = blockIo(dst);
    const p1 = portXY(src, wire.fromPort || 0, srcIo.outputs || 1, "out");
    const p2 = portXY(dst, wire.toPort || 0, dstIo.inputs || 1, "in");
    const line = svg("path", {
      d: `M ${p1.x} ${p1.y} C ${p1.x + 55} ${p1.y}, ${p2.x - 55} ${p2.y}, ${p2.x} ${p2.y}`,
      class: `wire ${Math.abs(values[wire.from] || 0) > 1e-9 ? "hot" : ""}`
    });
    stage.append(line);
  }
  for (const block of model.blocks) {
    const g = svg("g", { class: `block ${block.id === selectedId ? "selected" : ""}`, tabindex: "0" });
    g.addEventListener("click", () => { selectedId = block.id; renderCanvas(); renderInspector(); });
    const x = Number(block.x || 0);
    const y = Number(block.y || 0);
    const io = blockIo(block);
    const item = itemFor(block);
    const fill = item.category === "Sources" ? "#eef9f0" : item.category === "Sinks" ? "#fff0ef" : item.category === "Discrete Events" ? "#fff8ea" : "#eff5ff";
    g.append(svg("rect", { x, y, width: 132, height: 64, rx: 7, fill, stroke: "#64748b", "stroke-width": 1.4, class: "main-rect" }));
    g.append(svg("rect", { x, y, width: 132, height: 8, rx: 7, fill: item.category === "Sources" ? "#17803d" : item.category === "Sinks" ? "#c2413b" : item.category === "Discrete Events" ? "#a86403" : "#1f6feb" }));
    const title = svg("text", { x: x + 66, y: y + 26, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "#16202a" });
    title.textContent = blockLabel(block);
    g.append(title);
    const subtitle = svg("text", { x: x + 66, y: y + 42, "text-anchor": "middle", "font-size": 10, fill: "#5b6b7c" });
    subtitle.textContent = block.kind;
    g.append(subtitle);
    const live = svg("text", { x: x + 66, y: y + 57, "text-anchor": "middle", "font-size": 10, fill: "#0f4fba" });
    live.textContent = values[block.id] === undefined ? "" : fmt(values[block.id]);
    g.append(live);
    for (let p = 0; p < io.inputs; p++) {
      const pt = portXY(block, p, io.inputs, "in");
      g.append(svg("circle", { cx: pt.x, cy: pt.y, r: 3.6, class: "port" }));
    }
    for (let p = 0; p < io.outputs; p++) {
      const pt = portXY(block, p, io.outputs, "out");
      g.append(svg("circle", { cx: pt.x, cy: pt.y, r: 3.6, class: "port" }));
    }
    stage.append(g);
  }
}
function renderInspector() {
  const root = document.getElementById("inspector");
  root.replaceChildren();
  const block = selectedBlock();
  const status = document.getElementById("validation");
  status.replaceChildren(el("span", { class: analysis.validation.ok ? "dot" : "dot bad" }), document.createTextNode(analysis.validation.ok ? "Valid" : "Needs wiring"));
  if (!block) {
    root.append(el("div", { class: "error", text: "No block selected" }));
    return;
  }
  const labelInput = el("input", { value: block.label || block.id });
  labelInput.addEventListener("input", () => { block.label = labelInput.value; renderCanvas(); });
  root.append(el("div", { class: "field" }, [el("label", { text: "Label" }), labelInput]));
  root.append(el("div", { class: "pill", text: `${block.kind} / ${itemFor(block).category}` }));
  for (const param of itemFor(block).params || []) {
    const value = block.params && block.params[param.name] !== undefined ? block.params[param.name] : clone(param.defaultValue);
    const input = el("input", { value: Array.isArray(value) ? value.join(", ") : value, type: "text" });
    input.addEventListener("change", () => {
      if (!block.params) block.params = {};
      if (param.kind === "number-array") {
        block.params[param.name] = input.value.split(",").map(v => Number(v.trim())).filter(Number.isFinite);
      } else {
        const n = Number(input.value);
        if (Number.isFinite(n)) block.params[param.name] = param.kind === "integer" ? Math.max(1, Math.round(n)) : n;
      }
      refresh();
    });
    root.append(el("div", { class: "field" }, [el("label", { text: param.label }), input]));
  }
  const actions = el("div", { class: "actions" });
  const firstNumeric = (itemFor(block).params || []).find(p => p.kind === "number" || p.kind === "integer");
  if (firstNumeric) {
    actions.append(el("button", { type: "button", text: "Design Var", onclick: () => addDesignVariable(block, firstNumeric) }));
  }
  actions.append(el("button", { type: "button", text: "Objective", onclick: () => setObjective(block) }));
  actions.append(el("button", { type: "button", text: "Delete", onclick: () => deleteSelected() }));
  root.append(actions);
  if (model.designVariables && model.designVariables.length) {
    root.append(el("div", { class: "field" }, [el("label", { text: "Design variables" }), smallTable(model.designVariables.map(v => [v.name, `${v.lower}..${v.upper}`]))]));
  }
  if (model.objectives && model.objectives.length) {
    root.append(el("div", { class: "field" }, [el("label", { text: "Objectives" }), smallTable(model.objectives.map(o => [o.name, o.sense]))]));
  }
}
function addDesignVariable(block, param) {
  if (!model.designVariables) model.designVariables = [];
  const name = `${block.id}.${param.name}`;
  if (model.designVariables.some(v => v.name === name)) return;
  const value = Number(block.params && block.params[param.name] !== undefined ? block.params[param.name] : param.defaultValue);
  model.designVariables.push({ name, block: block.id, param: param.name, lower: Math.max(0, value - 1), upper: value + 1, samples: 9, units: null });
  refresh();
}
function setObjective(block) {
  if (!model.objectives) model.objectives = [];
  model.objectives = [{ name: `${block.id} final`, block: block.id, port: 0, sense: "track", target: lastRun && lastRun.finalSignals[block.id] !== undefined ? lastRun.finalSignals[block.id] : 0 }];
  refresh();
}
function deleteSelected() {
  const block = selectedBlock();
  if (!block) return;
  model.blocks = model.blocks.filter(b => b.id !== block.id);
  model.wires = model.wires.filter(w => w.from !== block.id && w.to !== block.id);
  selectedId = model.blocks[0] ? model.blocks[0].id : "";
  refresh();
}
function smallTable(rows) {
  const table = el("table");
  for (const row of rows) {
    table.append(el("tr", {}, [el("td", { text: row[0] }), el("td", { text: row[1] })]));
  }
  return table;
}
function compileModel() {
  const index = new Map(model.blocks.map((block, idx) => [block.id, idx]));
  const driver = new Map();
  const adj = model.blocks.map(() => []);
  const indeg = model.blocks.map(() => 0);
  for (const wire of model.wires) {
    const from = index.get(wire.from);
    const to = index.get(wire.to);
    if (from === undefined || to === undefined) throw new Error(`Unknown wire ${wire.from} -> ${wire.to}`);
    const srcIo = blockIo(model.blocks[from]);
    const dstIo = blockIo(model.blocks[to]);
    const fromPort = wire.fromPort || 0;
    const toPort = wire.toPort || 0;
    if (fromPort >= srcIo.outputs) throw new Error(`${wire.from} output ${fromPort} is not available`);
    if (toPort >= dstIo.inputs) throw new Error(`${wire.to} input ${toPort} is not available`);
    const key = `${to}:${toPort}`;
    if (driver.has(key)) throw new Error(`${wire.to} input ${toPort} is already driven`);
    driver.set(key, [from, fromPort]);
    if (from === to) throw new Error(`Cycle at ${wire.from}`);
    adj[from].push(to);
    indeg[to] += 1;
  }
  for (let idx = 0; idx < model.blocks.length; idx++) {
    const io = blockIo(model.blocks[idx]);
    for (let p = 0; p < io.inputs; p++) {
      if (!driver.has(`${idx}:${p}`)) throw new Error(`${model.blocks[idx].id} input ${p} is not connected`);
    }
  }
  const queue = [];
  indeg.forEach((d, idx) => { if (d === 0) queue.push(idx); });
  const order = [];
  for (let i = 0; i < queue.length; i++) {
    const u = queue[i];
    order.push(u);
    for (const v of adj[u]) {
      indeg[v] -= 1;
      if (indeg[v] === 0) queue.push(v);
    }
  }
  if (order.length !== model.blocks.length) throw new Error("The graph has a cycle");
  return { order, driver };
}
function param(block, name, fallback) {
  const value = block.params && block.params[name] !== undefined ? block.params[name] : fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
function stepBlock(block, t, inputs, state) {
  const first = inputs[0] || 0;
  if (block.kind === "constant") return [param(block, "value", 1)];
  if (block.kind === "step") return [t >= param(block, "t0", 1) ? param(block, "after", 1) : param(block, "before", 0)];
  if (block.kind === "ramp") return [param(block, "slope", 1) * t + param(block, "intercept", 0)];
  if (block.kind === "sine") return [param(block, "amp", 1) * Math.sin(2 * Math.PI * param(block, "freq", 1) * t) + param(block, "bias", 0)];
  if (block.kind === "gain") return [param(block, "k", 1) * first];
  if (block.kind === "affine") return [param(block, "m", 1) * first + param(block, "b", 0)];
  if (block.kind === "saturation") return [Math.max(param(block, "lo", -1), Math.min(param(block, "hi", 1), first))];
  if (block.kind === "sum") {
    const weights = block.params && Array.isArray(block.params.weights) ? block.params.weights : [1, 1];
    return [weights.reduce((acc, w, idx) => acc + Number(w || 0) * Number(inputs[idx] || 0), 0)];
  }
  if (block.kind === "integrator") {
    const s = state[block.id] || { value: param(block, "initial", 0), lastT: 0 };
    s.value += Math.max(0, t - s.lastT) * first;
    s.lastT = t;
    state[block.id] = s;
    return [s.value];
  }
  if (block.kind === "queue") {
    const s = state[block.id] || { backlog: 0 };
    s.backlog += Math.max(0, first);
    const served = Math.min(s.backlog, Math.max(0, param(block, "serviceRate", 1)));
    s.backlog -= served;
    state[block.id] = s;
    return [served];
  }
  if (block.kind === "transport-delay") {
    const delay = Math.max(1, Math.round(param(block, "delay", 1)));
    const s = state[block.id] || { buf: Array(delay).fill(0) };
    s.buf.push(first);
    const out = s.buf.shift() || 0;
    state[block.id] = s;
    return [out];
  }
  return [];
}
function runLocal(specModel = model) {
  const original = model;
  model = specModel;
  try {
    const compiled = compileModel();
    const state = {};
    const series = Object.fromEntries(model.blocks.map(block => [block.id, []]));
    const dt = Number(model.dt || 0.1);
    const steps = Math.max(1, Math.round(Number(model.steps || 1)));
    for (let k = 0; k < steps; k++) {
      const t = k * dt;
      const outs = model.blocks.map(() => []);
      for (const idx of compiled.order) {
        const block = model.blocks[idx];
        const io = blockIo(block);
        const inputs = Array(io.inputs).fill(0);
        for (let p = 0; p < io.inputs; p++) {
          const driven = compiled.driver.get(`${idx}:${p}`);
          if (driven) inputs[p] = (outs[driven[0]] || [])[driven[1]] || 0;
        }
        outs[idx] = stepBlock(block, t, inputs, state);
      }
      for (let idx = 0; idx < model.blocks.length; idx++) {
        const block = model.blocks[idx];
        let value = outs[idx][0];
        if (value === undefined) {
          const driven = compiled.driver.get(`${idx}:0`);
          value = driven ? ((outs[driven[0]] || [])[driven[1]] || 0) : 0;
        }
        series[block.id].push(value);
      }
    }
    const finalSignals = {};
    for (const [id, values] of Object.entries(series)) finalSignals[id] = values[values.length - 1] || 0;
    return { series, finalSignals };
  } finally {
    model = original;
  }
}
function runAndRender() {
  try {
    lastRun = runLocal(model);
    analysis = analyzeLocal();
    renderCanvas();
    renderBottom();
    renderDriver();
    renderInspector();
  } catch (err) {
    bottom.replaceChildren(el("div", { class: "error", text: err.message }));
  }
}
function renderBottom() {
  if (!lastRun) {
    bottom.replaceChildren(el("div", { class: "status" }, [el("span", { class: analysis.validation.ok ? "dot" : "dot bad" }), document.createTextNode(analysis.validation.message || "Ready")]));
    return;
  }
  const rows = Object.entries(lastRun.finalSignals).map(([id, value]) => [id, fmt(value)]);
  bottom.replaceChildren(el("div", { class: "results-grid" }, [
    el("div", { class: "metric" }, [el("strong", { text: "Final Signals" }), smallTable(rows)]),
    renderSeriesChart(lastRun.series)
  ]));
}
function renderSeriesChart(series) {
  const wrap = el("div", { class: "chart" });
  const chart = svg("svg", { viewBox: "0 0 640 180", width: "100%", height: "180" });
  const entries = Object.entries(series).filter(([, values]) => values.length);
  const all = entries.flatMap(([, values]) => values);
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 1);
  const span = Math.max(1e-9, max - min);
  chart.append(svg("line", { x1: 34, y1: 150, x2: 620, y2: 150, stroke: "#cad4df" }));
  chart.append(svg("line", { x1: 34, y1: 18, x2: 34, y2: 150, stroke: "#cad4df" }));
  const colors = ["#1f6feb", "#17803d", "#a86403", "#6d5bd0", "#c2413b"];
  entries.forEach(([id, values], idx) => {
    const pts = values.map((v, i) => {
      const x = 34 + (values.length === 1 ? 0 : i / (values.length - 1)) * 586;
      const y = 150 - ((v - min) / span) * 132;
      return `${x},${y}`;
    }).join(" ");
    chart.append(svg("polyline", { points: pts, fill: "none", stroke: colors[idx % colors.length], "stroke-width": 2 }));
    const label = svg("text", { x: 42 + idx * 110, y: 14, fill: colors[idx % colors.length], "font-size": 11 });
    label.textContent = id;
    chart.append(label);
  });
  wrap.append(chart);
  return wrap;
}
function renderN2() {
  const root = document.getElementById("n2-view");
  root.replaceChildren();
  const n = model.blocks.length;
  const grid = el("div", { class: "n2-grid" });
  grid.style.gridTemplateColumns = `120px repeat(${n}, 92px)`;
  grid.append(el("div", { class: "n2-cell header", text: "to / from" }));
  for (const col of model.blocks) grid.append(el("div", { class: "n2-cell header", text: col.id }));
  for (let r = 0; r < n; r++) {
    const row = model.blocks[r];
    grid.append(el("div", { class: "n2-cell header", text: row.id }));
    for (let c = 0; c < n; c++) {
      const col = model.blocks[c];
      const wires = model.wires.filter(w => w.from === col.id && w.to === row.id);
      const cell = el("div", { class: `n2-cell ${r === c ? "diag" : wires.length ? "link" : ""}`, text: r === c ? row.kind : wires.length ? wires.map(w => `${w.fromPort || 0}->${w.toPort || 0}`).join(", ") : "" });
      grid.append(cell);
    }
  }
  root.append(grid);
}
function runSweepLocal() {
  const dv = model.designVariables && model.designVariables[0];
  if (!dv) return null;
  const samples = Math.max(1, Math.round(Number(dv.samples || 1)));
  const values = samples === 1 ? [Number(dv.lower)] : Array.from({ length: samples }, (_, i) => Number(dv.lower) + (Number(dv.upper) - Number(dv.lower)) * i / (samples - 1));
  const cases = [];
  for (const value of values) {
    const spec = clone(model);
    const block = spec.blocks.find(b => b.id === dv.block);
    if (!block) continue;
    if (!block.params) block.params = {};
    block.params[dv.param] = value;
    const out = runLocal(spec);
    const objectives = (model.objectives || []).map(obj => {
      const y = out.finalSignals[obj.block] || 0;
      return { name: obj.name, block: obj.block, sense: obj.sense, value: y, target: obj.target ?? null, error: obj.target === null || obj.target === undefined ? null : y - obj.target };
    });
    const constraints = (model.constraints || []).map(con => {
      const y = out.finalSignals[con.block] || 0;
      const ok = (con.lower === null || con.lower === undefined || y >= con.lower) && (con.upper === null || con.upper === undefined || y <= con.upper);
      return { name: con.name, block: con.block, value: y, lower: con.lower ?? null, upper: con.upper ?? null, satisfied: ok };
    });
    cases.push({ value, finalSignals: out.finalSignals, objectives, constraints });
  }
  let bestCaseIndex = null;
  let bestScore = Infinity;
  cases.forEach((item, idx) => {
    if (!item.constraints.every(c => c.satisfied)) return;
    const obj = item.objectives[0];
    if (!obj) return;
    const score = obj.sense === "maximize" ? -obj.value : obj.sense === "track" ? Math.abs(obj.error ?? obj.value) : obj.value;
    if (score < bestScore) {
      bestScore = score;
      bestCaseIndex = idx;
    }
  });
  return { designVariable: dv, cases, bestCaseIndex };
}
function sweepAndRender() {
  try {
    lastSweep = runSweepLocal();
    renderDriver();
    showView("driver");
  } catch (err) {
    document.getElementById("driver-view").replaceChildren(el("div", { class: "error", text: err.message }));
  }
}
function renderDriver() {
  const root = document.getElementById("driver-view");
  root.replaceChildren();
  const sweep = lastSweep;
  if (!sweep) {
    root.append(el("div", { class: "error", text: "No design variable declared" }));
    return;
  }
  const rows = sweep.cases.map((item, idx) => [
    `${idx === sweep.bestCaseIndex ? "*" : ""}${fmt(item.value)}`,
    item.objectives[0] ? fmt(item.objectives[0].value) : "n/a",
    item.constraints.every(c => c.satisfied) ? "ok" : "bound"
  ]);
  const chart = sweepChart(sweep);
  root.append(el("div", { class: "driver-layout" }, [
    el("div", { class: "metric" }, [
      el("strong", { text: sweep.designVariable.name }),
      smallTable(rows)
    ]),
    chart
  ]));
}
function sweepChart(sweep) {
  const wrap = el("div", { class: "chart" });
  const chart = svg("svg", { viewBox: "0 0 640 220", width: "100%", height: "220" });
  const values = sweep.cases.map(c => c.objectives[0] ? c.objectives[0].value : 0);
  const xs = sweep.cases.map(c => c.value);
  const minY = Math.min(...values, 0);
  const maxY = Math.max(...values, 1);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 1);
  const sy = Math.max(1e-9, maxY - minY);
  const sx = Math.max(1e-9, maxX - minX);
  chart.append(svg("line", { x1: 42, y1: 182, x2: 612, y2: 182, stroke: "#cad4df" }));
  chart.append(svg("line", { x1: 42, y1: 22, x2: 42, y2: 182, stroke: "#cad4df" }));
  const points = sweep.cases.map(c => {
    const yv = c.objectives[0] ? c.objectives[0].value : 0;
    const x = 42 + ((c.value - minX) / sx) * 570;
    const y = 182 - ((yv - minY) / sy) * 160;
    return `${x},${y}`;
  }).join(" ");
  chart.append(svg("polyline", { points, fill: "none", stroke: "#1f6feb", "stroke-width": 2.4 }));
  sweep.cases.forEach((c, idx) => {
    const yv = c.objectives[0] ? c.objectives[0].value : 0;
    const x = 42 + ((c.value - minX) / sx) * 570;
    const y = 182 - ((yv - minY) / sy) * 160;
    chart.append(svg("circle", { cx: x, cy: y, r: idx === sweep.bestCaseIndex ? 5 : 3.5, fill: idx === sweep.bestCaseIndex ? "#17803d" : "#1f6feb" }));
  });
  wrap.append(chart);
  return wrap;
}
function renderJsonView() {
  const root = document.getElementById("json-view");
  root.replaceChildren();
  const area = el("textarea", { id: "json-area" });
  area.value = JSON.stringify(model, null, 2);
  const message = el("span", { class: "status", text: "" });
  const apply = el("button", { type: "button", class: "primary", text: "Apply", onclick: () => {
    try {
      model = JSON.parse(area.value);
      selectedId = model.blocks[0] ? model.blocks[0].id : "";
      lastRun = null;
      refresh();
      showView("json");
      message.textContent = "Applied";
    } catch (err) {
      message.textContent = err.message;
      message.className = "error";
    }
  }});
  root.append(el("div", { class: "field" }, [el("label", { text: "Model spec" }), area, el("div", { class: "json-actions" }, [apply, message])]));
}
function showView(view) {
  activeView = view;
  document.querySelectorAll(".tabs button").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view));
  document.getElementById("canvas-view").classList.toggle("hidden", view !== "canvas");
  document.getElementById("n2-view").classList.toggle("hidden", view !== "n2");
  document.getElementById("driver-view").classList.toggle("hidden", view !== "driver");
  document.getElementById("json-view").classList.toggle("hidden", view !== "json");
  if (view === "n2") renderN2();
  if (view === "driver") renderDriver();
  if (view === "json") renderJsonView();
}
function refresh() {
  analysis = analyzeLocal();
  document.getElementById("model-subtitle").textContent = `${model.name || "untitled"} / ${model.blocks.length} blocks / ${model.wires.length} wires`;
  renderTopbarControls();
  renderPalette();
  renderCanvas();
  renderInspector();
  renderBottom();
  if (activeView === "n2") renderN2();
  if (activeView === "driver") renderDriver();
  if (activeView === "json") renderJsonView();
}
window.showView = showView;
window.runAndRender = runAndRender;
window.sweepAndRender = sweepAndRender;
refresh();
runAndRender();
</script>
</body>
</html>
"###;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workbench_embeds_model_palette_and_views() {
        let html = render_starter_workbench_html().unwrap();
        assert!(html.contains("DES Modeling Studio"));
        assert!(html.contains("const PALETTE = ["));
        assert!(html.contains("view-tabs"));
        assert!(html.contains("gain.k"));
    }
}
