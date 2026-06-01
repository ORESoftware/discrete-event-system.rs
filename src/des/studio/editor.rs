//! Static browser editor for the first Modeling Studio surface.
//!
//! The page is intentionally self-contained so the existing site builder can
//! emit it alongside simulation artifacts without a web server.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::spec::{starter_model_spec, studio_palette};

pub const STUDIO_EDITOR_REL_PATH: &str = "studio/modeling-studio.html";

fn script_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .expect("studio editor bootstrap JSON should serialize")
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

pub fn studio_editor_html() -> String {
    STUDIO_EDITOR_TEMPLATE
        .replace("__PALETTE_JSON__", &script_json(&studio_palette()))
        .replace(
            "__STARTER_MODEL_JSON__",
            &script_json(&starter_model_spec()),
        )
}

pub fn write_studio_editor_html(out_root: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let path = out_root.as_ref().join(STUDIO_EDITOR_REL_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, studio_editor_html())?;
    Ok(path)
}

const STUDIO_EDITOR_TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Modeling Studio</title>
<style>
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #596575;
  --panel: #ffffff;
  --panel-soft: #f5f7fa;
  --line: #d5dbe4;
  --grid: #edf1f6;
  --accent: #0f766e;
  --accent-2: #b45309;
  --danger: #be123c;
  --blue: #2563eb;
  --shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #f7f8fb;
}
button, input, textarea {
  font: inherit;
}
.app {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 250px minmax(480px, 1fr) 300px;
  grid-template-rows: auto 1fr;
}
.topbar {
  grid-column: 1 / 4;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.brand-mark {
  width: 32px;
  height: 32px;
  border: 1px solid #99b6ae;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: #e6f3f0;
}
.brand h1 {
  margin: 0;
  font-size: 17px;
  line-height: 1.2;
  font-weight: 750;
  letter-spacing: 0;
}
.actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.icon-btn, .text-btn, .palette-btn {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  color: var(--ink);
  cursor: pointer;
}
.text-btn {
  min-height: 34px;
  padding: 0 12px;
}
.text-btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #ffffff;
}
.text-btn.warn {
  color: var(--danger);
}
.side, .inspector {
  min-height: 0;
  overflow: auto;
  background: var(--panel);
}
.side {
  border-right: 1px solid var(--line);
  padding: 14px;
}
.inspector {
  border-left: 1px solid var(--line);
  display: grid;
  grid-template-rows: auto auto minmax(150px, 1fr) auto;
}
.panel-section {
  padding: 14px;
  border-bottom: 1px solid var(--line);
}
.panel-section h2, .panel-section h3 {
  margin: 0 0 10px;
  font-size: 12px;
  line-height: 1.2;
  color: var(--muted);
  letter-spacing: 0;
  text-transform: uppercase;
}
.palette-group {
  margin-bottom: 14px;
}
.palette-list {
  display: grid;
  gap: 7px;
}
.palette-btn {
  width: 100%;
  min-height: 42px;
  display: grid;
  grid-template-columns: 32px 1fr;
  align-items: center;
  gap: 9px;
  padding: 5px 8px;
  text-align: left;
}
.palette-btn:hover, .text-btn:hover, .icon-btn:hover {
  border-color: #94a3b8;
  background: #f8fafc;
}
.palette-icon {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  display: grid;
  place-items: center;
}
.palette-label {
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.workspace {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(320px, 1fr) 210px;
  background:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px),
    #fbfcfe;
  background-size: 24px 24px;
}
.canvas-wrap {
  min-height: 0;
  position: relative;
  overflow: hidden;
}
#canvas {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
.plot {
  border-top: 1px solid var(--line);
  background: #ffffff;
  display: grid;
  grid-template-columns: 1fr 260px;
  min-height: 0;
}
#plotSvg {
  width: 100%;
  height: 100%;
  display: block;
}
.run-panel {
  border-left: 1px solid var(--line);
  padding: 12px;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 8px;
  min-width: 0;
}
#status {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}
.status-bad { color: var(--danger); }
.status-good { color: var(--accent); }
.json-box {
  width: 100%;
  min-height: 150px;
  resize: none;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fcfdff;
  color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
.field {
  display: grid;
  gap: 5px;
  margin-bottom: 10px;
}
.field label {
  color: var(--muted);
  font-size: 12px;
}
.field input {
  width: 100%;
  min-height: 32px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0 9px;
  background: #ffffff;
}
.empty {
  color: var(--muted);
  font-size: 13px;
}
.block-rect {
  fill: #ffffff;
  stroke: #64748b;
  stroke-width: 1.5;
  filter: drop-shadow(0 5px 10px rgba(15, 23, 42, 0.12));
}
.block-selected .block-rect {
  stroke: var(--accent);
  stroke-width: 2.3;
}
.block-title {
  fill: var(--ink);
  font-size: 13px;
  font-weight: 720;
  pointer-events: none;
}
.block-kind {
  fill: var(--muted);
  font-size: 10px;
  pointer-events: none;
}
.port {
  cursor: crosshair;
  stroke: #334155;
  stroke-width: 1.5;
}
.port.in { fill: #fef3c7; }
.port.out { fill: #dbeafe; }
.wire {
  fill: none;
  stroke: #475569;
  stroke-width: 2;
}
.wire.pending {
  stroke: var(--accent-2);
  stroke-dasharray: 5 5;
}
.glyph-stroke {
  fill: none;
  stroke: var(--accent);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.glyph-fill {
  fill: rgba(15, 118, 110, 0.12);
  stroke: var(--accent);
  stroke-width: 1.5;
}
.axis {
  stroke: #cbd5e1;
  stroke-width: 1;
}
.plot-line {
  fill: none;
  stroke: var(--blue);
  stroke-width: 2;
}
@media (max-width: 980px) {
  .app {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto minmax(420px, 1fr) auto;
  }
  .topbar, .side, .workspace, .inspector {
    grid-column: 1;
  }
  .side {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .inspector {
    border-left: 0;
    border-top: 1px solid var(--line);
  }
  .palette-list {
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" width="24" height="24">
          <path class="glyph-stroke" d="M5 20c5 0 5-9 10-9s5 9 10 9"></path>
          <path class="glyph-stroke" d="M6 25h20"></path>
        </svg>
      </div>
      <h1>Modeling Studio</h1>
    </div>
    <div class="actions">
      <button class="text-btn primary" id="runBtn" type="button">Run</button>
      <button class="text-btn" id="exportBtn" type="button">Export JSON</button>
      <button class="text-btn" id="loadBtn" type="button">Load JSON</button>
      <button class="text-btn warn" id="resetBtn" type="button">Reset</button>
    </div>
  </header>

  <aside class="side">
    <div class="panel-section">
      <h2>Palette</h2>
      <div id="palette"></div>
    </div>
  </aside>

  <main class="workspace">
    <div class="canvas-wrap">
      <svg id="canvas" viewBox="0 0 1000 620" role="img" aria-label="Model diagram canvas"></svg>
    </div>
    <section class="plot">
      <svg id="plotSvg" viewBox="0 0 760 210" role="img" aria-label="Selected signal plot"></svg>
      <div class="run-panel">
        <p id="status">Ready</p>
        <textarea id="jsonBox" class="json-box" spellcheck="false"></textarea>
      </div>
    </section>
  </main>

  <aside class="inspector">
    <div class="panel-section">
      <h2>Model</h2>
      <div class="field"><label for="modelName">Name</label><input id="modelName"></div>
      <div class="field"><label for="modelDt">dt</label><input id="modelDt" type="number" step="0.01"></div>
      <div class="field"><label for="modelSteps">Steps</label><input id="modelSteps" type="number" step="1" min="1"></div>
    </div>
    <div class="panel-section">
      <h2>Block</h2>
      <div id="blockInspector" class="empty">Select a block</div>
    </div>
    <div class="panel-section">
      <h2>Wires</h2>
      <div id="wireList" class="empty"></div>
    </div>
    <div class="panel-section">
      <h2>Selection</h2>
      <div id="selectionLabel" class="empty">None</div>
    </div>
  </aside>
</div>

<script>
window.STUDIO_PALETTE = __PALETTE_JSON__;
window.STUDIO_STARTER_MODEL = __STARTER_MODEL_JSON__;

(function () {
  const svgNS = "http://www.w3.org/2000/svg";
  const palette = window.STUDIO_PALETTE;
  const MAX_MODEL_BLOCKS = 128;
  const MAX_MODEL_WIRES = 512;
  const MAX_RUN_STEPS = 5000;
  const MAX_ID_LEN = 96;
  const MAX_LABEL_LEN = 160;
  const MAX_PARAM_ARRAY_LEN = 256;
  let model = normalizeModel(clone(window.STUDIO_STARTER_MODEL));
  let selectedId = model.blocks[0] ? model.blocks[0].id : null;
  let pendingWire = null;
  let drag = null;
  let lastSeries = [];

  const canvas = document.getElementById("canvas");
  const plotSvg = document.getElementById("plotSvg");
  const statusEl = document.getElementById("status");
  const jsonBox = document.getElementById("jsonBox");

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function el(id) {
    return document.getElementById(id);
  }

  function svgEl(name, attrs) {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs || {})) {
      node.setAttribute(key, String(value));
    }
    return node;
  }

  function setStatus(text, tone) {
    statusEl.textContent = text;
    statusEl.className = tone === "bad" ? "status-bad" : tone === "good" ? "status-good" : "";
  }

  function itemFor(kind) {
    return palette.find((item) => item.kind === kind);
  }

  function blockById(id) {
    return model.blocks.find((block) => block.id === id);
  }

  function blockLabel(block) {
    const item = itemFor(block.kind);
    return block.label || (item ? item.label : block.kind);
  }

  function blockInputs(block) {
    const item = itemFor(block.kind);
    if (!item) return 0;
    if (block.kind === "sum" && Array.isArray(block.params.weights)) {
      return block.params.weights.length;
    }
    return item.inputs;
  }

  function blockOutputs(block) {
    const item = itemFor(block.kind);
    return item ? item.outputs : 0;
  }

  function defaultParams(kind) {
    const item = itemFor(kind);
    const params = {};
    if (!item) return params;
    for (const p of item.params) {
      params[p.name] = clone(p.defaultValue);
    }
    return params;
  }

  function normalizeModel(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("model must be an object");
    if (!Array.isArray(raw.blocks)) throw new Error("model.blocks must be an array");
    if (!Array.isArray(raw.wires)) raw.wires = [];
    if (raw.blocks.length > MAX_MODEL_BLOCKS) throw new Error("model is limited to " + MAX_MODEL_BLOCKS + " blocks");
    if (raw.wires.length > MAX_MODEL_WIRES) throw new Error("model is limited to " + MAX_MODEL_WIRES + " wires");
    raw.name = checkedText(raw.name == null ? "untitled" : String(raw.name), "model name", MAX_LABEL_LEN, false);
    raw.dt = readFinite(raw.dt == null ? 0.1 : raw.dt, "model.dt");
    if (!(raw.dt > 0)) throw new Error("model.dt must be positive");
    raw.steps = readInteger(raw.steps == null ? 80 : raw.steps, "model.steps", 1, MAX_RUN_STEPS);
    const ids = new Set();
    for (const block of raw.blocks) {
      if (!block || typeof block !== "object" || Array.isArray(block)) throw new Error("blocks must be objects");
      block.id = checkedText(block.id == null ? "" : String(block.id), "block id", MAX_ID_LEN, false);
      if (ids.has(block.id)) throw new Error("duplicate block id: " + block.id);
      ids.add(block.id);
      if (!itemFor(block.kind)) throw new Error("unknown block kind: " + String(block.kind));
      if (block.label != null) block.label = checkedText(String(block.label), "block label", MAX_LABEL_LEN, true);
      block.params = block.params && typeof block.params === "object" && !Array.isArray(block.params) ? block.params : {};
      block.x = readFinite(block.x == null ? 0 : block.x, block.id + ".x");
      block.y = readFinite(block.y == null ? 0 : block.y, block.id + ".y");
      normalizeParams(block);
    }
    for (const wire of raw.wires) {
      if (!wire || typeof wire !== "object" || Array.isArray(wire)) throw new Error("wires must be objects");
      wire.from = checkedText(wire.from == null ? "" : String(wire.from), "wire.from", MAX_ID_LEN, false);
      wire.to = checkedText(wire.to == null ? "" : String(wire.to), "wire.to", MAX_ID_LEN, false);
      if (!ids.has(wire.from)) throw new Error("wire references unknown block: " + wire.from);
      if (!ids.has(wire.to)) throw new Error("wire references unknown block: " + wire.to);
      wire.fromPort = readInteger(wire.fromPort == null ? 0 : wire.fromPort, "wire.fromPort", 0, 255);
      wire.toPort = readInteger(wire.toPort == null ? 0 : wire.toPort, "wire.toPort", 0, 255);
    }
    return raw;
  }

  function normalizeParams(block) {
    const item = itemFor(block.kind);
    const allowed = new Set(item.params.map((param) => param.name));
    for (const key of Object.keys(block.params)) {
      if (!allowed.has(key)) throw new Error(block.id + " has unknown parameter " + key);
    }
    for (const param of item.params) {
      if (block.params[param.name] === undefined) block.params[param.name] = clone(param.defaultValue);
      if (param.kind === "number-array") {
        if (!Array.isArray(block.params[param.name])) throw new Error(block.id + "." + param.name + " must be an array");
        if (!block.params[param.name].length) throw new Error(block.id + "." + param.name + " must not be empty");
        if (block.params[param.name].length > MAX_PARAM_ARRAY_LEN) throw new Error(block.id + "." + param.name + " is too long");
        block.params[param.name] = block.params[param.name].map((value) => readFinite(value, block.id + "." + param.name));
      } else if (param.kind === "integer") {
        block.params[param.name] = readInteger(block.params[param.name], block.id + "." + param.name, Math.max(0, param.min || 0), 1000000);
      } else {
        block.params[param.name] = readFinite(block.params[param.name], block.id + "." + param.name);
      }
    }
    if (block.kind === "queue" && block.params.serviceRate < 0) throw new Error(block.id + ".serviceRate must be non-negative");
    if (block.kind === "saturation" && block.params.lo > block.params.hi) throw new Error(block.id + ".lo must be <= hi");
  }

  function checkedText(value, label, maxLen, allowEmpty) {
    if (!allowEmpty && !value.trim()) throw new Error(label + " must be non-empty");
    if (value.length > maxLen) throw new Error(label + " is too long");
    if (/[\x00-\x1f\x7f]/.test(value)) throw new Error(label + " must not contain control characters");
    return value;
  }

  function readFinite(value, label) {
    const n = Number(value);
    if (!Number.isFinite(n)) throw new Error(label + " must be a finite number");
    return n;
  }

  function readInteger(value, label, min, max) {
    const n = Number(value);
    if (!Number.isInteger(n) || n < min || n > max) {
      throw new Error(label + " must be an integer in " + min + ".." + max);
    }
    return n;
  }

  function uniqueId(kind) {
    const base = kind.replace(/[^a-z0-9]+/g, "_");
    let i = 1;
    let id = base;
    const ids = new Set(model.blocks.map((b) => b.id));
    while (ids.has(id)) {
      i += 1;
      id = base + "_" + i;
    }
    return id;
  }

  function renderPalette() {
    const root = el("palette");
    root.innerHTML = "";
    const categories = [...new Set(palette.map((item) => item.category))];
    for (const category of categories) {
      const group = document.createElement("div");
      group.className = "palette-group";
      const heading = document.createElement("h3");
      heading.textContent = category;
      const list = document.createElement("div");
      list.className = "palette-list";
      for (const item of palette.filter((entry) => entry.category === category)) {
        const button = document.createElement("button");
        button.className = "palette-btn";
        button.type = "button";
        button.title = item.description;
        button.innerHTML = '<span class="palette-icon">' + inlineGlyph(item.kind) + '</span><span class="palette-label"></span>';
        button.querySelector(".palette-label").textContent = item.label;
        button.addEventListener("click", () => addBlock(item.kind));
        list.appendChild(button);
      }
      group.appendChild(heading);
      group.appendChild(list);
      root.appendChild(group);
    }
  }

  function addBlock(kind) {
    if (model.blocks.length >= MAX_MODEL_BLOCKS) {
      setStatus("Model is limited to " + MAX_MODEL_BLOCKS + " blocks", "bad");
      return;
    }
    const id = uniqueId(kind);
    const offset = model.blocks.length * 24;
    const block = {
      id,
      kind,
      label: null,
      params: defaultParams(kind),
      x: 90 + (offset % 360),
      y: 80 + (offset % 220)
    };
    model.blocks.push(block);
    selectedId = id;
    pendingWire = null;
    renderAll();
    setStatus("Added " + blockLabel(block), "good");
  }

  function inlineGlyph(kind) {
    return '<svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">' + glyphMarkup(kind) + '</svg>';
  }

  function glyphMarkup(kind) {
    switch (kind) {
      case "integrator":
        return '<path class="glyph-fill" d="M7 23L7 19L11 17L15 12L20 9L25 8L25 23Z"></path><path class="glyph-stroke" d="M6 24H27M7 23C11 20 12 17 15 13S21 9 26 8"></path>';
      case "transport-delay":
        return '<path class="glyph-stroke" d="M7 10h9v6h9M7 22h18"></path><path class="glyph-stroke" d="M21 12l4 4l-4 4"></path>';
      case "sum":
        return '<path class="glyph-stroke" d="M10 8h13M10 24h13M10 8l9 8l-9 8"></path>';
      case "gain":
        return '<path class="glyph-stroke" d="M8 8l17 8l-17 8Z"></path><path class="glyph-stroke" d="M5 16h3M25 16h3"></path>';
      case "saturation":
        return '<path class="glyph-stroke" d="M6 23h5l10-14h5"></path><path class="glyph-stroke" d="M11 7v18M21 7v18"></path>';
      case "queue":
        return '<circle class="glyph-fill" cx="9" cy="16" r="3"></circle><circle class="glyph-fill" cx="16" cy="16" r="3"></circle><circle class="glyph-fill" cx="23" cy="16" r="3"></circle><path class="glyph-stroke" d="M5 23h22"></path>';
      case "sink":
        return '<path class="glyph-stroke" d="M7 10h14v12H7Z"></path><path class="glyph-stroke" d="M10 18c3-7 5 7 8 0M21 16h5"></path>';
      case "sine":
      case "step":
      case "ramp":
      case "constant":
        return '<path class="glyph-stroke" d="M5 22h22"></path><path class="glyph-stroke" d="M6 17c5-12 7 12 12 0s7 0 9-6"></path>';
      default:
        return '<path class="glyph-stroke" d="M6 16h20M16 6v20"></path>';
    }
  }

  function renderAll() {
    renderCanvas();
    renderInspector();
    renderWireList();
    syncModelInputs();
    exportJson(false);
    if (lastSeries.length) renderPlot(lastSeries);
  }

  function portPoint(block, side, index) {
    const w = 132;
    const h = 76;
    const count = side === "in" ? blockInputs(block) : blockOutputs(block);
    const y = block.y + ((index + 1) * h) / (count + 1);
    return { x: block.x + (side === "in" ? 0 : w), y };
  }

  function pathForWire(wire) {
    const from = blockById(wire.from);
    const to = blockById(wire.to);
    if (!from || !to) return "";
    const a = portPoint(from, "out", wire.fromPort || 0);
    const b = portPoint(to, "in", wire.toPort || 0);
    const dx = Math.max(60, Math.abs(b.x - a.x) * 0.45);
    return "M " + a.x + " " + a.y + " C " + (a.x + dx) + " " + a.y + ", " + (b.x - dx) + " " + b.y + ", " + b.x + " " + b.y;
  }

  function renderCanvas() {
    canvas.innerHTML = "";
    for (const wire of model.wires) {
      const path = svgEl("path", { class: "wire", d: pathForWire(wire) });
      canvas.appendChild(path);
    }
    for (const block of model.blocks) {
      const g = svgEl("g", { class: "block" + (block.id === selectedId ? " block-selected" : ""), transform: "translate(" + block.x + " " + block.y + ")", tabindex: 0 });
      g.addEventListener("pointerdown", (event) => {
        if (event.target.classList.contains("port")) return;
        selectedId = block.id;
        const p = canvasPoint(event);
        drag = { id: block.id, dx: p.x - block.x, dy: p.y - block.y };
        canvas.setPointerCapture(event.pointerId);
        renderAll();
      });
      const rect = svgEl("rect", { class: "block-rect", x: 0, y: 0, width: 132, height: 76, rx: 8 });
      const icon = svgEl("g", { transform: "translate(49 7)" });
      icon.innerHTML = glyphMarkup(block.kind);
      const title = svgEl("text", { class: "block-title", x: 66, y: 46, "text-anchor": "middle" });
      title.textContent = blockLabel(block);
      const kind = svgEl("text", { class: "block-kind", x: 66, y: 62, "text-anchor": "middle" });
      kind.textContent = block.kind;
      g.appendChild(rect);
      g.appendChild(icon);
      g.appendChild(title);
      g.appendChild(kind);
      for (let i = 0; i < blockInputs(block); i += 1) {
        const p = portPoint({ ...block, x: 0, y: 0 }, "in", i);
        const port = svgEl("circle", { class: "port in", cx: p.x, cy: p.y, r: 5, "data-id": block.id, "data-port": i, "data-side": "in" });
        port.addEventListener("pointerdown", onPortPointerDown);
        g.appendChild(port);
      }
      for (let i = 0; i < blockOutputs(block); i += 1) {
        const p = portPoint({ ...block, x: 0, y: 0 }, "out", i);
        const port = svgEl("circle", { class: "port out", cx: p.x, cy: p.y, r: 5, "data-id": block.id, "data-port": i, "data-side": "out" });
        port.addEventListener("pointerdown", onPortPointerDown);
        g.appendChild(port);
      }
      canvas.appendChild(g);
    }
  }

  function canvasPoint(event) {
    const pt = canvas.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    return pt.matrixTransform(canvas.getScreenCTM().inverse());
  }

  function onPortPointerDown(event) {
    event.stopPropagation();
    const id = event.target.getAttribute("data-id");
    const port = Number(event.target.getAttribute("data-port"));
    const side = event.target.getAttribute("data-side");
    selectedId = id;
    if (side === "out") {
      pendingWire = { from: id, fromPort: port };
      setStatus("Output selected", "good");
    } else if (pendingWire && pendingWire.from !== id) {
      model.wires = model.wires.filter((wire) => !(wire.to === id && (wire.toPort || 0) === port));
      model.wires.push({ from: pendingWire.from, fromPort: pendingWire.fromPort, to: id, toPort: port });
      pendingWire = null;
      setStatus("Wire connected", "good");
    }
    renderAll();
  }

  window.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const block = blockById(drag.id);
    if (!block) return;
    const p = canvasPoint(event);
    block.x = Math.max(12, Math.min(856, Math.round(p.x - drag.dx)));
    block.y = Math.max(12, Math.min(520, Math.round(p.y - drag.dy)));
    renderCanvas();
    exportJson(false);
  });

  window.addEventListener("pointerup", () => {
    drag = null;
  });

  function syncModelInputs() {
    el("modelName").value = model.name || "";
    el("modelDt").value = model.dt;
    el("modelSteps").value = model.steps;
    const selected = blockById(selectedId);
    el("selectionLabel").textContent = selected ? selected.id + " (" + selected.kind + ")" : "None";
  }

  function renderInspector() {
    const root = el("blockInspector");
    const block = blockById(selectedId);
    if (!block) {
      root.className = "empty";
      root.textContent = "Select a block";
      return;
    }
    root.className = "";
    root.innerHTML = "";
    root.appendChild(field("Id", block.id, (value) => {
      const next = value.trim();
      if (!next || next === block.id || blockById(next)) return;
      const old = block.id;
      block.id = next;
      for (const wire of model.wires) {
        if (wire.from === old) wire.from = next;
        if (wire.to === old) wire.to = next;
      }
      selectedId = next;
      renderAll();
    }));
    root.appendChild(field("Label", block.label || "", (value) => {
      block.label = value.trim() || null;
      renderAll();
    }));
    root.appendChild(field("x", block.x, (value) => {
      block.x = Number(value);
      renderAll();
    }, "number"));
    root.appendChild(field("y", block.y, (value) => {
      block.y = Number(value);
      renderAll();
    }, "number"));
    const item = itemFor(block.kind);
    for (const param of item ? item.params : []) {
      root.appendChild(field(param.label, Array.isArray(block.params[param.name]) ? block.params[param.name].join(", ") : block.params[param.name], (value) => {
        if (param.kind === "number-array") {
          block.params[param.name] = value.split(",").map((part) => Number(part.trim())).filter((n) => Number.isFinite(n));
        } else {
          block.params[param.name] = Number(value);
        }
        renderAll();
      }, param.kind === "number-array" ? "text" : "number"));
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "text-btn warn";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => {
      model.blocks = model.blocks.filter((entry) => entry.id !== block.id);
      model.wires = model.wires.filter((wire) => wire.from !== block.id && wire.to !== block.id);
      selectedId = model.blocks[0] ? model.blocks[0].id : null;
      renderAll();
    });
    root.appendChild(remove);
  }

  function field(label, value, onChange, type) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const id = "field_" + Math.random().toString(36).slice(2);
    const lab = document.createElement("label");
    lab.setAttribute("for", id);
    lab.textContent = label;
    const input = document.createElement("input");
    input.id = id;
    input.type = type || "text";
    input.value = value;
    input.addEventListener("change", () => onChange(input.value));
    wrap.appendChild(lab);
    wrap.appendChild(input);
    return wrap;
  }

  function renderWireList() {
    const root = el("wireList");
    if (!model.wires.length) {
      root.className = "empty";
      root.textContent = "No wires";
      return;
    }
    root.className = "";
    root.innerHTML = "";
    for (const wire of model.wires) {
      const row = document.createElement("div");
      row.style.display = "grid";
      row.style.gridTemplateColumns = "1fr auto";
      row.style.gap = "8px";
      row.style.alignItems = "center";
      row.style.marginBottom = "8px";
      const label = document.createElement("span");
      label.textContent = wire.from + ":" + (wire.fromPort || 0) + " -> " + wire.to + ":" + (wire.toPort || 0);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "icon-btn";
      button.textContent = "x";
      button.title = "Remove";
      button.addEventListener("click", () => {
        model.wires = model.wires.filter((entry) => entry !== wire);
        renderAll();
      });
      row.appendChild(label);
      row.appendChild(button);
      root.appendChild(row);
    }
  }

  function exportJson(updateStatus) {
    jsonBox.value = JSON.stringify(model, null, 2);
    if (updateStatus) setStatus("JSON exported", "good");
  }

  function loadJson() {
    try {
      const next = normalizeModel(JSON.parse(jsonBox.value));
      model = next;
      selectedId = model.blocks[0] ? model.blocks[0].id : null;
      pendingWire = null;
      lastSeries = [];
      renderAll();
      setStatus("JSON loaded", "good");
    } catch (err) {
      setStatus("Load failed: " + err.message, "bad");
    }
  }

  function validateModel() {
    try {
      normalizeModel(clone(model));
    } catch (err) {
      return err.message;
    }
    if (!Number.isFinite(Number(model.dt)) || Number(model.dt) <= 0) return "dt must be positive";
    if (!Number.isInteger(Number(model.steps)) || Number(model.steps) <= 0) return "steps must be a positive integer";
    const ids = new Set();
    for (const block of model.blocks) {
      if (!block.id || ids.has(block.id)) return "block ids must be unique";
      if (!itemFor(block.kind)) return "unknown block kind: " + block.kind;
      ids.add(block.id);
    }
    const driven = new Set();
    for (const wire of model.wires) {
      const from = blockById(wire.from);
      const to = blockById(wire.to);
      if (!from || !to) return "wire references an unknown block";
      if ((wire.fromPort || 0) >= blockOutputs(from)) return "wire output port is out of range";
      if ((wire.toPort || 0) >= blockInputs(to)) return "wire input port is out of range";
      const key = wire.to + ":" + (wire.toPort || 0);
      if (driven.has(key)) return "input port has multiple drivers";
      driven.add(key);
    }
    for (const block of model.blocks) {
      for (let i = 0; i < blockInputs(block); i += 1) {
        if (!driven.has(block.id + ":" + i)) return "undriven input on " + block.id;
      }
    }
    const ordered = topoOrder();
    if (ordered.length !== model.blocks.length) return "cycle detected";
    return null;
  }

  function topoOrder() {
    const indeg = new Map(model.blocks.map((block) => [block.id, 0]));
    const out = new Map(model.blocks.map((block) => [block.id, []]));
    for (const wire of model.wires) {
      indeg.set(wire.to, (indeg.get(wire.to) || 0) + 1);
      out.get(wire.from).push(wire.to);
    }
    const queue = model.blocks.filter((block) => (indeg.get(block.id) || 0) === 0).map((block) => block.id);
    const ordered = [];
    while (queue.length) {
      const id = queue.shift();
      ordered.push(id);
      for (const to of out.get(id) || []) {
        indeg.set(to, indeg.get(to) - 1);
        if (indeg.get(to) === 0) queue.push(to);
      }
    }
    return ordered.map(blockById).filter(Boolean);
  }

  function runModel() {
    model.name = el("modelName").value.trim() || model.name;
    model.dt = Number(el("modelDt").value);
    model.steps = Number(el("modelSteps").value);
    const issue = validateModel();
    if (issue) {
      setStatus(issue, "bad");
      return;
    }
    const series = simulate();
    lastSeries = series;
    renderPlot(series);
    setStatus("Ran " + model.steps + " steps", "good");
  }

  function inputValues(block, values) {
    const inputs = new Array(blockInputs(block)).fill(0);
    for (const wire of model.wires.filter((entry) => entry.to === block.id)) {
      const source = values.get(wire.from) || [];
      inputs[wire.toPort || 0] = source[wire.fromPort || 0] || 0;
    }
    return inputs;
  }

  function simulate() {
    const order = topoOrder();
    const dt = Number(model.dt);
    const states = new Map();
    for (const block of model.blocks) {
      if (block.kind === "integrator") states.set(block.id, { x: finite(block.params.initial, 0), lastT: 0 });
      if (block.kind === "queue") states.set(block.id, { q: 0 });
      if (block.kind === "transport-delay") {
        const n = Math.max(1, Math.floor(finite(block.params.delay, 1)));
        states.set(block.id, { buf: new Array(n).fill(0) });
      }
    }
    const series = [];
    for (let step = 0; step <= Number(model.steps); step += 1) {
      const t = step * dt;
      const values = new Map();
      const probeValues = new Map();
      for (const block of order) {
        const inputs = inputValues(block, values);
        let out = [];
        switch (block.kind) {
          case "constant":
            out = [finite(block.params.value, 1)];
            break;
          case "step":
            out = [t < finite(block.params.t0, 1) ? finite(block.params.before, 0) : finite(block.params.after, 1)];
            break;
          case "ramp":
            out = [finite(block.params.slope, 1) * t + finite(block.params.intercept, 0)];
            break;
          case "sine":
            out = [finite(block.params.amp, 1) * Math.sin(2 * Math.PI * finite(block.params.freq, 1) * t) + finite(block.params.bias, 0)];
            break;
          case "gain":
            out = [finite(block.params.k, 1) * inputs[0]];
            break;
          case "sum": {
            const weights = Array.isArray(block.params.weights) && block.params.weights.length ? block.params.weights : [1, 1];
            out = [weights.reduce((acc, w, i) => acc + finite(w, 0) * (inputs[i] || 0), 0)];
            break;
          }
          case "saturation": {
            const lo = finite(block.params.lo, -1);
            const hi = finite(block.params.hi, 1);
            out = [Math.min(Math.max(inputs[0], Math.min(lo, hi)), Math.max(lo, hi))];
            break;
          }
          case "affine":
            out = [finite(block.params.m, 1) * inputs[0] + finite(block.params.b, 0)];
            break;
          case "integrator": {
            const state = states.get(block.id);
            state.x += (t - state.lastT) * inputs[0];
            state.lastT = t;
            out = [state.x];
            break;
          }
          case "queue": {
            const state = states.get(block.id);
            state.q = Math.max(0, state.q + Math.max(0, inputs[0]));
            const served = Math.min(state.q, Math.max(0, finite(block.params.serviceRate, 1)));
            state.q -= served;
            out = [served];
            break;
          }
          case "transport-delay": {
            const state = states.get(block.id);
            out = [state.buf.shift() || 0];
            state.buf.push(inputs[0]);
            break;
          }
          case "sink":
            probeValues.set(block.id, inputs[0] || 0);
            out = [];
            break;
        }
        values.set(block.id, out);
      }
      const selected = blockById(selectedId) || order[order.length - 1];
      let y = 0;
      if (selected) {
        if (selected.kind === "sink") {
          y = probeValues.get(selected.id) || 0;
        } else {
          y = (values.get(selected.id) || [0])[0] || 0;
        }
      }
      series.push({ t, y });
    }
    return series;
  }

  function finite(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function renderPlot(series) {
    plotSvg.innerHTML = "";
    const w = 760;
    const h = 210;
    const pad = 28;
    plotSvg.appendChild(svgEl("line", { class: "axis", x1: pad, y1: h - pad, x2: w - 12, y2: h - pad }));
    plotSvg.appendChild(svgEl("line", { class: "axis", x1: pad, y1: 12, x2: pad, y2: h - pad }));
    if (!series.length) return;
    const minT = series[0].t;
    const maxT = series[series.length - 1].t || 1;
    let minY = Math.min(...series.map((p) => p.y));
    let maxY = Math.max(...series.map((p) => p.y));
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const x = (t) => pad + ((t - minT) / Math.max(1e-9, maxT - minT)) * (w - pad - 16);
    const y = (v) => h - pad - ((v - minY) / Math.max(1e-9, maxY - minY)) * (h - pad - 18);
    const d = series.map((p, i) => (i ? "L" : "M") + " " + x(p.t).toFixed(2) + " " + y(p.y).toFixed(2)).join(" ");
    plotSvg.appendChild(svgEl("path", { class: "plot-line", d }));
    const title = svgEl("text", { x: 42, y: 24, fill: "#596575", "font-size": 12 });
    const selected = blockById(selectedId);
    title.textContent = selected ? "selected: " + selected.id : "selected signal";
    plotSvg.appendChild(title);
  }

  el("runBtn").addEventListener("click", runModel);
  el("exportBtn").addEventListener("click", () => exportJson(true));
  el("loadBtn").addEventListener("click", loadJson);
  el("resetBtn").addEventListener("click", () => {
    model = normalizeModel(clone(window.STUDIO_STARTER_MODEL));
    selectedId = model.blocks[0] ? model.blocks[0].id : null;
    pendingWire = null;
    lastSeries = [];
    renderAll();
    renderPlot([]);
    setStatus("Reset", "good");
  });
  el("modelName").addEventListener("change", (event) => {
    model.name = event.target.value;
    exportJson(false);
  });
  el("modelDt").addEventListener("change", (event) => {
    model.dt = Number(event.target.value);
    exportJson(false);
  });
  el("modelSteps").addEventListener("change", (event) => {
    model.steps = Number(event.target.value);
    exportJson(false);
  });

  renderPalette();
  renderAll();
  runModel();
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn editor_html_embeds_palette_and_starter_model() {
        let html = studio_editor_html();
        assert!(html.contains("window.STUDIO_PALETTE"));
        assert!(html.contains("\"integrator\""));
        assert!(html.contains("window.STUDIO_STARTER_MODEL"));
        assert!(html.contains("Content-Security-Policy"));
        assert!(html.contains("MAX_MODEL_BLOCKS"));
        assert!(html.contains("normalizeModel"));
        assert!(!html.contains("__PALETTE_JSON__"));
        assert!(!html.contains("__STARTER_MODEL_JSON__"));
    }

    #[test]
    fn editor_bootstrap_json_escapes_script_breakout() {
        let encoded = script_json(&json!({
            "text": "</script><script>alert(1)</script>",
            "line": "\u{2028}\u{2029}",
            "amp": "&"
        }));
        assert!(!encoded.contains("</script>"));
        assert!(encoded.contains("\\u003c/script\\u003e"));
        assert!(encoded.contains("\\u2028\\u2029"));
        assert!(encoded.contains("\\u0026"));
    }

    #[test]
    fn editor_writer_creates_nested_file() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("des_engine_studio_editor_{nonce}"));
        let path = write_studio_editor_html(&root).unwrap();
        assert!(path.ends_with(STUDIO_EDITOR_REL_PATH));
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Modeling Studio"));
        let _ = std::fs::remove_dir_all(root);
    }
}
