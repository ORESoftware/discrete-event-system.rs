//! Render a plugin's captured output into a single self-contained HTML page.
//!
//! Like [`crate::des::animation::html_player`], the data is embedded in a
//! `<script type="application/json">` blob and a dependency-free vanilla-JS
//! player draws it offline. Two players, chosen by [`PlayerKind`]:
//!
//! * **sim** — a frame player with transport (play / pause / step / scrub /
//!   speed). Each frame renders as SVG when it carries a `shapes` array (the
//!   schema is the same as [`crate::des::animation::types`], so animation
//!   scenes can be emitted verbatim by a plugin), otherwise as a field
//!   inspector. A timeline charts every numeric frame field.
//! * **results** — metric cards for scalars, tables for arrays-of-objects, and
//!   a raw-JSON panel.
//!
//! Plugin-declared [`UiControl`](crate::des::plugin::manifest::UiControl)s are
//! rendered into a control panel with documented generic semantics:
//!
//! | control            | `target` / `id`        | effect                                   |
//! |--------------------|------------------------|------------------------------------------|
//! | toggle             | a series / section key | show/hide that series (sim) or section (results) |
//! | toggle             | `rawJson`              | show/hide the raw-JSON panel (results)   |
//! | select             | `metric`               | feature one series in the timeline (sim) |
//! | select             | `section`              | show only one results section            |
//! | range              | id `speed`             | playback rate in fps (sim)               |

use serde_json::{json, Value};

use super::manifest::{PlayerKind, PluginManifest};
use super::runner::{PluginOutput, PluginRun};

/// Render a complete HTML page for a plugin run.
pub fn render_player_html(manifest: &PluginManifest, run: &PluginRun) -> String {
    let payload = build_payload(manifest, run);
    let json = json_for_script(&payload);
    let title = escape_html(manifest.title.as_deref().unwrap_or(&manifest.name));
    let subtitle = escape_html(&manifest.description);
    TEMPLATE
        .replace("__TITLE__", &title)
        .replace("__SUBTITLE__", &subtitle)
        .replace("__PAYLOAD__", &json)
}

/// The embedded payload `{ player, title, subtitle, controls, frames|result }`.
fn build_payload(manifest: &PluginManifest, run: &PluginRun) -> Value {
    let controls =
        serde_json::to_value(&manifest.controls).unwrap_or_else(|_| Value::Array(vec![]));
    let mut obj = serde_json::Map::new();
    obj.insert(
        "player".to_string(),
        json!(match manifest.player {
            PlayerKind::Sim => "sim",
            PlayerKind::Results => "results",
        }),
    );
    obj.insert("pluginId".to_string(), json!(manifest.id));
    obj.insert(
        "title".to_string(),
        json!(manifest
            .title
            .clone()
            .unwrap_or_else(|| manifest.name.clone())),
    );
    obj.insert("subtitle".to_string(), json!(manifest.description.clone()));
    obj.insert("controls".to_string(), controls);
    match manifest.player {
        PlayerKind::Sim => {
            obj.insert(
                "frames".to_string(),
                Value::Array(normalize_frames(&run.output)),
            );
        }
        PlayerKind::Results => {
            obj.insert("result".to_string(), result_value(&run.output));
        }
    }
    Value::Object(obj)
}

/// Coerce any output into a frame list: JSONL → its lines; a JSON array → its
/// elements; a JSON object with a `frames` array → those; otherwise a single
/// one-element frame list.
fn normalize_frames(output: &PluginOutput) -> Vec<Value> {
    match output {
        PluginOutput::Jsonl(frames) => frames.clone(),
        PluginOutput::Json(value) => {
            if let Some(arr) = value.as_array() {
                arr.clone()
            } else if let Some(frames) = value.get("frames").and_then(Value::as_array) {
                frames.clone()
            } else {
                vec![value.clone()]
            }
        }
    }
}

/// Coerce any output into a single result value for the results player.
fn result_value(output: &PluginOutput) -> Value {
    match output {
        PluginOutput::Json(value) => value.clone(),
        PluginOutput::Jsonl(frames) => Value::Array(frames.clone()),
    }
}

/// `JSON.stringify` then make the text safe inside an inline `<script>`:
/// `</` → `<\/` (JSON-lossless) and the JS-illegal line separators U+2028/2029.
fn json_for_script(value: &Value) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|_| "{}".to_string())
        .replace("</", "<\\/")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #0f172a; background: #f1f5f9; }
  header { padding: 18px 22px; background: #0f172a; color: #f8fafc; }
  header h1 { margin: 0; font-size: 19px; font-weight: 650; }
  header p { margin: 4px 0 0; color: #94a3b8; font-size: 13px; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 18px 22px 60px; }
  .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
  .panel h2 { margin: 0 0 10px; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; color: #64748b; }
  .stage { width: 100%; min-height: 120px; display: flex; align-items: center; justify-content: center; }
  .stage-svg { width: 100%; max-height: 460px; background: #fff; }
  .chart-svg { width: 100%; height: auto; }
  .caption { margin-top: 8px; text-align: center; color: #475569; font-size: 13px; min-height: 18px; }
  .transport { display: grid; grid-template-columns: repeat(4, max-content) 30ch minmax(220px, 1fr); align-items: center; gap: 10px; }
  .transport button { font-size: 15px; line-height: 1; padding: 6px 11px; border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 7px; cursor: pointer; }
  .transport button:hover { background: #eef2f7; }
  .transport .frame-label { font-variant-numeric: tabular-nums; color: #475569; width: 30ch; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  input[type=range] { accent-color: #2563eb; }
  .scrub { width: 100%; min-width: 0; }
  @media (max-width: 720px) {
    .transport { grid-template-columns: repeat(4, max-content); }
    .transport .frame-label, .transport .scrub { grid-column: 1 / -1; width: 100%; }
  }
  .controls { display: flex; flex-wrap: wrap; gap: 14px 22px; }
  .ctrl { display: flex; flex-direction: column; gap: 4px; font-size: 13px; }
  .ctrl label { color: #475569; font-weight: 600; }
  .ctrl .row { display: flex; align-items: center; gap: 8px; }
  .ctrl select, .ctrl input[type=text] { padding: 4px 6px; border: 1px solid #cbd5e1; border-radius: 6px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; font-size: 12px; color: #334155; }
  .legend-item.off { opacity: .38; text-decoration: line-through; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; }
  .card { border: 1px solid #e2e8f0; border-radius: 9px; padding: 10px 14px; min-width: 120px; background: #f8fafc; }
  .card-label { font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: #64748b; }
  .card-value { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; margin-top: 2px; }
  .sect-title { margin: 16px 0 8px; font-size: 14px; color: #0f172a; }
  table.tbl { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.tbl th, table.tbl td { border: 1px solid #e2e8f0; padding: 5px 9px; text-align: left; }
  table.tbl th { background: #f1f5f9; font-weight: 600; }
  table.tbl td { font-variant-numeric: tabular-nums; }
  table.kv td:first-child { color: #64748b; width: 40%; }
  pre.raw { background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 9px; overflow: auto; font-size: 12px; max-height: 420px; }
  .inline-list { color: #334155; }
  .muted { color: #94a3b8; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>__SUBTITLE__</p>
</header>
<div class="wrap" id="app"></div>
<script type="application/json" id="plugin-payload">__PAYLOAD__</script>
<script>
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";
  var PAYLOAD = JSON.parse(document.getElementById("plugin-payload").textContent);
  var app = document.getElementById("app");
  var PALETTE = ["#2563eb","#16a34a","#dc2626","#9333ea","#ea580c","#0891b2","#ca8a04","#db2777","#65a30d","#475569"];
  function palette(i) { return PALETTE[i % PALETTE.length]; }

  // --- tiny DOM helpers ---
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      var v = attrs[k];
      if (v == null) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k === "style") e.setAttribute("style", v);
      else if (k.slice(0, 2) === "on" && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    (kids || []).forEach(function (c) {
      if (c == null) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return e;
  }
  function sv(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) { if (attrs[k] != null) e.setAttribute(k, attrs[k]); }
    return e;
  }
  function fmt(v) {
    if (typeof v !== "number" || !isFinite(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
    return (Math.round(v * 1000) / 1000).toString();
  }
  function fmtv(v) {
    if (v == null) return "";
    if (typeof v === "number") return fmt(v);
    if (typeof v === "boolean") return v ? "true" : "false";
    return String(v);
  }
  function fmtCell(v) {
    if (v == null) return "";
    if (typeof v === "object") return JSON.stringify(v);
    return fmtv(v);
  }

  // --- control state ---
  var controls = PAYLOAD.controls || [];
  var controlState = {};
  controls.forEach(function (c) { controlState[c.id] = (c.default == null ? null : c.default); });
  var controlInputs = {};

  function buildControlPanel(onChange) {
    if (!controls.length) return null;
    var row = h("div", { class: "controls" });
    controls.forEach(function (c) {
      var field;
      if (c.kind === "toggle") {
        var cb = h("input", { type: "checkbox" });
        cb.checked = !!controlState[c.id];
        cb.addEventListener("change", function () { controlState[c.id] = cb.checked; onChange(c); });
        controlInputs[c.id] = cb;
        field = h("div", { class: "ctrl" }, [h("label", null, [h("span", { class: "row" }, [cb, c.label])])]);
      } else if (c.kind === "select") {
        var sel = h("select");
        (c.options || []).forEach(function (o) {
          var opt = h("option", { value: o, text: o });
          if (o === controlState[c.id]) opt.selected = true;
          sel.appendChild(opt);
        });
        sel.addEventListener("change", function () { controlState[c.id] = sel.value; onChange(c); });
        controlInputs[c.id] = sel;
        field = h("div", { class: "ctrl" }, [h("label", { text: c.label }), sel]);
      } else { // range
        var out = h("span", { text: fmtv(controlState[c.id]) });
        var rng = h("input", { type: "range", min: c.min, max: c.max, step: c.step });
        rng.value = controlState[c.id];
        rng.addEventListener("input", function () {
          controlState[c.id] = parseFloat(rng.value);
          out.textContent = fmtv(controlState[c.id]);
          onChange(c);
        });
        controlInputs[c.id] = rng;
        field = h("div", { class: "ctrl" }, [h("label", { text: c.label }), h("div", { class: "row" }, [rng, out])]);
      }
      row.appendChild(field);
    });
    return h("div", { class: "panel" }, [h("h2", { text: "Controls" }), row]);
  }

  if (PAYLOAD.player === "sim") initSim(); else initResults();

  // ===================================================================== SIM
  function initSim() {
    var frames = PAYLOAD.frames || [];
    var n = frames.length;
    var idx = 0, playing = false, timer = null;

    var seriesKeys = collectNumericKeys(frames);
    var xs = frames.map(function (f, i) { return (f && typeof f.t === "number") ? f.t : i; });
    var hasShapes = frames.some(function (f) { return f && Array.isArray(f.shapes); });
    var bounds = hasShapes ? shapeBounds(frames) : null;
    var hidden = {};         // hidden series keys
    var featured = null;     // a single series to feature (select target=metric)

    function speed() {
      var sc = controls.find(function (c) { return c.id === "speed"; });
      var v = sc ? controlState["speed"] : null;
      return (typeof v === "number" && v > 0) ? v : 6;
    }

    // --- DOM ---
    var stage = h("div", { class: "stage" });
    var caption = h("div", { class: "caption" });
    var stagePanel = h("div", { class: "panel" }, [h("h2", { text: "Frame" }), stage, caption]);

    var playBtn = h("button", { text: "\u25B6", title: "Play / pause", onclick: togglePlay });
    var frameLabel = h("span", { class: "frame-label" });
    var scrub = h("input", { type: "range", class: "scrub", min: 0, max: Math.max(0, n - 1), step: 1 });
    scrub.value = 0;
    scrub.addEventListener("input", function () { stop(); idx = parseInt(scrub.value, 10) || 0; drawFrame(); });
    var transport = h("div", { class: "panel" }, [
      h("h2", { text: "Transport" }),
      h("div", { class: "transport" }, [
        h("button", { text: "\u23EE", title: "Restart", onclick: function () { stop(); idx = 0; drawFrame(); } }),
        h("button", { text: "\u25C0", title: "Step back", onclick: function () { stop(); idx = Math.max(0, idx - 1); drawFrame(); } }),
        playBtn,
        h("button", { text: "\u25B6\u2502", title: "Step forward", onclick: function () { stop(); idx = Math.min(n - 1, idx + 1); drawFrame(); } }),
        frameLabel, scrub
      ])
    ]);

    var timeline = h("div");
    var timelinePanel = h("div", { class: "panel" }, [h("h2", { text: "Timeline" }), timeline]);

    app.appendChild(transport);
    app.appendChild(stagePanel);
    if (seriesKeys.length) app.appendChild(timelinePanel);
    var cp = buildControlPanel(applyControls);
    if (cp) app.appendChild(cp);

    applyControls();
    drawFrame();

    function applyControls() {
      controls.forEach(function (c) {
        if (c.kind === "toggle" && c.target) {
          if (controlState[c.id]) delete hidden[c.target]; else hidden[c.target] = true;
        }
        if (c.kind === "select" && c.target === "metric") {
          featured = (controlState[c.id] && controlState[c.id] !== "all") ? controlState[c.id] : null;
        }
      });
      if (playing) schedule();
      drawTimeline();
      drawFrame();
    }

    function visibleSeries() {
      return seriesKeys.filter(function (k) {
        if (featured) return k === featured;
        return !hidden[k];
      });
    }

    function drawFrame() {
      if (n === 0) { stage.appendChild(h("div", { class: "muted", text: "(no frames)" })); return; }
      idx = Math.max(0, Math.min(n - 1, idx));
      scrub.value = idx;
      var f = frames[idx] || {};
      var t = (f && typeof f.t === "number") ? (" \u00B7 t=" + fmt(f.t)) : "";
      frameLabel.textContent = "frame " + (idx + 1) + " / " + n + t;
      stage.innerHTML = "";
      if (hasShapes && Array.isArray(f.shapes)) {
        var svg = sv("svg", { viewBox: bounds.x + " " + bounds.y + " " + bounds.w + " " + bounds.h, class: "stage-svg", preserveAspectRatio: "xMidYMid meet" });
        f.shapes.forEach(function (sh) {
          var e = renderShape(sh);
          if (e) svg.appendChild(e);
          if (sh.label != null && (sh.kind === "circle" || sh.kind === "rect")) {
            var lx = sh.kind === "circle" ? sh.x : sh.x + (sh.w || 0) / 2;
            var ly = sh.kind === "circle" ? sh.y : sh.y + (sh.h || 0) / 2;
            var lt = sv("text", { x: lx, y: ly, "text-anchor": "middle", "dominant-baseline": "central", "font-size": 11, fill: "#0f172a" });
            lt.textContent = sh.label;
            svg.appendChild(lt);
          }
        });
        stage.appendChild(svg);
      } else {
        stage.appendChild(fieldTable(f));
      }
      caption.textContent = (f && f.caption) ? f.caption : "";
      drawTimeline();
    }

    function fieldTable(f) {
      if (f == null || typeof f !== "object") return h("div", { class: "card-value", text: fmtv(f) });
      var t = h("table", { class: "tbl kv" });
      var body = h("tbody");
      Object.keys(f).forEach(function (k) {
        if (k === "shapes") return;
        body.appendChild(h("tr", null, [h("td", { text: k }), h("td", { text: fmtCell(f[k]) })]));
      });
      t.appendChild(body);
      return t;
    }

    function drawTimeline() {
      if (!seriesKeys.length) return;
      timeline.innerHTML = "";
      var vis = visibleSeries();
      var W = 760, H = 210, padL = 46, padR = 12, padT = 12, padB = 26;
      var ymin = Infinity, ymax = -Infinity;
      vis.forEach(function (k) {
        frames.forEach(function (f) {
          var v = f && f[k];
          if (typeof v === "number" && isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
        });
      });
      if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
      if (ymin === ymax) { ymax = ymin + 1; }
      var x0 = xs[0] || 0, x1 = xs[n - 1] || 1, xspan = (x1 - x0) || 1;
      function sx(v) { return padL + ((v - x0) / xspan) * (W - padL - padR); }
      function sy(v) { return H - padB - ((v - ymin) / (ymax - ymin)) * (H - padT - padB); }
      var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, class: "chart-svg", preserveAspectRatio: "xMidYMid meet" });
      svg.appendChild(sv("line", { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: "#cbd5e1" }));
      svg.appendChild(sv("line", { x1: padL, y1: padT, x2: padL, y2: H - padB, stroke: "#cbd5e1" }));
      [ymin, ymax].forEach(function (v) {
        var tx = sv("text", { x: padL - 6, y: sy(v), "text-anchor": "end", "dominant-baseline": "middle", "font-size": 10, fill: "#64748b" });
        tx.textContent = fmt(v); svg.appendChild(tx);
      });
      vis.forEach(function (k) {
        var col = palette(seriesKeys.indexOf(k));
        var pts = [];
        frames.forEach(function (f, i) {
          var v = f && f[k];
          if (typeof v === "number" && isFinite(v)) pts.push(sx(xs[i]) + "," + sy(v));
        });
        svg.appendChild(sv("polyline", { points: pts.join(" "), fill: "none", stroke: col, "stroke-width": 1.8 }));
      });
      var cx = sx(xs[idx] != null ? xs[idx] : x0);
      svg.appendChild(sv("line", { x1: cx, y1: padT, x2: cx, y2: H - padB, stroke: "#ef4444", "stroke-width": 1, "stroke-dasharray": "4,3" }));
      timeline.appendChild(svg);

      var leg = h("div", { class: "legend" });
      seriesKeys.forEach(function (k) {
        var on = featured ? (k === featured) : !hidden[k];
        leg.appendChild(h("span", { class: "legend-item" + (on ? "" : " off"), onclick: function () {
          if (featured) return; // featured mode is driven by the select control
          if (hidden[k]) delete hidden[k]; else hidden[k] = true;
          syncToggle(k);
          drawTimeline();
        } }, [h("span", { class: "swatch", style: "background:" + palette(seriesKeys.indexOf(k)) }), k]));
      });
      timeline.appendChild(leg);
    }

    function syncToggle(key) {
      controls.forEach(function (c) {
        if (c.kind === "toggle" && c.target === key && controlInputs[c.id]) {
          var on = !hidden[key];
          controlInputs[c.id].checked = on;
          controlState[c.id] = on;
        }
      });
    }

    function tickFwd() { if (idx < n - 1) { idx++; drawFrame(); } else stop(); }
    function schedule() { clearInterval(timer); timer = setInterval(tickFwd, 1000 / speed()); }
    function togglePlay() { if (playing) stop(); else play(); }
    function play() { if (playing || n === 0) return; if (idx >= n - 1) idx = 0; playing = true; playBtn.textContent = "\u23F8"; schedule(); }
    function stop() { playing = false; playBtn.textContent = "\u25B6"; clearInterval(timer); }
  }

  // ================================================================= RESULTS
  function initResults() {
    var result = PAYLOAD.result;
    var cont = h("div");
    var raw = h("pre", { class: "raw" });
    raw.textContent = JSON.stringify(result, null, 2);
    var resPanel = h("div", { class: "panel" }, [h("h2", { text: "Results" }), cont]);
    var rawPanel = h("div", { class: "panel" }, [h("h2", { text: "Raw JSON" }), raw]);

    var cp = buildControlPanel(render);
    if (cp) app.appendChild(cp);
    app.appendChild(resPanel);
    app.appendChild(rawPanel);
    render();

    function onlySection() {
      var sc = controls.find(function (c) { return c.kind === "select" && c.target === "section"; });
      return sc ? controlState[sc.id] : null;
    }
    function hiddenSections() {
      var set = {};
      controls.forEach(function (c) {
        if (c.kind === "toggle" && c.target && c.target !== "rawJson" && !controlState[c.id]) set[c.target] = true;
      });
      return set;
    }
    function sectionVisible(k) {
      var only = onlySection();
      if (only && only !== "all" && k !== only) return false;
      return !hiddenSections()[k];
    }
    function showRaw() {
      var rc = controls.find(function (c) { return c.kind === "toggle" && c.target === "rawJson"; });
      return rc ? !!controlState[rc.id] : true;
    }

    function render() {
      cont.innerHTML = "";
      if (result && typeof result === "object" && !Array.isArray(result)) {
        var scalarKeys = [], complexKeys = [];
        Object.keys(result).forEach(function (k) {
          var v = result[k];
          if (v != null && typeof v === "object") complexKeys.push(k); else scalarKeys.push(k);
        });
        var vs = scalarKeys.filter(sectionVisible);
        if (vs.length) cont.appendChild(h("div", { class: "cards" }, vs.map(function (k) { return metricCard(k, result[k]); })));
        complexKeys.filter(sectionVisible).forEach(function (k) {
          cont.appendChild(h("h3", { class: "sect-title", text: k }));
          renderValue(cont, result[k]);
        });
      } else {
        renderValue(cont, result);
      }
      rawPanel.style.display = showRaw() ? "" : "none";
    }
  }

  function metricCard(label, val) {
    return h("div", { class: "card" }, [h("div", { class: "card-label", text: label }), h("div", { class: "card-value", text: fmtv(val) })]);
  }
  function objectTable(arr) {
    var cols = [], seen = {};
    arr.forEach(function (o) { if (o && typeof o === "object" && !Array.isArray(o)) Object.keys(o).forEach(function (k) { if (!seen[k]) { seen[k] = 1; cols.push(k); } }); });
    return h("table", { class: "tbl" }, [
      h("thead", null, [h("tr", null, cols.map(function (c) { return h("th", { text: c }); }))]),
      h("tbody", null, arr.map(function (o) {
        return h("tr", null, cols.map(function (c) { return h("td", { text: (o && typeof o === "object") ? fmtCell(o[c]) : fmtCell(o) }); }));
      }))
    ]);
  }
  function renderValue(container, value) {
    if (Array.isArray(value)) {
      if (value.length && value.every(function (x) { return x && typeof x === "object" && !Array.isArray(x); })) {
        container.appendChild(objectTable(value));
      } else {
        container.appendChild(h("div", { class: "inline-list", text: value.map(fmtCell).join(", ") || "(empty)" }));
      }
    } else if (value && typeof value === "object") {
      var t = h("table", { class: "tbl kv" });
      var body = h("tbody");
      Object.keys(value).forEach(function (k) { body.appendChild(h("tr", null, [h("td", { text: k }), h("td", { text: fmtCell(value[k]) })])); });
      t.appendChild(body);
      container.appendChild(t);
    } else {
      container.appendChild(metricCard("value", value));
    }
  }

  // --- shared sim helpers ---
  function collectNumericKeys(frames) {
    var keys = [], seen = {};
    frames.forEach(function (f) {
      if (f && typeof f === "object" && !Array.isArray(f)) {
        Object.keys(f).forEach(function (k) {
          if (k === "t" || k === "tick" || k === "shapes" || k === "caption") return;
          var v = f[k];
          if (typeof v === "number" && isFinite(v) && !seen[k]) { seen[k] = 1; keys.push(k); }
        });
      }
    });
    return keys;
  }
  function shapeBounds(frames) {
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    function acc(x, y) { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y; }
    frames.forEach(function (f) {
      if (!f || !Array.isArray(f.shapes)) return;
      f.shapes.forEach(function (sh) {
        if (sh.kind === "circle") { var r = sh.r || 0; acc(sh.x - r, sh.y - r); acc(sh.x + r, sh.y + r); }
        else if (sh.kind === "rect") { acc(sh.x, sh.y); acc(sh.x + (sh.w || 0), sh.y + (sh.h || 0)); }
        else if (sh.kind === "line") { acc(sh.x1, sh.y1); acc(sh.x2, sh.y2); }
        else if (sh.kind === "text") { acc(sh.x, sh.y); }
      });
    });
    if (!isFinite(minX)) return { x: 0, y: 0, w: 800, h: 500 };
    var pad = 20;
    return { x: minX - pad, y: minY - pad, w: (maxX - minX) + 2 * pad, h: (maxY - minY) + 2 * pad };
  }
  function renderShape(sh) {
    if (sh.kind === "circle") return sv("circle", { cx: sh.x, cy: sh.y, r: sh.r, fill: sh.fill || "#94a3b8", stroke: sh.stroke, "stroke-width": sh.strokeWidth, opacity: sh.opacity });
    if (sh.kind === "rect") return sv("rect", { x: sh.x, y: sh.y, width: sh.w, height: sh.h, rx: sh.rx, fill: sh.fill || "#94a3b8", stroke: sh.stroke, "stroke-width": sh.strokeWidth, opacity: sh.opacity });
    if (sh.kind === "line") return sv("line", { x1: sh.x1, y1: sh.y1, x2: sh.x2, y2: sh.y2, stroke: sh.stroke || "#94a3b8", "stroke-width": sh.strokeWidth, "stroke-dasharray": sh.dasharray, opacity: sh.opacity });
    if (sh.kind === "text") { var t = sv("text", { x: sh.x, y: sh.y, "font-size": sh.fontSize || 12, fill: sh.fill || "#0f172a", "text-anchor": sh.anchor, "font-weight": sh.fontWeight, "font-family": sh.fontFamily }); t.textContent = sh.text || ""; return t; }
    if (sh.kind === "path") return sv("path", { d: sh.d, stroke: sh.stroke, "stroke-width": sh.strokeWidth, fill: sh.fill || "none", opacity: sh.opacity });
    return null;
  }
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::plugin::manifest::{
        OutputKind, PluginRuntimeKind, PluginTransportKind, RunSpec, UiControl,
    };
    use crate::des::plugin::runner::PluginOutput;

    fn sim_manifest() -> PluginManifest {
        PluginManifest {
            id: "mm1".to_string(),
            name: "M/M/1 queue".to_string(),
            version: "1.0.0".to_string(),
            description: "queue length over time".to_string(),
            runtime: PluginRuntimeKind::Rust,
            transport: PluginTransportKind::Stdio,
            language: None,
            run: RunSpec::new("./mm1", &[]),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls: vec![
                UiControl::toggle("show_n", "Show n", true, Some("n")),
                UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
            ],
            title: None,
        }
    }

    fn run_with(output: PluginOutput) -> PluginRun {
        PluginRun {
            plugin_id: "mm1".to_string(),
            output,
            exit_code: Some(0),
            stderr: String::new(),
        }
    }

    #[test]
    fn sim_html_embeds_frames_and_controls() {
        let frames = vec![
            json!({"t": 0.0, "n": 1}),
            json!({"t": 1.0, "n": 3}),
            json!({"t": 2.0, "n": 2}),
        ];
        let html = render_player_html(&sim_manifest(), &run_with(PluginOutput::Jsonl(frames)));
        assert!(html.contains("id=\"plugin-payload\""));
        assert!(html.contains("\"player\":\"sim\""));
        assert!(html.contains("M/M/1 queue"));
        assert!(html.contains("\"speed\""));
        // frame data made it into the embedded blob.
        assert!(html.contains("\"n\":3"));
    }

    #[test]
    fn results_html_uses_results_player() {
        let mut m = sim_manifest();
        m.player = PlayerKind::Results;
        m.output = OutputKind::Json;
        let value = json!({"objective": 42.0, "x": [1.0, 2.0], "status": "optimal"});
        let html = render_player_html(&m, &run_with(PluginOutput::Json(value)));
        assert!(html.contains("\"player\":\"results\""));
        assert!(html.contains("\"objective\":42"));
    }

    #[test]
    fn script_close_tag_in_data_is_escaped() {
        let mut m = sim_manifest();
        m.player = PlayerKind::Results;
        m.output = OutputKind::Json;
        let value = json!({"note": "</script><b>x"});
        let html = render_player_html(&m, &run_with(PluginOutput::Json(value)));
        // the literal closing tag must not appear inside the embedded blob.
        assert!(html.contains("<\\/script><b>x"));
    }

    #[test]
    fn json_doc_with_frames_field_is_normalized_for_sim() {
        let value = json!({"frames": [{"t": 0, "n": 1}, {"t": 1, "n": 2}]});
        let frames = normalize_frames(&PluginOutput::Json(value));
        assert_eq!(frames.len(), 2);
    }
}
