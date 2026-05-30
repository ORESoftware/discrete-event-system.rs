//! Port of `src/des/animation/html-player.ts`.
//!
//! Renders an [`Animation`] into a single self-contained HTML string: the
//! animation JSON is embedded in a `<script type="application/json">` blob and
//! a vanilla-JS player (no CDN / build step) draws each frame as SVG with
//! play / pause / step / scrub controls and optional time-series charts.
//!
//! ## Notes
//!
//! * `serde` is not available in this crate, so `JSON.stringify(value)` becomes
//!   [`JsonValue::to_string`] (compact) followed by the same three escaping
//!   replacements (`</script` → `<\/script`, U+2028 / U+2029).
//! * The large embedded `TEMPLATE` literal is a raw string; the
//!   `__PLACEHOLDER__` tokens are filled with [`str::replace`] (the same tokens,
//!   so output is byte-identical to the TS version).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::animation::types::Animation;
use crate::des::observability::logger::JsonValue;

/// One selectable animation in a multi-variant page.
#[derive(Clone, Debug, Default)]
pub struct AnimationVariant {
    pub id: String,
    pub label: String,
    pub animation: Animation,
    pub summary: Option<String>,
    /// Keyed control values (e.g. `{"policy": "greedy"}`). NOTE: a `HashMap`
    /// per the migration header — iteration order is not preserved, so the
    /// selector ordering in a multi-control page is unspecified (the single-
    /// variant `buildHTML` path is unaffected).
    pub controls: Option<HashMap<String, String>>,
}

/// Options for [`build_html_set`].
#[derive(Clone, Debug, Default)]
pub struct AnimationSetOptions {
    pub title: Option<String>,
    pub subtitle: Option<String>,
    pub selector_label: Option<String>,
}

/// Build a complete, self-contained HTML page for a single animation.
pub fn build_html(anim: &Animation) -> String {
    // Embed JSON in <script type="application/json">. Two precautions:
    //   1. Escape `</` to `<\/` so a literal "</script>" in any string field
    //      does NOT prematurely terminate the inline <script> block. JSON
    //      parses "\/" as "/" so this is loss-free.
    //   2. Escape U+2028 / U+2029 which are valid in JSON strings but
    //      illegal in JS string literals.
    let json = json_for_script(&anim.to_json());
    let title = escape_html(anim.title.as_deref().unwrap_or("Simulation"));
    let subtitle = match &anim.subtitle {
        Some(s) => escape_html(s),
        None => String::new(),
    };
    TEMPLATE
        .replace("__TITLE__", &title)
        .replace("__SUBTITLE__", &subtitle)
        .replace("__ANIMATION_JSON__", &json)
}

/// Build a page that lets the viewer switch between several animation variants.
///
/// Panics on an empty `variants` slice (a programmer error, matching the TS
/// `throw`).
pub fn build_html_set(variants: &[AnimationVariant], opts: &AnimationSetOptions) -> String {
    if variants.is_empty() {
        panic!("build_html_set requires at least one animation variant");
    }
    let title = escape_html(
        opts.title
            .as_deref()
            .or(variants[0].animation.title.as_deref())
            .unwrap_or("Simulation"),
    );
    let subtitle = escape_html(
        opts.subtitle
            .as_deref()
            .or(variants[0].animation.subtitle.as_deref())
            .unwrap_or(""),
    );

    let variant_json: Vec<JsonValue> = variants.iter().map(variant_to_json).collect();
    let payload = JsonValue::Object(vec![
        ("variants".to_string(), JsonValue::Array(variant_json)),
        (
            "selectorLabel".to_string(),
            JsonValue::String(opts.selector_label.clone().unwrap_or_else(|| "variant".to_string())),
        ),
    ]);
    let json = json_for_script(&payload);
    TEMPLATE
        .replace("__TITLE__", &title)
        .replace("__SUBTITLE__", &subtitle)
        .replace("__ANIMATION_JSON__", &json)
}

fn variant_to_json(v: &AnimationVariant) -> JsonValue {
    let mut o: Vec<(String, JsonValue)> = vec![
        ("id".to_string(), JsonValue::String(v.id.clone())),
        ("label".to_string(), JsonValue::String(v.label.clone())),
        ("animation".to_string(), v.animation.to_json()),
    ];
    if let Some(s) = &v.summary {
        o.push(("summary".to_string(), JsonValue::String(s.clone())));
    }
    if let Some(controls) = &v.controls {
        let entries: Vec<(String, JsonValue)> = controls
            .iter()
            .map(|(k, val)| (k.clone(), JsonValue::String(val.clone())))
            .collect();
        o.push(("controls".to_string(), JsonValue::Object(entries)));
    }
    JsonValue::Object(o)
}

fn json_for_script(value: &JsonValue) -> String {
    let s = value.to_string();
    let s = escape_script_close(&s);
    let s = s.replace('\u{2028}', "\\u2028");
    s.replace('\u{2029}', "\\u2029")
}

/// Replace `</` with `<\/` when immediately followed by `script` (case-
/// insensitive), mirroring the TS regex `/<\/(?=script)/gi`.
fn escape_script_close(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '/' {
            let end = (i + 2 + 6).min(chars.len());
            let follows: String = chars[i + 2..end].iter().collect();
            if follows.to_ascii_lowercase().starts_with("script") {
                out.push_str("<\\/");
                i += 2;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// -----------------------------------------------------------------------------
// HTML / CSS / JS template. Uses __PLACEHOLDERS__ instead of interpolation so
// `${...}` and backslashes pass through unmolested.
// -----------------------------------------------------------------------------
const TEMPLATE: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #f5f5f7;
    color: #111;
  }
  header {
    padding: 16px 24px 8px;
    border-bottom: 1px solid #ddd;
    background: #fafafa;
  }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .subtitle { margin: 4px 0 0; font-size: 13px; color: #666; }
  main {
    padding: 16px 24px;
    box-sizing: border-box;
    max-width: 100vw;
  }
  .stage-wrap {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 8px;
    display: block;
    box-sizing: border-box;
    width: 100%;
    max-width: calc(100vw - 48px);
    overflow: hidden;
  }
  svg.stage {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
  }
  .controls {
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  .controls button {
    padding: 6px 14px;
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }
  .controls button:hover { background: #eee; }
  .controls button:disabled { opacity: 0.4; cursor: default; }
  .controls input[type=range] { flex: 1; min-width: 200px; }
  .controls .ts-readout {
    font-family: SF Mono, Menlo, Consolas, monospace;
    color: #333;
    min-width: 220px;
  }
  .controls select {
    padding: 4px 6px;
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 4px;
    font-size: 13px;
  }
  .variant-controls {
    margin: 0 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  .variant-controls select {
    padding: 6px 8px;
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 4px;
    font-size: 13px;
  }
  .variant-selectors {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .variant-summary {
    color: #555;
    font-size: 12px;
  }
  .caption {
    margin-top: 6px;
    font-family: SF Mono, Menlo, Consolas, monospace;
    color: #444;
    font-size: 12px;
    min-height: 1.2em;
  }
  footer {
    padding: 12px 24px;
    color: #999;
    font-size: 12px;
    border-top: 1px solid #eee;
  }
  @media (max-width: 700px) {
    header { padding: 12px 14px 8px; }
    main { padding: 12px 14px; }
    .stage-wrap { max-width: calc(100vw - 28px); padding: 6px; }
    .controls { gap: 8px; }
    .controls button { padding: 6px 10px; }
    .controls .ts-readout { min-width: 160px; }
    .variant-controls { gap: 8px; }
  }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p class="subtitle">__SUBTITLE__</p>
</header>
<main>
  <div class="variant-controls" id="variant-controls" hidden>
    <span class="variant-selectors" id="variant-selectors"></span>
    <span class="variant-summary" id="variant-summary"></span>
  </div>
  <div class="stage-wrap"><svg id="stage" class="stage"></svg></div>
  <div class="caption" id="caption">&nbsp;</div>
  <div class="controls">
    <button id="play">Play</button>
    <button id="step-back">&laquo;</button>
    <button id="step-fwd">&raquo;</button>
    <input id="scrub" type="range" min="0" value="0" step="1">
    <span class="ts-readout" id="readout"></span>
    <label>speed
      <select id="speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
        <option value="5">5x</option>
        <option value="8">8x</option>
        <option value="10">10x</option>
        <option value="15">15x</option>
        <option value="16">16x</option>
      </select>
    </label>
  </div>
</main>
<footer>
  Generated by the DES animation plugin (<code>src/des/animation/</code>).
  Press space to play/pause, &larr;/&rarr; to step.
</footer>

<script type="application/json" id="anim-data">__ANIMATION_JSON__</script>
<script>
(function() {
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const PAYLOAD = JSON.parse(document.getElementById('anim-data').textContent);
  const VARIANTS = Array.isArray(PAYLOAD.variants) ? PAYLOAD.variants : null;
  let ANIM = VARIANTS ? VARIANTS[0].animation : PAYLOAD;
  const stage = document.getElementById('stage');
  const caption = document.getElementById('caption');
  const scrub = document.getElementById('scrub');
  const readout = document.getElementById('readout');
  const playBtn = document.getElementById('play');
  const speedSel = document.getElementById('speed');
  const stepBack = document.getElementById('step-back');
  const stepFwd  = document.getElementById('step-fwd');
  const variantControls = document.getElementById('variant-controls');
  const variantSelectors = document.getElementById('variant-selectors');
  const variantSummary = document.getElementById('variant-summary');

  let N = ANIM.frames.length;
  let i = 0;
  let playing = false;
  let lastTimestamp = null;
  let speed = 1;

  function applyAnimationConfig() {
    N = ANIM.frames.length;
    scrub.max = String(Math.max(0, N - 1));
    stage.setAttribute('width', ANIM.width);
    stage.setAttribute('height', ANIM.height);
    stage.setAttribute('viewBox', '0 0 ' + ANIM.width + ' ' + ANIM.height);
    stage.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    stage.style.background = ANIM.background || '#fff';
  }

  // --- Rendering -----------------------------------------------------------

  function clearStage() {
    while (stage.firstChild) stage.removeChild(stage.firstChild);
  }

  function applyAttrs(el, attrs) {
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) el.setAttribute(k, attrs[k]);
  }

  function renderShape(s) {
    if (s.kind === 'circle') {
      const c = document.createElementNS(SVG_NS, 'circle');
      applyAttrs(c, {cx: s.x, cy: s.y, r: s.r, fill: s.fill, stroke: s.stroke, 'stroke-width': s.strokeWidth, opacity: s.opacity});
      if (s.title) {
        const t = document.createElementNS(SVG_NS, 'title');
        t.textContent = s.title;
        c.appendChild(t);
      }
      stage.appendChild(c);
      if (s.label) {
        const text = document.createElementNS(SVG_NS, 'text');
        applyAttrs(text, {x: s.x, y: s.y + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#fff', 'font-weight': 'bold'});
        text.textContent = s.label;
        stage.appendChild(text);
      }
    } else if (s.kind === 'rect') {
      const r = document.createElementNS(SVG_NS, 'rect');
      applyAttrs(r, {x: s.x, y: s.y, width: s.w, height: s.h, fill: s.fill, stroke: s.stroke, 'stroke-width': s.strokeWidth, opacity: s.opacity, rx: s.rx});
      if (s.title) {
        const t = document.createElementNS(SVG_NS, 'title');
        t.textContent = s.title;
        r.appendChild(t);
      }
      stage.appendChild(r);
      if (s.label) {
        const text = document.createElementNS(SVG_NS, 'text');
        applyAttrs(text, {x: s.x + s.w / 2, y: s.y + s.h / 2 + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#fff'});
        text.textContent = s.label;
        stage.appendChild(text);
      }
    } else if (s.kind === 'line') {
      const l = document.createElementNS(SVG_NS, 'line');
      applyAttrs(l, {x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2, stroke: s.stroke, 'stroke-width': s.strokeWidth, opacity: s.opacity, 'stroke-dasharray': s.dasharray});
      stage.appendChild(l);
    } else if (s.kind === 'text') {
      const t = document.createElementNS(SVG_NS, 'text');
      applyAttrs(t, {x: s.x, y: s.y, 'text-anchor': s.anchor || 'start', 'font-size': s.fontSize || 12, fill: s.fill || '#000', 'font-weight': s.fontWeight, 'font-family': s.fontFamily});
      t.textContent = s.text;
      stage.appendChild(t);
    } else if (s.kind === 'path') {
      const p = document.createElementNS(SVG_NS, 'path');
      applyAttrs(p, {d: s.d, stroke: s.stroke, 'stroke-width': s.strokeWidth, fill: s.fill || 'none', opacity: s.opacity});
      stage.appendChild(p);
    }
  }

  function renderChart(c, currentT) {
    const x0 = c.x, y0 = c.y, w = c.w, h = c.h;
    const bg = document.createElementNS(SVG_NS, 'rect');
    applyAttrs(bg, {x: x0, y: y0, width: w, height: h, fill: '#fafafa', stroke: '#ccc', 'stroke-width': 1});
    stage.appendChild(bg);

    if (c.title) {
      const t = document.createElementNS(SVG_NS, 'text');
      applyAttrs(t, {x: x0 + 6, y: y0 + 14, 'font-size': 11, fill: '#444', 'font-weight': 'bold'});
      t.textContent = c.title;
      stage.appendChild(t);
    }

    let yMin = c.yMin, yMax = c.yMax;
    if (yMin === undefined || yMax === undefined) {
      let vMin = Infinity, vMax = -Infinity;
      for (const s of c.series) {
        for (const v of s.y) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
      }
      if (yMin === undefined) yMin = vMin === Infinity ? 0 : vMin;
      if (yMax === undefined) yMax = vMax === -Infinity ? 1 : vMax;
      if (yMax <= yMin) yMax = yMin + 1;
    }
    let tMin = Infinity, tMax = -Infinity;
    for (const s of c.series) {
      for (const t of s.t) { if (t < tMin) tMin = t; if (t > tMax) tMax = t; }
    }
    if (tMin === Infinity) { tMin = 0; tMax = 1; }
    if (tMax <= tMin) tMax = tMin + 1;

    const padTop = c.title ? 22 : 8, padBot = 18, padL = 36, padR = 8;
    const ix = x0 + padL, iy = y0 + padTop;
    const iw = w - padL - padR, ih = h - padTop - padBot;
    const sx = t => ix + iw * (t - tMin) / (tMax - tMin);
    const sy = v => iy + ih * (1 - (v - yMin) / (yMax - yMin));

    // Axes.
    const yAx = document.createElementNS(SVG_NS, 'line');
    applyAttrs(yAx, {x1: ix, y1: iy, x2: ix, y2: iy + ih, stroke: '#999', 'stroke-width': 1});
    stage.appendChild(yAx);
    const xAx = document.createElementNS(SVG_NS, 'line');
    applyAttrs(xAx, {x1: ix, y1: iy + ih, x2: ix + iw, y2: iy + ih, stroke: '#999', 'stroke-width': 1});
    stage.appendChild(xAx);

    // Y-axis labels.
    [yMin, yMax].forEach((v, k) => {
      const tx = document.createElementNS(SVG_NS, 'text');
      applyAttrs(tx, {x: ix - 4, y: k === 0 ? (iy + ih + 3) : (iy + 8), 'text-anchor': 'end', 'font-size': 10, fill: '#666'});
      tx.textContent = v.toLocaleString();
      stage.appendChild(tx);
    });

    // Series, clipped to t <= currentT.
    for (const s of c.series) {
      let d = '';
      for (let k = 0; k < s.t.length; k++) {
        if (s.t[k] > currentT) break;
        d += (k === 0 ? 'M' : 'L') + sx(s.t[k]).toFixed(2) + ',' + sy(s.y[k]).toFixed(2) + ' ';
      }
      if (d) {
        const p = document.createElementNS(SVG_NS, 'path');
        applyAttrs(p, {d: d, stroke: s.color, 'stroke-width': 1.5, fill: 'none'});
        stage.appendChild(p);
      }
    }

    // Legend.
    let legendY = iy + 14;
    for (const s of c.series) {
      const sw = document.createElementNS(SVG_NS, 'rect');
      applyAttrs(sw, {x: ix + 4, y: legendY - 8, width: 8, height: 8, fill: s.color});
      stage.appendChild(sw);
      const tx = document.createElementNS(SVG_NS, 'text');
      applyAttrs(tx, {x: ix + 16, y: legendY, 'font-size': 10, fill: '#444'});
      tx.textContent = s.label;
      stage.appendChild(tx);
      legendY += 12;
    }

    // Cursor.
    if (c.cursor !== false) {
      const cx = sx(currentT);
      if (cx >= ix && cx <= ix + iw) {
        const cur = document.createElementNS(SVG_NS, 'line');
        applyAttrs(cur, {x1: cx, y1: iy, x2: cx, y2: iy + ih, stroke: '#d22', 'stroke-width': 1, 'stroke-dasharray': '3,2'});
        stage.appendChild(cur);
      }
    }
  }

  function render(idx) {
    if (N === 0) {
      clearStage();
      caption.textContent = 'No frames';
      readout.textContent = 'frame 0 / 0';
      return;
    }
    const f = ANIM.frames[idx];
    clearStage();
    for (const s of f.shapes) renderShape(s);
    if (ANIM.charts) for (const c of ANIM.charts) renderChart(c, f.t);
    caption.textContent = f.caption || '\u00a0';
    readout.textContent = 'frame ' + (idx + 1) + ' / ' + N + '   t=' + f.t.toFixed(3) + '   tick=' + f.tick;
    scrub.value = String(idx);
  }

  // --- Playback loop -------------------------------------------------------

  function tick(ts) {
    if (!playing) { lastTimestamp = null; return; }
    if (lastTimestamp === null) lastTimestamp = ts;
    const dt = (ts - lastTimestamp) / 1000;
    const advance = dt * ANIM.fps * speed;
    if (advance >= 1) {
      i = Math.min(N - 1, i + Math.floor(advance));
      lastTimestamp = ts;
      render(i);
      if (i >= N - 1) { playing = false; playBtn.textContent = 'Play'; lastTimestamp = null; return; }
    }
    requestAnimationFrame(tick);
  }

  function setPlaying(p) {
    if (N === 0) p = false;
    playing = p;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) {
      if (i >= N - 1) i = 0;
      lastTimestamp = null;
      requestAnimationFrame(tick);
    }
  }

  // --- Wiring --------------------------------------------------------------

  function selectVariant(idx) {
    if (!VARIANTS) return;
    setPlaying(false);
    const variant = VARIANTS[idx];
    ANIM = variant.animation;
    i = 0;
    applyAnimationConfig();
    syncVariantControls(variant);
    if (variantSummary) variantSummary.textContent = variant.summary || '';
    render(i);
  }

  function displayControlName(name) {
    return String(name).replace(/[-_]+/g, ' ');
  }

  function controlKeys() {
    if (!VARIANTS || VARIANTS.length === 0) return [];
    const seen = new Set();
    const keys = [];
    for (const variant of VARIANTS) {
      if (!variant.controls) return [];
      for (const key of Object.keys(variant.controls)) {
        if (!seen.has(key)) {
          seen.add(key);
          keys.push(key);
        }
      }
    }
    return keys;
  }

  function uniqueControlValues(key) {
    const seen = new Set();
    const values = [];
    for (const variant of VARIANTS || []) {
      const value = variant.controls && variant.controls[key];
      if (value !== undefined && !seen.has(value)) {
        seen.add(value);
        values.push(value);
      }
    }
    return values;
  }

  function syncVariantControls(variant) {
    if (!variantSelectors || !variant || !variant.controls) return;
    const selects = variantSelectors.querySelectorAll('select[data-variant-control]');
    selects.forEach(function(select) {
      const key = select.getAttribute('data-variant-control');
      if (key && variant.controls[key] !== undefined) select.value = variant.controls[key];
    });
  }

  function selectVariantByControls() {
    if (!VARIANTS || !variantSelectors) return;
    const wanted = {};
    const selects = variantSelectors.querySelectorAll('select[data-variant-control]');
    selects.forEach(function(select) {
      const key = select.getAttribute('data-variant-control');
      if (key) wanted[key] = select.value;
    });
    const match = VARIANTS.findIndex(function(variant) {
      if (!variant.controls) return false;
      for (const key in wanted) {
        if (variant.controls[key] !== wanted[key]) return false;
      }
      return true;
    });
    if (match >= 0) selectVariant(match);
  }

  function populateVariantControls() {
    if (!VARIANTS || !variantSelectors) return;
    while (variantSelectors.firstChild) variantSelectors.removeChild(variantSelectors.firstChild);
    const keys = controlKeys();
    if (keys.length > 0) {
      for (const key of keys) {
        const label = document.createElement('label');
        label.textContent = displayControlName(key) + ' ';
        const select = document.createElement('select');
        select.setAttribute('data-variant-control', key);
        for (const value of uniqueControlValues(key)) {
          const option = document.createElement('option');
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        }
        select.addEventListener('change', selectVariantByControls);
        label.appendChild(select);
        variantSelectors.appendChild(label);
      }
      syncVariantControls(VARIANTS[0]);
      return;
    }

    const label = document.createElement('label');
    label.textContent = (PAYLOAD.selectorLabel || 'variant') + ' ';
    const select = document.createElement('select');
    select.id = 'variant-select';
    for (let k = 0; k < VARIANTS.length; k++) {
      const option = document.createElement('option');
      option.value = String(k);
      option.textContent = VARIANTS[k].label || VARIANTS[k].id || ('variant ' + (k + 1));
      select.appendChild(option);
    }
    select.addEventListener('change', function() { selectVariant(+select.value); });
    label.appendChild(select);
    variantSelectors.appendChild(label);
  }

  if (VARIANTS && VARIANTS.length > 0) {
    variantControls.hidden = false;
    populateVariantControls();
    variantSummary.textContent = VARIANTS[0].summary || '';
  }

  playBtn.addEventListener('click', function() { setPlaying(!playing); });
  scrub.addEventListener('input', function() { i = +scrub.value; setPlaying(false); render(i); });
  speedSel.addEventListener('change', function() { speed = +speedSel.value; });
  stepBack.addEventListener('click', function() { setPlaying(false); i = Math.max(0, i - 1); render(i); });
  stepFwd.addEventListener('click', function() { setPlaying(false); i = Math.min(N - 1, i + 1); render(i); });
  document.addEventListener('keydown', function(e) {
    if (e.key === ' ') { setPlaying(!playing); e.preventDefault(); }
    else if (e.key === 'ArrowLeft')  { setPlaying(false); i = Math.max(0, i - 1); render(i); }
    else if (e.key === 'ArrowRight') { setPlaying(false); i = Math.min(N - 1, i + 1); render(i); }
  });

  applyAnimationConfig();
  if (N > 0) render(0);
})();
</script>
</body>
</html>
"####;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::animation::types::{Frame, Shape, TextShape};

    fn tiny_anim() -> Animation {
        Animation {
            width: 100.0,
            height: 80.0,
            fps: 30.0,
            title: Some("Hi <there>".to_string()),
            subtitle: None,
            frames: vec![Frame {
                t: 0.0,
                tick: 0.0,
                shapes: vec![Shape::Text(TextShape {
                    x: 1.0,
                    y: 2.0,
                    text: "</script>".to_string(),
                    ..Default::default()
                })],
                caption: None,
            }],
            charts: None,
            background: None,
        }
    }

    #[test]
    fn build_html_fills_placeholders_and_escapes() {
        let html = build_html(&tiny_anim());
        // Title is HTML-escaped in both <title> and <h1>.
        assert!(html.contains("<title>Hi &lt;there&gt;</title>"));
        assert!(html.contains("<h1>Hi &lt;there&gt;</h1>"));
        // No leftover placeholder tokens.
        assert!(!html.contains("__TITLE__"));
        assert!(!html.contains("__ANIMATION_JSON__"));
        // The embedded JSON had `</script>` neutralised to `<\/script>`.
        assert!(html.contains("<\\/script>"));
    }

    #[test]
    fn escape_html_order() {
        assert_eq!(escape_html("a&b<c>\"d'"), "a&amp;b&lt;c&gt;&quot;d&#39;");
    }

    #[test]
    #[should_panic]
    fn build_html_set_rejects_empty() {
        build_html_set(&[], &AnimationSetOptions::default());
    }
}
