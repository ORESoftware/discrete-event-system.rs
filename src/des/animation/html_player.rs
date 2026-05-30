//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/animation/html-player.ts`
//! Rust target: `src/des/animation/html_player.rs`

#![allow(dead_code)]

use std::collections::BTreeMap;

use super::types::Animation;
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/animation/html-player.ts",
    "src/des/animation/html_player.rs",
    &["RUST MIGRATION:", "- Target: src/des/animation/html_player.rs", "- Keep AnimationVariant and AnimationSetOptions as serde structs; buildHTML/buildHTMLSet become build_html/build_html_set.", "- HTML/string builders should return Result<String, HtmlRenderError> and use a template writer rather than ad-hoc fallible panics.", "- jsonForScript and escapeHtml stay private module helpers; serde_json replaces JSON.stringify and explicit escaping stays tested.", "- The embedded JS template can remain a raw string constant or move behind an include_str! template without changing the module boundary.", "- embeds the Animation JSON as a `<script type=\"application/json\">` blob", "- uses vanilla JS (no CDN, no dependencies) to render frames as SVG", "- has a play / pause / step / scrub UI with a speed selector", "- draws optional time-series charts on the side, animated up to"],
    &["AnimationSetOptions", "AnimationVariant", "buildHTML", "buildHTMLSet"],
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationVariant {
    pub id: String,
    pub label: String,
    pub animation: Animation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controls: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationSetOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_label: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum HtmlRenderError {
    #[error("could not serialize animation JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("build_html_set requires at least one animation variant")]
    EmptyAnimationSet,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnimationSetPayload<'a> {
    variants: &'a [AnimationVariant],
    selector_label: &'a str,
}

pub fn build_html(anim: &Animation) -> Result<String, HtmlRenderError> {
    let json = json_for_script(anim)?;
    let title = escape_html(anim.title.as_deref().unwrap_or("Simulation"));
    let subtitle = escape_html(anim.subtitle.as_deref().unwrap_or(""));

    Ok(TEMPLATE
        .replace("__TITLE__", &title)
        .replace("__SUBTITLE__", &subtitle)
        .replace("__ANIMATION_JSON__", &json))
}

pub fn build_html_set(
    variants: &[AnimationVariant],
    opts: AnimationSetOptions,
) -> Result<String, HtmlRenderError> {
    let first = variants.first().ok_or(HtmlRenderError::EmptyAnimationSet)?;
    let selector_label = opts.selector_label.unwrap_or_else(|| "variant".to_owned());
    let payload = AnimationSetPayload {
        variants,
        selector_label: &selector_label,
    };
    let json = json_for_script(&payload)?;
    let title = escape_html(
        opts.title
            .as_deref()
            .or(first.animation.title.as_deref())
            .unwrap_or("Simulation"),
    );
    let subtitle = escape_html(
        opts.subtitle
            .as_deref()
            .or(first.animation.subtitle.as_deref())
            .unwrap_or(""),
    );

    Ok(TEMPLATE
        .replace("__TITLE__", &title)
        .replace("__SUBTITLE__", &subtitle)
        .replace("__ANIMATION_JSON__", &json))
}

fn json_for_script<T: Serialize + ?Sized>(value: &T) -> Result<String, serde_json::Error> {
    let json = serde_json::to_string(value)?;
    Ok(escape_script_tag_terminators(&json)
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
}

fn escape_script_tag_terminators(input: &str) -> String {
    const NEEDLE: &[u8] = b"</script";
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut last = 0usize;
    let mut i = 0usize;

    while i + NEEDLE.len() <= bytes.len() {
        if ascii_starts_with_ignore_case(&bytes[i..i + NEEDLE.len()], NEEDLE) {
            out.push_str(&input[last..i]);
            out.push_str("<\\/");
            last = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }

    if last == 0 {
        input.to_owned()
    } else {
        out.push_str(&input[last..]);
        out
    }
}

fn ascii_starts_with_ignore_case(candidate: &[u8], needle: &[u8]) -> bool {
    candidate.len() == needle.len()
        && candidate
            .iter()
            .zip(needle)
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
}

fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

const TEMPLATE: &str = r##"<!DOCTYPE html>
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
  main { padding: 16px 24px; box-sizing: border-box; max-width: 100vw; }
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
  svg.stage { display: block; width: 100%; max-width: 100%; height: auto; }
  .controls {
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  .controls button, .controls select, .variant-controls select {
    padding: 6px 10px;
    background: #fff;
    border: 1px solid #bbb;
    border-radius: 4px;
    font-size: 13px;
  }
  .controls button { cursor: pointer; }
  .controls button:hover { background: #eee; }
  .controls input[type=range] { flex: 1; min-width: 200px; }
  .controls .ts-readout {
    font-family: SF Mono, Menlo, Consolas, monospace;
    color: #333;
    min-width: 220px;
  }
  .variant-controls {
    margin: 0 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  .variant-selectors { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .variant-summary { color: #555; font-size: 12px; }
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
    .controls .ts-readout { min-width: 160px; }
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
        <option value="8">8x</option>
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

  function clearStage() {
    while (stage.firstChild) stage.removeChild(stage.firstChild);
  }

  function applyAttrs(el, attrs) {
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) el.setAttribute(k, attrs[k]);
  }

  function appendTitle(el, value) {
    if (!value) return;
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = value;
    el.appendChild(title);
  }

  function renderShape(s) {
    if (s.kind === 'circle') {
      const c = document.createElementNS(SVG_NS, 'circle');
      applyAttrs(c, {cx: s.x, cy: s.y, r: s.r, fill: s.fill, stroke: s.stroke, 'stroke-width': s.strokeWidth, opacity: s.opacity});
      appendTitle(c, s.title);
      stage.appendChild(c);
      if (s.label) {
        const label = document.createElementNS(SVG_NS, 'text');
        applyAttrs(label, {x: s.x, y: s.y + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#fff', 'font-weight': 'bold'});
        label.textContent = s.label;
        stage.appendChild(label);
      }
    } else if (s.kind === 'rect') {
      const r = document.createElementNS(SVG_NS, 'rect');
      applyAttrs(r, {x: s.x, y: s.y, width: s.w, height: s.h, fill: s.fill, stroke: s.stroke, 'stroke-width': s.strokeWidth, opacity: s.opacity, rx: s.rx});
      appendTitle(r, s.title);
      stage.appendChild(r);
      if (s.label) {
        const label = document.createElementNS(SVG_NS, 'text');
        applyAttrs(label, {x: s.x + s.w / 2, y: s.y + s.h / 2 + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#fff'});
        label.textContent = s.label;
        stage.appendChild(label);
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
      const title = document.createElementNS(SVG_NS, 'text');
      applyAttrs(title, {x: x0 + 6, y: y0 + 14, 'font-size': 11, fill: '#444', 'font-weight': 'bold'});
      title.textContent = c.title;
      stage.appendChild(title);
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
    const iw = Math.max(1, w - padL - padR), ih = Math.max(1, h - padTop - padBot);
    const sx = t => ix + iw * (t - tMin) / (tMax - tMin);
    const sy = v => iy + ih * (1 - (v - yMin) / (yMax - yMin));

    const yAxis = document.createElementNS(SVG_NS, 'line');
    applyAttrs(yAxis, {x1: ix, y1: iy, x2: ix, y2: iy + ih, stroke: '#999', 'stroke-width': 1});
    stage.appendChild(yAxis);
    const xAxis = document.createElementNS(SVG_NS, 'line');
    applyAttrs(xAxis, {x1: ix, y1: iy + ih, x2: ix + iw, y2: iy + ih, stroke: '#999', 'stroke-width': 1});
    stage.appendChild(xAxis);

    for (const s of c.series) {
      let d = '';
      for (let k = 0; k < s.t.length && k < s.y.length; k++) {
        if (s.t[k] > currentT) break;
        d += (d ? 'L' : 'M') + sx(s.t[k]).toFixed(2) + ',' + sy(s.y[k]).toFixed(2) + ' ';
      }
      if (d) {
        const p = document.createElementNS(SVG_NS, 'path');
        applyAttrs(p, {d: d, stroke: s.color, 'stroke-width': 1.5, fill: 'none'});
        stage.appendChild(p);
      }
    }

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
    const frame = ANIM.frames[idx];
    clearStage();
    for (const s of frame.shapes) renderShape(s);
    if (ANIM.charts) for (const chart of ANIM.charts) renderChart(chart, frame.t);
    caption.textContent = frame.caption || '\u00a0';
    readout.textContent = 'frame ' + (idx + 1) + ' / ' + N + '   t=' + frame.t.toFixed(3) + '   tick=' + frame.tick;
    scrub.value = String(idx);
  }

  function tick(ts) {
    if (!playing) { lastTimestamp = null; return; }
    if (lastTimestamp === null) lastTimestamp = ts;
    const advance = ((ts - lastTimestamp) / 1000) * ANIM.fps * speed;
    if (advance >= 1) {
      i = Math.min(N - 1, i + Math.floor(advance));
      lastTimestamp = ts;
      render(i);
      if (i >= N - 1) {
        playing = false;
        playBtn.textContent = 'Play';
        lastTimestamp = null;
        return;
      }
    }
    requestAnimationFrame(tick);
  }

  function setPlaying(next) {
    if (N === 0) next = false;
    playing = next;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) {
      if (i >= N - 1) i = 0;
      lastTimestamp = null;
      requestAnimationFrame(tick);
    }
  }

  function selectVariant(idx) {
    if (!VARIANTS) return;
    setPlaying(false);
    const variant = VARIANTS[idx];
    ANIM = variant.animation;
    i = 0;
    applyAnimationConfig();
    if (variantSummary) variantSummary.textContent = variant.summary || '';
    render(i);
  }

  function populateVariantControls() {
    if (!VARIANTS || !variantSelectors) return;
    variantControls.hidden = false;
    const label = document.createElement('label');
    label.textContent = (PAYLOAD.selectorLabel || 'variant') + ' ';
    const select = document.createElement('select');
    for (let k = 0; k < VARIANTS.length; k++) {
      const option = document.createElement('option');
      option.value = String(k);
      option.textContent = VARIANTS[k].label || VARIANTS[k].id || ('variant ' + (k + 1));
      select.appendChild(option);
    }
    select.addEventListener('change', function() { selectVariant(+select.value); });
    label.appendChild(select);
    variantSelectors.appendChild(label);
    if (variantSummary) variantSummary.textContent = VARIANTS[0].summary || '';
  }

  playBtn.addEventListener('click', function() { setPlaying(!playing); });
  scrub.addEventListener('input', function() { i = +scrub.value; setPlaying(false); render(i); });
  speedSel.addEventListener('change', function() { speed = +speedSel.value; });
  stepBack.addEventListener('click', function() { setPlaying(false); i = Math.max(0, i - 1); render(i); });
  stepFwd.addEventListener('click', function() { setPlaying(false); i = Math.min(N - 1, i + 1); render(i); });
  document.addEventListener('keydown', function(e) {
    if (e.key === ' ') { setPlaying(!playing); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { setPlaying(false); i = Math.max(0, i - 1); render(i); }
    else if (e.key === 'ArrowRight') { setPlaying(false); i = Math.min(N - 1, i + 1); render(i); }
  });

  populateVariantControls();
  applyAnimationConfig();
  render(0);
})();
</script>
</body>
</html>
"##;
