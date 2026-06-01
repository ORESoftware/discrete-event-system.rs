//! Interactive vehicle-jump planner.
//!
//! The domain kernel in `general::domain_application_models` is the canonical
//! Rust simulation. This module provides a self-contained browser player with
//! matching equations so users can adjust geometry, wind, mass, and drag inputs
//! and see the trajectory recomputed immediately.

use std::io;
use std::path::{Path, PathBuf};

pub const VEHICLE_JUMP_PLAYER_REL_PATH: &str = "vehicle-jump/player.html";

/// Render the self-contained HTML player.
pub fn vehicle_jump_player_html() -> String {
    VEHICLE_JUMP_HTML.to_string()
}

/// Write `out/vehicle-jump/player.html` under the requested output root.
pub fn write_vehicle_jump_player_html(out_root: impl AsRef<Path>) -> io::Result<PathBuf> {
    let path = out_root.as_ref().join(VEHICLE_JUMP_PLAYER_REL_PATH);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, vehicle_jump_player_html())?;
    Ok(path)
}

/// Binary/site-builder entry point.
pub fn run() {
    match write_vehicle_jump_player_html("out") {
        Ok(path) => eprintln!("Vehicle jump player: {}", path.display()),
        Err(e) => eprintln!("Vehicle jump player failed: {e}"),
    }
}

const VEHICLE_JUMP_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vehicle Jump Planner</title>
<style>
:root {
  color-scheme: light;
  --ink: #16202a;
  --muted: #5d6c7b;
  --line: #cbd5df;
  --surface: #ffffff;
  --band: #f2f5f8;
  --blue: #1c66d6;
  --green: #16845b;
  --red: #bd3f35;
  --amber: #a35f00;
  --violet: #6d58c9;
}
* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  color: var(--ink);
  background: #e8edf2;
  font: 14px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input { font: inherit; }
.shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}
.topbar {
  min-height: 54px;
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto;
  gap: 16px;
  align-items: center;
  padding: 10px 14px;
  background: #fbfcfe;
  border-bottom: 1px solid var(--line);
}
.brand { min-width: 0; }
.brand h1 {
  margin: 0;
  font-size: 17px;
  line-height: 1.1;
  letter-spacing: 0;
}
.brand p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 12px;
}
.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--green);
}
.status-dot.warn { background: var(--amber); }
.status-dot.bad { background: var(--red); }
.main {
  min-height: 0;
  display: grid;
  grid-template-columns: 316px minmax(0, 1fr);
}
.controls {
  min-height: 0;
  overflow: auto;
  background: var(--band);
  border-right: 1px solid var(--line);
  padding: 12px;
}
.control-group {
  display: grid;
  gap: 10px;
  padding: 0 0 16px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
}
.control-group:last-child { border-bottom: 0; }
.control-group h2 {
  margin: 0;
  color: #344354;
  font-size: 12px;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.field {
  display: grid;
  grid-template-columns: minmax(112px, 1fr) 76px;
  grid-template-rows: auto auto;
  gap: 5px 8px;
  align-items: center;
}
.field label {
  color: #314151;
  font-weight: 600;
  font-size: 12px;
}
.field output {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  text-align: right;
  font-size: 12px;
}
.field input[type="range"] {
  grid-column: 1 / -1;
  width: 100%;
  accent-color: var(--blue);
}
.workspace {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(380px, 1fr) auto;
}
.stage-band {
  min-height: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--line);
  display: grid;
}
#stage {
  width: 100%;
  height: 100%;
  min-height: 380px;
  display: block;
}
.bottom {
  background: #fbfcfe;
  padding: 12px;
  display: grid;
  gap: 12px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
}
.metric {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 9px 10px;
  min-width: 0;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.metric strong {
  display: block;
  margin-top: 3px;
  font-size: 18px;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.strip {
  display: grid;
  grid-template-columns: auto minmax(140px, 1fr) auto;
  gap: 10px;
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 9px 10px;
}
.icon-button {
  width: 32px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  display: inline-grid;
  place-items: center;
  cursor: pointer;
}
.icon-button:hover { border-color: var(--blue); color: var(--blue); }
#timeScrub { width: 100%; accent-color: var(--green); }
.strip label, .strip output {
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.note {
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 980px) {
  .main { grid-template-columns: 1fr; }
  .controls { border-right: 0; border-bottom: 1px solid var(--line); max-height: 42vh; }
  .metrics { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
}
@media (max-width: 620px) {
  .topbar { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .strip { grid-template-columns: auto minmax(100px, 1fr); }
  .strip output { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <h1>Vehicle Jump Planner</h1>
      <p>Non-linear drag, atmosphere, wind vector, ramp geometry, landing slope, and smooth transition curves.</p>
    </div>
    <div class="status"><span id="statusDot" class="status-dot"></span><span id="statusText">simulation only</span></div>
  </header>
  <main class="main">
    <aside class="controls" id="controls"></aside>
    <section class="workspace">
      <div class="stage-band">
        <svg id="stage" viewBox="0 0 960 520" role="img" aria-label="Vehicle jump trajectory"></svg>
      </div>
      <div class="bottom">
        <div class="metrics" id="metrics"></div>
        <div class="strip">
          <button class="icon-button" id="play" title="Play / pause" aria-label="Play / pause">▶</button>
          <input id="timeScrub" type="range" min="0" max="1" step="1" value="0" aria-label="Trajectory time">
          <output id="timeOut">t=0.00 s</output>
        </div>
        <div class="note">Simulation excludes rotation, suspension, tire contact limits, ramp structure, and field safety margins.</div>
      </div>
    </section>
  </main>
</div>
<script>
(function () {
  "use strict";
  const SVGNS = "http://www.w3.org/2000/svg";
  const G = 9.80665;
  const MPH = 2.2369362920544;
  const stage = document.getElementById("stage");
  const controlsEl = document.getElementById("controls");
  const metricsEl = document.getElementById("metrics");
  const playBtn = document.getElementById("play");
  const timeScrub = document.getElementById("timeScrub");
  const timeOut = document.getElementById("timeOut");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  const fields = [
    { group: "Geometry", id: "angle", label: "Ramp angle", min: 5, max: 45, step: 0.5, value: 18, unit: "deg" },
    { group: "Geometry", id: "distance", label: "Gap distance", min: 8, max: 80, step: 0.5, value: 28, unit: "m" },
    { group: "Geometry", id: "height", label: "Landing height", min: -8, max: 8, step: 0.25, value: 0, unit: "m" },
    { group: "Vehicle", id: "bikeMass", label: "Vehicle mass", min: 80, max: 1600, step: 5, value: 190, unit: "kg" },
    { group: "Vehicle", id: "riderMass", label: "Rider/driver mass", min: 40, max: 180, step: 1, value: 85, unit: "kg" },
    { group: "Vehicle", id: "cd", label: "Drag coefficient", min: 0.25, max: 1.6, step: 0.01, value: 0.90, unit: "" },
    { group: "Vehicle", id: "area", label: "Frontal area", min: 0.35, max: 3.2, step: 0.01, value: 0.75, unit: "m²" },
    { group: "Atmosphere", id: "altitude", label: "Altitude", min: -200, max: 4500, step: 25, value: 0, unit: "m" },
    { group: "Atmosphere", id: "temperature", label: "Temperature", min: -30, max: 45, step: 0.5, value: 15, unit: "°C" },
    { group: "Atmosphere", id: "seaLevelPressure", label: "Sea-level pressure", min: 930, max: 1045, step: 0.5, value: 1013.25, unit: "hPa" },
    { group: "Atmosphere", id: "densityScale", label: "Density scale", min: 0.65, max: 1.35, step: 0.01, value: 1, unit: "×" },
    { group: "Wind", id: "windSpeed", label: "Wind speed", min: 0, max: 25, step: 0.25, value: 0, unit: "m/s" },
    { group: "Wind", id: "windDirection", label: "Wind direction", min: -180, max: 180, step: 1, value: 0, unit: "°" },
    { group: "Wind", id: "windVertical", label: "Updraft", min: -8, max: 8, step: 0.25, value: 0, unit: "m/s" },
    { group: "Ramp Curves", id: "maxAccel", label: "Curve accel", min: 0.35, max: 2.5, step: 0.05, value: 1.0, unit: "g" },
    { group: "Ramp Curves", id: "maxSlope", label: "Max landing slope", min: 12, max: 60, step: 1, value: 45, unit: "deg" }
  ];

  let state = Object.fromEntries(fields.map(f => [f.id, f.value]));
  let last = null;
  let playing = false;
  let timer = null;

  function el(tag, attrs, kids) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else node.setAttribute(k, v);
      }
    }
    for (const kid of kids || []) node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    return node;
  }
  function sv(tag, attrs) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) node.setAttribute(k, v);
    return node;
  }
  function fmt(x, digits) {
    if (!Number.isFinite(x)) return "n/a";
    return x.toFixed(digits == null ? (Math.abs(x) >= 10 ? 1 : 2) : digits);
  }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function rad(deg) { return deg * Math.PI / 180; }
  function deg(r) { return r * 180 / Math.PI; }

  function buildControls() {
    const byGroup = new Map();
    for (const f of fields) {
      if (!byGroup.has(f.group)) byGroup.set(f.group, []);
      byGroup.get(f.group).push(f);
    }
    controlsEl.innerHTML = "";
    for (const [group, list] of byGroup) {
      const section = el("div", { class: "control-group" }, [el("h2", { text: group })]);
      for (const f of list) {
        const out = el("output", { text: labelValue(f, state[f.id]) });
        const input = el("input", { type: "range", min: f.min, max: f.max, step: f.step, value: state[f.id] });
        input.addEventListener("input", () => {
          state[f.id] = parseFloat(input.value);
          out.textContent = labelValue(f, state[f.id]);
          recompute();
        });
        section.appendChild(el("div", { class: "field" }, [el("label", { text: f.label }), out, input]));
      }
      controlsEl.appendChild(section);
    }
  }
  function labelValue(f, value) {
    const digits = f.step < 0.1 ? 2 : (f.step < 1 ? 1 : 0);
    return fmt(value, digits) + (f.unit ? " " + f.unit : "");
  }

  function solveSpeed(s) {
    const guess = ballisticGuess(s);
    let lo = Math.max(0.5, 0.35 * guess);
    let hi = Math.max(lo + 1, 1.65 * guess);
    let flo = landingError(s, lo);
    for (let i = 0; i < 12 && flo > 0; i++) {
      lo *= 0.65;
      flo = landingError(s, lo);
    }
    let fhi = landingError(s, hi);
    for (let i = 0; i < 16 && fhi < 0; i++) {
      hi *= 1.35;
      fhi = landingError(s, hi);
    }
    if (!(flo <= 0 && fhi >= 0)) return guess;
    for (let i = 0; i < 60; i++) {
      const mid = 0.5 * (lo + hi);
      const fm = landingError(s, mid);
      if (fm >= 0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
  }
  function ballisticGuess(s) {
    const theta = rad(s.angle);
    const dx = s.distance;
    const dy = s.height;
    const ct = Math.cos(theta);
    const denom = 2 * ct * ct * (dx * Math.tan(theta) - dy);
    return denom > 1e-9 ? Math.sqrt(G * dx * dx / denom) : Math.sqrt(G * dx) / Math.max(0.2, ct);
  }
  function landingError(s, speed) {
    const sim = simulate(s, speed, false);
    return sim.reached ? sim.target.y - s.height : -1e6;
  }
  function simulate(s, speed, keepTrace) {
    const theta = rad(s.angle);
    let q = { x: 0, y: 0, z: 0, vx: speed * Math.cos(theta), vy: speed * Math.sin(theta), vz: 0 };
    const dt = 0.006;
    const maxT = 8;
    let t = 0;
    const trace = keepTrace ? [{ t, ...q }] : [];
    for (let i = 0; i < Math.ceil(maxT / dt); i++) {
      const prev = q;
      const prevT = t;
      q = rk4(s, q, dt);
      t += dt;
      if (keepTrace) trace.push({ t, ...q });
      if (prev.x <= s.distance && q.x >= s.distance) {
        const u = Math.abs(q.x - prev.x) < 1e-12 ? 0 : clamp((s.distance - prev.x) / (q.x - prev.x), 0, 1);
        const target = lerp(prev, q, u);
        return { reached: true, target, targetTime: prevT + u * dt, trace };
      }
    }
    return { reached: false, target: q, targetTime: t, trace };
  }
  function deriv(s, q) {
    const mass = s.bikeMass + s.riderMass;
    const relx = q.vx - s.windForward;
    const rely = q.vy - s.windVertical;
    const relz = q.vz - s.windCross;
    const rel = Math.sqrt(relx * relx + rely * rely + relz * relz);
    const k = 0.5 * s.airDensity * s.cd * s.area / mass;
    return { x: q.vx, y: q.vy, z: q.vz, vx: -k * rel * relx, vy: -G - k * rel * rely, vz: -k * rel * relz };
  }
  function add(q, k, scale) {
    return { x: q.x + scale * k.x, y: q.y + scale * k.y, z: q.z + scale * k.z, vx: q.vx + scale * k.vx, vy: q.vy + scale * k.vy, vz: q.vz + scale * k.vz };
  }
  function rk4(s, q, dt) {
    const k1 = deriv(s, q), k2 = deriv(s, add(q, k1, dt / 2)), k3 = deriv(s, add(q, k2, dt / 2)), k4 = deriv(s, add(q, k3, dt));
    return {
      x: q.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6,
      y: q.y + dt * (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6,
      z: q.z + dt * (k1.z + 2 * k2.z + 2 * k3.z + k4.z) / 6,
      vx: q.vx + dt * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx) / 6,
      vy: q.vy + dt * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy) / 6,
      vz: q.vz + dt * (k1.vz + 2 * k2.vz + 2 * k3.vz + k4.vz) / 6
    };
  }
  function lerp(a, b, u) {
    return {
      x: a.x + (b.x - a.x) * u,
      y: a.y + (b.y - a.y) * u,
      z: a.z + (b.z - a.z) * u,
      vx: a.vx + (b.vx - a.vx) * u,
      vy: a.vy + (b.vy - a.vy) * u,
      vz: a.vz + (b.vz - a.vz) * u
    };
  }
  function smooth5(u) {
    u = clamp(u, 0, 1);
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5;
  }
  function transitionLength(angleRad, speed, maxAccelG) {
    return Math.max(0.25, 1.875 * Math.abs(angleRad) * speed * speed / Math.max(1e-9, maxAccelG * G));
  }
  function transitionProfile(length, startAngle, endAngle, samples) {
    const pts = [];
    let x = 0, y = 0;
    for (let i = 0; i < samples; i++) {
      const u = i / (samples - 1);
      if (i > 0) {
        const prevU = (i - 1) / (samples - 1);
        const mid = 0.5 * (prevU + u);
        const a = startAngle + (endAngle - startAngle) * smooth5(mid);
        const ds = length / (samples - 1);
        x += ds * Math.cos(a);
        y += ds * Math.sin(a);
      }
      const angle = startAngle + (endAngle - startAngle) * smooth5(u);
      pts.push({ s: length * u, x, y, angle });
    }
    return pts;
  }
  function localPressureHpa(s) {
    const lapseBase = Math.max(0.1, 1 - 2.25577e-5 * s.altitude);
    return s.seaLevelPressure * Math.pow(lapseBase, 5.25588);
  }
  function airDensity(s) {
    const pressurePa = localPressureHpa(s) * 100;
    const tempK = Math.max(1, s.temperature + 273.15);
    return pressurePa / (287.05 * tempK) * s.densityScale;
  }
  function windComponents(s) {
    const a = rad(s.windDirection);
    return {
      forward: s.windSpeed * Math.cos(a),
      cross: s.windSpeed * Math.sin(a)
    };
  }

  function recompute() {
    stop();
    const s = { ...state };
    s.localPressure = localPressureHpa(s);
    s.airDensity = airDensity(s);
    const wind = windComponents(s);
    s.windForward = wind.forward;
    s.windCross = wind.cross;
    const speed = solveSpeed(s);
    const sim = simulate(s, speed, true);
    const target = sim.target;
    const landingSlope = target.vy < 0 ? Math.atan2(-target.vy, Math.max(1e-9, target.vx)) : 0;
    const landingSpeed = Math.sqrt(target.vx ** 2 + target.vy ** 2 + target.vz ** 2);
    const takeoffLen = transitionLength(rad(s.angle), speed, s.maxAccel);
    const landingLen = transitionLength(landingSlope, landingSpeed, s.maxAccel);
    last = {
      params: s,
      speed,
      sim,
      landingSlope,
      landingSpeed,
      error: target.y - s.height,
      drift: target.z,
      takeoffCurve: transitionProfile(takeoffLen, 0, rad(s.angle), 18),
      landingCurve: transitionProfile(landingLen, -landingSlope, 0, 18)
    };
    timeScrub.max = Math.max(0, sim.trace.length - 1);
    timeScrub.value = Math.min(Number(timeScrub.value) || 0, Number(timeScrub.max));
    updateStatus();
    draw();
    drawMetrics();
  }
  function updateStatus() {
    const slopeDeg = deg(last.landingSlope);
    const ok = last.sim.reached && Math.abs(last.error) <= 0.15 && slopeDeg <= last.params.maxSlope && last.sim.target.vy < 0;
    statusDot.className = "status-dot" + (ok ? "" : (last.sim.reached ? " warn" : " bad"));
    statusText.textContent = ok ? "target matched" : (last.sim.reached ? "check slope/error" : "target not reached");
  }
  function drawMetrics() {
    const rows = [
      ["Takeoff speed", fmt(last.speed, 2) + " m/s"],
      ["Takeoff speed", fmt(last.speed * MPH, 1) + " mph"],
      ["Landing downslope", fmt(deg(last.landingSlope), 1) + "°"],
      ["Flight time", fmt(last.sim.targetTime, 2) + " s"],
      ["Landing speed", fmt(last.landingSpeed * MPH, 1) + " mph"],
      ["Landing error", fmt(last.error, 3) + " m"],
      ["Lateral drift", fmt(last.drift, 2) + " m"],
      ["Air density", fmt(last.params.airDensity, 3) + " kg/m³"],
      ["Local pressure", fmt(last.params.localPressure, 1) + " hPa"],
      ["Takeoff curve", fmt(last.takeoffCurve.at(-1).s, 1) + " m"],
      ["Landing curve", fmt(last.landingCurve.at(-1).s, 1) + " m"],
      ["Vehicle + rider", fmt(last.params.bikeMass + last.params.riderMass, 0) + " kg"],
      ["Wind vector", fmt(last.params.windSpeed, 1) + " m/s @ " + fmt(last.params.windDirection, 0) + "°"],
      ["Tailwind", fmt(last.params.windForward, 1) + " m/s"],
      ["Crosswind", fmt(last.params.windCross, 1) + " m/s"]
    ];
    metricsEl.innerHTML = "";
    for (const [label, value] of rows) {
      metricsEl.appendChild(el("div", { class: "metric" }, [el("span", { text: label }), el("strong", { text: value })]));
    }
  }
  function draw() {
    if (!last) return;
    const idx = clamp(parseInt(timeScrub.value, 10) || 0, 0, last.sim.trace.length - 1);
    const p = last.sim.trace[idx] || last.sim.trace[0];
    timeOut.textContent = "t=" + fmt(p.t, 2) + " s";

    stage.innerHTML = "";
    const s = last.params;
    const trace = last.sim.trace;
    const margin = { left: 68, right: 50, top: 34, bottom: 70 };
    const landingPreviewLen = Math.min(16, Math.max(8, s.distance * 0.35));
    const landingPreviewY = s.height - Math.tan(last.landingSlope) * landingPreviewLen;
    const ys = trace.map(d => d.y).concat([s.height, 0, landingPreviewY]);
    const minX = -18;
    const maxX = 92;
    const minY = Math.min(-18, ...ys) - 2;
    const maxY = Math.max(28, ...ys) + 2;
    const spanX = Math.max(1, maxX - minX);
    const spanY = Math.max(1, maxY - minY);
    const plotW = 960 - margin.left - margin.right;
    const plotH = 520 - margin.top - margin.bottom;
    const scale = Math.min(plotW / spanX, plotH / spanY);
    const ox = margin.left - minX * scale + (plotW - spanX * scale) * 0.5;
    const oy = margin.top + maxY * scale + (plotH - spanY * scale) * 0.5;
    const X = x => ox + x * scale;
    const Y = y => oy - y * scale;

    stage.appendChild(sv("rect", { x: 0, y: 0, width: 960, height: 520, fill: "#ffffff" }));
    grid(stage, X, Y, minX, maxX, minY, maxY);
    drawGround(stage, X, Y, s, last);
    drawCurve(stage, X, Y, trace, "#1c66d6", 3);
    drawCurve(stage, X, Y, trace.slice(0, idx + 1), "#16845b", 4);
    drawVehicle(stage, X(p.x), Y(p.y), rad(s.angle), deg(last.landingSlope), p);
    drawWind(stage, s);
    drawLabel(stage, X(0), Y(0) - 14, "takeoff " + fmt(s.angle, 1) + "°", "#314151", "middle");
    drawLabel(stage, X(s.distance), Y(s.height) - 14, "landing " + fmt(s.distance, 1) + " m", "#314151", "middle");
  }
  function grid(svg, X, Y, minX, maxX, minY, maxY) {
    const x0 = Math.floor(minX / 10) * 10, x1 = Math.ceil(maxX / 10) * 10;
    for (let x = x0; x <= x1; x += 10) {
      svg.appendChild(sv("line", { x1: X(x), y1: 28, x2: X(x), y2: 454, stroke: "#edf1f5", "stroke-width": 1 }));
      drawLabel(svg, X(x), 474, String(x) + "m", "#6b7a89", "middle");
    }
    const y0 = Math.floor(minY / 5) * 5, y1 = Math.ceil(maxY / 5) * 5;
    for (let y = y0; y <= y1; y += 5) {
      svg.appendChild(sv("line", { x1: 48, y1: Y(y), x2: 920, y2: Y(y), stroke: "#edf1f5", "stroke-width": 1 }));
    }
  }
  function drawGround(svg, X, Y, s, run) {
    svg.appendChild(sv("line", { x1: 42, y1: Y(0), x2: X(s.distance) - 8, y2: Y(0), stroke: "#6b7a89", "stroke-width": 2 }));
    const curveTip = run.takeoffCurve.at(-1);
    const takeoff = run.takeoffCurve
      .map(pt => [pt.x - curveTip.x, pt.y - curveTip.y])
      .filter(([x]) => x >= -17.5);
    drawPolyline(svg, takeoff.map(([x, y]) => [X(x), Y(y)]), "#6d58c9", 5);
    const rampGuideLen = Math.min(16, Math.max(8, s.distance * 0.28));
    svg.appendChild(sv("line", {
      x1: X(0),
      y1: Y(0),
      x2: X(-rampGuideLen * Math.cos(rad(s.angle))),
      y2: Y(-rampGuideLen * Math.sin(rad(s.angle))),
      stroke: "#6d58c9",
      "stroke-width": 2,
      "stroke-dasharray": "5,4",
      "stroke-linecap": "round"
    }));
    const landingLen = Math.min(16, Math.max(8, s.distance * 0.35));
    const drop = Math.tan(run.landingSlope) * landingLen;
    svg.appendChild(sv("line", { x1: X(s.distance), y1: Y(s.height), x2: X(s.distance + landingLen), y2: Y(s.height - drop), stroke: "#16845b", "stroke-width": 5, "stroke-linecap": "round" }));
    svg.appendChild(sv("circle", { cx: X(s.distance), cy: Y(s.height), r: 5, fill: "#bd3f35" }));
  }
  function drawCurve(svg, X, Y, trace, stroke, width) {
    if (trace.length < 2) return;
    drawPolyline(svg, trace.map(d => [X(d.x), Y(d.y)]), stroke, width);
  }
  function drawPolyline(svg, pts, stroke, width) {
    svg.appendChild(sv("polyline", { points: pts.map(p => p[0] + "," + p[1]).join(" "), fill: "none", stroke, "stroke-width": width, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  }
  function drawVehicle(svg, x, y, takeoffAngle, landingSlopeDeg, p) {
    const g = sv("g", { transform: "translate(" + x + " " + y + ")" });
    const pitch = Math.atan2(p.vy, Math.max(1e-9, p.vx)) * 0.35;
    g.setAttribute("transform", "translate(" + x + " " + y + ") rotate(" + deg(pitch) + ")");
    g.appendChild(sv("rect", { x: -17, y: -8, width: 34, height: 12, rx: 3, fill: "#263544" }));
    g.appendChild(sv("circle", { cx: -11, cy: 7, r: 6, fill: "#111827" }));
    g.appendChild(sv("circle", { cx: 12, cy: 7, r: 6, fill: "#111827" }));
    g.appendChild(sv("circle", { cx: -11, cy: 7, r: 2.5, fill: "#94a3b8" }));
    g.appendChild(sv("circle", { cx: 12, cy: 7, r: 2.5, fill: "#94a3b8" }));
    svg.appendChild(g);
  }
  function drawWind(svg, s) {
    const x = 760, y = 44;
    const len = clamp(s.windSpeed * 4.5, 8, 80);
    const a = rad(s.windDirection);
    const x2 = x + Math.cos(a) * len;
    const y2 = y - Math.sin(a) * len;
    const col = s.windForward >= 0 ? "#16845b" : "#a35f00";
    svg.appendChild(sv("line", { x1: x, y1: y, x2, y2, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }));
    svg.appendChild(sv("circle", { cx: x2, cy: y2, r: 4, fill: col }));
    drawLabel(svg, x, y + 22, "wind " + fmt(s.windSpeed, 1) + " m/s @ " + fmt(s.windDirection, 0) + "°", "#314151", "middle");
  }
  function drawLabel(svg, x, y, text, fill, anchor) {
    const t = sv("text", { x, y, fill, "font-size": 12, "text-anchor": anchor || "start", "font-weight": 600 });
    t.textContent = text;
    svg.appendChild(t);
  }

  playBtn.addEventListener("click", () => {
    if (playing) stop(); else play();
  });
  timeScrub.addEventListener("input", draw);
  function play() {
    if (!last || !last.sim.trace.length) return;
    playing = true;
    playBtn.textContent = "⏸";
    timer = setInterval(() => {
      let i = parseInt(timeScrub.value, 10) || 0;
      if (i >= Number(timeScrub.max)) {
        stop();
        return;
      }
      timeScrub.value = i + 1;
      draw();
    }, 24);
  }
  function stop() {
    playing = false;
    playBtn.textContent = "▶";
    clearInterval(timer);
  }

  buildControls();
  recompute();
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vehicle_jump_player_contains_interactive_solver() {
        let html = vehicle_jump_player_html();
        assert!(html.contains("Vehicle Jump Planner"));
        assert!(html.contains("function solveSpeed"));
        assert!(html.contains("Ramp angle"));
        assert!(html.contains("Altitude"));
        assert!(html.contains("Temperature"));
        assert!(html.contains("Wind direction"));
        assert!(html.contains("function airDensity"));
        assert!(html.contains("Trajectory time"));
    }
}
