//! Animated **FEL vs time-stepped** simulation of an open **tandem queuing
//! network** — a 4-router computer network (packets flow `src → R0 → R1 → R2 →
//! R3 → sink`, each router a single-server M/M/1 link).
//!
//! ```sh
//! cargo run --example fel_network_anim
//! # writes out/fel-network/animation.html  (open it in a browser)
//! ```
//!
//! The *same* network is simulated two ways and the frames are rendered
//! side-by-side so you can watch the paradigms differ:
//!
//! * **FEL (next-event):** uses the engine's real [`Engine`](des_engine::des::fel::engine::Engine)
//!   future-event-list scheduler. The clock *jumps* from event to event
//!   (arrival / link-transmission-complete); between events nothing happens.
//!   Exact in continuous time; work ≈ a few events per packet.
//! * **Time-stepped (Δt):** advances the whole network by a fixed Δt every tick,
//!   drawing `Poisson(μ·Δt)` link completions and `Poisson(λ·Δt)` arrivals per
//!   tick (the semantics of the engine's stepped entity stations). Work =
//!   `#nodes / Δt` updates regardless of load; accuracy is `O(Δt)`.
//!
//! A live chart tracks total packets-in-system over time for both, against the
//! closed-form open-Jackson-network steady-state mean.

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use des_engine::des::fel::engine::Engine;
use serde_json::{json, Value};

// --------------------------------------------------------------------------
// A tiny self-contained SplitMix64 RNG + samplers, so both paradigms draw from
// the same fair, reproducible source without coupling to engine internals.
// --------------------------------------------------------------------------
struct Rng {
    s: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { s: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0)
    }
    /// Exponential inter-event time with the given rate.
    fn exp(&mut self, rate: f64) -> f64 {
        -self.unit().ln() / rate
    }
    /// Knuth's Poisson sampler.
    fn poisson(&mut self, mean: f64) -> i64 {
        if mean <= 0.0 {
            return 0;
        }
        let l = (-mean).exp();
        let mut k = 0i64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= self.unit();
            if p <= l {
                break;
            }
        }
        k - 1
    }
}

// --------------------------------------------------------------------------
// FEL (next-event) tandem network on the engine's real Engine.
// --------------------------------------------------------------------------
struct NetWorld {
    rng: Rng,
    lambda: f64,
    mu: Vec<f64>,
    /// Packets waiting (NOT in service) at each node.
    waiting: Vec<VecDeque<u64>>,
    /// Packet currently in service at each node, if any.
    in_service: Vec<Option<u64>>,
    next_id: u64,
    arrivals: u64,
    completed: u64,
    frames: Vec<Value>,
}

impl NetWorld {
    fn new(seed: u64, lambda: f64, mu: Vec<f64>) -> Self {
        let n = mu.len();
        NetWorld {
            rng: Rng::new(seed),
            lambda,
            mu,
            waiting: vec![VecDeque::new(); n],
            in_service: vec![None; n],
            next_id: 0,
            arrivals: 0,
            completed: 0,
            frames: Vec::new(),
        }
    }
    /// In-system count at each node = waiting + (1 if a packet is in service).
    fn counts(&self) -> Vec<i64> {
        (0..self.mu.len())
            .map(|i| self.waiting[i].len() as i64 + self.in_service[i].is_some() as i64)
            .collect()
    }
}

fn record(eng: &mut Engine<NetWorld>, kind: &str, node: i64) {
    let counts = eng.world.counts();
    let sys: i64 = counts.iter().sum();
    let frame = json!({
        "t": eng.now(),
        "n": counts,
        "sys": sys,
        "work": eng.events_processed(),
        "kind": kind,
        "node": node,
    });
    eng.world.frames.push(frame);
}

/// Admit `pkt` at `node`: start service if the link is idle, else queue it.
fn admit(eng: &mut Engine<NetWorld>, node: usize, pkt: u64) {
    if eng.world.in_service[node].is_none() {
        eng.world.in_service[node] = Some(pkt);
        let svc = eng.world.rng.exp(eng.world.mu[node]);
        eng.schedule_after(svc, move |e| complete(e, node));
    } else {
        eng.world.waiting[node].push_back(pkt);
    }
}

/// A link finished transmitting its in-service packet: forward it downstream and
/// pull the next waiting packet (if any) into service.
fn complete(eng: &mut Engine<NetWorld>, node: usize) {
    let pkt = eng.world.in_service[node].take().expect("served packet");
    if let Some(next) = eng.world.waiting[node].pop_front() {
        eng.world.in_service[node] = Some(next);
        let svc = eng.world.rng.exp(eng.world.mu[node]);
        eng.schedule_after(svc, move |e| complete(e, node));
    }
    if node + 1 < eng.world.mu.len() {
        admit(eng, node + 1, pkt);
    } else {
        eng.world.completed += 1;
    }
    record(eng, "depart", node as i64);
}

fn arrival(eng: &mut Engine<NetWorld>) {
    // Schedule the next external arrival (Poisson stream into R0).
    let ia = eng.world.rng.exp(eng.world.lambda);
    eng.schedule_after(ia, arrival);

    let pkt = eng.world.next_id;
    eng.world.next_id += 1;
    eng.world.arrivals += 1;
    admit(eng, 0, pkt);
    record(eng, "arrive", 0);
}

fn run_fel(lambda: f64, mu: &[f64], horizon: f64, seed: u64) -> (Vec<Value>, u64) {
    let mut eng = Engine::new(NetWorld::new(seed, lambda, mu.to_vec()));
    // Initial empty frame.
    record(&mut eng, "start", -1);
    let first = eng.world.rng.exp(lambda);
    eng.schedule_after(first, arrival);
    eng.run_until(horizon);
    let events = eng.events_processed();
    // Park a final frame at the horizon so the animation holds the last state.
    let counts = eng.world.counts();
    let sys: i64 = counts.iter().sum();
    eng.world.frames.push(json!({
        "t": horizon, "n": counts, "sys": sys, "work": events, "kind": "end", "node": -1
    }));
    (eng.world.frames, events)
}

// --------------------------------------------------------------------------
// Time-stepped tandem network: fixed Δt, Poisson counts per tick.
// --------------------------------------------------------------------------
fn run_time_stepped(lambda: f64, mu: &[f64], dt: f64, ticks: u64, seed: u64) -> (Vec<Value>, u64) {
    let nn = mu.len();
    let mut rng = Rng::new(seed ^ 0x5DEE_CE66_D00D);
    let mut n = vec![0i64; nn];
    let mut frames = vec![json!({
        "t": 0.0, "n": n.clone(), "sys": 0, "work": 0u64, "kind": "start", "node": -1
    })];
    for tk in 0..ticks {
        // Departures are computed from the snapshot at the tick start (a packet
        // arriving this tick cannot also be served this tick).
        let mut dep = vec![0i64; nn];
        for i in 0..nn {
            dep[i] = rng.poisson(mu[i] * dt).min(n[i]);
        }
        let arr = rng.poisson(lambda * dt);
        for i in 0..nn {
            n[i] -= dep[i];
            if i + 1 < nn {
                n[i + 1] += dep[i];
            }
        }
        n[0] += arr;
        let t = (tk + 1) as f64 * dt;
        let sys: i64 = n.iter().sum();
        frames.push(json!({
            "t": t, "n": n.clone(), "sys": sys,
            "work": (tk + 1) * nn as u64, "kind": "tick", "node": -1
        }));
    }
    (frames, ticks * nn as u64)
}

fn main() {
    // Open tandem of 4 single-server links. R1 is the bottleneck (lowest μ),
    // so its queue visibly builds — the interesting bit to animate.
    let lambda = 0.85_f64;
    let mu = vec![1.5_f64, 1.1, 1.3, 1.6];
    let horizon = 60.0_f64;
    let dt = 0.5_f64;
    let seed = 0x1234_5678_9abc_def0_u64;
    let ticks = (horizon / dt).round() as u64;

    let (fel_frames, fel_events) = run_fel(lambda, &mu, horizon, seed);
    let (ts_frames, ts_updates) = run_time_stepped(lambda, &mu, dt, ticks, seed);

    // Closed-form open-Jackson-network steady state: a tandem feeds λ to every
    // node, so ρ_i = λ/μ_i and L_i = ρ_i/(1-ρ_i).
    let rho: Vec<f64> = mu.iter().map(|m| lambda / m).collect();
    let l_per: Vec<f64> = rho.iter().map(|r| r / (1.0 - r)).collect();
    let l_total: f64 = l_per.iter().sum();

    let data = json!({
        "meta": {
            "lambda": lambda,
            "mu": mu,
            "dt": dt,
            "horizon": horizon,
            "nodes": mu.len(),
            "felEvents": fel_events,
            "tsUpdates": ts_updates,
            "rho": rho,
            "analyticPerNode": l_per,
            "analyticL": l_total,
        },
        "fel": fel_frames,
        "ts": ts_frames,
    });

    let html = HTML_TEMPLATE.replace("__DES_DATA__", &serde_json::to_string(&data).unwrap());

    let dir = Path::new("out/fel-network");
    fs::create_dir_all(dir).expect("create out/fel-network");
    let path = dir.join("animation.html");
    fs::write(&path, html).expect("write animation.html");

    println!("FEL vs time-stepped tandem queuing network");
    println!(
        "  λ={lambda}  μ={mu:?}  horizon={horizon}s  Δt={dt}s  ({} nodes)",
        mu.len()
    );
    println!(
        "  FEL: {fel_events} events processed ({} frames)",
        fel_frames.len()
    );
    println!(
        "  Time-stepped: {ts_updates} station-updates over {ticks} ticks ({} frames)",
        ts_frames.len()
    );
    println!(
        "  work ratio (time-step / FEL): {:.1}x more updates than events",
        ts_updates as f64 / fel_events.max(1) as f64
    );
    println!("  analytic L (mean packets in system) = {l_total:.2}");
    println!("  wrote {}", path.display());
}

const HTML_TEMPLATE: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FEL vs time-stepped — tandem queuing network</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;background:#0b1021;color:#e6edf3}
main{max-width:1100px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:1.45rem;margin:0 0 4px}
.sub{color:#9aa4b2;margin:0 0 14px;font-size:.9rem;line-height:1.5;max-width:80ch}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.chip{font-size:.78rem;border:1px solid #2b3344;border-radius:6px;padding:3px 9px;color:#c9d4e3;background:#0f1422}
.chip b{color:#fff}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{border:1px solid #21262d;border-radius:12px;background:#0f1422;padding:14px}
.panel h2{font-size:1rem;margin:0 0 2px}
.panel .tag{font-size:.74rem;color:#8b949e;margin:0 0 8px}
.fel h2{color:#38bdf8}.ts h2{color:#f59e0b}
.stats{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:#9aa4b2;margin:0 0 8px}
.stats b{color:#e6edf3;font-variant-numeric:tabular-nums}
svg{display:block;width:100%}
.legend{display:flex;gap:16px;align-items:center;font-size:.78rem;color:#9aa4b2;margin:14px 0 4px}
.legend .k{display:inline-flex;align-items:center;gap:6px}
.legend .sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px;border-top:1px solid #21262d;padding-top:14px}
button{font:inherit;font-size:.85rem;cursor:pointer;border-radius:8px;padding:7px 12px;border:1px solid #2b3344;background:#161b22;color:#e6edf3}
button:hover{border-color:#3b82f6}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.controls label{font-size:.8rem;color:#9aa4b2;display:inline-flex;align-items:center;gap:8px}
input[type=range]{vertical-align:middle}
#scrub{flex:1;min-width:160px}
.clock{font-variant-numeric:tabular-nums;color:#fff;font-weight:600}
</style>
</head>
<body>
<main>
<h1>FEL vs time-stepped &mdash; open tandem queuing network</h1>
<p class="sub">The same 4-router packet network (<code>src&rarr;R0&rarr;R1&rarr;R2&rarr;R3&rarr;sink</code>, each a single-server M/M/1 link) simulated two ways on one shared clock. <b>FEL</b> jumps from event to event; <b>time-stepped</b> advances every node by a fixed &Delta;t. Watch the bottleneck <b>R1</b> back up, and the chart track total packets-in-system against the closed-form steady-state mean.</p>
<div class="chips" id="chips"></div>

<div class="panels">
  <div class="panel fel">
    <h2>FEL &mdash; next-event</h2>
    <div class="tag">clock leaps to the next arrival / transmission-complete event</div>
    <div class="stats">
      <span>clock <b class="clock" id="felClock">0.0s</b></span>
      <span>events <b id="felWork">0</b></span>
      <span>in&nbsp;system <b id="felSys">0</b></span>
    </div>
    <svg id="felSvg" viewBox="0 0 520 230"></svg>
  </div>
  <div class="panel ts">
    <h2>Time-stepped &mdash; &Delta;t</h2>
    <div class="tag">every node advances each tick whether or not anything happens</div>
    <div class="stats">
      <span>clock <b class="clock" id="tsClock">0.0s</b></span>
      <span>updates <b id="tsWork">0</b></span>
      <span>in&nbsp;system <b id="tsSys">0</b></span>
    </div>
    <svg id="tsSvg" viewBox="0 0 520 230"></svg>
  </div>
</div>

<div class="legend">
  <span class="k"><span class="sw" style="background:#38bdf8"></span>FEL in-system</span>
  <span class="k"><span class="sw" style="background:#f59e0b"></span>time-stepped in-system</span>
  <span class="k"><span class="sw" style="background:#6b7689;height:0;border-top:2px dashed #6b7689"></span>analytic mean L</span>
</div>
<svg id="chart" viewBox="0 0 1060 220"></svg>

<div class="controls">
  <button class="primary" id="play">&#9654; Play</button>
  <button id="stepb">&#9664;| step</button>
  <button id="stepf">|&#9654; step</button>
  <label>speed <input type="range" id="speed" min="0.5" max="12" step="0.5" value="4"> <span id="speedv">4&times;</span></label>
  <label>t <input type="range" id="scrub" min="0" max="1000" step="1" value="0"></label>
</div>
</main>

<script>
const DATA = __DES_DATA__;
const M = DATA.meta, NN = M.nodes, T = M.horizon;
const COL = {fel:'#38bdf8', ts:'#f59e0b'};

// ---- chips ----
(function(){
  const c = document.getElementById('chips');
  const items = [
    ['&lambda;', M.lambda+' /s'],
    ['&mu;', '['+M.mu.join(', ')+']'],
    ['&Delta;t', M.dt+' s'],
    ['horizon', T+' s'],
    ['FEL events', M.felEvents.toLocaleString()],
    ['time-step updates', M.tsUpdates.toLocaleString()],
    ['work ratio', (M.tsUpdates/Math.max(1,M.felEvents)).toFixed(1)+'&times;'],
    ['analytic L', M.analyticL.toFixed(2)],
  ];
  c.innerHTML = items.map(function(p){return '<span class="chip">'+p[0]+' <b>'+p[1]+'</b></span>';}).join('');
})();

// latest frame at or before time t (binary search)
function frameAt(frames, t){
  let lo=0, hi=frames.length-1, ans=0;
  while(lo<=hi){ const mid=(lo+hi)>>1; if(frames[mid].t<=t){ans=mid;lo=mid+1;} else hi=mid-1; }
  return frames[ans];
}

// ---- network renderer ----
const NODE_W=58, NODE_H=40, GAP=112, X0=70, YC=92, MAXDOTS=7;
function loadColor(n){
  if(n<=0) return '#1b2536';
  if(n<=2) return '#15803d';
  if(n<=4) return '#a16207';
  return '#b91c1c';
}
function drawNet(svgId, frame, accent){
  let s='';
  // src label + arrow into R0
  s+='<text x="14" y="'+(YC+4)+'" fill="#8b949e" font-size="11">src</text>';
  for(let i=0;i<NN;i++){
    const x = X0 + i*GAP;
    const n = frame.n[i];
    const waiting = Math.max(0, n-1);
    const inSvc = n>=1;
    // link arrow from previous node (or src) to this node
    const fromX = (i===0)? 40 : (X0+(i-1)*GAP+NODE_W);
    s+='<line x1="'+fromX+'" y1="'+YC+'" x2="'+(x-2)+'" y2="'+YC+'" stroke="#30363d" stroke-width="2" marker-end="url(#arr)"/>';
    // waiting queue: dots to the left of the server, stacked back from the link
    const shown = Math.min(waiting, MAXDOTS);
    for(let k=0;k<shown;k++){
      const dx = x - 12 - k*11;
      s+='<circle cx="'+dx+'" cy="'+YC+'" r="4.2" fill="'+accent+'" opacity="'+(0.55+0.45*(1-k/MAXDOTS))+'"/>';
    }
    if(waiting>MAXDOTS){
      s+='<text x="'+(x-12-MAXDOTS*11)+'" y="'+(YC+4)+'" fill="'+accent+'" font-size="10" text-anchor="end">+'+(waiting-MAXDOTS)+'</text>';
    }
    // server box
    s+='<rect x="'+x+'" y="'+(YC-NODE_H/2)+'" width="'+NODE_W+'" height="'+NODE_H+'" rx="8" fill="'+loadColor(n)+'" stroke="#3b4757" stroke-width="1.5"/>';
    if(inSvc){ s+='<circle cx="'+(x+NODE_W/2)+'" cy="'+YC+'" r="6" fill="#e6edf3"/>'; }
    // labels
    s+='<text x="'+(x+NODE_W/2)+'" y="'+(YC-NODE_H/2-7)+'" fill="#c9d4e3" font-size="12" text-anchor="middle" font-weight="600">R'+i+'</text>';
    s+='<text x="'+(x+NODE_W/2)+'" y="'+(YC+NODE_H/2+15)+'" fill="#e6edf3" font-size="12" text-anchor="middle" font-variant-numeric="tabular-nums">n='+n+'</text>';
    s+='<text x="'+(x+NODE_W/2)+'" y="'+(YC+NODE_H/2+30)+'" fill="#6b7689" font-size="10" text-anchor="middle">&mu;='+M.mu[i]+'</text>';
  }
  // sink arrow out of last node
  const lx = X0+(NN-1)*GAP+NODE_W;
  s+='<line x1="'+lx+'" y1="'+YC+'" x2="'+(lx+34)+'" y2="'+YC+'" stroke="#30363d" stroke-width="2" marker-end="url(#arr)"/>';
  s+='<text x="'+(lx+38)+'" y="'+(YC+4)+'" fill="#8b949e" font-size="11">sink</text>';
  const defs='<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#30363d"/></marker></defs>';
  document.getElementById(svgId).innerHTML = defs + s;
}

// ---- chart ----
const CW=1060, CH=220, ML=42, MR=14, MT=12, MB=26;
const PW=CW-ML-MR, PH=CH-MT-MB;
let maxSys = M.analyticL*1.4;
for(const f of DATA.fel) maxSys=Math.max(maxSys, f.sys);
for(const f of DATA.ts) maxSys=Math.max(maxSys, f.sys);
maxSys = Math.ceil(maxSys+1);
function sx(t){ return ML + PW*(t/T); }
function sy(v){ return MT + PH*(1 - v/maxSys); }
function stepPath(frames){
  let d='', px=null;
  for(const f of frames){
    const X=sx(f.t), Y=sy(f.sys);
    if(px===null){ d='M'+X+' '+Y; } else { d+=' L'+X+' '+py+' L'+X+' '+Y; }
    px=X; py=Y;
  }
  return d;
}
const felPath = stepPath(DATA.fel), tsPath = stepPath(DATA.ts);
function drawChart(tPlay){
  let s='';
  // axes
  s+='<line x1="'+ML+'" y1="'+sy(0)+'" x2="'+(CW-MR)+'" y2="'+sy(0)+'" stroke="#30363d"/>';
  s+='<line x1="'+ML+'" y1="'+MT+'" x2="'+ML+'" y2="'+sy(0)+'" stroke="#30363d"/>';
  for(let g=0; g<=maxSys; g+=Math.max(1,Math.round(maxSys/5))){
    s+='<line x1="'+ML+'" y1="'+sy(g)+'" x2="'+(CW-MR)+'" y2="'+sy(g)+'" stroke="#1b2230"/>';
    s+='<text x="'+(ML-6)+'" y="'+(sy(g)+4)+'" fill="#6b7689" font-size="10" text-anchor="end">'+g+'</text>';
  }
  for(let g=0; g<=T; g+=10){
    s+='<text x="'+sx(g)+'" y="'+(CH-8)+'" fill="#6b7689" font-size="10" text-anchor="middle">'+g+'s</text>';
  }
  // analytic L
  s+='<line x1="'+ML+'" y1="'+sy(M.analyticL)+'" x2="'+(CW-MR)+'" y2="'+sy(M.analyticL)+'" stroke="#6b7689" stroke-width="1.5" stroke-dasharray="5 4"/>';
  // series
  s+='<path d="'+tsPath+'" fill="none" stroke="'+COL.ts+'" stroke-width="1.6" opacity="0.95"/>';
  s+='<path d="'+felPath+'" fill="none" stroke="'+COL.fel+'" stroke-width="1.6" opacity="0.95"/>';
  // playhead + current dots
  const fx=sx(tPlay);
  s+='<line x1="'+fx+'" y1="'+MT+'" x2="'+fx+'" y2="'+sy(0)+'" stroke="#e6edf3" stroke-width="1" opacity="0.5"/>';
  const ff=frameAt(DATA.fel,tPlay), tf=frameAt(DATA.ts,tPlay);
  s+='<circle cx="'+fx+'" cy="'+sy(ff.sys)+'" r="4" fill="'+COL.fel+'"/>';
  s+='<circle cx="'+fx+'" cy="'+sy(tf.sys)+'" r="4" fill="'+COL.ts+'"/>';
  s+='<text x="'+(CW-MR)+'" y="'+(sy(M.analyticL)-5)+'" fill="#8b949e" font-size="10" text-anchor="end">L='+M.analyticL.toFixed(2)+'</text>';
  document.getElementById('chart').innerHTML = s;
}

// ---- playback ----
let tPlay=0, playing=false, last=null;
const scrub=document.getElementById('scrub');
function render(){
  const ff=frameAt(DATA.fel,tPlay), tf=frameAt(DATA.ts,tPlay);
  drawNet('felSvg', ff, COL.fel);
  drawNet('tsSvg', tf, COL.ts);
  document.getElementById('felClock').textContent=tPlay.toFixed(1)+'s';
  document.getElementById('tsClock').textContent=tPlay.toFixed(1)+'s';
  document.getElementById('felWork').textContent=ff.work.toLocaleString();
  document.getElementById('tsWork').textContent=tf.work.toLocaleString();
  document.getElementById('felSys').textContent=ff.sys;
  document.getElementById('tsSys').textContent=tf.sys;
  scrub.value=Math.round(1000*tPlay/T);
  drawChart(tPlay);
}
function tick(ts){
  if(!playing){ last=null; return; }
  if(last===null) last=ts;
  const dtReal=(ts-last)/1000; last=ts;
  const speed=parseFloat(document.getElementById('speed').value);
  tPlay+=dtReal*speed;
  if(tPlay>=T){ tPlay=T; playing=false; document.getElementById('play').innerHTML='&#9654; Play'; }
  render();
  if(playing) requestAnimationFrame(tick);
}
document.getElementById('play').onclick=function(){
  if(tPlay>=T) tPlay=0;
  playing=!playing;
  this.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';
  if(playing) requestAnimationFrame(tick);
};
document.getElementById('speed').oninput=function(){document.getElementById('speedv').textContent=this.value+'\u00d7';};
scrub.oninput=function(){ playing=false; document.getElementById('play').innerHTML='&#9654; Play'; tPlay=T*this.value/1000; render(); };
// step to the next / previous FEL event time (the finest granularity)
function stepTo(dir){
  playing=false; document.getElementById('play').innerHTML='&#9654; Play';
  const fr=DATA.fel; let idx=0;
  for(let i=0;i<fr.length;i++){ if(fr[i].t<=tPlay+1e-9) idx=i; else break; }
  idx=Math.min(fr.length-1, Math.max(0, idx+dir));
  tPlay=fr[idx].t; render();
}
document.getElementById('stepf').onclick=function(){stepTo(1);};
document.getElementById('stepb').onclick=function(){stepTo(-1);};
render();
</script>
</body>
</html>
"####;
