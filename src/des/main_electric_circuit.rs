//! Port of `src/des/main-electric-circuit.ts`.
//!
//! Series RLC step-response as a discrete-event system, where the DES tick
//! clock IS the numerical integrator (forward Euler):
//!
//! ```text
//!   V_step ─[VoltageSource]─[Resistor]─┬─[Inductor]─┬─[Capacitor]─┐
//! ```
//!
//! Three stationary entities communicate via SYNCHRONOUS DATAFLOW:
//!   * `VoltageSource` emits `V_in` (a Heaviside step at `t0`).
//!   * `Inductor` holds `I`:   `dI/dt   = (V_in − I·R − V_C) / L`.
//!   * `Capacitor` holds `V_C`: `dV_C/dt = I / C`.
//!
//! Each tick: (1) every station reads its inbox (FROZEN — values are last
//! tick's emissions), (2) computes its new state and stages emissions into a
//! `pending` map, (3) after all stations run, `pending` is committed into
//! target inboxes, visible next tick.
//!
//! ## Rust shape
//!   * The TS classes extend `SynchronousDataflowStation`
//!     (`crate::des::general::time_stepped_station`). Here they are plain
//!     structs that carry a `HashMap<String, f64>` inbox; [`run_rlc`] performs
//!     the two-phase (run → commit) routing explicitly, reproducing the frozen-
//!     inbox semantics without the `Rc<RefCell<dyn …>>` graph.
//!   * PORT NOTE: the TS `fisherYatesShuffle(order)` per tick is cosmetic — SDF
//!     reads come from the frozen inbox, so state updates are order-independent.
//!     We run in declared order (`src → L → C → rec`) so the recorder captures
//!     the freshly-updated `I`/`V_C` (the "recorder runs last" case) and drop
//!     the shuffle.

#![allow(dead_code)]

use std::collections::HashMap;

/// A recorded trace row.
#[derive(Clone, Copy, Debug)]
pub struct TraceRow {
    pub t: f64,
    pub i: f64,
    pub v_c: f64,
    pub v_in: f64,
}

struct VoltageSource {
    id: String,
    v_step: f64,
    t0: f64,
}

impl VoltageSource {
    fn new(id: &str, v_step: f64, t0: f64) -> Self {
        VoltageSource {
            id: id.to_string(),
            v_step,
            t0,
        }
    }
    /// Emits `V_in` = Heaviside step at `t0`.
    fn run_time_step(&self, t: f64) -> f64 {
        if t >= self.t0 {
            self.v_step
        } else {
            0.0
        }
    }
}

struct Inductor {
    id: String,
    l: f64,
    r: f64,
    i: f64,
    inbox: HashMap<String, f64>,
}

impl Inductor {
    fn new(id: &str, l: f64, r: f64) -> Self {
        Inductor {
            id: id.to_string(),
            l,
            r,
            i: 0.0,
            inbox: HashMap::new(),
        }
    }
    /// `dI/dt = (V_in − I·R − V_C) / L` (Kirchhoff's voltage law). Returns the
    /// emitted current.
    fn run_time_step(&mut self, step_size: f64) -> f64 {
        let v_in = *self.inbox.get("V_in").unwrap_or(&0.0);
        let v_c = *self.inbox.get("V_C").unwrap_or(&0.0);
        self.i += step_size * (v_in - self.i * self.r - v_c) / self.l;
        self.i
    }
}

struct Capacitor {
    id: String,
    c: f64,
    v_c: f64,
    inbox: HashMap<String, f64>,
}

impl Capacitor {
    fn new(id: &str, c: f64) -> Self {
        Capacitor {
            id: id.to_string(),
            c,
            v_c: 0.0,
            inbox: HashMap::new(),
        }
    }
    /// `dV_C/dt = I / C`. Returns the emitted capacitor voltage.
    fn run_time_step(&mut self, step_size: f64) -> f64 {
        let i = *self.inbox.get("I").unwrap_or(&0.0);
        self.v_c += step_size * i / self.c;
        self.v_c
    }
}

struct Recorder {
    id: String,
    inbox: HashMap<String, f64>,
    trace: Vec<TraceRow>,
}

impl Recorder {
    fn new(id: &str) -> Self {
        Recorder {
            id: id.to_string(),
            inbox: HashMap::new(),
            trace: Vec::new(),
        }
    }
    fn run_time_step(&mut self, step_size: f64, t: f64, inductor_i: f64, capacitor_v_c: f64) {
        let v_in = *self.inbox.get("V_in").unwrap_or(&0.0);
        self.trace.push(TraceRow {
            t: t * step_size,
            i: inductor_i,
            v_c: capacitor_v_c,
            v_in,
        });
    }
}

/// Configuration for the series RLC circuit.
#[derive(Clone, Copy, Debug)]
pub struct RLCConfig {
    pub r: f64,
    pub l: f64,
    pub c: f64,
    pub v_step: f64,
    pub t: f64,
    pub dt: f64,
}

/// Result of an RLC run.
#[derive(Clone, Debug)]
pub struct RLCResult {
    pub config: RLCConfig,
    pub ticks: usize,
    pub trace: Vec<TraceRow>,
}

/// Run the forward-Euler RLC step response.
pub fn run_rlc(cfg: RLCConfig) -> RLCResult {
    let src = VoltageSource::new("src", cfg.v_step, 0.0);
    let mut ind = Inductor::new("L", cfg.l, cfg.r);
    let mut cap = Capacitor::new("C", cfg.c);
    let mut rec = Recorder::new("rec");

    let n = (cfg.t / cfg.dt).round() as usize;
    for t in 0..n {
        let tf = t as f64;
        // Phase 1: run each station, reading frozen inboxes, staging emissions.
        let v = src.run_time_step(tf); // emits V_in -> ind, rec
        let i = ind.run_time_step(cfg.dt); // emits I -> cap
        let v_c = cap.run_time_step(cfg.dt); // emits V_C -> ind
        rec.run_time_step(cfg.dt, tf, ind.i, cap.v_c);

        // Phase 2: commit pending emissions into target inboxes for NEXT tick.
        ind.inbox.insert("V_in".to_string(), v);
        ind.inbox.insert("V_C".to_string(), v_c);
        cap.inbox.insert("I".to_string(), i);
        rec.inbox.insert("V_in".to_string(), v);
    }

    RLCResult {
        config: cfg,
        ticks: n,
        trace: rec.trace,
    }
}

/// Underdamped default: ω0 ≈ 1 rad/s, α = R/(2L) ≈ 0.1 (mild damping).
fn default_config(dt: f64) -> RLCConfig {
    RLCConfig {
        r: 0.2,
        l: 1.0,
        c: 1.0,
        v_step: 1.0,
        t: 30.0,
        dt,
    }
}

/// Entry point (TS top-level `main`). Env vars: `DTS`, `T`.
pub fn run() {
    let dts: Vec<f64> = std::env::var("DTS")
        .unwrap_or_else(|_| "0.5,0.1,0.05,0.01,0.005,0.001".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let t = std::env::var("T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0_f64);

    println!("# Series RLC step response sweep");
    println!("#   R=0.2 ohm, L=1 H, C=1 F, V_step=1 V");
    println!("#   ω0 = 1 rad/s, α = R/(2L) = 0.1, period = 2π s");
    println!("#   T = {t} s");

    for dt in dts {
        let mut cfg = default_config(dt);
        cfg.t = t;
        let result = run_rlc(cfg);
        let last = *result.trace.last().expect("non-empty trace");
        println!(
            "  dt={:<8}  ticks={:>6}  V_C({:.3})={:.6}  I={:.2e}",
            dt, result.ticks, last.t, last.v_c, last.i
        );
    }

    // PORT NOTE: the TS writes out/electric-circuit-framework.json via fs; the
    // JSON serialization is omitted here (no serde dependency assumed).
    println!("# (JSON artifact write omitted in port — see PORT NOTE)");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At small dt the underdamped RLC settles toward V_C → V_step.
    #[test]
    fn settles_to_step_amplitude() {
        let cfg = RLCConfig {
            r: 0.2,
            l: 1.0,
            c: 1.0,
            v_step: 1.0,
            t: 200.0,
            dt: 0.001,
        };
        let result = run_rlc(cfg);
        let last = *result.trace.last().unwrap();
        assert!(
            (last.v_c - 1.0).abs() < 0.05,
            "V_C={} not near step",
            last.v_c
        );
    }
}
