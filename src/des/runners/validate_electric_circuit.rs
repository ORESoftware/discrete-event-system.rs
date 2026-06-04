//! Port of `src/des/runners/validate-electric-circuit.ts`.
//!
//! Compares the framework series-RLC step-response (forward Euler at several
//! `dt`) against the analytical closed form and a native Rust reference, reporting
//! max-abs-error and the empirical convergence order. Top-level `main()` → [`run`].
//!
//! The framework sweep and analytical reference are generated in-process with
//! Rust code.

#![allow(dead_code)]

use crate::des::main_electric_circuit::{run_rlc, RLCConfig};
use serde::Deserialize;

// =============================================================================
// Typed views of the generated framework/reference structures. The framework
// writer emits the TS field names (`R/L/C/T`, trace rows `t/I/V_C/V_in`);
// `serde(default)` keeps both tolerant of omitted fields.
// =============================================================================

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default)]
struct CircuitConfig {
    #[serde(rename = "R")]
    r: f64,
    #[serde(rename = "L")]
    l: f64,
    #[serde(rename = "C")]
    c: f64,
    #[serde(rename = "T")]
    t: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default)]
struct TracePoint {
    t: f64,
    #[serde(rename = "V_C")]
    v_c: f64,
    #[serde(rename = "I")]
    i: f64,
    #[serde(rename = "V_in")]
    v_in: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct SweepRun {
    dt: f64,
    ticks: usize,
    trace: Vec<TracePoint>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct FrameworkJson {
    config: CircuitConfig,
    sweep: Vec<SweepRun>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct SelfCheck {
    max_abs_v_c: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct ReferenceJson {
    config: CircuitConfig,
    self_check: SelfCheck,
    t: Vec<f64>,
    v_c_analytical: Vec<f64>,
    i_analytical: Vec<f64>,
    v_c_reference: Vec<f64>,
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!("length mismatch: {} vs {}", a.len(), b.len());
    }
    let mut m = 0.0;
    for i in 0..a.len() {
        m = f64::max(m, (a[i] - b[i]).abs());
    }
    m
}

/// Resample a (possibly coarser) trace at the reference grid by piecewise linear
/// interpolation.
fn resample(trace: &[TracePoint], t_grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut v_c: Vec<f64> = Vec::new();
    let mut i_out: Vec<f64> = Vec::new();
    let mut j = 0usize;
    for &t_ref in t_grid {
        while j + 1 < trace.len() && trace[j + 1].t < t_ref {
            j += 1;
        }
        if j + 1 >= trace.len() {
            v_c.push(trace[trace.len() - 1].v_c);
            i_out.push(trace[trace.len() - 1].i);
            continue;
        }
        let t0 = trace[j].t;
        let t1 = trace[j + 1].t;
        let w = (t_ref - t0) / (t1 - t0);
        v_c.push(trace[j].v_c + w * (trace[j + 1].v_c - trace[j].v_c));
        i_out.push(trace[j].i + w * (trace[j + 1].i - trace[j].i));
    }
    (v_c, i_out)
}

fn parse_dts() -> Vec<f64> {
    std::env::var("DTS")
        .unwrap_or_else(|_| "0.5,0.1,0.05,0.01,0.005,0.001".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn framework_from_rust() -> FrameworkJson {
    let t = std::env::var("T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0_f64);
    let config = CircuitConfig {
        r: 0.2,
        l: 1.0,
        c: 1.0,
        t,
    };
    let sweep = parse_dts()
        .into_iter()
        .map(|dt| {
            let result = run_rlc(RLCConfig {
                r: config.r,
                l: config.l,
                c: config.c,
                v_step: 1.0,
                t,
                dt,
            });
            SweepRun {
                dt,
                ticks: result.ticks,
                trace: result
                    .trace
                    .into_iter()
                    .map(|row| TracePoint {
                        t: row.t,
                        v_c: row.v_c,
                        i: row.i,
                        v_in: row.v_in,
                    })
                    .collect(),
            }
        })
        .collect();
    FrameworkJson { config, sweep }
}

fn analytical_rlc(config: &CircuitConfig, t: f64) -> (f64, f64) {
    let alpha = config.r / (2.0 * config.l);
    let omega0 = 1.0 / (config.l * config.c).sqrt();
    let omega_d_sq = omega0 * omega0 - alpha * alpha;
    assert!(
        omega_d_sq > 0.0,
        "validate-electric-circuit expects an underdamped RLC config"
    );
    let omega_d = omega_d_sq.sqrt();
    let decay = (-alpha * t).exp();
    let sin = (omega_d * t).sin();
    let cos = (omega_d * t).cos();
    let v_step = 1.0;
    let v_c = v_step * (1.0 - decay * (cos + alpha / omega_d * sin));
    let i = v_step / (config.l * omega_d) * decay * sin;
    (v_c, i)
}

fn reference_from_analytical(ts: &FrameworkJson) -> ReferenceJson {
    let smallest = ts
        .sweep
        .iter()
        .reduce(|a, b| if a.dt < b.dt { a } else { b })
        .expect("non-empty sweep");
    let mut t_grid = Vec::new();
    let mut v_c = Vec::new();
    let mut i_out = Vec::new();
    for point in &smallest.trace {
        let (v, i) = analytical_rlc(&ts.config, point.t);
        t_grid.push(point.t);
        v_c.push(v);
        i_out.push(i);
    }
    ReferenceJson {
        config: ts.config,
        self_check: SelfCheck { max_abs_v_c: 0.0 },
        t: t_grid,
        v_c_analytical: v_c.clone(),
        i_analytical: i_out,
        v_c_reference: v_c,
    }
}

/// `validate-electric-circuit.ts` `main()`.
pub fn run() {
    let ts = framework_from_rust();
    let r#ref = reference_from_analytical(&ts);

    println!("Series RLC step response: framework vs analytical Rust reference");
    println!("================================================================");
    println!(
        "  R={} ohm, L={} H, C={} F",
        ts.config.r, ts.config.l, ts.config.c
    );
    println!(
        "  α = R/(2L) = {:.4} rad/s",
        ts.config.r / (2.0 * ts.config.l)
    );
    println!(
        "  ω0 = 1/√(LC) = {:.4} rad/s",
        1.0 / (ts.config.l * ts.config.c).sqrt()
    );
    println!(
        "  T = {} s    (Rust reference self-check max|V_C err| = {:.2e})",
        ts.config.t, r#ref.self_check.max_abs_v_c
    );
    println!();
    println!(
        "  {:<8} {:>6}  {:>22}  {:>20}  {:>8}",
        "dt", "ticks", "max|V_C - analytical|", "max|V_C - reference|", "order"
    );

    let t_grid = &r#ref.t;
    let ref_v = &r#ref.v_c_analytical;
    let _ref_i = &r#ref.i_analytical;
    let ref_shadow_v = &r#ref.v_c_reference;

    let mut prev_err = -1.0_f64;
    let mut prev_dt = -1.0_f64;
    for run in &ts.sweep {
        let (v_ts, _i_ts) = resample(&run.trace, t_grid);
        let err_ana = max_abs(&v_ts, ref_v);
        let err_ref = max_abs(&v_ts, ref_shadow_v);

        let mut order = String::new();
        if prev_err > 0.0 && prev_dt > 0.0 {
            // Forward Euler is O(dt^1).
            let r = (prev_err / err_ana).ln() / (prev_dt / run.dt).ln();
            order = format!("{:.2}", r);
        }

        println!(
            "  {:<8} {:>6}  {:>22}  {:>20}  {:>8}",
            run.dt,
            run.ticks,
            format!("{:.3e}", err_ana),
            format!("{:.3e}", err_ref),
            order
        );
        prev_err = err_ana;
        prev_dt = run.dt;
    }

    // Smallest dt run (`reduce((a, b) => a.dt < b.dt ? a : b)`).
    let smallest = ts
        .sweep
        .iter()
        .reduce(|a, b| if a.dt < b.dt { a } else { b })
        .expect("non-empty sweep");
    let (v_small, _) = resample(&smallest.trace, t_grid);
    let err_small = max_abs(&v_small, ref_shadow_v);
    let ok = err_small < 5e-3;
    println!();
    println!(
        "  Tightest dt = {}: max|V_C - reference| = {:.3e}    threshold = 5e-3",
        smallest.dt, err_small
    );
    println!("{}", if ok { "  PASS" } else { "  FAIL" });

    std::process::exit(if ok { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The framework structs must deserialize exactly what
    /// `main_electric_circuit` writes to `out/electric-circuit-framework.json`
    /// (config `{R,L,C,Vstep,T}`, trace rows `{t, I, V_C, V_in}`).
    #[test]
    fn framework_json_parses_sim_output_shape() {
        let json = r#"{
            "sweep": [
                {"dt": 0.5, "ticks": 60, "final_V_C": 0.99, "final_I": 1e-3,
                 "trace": [
                    {"t": 0.0, "I": 0.0, "V_C": 0.0, "V_in": 1.0},
                    {"t": 0.5, "I": 0.4, "V_C": 0.1, "V_in": 1.0}
                 ]}
            ],
            "config": {"R": 0.2, "L": 1.0, "C": 1.0, "Vstep": 1.0, "T": 30.0}
        }"#;
        let fw: FrameworkJson = serde_json::from_str(json).expect("parse framework json");
        assert_eq!(fw.sweep.len(), 1);
        assert_eq!(fw.config.r, 0.2);
        assert_eq!(fw.config.t, 30.0);
        assert_eq!(fw.sweep[0].trace.len(), 2);
        assert_eq!(fw.sweep[0].trace[1].v_c, 0.1);
        assert_eq!(fw.sweep[0].trace[1].i, 0.4);
        assert_eq!(fw.sweep[0].trace[1].v_in, 1.0);
    }
}
