//! Port of `src/des/runners/validate-electric-circuit.ts`.
//!
//! Compares the framework series-RLC step-response (forward Euler at several
//! `dt`) against the analytical closed form and, when present, scipy LSODA,
//! reporting max-abs-error and the empirical convergence order. Top-level
//! `main()` → [`run`].

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::main_electric_circuit::{run_rlc, RLCConfig};

// =============================================================================
// Typed views of the two JSON files. The framework writer emits the TS field
// names (`R/L/C/T`, trace rows `t/I/V_C/V_in`); the scipy/analytic reference is
// snake_case. `serde(default)` keeps both tolerant of omitted fields.
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
    #[serde(rename = "Vstep")]
    v_step: f64,
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
    v_c_scipy: Vec<f64>,
}

fn load_optional_json<T: serde::de::DeserializeOwned>(p: &Path) -> Option<T> {
    if !p.exists() {
        return None;
    }
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
        eprintln!(
            "[validate-electric-circuit] read error {}: {e}",
            p.display()
        );
        std::process::exit(1);
    });
    Some(serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!(
            "[validate-electric-circuit] parse error {}: {e}",
            p.display()
        );
        std::process::exit(1);
    }))
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_dts() -> Vec<f64> {
    let dts: Vec<f64> = std::env::var("DTS")
        .unwrap_or_else(|_| "0.5,0.1,0.05,0.01,0.005,0.001".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|dt: &f64| *dt > 0.0)
        .collect();
    if dts.is_empty() {
        eprintln!("[validate-electric-circuit] DTS did not contain any positive values");
        std::process::exit(1);
    }
    dts
}

fn default_rlc_config(dt: f64, t: f64) -> RLCConfig {
    RLCConfig {
        r: 0.2,
        l: 1.0,
        c: 1.0,
        v_step: 1.0,
        t,
        dt,
    }
}

fn build_framework_sweep() -> FrameworkJson {
    let t = env_f64("T", 30.0);
    let mut sweep = Vec::new();
    for dt in env_dts() {
        let result = run_rlc(default_rlc_config(dt, t));
        let trace = result
            .trace
            .iter()
            .map(|row| TracePoint {
                t: row.t,
                v_c: row.v_c,
                i: row.i,
                v_in: row.v_in,
            })
            .collect();
        sweep.push(SweepRun {
            dt,
            ticks: result.ticks,
            trace,
        });
    }
    FrameworkJson {
        config: CircuitConfig {
            r: 0.2,
            l: 1.0,
            c: 1.0,
            v_step: 1.0,
            t,
        },
        sweep,
    }
}

fn analytical_step_response(cfg: CircuitConfig, t: f64) -> (f64, f64) {
    let alpha = cfg.r / (2.0 * cfg.l);
    let omega0 = 1.0 / (cfg.l * cfg.c).sqrt();
    let disc = alpha * alpha - omega0 * omega0;
    if disc.abs() <= 1e-12 {
        let exp_term = (-alpha * t).exp();
        let v_c = cfg.v_step * (1.0 - (1.0 + alpha * t) * exp_term);
        let i = cfg.c * cfg.v_step * alpha * alpha * t * exp_term;
        return (v_c, i);
    }
    if disc < 0.0 {
        let omega_d = (-disc).sqrt();
        let exp_term = (-alpha * t).exp();
        let sin = (omega_d * t).sin();
        let cos = (omega_d * t).cos();
        let v_c = cfg.v_step * (1.0 - exp_term * (cos + alpha / omega_d * sin));
        let i = cfg.v_step / (cfg.l * omega_d) * exp_term * sin;
        return (v_c, i);
    }

    let beta = disc.sqrt();
    let r1 = -alpha + beta;
    let r2 = -alpha - beta;
    let a = -cfg.v_step * r2 / (r1 - r2);
    let b = cfg.v_step * r1 / (r1 - r2);
    let y = a * (r1 * t).exp() + b * (r2 * t).exp();
    let dy = a * r1 * (r1 * t).exp() + b * r2 * (r2 * t).exp();
    (cfg.v_step - y, -cfg.c * dy)
}

fn fallback_reference_grid(ts: &FrameworkJson) -> Vec<f64> {
    ts.sweep
        .iter()
        .reduce(|a, b| if a.dt < b.dt { a } else { b })
        .map(|run| run.trace.iter().map(|row| row.t).collect())
        .unwrap_or_default()
}

fn analytical_series(cfg: CircuitConfig, t_grid: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut v_c = Vec::with_capacity(t_grid.len());
    let mut i = Vec::with_capacity(t_grid.len());
    for &t in t_grid {
        let (vc, current) = analytical_step_response(cfg, t);
        v_c.push(vc);
        i.push(current);
    }
    (v_c, i)
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

/// `validate-electric-circuit.ts` `main()`.
pub fn run() {
    let ref_path = root()
        .join("out")
        .join("external")
        .join("electric-circuit")
        .join("reference.json");

    let ts = build_framework_sweep();
    let external_ref: Option<ReferenceJson> = load_optional_json(&ref_path);
    let t_grid = external_ref
        .as_ref()
        .filter(|r| !r.t.is_empty())
        .map(|r| r.t.clone())
        .unwrap_or_else(|| fallback_reference_grid(&ts));
    if t_grid.is_empty() {
        eprintln!("[validate-electric-circuit] no reference times available");
        std::process::exit(1);
    }
    let (ref_v, _ref_i) = analytical_series(ts.config, &t_grid);
    let sci_v = external_ref.as_ref().and_then(|r| {
        if r.v_c_scipy.len() == t_grid.len() {
            Some(r.v_c_scipy.clone())
        } else {
            println!(
                "  SKIP  scipy LSODA reference length mismatch: {} vs {}",
                r.v_c_scipy.len(),
                t_grid.len()
            );
            None
        }
    });

    println!("Series RLC step response: framework vs analytical + scipy LSODA");
    println!("=================================================================");
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
        "  T = {} s{}",
        ts.config.t,
        external_ref
            .as_ref()
            .map(|r| format!(
                "    (LSODA self-check max|V_C err| = {:.2e})",
                r.self_check.max_abs_v_c
            ))
            .unwrap_or_else(|| "    (SKIP scipy LSODA reference unavailable)".to_string())
    );
    println!();
    println!(
        "  {:<8} {:>6}  {:>22}  {:>20}  {:>8}",
        "dt", "ticks", "max|V_C - analytical|", "max|V_C - scipy|", "order"
    );

    let mut prev_err = -1.0_f64;
    let mut prev_dt = -1.0_f64;
    for run in &ts.sweep {
        let (v_ts, _i_ts) = resample(&run.trace, &t_grid);
        let err_ana = max_abs(&v_ts, &ref_v);
        let err_sci = sci_v.as_ref().map(|sci| max_abs(&v_ts, sci));

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
            err_sci
                .map(|err| format!("{:.3e}", err))
                .unwrap_or_else(|| "SKIP".to_string()),
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
    let (v_small, _) = resample(&smallest.trace, &t_grid);
    let err_small_ana = max_abs(&v_small, &ref_v);
    let ok_ana = err_small_ana < 5e-3;
    println!();
    println!(
        "  Tightest dt = {}: max|V_C - analytical| = {:.3e}    threshold = 5e-3",
        smallest.dt, err_small_ana
    );
    let ok_sci = match sci_v.as_ref() {
        Some(sci) => {
            let err_small_sci = max_abs(&v_small, sci);
            println!(
                "  Tightest dt = {}: max|V_C - scipy|      = {:.3e}    threshold = 5e-3",
                smallest.dt, err_small_sci
            );
            err_small_sci < 5e-3
        }
        None => {
            println!("  SKIP  scipy LSODA comparison (reference JSON unavailable)");
            true
        }
    };
    let ok = ok_ana && ok_sci;
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
