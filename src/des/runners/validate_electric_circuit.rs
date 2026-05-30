//! Port of `src/des/runners/validate-electric-circuit.ts`.
//!
//! Compares the framework series-RLC step-response (forward Euler at several
//! `dt`) against the analytical closed form and scipy LSODA, reporting
//! max-abs-error and the empirical convergence order. Top-level `main()` → [`run`].
//!
//! PORT NOTES:
//!   * JSON loading is stubbed (no `serde`/`serde_json` dependency yet). The
//!     `load_json` helper reproduces the missing-file `exit(1)` and documents the
//!     `serde_json::from_str` call to wire.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::{Path, PathBuf};

// =============================================================================
// Typed views of the two JSON files (PORT NOTE: `#[derive(Deserialize)]`).
// =============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct CircuitConfig {
    r: f64,
    l: f64,
    c: f64,
    t: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TracePoint {
    t: f64,
    v_c: f64,
    i: f64,
    v_in: f64,
}

#[derive(Clone, Debug, Default)]
struct SweepRun {
    dt: f64,
    ticks: usize,
    trace: Vec<TracePoint>,
}

#[derive(Clone, Debug, Default)]
struct FrameworkJson {
    config: CircuitConfig,
    sweep: Vec<SweepRun>,
}

#[derive(Clone, Debug, Default)]
struct SelfCheck {
    max_abs_v_c: f64,
}

#[derive(Clone, Debug, Default)]
struct ReferenceJson {
    config: CircuitConfig,
    self_check: SelfCheck,
    t: Vec<f64>,
    v_c_analytical: Vec<f64>,
    i_analytical: Vec<f64>,
    v_c_scipy: Vec<f64>,
}

fn load_json<T>(p: &Path) -> T {
    if !p.exists() {
        eprintln!("[validate-electric-circuit] missing {}", p.display());
        std::process::exit(1);
    }
    // PORT NOTE: `serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()`.
    eprintln!(
        "[validate-electric-circuit] PORT NOTE: JSON parsing not wired (needs serde_json): {}",
        p.display()
    );
    std::process::exit(1);
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
    let ts_path = root().join("out").join("electric-circuit-framework.json");
    let ref_path = root()
        .join("out")
        .join("external")
        .join("electric-circuit")
        .join("reference.json");

    let ts: FrameworkJson = load_json(&ts_path);
    let r#ref: ReferenceJson = load_json(&ref_path);

    println!("Series RLC step response: framework vs analytical + scipy LSODA");
    println!("=================================================================");
    println!("  R={} ohm, L={} H, C={} F", ts.config.r, ts.config.l, ts.config.c);
    println!(
        "  α = R/(2L) = {:.4} rad/s",
        ts.config.r / (2.0 * ts.config.l)
    );
    println!(
        "  ω0 = 1/√(LC) = {:.4} rad/s",
        1.0 / (ts.config.l * ts.config.c).sqrt()
    );
    println!(
        "  T = {} s    (LSODA self-check max|V_C err| = {:.2e})",
        ts.config.t, r#ref.self_check.max_abs_v_c
    );
    println!();
    println!(
        "  {:<8} {:>6}  {:>22}  {:>20}  {:>8}",
        "dt", "ticks", "max|V_C - analytical|", "max|V_C - scipy|", "order"
    );

    let t_grid = &r#ref.t;
    let ref_v = &r#ref.v_c_analytical;
    let _ref_i = &r#ref.i_analytical;
    let sci_v = &r#ref.v_c_scipy;

    let mut prev_err = -1.0_f64;
    let mut prev_dt = -1.0_f64;
    for run in &ts.sweep {
        let (v_ts, _i_ts) = resample(&run.trace, t_grid);
        let err_ana = max_abs(&v_ts, ref_v);
        let err_sci = max_abs(&v_ts, sci_v);

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
            format!("{:.3e}", err_sci),
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
    let err_small = max_abs(&v_small, sci_v);
    let ok = err_small < 5e-3;
    println!();
    println!(
        "  Tightest dt = {}: max|V_C - scipy| = {:.3e}    threshold = 5e-3",
        smallest.dt, err_small
    );
    println!("{}", if ok { "  PASS" } else { "  FAIL" });

    std::process::exit(if ok { 0 } else { 1 });
}
