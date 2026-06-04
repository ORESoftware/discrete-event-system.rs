//! Port of `src/des/runners/validate-two-disease.ts`.
//!
//! Compares the framework two-disease ensemble mean against the scipy LSODA ODE
//! and the Python Gillespie SSA ensemble: per-tick max-relative error, the
//! time-integrated populations, and a Welch t-test on the final death count.
//! The framework ensemble is generated from the Rust engine in-process; the
//! Python comparison is skipped when the reference artifact is absent.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::main_two_disease::{
    run_two_disease, CompartmentId, TwoDiseaseParams as EngineTwoDiseaseParams,
    TwoDiseaseTrace as EngineTwoDiseaseTrace,
};

// =============================================================================
// Typed views of the two JSON files. The framework writer emits uppercase
// compartment keys (`S/A/B/AB/R/D`) and camelCase params; the python reference
// is snake_case. `serde(default)` keeps both tolerant of omitted fields.
// =============================================================================

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct MeanTrace {
    t: Vec<f64>,
    #[serde(rename = "S")]
    s: Vec<f64>,
    #[serde(rename = "A")]
    a: Vec<f64>,
    #[serde(rename = "B")]
    b: Vec<f64>,
    #[serde(rename = "AB")]
    ab: Vec<f64>,
    #[serde(rename = "R")]
    r: Vec<f64>,
    #[serde(rename = "D")]
    d: Vec<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct Compartments {
    #[serde(rename = "S")]
    s: Vec<f64>,
    #[serde(rename = "A")]
    a: Vec<f64>,
    #[serde(rename = "B")]
    b: Vec<f64>,
    #[serde(rename = "AB")]
    ab: Vec<f64>,
    #[serde(rename = "R")]
    r: Vec<f64>,
    #[serde(rename = "D")]
    d: Vec<f64>,
}

impl MeanTrace {
    fn get(&self, k: &str) -> &Vec<f64> {
        match k {
            "S" => &self.s,
            "A" => &self.a,
            "B" => &self.b,
            "AB" => &self.ab,
            "R" => &self.r,
            "D" => &self.d,
            _ => &self.t,
        }
    }
}

impl Compartments {
    fn get(&self, k: &str) -> &Vec<f64> {
        match k {
            "S" => &self.s,
            "A" => &self.a,
            "B" => &self.b,
            "AB" => &self.ab,
            "R" => &self.r,
            "D" => &self.d,
            _ => &self.s,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct TwoDiseaseParams {
    n: f64,
    sim_t: f64,
    step_size: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct FrameworkJson {
    mean_trace: MeanTrace,
    params: TwoDiseaseParams,
    reps: usize,
    final_deaths: Vec<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct PythonJson {
    ode: Compartments,
    ssa_mean: Compartments,
    ssa_final_d_mean: f64,
    ssa_final_d_std: f64,
    ssa_reps: f64,
}

fn load_optional_json<T: serde::de::DeserializeOwned>(p: &Path) -> Option<T> {
    if !p.exists() {
        return None;
    }
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
        eprintln!("[validate-two-disease] read error {}: {e}", p.display());
        std::process::exit(1);
    });
    Some(serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("[validate-two-disease] parse error {}: {e}", p.display());
        std::process::exit(1);
    }))
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn framework_engine_params() -> EngineTwoDiseaseParams {
    EngineTwoDiseaseParams {
        n: env_usize("N", 1000),
        initial_a: env_usize("INIT_A", 5),
        initial_b: env_usize("INIT_B", 5),
        initial_ab: env_usize("INIT_AB", 0),
        beta_a: env_f64("BETA_A", 0.5),
        beta_b: env_f64("BETA_B", 0.4),
        gamma_a: env_f64("GAMMA_A", 1.0 / 7.0),
        gamma_b: env_f64("GAMMA_B", 1.0 / 10.0),
        gamma_ab: env_f64("GAMMA_AB", 1.0 / 8.0),
        p_death_a: env_f64("P_D_A", 0.40),
        p_death_b: env_f64("P_D_B", 0.60),
        p_death_ab: env_f64("P_D_AB", 0.50),
        sim_t: env_f64("SIM_T", 200.0),
        step_size: env_f64("STEPSIZE", 0.1),
        seed: env_usize("SEED", 1) as u32,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn build_mean_trace(traces: &[EngineTwoDiseaseTrace]) -> MeanTrace {
    let t_len = traces.first().map(|tr| tr.t.len()).unwrap_or(0);
    let mut mean_trace = MeanTrace {
        t: traces.first().map(|tr| tr.t.clone()).unwrap_or_default(),
        ..Default::default()
    };
    for i in 0..t_len {
        mean_trace
            .s
            .push(mean(&traces.iter().map(|tr| tr.s[i]).collect::<Vec<_>>()));
        mean_trace
            .a
            .push(mean(&traces.iter().map(|tr| tr.a[i]).collect::<Vec<_>>()));
        mean_trace
            .b
            .push(mean(&traces.iter().map(|tr| tr.b[i]).collect::<Vec<_>>()));
        mean_trace
            .ab
            .push(mean(&traces.iter().map(|tr| tr.ab[i]).collect::<Vec<_>>()));
        mean_trace
            .r
            .push(mean(&traces.iter().map(|tr| tr.r[i]).collect::<Vec<_>>()));
        mean_trace
            .d
            .push(mean(&traces.iter().map(|tr| tr.d[i]).collect::<Vec<_>>()));
    }
    mean_trace
}

fn build_framework_json() -> FrameworkJson {
    let params = framework_engine_params();
    let reps = env_usize("REPS", 30).max(1);
    let mut traces = Vec::with_capacity(reps);
    let mut final_deaths = Vec::with_capacity(reps);
    for rep in 0..reps {
        let mut cfg = params;
        cfg.seed = params.seed + rep as u32;
        let result = run_two_disease(&cfg);
        final_deaths.push(result.final_counts.d as f64);
        traces.push(result.trace);
    }
    FrameworkJson {
        mean_trace: build_mean_trace(&traces),
        params: TwoDiseaseParams {
            n: params.n as f64,
            sim_t: params.sim_t,
            step_size: params.step_size,
        },
        reps,
        final_deaths,
    }
}

fn framework_ok(ts: &FrameworkJson) -> bool {
    let len = ts.mean_trace.t.len();
    len > 0
        && ts.reps > 0
        && ts.final_deaths.len() == ts.reps
        && ["S", "A", "B", "AB", "R", "D"]
            .iter()
            .all(|k| ts.mean_trace.get(k).len() == len)
        && ts
            .final_deaths
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= ts.params.n)
}

/// `maxRelDiff` → `(max, mean_abs)`.
fn max_rel_diff(a: &[f64], b: &[f64], floor: f64) -> (f64, f64) {
    if a.len() != b.len() {
        panic!("length mismatch {} vs {}", a.len(), b.len());
    }
    let mut mx = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for i in 0..a.len() {
        let denom = floor.max(a[i].abs() + b[i].abs());
        let r = (a[i] - b[i]).abs() / denom;
        if r > mx {
            mx = r;
        }
        sum += (a[i] - b[i]).abs();
        n += 1;
    }
    (mx, sum / (n.max(1) as f64))
}

/// Time-integrated population: trapezoid rule on (t, x).
fn time_integrate(t: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 1..t.len() {
        sum += 0.5 * (x[i] + x[i - 1]) * (t[i] - t[i - 1]);
    }
    sum
}

/// Abramowitz & Stegun erf.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let tt = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * (-x * x).exp();
    sign * y
}

struct WelchOut {
    t: f64,
    df: f64,
    p: f64,
}

/// `welchT` (defined in the TS but kept for parity — primarily diagnostic).
fn welch_t(xs: &[f64], ys: &[f64]) -> WelchOut {
    let m = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
    let v = |a: &[f64]| {
        let mu = m(a);
        a.iter().map(|&vv| (vv - mu) * (vv - mu)).sum::<f64>() / (1.0_f64).max(a.len() as f64 - 1.0)
    };
    let mx = m(xs);
    let my = m(ys);
    let vx = v(xs);
    let vy = v(ys);
    let nx = xs.len() as f64;
    let ny = ys.len() as f64;
    let se = (vx / nx + vy / ny).sqrt();
    let t = if se == 0.0 { 0.0 } else { (mx - my) / se };
    let num = (vx / nx + vy / ny).powi(2);
    let den =
        (vx / nx).powi(2) / (1.0_f64).max(nx - 1.0) + (vy / ny).powi(2) / (1.0_f64).max(ny - 1.0);
    let df = if den == 0.0 { f64::INFINITY } else { num / den };
    let z = t.abs();
    let phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    WelchOut {
        t,
        df,
        p: 2.0 * (1.0 - phi),
    }
}

/// `validate-two-disease.ts` `main()`.
pub fn run() {
    let py_path = root()
        .join("out")
        .join("external")
        .join("two-disease")
        .join("python.json");

    let ts = build_framework_json();
    let py: Option<PythonJson> = load_optional_json(&py_path);

    let mean_ts = &ts.mean_trace;

    println!("Two-disease framework vs Python (LSODA + Gillespie SSA)");
    println!("==========================================================================");
    println!(
        "  N={}  reps={}  simT={}  dt={}",
        ts.params.n, ts.reps, ts.params.sim_t, ts.params.step_size
    );
    println!();

    if py.is_none() {
        let ok = framework_ok(&ts);
        let final_d_mean = mean(&ts.final_deaths);
        println!(
            "  framework ensemble produced {} time points and {} final-D samples",
            mean_ts.t.len(),
            ts.final_deaths.len()
        );
        println!("  final D mean = {:.2}", final_d_mean);
        println!(
            "  SKIP  Python LSODA/Gillespie comparison (reference JSON unavailable: {})",
            py_path.display()
        );
        println!("{}", if ok { "  PASS" } else { "  FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    let py = py.expect("checked present");
    let ode = &py.ode;
    let ssa = &py.ssa_mean;

    let compartments = ["S", "A", "B", "AB", "R", "D"];
    println!(
        "  Trajectory (ensemble mean) — framework ({} reps) vs LSODA / SSA-mean",
        ts.reps
    );
    println!("  Compartment  |  max-rel  |  ∫framework  |   ∫LSODA   |   ∫SSA   | rel-err vs LSODA | rel-err vs SSA");
    println!("  ─────────────┼───────────┼──────────────┼────────────┼──────────┼──────────────────┼────────────────");

    let mut int_err_ode: HashMap<String, f64> = HashMap::new();
    let mut int_err_ssa: HashMap<String, f64> = HashMap::new();
    let mut worst_peak_ode = 0.0_f64;
    for k in compartments {
        if k == "t" {
            continue;
        }
        let f = mean_ts.get(k);
        let o = ode.get(k);
        let s = ssa.get(k);
        let t_arr = &mean_ts.t;
        let (o_r, _) = max_rel_diff(f, o, 5.0);
        if o_r > worst_peak_ode {
            worst_peak_ode = o_r;
        }
        let int_f = time_integrate(t_arr, f);
        let int_o = time_integrate(t_arr, o);
        let int_s = time_integrate(t_arr, s);
        let int_r_ode = (int_f - int_o).abs() / (1.0_f64).max(int_o);
        let int_r_ssa = (int_f - int_s).abs() / (1.0_f64).max(int_s);
        int_err_ode.insert(k.to_string(), int_r_ode);
        int_err_ssa.insert(k.to_string(), int_r_ssa);
        println!(
            "  {:<11}  |  {:>6} %  |  {:>8}    |  {:>8}  |  {:>6}  |  {:>13} %  |  {:>11} %",
            k,
            format!("{:.1}", o_r * 100.0),
            format!("{:.0}", int_f),
            format!("{:.0}", int_o),
            format!("{:.0}", int_s),
            format!("{:.2}", int_r_ode * 100.0),
            format!("{:.2}", int_r_ssa * 100.0)
        );
    }

    // Final-state Welch test on D.
    let ts_final_d = &ts.final_deaths;
    let py_ssa_mean_d = py.ssa_final_d_mean;
    let py_ssa_std_d = py.ssa_final_d_std;
    let py_ssa_reps = py.ssa_reps;
    let ts_mean_d = ts_final_d.iter().sum::<f64>() / ts_final_d.len() as f64;
    let ts_std_d = (ts_final_d
        .iter()
        .map(|&v| (v - ts_mean_d) * (v - ts_mean_d))
        .sum::<f64>()
        / (1.0_f64).max(ts_final_d.len() as f64 - 1.0))
    .sqrt();
    let se_gap = (ts_std_d * ts_std_d / ts_final_d.len() as f64
        + py_ssa_std_d * py_ssa_std_d / py_ssa_reps)
        .sqrt();
    let t_stat = if se_gap == 0.0 {
        0.0
    } else {
        (ts_mean_d - py_ssa_mean_d) / se_gap
    };
    let z = t_stat.abs();
    let p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2)));
    println!();
    println!("  Welch test on final-D (framework reps vs SSA reps):");
    println!(
        "    framework: mean={:.2}  std={:.2}  n={}",
        ts_mean_d,
        ts_std_d,
        ts_final_d.len()
    );
    println!(
        "    Python SSA: mean={:.2}  std={:.2}  n={}",
        py_ssa_mean_d, py_ssa_std_d, py_ssa_reps
    );
    println!("    t = {:.3}    p ≈ {:.3}", t_stat, p);

    let ode_final_d = ode.d[ode.d.len() - 1];
    println!(
        "    LSODA mean-field final D = {:.2} (compare to SSA mean {:.2})",
        ode_final_d, py_ssa_mean_d
    );

    let tol_int_ode_mon = 0.05;
    let tol_int_ode_transient = 0.20;
    let tol_int_ssa = 0.10;
    let tol_peak = 0.50;
    let mon_ok = ["R", "D"].iter().all(|k| int_err_ode[*k] < tol_int_ode_mon);
    let transient_ok = ["S", "A", "B", "AB"]
        .iter()
        .all(|k| int_err_ode[*k] < tol_int_ode_transient);
    let large_ok = mon_ok;
    let small_ok = transient_ok;
    let ssa_ok = compartments
        .iter()
        .filter(|k| **k != "t")
        .all(|k| int_err_ssa[*k] < tol_int_ssa);
    let ok_peak = worst_peak_ode < tol_peak;
    let ok_welch = p > 0.01;
    println!();
    println!(
        "  ∫-rel-err vs LSODA, monotonic (R,D)        < {:.0}%: {}",
        tol_int_ode_mon * 100.0,
        if mon_ok { "yes" } else { "NO" }
    );
    println!(
        "  ∫-rel-err vs LSODA, transient (S,A,B,AB)   < {:.0}%: {}",
        tol_int_ode_transient * 100.0,
        if transient_ok { "yes" } else { "NO" }
    );
    println!(
        "  ∫-rel-err vs SSA-mean (all)       < {:.0}%: {}",
        tol_int_ssa * 100.0,
        if ssa_ok { "yes" } else { "NO" }
    );
    println!(
        "  max peak-rel-err vs LSODA         < {:.0}%: {}  (got {:.2}%)",
        tol_peak * 100.0,
        if ok_peak { "yes" } else { "NO" },
        worst_peak_ode * 100.0
    );
    println!(
        "  Welch p > 0.01 (final D)              : {}  (got p={:.3})",
        if ok_welch { "yes" } else { "NO" },
        p
    );

    let ok = large_ok && small_ok && ssa_ok && ok_peak && ok_welch;
    println!("{}", if ok { "  PASS" } else { "  FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
