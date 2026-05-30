//! Port of `src/des/runners/validate-convolution.ts`.
//!
//! Compares the framework's convolution output (`out/convolution-framework.json`)
//! against `numpy.convolve` (`out/external/convolution/numpy.json`): reports
//! max-abs error + RMSE and asserts agreement to within a ULP-scaled tolerance.
//! The TS top-level `main()` becomes [`run`], returning the process exit code.
//!
//! ## PORT NOTE
//!   * `__dirname/../../..` repo root → `REPO_ROOT` env var or the current
//!     working directory.
//!   * `fs`/`JSON.parse` → `std::fs` + [`parse_json`].
//!   * `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::observability::logger::{parse_json, JsonValue};

fn root() -> PathBuf {
    match std::env::var("REPO_ROOT") {
        Ok(r) => PathBuf::from(r),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

fn load_json(p: &std::path::Path) -> Result<JsonValue, i32> {
    if !p.exists() {
        eprintln!("[validate-convolution] missing {}", p.display());
        return Err(1);
    }
    let text = std::fs::read_to_string(p).map_err(|_| 1)?;
    parse_json(&text).map_err(|e| {
        eprintln!("[validate-convolution] parse error: {e}");
        1
    })
}

fn arr_f64(v: &JsonValue, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(|a| a.as_array())
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default()
}

fn arr_len(v: &JsonValue, key: &str) -> usize {
    v.get(key)
        .and_then(|a| a.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

/// `main()` — returns the exit code (0 = PASS).
pub fn run() -> i32 {
    let root = root();
    let ts_path = root.join("out").join("convolution-framework.json");
    let np_path = root
        .join("out")
        .join("external")
        .join("convolution")
        .join("numpy.json");

    let ts = match load_json(&ts_path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let np = match load_json(&np_path) {
        Ok(v) => v,
        Err(code) => return code,
    };

    let y_ts = arr_f64(&ts, "y");
    let y_np = arr_f64(&np, "y");

    if y_ts.len() != y_np.len() {
        eprintln!(
            "length mismatch: framework={} numpy={}",
            y_ts.len(),
            y_np.len()
        );
        return 1;
    }

    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut arg_max: i64 = -1;
    for i in 0..y_ts.len() {
        let e = (y_ts[i] - y_np[i]).abs();
        sum_sq += e * e;
        if e > max_abs {
            max_abs = e;
            arg_max = i as i64;
        }
    }
    let rmse = (sum_sq / y_ts.len().max(1) as f64).sqrt();

    println!("Convolution: framework vs numpy.convolve");
    println!("==========================================");
    println!("  signal length     = {}", arr_len(&ts, "signal"));
    println!("  kernel length     = {}", arr_len(&ts, "kernel"));
    println!("  output length     = {}", y_ts.len());
    println!("  max-abs-error     = {:e}  (at i={arg_max})", max_abs);
    println!("  RMSE              = {:e}", rmse);

    let peak = y_ts.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let ulp_at_peak = peak.max(1.0) * 2f64.powi(-52);
    let tolerance = (1e-12_f64).max(1024.0 * ulp_at_peak);

    println!("  peak |y|          = {:e}", peak);
    println!("  1024 * ULP(peak)  = {:e}", 1024.0 * ulp_at_peak);
    println!("  tolerance         = {:e}", tolerance);

    let ok = max_abs < tolerance;
    println!();
    println!("{}", if ok { "  PASS" } else { "  FAIL" });
    if ok {
        0
    } else {
        1
    }
}
