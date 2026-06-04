//! Port of `src/des/runners/validate-convolution.ts`.
//!
//! Compares the framework's convolution output (`out/convolution-framework.json`)
//! against a direct Rust convolution reference. If
//! `out/external/convolution/numpy.json` exists, it is used as an optional
//! sidecar reference instead. The runner reports max-abs error + RMSE and
//! asserts agreement to within a ULP-scaled tolerance. The TS top-level
//! `main()` becomes [`run`], returning the process exit code.
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

fn direct_convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }
    let mut y = vec![0.0; signal.len() + kernel.len() - 1];
    for n in 0..y.len() {
        let mut acc = 0.0;
        for k in 0..kernel.len() {
            if n >= k {
                let i = n - k;
                if i < signal.len() {
                    acc += kernel[k] * signal[i];
                }
            }
        }
        y[n] = acc;
    }
    y
}

fn write_rust_reference(root: &std::path::Path, y: &[f64]) {
    let dir = root.join("out").join("external").join("convolution");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let json = JsonValue::Object(vec![
        (
            "solver".to_string(),
            JsonValue::String("rust:direct-convolution".to_string()),
        ),
        (
            "y".to_string(),
            JsonValue::Array(y.iter().copied().map(JsonValue::Number).collect()),
        ),
    ]);
    let _ = std::fs::write(dir.join("rust-direct.json"), json.to_string_pretty(2));
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
    let y_ts = arr_f64(&ts, "y");
    let signal = arr_f64(&ts, "signal");
    let kernel = arr_f64(&ts, "kernel");
    let (reference_name, y_ref) = if np_path.exists() {
        match load_json(&np_path) {
            Ok(np) => ("NumPy sidecar", arr_f64(&np, "y")),
            Err(code) => return code,
        }
    } else {
        if signal.is_empty() || kernel.is_empty() {
            eprintln!(
                "[validate-convolution] framework JSON must include non-empty signal and kernel arrays for Rust reference"
            );
            return 1;
        }
        let y = direct_convolve(&signal, &kernel);
        write_rust_reference(&root, &y);
        ("Rust direct convolution", y)
    };

    if y_ts.len() != y_ref.len() {
        eprintln!(
            "length mismatch: framework={} reference({})={}",
            y_ts.len(),
            reference_name,
            y_ref.len()
        );
        return 1;
    }

    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut arg_max: i64 = -1;
    for i in 0..y_ts.len() {
        let e = (y_ts[i] - y_ref[i]).abs();
        sum_sq += e * e;
        if e > max_abs {
            max_abs = e;
            arg_max = i as i64;
        }
    }
    let rmse = (sum_sq / y_ts.len().max(1) as f64).sqrt();

    println!("Convolution: framework vs {reference_name}");
    println!("===========================================");
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
