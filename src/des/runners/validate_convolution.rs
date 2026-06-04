//! Port of `src/des/runners/validate-convolution.ts`.
//!
//! Compares the framework's streaming convolution output against an independent
//! direct convolution reference and, when present, `numpy.convolve`
//! (`out/external/convolution/numpy.json`): reports max-abs error + RMSE and
//! asserts agreement to within a ULP-scaled tolerance. The TS top-level `main()`
//! becomes [`run`], returning the process exit code.
//!
//! ## PORT NOTE
//!   * `__dirname/../../..` repo root → `REPO_ROOT` env var or the current
//!     working directory.
//!   * `fs`/`JSON.parse` → `std::fs` + [`parse_json`].
//!   * `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use crate::des::general::prng::mulberry32;
use crate::des::main_convolution::run_convolution;
use crate::des::observability::logger::{parse_json, JsonValue};
use crate::des::shared::capabilities::RandomSource;

fn root() -> PathBuf {
    match std::env::var("REPO_ROOT") {
        Ok(r) => PathBuf::from(r),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

fn load_optional_json(p: &Path) -> Result<Option<JsonValue>, i32> {
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(p).map_err(|_| 1)?;
    parse_json(&text).map(Some).map_err(|e| {
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

struct FrameworkRun {
    signal: Vec<f64>,
    kernel: Vec<f64>,
    y: Vec<f64>,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn make_triangle_kernel(k: usize) -> Vec<f64> {
    let k = k.max(1);
    let mut h = vec![0.0; k];
    let peak = (k as f64 - 1.0) / 2.0;
    let mut sum = 0.0;
    for (i, hi) in h.iter_mut().enumerate() {
        *hi = 1.0 - (i as f64 - peak).abs() / (peak + 1.0);
        sum += *hi;
    }
    h.iter().map(|v| v / sum).collect()
}

fn make_test_signal(n: usize, seed: u32) -> Vec<f64> {
    let mut rng = mulberry32(seed);
    let mut out = vec![0.0; n];
    for (i, oi) in out.iter_mut().enumerate() {
        let i_f = i as f64;
        *oi = (2.0 * PI * 0.1 * i_f).sin()
            + 0.5 * (2.0 * PI * 0.4 * i_f).cos()
            + 0.1 * (rng.next_float() - 0.5);
    }
    out
}

fn build_framework_run() -> FrameworkRun {
    let n = env_usize("N", 64);
    let k = env_usize("K", 7);
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42u32);
    let signal = make_test_signal(n, seed);
    let kernel = make_triangle_kernel(k);
    let result = run_convolution(&signal, &kernel);
    FrameworkRun {
        signal,
        kernel,
        y: result.y,
    }
}

fn direct_convolution(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }
    let mut y = vec![0.0; signal.len() + kernel.len() - 1];
    for (i, &x) in signal.iter().enumerate() {
        for (j, &h) in kernel.iter().enumerate() {
            y[i + j] += x * h;
        }
    }
    y
}

struct ErrorStats {
    max_abs: f64,
    rmse: f64,
    arg_max: i64,
}

fn error_stats(a: &[f64], b: &[f64]) -> Result<ErrorStats, i32> {
    if a.len() != b.len() {
        eprintln!(
            "length mismatch: framework={} reference={}",
            a.len(),
            b.len()
        );
        return Err(1);
    }
    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut arg_max: i64 = -1;
    for i in 0..a.len() {
        let e = (a[i] - b[i]).abs();
        sum_sq += e * e;
        if e > max_abs {
            max_abs = e;
            arg_max = i as i64;
        }
    }
    Ok(ErrorStats {
        max_abs,
        rmse: (sum_sq / a.len().max(1) as f64).sqrt(),
        arg_max,
    })
}

/// `main()` — returns the exit code (0 = PASS).
pub fn run() -> i32 {
    let root = root();
    let np_path = root
        .join("out")
        .join("external")
        .join("convolution")
        .join("numpy.json");

    let ts = build_framework_run();
    let reference = direct_convolution(&ts.signal, &ts.kernel);
    let direct_stats = match error_stats(&ts.y, &reference) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let np = match load_optional_json(&np_path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let numpy_stats = match np.as_ref() {
        Some(np) => match error_stats(&ts.y, &arr_f64(np, "y")) {
            Ok(stats) => Some(stats),
            Err(code) => return code,
        },
        None => None,
    };

    println!("Convolution: framework vs numpy.convolve");
    println!("==========================================");
    println!("  signal length     = {}", ts.signal.len());
    println!("  kernel length     = {}", ts.kernel.len());
    println!("  output length     = {}", ts.y.len());
    println!(
        "  direct max-abs-error = {:e}  (at i={})",
        direct_stats.max_abs, direct_stats.arg_max
    );
    println!("  direct RMSE          = {:e}", direct_stats.rmse);
    match &numpy_stats {
        Some(stats) => {
            println!(
                "  numpy max-abs-error  = {:e}  (at i={})",
                stats.max_abs, stats.arg_max
            );
            println!("  numpy RMSE           = {:e}", stats.rmse);
        }
        None => println!(
            "  SKIP  numpy.convolve comparison (reference JSON unavailable: {})",
            np_path.display()
        ),
    }

    let peak = ts.y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let ulp_at_peak = peak.max(1.0) * 2f64.powi(-52);
    let tolerance = (1e-12_f64).max(1024.0 * ulp_at_peak);

    println!("  peak |y|          = {:e}", peak);
    println!("  1024 * ULP(peak)  = {:e}", 1024.0 * ulp_at_peak);
    println!("  tolerance         = {:e}", tolerance);

    let ok = direct_stats.max_abs < tolerance
        && numpy_stats
            .as_ref()
            .map(|stats| stats.max_abs < tolerance)
            .unwrap_or(true);
    println!();
    println!("{}", if ok { "  PASS" } else { "  FAIL" });
    if ok {
        0
    } else {
        1
    }
}
