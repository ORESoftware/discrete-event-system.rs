//! Port of `src/des/runners/validate-backpropagation.ts`.
//!
//! Compares the framework's backprop output (`out/backprop-framework.json`)
//! against a deterministic Rust recomputation from the same config. If the
//! optional NumPy sidecar (`out/external/backpropagation/numpy.json`) is present,
//! it can be used as the reference artifact instead. The validator compares
//! per-tensor max-abs error on `W1/b1/W2/b2`, the loss history, and the four XOR
//! predictions, asserting agreement within `1e-12`. The TS `main()` becomes
//! [`run`], returning the process exit code.
//!
//! ## PORT NOTE
//!   * `__dirname/../../..` repo root → `REPO_ROOT` env var or the cwd.
//!   * `fs`/`JSON.parse` → `std::fs` + [`parse_json`].
//!   * weight tensors `as any` → `Vec<Vec<f64>>` extracted from [`JsonValue`].

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::main_backpropagation::{init_weights, run_backprop};
use crate::des::observability::logger::{parse_json, JsonValue};

fn root() -> PathBuf {
    match std::env::var("REPO_ROOT") {
        Ok(r) => PathBuf::from(r),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

fn load_json(p: &std::path::Path) -> Result<JsonValue, i32> {
    if !p.exists() {
        eprintln!("[validate-backpropagation] missing {}", p.display());
        return Err(1);
    }
    let text = std::fs::read_to_string(p).map_err(|_| 1)?;
    parse_json(&text).map_err(|e| {
        eprintln!("[validate-backpropagation] parse error: {e}");
        1
    })
}

fn as_vec(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default()
}

fn as_mat(v: &JsonValue) -> Vec<Vec<f64>> {
    v.as_array()
        .map(|a| a.iter().map(as_vec).collect())
        .unwrap_or_default()
}

/// Path lookup `v[a][b]...` returning a borrowed value.
fn at<'a>(v: &'a JsonValue, keys: &[&str]) -> Option<&'a JsonValue> {
    let mut cur = v;
    for k in keys {
        cur = cur.get(k)?;
    }
    Some(cur)
}

struct Diff1 {
    max: f64,
    idx: i64,
}

struct Diff2 {
    max: f64,
    row: i64,
    col: i64,
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> Result<Diff1, i32> {
    if a.len() != b.len() {
        eprintln!("length mismatch: {} vs {}", a.len(), b.len());
        return Err(1);
    }
    let mut max = 0.0_f64;
    let mut idx: i64 = -1;
    for i in 0..a.len() {
        let e = (a[i] - b[i]).abs();
        if e > max {
            max = e;
            idx = i as i64;
        }
    }
    Ok(Diff1 { max, idx })
}

fn max_abs_diff_2d(a: &[Vec<f64>], b: &[Vec<f64>]) -> Diff2 {
    let mut max = 0.0_f64;
    let mut row: i64 = -1;
    let mut col: i64 = -1;
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            let bij = b.get(i).and_then(|r| r.get(j)).copied().unwrap_or(f64::NAN);
            let e = (a[i][j] - bij).abs();
            if e > max {
                max = e;
                row = i as i64;
                col = j as i64;
            }
        }
    }
    Diff2 { max, row, col }
}

fn fmt_num(v: Option<&JsonValue>) -> String {
    match v.and_then(|x| x.as_f64()) {
        Some(n) if n.fract() == 0.0 && n.abs() < 1e15 => format!("{}", n as i64),
        Some(n) => format!("{n}"),
        None => "undefined".to_string(),
    }
}

fn config_number(v: &JsonValue, key: &str, default: f64) -> f64 {
    at(v, &["config", key])
        .and_then(|x| x.as_f64())
        .unwrap_or(default)
}

fn rust_reference_from_framework_config(framework: &JsonValue) -> Result<JsonValue, i32> {
    let seed = config_number(framework, "seed", 7.0) as u32;
    let n = config_number(framework, "N", 10000.0) as usize;
    let lr = config_number(framework, "lr", 0.5);
    let init = init_weights(seed, 3);
    let result = run_backprop(&init, n, lr);
    let out = serde_json::json!({
        "config": {"seed": seed, "N": n, "lr": lr},
        "init": {"W1": init.w1, "b1": init.b1, "W2": init.w2, "b2": init.b2},
        "final": {"W1": result.w1, "b1": result.b1, "W2": result.w2, "b2": result.b2},
        "predictions": result.predictions,
        "lossHistory": result.loss_history,
        "ticks": result.ticks,
    });
    parse_json(&out.to_string()).map_err(|e| {
        eprintln!("[validate-backpropagation] Rust reference serialization error: {e}");
        1
    })
}

/// `main()` — returns the exit code (0 = PASS).
pub fn run() -> i32 {
    let root = root();
    let ts_path = root.join("out").join("backprop-framework.json");
    let reference_path = root
        .join("out")
        .join("external")
        .join("backpropagation")
        .join("numpy.json");

    let ts = match load_json(&ts_path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let (reference, reference_source) = if reference_path.exists() {
        match load_json(&reference_path) {
            Ok(v) => (v, reference_path.display().to_string()),
            Err(code) => return code,
        }
    } else {
        match rust_reference_from_framework_config(&ts) {
            Ok(v) => (v, "Rust recomputation from framework config".to_string()),
            Err(code) => return code,
        }
    };

    println!("Backpropagation: framework vs Rust/optional NumPy reference artifact");
    println!("===================================================");
    println!(
        "  config = seed={}, N={}, lr={}",
        fmt_num(at(&ts, &["config", "seed"])),
        fmt_num(at(&ts, &["config", "N"])),
        fmt_num(at(&ts, &["config", "lr"])),
    );
    println!("  reference = {reference_source}");

    let loss_ts = as_vec(ts.get("lossHistory").unwrap_or(&JsonValue::Null));
    let loss_reference = as_vec(reference.get("lossHistory").unwrap_or(&JsonValue::Null));
    let pred_ts = as_vec(ts.get("predictions").unwrap_or(&JsonValue::Null));
    let pred_reference = as_vec(reference.get("predictions").unwrap_or(&JsonValue::Null));

    let loss_diff = match max_abs_diff(&loss_ts, &loss_reference) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let pred_diff = match max_abs_diff(&pred_ts, &pred_reference) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let w1_diff = max_abs_diff_2d(
        &as_mat(at(&ts, &["final", "W1"]).unwrap_or(&JsonValue::Null)),
        &as_mat(at(&reference, &["final", "W1"]).unwrap_or(&JsonValue::Null)),
    );
    let b1_diff = match max_abs_diff(
        &as_vec(at(&ts, &["final", "b1"]).unwrap_or(&JsonValue::Null)),
        &as_vec(at(&reference, &["final", "b1"]).unwrap_or(&JsonValue::Null)),
    ) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let w2_diff = max_abs_diff_2d(
        &as_mat(at(&ts, &["final", "W2"]).unwrap_or(&JsonValue::Null)),
        &as_mat(at(&reference, &["final", "W2"]).unwrap_or(&JsonValue::Null)),
    );
    let b2_diff = match max_abs_diff(
        &as_vec(at(&ts, &["final", "b2"]).unwrap_or(&JsonValue::Null)),
        &as_vec(at(&reference, &["final", "b2"]).unwrap_or(&JsonValue::Null)),
    ) {
        Ok(d) => d,
        Err(c) => return c,
    };

    println!(
        "  W1   max-abs-error  = {:e}  (at [{}][{}])",
        w1_diff.max, w1_diff.row, w1_diff.col
    );
    println!(
        "  b1   max-abs-error  = {:e}  (at [{}])",
        b1_diff.max, b1_diff.idx
    );
    println!(
        "  W2   max-abs-error  = {:e}  (at [{}][{}])",
        w2_diff.max, w2_diff.row, w2_diff.col
    );
    println!(
        "  b2   max-abs-error  = {:e}  (at [{}])",
        b2_diff.max, b2_diff.idx
    );
    println!(
        "  loss max-abs-error  = {:e}  (at sample {})",
        loss_diff.max, loss_diff.idx
    );
    println!(
        "  pred max-abs-error  = {:e}  (at case {})",
        pred_diff.max, pred_diff.idx
    );

    let tol = 1e-12_f64;
    let all_diffs = [
        w1_diff.max,
        b1_diff.max,
        w2_diff.max,
        b2_diff.max,
        loss_diff.max,
        pred_diff.max,
    ];
    let worst = all_diffs.iter().copied().fold(0.0_f64, f64::max);
    println!();
    println!("  worst diff = {:e}    tolerance = {:e}", worst, tol);
    let ok = worst < tol;
    println!("{}", if ok { "  PASS" } else { "  FAIL" });

    // Convergence sanity: avg loss over last 100 samples.
    let last100: Vec<f64> = if loss_ts.len() > 100 {
        loss_ts[loss_ts.len() - 100..].to_vec()
    } else {
        loss_ts.clone()
    };
    let avg_loss = if last100.is_empty() {
        0.0
    } else {
        last100.iter().sum::<f64>() / last100.len() as f64
    };
    println!("  avg loss (last 100) = {:e}", avg_loss);
    if avg_loss > 0.05 {
        println!("  WARN: avg loss > 0.05, network may not have converged");
    }

    if ok {
        0
    } else {
        1
    }
}
