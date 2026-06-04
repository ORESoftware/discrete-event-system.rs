//! Port of `src/des/runners/validate-backpropagation.ts`.
//!
//! Trains the framework's backprop DES in-process and compares it against the
//! numpy-style reference (`out/external/backpropagation/numpy.json`) when that
//! artifact is present: per-tensor max-abs error on `W1/b1/W2/b2`, the loss
//! history, and the four XOR predictions. The TS `main()` becomes [`run`],
//! returning the process exit code.
//!
//! ## PORT NOTE
//!   * `__dirname/../../..` repo root → `REPO_ROOT` env var or the cwd.
//!   * `fs`/`JSON.parse` → `std::fs` + [`parse_json`].
//!   * weight tensors `as any` → `Vec<Vec<f64>>` extracted from [`JsonValue`].

#![allow(dead_code)]

use std::path::PathBuf;

use crate::des::main_backpropagation::{init_weights, run_backprop, BackpropResult};
use crate::des::observability::logger::{parse_json, JsonValue};

fn root() -> PathBuf {
    match std::env::var("REPO_ROOT") {
        Ok(r) => PathBuf::from(r),
        Err(_) => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

fn load_optional_json(p: &std::path::Path) -> Result<Option<JsonValue>, i32> {
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(p).map_err(|_| 1)?;
    parse_json(&text).map(Some).map_err(|e| {
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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn finite_vec(xs: &[f64]) -> bool {
    xs.iter().all(|v| v.is_finite())
}

fn finite_mat(xs: &[Vec<f64>]) -> bool {
    xs.iter().all(|row| finite_vec(row))
}

fn framework_sane(result: &BackpropResult) -> bool {
    !result.loss_history.is_empty()
        && result.predictions.len() == 4
        && finite_mat(&result.w1)
        && finite_vec(&result.b1)
        && finite_mat(&result.w2)
        && finite_vec(&result.b2)
        && finite_vec(&result.loss_history)
        && finite_vec(&result.predictions)
}

fn avg_last_loss(loss_history: &[f64], n: usize) -> f64 {
    let last: Vec<f64> = if loss_history.len() > n {
        loss_history[loss_history.len() - n..].to_vec()
    } else {
        loss_history.to_vec()
    };
    if last.is_empty() {
        0.0
    } else {
        last.iter().sum::<f64>() / last.len() as f64
    }
}

/// `main()` — returns the exit code (0 = PASS).
pub fn run() -> i32 {
    let root = root();
    let py_path = root
        .join("out")
        .join("external")
        .join("backpropagation")
        .join("numpy.json");

    let seed = env_u32("SEED", 7);
    let n = env_usize("N", 10000);
    let lr = env_f64("LR", 0.5);
    let init = init_weights(seed, 3);
    let ts = run_backprop(&init, n, lr);
    let py = match load_optional_json(&py_path) {
        Ok(v) => v,
        Err(code) => return code,
    };

    println!("Backpropagation: framework vs numpy-naive python");
    println!("===================================================");
    println!("  config = seed={seed}, N={n}, lr={lr}");

    if py.is_none() {
        let ok = framework_sane(&ts);
        println!("  ticks = {}", ts.ticks);
        println!(
            "  predictions = [{:.4}, {:.4}, {:.4}, {:.4}]",
            ts.predictions[0], ts.predictions[1], ts.predictions[2], ts.predictions[3]
        );
        let avg_loss = avg_last_loss(&ts.loss_history, 100);
        println!("  avg loss (last 100) = {:e}", avg_loss);
        if avg_loss > 0.05 {
            println!("  WARN: avg loss > 0.05, network may not have converged");
        }
        println!(
            "  SKIP  numpy-naive comparison (reference JSON unavailable: {})",
            py_path.display()
        );
        println!("{}", if ok { "  PASS" } else { "  FAIL" });
        return if ok { 0 } else { 1 };
    }
    let py = py.expect("checked present");

    let loss_ts = &ts.loss_history;
    let loss_py = as_vec(py.get("lossHistory").unwrap_or(&JsonValue::Null));
    let pred_ts = &ts.predictions;
    let pred_py = as_vec(py.get("predictions").unwrap_or(&JsonValue::Null));

    let loss_diff = match max_abs_diff(&loss_ts, &loss_py) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let pred_diff = match max_abs_diff(&pred_ts, &pred_py) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let w1_diff = max_abs_diff_2d(
        &ts.w1,
        &as_mat(at(&py, &["final", "W1"]).unwrap_or(&JsonValue::Null)),
    );
    let b1_diff = match max_abs_diff(
        &ts.b1,
        &as_vec(at(&py, &["final", "b1"]).unwrap_or(&JsonValue::Null)),
    ) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let w2_diff = max_abs_diff_2d(
        &ts.w2,
        &as_mat(at(&py, &["final", "W2"]).unwrap_or(&JsonValue::Null)),
    );
    let b2_diff = match max_abs_diff(
        &ts.b2,
        &as_vec(at(&py, &["final", "b2"]).unwrap_or(&JsonValue::Null)),
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
    let avg_loss = avg_last_loss(loss_ts, 100);
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
