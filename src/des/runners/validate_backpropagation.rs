//! Port of `src/des/runners/validate-backpropagation.ts`.
//!
//! Compares the framework's backprop output (`out/backprop-framework.json`)
//! against the numpy-style reference (`out/external/backpropagation/numpy.json`):
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
    v.as_array().map(|a| a.iter().map(as_vec).collect()).unwrap_or_default()
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

/// `main()` — returns the exit code (0 = PASS).
pub fn run() -> i32 {
    let root = root();
    let ts_path = root.join("out").join("backprop-framework.json");
    let py_path = root.join("out").join("external").join("backpropagation").join("numpy.json");

    let ts = match load_json(&ts_path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let py = match load_json(&py_path) {
        Ok(v) => v,
        Err(code) => return code,
    };

    println!("Backpropagation: framework vs numpy-naive python");
    println!("===================================================");
    println!(
        "  config = seed={}, N={}, lr={}",
        fmt_num(at(&ts, &["config", "seed"])),
        fmt_num(at(&ts, &["config", "N"])),
        fmt_num(at(&ts, &["config", "lr"])),
    );

    let loss_ts = as_vec(ts.get("lossHistory").unwrap_or(&JsonValue::Null));
    let loss_py = as_vec(py.get("lossHistory").unwrap_or(&JsonValue::Null));
    let pred_ts = as_vec(ts.get("predictions").unwrap_or(&JsonValue::Null));
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
        &as_mat(at(&ts, &["final", "W1"]).unwrap_or(&JsonValue::Null)),
        &as_mat(at(&py, &["final", "W1"]).unwrap_or(&JsonValue::Null)),
    );
    let b1_diff = match max_abs_diff(
        &as_vec(at(&ts, &["final", "b1"]).unwrap_or(&JsonValue::Null)),
        &as_vec(at(&py, &["final", "b1"]).unwrap_or(&JsonValue::Null)),
    ) {
        Ok(d) => d,
        Err(c) => return c,
    };
    let w2_diff = max_abs_diff_2d(
        &as_mat(at(&ts, &["final", "W2"]).unwrap_or(&JsonValue::Null)),
        &as_mat(at(&py, &["final", "W2"]).unwrap_or(&JsonValue::Null)),
    );
    let b2_diff = match max_abs_diff(
        &as_vec(at(&ts, &["final", "b2"]).unwrap_or(&JsonValue::Null)),
        &as_vec(at(&py, &["final", "b2"]).unwrap_or(&JsonValue::Null)),
    ) {
        Ok(d) => d,
        Err(c) => return c,
    };

    println!("  W1   max-abs-error  = {:e}  (at [{}][{}])", w1_diff.max, w1_diff.row, w1_diff.col);
    println!("  b1   max-abs-error  = {:e}  (at [{}])", b1_diff.max, b1_diff.idx);
    println!("  W2   max-abs-error  = {:e}  (at [{}][{}])", w2_diff.max, w2_diff.row, w2_diff.col);
    println!("  b2   max-abs-error  = {:e}  (at [{}])", b2_diff.max, b2_diff.idx);
    println!("  loss max-abs-error  = {:e}  (at sample {})", loss_diff.max, loss_diff.idx);
    println!("  pred max-abs-error  = {:e}  (at case {})", pred_diff.max, pred_diff.idx);

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
