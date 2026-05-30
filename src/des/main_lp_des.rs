//! Port of `src/des/main-lp-des.ts`.
//!
//! Solve an LP by running simplex as a DES (one pivot = one tick) via
//! Entering/Leaving/Pivot/Observer stations, and cross-check against the
//! in-process simplex and scipy:HiGHS.
//!
//! Delegates to `crate::des::general::{lp, lp_des}`. The 2-D polytope-walk
//! animation is omitted.
//!
//! PORT NOTE: `ANIMATE=1` rendering (FrameRecorder polytope / objective scenes)
//! is omitted — no animation engine is ported. The console trace (the substance
//! of the script) is reproduced faithfully.

#![allow(dead_code)]

use crate::des::general::lp::{
    lp_to_string, solve_lp_external, solve_lp_internal, ExternalSolverOptions,
    InternalSimplexOptions, LPProblem, Sense,
};
use crate::des::general::lp_des::{solve_lp_via_des, DESSimplexOptions, PivotRule};

/// The named library LP, or `None` for an unknown key.
fn problem(name: &str) -> Option<LPProblem> {
    let names = |xs: &[&str]| Some(xs.iter().map(|s| s.to_string()).collect());
    Some(match name {
        "2var" => LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 1.0], vec![1.0, 3.0]]),
            b_ub: Some(vec![4.0, 6.0]),
            a_eq: None,
            b_eq: None,
            lb: None,
            ub: None,
            var_names: names(&["x", "y"]),
            con_names: None,
        },
        "2var-diamond" => LPProblem {
            sense: Sense::Max,
            c: vec![2.0, 3.0],
            a_ub: Some(vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![1.0, 2.0],
            ]),
            b_ub: Some(vec![4.0, 5.0, 6.0, 9.0]),
            a_eq: None,
            b_eq: None,
            lb: None,
            ub: None,
            var_names: names(&["x", "y"]),
            con_names: None,
        },
        "diet" => LPProblem {
            sense: Sense::Min,
            c: vec![0.5, 0.3, 0.7, 0.2],
            a_ub: Some(vec![
                vec![-2.0, -3.0, -1.0, -4.0],
                vec![-1.0, -2.0, -3.0, -1.0],
                vec![-3.0, -1.0, -2.0, 0.0],
            ]),
            b_ub: Some(vec![-12.0, -6.0, -4.0]),
            a_eq: None,
            b_eq: None,
            lb: None,
            ub: None,
            var_names: names(&["bread", "cheese", "meat", "rice"]),
            con_names: names(&["protein", "vit-A", "vit-C"]),
        },
        "transport" => LPProblem {
            sense: Sense::Min,
            c: vec![4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0, 2.0, 1.0],
            a_ub: None,
            b_ub: None,
            a_eq: Some(vec![
                vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]),
            b_eq: Some(vec![20.0, 30.0, 25.0, 25.0, 25.0, 25.0]),
            lb: None,
            ub: None,
            var_names: names(&[
                "x11", "x12", "x13", "x21", "x22", "x23", "x31", "x32", "x33",
            ]),
            con_names: None,
        },
        _ => return None,
    })
}

const KNOWN: &[&str] = &["2var", "2var-diamond", "diet", "transport"];

fn opt_obj(o: f64) -> String {
    if o.is_nan() {
        "-".to_string()
    } else {
        format!("{o:.8}")
    }
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let which = std::env::var("PROBLEM").unwrap_or_else(|_| "2var".into());
    let which = which.trim().to_string();
    let lp = match problem(&which) {
        Some(lp) => lp,
        None => {
            eprintln!(
                "unknown PROBLEM='{which}'; expected one of: {}",
                KNOWN.join(", ")
            );
            return;
        }
    };
    let pivot_rule = match std::env::var("PIVOT_RULE")
        .unwrap_or_else(|_| "dantzig".into())
        .as_str()
    {
        "bland" => PivotRule::Bland,
        _ => PivotRule::Dantzig,
    };

    println!(
        "# DES-driven simplex on '{which}' problem  (pivotRule = {})",
        pivot_rule.as_str()
    );
    println!("#");
    println!("# LP:");
    for line in lp_to_string(&lp).split('\n') {
        println!("#   {line}");
    }
    println!();

    let des = solve_lp_via_des(
        &lp,
        &DESSimplexOptions {
            pivot_rule: Some(pivot_rule),
            max_iter: Some(1000),
            tol: None,
        },
    );
    let internal = solve_lp_internal(&lp, &InternalSimplexOptions::default());
    let external = solve_lp_external(
        &lp,
        &ExternalSolverOptions {
            method: Some("highs".into()),
            ..Default::default()
        },
    );

    println!("# DES simplex (this engine):");
    println!("#   status     = {}", des.status.as_str());
    println!("#   pivots     = {}", des.trace.pivot_history.len());
    println!(
        "#   x*         = [ {} ]",
        des.x
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("#   objective  = {:.8}", des.objective);
    println!("#   wall time  = {}ms", des.elapsed_ms);
    println!();

    println!("# In-process simplex (textbook two-phase, NOT through DES):");
    println!(
        "#   status     = {}    obj = {}    iters = {}    Δ = {:.3e}",
        internal.status.as_str(),
        opt_obj(internal.objective),
        internal.iters.map(|v| v.to_string()).unwrap_or_default(),
        (des.objective - internal.objective).abs()
    );

    println!("# scipy:HiGHS (external simplex):");
    println!(
        "#   status     = {}    obj = {}    iters = {}    Δ = {:.3e}",
        external.status.as_str(),
        opt_obj(external.objective),
        external.iters.map(|v| v.to_string()).unwrap_or_default(),
        (des.objective - external.objective).abs()
    );
    if let Some(dual) = &external.dual_ub {
        if !dual.is_empty() {
            println!(
                "#   shadow prices on each ≤ constraint = [ {} ]",
                dual.iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    println!();

    println!("# Pivot trajectory (each row = one tick / one vertex visit):");
    println!("#   tick  phase   enter   leave    pivot         vertex (x*)");
    println!(
        "#   {:<6} {:<7} {:<7} {:<7} {:<13} [ {} ]    obj = {:.4}",
        "init",
        "",
        "",
        "",
        "",
        des.trace.vertex_history[0]
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
        des.trace.obj_history[0]
    );
    for (i, p) in des.trace.pivot_history.iter().enumerate() {
        let v = &des.trace.vertex_history[i + 1];
        let obj_str = if p.obj.is_nan() {
            "(phase-1)".to_string()
        } else {
            format!("{:.4}", p.obj)
        };
        println!(
            "#   {:<6} {:<7} {:<7} {:<7} {:<13} [ {} ]    obj = {}",
            p.tick,
            p.phase,
            format!("col={}", p.enter),
            format!("row={}", p.leave),
            format!("{:.3e}", p.pivot_elt),
            v.iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            obj_str
        );
    }
    println!();

    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        println!("# (animation omitted in Rust port — see PORT NOTE)");
    }
}
