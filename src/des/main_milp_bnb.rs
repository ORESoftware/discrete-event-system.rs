//! Port of `src/des/main-milp-bnb.ts`.
//!
//! CLI driver for MILP via Branch-and-Bound: a textbook knapsack with full
//! B&B trace, a larger knapsack vs brute force, a generic MILP, a pure-LP
//! fallback, and a performance-scaling study.
//!
//! Conversion notes:
//!   - bitmask brute enumeration `1 << n` → `usize` bitsets.
//!   - delegates to `general::milp_bnb`; top-level `main()` → [`run`].

use std::time::Instant;

use crate::des::general::incremental_lp::Sense;
use crate::des::general::milp_bnb::{
    build_knapsack_milp, solve_milp, MILPProblem, MILPSolution, MILPSolveOptions, MILPStatus,
};

fn header(s: &str) {
    println!();
    println!("{}", "═".repeat(96));
    println!("  {}", s);
    println!("{}", "═".repeat(96));
}

/// TS string spelling of an [`MILPStatus`].
fn status_str(s: MILPStatus) -> &'static str {
    match s {
        MILPStatus::Optimal => "optimal",
        MILPStatus::Infeasible => "infeasible",
        MILPStatus::Unbounded => "unbounded",
        MILPStatus::MaxNodes => "max-nodes",
    }
}

fn brute_knapsack(values: &[f64], weights: &[f64], capacity: f64) -> (f64, Vec<i32>) {
    let n = values.len();
    let mut best_z = 0.0;
    let mut best_x = vec![0_i32; n];
    for mask in 0..(1usize << n) {
        let mut v = 0.0;
        let mut w = 0.0;
        for i in 0..n {
            if mask & (1 << i) != 0 {
                v += values[i];
                w += weights[i];
            }
        }
        if w <= capacity && v > best_z {
            best_z = v;
            best_x = (0..n).map(|i| if mask & (1 << i) != 0 { 1 } else { 0 }).collect();
        }
    }
    (best_z, best_x)
}

fn print_solution(label: &str, r: &MILPSolution) {
    let x_pretty: Vec<String> = r
        .x
        .iter()
        .take(16)
        .map(|&v| if v.is_finite() { format!("{:.3}", v) } else { "N/A".to_string() })
        .collect();
    let z_str = if r.z.is_finite() {
        format!("{:.4}", r.z)
    } else if r.z.is_infinite() {
        if r.z > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        "NaN".to_string()
    };
    println!("  {}", label);
    println!("    status:   {}", status_str(r.status));
    println!("    z*:       {}", z_str);
    println!("    bestBound:{:.4}    gap: {:.2e}", r.best_bound, r.gap);
    println!(
        "    x* (first 16):  [{}{}]",
        x_pretty.join(", "),
        if r.x.len() > 16 { ", …" } else { "" }
    );
    println!("    nodes:    {}    LP pivots: {}", r.nodes_explored, r.total_pivots);
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    header("STUDY 1 — Textbook 0/1 knapsack (4 items)");
    println!("  values  v = [10, 40, 30, 50]");
    println!("  weights w = [ 5,  4,  6,  3]");
    println!("  capacity W = 10");
    {
        let milp = build_knapsack_milp(vec![10.0, 40.0, 30.0, 50.0], vec![5.0, 4.0, 6.0, 3.0], 10.0);
        let t0 = Instant::now();
        let r = solve_milp(&milp, MILPSolveOptions { verbose: Some(true), ..Default::default() });
        let dt = t0.elapsed().as_millis();
        println!();
        print_solution(&format!("B&B solution (wall={}ms):", dt), &r);
    }

    header("STUDY 2 — 12-item knapsack vs brute force (4096 enumerations)");
    {
        let v = vec![12.0, 18.0, 9.0, 4.0, 21.0, 35.0, 14.0, 25.0, 30.0, 8.0, 17.0, 6.0];
        let w = vec![5.0, 8.0, 4.0, 3.0, 9.0, 13.0, 6.0, 7.0, 11.0, 3.0, 6.0, 4.0];
        let cap = 30.0;
        let milp = build_knapsack_milp(v.clone(), w.clone(), cap);
        let t0 = Instant::now();
        let r = solve_milp(&milp, MILPSolveOptions::default());
        let t1 = t0.elapsed().as_millis();
        let t0b = Instant::now();
        let brute = brute_knapsack(&v, &w, cap);
        let t2 = t0b.elapsed().as_millis();
        print_solution("B&B", &r);
        let bx: Vec<String> = brute.1.iter().map(|x| x.to_string()).collect();
        println!("    brute force z={}, x=[{}]", num_str(brute.0), bx.join(", "));
        let is_match = (r.z - brute.0).abs() < 1e-6;
        println!(
            "    match: {}    (B&B={}ms, brute={}ms)",
            if is_match { "YES" } else { "NO" },
            t1,
            t2
        );
    }

    header("STUDY 3 — Generic MILP (2 integer + 1 continuous var, min sense)");
    println!("  min  3 x_0 + 5 x_1 + 7 x_2");
    println!("  s.t. x_0 + x_1 + x_2 ≤ 10");
    println!("       2 x_0 + x_1     ≤ 8");
    println!("       x_0 + 2 x_1 + 3 x_2 ≤ 15");
    println!("  x_0, x_1 ∈ ℤ_≥0,  x_2 ∈ ℝ_≥0");
    {
        let milp = MILPProblem {
            sense: Sense::Max,
            c: vec![3.0, 5.0, 7.0],
            a: vec![vec![1.0, 1.0, 1.0], vec![2.0, 1.0, 0.0], vec![1.0, 2.0, 3.0]],
            b: vec![10.0, 8.0, 15.0],
            integer_vars: vec![true, true, false],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(&milp, MILPSolveOptions { verbose: Some(false), ..Default::default() });
        print_solution("B&B", &r);
    }

    header("STUDY 4 — Pure LP (all variables continuous) — should run only the root");
    {
        let lp = MILPProblem {
            sense: Sense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]],
            b: vec![4.0, 12.0, 18.0],
            integer_vars: vec![false, false],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(&lp, MILPSolveOptions::default());
        print_solution("B&B (no integrality)", &r);
        println!("    expected: z = 36, x = (2, 6) — classic textbook 2-D LP.");
    }

    header("STUDY 5 — Knapsack scaling: B&B nodes vs n");
    println!("  n         nodes  pivots   z*       wall(ms)");
    for n in [6usize, 10, 14, 18, 22, 26] {
        let mut v: Vec<f64> = Vec::new();
        let mut w: Vec<f64> = Vec::new();
        let mut s: u32 = 1;
        let mut rng = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s as f64 / 4_294_967_296.0
        };
        for _ in 0..n {
            v.push((rng() * 40.0 + 1.0).floor());
            w.push((rng() * 25.0 + 1.0).floor());
        }
        let cap = (w.iter().sum::<f64>() * 0.4).floor();
        let milp = build_knapsack_milp(v, w, cap);
        let t0 = Instant::now();
        let r = solve_milp(&milp, MILPSolveOptions { max_nodes: Some(50000), ..Default::default() });
        let dt = t0.elapsed().as_millis();
        println!(
            "  {:>2}     {:>7}  {:>7}  {:>7}    {:>5}",
            n,
            r.nodes_explored,
            r.total_pivots,
            format!("{:.2}", r.z),
            dt
        );
    }
}

/// JS `String(x)` for a number: integer-valued floats print bare.
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}
