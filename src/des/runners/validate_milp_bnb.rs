//! Port of `src/des/runners/validate-milp-bnb.ts`.
//!
//! Verifies the branch-and-bound MILP solver against brute-force enumeration on
//! small instances and against the LP solver when integrality is dropped.
//! Top-level driver → [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust branch-and-bound MILP solver and simplex LP solver.
//!   * `bruteKnapsack`/`feasible`/`close` are file-local helpers, ported faithfully.

#![allow(dead_code)]

use std::time::Instant;

use crate::des::general::lp::{
    solve_lp_internal as solve_lp_internal_model, InternalSimplexOptions, LPProblem,
    Sense as LPSense,
};
use crate::des::general::milp_bnb::{
    build_knapsack_milp as build_knapsack_milp_model, solve_milp as solve_milp_model, MILPProblem,
    MILPSolution, MILPSolveOptions, MILPStatus, Sense as MILPSense,
};

type MilpProblem = MILPProblem;
type MilpResult = MILPSolution;
type LpProblem = LPProblem;

fn build_knapsack_milp(values: &[f64], weights: &[f64], capacity: f64) -> MilpProblem {
    build_knapsack_milp_model(values.to_vec(), weights.to_vec(), capacity)
}

fn solve_milp(milp: &MilpProblem, max_nodes: Option<usize>) -> MilpResult {
    solve_milp_model(
        milp,
        MILPSolveOptions {
            max_nodes,
            ..Default::default()
        },
    )
}

fn solve_lp_internal(lp: &LpProblem) -> crate::des::general::lp::LPSolution {
    solve_lp_internal_model(lp, &InternalSimplexOptions::default())
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!(
            "{}  {}{}",
            if ok { "  PASS" } else { "  FAIL" },
            label,
            tail
        );
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * f64::max(1.0, f64::max(a.abs(), b.abs()))
}

fn brute_knapsack(values: &[f64], weights: &[f64], capacity: f64) -> (f64, Vec<f64>) {
    let n = values.len();
    let mut best_z = 0.0;
    let mut best_x = vec![0.0; n];
    for mask in 0u32..(1u32 << n) {
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
            best_x = (0..n)
                .map(|i| if mask & (1 << i) != 0 { 1.0 } else { 0.0 })
                .collect();
        }
    }
    (best_z, best_x)
}

fn feasible(milp: &MilpProblem, x: &[f64]) -> bool {
    let tol = 1e-6;
    for i in 0..milp.a.len() {
        let mut s = 0.0;
        for j in 0..x.len() {
            s += milp.a[i][j] * x[j];
        }
        if s > milp.b[i] + tol {
            return false;
        }
    }
    if let Some(ub) = &milp.ub {
        for j in 0..x.len() {
            if ub[j].is_finite() && x[j] > ub[j] + tol {
                return false;
            }
        }
    }
    for j in 0..x.len() {
        if x[j] < -tol {
            return false;
        }
    }
    for j in 0..x.len() {
        if milp.integer_vars[j] && (x[j] - x[j].round()).abs() > 1e-4 {
            return false;
        }
    }
    true
}

/// `validate-milp-bnb.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    // Study 1 — Textbook 4-item knapsack.
    println!("\nStudy 1 — Textbook 4-item knapsack");
    {
        let milp = build_knapsack_milp(&[10.0, 40.0, 30.0, 50.0], &[5.0, 4.0, 6.0, 3.0], 10.0);
        let r = solve_milp(&milp, None);
        let (brute_z, _brute_x) =
            brute_knapsack(&[10.0, 40.0, 30.0, 50.0], &[5.0, 4.0, 6.0, 3.0], 10.0);
        c.check("1.1 status optimal", r.status == MILPStatus::Optimal, "");
        c.check(
            "1.2 z matches brute force",
            close(r.z, brute_z),
            &format!("B&B={}, brute={}", r.z, brute_z),
        );
        c.check("1.3 solution feasible", feasible(&milp, &r.x), "");
        c.check(
            "1.4 gap = 0 at optimal",
            r.gap < 1e-9,
            &format!("gap={}", r.gap),
        );
        c.check(
            "1.5 explores ≤ 16 nodes",
            r.nodes_explored <= 16,
            &format!("nodes={}", r.nodes_explored),
        );
    }

    // Study 2 — Random knapsacks vs brute force (n=8 to 14).
    println!("\nStudy 2 — Random knapsacks vs brute force (n=8 to 14)");
    {
        let mut s: u32 = 17;
        let mut rng = move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s as f64 / 4294967296.0
        };
        let mut all_match = true;
        let mut total_nodes = 0usize;
        let mut total_brute_perm = 0usize;
        for n in [8usize, 10, 12, 14] {
            for _trial in 0..5 {
                let values: Vec<f64> = (0..n).map(|_| (rng() * 50.0 + 1.0).floor()).collect();
                let weights: Vec<f64> = (0..n).map(|_| (rng() * 30.0 + 1.0).floor()).collect();
                let cap = (weights.iter().sum::<f64>() * 0.4).floor();
                let milp = build_knapsack_milp(&values, &weights, cap);
                let r = solve_milp(&milp, None);
                let (brute_z, _) = brute_knapsack(&values, &weights, cap);
                if !close(r.z, brute_z) {
                    all_match = false;
                    println!("    MISMATCH n={} : B&B={}, brute={}", n, r.z, brute_z);
                }
                total_nodes += r.nodes_explored;
                total_brute_perm += 1usize << n;
            }
        }
        c.check(
            "2.1 all 20 random knapsack instances match brute force",
            all_match,
            "",
        );
        c.check(
            "2.2 total B&B nodes far less than total enumerations",
            total_nodes < total_brute_perm / 5,
            &format!("nodes={}, enum={}", total_nodes, total_brute_perm),
        );
    }

    // Study 3 — Pure LP (no integer vars) reduces to root LP.
    println!("\nStudy 3 — Pure LP (no integer vars) reduces to root LP");
    {
        let lp = MilpProblem {
            sense: MILPSense::Max,
            c: vec![3.0, 5.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]],
            b: vec![4.0, 12.0, 18.0],
            integer_vars: vec![false, false],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let milp_r = solve_milp(&lp, None);
        let lp_r = solve_lp_internal(&LpProblem {
            sense: LPSense::Max,
            c: vec![3.0, 5.0],
            a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]]),
            b_ub: Some(vec![4.0, 12.0, 18.0]),
            a_eq: None,
            b_eq: None,
            lb: None,
            ub: None,
            var_names: None,
            con_names: None,
        });
        c.check(
            "3.1 MILP-no-integers status optimal",
            milp_r.status == MILPStatus::Optimal,
            "",
        );
        c.check(
            "3.2 z agrees with solveLPInternal",
            close(milp_r.z, lp_r.objective),
            &format!("MILP={}, LP={}", milp_r.z, lp_r.objective),
        );
        c.check(
            "3.3 only the root node was explored",
            milp_r.nodes_explored == 1,
            &format!("nodes={}", milp_r.nodes_explored),
        );
    }

    // Study 4 — Mixed integer/continuous (3 vars).
    println!("\nStudy 4 — Mixed integer/continuous (3 vars)");
    {
        let milp = MilpProblem {
            sense: MILPSense::Max,
            c: vec![3.0, 5.0, 7.0],
            a: vec![
                vec![1.0, 1.0, 1.0],
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 3.0],
            ],
            b: vec![10.0, 8.0, 15.0],
            integer_vars: vec![true, true, false],
            ub: None,
            var_names: None,
            con_names: None,
        };
        let r = solve_milp(&milp, None);
        c.check(
            "4.1 mixed MILP optimal",
            r.status == MILPStatus::Optimal,
            "",
        );
        c.check(
            "4.2 x_0, x_1 are integer",
            (r.x[0] - r.x[0].round()).abs() < 1e-4 && (r.x[1] - r.x[1].round()).abs() < 1e-4,
            &format!("x_0={}, x_1={}", r.x[0], r.x[1]),
        );
        c.check("4.3 solution feasible", feasible(&milp, &r.x), "");
        let lp = solve_lp_internal(&LpProblem {
            sense: LPSense::Max,
            c: vec![3.0, 5.0, 7.0],
            a_ub: Some(vec![
                vec![1.0, 1.0, 1.0],
                vec![2.0, 1.0, 0.0],
                vec![1.0, 2.0, 3.0],
            ]),
            b_ub: Some(vec![10.0, 8.0, 15.0]),
            a_eq: None,
            b_eq: None,
            lb: None,
            ub: None,
            var_names: None,
            con_names: None,
        });
        c.check(
            "4.4 MILP z ≤ LP relaxation z (max)",
            r.z <= lp.objective + 1e-6,
            &format!("MILP={}, LP={}", r.z, lp.objective),
        );
    }

    // Study 5 — Infeasibility / zero-capacity knapsack.
    println!("\nStudy 5 — Infeasibility detection");
    {
        let milp = build_knapsack_milp(&[1.0, 1.0, 1.0], &[2.0, 3.0, 5.0], 0.0);
        let r = solve_milp(&milp, None);
        c.check(
            "5.1 zero-capacity knapsack: optimal",
            r.status == MILPStatus::Optimal,
            "",
        );
        c.check("5.2 z = 0 (no items selected)", close(r.z, 0.0), "");
        c.check("5.3 x = 0 vector", r.x.iter().all(|v| v.abs() < 1e-9), "");
    }

    // Study 6 — Scaling: B&B much faster than 2^n on knapsack.
    println!("\nStudy 6 — Scaling: B&B much faster than 2^n on knapsack");
    {
        let mut s: u32 = 1234;
        let mut rng = move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s as f64 / 4294967296.0
        };
        let v: Vec<f64> = (0..24).map(|_| (rng() * 40.0 + 1.0).floor()).collect();
        let w: Vec<f64> = (0..24).map(|_| (rng() * 25.0 + 1.0).floor()).collect();
        let cap = (w.iter().sum::<f64>() * 0.4).floor();
        let milp = build_knapsack_milp(&v, &w, cap);
        let t0 = Instant::now();
        let r = solve_milp(&milp, Some(100_000));
        let dt = t0.elapsed().as_millis();
        c.check(
            "6.1 24-item knapsack solves to optimum",
            r.status == MILPStatus::Optimal,
            &format!("dt={}ms, nodes={}", dt, r.nodes_explored),
        );
        c.check(
            "6.2 nodes ≪ 2^24 = 16.7M",
            r.nodes_explored < 1000,
            &format!("nodes={}", r.nodes_explored),
        );
        c.check("6.3 wall < 1 second", dt < 1000, &format!("dt={}ms", dt));
    }

    // Study 7 — Bound monotonicity & fewer-node-on-warm-start sanity.
    println!("\nStudy 7 — Bound monotonicity & fewer-node-on-warm-start sanity");
    {
        let v = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let w = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r1 = solve_milp(&build_knapsack_milp(&v, &w, 30.0), None);
        let r2 = solve_milp(&build_knapsack_milp(&v, &w, 5.0), None);
        c.check(
            "7.1 tighter capacity ⇒ smaller (or equal) optimal",
            r2.z <= r1.z,
            "",
        );
        c.check(
            "7.2 both solutions feasible",
            feasible(&build_knapsack_milp(&v, &w, 30.0), &r1.x)
                && feasible(&build_knapsack_milp(&v, &w, 5.0), &r2.x),
            "",
        );
    }

    println!("\n  ─────────────────────────────────────────────────────────────────────────");
    println!("  {} passed, {} failed", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}
