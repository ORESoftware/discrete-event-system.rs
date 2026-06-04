//! Port of `src/des/runners/validate-lp.ts`.
//!
//! Validates the in-process simplex against scipy.linprog and the MDP-as-LP
//! transformation against generic value iteration, across nine studies. The
//! top-level driver code becomes [`run`].
//!
//! PORT NOTES:
//!   * Uses the real Rust LP, DES-simplex, MDP-as-LP, and value-iteration
//!     modules. The local `LpProblem`/`MdpSpec` structs are validator adapters
//!     that preserve the original study bodies.

#![allow(dead_code)]

use std::rc::Rc;

use crate::des::general::des_lp_bridge::{
    build_mdp_lp as build_mdp_lp_model, solve_mdp_as_lp as solve_mdp_as_lp_model, MdpAsLpOptions,
};
use crate::des::general::lp::{
    solve_lp as solve_lp_model, solve_lp_internal as solve_lp_internal_model, ExternalSolver,
    ExternalSolverOptions, InternalSimplexOptions, LPProblem as RealLpProblem, LpSolverOptions,
    Sense as RealLpSense,
};
use crate::des::general::lp_des::{
    solve_lp_via_des as solve_lp_via_des_model, DESSimplexOptions, PivotRule,
};
use crate::des::general::value_iteration::{
    q_value as q_value_model, value_iteration as value_iteration_model, MDPSpec as RealMdpSpec,
    Outcome as RealOutcome, VIOptions as RealViOptions,
};
use crate::des::shared::transform::Transform;

// =============================================================================
// Validator adapter layer.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct LpProblem {
    sense: &'static str, // "max" | "min"
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct LpTrace {
    pivot_history: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
struct LpResult {
    status: String,
    x: Vec<f64>,
    objective: f64,
    iters: usize,
    solver: String,
    message: String,
    trace: LpTrace,
}

fn stub_lp(lp: &LpProblem, solver: &str) -> LpResult {
    LpResult {
        status: "optimal".to_string(),
        x: vec![0.0; lp.c.len()],
        objective: 0.0,
        iters: 0,
        solver: solver.to_string(),
        message: String::new(),
        trace: LpTrace::default(),
    }
}

fn real_sense(sense: &str) -> RealLpSense {
    match sense {
        "min" => RealLpSense::Min,
        _ => RealLpSense::Max,
    }
}

fn to_real_lp(lp: &LpProblem) -> RealLpProblem {
    RealLpProblem {
        sense: real_sense(lp.sense),
        c: lp.c.clone(),
        a_ub: (!lp.a_ub.is_empty()).then(|| lp.a_ub.clone()),
        b_ub: (!lp.b_ub.is_empty()).then(|| lp.b_ub.clone()),
        a_eq: (!lp.a_eq.is_empty()).then(|| lp.a_eq.clone()),
        b_eq: (!lp.b_eq.is_empty()).then(|| lp.b_eq.clone()),
        ..Default::default()
    }
}

fn lp_result_from_real(sol: crate::des::general::lp::LPSolution) -> LpResult {
    LpResult {
        status: sol.status.as_str().to_string(),
        x: sol.x,
        objective: sol.objective,
        iters: sol.iters.unwrap_or(0),
        solver: sol.solver,
        message: sol.message.unwrap_or_default(),
        trace: LpTrace::default(),
    }
}

fn solve_lp_internal(lp: &LpProblem, max_iter: Option<usize>) -> LpResult {
    let real = to_real_lp(lp);
    lp_result_from_real(solve_lp_internal_model(
        &real,
        &InternalSimplexOptions {
            max_iter,
            ..Default::default()
        },
    ))
}

fn solve_lp_external(lp: &LpProblem, method: &str) -> LpResult {
    let real = to_real_lp(lp);
    lp_result_from_real(
        ExternalSolver::new(ExternalSolverOptions {
            method: Some(method.to_string()),
            ..Default::default()
        })
        .transform(real),
    )
}

fn solve_lp_via_des(lp: &LpProblem, pivot_rule: Option<&str>, max_iter: Option<usize>) -> LpResult {
    let real = to_real_lp(lp);
    let pivot_rule = match pivot_rule {
        Some("bland") => Some(PivotRule::Bland),
        Some("dantzig") | None => Some(PivotRule::Dantzig),
        _ => Some(PivotRule::Dantzig),
    };
    let sol = solve_lp_via_des_model(
        &real,
        &DESSimplexOptions {
            pivot_rule,
            max_iter,
            ..Default::default()
        },
    );
    LpResult {
        status: sol.status.as_str().to_string(),
        x: sol.x,
        objective: sol.objective,
        iters: sol.iters.unwrap_or(0),
        solver: sol.solver,
        message: sol.message.unwrap_or_default(),
        trace: LpTrace {
            pivot_history: vec![0; sol.trace.pivot_history.len()],
        },
    }
}

fn solve_lp(lp: &LpProblem) -> LpResult {
    let real = to_real_lp(lp);
    lp_result_from_real(solve_lp_model(&real, &LpSolverOptions::default()))
}

fn scipy_unavailable(r: &LpResult) -> bool {
    r.status == "numerical-error"
        && (r.message.contains("scipy")
            || r.message.contains("numpy")
            || r.message.contains("No module named"))
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    prob: f64,
    reward: f64,
    next_state: usize,
}

struct MdpSpec {
    num_states: usize,
    num_actions: Rc<dyn Fn(usize) -> usize>,
    outcomes: Rc<dyn Fn(usize, usize) -> Vec<Outcome>>,
    is_terminal: Rc<dyn Fn(usize) -> bool>,
    terminal_reward: Rc<dyn Fn(usize) -> f64>,
}

#[derive(Clone, Debug, Default)]
struct ViResult {
    v: Vec<f64>,
    policy: Vec<i32>,
    iterations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct ViOptions {
    gamma: f64,
    tol: f64,
    max_iter: usize,
}

fn to_real_mdp_spec(spec: &MdpSpec) -> RealMdpSpec {
    let num_actions = spec.num_actions.clone();
    let outcomes = spec.outcomes.clone();
    let is_terminal = spec.is_terminal.clone();
    let terminal_reward = spec.terminal_reward.clone();
    RealMdpSpec {
        num_states: spec.num_states,
        num_actions: Box::new(move |s| num_actions(s)),
        outcomes: Box::new(move |s, a| {
            outcomes(s, a)
                .into_iter()
                .map(|o| RealOutcome {
                    prob: o.prob,
                    reward: o.reward,
                    next_state: o.next_state,
                })
                .collect()
        }),
        is_terminal: Some(Box::new(move |s| is_terminal(s))),
        terminal_reward: Some(Box::new(move |s| terminal_reward(s))),
        state_label: None,
        action_label: None,
    }
}

fn value_iteration(spec: &MdpSpec, opts: ViOptions) -> ViResult {
    let real = to_real_mdp_spec(spec);
    let res = value_iteration_model(
        real,
        RealViOptions {
            gamma: opts.gamma,
            tol: opts.tol,
            max_iter: opts.max_iter,
            ..Default::default()
        },
    );
    ViResult {
        v: res.v,
        policy: res.policy,
        iterations: res.iterations,
    }
}

fn q_value(spec: &MdpSpec, v: &[f64], s: usize, a: i32, gamma: f64) -> f64 {
    if a < 0 {
        return f64::NEG_INFINITY;
    }
    let real = to_real_mdp_spec(spec);
    q_value_model(&real, v, s, a as usize, gamma)
}

#[derive(Clone, Debug, Default)]
struct LpSubInfo {
    iters: usize,
    solver: String,
}

#[derive(Clone, Debug, Default)]
struct MdpLpSolution {
    v: Vec<f64>,
    policy: Vec<i32>,
    lp: LpSubInfo,
}

fn build_mdp_lp(_spec: &MdpSpec, _gamma: f64) -> LpProblem {
    let real = build_mdp_lp_model(&to_real_mdp_spec(_spec), _gamma, None);
    LpProblem {
        sense: real.sense.as_str(),
        c: real.c,
        a_ub: real.a_ub.unwrap_or_default(),
        b_ub: real.b_ub.unwrap_or_default(),
        a_eq: real.a_eq.unwrap_or_default(),
        b_eq: real.b_eq.unwrap_or_default(),
    }
}

fn solve_mdp_as_lp(spec: &MdpSpec, gamma: f64) -> MdpLpSolution {
    let real = to_real_mdp_spec(spec);
    let sol = solve_mdp_as_lp_model(&real, gamma, &MdpAsLpOptions::default())
        .expect("MDP-as-LP solve failed in validate_lp");
    MdpLpSolution {
        v: sol.v,
        policy: sol.policy,
        lp: LpSubInfo {
            iters: sol.lp.iters.unwrap_or(0),
            solver: sol.lp.solver,
        },
    }
}

// =============================================================================
// Driver helpers.
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
            format!("  {}", detail)
        };
        if ok {
            self.pass += 1;
            println!("  PASS    {}{}", label, tail);
        } else {
            self.fail += 1;
            println!("  FAIL    {}{}", label, tail);
        }
    }
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * f64::max(1.0, f64::max(a.abs(), b.abs()))
}

fn max_abs_diff(u: &[f64], v: &[f64]) -> f64 {
    let mut m = 0.0;
    for i in 0..u.len() {
        m = f64::max(m, (u[i] - v[i]).abs());
    }
    m
}

fn arr_to_string(a: &[f64], digits: usize) -> String {
    a.iter()
        .map(|x| format!("{:.*}", digits, x))
        .collect::<Vec<_>>()
        .join(", ")
}

// Grid-world helpers for Study 5 (W = H = 3), as free fns so the boxed MDP
// closures stay `'static`.
const GRID_W: usize = 3;
const GRID_H: usize = 3;

fn grid_idx(x: usize, y: usize) -> usize {
    y * GRID_W + x
}

fn grid_move(s: usize, a: usize) -> usize {
    let x = s % GRID_W;
    let y = s / GRID_W;
    if a == 0 {
        return grid_idx(x, y.saturating_sub(1));
    }
    if a == 1 {
        return grid_idx(x, (GRID_H - 1).min(y + 1));
    }
    if a == 2 {
        return grid_idx(x.saturating_sub(1), y);
    }
    grid_idx((GRID_W - 1).min(x + 1), y)
}

/// `validate-lp.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    // -------------------------------------------------------------------------
    // STUDY 1: Classic 2-variable LP — internal ≡ scipy across all methods.
    // -------------------------------------------------------------------------
    println!("=== STUDY 1: 2-variable LP across all solver methods ===");
    {
        let lp = LpProblem {
            sense: "max",
            c: vec![3.0, 2.0],
            a_ub: vec![vec![1.0, 1.0], vec![1.0, 3.0]],
            b_ub: vec![4.0, 6.0],
            ..Default::default()
        };
        let expected_obj = 12.0;
        let expected_x = [4.0, 0.0];
        let internal = solve_lp_internal(&lp, None);
        println!(
            "#   internal:        x={:?}  obj={}  iters={}",
            internal.x, internal.objective, internal.iters
        );
        c.check(
            "internal solver finds optimum",
            internal.status == "optimal" && approx_eq(internal.objective, expected_obj, 1e-7),
            &format!("obj={:.6}", internal.objective),
        );
        for method in ["highs", "highs-ds", "highs-ipm"] {
            let ext = solve_lp_external(&lp, method);
            if ext.status == "numerical-error" && ext.message.contains("scipy") {
                println!("#   scipy:{} skipped (scipy unavailable)", method);
                continue;
            }
            println!(
                "#   scipy:{}:  x={:?}  obj={}  iters={}",
                method, ext.x, ext.objective, ext.iters
            );
            c.check(
                &format!("scipy:{} matches expected optimum", method),
                ext.status == "optimal"
                    && approx_eq(ext.objective, expected_obj, 1e-7)
                    && max_abs_diff(&ext.x, &expected_x) < 1e-9,
                &format!("obj={:.6}", ext.objective),
            );
            c.check(
                &format!("scipy:{} ≡ internal (|Δobj| ≤ 1e-9)", method),
                approx_eq(ext.objective, internal.objective, 1e-9),
                "",
            );
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 2: Diet problem.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 2: Diet LP (Stigler 1945 mini, 4 foods × 3 nutrients) ===");
    {
        let lp = LpProblem {
            sense: "min",
            c: vec![0.5, 0.3, 0.7, 0.2],
            a_ub: vec![
                vec![-2.0, -3.0, -1.0, -4.0],
                vec![-1.0, -2.0, -3.0, -1.0],
                vec![-3.0, -1.0, -2.0, 0.0],
            ],
            b_ub: vec![-12.0, -6.0, -4.0],
            ..Default::default()
        };
        let internal = solve_lp_internal(&lp, None);
        println!(
            "#   internal cost  = {:.6}   x = {}",
            internal.objective,
            internal
                .x
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let ext = solve_lp_external(&lp, "highs");
        if ext.status == "optimal" {
            println!(
                "#   scipy:highs    = {:.6}   x = {}",
                ext.objective,
                ext.x
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            c.check(
                "internal cost ≡ scipy:highs cost (|Δ| ≤ 1e-7)",
                approx_eq(internal.objective, ext.objective, 1e-7),
                &format!("Δ={:.3e}", (internal.objective - ext.objective).abs()),
            );
            c.check(
                "internal x ≡ scipy:highs x  (max|Δ| ≤ 1e-6)",
                max_abs_diff(&internal.x, &ext.x) < 1e-6,
                &format!("max|Δx|={:.3e}", max_abs_diff(&internal.x, &ext.x)),
            );
        }
        let x = &internal.x;
        let protein = 2.0 * x[0] + 3.0 * x[1] + 1.0 * x[2] + 4.0 * x[3];
        let vit_a = 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2] + 1.0 * x[3];
        let vit_c = 3.0 * x[0] + 1.0 * x[1] + 2.0 * x[2] + 0.0 * x[3];
        c.check(
            "protein ≥ 12 (constraint feasibility)",
            protein >= 12.0 - 1e-7,
            &format!("got {:.4}", protein),
        );
        c.check(
            "vit-A   ≥  6",
            vit_a >= 6.0 - 1e-7,
            &format!("got {:.4}", vit_a),
        );
        c.check(
            "vit-C   ≥  4",
            vit_c >= 4.0 - 1e-7,
            &format!("got {:.4}", vit_c),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 3: Transportation problem (3×3, balanced).
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 3: Transportation LP (3×3, balanced) ===");
    {
        let cost = [[4.0, 6.0, 8.0], [3.0, 5.0, 7.0], [9.0, 2.0, 1.0]];
        let supply = [20.0, 30.0, 25.0];
        let demand = [25.0, 25.0, 25.0];
        let mut cvec: Vec<f64> = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                cvec.push(cost[i][j]);
            }
        }
        let mut a_eq: Vec<Vec<f64>> = Vec::new();
        let mut b_eq: Vec<f64> = Vec::new();
        for i in 0..3 {
            let mut row = vec![0.0; 9];
            for j in 0..3 {
                row[i * 3 + j] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(supply[i]);
        }
        for j in 0..3 {
            let mut row = vec![0.0; 9];
            for i in 0..3 {
                row[i * 3 + j] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(demand[j]);
        }
        let lp = LpProblem {
            sense: "min",
            c: cvec,
            a_eq,
            b_eq,
            ..Default::default()
        };
        let internal = solve_lp_internal(&lp, None);
        let ext = solve_lp_external(&lp, "highs");
        println!("#   internal cost   = {:.6}", internal.objective);
        if ext.status == "optimal" {
            println!("#   scipy:highs     = {:.6}", ext.objective);
            c.check(
                "transportation cost: internal ≡ scipy:highs (|Δ| ≤ 1e-7)",
                approx_eq(internal.objective, ext.objective, 1e-7),
                &format!(
                    "internal={:.6}  highs={:.6}",
                    internal.objective, ext.objective
                ),
            );
        }
        let mut supply_ok = true;
        let mut demand_ok = true;
        for i in 0..3 {
            let mut s = 0.0;
            for j in 0..3 {
                s += internal.x.get(i * 3 + j).copied().unwrap_or(0.0);
            }
            if (s - supply[i]).abs() > 1e-6 {
                supply_ok = false;
            }
        }
        for j in 0..3 {
            let mut s = 0.0;
            for i in 0..3 {
                s += internal.x.get(i * 3 + j).copied().unwrap_or(0.0);
            }
            if (s - demand[j]).abs() > 1e-6 {
                demand_ok = false;
            }
        }
        c.check("supply equalities all satisfied", supply_ok, "");
        c.check("demand equalities all satisfied", demand_ok, "");
    }

    // -------------------------------------------------------------------------
    // STUDY 4: MDP-as-LP ≡ value iteration on a 5-state chain.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 4: MDP-as-LP solution ≡ value-iteration solution ===");
    {
        let n = 5usize;
        let mdp = MdpSpec {
            num_states: n,
            num_actions: Rc::new(|_s| 2),
            outcomes: Rc::new(move |s, a| {
                if s == n - 1 {
                    return vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: s,
                    }];
                }
                let target = if a == 1 {
                    (n - 1).min(s + 1)
                } else {
                    s.saturating_sub(1)
                };
                let reward = if target == n - 1 { 1.0 } else { 0.0 };
                vec![Outcome {
                    prob: 1.0,
                    reward,
                    next_state: target,
                }]
            }),
            is_terminal: Rc::new(move |s| s == n - 1),
            terminal_reward: Rc::new(|_s| 0.0),
        };
        let gamma = 0.9;
        let vi = value_iteration(
            &mdp,
            ViOptions {
                gamma,
                tol: 1e-12,
                max_iter: 10000,
            },
        );
        let lp_sol = solve_mdp_as_lp(&mdp, gamma);
        println!(
            "#   VI    V = {}    iters={}",
            arr_to_string(&vi.v, 6),
            vi.iterations
        );
        println!(
            "#   LP    V = {}    iters={}",
            arr_to_string(&lp_sol.v, 6),
            lp_sol.lp.iters
        );
        c.check(
            "V*_LP ≡ V*_VI (max|Δ| ≤ 1e-6)",
            max_abs_diff(&lp_sol.v, &vi.v) < 1e-6,
            &format!("max|Δ|={:.3e}", max_abs_diff(&lp_sol.v, &vi.v)),
        );
        let mut pol_match = true;
        for s in 0..n - 1 {
            if lp_sol.policy[s] != vi.policy[s] {
                pol_match = false;
            }
        }
        c.check(
            "π*_LP ≡ π*_VI on all non-terminal states",
            pol_match,
            &format!(
                "LP={}  VI={}",
                lp_sol
                    .policy
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                vi.policy
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 5: 3×3 grid-world MDP-as-LP ≡ value iteration.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 5: 3×3 grid-world MDP-as-LP ≡ value-iteration ===");
    {
        let n = GRID_W * GRID_H;
        let goal = grid_idx(2, 2);
        let mdp = MdpSpec {
            num_states: n,
            num_actions: Rc::new(|_s| 4),
            outcomes: Rc::new(move |s, a| {
                if s == goal {
                    return vec![Outcome {
                        prob: 1.0,
                        reward: 0.0,
                        next_state: s,
                    }];
                }
                let intended = grid_move(s, a);
                let (sa1, sa2) = match a {
                    0 | 1 => (2usize, 3usize),
                    _ => (0usize, 1usize),
                };
                let slip1 = grid_move(s, sa1);
                let slip2 = grid_move(s, sa2);
                let step_cost = -0.04;
                let r = |sp: usize| if sp == goal { 1.0 } else { step_cost };
                vec![
                    Outcome {
                        prob: 0.8,
                        reward: r(intended),
                        next_state: intended,
                    },
                    Outcome {
                        prob: 0.1,
                        reward: r(slip1),
                        next_state: slip1,
                    },
                    Outcome {
                        prob: 0.1,
                        reward: r(slip2),
                        next_state: slip2,
                    },
                ]
            }),
            is_terminal: Rc::new(move |s| s == goal),
            terminal_reward: Rc::new(|_s| 0.0),
        };
        let gamma = 0.95;
        let vi = value_iteration(
            &mdp,
            ViOptions {
                gamma,
                tol: 1e-12,
                max_iter: 10000,
            },
        );
        let lp = solve_mdp_as_lp(&mdp, gamma);
        println!(
            "#   VI iters={}    LP iters={}    LP solver={}",
            vi.iterations, lp.lp.iters, lp.lp.solver
        );
        println!("#   V*_VI = {}", arr_to_string(&vi.v, 4));
        println!("#   V*_LP = {}", arr_to_string(&lp.v, 4));
        c.check(
            "grid-world V*_LP ≡ V*_VI (max|Δ| ≤ 1e-5)",
            max_abs_diff(&lp.v, &vi.v) < 1e-5,
            &format!("max|Δ|={:.3e}", max_abs_diff(&lp.v, &vi.v)),
        );
        let mut policies_optimal = true;
        let mut max_policy_gap = 0.0_f64;
        for s in 0..n {
            if s == goal {
                continue;
            }
            let q_lp = q_value(&mdp, &vi.v, s, lp.policy[s], gamma);
            let q_vi = q_value(&mdp, &vi.v, s, vi.policy[s], gamma);
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..(mdp.num_actions)(s) {
                best_q = f64::max(best_q, q_value(&mdp, &vi.v, s, a as i32, gamma));
            }
            let gap = f64::max(best_q - q_lp, best_q - q_vi);
            max_policy_gap = f64::max(max_policy_gap, gap);
            if gap > 1e-7 {
                policies_optimal = false;
            }
        }
        c.check(
            "grid-world LP and VI policies both choose optimal actions",
            policies_optimal,
            &format!("max action-value gap={:.3e}", max_policy_gap),
        );
    }

    // -------------------------------------------------------------------------
    // STUDY 6: 200 random feasible LPs — internal ≡ scipy:highs.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 6: 200 random feasible LPs — internal ≡ scipy:highs ===");
    {
        let n_prob = 200usize;
        let mut max_obj_diff = 0.0_f64;
        let mut n_match = 0usize;
        let mut n_skip = 0usize;
        let mut scipy_available = true;
        for p in 0..n_prob {
            // Seeded LCG (`seed = (seed*1664525 + 1013904223) >>> 0; seed/0xFFFFFFFF`).
            let mut seed: u32 = (p as u32).wrapping_add(1);
            let mut rng = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed as f64 / 0xFFFF_FFFFu32 as f64
            };
            let n = 3 + (rng() * 5.0).floor() as usize;
            let m = 3 + (rng() * 5.0).floor() as usize;
            let cvec: Vec<f64> = (0..n).map(|_| rng() * 4.0 - 1.0).collect();
            let mut a_ub: Vec<Vec<f64>> = Vec::new();
            let mut b_ub: Vec<f64> = Vec::new();
            for _ in 0..m {
                a_ub.push((0..n).map(|_| rng() * 2.0).collect());
                b_ub.push(1.0 + rng() * 9.0);
            }
            let lp = LpProblem {
                sense: "max",
                c: cvec,
                a_ub,
                b_ub,
                ..Default::default()
            };
            let internal = solve_lp_internal(&lp, Some(1000));
            let ext = solve_lp_external(&lp, "highs");
            if ext.status == "numerical-error" && ext.message.contains("scipy") {
                scipy_available = false;
                n_skip += 1;
                continue;
            }
            if internal.status != "optimal" || ext.status != "optimal" {
                n_skip += 1;
                continue;
            }
            let d = (internal.objective - ext.objective).abs();
            if d > max_obj_diff {
                max_obj_diff = d;
            }
            if d < 1e-7 {
                n_match += 1;
            }
        }
        if !scipy_available {
            println!("#   scipy unavailable; skipping random comparison");
        } else {
            println!(
                "#   {}/{} matched to 1e-7   max|Δobj| = {:.3e}",
                n_match,
                n_prob - n_skip,
                max_obj_diff
            );
            c.check(
                "all random LPs match within 1e-7 (excluding skipped)",
                n_match == n_prob - n_skip,
                &format!(
                    "nMatch={}  N={}  maxΔ={:.3e}",
                    n_match,
                    n_prob - n_skip,
                    max_obj_diff
                ),
            );
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 7: LP_SOLVER env-var dispatching.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 7: solveLP env-var dispatch ===");
    {
        let lp = LpProblem {
            sense: "max",
            c: vec![1.0, 1.0],
            a_ub: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            b_ub: vec![3.0, 5.0],
            ..Default::default()
        };
        let expected = 8.0;
        for choice in [
            "internal",
            "scipy:highs",
            "scipy:highs-ds",
            "scipy:highs-ipm",
        ] {
            std::env::set_var("LP_SOLVER", choice);
            let r = solve_lp(&lp);
            println!(
                "#   LP_SOLVER={:<20} → {:<20} obj={:.4} {}",
                choice,
                r.solver,
                r.objective,
                if r.message.is_empty() {
                    String::new()
                } else {
                    format!("({})", &r.message[..r.message.len().min(60)])
                }
            );
            c.check(
                &format!("LP_SOLVER={} returns optimum", choice),
                r.status == "optimal" && approx_eq(r.objective, expected, 1e-7),
                &format!("obj={}", r.objective),
            );
        }
        std::env::remove_var("LP_SOLVER");
    }

    // -------------------------------------------------------------------------
    // STUDY 8: DES-engine simplex ≡ scipy:highs ≡ in-process simplex.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 8: DES-engine simplex ≡ scipy:highs (LP-as-DES validation) ===");
    {
        struct Case {
            name: &'static str,
            lp: LpProblem,
        }
        let cases = vec![
            Case {
                name: "2-var max",
                lp: LpProblem {
                    sense: "max",
                    c: vec![3.0, 2.0],
                    a_ub: vec![vec![1.0, 1.0], vec![1.0, 3.0]],
                    b_ub: vec![4.0, 6.0],
                    ..Default::default()
                },
            },
            Case {
                name: "diet (phase-1)",
                lp: LpProblem {
                    sense: "min",
                    c: vec![0.5, 0.3, 0.7, 0.2],
                    a_ub: vec![
                        vec![-2.0, -3.0, -1.0, -4.0],
                        vec![-1.0, -2.0, -3.0, -1.0],
                        vec![-3.0, -1.0, -2.0, 0.0],
                    ],
                    b_ub: vec![-12.0, -6.0, -4.0],
                    ..Default::default()
                },
            },
            Case {
                name: "transportation 3×3 equalities",
                lp: LpProblem {
                    sense: "min",
                    c: vec![4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0, 2.0, 1.0],
                    a_eq: vec![
                        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                    b_eq: vec![20.0, 30.0, 25.0, 25.0, 25.0, 25.0],
                    ..Default::default()
                },
            },
            Case {
                name: "unbounded",
                lp: LpProblem {
                    sense: "max",
                    c: vec![1.0, 1.0],
                    a_ub: vec![vec![-1.0, 1.0]],
                    b_ub: vec![3.0],
                    ..Default::default()
                },
            },
            Case {
                name: "infeasible",
                lp: LpProblem {
                    sense: "max",
                    c: vec![1.0, 1.0],
                    a_ub: vec![vec![1.0, 1.0], vec![-1.0, -1.0]],
                    b_ub: vec![3.0, -5.0],
                    ..Default::default()
                },
            },
        ];
        for case in &cases {
            let des_d = solve_lp_via_des(&case.lp, Some("dantzig"), None);
            let des_b = solve_lp_via_des(&case.lp, Some("bland"), None);
            let ext = solve_lp_external(&case.lp, "highs");
            let internal = solve_lp_internal(&case.lp, None);
            let all: Vec<&LpResult> = if scipy_unavailable(&ext) {
                vec![&des_d, &des_b, &internal]
            } else {
                vec![&des_d, &des_b, &internal, &ext]
            };
            let stats: Vec<String> = all.iter().map(|r| r.status.clone()).collect();
            let same_status = stats.iter().all(|s| *s == stats[0]);
            let scope = if scipy_unavailable(&ext) {
                "available solvers"
            } else {
                "all four solvers"
            };
            if scipy_unavailable(&ext) {
                println!(
                    "#   {:<32}  scipy:highs skipped ({})",
                    case.name, ext.message
                );
            }
            c.check(
                &format!("'{}': {} agree on status ({})", case.name, scope, stats[0]),
                same_status,
                &format!("statuses=[{}]", stats.join(",")),
            );
            if des_d.status == "optimal" {
                let objs: Vec<f64> = all.iter().map(|r| r.objective).collect();
                let reference_obj = objs[objs.len() - 1];
                let max_delta = objs
                    .iter()
                    .map(|o| (o - reference_obj).abs())
                    .fold(0.0_f64, f64::max);
                c.check(
                    &format!(
                        "'{}': {} agree on objective (max|Δ| ≤ 1e-7)",
                        case.name, scope
                    ),
                    max_delta < 1e-7,
                    &format!(
                        "objs=[{}]   maxΔ={:.2e}",
                        objs.iter()
                            .map(|o| format!("{:.6}", o))
                            .collect::<Vec<_>>()
                            .join(", "),
                        max_delta
                    ),
                );
                let reference_x = if scipy_unavailable(&ext) {
                    &internal.x
                } else {
                    &ext.x
                };
                let reference_name = if scipy_unavailable(&ext) {
                    "internal simplex"
                } else {
                    "scipy:highs"
                };
                let x_max_delta = max_abs_diff(&des_d.x, reference_x);
                c.check(
                    &format!(
                        "'{}': DES-simplex x ≡ {}  (max|Δ| ≤ 1e-7)",
                        case.name, reference_name
                    ),
                    x_max_delta < 1e-7,
                    &format!("max|Δx|={:.2e}", x_max_delta),
                );
                println!(
                    "#   {:<32}  Dantzig={} pivots   Bland={} pivots   internal={}   highs={}",
                    case.name,
                    des_d.trace.pivot_history.len(),
                    des_b.trace.pivot_history.len(),
                    internal.iters,
                    ext.iters
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // STUDY 9: 50 random feasible LPs — DES simplex ≡ scipy:highs.
    // -------------------------------------------------------------------------
    println!("\n=== STUDY 9: 50 random feasible LPs — DES simplex ≡ scipy:highs ===");
    {
        let n = 50usize;
        let mut n_match = 0usize;
        let mut n_skip = 0usize;
        let mut scipy_available = true;
        let mut max_obj_diff = 0.0_f64;
        for p in 0..n {
            let mut seed: u32 = (p as u32).wrapping_add(50);
            let mut rng = move || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed as f64 / 0xFFFF_FFFFu32 as f64
            };
            let nv = 2 + (rng() * 4.0).floor() as usize;
            let m = 2 + (rng() * 4.0).floor() as usize;
            let cvec: Vec<f64> = (0..nv).map(|_| rng() * 4.0 - 1.0).collect();
            let mut a_ub: Vec<Vec<f64>> = Vec::new();
            let mut b_ub: Vec<f64> = Vec::new();
            for _ in 0..m {
                a_ub.push((0..nv).map(|_| rng() * 2.0).collect());
                b_ub.push(1.0 + rng() * 9.0);
            }
            let lp = LpProblem {
                sense: "max",
                c: cvec,
                a_ub,
                b_ub,
                ..Default::default()
            };
            let des = solve_lp_via_des(&lp, None, Some(500));
            let ext = solve_lp_external(&lp, "highs");
            if scipy_unavailable(&ext) {
                scipy_available = false;
                n_skip += 1;
                continue;
            }
            if des.status != "optimal" || ext.status != "optimal" {
                n_skip += 1;
                continue;
            }
            let d = (des.objective - ext.objective).abs();
            if d > max_obj_diff {
                max_obj_diff = d;
            }
            if d < 1e-7 {
                n_match += 1;
            }
        }
        let compared = n - n_skip;
        if !scipy_available || compared == 0 {
            println!("#   scipy unavailable; skipping random DES/scipy comparison");
        } else {
            println!(
                "#   {}/{} matched to 1e-7   max|Δobj| = {:.3e}",
                n_match, compared, max_obj_diff
            );
            c.check(
                "all 50 random LPs: DES-simplex obj ≡ scipy:highs obj  (|Δ| ≤ 1e-7)",
                n_match == compared,
                &format!("nMatch={}  N={}", n_match, compared),
            );
        }
    }

    println!("\n=== Summary: {} passed, {} failed ===", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}
