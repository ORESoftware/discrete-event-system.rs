//! Port of `src/des/main-mdp-lp.ts`.
//!
//! Solves a small MDP via its LP formulation, cross-checked against value
//! iteration. (The TS optional external `scipy:highs` solve is handled inside
//! `general::lp` / `general::des_lp_bridge`; here we use the default solver.)
//!
//! Conversion notes:
//!   - the TS `MDPSpec` callbacks (`numActions`, `outcomes`, …) become the
//!     boxed closures of `general::value_iteration::MDPSpec`.
//!   - value-iteration vs LP cross-check are pure calls; top-level `main()` →
//!     [`run`]. `process.exit(2)` on unknown PROBLEM → early `return`.

use std::time::Instant;

use crate::des::general::des_lp_bridge::{solve_mdp_as_lp, MdpAsLpOptions};
use crate::des::general::lp::Sense;
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};

// -----------------------------------------------------------------------------
// Problem 1: inventory control.
// -----------------------------------------------------------------------------
fn build_inventory_mdp() -> MDPSpec {
    let s_max = 10usize;
    let k_max = 6usize;
    let lambda = 4.0_f64;
    let p = 6.0_f64; // selling price
    let c = 2.0_f64; // order cost per unit
    let h = 1.0_f64; // holding cost per unit per period
    let q = 5.0_f64; // stockout penalty per missed unit

    let d_cap = 10usize;
    let mut pmf: Vec<f64> = Vec::with_capacity(d_cap + 1);
    let mut factorial = 1.0_f64;
    for k in 0..=d_cap {
        if k > 0 {
            factorial *= k as f64;
        }
        pmf.push(lambda.powi(k as i32) * (-lambda).exp() / factorial);
    }
    let norm: f64 = pmf.iter().sum();
    for v in pmf.iter_mut() {
        *v /= norm;
    }

    let num_states = s_max + 1;
    let pmf_out = pmf.clone();
    MDPSpec {
        num_states,
        num_actions: Box::new(move |_s| k_max + 1),
        outcomes: Box::new(move |s, a| {
            let stock_after_order = s_max.min(s + a);
            let order_cost = c * a as f64;
            let hold_cost = h * stock_after_order as f64;
            let mut out = Vec::new();
            for d in 0..=d_cap {
                if pmf_out[d] == 0.0 {
                    continue;
                }
                let sold = stock_after_order.min(d);
                let stockout = d - sold;
                let revenue = p * sold as f64;
                let stockout_penalty = q * stockout as f64;
                let reward = revenue - order_cost - hold_cost - stockout_penalty;
                let next_state = stock_after_order - sold;
                out.push(Outcome {
                    prob: pmf_out[d],
                    reward,
                    next_state,
                });
            }
            out
        }),
        is_terminal: None,
        terminal_reward: None,
        state_label: Some(Box::new(|s| format!("inv={}", s))),
        action_label: Some(Box::new(|a| format!("order={}", a))),
    }
}

// -----------------------------------------------------------------------------
// Problem 2: chain MDP.
// -----------------------------------------------------------------------------
fn build_chain_mdp() -> MDPSpec {
    let n = 5usize;
    MDPSpec {
        num_states: n,
        num_actions: Box::new(|_s| 2),
        outcomes: Box::new(move |s, a| {
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
        is_terminal: Some(Box::new(move |s| s == n - 1)),
        terminal_reward: Some(Box::new(|_| 0.0)),
        state_label: Some(Box::new(|s| format!("s{}", s))),
        action_label: Some(Box::new(|a| {
            if a == 0 {
                "left".to_string()
            } else {
                "right".to_string()
            }
        })),
    }
}

// -----------------------------------------------------------------------------
// Problem 3: 4×4 grid-world with stochastic transitions.
// -----------------------------------------------------------------------------
fn build_grid_mdp() -> MDPSpec {
    let w = 4usize;
    let h = 4usize;
    let n = w * h;
    let idx = move |x: usize, y: usize| y * w + x;
    let move_fn = move |s: usize, a: usize| -> usize {
        let x = s % w;
        let y = s / w;
        match a {
            0 => idx(x, y.saturating_sub(1)), // up
            1 => idx(x, (h - 1).min(y + 1)),  // down
            2 => idx(x.saturating_sub(1), y), // left
            _ => idx((w - 1).min(x + 1), y),  // right
        }
    };
    let slip: [[usize; 2]; 4] = [[2, 3], [2, 3], [0, 1], [0, 1]];
    let goal = idx(3, 3);
    let pit = idx(1, 2);
    MDPSpec {
        num_states: n,
        num_actions: Box::new(|_s| 4),
        outcomes: Box::new(move |s, a| {
            if s == goal || s == pit {
                return vec![Outcome {
                    prob: 1.0,
                    reward: 0.0,
                    next_state: s,
                }];
            }
            let intended = move_fn(s, a);
            let s1 = slip[a][0];
            let s2 = slip[a][1];
            let sl1 = move_fn(s, s1);
            let sl2 = move_fn(s, s2);
            let r = |sp: usize| {
                if sp == goal {
                    1.0
                } else if sp == pit {
                    -1.0
                } else {
                    -0.04
                }
            };
            vec![
                Outcome {
                    prob: 0.8,
                    reward: r(intended),
                    next_state: intended,
                },
                Outcome {
                    prob: 0.1,
                    reward: r(sl1),
                    next_state: sl1,
                },
                Outcome {
                    prob: 0.1,
                    reward: r(sl2),
                    next_state: sl2,
                },
            ]
        }),
        is_terminal: Some(Box::new(move |s| s == goal || s == pit)),
        terminal_reward: Some(Box::new(|_| 0.0)),
        state_label: Some(Box::new(move |s| format!("({},{})", s % w, s / w))),
        action_label: Some(Box::new(|a| ["up", "down", "left", "right"][a].to_string())),
    }
}

fn build_mdp(which: &str) -> Option<MDPSpec> {
    match which {
        "inventory" => Some(build_inventory_mdp()),
        "chain" => Some(build_chain_mdp()),
        "gridworld" => Some(build_grid_mdp()),
        _ => None,
    }
}

/// `maxAbs(u, v)` — max element-wise absolute difference.
fn max_abs(u: &[f64], v: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..u.len() {
        m = m.max((u[i] - v[i]).abs());
    }
    m
}

fn js_exp3(x: f64) -> String {
    format!("{:.3e}", x)
}

/// Entry point (`main()` in the TS source). Sense import kept since the LP
/// formulation uses `general::lp::Sense` (re-exported for downstream wiring).
pub fn run() {
    let _ = Sense::Max; // LP formulation sense is fixed inside des_lp_bridge.
    let which = std::env::var("PROBLEM").unwrap_or_else(|_| "inventory".to_string());
    let gamma: f64 = std::env::var("GAMMA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let solver_env = std::env::var("LP_SOLVER").unwrap_or_else(|_| "scipy:highs".to_string());

    let Some(mdp_vi) = build_mdp(&which) else {
        eprintln!(
            "unknown PROBLEM='{}'; expected one of: inventory, chain, gridworld",
            which
        );
        return;
    };

    let mdp = build_mdp(&which).expect("PROBLEM validated above");
    println!(
        "# MDP-as-LP: solving '{}' MDP with {} states, γ={}",
        which, mdp.num_states, gamma
    );
    println!("#   LP_SOLVER={}", solver_env);
    println!();

    // ---- 1. Value iteration (reference) ----
    let t0 = Instant::now();
    let vi = value_iteration(
        mdp_vi,
        VIOptions {
            gamma,
            tol: 1e-12,
            max_iter: 100000,
            ..Default::default()
        },
    );
    let t_vi = t0.elapsed().as_millis();
    println!("# Value iteration (reference):");
    println!("#   iterations  = {}", vi.iterations);
    println!("#   final delta = {}", js_exp3(vi.final_delta));
    println!("#   wall time   = {}ms", t_vi);
    println!();

    // ---- 2. MDP-as-LP ----
    let t1 = Instant::now();
    let lp =
        solve_mdp_as_lp(&mdp, gamma, &MdpAsLpOptions::default()).expect("MDP-as-LP solve failed");
    let t_lp = t1.elapsed().as_millis();
    println!("# MDP-as-LP:");
    println!("#   solver      = {}", lp.lp.solver);
    println!("#   iterations  = {}", lp.lp.iters.unwrap_or(0));
    println!(
        "#   wall time   = {}ms (incl. Python startup if external)",
        t_lp
    );
    println!();

    // ---- 3. Compare ----
    let d_v = max_abs(&vi.v, &lp.v);
    println!("# Comparison:");
    println!("#   max|V_LP − V_VI|  = {}", js_exp3(d_v));
    let mut pol_match = true;
    for s in 0..mdp.num_states {
        if mdp.is_terminal.as_ref().map(|f| f(s)).unwrap_or(false) {
            continue;
        }
        if vi.policy[s] != lp.policy[s] {
            pol_match = false;
        }
    }
    println!("#   π_LP ≡ π_VI?      = {}", pol_match);
    println!();

    // ---- 4. Pretty-print V* and π* ----
    let n = mdp.num_states;
    println!("# Optimal V* and π*:");
    println!("#   {:<14} {:>12}  {:<12}", "state", "V*", "π*");
    for s in 0..n {
        let lbl = mdp
            .state_label
            .as_ref()
            .map(|f| f(s))
            .unwrap_or_else(|| format!("s{}", s));
        let v = format!("{:.6}", lp.v[s]);
        let a = lp.policy[s];
        let al = if a < 0 {
            "(terminal)".to_string()
        } else {
            mdp.action_label
                .as_ref()
                .map(|f| f(a as usize))
                .unwrap_or_else(|| format!("a{}", a))
        };
        println!("#   {:<14} {:>12}  {}", lbl, v, al);
    }
}
