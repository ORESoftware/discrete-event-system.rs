//! Port of `src/des/mdp/value-iteration.ts`.
//!
//! Generic Bellman value iteration for finite-state, finite-action MDPs,
//! specialised to the USACC court-case model in `usacc_mdp`.
//!
//! Bellman optimality: V*(s) = max_a sum_s' T(s'|s,a) * (r(s,a,s') + gamma *
//! V*(s')), with the greedy policy pi*(s) = argmax_a of the same quantity.
//!
//! Updates are synchronous: V_next is built into a fresh array, then swapped.
//! Sum-coalescing in `usacc_mdp::outcomes` keeps each (s, a) to a handful of
//! outcomes, so the inner loop is fast. Convergence stops when
//! max_s |V_next(s) - V(s)| < tol.

#![allow(dead_code)]

use crate::des::mdp::usacc_mdp::{
    is_terminal, outcomes, terminal_reward, Outcome, ACCEPTED, CLOSED, EXHAUSTED, N_ACTIONS,
    N_STATES,
};

/// Options controlling value iteration. The TS optional fields with `??`
/// defaults are modelled as concrete fields plus a `Default` impl.
#[derive(Clone, Copy, Debug)]
pub struct VIOptions {
    /// Discount factor (default 0.95).
    pub gamma: f64,
    /// Convergence tolerance on max |dV| (default 1e-9).
    pub tol: f64,
    /// Hard cap on iterations (default 5000).
    pub max_iter: usize,
}

impl Default for VIOptions {
    fn default() -> Self {
        VIOptions {
            gamma: 0.95,
            tol: 1e-9,
            max_iter: 5000,
        }
    }
}

/// Result of value iteration.
#[derive(Clone, Debug)]
pub struct VIResult {
    /// Optimal value function, one entry per state id.
    pub v: Vec<f64>,
    /// Action index per state id (-1 for absorbing terminals).
    pub policy: Vec<i32>,
    pub iterations: usize,
    pub final_delta: f64,
    pub gamma: f64,
}

/// Pre-build a transition table so value iteration doesn't recompute
/// `outcomes()` on every sweep (a substantial speedup for even moderate state
/// spaces).
pub fn build_transition_table() -> Vec<Vec<Vec<Outcome>>> {
    let mut table: Vec<Vec<Vec<Outcome>>> = Vec::with_capacity(N_STATES);
    for s in 0..N_STATES {
        let mut per_action: Vec<Vec<Outcome>> = Vec::with_capacity(N_ACTIONS);
        for a in 0..N_ACTIONS {
            per_action.push(outcomes(s, a));
        }
        table.push(per_action);
    }
    table
}

pub fn value_iteration(opts: VIOptions) -> VIResult {
    let gamma = opts.gamma;
    assert!(
        gamma.is_finite() && (0.0..=1.0).contains(&gamma),
        "mdp value iteration: discount gamma must be finite and in [0, 1], got {gamma}"
    );
    let tol = opts.tol;
    let max_iter = opts.max_iter;
    assert!(
        tol.is_finite() && tol > 0.0,
        "mdp value iteration: convergence tolerance must be finite and > 0, got {tol}"
    );
    assert!(
        max_iter > 0,
        "mdp value iteration: max_iter must be greater than zero"
    );

    let t = build_transition_table();
    let mut v = vec![0.0_f64; N_STATES];
    // Initialize terminal states to their terminal reward; non-terminal to 0.
    v[ACCEPTED] = terminal_reward(ACCEPTED);
    v[CLOSED] = terminal_reward(CLOSED);
    v[EXHAUSTED] = terminal_reward(EXHAUSTED);

    let mut iterations = 0usize;
    let mut final_delta = f64::INFINITY;
    for iter in 0..max_iter {
        let mut vn = vec![0.0_f64; N_STATES];
        vn[ACCEPTED] = v[ACCEPTED];
        vn[CLOSED] = v[CLOSED];
        vn[EXHAUSTED] = v[EXHAUSTED];
        let mut delta = 0.0_f64;
        for s in 0..N_STATES {
            if is_terminal(s) {
                continue;
            }
            let mut best = f64::NEG_INFINITY;
            for a in 0..N_ACTIONS {
                let ol = &t[s][a];
                let mut q = 0.0;
                for o in ol {
                    q += o.prob * (o.reward + gamma * v[o.next_state]);
                }
                if q > best {
                    best = q;
                }
            }
            vn[s] = best;
            let d = (vn[s] - v[s]).abs();
            if d > delta {
                delta = d;
            }
        }
        v = vn;
        iterations = iter + 1;
        final_delta = delta;
        if delta < tol {
            break;
        }
    }

    if final_delta >= tol {
        eprintln!(
            "[mdp.valueIteration] did not converge: maxIter={max_iter} reached with max|\u{0394}V|={final_delta} \u{2265} tol={tol} (gamma={gamma}). Increase maxIter or check the model."
        );
    } else {
        eprintln!(
            "[mdp.valueIteration] converged in {iterations} iterations (max|\u{0394}V|={final_delta} < tol={tol}, gamma={gamma})."
        );
    }

    // Extract greedy policy from V.
    let mut policy = vec![0_i32; N_STATES];
    for s in 0..N_STATES {
        if is_terminal(s) {
            policy[s] = -1;
            continue;
        }
        let mut best_a = 0_i32;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..N_ACTIONS {
            let ol = &t[s][a];
            let mut q = 0.0;
            for o in ol {
                q += o.prob * (o.reward + gamma * v[o.next_state]);
            }
            if q > best_q {
                best_q = q;
                best_a = a as i32;
            }
        }
        policy[s] = best_a;
    }
    VIResult {
        v,
        policy,
        iterations,
        final_delta,
        gamma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_table_shape() {
        let t = build_transition_table();
        assert_eq!(t.len(), N_STATES);
        for per_action in &t {
            assert_eq!(per_action.len(), N_ACTIONS);
        }
    }

    #[test]
    #[should_panic(expected = "discount gamma must be finite and in [0, 1]")]
    fn rejects_out_of_range_discount() {
        let _ = value_iteration(VIOptions {
            gamma: 1.5,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "convergence tolerance must be finite and > 0")]
    fn rejects_non_finite_convergence_tolerance() {
        let _ = value_iteration(VIOptions {
            tol: f64::NAN,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_iter must be greater than zero")]
    fn rejects_zero_iteration_budget() {
        let _ = value_iteration(VIOptions {
            max_iter: 0,
            ..Default::default()
        });
    }

    #[test]
    fn converges_and_policy_well_formed() {
        let res = value_iteration(VIOptions::default());
        assert_eq!(res.v.len(), N_STATES);
        assert_eq!(res.policy.len(), N_STATES);
        // Terminals are absorbing with policy -1.
        assert_eq!(res.policy[ACCEPTED], -1);
        assert_eq!(res.policy[CLOSED], -1);
        assert_eq!(res.policy[EXHAUSTED], -1);
        // It should converge within the default cap.
        assert!(
            res.final_delta < VIOptions::default().tol,
            "delta={}",
            res.final_delta
        );
        for &x in &res.v {
            assert!(x.is_finite(), "value not finite: {x}");
        }
        // Non-terminal states pick a valid action index.
        for s in 0..N_STATES {
            if !is_terminal(s) {
                assert!(
                    (0..N_ACTIONS as i32).contains(&res.policy[s]),
                    "bad policy at {s}: {}",
                    res.policy[s]
                );
            }
        }
    }
}
