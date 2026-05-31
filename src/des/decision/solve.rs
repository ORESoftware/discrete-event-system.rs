//! Unified solving for the canonical specs. Each function reuses an existing
//! solver and returns a serializable solution (for the results payload) or a
//! [`PomdpPlan`] (a policy you can act with during a rollout).

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde::Serialize;

use crate::des::general::belief::DiscreteBelief;
use crate::des::general::pomdp::{
    mdp_value_iteration, pomdp_exact_finite_horizon, BeliefLookaheadOptions, BeliefLookaheadSolver,
    MDPVIOptions, MostLikelyStateSolver, POMDPExactResult, QMDPSolver,
};
use crate::des::general::value_iteration::{value_iteration, VIOptions};

use super::spec::{MdpSpec, PomdpSpec};

/// MDP solve methods. (Value iteration today; the enum leaves room for policy
/// iteration / LP without changing the contract.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MdpMethod {
    ValueIteration,
}

impl Default for MdpMethod {
    fn default() -> Self {
        MdpMethod::ValueIteration
    }
}

/// MDP solution: optimal value function, greedy policy, and Q-values.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MdpSolution {
    pub value: Vec<f64>,
    /// Greedy action per state; `-1` for terminal/no-action states.
    pub policy: Vec<i32>,
    /// `q[state][action]` under the optimal value function.
    pub q: Vec<Vec<f64>>,
    pub iterations: usize,
    pub final_delta: f64,
    pub discount: f64,
}

/// Solve an MDP with value iteration (discount taken from the spec).
pub fn solve_mdp(spec: &MdpSpec, _method: MdpMethod) -> Result<MdpSolution, String> {
    spec.validate()?;
    let vi_spec = spec.to_value_iteration_spec();
    let opts = VIOptions {
        gamma: spec.discount,
        ..VIOptions::default()
    };
    let result = catch_unwind(AssertUnwindSafe(|| value_iteration(vi_spec, opts)))
        .map_err(|_| "value iteration failed (check transition probabilities)".to_string())?;

    // Q-values from the tabular spec (cheap, avoids rebuilding the closure spec).
    let gamma = spec.discount;
    let mut q = vec![Vec::new(); spec.num_states];
    for s in 0..spec.num_states {
        let actions = &spec.transitions[s];
        let mut qs = Vec::with_capacity(actions.len());
        for outcomes in actions {
            let mut qv = 0.0;
            for o in outcomes {
                qv += o.prob * (o.reward + gamma * result.v[o.next]);
            }
            qs.push(qv);
        }
        q[s] = qs;
    }

    Ok(MdpSolution {
        value: result.v,
        policy: result.policy,
        q,
        iterations: result.iterations,
        final_delta: result.final_delta,
        discount: gamma,
    })
}

/// POMDP solve methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PomdpMethod {
    /// QMDP heuristic (solve the underlying MDP, act greedily under the belief).
    Qmdp,
    /// Act as if the modal hidden state is the truth.
    MostLikelyState,
    /// Finite-horizon belief-tree lookahead (QMDP leaves).
    Lookahead,
    /// Exact α-vector finite-horizon value iteration (small problems only).
    ExactFiniteHorizon,
}

impl Default for PomdpMethod {
    fn default() -> Self {
        PomdpMethod::Qmdp
    }
}

/// A solved POMDP policy you can act with given a belief. Owns its solver.
pub enum PomdpPlan {
    Qmdp(QMDPSolver<usize, usize, usize>),
    MostLikely(MostLikelyStateSolver<usize, usize, usize>),
    Lookahead(BeliefLookaheadSolver<usize, usize, usize>),
    Exact(POMDPExactResult),
}

impl PomdpPlan {
    /// The action this plan chooses at belief `b`.
    pub fn act(&mut self, b: &DiscreteBelief<usize>) -> usize {
        match self {
            PomdpPlan::Qmdp(s) => s.act(b, None, 0.0),
            PomdpPlan::MostLikely(s) => s.act(b),
            PomdpPlan::Lookahead(s) => s.act(b, None, 0.0),
            PomdpPlan::Exact(r) => r.act(b),
        }
    }
}

/// Build a POMDP plan with the chosen method. `horizon` applies to the
/// lookahead and exact methods (ignored otherwise).
pub fn solve_pomdp(
    spec: &PomdpSpec,
    method: PomdpMethod,
    horizon: usize,
) -> Result<PomdpPlan, String> {
    spec.validate()?;
    let make = || spec.to_pomdp_spec();
    match method {
        PomdpMethod::Qmdp => Ok(PomdpPlan::Qmdp(QMDPSolver::new(
            make(),
            &MDPVIOptions::default(),
        ))),
        PomdpMethod::MostLikelyState => {
            Ok(PomdpPlan::MostLikely(MostLikelyStateSolver::new(make())))
        }
        PomdpMethod::Lookahead => {
            let opts = BeliefLookaheadOptions {
                horizon: horizon.max(1),
                ..BeliefLookaheadOptions::default()
            };
            Ok(PomdpPlan::Lookahead(BeliefLookaheadSolver::new(
                make(),
                opts,
            )))
        }
        PomdpMethod::ExactFiniteHorizon => {
            let closure = make();
            let h = horizon.max(1);
            let result = catch_unwind(AssertUnwindSafe(|| {
                pomdp_exact_finite_horizon(&closure, h)
            }))
            .map_err(|_| {
                "exact finite-horizon VI blew up (reduce horizon or use qmdp)".to_string()
            })?;
            Ok(PomdpPlan::Exact(result))
        }
    }
}

/// The underlying-MDP value function / policy of a POMDP, for the results
/// payload (treats the hidden state as observable).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PomdpSolution {
    pub underlying_value: Vec<f64>,
    pub underlying_policy: Vec<usize>,
    pub q: Vec<Vec<f64>>,
    pub discount: f64,
}

/// Solve the underlying MDP of a POMDP (for inspection / results).
pub fn solve_pomdp_underlying(spec: &PomdpSpec) -> Result<PomdpSolution, String> {
    spec.validate()?;
    let closure = spec.to_pomdp_spec();
    let r = mdp_value_iteration(&closure, &MDPVIOptions::default());
    Ok(PomdpSolution {
        underlying_value: r.v,
        underlying_policy: r.policy,
        q: r.q,
        discount: spec.discount,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::spec::{MdpTransition, TerminalState};

    fn chain_mdp() -> MdpSpec {
        // s0 --go--> s1 (terminal, reward 1), discount 0.9.
        MdpSpec {
            schema: "des/mdp/v1".into(),
            num_states: 2,
            transitions: vec![
                vec![vec![MdpTransition { prob: 1.0, reward: 0.0, next: 1 }]],
                vec![],
            ],
            discount: 0.9,
            terminal: vec![TerminalState { state: 1, reward: 1.0 }],
            state_labels: vec![],
            action_labels: vec![],
        }
    }

    #[test]
    fn solve_mdp_matches_closed_form() {
        let sol = solve_mdp(&chain_mdp(), MdpMethod::ValueIteration).unwrap();
        assert!((sol.value[1] - 1.0).abs() < 1e-6, "V[1]={}", sol.value[1]);
        assert!((sol.value[0] - 0.9).abs() < 1e-6, "V[0]={}", sol.value[0]);
        assert_eq!(sol.policy[0], 0);
        assert_eq!(sol.policy[1], -1);
        assert!((sol.q[0][0] - 0.9).abs() < 1e-6);
    }
}
