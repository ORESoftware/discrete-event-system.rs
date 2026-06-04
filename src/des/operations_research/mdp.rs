//! Markov decision process helpers beyond the first-class JSON runner.
//!
//! The crate already exposes discounted value iteration through
//! [`crate::des::decision`]. This module adds course-useful tabular kernels:
//! finite-horizon backward induction, policy evaluation, and Q-table
//! construction over the canonical [`MdpSpec`].

use crate::des::decision::spec::MdpSpec;

const EPS: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct PolicyEvaluationOptions {
    pub discount: f64,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for PolicyEvaluationOptions {
    fn default() -> Self {
        PolicyEvaluationOptions {
            discount: 0.95,
            tol: 1e-9,
            max_iter: 5000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEvaluationResult {
    pub value: Vec<f64>,
    pub iterations: usize,
    pub final_delta: f64,
    pub converged: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteHorizonMdpResult {
    /// `value_by_stage[t][s]` is the optimal value with `horizon - t` decisions
    /// remaining. The final row is the terminal value vector.
    pub value_by_stage: Vec<Vec<f64>>,
    /// `policy_by_stage[t][s]` is the greedy action at stage `t`; `None` means
    /// terminal/no legal action.
    pub policy_by_stage: Vec<Vec<Option<usize>>>,
}

/// Compute `Q(s,a)` for a supplied value function.
pub fn q_values(spec: &MdpSpec, values: &[f64], discount: f64) -> Result<Vec<Vec<f64>>, String> {
    spec.validate()?;
    if values.len() != spec.num_states {
        return Err(format!(
            "values length {} != numStates {}",
            values.len(),
            spec.num_states
        ));
    }
    if !(0.0..=1.0).contains(&discount) {
        return Err(format!("discount must be in [0, 1]; got {discount}"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("values must all be finite".to_string());
    }

    let mut q = vec![Vec::new(); spec.num_states];
    for (s, actions) in spec.transitions.iter().enumerate() {
        let mut row = Vec::with_capacity(actions.len());
        for outcomes in actions {
            let mut qv = 0.0;
            for o in outcomes {
                qv += o.prob * (o.reward + discount * values[o.next]);
            }
            row.push(qv);
        }
        q[s] = row;
    }
    Ok(q)
}

/// Evaluate a stationary policy by fixed-point iteration.
pub fn evaluate_policy(
    spec: &MdpSpec,
    policy: &[Option<usize>],
    options: PolicyEvaluationOptions,
) -> Result<PolicyEvaluationResult, String> {
    spec.validate()?;
    if policy.len() != spec.num_states {
        return Err(format!(
            "policy length {} != numStates {}",
            policy.len(),
            spec.num_states
        ));
    }
    if !(0.0..=1.0).contains(&options.discount) {
        return Err(format!(
            "discount must be in [0, 1]; got {}",
            options.discount
        ));
    }
    if options.tol < 0.0 || !options.tol.is_finite() {
        return Err(format!(
            "tol must be non-negative and finite; got {}",
            options.tol
        ));
    }
    if options.max_iter == 0 {
        return Err("max_iter must be positive".to_string());
    }
    validate_stationary_policy(spec, policy)?;

    let mut value = terminal_values(spec);
    let mut final_delta = f64::INFINITY;
    let mut iterations = 0usize;
    for iter in 0..options.max_iter {
        let mut next = value.clone();
        let mut delta = 0.0_f64;
        for s in 0..spec.num_states {
            if spec.is_terminal(s) {
                continue;
            }
            let a = policy[s].expect("validated stationary policy has an action here");
            let outcomes = spec.transitions[s]
                .get(a)
                .ok_or_else(|| format!("policy[{s}]={a} is not a legal action"))?;
            let mut v = 0.0;
            for o in outcomes {
                v += o.prob * (o.reward + options.discount * value[o.next]);
            }
            delta = delta.max((v - value[s]).abs());
            next[s] = v;
        }
        value = next;
        iterations = iter + 1;
        final_delta = delta;
        if delta <= options.tol {
            break;
        }
    }

    Ok(PolicyEvaluationResult {
        value,
        iterations,
        final_delta,
        converged: final_delta <= options.tol,
    })
}

/// Finite-horizon dynamic programming by backward induction.
pub fn finite_horizon_backward_induction(
    spec: &MdpSpec,
    horizon: usize,
    terminal_value: Option<Vec<f64>>,
) -> Result<FiniteHorizonMdpResult, String> {
    spec.validate()?;
    let mut value_by_stage = vec![vec![0.0; spec.num_states]; horizon + 1];
    value_by_stage[horizon] = match terminal_value {
        Some(v) => {
            if v.len() != spec.num_states {
                return Err(format!(
                    "terminal_value length {} != numStates {}",
                    v.len(),
                    spec.num_states
                ));
            }
            v
        }
        None => terminal_values(spec),
    };
    if value_by_stage[horizon].iter().any(|v| !v.is_finite()) {
        return Err("terminal_value must contain only finite values".to_string());
    }
    let mut policy_by_stage = vec![vec![None; spec.num_states]; horizon];

    for t in (0..horizon).rev() {
        for s in 0..spec.num_states {
            if spec.is_terminal(s) || spec.transitions[s].is_empty() {
                value_by_stage[t][s] = value_by_stage[t + 1][s];
                continue;
            }
            let mut best_value = f64::NEG_INFINITY;
            let mut best_action = None;
            for (a, outcomes) in spec.transitions[s].iter().enumerate() {
                if outcomes.is_empty() {
                    continue;
                }
                let mut q = 0.0;
                for o in outcomes {
                    q += o.prob * (o.reward + spec.discount * value_by_stage[t + 1][o.next]);
                }
                if q > best_value + EPS {
                    best_value = q;
                    best_action = Some(a);
                }
            }
            if let Some(a) = best_action {
                value_by_stage[t][s] = best_value;
                policy_by_stage[t][s] = Some(a);
            } else {
                value_by_stage[t][s] = value_by_stage[t + 1][s];
            }
        }
    }

    Ok(FiniteHorizonMdpResult {
        value_by_stage,
        policy_by_stage,
    })
}

fn terminal_values(spec: &MdpSpec) -> Vec<f64> {
    let mut values = vec![0.0; spec.num_states];
    for terminal in &spec.terminal {
        values[terminal.state] = terminal.reward;
    }
    values
}

fn validate_stationary_policy(spec: &MdpSpec, policy: &[Option<usize>]) -> Result<(), String> {
    for (s, action) in policy.iter().enumerate() {
        if spec.is_terminal(s) {
            if action.is_some() {
                return Err(format!("policy[{s}] must be None for terminal states"));
            }
            continue;
        }
        let has_legal_action = spec.transitions[s]
            .iter()
            .any(|outcomes| !outcomes.is_empty());
        match (has_legal_action, action) {
            (true, Some(a)) => {
                let outcomes = spec.transitions[s]
                    .get(*a)
                    .ok_or_else(|| format!("policy[{s}]={a} is not a legal action"))?;
                if outcomes.is_empty() {
                    return Err(format!("policy[{s}]={a} points to an empty action"));
                }
            }
            (true, None) => {
                return Err(format!(
                    "policy[{s}] must choose an action for a non-terminal state"
                ));
            }
            (false, Some(_)) => {
                return Err(format!(
                    "policy[{s}] must be None because the state has no legal actions"
                ));
            }
            (false, None) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::spec::{MdpTransition, TerminalState};

    fn chain() -> MdpSpec {
        MdpSpec {
            schema: "des/mdp/v1".to_string(),
            num_states: 3,
            transitions: vec![
                vec![
                    vec![MdpTransition {
                        prob: 1.0,
                        reward: 0.0,
                        next: 1,
                    }],
                    vec![MdpTransition {
                        prob: 1.0,
                        reward: 0.0,
                        next: 2,
                    }],
                ],
                vec![vec![MdpTransition {
                    prob: 1.0,
                    reward: 5.0,
                    next: 2,
                }]],
                vec![],
            ],
            discount: 0.9,
            terminal: vec![TerminalState {
                state: 2,
                reward: 1.0,
            }],
            state_labels: Vec::new(),
            action_labels: Vec::new(),
        }
    }

    #[test]
    fn finite_horizon_prefers_waiting_when_horizon_allows() {
        let result = finite_horizon_backward_induction(&chain(), 2, None).unwrap();
        assert_eq!(result.policy_by_stage[0][0], Some(0));
        assert!((result.value_by_stage[0][0] - 5.31).abs() < 1e-10);
    }

    #[test]
    fn policy_evaluation_scores_stationary_policy() {
        let result = evaluate_policy(
            &chain(),
            &[Some(0), Some(0), None],
            PolicyEvaluationOptions {
                discount: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert!((result.value[0] - 5.31).abs() < 1e-7);
    }

    #[test]
    fn policy_evaluation_rejects_incomplete_policy() {
        let err = evaluate_policy(
            &chain(),
            &[None, Some(0), None],
            PolicyEvaluationOptions {
                discount: 0.9,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("policy[0] must choose an action"));
    }

    #[test]
    fn q_values_reject_non_finite_values() {
        let err = q_values(&chain(), &[0.0, f64::NAN, 1.0], 0.9).unwrap_err();
        assert!(err.contains("finite"));
    }
}
