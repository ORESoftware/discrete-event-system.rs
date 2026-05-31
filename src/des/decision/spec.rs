//! Canonical, JSON-first MDP / POMDP specifications.
//!
//! These are the declarative contracts an LLM or UI targets. They are tabular
//! (dense transition/observation/reward arrays) and `serde`-(de)serializable, so
//! a spec round-trips to JSON and validates with messages designed for a caller
//! to self-correct against. Each spec [`bridges`](MdpSpec::to_value_iteration_spec)
//! to the crate's existing closure-based solver types, so all solver logic is
//! reused verbatim — these specs add a canonical surface, not a new solver.

use std::collections::HashMap;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::des::general::pomdp::POMDPSpec;
use crate::des::general::value_iteration::{MDPSpec, Outcome};

pub const MDP_SCHEMA: &str = "des/mdp/v1";
pub const POMDP_SCHEMA: &str = "des/pomdp/v1";

fn default_discount() -> f64 {
    0.95
}
fn mdp_schema() -> String {
    MDP_SCHEMA.to_string()
}
fn pomdp_schema() -> String {
    POMDP_SCHEMA.to_string()
}

/// One probabilistic transition outcome of taking an action in a state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MdpTransition {
    /// Probability of this outcome, in `[0, 1]`.
    pub prob: f64,
    /// Immediate reward on this transition.
    #[serde(default)]
    pub reward: f64,
    /// Index of the next state.
    pub next: usize,
}

/// A terminal (absorbing) state with a pinned value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TerminalState {
    pub state: usize,
    #[serde(default)]
    pub reward: f64,
}

/// Canonical tabular Markov Decision Process.
///
/// `transitions[state][action]` is the list of [`MdpTransition`] outcomes; an
/// empty action list means "no legal action" (e.g. terminal states).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MdpSpec {
    #[serde(rename = "$schema", default = "mdp_schema")]
    pub schema: String,
    pub num_states: usize,
    /// `transitions[state][action] -> outcomes`.
    pub transitions: Vec<Vec<Vec<MdpTransition>>>,
    #[serde(default = "default_discount")]
    pub discount: f64,
    #[serde(default)]
    pub terminal: Vec<TerminalState>,
    #[serde(default)]
    pub state_labels: Vec<String>,
    #[serde(default)]
    pub action_labels: Vec<String>,
}

impl MdpSpec {
    /// Validate dimensions, probabilities, indices, and the discount factor.
    /// Errors are phrased for an LLM/user to repair.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_states == 0 {
            return Err("numStates must be > 0".to_string());
        }
        if self.transitions.len() != self.num_states {
            return Err(format!(
                "transitions has {} state rows but numStates is {}",
                self.transitions.len(),
                self.num_states
            ));
        }
        if !(self.discount > 0.0 && self.discount <= 1.0) {
            return Err(format!("discount must be in (0, 1]; got {}", self.discount));
        }
        for (s, actions) in self.transitions.iter().enumerate() {
            for (a, outcomes) in actions.iter().enumerate() {
                if outcomes.is_empty() {
                    continue; // a legal "no-op"/illegal action
                }
                let mut sum = 0.0;
                for o in outcomes {
                    if o.prob < 0.0 {
                        return Err(format!("transitions[{s}][{a}] has a negative probability"));
                    }
                    if o.next >= self.num_states {
                        return Err(format!(
                            "transitions[{s}][{a}] outcome `next`={} is out of range (numStates={})",
                            o.next, self.num_states
                        ));
                    }
                    sum += o.prob;
                }
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(format!(
                        "transitions[{s}][{a}] probabilities sum to {sum}, expected 1.0"
                    ));
                }
            }
        }
        for t in &self.terminal {
            if t.state >= self.num_states {
                return Err(format!("terminal state {} is out of range", t.state));
            }
        }
        Ok(())
    }

    pub fn state_label(&self, s: usize) -> String {
        self.state_labels
            .get(s)
            .cloned()
            .unwrap_or_else(|| format!("s{s}"))
    }

    pub fn action_label(&self, a: usize) -> String {
        self.action_labels
            .get(a)
            .cloned()
            .unwrap_or_else(|| format!("a{a}"))
    }

    /// Largest action count across states (for layout / dense tables).
    pub fn max_actions(&self) -> usize {
        self.transitions
            .iter()
            .map(|av| av.len())
            .max()
            .unwrap_or(0)
    }

    /// Bridge to the closure-based [`MDPSpec`] consumed by `value_iteration`.
    pub fn to_value_iteration_spec(&self) -> MDPSpec {
        let transitions = Rc::new(self.transitions.clone());
        let terminal: Rc<HashMap<usize, f64>> =
            Rc::new(self.terminal.iter().map(|t| (t.state, t.reward)).collect());
        let labels = Rc::new(self.state_labels.clone());
        let alabels = Rc::new(self.action_labels.clone());
        MDPSpec {
            num_states: self.num_states,
            num_actions: {
                let t = transitions.clone();
                Box::new(move |s: usize| t.get(s).map(|av| av.len()).unwrap_or(0))
            },
            outcomes: {
                let t = transitions.clone();
                Box::new(move |s: usize, a: usize| {
                    t.get(s)
                        .and_then(|av| av.get(a))
                        .map(|outs| {
                            outs.iter()
                                .map(|o| Outcome {
                                    prob: o.prob,
                                    reward: o.reward,
                                    next_state: o.next,
                                })
                                .collect::<Vec<Outcome>>()
                        })
                        .unwrap_or_default()
                })
            },
            is_terminal: {
                let m = terminal.clone();
                Some(Box::new(move |s: usize| m.contains_key(&s)))
            },
            terminal_reward: {
                let m = terminal.clone();
                Some(Box::new(move |s: usize| m.get(&s).copied().unwrap_or(0.0)))
            },
            state_label: if labels.is_empty() {
                None
            } else {
                let l = labels.clone();
                Some(Box::new(move |s: usize| {
                    l.get(s).cloned().unwrap_or_else(|| format!("s{s}"))
                }))
            },
            action_label: if alabels.is_empty() {
                None
            } else {
                let l = alabels.clone();
                Some(Box::new(move |a: usize| {
                    l.get(a).cloned().unwrap_or_else(|| format!("a{a}"))
                }))
            },
        }
    }

    pub fn is_terminal(&self, s: usize) -> bool {
        self.terminal.iter().any(|t| t.state == s)
    }
}

/// Canonical tabular Partially Observable Markov Decision Process
/// ⟨S, A, Ω, T, O, R, γ⟩.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PomdpSpec {
    #[serde(rename = "$schema", default = "pomdp_schema")]
    pub schema: String,
    pub num_states: usize,
    pub num_actions: usize,
    pub num_observations: usize,
    /// `transition[state][action]` = distribution over next states (len = numStates).
    pub transition: Vec<Vec<Vec<f64>>>,
    /// `observation[nextState][action]` = distribution over observations (len = numObservations).
    pub observation: Vec<Vec<Vec<f64>>>,
    /// `reward[state][action]`.
    pub reward: Vec<Vec<f64>>,
    #[serde(default = "default_discount")]
    pub discount: f64,
    /// Optional initial belief (len = numStates); defaults to uniform.
    #[serde(default)]
    pub initial_belief: Option<Vec<f64>>,
    #[serde(default)]
    pub state_labels: Vec<String>,
    #[serde(default)]
    pub action_labels: Vec<String>,
    #[serde(default)]
    pub observation_labels: Vec<String>,
}

impl PomdpSpec {
    pub fn validate(&self) -> Result<(), String> {
        let (ns, na, no) = (self.num_states, self.num_actions, self.num_observations);
        if ns == 0 || na == 0 || no == 0 {
            return Err("numStates, numActions and numObservations must all be > 0".to_string());
        }
        if !(self.discount > 0.0 && self.discount <= 1.0) {
            return Err(format!("discount must be in (0, 1]; got {}", self.discount));
        }
        check_dist3(&self.transition, ns, na, ns, "transition")?;
        check_dist3(&self.observation, ns, na, no, "observation")?;
        if self.reward.len() != ns {
            return Err(format!(
                "reward has {} rows, expected numStates={ns}",
                self.reward.len()
            ));
        }
        for (s, row) in self.reward.iter().enumerate() {
            if row.len() != na {
                return Err(format!(
                    "reward[{s}] has {} entries, expected numActions={na}",
                    row.len()
                ));
            }
        }
        if let Some(b) = &self.initial_belief {
            if b.len() != ns {
                return Err(format!(
                    "initialBelief has {} entries, expected numStates={ns}",
                    b.len()
                ));
            }
            let sum: f64 = b.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("initialBelief sums to {sum}, expected 1.0"));
            }
        }
        Ok(())
    }

    pub fn state_label(&self, s: usize) -> String {
        self.state_labels
            .get(s)
            .cloned()
            .unwrap_or_else(|| format!("s{s}"))
    }
    pub fn action_label(&self, a: usize) -> String {
        self.action_labels
            .get(a)
            .cloned()
            .unwrap_or_else(|| format!("a{a}"))
    }
    pub fn observation_label(&self, o: usize) -> String {
        self.observation_labels
            .get(o)
            .cloned()
            .unwrap_or_else(|| format!("o{o}"))
    }

    /// The initial belief vector (uniform if unspecified).
    pub fn initial_belief_vec(&self) -> Vec<f64> {
        self.initial_belief
            .clone()
            .unwrap_or_else(|| vec![1.0 / self.num_states as f64; self.num_states])
    }

    /// Bridge to the closure-based [`POMDPSpec`] consumed by the POMDP solvers.
    pub fn to_pomdp_spec(&self) -> POMDPSpec<usize, usize, usize> {
        let trans = Rc::new(self.transition.clone());
        let obs = Rc::new(self.observation.clone());
        let rew = Rc::new(self.reward.clone());
        POMDPSpec {
            states: (0..self.num_states).collect(),
            actions: (0..self.num_actions).collect(),
            observations: (0..self.num_observations).collect(),
            transition: {
                let t = trans.clone();
                Box::new(move |s: usize, a: usize| t[s][a].clone())
            },
            observation: {
                let o = obs.clone();
                Box::new(move |sp: usize, a: usize| o[sp][a].clone())
            },
            reward: {
                let r = rew.clone();
                Box::new(move |s: usize, a: usize| r[s][a])
            },
            discount: self.discount,
            initial_belief: self.initial_belief.clone(),
            is_terminal: None,
        }
    }
}

/// Validate a `[d0][d1] -> Vec<f64> of len d2` tensor of probability rows.
fn check_dist3(
    t: &[Vec<Vec<f64>>],
    d0: usize,
    d1: usize,
    d2: usize,
    name: &str,
) -> Result<(), String> {
    if t.len() != d0 {
        return Err(format!("{name} has {} rows, expected {d0}", t.len()));
    }
    for (i, row) in t.iter().enumerate() {
        if row.len() != d1 {
            return Err(format!(
                "{name}[{i}] has {} action entries, expected {d1}",
                row.len()
            ));
        }
        for (j, dist) in row.iter().enumerate() {
            if dist.len() != d2 {
                return Err(format!(
                    "{name}[{i}][{j}] has {} entries, expected {d2}",
                    dist.len()
                ));
            }
            let sum: f64 = dist.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(format!("{name}[{i}][{j}] sums to {sum}, expected 1.0"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdp_spec_round_trips_through_json() {
        let spec = MdpSpec {
            schema: mdp_schema(),
            num_states: 2,
            transitions: vec![
                vec![vec![MdpTransition {
                    prob: 1.0,
                    reward: 0.0,
                    next: 1,
                }]],
                vec![],
            ],
            discount: 0.9,
            terminal: vec![TerminalState {
                state: 1,
                reward: 1.0,
            }],
            state_labels: vec!["start".into(), "goal".into()],
            action_labels: vec!["go".into()],
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: MdpSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_states, 2);
        assert!(back.validate().is_ok());
        assert!(json.contains("\"$schema\":\"des/mdp/v1\""));
    }

    #[test]
    fn mdp_validation_catches_bad_probabilities() {
        let spec = MdpSpec {
            schema: mdp_schema(),
            num_states: 1,
            transitions: vec![vec![vec![MdpTransition {
                prob: 0.5,
                reward: 0.0,
                next: 0,
            }]]],
            discount: 0.9,
            terminal: vec![],
            state_labels: vec![],
            action_labels: vec![],
        };
        let err = spec.validate().unwrap_err();
        assert!(err.contains("sum to 0.5"), "{err}");
    }

    #[test]
    fn pomdp_validation_catches_dimension_mismatch() {
        let spec = PomdpSpec {
            schema: pomdp_schema(),
            num_states: 2,
            num_actions: 1,
            num_observations: 2,
            transition: vec![vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]],
            observation: vec![vec![vec![1.0]], vec![vec![1.0]]], // wrong obs length
            reward: vec![vec![0.0], vec![0.0]],
            discount: 0.9,
            initial_belief: None,
            state_labels: vec![],
            action_labels: vec![],
            observation_labels: vec![],
        };
        let err = spec.validate().unwrap_err();
        assert!(err.contains("observation"), "{err}");
    }
}
