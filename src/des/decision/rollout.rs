//! Unified rollout: simulate a policy/plan in an MDP or POMDP and record one
//! [`EpisodeTrace`] — the trajectory that drives both visualization and
//! analysis. (Previously each domain model copy-pasted its own rollout loop.)

use serde::Serialize;

use crate::des::general::belief::DiscreteBelief;
use crate::des::general::pomdp::belief_update;

use super::solve::PomdpPlan;
use super::spec::{MdpSpec, PomdpSpec};

/// Tiny deterministic xorshift64* RNG yielding `f64` in `[0, 1)`. Seeded for
/// reproducible rollouts.
pub struct Prng {
    state: u64,
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        // Avoid the zero fixed point.
        Prng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    pub fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Sample an index from a probability distribution using a uniform draw `u`.
fn sample_index(dist: &[f64], u: f64) -> usize {
    let mut acc = 0.0;
    for (i, &p) in dist.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    dist.len().saturating_sub(1)
}

/// A recorded trajectory. `states` has one more entry than `actions` (it
/// includes the start state). `observations`/`beliefs` are populated for POMDP
/// rollouts only.
#[derive(Clone, Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeTrace {
    pub states: Vec<usize>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f64>,
    /// Cumulative (undiscounted) reward after each step.
    pub returns: Vec<f64>,
    /// Total discounted return `Σ γ^t r_t`.
    pub discounted_return: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<usize>,
    /// Belief vector at each step (POMDP only); `beliefs[t]` is the belief
    /// *before* taking `actions[t]`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub beliefs: Vec<Vec<f64>>,
}

/// Roll out a (greedy) MDP policy from `start`, sampling stochastic transitions.
/// Stops early at a terminal/no-action state.
pub fn rollout_mdp(
    spec: &MdpSpec,
    policy: &[i32],
    start: usize,
    steps: usize,
    seed: u64,
) -> EpisodeTrace {
    let mut rng = Prng::new(seed);
    let mut trace = EpisodeTrace::default();
    let mut s = start;
    let mut cum = 0.0;
    let mut disc = 0.0;
    let gamma = spec.discount;
    trace.states.push(s);
    for t in 0..steps {
        if spec.is_terminal(s) {
            break;
        }
        let a = policy.get(s).copied().unwrap_or(-1);
        if a < 0 {
            break;
        }
        let outcomes = match spec.transitions.get(s).and_then(|av| av.get(a as usize)) {
            Some(o) if !o.is_empty() => o,
            _ => break,
        };
        let u = rng.next_f64();
        let mut acc = 0.0;
        let mut chosen = &outcomes[outcomes.len() - 1];
        for o in outcomes {
            acc += o.prob;
            if u <= acc {
                chosen = o;
                break;
            }
        }
        cum += chosen.reward;
        disc += gamma.powi(t as i32) * chosen.reward;
        trace.actions.push(a as usize);
        trace.rewards.push(chosen.reward);
        trace.returns.push(cum);
        s = chosen.next;
        trace.states.push(s);
    }
    trace.discounted_return = disc;
    trace
}

/// Roll out a POMDP plan: sample the hidden state, act on the belief, sample the
/// next state + observation, update the belief, and record everything.
pub fn rollout_pomdp(
    spec: &PomdpSpec,
    plan: &mut PomdpPlan,
    start_state: Option<usize>,
    steps: usize,
    seed: u64,
) -> EpisodeTrace {
    let mut rng = Prng::new(seed);
    let closure = spec.to_pomdp_spec();
    let init = spec.initial_belief_vec();
    let mut belief = DiscreteBelief::new((0..spec.num_states).collect(), Some(&init));

    let mut s = start_state.unwrap_or_else(|| sample_index(&init, rng.next_f64()));
    let mut trace = EpisodeTrace::default();
    let mut cum = 0.0;
    let mut disc = 0.0;
    let gamma = spec.discount;
    trace.states.push(s);
    trace.beliefs.push(belief.weights.clone());

    for t in 0..steps {
        let a = plan.act(&belief);
        let reward = spec.reward[s][a];
        let s_next = sample_index(&spec.transition[s][a], rng.next_f64());
        let o = sample_index(&spec.observation[s_next][a], rng.next_f64());

        cum += reward;
        disc += gamma.powi(t as i32) * reward;
        belief = belief_update(&closure, &belief, a, o);

        trace.actions.push(a);
        trace.rewards.push(reward);
        trace.returns.push(cum);
        trace.observations.push(o);
        trace.states.push(s_next);
        trace.beliefs.push(belief.weights.clone());
        s = s_next;
    }
    trace.discounted_return = disc;
    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::spec::{MdpTransition, TerminalState};

    #[test]
    fn mdp_rollout_reaches_terminal_and_collects_reward() {
        // s0 --go(reward 0)--> s1 (terminal). Policy: take action 0 in s0.
        let spec = MdpSpec {
            schema: "des/mdp/v1".into(),
            num_states: 2,
            transitions: vec![
                vec![vec![MdpTransition {
                    prob: 1.0,
                    reward: 2.0,
                    next: 1,
                }]],
                vec![],
            ],
            discount: 0.9,
            terminal: vec![TerminalState {
                state: 1,
                reward: 0.0,
            }],
            state_labels: vec![],
            action_labels: vec![],
        };
        let trace = rollout_mdp(&spec, &[0, -1], 0, 10, 42);
        assert_eq!(trace.states, vec![0, 1]);
        assert_eq!(trace.actions, vec![0]);
        assert!((trace.rewards[0] - 2.0).abs() < 1e-12);
        assert!((trace.discounted_return - 2.0).abs() < 1e-12);
    }
}
