//! Port of `src/des/general/des-base/actor-critic.ts`.
//!
//! One-step tabular ACTOR-CRITIC agent (Sutton & Barto §13.5). The agent
//! simultaneously learns:
//!
//!   * a state-value function `V_w(s)`        ("CRITIC", `w ∈ ℝ`)
//!   * a parameterised policy `π_θ(a|s)`      ("ACTOR",  `θ ∈ ℝ`)
//!
//! On every transition `(s, a, r, s', done)` it computes a TD error
//!
//! ```text
//!     δ = r + γ V_w(s') − V_w(s)            (0 if done)
//! ```
//!
//! and applies the canonical updates
//!
//! ```text
//!     w  ← w + α_w δ ∇_w V_w(s)
//!     θ  ← θ + α_θ δ ∇_θ log π_θ(a|s)
//! ```
//!
//! ## TS → Rust mapping
//!
//! TypeScript modelled this as a CONCRETE `class TabularActorCritic extends
//! RLAgentStation<number, number>`. As per `rl_agent.rs`, a concrete agent
//! EMBEDS a [`StationCore`] and an [`RLAgentCore`], implements [`DESStation`]
//! (delegating `run_time_step` → `rl_agent_run_time_step`), and implements the
//! [`RLAgentStation`] hooks (`pick_action`, `update`, `end_of_episode`).
//!
//!   * `interface ActorCriticOptions` → [`ActorCriticOptions`] (`#[derive(Default)]`;
//!     the injected `rng` is passed to the constructor rather than living in the
//!     options struct).
//!   * `V: Float64Array` / `logits: Float64Array` (flat `N×A`) → `Vec<f64>`
//!     indexed `s*A + a`.
//!   * `pi()` returns a fresh softmax `Float64Array` → returns `Vec<f64>`.
//!   * `rng: () => number` → injected [`RandomSource`]; categorical sampling in
//!     `pick_action` uses the threaded `&mut dyn RandomSource`, and
//!     [`TabularActorCritic::greedy_action`] reuses [`arg_max_with_tie_break`].
//!   * non-ASCII `δ` → `delta`; `Math.exp` → `f64::exp`; `number` → `f64`,
//!     state/action indices → `usize`.

use std::any::Any;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation, RngRef};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::shared::capabilities::RandomSource;

/// Construction options for [`TabularActorCritic`]. `#[derive(Default)]`
/// supplies `None`/`0` for every field; the required `num_states` / `num_actions`
/// must be set explicitly, and the injected RNG is a separate constructor
/// argument (a boxed `dyn RandomSource` cannot be `Default`).
#[derive(Default)]
pub struct ActorCriticOptions {
    pub num_states: usize,
    pub num_actions: usize,
    /// Critic learning rate `α_v`. Default 0.1.
    pub alpha_v: Option<f64>,
    /// Actor learning rate `α_p`. Default 0.05.
    pub alpha_p: Option<f64>,
    /// Discount `γ`. Default 0.95.
    pub gamma: Option<f64>,
    /// Initial logits value (broadcast). Default 0.
    pub init_logits: Option<f64>,
    /// Initial value estimate (broadcast). Default 0.
    pub init_v: Option<f64>,
    /// Optional entropy coefficient `β`; adds `β · H(π(·|s))` to the actor
    /// gradient. Default 0 (disabled).
    pub entropy_coef: Option<f64>,
}

/// Tabular one-step actor-critic over finite (state, action) spaces.
///
/// Implements the [`RLAgentStation`] hooks `pick_action` (categorical sampling
/// from the softmax policy), `update` (TD-error critic + policy-gradient actor),
/// and `end_of_episode` (mean-|δ| logging).
pub struct TabularActorCritic {
    core: StationCore,
    agent: RLAgentCore,
    /// Number of states (configuration; tables are sized from it at construction).
    #[allow(dead_code)]
    n: usize,
    a: usize,
    /// Critic table `V[s]`.
    v: Vec<f64>,
    /// Actor logits, flat `N × A` indexed `s*A + a`.
    logits: Vec<f64>,
    alpha_v: f64,
    alpha_p: f64,
    gamma: f64,
    entropy_coef: f64,
    /// Per-episode mean `|δ|` (TD error) for diagnostics.
    pub td_error_history: Vec<f64>,
    ep_abs_td: f64,
    ep_updates: f64,
}

impl TabularActorCritic {
    /// Mirrors `new TabularActorCritic(id, opts)`: the injected `rng` becomes the
    /// boxed `dyn RandomSource` stored in [`RLAgentCore`].
    pub fn new(
        id: impl Into<String>,
        rng: Box<dyn RandomSource>,
        opts: ActorCriticOptions,
    ) -> Self {
        let n = opts.num_states;
        let a = opts.num_actions;
        let mut v = vec![0.0; n];
        let init_v = opts.init_v.unwrap_or(0.0);
        if init_v != 0.0 {
            v.iter_mut().for_each(|x| *x = init_v);
        }
        let mut logits = vec![0.0; n * a];
        let init_logits = opts.init_logits.unwrap_or(0.0);
        if init_logits != 0.0 {
            logits.iter_mut().for_each(|x| *x = init_logits);
        }
        TabularActorCritic {
            core: StationCore::new(id),
            agent: RLAgentCore::new(rng),
            n,
            a,
            v,
            logits,
            alpha_v: opts.alpha_v.unwrap_or(0.1),
            alpha_p: opts.alpha_p.unwrap_or(0.05),
            gamma: opts.gamma.unwrap_or(0.95),
            entropy_coef: opts.entropy_coef.unwrap_or(0.0),
            td_error_history: Vec::new(),
            ep_abs_td: 0.0,
            ep_updates: 0.0,
        }
    }

    /// `π(·|s)` — softmax over `logits[s]`.
    pub fn pi(&self, state: usize) -> Vec<f64> {
        let off = state * self.a;
        let mut mx = f64::NEG_INFINITY;
        for a in 0..self.a {
            if self.logits[off + a] > mx {
                mx = self.logits[off + a];
            }
        }
        let mut buf = vec![0.0; self.a];
        let mut z = 0.0;
        for a in 0..self.a {
            buf[a] = (self.logits[off + a] - mx).exp();
            z += buf[a];
        }
        for a in 0..self.a {
            buf[a] /= z;
        }
        buf
    }

    /// Argmax over the policy distribution at state `s` (uniform tie-break).
    ///
    /// Takes `&mut self` because the tie-break consumes the injected RNG, which
    /// is moved transiently out of the embedded [`RLAgentCore`] (same trick as
    /// the template method in `rl_agent.rs`).
    pub fn greedy_action(&mut self, state: usize) -> usize {
        let probs = self.pi(state);
        let mut rng = self.agent.rng.take().expect("rng already in use");
        let a =
            arg_max_with_tie_break(&probs, &mut RngRef(&mut *rng), ARGMAX_EPS_DEFAULT).unwrap_or(0);
        self.agent.rng = Some(rng);
        a
    }

    // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

    pub fn get_v(&self) -> &[f64] {
        &self.v
    }
    pub fn get_logits(&self) -> &[f64] {
        &self.logits
    }
    pub fn get_policy_prob(&self, state: usize, action: usize) -> f64 {
        self.pi(state)[action]
    }
}

impl DESStation for TabularActorCritic {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.rl_agent_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.rl_agent_has_work()
    }
}

impl RLAgentStation<usize, usize> for TabularActorCritic {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }

    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        let probs = self.pi(*state);
        let u = rng.next_float();
        let mut acc = 0.0;
        for a in 0..self.a {
            acc += probs[a];
            if u < acc {
                return a;
            }
        }
        self.a - 1
    }

    fn update(
        &mut self,
        state: &usize,
        action: &usize,
        reward: f64,
        next_state: &usize,
        done: bool,
    ) {
        let delta = reward
            + (if done {
                0.0
            } else {
                self.gamma * self.v[*next_state]
            })
            - self.v[*state];
        // Critic step (tabular ∇V(s) = e_s).
        self.v[*state] += self.alpha_v * delta;
        // Actor step (∇log π(a|s) = e_a − π(·|s)).
        let probs = self.pi(*state);
        let off = *state * self.a;
        for b in 0..self.a {
            let grad = (if b == *action { 1.0 } else { 0.0 }) - probs[b];
            self.logits[off + b] += self.alpha_p * delta * grad;
            if self.entropy_coef != 0.0 && probs[b] > 0.0 {
                // Entropy bonus pushing the distribution toward uniform.
                self.logits[off + b] += self.entropy_coef * (probs[b] - 1.0 / self.a as f64);
            }
        }
        self.ep_abs_td += delta.abs();
        self.ep_updates += 1.0;
    }

    fn end_of_episode(&mut self, _episode_id: f64) {
        if self.ep_updates > 0.0 {
            self.td_error_history.push(self.ep_abs_td / self.ep_updates);
        }
        self.ep_abs_td = 0.0;
        self.ep_updates = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::rl_tokens::TransitionToken;
    use crate::des::shared::capabilities::SeededRandom;
    use std::rc::Rc;

    /// Contextual bandit: 2 states × 2 actions, action 1 yields +1 and action 0
    /// yields 0; every step is terminal. Optimal policy picks action 1.
    fn reward_for(action: usize) -> f64 {
        if action == 1 {
            1.0
        } else {
            0.0
        }
    }

    fn agent(seed: u32) -> TabularActorCritic {
        TabularActorCritic::new(
            "ac",
            Box::new(SeededRandom::new(seed)),
            ActorCriticOptions {
                num_states: 2,
                num_actions: 2,
                alpha_v: Some(0.1),
                alpha_p: Some(0.2),
                gamma: Some(0.9),
                ..Default::default()
            },
        )
    }

    #[test]
    fn actor_critic_improves_on_two_state_task() {
        let mut a = agent(1);
        for ep in 0..500 {
            for state in 0..2usize {
                for action in 0..2usize {
                    let t = TransitionToken::new(
                        state,
                        action,
                        reward_for(action),
                        state,
                        true,
                        ep as f64,
                    );
                    a.core_mut()
                        .take(Rc::new(t), TabularActorCritic::CH_TRANSITION);
                    a.run_time_step();
                }
            }
        }
        // The actor must prefer the rewarding action in both states.
        for state in 0..2usize {
            assert!(
                a.get_policy_prob(state, 1) > a.get_policy_prob(state, 0),
                "state {state}: π = [{}, {}]",
                a.get_policy_prob(state, 0),
                a.get_policy_prob(state, 1)
            );
            assert!(a.get_policy_prob(state, 1) > 0.5);
            assert_eq!(a.greedy_action(state), 1);
        }
    }

    #[test]
    fn critic_value_tracks_reward_and_history_logged() {
        let mut a = agent(2);
        // Consistent per-state terminal target so the critic can actually fit it
        // (state 0 -> 1.0, state 1 -> 0.0); then V[s] converges and |δ| shrinks.
        let target = |state: usize| if state == 0 { 1.0 } else { 0.0 };
        for ep in 0..500 {
            for state in 0..2usize {
                let t = TransitionToken::new(state, 0usize, target(state), state, true, ep as f64);
                a.core_mut()
                    .take(Rc::new(t), TabularActorCritic::CH_TRANSITION);
                a.run_time_step();
            }
        }
        // V(s) converges toward its consistent terminal target.
        for state in 0..2usize {
            assert!(
                (a.get_v()[state] - target(state)).abs() < 0.1,
                "V[{state}] = {}",
                a.get_v()[state]
            );
        }
        assert_eq!(a.td_error_history.len(), 500 * 2);
        // Mean |δ| shrinks over training as V approaches the target.
        let h = &a.td_error_history;
        let window = 40;
        let mean_first: f64 = h[..window].iter().sum::<f64>() / window as f64;
        let mean_last: f64 = h[h.len() - window..].iter().sum::<f64>() / window as f64;
        assert!(
            mean_last < mean_first,
            "mean TD error should shrink: first {mean_first}, last {mean_last}"
        );
    }

    #[test]
    fn pi_is_a_normalised_distribution() {
        let a = agent(3);
        let probs = a.pi(0);
        assert_eq!(probs.len(), 2);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // Zero-initialised logits → uniform policy.
        assert!((probs[0] - 0.5).abs() < 1e-12);
    }
}
