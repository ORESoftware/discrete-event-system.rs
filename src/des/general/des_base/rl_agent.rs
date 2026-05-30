//! Port of `src/des/general/des-base/rl-agent.ts`.
//!
//! Template-method base for ONLINE TEMPORAL-DIFFERENCE agents: Q-learning,
//! SARSA, expected-SARSA, Double-Q, Q(λ), … An agent:
//!
//!   * receives a [`StateToken`] at the start of each episode,
//!   * receives a [`TransitionToken`] `(s, a, r, s', done)` after every step,
//!   * emits an [`ActionToken`] `(s, a)` whenever it needs to act,
//!   * applies an UPDATE rule on each transition.
//!
//! The DIFFERENTIATOR among agents is the UPDATE rule (the `update` hook):
//!
//!   * Q-learning:     `Q[s,a] ← Q[s,a] + α(r + γ max_a' Q[s',a'] − Q[s,a])`
//!   * SARSA:          `Q[s,a] ← Q[s,a] + α(r + γ Q[s',a'] − Q[s,a])`
//!   * Expected SARSA: `Q[s,a] ← Q[s,a] + α(r + γ Σ_a' π(a'|s') Q[s',a'] − Q[s,a])`
//!
//! ## Template-method mapping (TS `abstract class` → Rust)
//!
//! TypeScript modelled this as `abstract class RLAgentStation<S, A> extends
//! DESStation` whose `runTimeStep` is a FINAL template method calling abstract
//! hooks (`pickAction`, `update`) plus an optional hook (`endOfEpisode`).
//! Concrete agents subclass and override the hooks. Rust has no
//! abstract-method inheritance, so we split the class:
//!
//!   * [`RLAgentCore`] — a plain struct holding the bookkeeping the TS base
//!     owned (the embedded [`EpisodeAccounting`] and the injected RNG). A
//!     concrete agent EMBEDS one and exposes it via `agent_core()` /
//!     `agent_core_mut()`.
//!   * [`RLAgentStation`] — the hook trait (`: DESStation`). REQUIRED methods
//!     are the abstract hooks (`pick_action`, `update`); the optional hook
//!     (`end_of_episode`) has a default impl. The template method itself is the
//!     PROVIDED method [`RLAgentStation::rl_agent_run_time_step`] (plus the
//!     accessors and `rl_agent_has_work`), which must NOT be overridden. A
//!     concrete agent delegates `DESStation::run_time_step` →
//!     `self.rl_agent_run_time_step()` and `DESStation::has_work` →
//!     `self.rl_agent_has_work()`.
//!
//! The injected `rng: () => number` becomes a boxed
//! [`RandomSource`](crate::des::shared::capabilities::RandomSource) stored in
//! [`RLAgentCore`]; the template threads it into `pick_action` as `&mut dyn
//! RandomSource` (moved out of the core transiently so `&mut self` and the RNG
//! do not alias — same trick as `single_state_optimizer`).
//!
//! Getter/setter pairs (`get totalSteps` / `set totalSteps`, `episodeReward`,
//! `episodeLength`) become plain methods proxying the inner
//! [`EpisodeAccounting`]. The TS `rewardHistory` / `lengthHistory` aliases of
//! the inner `Vec`s (Rust can't own a shared alias) become borrowing accessors.
//! `number` → `f64`, action index → `usize`. Generic defaults `S = number` /
//! `A = number` map to `S = f64` / `A = usize` per the migration rules.

use std::rc::Rc;

use crate::des::general::des_base::episode_accounting::EpisodeAccounting;
use crate::des::general::des_base::rl_tokens::{ActionToken, StateToken, TransitionToken};
use crate::des::general::des_base::station::{AnyToken, DESStation};
use crate::des::shared::capabilities::RandomSource;

/// Bookkeeping fields the TS `abstract class` owned, factored into a struct the
/// concrete agent embeds. Holds the episode accounting and the injected RNG
/// (moved out transiently while a hook runs). Non-generic: it carries no `S`/`A`
/// state, only reward/length bookkeeping and randomness.
pub struct RLAgentCore {
    /// Reward/length bookkeeping across episodes.
    pub episode_accounting: EpisodeAccounting,
    /// RNG handed to `pick_action` (moved out transiently during a step).
    pub rng: Option<Box<dyn RandomSource>>,
}

impl RLAgentCore {
    /// Mirrors `new RLAgentStation(id, {rng})`: stores the injected RNG and a
    /// fresh [`EpisodeAccounting`].
    pub fn new(rng: Box<dyn RandomSource>) -> Self {
        RLAgentCore {
            episode_accounting: EpisodeAccounting::new(),
            rng: Some(rng),
        }
    }
}

/// Adapter so the `&mut dyn RandomSource` threaded into `pick_action` can be
/// passed to the generic `argmax` helpers (which want `&mut impl
/// RandomSource`). `dyn RandomSource` does not itself implement `RandomSource`,
/// so this concrete sized newtype bridges the gap by delegating through dynamic
/// dispatch. Concrete agents wrap their `rng` argument: e.g.
/// `arg_max_with_tie_break(&scores, &mut RngRef(rng), eps)`.
pub struct RngRef<'a>(pub &'a mut dyn RandomSource);

impl RandomSource for RngRef<'_> {
    fn next_float(&mut self) -> f64 {
        self.0.next_float()
    }
}

/// The online-TD agent hook trait. REQUIRED methods are the TS abstract hooks;
/// the optional hook has a default impl. The PROVIDED methods
/// (`rl_agent_run_time_step`, `rl_agent_has_work`, accessors) make up the
/// template method and must NOT be overridden by concrete agents.
///
/// Generic over the state type `S` and action type `A`; both must be `'static`
/// (so the tokens qualify as `Rc<dyn Any>`) and `Clone` (the template clones a
/// state into an outgoing [`ActionToken`]).
pub trait RLAgentStation<S: Clone + 'static = f64, A: Clone + 'static = usize>: DESStation {
    /// Channel carrying a [`StateToken`] at episode start.
    const CH_STATE: &'static str = "state";
    /// Channel carrying a [`TransitionToken`] after each env step.
    const CH_TRANSITION: &'static str = "transition";
    /// Channel onto which [`ActionToken`]s are emitted.
    const CH_ACTION: &'static str = "action";

    /// Borrow the embedded RL bookkeeping state.
    fn agent_core(&self) -> &RLAgentCore;
    /// Mutably borrow the embedded RL bookkeeping state.
    fn agent_core_mut(&mut self) -> &mut RLAgentCore;

    // ── HOOKS (required) ───────────────────────────────────────────────────────

    /// Pick an action in `state`. Typically ε-greedy; subclasses encode policy.
    fn pick_action(&self, state: &S, rng: &mut dyn RandomSource) -> A;
    /// Apply a TD update from a transition.
    fn update(&mut self, state: &S, action: &A, reward: f64, next_state: &S, done: bool);

    // ── HOOKS (optional) ───────────────────────────────────────────────────────

    /// Called when an episode ends, with the just-finished episode's id. Decay
    /// ε here, log, etc. Default: no-op.
    fn end_of_episode(&mut self, _episode_id: f64) {}

    // ── TEMPLATE METHOD (do NOT override) ──────────────────────────────────────

    /// Single tick. Concrete agents delegate `DESStation::run_time_step` here.
    ///
    /// 1. Apply transitions (`update`, then act on `s'` if not done).
    /// 2. Process [`StateToken`]s (start of an episode) by acting.
    fn rl_agent_run_time_step(&mut self) {
        // 1. Apply transitions (TD update, then act on s' if not done).
        let transitions = self
            .core_mut()
            .drain::<TransitionToken<S, A>>(Self::CH_TRANSITION);
        for t in transitions {
            self.update(&t.state, &t.action, t.reward, &t.next_state, t.done);
            self.agent_core_mut()
                .episode_accounting
                .record_step(t.reward);
            if t.done {
                self.agent_core_mut().episode_accounting.finish_episode();
                self.end_of_episode(t.episode_id);
                // Note: do NOT emit on done — wait for the env's next
                // StateToken (it will arrive on the same tick or the next).
            } else {
                let mut rng = self
                    .agent_core_mut()
                    .rng
                    .take()
                    .expect("rng already in use");
                let a = self.pick_action(&t.next_state, &mut *rng);
                self.agent_core_mut().rng = Some(rng);
                let token: AnyToken = Rc::new(ActionToken::<S, A>::new(
                    t.next_state.clone(),
                    a,
                    t.episode_id,
                ));
                self.core_mut().emit(token, Self::CH_ACTION);
            }
        }
        // 2. Process StateTokens (start of an episode).
        let states = self.core_mut().drain::<StateToken<S>>(Self::CH_STATE);
        for s in states {
            let mut rng = self
                .agent_core_mut()
                .rng
                .take()
                .expect("rng already in use");
            let a = self.pick_action(&s.state, &mut *rng);
            self.agent_core_mut().rng = Some(rng);
            let token: AnyToken =
                Rc::new(ActionToken::<S, A>::new(s.state.clone(), a, s.episode_id));
            self.core_mut().emit(token, Self::CH_ACTION);
        }
    }

    /// `hasWork` override: any pending transition or state token is work.
    fn rl_agent_has_work(&self) -> bool {
        self.core().inbox_size(Self::CH_TRANSITION) > 0
            || self.core().inbox_size(Self::CH_STATE) > 0
    }

    // ── ACCESSORS (proxy the embedded EpisodeAccounting) ───────────────────────

    fn total_steps(&self) -> u64 {
        self.agent_core().episode_accounting.total_steps
    }
    fn set_total_steps(&mut self, value: u64) {
        self.agent_core_mut().episode_accounting.total_steps = value;
    }
    fn episode_reward(&self) -> f64 {
        self.agent_core().episode_accounting.current_reward
    }
    fn set_episode_reward(&mut self, value: f64) {
        self.agent_core_mut().episode_accounting.current_reward = value;
    }
    fn episode_length(&self) -> f64 {
        self.agent_core().episode_accounting.current_length
    }
    fn set_episode_length(&mut self, value: f64) {
        self.agent_core_mut().episode_accounting.current_length = value;
    }
    /// Borrowing accessor for the TS `rewardHistory` alias.
    fn reward_history(&self) -> &[f64] {
        &self.agent_core().episode_accounting.reward_history
    }
    /// Borrowing accessor for the TS `lengthHistory` alias.
    fn length_history(&self) -> &[f64] {
        &self.agent_core().episode_accounting.length_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
    use crate::des::general::des_base::station::StationCore;
    use crate::des::shared::capabilities::SeededRandom;
    use std::any::Any;
    use std::cell::RefCell;

    /// Tiny tabular Q-learning agent over `2 states × 2 actions` implementing
    /// the [`RLAgentStation`] hooks. `update` is the Q-learning rule; the policy
    /// is ε-greedy with tie-broken argmax.
    struct QLearner {
        core: StationCore,
        agent: RLAgentCore,
        /// `q[state][action]`.
        q: Vec<Vec<f64>>,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
    }

    impl QLearner {
        fn new(seed: u32, alpha: f64, gamma: f64, epsilon: f64) -> Self {
            QLearner {
                core: StationCore::new("q"),
                agent: RLAgentCore::new(Box::new(SeededRandom::new(seed))),
                q: vec![vec![0.0; 2]; 2],
                alpha,
                gamma,
                epsilon,
            }
        }

        fn greedy_action(&self, state: usize) -> usize {
            // Deterministic greedy read (state 0/1) ignoring ε; first max wins.
            if self.q[state][1] > self.q[state][0] {
                1
            } else {
                0
            }
        }
    }

    impl DESStation for QLearner {
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

    impl RLAgentStation<usize, usize> for QLearner {
        fn agent_core(&self) -> &RLAgentCore {
            &self.agent
        }
        fn agent_core_mut(&mut self) -> &mut RLAgentCore {
            &mut self.agent
        }
        fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
            if rng.next_float() < self.epsilon {
                (rng.next_float() * 2.0).floor() as usize
            } else {
                arg_max_with_tie_break(&self.q[*state], &mut RngRef(rng), ARGMAX_EPS_DEFAULT)
                    .unwrap_or(0)
            }
        }
        fn update(
            &mut self,
            state: &usize,
            action: &usize,
            reward: f64,
            next_state: &usize,
            done: bool,
        ) {
            let best_next = if done {
                0.0
            } else {
                self.q[*next_state]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let target = reward + self.gamma * best_next;
            let q = &mut self.q[*state][*action];
            *q += self.alpha * (target - *q);
        }
    }

    /// The toy MDP: in either state, action 1 yields reward +1 and action 0
    /// yields 0; every step is terminal (a contextual bandit). Optimal policy
    /// picks action 1 in both states.
    fn reward_for(action: usize) -> f64 {
        if action == 1 {
            1.0
        } else {
            0.0
        }
    }

    #[test]
    fn q_learning_learns_best_action() {
        let mut agent = QLearner::new(1, 0.2, 0.9, 0.0);
        // Feed both actions in both states repeatedly via the template method.
        for ep in 0..200 {
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
                    agent.core_mut().take(Rc::new(t), QLearner::CH_TRANSITION);
                    agent.run_time_step();
                }
            }
        }
        // Q[s][1] (rewarding) should dominate Q[s][0], so greedy picks 1.
        for state in 0..2usize {
            assert!(
                agent.q[state][1] > agent.q[state][0],
                "state {state}: {:?}",
                agent.q[state]
            );
            assert_eq!(agent.greedy_action(state), 1);
        }
        // Bookkeeping: 200 episodes × 2 states × 2 actions terminal steps.
        assert_eq!(agent.total_steps(), 800);
        assert_eq!(agent.reward_history().len(), 800);
    }

    #[test]
    fn transition_emits_next_action_and_records_step() {
        // Pre-train so the greedy action in state 1 is action 1.
        let mut agent = QLearner::new(2, 0.5, 0.9, 0.0);
        agent.q = vec![vec![0.0, 0.0], vec![0.0, 5.0]];

        let sink = Rc::new(RefCell::new(ActionSink::new("sink")));
        agent
            .core_mut()
            .pipe(sink.clone(), QLearner::CH_ACTION, ActionSink::CH_IN);

        // A non-done transition: agent must act on next_state (=1) and record.
        let t = TransitionToken::new(0usize, 0usize, 0.3, 1usize, false, 0.0);
        agent.core_mut().take(Rc::new(t), QLearner::CH_TRANSITION);
        agent.run_time_step();

        assert_eq!(agent.total_steps(), 1);
        assert!((agent.episode_reward() - 0.3).abs() < 1e-12);
        assert!(
            agent.reward_history().is_empty(),
            "episode not finished yet"
        );

        // `emit` placed the ActionToken in the sink's inbox; run it to capture.
        sink.borrow_mut().run_time_step();
        let captured = sink.borrow().last.clone().expect("an action was emitted");
        assert_eq!(captured.state, 1);
        assert_eq!(captured.action, 1, "ε=0 greedy picks the high-value action");
    }

    #[test]
    fn state_token_starts_episode_with_action() {
        let mut agent = QLearner::new(3, 0.5, 0.9, 0.0);
        agent.q = vec![vec![2.0, 0.0], vec![0.0, 0.0]];

        let sink = Rc::new(RefCell::new(ActionSink::new("sink")));
        agent
            .core_mut()
            .pipe(sink.clone(), QLearner::CH_ACTION, ActionSink::CH_IN);

        let s = StateToken::new(0usize, 42.0);
        agent.core_mut().take(Rc::new(s), QLearner::CH_STATE);
        assert!(agent.has_work());
        agent.run_time_step();

        sink.borrow_mut().run_time_step();
        let captured = sink.borrow().last.clone().expect("an action was emitted");
        assert_eq!(captured.state, 0);
        assert_eq!(captured.action, 0, "ε=0 greedy picks the high-value action");
        assert_eq!(captured.episode_id, 42.0);
    }

    /// Minimal sink that records the latest [`ActionToken`] it receives.
    struct ActionSink {
        core: StationCore,
        last: Option<Rc<ActionToken<usize, usize>>>,
    }

    impl ActionSink {
        const CH_IN: &'static str = "action";
        fn new(id: &str) -> Self {
            ActionSink {
                core: StationCore::new(id),
                last: None,
            }
        }
    }

    impl DESStation for ActionSink {
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
            let toks = self.core.drain::<ActionToken<usize, usize>>(Self::CH_IN);
            if let Some(last) = toks.into_iter().last() {
                self.last = Some(last);
            }
        }
    }
}
