//! Port of `src/des/general/des-base/policy-gradient-agent.ts` —
//! template-method bases for policy-gradient methods (REINFORCE / A2C / PPO /
//! TRPO …) plus the paired policy-update station.
//!
//! The agent samples `a ~ π_θ(·|s)`, records `(s, a, log π_old, V_φ(s))` in a
//! rollout buffer, and after `rollout_len` env steps it PAUSES, fires a
//! [`TrainTriggerToken`], and waits for a [`ResumeToken`]. A separate
//! [`PolicyUpdateStation`] consumes the trigger, mutates the shared parameters,
//! clears the buffer, and resumes the agent.
//!
//! ## Rust shape
//!
//!   * `interface RolloutEntry<S,A>` → [`RolloutEntry`] (`r` / `done` / `s_next`
//!     are `Option`, populated by the matching transition).
//!   * `abstract class PolicyGradientAgent` → the [`PolicyGradientAgent`] trait
//!     (template-method `runTimeStep` → provided
//!     [`PolicyGradientAgent::policy_gradient_run_time_step`]; the abstract hook
//!     `samplePolicyAndValue` → the required
//!     [`PolicyGradientAgent::sample_policy_and_value`]). Per-agent state lives
//!     in [`PolicyGradientCore`].
//!   * `abstract class PolicyUpdateStation` → the [`PolicyUpdateStation`] trait
//!     (`runUpdate` → required [`PolicyUpdateStation::run_update`]; `runTimeStep`
//!     → provided template). Counter state lives in [`PolicyUpdateCore`].
//!   * `rng: () => number` → an injected `Option<Box<dyn RandomSource>>` stored
//!     in the core and threaded into the hook as `&mut dyn RandomSource` (moved
//!     out transiently while the hook runs), matching the sibling `RLAgentCore`.
//!   * `pendingActionState: {state; episodeId} | null` → `Option<PendingAction>`.
//!   * The buffer search `e.s === t.state && e.a === t.action` (value equality)
//!     → bounds `S: PartialEq, A: PartialEq` (no JS reference identity).
//!   * Shared agent↔update parameters (TS shared object handle) → the concrete
//!     subclasses wire an `Rc<RefCell<…>>`; the base only defines the contract.

use std::rc::Rc;

use crate::des::shared::capabilities::RandomSource;

use super::episode_accounting::EpisodeAccounting;
use super::rl_tokens::{ActionToken, ResumeToken, StateToken, TrainTriggerToken, TransitionToken};
use super::station::DESStation;

/// Inbox: state tokens (start of each episode).
pub const CH_STATE: &str = "state";
/// Inbox: transition tokens (after every env step).
pub const CH_TRANSITION: &str = "transition";
/// Outbox: action tokens (to the environment).
pub const CH_ACTION: &str = "action";
/// Outbox: train-trigger tokens (to the update station).
pub const CH_TRAIN: &str = "train";
/// Inbox: resume tokens (from the update station).
pub const CH_RESUME: &str = "resume";

/// A single `(s, a, logp_old, v, r, done, sNext)` tuple in the buffer.
#[derive(Clone, Debug)]
pub struct RolloutEntry<S, A> {
    pub s: S,
    pub a: A,
    pub log_prob_old: f64,
    pub v: f64,
    pub r: Option<f64>,
    pub done: Option<bool>,
    pub s_next: Option<S>,
}

/// Output of the policy hook: a sampled action plus its log-prob and value.
#[derive(Clone, Copy, Debug)]
pub struct PolicyOutput<A = usize> {
    pub action: A,
    pub log_prob: f64,
    pub value: f64,
}

/// State owed an action after the next resume (TS `pendingActionState`).
#[derive(Clone, Debug)]
pub struct PendingAction<S> {
    pub state: S,
    pub episode_id: f64,
}

/// Per-agent state (the non-shared fields of the TS abstract class).
pub struct PolicyGradientCore<S, A> {
    /// Injected randomness (TS `rng: () => number`). Moved out transiently while
    /// the policy hook runs (see [`PolicyGradientAgent::sample_policy_and_value`]).
    pub rng: Option<Box<dyn RandomSource>>,
    pub rollout_len: usize,
    /// True while waiting for a [`ResumeToken`] after triggering an update.
    pub paused: bool,
    pub buffer: Vec<RolloutEntry<S, A>>,
    pending_action_state: Option<PendingAction<S>>,
    pub num_updates: u64,
    pub episode_accounting: EpisodeAccounting,
}

impl<S, A> PolicyGradientCore<S, A> {
    pub fn new(rollout_len: usize, rng: Box<dyn RandomSource>) -> Self {
        PolicyGradientCore {
            rng: Some(rng),
            rollout_len,
            paused: false,
            buffer: Vec::new(),
            pending_action_state: None,
            num_updates: 0,
            episode_accounting: EpisodeAccounting::new(),
        }
    }
}

/// Template-method base for policy-gradient agents.
pub trait PolicyGradientAgent<S: 'static = f64, A: 'static = usize>: DESStation {
    /// Borrow per-agent state.
    fn pg_core(&self) -> &PolicyGradientCore<S, A>;
    /// Mutably borrow per-agent state.
    fn pg_core_mut(&mut self) -> &mut PolicyGradientCore<S, A>;

    // ── HOOK (abstract) ───────────────────────────────────────────────────────

    /// Sample `a ~ π_θ(·|s)` and report `(a, log π(a|s), V_φ(s))`. The injected
    /// RNG is threaded in (matching the TS `samplePolicyAndValue(state, rng)`).
    fn sample_policy_and_value(&self, state: &S, rng: &mut dyn RandomSource) -> PolicyOutput<A>;

    // ── TEMPLATE METHOD (final) ────────────────────────────────────────────────

    fn policy_gradient_run_time_step(&mut self)
    where
        S: Clone + PartialEq + 'static,
        A: Clone + PartialEq + 'static,
    {
        if self.pg_core().paused {
            let resumes = self.core_mut().drain::<ResumeToken>(CH_RESUME);
            if resumes.is_empty() {
                return;
            }
            self.pg_core_mut().paused = false;
            // Resume the rollout: act on the state we owe.
            if self.pg_core().pending_action_state.is_some() {
                let ps = self.pg_core_mut().pending_action_state.take().unwrap();
                let mut rng = self.pg_core_mut().rng.take().expect("rng already in use");
                let out = self.sample_policy_and_value(&ps.state, &mut *rng);
                self.pg_core_mut().rng = Some(rng);
                self.pg_core_mut().buffer.push(RolloutEntry {
                    s: ps.state.clone(),
                    a: out.action.clone(),
                    log_prob_old: out.log_prob,
                    v: out.value,
                    r: None,
                    done: None,
                    s_next: None,
                });
                let tok = ActionToken::new(ps.state.clone(), out.action.clone(), ps.episode_id);
                self.core_mut().emit(Rc::new(tok), CH_ACTION);
            }
        }
        // 1. Process transitions: fill the buffer entry for (s, a) just emitted.
        let transitions = self
            .core_mut()
            .drain::<TransitionToken<S, A>>(CH_TRANSITION);
        for t in transitions {
            // Find the most-recent un-completed entry; this should be the last.
            {
                let buffer = &mut self.pg_core_mut().buffer;
                for i in (0..buffer.len()).rev() {
                    if buffer[i].r.is_none() && buffer[i].s == t.state && buffer[i].a == t.action {
                        buffer[i].r = Some(t.reward);
                        buffer[i].done = Some(t.done);
                        buffer[i].s_next = Some(t.next_state.clone());
                        break;
                    }
                }
            }
            self.pg_core_mut().episode_accounting.record_step(t.reward);
            if t.done {
                self.pg_core_mut().episode_accounting.finish_episode();
            }
            // If buffer is full → trigger train phase and pause.
            if self.pg_core().buffer.len() >= self.pg_core().rollout_len {
                self.pg_core_mut().paused = true;
                // If the just-completed transition was NON-terminal we owe an
                // action on s' once the update station finishes — stash it.
                if !t.done {
                    self.pg_core_mut().pending_action_state = Some(PendingAction {
                        state: t.next_state.clone(),
                        episode_id: t.episode_id,
                    });
                }
                self.core_mut().emit(Rc::new(TrainTriggerToken), CH_TRAIN);
                return;
            }
            // If !done, sample next action on s'. If done, env will emit a new
            // StateToken — handled below in this same tick.
            if !t.done {
                let mut rng = self.pg_core_mut().rng.take().expect("rng already in use");
                let out = self.sample_policy_and_value(&t.next_state, &mut *rng);
                self.pg_core_mut().rng = Some(rng);
                self.pg_core_mut().buffer.push(RolloutEntry {
                    s: t.next_state.clone(),
                    a: out.action.clone(),
                    log_prob_old: out.log_prob,
                    v: out.value,
                    r: None,
                    done: None,
                    s_next: None,
                });
                let tok = ActionToken::new(t.next_state.clone(), out.action.clone(), t.episode_id);
                self.core_mut().emit(Rc::new(tok), CH_ACTION);
            }
        }
        // 2. Process new-episode states.
        let states = self.core_mut().drain::<StateToken<S>>(CH_STATE);
        for s in states {
            let mut rng = self.pg_core_mut().rng.take().expect("rng already in use");
            let out = self.sample_policy_and_value(&s.state, &mut *rng);
            self.pg_core_mut().rng = Some(rng);
            self.pg_core_mut().buffer.push(RolloutEntry {
                s: s.state.clone(),
                a: out.action.clone(),
                log_prob_old: out.log_prob,
                v: out.value,
                r: None,
                done: None,
                s_next: None,
            });
            let tok = ActionToken::new(s.state.clone(), out.action.clone(), s.episode_id);
            self.core_mut().emit(Rc::new(tok), CH_ACTION);
        }
    }

    /// `hasWork` override.
    fn policy_gradient_has_work(&self) -> bool {
        if self.pg_core().paused {
            return self.core().inbox_size(CH_RESUME) > 0;
        }
        self.core().inbox_size(CH_STATE) > 0 || self.core().inbox_size(CH_TRANSITION) > 0
    }

    // ── INTROSPECTION HELPERS USED BY UPDATE STATIONS ──────────────────────────

    fn get_buffer(&self) -> &[RolloutEntry<S, A>] {
        &self.pg_core().buffer
    }
    fn clear_buffer(&mut self) {
        self.pg_core_mut().buffer = Vec::new();
    }
    fn is_paused(&self) -> bool {
        self.pg_core().paused
    }
    fn num_queued_train(&self) -> usize {
        self.core().inbox_size(CH_RESUME)
    }

    // ── ACCESSORS (TS getter/setter pairs + episodeAccounting aliases) ─────────

    fn reward_history(&self) -> &[f64] {
        &self.pg_core().episode_accounting.reward_history
    }
    fn length_history(&self) -> &[f64] {
        &self.pg_core().episode_accounting.length_history
    }
    fn total_steps(&self) -> u64 {
        self.pg_core().episode_accounting.total_steps
    }
    fn set_total_steps(&mut self, value: u64) {
        self.pg_core_mut().episode_accounting.total_steps = value;
    }
    fn episode_reward(&self) -> f64 {
        self.pg_core().episode_accounting.current_reward
    }
    fn set_episode_reward(&mut self, value: f64) {
        self.pg_core_mut().episode_accounting.current_reward = value;
    }
}

// -----------------------------------------------------------------------------
// PolicyUpdateStation — counterpart to PolicyGradientAgent. Listens for
// CH_TRAIN, runs an update on the agent's parameters, emits CH_RESUME.
// -----------------------------------------------------------------------------

/// Counter state for the update station (TS `numUpdates`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyUpdateCore {
    pub num_updates: u64,
}

impl PolicyUpdateCore {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Template-method base for the policy-update station.
pub trait PolicyUpdateStation: DESStation {
    /// Borrow counter state.
    fn pu_core(&self) -> &PolicyUpdateCore;
    /// Mutably borrow counter state.
    fn pu_core_mut(&mut self) -> &mut PolicyUpdateCore;

    // ── HOOK (abstract) ───────────────────────────────────────────────────────

    /// Mutate params using the rollout buffer of the attached
    /// [`PolicyGradientAgent`] (held by the concrete subclass).
    fn run_update(&mut self);

    // ── TEMPLATE METHOD (final) ────────────────────────────────────────────────

    fn policy_update_run_time_step(&mut self) {
        let triggers = self.core_mut().drain::<TrainTriggerToken>(CH_TRAIN);
        for _ in triggers {
            self.run_update();
            self.pu_core_mut().num_updates += 1;
            self.core_mut().emit(Rc::new(ResumeToken), CH_RESUME);
        }
    }

    /// `hasWork` override.
    fn policy_update_has_work(&self) -> bool {
        self.core().inbox_size(CH_TRAIN) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::super::rl_tokens::{
        ActionToken, ResumeToken, StateToken, TrainTriggerToken, TransitionToken,
    };
    use super::super::station::{DESStation, StationCore, StationRef};
    use super::*;
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// 2-elem softmax over `theta`; returns `(probs, sampled_action)`.
    fn softmax2(theta: &[f64]) -> [f64; 2] {
        let m = theta[0].max(theta[1]);
        let e0 = (theta[0] - m).exp();
        let e1 = (theta[1] - m).exp();
        let z = e0 + e1;
        [e0 / z, e1 / z]
    }

    /// Softmax-policy agent sharing `theta` with its update station.
    struct BanditAgent {
        core: StationCore,
        pg: PolicyGradientCore<f64, usize>,
        theta: Rc<RefCell<Vec<f64>>>,
    }

    impl BanditAgent {
        fn new(rollout_len: usize, seed: u32, theta: Rc<RefCell<Vec<f64>>>) -> Self {
            BanditAgent {
                core: StationCore::new("agent"),
                pg: PolicyGradientCore::new(rollout_len, Box::new(SeededRandom::new(seed))),
                theta,
            }
        }
    }

    impl DESStation for BanditAgent {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.policy_gradient_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.policy_gradient_has_work()
        }
    }

    impl PolicyGradientAgent<f64, usize> for BanditAgent {
        fn pg_core(&self) -> &PolicyGradientCore<f64, usize> {
            &self.pg
        }
        fn pg_core_mut(&mut self) -> &mut PolicyGradientCore<f64, usize> {
            &mut self.pg
        }
        fn sample_policy_and_value(
            &self,
            _state: &f64,
            rng: &mut dyn RandomSource,
        ) -> PolicyOutput<usize> {
            let probs = softmax2(&self.theta.borrow());
            let u = rng.next_float();
            let action = if u < probs[0] { 0 } else { 1 };
            PolicyOutput {
                action,
                log_prob: probs[action].ln(),
                value: 0.0,
            }
        }
    }

    /// REINFORCE update on the shared `theta` (no baseline, no critic).
    struct BanditUpdate {
        core: StationCore,
        pu: PolicyUpdateCore,
        agent: Rc<RefCell<BanditAgent>>,
        theta: Rc<RefCell<Vec<f64>>>,
        alpha: f64,
    }

    impl BanditUpdate {
        fn new(agent: Rc<RefCell<BanditAgent>>, theta: Rc<RefCell<Vec<f64>>>, alpha: f64) -> Self {
            BanditUpdate {
                core: StationCore::new("update"),
                pu: PolicyUpdateCore::new(),
                agent,
                theta,
                alpha,
            }
        }
    }

    impl DESStation for BanditUpdate {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.policy_update_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.policy_update_has_work()
        }
    }

    impl PolicyUpdateStation for BanditUpdate {
        fn pu_core(&self) -> &PolicyUpdateCore {
            &self.pu
        }
        fn pu_core_mut(&mut self) -> &mut PolicyUpdateCore {
            &mut self.pu
        }
        fn run_update(&mut self) {
            let mut agent = self.agent.borrow_mut();
            let mut theta = self.theta.borrow_mut();
            for e in agent.get_buffer() {
                let Some(g) = e.r else { continue };
                let probs = softmax2(&theta);
                for j in 0..2 {
                    let indicator = if j == e.a { 1.0 } else { 0.0 };
                    theta[j] += self.alpha * g * (indicator - probs[j]);
                }
            }
            agent.clear_buffer();
        }
    }

    /// Captures actions emitted by the agent so the test driver (an inline
    /// 2-action bandit) can score them and feed transitions back.
    #[derive(Default)]
    struct ActionCollector {
        core: StationCore,
        actions: Vec<Rc<ActionToken<f64, usize>>>,
    }

    impl DESStation for ActionCollector {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            let mut drained = self.core.drain::<ActionToken<f64, usize>>("in");
            self.actions.append(&mut drained);
        }
    }

    #[test]
    fn bandit_agent_improves_action_one() {
        // Inline 2-action bandit: single state 0.0, every step terminal,
        // action 1 pays 1.0, action 0 pays 0.0.
        let theta = Rc::new(RefCell::new(vec![0.0, 0.0]));
        let agent = Rc::new(RefCell::new(BanditAgent::new(8, 12345, theta.clone())));
        let update = Rc::new(RefCell::new(BanditUpdate::new(
            agent.clone(),
            theta.clone(),
            0.1,
        )));
        let actions: Rc<RefCell<ActionCollector>> =
            Rc::new(RefCell::new(ActionCollector::default()));

        // Wire agent↔update; route the agent's actions to the collector.
        agent
            .borrow_mut()
            .core_mut()
            .pipe(update.clone() as StationRef, CH_TRAIN, CH_TRAIN);
        update
            .borrow_mut()
            .core_mut()
            .pipe(agent.clone() as StationRef, CH_RESUME, CH_RESUME);
        agent
            .borrow_mut()
            .core_mut()
            .pipe(actions.clone() as StationRef, CH_ACTION, "in");

        // Seed the first episode start.
        let mut episode = 0.0_f64;
        agent
            .borrow_mut()
            .core_mut()
            .take(Rc::new(StateToken::new(0.0_f64, episode)), CH_STATE);

        for _ in 0..8000 {
            if update.borrow().has_work() {
                update.borrow_mut().run_time_step();
            }
            if agent.borrow().has_work() {
                agent.borrow_mut().run_time_step();
            }
            actions.borrow_mut().run_time_step();
            let emitted: Vec<_> = actions.borrow_mut().actions.drain(..).collect();
            for a in emitted {
                let reward = if a.action == 1 { 1.0 } else { 0.0 };
                // Terminal transition, then reset → next-episode StateToken.
                agent.borrow_mut().core_mut().take(
                    Rc::new(TransitionToken::new(
                        a.state,
                        a.action,
                        reward,
                        0.0,
                        true,
                        a.episode_id,
                    )),
                    CH_TRANSITION,
                );
                episode += 1.0;
                agent
                    .borrow_mut()
                    .core_mut()
                    .take(Rc::new(StateToken::new(0.0_f64, episode)), CH_STATE);
            }
        }

        assert!(update.borrow().pu_core().num_updates > 0);
        let final_theta = theta.borrow();
        assert!(final_theta[1] > final_theta[0], "theta = {final_theta:?}");
        let probs = softmax2(&final_theta);
        assert!(probs[1] > 0.8, "P(action 1) = {}", probs[1]);
    }

    /// Deterministic agent (always action 0) to verify the pause/train trigger.
    struct FixedAgent {
        core: StationCore,
        pg: PolicyGradientCore<f64, usize>,
    }

    impl DESStation for FixedAgent {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.policy_gradient_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.policy_gradient_has_work()
        }
    }

    impl PolicyGradientAgent<f64, usize> for FixedAgent {
        fn pg_core(&self) -> &PolicyGradientCore<f64, usize> {
            &self.pg
        }
        fn pg_core_mut(&mut self) -> &mut PolicyGradientCore<f64, usize> {
            &mut self.pg
        }
        fn sample_policy_and_value(
            &self,
            _state: &f64,
            _rng: &mut dyn RandomSource,
        ) -> PolicyOutput<usize> {
            PolicyOutput {
                action: 0,
                log_prob: (0.5_f64).ln(),
                value: 0.0,
            }
        }
    }

    #[test]
    fn agent_pauses_and_triggers_train_when_buffer_full() {
        let mut agent = FixedAgent {
            core: StationCore::new("fixed"),
            pg: PolicyGradientCore::new(3, Box::new(SeededRandom::new(1))),
        };
        let trains: Rc<RefCell<TrainCollector>> = Rc::new(RefCell::new(TrainCollector::default()));
        agent
            .core_mut()
            .pipe(trains.clone() as StationRef, CH_TRAIN, "in");

        // Seed one episode start, then feed three non-terminal transitions.
        agent
            .core_mut()
            .take(Rc::new(StateToken::new(0.0_f64, 0.0)), CH_STATE);
        agent.run_time_step();
        for _ in 0..3 {
            agent.core_mut().take(
                Rc::new(TransitionToken::new(0.0_f64, 0usize, 0.5, 0.0, false, 0.0)),
                CH_TRANSITION,
            );
        }
        agent.run_time_step();
        trains.borrow_mut().run_time_step();

        assert!(agent.is_paused());
        assert_eq!(agent.get_buffer().len(), 3);
        assert_eq!(trains.borrow().count, 1);
    }

    #[derive(Default)]
    struct TrainCollector {
        core: StationCore,
        count: usize,
    }

    impl DESStation for TrainCollector {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.count += self.core.drain::<TrainTriggerToken>("in").len();
        }
    }

    /// No-op update that just records that it fired.
    struct CountingUpdate {
        core: StationCore,
        pu: PolicyUpdateCore,
        ran: usize,
    }

    impl DESStation for CountingUpdate {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.policy_update_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.policy_update_has_work()
        }
    }

    impl PolicyUpdateStation for CountingUpdate {
        fn pu_core(&self) -> &PolicyUpdateCore {
            &self.pu
        }
        fn pu_core_mut(&mut self) -> &mut PolicyUpdateCore {
            &mut self.pu
        }
        fn run_update(&mut self) {
            self.ran += 1;
        }
    }

    #[test]
    fn update_station_runs_hook_and_emits_resume() {
        let mut update = CountingUpdate {
            core: StationCore::new("upd"),
            pu: PolicyUpdateCore::new(),
            ran: 0,
        };
        let resumes: Rc<RefCell<ResumeCollector>> =
            Rc::new(RefCell::new(ResumeCollector::default()));
        update
            .core_mut()
            .pipe(resumes.clone() as StationRef, CH_RESUME, "in");

        assert!(!update.has_work());
        update.core_mut().take(Rc::new(TrainTriggerToken), CH_TRAIN);
        assert!(update.has_work());
        update.run_time_step();
        resumes.borrow_mut().run_time_step();

        assert_eq!(update.ran, 1);
        assert_eq!(update.pu_core().num_updates, 1);
        assert_eq!(resumes.borrow().count, 1);
    }

    #[derive(Default)]
    struct ResumeCollector {
        core: StationCore,
        count: usize,
    }

    impl DESStation for ResumeCollector {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn run_time_step(&mut self) {
            self.count += self.core.drain::<ResumeToken>("in").len();
        }
    }
}
