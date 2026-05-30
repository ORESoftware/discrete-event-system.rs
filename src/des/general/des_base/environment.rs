//! Port of `src/des/general/des-base/environment.ts` — generic Environment
//! Station for RL.
//!
//! Wraps a pure (non-DES) [`PureEnvironment`] in a station that:
//!   * emits a [`StateToken`] on its `state` channel at the start of every
//!     episode (and once at the very beginning to seed the loop)
//!   * accepts [`ActionToken`] on its `action` channel; calls `env.step`; emits
//!     a [`TransitionToken`] on its `transition` channel
//!   * optionally truncates episodes longer than `max_steps_per_episode`
//!   * stops emitting once `num_episodes` is reached OR an external observer
//!     flips `done = true` (used by PPO when the global step budget is reached)
//!
//! ## Rust shape (faithful translation of the TS class)
//!
//!   * `interface PureEnvironment<S,A>` → the [`PureEnvironment`] trait (the
//!     hook trait). `step`'s inline `{nextState, reward, done}` becomes the
//!     named [`StepResult`] struct. `render?` → a provided default returning
//!     `String`.
//!   * `interface EnvironmentStationOptions` → [`EnvironmentStationOptions`]
//!     (`#[derive(Default)]`, `Option` fields). `Infinity` default →
//!     `f64::INFINITY`.
//!   * `class EnvironmentStation<S,A> extends DESStation` → a concrete struct
//!     embedding [`StationCore`] + holding a `Box<dyn PureEnvironment>` and
//!     `impl DESStation`.
//!   * getter/setter pairs + the `rewardHistory` alias of `episodeAccounting`
//!     → plain accessor methods borrowing the inner [`EpisodeAccounting`].
//!   * `done` (externally-set flag) → `pub done: bool`.

use std::rc::Rc;

use super::episode_accounting::EpisodeAccounting;
use super::rl_tokens::{ActionToken, StateToken, TransitionToken};
use super::station::{DESStation, StationCore};

/// Action inbox channel.
pub const CH_ACTION: &str = "action";
/// State outbox channel (emitted only at episode start).
pub const CH_STATE: &str = "state";
/// Transition outbox channel (emitted after every `env.step`).
pub const CH_TRANSITION: &str = "transition";

/// Result of a single environment step (TS returned an inline object).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResult<S = f64> {
    pub next_state: S,
    pub reward: f64,
    pub done: bool,
}

/// A pure (non-DES) environment: the hook trait wrapped by the station.
///
/// `numStates` / `numActions` (TS fields) become accessor methods. `reset` and
/// `step` take `&mut self` so stateful environments are permitted (the TS
/// interface left mutability unconstrained).
pub trait PureEnvironment<S = f64, A = usize> {
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn reset(&mut self) -> S;
    fn step(&mut self, state: S, action: A) -> StepResult<S>;
    /// Optional human-readable render of a state. Default: empty string.
    fn render(&self, _state: &S) -> String {
        String::new()
    }
}

/// Optional configuration (TS `interface EnvironmentStationOptions`).
#[derive(Clone, Copy, Debug, Default)]
pub struct EnvironmentStationOptions {
    /// Maximum episodes to run. `None` → `Infinity` ("until external stop").
    pub num_episodes: Option<f64>,
    /// Truncate episodes longer than this many steps. `None` → `1_000_000`.
    pub max_steps_per_episode: Option<usize>,
}

/// Generic Environment Station for RL (the concrete TS `EnvironmentStation`).
pub struct EnvironmentStation<S = f64, A = usize> {
    core: StationCore,
    env: Box<dyn PureEnvironment<S, A>>,
    /// Resolved options (TS `Required<EnvironmentStationOptions>`).
    num_episodes: f64,
    max_steps_per_episode: usize,

    cur_state: S,
    episode_id: usize,
    step_in_episode: usize,
    emitted_start: bool,
    /// Externally settable termination flag — used by step-budget drivers.
    pub done: bool,

    episode_accounting: EpisodeAccounting,
}

impl<S, A> EnvironmentStation<S, A> {
    pub fn new(
        id: impl Into<String>,
        mut env: Box<dyn PureEnvironment<S, A>>,
        opts: EnvironmentStationOptions,
    ) -> Self {
        let cur_state = env.reset();
        EnvironmentStation {
            core: StationCore::new(id),
            env,
            num_episodes: opts.num_episodes.unwrap_or(f64::INFINITY),
            max_steps_per_episode: opts.max_steps_per_episode.unwrap_or(1_000_000),
            cur_state,
            episode_id: 0,
            step_in_episode: 0,
            emitted_start: false,
            done: false,
            episode_accounting: EpisodeAccounting::new(),
        }
    }

    // ── ACCESSORS (TS getter/setter pairs + episodeAccounting aliases) ─────────

    pub fn reward_history(&self) -> &[f64] {
        &self.episode_accounting.reward_history
    }
    pub fn length_history(&self) -> &[f64] {
        &self.episode_accounting.length_history
    }
    pub fn total_steps(&self) -> u64 {
        self.episode_accounting.total_steps
    }
    pub fn set_total_steps(&mut self, value: u64) {
        self.episode_accounting.total_steps = value;
    }
    pub fn cur_return(&self) -> f64 {
        self.episode_accounting.current_reward
    }
    pub fn set_cur_return(&mut self, value: f64) {
        self.episode_accounting.current_reward = value;
    }
    pub fn cur_length(&self) -> f64 {
        self.episode_accounting.current_length
    }
    pub fn set_cur_length(&mut self, value: f64) {
        self.episode_accounting.current_length = value;
    }
    pub fn episode_id(&self) -> usize {
        self.episode_id
    }
}

impl<S: Clone + 'static, A: Clone + 'static> DESStation for EnvironmentStation<S, A> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn has_work(&self) -> bool {
        if self.done {
            return false;
        }
        if !self.emitted_start {
            return true;
        }
        if (self.episode_id as f64) >= self.num_episodes {
            return false;
        }
        self.core.inbox_size(CH_ACTION) > 0
    }

    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        if !self.emitted_start {
            self.emitted_start = true;
            let tok = StateToken::new(self.cur_state.clone(), self.episode_id as f64);
            self.core.emit(Rc::new(tok), CH_STATE);
            return;
        }
        if (self.episode_id as f64) >= self.num_episodes {
            return;
        }
        let actions = self.core.drain::<ActionToken<S, A>>(CH_ACTION);
        for a in actions {
            if a.episode_id != self.episode_id as f64 {
                continue;
            }
            let r = self.env.step(a.state.clone(), a.action.clone());
            self.episode_accounting.record_step(r.reward);
            self.step_in_episode += 1;
            let truncated = self.step_in_episode >= self.max_steps_per_episode;
            let is_done = r.done || truncated;
            let tok = TransitionToken::new(
                a.state.clone(),
                a.action.clone(),
                r.reward,
                r.next_state,
                is_done,
                self.episode_id as f64,
            );
            self.core.emit(Rc::new(tok), CH_TRANSITION);
            if is_done {
                self.episode_accounting.finish_episode();
                self.step_in_episode = 0;
                self.episode_id += 1;
                if (self.episode_id as f64) < self.num_episodes {
                    self.cur_state = self.env.reset();
                    let st = StateToken::new(self.cur_state.clone(), self.episode_id as f64);
                    self.core.emit(Rc::new(st), CH_STATE);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::rl_tokens::{ActionToken, TransitionToken};
    use super::super::station::{DESStation, StationCore, StationRef};
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A trivial 1-D walk: positions `0..n`, action `1` = right, `0` = left.
    /// Reaching the rightmost cell (`goal = n - 1`) yields reward `1.0` and
    /// ends the episode; every other step yields `0.0`.
    struct LineWalk {
        n: usize,
    }

    impl LineWalk {
        fn new(n: usize) -> Self {
            LineWalk { n }
        }
    }

    impl PureEnvironment<f64, usize> for LineWalk {
        fn num_states(&self) -> usize {
            self.n
        }
        fn num_actions(&self) -> usize {
            2
        }
        fn reset(&mut self) -> f64 {
            0.0
        }
        fn step(&mut self, state: f64, action: usize) -> StepResult<f64> {
            let delta = if action == 1 { 1.0 } else { -1.0 };
            let mut pos = state + delta;
            if pos < 0.0 {
                pos = 0.0;
            }
            let goal = (self.n - 1) as f64;
            if pos > goal {
                pos = goal;
            }
            let done = pos >= goal;
            StepResult {
                next_state: pos,
                reward: if done { 1.0 } else { 0.0 },
                done,
            }
        }
    }

    /// Drains a channel and counts the typed tokens it receives.
    struct Collector<T: 'static> {
        core: StationCore,
        items: Vec<Rc<T>>,
    }

    impl<T: 'static> Collector<T> {
        fn new() -> Self {
            Collector {
                core: StationCore::new("collector"),
                items: Vec::new(),
            }
        }
    }

    impl<T: 'static> DESStation for Collector<T> {
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
            let mut drained = self.core.drain::<T>("in");
            self.items.append(&mut drained);
        }
    }

    #[test]
    fn emits_start_then_completes_episode() {
        let mut env: EnvironmentStation = EnvironmentStation::new(
            "env",
            Box::new(LineWalk::new(4)),
            EnvironmentStationOptions {
                num_episodes: Some(1.0),
                ..Default::default()
            },
        );
        let transitions: Rc<RefCell<Collector<TransitionToken<f64, usize>>>> =
            Rc::new(RefCell::new(Collector::new()));
        env.core_mut()
            .pipe(transitions.clone() as StationRef, CH_TRANSITION, "in");

        assert!(env.has_work());
        // First tick emits only the start state (no transition yet).
        env.run_time_step();
        assert_eq!(transitions.borrow().items.len(), 0);

        // Feed three "go right" actions; from 0 → 1 → 2 → 3 (= goal).
        env.core_mut()
            .take(Rc::new(ActionToken::new(0.0_f64, 1usize, 0.0)), CH_ACTION);
        env.core_mut()
            .take(Rc::new(ActionToken::new(1.0_f64, 1usize, 0.0)), CH_ACTION);
        env.core_mut()
            .take(Rc::new(ActionToken::new(2.0_f64, 1usize, 0.0)), CH_ACTION);
        env.run_time_step();
        transitions.borrow_mut().run_time_step();

        assert_eq!(env.total_steps(), 3);
        assert_eq!(env.reward_history(), &[1.0]);
        assert_eq!(env.episode_id(), 1);
        let captured = transitions.borrow();
        assert_eq!(captured.items.len(), 3);
        assert!(captured.items[2].done);
        // Episode budget reached → no more work.
        assert!(!env.has_work());
    }

    #[test]
    fn truncates_long_episodes() {
        // Walk of length 10; goal is far. Cap episodes at 2 steps → truncation.
        let mut env: EnvironmentStation = EnvironmentStation::new(
            "env",
            Box::new(LineWalk::new(10)),
            EnvironmentStationOptions {
                num_episodes: Some(5.0),
                max_steps_per_episode: Some(2),
            },
        );
        env.run_time_step(); // emit start
        env.core_mut()
            .take(Rc::new(ActionToken::new(0.0_f64, 1usize, 0.0)), CH_ACTION);
        env.core_mut()
            .take(Rc::new(ActionToken::new(1.0_f64, 1usize, 0.0)), CH_ACTION);
        env.run_time_step();
        // Two steps → truncated done, episode finished without reaching goal.
        assert_eq!(env.reward_history().len(), 1);
        assert_eq!(env.length_history(), &[2.0]);
        assert_eq!(env.episode_id(), 1);
    }

    #[test]
    fn external_done_flag_short_circuits() {
        let mut env: EnvironmentStation = EnvironmentStation::new(
            "env",
            Box::new(LineWalk::new(4)),
            EnvironmentStationOptions::default(),
        );
        env.done = true;
        assert!(!env.has_work());
        env.run_time_step(); // no-op
        assert!(!env.emitted_start);
    }
}
