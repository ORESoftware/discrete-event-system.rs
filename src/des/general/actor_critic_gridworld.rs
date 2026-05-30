//! Port of `src/des/general/actor-critic-gridworld.ts` — the runnable driver for
//! a ONE-STEP TABULAR ACTOR-CRITIC (Sutton & Barto §13.5) on a small GridWorld.
//!
//! The [`TabularActorCritic`] base lives in `des_base::actor_critic`. This module
//! is the driver: wire the agent to an [`EnvironmentStation`], run the training
//! loop, then report the learned `V(s)` / greedy-rollout outcome at the start
//! state.
//!
//! ## TS → Rust mapping
//!
//!   * `interface ActorCriticTrainOpts` → [`ActorCriticTrainOpts`] (`Default`;
//!     optionals → `Option<T>`).
//!   * `interface ActorCriticResult` → [`ActorCriticResult`] (`#[derive(Clone)]`;
//!     `readonly number[]` history fields → `Vec<f64>` returned by value).
//!   * `fn runActorCriticGridworld` → the free fn [`run_actor_critic_gridworld`].
//!   * `mulberry32(seed)` (a `() => number` closure shared by the agent AND the
//!     runner) → a single [`SeededRandom`] behind `Rc<RefCell<…>>`, bridged into
//!     both via the [`SharedRng`] newtype so the agent's categorical sampling and
//!     the runner's tick-shuffle draw from the SAME stream (faithful to the TS
//!     shared closure). FLAGGED local equivalent — see the sibling
//!     `rl_learning_models.rs` / `qlearning_des.rs`.
//!   * `GridWorld` implements the rl-environments [`Environment`] trait, not the
//!     [`PureEnvironment`] hook trait that [`EnvironmentStation`] wraps; the local
//!     [`EnvAdapter`] bridges the two (FLAGGED — in TS the interfaces were
//!     structurally identical so one class satisfied both).
//!   * `Preconditions.*` guards `throw` in TS (invariant violations) → `panic!`
//!     here (the Rust guards return `Result`; the driver unwraps to a panic).
//!   * `number` → `f64`, state/action indices → `usize`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::actor_critic::{ActorCriticOptions, TabularActorCritic};
use crate::des::general::des_base::environment::{
    self, EnvironmentStation, EnvironmentStationOptions, PureEnvironment, StepResult,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::rl_agent::RLAgentStation;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::general::rl_environments::{Environment, GridWorld, GridWorldOptions};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Bridges one shared [`SeededRandom`] (behind `Rc<RefCell<…>>`) into a boxed
/// [`RandomSource`]. FLAGGED local equivalent: the TS code shared a single
/// `() => number` closure between the agent and the runner; Rust cannot alias an
/// owned `Box<dyn RandomSource>`, so this newtype hands out a [`RandomSource`]
/// view backed by the same generator state.
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

/// Adapts a pure [`Environment`] (rl-environments) into the [`PureEnvironment`]
/// hook trait that [`EnvironmentStation`] wraps. FLAGGED local equivalent: in TS
/// the two interfaces were structurally identical so `GridWorld` satisfied both;
/// Rust models them as distinct traits, so this thin adapter bridges
/// `Environment` → `PureEnvironment<usize, usize>`.
struct EnvAdapter {
    env: Rc<dyn Environment>,
}

impl PureEnvironment<usize, usize> for EnvAdapter {
    fn num_states(&self) -> usize {
        self.env.num_states()
    }
    fn num_actions(&self) -> usize {
        self.env.num_actions()
    }
    fn reset(&mut self) -> usize {
        self.env.reset()
    }
    fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
        let o = self.env.step(state, action);
        StepResult { next_state: o.next_state, reward: o.reward, done: o.done }
    }
}

/// `interface ActorCriticTrainOpts`. `num_episodes` is required; the rest fall
/// back to the [`TabularActorCritic`] defaults when `None`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ActorCriticTrainOpts {
    pub num_episodes: usize,
    pub max_steps_per_episode: Option<usize>,
    pub alpha_v: Option<f64>,
    pub alpha_p: Option<f64>,
    pub gamma: Option<f64>,
    pub entropy_coef: Option<f64>,
    pub seed: Option<u32>,
    /// GridWorld dimensions. Default 4×4.
    pub width: Option<usize>,
    pub height: Option<usize>,
}

/// `interface ActorCriticResult`.
#[derive(Clone, Debug, Default)]
pub struct ActorCriticResult {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub td_error_history: Vec<f64>,
    /// `V_θ` at the start state — proxy for "how good is the start".
    pub v_start: f64,
    /// Whether the GREEDY policy reaches the goal from the start.
    pub greedy_reached: bool,
    pub greedy_len: usize,
    pub ticks: usize,
}

/// Wire a [`TabularActorCritic`] to an [`EnvironmentStation`] over a GridWorld,
/// run the DES, then report the learned `V`/greedy rollout from the start state.
pub fn run_actor_critic_gridworld(opts: ActorCriticTrainOpts) -> ActorCriticResult {
    let cls = "runActorCriticGridworld";
    // TS `Preconditions.*` throw on violation → panic here (invariant guards).
    let must = |c: Check| {
        if let Err(e) = c {
            panic!("{e}");
        }
    };
    must(Preconditions::integer_in_range(cls, "numEpisodes", opts.num_episodes as f64, 1.0, 1e9));
    if let Some(m) = opts.max_steps_per_episode {
        must(Preconditions::integer_in_range(cls, "maxStepsPerEpisode", m as f64, 1.0, 1e9));
    }
    if let Some(v) = opts.alpha_v {
        must(Preconditions::positive(cls, "alphaV", v));
    }
    if let Some(v) = opts.alpha_p {
        must(Preconditions::positive(cls, "alphaP", v));
    }
    if let Some(v) = opts.gamma {
        must(Preconditions::in_range(cls, "gamma", v, 0.0, 1.0));
    }
    if let Some(v) = opts.entropy_coef {
        must(Preconditions::non_negative(cls, "entropyCoef", v));
    }
    if let Some(v) = opts.width {
        must(Preconditions::integer_in_range(cls, "width", v as f64, 1.0, 10000.0));
    }
    if let Some(v) = opts.height {
        must(Preconditions::integer_in_range(cls, "height", v as f64, 1.0, 10000.0));
    }

    let width = opts.width.unwrap_or(4);
    let height = opts.height.unwrap_or(4);
    let max_steps = opts.max_steps_per_episode.unwrap_or(100);
    let shared = Rc::new(RefCell::new(mulberry32(opts.seed.unwrap_or(1))));

    let env_concrete: Rc<GridWorld> =
        Rc::new(GridWorld::new(GridWorldOptions { width: Some(width), height: Some(height), ..Default::default() }));
    let num_states = env_concrete.num_states();
    let num_actions = env_concrete.num_actions();

    let agent = Rc::new(RefCell::new(TabularActorCritic::new(
        "ac-grid",
        Box::new(SharedRng(shared.clone())),
        ActorCriticOptions {
            num_states,
            num_actions,
            alpha_v: opts.alpha_v,
            alpha_p: opts.alpha_p,
            gamma: opts.gamma,
            entropy_coef: opts.entropy_coef,
            ..Default::default()
        },
    )));
    let env_station = Rc::new(RefCell::new(EnvironmentStation::new(
        "env",
        Box::new(EnvAdapter { env: env_concrete.clone() }),
        EnvironmentStationOptions {
            num_episodes: Some(opts.num_episodes as f64),
            max_steps_per_episode: Some(max_steps),
        },
    )));

    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_STATE,
        <TabularActorCritic as RLAgentStation<usize, usize>>::CH_STATE,
    );
    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_TRANSITION,
        <TabularActorCritic as RLAgentStation<usize, usize>>::CH_TRANSITION,
    );
    agent.borrow_mut().core_mut().pipe(
        env_station.clone() as StationRef,
        <TabularActorCritic as RLAgentStation<usize, usize>>::CH_ACTION,
        environment::CH_ACTION,
    );

    let mut des = IterativeRunOptions::default();
    let r = shared.clone();
    des.rng = Some(Box::new(move || r.borrow_mut().next_float()));
    let summary = run_iterative_des(
        vec![env_station.clone() as StationRef, agent.clone() as StationRef],
        des,
    );

    // Greedy rollout from a fresh evaluation environment.
    let eval_env = GridWorld::new(GridWorldOptions { width: Some(width), height: Some(height), ..Default::default() });
    let mut s = eval_env.reset();
    let mut len = 0usize;
    let mut reached = false;
    for _ in 0..max_steps {
        let a = agent.borrow_mut().greedy_action(s);
        let r = eval_env.step(s, a);
        len += 1;
        if r.done {
            reached = r.reward > 0.0;
            break;
        }
        s = r.next_state;
    }

    let reward_history = agent.borrow().reward_history().to_vec();
    let length_history = agent.borrow().length_history().to_vec();
    let td_error_history = agent.borrow().td_error_history.clone();
    let v_start = agent.borrow().get_v()[env_concrete.start];
    ActorCriticResult {
        reward_history,
        length_history,
        td_error_history,
        v_start,
        greedy_reached: reached,
        greedy_len: len,
        ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! The actor-critic driver learns to navigate the 4×4 GridWorld.
    //!
    //! After enough episodes the greedy policy should reach the goal from the
    //! start in a short path, the learned start-state value should be positive
    //! (the goal is reachable), and the per-episode mean TD error should shrink
    //! as the critic's value estimates settle.
    use super::*;

    fn train(num_episodes: usize, seed: u32) -> ActorCriticResult {
        run_actor_critic_gridworld(ActorCriticTrainOpts {
            num_episodes,
            seed: Some(seed),
            ..Default::default()
        })
    }

    #[test]
    fn greedy_policy_reaches_goal_quickly() {
        let res = train(3000, 1);
        assert!(res.greedy_reached, "greedy policy should reach the goal");
        // Optimal path from start (0) to goal (15) on a 4×4 grid is 6 steps.
        assert!(res.greedy_len <= 8, "greedy path should be short: {}", res.greedy_len);
        assert!(res.v_start > 0.0, "start value should be positive: {}", res.v_start);
    }

    #[test]
    fn logs_one_history_entry_per_episode() {
        let res = train(400, 2);
        assert_eq!(res.reward_history.len(), 400);
        assert_eq!(res.length_history.len(), 400);
        assert_eq!(res.td_error_history.len(), 400);
        assert!(res.ticks > 0);
    }

    #[test]
    fn mean_td_error_shrinks_over_training() {
        let res = train(3000, 3);
        let h = &res.td_error_history;
        let window = 100;
        let first: f64 = h[..window].iter().sum::<f64>() / window as f64;
        let last: f64 = h[h.len() - window..].iter().sum::<f64>() / window as f64;
        assert!(last < first, "mean TD error should shrink: first {first}, last {last}");
    }
}
