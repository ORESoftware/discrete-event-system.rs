//! Port of `src/des/general/mountain-car.ts` — the classic MOUNTAIN CAR control
//! problem (Moore 1990; Sutton & Barto 1998 section 10.1) solved with LINEAR
//! function approximation over a TILE-CODED feature set.
//!
//! PROBLEM: continuous state (position, velocity). An underpowered car at the
//! bottom of a valley must rock back and forth to build momentum and reach the
//! goal at position >= 0.5. Reward is minus-one per step until the goal; the
//! episode terminates at the goal. Three actions: reverse, coast, forward.
//! Dynamics: v_next = clamp(v + 0.001*a - 0.0025*cos(3*x), -0.07, 0.07) and
//! x_next = clamp(x + v_next, -1.2, 0.5); hitting the left wall zeroes v.
//!
//! TILE CODING (Sutton-Albus CMAC): discretise (position, velocity) into a
//! coarse grid with several offset tilings; each tiling contributes one active
//! binary feature per state, giving featureDim = numTilings * posBins * velBins.
//!
//! Declarations -> Rust:
//!   * `interface MountainCarOpts/TrainOpts/Result` -> structs
//!   * `class MountainCarEnv implements Environment` -> struct + an
//!     [`McStationEnv`] adapter implementing `PureEnvironment`
//!   * `class MountainCarLinearVFA extends LinearVFAStation` -> struct + the
//!     [`LinearVFAStation`] hook trait
//!   * `fn runMountainCar` -> free fn [`run_mountain_car`]
//!
//! Conversion notes:
//!   * The agent and the environment station SHARE the env (the agent's
//!     `features()` reads the continuous state the station's `step` wrote into
//!     the side table). This shared mutable env is an `Rc<RefCell<MountainCarEnv>>`.
//!   * `states: Map<number,[number,number]>` -> `HashMap<usize,(f64,f64)>`.
//!   * The TS greedy-eval `(greedyEnv as any).nextId++` / `.states.set(...)`
//!     reach into private fields; replaced by the typed accessors
//!     [`MountainCarEnv::alloc_id`] / [`MountainCarEnv::set_state`] (no casts).
//!   * The single shared TS `rng` closure is a cloneable [`SharedRng`]. The TS
//!     default `reset()` consumed `DEFAULT_RANDOM` (absent from the ported
//!     capabilities) but is never called by `runMountainCar` (which uses
//!     `resetWithRng`), so only [`MountainCarEnv::reset_with_rng`] is provided.
//!     FLAGGED in the return notes.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::environment::{
    EnvironmentStation, EnvironmentStationOptions, PureEnvironment, StepResult, CH_ACTION, CH_STATE,
    CH_TRANSITION,
};
use crate::des::general::des_base::linear_vfa::{LinearVFACore, LinearVFAOptions, LinearVFAStation};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// SHARED RNG
// -----------------------------------------------------------------------------

/// Cloneable handle to one PRNG stream (mirrors the single shared TS `rng`).
#[derive(Clone)]
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl SharedRng {
    fn new(seed: u32) -> Self {
        SharedRng(Rc::new(RefCell::new(SeededRandom::new(seed))))
    }
}

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

// -----------------------------------------------------------------------------
// OPTIONS
// -----------------------------------------------------------------------------

/// Environment options (TS `MountainCarOpts`); `None` fields use the defaults.
#[derive(Clone, Debug, Default)]
pub struct MountainCarOpts {
    /// Number of tilings stacked over (position, velocity). Default 8.
    pub num_tilings: Option<usize>,
    /// Tiles per tiling (per dimension). Default 8.
    pub num_tiles_per_dim: Option<usize>,
    /// Position range. Default (-1.2, 0.5).
    pub pos_range: Option<(f64, f64)>,
    /// Velocity range. Default (-0.07, 0.07).
    pub vel_range: Option<(f64, f64)>,
    /// Max steps per episode. Default 1000.
    pub max_steps_per_episode: Option<usize>,
    /// Goal position. Default 0.5.
    pub goal_pos: Option<f64>,
}

/// `Required<MountainCarOpts>` after defaults are resolved.
#[derive(Clone, Debug)]
struct ResolvedMcOpts {
    num_tilings: usize,
    num_tiles_per_dim: usize,
    pos_range: (f64, f64),
    vel_range: (f64, f64),
    #[allow(dead_code)]
    max_steps_per_episode: usize,
    goal_pos: f64,
}

// -----------------------------------------------------------------------------
// ENVIRONMENT
// -----------------------------------------------------------------------------

/// Mountain Car environment. Continuous state is encoded as an integer id; the
/// (position, velocity) pair is held in a side table keyed by id.
pub struct MountainCarEnv {
    opts: ResolvedMcOpts,
    states: HashMap<usize, (f64, f64)>,
    next_id: usize,
}

impl MountainCarEnv {
    /// Number of discrete actions (reverse, coast, forward).
    pub const NUM_ACTIONS: usize = 3;

    fn new(opts: MountainCarOpts) -> Self {
        MountainCarEnv {
            opts: ResolvedMcOpts {
                num_tilings: opts.num_tilings.unwrap_or(8),
                num_tiles_per_dim: opts.num_tiles_per_dim.unwrap_or(8),
                pos_range: opts.pos_range.unwrap_or((-1.2, 0.5)),
                vel_range: opts.vel_range.unwrap_or((-0.07, 0.07)),
                max_steps_per_episode: opts.max_steps_per_episode.unwrap_or(1000),
                goal_pos: opts.goal_pos.unwrap_or(0.5),
            },
            states: HashMap::new(),
            next_id: 0,
        }
    }

    /// Reset using an injected RNG (for reproducibility). Standard MC start:
    /// uniform position on [-0.6, -0.4] with zero velocity.
    fn reset_with_rng(&mut self, rng: &mut dyn RandomSource) -> usize {
        let id = self.alloc_id();
        let x = -0.6 + 0.2 * rng.next_float();
        self.states.insert(id, (x, 0.0));
        id
    }

    fn step(&mut self, state_id: usize, action: usize) -> StepResult<usize> {
        let (x, v) = *self.states.get(&state_id).expect("unknown stateId");
        let a = action as f64 - 1.0; // 0,1,2 -> -1,0,+1
        let mut v_new = v + 0.001 * a - 0.0025 * (3.0 * x).cos();
        let lo = self.opts.vel_range.0;
        let hi = self.opts.vel_range.1;
        v_new = lo.max(hi.min(v_new));
        let mut x_new = x + v_new;
        if x_new < self.opts.pos_range.0 {
            x_new = self.opts.pos_range.0;
            v_new = 0.0;
        }
        if x_new > self.opts.pos_range.1 {
            x_new = self.opts.pos_range.1;
        }
        let done = x_new >= self.opts.goal_pos;
        let id = self.alloc_id();
        self.states.insert(id, (x_new, v_new));
        StepResult { next_state: id, reward: -1.0, done }
    }

    /// (position, velocity) for an id. Panics on an unknown id (TS `throw`).
    pub fn get_continuous_state(&self, state_id: usize) -> (f64, f64) {
        *self.states.get(&state_id).expect("unknown stateId")
    }

    fn get_opts(&self) -> &ResolvedMcOpts {
        &self.opts
    }

    /// Hand out a fresh integer id (typed replacement for `(env as any).nextId++`).
    pub fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Store a continuous state under an id (typed replacement for
    /// `(env as any).states.set(...)`).
    pub fn set_state(&mut self, id: usize, state: (f64, f64)) {
        self.states.insert(id, state);
    }
}

/// `PureEnvironment` adapter wrapping the shared [`MountainCarEnv`] (the TS
/// `pureEnv` object whose `reset` closes over the same env + shared rng).
struct McStationEnv {
    env: Rc<RefCell<MountainCarEnv>>,
    rng: SharedRng,
}

impl PureEnvironment<usize, usize> for McStationEnv {
    fn num_states(&self) -> usize {
        1 // not used (states are re-keyed via the side table)
    }

    fn num_actions(&self) -> usize {
        MountainCarEnv::NUM_ACTIONS
    }

    fn reset(&mut self) -> usize {
        let mut rng = self.rng.clone();
        self.env.borrow_mut().reset_with_rng(&mut rng)
    }

    fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
        self.env.borrow_mut().step(state, action)
    }
}

// -----------------------------------------------------------------------------
// LINEAR VFA AGENT WITH TILE CODING
// -----------------------------------------------------------------------------

struct MountainCarLinearVFA {
    core: StationCore,
    agent: RLAgentCore,
    vfa: LinearVFACore,
    env: Rc<RefCell<MountainCarEnv>>,
    num_tilings: usize,
    num_tiles_per_dim: usize,
    pos_low: f64,
    pos_span: f64,
    vel_low: f64,
    vel_span: f64,
}

impl MountainCarLinearVFA {
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: Rc<RefCell<MountainCarEnv>>,
        rng: Box<dyn RandomSource>,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
        epsilon_min: f64,
    ) -> Self {
        let o = env.borrow().get_opts().clone();
        let feature_dim = o.num_tilings * o.num_tiles_per_dim * o.num_tiles_per_dim;
        let vfa = LinearVFACore::new(LinearVFAOptions {
            feature_dim,
            num_actions: 3,
            alpha: Some(alpha / o.num_tilings as f64), // canonical: divide by numTilings
            gamma: Some(gamma),
            epsilon: Some(epsilon),
            epsilon_decay: Some(epsilon_decay),
            epsilon_min: Some(epsilon_min),
            ..Default::default()
        });
        MountainCarLinearVFA {
            core: StationCore::new("mc-vfa"),
            agent: RLAgentCore::new(rng),
            vfa,
            num_tilings: o.num_tilings,
            num_tiles_per_dim: o.num_tiles_per_dim,
            pos_low: o.pos_range.0,
            pos_span: o.pos_range.1 - o.pos_range.0,
            vel_low: o.vel_range.0,
            vel_span: o.vel_range.1 - o.vel_range.0,
            env,
        }
    }
}

impl DESStation for MountainCarLinearVFA {
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
        self.rl_agent_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.rl_agent_has_work()
    }
}

impl RLAgentStation<usize, usize> for MountainCarLinearVFA {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }
    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        self.linear_vfa_pick_action(state, rng)
    }
    fn update(&mut self, state: &usize, action: &usize, reward: f64, next_state: &usize, done: bool) {
        self.linear_vfa_update(state, *action, reward, next_state, done);
    }
    fn end_of_episode(&mut self, _episode_id: f64) {
        self.linear_vfa_end_of_episode();
    }
}

impl LinearVFAStation<usize> for MountainCarLinearVFA {
    fn vfa_core(&self) -> &LinearVFACore {
        &self.vfa
    }
    fn vfa_core_mut(&mut self) -> &mut LinearVFACore {
        &mut self.vfa
    }

    /// Sutton-Albus tile coding: each tiling k is offset by k/numTilings of a
    /// single tile width; the active tile for tiling k at (x, v) is
    /// ((p_idx + offset) clamped) * n + ((v_idx + offset) clamped).
    fn features(&self, state: &usize) -> Vec<f64> {
        let (x, v) = self.env.borrow().get_continuous_state(*state);
        let px = (x - self.pos_low) / self.pos_span;
        let vy = (v - self.vel_low) / self.vel_span;
        let n = self.num_tiles_per_dim;
        let feature_dim = self.num_tilings * n * n;
        let mut buf = vec![0.0_f64; feature_dim];
        for k in 0..self.num_tilings {
            let offset = k as f64 / self.num_tilings as f64;
            let p_idx = ((px + offset) * n as f64).floor() as i64;
            let v_idx = ((vy + offset) * n as f64).floor() as i64;
            let p_idx = p_idx.clamp(0, n as i64 - 1);
            let v_idx = v_idx.clamp(0, n as i64 - 1);
            let tile_idx = (p_idx * n as i64 + v_idx) as usize;
            let feat_idx = k * n * n + tile_idx;
            buf[feat_idx] = 1.0;
        }
        buf
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Training options for [`run_mountain_car`]. `None` fields use TS defaults.
#[derive(Clone, Debug)]
pub struct MountainCarTrainOpts {
    pub num_episodes: usize,
    pub max_steps_per_episode: Option<usize>,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub seed: Option<u32>,
    pub num_tilings: Option<usize>,
    pub num_tiles_per_dim: Option<usize>,
}

impl Default for MountainCarTrainOpts {
    fn default() -> Self {
        MountainCarTrainOpts {
            num_episodes: 1,
            max_steps_per_episode: None,
            alpha: None,
            gamma: None,
            epsilon: None,
            epsilon_decay: None,
            epsilon_min: None,
            seed: None,
            num_tilings: None,
            num_tiles_per_dim: None,
        }
    }
}

/// Result of [`run_mountain_car`].
#[derive(Clone, Debug, PartialEq)]
pub struct MountainCarResult {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub td_error_history: Vec<f64>,
    /// True iff the GREEDY policy reaches the goal from a quiet start
    /// (x = -0.5, v = 0) within `max_steps_per_episode`.
    pub greedy_solves: bool,
    pub greedy_episode_length: f64,
    pub final_epsilon: f64,
    pub theta_norm: f64,
    pub ticks: usize,
}

/// Train a tile-coded linear-VFA agent on Mountain Car, then roll out the greedy
/// policy from a quiet start to evaluate it.
pub fn run_mountain_car(opts: MountainCarTrainOpts) -> MountainCarResult {
    let cls = "run_mountain_car";
    Preconditions::integer_in_range(cls, "numEpisodes", opts.num_episodes as f64, 1.0, 1e9).unwrap();
    if let Some(m) = opts.max_steps_per_episode {
        Preconditions::integer_in_range(cls, "maxStepsPerEpisode", m as f64, 1.0, 1e9).unwrap();
    }
    if let Some(a) = opts.alpha {
        Preconditions::positive(cls, "alpha", a).unwrap();
    }
    if let Some(g) = opts.gamma {
        Preconditions::in_range(cls, "gamma", g, 0.0, 1.0).unwrap();
    }
    if let Some(e) = opts.epsilon {
        Preconditions::in_range(cls, "epsilon", e, 0.0, 1.0).unwrap();
    }
    if let Some(e) = opts.epsilon_decay {
        Preconditions::in_range(cls, "epsilonDecay", e, 0.0, 1.0).unwrap();
    }
    if let Some(e) = opts.epsilon_min {
        Preconditions::in_range(cls, "epsilonMin", e, 0.0, 1.0).unwrap();
    }
    if let Some(t) = opts.num_tilings {
        Preconditions::integer_in_range(cls, "numTilings", t as f64, 1.0, 1e6).unwrap();
    }
    if let Some(t) = opts.num_tiles_per_dim {
        Preconditions::integer_in_range(cls, "numTilesPerDim", t as f64, 1.0, 1e6).unwrap();
    }

    let rng = SharedRng::new(opts.seed.unwrap_or(1));
    let env = Rc::new(RefCell::new(MountainCarEnv::new(MountainCarOpts {
        num_tilings: opts.num_tilings,
        num_tiles_per_dim: opts.num_tiles_per_dim,
        max_steps_per_episode: opts.max_steps_per_episode,
        ..Default::default()
    })));
    let agent = MountainCarLinearVFA::new(
        env.clone(),
        Box::new(rng.clone()),
        opts.alpha.unwrap_or(0.5),
        opts.gamma.unwrap_or(1.0),
        opts.epsilon.unwrap_or(0.0), // canonical MC: greedy is enough with tile coding
        opts.epsilon_decay.unwrap_or(1.0),
        opts.epsilon_min.unwrap_or(0.0),
    );

    let max = opts.max_steps_per_episode.unwrap_or(1000);
    let station_env = McStationEnv { env: env.clone(), rng: rng.clone() };
    let env_station = EnvironmentStation::<usize, usize>::new(
        "env",
        Box::new(station_env),
        EnvironmentStationOptions {
            num_episodes: Some(opts.num_episodes as f64),
            max_steps_per_episode: Some(max),
        },
    );

    let agent: Rc<RefCell<MountainCarLinearVFA>> = Rc::new(RefCell::new(agent));
    let env_station: Rc<RefCell<EnvironmentStation<usize, usize>>> = Rc::new(RefCell::new(env_station));

    env_station.borrow_mut().core_mut().pipe(agent.clone() as StationRef, CH_STATE, CH_STATE);
    env_station.borrow_mut().core_mut().pipe(agent.clone() as StationRef, CH_TRANSITION, CH_TRANSITION);
    agent.borrow_mut().core_mut().pipe(env_station.clone() as StationRef, CH_ACTION, CH_ACTION);

    let summary = run_iterative_des(
        vec![env_station.clone() as StationRef, agent.clone() as StationRef],
        IterativeRunOptions {
            rng: Some({
                let mut r = rng.clone();
                Box::new(move || r.next_float())
            }),
            ..Default::default()
        },
    );

    // Greedy rollout from a quiet start (x = -0.5, v = 0). The agent's env gets
    // the state injected so `features()` can read it.
    let mut greedy_env = MountainCarEnv::new(MountainCarOpts {
        num_tilings: opts.num_tilings,
        num_tiles_per_dim: opts.num_tiles_per_dim,
        max_steps_per_episode: opts.max_steps_per_episode,
        ..Default::default()
    });
    let start_id = greedy_env.alloc_id();
    greedy_env.set_state(start_id, (-0.5, 0.0));
    let mut s = start_id;
    let mut solves = false;
    let mut len = 0usize;
    for _ in 0..max {
        let st = greedy_env.get_continuous_state(s);
        env.borrow_mut().set_state(s, st);
        let a = agent.borrow_mut().vfa_greedy_action(&s);
        let stp = greedy_env.step(s, a);
        len += 1;
        if stp.done {
            solves = true;
            break;
        }
        s = stp.next_state;
    }

    let theta_norm = {
        let b = agent.borrow();
        b.get_theta().iter().map(|x| x * x).sum::<f64>().sqrt()
    };
    let reward_history = agent.borrow().reward_history().to_vec();
    let length_history = agent.borrow().length_history().to_vec();
    let td_error_history = agent.borrow().vfa_core().td_error_history.clone();
    let final_epsilon = agent.borrow().get_epsilon();

    MountainCarResult {
        reward_history,
        length_history,
        td_error_history,
        greedy_solves: solves,
        greedy_episode_length: len as f64,
        final_epsilon,
        theta_norm,
        ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! Env reset/step dynamics + tile-coded training episode termination.

    use super::*;

    #[test]
    fn reset_starts_in_the_valley() {
        let mut env = MountainCarEnv::new(MountainCarOpts::default());
        let mut rng = SeededRandom::new(1);
        let id = env.reset_with_rng(&mut rng);
        let (x, v) = env.get_continuous_state(id);
        assert!((-0.6..=-0.4).contains(&x), "x = {x}");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn forward_at_the_goal_terminates() {
        let mut env = MountainCarEnv::new(MountainCarOpts::default());
        let id = env.alloc_id();
        env.set_state(id, (0.5, 0.0));
        let r = env.step(id, 2); // forward
        assert!(r.done, "stepping forward at the goal position terminates");
        assert_eq!(r.reward, -1.0);
    }

    #[test]
    fn training_runs_episodes_to_completion() {
        let res = run_mountain_car(MountainCarTrainOpts {
            num_episodes: 3,
            seed: Some(1),
            num_tilings: Some(4),
            num_tiles_per_dim: Some(6),
            max_steps_per_episode: Some(200),
            ..Default::default()
        });
        assert_eq!(res.reward_history.len(), 3, "one reward logged per episode");
        assert_eq!(res.length_history.len(), 3);
        assert!(res.ticks > 0);
        assert!(res.theta_norm.is_finite());
    }
}
