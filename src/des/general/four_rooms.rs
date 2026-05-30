//! Port of `src/des/general/four-rooms.ts` — the canonical FOUR ROOMS benchmark
//! (Sutton, Precup, Singh 1999) for SEMI-MDPs / the OPTIONS framework.
//!
//! An 11 by 11 grid of four rooms joined by four hallway cells. States = 121
//! cells minus walls; actions = N, E, S, W with an optional perpendicular slip
//! probability. Reward is plus-one at the goal (10, 10) and zero otherwise; the
//! episode terminates at the goal.
//!
//! HALLWAY OPTIONS: eight temporally-extended options (two per room, "go to the
//! adjacent NORTH/SOUTH or EAST/WEST hallway"), each with a hard-coded
//! shortest-path internal policy and a termination probability of 1 at the
//! hallway cell. With these options, SMDP Q-learning over the 8-action
//! option-MDP learns the goal in O(few) episodes.
//!
//! Declarations -> Rust:
//!   * `interface FourRoomsOpts/TrainOpts/Result`        -> structs
//!   * `class FourRoomsEnv implements Environment`        -> struct + `impl PureEnvironment`
//!   * `class FourRoomsSMDPAgent extends SemiMDPAgentStation` -> struct + the
//!     [`SemiMDPAgentStation`] hook trait (delegating the RL hooks)
//!   * `const FOUR_ROOMS_MAP/HALLWAYS/GOAL/DR/DC`          -> consts / a derived
//!     [`is_free`] wall predicate (no stored map needed)
//!   * the RL `Option<S, A>` type (init/policy/terminate)  -> the crate's
//!     [`Opt`] trait (renamed from `Option` in the SMDP module)
//!
//! Conversion notes:
//!   * The single TS `rng` closure shared by env + agent + runner is modelled as
//!     a cloneable [`SharedRng`]; the `() => 0` deterministic eval RNG becomes
//!     [`ZeroRng`].
//!   * `Map<number,number>` first-action tables -> dense `Vec<usize>`.
//!   * The TS greedy-eval block constructs an `evalAgent` and copies the trained
//!     Q into it via `(evalAgent as any).Q` — but that copy is never read (the
//!     rollout uses `Qtrained` + `options` directly). Per the "avoid as any"
//!     migration rule the dead `evalAgent` is omitted here. FLAGGED in the
//!     return notes.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;

use crate::des::general::des_base::argmax::{scan_arg_max_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::environment::{
    EnvironmentStation, EnvironmentStationOptions, PureEnvironment, StepResult, CH_ACTION, CH_STATE,
    CH_TRANSITION,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::semi_mdp::{Opt, SemiMDPAgentStation, SemiMDPCore, SemiMDPOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// SHARED / ZERO RNG
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

/// Deterministic RNG that always returns 0 (the TS `() => 0` eval policy RNG).
struct ZeroRng;

impl RandomSource for ZeroRng {
    fn next_float(&mut self) -> f64 {
        0.0
    }
}

// -----------------------------------------------------------------------------
// MAP DEFINITION
// -----------------------------------------------------------------------------

/// Hallway cells linking the rooms: top/bottom vertical, left/right horizontal.
const HALLWAYS: [(usize, usize); 4] = [(2, 5), (6, 5), (5, 1), (5, 8)];

/// Goal cell.
const GOAL: (usize, usize) = (10, 10);

/// Action displacements: 0=N, 1=E, 2=S, 3=W.
const DR: [i64; 4] = [-1, 0, 1, 0];
const DC: [i64; 4] = [0, 1, 0, -1];

fn rc_to_idx(r: usize, c: usize) -> usize {
    r * 11 + c
}

fn idx_to_rc(i: usize) -> (usize, usize) {
    (i / 11, i % 11)
}

/// Derived wall predicate equivalent to `FOUR_ROOMS_MAP[r][c] === 0`. Column 5
/// is a wall except at rows 2 and 6; row 5 is a wall except at columns 1 and 8;
/// the (5, 5) corner is a wall (covered by both clauses).
fn is_free(r: i64, c: i64) -> bool {
    if !(0..=10).contains(&r) || !(0..=10).contains(&c) {
        return false;
    }
    let wall = (c == 5 && r != 2 && r != 6) || (r == 5 && c != 1 && c != 8);
    !wall
}

/// Room id: 0=NW, 1=NE, 2=SW, 3=SE.
fn room(r: usize, c: usize) -> usize {
    (if r >= 5 { 2 } else { 0 }) + (if c >= 5 { 1 } else { 0 })
}

// -----------------------------------------------------------------------------
// ENVIRONMENT
// -----------------------------------------------------------------------------

/// Construction options for [`FourRoomsEnv`].
#[derive(Clone, Debug, Default)]
pub struct FourRoomsOpts {
    /// Probability the actuator slips perpendicularly. Default 0.
    pub slip: Option<f64>,
    /// Start state. Default top-left corner (0, 0).
    pub start_state: Option<usize>,
}

/// Four-rooms gridworld (`class FourRoomsEnv implements Environment`).
pub struct FourRoomsEnv {
    slip: f64,
    start: usize,
    rng: SharedRng,
}

impl FourRoomsEnv {
    /// Number of grid cells (states).
    pub const NUM_STATES: usize = 121;
    /// Number of primitive actions.
    pub const NUM_ACTIONS: usize = 4;

    fn new(opts: FourRoomsOpts, rng: SharedRng) -> Self {
        FourRoomsEnv {
            slip: opts.slip.unwrap_or(0.0),
            start: opts.start_state.unwrap_or(rc_to_idx(0, 0)),
            rng,
        }
    }
}

impl PureEnvironment<usize, usize> for FourRoomsEnv {
    fn num_states(&self) -> usize {
        FourRoomsEnv::NUM_STATES
    }

    fn num_actions(&self) -> usize {
        FourRoomsEnv::NUM_ACTIONS
    }

    fn reset(&mut self) -> usize {
        self.start
    }

    fn step(&mut self, state: usize, action: usize) -> StepResult<usize> {
        let (r, c) = idx_to_rc(state);
        let mut a_eff = action;
        if self.slip > 0.0 && self.rng.next_float() < self.slip {
            a_eff = if self.rng.next_float() < 0.5 {
                (action + 1) % 4
            } else {
                (action + 3) % 4
            };
        }
        let nr = r as i64 + DR[a_eff];
        let nc = c as i64 + DC[a_eff];
        let next_state = if is_free(nr, nc) {
            rc_to_idx(nr as usize, nc as usize)
        } else {
            state
        };
        let (gr, gc) = GOAL;
        let done = nr == gr as i64 && nc == gc as i64;
        let reward = if done { 1.0 } else { 0.0 };
        StepResult { next_state, reward, done }
    }

    fn render(&self, state: &usize) -> String {
        let (r, c) = idx_to_rc(*state);
        let mut lines: Vec<String> = Vec::new();
        for i in 0..11 {
            let mut line = String::new();
            for j in 0..11 {
                if i == r && j == c {
                    line.push('@');
                } else if !is_free(i as i64, j as i64) {
                    line.push('\u{2588}');
                } else if i == GOAL.0 && j == GOAL.1 {
                    line.push('G');
                } else {
                    line.push('.');
                }
            }
            lines.push(line);
        }
        lines.join("\n")
    }
}

// -----------------------------------------------------------------------------
// HALLWAY OPTIONS
// -----------------------------------------------------------------------------

/// Pre-compute the first-step-action lookup table to a hallway via BFS. For
/// each reachable free cell we record which primitive action takes one step
/// closer to the hallway. Unreachable cells default to action 0 (the TS
/// `lookup.get(s) ?? 0`).
fn hallway_first_action(hallway_idx: usize) -> Vec<usize> {
    let (hr, hc) = HALLWAYS[hallway_idx];
    let mut dist: Vec<i64> = vec![i64::MAX; 121];
    let mut parent: Vec<i64> = vec![-1; 121];
    let start = rc_to_idx(hr, hc);
    dist[start] = 0;
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(start);
    while let Some(cur) = queue.pop_front() {
        let (r, c) = idx_to_rc(cur);
        for a in 0..4 {
            let nr = r as i64 + DR[a];
            let nc = c as i64 + DC[a];
            if !is_free(nr, nc) {
                continue;
            }
            let ni = rc_to_idx(nr as usize, nc as usize);
            if dist[ni] == i64::MAX {
                dist[ni] = dist[cur] + 1;
                parent[ni] = cur as i64;
                queue.push_back(ni);
            }
        }
    }
    let mut m: Vec<usize> = vec![0; 121];
    for s in 0..121 {
        if s == start {
            m[s] = 0;
            continue;
        }
        if parent[s] < 0 {
            continue;
        }
        let (sr, sc) = idx_to_rc(s);
        let (pr, pc) = idx_to_rc(parent[s] as usize);
        let dr = pr as i64 - sr as i64;
        let dc = pc as i64 - sc as i64;
        let a = if dr == -1 {
            0
        } else if dc == 1 {
            1
        } else if dr == 1 {
            2
        } else if dc == -1 {
            3
        } else {
            0
        };
        m[s] = a;
    }
    m
}

/// A hallway-targeting option (`makeHallwayOption`).
struct HallwayOption {
    name: String,
    hr: usize,
    hc: usize,
    lookup: Vec<usize>,
    owner_rooms: HashSet<usize>,
}

impl Opt<usize, usize> for HallwayOption {
    fn name(&self) -> &str {
        &self.name
    }

    fn init(&self, s: &usize) -> bool {
        let (r, c) = idx_to_rc(*s);
        self.owner_rooms.contains(&room(r, c)) || (r == self.hr && c == self.hc)
    }

    fn policy(&self, s: &usize, _rng: &mut dyn RandomSource) -> usize {
        self.lookup[*s]
    }

    fn terminate(&self, s: &usize) -> f64 {
        let (r, c) = idx_to_rc(*s);
        if r == self.hr && c == self.hc {
            return 1.0;
        }
        if r == GOAL.0 && c == GOAL.1 {
            return 1.0;
        }
        if !self.owner_rooms.contains(&room(r, c)) {
            return 1.0;
        }
        0.0
    }
}

/// A primitive single-direction option (terminates after one step).
struct PrimitiveOption {
    name: String,
    action: usize,
}

impl Opt<usize, usize> for PrimitiveOption {
    fn name(&self) -> &str {
        &self.name
    }

    fn init(&self, _s: &usize) -> bool {
        true
    }

    fn policy(&self, _s: &usize, _rng: &mut dyn RandomSource) -> usize {
        self.action
    }

    fn terminate(&self, _s: &usize) -> f64 {
        1.0
    }
}

fn make_hallway_option(name: &str, hallway_idx: usize, owner_rooms: &[usize]) -> Box<dyn Opt<usize, usize>> {
    let (hr, hc) = HALLWAYS[hallway_idx];
    Box::new(HallwayOption {
        name: name.to_string(),
        hr,
        hc,
        lookup: hallway_first_action(hallway_idx),
        owner_rooms: owner_rooms.iter().copied().collect(),
    })
}

/// Eight hallway options (two per room) plus, optionally, four primitive
/// single-direction options.
pub fn build_four_rooms_options(include_primitive: bool) -> Vec<Box<dyn Opt<usize, usize>>> {
    let mut opts: Vec<Box<dyn Opt<usize, usize>>> = Vec::new();
    opts.push(make_hallway_option("NW->top", 0, &[0]));
    opts.push(make_hallway_option("NW->left", 2, &[0]));
    opts.push(make_hallway_option("NE->top", 0, &[1]));
    opts.push(make_hallway_option("NE->right", 3, &[1]));
    opts.push(make_hallway_option("SW->left", 2, &[2]));
    opts.push(make_hallway_option("SW->bottom", 1, &[2]));
    opts.push(make_hallway_option("SE->right", 3, &[3]));
    opts.push(make_hallway_option("SE->bottom", 1, &[3]));
    if include_primitive {
        let dirs = ["N", "E", "S", "W"];
        for (a, d) in dirs.iter().enumerate() {
            opts.push(Box::new(PrimitiveOption { name: format!("prim-{d}"), action: a }));
        }
    }
    opts
}

// -----------------------------------------------------------------------------
// SMDP Q-LEARNING AGENT
// -----------------------------------------------------------------------------

struct FourRoomsSMDPAgent {
    core: StationCore,
    agent: RLAgentCore,
    semi: SemiMDPCore<usize>,
    opt_lib: Vec<Box<dyn Opt<usize, usize>>>,
}

impl FourRoomsSMDPAgent {
    fn new(
        rng: Box<dyn RandomSource>,
        semi_opts: SemiMDPOptions,
        options: Vec<Box<dyn Opt<usize, usize>>>,
        init_q: f64,
    ) -> Self {
        let num_options = options.len();
        let semi = SemiMDPCore::new(semi_opts);
        // Optimistic Q init across all 121 states (drives exploration).
        *semi.q.borrow_mut() = vec![vec![init_q; num_options]; 121];
        FourRoomsSMDPAgent {
            core: StationCore::new("four-rooms-smdp"),
            agent: RLAgentCore::new(rng),
            semi,
            opt_lib: options,
        }
    }
}

impl DESStation for FourRoomsSMDPAgent {
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

impl RLAgentStation<usize, usize> for FourRoomsSMDPAgent {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }
    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        self.semi_pick_action(state, rng)
    }
    fn update(&mut self, _state: &usize, _action: &usize, reward: f64, next_state: &usize, done: bool) {
        self.semi_update(reward, next_state, done);
    }
    fn end_of_episode(&mut self, _episode_id: f64) {
        self.semi_end_of_episode();
    }
}

impl SemiMDPAgentStation<usize, usize> for FourRoomsSMDPAgent {
    fn semi_core(&self) -> &SemiMDPCore<usize> {
        &self.semi
    }
    fn semi_core_mut(&mut self) -> &mut SemiMDPCore<usize> {
        &mut self.semi
    }
    fn options(&self) -> &[Box<dyn Opt<usize, usize>>] {
        &self.opt_lib
    }
    fn state_key(&self, s: &usize) -> usize {
        *s
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Training options for [`run_four_rooms_smdp`]. `None` fields use TS defaults.
#[derive(Clone, Debug)]
pub struct FourRoomsTrainOpts {
    pub num_episodes: usize,
    pub max_steps_per_episode: Option<usize>,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub seed: Option<u32>,
    /// Slip probability of the env. Default 0.
    pub slip: Option<f64>,
    /// Include primitive 4-direction options. Default true.
    pub include_primitive: Option<bool>,
    /// Optimistic Q initial value. Default 1.0.
    pub init_q: Option<f64>,
}

impl Default for FourRoomsTrainOpts {
    fn default() -> Self {
        FourRoomsTrainOpts {
            num_episodes: 1,
            max_steps_per_episode: None,
            alpha: None,
            gamma: None,
            epsilon: None,
            epsilon_decay: None,
            epsilon_min: None,
            seed: None,
            slip: None,
            include_primitive: None,
            init_q: None,
        }
    }
}

/// Result of [`run_four_rooms_smdp`].
#[derive(Clone, Debug, PartialEq)]
pub struct FourRoomsResult {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    /// Steps used by the GREEDY policy from start to goal
    /// (`f64::INFINITY` if it never reaches the goal within the step budget).
    pub greedy_episode_length: f64,
    pub greedy_reached_goal: bool,
    pub final_epsilon: f64,
    pub ticks: usize,
}

/// Train an intra-option SMDP Q-learning agent on the four-rooms task, then roll
/// out the greedy option policy from the start.
pub fn run_four_rooms_smdp(opts: FourRoomsTrainOpts) -> FourRoomsResult {
    let cls = "run_four_rooms_smdp";
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
    if let Some(s) = opts.slip {
        Preconditions::in_range(cls, "slip", s, 0.0, 1.0).unwrap();
    }
    if let Some(q) = opts.init_q {
        Preconditions::finite(cls, "initQ", q).unwrap();
    }

    let seed = opts.seed.unwrap_or(1);
    let rng = SharedRng::new(seed);
    let env = FourRoomsEnv::new(FourRoomsOpts { slip: Some(opts.slip.unwrap_or(0.0)), start_state: None }, rng.clone());
    let options = build_four_rooms_options(opts.include_primitive.unwrap_or(true));
    let agent = FourRoomsSMDPAgent::new(
        Box::new(rng.clone()),
        SemiMDPOptions {
            alpha: Some(opts.alpha.unwrap_or(0.25)),
            gamma: Some(opts.gamma.unwrap_or(0.99)),
            epsilon: Some(opts.epsilon.unwrap_or(0.1)),
            epsilon_decay: Some(opts.epsilon_decay.unwrap_or(1.0)),
            epsilon_min: Some(opts.epsilon_min.unwrap_or(0.01)),
        },
        options,
        opts.init_q.unwrap_or(1.0),
    );

    let max_steps = opts.max_steps_per_episode.unwrap_or(5000);
    let env_station = EnvironmentStation::<usize, usize>::new(
        "env",
        Box::new(env),
        EnvironmentStationOptions {
            num_episodes: Some(opts.num_episodes as f64),
            max_steps_per_episode: Some(max_steps),
        },
    );

    let agent: Rc<RefCell<FourRoomsSMDPAgent>> = Rc::new(RefCell::new(agent));
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

    // Greedy rollout from start. (The TS `evalAgent` whose Q is copied but never
    // read is omitted — see module docs.)
    let mut eval_env = FourRoomsEnv::new(FourRoomsOpts { slip: Some(0.0), start_state: None }, rng.clone());
    let mut s = eval_env.reset();
    let mut len = 0usize;
    let mut reached = false;
    let q_trained = agent.borrow().get_q();
    let options_eval = build_four_rooms_options(opts.include_primitive.unwrap_or(true));
    let mut cur_option: i64 = -1;
    let mut eval_rng = SeededRandom::new(seed.wrapping_add(17));
    for _ in 0..max_steps {
        if cur_option < 0 || options_eval[cur_option as usize].terminate(&s) >= 1.0 {
            let s_cur = s;
            cur_option = scan_arg_max_tie_break(
                options_eval.len(),
                |i| {
                    if options_eval[i].init(&s_cur) {
                        q_trained[s_cur][i]
                    } else {
                        f64::NEG_INFINITY
                    }
                },
                &mut eval_rng,
                ARGMAX_EPS_DEFAULT,
            )
            .map(|x| x as i64)
            .unwrap_or(-1);
            if cur_option < 0 {
                cur_option = 0;
            }
        }
        let a = options_eval[cur_option as usize].policy(&s, &mut ZeroRng);
        let r = eval_env.step(s, a);
        len += 1;
        if r.done {
            reached = true;
            break;
        }
        s = r.next_state;
    }

    let reward_history = agent.borrow().reward_history().to_vec();
    let length_history = agent.borrow().length_history().to_vec();
    let final_epsilon = agent.borrow().get_epsilon();

    FourRoomsResult {
        reward_history,
        length_history,
        greedy_episode_length: if reached { len as f64 } else { f64::INFINITY },
        greedy_reached_goal: reached,
        final_epsilon,
        ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! Env reset/step/wall semantics + SMDP episode termination.

    use super::*;

    #[test]
    fn reset_and_wall_clamp() {
        let mut env = FourRoomsEnv::new(FourRoomsOpts::default(), SharedRng::new(1));
        assert_eq!(env.reset(), rc_to_idx(0, 0));
        // From (0, 4) moving East steps into the col-5 wall -> stays put.
        let s = rc_to_idx(0, 4);
        let r = env.step(s, 1);
        assert_eq!(r.next_state, s, "wall clamp keeps the agent in place");
        assert!(!r.done);
        assert_eq!(r.reward, 0.0);
    }

    #[test]
    fn stepping_into_goal_terminates() {
        let mut env = FourRoomsEnv::new(FourRoomsOpts::default(), SharedRng::new(2));
        // From (10, 9) moving East reaches the goal (10, 10).
        let s = rc_to_idx(10, 9);
        let r = env.step(s, 1);
        assert_eq!(r.next_state, rc_to_idx(10, 10));
        assert!(r.done);
        assert_eq!(r.reward, 1.0);
    }

    #[test]
    fn smdp_training_runs_and_greedy_reaches_goal() {
        let res = run_four_rooms_smdp(FourRoomsTrainOpts {
            num_episodes: 800,
            seed: Some(1),
            ..Default::default()
        });
        assert_eq!(res.reward_history.len(), 800);
        assert_eq!(res.length_history.len(), 800);
        assert!(res.ticks > 0);
        // With hallway options the greedy policy should solve the task.
        assert!(res.greedy_reached_goal, "greedy option policy reaches the goal");
        assert!(res.greedy_episode_length.is_finite());
    }
}
