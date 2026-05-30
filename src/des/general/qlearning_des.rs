//! Port of `src/des/general/qlearning-des.ts` — Q-learning as a DES.
//!
//! A concrete leaf agent on the [`RLAgentStation`] template-method base: it
//! implements ε-greedy `pick_action` and the textbook off-policy TD(0) update
//!
//!   Q[s,a] ← Q[s,a] + α · ( r + γ · max_a' Q[s',a'] − Q[s,a] )
//!
//! Topology: a [`QLearningAgent`] exchanges actions / states / transitions with
//! an [`EnvironmentStation`] wrapping a pure environment.
//!
//! ## TS → Rust mapping
//!
//!   * `interface QLearningOptions` / `QLearningResult` → structs.
//!   * `class QLearningAgent extends RLAgentStation<number, number>` → a struct
//!     embedding [`StationCore`] + [`RLAgentCore`], delegating
//!     `DESStation::run_time_step` → `rl_agent_run_time_step` and implementing
//!     the [`RLAgentStation`] hooks (`pick_action`, `update`, `end_of_episode`).
//!     State/action indices are `usize`; rewards/values are `f64`.
//!   * `fn runQLearningDES` → the free fn [`run_qlearning_des`].
//!   * the injected `rng: () => number` (a `mulberry32` closure shared by the
//!     agent AND the runner) → a single [`SeededRandom`] behind an
//!     `Rc<RefCell<…>>`, bridged into both via the [`SharedRng`] newtype so the
//!     agent's `pick_action` draws and the runner's tick-shuffle draws come from
//!     the SAME stream (faithful to the TS shared closure).

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::environment::{
    self, EnvironmentStation, EnvironmentStationOptions, PureEnvironment,
};
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation, RngRef};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
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

/// `interface QLearningOptions`. The injected `rng` is a separate constructor
/// argument (matching the sibling RL bases); `q_init` is an optional Q-table
/// initialiser threaded the injected RNG.
pub struct QLearningOptions {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: Option<f64>,
    /// Per-EPISODE multiplicative decay of ε.
    pub epsilon_decay: Option<f64>,
    pub num_states: usize,
    pub num_actions: usize,
    /// Optional Q-init function (default: zeros).
    #[allow(clippy::type_complexity)]
    pub q_init: Option<Box<dyn Fn(usize, usize, &mut dyn RandomSource) -> f64>>,
}

/// Tabular off-policy Q-learning agent driven as a DES.
pub struct QLearningAgent {
    core: StationCore,
    agent: RLAgentCore,
    /// `q[state][action]`.
    pub q: Vec<Vec<f64>>,
    pub num_states: usize,
    pub num_actions: usize,
    alpha: f64,
    gamma: f64,
    epsilon_min: Option<f64>,
    epsilon_decay: Option<f64>,
    current_epsilon: f64,
}

impl QLearningAgent {
    /// Mirrors `new QLearningAgent(id, {...opts, rng})`. The `rng` is threaded
    /// through `q_init` at construction and then stored in [`RLAgentCore`].
    pub fn new(id: impl Into<String>, opts: QLearningOptions, mut rng: Box<dyn RandomSource>) -> Self {
        let num_states = opts.num_states;
        let num_actions = opts.num_actions;
        let q: Vec<Vec<f64>> = (0..num_states)
            .map(|s| {
                (0..num_actions)
                    .map(|a| match &opts.q_init {
                        Some(f) => f(s, a, &mut *rng),
                        None => 0.0,
                    })
                    .collect()
            })
            .collect();
        QLearningAgent {
            core: StationCore::new(id),
            agent: RLAgentCore::new(rng),
            q,
            num_states,
            num_actions,
            alpha: opts.alpha,
            gamma: opts.gamma,
            epsilon_min: opts.epsilon_min,
            epsilon_decay: opts.epsilon_decay,
            current_epsilon: opts.epsilon,
        }
    }

    /// Greedy action per state (random tie-break, drawing the agent's RNG).
    pub fn greedy_policy(&mut self) -> Vec<usize> {
        let mut rng = self.agent.rng.take().expect("rng already in use");
        let policy = self
            .q
            .iter()
            .map(|row| arg_max_with_tie_break(row, &mut RngRef(&mut *rng), ARGMAX_EPS_DEFAULT).unwrap_or(0))
            .collect();
        self.agent.rng = Some(rng);
        policy
    }

    pub fn get_epsilon(&self) -> f64 {
        self.current_epsilon
    }
}

impl DESStation for QLearningAgent {
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

impl RLAgentStation<usize, usize> for QLearningAgent {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }

    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        if rng.next_float() < self.current_epsilon {
            return (rng.next_float() * self.num_actions as f64).floor() as usize;
        }
        // Random tie-breaking on argmax: zero-init Q-tables tie initially, and a
        // deterministic `>` would always pick action 0 under ε=0 evaluation.
        arg_max_with_tie_break(&self.q[*state], &mut RngRef(rng), ARGMAX_EPS_DEFAULT).unwrap_or(0)
    }

    fn update(&mut self, state: &usize, action: &usize, reward: f64, next_state: &usize, done: bool) {
        let qsa = self.q[*state][*action];
        let best_next = if done {
            0.0
        } else {
            self.q[*next_state].iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        let target = reward + if done { 0.0 } else { self.gamma * best_next };
        self.q[*state][*action] = qsa + self.alpha * (target - qsa);
    }

    fn end_of_episode(&mut self, _episode_id: f64) {
        if let Some(decay) = self.epsilon_decay {
            self.current_epsilon = self.epsilon_min.unwrap_or(0.0).max(self.current_epsilon * decay);
        }
    }
}

// -----------------------------------------------------------------------------
// PUBLIC DRIVER
// -----------------------------------------------------------------------------

/// `interface QLearningResult`.
#[derive(Clone, Debug, Default)]
pub struct QLearningResult {
    pub q: Vec<Vec<f64>>,
    pub policy: Vec<usize>,
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub total_episodes: usize,
    pub total_steps: u64,
    pub total_ticks: usize,
}

/// Options bag for [`run_qlearning_des`] (the TS `opts` object). `des_options`
/// is `None` to use defaults.
#[derive(Default)]
pub struct RunQLearningOptions {
    pub num_episodes: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub max_steps_per_episode: Option<usize>,
    pub seed: Option<u32>,
    pub des_options: Option<IterativeRunOptions>,
}

/// Wire a [`QLearningAgent`] to an [`EnvironmentStation`] and run the DES.
pub fn run_qlearning_des(
    env: Box<dyn PureEnvironment<usize, usize>>,
    opts: RunQLearningOptions,
) -> QLearningResult {
    let num_states = env.num_states();
    let num_actions = env.num_actions();
    let shared = Rc::new(RefCell::new(mulberry32(opts.seed.unwrap_or(1))));

    let agent = QLearningAgent::new(
        "q-agent",
        QLearningOptions {
            alpha: opts.alpha,
            gamma: opts.gamma,
            epsilon: opts.epsilon,
            epsilon_min: opts.epsilon_min,
            epsilon_decay: opts.epsilon_decay,
            num_states,
            num_actions,
            q_init: None,
        },
        Box::new(SharedRng(shared.clone())),
    );
    let agent = Rc::new(RefCell::new(agent));

    let env_st = EnvironmentStation::new(
        "env",
        env,
        EnvironmentStationOptions {
            num_episodes: Some(opts.num_episodes),
            max_steps_per_episode: opts.max_steps_per_episode,
        },
    );
    let env_st = Rc::new(RefCell::new(env_st));

    // Wire channels: env → agent (state, transition), agent → env (action).
    env_st.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_STATE,
        <QLearningAgent as RLAgentStation<usize, usize>>::CH_STATE,
    );
    env_st.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_TRANSITION,
        <QLearningAgent as RLAgentStation<usize, usize>>::CH_TRANSITION,
    );
    agent.borrow_mut().core_mut().pipe(
        env_st.clone() as StationRef,
        <QLearningAgent as RLAgentStation<usize, usize>>::CH_ACTION,
        environment::CH_ACTION,
    );

    let mut des_options = opts.des_options.unwrap_or_default();
    if des_options.rng.is_none() {
        let r = shared.clone();
        des_options.rng = Some(Box::new(move || r.borrow_mut().next_float()));
    }
    let summary = run_iterative_des(
        vec![env_st.clone() as StationRef, agent.clone() as StationRef],
        des_options,
    );

    let q = agent.borrow().q.clone();
    let policy = agent.borrow_mut().greedy_policy();
    let reward_history = agent.borrow().reward_history().to_vec();
    let length_history = agent.borrow().length_history().to_vec();
    let total_steps = agent.borrow().total_steps();
    QLearningResult {
        total_episodes: reward_history.len(),
        q,
        policy,
        reward_history,
        length_history,
        total_steps,
        total_ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! Q-learning over a small contextual bandit DES learns the better action.
    //!
    //! The bandit has two states and two actions; action 1 always pays +1 and
    //! action 0 pays 0, every step terminal. The greedy policy must converge to
    //! action 1 in both states, and the running reward should rise over training.
    use super::*;
    use crate::des::general::des_base::environment::StepResult;

    /// Two-state contextual bandit: each step is terminal, action 1 pays +1.
    struct Bandit {
        next: usize,
    }

    impl PureEnvironment<usize, usize> for Bandit {
        fn num_states(&self) -> usize {
            2
        }
        fn num_actions(&self) -> usize {
            2
        }
        fn reset(&mut self) -> usize {
            // Alternate the start state so both rows of Q get trained.
            self.next = 1 - self.next;
            self.next
        }
        fn step(&mut self, _state: usize, action: usize) -> StepResult<usize> {
            StepResult { next_state: 0, reward: if action == 1 { 1.0 } else { 0.0 }, done: true }
        }
    }

    #[test]
    fn learns_best_action_in_both_states() {
        let res = run_qlearning_des(
            Box::new(Bandit { next: 0 }),
            RunQLearningOptions {
                num_episodes: 400.0,
                alpha: 0.2,
                gamma: 0.9,
                epsilon: 0.2,
                epsilon_min: Some(0.0),
                epsilon_decay: Some(0.99),
                max_steps_per_episode: Some(1),
                seed: Some(7),
                des_options: None,
            },
        );
        for state in 0..2usize {
            assert!(
                res.q[state][1] > res.q[state][0],
                "state {state}: Q = {:?}",
                res.q[state]
            );
            assert_eq!(res.policy[state], 1);
        }
        assert_eq!(res.total_episodes, 400);
    }

    #[test]
    fn epsilon_decays_each_episode() {
        // ε starts at 0.5, decays ×0.5/episode toward a floor of 0.05.
        let mut agent = QLearningAgent::new(
            "q",
            QLearningOptions {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.5,
                epsilon_min: Some(0.05),
                epsilon_decay: Some(0.5),
                num_states: 1,
                num_actions: 2,
                q_init: None,
            },
            Box::new(SeededRandom::new(1)),
        );
        let before = agent.get_epsilon();
        agent.end_of_episode(0.0);
        let after = agent.get_epsilon();
        assert!(after < before);
        // Decay clamps at the floor after enough episodes.
        for _ in 0..20 {
            agent.end_of_episode(0.0);
        }
        assert!((agent.get_epsilon() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn reward_history_improves_over_training() {
        let res = run_qlearning_des(
            Box::new(Bandit { next: 0 }),
            RunQLearningOptions {
                num_episodes: 600.0,
                alpha: 0.3,
                gamma: 0.9,
                epsilon: 0.3,
                epsilon_min: Some(0.0),
                epsilon_decay: Some(0.99),
                max_steps_per_episode: Some(1),
                seed: Some(3),
                des_options: None,
            },
        );
        let h = &res.reward_history;
        let window = 100;
        let first: f64 = h[..window].iter().sum::<f64>() / window as f64;
        let last: f64 = h[h.len() - window..].iter().sum::<f64>() / window as f64;
        assert!(last > first, "mean reward should rise: first {first}, last {last}");
    }
}
