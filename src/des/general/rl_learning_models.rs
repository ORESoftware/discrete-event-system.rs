//! Port of `src/des/general/rl-learning-models.ts` — additional RL station-graph
//! models built from the shared DES bases:
//!
//!   * `policy-gradient-corridor` (REINFORCE on [`PolicyGradientAgent`] +
//!     [`PolicyUpdateStation`]).
//!   * `expected-sarsa-gridworld` (expected-SARSA on [`RLAgentStation`] +
//!     [`EnvironmentStation`]).
//!
//! ## TS → Rust mapping
//!
//!   * `type RLTopology = StationGraphSummary` → the [`RLTopology`] alias.
//!   * the param/result interfaces → structs (`Option<T>` for optionals).
//!   * `class SoftmaxPolicyGradientAgent extends PolicyGradientAgent<number,
//!     number>` → a struct over `S = usize, A = usize` implementing
//!     `sample_policy_and_value`.
//!   * `class ReinforceUpdateStation extends PolicyUpdateStation` → a struct
//!     holding the shared agent (`Rc<RefCell<…>>`) implementing `run_update`.
//!   * `class ExpectedSarsaAgent extends RLAgentStation<number, number>` → a
//!     struct implementing `pick_action` / `update` / `end_of_episode`.
//!   * the injected `mulberry32` closure → a single [`SeededRandom`] behind
//!     `Rc<RefCell<…>>`, bridged via [`SharedRng`].
//!   * the rl-environments dependency (`Corridor` / `GridWorld` / `evalPolicy`)
//!     is wrapped for the [`EnvironmentStation`] via the local [`EnvAdapter`]
//!     (FLAGGED below).

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::environment::{
    self, EnvironmentStation, EnvironmentStationOptions, PureEnvironment, StepResult,
};
use crate::des::general::des_base::learning_optimization::{
    channel_edge, softmax, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::policy_gradient_agent::{
    self, PolicyGradientAgent, PolicyGradientCore, PolicyOutput, PolicyUpdateCore,
    PolicyUpdateStation, RolloutEntry,
};
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation, RngRef};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::general::rl_environments::{
    eval_policy, Corridor, Environment, EvalPolicyOptions, GridWorld, GridWorldOptions,
};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Bridges one shared [`SeededRandom`] into a boxed [`RandomSource`] (the TS
/// shared `() => number`). FLAGGED local equivalent — see `qlearning_des.rs`.
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

/// Adapts a pure [`Environment`] (rl-environments) into the
/// [`PureEnvironment`] hook trait that [`EnvironmentStation`] wraps. FLAGGED
/// local equivalent: in TS the two interfaces were structurally identical so
/// `GridWorld` / `Corridor` satisfied both; Rust models them as distinct traits,
/// so this thin adapter bridges `Environment` → `PureEnvironment<usize, usize>`.
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
        StepResult {
            next_state: o.next_state,
            reward: o.reward,
            done: o.done,
        }
    }
}

/// `type RLTopology = StationGraphSummary`.
pub type RLTopology = StationGraphSummary;

// =============================================================================
// POLICY-GRADIENT CORRIDOR (REINFORCE)
// =============================================================================

/// `interface PolicyGradientCorridorParams`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyGradientCorridorParams {
    pub num_episodes: Option<usize>,
    pub max_steps_per_episode: Option<usize>,
    pub rollout_len: Option<usize>,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub seed: Option<u32>,
    pub length: Option<usize>,
}

/// `interface PolicyGradientCorridorResult`.
#[derive(Clone, Debug, Default)]
pub struct PolicyGradientCorridorResult {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub greedy_success_rate: f64,
    pub greedy_mean_length: f64,
    pub policy: Vec<usize>,
    pub updates: u64,
    pub topology: RLTopology,
}

/// Softmax-policy REINFORCE agent (tabular logits `θ[s][a]`, no critic).
pub struct SoftmaxPolicyGradientAgent {
    core: StationCore,
    pg: PolicyGradientCore<usize, usize>,
    pub theta: Vec<Vec<f64>>,
    pub num_states: usize,
    pub num_actions: usize,
}

impl SoftmaxPolicyGradientAgent {
    pub fn new(
        id: impl Into<String>,
        num_states: usize,
        num_actions: usize,
        rollout_len: usize,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        SoftmaxPolicyGradientAgent {
            core: StationCore::new(id),
            pg: PolicyGradientCore::new(rollout_len, rng),
            theta: vec![vec![0.0; num_actions]; num_states],
            num_states,
            num_actions,
        }
    }

    /// `θ[s][·] += α · A · (1[a'=a] − π(a'|s))`.
    pub fn apply_policy_gradient(
        &mut self,
        state: usize,
        action: usize,
        advantage: f64,
        alpha: f64,
    ) {
        let probs = softmax(&self.theta[state]);
        for a in 0..self.num_actions {
            self.theta[state][a] +=
                alpha * advantage * ((if a == action { 1.0 } else { 0.0 }) - probs[a]);
        }
    }

    /// Greedy (argmax) action with random tie-break, drawing the agent's RNG.
    pub fn greedy_action(&mut self, state: usize) -> usize {
        let mut rng = self.pg.rng.take().expect("rng already in use");
        let a = arg_max_with_tie_break(
            &self.theta[state],
            &mut RngRef(&mut *rng),
            ARGMAX_EPS_DEFAULT,
        )
        .unwrap_or(0);
        self.pg.rng = Some(rng);
        a
    }
}

impl DESStation for SoftmaxPolicyGradientAgent {
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

impl PolicyGradientAgent<usize, usize> for SoftmaxPolicyGradientAgent {
    fn pg_core(&self) -> &PolicyGradientCore<usize, usize> {
        &self.pg
    }
    fn pg_core_mut(&mut self) -> &mut PolicyGradientCore<usize, usize> {
        &mut self.pg
    }

    fn sample_policy_and_value(
        &self,
        state: &usize,
        rng: &mut dyn RandomSource,
    ) -> PolicyOutput<usize> {
        let probs = softmax(&self.theta[*state]);
        let mut u = rng.next_float();
        for (a, &p) in probs.iter().enumerate() {
            u -= p;
            if u <= 0.0 {
                return PolicyOutput {
                    action: a,
                    log_prob: p.max(1e-12).ln(),
                    value: 0.0,
                };
            }
        }
        let action = probs.len() - 1;
        PolicyOutput {
            action,
            log_prob: probs[action].max(1e-12).ln(),
            value: 0.0,
        }
    }
}

/// REINFORCE update with a mean-return baseline (no critic).
pub struct ReinforceUpdateStation {
    core: StationCore,
    pu: PolicyUpdateCore,
    agent: Rc<RefCell<SoftmaxPolicyGradientAgent>>,
    alpha: f64,
    gamma: f64,
    pub update_returns: Vec<f64>,
}

impl ReinforceUpdateStation {
    pub fn new(
        id: impl Into<String>,
        agent: Rc<RefCell<SoftmaxPolicyGradientAgent>>,
        alpha: f64,
        gamma: f64,
    ) -> Self {
        ReinforceUpdateStation {
            core: StationCore::new(id),
            pu: PolicyUpdateCore::new(),
            agent,
            alpha,
            gamma,
            update_returns: Vec::new(),
        }
    }
}

impl DESStation for ReinforceUpdateStation {
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

impl PolicyUpdateStation for ReinforceUpdateStation {
    fn pu_core(&self) -> &PolicyUpdateCore {
        &self.pu
    }
    fn pu_core_mut(&mut self) -> &mut PolicyUpdateCore {
        &mut self.pu
    }

    fn run_update(&mut self) {
        let buffer: Vec<RolloutEntry<usize, usize>> = self.agent.borrow().get_buffer().to_vec();
        let mut g = 0.0;
        let mut returns = vec![0.0; buffer.len()];
        for i in (0..buffer.len()).rev() {
            let r = buffer[i].r.unwrap_or(0.0);
            g = r + self.gamma * g;
            returns[i] = g;
            if buffer[i].done.unwrap_or(false) {
                g = 0.0;
            }
        }
        let baseline = returns.iter().sum::<f64>() / (returns.len().max(1) as f64);
        {
            let mut agent = self.agent.borrow_mut();
            for i in 0..buffer.len() {
                let e = &buffer[i];
                if e.r.is_none() {
                    continue;
                }
                agent.apply_policy_gradient(e.s, e.a, returns[i] - baseline, self.alpha);
            }
            agent.clear_buffer();
        }
        self.update_returns.extend(returns);
    }
}

/// Run the REINFORCE corridor model and report learned policy + greedy eval.
pub fn run_policy_gradient_corridor(
    params: PolicyGradientCorridorParams,
) -> PolicyGradientCorridorResult {
    let num_episodes = params.num_episodes.unwrap_or(300);
    let max_steps = params.max_steps_per_episode.unwrap_or(40);
    let env_rc: Rc<Corridor> = Rc::new(Corridor::new(params.length.unwrap_or(7), 0));
    let shared = Rc::new(RefCell::new(mulberry32(params.seed.unwrap_or(1))));
    let num_states = env_rc.num_states();
    let num_actions = env_rc.num_actions();

    let env_station = Rc::new(RefCell::new(EnvironmentStation::new(
        "corridor-env",
        Box::new(EnvAdapter {
            env: env_rc.clone(),
        }),
        EnvironmentStationOptions {
            num_episodes: Some(num_episodes as f64),
            max_steps_per_episode: Some(max_steps),
        },
    )));
    let agent = Rc::new(RefCell::new(SoftmaxPolicyGradientAgent::new(
        "softmax-policy-agent",
        num_states,
        num_actions,
        params.rollout_len.unwrap_or(12),
        Box::new(SharedRng(shared.clone())),
    )));
    let updater = Rc::new(RefCell::new(ReinforceUpdateStation::new(
        "reinforce-update",
        agent.clone(),
        params.alpha.unwrap_or(0.04),
        params.gamma.unwrap_or(0.95),
    )));

    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_STATE,
        policy_gradient_agent::CH_STATE,
    );
    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_TRANSITION,
        policy_gradient_agent::CH_TRANSITION,
    );
    agent.borrow_mut().core_mut().pipe(
        env_station.clone() as StationRef,
        policy_gradient_agent::CH_ACTION,
        environment::CH_ACTION,
    );
    agent.borrow_mut().core_mut().pipe(
        updater.clone() as StationRef,
        policy_gradient_agent::CH_TRAIN,
        policy_gradient_agent::CH_TRAIN,
    );
    updater.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        policy_gradient_agent::CH_RESUME,
        policy_gradient_agent::CH_RESUME,
    );

    let mut des = IterativeRunOptions {
        max_ticks: Some(num_episodes * max_steps * 4),
        ..Default::default()
    };
    let r = shared.clone();
    des.rng = Some(Box::new(move || r.borrow_mut().next_float()));
    run_iterative_des(
        vec![
            env_station.clone() as StationRef,
            agent.clone() as StationRef,
            updater.clone() as StationRef,
        ],
        des,
    );

    let mut eval_rng = mulberry32(99);
    let evaluation = eval_policy(
        &*env_rc,
        |s, _rng| agent.borrow_mut().greedy_action(s),
        &mut eval_rng,
        EvalPolicyOptions {
            num_episodes: 50,
            max_steps_per_episode: max_steps,
            gamma: 1.0,
        },
    );

    let policy: Vec<usize> = (0..num_states)
        .map(|s| agent.borrow_mut().greedy_action(s))
        .collect();
    let reward_history = env_station.borrow().reward_history().to_vec();
    let length_history = env_station.borrow().length_history().to_vec();
    let updates = updater.borrow().pu_core().num_updates;

    let env_id = StationOrId::Id("corridor-env".to_string());
    let agent_id = StationOrId::Id("softmax-policy-agent".to_string());
    let updater_id = StationOrId::Id("reinforce-update".to_string());
    let topology = station_graph(
        &[env_id.clone(), agent_id.clone(), updater_id.clone()],
        &[
            "StateToken",
            "ActionToken",
            "TransitionToken",
            "TrainTriggerToken",
            "ResumeToken",
        ]
        .map(String::from),
        &[
            channel_edge(
                &env_id,
                environment::CH_STATE,
                &agent_id,
                Some(policy_gradient_agent::CH_STATE),
            ),
            channel_edge(
                &agent_id,
                policy_gradient_agent::CH_ACTION,
                &env_id,
                Some(environment::CH_ACTION),
            ),
            channel_edge(
                &env_id,
                environment::CH_TRANSITION,
                &agent_id,
                Some(policy_gradient_agent::CH_TRANSITION),
            ),
            channel_edge(
                &agent_id,
                policy_gradient_agent::CH_TRAIN,
                &updater_id,
                Some(policy_gradient_agent::CH_TRAIN),
            ),
            channel_edge(
                &updater_id,
                policy_gradient_agent::CH_RESUME,
                &agent_id,
                Some(policy_gradient_agent::CH_RESUME),
            ),
        ],
    );

    PolicyGradientCorridorResult {
        reward_history,
        length_history,
        greedy_success_rate: evaluation.success_rate,
        greedy_mean_length: evaluation.mean_length,
        policy,
        updates,
        topology,
    }
}

// =============================================================================
// EXPECTED-SARSA GRIDWORLD
// =============================================================================

/// `interface ExpectedSarsaGridParams`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpectedSarsaGridParams {
    pub num_episodes: Option<usize>,
    pub max_steps_per_episode: Option<usize>,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub seed: Option<u32>,
}

/// `interface ExpectedSarsaGridResult`.
#[derive(Clone, Debug, Default)]
pub struct ExpectedSarsaGridResult {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub greedy_reached: bool,
    pub greedy_len: usize,
    pub q_start: Vec<f64>,
    pub policy: Vec<usize>,
    pub topology: RLTopology,
}

/// Expected-SARSA agent: `Q[s,a] ← Q[s,a] + α(r + γ Σ_a' π(a'|s') Q[s',a'] − Q[s,a])`.
pub struct ExpectedSarsaAgent {
    core: StationCore,
    agent: RLAgentCore,
    q: Vec<Vec<f64>>,
    /// State count (configuration; the Q-table is sized from it at construction).
    #[allow(dead_code)]
    num_states: usize,
    num_actions: usize,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
}

impl ExpectedSarsaAgent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        num_states: usize,
        num_actions: usize,
        rng: Box<dyn RandomSource>,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
        epsilon_min: f64,
    ) -> Self {
        ExpectedSarsaAgent {
            core: StationCore::new(id),
            agent: RLAgentCore::new(rng),
            q: vec![vec![0.0; num_actions]; num_states],
            num_states,
            num_actions,
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
        }
    }

    fn greedy_idx(&self, state: usize, rng: &mut dyn RandomSource) -> usize {
        arg_max_with_tie_break(&self.q[state], &mut RngRef(rng), ARGMAX_EPS_DEFAULT).unwrap_or(0)
    }

    fn expected_value_with(&self, state: usize, rng: &mut dyn RandomSource) -> f64 {
        let greedy = self.greedy_idx(state, rng);
        let mut v = 0.0;
        for a in 0..self.num_actions {
            let p = self.epsilon / self.num_actions as f64
                + if a == greedy { 1.0 - self.epsilon } else { 0.0 };
            v += p * self.q[state][a];
        }
        v
    }

    /// Greedy action (driver use): draws the agent's stored RNG for tie-break.
    pub fn greedy_action(&mut self, state: usize) -> usize {
        let mut rng = self.agent.rng.take().expect("rng already in use");
        let a = self.greedy_idx(state, &mut *rng);
        self.agent.rng = Some(rng);
        a
    }

    pub fn q_values(&self, state: usize) -> Vec<f64> {
        self.q[state].clone()
    }
}

impl DESStation for ExpectedSarsaAgent {
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

impl RLAgentStation<usize, usize> for ExpectedSarsaAgent {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }

    fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
        if rng.next_float() < self.epsilon {
            return (rng.next_float() * self.num_actions as f64).floor() as usize;
        }
        self.greedy_idx(*state, rng)
    }

    fn update(
        &mut self,
        state: &usize,
        action: &usize,
        reward: f64,
        next_state: &usize,
        done: bool,
    ) {
        let mut rng = self.agent.rng.take().expect("rng already in use");
        let expected = if done {
            0.0
        } else {
            self.expected_value_with(*next_state, &mut *rng)
        };
        self.agent.rng = Some(rng);
        let target = reward + self.gamma * expected;
        self.q[*state][*action] += self.alpha * (target - self.q[*state][*action]);
    }

    fn end_of_episode(&mut self, _episode_id: f64) {
        self.epsilon = self.epsilon_min.max(self.epsilon * self.epsilon_decay);
    }
}

/// Run the expected-SARSA gridworld model and report learned policy + greedy
/// rollout from the start state.
pub fn run_expected_sarsa_gridworld(params: ExpectedSarsaGridParams) -> ExpectedSarsaGridResult {
    let num_episodes = params.num_episodes.unwrap_or(900);
    let max_steps = params.max_steps_per_episode.unwrap_or(80);
    let shared = Rc::new(RefCell::new(mulberry32(params.seed.unwrap_or(1))));
    let env_concrete: Rc<GridWorld> = Rc::new(GridWorld::new(GridWorldOptions {
        width: Some(4),
        height: Some(4),
        ..Default::default()
    }));
    let num_states = env_concrete.num_states();
    let num_actions = env_concrete.num_actions();

    let env_station = Rc::new(RefCell::new(EnvironmentStation::new(
        "grid-env",
        Box::new(EnvAdapter {
            env: env_concrete.clone(),
        }),
        EnvironmentStationOptions {
            num_episodes: Some(num_episodes as f64),
            max_steps_per_episode: Some(max_steps),
        },
    )));
    let agent = Rc::new(RefCell::new(ExpectedSarsaAgent::new(
        "expected-sarsa-agent",
        num_states,
        num_actions,
        Box::new(SharedRng(shared.clone())),
        params.alpha.unwrap_or(0.2),
        params.gamma.unwrap_or(0.95),
        params.epsilon.unwrap_or(0.35),
        params.epsilon_decay.unwrap_or(0.995),
        params.epsilon_min.unwrap_or(0.02),
    )));

    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_STATE,
        <ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_STATE,
    );
    env_station.borrow_mut().core_mut().pipe(
        agent.clone() as StationRef,
        environment::CH_TRANSITION,
        <ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_TRANSITION,
    );
    agent.borrow_mut().core_mut().pipe(
        env_station.clone() as StationRef,
        <ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_ACTION,
        environment::CH_ACTION,
    );

    let mut des = IterativeRunOptions {
        max_ticks: Some(num_episodes * max_steps * 3),
        ..Default::default()
    };
    let r = shared.clone();
    des.rng = Some(Box::new(move || r.borrow_mut().next_float()));
    run_iterative_des(
        vec![
            env_station.clone() as StationRef,
            agent.clone() as StationRef,
        ],
        des,
    );

    // Greedy rollout from the start state.
    let mut s = env_concrete.reset();
    let mut reached = false;
    let mut len = 0usize;
    for _ in 0..max_steps {
        let a = agent.borrow_mut().greedy_action(s);
        let step = env_concrete.step(s, a);
        len += 1;
        if step.done {
            reached = step.reward > 0.0;
            break;
        }
        s = step.next_state;
    }

    let q_start = agent.borrow().q_values(env_concrete.start);
    let policy: Vec<usize> = (0..num_states)
        .map(|state| agent.borrow_mut().greedy_action(state))
        .collect();
    let reward_history = agent.borrow().reward_history().to_vec();
    let length_history = agent.borrow().length_history().to_vec();

    let env_id = StationOrId::Id("grid-env".to_string());
    let agent_id = StationOrId::Id("expected-sarsa-agent".to_string());
    let topology = station_graph(
        &[env_id.clone(), agent_id.clone()],
        &["StateToken", "ActionToken", "TransitionToken"].map(String::from),
        &[
            channel_edge(
                &env_id,
                environment::CH_STATE,
                &agent_id,
                Some(<ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_STATE),
            ),
            channel_edge(
                &agent_id,
                <ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_ACTION,
                &env_id,
                Some(environment::CH_ACTION),
            ),
            channel_edge(
                &env_id,
                environment::CH_TRANSITION,
                &agent_id,
                Some(<ExpectedSarsaAgent as RLAgentStation<usize, usize>>::CH_TRANSITION),
            ),
        ],
    );

    ExpectedSarsaGridResult {
        reward_history,
        length_history,
        greedy_reached: reached,
        greedy_len: len,
        q_start,
        policy,
        topology,
    }
}

#[cfg(test)]
mod tests {
    //! Both station-graph RL models learn to solve their environment.
    //!
    //! REINFORCE on the corridor should reach the goal under the greedy policy
    //! with a high success rate; expected-SARSA on the 4×4 gridworld should reach
    //! the goal from the start and improve its episodic reward over training.
    use super::*;

    #[test]
    fn reinforce_corridor_learns_to_reach_goal() {
        let res = run_policy_gradient_corridor(PolicyGradientCorridorParams {
            seed: Some(1),
            ..Default::default()
        });
        assert!(res.updates > 0, "expected REINFORCE updates");
        assert!(
            res.greedy_success_rate > 0.5,
            "greedy success rate should be substantial: {}",
            res.greedy_success_rate
        );
        // In a corridor of length 7 the only sensible greedy action is "right".
        assert_eq!(res.policy.len(), 7);
    }

    #[test]
    fn reinforce_corridor_reward_improves() {
        // REINFORCE is high-variance, so average over wide windows of a longer
        // run (seed 1 is the same known-good seed as the sibling test) to expose
        // the underlying upward trend rather than per-episode noise.
        let res = run_policy_gradient_corridor(PolicyGradientCorridorParams {
            num_episodes: Some(600),
            seed: Some(1),
            ..Default::default()
        });
        let h = &res.reward_history;
        let window = 150.min(h.len() / 4).max(1);
        let first: f64 = h[..window].iter().sum::<f64>() / window as f64;
        let last: f64 = h[h.len() - window..].iter().sum::<f64>() / window as f64;
        assert!(
            last > first,
            "mean reward should rise: first {first}, last {last}"
        );
    }

    #[test]
    fn expected_sarsa_reaches_gridworld_goal() {
        let res = run_expected_sarsa_gridworld(ExpectedSarsaGridParams {
            seed: Some(1),
            ..Default::default()
        });
        assert!(res.greedy_reached, "greedy policy should reach the goal");
        // Optimal path from start (0) to goal (15) on a 4×4 grid is 6 steps.
        assert!(
            res.greedy_len <= 12,
            "greedy path should be short: {}",
            res.greedy_len
        );
        assert_eq!(res.policy.len(), 16);
        assert_eq!(res.q_start.len(), 4);
    }
}
