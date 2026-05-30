//! Port of `src/des/general/des-base/multi-agent.ts`.
//!
//! Base for SIMULTANEOUS-MOVE MULTI-AGENT REINFORCEMENT LEARNING on a shared
//! joint environment (Independent Q-Learning, IQL).
//!
//! ## Problem shape
//!
//! `N` agents share an environment. At each tick: every agent observes its own
//! state `s_i`, every agent picks an action `a_i` in parallel, the environment
//! applies the joint action `a = (a_1, …, a_N)` and emits a joint next state
//! plus a per-agent reward vector and per-agent done flags. Each agent then
//! updates its own table on its own transition — NO cross-agent updates.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//!   * `interface JointEnvironment<S,A>` → trait [`JointEnvironment`]. Its
//!     `step` returns a named [`JointStepResult`] instead of an inline
//!     `{nextStates, rewards, dones}` object. `numAgents` (a readonly property)
//!     → `num_agents(&self)`. `reset`/`step` take `&mut self` (implementations
//!     are stateful).
//!   * `class JointEnvStation<S,A>` → struct + `impl DESStation`. Per-agent
//!     channels (`agent-action-<i>`, …) are built with `format!`; the static
//!     `*Channel(i)` helpers become associated fns. `pendingActions:
//!     Map<number,A>` → `HashMap<usize, A>`. `env['env'].numAgents` (bracket
//!     access to a protected field) → the public accessor [`JointEnvStation::num_agents`].
//!   * `interface MultiAgentSystemOpts` → struct ([`MultiAgentSystemOpts`],
//!     `#[derive(Default)]`).
//!   * `class MultiAgentSystem<S,A>` → struct owning the env handle
//!     (`Rc<RefCell<JointEnvStation>>`) and a `Vec<StationRef>` of agents (the
//!     TS held `RLAgentStation<S,A>[]` DESStation references). `pipe` builds the
//!     graph edges over shared handles. The agent-count mismatch `throw` →
//!     `panic!`.
//!   * The TS `rewardHistory` / `lengthHistory` fields aliased the inner
//!     [`VectorEpisodeAccounting`] vectors; Rust cannot own a shared alias, so
//!     they become borrowing accessor methods. Getter/setter pairs proxy the
//!     inner accounting.
//!
//! Generic defaults `S = number` / `A = number` map to `S = f64` / `A = usize`
//! per the migration rules. `S`/`A` are `'static` (tokens are `Rc<dyn Any>`)
//! and `Clone` (the env station clones states/actions into outgoing tokens).

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::episode_accounting::VectorEpisodeAccounting;
use crate::des::general::des_base::rl_tokens::{ActionToken, StateToken, TransitionToken};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{
    AnyToken, ChannelName, DESStation, StationCore, StationRef,
};

// FLAG (unported-const reference): `MultiAgentSystem` wires its agents to the
// channel names declared as associated consts on the `RLAgentStation` trait
// (`CH_STATE` / `CH_TRANSITION` / `CH_ACTION` in `rl_agent.rs`). Those consts
// cannot be named in this generic context (agents are held as `dyn DESStation`,
// not a concrete `RLAgentStation` type), so their default string values are
// restated here. These MUST stay in sync with `rl_agent.rs`.
const RL_AGENT_CH_STATE: &str = "state";
const RL_AGENT_CH_TRANSITION: &str = "transition";
const RL_AGENT_CH_ACTION: &str = "action";

/// The outcome of one [`JointEnvironment::step`]: joint next state, per-agent
/// reward vector, and per-agent done flags (all of length `num_agents`).
///
/// Replaces the inline TS return object `{nextStates, rewards, dones}`.
#[derive(Clone, Debug)]
pub struct JointStepResult<S = f64> {
    /// Joint next state `s' = [s'_1, …, s'_N]`.
    pub next_states: Vec<S>,
    /// Per-agent reward `r = [r_1, …, r_N]`.
    pub rewards: Vec<f64>,
    /// Per-agent done flag. The episode ENDS when all are `true` (or some
    /// env-wide termination fires, in which case all flags should be `true`).
    pub dones: Vec<bool>,
}

/// An N-agent environment with a shared joint state/action per step.
pub trait JointEnvironment<S = f64, A = usize> {
    /// Number of agents sharing this environment.
    fn num_agents(&self) -> usize;
    /// Reset returns the joint state `s = [s_1, …, s_N]`.
    fn reset(&mut self) -> Vec<S>;
    /// Apply the joint action `a = [a_1, …, a_N]` and return the joint
    /// transition (next state, per-agent reward, per-agent done).
    fn step(&mut self, states: &[S], actions: &[A]) -> JointStepResult<S>;
}

/// Options for constructing a [`JointEnvStation`] (the TS inline `opts` object).
pub struct JointEnvStationOpts {
    /// Number of episodes to run before the station goes quiescent. TS default
    /// `Infinity`.
    pub num_episodes: Option<f64>,
    /// Per-episode step cap (truncation). TS default `1_000_000`.
    pub max_steps_per_episode: Option<usize>,
}

impl Default for JointEnvStationOpts {
    fn default() -> Self {
        JointEnvStationOpts {
            num_episodes: None,
            max_steps_per_episode: None,
        }
    }
}

/// A single tick of the joint env: deal joint state to every agent, collect
/// every agent's action, apply `env.step`, dispatch transitions + new states.
pub struct JointEnvStation<S = f64, A = usize> {
    core: StationCore,
    env: Box<dyn JointEnvironment<S, A>>,
    num_episodes: f64,
    max_steps_per_episode: usize,

    cur_states: Vec<S>,
    episode_id: f64,
    step_in_episode: usize,
    emitted_start: bool,
    pending_actions: HashMap<usize, A>,

    /// Per-episode return per agent, recorded at episode end.
    episode_accounting: VectorEpisodeAccounting,
}

impl<S: Clone + 'static, A: Clone + 'static> JointEnvStation<S, A> {
    /// Prefix for the per-agent ACTION inbox channels (`agent-action-<i>`).
    pub const CH_AGENT_ACTION_PREFIX: &'static str = "agent-action-";
    /// Prefix for the per-agent STATE output channels (`agent-state-<i>`).
    pub const CH_AGENT_STATE_PREFIX: &'static str = "agent-state-";
    /// Prefix for the per-agent TRANSITION output channels (`agent-transition-<i>`).
    pub const CH_AGENT_TRANSITION_PREFIX: &'static str = "agent-transition-";

    /// Mirrors `new JointEnvStation(id, env, opts)`. Resets the env to obtain
    /// the initial joint state and sizes the vector accounting by
    /// `env.num_agents()`.
    pub fn new(
        id: impl Into<String>,
        env: Box<dyn JointEnvironment<S, A>>,
        opts: JointEnvStationOpts,
    ) -> Self {
        let mut env = env;
        let cur_states = env.reset();
        let num_agents = env.num_agents();
        JointEnvStation {
            core: StationCore::new(id),
            env,
            num_episodes: opts.num_episodes.unwrap_or(f64::INFINITY),
            max_steps_per_episode: opts.max_steps_per_episode.unwrap_or(1_000_000),
            cur_states,
            episode_id: 0.0,
            step_in_episode: 0,
            emitted_start: false,
            pending_actions: HashMap::new(),
            episode_accounting: VectorEpisodeAccounting::new(num_agents),
        }
    }

    /// Public accessor for the inner env's agent count (TS `env['env'].numAgents`).
    pub fn num_agents(&self) -> usize {
        self.env.num_agents()
    }

    /// Each agent calls this with its action for the current joint state.
    pub fn take_agent_action(&mut self, agent_idx: usize, action: A) {
        self.pending_actions.insert(agent_idx, action);
    }

    /// Total steps recorded across episodes (proxies the inner accounting).
    pub fn total_steps(&self) -> u64 {
        self.episode_accounting.total_steps
    }
    /// Setter for `total_steps` (TS `set totalSteps`).
    pub fn set_total_steps(&mut self, value: u64) {
        self.episode_accounting.total_steps = value;
    }

    /// Current accumulating per-agent return (TS protected `get curReturn`).
    pub fn cur_return(&self) -> &[f64] {
        &self.episode_accounting.current_rewards
    }
    /// Setter for the current per-agent return (TS protected `set curReturn`).
    pub fn set_cur_return(&mut self, value: &[f64]) {
        self.episode_accounting.set_current_rewards(value);
    }

    /// Borrowing accessor for the TS `rewardHistory` alias (per-episode,
    /// per-agent returns).
    pub fn reward_history(&self) -> &[Vec<f64>] {
        &self.episode_accounting.reward_history
    }
    /// Borrowing accessor for the TS `lengthHistory` alias (per-episode length).
    pub fn length_history(&self) -> &[f64] {
        &self.episode_accounting.length_history
    }

    /// Emit the current joint state, one [`StateToken`] per agent channel.
    fn emit_state_for_all_agents(&mut self) {
        for i in 0..self.env.num_agents() {
            let token: AnyToken =
                Rc::new(StateToken::new(self.cur_states[i].clone(), self.episode_id));
            let ch = format!("{}{i}", Self::CH_AGENT_STATE_PREFIX);
            self.core.emit(token, &ch);
        }
    }

    /// Action channel name for agent `i`.
    pub fn action_channel(i: usize) -> ChannelName {
        format!("{}{i}", Self::CH_AGENT_ACTION_PREFIX)
    }
    /// State channel name for agent `i`.
    pub fn state_channel(i: usize) -> ChannelName {
        format!("{}{i}", Self::CH_AGENT_STATE_PREFIX)
    }
    /// Transition channel name for agent `i`.
    pub fn transition_channel(i: usize) -> ChannelName {
        format!("{}{i}", Self::CH_AGENT_TRANSITION_PREFIX)
    }
}

impl<S: Clone + 'static, A: Clone + 'static> DESStation for JointEnvStation<S, A> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn has_work(&self) -> bool {
        if !self.emitted_start {
            return true;
        }
        if self.episode_id >= self.num_episodes {
            return false;
        }
        // We have work if our action channels have tokens we haven't drained
        // yet, OR if we already drained enough into pending_actions earlier.
        if self.pending_actions.len() == self.env.num_agents() {
            return true;
        }
        for i in 0..self.env.num_agents() {
            let ch = format!("{}{i}", Self::CH_AGENT_ACTION_PREFIX);
            if self.core.inbox_size(&ch) > 0 {
                return true;
            }
        }
        false
    }

    fn run_time_step(&mut self) {
        if !self.emitted_start {
            self.emitted_start = true;
            self.emit_state_for_all_agents();
            return;
        }
        if self.episode_id >= self.num_episodes {
            return;
        }
        let n = self.env.num_agents();
        if self.pending_actions.len() < n {
            // Drain any pending action tokens.
            for i in 0..n {
                let ch = format!("{}{i}", Self::CH_AGENT_ACTION_PREFIX);
                for tok in self.core.drain::<ActionToken<S, A>>(&ch) {
                    self.pending_actions.insert(i, tok.action.clone());
                }
            }
            if self.pending_actions.len() < n {
                return;
            }
        }
        let mut actions: Vec<A> = Vec::with_capacity(n);
        for i in 0..n {
            actions.push(
                self.pending_actions
                    .get(&i)
                    .expect("missing pending action")
                    .clone(),
            );
        }
        self.pending_actions.clear();
        let r = self.env.step(&self.cur_states, &actions);
        self.episode_accounting.record_step(&r.rewards);
        self.step_in_episode += 1;
        let truncated = self.step_in_episode >= self.max_steps_per_episode;
        let all_done = r.dones.iter().all(|&d| d) || truncated;
        for i in 0..n {
            let token: AnyToken = Rc::new(TransitionToken::new(
                self.cur_states[i].clone(),
                actions[i].clone(),
                r.rewards[i],
                r.next_states[i].clone(),
                all_done,
                self.episode_id,
            ));
            let ch = format!("{}{i}", Self::CH_AGENT_TRANSITION_PREFIX);
            self.core.emit(token, &ch);
        }
        self.cur_states = r.next_states;
        if all_done {
            self.episode_accounting
                .finish_episode(self.step_in_episode as f64);
            self.step_in_episode = 0;
            self.episode_id += 1.0;
            if self.episode_id < self.num_episodes {
                self.cur_states = self.env.reset();
                self.emit_state_for_all_agents();
            }
        }
    }
}

// -----------------------------------------------------------------------------
// MULTI-AGENT SYSTEM ORCHESTRATOR
// -----------------------------------------------------------------------------

/// Options for [`MultiAgentSystem::run`] (TS `MultiAgentSystemOpts`). `rng` is
/// threaded into the run loop's tick-order shuffling.
#[derive(Default)]
pub struct MultiAgentSystemOpts {
    /// RNG forwarded to [`run_iterative_des`]; `None` uses the runner default.
    pub rng: Option<Box<dyn FnMut() -> f64>>,
}

/// The result of one full multi-agent training loop (TS `run` return object).
#[derive(Clone, Debug, Default)]
pub struct MultiAgentRunResult {
    /// Number of simulation ticks executed.
    pub ticks: usize,
    /// Per-episode, per-agent return (a copy of the env's `reward_history`).
    pub reward_history: Vec<Vec<f64>>,
    /// Re-indexed by agent: `per_agent_reward_history[i][e]` is agent `i`'s
    /// return in episode `e`.
    pub per_agent_reward_history: Vec<Vec<f64>>,
}

/// Orchestrator that owns a [`JointEnvStation`] plus `N` agent handles, wires
/// their channels, and drives the train loop via [`run_iterative_des`].
pub struct MultiAgentSystem<S = f64, A = usize> {
    /// The joint environment station.
    pub env: Rc<RefCell<JointEnvStation<S, A>>>,
    /// The agents (held as `DESStation` handles, mirroring the TS
    /// `RLAgentStation<S,A>[]` references).
    pub agents: Vec<StationRef>,
}

impl<S: Clone + 'static, A: Clone + 'static> MultiAgentSystem<S, A> {
    /// Mirrors `new MultiAgentSystem(env, agents)`: validates the agent count
    /// against `env.num_agents()` (`panic!` on mismatch) and wires the
    /// state/transition/action channels.
    pub fn new(env: Rc<RefCell<JointEnvStation<S, A>>>, agents: Vec<StationRef>) -> Self {
        let expected = env.borrow().num_agents();
        if agents.len() != expected {
            panic!("expected {} agents, got {}", expected, agents.len());
        }
        // Wire the channels (Rc handles).
        for (i, agent) in agents.iter().enumerate() {
            env.borrow_mut().core_mut().pipe(
                agent.clone(),
                &JointEnvStation::<S, A>::state_channel(i),
                RL_AGENT_CH_STATE,
            );
            env.borrow_mut().core_mut().pipe(
                agent.clone(),
                &JointEnvStation::<S, A>::transition_channel(i),
                RL_AGENT_CH_TRANSITION,
            );
            agent.borrow_mut().core_mut().pipe(
                env.clone() as StationRef,
                RL_AGENT_CH_ACTION,
                &JointEnvStation::<S, A>::action_channel(i),
            );
        }
        MultiAgentSystem { env, agents }
    }

    /// Run the full multi-agent training loop.
    pub fn run(&self, opts: MultiAgentSystemOpts) -> MultiAgentRunResult {
        let mut participants: Vec<StationRef> = Vec::with_capacity(1 + self.agents.len());
        participants.push(self.env.clone() as StationRef);
        for a in &self.agents {
            participants.push(a.clone());
        }
        let summary = run_iterative_des(
            participants,
            IterativeRunOptions {
                rng: opts.rng,
                ..Default::default()
            },
        );

        let env = self.env.borrow();
        let reward_hist = env.reward_history();
        let mut per_agent: Vec<Vec<f64>> = vec![Vec::new(); self.agents.len()];
        for ep in reward_hist {
            for i in 0..ep.len() {
                per_agent[i].push(ep[i]);
            }
        }
        let reward_history: Vec<Vec<f64>> = reward_hist.iter().map(|r| r.clone()).collect();
        MultiAgentRunResult {
            ticks: summary.ticks,
            reward_history,
            per_agent_reward_history: per_agent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation};
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};

    /// A 2+ agent coordination game: every step is terminal, and each agent is
    /// rewarded `+1` iff all agents picked the same action (else `0`).
    struct CoordEnv {
        n: usize,
    }

    impl JointEnvironment<usize, usize> for CoordEnv {
        fn num_agents(&self) -> usize {
            self.n
        }
        fn reset(&mut self) -> Vec<usize> {
            vec![0; self.n]
        }
        fn step(&mut self, _states: &[usize], actions: &[usize]) -> JointStepResult<usize> {
            let coordinated = actions.iter().all(|&a| a == actions[0]);
            let r = if coordinated { 1.0 } else { 0.0 };
            JointStepResult {
                next_states: actions.to_vec(),
                rewards: vec![r; actions.len()],
                dones: vec![true; actions.len()],
            }
        }
    }

    /// A trivial RL agent that always plays a fixed action (ignores its state)
    /// and never learns — enough to exercise the coordination loop.
    struct DummyAgent {
        core: StationCore,
        agent: RLAgentCore,
        fixed: usize,
    }

    impl DummyAgent {
        fn new(id: &str, seed: u32, fixed: usize) -> Self {
            DummyAgent {
                core: StationCore::new(id),
                agent: RLAgentCore::new(Box::new(SeededRandom::new(seed))),
                fixed,
            }
        }
    }

    impl DESStation for DummyAgent {
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

    impl RLAgentStation<usize, usize> for DummyAgent {
        fn agent_core(&self) -> &RLAgentCore {
            &self.agent
        }
        fn agent_core_mut(&mut self) -> &mut RLAgentCore {
            &mut self.agent
        }
        fn pick_action(&self, _state: &usize, _rng: &mut dyn RandomSource) -> usize {
            self.fixed
        }
        fn update(&mut self, _s: &usize, _a: &usize, _r: f64, _ns: &usize, _done: bool) {}
    }

    /// Sink that records every [`TransitionToken`] it receives on `"in"`.
    struct TransitionSink {
        core: StationCore,
        count: usize,
        last_reward: f64,
    }

    impl TransitionSink {
        const CH_IN: &'static str = "in";
        fn new(id: &str) -> Self {
            TransitionSink {
                core: StationCore::new(id),
                count: 0,
                last_reward: f64::NAN,
            }
        }
    }

    impl DESStation for TransitionSink {
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
            for t in self
                .core
                .drain::<TransitionToken<usize, usize>>(Self::CH_IN)
            {
                self.count += 1;
                self.last_reward = t.reward;
            }
        }
    }

    #[test]
    fn single_round_emits_transitions() {
        let mut station = JointEnvStation::<usize, usize>::new(
            "env",
            Box::new(CoordEnv { n: 2 }),
            JointEnvStationOpts::default(),
        );
        let sink = Rc::new(RefCell::new(TransitionSink::new("sink")));
        station.core_mut().pipe(
            sink.clone(),
            &JointEnvStation::<usize, usize>::transition_channel(0),
            TransitionSink::CH_IN,
        );
        station.core_mut().pipe(
            sink.clone(),
            &JointEnvStation::<usize, usize>::transition_channel(1),
            TransitionSink::CH_IN,
        );

        // First tick emits start states (no agent wired to the state channels).
        station.run_time_step();
        // Two agents coordinate on action 1.
        station.take_agent_action(0, 1);
        station.take_agent_action(1, 1);
        assert!(station.has_work());
        station.run_time_step();

        sink.borrow_mut().run_time_step();
        assert_eq!(sink.borrow().count, 2, "one transition per agent");
        assert_eq!(sink.borrow().last_reward, 1.0, "coordinated -> reward 1");
        assert_eq!(station.reward_history().len(), 1, "one episode finished");
        assert_eq!(station.reward_history()[0], vec![1.0, 1.0]);
    }

    #[test]
    fn full_system_runs_and_records_histories() {
        let env = Rc::new(RefCell::new(JointEnvStation::<usize, usize>::new(
            "env",
            Box::new(CoordEnv { n: 2 }),
            JointEnvStationOpts {
                num_episodes: Some(5.0),
                max_steps_per_episode: None,
            },
        )));
        let a0: StationRef = Rc::new(RefCell::new(DummyAgent::new("a0", 1, 1)));
        let a1: StationRef = Rc::new(RefCell::new(DummyAgent::new("a1", 2, 1)));
        let sys = MultiAgentSystem::new(env.clone(), vec![a0, a1]);

        let res = sys.run(MultiAgentSystemOpts::default());

        assert_eq!(res.reward_history.len(), 5, "five episodes recorded");
        assert_eq!(res.per_agent_reward_history.len(), 2);
        assert_eq!(res.per_agent_reward_history[0].len(), 5);
        assert!(res.per_agent_reward_history[0].iter().all(|&r| r == 1.0));
        assert!(res.per_agent_reward_history[1].iter().all(|&r| r == 1.0));
        assert!(res.ticks > 0);
    }

    #[test]
    #[should_panic(expected = "expected 2 agents")]
    fn mismatched_agent_count_panics() {
        let env = Rc::new(RefCell::new(JointEnvStation::<usize, usize>::new(
            "env",
            Box::new(CoordEnv { n: 2 }),
            JointEnvStationOpts::default(),
        )));
        let a0: StationRef = Rc::new(RefCell::new(DummyAgent::new("a0", 1, 1)));
        let _ = MultiAgentSystem::new(env, vec![a0]);
    }
}
