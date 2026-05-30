//! Port of `src/des/general/stag-hunt.ts` — module `des::general::stag_hunt`.
//!
//! The canonical STAG HUNT coordination game (Rousseau 1755 / Skyrms 2004)
//! solved with INDEPENDENT Q-LEARNING (Tan 1993) on top of the multi-agent DES
//! base.
//!
//! The payoff matrix (rows = player 1, columns = player 2) is Stag/Stag = (4,4),
//! Stag/Hare = (0,3), Hare/Stag = (3,0), Hare/Hare = (3,3). It has two pure
//! Nash equilibria: (Stag, Stag) is payoff-dominant but needs coordination, and
//! (Hare, Hare) is risk-dominant but safe. Independent Q-learners can converge
//! to either depending on initialisation, epsilon, and alpha.
//!
//! As a multi-agent DES system this is a single-state stateless game (state is
//! always 0). Each agent is a tabular Q-learner over one state and two actions;
//! the joint-environment station samples the matrix payoff and each agent
//! updates independently.
//!
//! Mapping notes (from the TS "RUST MIGRATION" header):
//!   * `interface StagHuntOpts` / `StagHuntResult` -> structs.
//!   * `class StagHuntEnv implements JointEnvironment<number, number>` -> a
//!     private struct implementing [`JointEnvironment`].
//!   * `class StagHuntQLearner extends RLAgentStation<number, number>` -> a
//!     private struct embedding [`RLAgentCore`] + [`StationCore`] and
//!     implementing [`RLAgentStation`].
//!   * `fn runStagHunt` -> [`run_stag_hunt`].
//!   * INJECT RNG: the TS shared one `mulberry32` closure across both agents and
//!     the runner's tick-order shuffle; here a single
//!     `Rc<RefCell<SeededRandom>>` is shared through the [`SharedRng`] wrapper.
//!   * `number` -> `f64`; state/action indices -> `usize`; the payoff matrix is
//!     a fixed `[[(f64, f64); 2]; 2]` constant.

use std::cell::RefCell;
use std::rc::Rc;

use std::any::Any;

use crate::des::general::des_base::argmax::{arg_max_with_tie_break, ARGMAX_EPS_DEFAULT};
use crate::des::general::des_base::multi_agent::{
    JointEnvStation, JointEnvStationOpts, JointEnvironment, JointStepResult, MultiAgentSystem,
    MultiAgentSystemOpts,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::rl_agent::{RLAgentCore, RLAgentStation, RngRef};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// MATRIX-GAME ENVIRONMENT
// -----------------------------------------------------------------------------

/// Action index: hunt the stag (payoff-dominant, needs coordination).
pub const STAG: usize = 0;
/// Action index: hunt the hare (risk-dominant, safe).
pub const HARE: usize = 1;

/// `PAYOFF[a1][a2] = (r1, r2)` — row is player 1's action, column is player 2's.
const PAYOFF: [[(f64, f64); 2]; 2] = [
    // a1 = Stag
    [(4.0, 4.0), (0.0, 3.0)],
    // a1 = Hare
    [(3.0, 0.0), (3.0, 3.0)],
];

/// A single-state, single-step matrix game. `reset` returns the joint state
/// `[0, 0]` and `step` returns the matrix payoff with both agents done.
struct StagHuntEnv;

impl JointEnvironment<usize, usize> for StagHuntEnv {
    fn num_agents(&self) -> usize {
        2
    }
    fn reset(&mut self) -> Vec<usize> {
        vec![0, 0]
    }
    fn step(&mut self, _states: &[usize], actions: &[usize]) -> JointStepResult<usize> {
        let a1 = actions[0];
        let a2 = actions[1];
        let (r1, r2) = PAYOFF[a1][a2];
        JointStepResult {
            next_states: vec![0, 0],
            rewards: vec![r1, r2],
            dones: vec![true, true],
        }
    }
}

// -----------------------------------------------------------------------------
// SHARED RNG
// -----------------------------------------------------------------------------

/// A clonable handle onto one shared mulberry32 stream. The TS file shared a
/// single `rng` closure across both agents and the run loop; this wrapper gives
/// each consumer its own [`RandomSource`] view of the same underlying state.
#[derive(Clone)]
struct SharedRng(Rc<RefCell<SeededRandom>>);

impl RandomSource for SharedRng {
    fn next_float(&mut self) -> f64 {
        self.0.borrow_mut().next_float()
    }
}

// -----------------------------------------------------------------------------
// SIMPLE TABULAR Q-LEARNING AGENT (1 state x 2 actions)
// -----------------------------------------------------------------------------

/// Tabular Q-learner over a single state and two actions. The differentiator is
/// the (no-bootstrap) `update` rule of the stateless one-step game.
struct StagHuntQLearner {
    core: StationCore,
    agent: RLAgentCore,
    /// `Q[action]` for the single state.
    q: Vec<f64>,
    a: usize,
    alpha: f64,
    #[allow(dead_code)]
    gamma: f64,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
}

impl StagHuntQLearner {
    fn new(
        id: &str,
        rng: SharedRng,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        epsilon_decay: f64,
        epsilon_min: f64,
    ) -> Self {
        StagHuntQLearner {
            core: StationCore::new(id),
            agent: RLAgentCore::new(Box::new(rng)),
            q: vec![0.0; 2],
            a: 2,
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
        }
    }

    /// The current greedy action (tie broken via the agent's own RNG, mirroring
    /// the TS `argMaxWithTieBreak(this.Q, this.rng)`).
    fn greedy_action(&mut self) -> usize {
        let mut rng = self.agent.rng.take().expect("rng already in use");
        let a = arg_max_with_tie_break(&self.q, &mut RngRef(&mut *rng), ARGMAX_EPS_DEFAULT)
            .unwrap_or(0);
        self.agent.rng = Some(rng);
        a
    }

    #[allow(dead_code)]
    fn get_q(&self) -> &[f64] {
        &self.q
    }

    #[allow(dead_code)]
    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl DESStation for StagHuntQLearner {
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

impl RLAgentStation<usize, usize> for StagHuntQLearner {
    fn agent_core(&self) -> &RLAgentCore {
        &self.agent
    }
    fn agent_core_mut(&mut self) -> &mut RLAgentCore {
        &mut self.agent
    }
    fn pick_action(&self, _state: &usize, rng: &mut dyn RandomSource) -> usize {
        if rng.next_float() < self.epsilon {
            return (rng.next_float() * self.a as f64).floor() as usize;
        }
        arg_max_with_tie_break(&self.q, &mut RngRef(rng), ARGMAX_EPS_DEFAULT).unwrap_or(0)
    }
    fn update(&mut self, _s: &usize, action: &usize, reward: f64, _ns: &usize, _done: bool) {
        // No bootstrap: stateless single-step game.
        self.q[*action] += self.alpha * (reward - self.q[*action]);
    }
    fn end_of_episode(&mut self, _episode_id: f64) {
        self.epsilon = self.epsilon_min.max(self.epsilon * self.epsilon_decay);
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for [`run_stag_hunt`]. Absent optionals fall back to the TS defaults
/// (alpha 0.05, gamma 0, epsilon 0.2, decay 0.999, min 0.01, seed 1).
#[derive(Clone, Debug)]
pub struct StagHuntOpts {
    pub num_episodes: usize,
    pub alpha: Option<f64>,
    pub gamma: Option<f64>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub seed: Option<u32>,
}

impl StagHuntOpts {
    /// Construct with all-default hyperparameters and the given episode count.
    pub fn new(num_episodes: usize) -> Self {
        StagHuntOpts {
            num_episodes,
            alpha: None,
            gamma: None,
            epsilon: None,
            epsilon_decay: None,
            epsilon_min: None,
            seed: None,
        }
    }
}

/// Outcome of a [`run_stag_hunt`] training run.
#[derive(Clone, Debug)]
pub struct StagHuntResult {
    /// Per-episode joint reward `[r1, r2]`.
    pub reward_history: Vec<Vec<f64>>,
    /// Final greedy joint action `[a1, a2]`.
    pub final_joint_action: [usize; 2],
    /// Per-agent mean return over the last (up to) 100 episodes.
    pub recent_mean_return: [f64; 2],
    /// True iff both agents converged to STAG (payoff-dominant equilibrium).
    pub coordinated_on_stag: bool,
    /// True iff both agents converged to HARE (risk-dominant equilibrium).
    pub coordinated_on_hare: bool,
    pub ticks: usize,
}

/// Train two independent Q-learners on the stag-hunt game and report the
/// equilibrium they coordinate on.
pub fn run_stag_hunt(opts: &StagHuntOpts) -> StagHuntResult {
    let cls = "runStagHunt";
    Preconditions::integer_in_range(cls, "numEpisodes", opts.num_episodes as f64, 1.0, 1e9)
        .unwrap();
    if let Some(alpha) = opts.alpha {
        Preconditions::positive(cls, "alpha", alpha).unwrap();
    }
    if let Some(gamma) = opts.gamma {
        Preconditions::in_range(cls, "gamma", gamma, 0.0, 1.0).unwrap();
    }
    if let Some(epsilon) = opts.epsilon {
        Preconditions::in_range(cls, "epsilon", epsilon, 0.0, 1.0).unwrap();
    }
    if let Some(decay) = opts.epsilon_decay {
        Preconditions::in_range(cls, "epsilonDecay", decay, 0.0, 1.0).unwrap();
    }
    if let Some(min) = opts.epsilon_min {
        Preconditions::in_range(cls, "epsilonMin", min, 0.0, 1.0).unwrap();
    }

    let rng = SharedRng(Rc::new(RefCell::new(mulberry32(opts.seed.unwrap_or(1)))));

    let env_station = Rc::new(RefCell::new(JointEnvStation::<usize, usize>::new(
        "stag-hunt-env",
        Box::new(StagHuntEnv),
        JointEnvStationOpts {
            num_episodes: Some(opts.num_episodes as f64),
            max_steps_per_episode: Some(1),
        },
    )));

    let alpha = opts.alpha.unwrap_or(0.05);
    let gamma = opts.gamma.unwrap_or(0.0);
    let epsilon = opts.epsilon.unwrap_or(0.2);
    let decay = opts.epsilon_decay.unwrap_or(0.999);
    let min = opts.epsilon_min.unwrap_or(0.01);

    let a1 = Rc::new(RefCell::new(StagHuntQLearner::new(
        "agent-0",
        rng.clone(),
        alpha,
        gamma,
        epsilon,
        decay,
        min,
    )));
    let a2 = Rc::new(RefCell::new(StagHuntQLearner::new(
        "agent-1",
        rng.clone(),
        alpha,
        gamma,
        epsilon,
        decay,
        min,
    )));

    let sys = MultiAgentSystem::new(
        env_station,
        vec![a1.clone() as StationRef, a2.clone() as StationRef],
    );

    // The runner's tick-order shuffle draws from the same shared stream.
    let shuffle_rng = rng.clone();
    let summary = sys.run(MultiAgentSystemOpts {
        rng: Some(Box::new({
            let mut r = shuffle_rng;
            move || r.next_float()
        })),
    });

    let greedy = [
        a1.borrow_mut().greedy_action(),
        a2.borrow_mut().greedy_action(),
    ];

    // Recent returns over the last (up to) 100 episodes.
    let hist = &summary.reward_history;
    let take = hist.len().min(100);
    let last = &hist[hist.len() - take..];
    let denom = (last.len().max(1)) as f64;
    let r1: f64 = last.iter().map(|e| e[0]).sum::<f64>() / denom;
    let r2: f64 = last.iter().map(|e| e[1]).sum::<f64>() / denom;

    StagHuntResult {
        reward_history: summary.reward_history.clone(),
        final_joint_action: greedy,
        recent_mean_return: [r1, r2],
        coordinated_on_stag: greedy[0] == STAG && greedy[1] == STAG,
        coordinated_on_hare: greedy[0] == HARE && greedy[1] == HARE,
        ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! Stag-hunt convergence checks. With a fixed seed the two independent
    //! Q-learners settle on one of the two pure equilibria, so the final greedy
    //! joint action is coordinated (both Stag or both Hare).

    use super::*;

    #[test]
    fn reaches_an_equilibrium() {
        let result = run_stag_hunt(&StagHuntOpts {
            num_episodes: 5000,
            seed: Some(7),
            ..StagHuntOpts::new(5000)
        });
        // The greedy actions agree -> a coordinated pure-strategy equilibrium.
        assert_eq!(
            result.final_joint_action[0], result.final_joint_action[1],
            "agents failed to coordinate: {:?}",
            result.final_joint_action
        );
        assert!(result.coordinated_on_stag || result.coordinated_on_hare);
        assert_eq!(result.reward_history.len(), 5000);
        assert!(result.ticks > 0);
    }

    #[test]
    fn equilibrium_payoff_is_consistent() {
        let result = run_stag_hunt(&StagHuntOpts {
            num_episodes: 5000,
            seed: Some(1),
            ..StagHuntOpts::new(5000)
        });
        // Whichever equilibrium they reach, the symmetric recent returns are
        // both near the equilibrium payoff (Hare = 3, Stag = 4); never below the
        // safe value once coordinated.
        let [r1, r2] = result.recent_mean_return;
        assert!(r1 > 2.5, "agent-0 recent return too low: {r1}");
        assert!(r2 > 2.5, "agent-1 recent return too low: {r2}");
        assert!(result.coordinated_on_stag || result.coordinated_on_hare);
    }

    #[test]
    fn single_episode_runs() {
        let result = run_stag_hunt(&StagHuntOpts::new(1));
        assert_eq!(result.reward_history.len(), 1);
        assert!(result.ticks > 0);
    }
}
