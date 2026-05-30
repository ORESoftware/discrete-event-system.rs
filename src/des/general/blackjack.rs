//! Port of `src/des/general/blackjack.ts` — the textbook BLACKJACK environment
//! from Sutton & Barto section 5.1, solved by ON-POLICY MONTE CARLO CONTROL.
//!
//! RULES: infinite deck (cards drawn with replacement, ace = 1 or 11). The
//! player sees their own hand sum plus the dealer's upcard and chooses HIT or
//! STICK. On STICK the dealer reveals the hole card and hits while sum is below
//! 17. Closer to 21 wins; ties draw; bust loses. Reward only at terminal:
//! plus-one win, zero draw, minus-one lose.
//!
//! STATE ENCODING (200 states, the canonical S&B coding): player_sum in 12..21
//! (10 values), dealer_up in 1..10, usable_ace in 0..1, giving
//! stateId = (player_sum - 12) * 20 + (dealer_up - 1) * 2 + usable_ace.
//!
//! Declarations -> Rust:
//!   * `fn drawCard` / `fn handTotal`            -> free fns ([`hand_total`] returns [`HandTotal`])
//!   * `class Blackjack implements Environment`  -> struct + `impl PureEnvironment`
//!   * `interface BlackjackTrainOpts/Result`     -> structs (optionals -> `Option<T>`)
//!   * `fn runBlackjackMC`                        -> free fn [`run_blackjack_mc`]
//!   * `static Blackjack.dealerStickPolicy`      -> associated fn
//!
//! Conversion notes:
//!   * The single TS `rng` closure is shared by env + agent + runner; modelled
//!     as a cloneable [`SharedRng`] (an `Rc<RefCell<SeededRandom>>`) so every
//!     clone advances the SAME stream, faithful to the TS aliasing.
//!   * The TS `Environment` interface is consumed by `EnvironmentStation`, which
//!     in this crate is the `PureEnvironment` trait; `Blackjack` implements that
//!     (`&mut self`, stateful card buffers) rather than the `&self`/rng-free
//!     `rl_environments::Environment`. See the migration note in the return.
//!   * `step`'s inline object -> [`StepResult`]; `throw` invariants -> `panic!`
//!     (here surfaced via `.expect`/`.unwrap` on the preconditions).

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::environment::{
    EnvironmentStation, EnvironmentStationOptions, PureEnvironment, StepResult, CH_ACTION, CH_STATE,
    CH_TRANSITION,
};
use crate::des::general::des_base::monte_carlo_rl::{MonteCarloAgent, MonteCarloOptions};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::rl_agent::RLAgentStation;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// SHARED RNG (mirrors the single TS `rng` closure shared by env/agent/runner)
// -----------------------------------------------------------------------------

/// Cloneable handle to one PRNG stream. The TS code passes a single
/// `mulberry32` closure into the env, the agent, and the runner; every clone
/// here shares the same underlying `SeededRandom`, so they consume one stream.
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
// CARD UTILITIES
// -----------------------------------------------------------------------------

/// Draw a single card from an infinite deck. Card values 1..10 (10s are
/// quad-weighted: 11=J, 12=Q, 13=K all collapse to 10).
fn draw_card(rng: &mut dyn RandomSource) -> usize {
    let u = (rng.next_float() * 13.0).floor() as usize + 1;
    u.min(10)
}

/// Result of summing a hand: the total interpreting an ace as 11 when it does
/// not bust, and whether a usable (11-valued) ace remains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandTotal {
    pub sum: usize,
    pub usable_ace: bool,
}

/// Sum a hand interpreting any ace as 11 if it doesn't bust.
fn hand_total(cards: &[usize]) -> HandTotal {
    let mut s = 0usize;
    let mut aces = 0usize;
    for &c in cards {
        if c == 1 {
            aces += 1;
            s += 11;
        } else {
            s += c;
        }
    }
    while s > 21 && aces > 0 {
        s -= 10;
        aces -= 1;
    }
    HandTotal { sum: s, usable_ace: aces > 0 }
}

// -----------------------------------------------------------------------------
// ENVIRONMENT
// -----------------------------------------------------------------------------

/// Blackjack environment (`class Blackjack implements Environment`). State is
/// the compact 200-value encoding; the internal card buffers are the "deal"
/// cache for the current episode.
pub struct Blackjack {
    rng: SharedRng,
    dealer_cards: Vec<usize>,
    player_cards: Vec<usize>,
}

impl Blackjack {
    /// Number of encoded states.
    pub const NUM_STATES: usize = 200;
    /// Number of actions (0 = STICK, 1 = HIT).
    pub const NUM_ACTIONS: usize = 2;

    fn new(rng: SharedRng) -> Self {
        Blackjack { rng, dealer_cards: Vec::new(), player_cards: Vec::new() }
    }

    /// Reset using an injected RNG (for reproducibility). Deal until the player
    /// sum reaches 12 (below-12 actions are a deterministic hit).
    fn reset_with_rng(&mut self, rng: &mut dyn RandomSource) -> usize {
        self.player_cards = vec![draw_card(rng), draw_card(rng)];
        self.dealer_cards = vec![draw_card(rng), draw_card(rng)];
        while hand_total(&self.player_cards).sum < 12 {
            self.player_cards.push(draw_card(rng));
        }
        self.encode_state(false)
    }

    /// Compact 200-state encoding. A bust collapses to a sentinel (state 0,
    /// never visited again because the episode is then done).
    pub fn encode_state(&self, busted: bool) -> usize {
        let t = hand_total(&self.player_cards);
        if busted || t.sum > 21 {
            return 0;
        }
        let player_idx = t.sum - 12; // 0..9
        let ua = if t.usable_ace { 1 } else { 0 };
        let dealer_up = self.dealer_cards[0]; // 1..10
        player_idx * 20 + (dealer_up - 1) * 2 + ua
    }

    /// Deterministic STICK-on-20+ baseline policy.
    pub fn dealer_stick_policy(state: usize) -> usize {
        let ps = state / 20 + 12;
        if ps >= 20 {
            0
        } else {
            1
        }
    }
}

impl PureEnvironment<usize, usize> for Blackjack {
    fn num_states(&self) -> usize {
        Blackjack::NUM_STATES
    }

    fn num_actions(&self) -> usize {
        Blackjack::NUM_ACTIONS
    }

    fn reset(&mut self) -> usize {
        // The TS `reset()` uses the env's own (shared) rng.
        let mut rng = self.rng.clone();
        self.reset_with_rng(&mut rng)
    }

    fn step(&mut self, _state: usize, action: usize) -> StepResult<usize> {
        let mut rng = self.rng.clone();
        if action == 1 {
            // HIT: draw a card.
            self.player_cards.push(draw_card(&mut rng));
            let t = hand_total(&self.player_cards);
            if t.sum > 21 {
                return StepResult { next_state: self.encode_state(true), reward: -1.0, done: true };
            }
            return StepResult { next_state: self.encode_state(false), reward: 0.0, done: false };
        }
        // STICK: dealer plays, then settle.
        let player_sum = hand_total(&self.player_cards).sum;
        let mut dealer_sum = hand_total(&self.dealer_cards).sum;
        while dealer_sum < 17 {
            self.dealer_cards.push(draw_card(&mut rng));
            dealer_sum = hand_total(&self.dealer_cards).sum;
        }
        let r = if dealer_sum > 21 {
            1.0
        } else if player_sum > dealer_sum {
            1.0
        } else if player_sum == dealer_sum {
            0.0
        } else {
            -1.0
        };
        StepResult { next_state: self.encode_state(true), reward: r, done: true }
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Training options for [`run_blackjack_mc`] (TS `BlackjackTrainOpts &
/// {evalEpisodes?}`). `None` fields fall back to the TS defaults.
#[derive(Clone, Debug)]
pub struct BlackjackTrainOpts {
    pub num_episodes: usize,
    pub seed: Option<u32>,
    pub epsilon: Option<f64>,
    pub epsilon_decay: Option<f64>,
    pub epsilon_min: Option<f64>,
    pub first_visit: Option<bool>,
    pub gamma: Option<f64>,
    pub eval_episodes: Option<usize>,
}

impl Default for BlackjackTrainOpts {
    fn default() -> Self {
        BlackjackTrainOpts {
            num_episodes: 100_000,
            seed: None,
            epsilon: None,
            epsilon_decay: None,
            epsilon_min: None,
            first_visit: None,
            gamma: None,
            eval_episodes: None,
        }
    }
}

/// Result of [`run_blackjack_mc`] (TS `BlackjackResult`).
#[derive(Clone, Debug, PartialEq)]
pub struct BlackjackResult {
    pub reward_history: Vec<f64>,
    /// Final mean return over the last `eval_episodes` greedy episodes.
    pub greedy_mean_return: f64,
    /// Baseline mean return over the same eval window (stick on 20+).
    pub baseline_mean_return: f64,
    /// Number of (s, a) cells visited at least once during training.
    pub visited_cells: usize,
    pub final_epsilon: f64,
    pub ticks: usize,
}

/// Train an on-policy Monte-Carlo control agent on blackjack, then evaluate the
/// greedy policy and the stick-on-20+ baseline.
pub fn run_blackjack_mc(opts: BlackjackTrainOpts) -> BlackjackResult {
    let cls = "run_blackjack_mc";
    Preconditions::integer_in_range(cls, "numEpisodes", opts.num_episodes as f64, 1.0, 1e9).unwrap();
    if let Some(e) = opts.eval_episodes {
        Preconditions::integer_in_range(cls, "evalEpisodes", e as f64, 1.0, 1e9).unwrap();
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

    let rng = SharedRng::new(opts.seed.unwrap_or(1));
    let env = Blackjack::new(rng.clone());
    let agent = MonteCarloAgent::new(
        "blackjack-mc",
        Box::new(rng.clone()),
        MonteCarloOptions {
            num_states: 200,
            num_actions: 2,
            first_visit: Some(opts.first_visit.unwrap_or(true)),
            gamma: Some(opts.gamma.unwrap_or(1.0)),
            epsilon: Some(opts.epsilon.unwrap_or(0.1)),
            epsilon_decay: Some(opts.epsilon_decay.unwrap_or(1.0)),
            epsilon_min: Some(opts.epsilon_min.unwrap_or(0.05)),
            ..Default::default()
        },
    );
    let env_station = EnvironmentStation::<usize, usize>::new(
        "env",
        Box::new(env),
        EnvironmentStationOptions {
            num_episodes: Some(opts.num_episodes as f64),
            max_steps_per_episode: Some(50), // a hand never exceeds ~10 hits
        },
    );

    let agent: Rc<RefCell<MonteCarloAgent>> = Rc::new(RefCell::new(agent));
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

    // Greedy + baseline evaluation.
    let eval_n = opts.eval_episodes.unwrap_or(10_000);
    let mut eval_rng = rng.clone();
    let mut eval_env = Blackjack::new(rng.clone());
    let mut greedy_total = 0.0_f64;
    for _ in 0..eval_n {
        let mut s = eval_env.reset_with_rng(&mut eval_rng);
        let mut step_count = 0;
        while step_count < 50 {
            let a = agent.borrow().greedy_action(s);
            let r = eval_env.step(s, a);
            step_count += 1;
            if r.done {
                greedy_total += r.reward;
                break;
            }
            s = r.next_state;
        }
    }
    let mut baseline_total = 0.0_f64;
    let mut base_env = Blackjack::new(rng.clone());
    for _ in 0..eval_n {
        let mut s = base_env.reset_with_rng(&mut eval_rng);
        let mut step_count = 0;
        while step_count < 50 {
            let a = Blackjack::dealer_stick_policy(s);
            let r = base_env.step(s, a);
            step_count += 1;
            if r.done {
                baseline_total += r.reward;
                break;
            }
            s = r.next_state;
        }
    }

    let visited = agent.borrow().get_visit_counts().iter().filter(|&&c| c > 0).count();
    let reward_history = agent.borrow().reward_history().to_vec();
    let final_epsilon = agent.borrow().get_epsilon();

    BlackjackResult {
        reward_history,
        greedy_mean_return: greedy_total / eval_n as f64,
        baseline_mean_return: baseline_total / eval_n as f64,
        visited_cells: visited,
        final_epsilon,
        ticks: summary.ticks,
    }
}

#[cfg(test)]
mod tests {
    //! Env reset/step semantics + Monte-Carlo episode termination.

    use super::*;

    #[test]
    fn reset_encodes_a_valid_state() {
        let mut env = Blackjack::new(SharedRng::new(7));
        for _ in 0..20 {
            let s = env.reset();
            assert!(s < Blackjack::NUM_STATES, "state {s} out of range");
        }
    }

    #[test]
    fn hitting_eventually_terminates_with_a_bust() {
        let mut env = Blackjack::new(SharedRng::new(3));
        env.reset();
        let mut done = false;
        let mut last_reward = 0.0;
        for _ in 0..50 {
            let r = env.step(0, 1); // always HIT
            last_reward = r.reward;
            if r.done {
                done = true;
                break;
            }
        }
        assert!(done, "hitting forever should bust and terminate");
        assert_eq!(last_reward, -1.0, "a bust loses");
    }

    #[test]
    fn monte_carlo_training_runs_episodes_to_completion() {
        let res = run_blackjack_mc(BlackjackTrainOpts {
            num_episodes: 300,
            seed: Some(1),
            eval_episodes: Some(100),
            ..Default::default()
        });
        assert_eq!(res.reward_history.len(), 300, "one reward logged per episode");
        assert!(res.ticks > 0);
        assert!(res.visited_cells > 0);
        assert!((-1.0..=1.0).contains(&res.greedy_mean_return));
        assert!((-1.0..=1.0).contains(&res.baseline_mean_return));
    }
}
