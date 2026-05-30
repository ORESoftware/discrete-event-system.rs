//! Port of `src/des/general/tiger-pomdp.ts` — module `des::general::tiger_pomdp`.
//!
//! The canonical TIGER PROBLEM (Cassandra, Kaelbling, Littman 1994) wrapped on
//! the [`BeliefStateStation`] base. Two doors hide a tiger and gold. The agent
//! may LISTEN (cheap, noisy, keeps the latent state) or OPEN_LEFT / OPEN_RIGHT
//! (definitive but risky; opening resets the world to a uniform belief).
//!
//! States are {tiger-left, tiger-right}, actions are {listen, open-left,
//! open-right}, observations are {hear-left, hear-right}. Listen accuracy is the
//! probability of hearing the tiger behind its true door. We solve it with two
//! policies built on the ported `pomdp` infrastructure: the QMDP heuristic and a
//! one-step belief lookahead.
//!
//! Mapping notes (from the TS "RUST MIGRATION" header):
//!   * `interface TigerOpts` / `TigerSimResult` / `TigerSimOpts` -> structs.
//!   * `buildTigerSpec` -> [`build_tiger_spec`] returning a classic
//!     `POMDPSpec<String, String, String>`; `specToCore` -> the private
//!     [`SpecPomdpCore`] adapter implementing [`POMDPCore`].
//!   * `class QMDPStation extends BeliefStateStation<number, number>` -> a
//!     struct + `impl BeliefStateStation<usize, usize>`.
//!   * INHERITANCE: `OneStepLookAheadStation extends QMDPStation` -> a struct
//!     EMBEDDING a [`QMDPStation`] (`base`) and overriding only `pick_action`.
//!   * INJECT RNG: `simulateTiger` samples the latent door + noisy observations
//!     through an injected [`SeededRandom`].
//!   * String-keyed sets -> the closures index by `usize`; the string-union
//!     `solver` field -> the [`TigerSolver`] enum.
//!
//! FLAG (reused-API divergence): the TS `QMDPStation` held a
//! `new QMDPSolver(spec)`. The Rust [`QMDPSolver`](crate::des::general::pomdp)
//! takes its `POMDPSpec` BY VALUE, which conflicts with the shared `Rc<spec>`
//! also needed by the belief-station adapter. We therefore compute the QMDP
//! Q-table directly with [`mdp_value_iteration`] (exactly what `QMDPSolver::new`
//! does internally) and inline the 3-line greedy `act` / `qmdp_value`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::belief_state::{
    ActionObservationToken, BeliefCore, BeliefStateStation, POMDPCore, CH_INPUT,
};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::pomdp::{mdp_value_iteration, MDPVIOptions, POMDPSpec};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// TIGER PROBLEM CONSTANTS
// -----------------------------------------------------------------------------

pub const TIGER_LEFT: usize = 0;
pub const TIGER_RIGHT: usize = 1;
pub const ACT_LISTEN: usize = 0;
pub const ACT_OPEN_LEFT: usize = 1;
pub const ACT_OPEN_RIGHT: usize = 2;
pub const OBS_HEAR_LEFT: usize = 0;
pub const OBS_HEAR_RIGHT: usize = 1;

/// Knobs for [`build_tiger_spec`]. Absent optionals use the classic defaults.
#[derive(Clone, Debug, Default)]
pub struct TigerOpts {
    /// `P(hear-left | tiger-left, listen)`. Default 0.85.
    pub listen_accuracy: Option<f64>,
    /// Reward for opening the gold door. Default +10.
    pub open_good: Option<f64>,
    /// Reward for opening the tiger door. Default -100.
    pub open_bad: Option<f64>,
    /// Per-listen cost. Default -1.
    pub listen_cost: Option<f64>,
    /// Discount gamma. Default 0.95.
    pub discount: Option<f64>,
}

/// Build the Tiger problem in the classic `POMDPSpec` shape.
pub fn build_tiger_spec(opts: &TigerOpts) -> POMDPSpec<String, String, String> {
    let acc = opts.listen_accuracy.unwrap_or(0.85);
    let good = opts.open_good.unwrap_or(10.0);
    let bad = opts.open_bad.unwrap_or(-100.0);
    let lc = opts.listen_cost.unwrap_or(-1.0);
    let gamma = opts.discount.unwrap_or(0.95);
    POMDPSpec {
        states: vec!["tiger-left".to_string(), "tiger-right".to_string()],
        actions: vec![
            "listen".to_string(),
            "open-left".to_string(),
            "open-right".to_string(),
        ],
        observations: vec!["hear-left".to_string(), "hear-right".to_string()],
        transition: Box::new(move |s_idx, a_idx| {
            if a_idx == ACT_LISTEN {
                if s_idx == TIGER_LEFT {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            } else {
                // Opening either door resets the world to uniform.
                vec![0.5, 0.5]
            }
        }),
        observation: Box::new(move |sn_idx, a_idx| {
            if a_idx != ACT_LISTEN {
                return vec![0.5, 0.5];
            }
            if sn_idx == TIGER_LEFT {
                vec![acc, 1.0 - acc]
            } else {
                vec![1.0 - acc, acc]
            }
        }),
        reward: Box::new(move |s_idx, a_idx| {
            if a_idx == ACT_LISTEN {
                return lc;
            }
            if a_idx == ACT_OPEN_LEFT {
                return if s_idx == TIGER_LEFT { bad } else { good };
            }
            if a_idx == ACT_OPEN_RIGHT {
                return if s_idx == TIGER_RIGHT { bad } else { good };
            }
            0.0
        }),
        discount: gamma,
        initial_belief: Some(vec![0.5, 0.5]),
        is_terminal: None,
    }
}

// -----------------------------------------------------------------------------
// CORE ADAPTER: classic POMDPSpec -> POMDPCore (the BeliefStateStation API)
// -----------------------------------------------------------------------------

/// Adapts a shared classic `POMDPSpec` into the [`POMDPCore`] the belief station
/// consumes (the TS `specToCore`). Indices are `usize`; missing entries read as
/// zero (the TS `[...] ?? 0`).
struct SpecPomdpCore {
    spec: Rc<POMDPSpec<String, String, String>>,
}

impl POMDPCore<usize, usize> for SpecPomdpCore {
    fn num_states(&self) -> usize {
        self.spec.states.len()
    }
    fn num_actions(&self) -> usize {
        self.spec.actions.len()
    }
    fn num_observations(&self) -> usize {
        self.spec.observations.len()
    }
    fn transition_prob(&self, s: usize, a: &usize, sp: usize) -> f64 {
        (self.spec.transition)(s, *a)
            .get(sp)
            .copied()
            .unwrap_or(0.0)
    }
    fn observation_prob(&self, sp: usize, a: &usize, o: &usize) -> f64 {
        (self.spec.observation)(sp, *a)
            .get(*o)
            .copied()
            .unwrap_or(0.0)
    }
}

// -----------------------------------------------------------------------------
// QMDP STATION
// -----------------------------------------------------------------------------

/// QMDP belief station: solves the underlying MDP, then acts greedily under the
/// belief.
pub struct QMDPStation {
    core: StationCore,
    belief: BeliefCore<usize, usize>,
    classic_spec: Rc<POMDPSpec<String, String, String>>,
    /// QMDP `Q[s][a]`.
    q: Vec<Vec<f64>>,
}

impl QMDPStation {
    /// Mirrors `new QMDPStation(spec, b0?)`. The initial belief is `b0`, else the
    /// spec's `initial_belief`, else uniform.
    pub fn new(spec: Rc<POMDPSpec<String, String, String>>, b0: Option<Vec<f64>>) -> Self {
        let core_pomdp: Box<dyn POMDPCore<usize, usize>> =
            Box::new(SpecPomdpCore { spec: spec.clone() });
        let initial = b0.or_else(|| spec.initial_belief.clone());
        let belief = BeliefCore::new(core_pomdp, initial.as_deref());
        let r = mdp_value_iteration(&*spec, &MDPVIOptions::default());
        QMDPStation {
            core: StationCore::new("qmdp"),
            belief,
            classic_spec: spec,
            q: r.q,
        }
    }

    /// `a* = argmax_a Σ_s b(s) Q(s, a)` (the QMDP policy).
    fn qmdp_act(&self, b: &[f64]) -> usize {
        let num_a = self.classic_spec.actions.len();
        let mut best_a = 0;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..num_a {
            let mut q = 0.0;
            for s in 0..self.q.len() {
                q += b[s] * self.q[s][a];
            }
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        best_a
    }

    /// The QMDP value function `V_QMDP(b) = Σ_s b(s) max_a Q(s, a)`.
    pub fn qmdp_value(&self, b: &[f64]) -> f64 {
        let mut v = 0.0;
        for s in 0..self.q.len() {
            let mut best = f64::NEG_INFINITY;
            for a in 0..self.q[s].len() {
                if self.q[s][a] > best {
                    best = self.q[s][a];
                }
            }
            v += b[s] * best;
        }
        v
    }
}

impl DESStation for QMDPStation {
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
        self.belief_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.belief_has_work()
    }
}

impl BeliefStateStation<usize, usize> for QMDPStation {
    fn belief_core(&self) -> &BeliefCore<usize, usize> {
        &self.belief
    }
    fn belief_core_mut(&mut self) -> &mut BeliefCore<usize, usize> {
        &mut self.belief
    }
    fn pick_action(&self, b: &[f64]) -> usize {
        self.qmdp_act(b)
    }
}

// -----------------------------------------------------------------------------
// 1-STEP LOOK-AHEAD STATION
// -----------------------------------------------------------------------------

/// Information-gathering aware policy that picks
/// `a* = argmax_a R(b, a) + γ Σ_o P(o | b, a) V_QMDP(τ(b, a, o))`. Reuses the
/// QMDP machinery via an embedded [`QMDPStation`] (`base`) and overrides only
/// the action selection.
pub struct OneStepLookAheadStation {
    base: QMDPStation,
}

impl OneStepLookAheadStation {
    pub fn new(spec: Rc<POMDPSpec<String, String, String>>, b0: Option<Vec<f64>>) -> Self {
        let mut base = QMDPStation::new(spec, b0);
        base.core.id = "pomdp-1step-lookahead".to_string();
        OneStepLookAheadStation { base }
    }
}

impl DESStation for OneStepLookAheadStation {
    fn core(&self) -> &StationCore {
        &self.base.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.base.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.belief_run_time_step();
    }
    fn has_work(&self) -> bool {
        self.belief_has_work()
    }
}

impl BeliefStateStation<usize, usize> for OneStepLookAheadStation {
    fn belief_core(&self) -> &BeliefCore<usize, usize> {
        &self.base.belief
    }
    fn belief_core_mut(&mut self) -> &mut BeliefCore<usize, usize> {
        &mut self.base.belief
    }
    fn pick_action(&self, b: &[f64]) -> usize {
        let spec = &self.base.classic_spec;
        let gamma = spec.discount;
        let n = spec.states.len();
        let num_a = spec.actions.len();
        let num_o = spec.observations.len();
        let mut best_a = 0;
        let mut best_q = f64::NEG_INFINITY;
        for a in 0..num_a {
            // Expected immediate reward.
            let mut r_imm = 0.0;
            for s in 0..n {
                r_imm += b[s] * (spec.reward)(s, a);
            }
            // Discounted expected QMDP value over the next-belief distribution.
            let mut exp = 0.0;
            for o in 0..num_o {
                let p_o = self.observation_likelihood(b, &a, &o);
                if p_o == 0.0 {
                    continue;
                }
                let bp = self.belief_update(b, &a, &o);
                exp += p_o * self.base.qmdp_value(&bp);
            }
            let q = r_imm + gamma * exp;
            if q > best_q {
                best_q = q;
                best_a = a;
            }
        }
        best_a
    }
}

// -----------------------------------------------------------------------------
// SIMULATION DRIVER
// -----------------------------------------------------------------------------

/// Which solver `simulateTiger` runs (the TS `'qmdp' | 'one-step-lookahead'`
/// string union).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TigerSolver {
    Qmdp,
    OneStepLookahead,
}

/// Result of a fixed-step Tiger simulation.
#[derive(Clone, Debug)]
pub struct TigerSimResult {
    /// Discounted return.
    pub total_return: f64,
    pub actions: Vec<usize>,
    pub observations: Vec<usize>,
    pub states: Vec<usize>,
    pub belief_p0: Vec<f64>,
    pub steps: usize,
    /// Number of doors opened during the run.
    pub num_opens: usize,
    /// Number of times the bad (tiger) door was opened.
    pub num_bad_opens: usize,
}

/// Options for [`simulate_tiger`]. `spec` defaults to [`build_tiger_spec`].
pub struct TigerSimOpts {
    pub spec: Option<POMDPSpec<String, String, String>>,
    pub solver: TigerSolver,
    pub num_steps: usize,
    pub seed: Option<u32>,
    pub initial_state: Option<usize>,
    pub initial_belief: Option<Vec<f64>>,
}

/// Shared step loop, monomorphised per concrete solver station. The "world"
/// outside the agent is the same Tiger spec driven by `rng`.
fn run_tiger_sim<St: BeliefStateStation<usize, usize> + 'static>(
    station: St,
    spec: &Rc<POMDPSpec<String, String, String>>,
    num_steps: usize,
    initial_state: Option<usize>,
    rng: &mut SeededRandom,
) -> TigerSimResult {
    let gamma = spec.discount;
    let station = Rc::new(RefCell::new(station));
    let mut s = initial_state.unwrap_or_else(|| if rng.next_float() < 0.5 { 0 } else { 1 });
    let mut actions: Vec<usize> = Vec::new();
    let mut observations: Vec<usize> = Vec::new();
    let mut states: Vec<usize> = vec![s];
    let mut belief_p0: Vec<f64> = vec![station.borrow().get_belief()[0]];
    let mut total_ret = 0.0;
    let mut discount = 1.0;
    let mut num_opens = 0;
    let mut num_bad_opens = 0;

    for _t in 0..num_steps {
        let b = station.borrow().get_belief().to_vec();
        let a = station.borrow().pick_action(&b);
        actions.push(a);
        // Sample next state (latent dynamics).
        let t_row = (spec.transition)(s, a);
        let u1 = rng.next_float();
        let mut sp = 0;
        let mut acc = 0.0;
        for sn in 0..spec.states.len() {
            acc += t_row[sn];
            if u1 < acc {
                sp = sn;
                break;
            }
        }
        // Sample observation conditional on s'.
        let o_row = (spec.observation)(sp, a);
        let u2 = rng.next_float();
        let mut o = 0;
        acc = 0.0;
        for on in 0..spec.observations.len() {
            acc += o_row[on];
            if u2 < acc {
                o = on;
                break;
            }
        }
        observations.push(o);
        states.push(sp);
        let r = (spec.reward)(s, a);
        total_ret += discount * r;
        discount *= gamma;
        if a == ACT_OPEN_LEFT || a == ACT_OPEN_RIGHT {
            num_opens += 1;
            if r < 0.0 {
                num_bad_opens += 1;
            }
        }
        // Inject the (action, observation) and run one tick of the filter.
        station
            .borrow_mut()
            .core_mut()
            .take(Rc::new(ActionObservationToken::new(a, o)), CH_INPUT);
        let handle: StationRef = station.clone();
        run_iterative_des(
            vec![handle],
            IterativeRunOptions {
                max_ticks: Some(1),
                run_validators: false,
                ..Default::default()
            },
        );
        belief_p0.push(station.borrow().get_belief()[0]);
        s = sp;
    }

    TigerSimResult {
        total_return: total_ret,
        actions,
        observations,
        states,
        belief_p0,
        steps: num_steps,
        num_opens,
        num_bad_opens,
    }
}

/// Run a fixed-step Tiger simulation under the chosen solver.
pub fn simulate_tiger(opts: TigerSimOpts) -> TigerSimResult {
    let cls = "simulateTiger";
    Preconditions::integer_in_range(cls, "numSteps", opts.num_steps as f64, 1.0, 1e9).unwrap();
    // FLAG: the TS runtime check that `solver` is one of two strings is vacuous
    // for the `TigerSolver` enum.
    if let Some(ib) = &opts.initial_belief {
        Preconditions::probability_vector(cls, "initialBelief", ib, 1e-9).unwrap();
    }
    let mut rng = mulberry32(opts.seed.unwrap_or(1));
    let spec = Rc::new(
        opts.spec
            .unwrap_or_else(|| build_tiger_spec(&TigerOpts::default())),
    );
    Preconditions::in_range(cls, "spec.discount", spec.discount, 0.0, 1.0).unwrap();
    Preconditions::non_empty(cls, "spec.states", &spec.states).unwrap();
    Preconditions::non_empty(cls, "spec.actions", &spec.actions).unwrap();
    Preconditions::non_empty(cls, "spec.observations", &spec.observations).unwrap();
    if let Some(ib) = &opts.initial_belief {
        Preconditions::length_eq(cls, "initialBelief", ib, spec.states.len()).unwrap();
    }
    if let Some(is) = opts.initial_state {
        Preconditions::integer_in_range(
            cls,
            "initialState",
            is as f64,
            0.0,
            (spec.states.len() - 1) as f64,
        )
        .unwrap();
    }

    let b0 = opts.initial_belief.clone();
    match opts.solver {
        TigerSolver::Qmdp => {
            let station = QMDPStation::new(spec.clone(), b0);
            run_tiger_sim(station, &spec, opts.num_steps, opts.initial_state, &mut rng)
        }
        TigerSolver::OneStepLookahead => {
            let station = OneStepLookAheadStation::new(spec.clone(), b0);
            run_tiger_sim(station, &spec, opts.num_steps, opts.initial_state, &mut rng)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tiger-problem checks. The Bayesian belief update concentrates after a
    //! noisy listen, the QMDP policy listens from a uniform prior and opens the
    //! safe door once confident, and a one-step-lookahead simulation listens
    //! first and opens at least once over a fixed-seed run.

    use super::*;

    fn spec() -> Rc<POMDPSpec<String, String, String>> {
        Rc::new(build_tiger_spec(&TigerOpts::default()))
    }

    #[test]
    fn belief_update_concentrates_after_listen() {
        let station = QMDPStation::new(spec(), None);
        // Uniform prior, listen, hear the tiger on the left: belief shifts to the
        // listen accuracy (0.85) on tiger-left.
        let updated = station.belief_update(&[0.5, 0.5], &ACT_LISTEN, &OBS_HEAR_LEFT);
        assert!((updated.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((updated[TIGER_LEFT] - 0.85).abs() < 1e-9, "{updated:?}");
        assert!(updated[TIGER_LEFT] > updated[TIGER_RIGHT]);
    }

    #[test]
    fn qmdp_listens_then_opens_the_safe_door() {
        let station = QMDPStation::new(spec(), None);
        // From a uniform prior, opening is too risky: QMDP listens first.
        assert_eq!(station.pick_action(&[0.5, 0.5]), ACT_LISTEN);
        // Confident the tiger is on the left -> open the RIGHT (gold) door.
        assert_eq!(station.pick_action(&[0.99, 0.01]), ACT_OPEN_RIGHT);
        // Symmetrically, confident tiger-right -> open the LEFT door.
        assert_eq!(station.pick_action(&[0.01, 0.99]), ACT_OPEN_LEFT);
    }

    #[test]
    fn one_step_lookahead_simulation_listens_then_opens() {
        let result = simulate_tiger(TigerSimOpts {
            spec: None,
            solver: TigerSolver::OneStepLookahead,
            num_steps: 30,
            seed: Some(3),
            initial_state: Some(TIGER_LEFT),
            initial_belief: None,
        });
        assert_eq!(result.steps, 30);
        assert_eq!(result.belief_p0.len(), 31);
        // Information-gathering: the first move is always to LISTEN.
        assert_eq!(result.actions[0], ACT_LISTEN);
        // Over 30 steps the belief concentrates enough to commit at least once.
        assert!(
            result.num_opens >= 1,
            "never opened a door: {:?}",
            result.actions
        );
        assert!(result.total_return.is_finite());
    }
}
