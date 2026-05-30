//! Port of `src/des/general/des-base/linear-vfa.ts`.
//!
//! Approximate dynamic programming with LINEAR FUNCTION APPROXIMATION ("linear
//! VFA"). When `|S|` is too large for a tabular `V[s]`, approximate
//!
//! ```text
//!     V_θ(s) = θ · φ(s),         (or Q_θ(s, a) = θ_a · φ(s))
//! ```
//!
//! with a feature vector `φ(s) ∈ ℝ^d` and learn `θ` via the semi-gradient
//! Q-learning update (max over next actions):
//!
//! ```text
//!     δ_t = r_t + γ max_a' Q_θ(s', a') − Q_θ(s, a)
//!     θ_a ← θ_a + α δ_t φ(s)
//! ```
//!
//! ## TS → Rust mapping
//!
//! TypeScript modelled this as `abstract class LinearVFAStation<S> extends
//! RLAgentStation<S, number>` with one abstract hook (`features`) and a tabular
//! base implementation of `pickAction` / `update` / `endOfEpisode`. Rust has no
//! abstract-method inheritance, so (mirroring how `rl_agent.rs` factors the
//! template method out of the trait) we split it into:
//!
//!   * [`LinearVFACore`] — the parameters/bookkeeping the abstract class owned
//!     (`θ`, dimensions, learning rates, `ε` schedule, TD-error history). A
//!     concrete agent EMBEDS one and exposes it via `vfa_core()` /
//!     `vfa_core_mut()`.
//!   * [`LinearVFAStation`] — the hook trait (`: RLAgentStation<S, usize>`). It
//!     ADDS one REQUIRED hook (`features`) and one PROVIDED hook (`legal_actions`,
//!     defaulting to `None` = all actions legal). The TS base bodies become
//!     PROVIDED methods (`q`, `greedy_with_rng`, `vfa_greedy_action`,
//!     `linear_vfa_pick_action`, `linear_vfa_update`, `linear_vfa_end_of_episode`).
//!     A concrete agent satisfies [`RLAgentStation`] by delegating
//!     `pick_action` → `linear_vfa_pick_action`, `update` → `linear_vfa_update`,
//!     `end_of_episode` → `linear_vfa_end_of_episode`.
//!
//! Other conversions: `theta: Float64Array` (flat `A×d`) → `Vec<f64>` indexed
//! `a*d + i`; `rng: () => number` → injected [`RandomSource`] (the greedy
//! tie-break reuses `argmax.rs`'s `ARGMAX_EPS_DEFAULT`); non-ASCII `φ`/`δ` →
//! `phi`/`delta`; `legalActions(): readonly number[] | null` →
//! `Option<Vec<usize>>`; bad-dimension `throw new Error` → `panic!`.

use crate::des::general::des_base::argmax::ARGMAX_EPS_DEFAULT;
use crate::des::general::des_base::rl_agent::RLAgentStation;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::linalg::VecOps;

/// Construction options for a [`LinearVFAStation`]. `#[derive(Default)]` supplies
/// `None`/`0`; the required `feature_dim` / `num_actions` must be set explicitly.
#[derive(Default)]
pub struct LinearVFAOptions {
    /// Feature dimension `d`. Required (the constructor cannot probe `φ`).
    pub feature_dim: usize,
    pub num_actions: usize,
    /// Step size `α`. Default 0.1.
    pub alpha: Option<f64>,
    /// Discount factor `γ`. Default 0.95.
    pub gamma: Option<f64>,
    /// ε-greedy exploration probability. Default 0.1.
    pub epsilon: Option<f64>,
    /// ε decay multiplier per episode. Default 1 (no decay).
    pub epsilon_decay: Option<f64>,
    /// Floor for ε. Default 0.01.
    pub epsilon_min: Option<f64>,
    /// Initial `θ` value (broadcast to all entries). Default 0.
    pub init_theta: Option<f64>,
}

/// Parameters and bookkeeping the TS abstract class owned, factored into a
/// struct a concrete linear-VFA agent embeds.
pub struct LinearVFACore {
    /// `θ_a ∈ ℝ^d` for each action `a`. Stored as a flat `A × d` matrix.
    pub theta: Vec<f64>,
    pub d: usize,
    pub a: usize,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    /// Per-episode TD-error history (mean `|δ|` over the episode).
    pub td_error_history: Vec<f64>,
    episode_abs_td: f64,
    episode_updates: f64,
}

impl LinearVFACore {
    /// Mirrors the TS `super(id, {rng}); …` body. Panics on invalid dimensions
    /// (TS `throw new Error`).
    pub fn new(opts: LinearVFAOptions) -> Self {
        if opts.feature_dim < 1 {
            panic!("featureDim must be ≥ 1");
        }
        if opts.num_actions < 1 {
            panic!("numActions must be ≥ 1");
        }
        let d = opts.feature_dim;
        let a = opts.num_actions;
        let mut theta = vec![0.0; a * d];
        let init = opts.init_theta.unwrap_or(0.0);
        if init != 0.0 {
            theta.iter_mut().for_each(|x| *x = init);
        }
        LinearVFACore {
            theta,
            d,
            a,
            alpha: opts.alpha.unwrap_or(0.1),
            gamma: opts.gamma.unwrap_or(0.95),
            epsilon: opts.epsilon.unwrap_or(0.1),
            epsilon_decay: opts.epsilon_decay.unwrap_or(1.0),
            epsilon_min: opts.epsilon_min.unwrap_or(0.01),
            td_error_history: Vec::new(),
            episode_abs_td: 0.0,
            episode_updates: 0.0,
        }
    }
}

/// Linear value-function-approximation agent hooks. Extends [`RLAgentStation`]:
/// a concrete agent satisfies the RL hooks by delegating to the provided
/// `linear_vfa_*` methods, and need only supply [`LinearVFAStation::features`].
///
/// `S: 'static` is required transitively by [`RLAgentStation`] (its tokens are
/// `Rc<dyn Any>`).
pub trait LinearVFAStation<S: Clone + 'static = f64>: RLAgentStation<S, usize> {
    /// Borrow the embedded linear-VFA parameters/bookkeeping.
    fn vfa_core(&self) -> &LinearVFACore;
    /// Mutably borrow the embedded linear-VFA parameters/bookkeeping.
    fn vfa_core_mut(&mut self) -> &mut LinearVFACore;

    // ── HOOKS ────────────────────────────────────────────────────────────────

    /// Required hook: feature map `φ(s) ∈ ℝ^d`.
    fn features(&self, state: &S) -> Vec<f64>;

    /// Optional override: action mask. Default: all actions legal (`None`).
    fn legal_actions(&self, _state: &S) -> Option<Vec<usize>> {
        None
    }

    // ── Q AND POLICY ─────────────────────────────────────────────────────────

    /// `Q_θ(s, a) = θ_a · φ(s)`.
    fn q(&self, state: &S, action: usize) -> f64 {
        let phi = self.features(state);
        let core = self.vfa_core();
        if phi.len() != core.d {
            panic!("features() returned dim {}, expected {}", phi.len(), core.d);
        }
        let off = action * core.d;
        VecOps::dot(&core.theta[off..off + core.d], &phi)
    }

    /// Argmax over actions of `Q_θ(s, ·)` with uniform random tie-breaking,
    /// using the supplied RNG. Critical for linear VFA: with `θ = 0` every
    /// Q-value is 0, so deterministic argmax would always return action 0.
    fn greedy_with_rng(&self, state: &S, rng: &mut dyn RandomSource) -> usize {
        let legal = self.legal_actions(state);
        let eps = ARGMAX_EPS_DEFAULT;
        let mut best_a: i64 = -1;
        let mut best_q = f64::NEG_INFINITY;
        let mut tie_count = 0.0;
        let actions: Vec<usize> = match &legal {
            Some(l) => l.clone(),
            None => (0..self.vfa_core().a).collect(),
        };
        for a in actions {
            let q = self.q(state, a);
            if best_a < 0 || q > best_q + eps {
                best_q = q;
                best_a = a as i64;
                tie_count = 1.0;
            } else if q >= best_q - eps {
                tie_count += 1.0;
                if rng.next_float() * tie_count < 1.0 {
                    best_a = a as i64;
                }
            }
        }
        if best_a < 0 {
            0
        } else {
            best_a as usize
        }
    }

    /// Public greedy read using the embedded (injected) RNG, moved out
    /// transiently for the tie-break.
    fn vfa_greedy_action(&mut self, state: &S) -> usize {
        let mut rng = self
            .agent_core_mut()
            .rng
            .take()
            .expect("rng already in use");
        let a = self.greedy_with_rng(state, &mut *rng);
        self.agent_core_mut().rng = Some(rng);
        a
    }

    // ── RL HOOK BODIES (delegate to these from RLAgentStation impl) ────────────

    /// ε-greedy action selection (the TS base `pickAction`).
    fn linear_vfa_pick_action(&self, state: &S, rng: &mut dyn RandomSource) -> usize {
        let legal = self.legal_actions(state);
        if rng.next_float() < self.vfa_core().epsilon {
            if let Some(l) = &legal {
                if !l.is_empty() {
                    let idx = (rng.next_float() * l.len() as f64).floor() as usize;
                    return l[idx];
                }
            }
            return (rng.next_float() * self.vfa_core().a as f64).floor() as usize;
        }
        self.greedy_with_rng(state, rng)
    }

    /// Semi-gradient TD(0) update with Q-learning (max over next actions).
    fn linear_vfa_update(
        &mut self,
        state: &S,
        action: usize,
        reward: f64,
        next_state: &S,
        done: bool,
    ) {
        let phi = self.features(state);
        let q_sa = self.q(state, action);
        let mut bootstrap = 0.0;
        if !done {
            // max_a' Q(s', a')
            let mut max_q = f64::NEG_INFINITY;
            match self.legal_actions(next_state) {
                Some(legal_next) => {
                    for a in legal_next {
                        let q = self.q(next_state, a);
                        if q > max_q {
                            max_q = q;
                        }
                    }
                }
                None => {
                    for a in 0..self.vfa_core().a {
                        let q = self.q(next_state, a);
                        if q > max_q {
                            max_q = q;
                        }
                    }
                }
            }
            bootstrap = self.vfa_core().gamma * max_q;
        }
        let delta = reward + bootstrap - q_sa;
        let core = self.vfa_core_mut();
        let off = action * core.d;
        for i in 0..core.d {
            core.theta[off + i] += core.alpha * delta * phi[i];
        }
        core.episode_abs_td += delta.abs();
        core.episode_updates += 1.0;
    }

    /// Log mean `|δ|` and decay `ε` (the TS base `endOfEpisode`).
    fn linear_vfa_end_of_episode(&mut self) {
        let core = self.vfa_core_mut();
        if core.episode_updates > 0.0 {
            core.td_error_history
                .push(core.episode_abs_td / core.episode_updates);
        }
        core.episode_abs_td = 0.0;
        core.episode_updates = 0.0;
        core.epsilon = core.epsilon_min.max(core.epsilon * core.epsilon_decay);
    }

    // ── PUBLIC ACCESSORS ─────────────────────────────────────────────────────

    fn get_theta(&self) -> &[f64] {
        &self.vfa_core().theta
    }
    fn get_epsilon(&self) -> f64 {
        self.vfa_core().epsilon
    }
    fn get_feature_dim(&self) -> usize {
        self.vfa_core().d
    }
    fn get_num_actions(&self) -> usize {
        self.vfa_core().a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::rl_agent::RLAgentCore;
    use crate::des::general::des_base::rl_tokens::TransitionToken;
    use crate::des::general::des_base::station::{DESStation, StationCore};
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};
    use std::any::Any;
    use std::rc::Rc;

    /// Concrete linear-VFA agent over integer states `s`, with one action and a
    /// 2-dim feature map `φ(s) = [1, s]`. With a single (terminal) action the
    /// semi-gradient update is plain online linear regression of the reward onto
    /// `φ`, so `θ` should converge to the target's coefficients.
    struct LinearRegressor {
        core: StationCore,
        agent: RLAgentCore,
        vfa: LinearVFACore,
    }

    impl LinearRegressor {
        fn new(seed: u32, opts: LinearVFAOptions) -> Self {
            LinearRegressor {
                core: StationCore::new("vfa"),
                agent: RLAgentCore::new(Box::new(SeededRandom::new(seed))),
                vfa: LinearVFACore::new(opts),
            }
        }
    }

    impl DESStation for LinearRegressor {
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

    impl RLAgentStation<usize, usize> for LinearRegressor {
        fn agent_core(&self) -> &RLAgentCore {
            &self.agent
        }
        fn agent_core_mut(&mut self) -> &mut RLAgentCore {
            &mut self.agent
        }
        fn pick_action(&self, state: &usize, rng: &mut dyn RandomSource) -> usize {
            self.linear_vfa_pick_action(state, rng)
        }
        fn update(
            &mut self,
            state: &usize,
            action: &usize,
            reward: f64,
            next_state: &usize,
            done: bool,
        ) {
            self.linear_vfa_update(state, *action, reward, next_state, done);
        }
        fn end_of_episode(&mut self, _episode_id: f64) {
            self.linear_vfa_end_of_episode();
        }
    }

    impl LinearVFAStation<usize> for LinearRegressor {
        fn vfa_core(&self) -> &LinearVFACore {
            &self.vfa
        }
        fn vfa_core_mut(&mut self) -> &mut LinearVFACore {
            &mut self.vfa
        }
        fn features(&self, state: &usize) -> Vec<f64> {
            vec![1.0, *state as f64]
        }
    }

    /// Linear target the regressor must fit: `value(s) = 1 + 0.5 s`.
    fn target(state: usize) -> f64 {
        1.0 + 0.5 * state as f64
    }

    fn regressor(seed: u32) -> LinearRegressor {
        LinearRegressor::new(
            seed,
            LinearVFAOptions {
                feature_dim: 2,
                num_actions: 1,
                alpha: Some(0.02),
                gamma: Some(0.0),
                epsilon: Some(0.0),
                ..Default::default()
            },
        )
    }

    #[test]
    fn fits_a_linear_value_target() {
        let mut a = regressor(7);
        for ep in 0..4000 {
            for s in 0..4usize {
                let t = TransitionToken::new(s, 0usize, target(s), s, true, ep as f64);
                a.core_mut()
                    .take(Rc::new(t), LinearRegressor::CH_TRANSITION);
                a.run_time_step();
            }
        }
        // θ should approach the target coefficients [1, 0.5] and predict well.
        let theta = a.get_theta();
        assert!((theta[0] - 1.0).abs() < 0.05, "theta0 = {}", theta[0]);
        assert!((theta[1] - 0.5).abs() < 0.05, "theta1 = {}", theta[1]);
        for s in 0..4usize {
            assert!(
                (a.q(&s, 0) - target(s)).abs() < 0.05,
                "Q({s}) = {}",
                a.q(&s, 0)
            );
        }
    }

    #[test]
    fn td_error_shrinks_over_training() {
        let mut a = regressor(11);
        for ep in 0..4000 {
            for s in 0..4usize {
                let t = TransitionToken::new(s, 0usize, target(s), s, true, ep as f64);
                a.core_mut()
                    .take(Rc::new(t), LinearRegressor::CH_TRANSITION);
                a.run_time_step();
            }
        }
        let hist = &a.vfa_core().td_error_history;
        assert_eq!(hist.len(), 4000 * 4);
        assert!(*hist.last().unwrap() < hist[0], "TD error should shrink");
    }

    #[test]
    fn epsilon_decays_and_greedy_breaks_ties() {
        let mut a = LinearRegressor::new(
            3,
            LinearVFAOptions {
                feature_dim: 2,
                num_actions: 2,
                epsilon: Some(1.0),
                epsilon_decay: Some(0.5),
                epsilon_min: Some(0.01),
                ..Default::default()
            },
        );
        // With θ = 0 every Q is 0; greedy must still return a valid action.
        let g = a.vfa_greedy_action(&0usize);
        assert!(g < 2);
        // One terminal transition finishes one episode → ε halves.
        let t = TransitionToken::new(0usize, 0usize, 0.0, 0usize, true, 0.0);
        a.core_mut()
            .take(Rc::new(t), LinearRegressor::CH_TRANSITION);
        a.run_time_step();
        assert!(
            (a.get_epsilon() - 0.5).abs() < 1e-12,
            "epsilon = {}",
            a.get_epsilon()
        );
    }
}
