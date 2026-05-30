//! Port of `src/des/general/des-base/finite-horizon-dp.ts`.
//!
//! Template-method base for FINITE-HORIZON DYNAMIC PROGRAMMING via backward
//! induction. With horizon `T < ∞` and a known terminal value `V_T(s)`,
//!
//! ```text
//!   V_t(s) = max_a Σ_{s'} p(s'|s,a) [ r(s,a,s',t) + γ_t V_{t+1}(s') ],  t < T
//!   π_t(s) = argmax_a (same expression)
//! ```
//!
//! is computed by walking BACKWARDS from `t = T-1` down to `t = 0`. As a DES
//! station each tick (`run_dp_step`) performs ONE backward sweep at the current
//! stage; the station finishes once stage 0 is computed.
//!
//! ## Rust shape (faithful translation of the TS abstract class)
//!
//!   * `interface DPOutcome`  → struct [`DPOutcome`] (`nextState` → `usize`).
//!   * `interface DPOptions`  → struct [`DpOptions`] (manual `Default` — the TS
//!     defaults are `Infinity` / `true` / `ARGMAX_EPS_DEFAULT`, which a
//!     `#[derive(Default)]` cannot express).
//!   * `abstract class FiniteHorizonDPStation extends DESStation` → trait
//!     [`FiniteHorizonDPStation`]`: DESStation`. Rust traits hold no fields, so
//!     the shared protected state (`V`, `policy`, `finished`, `currentStage`,
//!     `stageHistory`, options, injected RNG) lives in [`DpState`], surfaced via
//!     the required `dp_state` / `dp_state_mut` accessors.
//!   * TEMPLATE METHOD: the `final runTimeStep` (one backward sweep) →
//!     provided [`FiniteHorizonDPStation::run_dp_step`]; required hooks
//!     `horizon` / `num_states` / `num_actions` / `transitions` are required
//!     trait fns; `terminal_reward` / `stage_discount` / `on_stage_computed`
//!     are provided defaults. `bootstrap` is a provided template helper.
//!   * `rng: () => number` (reservoir tie-break) → an injected boxed
//!     [`RandomSource`](crate::des::shared::capabilities::RandomSource), threaded
//!     out of the state during a sweep (take/put pattern).
//!   * `policy: number[][]` with a `-1` sentinel → `Vec<Vec<Option<usize>>>`.
//!   * non-ASCII `γ` → `gamma`. `throw new Error` (horizon < 1) → `panic!`;
//!     `Preconditions.*` guards → [`assert_preconditions_dp`](FiniteHorizonDPStation::assert_preconditions_dp)
//!     returning [`Check`].

use crate::des::general::des_base::argmax::ARGMAX_EPS_DEFAULT;
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::DESStation;
use crate::des::shared::capabilities::{RandomSource, SystemRandom};

/// One `(prob, reward, nextState)` branch of a stage transition.
#[derive(Clone, Copy, Debug)]
pub struct DPOutcome {
    pub prob: f64,
    pub reward: f64,
    pub next_state: usize,
}

/// `{stage, maxV, minV}` diagnostic row (TS inline object → named struct).
#[derive(Clone, Copy, Debug)]
pub struct StageStat {
    pub stage: usize,
    pub max_v: f64,
    pub min_v: f64,
}

/// Construction options (TS `interface DPOptions`).
pub struct DpOptions {
    /// Optional cap on max history length (`Infinity` = unbounded).
    pub max_history_len: f64,
    /// Break argmax ties uniformly at random (default `true`); `false` reverts
    /// to first-action-wins (matches a deterministic textbook DP).
    pub random_tie_break: bool,
    pub tie_break_eps: f64,
    /// Injected randomness; `None` → [`SystemRandom`].
    pub rng: Option<Box<dyn RandomSource>>,
}

impl Default for DpOptions {
    fn default() -> Self {
        DpOptions {
            max_history_len: f64::INFINITY,
            random_tie_break: true,
            tie_break_eps: ARGMAX_EPS_DEFAULT,
            rng: None,
        }
    }
}

/// Shared protected state of the TS `abstract class`. A concrete DP station
/// embeds one and exposes it via the trait's `dp_state` / `dp_state_mut`.
pub struct DpState {
    /// `V[t][s]` for `t = 0 … T` (length `T+1`). Built incrementally.
    pub v: Vec<Vec<f64>>,
    /// `π[t][s]` for `t = 0 … T-1` (length `T`); `None` in unset positions.
    pub policy: Vec<Vec<Option<usize>>>,
    /// True after stage 0 has been computed.
    pub finished: bool,
    /// Current stage being processed (counts down from `T-1` to 0).
    pub current_stage: usize,
    /// History of `{stage, maxV, minV}` for diagnostics.
    pub stage_history: Vec<StageStat>,
    pub max_history_len: f64,
    pub random_tie_break: bool,
    pub tie_break_eps: f64,
    pub rng: Option<Box<dyn RandomSource>>,
}

impl DpState {
    pub fn new(opts: DpOptions) -> Self {
        DpState {
            v: Vec::new(),
            policy: Vec::new(),
            finished: false,
            current_stage: 0,
            stage_history: Vec::new(),
            max_history_len: opts.max_history_len,
            random_tie_break: opts.random_tie_break,
            tie_break_eps: opts.tie_break_eps,
            rng: Some(opts.rng.unwrap_or_else(|| Box::new(SystemRandom::new()))),
        }
    }
}

impl Default for DpState {
    fn default() -> Self {
        DpState::new(DpOptions::default())
    }
}

/// The finite-horizon DP hook trait. REQUIRED methods are the TS abstract
/// hooks; optional hooks have defaults. The PROVIDED methods make up the
/// template method (`run_dp_step`) plus `bootstrap`, guards and accessors, and
/// must NOT be overridden by concrete algorithms.
pub trait FiniteHorizonDPStation: DESStation {
    fn dp_state(&self) -> &DpState;
    fn dp_state_mut(&mut self) -> &mut DpState;

    // ── HOOKS (required) ───────────────────────────────────────────────────────

    /// Horizon `T` (≥ 1).
    fn horizon(&self) -> usize;
    /// Number of states `|S|`.
    fn num_states(&self) -> usize;
    /// Legal action count `A(s, t)` (0-indexed; may vary by stage).
    fn num_actions(&self, state: usize, stage: usize) -> usize;
    /// Transition branches for `(state, action)` at STAGE `t`.
    fn transitions(&self, state: usize, action: usize, stage: usize) -> Vec<DPOutcome>;

    // ── HOOKS (optional override) ──────────────────────────────────────────────

    /// Terminal reward `V_T(s)`. Default 0.
    fn terminal_reward(&self, _state: usize) -> f64 {
        0.0
    }
    /// Per-stage discount `γ_t`. Default 1 (undiscounted finite horizon).
    fn stage_discount(&self, _stage: usize) -> f64 {
        1.0
    }
    /// Instrumentation fired after stage `t`'s value row is computed.
    fn on_stage_computed(&mut self, _stage: usize, _v: &[f64]) {}

    // ── TEMPLATE HELPERS ───────────────────────────────────────────────────────

    /// Install `V_T` from the terminal reward and set `currentStage = T-1`.
    /// Concrete stations MUST call this once after construction.
    fn bootstrap(&mut self) {
        let t = self.horizon();
        let n = self.num_states();
        if t < 1 {
            panic!("finite-horizon-dp: horizon must be >= 1, got {t}");
        }
        let mut v_terminal = vec![0.0_f64; n];
        for (s, slot) in v_terminal.iter_mut().enumerate() {
            *slot = self.terminal_reward(s);
        }
        let max_v = max_arr(&v_terminal);
        let min_v = min_arr(&v_terminal);
        let st = self.dp_state_mut();
        st.v = vec![Vec::new(); t + 1];
        st.policy = vec![Vec::new(); t];
        st.v[t] = v_terminal;
        st.current_stage = t - 1;
        if (st.stage_history.len() as f64) < st.max_history_len {
            st.stage_history.push(StageStat { stage: t, max_v, min_v });
        }
    }

    /// One backward sweep at the current stage (the TS `final runTimeStep`).
    fn run_dp_step(&mut self) {
        if self.dp_state().finished {
            return;
        }
        let t = self.dp_state().current_stage;
        let n = self.num_states();
        let gamma = self.stage_discount(t);
        let eps = self.dp_state().tie_break_eps;
        let use_tie_break = self.dp_state().random_tie_break;
        let v_next = self.dp_state().v[t + 1].clone();

        let mut vt = vec![0.0_f64; n];
        let mut pol: Vec<Option<usize>> = vec![None; n];
        let mut rng = self.dp_state_mut().rng.take().expect("finite-horizon-dp: rng already in use");

        for s in 0..n {
            let a_count = self.num_actions(s, t);
            let mut best_q = f64::NEG_INFINITY;
            let mut best_a: Option<usize> = None;
            let mut tie_count = 0.0_f64;
            for a in 0..a_count {
                let outs = self.transitions(s, a, t);
                if outs.is_empty() {
                    continue;
                }
                let mut q = 0.0;
                for o in &outs {
                    q += o.prob * (o.reward + gamma * v_next[o.next_state]);
                }
                if best_a.is_none() || q > best_q + eps {
                    best_q = q;
                    best_a = Some(a);
                    tie_count = 1.0;
                } else if use_tie_break && q >= best_q - eps {
                    tie_count += 1.0;
                    // Reservoir sampling: keep the new index with prob 1/tieCount.
                    if rng.next_float() * tie_count < 1.0 {
                        best_a = Some(a);
                    }
                }
            }
            vt[s] = if best_a.is_some() { best_q } else { 0.0 };
            pol[s] = best_a;
        }

        self.dp_state_mut().rng = Some(rng);

        let max_v = max_arr(&vt);
        let min_v = min_arr(&vt);
        let vt_for_hook = vt.clone();
        {
            let st = self.dp_state_mut();
            st.v[t] = vt;
            st.policy[t] = pol;
            if (st.stage_history.len() as f64) < st.max_history_len {
                st.stage_history.push(StageStat { stage: t, max_v, min_v });
            }
        }
        self.on_stage_computed(t, &vt_for_hook);
        if t == 0 {
            self.dp_state_mut().finished = true;
            return;
        }
        self.dp_state_mut().current_stage = t - 1;
    }

    /// Default `has_work`: not yet finished.
    fn dp_has_work(&self) -> bool {
        !self.dp_state().finished
    }

    /// Pre-run guards (TS `assertPreconditions` override). Recoverable
    /// construction-time failures → [`Check`]; callers `?`/`.expect()` at the
    /// edge. Model name uses `id()` (the TS used `this.constructor.name`).
    fn assert_preconditions_dp(&self) -> Check {
        let cls = self.id().to_string();
        let t = self.horizon();
        let n = self.num_states();
        Preconditions::check(&cls, "horizon()", "be an integer >= 1", t >= 1, Some(t.to_string()))?;
        Preconditions::check(&cls, "numStates()", "be an integer >= 1", n >= 1, Some(n.to_string()))?;
        for stage in 0..t {
            for s in 0..n {
                let a_count = self.num_actions(s, stage);
                for a in 0..a_count {
                    let outs = self.transitions(s, a, stage);
                    if outs.is_empty() {
                        continue;
                    }
                    let mut sum = 0.0;
                    for (i, o) in outs.iter().enumerate() {
                        Preconditions::check(
                            &cls,
                            &format!("transitions({s},{a},{stage})[{i}].prob"),
                            "be in [0, 1]",
                            o.prob.is_finite() && o.prob >= 0.0 && o.prob <= 1.0 + 1e-9,
                            Some(o.prob.to_string()),
                        )?;
                        Preconditions::finite(
                            &cls,
                            &format!("transitions({s},{a},{stage})[{i}].reward"),
                            o.reward,
                        )?;
                        Preconditions::check(
                            &cls,
                            &format!("transitions({s},{a},{stage})[{i}].nextState"),
                            "be a valid state index",
                            o.next_state < n,
                            Some(o.next_state.to_string()),
                        )?;
                        sum += o.prob;
                    }
                    if (sum - 1.0).abs() > 1e-6 {
                        Preconditions::check(
                            &cls,
                            &format!("transitions({s},{a},{stage}) probs"),
                            "sum to 1",
                            false,
                            Some(sum.to_string()),
                        )?;
                    }
                }
            }
        }
        for stage in 0..t {
            let gamma = self.stage_discount(stage);
            Preconditions::in_range(&cls, &format!("stageDiscount({stage})"), gamma, 0.0, 1.0)?;
        }
        Ok(())
    }

    // ── PUBLIC ACCESSORS ───────────────────────────────────────────────────────

    /// Value function at stage `t` (stage `T` returns the terminal rewards).
    fn get_v(&self, t: usize) -> &[f64] {
        &self.dp_state().v[t]
    }
    /// Optimal action at stage `t` (`0 ≤ t ≤ T-1`) in state `s`.
    fn get_action(&self, t: usize, s: usize) -> Option<usize> {
        self.dp_state().policy[t][s]
    }
    fn is_finished(&self) -> bool {
        self.dp_state().finished
    }
    fn get_current_stage(&self) -> usize {
        self.dp_state().current_stage
    }
}

fn max_arr(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn min_arr(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::station::StationCore;
    use crate::des::shared::capabilities::SeededRandom;
    use std::any::Any;

    /// Two-state, two-action, deterministic 2-stage problem with a known
    /// optimum. Transitions are stage-independent:
    ///   state 0: a0 → (0, r=1), a1 → (1, r=0)
    ///   state 1: a0 → (0, r=2), a1 → (1, r=3)
    /// terminal reward 0, γ = 1 ⇒ V_0 = [3, 6], π_0 = [1, 1], π_1 = [0, 1].
    struct TwoStage {
        core: StationCore,
        state: DpState,
    }

    impl TwoStage {
        fn new() -> Self {
            TwoStage {
                core: StationCore::new("two-stage"),
                // deterministic policy (no random tie-break needed here)
                state: DpState::new(DpOptions {
                    random_tie_break: false,
                    rng: Some(Box::new(SeededRandom::new(1))),
                    ..Default::default()
                }),
            }
        }
    }

    impl DESStation for TwoStage {
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
            self.run_dp_step();
        }
        fn has_work(&self) -> bool {
            self.dp_has_work()
        }
    }

    impl FiniteHorizonDPStation for TwoStage {
        fn dp_state(&self) -> &DpState {
            &self.state
        }
        fn dp_state_mut(&mut self) -> &mut DpState {
            &mut self.state
        }
        fn horizon(&self) -> usize {
            2
        }
        fn num_states(&self) -> usize {
            2
        }
        fn num_actions(&self, _state: usize, _stage: usize) -> usize {
            2
        }
        fn transitions(&self, state: usize, action: usize, _stage: usize) -> Vec<DPOutcome> {
            let (next_state, reward) = match (state, action) {
                (0, 0) => (0, 1.0),
                (0, _) => (1, 0.0),
                (1, 0) => (0, 2.0),
                (_, _) => (1, 3.0),
            };
            vec![DPOutcome { prob: 1.0, reward, next_state }]
        }
    }

    /// Single-state, two-action, discounted problem: a0 self-loop r=1, a1
    /// self-loop r=10, γ = 0.5 ⇒ V_1 = 10, V_0 = 1 + 0.5·10 vs 10 + 0.5·10 = 15.
    struct DiscountedLoop {
        core: StationCore,
        state: DpState,
    }

    impl DiscountedLoop {
        fn new() -> Self {
            DiscountedLoop {
                core: StationCore::new("discounted-loop"),
                state: DpState::new(DpOptions {
                    random_tie_break: false,
                    rng: Some(Box::new(SeededRandom::new(2))),
                    ..Default::default()
                }),
            }
        }
    }

    impl DESStation for DiscountedLoop {
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
            self.run_dp_step();
        }
        fn has_work(&self) -> bool {
            self.dp_has_work()
        }
    }

    impl FiniteHorizonDPStation for DiscountedLoop {
        fn dp_state(&self) -> &DpState {
            &self.state
        }
        fn dp_state_mut(&mut self) -> &mut DpState {
            &mut self.state
        }
        fn horizon(&self) -> usize {
            2
        }
        fn num_states(&self) -> usize {
            1
        }
        fn num_actions(&self, _state: usize, _stage: usize) -> usize {
            2
        }
        fn stage_discount(&self, _stage: usize) -> f64 {
            0.5
        }
        fn transitions(&self, _state: usize, action: usize, _stage: usize) -> Vec<DPOutcome> {
            let reward = if action == 0 { 1.0 } else { 10.0 };
            vec![DPOutcome { prob: 1.0, reward, next_state: 0 }]
        }
    }

    fn drive<S: FiniteHorizonDPStation>(s: &mut S) {
        s.bootstrap();
        let mut guard = 0;
        while !s.is_finished() {
            s.run_dp_step();
            guard += 1;
            assert!(guard < 1000, "DP did not finish");
        }
    }

    #[test]
    fn solves_two_stage_to_known_optimum() {
        let mut dp = TwoStage::new();
        assert!(dp.assert_preconditions_dp().is_ok());
        drive(&mut dp);
        assert_eq!(dp.get_v(0), &[3.0, 6.0]);
        assert_eq!(dp.get_v(1), &[1.0, 3.0]);
        assert_eq!(dp.get_action(0, 0), Some(1));
        assert_eq!(dp.get_action(0, 1), Some(1));
        assert_eq!(dp.get_action(1, 0), Some(0));
        assert_eq!(dp.get_action(1, 1), Some(1));
    }

    #[test]
    fn discount_factor_shapes_value() {
        let mut dp = DiscountedLoop::new();
        drive(&mut dp);
        assert_eq!(dp.get_v(1)[0], 10.0);
        assert_eq!(dp.get_v(0)[0], 15.0);
        assert_eq!(dp.get_action(0, 0), Some(1));
    }

    #[test]
    fn finishes_and_counts_down_to_stage_zero() {
        let mut dp = TwoStage::new();
        dp.bootstrap();
        assert_eq!(dp.get_current_stage(), 1);
        assert!(!dp.is_finished());
        dp.run_dp_step(); // stage 1
        assert_eq!(dp.get_current_stage(), 0);
        assert!(!dp.is_finished());
        dp.run_dp_step(); // stage 0
        assert!(dp.is_finished());
        assert_eq!(dp.get_current_stage(), 0);
        // stage history holds terminal (T=2) + stage 1 + stage 0 rows.
        assert_eq!(dp.dp_state().stage_history.len(), 3);
    }
}
