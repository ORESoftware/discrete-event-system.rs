//! Port of `src/des/general/des-base/belief-state.ts` — belief-state base for
//! POMDP / HMM filtering: a Bayesian belief update plus a `pickAction` policy
//! hook. The belief `b ∈ Δ(S)` evolves via
//!
//! ```text
//! b_{k+1}(s') = (η · O(o|s', a)) · Σ_s T(s'|s, a) b_k(s)
//! ```
//!
//! ## Rust shape
//!
//! `abstract class BeliefStateStation<A, O> extends DESStation` becomes the
//! [`BeliefStateStation`] trait extending [`DESStation`]:
//!
//!   * Belief state (`belief: number[]`, the `core: POMDPCore` held by
//!     composition, and the belief trace) lives in [`BeliefCore`]; the model is
//!     a `Box<dyn POMDPCore<A, O>>`.
//!   * Only `pickAction` is abstract → the required
//!     [`BeliefStateStation::pick_action`]. `beliefUpdate` /
//!     `observationLikelihood` are concrete → provided defaults.
//!   * `runTimeStep` consumes `(action, observation)` tuples and emits a
//!     [`BeliefToken`]; concrete stations delegate from
//!     [`DESStation::run_time_step`].
//!   * `throw new Error` on length mismatch → `panic!` (invariant violation).

use std::rc::Rc;

use super::station::DESStation;

/// `(action, observation)` inbox channel.
pub const CH_INPUT: &str = "ao";
/// Posterior-belief outbox channel.
pub const CH_BELIEF: &str = "belief";

/// One interaction tuple delivered to the filter.
pub struct ActionObservationToken<A = f64, O = f64> {
    pub action: A,
    pub observation: O,
}

impl<A, O> ActionObservationToken<A, O> {
    pub fn new(action: A, observation: O) -> Self {
        ActionObservationToken { action, observation }
    }
}

/// The posterior belief emitted after an update.
pub struct BeliefToken {
    pub belief: Vec<f64>,
}

/// The POMDP / HMM model (functions, not dense arrays, so large `S` is cheap).
pub trait POMDPCore<A = f64, O = f64> {
    /// `|S|`.
    fn num_states(&self) -> usize;
    /// `|A|`.
    fn num_actions(&self) -> usize;
    /// `|Ω|`.
    fn num_observations(&self) -> usize;
    /// `T(s, a, s')`.
    fn transition_prob(&self, s: usize, a: &A, sp: usize) -> f64;
    /// `O(s', a, o)` — observation likelihood after taking `a`, ending in `s'`.
    fn observation_prob(&self, sp: usize, a: &A, o: &O) -> f64;
}

/// Belief state (the fields of the TS abstract class).
pub struct BeliefCore<A, O> {
    pub belief: Vec<f64>,
    pub core: Box<dyn POMDPCore<A, O>>,
    /// Trace of beliefs (always recorded; seeded with the initial belief).
    pub belief_history: Vec<Vec<f64>>,
}

impl<A, O> BeliefCore<A, O> {
    /// Build the belief state. With no `initial`, a uniform prior is used; an
    /// explicit prior whose length disagrees with `num_states` panics.
    pub fn new(core: Box<dyn POMDPCore<A, O>>, initial: Option<&[f64]>) -> Self {
        let belief = match initial {
            Some(b) => {
                if b.len() != core.num_states() {
                    panic!(
                        "initial belief length {} != numStates {}",
                        b.len(),
                        core.num_states()
                    );
                }
                b.to_vec()
            }
            None => {
                let n = core.num_states();
                vec![1.0 / n as f64; n]
            }
        };
        let mut s = BeliefCore {
            belief: belief.clone(),
            core,
            belief_history: Vec::new(),
        };
        s.belief_history.push(belief);
        s
    }
}

/// Belief-tracking base for POMDP / filtering stations.
pub trait BeliefStateStation<A: 'static = f64, O: 'static = f64>: DESStation {
    /// Borrow belief state.
    fn belief_core(&self) -> &BeliefCore<A, O>;
    /// Mutably borrow belief state.
    fn belief_core_mut(&mut self) -> &mut BeliefCore<A, O>;

    // ── HOOK (abstract) ────────────────────────────────────────────────────────

    /// Pick the next action given the current belief — the "policy" half of any
    /// POMDP algorithm.
    fn pick_action(&self, b: &[f64]) -> A;

    // ── PROVIDED ───────────────────────────────────────────────────────────────

    /// `hasWork` override: any pending `(a, o)` tuple counts as work.
    fn belief_has_work(&self) -> bool {
        self.core().inbox_size(CH_INPUT) > 0
    }

    fn belief_run_time_step(&mut self)
    where
        A: 'static,
        O: 'static,
    {
        let tokens = self.core_mut().drain::<ActionObservationToken<A, O>>(CH_INPUT);
        for t in tokens {
            let cur = self.belief_core().belief.clone();
            let next = self.belief_update(&cur, &t.action, &t.observation);
            self.belief_core_mut().belief = next.clone();
            self.belief_core_mut().belief_history.push(next.clone());
            self.core_mut().emit(Rc::new(BeliefToken { belief: next }), CH_BELIEF);
        }
    }

    /// Bayesian belief update `b' = η · O(o|s',a) Σ_s T(s'|s,a) b(s)`.
    fn belief_update(&self, b: &[f64], a: &A, o: &O) -> Vec<f64> {
        let core = &self.belief_core().core;
        let n = core.num_states();
        let mut bp = vec![0.0_f64; n];
        let mut total = 0.0;
        for sp in 0..n {
            let mut p_trans = 0.0;
            for s in 0..n {
                p_trans += core.transition_prob(s, a, sp) * b[s];
            }
            let v = core.observation_prob(sp, a, o) * p_trans;
            bp[sp] = v;
            total += v;
        }
        if total > 0.0 {
            for x in bp.iter_mut() {
                *x /= total;
            }
        } else {
            for x in bp.iter_mut() {
                *x = 1.0 / n as f64;
            }
        }
        bp
    }

    /// `P(o | b, a)`.
    fn observation_likelihood(&self, b: &[f64], a: &A, o: &O) -> f64 {
        let core = &self.belief_core().core;
        let n = core.num_states();
        let mut total = 0.0;
        for sp in 0..n {
            let mut p_trans = 0.0;
            for s in 0..n {
                p_trans += core.transition_prob(s, a, sp) * b[s];
            }
            total += core.observation_prob(sp, a, o) * p_trans;
        }
        total
    }

    // ── PUBLIC ACCESSORS ───────────────────────────────────────────────────────

    fn get_belief(&self) -> &[f64] {
        &self.belief_core().belief
    }

    fn set_belief(&mut self, b: &[f64]) {
        let n = self.belief_core().core.num_states();
        if b.len() != n {
            panic!("belief length {} != numStates {}", b.len(), n);
        }
        self.belief_core_mut().belief = b.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::station::{StationCore, DESStation};
    use std::any::Any;

    /// 2-state identity-transition HMM; the observation favours the matching
    /// state with weight 0.9 (vs 0.1 for the other).
    struct TwoStateCore;

    impl POMDPCore<usize, usize> for TwoStateCore {
        fn num_states(&self) -> usize {
            2
        }
        fn num_actions(&self) -> usize {
            1
        }
        fn num_observations(&self) -> usize {
            2
        }
        fn transition_prob(&self, s: usize, _a: &usize, sp: usize) -> f64 {
            if s == sp {
                1.0
            } else {
                0.0
            }
        }
        fn observation_prob(&self, sp: usize, _a: &usize, o: &usize) -> f64 {
            if sp == *o {
                0.9
            } else {
                0.1
            }
        }
    }

    struct Filter {
        core: StationCore,
        bc: BeliefCore<usize, usize>,
    }

    impl Filter {
        fn new(initial: Option<&[f64]>) -> Self {
            Filter {
                core: StationCore::new("filter"),
                bc: BeliefCore::new(Box::new(TwoStateCore), initial),
            }
        }
    }

    impl DESStation for Filter {
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

    impl BeliefStateStation<usize, usize> for Filter {
        fn belief_core(&self) -> &BeliefCore<usize, usize> {
            &self.bc
        }
        fn belief_core_mut(&mut self) -> &mut BeliefCore<usize, usize> {
            &mut self.bc
        }
        fn pick_action(&self, b: &[f64]) -> usize {
            // Argmax belief (greedy MLS policy).
            let mut best = 0;
            for i in 1..b.len() {
                if b[i] > b[best] {
                    best = i;
                }
            }
            best
        }
    }

    #[test]
    fn uniform_prior_seeds_history() {
        let f = Filter::new(None);
        assert_eq!(f.get_belief(), &[0.5, 0.5]);
        assert_eq!(f.bc.belief_history.len(), 1);
    }

    #[test]
    fn belief_update_normalizes() {
        let f = Filter::new(None);
        let updated = f.belief_update(&[0.5, 0.5], &0, &0);
        let sum: f64 = updated.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "belief did not normalize: {sum}");
        // observing o=0 with identity transition => 0.9*0.5 vs 0.1*0.5 -> 0.9 / 0.1
        assert!((updated[0] - 0.9).abs() < 1e-9);
        assert!((updated[1] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn run_time_step_updates_and_picks() {
        let mut f = Filter::new(None);
        assert!(!f.has_work());
        f.core_mut()
            .take(Rc::new(ActionObservationToken::new(0usize, 0usize)), CH_INPUT);
        assert!(f.has_work());
        f.run_time_step();
        let b = f.get_belief().to_vec();
        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(f.pick_action(&b), 0);
        // initial + one update recorded
        assert_eq!(f.bc.belief_history.len(), 2);
        assert!(!f.has_work());
    }
}
