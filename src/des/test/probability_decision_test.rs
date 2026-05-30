//! Port of src/des/test/probability-decision-test.ts
//!
//! RECONCILIATION HARVEST (main + main-lb): `origin/main-lb`'s
//! `tests/probability_decision_test.rs` exercised exact branch selection at RNG
//! boundaries — a scenario `main`'s suite lacked (the inline tests only cover
//! construction/validation and the empty-queue no-op). Re-expressed against
//! `main`'s `ProbabilityDecisionEntity`, whose `run_time_step` draws from an
//! injected `RandomSource` and routes via the cumulative `r < sum` rule.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::entity_decision::probability_decision::{Branch, ProbabilityDecisionEntity};
    use crate::des::entity_moving::moving::{BasicMovingEntity, MovingEntity};
    use crate::des::entity_sink::sink::EntitySink;
    use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
    use crate::des::r#abstract::r#abstract::Entity;
    use crate::des::random_variables::rv::{BernoulliRandomVariable, RandomVariable};
    use crate::des::shared::capabilities::{RandomSource, SeededRandom};
    use crate::des::shared::precision::bgn;

    /// Deterministic RNG that replays a fixed float sequence (cycling), so we can
    /// pin which branch each draw selects.
    struct FixedSequenceRandom {
        seq: Vec<f64>,
        i: usize,
    }
    impl RandomSource for FixedSequenceRandom {
        fn next_float(&mut self) -> f64 {
            let v = self.seq[self.i % self.seq.len()];
            self.i += 1;
            v
        }
    }

    fn rv() -> Box<dyn RandomVariable> {
        Box::new(BernoulliRandomVariable::new(Box::new(SeededRandom::new(7))))
    }

    #[test]
    fn decimal_probabilities_route_at_rng_boundaries() {
        // 25% to branch 0, 75% to branch 1. Cumulative rule (`r < sum`):
        //   draw 0.20 -> 0.20 < 0.25 -> branch 0
        //   draw 0.90 -> not < 0.25, but < 1.00 -> branch 1
        let probs = vec![
            Branch { index: 0, prob: bgn(0.25) },
            Branch { index: 1, prob: bgn(0.75) },
        ];
        let rng = Box::new(FixedSequenceRandom { seq: vec![0.20, 0.90], i: 0 });
        let mut d = ProbabilityDecisionEntity::new("pd".to_string(), probs, rv(), rng);

        let b0 = Rc::new(RefCell::new(EntitySink::new("b0".to_string())));
        let b1 = Rc::new(RefCell::new(EntitySink::new("b1".to_string())));
        let t0: Rc<RefCell<dyn HasInput>> = b0.clone();
        let t1: Rc<RefCell<dyn HasInput>> = b1.clone();
        d.add_out_connection(t0);
        d.add_out_connection(t1);
        // Index the out-connections so the sampled branch index resolves to an
        // edge (both trait impls delegate to the same private indexer).
        HasOutput::do_setup_after_input_conn(&mut d);

        for _ in 0..2 {
            let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
            d.take_item(m);
        }
        d.run_time_step(bgn(1.0));

        // First token (draw 0.20) -> b0; second (draw 0.90) -> b1.
        assert_eq!(b0.borrow().destroyed_count, 1);
        assert_eq!(b1.borrow().destroyed_count, 1);
        assert_eq!(d.queue.len(), 0);
    }
}
