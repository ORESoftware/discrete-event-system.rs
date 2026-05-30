//! Canonical use path: `crate::des::entity_decision::probability_decision::*`
//!
//! Port of `src/des/entity-decision/probability-decision.ts` — routes each queued
//! item to ONE out-connection sampled from a probability vector.
//!
//! TS `class ProbabilityDecisionEntity extends AbstractBidirectionalEntity
//! implements HasComputedProperties, HasInternalQueue`.
//!
//! PORT NOTES:
//!   * DETERMINISM: `bgn(Math.random())` becomes a draw from an injected
//!     `RandomSource`. The TS defaulted the ctor param to `DEFAULT_RANDOM`; the
//!     Rust `capabilities` module has NO global default, so the `RandomSource` is
//!     injected explicitly (and the unused `rv` is still stored for parity).
//!   * `process.exit(0)` on a void-dequeue is a HARD ABORT — NOT ported. We drain
//!     a `VecDeque`, so there is no void sentinel to hit.
//!   * the inner `for (const v of probabilities)` shadowed the outer item `v`; the
//!     port names them `branch` (probability) and `item` (token) so the token is
//!     routed (the intended behaviour after the inner block's scope ends).
//!   * `connectionsOutByIndex` / `connectionsInByIndex` (`Map<number, _>`) ->
//!     `HashMap<usize, Rc<RefCell<EntityConnection>>>`. The `doSetupAfter*Conn`
//!     indexing lives in inherent helpers that all trait impls delegate to.
//!   * the constructor sums the branch probabilities and `panic!`s if the total
//!     is `> 1.00001` or `< 0.9999`; `getWithComputedProperties()` returns an
//!     `EntityGraphData` payload; `reg.registerDecision(this)` is omitted (no
//!     self-`Rc` in a constructor).

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::des::entity_decision::decision::DecisionEntityGraph;
use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::entity_travel::time_delay::TimeDelayEntityGraphData;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{BidirectionalCore, Entity, EntityConnection, EntityCore};
use crate::des::random_variables::rv::RandomVariable;
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::precision::{bgn, Decimal};

/// One `opts.probabilities` entry: `{ index, prob }`.
#[derive(Clone, Copy, Debug)]
pub struct Branch {
    pub index: i64,
    pub prob: Decimal,
}

/// `class ProbabilityDecisionEntity`.
pub struct ProbabilityDecisionEntity {
    pub bi: BidirectionalCore,
    pub queue: VecDeque<Rc<RefCell<dyn MovingEntity>>>,
    pub connections_out_by_index: HashMap<usize, Rc<RefCell<EntityConnection>>>,
    pub connections_in_by_index: HashMap<usize, Rc<RefCell<EntityConnection>>>,
    pub rv: Box<dyn RandomVariable>,
    pub rng: Box<dyn RandomSource>,
    pub probabilities: Vec<Branch>,
    pub max_queue_size: i64,
}

impl ProbabilityDecisionEntity {
    /// `new(id, { rv, probabilities }, rng)`. Panics if the branch probabilities
    /// do not sum to ~1.
    pub fn new(
        id: String,
        probabilities: Vec<Branch>,
        rv: Box<dyn RandomVariable>,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        // PORT NOTE: `reg.registerDecision(this)` is deferred to the integrator
        // (a constructor has no `Rc<RefCell<Self>>` to register).

        let mut check_sum = bgn(0.0);
        for branch in &probabilities {
            check_sum += branch.prob;
        }

        if check_sum > bgn(1.00001) {
            eprintln!(
                "[decision:{id}] branch probabilities sum to {check_sum} (> 1) across {} branches — they must sum to 1.",
                probabilities.len()
            );
            panic!("probability sum too high");
        }

        if check_sum < bgn(0.9999) {
            eprintln!(
                "[decision:{id}] branch probabilities sum to {check_sum} (< 1) across {} branches — they must sum to 1.",
                probabilities.len()
            );
            panic!("probability sum too high");
        }

        ProbabilityDecisionEntity {
            bi: BidirectionalCore::new(id),
            queue: VecDeque::new(),
            connections_out_by_index: HashMap::new(),
            connections_in_by_index: HashMap::new(),
            rv,
            rng,
            probabilities,
            max_queue_size: -1,
        }
    }

    pub fn get_graph_data_typed(&self) -> TimeDelayEntityGraphData {
        TimeDelayEntityGraphData { processed_count: 3 }
    }

    pub fn decision_graph(&self) -> DecisionEntityGraph {
        DecisionEntityGraph
    }

    /// `doAudit()` -> `{ totalSize }`.
    pub fn do_audit(&self) -> usize {
        self.queue.len()
    }

    /// `doValidationBeforeRun()` — connection count must match branch count.
    pub fn validate_before_run(&self) -> bool {
        if self.bi.connections_out.len() != self.probabilities.len() {
            eprintln!(
                "[decision:{}] validation failed: {} out-connections but {} branch probabilities — these must match.",
                self.bi.entity.id,
                self.bi.connections_out.len(),
                self.probabilities.len()
            );
            panic!("connections out size must be the same size as probabilities.");
        }
        true
    }

    /// `doSetupAfterInputConn()` — index the out-connections by position.
    fn setup_after_input_conn(&mut self) -> bool {
        self.connections_out_by_index.clear();
        let mut index: i64 = -1;
        for v in self.bi.connections_out.clone() {
            index += 1;
            self.connections_out_by_index.insert(index as usize, v);
        }
        true
    }

    /// `doSetupAfterOutputConn()` — index the in-connections by position.
    fn setup_after_output_conn(&mut self) -> bool {
        self.connections_in_by_index.clear();
        let mut index: i64 = -1;
        for v in self.bi.connections_in.clone() {
            index += 1;
            self.connections_in_by_index.insert(index as usize, v);
        }
        true
    }
}

impl Entity for ProbabilityDecisionEntity {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {
        // TS: `throw new Error("Method not implemented.")`.
        panic!("Method not implemented.");
    }
    fn do_validation_before_run(&mut self) -> bool {
        self.validate_before_run()
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("processedCount", 3.0)
    }
    fn get_with_computed_properties(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("queue.size", self.queue.len() as f64)
            .with("branches", self.probabilities.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        let mut rejected: Vec<Rc<RefCell<dyn MovingEntity>>> = Vec::new();

        while let Some(item) = self.queue.pop_front() {
            let r = bgn(self.rng.next_float());

            let mut sum = bgn(0.0);
            let mut index: i64 = -1;
            for branch in &self.probabilities {
                index += 1;
                sum += branch.prob;
                if r < sum {
                    break;
                }
            }

            let out_conn = self
                .connections_out_by_index
                .get(&(index as usize))
                .cloned();
            let out_conn = match out_conn {
                Some(c) => c,
                None => {
                    eprintln!(
                        "[decision:{}] sampled branch index {index} has no out-connection (have {} indexed connections, {} probabilities) — branch/connection mismatch.",
                        self.bi.entity.id,
                        self.connections_out_by_index.len(),
                        self.probabilities.len()
                    );
                    panic!("missing connection with index:");
                }
            };

            let target = out_conn.borrow().get_target();
            let routed = match target {
                Some(t) if t.borrow_mut().accept_item(item.clone()) => {
                    t.borrow_mut().take_item(item.clone());
                    true
                }
                _ => false,
            };
            if !routed {
                rejected.push(item);
            }
        }

        for item in rejected {
            self.queue.push_back(item);
        }
    }
}

impl HasInternalQueue for ProbabilityDecisionEntity {
    fn max_queue_size(&self) -> usize {
        if self.max_queue_size < 0 {
            usize::MAX
        } else {
            self.max_queue_size as usize
        }
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.queue.len() < 1
    }
}

impl HasInput for ProbabilityDecisionEntity {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        self.bi.accept_item()
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        self.queue.push_back(m);
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        self.setup_after_input_conn()
    }
    fn notify_sources(&mut self) {
        self.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        self.setup_after_output_conn()
    }
    fn add_in_connection(
        &mut self,
        _source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_in_conn();
        self.bi.add_in_connection(conn.clone());
        Some(conn)
    }
}

impl HasManyInputConnections for ProbabilityDecisionEntity {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for ProbabilityDecisionEntity {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_out_conn(target);
        self.bi.add_out_connection(conn.clone());
        Some(conn)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        self.setup_after_input_conn()
    }
    fn notify_targets(&mut self) {
        self.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        self.setup_after_output_conn()
    }
}

impl HasManyOutputConnections for ProbabilityDecisionEntity {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::entity_moving::moving::BasicMovingEntity;
    use crate::des::random_variables::rv::BernoulliRandomVariable;
    use crate::des::shared::capabilities::SeededRandom;

    fn rv() -> Box<dyn RandomVariable> {
        Box::new(BernoulliRandomVariable::new(Box::new(SeededRandom::new(2))))
    }

    #[test]
    fn constructs_with_valid_probabilities() {
        let probs = vec![
            Branch {
                index: 0,
                prob: bgn(0.5),
            },
            Branch {
                index: 1,
                prob: bgn(0.5),
            },
        ];
        let d = ProbabilityDecisionEntity::new(
            "prob1".to_string(),
            probs,
            rv(),
            Box::new(SeededRandom::new(3)),
        );
        assert_eq!(d.probabilities.len(), 2);
    }

    #[test]
    #[should_panic(expected = "probability sum too high")]
    fn rejects_probabilities_that_do_not_sum_to_one() {
        let probs = vec![Branch {
            index: 0,
            prob: bgn(0.5),
        }];
        let _ = ProbabilityDecisionEntity::new(
            "prob2".to_string(),
            probs,
            rv(),
            Box::new(SeededRandom::new(3)),
        );
    }

    #[test]
    fn empty_queue_run_step_is_noop() {
        let probs = vec![Branch {
            index: 0,
            prob: bgn(1.0),
        }];
        let mut d = ProbabilityDecisionEntity::new(
            "prob3".to_string(),
            probs,
            rv(),
            Box::new(SeededRandom::new(3)),
        );
        // No items enqueued -> stepping is a no-op (the while-loop body, which
        // would panic on a missing out-connection, never runs).
        d.run_time_step(bgn(0.1));
        assert_eq!(d.queue.len(), 0);
    }

    #[test]
    #[should_panic(expected = "missing connection with index")]
    fn run_step_without_wired_branch_panics() {
        // Faithful to the TS source, which throws `missing connection with index`
        // when a sampled branch has no out-connection.
        let probs = vec![Branch {
            index: 0,
            prob: bgn(1.0),
        }];
        let mut d = ProbabilityDecisionEntity::new(
            "prob4".to_string(),
            probs,
            rv(),
            Box::new(SeededRandom::new(3)),
        );
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
        d.take_item(m);
        d.run_time_step(bgn(0.1));
    }
}
