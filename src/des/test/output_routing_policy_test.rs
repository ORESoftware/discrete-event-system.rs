//! Port of src/des/test/output-routing-policy-test.ts
//!
//! RECONCILIATION HARVEST (main + main-lb): the original port deferred this
//! bucket because `entity_routing`/`entity_processing` were not yet ported. Both
//! now exist, and `origin/main-lb`'s `tests/output_routing_policy_test.rs`
//! carried several routing-distribution scenarios this file lacked. They are
//! re-expressed here against `main`'s `OutputConnectionRouter` +
//! `PerIndividualProcessor` API (which routes by INDEX into `Rc<RefCell<…>>`
//! edges rather than `main-lb`'s `PartialEq` chain).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::des::entity_moving::moving::{BasicMovingEntity, MovingEntity};
    use crate::des::entity_processing::per_individual_processor::{
        PerIndividualProcessor, PerIndividualProcessorOpts,
    };
    use crate::des::entity_routing::output_routing_policy::{
        OutputConnectionRouter, OutputRoutingPolicy,
    };
    use crate::des::entity_sink::sink::EntitySink;
    use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
    use crate::des::r#abstract::r#abstract::Entity;
    use crate::des::shared::precision::bgn;

    // --- router-level policy behaviour -------------------------------------

    #[test]
    fn round_robin_full_rotation_sequence() {
        // Seven consecutive single-target picks cycle A,B,C,A,B,C,A.
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::RoundRobin);
        let conns = vec!["A", "B", "C"];
        let mut picks = String::new();
        for _ in 0..7 {
            let chosen = r.order(&conns)[0];
            picks.push_str(chosen);
            r.mark_accepted(&conns, &chosen);
        }
        assert_eq!(picks, "ABCABCA");
    }

    #[test]
    fn default_policy_is_random() {
        assert_eq!(OutputRoutingPolicy::default(), OutputRoutingPolicy::Random);
    }

    #[test]
    fn round_robin_unknown_accept_leaves_cursor_unchanged() {
        let mut r = OutputConnectionRouter::new(OutputRoutingPolicy::RoundRobin);
        let conns = vec!["A", "B", "C"];
        let _ = r.order(&conns);
        r.mark_accepted(&conns, &"Z"); // not present -> cursor must not advance
        assert_eq!(r.get_cursor(), 0);
    }

    // --- end-to-end distribution through PerIndividualProcessor ------------

    fn opts(policy: OutputRoutingPolicy, duration: f64) -> PerIndividualProcessorOpts {
        PerIndividualProcessorOpts {
            draw_duration: Box::new(move || duration),
            rv: None,
            output_routing: Some(policy),
        }
    }

    fn wire_three_sinks(p: &mut PerIndividualProcessor) -> [Rc<RefCell<EntitySink>>; 3] {
        let sinks = [
            Rc::new(RefCell::new(EntitySink::new("s0".to_string()))),
            Rc::new(RefCell::new(EntitySink::new("s1".to_string()))),
            Rc::new(RefCell::new(EntitySink::new("s2".to_string()))),
        ];
        for sink in &sinks {
            let target: Rc<RefCell<dyn HasInput>> = sink.clone();
            p.add_out_connection(target);
        }
        sinks
    }

    fn feed(p: &mut PerIndividualProcessor, n: usize) {
        for _ in 0..n {
            let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(BasicMovingEntity::new()));
            p.take_item(m);
        }
    }

    #[test]
    fn per_individual_round_robin_balances_three_sinks() {
        let mut p = PerIndividualProcessor::new(
            "rr".to_string(),
            opts(OutputRoutingPolicy::RoundRobin, 1.0),
        );
        let sinks = wire_three_sinks(&mut p);
        feed(&mut p, 6);
        // dt == draw duration -> all six finish service this tick and route.
        p.run_time_step(bgn(1.0));
        assert_eq!(sinks[0].borrow().destroyed_count, 2);
        assert_eq!(sinks[1].borrow().destroyed_count, 2);
        assert_eq!(sinks[2].borrow().destroyed_count, 2);
    }

    #[test]
    fn per_individual_ordered_floods_first_sink() {
        let mut p = PerIndividualProcessor::new(
            "ord".to_string(),
            opts(OutputRoutingPolicy::Ordered, 1.0),
        );
        let sinks = wire_three_sinks(&mut p);
        feed(&mut p, 6);
        p.run_time_step(bgn(1.0));
        // Ordered always offers the first accepting target first; sinks always
        // accept, so every token lands in s0.
        assert_eq!(sinks[0].borrow().destroyed_count, 6);
        assert_eq!(sinks[1].borrow().destroyed_count, 0);
        assert_eq!(sinks[2].borrow().destroyed_count, 0);
    }
}
