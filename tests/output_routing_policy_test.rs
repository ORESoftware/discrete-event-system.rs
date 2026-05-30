//! TypeScript source: `src/des/test/output-routing-policy-test.ts`
//! Rust target: `tests/output_routing_policy_test.rs`

use discrete_event_system_rs::core::EntityConnection;
use discrete_event_system_rs::des::entity_processing::per_individual_processor::{
    PerIndividualProcessor, PerIndividualProcessorOpts, PerIndividualSink,
};
use discrete_event_system_rs::des::entity_routing::output_routing_policy::{
    OutputConnectionRouter, OutputRoutingPolicy,
};
use discrete_event_system_rs::DesDecimal;

fn conn(id: &str) -> EntityConnection {
    EntityConnection {
        id: id.to_owned(),
        source: "source".to_owned(),
        target: id.to_owned(),
        channel: None,
    }
}

fn ids(connections: &[EntityConnection]) -> String {
    connections
        .iter()
        .map(|connection| connection.target.as_str())
        .collect::<Vec<_>>()
        .join("")
}

#[derive(Debug)]
struct Sink<T> {
    id: String,
    cap: usize,
    received: Vec<T>,
}

impl<T> Sink<T> {
    fn new(id: &str) -> Self {
        Self::with_cap(id, usize::MAX)
    }

    fn with_cap(id: &str, cap: usize) -> Self {
        Self {
            id: id.to_owned(),
            cap,
            received: Vec::new(),
        }
    }
}

impl<T> PerIndividualSink<T> for Sink<T> {
    fn id(&self) -> &str {
        &self.id
    }

    fn accept_item(&self, _item: &T) -> bool {
        self.received.len() < self.cap
    }

    fn take_item(&mut self, item: T) -> discrete_event_system_rs::core::DesResult<()> {
        self.received.push(item);
        Ok(())
    }
}

#[test]
fn round_robin_rotates_through_declared_order() {
    let conns = vec![conn("A"), conn("B"), conn("C")];
    let mut router = OutputConnectionRouter::new(OutputRoutingPolicy::RoundRobin);
    let mut rng = || 0.0;
    let mut picks = String::new();

    for _ in 0..7 {
        let ordered = router.ordered_connections(&conns, &mut rng);
        let accepted = ordered[0].clone();
        picks.push_str(&accepted.target);
        router.mark_accepted(&conns, &accepted);
    }

    assert_eq!(picks, "ABCABCA");
}

#[test]
fn ordered_keeps_declared_order_and_does_not_rotate_after_accept() {
    let conns = vec![conn("A"), conn("B"), conn("C")];
    let mut router = OutputConnectionRouter::new(OutputRoutingPolicy::Ordered);
    let mut rng = || 0.0;

    assert_eq!(ids(&router.ordered_connections(&conns, &mut rng)), "ABC");
    router.mark_accepted(&conns, &conns[0]);
    assert_eq!(ids(&router.ordered_connections(&conns, &mut rng)), "ABC");
}

#[test]
fn default_policy_matches_typescript_random_default() {
    let router = OutputConnectionRouter::new(OutputRoutingPolicy::default());
    assert_eq!(router.policy, OutputRoutingPolicy::Random);
}

#[test]
fn round_robin_ignores_unknown_accepted_connection() {
    let conns = vec![conn("A"), conn("B")];
    let mut router = OutputConnectionRouter::new(OutputRoutingPolicy::RoundRobin);
    router.mark_accepted(&conns, &conn("missing"));
    assert_eq!(router.next_round_robin_index, 0);
}

#[test]
fn per_individual_processor_round_robin_sends_two_to_each_declared_sink() {
    let mut processor = PerIndividualProcessor::new(
        "P-rr",
        PerIndividualProcessorOpts::with_output_routing(
            || DesDecimal::ZERO,
            OutputRoutingPolicy::RoundRobin,
        ),
    );
    processor.add_out_connection("A");
    processor.add_out_connection("B");
    processor.add_out_connection("C");
    for entity in 0..6 {
        processor.take_item(entity);
    }

    let mut a = Sink::new("A");
    let mut b = Sink::new("B");
    let mut c = Sink::new("C");
    {
        let mut sinks: [&mut dyn PerIndividualSink<usize>; 3] = [&mut a, &mut b, &mut c];
        processor
            .run_time_step_with_sinks(DesDecimal::ONE, &mut sinks)
            .expect("routing should succeed");
    }

    assert_eq!(a.received.len(), 2);
    assert_eq!(b.received.len(), 2);
    assert_eq!(c.received.len(), 2);
}

#[test]
fn per_individual_processor_ordered_keeps_first_sink_as_priority() {
    let mut processor = PerIndividualProcessor::new(
        "P-ordered",
        PerIndividualProcessorOpts::with_output_routing(
            || DesDecimal::ZERO,
            OutputRoutingPolicy::Ordered,
        ),
    );
    processor.add_out_connection("A");
    processor.add_out_connection("B");
    processor.add_out_connection("C");
    for entity in 0..6 {
        processor.take_item(entity);
    }

    let mut a = Sink::new("A");
    let mut b = Sink::new("B");
    let mut c = Sink::new("C");
    {
        let mut sinks: [&mut dyn PerIndividualSink<usize>; 3] = [&mut a, &mut b, &mut c];
        processor
            .run_time_step_with_sinks(DesDecimal::ONE, &mut sinks)
            .expect("routing should succeed");
    }

    assert_eq!(a.received.len(), 6);
    assert_eq!(b.received.len(), 0);
    assert_eq!(c.received.len(), 0);
}

#[test]
fn per_individual_processor_round_robin_skips_full_acceptors() {
    let mut processor = PerIndividualProcessor::new(
        "P-cap",
        PerIndividualProcessorOpts::with_output_routing(
            || DesDecimal::ZERO,
            OutputRoutingPolicy::RoundRobin,
        ),
    );
    processor.add_out_connection("A");
    processor.add_out_connection("B");
    for entity in 0..4 {
        processor.take_item(entity);
    }

    let mut a = Sink::with_cap("A", 1);
    let mut b = Sink::new("B");
    {
        let mut sinks: [&mut dyn PerIndividualSink<usize>; 2] = [&mut a, &mut b];
        processor
            .run_time_step_with_sinks(DesDecimal::ONE, &mut sinks)
            .expect("routing should succeed");
    }

    assert_eq!(a.received.len(), 1);
    assert_eq!(b.received.len(), 3);
}
