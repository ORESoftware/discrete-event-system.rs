//! Focused Decimal coverage for `src/des/entity-decision/probability-decision.ts`.

use discrete_event_system_rs::core::EntityConnection;
use discrete_event_system_rs::des::entity_decision::probability_decision::{
    ProbabilisticDecision, ProbabilityDecisionOpts,
};
use discrete_event_system_rs::DesDecimal;

fn conn(id: &str) -> EntityConnection {
    EntityConnection {
        id: id.to_owned(),
        source: "decision".to_owned(),
        target: id.to_owned(),
        channel: None,
    }
}

#[test]
fn probabilities_validate_with_exact_decimal_sums() {
    let opts = ProbabilityDecisionOpts {
        probabilities: vec![
            DesDecimal::new(1, 1),
            DesDecimal::new(2, 1),
            DesDecimal::new(7, 1),
        ],
    };

    assert!(opts.validate().is_ok());
}

#[test]
fn probabilities_reject_non_unit_decimal_sums() {
    let opts = ProbabilityDecisionOpts {
        probabilities: vec![DesDecimal::new(3, 1), DesDecimal::new(3, 1)],
    };

    assert!(opts.validate().is_err());
}

#[test]
fn decimal_probabilities_route_against_float_rng_boundary() {
    let opts = ProbabilityDecisionOpts {
        probabilities: vec![DesDecimal::new(25, 2), DesDecimal::new(75, 2)],
    };
    let mut decision = ProbabilisticDecision::<usize>::new("D", opts).expect("valid opts");
    decision.decision.add_out_connection(conn("A"));
    decision.decision.add_out_connection(conn("B"));

    let mut first_sample = || 0.20;
    let mut second_sample = || 0.90;

    assert_eq!(
        decision
            .choose_connection(&mut first_sample)
            .map(|c| c.target.as_str()),
        Some("A")
    );
    assert_eq!(
        decision
            .choose_connection(&mut second_sample)
            .map(|c| c.target.as_str()),
        Some("B")
    );
}
