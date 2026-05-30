//! TypeScript source: `src/des/test/network-flow-test.ts`
//! Rust target: `tests/network_flow_test.rs`

use discrete_event_system_rs::des::general::network_flow::{run_max_flow, FlowEdge, MaxFlowParams};
use discrete_event_system_rs::DesDecimal;

fn teaching_network() -> MaxFlowParams {
    MaxFlowParams {
        num_nodes: 6,
        source: 0,
        sink: 5,
        node_names: Some(vec![
            "s".to_owned(),
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
            "t".to_owned(),
        ]),
        node_coordinates: Some(vec![
            (90.0, 260.0),
            (260.0, 160.0),
            (260.0, 360.0),
            (520.0, 160.0),
            (520.0, 360.0),
            (760.0, 260.0),
        ]),
        edges: vec![
            edge(0, 1, 16, "s-a"),
            edge(0, 2, 13, "s-b"),
            edge(1, 2, 10, "a-b"),
            edge(2, 1, 4, "b-a"),
            edge(1, 3, 12, "a-c"),
            edge(3, 2, 9, "c-b"),
            edge(2, 4, 14, "b-d"),
            edge(4, 3, 7, "d-c"),
            edge(3, 5, 20, "c-t"),
            edge(4, 5, 4, "d-t"),
        ],
        max_augmentations: None,
    }
}

fn edge(from: usize, to: usize, capacity: i64, name: &str) -> FlowEdge {
    FlowEdge {
        from,
        to,
        capacity: DesDecimal::from(capacity),
        name: Some(name.to_owned()),
    }
}

#[test]
fn animated_logged_max_flow_des_optimization_matches_typescript() {
    let result = run_max_flow(teaching_network()).expect("max-flow should solve");
    assert_eq!(result.max_flow, DesDecimal::from(23));
    assert_eq!(result.min_cut.capacity, result.max_flow);
    assert!(result.validation.iter().all(|check| check.passed));
    assert!(!result.trace.is_empty());
    assert!(result
        .trace
        .windows(2)
        .all(|rows| rows[1].value >= rows[0].value));
    assert!(result
        .edge_flows
        .iter()
        .all(|edge| edge.flow >= DesDecimal::ZERO && edge.flow <= edge.capacity));
}

#[test]
fn max_flow_min_cut_exposes_expected_partition_shape() {
    let result = run_max_flow(teaching_network()).expect("max-flow should solve");
    assert!(result.min_cut.source_side.contains(&0));
    assert!(result.min_cut.sink_side.contains(&5));
    assert!(!result.min_cut.cut_edges.is_empty());
    assert_eq!(result.params.node_names.as_ref().unwrap()[0], "s");
    assert_eq!(
        result.params.node_coordinates.as_ref().unwrap()[5],
        (760.0, 260.0)
    );
}

#[test]
fn max_flow_rejects_negative_capacities_before_running() {
    let err = run_max_flow(MaxFlowParams {
        num_nodes: 2,
        source: 0,
        sink: 1,
        edges: vec![FlowEdge {
            from: 0,
            to: 1,
            capacity: DesDecimal::from(-1),
            name: None,
        }],
        max_augmentations: None,
        node_coordinates: None,
        node_names: None,
    })
    .expect_err("negative capacities should fail");
    assert!(err.contains("capacity"), "unexpected error: {err}");
}
