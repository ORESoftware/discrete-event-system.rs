//! TypeScript source: `src/des/test/shortest-path-test.ts`
//! Rust target: `tests/shortest_path_test.rs`

use discrete_event_system_rs::des::general::shortest_path_des::{
    build_random_graph, build_small_chain_graph, reconstruct_path, shortest_path_bellman_ford_des,
    shortest_path_dijkstra_des, BellmanFordOptions, Edge, Graph, ShortestPathError,
};
use discrete_event_system_rs::DesDecimal;

fn dec(value: i64) -> DesDecimal {
    DesDecimal::from(value)
}

fn close(actual: Option<DesDecimal>, expected: i64) {
    assert_eq!(actual, Some(dec(expected)));
}

#[test]
fn small_chain_distances_bellman_ford() {
    let graph = build_small_chain_graph();
    let result = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    close(result.distance[0], 0);
    close(result.distance[1], 1);
    close(result.distance[2], 3);
    close(result.distance[3], 5);
    close(result.distance[4], 6);
}

#[test]
fn small_chain_distances_dijkstra() {
    let graph = build_small_chain_graph();
    let result = shortest_path_dijkstra_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    close(result.distance[0], 0);
    close(result.distance[1], 1);
    close(result.distance[2], 3);
    close(result.distance[3], 5);
    close(result.distance[4], 6);
}

#[test]
fn reconstructs_paths() {
    let graph = build_small_chain_graph();
    let result = shortest_path_dijkstra_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    assert_eq!(reconstruct_path(&result, 0, 4), Some(vec![0, 1, 2, 3, 4]));
    assert_eq!(reconstruct_path(&result, 0, 0), Some(vec![0]));
}

#[test]
fn unreachable_nodes_have_infinite_distance() {
    let graph = Graph {
        num_nodes: 4,
        edges: vec![
            vec![Edge {
                to: 1,
                weight: dec(5),
            }],
            vec![],
            vec![Edge {
                to: 3,
                weight: dec(1),
            }],
            vec![],
        ],
        coordinates: None,
        node_names: None,
    };
    let result = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    assert_eq!(result.distance[2], None);
    assert_eq!(result.distance[3], None);
    assert_eq!(result.distance[1], Some(dec(5)));
    assert_eq!(reconstruct_path(&result, 0, 3), None);
}

#[test]
fn bellman_ford_and_dijkstra_agree_on_random_non_negative_graphs() {
    for seed in [1, 7, 13, 42, 99] {
        let graph = build_random_graph(12, 0.4, 1.0, 9.0, seed);
        let bellman =
            shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
        let dijkstra =
            shortest_path_dijkstra_des(&graph, 0, BellmanFordOptions::default()).unwrap();
        for (a, b) in bellman.distance.iter().zip(dijkstra.distance.iter()) {
            assert_eq!(a, b);
        }
    }
}

#[test]
fn negative_weights_are_bellman_ford_only() {
    let graph = Graph {
        num_nodes: 3,
        edges: vec![
            vec![
                Edge {
                    to: 1,
                    weight: dec(5),
                },
                Edge {
                    to: 2,
                    weight: dec(-2),
                },
            ],
            vec![Edge {
                to: 2,
                weight: dec(1),
            }],
            vec![],
        ],
        coordinates: None,
        node_names: None,
    };
    let bellman = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    close(bellman.distance[2], -2);
    assert!(!bellman.has_negative_cycle_from_source);
    assert!(matches!(
        shortest_path_dijkstra_des(&graph, 0, BellmanFordOptions::default()),
        Err(ShortestPathError::NegativeWeightForDijkstra { .. })
    ));
}

#[test]
fn bellman_ford_detects_reachable_negative_cycle() {
    let graph = Graph {
        num_nodes: 3,
        edges: vec![
            vec![Edge {
                to: 1,
                weight: dec(1),
            }],
            vec![Edge {
                to: 2,
                weight: dec(-3),
            }],
            vec![Edge {
                to: 1,
                weight: dec(1),
            }],
        ],
        coordinates: None,
        node_names: None,
    };
    let result = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    assert!(result.has_negative_cycle_from_source);
}

#[test]
fn trace_and_wave_events_have_consistent_shape() {
    let graph = build_small_chain_graph();
    let result = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    assert_eq!(result.trace.len(), result.iterations + 1);
    assert_eq!(result.wave_events.len(), result.iterations);
    for snapshot in &result.trace {
        assert_eq!(snapshot.len(), graph.num_nodes);
    }
}

#[test]
fn bellman_ford_is_reproducible() {
    let graph = build_random_graph(15, 0.4, 1.0, 5.0, 99);
    let first = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    let second = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    assert_eq!(first.distance, second.distance);
    assert_eq!(first.predecessor, second.predecessor);
    assert_eq!(first.iterations, second.iterations);
}

#[test]
fn bellman_ford_respects_worst_case_bound() {
    let n = 6;
    let mut edges = Vec::new();
    for node in 0..(n - 1) {
        edges.push(vec![Edge {
            to: node + 1,
            weight: dec(1),
        }]);
    }
    edges.push(vec![]);
    let graph = Graph {
        num_nodes: n,
        edges,
        coordinates: None,
        node_names: None,
    };
    let result = shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
    for node in 0..n {
        close(result.distance[node], node as i64);
    }
    assert!(result.iterations <= n);
}
