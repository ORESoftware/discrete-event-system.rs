//! Network-optimization facade over shortest path, max-flow/min-cut,
//! minimum-cost flow, and assignment auction algorithms.

pub use crate::des::general::classical_optimization_models::{
    run_auction_assignment, AuctionAssignmentParams,
};
pub use crate::des::general::max_flow::{
    solve_max_flow, MaxFlowEdge, MaxFlowEdgeFlow, MaxFlowProblem, MaxFlowResult, MaxFlowStatus,
    MaxFlowTraceEntry, MinCut,
};
pub use crate::des::general::min_cost_flow::{
    min_cost_flow_to_lp, solve_min_cost_flow, MinCostFlowArc, MinCostFlowArcResult,
    MinCostFlowProblem, MinCostFlowResult, MinCostFlowStatus, MinCostFlowTraceEntry,
};
pub use crate::des::general::shortest_path_des::{
    reconstruct_path, shortest_path_bellman_ford_des, shortest_path_dijkstra_des, Algorithm,
    BellmanFordOptions, Edge, Graph, SPResult, WaveEvent,
};

/// Compact all-pairs shortest-path distances via repeated Bellman-Ford DES.
pub fn all_pairs_shortest_path(graph: &Graph) -> Vec<Vec<f64>> {
    try_all_pairs_shortest_path(graph).expect("all_pairs_shortest_path: invalid graph")
}

/// Checked all-pairs shortest-path distances via repeated Bellman-Ford DES.
pub fn try_all_pairs_shortest_path(graph: &Graph) -> Result<Vec<Vec<f64>>, String> {
    validate_graph(graph)?;
    Ok((0..graph.num_nodes)
        .map(|source| {
            shortest_path_bellman_ford_des(
                graph,
                source,
                BellmanFordOptions {
                    record_trace: false,
                    ..Default::default()
                },
            )
            .distance
        })
        .collect())
}

/// Convenience constructor for a directed graph from `(from, to, weight)` arcs.
pub fn graph_from_arcs(num_nodes: usize, arcs: &[(usize, usize, f64)]) -> Result<Graph, String> {
    if num_nodes == 0 {
        return Err("num_nodes must be positive".to_string());
    }
    let mut edges = vec![Vec::new(); num_nodes];
    for (i, &(from, to, weight)) in arcs.iter().enumerate() {
        if from >= num_nodes || to >= num_nodes {
            return Err(format!("arc {i} endpoint out of range"));
        }
        if !weight.is_finite() {
            return Err(format!("arc {i} has non-finite weight"));
        }
        edges[from].push(Edge { to, weight });
    }
    Ok(Graph {
        num_nodes,
        edges,
        coordinates: None,
        node_names: None,
    })
}

pub fn validate_graph(graph: &Graph) -> Result<(), String> {
    if graph.num_nodes == 0 {
        return Err("graph must have at least one node".to_string());
    }
    if graph.edges.len() != graph.num_nodes {
        return Err(format!(
            "graph has {} edge rows, expected {}",
            graph.edges.len(),
            graph.num_nodes
        ));
    }
    for (from, edges) in graph.edges.iter().enumerate() {
        for (i, edge) in edges.iter().enumerate() {
            if edge.to >= graph.num_nodes {
                return Err(format!(
                    "edge row {from} entry {i} has endpoint out of range"
                ));
            }
            if !edge.weight.is_finite() {
                return Err(format!("edge row {from} entry {i} has non-finite weight"));
            }
        }
    }
    if let Some(coords) = &graph.coordinates {
        if coords.len() != graph.num_nodes {
            return Err(format!(
                "coordinates has length {}, expected {}",
                coords.len(),
                graph.num_nodes
            ));
        }
    }
    if let Some(names) = &graph.node_names {
        if names.len() != graph.num_nodes {
            return Err(format!(
                "node_names has length {}, expected {}",
                names.len(),
                graph.num_nodes
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_constructor_and_shortest_path_work() {
        let graph =
            graph_from_arcs(4, &[(0, 1, 2.0), (1, 3, 2.0), (0, 2, 10.0), (2, 3, 1.0)]).unwrap();
        let result = shortest_path_dijkstra_des(&graph, 0, Default::default());
        assert!((result.distance[3] - 4.0).abs() < 1e-12);
        assert_eq!(reconstruct_path(&result, 0, 3), Some(vec![0, 1, 3]));
    }

    #[test]
    fn max_flow_and_min_cost_flow_are_reachable_from_facade() {
        let max_flow = solve_max_flow(MaxFlowProblem {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 0,
                    to: 2,
                    capacity: 1.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 3,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 2,
                    to: 3,
                    capacity: 1.0,
                    name: None,
                },
            ],
        });
        assert_eq!(max_flow.status, MaxFlowStatus::Optimal);
        assert!((max_flow.max_flow - 3.0).abs() < 1e-10);

        let min_cost = solve_min_cost_flow(MinCostFlowProblem {
            num_nodes: 3,
            supplies: vec![3.0, 0.0, -3.0],
            arcs: vec![
                MinCostFlowArc {
                    from: 0,
                    to: 1,
                    lower_bound: 0.0,
                    capacity: 3.0,
                    cost: 1.0,
                    name: None,
                },
                MinCostFlowArc {
                    from: 1,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 3.0,
                    cost: 1.0,
                    name: None,
                },
                MinCostFlowArc {
                    from: 0,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 3.0,
                    cost: 5.0,
                    name: None,
                },
            ],
        });
        assert_eq!(min_cost.status, MinCostFlowStatus::Optimal);
        assert!((min_cost.total_cost - 6.0).abs() < 1e-10);
    }

    #[test]
    fn checked_all_pairs_rejects_bad_graph_shape() {
        let err = try_all_pairs_shortest_path(&Graph {
            num_nodes: 2,
            edges: vec![vec![Edge { to: 2, weight: 1.0 }]],
            coordinates: None,
            node_names: None,
        })
        .unwrap_err();
        assert!(err.contains("edge rows"));
    }
}
