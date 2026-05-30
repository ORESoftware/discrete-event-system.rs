//! TypeScript source: `src/des/general/shortest-path-des.ts`
//! Rust target: `src/des/general/shortest_path_des.rs`
//!
//! Porting note: this file intentionally keeps the TypeScript module surface
//! recognizable while making failure modes explicit with `Result`.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::DesDecimal;
use crate::des::general::prng::Mulberry32;
use crate::migration::MigrationFile;
use crate::numeric::decimal_from_f64;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/shortest-path-des.ts",
    "src/des/general/shortest_path_des.rs",
    &[
        "RUST MIGRATION: `Graph`, `Edge`, trace rows, and wave events are nominal serde structs.",
        "RUST MIGRATION: Algorithm string unions are represented by `ShortestPathAlgorithm`.",
        "RUST MIGRATION: Edge weights and accumulated distances use DesDecimal; unreachable nodes are represented as None.",
        "RUST MIGRATION: Dijkstra returns `Result` for negative weights instead of throwing.",
        "RUST MIGRATION: The DES station/message framing is preserved in trace/wave data; a later deeper port can wrap node state in concrete station structs.",
    ],
    &[
        "BellmanFordOptions",
        "Edge",
        "Graph",
        "ShortestPathResult",
        "build_random_graph",
        "build_small_chain_graph",
        "reconstruct_path",
        "shortest_path_bellman_ford_des",
        "shortest_path_dijkstra_des",
    ],
);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub to: usize,
    pub weight: DesDecimal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub num_nodes: usize,
    pub edges: Vec<Vec<Edge>>,
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub node_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShortestPathAlgorithm {
    BellmanFordDes,
    DijkstraDes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveEvent {
    pub from: usize,
    pub to: usize,
    pub new_distance: DesDecimal,
    pub improved: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortestPathResult {
    pub distance: Vec<Option<DesDecimal>>,
    pub predecessor: Vec<Option<usize>>,
    pub iterations: usize,
    pub waves_emitted: usize,
    pub has_negative_cycle_from_source: bool,
    pub trace: Vec<Vec<Option<DesDecimal>>>,
    pub wave_events: Vec<Vec<WaveEvent>>,
    pub algorithm: ShortestPathAlgorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BellmanFordOptions {
    pub max_iterations: Option<usize>,
    pub record_trace: bool,
}

impl Default for BellmanFordOptions {
    fn default() -> Self {
        Self {
            max_iterations: None,
            record_trace: true,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum ShortestPathError {
    #[error("source node {source_node} is outside graph with {num_nodes} nodes")]
    SourceOutOfBounds {
        source_node: usize,
        num_nodes: usize,
    },
    #[error("graph.edges length {edge_rows} does not match graph.num_nodes {num_nodes}")]
    EdgeRowCountMismatch { edge_rows: usize, num_nodes: usize },
    #[error("edge {from}->{to} points outside graph with {num_nodes} nodes")]
    EdgeTargetOutOfBounds {
        from: usize,
        to: usize,
        num_nodes: usize,
    },
    #[error("Dijkstra requires non-negative weights, got {weight} on edge {from}->{to}")]
    NegativeWeightForDijkstra {
        from: usize,
        to: usize,
        weight: DesDecimal,
    },
}

pub fn reconstruct_path(
    result: &ShortestPathResult,
    source: usize,
    target: usize,
) -> Option<Vec<usize>> {
    if target >= result.distance.len() || result.distance[target].is_none() {
        return None;
    }
    let mut path = Vec::new();
    let mut current = Some(target);
    let mut seen = HashSet::new();
    while let Some(node) = current {
        if !seen.insert(node) {
            return None;
        }
        path.push(node);
        if node == source {
            break;
        }
        current = result.predecessor.get(node).copied().flatten();
    }
    if path.last().copied() != Some(source) {
        return None;
    }
    path.reverse();
    Some(path)
}

pub fn shortest_path_bellman_ford_des(
    graph: &Graph,
    source: usize,
    options: BellmanFordOptions,
) -> Result<ShortestPathResult, ShortestPathError> {
    validate_graph(graph, source)?;
    let n = graph.num_nodes;
    let mut distance = vec![None; n];
    let mut predecessor = vec![None; n];
    let mut dirty = vec![false; n];
    distance[source] = Some(DesDecimal::ZERO);
    dirty[source] = true;

    let mut trace = Vec::new();
    let mut wave_events = Vec::new();
    if options.record_trace {
        trace.push(distance.clone());
    }

    let max_iterations = options.max_iterations.unwrap_or(n);
    let mut waves_emitted = 0;
    let mut iterations = 0;
    let mut has_negative_cycle = false;

    while iterations < max_iterations {
        iterations += 1;
        let mut new_dirty = vec![false; n];
        let mut events_this_tick = Vec::new();
        let mut any_change = false;

        for from in 0..n {
            if !dirty[from] {
                continue;
            }
            let Some(source_distance) = distance[from] else {
                continue;
            };
            for edge in &graph.edges[from] {
                waves_emitted += 1;
                let candidate = source_distance + edge.weight;
                let improved = distance[edge.to]
                    .map(|current| candidate < current)
                    .unwrap_or(true);
                events_this_tick.push(WaveEvent {
                    from,
                    to: edge.to,
                    new_distance: candidate,
                    improved,
                });
                if improved {
                    distance[edge.to] = Some(candidate);
                    predecessor[edge.to] = Some(from);
                    new_dirty[edge.to] = true;
                    any_change = true;
                }
            }
        }

        wave_events.push(events_this_tick);
        if options.record_trace {
            trace.push(distance.clone());
        }
        dirty = new_dirty;
        if !any_change {
            break;
        }
        if iterations >= n {
            has_negative_cycle = true;
            break;
        }
    }

    Ok(ShortestPathResult {
        distance,
        predecessor,
        iterations,
        waves_emitted,
        has_negative_cycle_from_source: has_negative_cycle,
        trace,
        wave_events,
        algorithm: ShortestPathAlgorithm::BellmanFordDes,
    })
}

pub fn shortest_path_dijkstra_des(
    graph: &Graph,
    source: usize,
    options: BellmanFordOptions,
) -> Result<ShortestPathResult, ShortestPathError> {
    validate_graph(graph, source)?;
    for (from, edges) in graph.edges.iter().enumerate() {
        for edge in edges {
            if edge.weight < DesDecimal::ZERO {
                return Err(ShortestPathError::NegativeWeightForDijkstra {
                    from,
                    to: edge.to,
                    weight: edge.weight,
                });
            }
        }
    }

    let n = graph.num_nodes;
    let mut distance = vec![None; n];
    let mut predecessor = vec![None; n];
    let mut settled = vec![false; n];
    let mut queue = IndexedMinHeap::new();
    distance[source] = Some(DesDecimal::ZERO);
    queue.push(PqEntry {
        distance: DesDecimal::ZERO,
        node_id: source,
    });

    let mut trace = Vec::new();
    let mut wave_events = Vec::new();
    if options.record_trace {
        trace.push(distance.clone());
    }

    let mut waves_emitted = 0;
    let mut iterations = 0;

    while let Some(top) = queue.pop() {
        iterations += 1;
        if settled[top.node_id] {
            iterations -= 1;
            continue;
        }
        settled[top.node_id] = true;
        let mut events_this_tick = Vec::new();

        for edge in &graph.edges[top.node_id] {
            waves_emitted += 1;
            let candidate = top.distance + edge.weight;
            let improved = distance[edge.to]
                .map(|current| candidate < current)
                .unwrap_or(true);
            events_this_tick.push(WaveEvent {
                from: top.node_id,
                to: edge.to,
                new_distance: candidate,
                improved,
            });
            if improved {
                distance[edge.to] = Some(candidate);
                predecessor[edge.to] = Some(top.node_id);
                queue.push(PqEntry {
                    distance: candidate,
                    node_id: edge.to,
                });
            }
        }

        wave_events.push(events_this_tick);
        if options.record_trace {
            trace.push(distance.clone());
        }
    }

    Ok(ShortestPathResult {
        distance,
        predecessor,
        iterations,
        waves_emitted,
        has_negative_cycle_from_source: false,
        trace,
        wave_events,
        algorithm: ShortestPathAlgorithm::DijkstraDes,
    })
}

pub fn build_random_graph(
    num_nodes: usize,
    edge_probability: f64,
    weight_min: f64,
    weight_max: f64,
    seed: u32,
) -> Graph {
    let mut rng = Mulberry32::new(seed);
    let weight_min = decimal_from_f64(weight_min, "build_random_graph")
        .expect("random graph minimum weight must be finite");
    let weight_max = decimal_from_f64(weight_max, "build_random_graph")
        .expect("random graph maximum weight must be finite");
    let weight_span = weight_max - weight_min;
    let mut edges = vec![Vec::new(); num_nodes];
    for from in 0..num_nodes {
        for to in 0..num_nodes {
            if from == to {
                continue;
            }
            if rng.next_f64() < edge_probability {
                let sample = decimal_from_f64(rng.next_f64(), "build_random_graph")
                    .expect("Mulberry32 sample should be finite");
                edges[from].push(Edge {
                    to,
                    weight: weight_min + weight_span * sample,
                });
            }
        }
    }

    for (from, outgoing) in edges.iter_mut().enumerate() {
        if outgoing.is_empty() && num_nodes > 1 {
            let mut to = (rng.next_f64() * num_nodes as f64).floor() as usize;
            if to == from {
                to = (to + 1) % num_nodes;
            }
            let sample = decimal_from_f64(rng.next_f64(), "build_random_graph")
                .expect("Mulberry32 sample should be finite");
            outgoing.push(Edge {
                to,
                weight: weight_min + weight_span * sample,
            });
        }
    }

    let coordinates = (0..num_nodes)
        .map(|_| (rng.next_f64() * 100.0, rng.next_f64() * 100.0))
        .collect();

    Graph {
        num_nodes,
        edges,
        coordinates: Some(coordinates),
        node_names: None,
    }
}

pub fn build_small_chain_graph() -> Graph {
    Graph {
        num_nodes: 5,
        edges: vec![
            vec![
                Edge {
                    to: 1,
                    weight: DesDecimal::from(1),
                },
                Edge {
                    to: 2,
                    weight: DesDecimal::from(4),
                },
                Edge {
                    to: 4,
                    weight: DesDecimal::from(10),
                },
            ],
            vec![
                Edge {
                    to: 2,
                    weight: DesDecimal::from(2),
                },
                Edge {
                    to: 3,
                    weight: DesDecimal::from(5),
                },
            ],
            vec![Edge {
                to: 3,
                weight: DesDecimal::from(2),
            }],
            vec![Edge {
                to: 4,
                weight: DesDecimal::from(1),
            }],
            vec![],
        ],
        coordinates: None,
        node_names: Some(["s", "a", "b", "c", "t"].map(String::from).to_vec()),
    }
}

fn validate_graph(graph: &Graph, source: usize) -> Result<(), ShortestPathError> {
    if source >= graph.num_nodes {
        return Err(ShortestPathError::SourceOutOfBounds {
            source_node: source,
            num_nodes: graph.num_nodes,
        });
    }
    if graph.edges.len() != graph.num_nodes {
        return Err(ShortestPathError::EdgeRowCountMismatch {
            edge_rows: graph.edges.len(),
            num_nodes: graph.num_nodes,
        });
    }
    for (from, edges) in graph.edges.iter().enumerate() {
        for edge in edges {
            if edge.to >= graph.num_nodes {
                return Err(ShortestPathError::EdgeTargetOutOfBounds {
                    from,
                    to: edge.to,
                    num_nodes: graph.num_nodes,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PqEntry {
    distance: DesDecimal,
    node_id: usize,
}

#[derive(Debug, Default)]
struct IndexedMinHeap {
    heap: Vec<PqEntry>,
}

impl IndexedMinHeap {
    fn new() -> Self {
        Self { heap: Vec::new() }
    }

    fn push(&mut self, entry: PqEntry) {
        self.heap.push(entry);
        self.bubble_up(self.heap.len() - 1);
    }

    fn pop(&mut self) -> Option<PqEntry> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        let last = self.heap.pop().expect("heap is non-empty");
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.bubble_down(0);
        }
        Some(top)
    }

    fn bubble_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) >> 1;
            if self.heap[index].distance < self.heap[parent].distance {
                self.heap.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn bubble_down(&mut self, mut index: usize) {
        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut best = index;
            if left < self.heap.len() && self.heap[left].distance < self.heap[best].distance {
                best = left;
            }
            if right < self.heap.len() && self.heap[right].distance < self.heap[best].distance {
                best = right;
            }
            if best == index {
                break;
            }
            self.heap.swap(index, best);
            index = best;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_distances(values: &[i64]) -> Vec<Option<DesDecimal>> {
        values
            .iter()
            .map(|value| Some(DesDecimal::from(*value)))
            .collect()
    }

    #[test]
    fn small_chain_bellman_ford_distances_match_typescript() {
        let graph = build_small_chain_graph();
        let result =
            shortest_path_bellman_ford_des(&graph, 0, BellmanFordOptions::default()).unwrap();
        assert_eq!(result.distance, path_distances(&[0, 1, 3, 5, 6]));
    }

    #[test]
    fn small_chain_dijkstra_distances_match_typescript() {
        let graph = build_small_chain_graph();
        let result = shortest_path_dijkstra_des(&graph, 0, BellmanFordOptions::default()).unwrap();
        assert_eq!(result.distance, path_distances(&[0, 1, 3, 5, 6]));
    }
}
