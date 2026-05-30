//! Port of `src/des/general/shortest-path-des.ts` — shortest path on a weighted
//! directed graph computed BY THE DES, where every graph node IS a stationary
//! entity and every distance update IS a movable "wave" message flowing along an
//! edge.
//!
//! THE PROBLEM
//!   Given `G = (V, E, w)` directed (or symmetric) with edge weights `w`
//!   (allowing negatives for Bellman-Ford, non-negatives for Dijkstra), find the
//!   shortest distance from a source `s ∈ V` to every other node.
//!
//! AS A DES
//!   Nodes are stations (each holds a distance estimate, a predecessor and a
//!   `dirty` flag). Movables are "Wave" messages carrying a `distanceProposal =
//!   sourceNode.distance + edge.weight`. Two tick models are provided:
//!     1. [`shortest_path_bellman_ford_des`] — every tick, dirty nodes emit waves
//!        along outgoing edges; receivers relax. Converges in `≤ |V|-1`
//!        iterations without negative cycles; iteration `|V|` detects them.
//!     2. [`shortest_path_dijkstra_des`] — a global min-priority queue settles the
//!        closest node each tick, then emits waves. Requires non-negative weights.
//!
//! MIGRATION NOTES
//!   * Node ids are `usize`; weights/distances are `f64`. Unreachable distances
//!     stay as the `f64::INFINITY` sentinel (NOT `Option`) to match the algorithm.
//!   * `predecessor` keeps the `-1` "none" sentinel as `isize` (1:1 with the TS
//!     `number[]`), so `reconstruct_path` walks it exactly like the source.
//!   * `IndexedMinHeap` is a hand-rolled binary min-heap, faithful to the TS
//!     `bubbleUp` / `bubbleDown`.
//!   * `reconstruct_path` returns `Option<Vec<usize>>` (TS `number[] | null`).
//!   * `build_random_graph` injects a [`RandomSource`] instead of inlining
//!     mulberry32 (per the migration header). `SeededRandom` IS mulberry32, so
//!     seeding it reproduces the TS stream.
//!   * Dijkstra's negative-weight guard is an invariant violation → `panic!`.

use crate::des::shared::capabilities::RandomSource;

// =============================================================================
// Declarations
// =============================================================================

/// A single outgoing edge `u → to` with `weight`. (TS `interface Edge`.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

/// A weighted directed graph. (TS `interface Graph`.)
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub num_nodes: usize,
    /// `edges[u]` = list of outgoing edges from `u`.
    pub edges: Vec<Vec<Edge>>,
    /// Optional 2-D coordinates for animation only.
    pub coordinates: Option<Vec<(f64, f64)>>,
    /// Optional names for animation captions.
    pub node_names: Option<Vec<String>>,
}

/// Which algorithm produced an [`SPResult`]. (TS string-union.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    BellmanFordDes,
    DijkstraDes,
}

/// One relaxation event fired during a tick (TS inline object literal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveEvent {
    pub from: usize,
    pub to: usize,
    pub new_distance: f64,
    pub improved: bool,
}

/// Shortest-path computation result. (TS `interface SPResult`.)
#[derive(Clone, Debug)]
pub struct SPResult {
    /// `distance[v]` = shortest distance from source to `v` (`INFINITY` if
    /// unreachable).
    pub distance: Vec<f64>,
    /// `predecessor[v]` = previous node on the shortest path (`-1` if source /
    /// unreachable).
    pub predecessor: Vec<isize>,
    /// Number of full tick rounds run.
    pub iterations: usize,
    /// Total number of waves emitted (edge relaxations performed).
    pub waves_emitted: usize,
    /// True iff a negative cycle reachable from the source was detected.
    pub has_negative_cycle_from_source: bool,
    /// Per-tick distance-vector snapshots (for animation). `trace[t][v]`.
    pub trace: Vec<Vec<f64>>,
    /// Per-tick wave events: which `(u → v)` relaxations fired this tick.
    pub wave_events: Vec<Vec<WaveEvent>>,
    /// Algorithm used for the result.
    pub algorithm: Algorithm,
}

/// Reconstruct the shortest path from `source` to `target` using the
/// predecessor array. Returns `None` if `target` is unreachable. (TS
/// `reconstructPath`.)
pub fn reconstruct_path(result: &SPResult, source: usize, target: usize) -> Option<Vec<usize>> {
    if !result.distance[target].is_finite() {
        return None;
    }
    let mut path: Vec<usize> = Vec::new();
    let mut cur: isize = target as isize;
    let mut seen: std::collections::HashSet<isize> = std::collections::HashSet::new();
    while cur != -1 {
        if seen.contains(&cur) {
            return None; // cycle in predecessors → broken
        }
        seen.insert(cur);
        path.push(cur as usize);
        if cur as usize == source {
            break;
        }
        cur = result.predecessor[cur as usize];
    }
    if *path.last().unwrap() != source {
        return None;
    }
    path.reverse();
    Some(path)
}

// =============================================================================
// Options
// =============================================================================

/// Options shared by both DES modes. (TS `interface BellmanFordOptions`.)
#[derive(Clone, Copy, Debug)]
pub struct BellmanFordOptions {
    /// Hard cap on the number of tick rounds. Default = `num_nodes` (enough for
    /// non-negative-cycle convergence; iteration `num_nodes` itself detects
    /// negative cycles reachable from source).
    pub max_iterations: Option<usize>,
    /// If `false`, suppresses the per-tick trace (saves memory). Default `true`.
    pub record_trace: bool,
}

impl Default for BellmanFordOptions {
    fn default() -> Self {
        BellmanFordOptions {
            max_iterations: None,
            record_trace: true,
        }
    }
}

// =============================================================================
// MODE 1 — BELLMAN-FORD as DES
// =============================================================================

pub fn shortest_path_bellman_ford_des(
    graph: &Graph,
    source: usize,
    opts: BellmanFordOptions,
) -> SPResult {
    let n = graph.num_nodes;
    let mut distance = vec![f64::INFINITY; n];
    let mut predecessor = vec![-1isize; n];
    let mut dirty = vec![false; n];
    distance[source] = 0.0;
    dirty[source] = true;
    let mut trace: Vec<Vec<f64>> = Vec::new();
    let mut wave_events: Vec<Vec<WaveEvent>> = Vec::new();
    if opts.record_trace {
        trace.push(distance.clone());
    }
    let mut waves_emitted = 0usize;
    let max_iter = opts.max_iterations.unwrap_or(n);
    let mut iter = 0usize;
    let mut has_negative_cycle = false;

    while iter < max_iter {
        iter += 1;
        let mut new_dirty = vec![false; n];
        let mut events_this_tick: Vec<WaveEvent> = Vec::new();
        let mut any_change = false;
        for u in 0..n {
            if !dirty[u] {
                continue;
            }
            let du = distance[u];
            for edge in &graph.edges[u] {
                waves_emitted += 1;
                let cand = du + edge.weight;
                let before = distance[edge.to];
                let improved = cand < before - 1e-12;
                events_this_tick.push(WaveEvent {
                    from: u,
                    to: edge.to,
                    new_distance: cand,
                    improved,
                });
                if improved {
                    distance[edge.to] = cand;
                    predecessor[edge.to] = u as isize;
                    new_dirty[edge.to] = true;
                    any_change = true;
                }
            }
        }
        wave_events.push(events_this_tick);
        if opts.record_trace {
            trace.push(distance.clone());
        }
        dirty = new_dirty;
        if !any_change {
            break;
        }
        // Iteration n+1 onwards detecting any change ⇒ negative cycle.
        if iter >= n {
            has_negative_cycle = any_change;
            break;
        }
    }

    SPResult {
        distance,
        predecessor,
        iterations: iter,
        waves_emitted,
        has_negative_cycle_from_source: has_negative_cycle,
        trace,
        wave_events,
        algorithm: Algorithm::BellmanFordDes,
    }
}

// =============================================================================
// MODE 2 — DIJKSTRA as DES
// =============================================================================

/// A `(distance, nodeId)` priority-queue entry. (TS private `interface PQEntry`.)
#[derive(Clone, Copy, Debug)]
struct PQEntry {
    distance: f64,
    node_id: usize,
}

/// Tiny indexed binary-heap min-priority-queue for Dijkstra. (TS private
/// `class IndexedMinHeap`.)
struct IndexedMinHeap {
    heap: Vec<PQEntry>,
}

impl IndexedMinHeap {
    fn new() -> Self {
        IndexedMinHeap { heap: Vec::new() }
    }

    fn push(&mut self, entry: PQEntry) {
        self.heap.push(entry);
        self.bubble_up(self.heap.len() - 1);
    }

    fn pop(&mut self) -> Option<PQEntry> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        let last = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.bubble_down(0);
        }
        Some(top)
    }

    fn size(&self) -> usize {
        self.heap.len()
    }

    fn bubble_up(&mut self, mut i: usize) {
        while i > 0 {
            let par = (i - 1) >> 1;
            if self.heap[i].distance < self.heap[par].distance {
                self.heap.swap(i, par);
                i = par;
            } else {
                break;
            }
        }
    }

    fn bubble_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut best = i;
            if l < n && self.heap[l].distance < self.heap[best].distance {
                best = l;
            }
            if r < n && self.heap[r].distance < self.heap[best].distance {
                best = r;
            }
            if best == i {
                break;
            }
            self.heap.swap(i, best);
            i = best;
        }
    }
}

pub fn shortest_path_dijkstra_des(
    graph: &Graph,
    source: usize,
    opts: BellmanFordOptions,
) -> SPResult {
    let n = graph.num_nodes;
    let mut distance = vec![f64::INFINITY; n];
    let mut predecessor = vec![-1isize; n];
    let mut settled = vec![false; n];
    distance[source] = 0.0;
    let mut pq = IndexedMinHeap::new();
    pq.push(PQEntry {
        distance: 0.0,
        node_id: source,
    });
    let mut trace: Vec<Vec<f64>> = Vec::new();
    let mut wave_events: Vec<Vec<WaveEvent>> = Vec::new();
    if opts.record_trace {
        trace.push(distance.clone());
    }
    let mut waves_emitted = 0usize;
    let mut iter = 0usize;

    // Sanity: weights must be non-negative for Dijkstra to be correct.
    for u in 0..n {
        for e in &graph.edges[u] {
            if e.weight < -1e-12 {
                panic!(
                    "Dijkstra requires non-negative weights, got {} on edge {}→{}",
                    e.weight, u, e.to
                );
            }
        }
    }

    while pq.size() > 0 {
        iter += 1;
        let top = pq.pop().unwrap();
        if settled[top.node_id] {
            iter -= 1;
            continue;
        }
        settled[top.node_id] = true;
        let mut events_this_tick: Vec<WaveEvent> = Vec::new();
        for edge in &graph.edges[top.node_id] {
            waves_emitted += 1;
            let cand = top.distance + edge.weight;
            let improved = cand < distance[edge.to] - 1e-12;
            events_this_tick.push(WaveEvent {
                from: top.node_id,
                to: edge.to,
                new_distance: cand,
                improved,
            });
            if improved {
                distance[edge.to] = cand;
                predecessor[edge.to] = top.node_id as isize;
                pq.push(PQEntry {
                    distance: cand,
                    node_id: edge.to,
                });
            }
        }
        wave_events.push(events_this_tick);
        if opts.record_trace {
            trace.push(distance.clone());
        }
    }

    SPResult {
        distance,
        predecessor,
        iterations: iter,
        waves_emitted,
        has_negative_cycle_from_source: false,
        trace,
        wave_events,
        algorithm: Algorithm::DijkstraDes,
    }
}

// =============================================================================
// GRAPH BUILDERS / HELPERS
// =============================================================================

/// Build a directed Erdős–Rényi-style random graph with edge probability
/// `edge_prob`, where each edge has uniform weight in `[w_min, w_max]`.
///
/// Per the migration header the inlined mulberry32 is replaced by an injected
/// [`RandomSource`]; pass a `SeededRandom` to reproduce the TS stream.
pub fn build_random_graph(
    num_nodes: usize,
    edge_prob: f64,
    w_min: f64,
    w_max: f64,
    rng: &mut impl RandomSource,
) -> Graph {
    let mut edges: Vec<Vec<Edge>> = (0..num_nodes).map(|_| Vec::new()).collect();
    for u in 0..num_nodes {
        for v in 0..num_nodes {
            if u == v {
                continue;
            }
            if rng.next_float() < edge_prob {
                edges[u].push(Edge {
                    to: v,
                    weight: w_min + (w_max - w_min) * rng.next_float(),
                });
            }
        }
    }
    // Ensure every node has at least one outgoing edge (avoid trivially
    // disconnected sinks for small p).
    for u in 0..num_nodes {
        if edges[u].is_empty() {
            let mut v = (rng.next_float() * num_nodes as f64).floor() as usize;
            if v == u {
                v = (v + 1) % num_nodes;
            }
            edges[u].push(Edge {
                to: v,
                weight: w_min + (w_max - w_min) * rng.next_float(),
            });
        }
    }
    // Random 2-D layout for animation.
    let mut coordinates: Vec<(f64, f64)> = Vec::new();
    for _ in 0..num_nodes {
        coordinates.push((rng.next_float() * 100.0, rng.next_float() * 100.0));
    }
    Graph {
        num_nodes,
        edges,
        coordinates: Some(coordinates),
        node_names: None,
    }
}

/// Build a small canonical graph used by examples and tests. The optimum path
/// from `0` to `4` has length `6`: `0→1(1)→2(2)→3(2)→4(1)`.
pub fn build_small_chain_graph() -> Graph {
    let edges: Vec<Vec<Edge>> = vec![
        vec![
            Edge { to: 1, weight: 1.0 },
            Edge { to: 2, weight: 4.0 },
            Edge { to: 4, weight: 10.0 },
        ], // 0
        vec![
            Edge { to: 2, weight: 2.0 },
            Edge { to: 3, weight: 5.0 },
        ], // 1
        vec![Edge { to: 3, weight: 2.0 }], // 2
        vec![Edge { to: 4, weight: 1.0 }], // 3
        vec![],                            // 4
    ];
    Graph {
        num_nodes: 5,
        edges,
        coordinates: None,
        node_names: Some(vec![
            "s".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "t".to_string(),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn bellman_ford_small_chain() {
        let g = build_small_chain_graph();
        let res = shortest_path_bellman_ford_des(&g, 0, BellmanFordOptions::default());
        // 0→1(1)→2(2)→3(2)→4(1) = 6
        assert_eq!(res.distance, vec![0.0, 1.0, 3.0, 5.0, 6.0]);
        assert!(!res.has_negative_cycle_from_source);
        assert_eq!(reconstruct_path(&res, 0, 4), Some(vec![0, 1, 2, 3, 4]));
        assert_eq!(res.algorithm, Algorithm::BellmanFordDes);
    }

    #[test]
    fn dijkstra_matches_bellman_ford() {
        let g = build_small_chain_graph();
        let res = shortest_path_dijkstra_des(&g, 0, BellmanFordOptions::default());
        assert_eq!(res.distance, vec![0.0, 1.0, 3.0, 5.0, 6.0]);
        assert_eq!(res.algorithm, Algorithm::DijkstraDes);
        assert_eq!(reconstruct_path(&res, 0, 4), Some(vec![0, 1, 2, 3, 4]));
    }

    #[test]
    fn negative_cycle_is_detected() {
        // 0→1(1), 1→2(-1), 2→1(-1): cycle 1↔2 has total weight -2.
        let g = Graph {
            num_nodes: 3,
            edges: vec![
                vec![Edge { to: 1, weight: 1.0 }],
                vec![Edge { to: 2, weight: -1.0 }],
                vec![Edge { to: 1, weight: -1.0 }],
            ],
            coordinates: None,
            node_names: None,
        };
        let res = shortest_path_bellman_ford_des(&g, 0, BellmanFordOptions::default());
        assert!(res.has_negative_cycle_from_source);
    }

    #[test]
    fn random_graph_is_reproducible() {
        let mut a = SeededRandom::new(123);
        let mut b = SeededRandom::new(123);
        let ga = build_random_graph(6, 0.5, 1.0, 5.0, &mut a);
        let gb = build_random_graph(6, 0.5, 1.0, 5.0, &mut b);
        assert_eq!(ga.num_nodes, gb.num_nodes);
        assert_eq!(ga.edges, gb.edges);
        for u in 0..ga.num_nodes {
            assert!(!ga.edges[u].is_empty());
        }
    }
}
