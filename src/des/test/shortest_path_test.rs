//! Port of src/des/test/shortest-path-test.ts
//!
//! Unit tests for the shortest-path-DES module (Bellman-Ford, Dijkstra, path
//! reconstruction). The TS file's expect()/close() tally becomes `#[test]`
//! functions using `assert!`/`assert_eq!`.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::prng::mulberry32;
    use crate::des::general::shortest_path_des::{
        build_random_graph, build_small_chain_graph, reconstruct_path,
        shortest_path_bellman_ford_des, shortest_path_dijkstra_des, BellmanFordOptions, Edge, Graph,
    };

    const TOL: f64 = 1e-12;

    fn opts() -> BellmanFordOptions {
        BellmanFordOptions::default()
    }

    // Group 1 — Small chain graph distances (Bellman-Ford)
    #[test]
    fn small_chain_bellman_ford() {
        let g = build_small_chain_graph();
        let r = shortest_path_bellman_ford_des(&g, 0, opts());
        assert!((r.distance[0] - 0.0).abs() <= TOL);
        assert!((r.distance[1] - 1.0).abs() <= TOL);
        assert!((r.distance[2] - 3.0).abs() <= TOL);
        assert!((r.distance[3] - 5.0).abs() <= TOL);
        assert!((r.distance[4] - 6.0).abs() <= TOL);
    }

    // Group 2 — Small chain graph distances (Dijkstra)
    #[test]
    fn small_chain_dijkstra() {
        let g = build_small_chain_graph();
        let r = shortest_path_dijkstra_des(&g, 0, opts());
        assert!((r.distance[0] - 0.0).abs() <= TOL);
        assert!((r.distance[1] - 1.0).abs() <= TOL);
        assert!((r.distance[2] - 3.0).abs() <= TOL);
        assert!((r.distance[3] - 5.0).abs() <= TOL);
        assert!((r.distance[4] - 6.0).abs() <= TOL);
    }

    // Group 3 — Path reconstruction
    #[test]
    fn path_reconstruction() {
        let g = build_small_chain_graph();
        let r = shortest_path_dijkstra_des(&g, 0, opts());
        let path = reconstruct_path(&r, 0, 4);
        assert_eq!(path, Some(vec![0, 1, 2, 3, 4]));
        let p_self = reconstruct_path(&r, 0, 0);
        assert_eq!(p_self, Some(vec![0]));
    }

    // Group 4 — Unreachable nodes have infinite distance
    #[test]
    fn unreachable_nodes_infinite() {
        let g = Graph {
            num_nodes: 4,
            edges: vec![
                vec![Edge { to: 1, weight: 5.0 }],
                vec![],
                vec![Edge { to: 3, weight: 1.0 }],
                vec![],
            ],
            ..Default::default()
        };
        let r = shortest_path_bellman_ford_des(&g, 0, opts());
        assert!(!r.distance[2].is_finite());
        assert!(!r.distance[3].is_finite());
        assert_eq!(r.distance[1], 5.0);
        assert_eq!(reconstruct_path(&r, 0, 3), None);
    }

    // Group 5 — BF and Dijkstra agree on random non-negative graphs
    #[test]
    fn bf_and_dijkstra_agree() {
        for seed in [1u32, 7, 13, 42, 99] {
            let n = 12;
            let mut rng = mulberry32(seed);
            let g = build_random_graph(n, 0.4, 1.0, 9.0, &mut rng);
            let bf = shortest_path_bellman_ford_des(&g, 0, opts());
            let dj = shortest_path_dijkstra_des(&g, 0, opts());
            for v in 0..n {
                let a = bf.distance[v];
                let b = dj.distance[v];
                if !a.is_finite() && !b.is_finite() {
                    continue;
                }
                assert!(a.is_finite() && b.is_finite(), "seed={seed} node={v}");
                assert!((a - b).abs() <= TOL, "seed={seed} node={v}: {a} vs {b}");
            }
        }
    }

    // Group 6 — Negative weights handled correctly
    #[test]
    fn negative_weights() {
        let g = Graph {
            num_nodes: 3,
            edges: vec![
                vec![Edge { to: 1, weight: 5.0 }, Edge { to: 2, weight: -2.0 }],
                vec![Edge { to: 2, weight: 1.0 }],
                vec![],
            ],
            ..Default::default()
        };
        let bf = shortest_path_bellman_ford_des(&g, 0, opts());
        assert!((bf.distance[2] - (-2.0)).abs() <= TOL);
        assert!(!bf.has_negative_cycle_from_source);

        // Dijkstra panics on negative weights.
        let threw = std::panic::catch_unwind(|| shortest_path_dijkstra_des(&g, 0, opts())).is_err();
        assert!(threw);
    }

    // Group 7 — Negative cycle detection
    #[test]
    fn negative_cycle_detection() {
        let g = Graph {
            num_nodes: 3,
            edges: vec![
                vec![Edge { to: 1, weight: 1.0 }],
                vec![Edge { to: 2, weight: -3.0 }],
                vec![Edge { to: 1, weight: 1.0 }],
            ],
            ..Default::default()
        };
        let bf = shortest_path_bellman_ford_des(&g, 0, opts());
        assert!(bf.has_negative_cycle_from_source);
    }

    // Group 8 — Trace and wave_events have consistent shape
    #[test]
    fn trace_and_wave_events_shape() {
        let g = build_small_chain_graph();
        let bf = shortest_path_bellman_ford_des(&g, 0, opts());
        assert_eq!(bf.trace.len(), bf.iterations + 1);
        assert_eq!(bf.wave_events.len(), bf.iterations);
        for snap in &bf.trace {
            assert_eq!(snap.len(), g.num_nodes);
        }
    }

    // Group 9 — Reproducibility
    #[test]
    fn reproducibility() {
        let mut rng = mulberry32(99);
        let g = build_random_graph(15, 0.4, 1.0, 5.0, &mut rng);
        let r1 = shortest_path_bellman_ford_des(&g, 0, opts());
        let r2 = shortest_path_bellman_ford_des(&g, 0, opts());
        assert_eq!(r1.distance, r2.distance);
        assert_eq!(r1.predecessor, r2.predecessor);
        assert_eq!(r1.iterations, r2.iterations);
    }

    // Group 10 — Bellman-Ford respects |V|-1 worst-case bound
    #[test]
    fn bellman_ford_worst_case_bound() {
        let n = 6;
        let mut edges: Vec<Vec<Edge>> = Vec::new();
        for i in 0..n - 1 {
            edges.push(vec![Edge { to: i + 1, weight: 1.0 }]);
        }
        edges.push(vec![]);
        let g = Graph { num_nodes: n, edges, ..Default::default() };
        let bf = shortest_path_bellman_ford_des(&g, 0, opts());
        for v in 0..n {
            assert!((bf.distance[v] - v as f64).abs() <= TOL);
        }
        assert!(bf.iterations <= n);
    }
}
