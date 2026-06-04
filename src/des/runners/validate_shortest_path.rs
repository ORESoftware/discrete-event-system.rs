//! Port of `src/des/runners/validate-shortest-path.ts`.
//!
//! Verifies the DES-driven Bellman-Ford ≡ Dijkstra on non-negative graphs,
//! Dijkstra's refusal of negative weights, Bellman-Ford negative-cycle
//! detection, iteration bounds, and wave-count ordering. Driver → [`run`].
//!
//! The runner now delegates to `crate::des::general::shortest_path_des`.

#![allow(dead_code)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::des::general::shortest_path_des::{
    build_random_graph as build_random_graph_model,
    build_small_chain_graph as build_small_chain_graph_model,
    shortest_path_bellman_ford_des as shortest_path_bellman_ford_des_model,
    shortest_path_dijkstra_des as shortest_path_dijkstra_des_model, BellmanFordOptions, Edge,
    Graph, SPResult,
};
use crate::des::shared::capabilities::SeededRandom;

type BfResult = SPResult;
type DijkstraResult = SPResult;

fn build_small_chain_graph() -> Graph {
    build_small_chain_graph_model()
}

fn build_random_graph(n: usize, density: f64, wmin: f64, wmax: f64, seed: u32) -> Graph {
    let mut rng = SeededRandom::new(seed);
    build_random_graph_model(n, density, wmin, wmax, &mut rng)
}

fn shortest_path_bellman_ford_des(g: &Graph, source: usize) -> BfResult {
    shortest_path_bellman_ford_des_model(g, source, BellmanFordOptions::default())
}

fn shortest_path_dijkstra_des(g: &Graph, source: usize) -> Result<DijkstraResult, String> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| {
        shortest_path_dijkstra_des_model(g, source, BellmanFordOptions::default())
    }));
    std::panic::set_hook(previous_hook);
    result.map_err(|panic| {
        panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "Dijkstra rejected graph".to_string())
    })
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!(
            "{}  {}{}",
            if ok { "  PASS" } else { "  FAIL" },
            label,
            tail
        );
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
    fn close(&mut self, label: &str, a: f64, b: f64) {
        self.check(
            label,
            (a - b).abs() <= 1e-12,
            &format!("|{} − {}| = {:.2e}", a, b, (a - b).abs()),
        );
    }
}

/// `validate-shortest-path.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\nStudy 1 — Small chain graph: textbook distances");
    {
        let g = build_small_chain_graph();
        let r = shortest_path_bellman_ford_des(&g, 0);
        c.close("d(s) = 0", r.distance[0], 0.0);
        c.close("d(a) = 1", r.distance[1], 1.0);
        c.close("d(b) = 3", r.distance[2], 3.0);
        c.close("d(c) = 5", r.distance[3], 5.0);
        c.close("d(t) = 6", r.distance[4], 6.0);
        c.check(
            "Bellman-Ford terminates in ≤ 4 iterations on 5-node graph",
            r.iterations <= 4,
            &format!("iterations = {}", r.iterations),
        );
        c.check(
            "no negative cycle on positive-weight graph",
            !r.has_negative_cycle_from_source,
            "",
        );
    }

    println!("\nStudy 2 — Bellman-Ford-DES ≡ Dijkstra-DES on non-negative graphs");
    {
        for seed in [1u32, 7, 13, 42, 99] {
            let n = 12usize;
            let g = build_random_graph(n, 0.4, 1.0, 9.0, seed);
            let bf = shortest_path_bellman_ford_des(&g, 0);
            let dj = shortest_path_dijkstra_des(&g, 0).expect("non-negative graph");
            let mut max_diff = 0.0_f64;
            for v in 0..n {
                let a = bf.distance[v];
                let b = dj.distance[v];
                if !a.is_finite() && !b.is_finite() {
                    continue;
                }
                if !a.is_finite() || !b.is_finite() {
                    max_diff = f64::INFINITY;
                } else {
                    max_diff = f64::max(max_diff, (a - b).abs());
                }
            }
            c.check(
                &format!(
                    "seed={}: Bellman-Ford and Dijkstra agree on every distance",
                    seed
                ),
                max_diff < 1e-12,
                &format!("max |Δ| = {:.2e}", max_diff),
            );
        }
    }

    println!("\nStudy 3 — Dijkstra refuses negative weights");
    {
        let g = Graph {
            num_nodes: 3,
            edges: vec![
                vec![
                    Edge { to: 1, weight: 5.0 },
                    Edge {
                        to: 2,
                        weight: -2.0,
                    },
                ],
                vec![Edge { to: 2, weight: 1.0 }],
                vec![],
            ],
            coordinates: None,
            node_names: None,
        };
        let threw = shortest_path_dijkstra_des(&g, 0).is_err();
        c.check("Dijkstra throws on negative-weight edge", threw, "");
        let bf = shortest_path_bellman_ford_des(&g, 0);
        c.close(
            "Bellman-Ford handles negative edge: d(2) = -2",
            bf.distance[2],
            -2.0,
        );
        c.check(
            "Bellman-Ford does not flag negative cycle",
            !bf.has_negative_cycle_from_source,
            "",
        );
    }

    println!("\nStudy 4 — Bellman-Ford detects negative cycles reachable from source");
    {
        let g = Graph {
            num_nodes: 3,
            edges: vec![
                vec![Edge { to: 1, weight: 1.0 }],
                vec![Edge {
                    to: 2,
                    weight: -3.0,
                }],
                vec![Edge { to: 1, weight: 1.0 }],
            ],
            coordinates: None,
            node_names: None,
        };
        let bf = shortest_path_bellman_ford_des(&g, 0);
        c.check(
            "negative cycle reachable from source flagged",
            bf.has_negative_cycle_from_source,
            "",
        );
    }

    println!("\nStudy 5 — Bellman-Ford terminates within |V| DES ticks on positive-weight graphs");
    {
        for n in [5usize, 10, 20, 30] {
            let g = build_random_graph(n, 0.3, 1.0, 5.0, 42 + n as u32);
            let bf = shortest_path_bellman_ford_des(&g, 0);
            c.check(
                &format!(
                    "n={}: Bellman-Ford ran in {} DES ticks (≤ {})",
                    n, bf.iterations, n
                ),
                bf.iterations <= n,
                &format!("iterations={}, |V|={}", bf.iterations, n),
            );
        }
    }

    println!("\nStudy 6 — Wave count: Dijkstra waves ≤ Bellman-Ford waves on dense graphs");
    {
        for seed in [1u32, 7, 42] {
            let g = build_random_graph(15, 0.5, 1.0, 10.0, seed);
            let bf = shortest_path_bellman_ford_des(&g, 0);
            let dj = shortest_path_dijkstra_des(&g, 0).expect("non-negative graph");
            println!(
                "    seed={}: BF waves = {}, Dij waves = {}",
                seed, bf.waves_emitted, dj.waves_emitted
            );
            c.check(
                &format!("seed={}: Dijkstra waves ≤ Bellman-Ford waves", seed),
                dj.waves_emitted <= bf.waves_emitted,
                &format!("dij={}, bf={}", dj.waves_emitted, bf.waves_emitted),
            );
        }
    }

    println!(
        "\n{} checks: {} passed, {} failed",
        c.pass + c.fail,
        c.pass,
        c.fail
    );
    if c.fail > 0 {
        std::process::exit(1);
    }
}
