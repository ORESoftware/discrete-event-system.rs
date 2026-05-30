//! Port of `src/des/main-shortest-path.ts`.
//!
//! Shortest-path-as-DES on a directed weighted graph. Each node IS a stationary
//! entity holding its best distance estimate; each "wave" message IS a movable
//! carrying a distance update along an edge. Two algorithms (Bellman-Ford-DES,
//! Dijkstra-DES) and an optional animation.
//!
//! Conversion notes:
//!   - `process.env` (`N_NODES`, `EDGE_PROB`, `ALGO`, `SOURCE`, `ANIMATE`) →
//!     `std::env::var`.
//!   - random graph generation routes through the seeded `SeededRandom`.
//!   - `async main` → [`run`].
//!   - delegates to `general::shortest_path_des` and `animation`.

use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::shortest_path_scene as scene;
use crate::des::general::shortest_path_des::{
    build_random_graph, build_small_chain_graph, reconstruct_path, shortest_path_bellman_ford_des,
    shortest_path_dijkstra_des, Algorithm, BellmanFordOptions, Graph, SPResult, WaveEvent,
};
use crate::des::shared::capabilities::SeededRandom;

/// `Number.prototype.toExponential(digits)` (signed exponent, JS `Infinity`).
fn to_exponential(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    let s = format!("{:.*e}", digits, x);
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{}e{}{}", mant, sign, exp_num.abs())
}

/// `graph.nodeNames?.[v] ?? String(v)`.
fn node_name(graph: &Graph, v: usize) -> String {
    graph
        .node_names
        .as_ref()
        .and_then(|n| n.get(v))
        .cloned()
        .unwrap_or_else(|| v.to_string())
}

fn to_scene_graph(g: &Graph) -> scene::Graph {
    scene::Graph {
        num_nodes: g.num_nodes,
        edges: g
            .edges
            .iter()
            .map(|es| {
                es.iter()
                    .map(|e| scene::Edge {
                        to: e.to,
                        weight: e.weight,
                    })
                    .collect()
            })
            .collect(),
        coordinates: g
            .coordinates
            .as_ref()
            .map(|cs| cs.iter().map(|c| [c.0, c.1]).collect()),
        node_names: g.node_names.clone(),
    }
}

fn to_scene_wave(w: &WaveEvent) -> scene::WaveEvent {
    scene::WaveEvent {
        from: w.from,
        to: w.to,
        new_distance: w.new_distance,
        improved: w.improved,
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let algo = std::env::var("ALGO").unwrap_or_else(|_| "bellman-ford".to_string());
    let n_nodes: usize = std::env::var("N_NODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let edge_prob: f64 = std::env::var("EDGE_PROB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35);
    let seed: u32 = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(13);
    let source: usize = std::env::var("SOURCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let animate = std::env::var("ANIMATE").map(|v| v == "1").unwrap_or(false);

    let graph: Graph = if n_nodes > 0 {
        let mut rng = SeededRandom::new(seed);
        build_random_graph(n_nodes, edge_prob, 1.0, 10.0, &mut rng)
    } else {
        build_small_chain_graph()
    };

    // ── Banner ──
    println!("# Shortest-path solver as DES (each node is a station, waves are movables)");
    println!(
        "# graph: {} nodes, source = {}",
        graph.num_nodes,
        node_name(&graph, source)
    );
    let mut edge_count = 0usize;
    for e in &graph.edges {
        edge_count += e.len();
    }
    println!("# edges: {}", edge_count);
    println!();

    // ── Run requested algorithm(s) ──
    let mut runs: Vec<(String, SPResult)> = Vec::new();
    if algo == "bellman-ford" || algo == "both" {
        let t0 = Instant::now();
        let r = shortest_path_bellman_ford_des(&graph, source, BellmanFordOptions::default());
        println!(
            "# Bellman-Ford-DES finished in {}ms",
            t0.elapsed().as_millis()
        );
        println!("#   iterations  = {}", r.iterations);
        println!("#   waves emitted = {}", r.waves_emitted);
        println!("#   negative cycle = {}", r.has_negative_cycle_from_source);
        runs.push(("bellman-ford".to_string(), r));
    }
    if algo == "dijkstra" || algo == "both" {
        let t0 = Instant::now();
        let r = shortest_path_dijkstra_des(&graph, source, BellmanFordOptions::default());
        println!(
            "# Dijkstra-DES       finished in {}ms",
            t0.elapsed().as_millis()
        );
        println!("#   priority-queue pops = {}", r.iterations);
        println!("#   waves emitted       = {}", r.waves_emitted);
        runs.push(("dijkstra".to_string(), r));
    }
    println!();

    // ── Cross-validate Bellman-Ford and Dijkstra distances if both ran ──
    if runs.len() == 2 {
        let mut max_diff = 0.0_f64;
        for v in 0..graph.num_nodes {
            let a = runs[0].1.distance[v];
            let b = runs[1].1.distance[v];
            if a.is_finite() && b.is_finite() {
                max_diff = max_diff.max((a - b).abs());
            } else if a != b {
                max_diff = f64::INFINITY;
            }
        }
        println!(
            "# Bellman-Ford vs Dijkstra:  max |Δ distance| = {}",
            to_exponential(max_diff, 2)
        );
        println!();
    }

    // ── Per-node distance + path report from the first run ──
    let r = &runs[0].1;
    println!("# Distances from source {}:", node_name(&graph, source));
    for v in 0..graph.num_nodes {
        let name = node_name(&graph, v);
        if !r.distance[v].is_finite() {
            println!("#   {} d = ∞       (unreachable)", format!("{:<6}", name));
        } else {
            let path = reconstruct_path(r, source, v);
            let path_str = match path {
                Some(p) => p
                    .iter()
                    .map(|&node| node_name(&graph, node))
                    .collect::<Vec<_>>()
                    .join(" → "),
                None => "-".to_string(),
            };
            println!(
                "#   {} d = {}   path: {}",
                format!("{:<6}", name),
                format!("{:>6}", format!("{:.2}", r.distance[v])),
                path_str
            );
        }
    }
    println!();

    // ── Animation ──
    if animate {
        let target = runs.last().unwrap(); // animate the last (so 'both' shows Dijkstra)
        let target_name = target.0.clone();
        let target_result = &target.1;
        let out_dir = std::path::Path::new("out");
        let frames_path = out_dir.join(format!("shortest-path-{}.frames.jsonl", target_name));
        let html_path = out_dir.join(format!("shortest-path-{}.html", target_name));
        let mut rec = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.to_string_lossy().into_owned(),
            html_path: Some(html_path.to_string_lossy().into_owned()),
            width: scene::STAGE_W,
            height: scene::STAGE_H,
            fps: Some(2.0),
            title: Some(format!("Shortest-path-DES ({})", target_name)),
            subtitle: Some(format!(
                "{} nodes, {} edges, {} iterations",
                graph.num_nodes, edge_count, target_result.iterations
            )),
            background: Some("#020617".to_string()),
            ..Default::default()
        })
        .expect("create frame recorder");

        let scene_graph = to_scene_graph(&graph);
        let sp_algo = match target_result.algorithm {
            Algorithm::BellmanFordDes => scene::SpAlgorithm::BellmanFordDes,
            Algorithm::DijkstraDes => scene::SpAlgorithm::DijkstraDes,
        };
        let mut ticks: Vec<f64> = Vec::new();
        let mut min_d: Vec<f64> = Vec::new();
        let mut max_d: Vec<f64> = Vec::new();
        for i in 0..target_result.trace.len() {
            let dist_now = target_result.trace[i].clone();
            let events: Vec<scene::WaveEvent> = if i > 0 {
                target_result
                    .wave_events
                    .get(i - 1)
                    .map(|evs| evs.iter().map(to_scene_wave).collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            ticks.push(i as f64);
            let finite: Vec<f64> = dist_now.iter().copied().filter(|d| d.is_finite()).collect();
            min_d.push(if finite.is_empty() {
                0.0
            } else {
                finite.iter().copied().fold(f64::INFINITY, f64::min)
            });
            max_d.push(if finite.is_empty() {
                0.0
            } else {
                finite.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            });
            let i_f = i as f64;
            let sg = &scene_graph;
            rec.frame(i_f, i_f, || {
                scene::build_shortest_path_frame(
                    i_f, i_f, sg, &dist_now, &events, source, i_f, sp_algo,
                )
            });
        }
        rec.set_charts(scene::build_shortest_path_charts(&ticks, &min_d, &max_d));
        rec.finish().expect("finish recorder");
        println!("# Animation written to {}", html_path.display());
    }
}
