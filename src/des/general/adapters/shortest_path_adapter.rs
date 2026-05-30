//! Port of `src/des/general/adapters/shortest-path-adapter.ts`
//! (module `des::general::adapters::shortest_path_adapter`).
//!
//! JSON adapter registering the Bellman–Ford / Dijkstra DES shortest-path
//! solver. Demonstrates that the registry can host a wholly different model
//! without changes to the registry itself.
//!
//! ## Conversion notes
//!
//!   * `algorithm: 'bellman-ford' | 'dijkstra'` -> [`SPAlgorithm`] enum,
//!     dispatched with `match`.
//!   * `builtin?: 'small-chain'` -> `Option<SPBuiltin>`.
//!   * `buildRandomGraph(.., seed)` in the Rust tree takes an injected
//!     `RandomSource` rather than a raw seed, so we seed a `mulberry32(seed)`
//!     (matching `internal_solver_network.rs`).
//!   * The two DES entry points gained a `BellmanFordOptions` arg in the Rust
//!     port; the TS used the defaults, so we pass `BellmanFordOptions::default()`.
//!   * `throw new Error(...)` when none of `{builtin, graph, randomGraph}` is
//!     provided -> `panic!` (an invariant the registry would have rejected).
//!
//! PORT NOTE: `registerModel` / the model registry (`des-registry.ts`) is not
//! ported yet, and Rust has no module-load side effects. The adapter is exposed
//! via [`adapter()`]; the integrator should call `register_model(adapter())`
//! once the registry exists.

#![allow(dead_code)]

use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};
use crate::des::general::prng::mulberry32;
use crate::des::general::shortest_path_des::{
    build_random_graph, build_small_chain_graph, shortest_path_bellman_ford_des,
    shortest_path_dijkstra_des, BellmanFordOptions, Edge, Graph, SPResult,
};

/// `'bellman-ford' | 'dijkstra'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SPAlgorithm {
    BellmanFord,
    Dijkstra,
}

impl SPAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            SPAlgorithm::BellmanFord => "bellman-ford",
            SPAlgorithm::Dijkstra => "dijkstra",
        }
    }
}

/// `builtin?: 'small-chain'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SPBuiltin {
    SmallChain,
}

/// One adjacency entry in the explicit-graph param (`{to, weight}`).
#[derive(Clone, Copy, Debug)]
pub struct EdgeParam {
    pub to: usize,
    pub weight: f64,
}

/// Explicit graph param.
#[derive(Clone, Debug, Default)]
pub struct GraphParam {
    pub num_nodes: usize,
    pub edges: Vec<Vec<EdgeParam>>,
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub node_names: Option<Vec<String>>,
}

/// Random-graph param.
#[derive(Clone, Copy, Debug)]
pub struct RandomGraphParam {
    pub num_nodes: usize,
    pub edge_prob: f64,
    pub w_min: f64,
    pub w_max: f64,
    pub seed: u32,
}

/// `interface SPParams`.
#[derive(Clone, Debug)]
pub struct SPParams {
    pub algorithm: SPAlgorithm,
    pub source: usize,
    pub graph: Option<GraphParam>,
    pub random_graph: Option<RandomGraphParam>,
    pub builtin: Option<SPBuiltin>,
}

/// `const spSchema`.
pub fn sp_schema() -> ParamSchema {
    let edge_obj = ParamSchema::Object {
        fields: vec![
            (
                "to".to_string(),
                ParamSchema::Number {
                    min: Some(0.0),
                    max: None,
                    integer: Some(true),
                    default: None,
                    description: None,
                },
            ),
            (
                "weight".to_string(),
                ParamSchema::Number {
                    min: None,
                    max: None,
                    integer: None,
                    default: None,
                    description: None,
                },
            ),
        ],
        required: Some(vec!["to".to_string(), "weight".to_string()]),
        description: None,
    };
    let graph_schema = ParamSchema::Object {
        fields: vec![
            (
                "numNodes".to_string(),
                ParamSchema::Number {
                    min: Some(1.0),
                    max: None,
                    integer: Some(true),
                    default: None,
                    description: None,
                },
            ),
            (
                "edges".to_string(),
                ParamSchema::Array {
                    items: Box::new(ParamSchema::Array {
                        items: Box::new(edge_obj),
                        min_length: None,
                        max_length: None,
                        description: None,
                    }),
                    min_length: None,
                    max_length: None,
                    description: None,
                },
            ),
        ],
        required: Some(vec![]),
        description: None,
    };
    let random_graph_schema = ParamSchema::Object {
        fields: vec![
            (
                "numNodes".to_string(),
                ParamSchema::Number {
                    min: Some(2.0),
                    max: Some(1000.0),
                    integer: Some(true),
                    default: None,
                    description: None,
                },
            ),
            (
                "edgeProb".to_string(),
                ParamSchema::Number {
                    min: Some(0.0),
                    max: Some(1.0),
                    integer: None,
                    default: None,
                    description: None,
                },
            ),
            (
                "wMin".to_string(),
                ParamSchema::Number {
                    min: None,
                    max: None,
                    integer: None,
                    default: None,
                    description: None,
                },
            ),
            (
                "wMax".to_string(),
                ParamSchema::Number {
                    min: None,
                    max: None,
                    integer: None,
                    default: None,
                    description: None,
                },
            ),
            (
                "seed".to_string(),
                ParamSchema::Number {
                    min: None,
                    max: None,
                    integer: Some(true),
                    default: None,
                    description: None,
                },
            ),
        ],
        required: Some(vec![]),
        description: None,
    };
    ParamSchema::Object {
        fields: vec![
            (
                "algorithm".to_string(),
                ParamSchema::String {
                    allowed: Some(vec!["bellman-ford".to_string(), "dijkstra".to_string()]),
                    default: None,
                    description: Some("Which DES variant to run".to_string()),
                },
            ),
            (
                "source".to_string(),
                ParamSchema::Number {
                    min: Some(0.0),
                    max: None,
                    integer: Some(true),
                    default: None,
                    description: Some("Source node id".to_string()),
                },
            ),
            ("graph".to_string(), graph_schema),
            ("randomGraph".to_string(), random_graph_schema),
            (
                "builtin".to_string(),
                ParamSchema::String {
                    allowed: Some(vec!["small-chain".to_string()]),
                    default: None,
                    description: None,
                },
            ),
        ],
        required: Some(vec!["algorithm".to_string(), "source".to_string()]),
        description: Some(
            "Shortest path on a directed graph using a DES wave-propagation solver.".to_string(),
        ),
    }
}

/// `const adapter: DESModelRegistration<SPParams, SPResult>`.
pub struct ShortestPathAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> ShortestPathAdapter {
    ShortestPathAdapter
}

impl DESModelRegistration<SPParams, SPResult> for ShortestPathAdapter {
    fn id(&self) -> &str {
        "shortest-path"
    }

    fn description(&self) -> &str {
        "Shortest path solved by DES wave-propagation (Bellman-Ford or Dijkstra)."
    }

    fn schema(&self) -> ParamSchema {
        sp_schema()
    }

    fn run(&self, params: SPParams, _runtime: &DESRuntimeConfig) -> SPResult {
        let g: Graph = if params.builtin == Some(SPBuiltin::SmallChain) {
            build_small_chain_graph()
        } else if let Some(rg) = &params.random_graph {
            let mut rng = mulberry32(rg.seed);
            build_random_graph(rg.num_nodes, rg.edge_prob, rg.w_min, rg.w_max, &mut rng)
        } else if let Some(gr) = &params.graph {
            Graph {
                num_nodes: gr.num_nodes,
                edges: gr
                    .edges
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|e| Edge {
                                to: e.to,
                                weight: e.weight,
                            })
                            .collect()
                    })
                    .collect(),
                coordinates: gr.coordinates.clone(),
                node_names: gr.node_names.clone(),
            }
        } else {
            panic!("shortest-path: provide one of {{builtin, graph, randomGraph}}");
        };

        match params.algorithm {
            SPAlgorithm::BellmanFord => {
                shortest_path_bellman_ford_des(&g, params.source, BellmanFordOptions::default())
            }
            SPAlgorithm::Dijkstra => {
                shortest_path_dijkstra_des(&g, params.source, BellmanFordOptions::default())
            }
        }
    }

    fn summarize(&self, result: &SPResult, params: &SPParams) -> String {
        let reachable = result.distance.iter().filter(|d| d.is_finite()).count();
        let distances_preview = result
            .distance
            .iter()
            .take(12)
            .map(|&d| {
                if d.is_finite() {
                    format!("{d:.2}")
                } else {
                    "∞".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let lines = vec![
            "SHORTEST-PATH RUN SUMMARY".to_string(),
            "──────────────────────────────────".to_string(),
            format!("  Algorithm:       {}", params.algorithm.as_str()),
            format!("  Source:          {}", params.source),
            format!("  Iterations:      {}", result.iterations),
            format!("  Waves emitted:   {}", result.waves_emitted),
            format!(
                "  Reachable nodes: {} / {}",
                reachable,
                result.distance.len()
            ),
            format!(
                "  Negative cycle:  {}",
                if result.has_negative_cycle_from_source {
                    "YES (from source)"
                } else {
                    "no"
                }
            ),
            "".to_string(),
            format!("  Distances (first 12 nodes):  {distances_preview}"),
        ];
        lines.join("\n")
    }

    fn write_csv(&self, result: &SPResult, csv_path: &str) {
        let mut lines = vec!["node,distance,predecessor".to_string()];
        for v in 0..result.distance.len() {
            let dist = if result.distance[v].is_finite() {
                format!("{:.6}", result.distance[v])
            } else {
                "inf".to_string()
            };
            lines.push(format!("{},{},{}", v, dist, result.predecessor[v]));
        }
        crate::des::general::adapters::adapter_utils::write_csv_lines(csv_path, &lines);
    }
}
