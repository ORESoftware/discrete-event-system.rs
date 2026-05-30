//! Port of `src/des/general/adapters/internal-solver-network-adapter.ts`
//! (module `des::general::adapters::internal_solver_network_adapter`).
//!
//! JSON adapter exposing GA / SA / DP / shortest-path / TSP solvers as DES
//! station networks.
//!
//! ## Conversion notes
//!
//!   * `summarizeBestState` duck-typed an erased `bestState: unknown`. Here the
//!     engine models it as the [`SolverBestState`] enum, so we `match` instead.
//!   * `JSON.stringify(row.bestState)` (CSV) -> [`best_state_json`] and
//!     [`metadata_json`] serialise the enum / metadata map by hand.
//!   * `withLogger` structured logging -> the ported [`JsonlLogger`].
//!   * `?? default` chains (`timeLimitMs ?? 180000`) -> `Option::unwrap_or`.
//!
//! PORT NOTE: the `jsonCsvRow` helper in the Rust `adapter_utils` only applies
//! CSV quoting; per the TS `jsonCsvRow` every cell is `JSON.stringify`-d first,
//! so string cells are emitted quoted (e.g. `"knapsack-dp"`) and numbers /
//! booleans use their JSON spellings.
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; the adapter is
//! exposed via [`adapter()`].
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and `drawSolverNetwork`) is not ported, so `animate` is a
//! no-op here.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{
    json_csv_row, validation_line, with_logger, write_csv_lines,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, OneOfVariant, ParamSchema,
    RegistrationExample, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::genetic_tsp::InitMode;
use crate::des::general::internal_solver_network::{
    run_internal_solver_network, InternalSolverKind, InternalSolverRunParams,
    InternalSolverRunResult, InternalSolverStatus, KnapsackParams, MetaValue, ShortestPathAlgorithm,
    SolverBestState, SolverProgressPayload, TSPGAOptionsPartial, TSPSolverParams, TspBuiltin,
};
use crate::des::observability::logger::JsonValue as LogJson;

// =============================================================================
// Number / JSON formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `JSON.stringify(number)` (non-finite -> `null`).
fn json_num(v: f64) -> String {
    if v.is_finite() { js_number(v) } else { "null".to_string() }
}

fn bool_json(b: bool) -> String {
    if b { "true".to_string() } else { "false".to_string() }
}

/// `v.toExponential(digits)`.
fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

/// `function formatNumber(x)`.
fn format_number(x: f64) -> String {
    if !x.is_finite() {
        return js_number(x);
    }
    if x.abs() >= 1e9 || (x.abs() < 1e-3 && x != 0.0) {
        return to_exponential(x, 3);
    }
    format!("{x:.4}")
}

/// JSON-quote a string (`JSON.stringify(string)`).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn status_str(s: InternalSolverStatus) -> &'static str {
    match s {
        InternalSolverStatus::Complete => "complete",
        InternalSolverStatus::TimeLimit => "time-limit",
        InternalSolverStatus::TickLimit => "tick-limit",
    }
}

fn algorithm_str(a: ShortestPathAlgorithm) -> &'static str {
    match a {
        ShortestPathAlgorithm::BellmanFord => "bellman-ford",
        ShortestPathAlgorithm::Dijkstra => "dijkstra",
    }
}

/// `function summarizeBestState(row)`.
fn summarize_best_state(row: &SolverProgressPayload) -> String {
    match &row.best_state {
        SolverBestState::ShortestPath { distance, .. } => {
            let ds: Vec<String> = distance
                .iter()
                .take(8)
                .map(|v| if v.is_finite() { format!("{v:.2}") } else { "inf".to_string() })
                .collect();
            format!("dist=[{}{}]", ds.join(", "), if distance.len() > 8 { ", ..." } else { "" })
        }
        SolverBestState::Knapsack { value, weight, capacity, .. } => format!(
            "value={} weight={}/{}",
            format_number(*value),
            format_number(*weight),
            format_number(*capacity)
        ),
        SolverBestState::Tour { tour, length } => {
            let ts: Vec<String> = tour.iter().take(9).map(|v| v.to_string()).collect();
            format!(
                "length={} tour=[{}{}]",
                format_number(*length),
                ts.join(" -> "),
                if tour.len() > 9 { " -> ..." } else { "" }
            )
        }
    }
}

/// `JSON.stringify(row.bestState)`.
fn best_state_json(s: &SolverBestState) -> String {
    match s {
        SolverBestState::ShortestPath { distance, predecessor, algorithm, has_negative_cycle_from_source } => {
            let d = distance.iter().map(|v| json_num(*v)).collect::<Vec<_>>().join(",");
            let p = predecessor.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            format!(
                "{{\"distance\":[{}],\"predecessor\":[{}],\"algorithm\":{},\"hasNegativeCycleFromSource\":{}}}",
                d,
                p,
                json_str(algorithm_str(*algorithm)),
                bool_json(*has_negative_cycle_from_source)
            )
        }
        SolverBestState::Knapsack { selected, value, weight, capacity } => {
            let sel = selected.iter().map(|v| json_num(*v)).collect::<Vec<_>>().join(",");
            format!(
                "{{\"selected\":[{}],\"value\":{},\"weight\":{},\"capacity\":{}}}",
                sel,
                json_num(*value),
                json_num(*weight),
                json_num(*capacity)
            )
        }
        SolverBestState::Tour { tour, length } => {
            let t = tour.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            format!("{{\"tour\":[{}],\"length\":{}}}", t, json_num(*length))
        }
    }
}

/// `JSON.stringify(row.metadata ?? {})`.
fn metadata_json(meta: &[(String, MetaValue)]) -> String {
    let inner = meta
        .iter()
        .map(|(k, v)| {
            let vs = match v {
                MetaValue::Number(n) => json_num(*n),
                MetaValue::Bool(b) => bool_json(*b),
            };
            format!("{}:{}", json_str(k), vs)
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{}}}", inner)
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn str_enum(allowed: &[&str], default: Option<&str>) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: default.map(|s| s.to_string()),
        description: None,
    }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn obj_desc(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>, description: &str) -> ParamSchema {
    match obj(fields, required) {
        ParamSchema::Object { fields, required, .. } => {
            ParamSchema::Object { fields, required, description: Some(description.to_string()) }
        }
        other => other,
    }
}

/// `const coolingSchema` (a `oneOf` over four cooling kinds).
fn cooling_schema() -> ParamSchema {
    let variant = |tag: &str, fields: Vec<(&str, ParamSchema)>, required: Vec<&str>| OneOfVariant {
        tag: tag.to_string(),
        tag_field: None,
        schema: obj(fields, required),
        description: None,
    };
    ParamSchema::OneOf {
        variants: vec![
            variant(
                "geometric",
                vec![
                    ("kind", str_enum(&["geometric"], None)),
                    ("T0", num(Some(0.0), None, None, None)),
                    ("alpha", num(Some(0.0), Some(1.0), None, None)),
                    ("Tmin", num(Some(0.0), None, None, None)),
                ],
                vec!["kind", "T0", "alpha"],
            ),
            variant(
                "logarithmic",
                vec![
                    ("kind", str_enum(&["logarithmic"], None)),
                    ("T0", num(Some(0.0), None, None, None)),
                    ("Tmin", num(Some(0.0), None, None, None)),
                ],
                vec!["kind", "T0"],
            ),
            variant(
                "linear",
                vec![
                    ("kind", str_enum(&["linear"], None)),
                    ("T0", num(Some(0.0), None, None, None)),
                    ("rate", num(Some(0.0), None, None, None)),
                    ("Tmin", num(Some(0.0), None, None, None)),
                ],
                vec!["kind", "T0", "rate"],
            ),
            variant(
                "exp-restart",
                vec![
                    ("kind", str_enum(&["exp-restart"], None)),
                    ("T0", num(Some(0.0), None, None, None)),
                    ("alpha", num(Some(0.0), Some(1.0), None, None)),
                    ("period", num(Some(1.0), None, Some(true), None)),
                    ("Tmin", num(Some(0.0), None, None, None)),
                ],
                vec!["kind", "T0", "alpha", "period"],
            ),
        ],
        description: None,
    }
}

fn graph_edge_schema() -> ParamSchema {
    obj(
        vec![("to", num(Some(0.0), None, Some(true), None)), ("weight", num(None, None, None, None))],
        vec!["to", "weight"],
    )
}

fn graph_schema() -> ParamSchema {
    obj(
        vec![
            ("numNodes", num(Some(1.0), None, Some(true), None)),
            ("edges", arr(arr(graph_edge_schema(), None, None), None, None)),
            (
                "coordinates",
                arr(arr(num(None, None, None, None), Some(2), Some(2)), None, None),
            ),
            ("nodeNames", arr(string_field(), None, None)),
        ],
        vec!["numNodes", "edges"],
    )
}

fn shortest_path_schema() -> ParamSchema {
    obj(
        vec![
            ("algorithm", str_enum(&["bellman-ford", "dijkstra"], Some("dijkstra"))),
            ("source", num(Some(0.0), None, Some(true), Some(0.0))),
            ("builtin", str_enum(&["small-chain"], None)),
            ("graph", graph_schema()),
            (
                "randomGraph",
                obj(
                    vec![
                        ("numNodes", num(Some(2.0), Some(100000.0), Some(true), None)),
                        ("edgeProb", num(Some(0.0), Some(1.0), None, None)),
                        ("wMin", num(None, None, None, None)),
                        ("wMax", num(None, None, None, None)),
                        ("seed", num(None, None, Some(true), None)),
                    ],
                    vec!["numNodes", "edgeProb", "wMin", "wMax", "seed"],
                ),
            ),
        ],
        vec!["algorithm", "source"],
    )
}

fn knapsack_schema() -> ParamSchema {
    obj(
        vec![
            ("values", arr(num(None, None, None, None), Some(1), None)),
            ("weights", arr(num(None, None, None, None), Some(1), None)),
            ("capacity", num(Some(0.0), None, Some(true), None)),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("maxIterations", num(Some(1.0), None, Some(true), Some(5000.0))),
            ("cooling", cooling_schema()),
            ("stallLimit", num(Some(0.0), None, Some(true), Some(0.0))),
            ("penalty", num(Some(0.0), None, None, Some(1_000_000.0))),
        ],
        vec!["values", "weights", "capacity"],
    )
}

fn tsp_sa_schema() -> ParamSchema {
    obj(
        vec![
            ("cooling", cooling_schema()),
            ("maxIterations", num(Some(1.0), None, Some(true), Some(5000.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("init", str_enum(&["random", "nearest-neighbor"], Some("nearest-neighbor"))),
            ("moves", str_enum(&["2-opt", "or-opt", "mixed"], Some("mixed"))),
            ("penaltyPerViolation", num(Some(0.0), None, None, Some(1_000_000.0))),
            ("traceStride", num(Some(1.0), None, Some(true), None)),
            ("stallLimit", num(Some(0.0), None, Some(true), Some(0.0))),
        ],
        vec!["maxIterations", "seed"],
    )
}

fn tsp_ga_schema() -> ParamSchema {
    obj(
        vec![
            ("popSize", num(Some(2.0), None, Some(true), Some(60.0))),
            ("numGenerations", num(Some(1.0), None, Some(true), Some(200.0))),
            ("tournamentSize", num(Some(1.0), None, Some(true), Some(3.0))),
            ("crossoverProb", num(Some(0.0), Some(1.0), None, Some(0.95))),
            ("mutationProb", num(Some(0.0), Some(1.0), None, Some(0.3))),
            ("elitism", num(Some(0.0), None, Some(true), Some(2.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("init", str_enum(&["random", "nearest-neighbor"], Some("nearest-neighbor"))),
            ("penaltyPerViolation", num(Some(0.0), None, None, Some(1_000_000.0))),
        ],
        vec!["popSize", "numGenerations", "seed"],
    )
}

fn tsp_schema() -> ParamSchema {
    obj(
        vec![
            ("builtin", str_enum(&["pentagon", "random"], Some("pentagon"))),
            ("n", num(Some(3.0), None, Some(true), Some(5.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            (
                "coordinates",
                arr(arr(num(None, None, None, None), Some(2), Some(2)), None, None),
            ),
            ("distance", arr(arr(num(None, None, None, None), None, None), None, None)),
            (
                "precedence",
                arr(arr(num(Some(0.0), None, Some(true), None), Some(2), Some(2)), None, None),
            ),
            ("sa", tsp_sa_schema()),
            ("ga", tsp_ga_schema()),
        ],
        vec![],
    )
}

/// `const internalSolverSchema`.
pub fn internal_solver_schema() -> ParamSchema {
    obj_desc(
        vec![
            (
                "kind",
                str_enum(
                    &["shortest-path", "knapsack-dp", "knapsack-sa", "tsp-sa", "tsp-ga", "tsp-held-karp"],
                    None,
                ),
            ),
            ("timeLimitMs", num(Some(0.0), None, None, Some(180000.0))),
            ("maxTicks", num(Some(1.0), None, Some(true), None)),
            ("checkEveryTicks", num(Some(1.0), None, Some(true), Some(1.0))),
            ("shortestPath", shortest_path_schema()),
            ("knapsack", knapsack_schema()),
            ("tsp", tsp_schema()),
        ],
        vec!["kind"],
        "Internal optimization/search solvers represented as DES station networks.",
    )
}

// =============================================================================
// Adapter
// =============================================================================

pub struct InternalSolverNetworkAdapter;

pub fn adapter() -> InternalSolverNetworkAdapter {
    InternalSolverNetworkAdapter
}

impl DESModelRegistration<InternalSolverRunParams, InternalSolverRunResult>
    for InternalSolverNetworkAdapter
{
    fn id(&self) -> &str {
        "internal-solver-network"
    }
    fn description(&self) -> &str {
        "Internal GA, SA, knapsack, shortest-path, and TSP solvers as DES station/movable networks."
    }
    fn schema(&self) -> ParamSchema {
        internal_solver_schema()
    }

    fn run(
        &self,
        params: InternalSolverRunParams,
        runtime: &DESRuntimeConfig,
    ) -> InternalSolverRunResult {
        let solver_kind = params.kind;
        let time_limit = params.time_limit_ms.unwrap_or(180000.0);
        with_logger(runtime, move |mut logger| {
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("internal-solver-start".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("solverKind".to_string(), LogJson::String(solver_kind.as_str().to_string())),
                    ("timeLimitMs".to_string(), LogJson::Number(time_limit)),
                ]));
            }
            let result = run_internal_solver_network(params);
            let stride = (result.trace.len() / 50).max(1);
            let mut i = 0;
            while i < result.trace.len() {
                let row = &result.trace[i];
                if let Some(l) = logger.as_deref_mut() {
                    l.log(LogJson::Object(vec![
                        ("kind".to_string(), LogJson::String("internal-solver-trace".to_string())),
                        ("level".to_string(), LogJson::String("debug".to_string())),
                        ("solverKind".to_string(), LogJson::String(row.solver_kind.as_str().to_string())),
                        ("iteration".to_string(), LogJson::Number(row.iteration as f64)),
                        ("objective".to_string(), LogJson::Number(row.objective)),
                        ("feasible".to_string(), LogJson::Bool(row.feasible)),
                        ("done".to_string(), LogJson::Bool(row.done)),
                    ]));
                }
                i += stride;
            }
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("internal-solver-finish".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("solverKind".to_string(), LogJson::String(result.kind.as_str().to_string())),
                    ("status".to_string(), LogJson::String(status_str(result.status).to_string())),
                    ("objective".to_string(), LogJson::Number(result.best.objective)),
                    ("iterations".to_string(), LogJson::Number(result.best.iteration as f64)),
                    (
                        "validationOk".to_string(),
                        LogJson::Bool(result.run_summary.validation_ok.unwrap_or(true)),
                    ),
                ]));
            }
            result
        })
    }

    fn summarize(&self, result: &InternalSolverRunResult, _params: &InternalSolverRunParams) -> String {
        let reason = result.run_summary.reason.map(|r| r.as_str()).unwrap_or("done");
        [
            "INTERNAL SOLVER NETWORK".to_string(),
            "------------------------".to_string(),
            format!("  kind={} status={}", result.kind.as_str(), status_str(result.status)),
            format!(
                "  iterations={} ticks={} reason={}",
                result.best.iteration, result.run_summary.ticks, reason
            ),
            format!(
                "  objective={} feasible={} done={}",
                format_number(result.best.objective),
                result.best.feasible,
                result.best.done
            ),
            format!(
                "  wall-clock={} / {} ms checks={}",
                format_number(result.wall_clock.elapsed_ms),
                format_number(result.wall_clock.budget_ms),
                result.wall_clock.checks
            ),
            format!(
                "  network stationary={} moving={} edges={}",
                result.network.stationary_entities.len(),
                result.network.moving_entities.len(),
                result.network.edges.len()
            ),
            format!("  validation: {}", validation_line(&result.validation)),
            format!("  best: {}", summarize_best_state(&result.best)),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &InternalSolverRunResult, csv_path: &str) {
        let mut lines =
            vec!["tick,iteration,solver_kind,objective,feasible,done,best_state,metadata".to_string()];
        for row in &result.trace {
            lines.push(json_csv_row([
                json_num(row.tick as f64),
                json_num(row.iteration as f64),
                json_str(row.solver_kind.as_str()),
                json_num(row.objective),
                bool_json(row.feasible),
                bool_json(row.done),
                best_state_json(&row.best_state),
                metadata_json(&row.metadata),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn animate(
        &self,
        _result: &InternalSolverRunResult,
        _params: &InternalSolverRunParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }

    fn examples(&self) -> Vec<RegistrationExample<InternalSolverRunParams>> {
        let knapsack = RegistrationExample {
            name: "knapsack dynamic programming".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "internal-solver-network".to_string(),
                description: None,
                parameters: InternalSolverRunParams {
                    kind: InternalSolverKind::KnapsackDp,
                    time_limit_ms: Some(180000.0),
                    max_ticks: None,
                    check_every_ticks: None,
                    shortest_path: None,
                    knapsack: Some(KnapsackParams {
                        values: vec![20.0, 30.0, 35.0, 12.0, 3.0],
                        weights: vec![2.0, 5.0, 7.0, 3.0, 1.0],
                        capacity: 10.0,
                        seed: None,
                        max_iterations: None,
                        cooling: None,
                        stall_limit: None,
                        penalty: None,
                    }),
                    tsp: None,
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        };

        let tsp_ga = RegistrationExample {
            name: "tsp genetic algorithm".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "internal-solver-network".to_string(),
                description: None,
                parameters: InternalSolverRunParams {
                    kind: InternalSolverKind::TspGa,
                    time_limit_ms: Some(180000.0),
                    max_ticks: None,
                    check_every_ticks: None,
                    shortest_path: None,
                    knapsack: None,
                    tsp: Some(TSPSolverParams {
                        builtin: Some(TspBuiltin::Pentagon),
                        n: Some(8),
                        seed: Some(7),
                        coordinates: None,
                        distance: None,
                        precedence: None,
                        sa: None,
                        ga: Some(TSPGAOptionsPartial {
                            pop_size: Some(40),
                            num_generations: Some(80),
                            tournament_size: None,
                            crossover_prob: None,
                            mutation_prob: None,
                            elitism: None,
                            seed: Some(11),
                            init: Some(InitMode::NearestNeighbor),
                            penalty_per_violation: None,
                        }),
                    }),
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        };

        vec![knapsack, tsp_ga]
    }
}
