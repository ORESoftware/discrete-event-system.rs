//! Port of `src/des/general/adapters/network-flow-adapters.ts`
//! (module `des::general::adapters::network_flow_adapters`).
//!
//! Registers three JSON adapters: `max-flow`, `stochastic-flow-mdp`, and
//! `traffic-flow`.
//!
//! ## Conversion notes
//!
//!   * `params.problem ?? buildDefault…()` -> `Option::unwrap_or_else`.
//!   * `runtime.seed ?? params.seed ?? 7` -> `runtime.seed.or(params.seed)…`.
//!   * `builtin` string-literal enums become small marker enums (the field is
//!     decorative; `run` always falls back to the engine default builder).
//!   * `csvRow` cells are stringified with `String(v)` ([`js_number`]);
//!     `jsonCsvRow` cells are `JSON.stringify`-d ([`json_num`] / [`json_str`] /
//!     [`json_num_array`]).
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; the three
//! adapters are exposed via the `*_adapter()` constructors.
//!
//! PORT NOTE: the `stochastic-flow-mdp` model registered here is the same model
//! id as the standalone `stochastic_flow_mdp_adapter.rs` port; the integrator
//! should register only one of them.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, json_csv_row, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::max_flow::{
    build_textbook_max_flow_problem, solve_max_flow, MaxFlowProblem, MaxFlowResult, MaxFlowStatus,
};
use crate::des::general::stochastic_flow_mdp::{
    build_default_stochastic_flow_mdp_problem, solve_stochastic_flow_mdp,
    SolveStochasticFlowMDPOptions, StochasticFlowMDPProblem, StochasticFlowMDPResult,
};
use crate::des::general::traffic_flow::{
    build_default_traffic_problem, run_traffic_simulation, TrafficProblem, TrafficSimulationResult,
};

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        let s = v.to_string();
        if s == "-0" {
            "0".to_string()
        } else {
            s
        }
    }
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        js_number(v)
    } else {
        "null".to_string()
    }
}

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

/// `JSON.stringify(number[])`.
fn json_num_array(xs: &[f64]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|v| json_num(*v))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn max_flow_status_str(s: MaxFlowStatus) -> &'static str {
    match s {
        MaxFlowStatus::Optimal => "optimal",
        MaxFlowStatus::Infeasible => "infeasible",
    }
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: None,
    }
}

fn str_enum(allowed: &[&str], default: Option<&str>) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: default.map(|s| s.to_string()),
        description: None,
    }
}

fn string_field() -> ParamSchema {
    ParamSchema::String {
        allowed: None,
        default: None,
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length,
        max_length: None,
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn obj_desc(
    fields: Vec<(&str, ParamSchema)>,
    required: Vec<&str>,
    description: &str,
) -> ParamSchema {
    match obj(fields, required) {
        ParamSchema::Object {
            fields, required, ..
        } => ParamSchema::Object {
            fields,
            required,
            description: Some(description.to_string()),
        },
        other => other,
    }
}

// =============================================================================
// 1. max-flow
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub enum MaxFlowBuiltin {
    Textbook,
}

#[derive(Clone, Debug, Default)]
pub struct MaxFlowParams {
    pub builtin: Option<MaxFlowBuiltin>,
    pub problem: Option<MaxFlowProblem>,
}

fn max_flow_edge_schema() -> ParamSchema {
    obj(
        vec![
            ("from", num(Some(0.0), None, Some(true), None)),
            ("to", num(Some(0.0), None, Some(true), None)),
            ("capacity", num(Some(0.0), None, None, None)),
            ("name", string_field()),
        ],
        vec!["from", "to", "capacity"],
    )
}

fn max_flow_problem_schema() -> ParamSchema {
    obj_desc(
        vec![
            ("numNodes", num(Some(2.0), None, Some(true), None)),
            ("source", num(Some(0.0), None, Some(true), None)),
            ("sink", num(Some(0.0), None, Some(true), None)),
            ("edges", arr(max_flow_edge_schema(), Some(1))),
        ],
        vec!["numNodes", "source", "sink", "edges"],
        "Directed capacitated network with a single source and sink.",
    )
}

pub struct MaxFlowAdapter;

pub fn max_flow_adapter() -> MaxFlowAdapter {
    MaxFlowAdapter
}

impl DESModelRegistration<MaxFlowParams, MaxFlowResult> for MaxFlowAdapter {
    fn id(&self) -> &str {
        "max-flow"
    }
    fn description(&self) -> &str {
        "Maximum flow/min-cut optimisation via Edmonds-Karp DES ticks."
    }
    fn schema(&self) -> ParamSchema {
        obj_desc(
            vec![
                ("builtin", str_enum(&["textbook"], Some("textbook"))),
                ("problem", max_flow_problem_schema()),
            ],
            vec![],
            "Maximum flow solved by DES augmenting-path iterations.",
        )
    }
    fn run(&self, params: MaxFlowParams, _runtime: &DESRuntimeConfig) -> MaxFlowResult {
        solve_max_flow(
            params
                .problem
                .unwrap_or_else(build_textbook_max_flow_problem),
        )
    }
    fn summarize(&self, result: &MaxFlowResult, _params: &MaxFlowParams) -> String {
        let cut_edges = result
            .min_cut
            .cut_edges
            .iter()
            .map(|e| {
                e.name
                    .clone()
                    .unwrap_or_else(|| format!("{}->{}", e.from, e.to))
            })
            .collect::<Vec<_>>()
            .join(", ");
        [
            "MAX-FLOW OPTIMISATION".to_string(),
            "---------------------".to_string(),
            format!("  Status:          {}", max_flow_status_str(result.status)),
            format!("  Max flow:        {:.6}", result.max_flow),
            format!("  Iterations:      {}", result.iterations),
            format!("  Augmentations:   {}", result.trace.len()),
            format!("  Min-cut cap:     {:.6}", result.min_cut.capacity),
            format!(
                "  Source side:     {{{}}}",
                result
                    .min_cut
                    .source_side
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!("  Cut edges:       {cut_edges}"),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &MaxFlowResult, csv_path: &str) {
        let mut lines = vec!["from,to,name,capacity,flow".to_string()];
        for e in &result.edge_flows {
            lines.push(csv_row([
                e.from.to_string(),
                e.to.to_string(),
                e.name.clone().unwrap_or_default(),
                js_number(e.capacity),
                js_number(e.flow),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn examples(&self) -> Vec<RegistrationExample<MaxFlowParams>> {
        vec![RegistrationExample {
            name: "textbook".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "max-flow".to_string(),
                description: Some("Textbook six-node maximum-flow/min-cut example.".to_string()),
                parameters: MaxFlowParams {
                    builtin: Some(MaxFlowBuiltin::Textbook),
                    problem: None,
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}

// =============================================================================
// 2. stochastic-flow-mdp
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub enum StochasticFlowBuiltin {
    SmallStochasticNetwork,
}

#[derive(Clone, Debug, Default)]
pub struct StochasticFlowMDPParams {
    pub builtin: Option<StochasticFlowBuiltin>,
    pub problem: Option<StochasticFlowMDPProblem>,
    pub seed: Option<u32>,
    pub max_policy_rows: Option<usize>,
}

fn stochastic_flow_edge_schema() -> ParamSchema {
    obj(
        vec![
            ("from", num(Some(0.0), None, Some(true), None)),
            ("to", num(Some(0.0), None, Some(true), None)),
            ("capacity", num(Some(0.0), None, Some(true), None)),
            ("successProb", num(Some(0.0), Some(1.0), None, None)),
            ("cost", num(Some(0.0), None, None, None)),
            ("name", string_field()),
        ],
        vec!["from", "to", "capacity", "successProb"],
    )
}

fn stochastic_flow_problem_schema() -> ParamSchema {
    obj_desc(
        vec![
            ("numNodes", num(Some(2.0), None, Some(true), None)),
            ("source", num(Some(0.0), None, Some(true), None)),
            ("sink", num(Some(0.0), None, Some(true), None)),
            ("edges", arr(stochastic_flow_edge_schema(), Some(1))),
            ("horizon", num(Some(1.0), None, Some(true), None)),
            ("deliveredReward", num(Some(1e-9), None, None, None)),
            ("waitPenalty", num(Some(0.0), None, None, None)),
            ("failurePenalty", num(Some(0.0), None, None, None)),
            ("discount", num(Some(0.0), Some(1.0), None, None)),
            ("maxStates", num(Some(1.0), None, Some(true), None)),
        ],
        vec!["numNodes", "source", "sink", "edges", "horizon"],
        "Finite-horizon stochastic flow-control MDP on a directed network.",
    )
}

pub struct StochasticFlowMdpAdapter;

pub fn stochastic_flow_mdp_adapter() -> StochasticFlowMdpAdapter {
    StochasticFlowMdpAdapter
}

impl DESModelRegistration<StochasticFlowMDPParams, StochasticFlowMDPResult>
    for StochasticFlowMdpAdapter
{
    fn id(&self) -> &str {
        "stochastic-flow-mdp"
    }
    fn description(&self) -> &str {
        "MDP interpretation of max-flow: stochastic capacities/availability with sequential routing control."
    }
    fn schema(&self) -> ParamSchema {
        obj_desc(
            vec![
                (
                    "builtin",
                    str_enum(
                        &["small-stochastic-network"],
                        Some("small-stochastic-network"),
                    ),
                ),
                ("problem", stochastic_flow_problem_schema()),
                ("seed", num(None, None, Some(true), Some(7.0))),
                (
                    "maxPolicyRows",
                    num(Some(1.0), None, Some(true), Some(24.0)),
                ),
            ],
            vec![],
            "MDP interpretation of max-flow when edge availability/capacity is stochastic.",
        )
    }
    fn run(
        &self,
        params: StochasticFlowMDPParams,
        runtime: &DESRuntimeConfig,
    ) -> StochasticFlowMDPResult {
        let seed = runtime.seed.map(|s| s as u32).or(params.seed).unwrap_or(7);
        solve_stochastic_flow_mdp(
            params
                .problem
                .unwrap_or_else(build_default_stochastic_flow_mdp_problem),
            SolveStochasticFlowMDPOptions {
                seed: Some(seed),
                max_policy_rows: Some(params.max_policy_rows.unwrap_or(24)),
            },
        )
    }
    fn summarize(
        &self,
        result: &StochasticFlowMDPResult,
        _params: &StochasticFlowMDPParams,
    ) -> String {
        let first = result
            .initial_policy
            .iter()
            .take(5)
            .map(|row| format!("t{}:{}", row.stage, row.action.label))
            .collect::<Vec<_>>()
            .join(" -> ");
        [
            "STOCHASTIC FLOW-CONTROL MDP".to_string(),
            "---------------------------".to_string(),
            format!("  Horizon:         {}", result.horizon),
            format!("  States:          {}", result.num_states),
            format!("  E[reward]*:      {:.6}", result.expected_reward),
            format!(
                "  Static max-flow: {:.6}  (deterministic upper bound)",
                result.deterministic_max_flow
            ),
            format!("  First policy:    {first}"),
            format!(
                "  Sim delivered:   {}",
                js_number(result.simulation.delivered)
            ),
            format!("  Sim reward:      {:.6}", result.simulation.total_reward),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &StochasticFlowMDPResult, csv_path: &str) {
        let mut lines = vec!["stage,state_index,node,capacities,action,value".to_string()];
        for row in &result.policy {
            lines.push(json_csv_row([
                json_num(row.stage as f64),
                json_num(row.state_index as f64),
                json_num(row.state.node as f64),
                json_num_array(&row.state.capacities),
                json_str(&row.action.label),
                json_num(row.value),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn examples(&self) -> Vec<RegistrationExample<StochasticFlowMDPParams>> {
        vec![RegistrationExample {
            name: "small-stochastic-network".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "stochastic-flow-mdp".to_string(),
                description: Some(
                    "MDP interpretation of max-flow with stochastic edge availability.".to_string(),
                ),
                parameters: StochasticFlowMDPParams {
                    builtin: Some(StochasticFlowBuiltin::SmallStochasticNetwork),
                    problem: None,
                    seed: Some(7),
                    max_policy_rows: None,
                },
                runtime: Some(DESRuntimeConfig {
                    seed: Some(7.0),
                    ..Default::default()
                }),
                metadata: None,
            },
        }]
    }
}

// =============================================================================
// 3. traffic-flow
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub enum TrafficBuiltin {
    FiveIntersection,
}

#[derive(Clone, Debug, Default)]
pub struct TrafficParams {
    pub builtin: Option<TrafficBuiltin>,
    pub problem: Option<TrafficProblem>,
}

fn traffic_node_schema() -> ParamSchema {
    obj(
        vec![
            ("id", num(Some(0.0), None, Some(true), None)),
            ("name", string_field()),
            ("x", num(None, None, None, None)),
            ("y", num(None, None, None, None)),
            ("signalOffsetSec", num(Some(0.0), None, None, None)),
        ],
        vec!["id", "name", "x", "y"],
    )
}

fn traffic_link_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("from", num(Some(0.0), None, Some(true), None)),
            ("to", num(Some(0.0), None, Some(true), None)),
            ("lengthM", num(Some(1e-9), None, None, None)),
            ("speedLimitMps", num(Some(1e-9), None, None, None)),
            ("capacity", num(Some(1.0), Some(299.0), Some(true), None)),
            ("dischargePerMin", num(Some(1e-9), None, None, None)),
        ],
        vec!["id", "from", "to", "lengthM", "speedLimitMps"],
    )
}

fn traffic_source_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("node", num(Some(0.0), None, Some(true), None)),
            ("destNode", num(Some(0.0), None, Some(true), None)),
            ("ratePerMin", num(Some(0.0), None, None, None)),
            ("maxGenerated", num(Some(0.0), None, Some(true), None)),
            ("startSec", num(Some(0.0), None, None, None)),
            ("endSec", num(Some(0.0), None, None, None)),
        ],
        vec!["id", "node", "destNode", "ratePerMin"],
    )
}

fn traffic_problem_schema() -> ParamSchema {
    obj_desc(
        vec![
            ("nodes", arr(traffic_node_schema(), Some(2))),
            ("links", arr(traffic_link_schema(), Some(1))),
            ("sources", arr(traffic_source_schema(), Some(1))),
            ("durationSec", num(Some(1e-9), None, None, None)),
            ("dtSec", num(Some(1e-9), None, None, None)),
            ("maxCars", num(Some(1.0), Some(299.0), Some(true), None)),
            ("minGapM", num(Some(1e-9), None, None, None)),
            ("accelMps2", num(Some(1e-9), None, None, None)),
            ("signalCycleSec", num(Some(1e-9), None, None, None)),
            ("drainAfterSourcesSec", num(Some(0.0), None, None, None)),
            ("seed", num(None, None, Some(true), None)),
        ],
        vec![
            "nodes",
            "links",
            "sources",
            "durationSec",
            "dtSec",
            "maxCars",
            "minGapM",
            "accelMps2",
            "signalCycleSec",
        ],
        "Small directed road network with signalized intersections and moving cars.",
    )
}

pub struct TrafficFlowAdapter;

pub fn traffic_flow_adapter() -> TrafficFlowAdapter {
    TrafficFlowAdapter
}

impl DESModelRegistration<TrafficParams, TrafficSimulationResult> for TrafficFlowAdapter {
    fn id(&self) -> &str {
        "traffic-flow"
    }
    fn description(&self) -> &str {
        "Continuous-position traffic simulation on a five-intersection grid with max-flow upper bound."
    }
    fn schema(&self) -> ParamSchema {
        obj_desc(
            vec![
                ("builtin", str_enum(&["five-intersection"], Some("five-intersection"))),
                ("problem", traffic_problem_schema()),
            ],
            vec![],
            "Traffic-flow simulation with stationary grid/link/intersection entities and moving cars.",
        )
    }
    fn run(&self, params: TrafficParams, runtime: &DESRuntimeConfig) -> TrafficSimulationResult {
        let mut problem = params.problem.unwrap_or_else(build_default_traffic_problem);
        if let Some(s) = runtime.seed {
            problem.seed = Some(s as u32);
        }
        run_traffic_simulation(&problem)
    }
    fn summarize(&self, result: &TrafficSimulationResult, _params: &TrafficParams) -> String {
        [
            "TRAFFIC-FLOW DES".to_string(),
            "----------------".to_string(),
            format!("  Generated:       {}", js_number(result.generated_cars)),
            format!("  Completed:       {}", js_number(result.completed_cars)),
            format!("  Active at stop:  {}", js_number(result.active_cars)),
            format!("  Max active:      {}", js_number(result.max_active_cars)),
            format!(
                "  Blocked tries:   {}",
                js_number(result.blocked_source_attempts)
            ),
            format!("  Mean travel:     {:.3} sec", result.mean_travel_time_sec),
            format!("  P95 travel:      {:.3} sec", result.p95_travel_time_sec),
            format!(
                "  Throughput:      {:.3} cars/hour",
                result.throughput_per_hour
            ),
            format!(
                "  Max-flow bound:  {:.3} cars/min",
                result.max_flow_upper_bound_per_min
            ),
            format!("  Throughput/bnd:  {:.3}", result.throughput_vs_max_flow),
            format!(
                "  Invariants:      {}",
                if result.invariant_violations.is_empty() {
                    "ok".to_string()
                } else {
                    format!("{} violations", result.invariant_violations.len())
                }
            ),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &TrafficSimulationResult, csv_path: &str) {
        let mut lines = vec![
            "id,from,to,capacity,entered,exited,final_occupancy,max_occupancy,avg_occupancy"
                .to_string(),
        ];
        for l in &result.link_stats {
            lines.push(csv_row([
                l.id.clone(),
                l.from.to_string(),
                l.to.to_string(),
                js_number(l.capacity),
                js_number(l.entered),
                js_number(l.exited),
                js_number(l.final_occupancy),
                js_number(l.max_occupancy),
                js_number(l.avg_occupancy),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn examples(&self) -> Vec<RegistrationExample<TrafficParams>> {
        vec![RegistrationExample {
            name: "five-intersection".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "traffic-flow".to_string(),
                description: Some(
                    "Five-intersection traffic-flow scenario with fewer than 300 cars.".to_string(),
                ),
                parameters: TrafficParams {
                    builtin: Some(TrafficBuiltin::FiveIntersection),
                    problem: None,
                },
                runtime: Some(DESRuntimeConfig {
                    seed: Some(7.0),
                    ..Default::default()
                }),
                metadata: None,
            },
        }]
    }
}
