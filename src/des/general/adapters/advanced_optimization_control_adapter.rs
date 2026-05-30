//! Port of `src/des/general/adapters/advanced-optimization-control-adapter.ts`
//! (module `des::general::adapters::advanced_optimization_control_adapter`).
//!
//! Registers seven JSON adapters: particle-swarm, ant-colony-tsp,
//! map-coloring-csp, max-sat-local-search, sdp-maxcut-relaxation,
//! pareto-portfolio, hinfinity-robust-control, and pursuit-evasion-game.
//!
//! ## Conversion notes
//!
//!   * Params shapes reuse the engine `*Params` structs; the JSON `empty array
//!     -> undefined` guards become `Option::filter(|v| !v.is_empty())`.
//!   * `runHInfinityRobustControl` / `runPursuitEvasionGame` return
//!     `Result<_, PreconditionError>`; the TS threw on failure, so the Rust
//!     `run` unwraps with `panic!`.
//!   * `result.captureTick === null` -> `Option<usize>`.
//!   * `JSON.stringify(point.weights)` (CSV) -> [`json_num_array`].
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; the adapters
//! are exposed via the `*_adapter()` constructors.
//!
//! PORT NOTE: `MapColoringCSPResult::assignment` is a `HashMap` (the TS used an
//! insertion-ordered object). The summary sorts keys for determinism, which can
//! differ in ordering from the TS `Object.entries` output.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::advanced_control_models::{
    run_h_infinity_robust_control, run_pursuit_evasion_game, HInfinityRobustControlParams,
    HInfinityRobustControlResult, PursuitEvasionGameParams, PursuitEvasionGameResult,
};
use crate::des::general::advanced_optimization_models::{
    run_ant_colony_tsp, run_map_coloring_csp, run_max_sat_local_search, run_pareto_portfolio,
    run_particle_swarm, run_sdp_max_cut_relaxation, AntColonyTSPParams, AntColonyTSPResult,
    MapColoringCSPParams, MapColoringCSPResult, MaxSATParams, MaxSATResult, ParetoPortfolioParams,
    ParetoPortfolioResult, ParticleSwarmParams, ParticleSwarmResult, SDPMaxCutParams,
    SDPMaxCutResult,
};
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};

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

fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length,
        max_length,
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

fn number_vector_schema() -> ParamSchema {
    arr(num(None, None, None, None), Some(1), None)
}

fn point_schema() -> ParamSchema {
    obj(
        vec![
            ("x", num(None, None, None, None)),
            ("y", num(None, None, None, None)),
        ],
        vec!["x", "y"],
    )
}

fn string_pair_schema() -> ParamSchema {
    arr(string_field(), Some(2), Some(2))
}

fn weighted_edge_schema() -> ParamSchema {
    obj(
        vec![
            ("i", num(Some(0.0), None, Some(true), None)),
            ("j", num(Some(0.0), None, Some(true), None)),
            ("weight", num(Some(1e-12), None, None, None)),
        ],
        vec!["i", "j", "weight"],
    )
}

fn portfolio_asset_schema() -> ParamSchema {
    obj(
        vec![
            ("name", string_field()),
            ("expectedReturn", num(None, None, None, None)),
            ("risk", num(Some(0.0), None, None, None)),
        ],
        vec!["name", "expectedReturn", "risk"],
    )
}

// =============================================================================
// 1. particle-swarm
// =============================================================================

pub struct ParticleSwarmAdapter;

pub fn particle_swarm_adapter() -> ParticleSwarmAdapter {
    ParticleSwarmAdapter
}

impl DESModelRegistration<ParticleSwarmParams, ParticleSwarmResult> for ParticleSwarmAdapter {
    fn id(&self) -> &str {
        "particle-swarm"
    }
    fn description(&self) -> &str {
        "Particle Swarm Optimization using a shared numeric-swarm station and particle movables."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                (
                    "objective",
                    str_enum(&["sphere", "rastrigin", "rosenbrock"], Some("sphere")),
                ),
                ("dimension", num(Some(1.0), None, Some(true), Some(3.0))),
                ("particles", num(Some(1.0), None, Some(true), Some(32.0))),
                ("iterations", num(Some(1.0), None, Some(true), Some(120.0))),
                ("lower", num(None, None, None, Some(-5.0))),
                ("upper", num(None, None, None, Some(5.0))),
                ("inertia", num(Some(0.0), None, None, Some(0.68))),
                ("cognitive", num(Some(0.0), None, None, Some(1.45))),
                ("social", num(Some(0.0), None, None, Some(1.45))),
                ("seed", num(None, None, Some(true), Some(11.0))),
            ],
            vec![],
        )
    }
    fn run(&self, params: ParticleSwarmParams, _runtime: &DESRuntimeConfig) -> ParticleSwarmResult {
        run_particle_swarm(params)
    }
    fn summarize(&self, result: &ParticleSwarmResult, _params: &ParticleSwarmParams) -> String {
        [
            "PARTICLE SWARM OPTIMIZATION (DES)".to_string(),
            "---------------------------------".to_string(),
            format!("  Best value:     {}", to_exponential(result.best_value, 4)),
            format!(
                "  Best position:  [{}]",
                result
                    .best_position
                    .iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!("  Iterations:     {}", result.iterations),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &ParticleSwarmResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "iteration",
            "best_value",
            "mean_value",
            "worst_value",
        ])];
        for row in &result.trace {
            lines.push(csv_row([
                row.iteration.to_string(),
                js_number(row.best_value),
                js_number(row.mean_value),
                js_number(row.worst_value),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 2. ant-colony-tsp
// =============================================================================

pub struct AntColonyTspAdapter;

pub fn ant_colony_tsp_adapter() -> AntColonyTspAdapter {
    AntColonyTspAdapter
}

impl DESModelRegistration<AntColonyTSPParams, AntColonyTSPResult> for AntColonyTspAdapter {
    fn id(&self) -> &str {
        "ant-colony-tsp"
    }
    fn description(&self) -> &str {
        "Ant Colony Optimization on TSP using pheromone graph-search stations and walk tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("points", arr(point_schema(), Some(2), None)),
                ("ants", num(Some(1.0), None, Some(true), Some(18.0))),
                ("iterations", num(Some(1.0), None, Some(true), Some(80.0))),
                ("alpha", num(Some(0.0), None, None, Some(1.0))),
                ("beta", num(Some(0.0), None, None, Some(3.0))),
                ("evaporation", num(Some(0.0), Some(1.0), None, Some(0.28))),
                ("deposit", num(Some(1e-12), None, None, Some(1.0))),
                ("seed", num(None, None, Some(true), Some(5.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        mut params: AntColonyTSPParams,
        _runtime: &DESRuntimeConfig,
    ) -> AntColonyTSPResult {
        params.points = params.points.filter(|p| !p.is_empty());
        run_ant_colony_tsp(params)
    }
    fn summarize(&self, result: &AntColonyTSPResult, _params: &AntColonyTSPParams) -> String {
        [
            "ANT COLONY TSP (DES)".to_string(),
            "--------------------".to_string(),
            format!("  Best length:    {:.6}", result.best_length),
            format!(
                "  Best tour:      {}",
                result
                    .best_tour
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            ),
            format!("  Iterations:     {}", result.iterations),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &AntColonyTSPResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "iteration",
            "best_length",
            "mean_length",
            "worst_length",
        ])];
        for row in &result.trace {
            lines.push(csv_row([
                row.iteration.to_string(),
                js_number(row.best_length),
                js_number(row.mean_length),
                js_number(row.worst_length),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 3. map-coloring-csp
// =============================================================================

pub struct MapColoringCspAdapter;

pub fn map_coloring_csp_adapter() -> MapColoringCspAdapter {
    MapColoringCspAdapter
}

impl DESModelRegistration<MapColoringCSPParams, MapColoringCSPResult> for MapColoringCspAdapter {
    fn id(&self) -> &str {
        "map-coloring-csp"
    }
    fn description(&self) -> &str {
        "Constraint Satisfaction Problem solved by shared MRV backtracking tree-search station."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("variables", arr(string_field(), Some(1), None)),
                ("colors", arr(string_field(), Some(1), None)),
                ("edges", arr(string_pair_schema(), Some(1), None)),
                ("maxNodes", num(Some(1.0), None, Some(true), Some(10000.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        mut params: MapColoringCSPParams,
        _runtime: &DESRuntimeConfig,
    ) -> MapColoringCSPResult {
        params.variables = params.variables.filter(|v| !v.is_empty());
        params.colors = params.colors.filter(|c| !c.is_empty());
        params.edges = params.edges.filter(|e| !e.is_empty());
        run_map_coloring_csp(params)
    }
    fn summarize(&self, result: &MapColoringCSPResult, _params: &MapColoringCSPParams) -> String {
        let mut keys: Vec<&String> = result.assignment.keys().collect();
        keys.sort();
        let assignment = keys
            .iter()
            .map(|k| format!("{}={}", k, result.assignment[*k]))
            .collect::<Vec<_>>()
            .join(", ");
        [
            "MAP COLORING CSP (DES)".to_string(),
            "----------------------".to_string(),
            format!(
                "  Satisfied:      {}",
                if result.satisfied { "yes" } else { "no" }
            ),
            format!("  Assignment:     {assignment}"),
            format!("  Nodes:          {}", result.nodes_processed),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
}

// =============================================================================
// 4. max-sat-local-search
// =============================================================================

pub struct MaxSatLocalSearchAdapter;

pub fn max_sat_local_search_adapter() -> MaxSatLocalSearchAdapter {
    MaxSatLocalSearchAdapter
}

impl DESModelRegistration<MaxSATParams, MaxSATResult> for MaxSatLocalSearchAdapter {
    fn id(&self) -> &str {
        "max-sat-local-search"
    }
    fn description(&self) -> &str {
        "SAT/MAX-SAT local search using the shared single-state optimizer station."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("numVars", num(Some(1.0), None, Some(true), None)),
                ("clauses", arr(number_vector_schema(), Some(1), None)),
                ("iterations", num(Some(1.0), None, Some(true), Some(300.0))),
                ("noise", num(Some(0.0), Some(1.0), None, Some(0.25))),
                ("seed", num(None, None, Some(true), Some(13.0))),
            ],
            vec![],
        )
    }
    fn run(&self, mut params: MaxSATParams, _runtime: &DESRuntimeConfig) -> MaxSATResult {
        params.clauses = params.clauses.filter(|c| !c.is_empty());
        run_max_sat_local_search(params)
    }
    fn summarize(&self, result: &MaxSATResult, _params: &MaxSATParams) -> String {
        [
            "MAX-SAT LOCAL SEARCH (DES)".to_string(),
            "--------------------------".to_string(),
            format!(
                "  Satisfied:      {}/{}",
                result.satisfied_clauses, result.total_clauses
            ),
            format!(
                "  Complete SAT:   {}",
                if result.all_satisfied { "yes" } else { "no" }
            ),
            format!("  Iterations:     {}", result.iterations),
            format!(
                "  Assignment:     [{}]",
                result
                    .assignment
                    .iter()
                    .map(|v| if *v { "T" } else { "F" })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &MaxSATResult, csv_path: &str) {
        let mut lines = vec![csv_row(["iteration", "unsatisfied"])];
        for row in &result.trace {
            lines.push(csv_row([
                row.iteration.to_string(),
                js_number(row.unsatisfied),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 5. sdp-maxcut-relaxation
// =============================================================================

pub struct SdpMaxcutRelaxationAdapter;

pub fn sdp_maxcut_relaxation_adapter() -> SdpMaxcutRelaxationAdapter {
    SdpMaxcutRelaxationAdapter
}

impl DESModelRegistration<SDPMaxCutParams, SDPMaxCutResult> for SdpMaxcutRelaxationAdapter {
    fn id(&self) -> &str {
        "sdp-maxcut-relaxation"
    }
    fn description(&self) -> &str {
        "Semidefinite Max-Cut relaxation through rank-constrained unit-vector station updates."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("nodes", num(Some(2.0), None, Some(true), Some(5.0))),
                ("edges", arr(weighted_edge_schema(), Some(1), None)),
                ("rank", num(Some(1.0), None, Some(true), Some(3.0))),
                ("iterations", num(Some(1.0), None, Some(true), Some(250.0))),
                ("stepSize", num(Some(1e-12), None, None, Some(0.08))),
                ("seed", num(None, None, Some(true), Some(17.0))),
            ],
            vec![],
        )
    }
    fn run(&self, mut params: SDPMaxCutParams, _runtime: &DESRuntimeConfig) -> SDPMaxCutResult {
        params.edges = params.edges.filter(|e| !e.is_empty());
        run_sdp_max_cut_relaxation(params)
    }
    fn summarize(&self, result: &SDPMaxCutResult, _params: &SDPMaxCutParams) -> String {
        [
            "SDP MAX-CUT RELAXATION (DES)".to_string(),
            "----------------------------".to_string(),
            format!("  SDP value:      {:.6}", result.sdp_value),
            format!("  Rounded cut:    {:.6}", result.rounded_cut_value),
            format!(
                "  Cut signs:      [{}]",
                result
                    .cut
                    .iter()
                    .map(|v| js_number(*v))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!("  Iterations:     {}", result.iterations),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &SDPMaxCutResult, csv_path: &str) {
        let mut lines = vec![csv_row(["iteration", "objective"])];
        for row in &result.trace {
            lines.push(csv_row([
                row.iteration.to_string(),
                js_number(row.objective),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 6. pareto-portfolio
// =============================================================================

pub struct ParetoPortfolioAdapter;

pub fn pareto_portfolio_adapter() -> ParetoPortfolioAdapter {
    ParetoPortfolioAdapter
}

impl DESModelRegistration<ParetoPortfolioParams, ParetoPortfolioResult> for ParetoPortfolioAdapter {
    fn id(&self) -> &str {
        "pareto-portfolio"
    }
    fn description(&self) -> &str {
        "Multi-objective risk/return optimization with a reusable Pareto archive station."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("assets", arr(portfolio_asset_schema(), Some(1), None)),
                ("samples", num(Some(1.0), None, Some(true), Some(240.0))),
                ("seed", num(None, None, Some(true), Some(19.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        mut params: ParetoPortfolioParams,
        _runtime: &DESRuntimeConfig,
    ) -> ParetoPortfolioResult {
        params.assets = params.assets.filter(|a| !a.is_empty());
        run_pareto_portfolio(params)
    }
    fn summarize(&self, result: &ParetoPortfolioResult, _params: &ParetoPortfolioParams) -> String {
        [
            "PARETO PORTFOLIO (DES)".to_string(),
            "----------------------".to_string(),
            format!("  Candidates:     {}", result.candidate_count),
            format!("  Pareto points:  {}", result.pareto_front.len()),
            format!(
                "  Hypervolume:    {}",
                to_exponential(result.hypervolume, 4)
            ),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &ParetoPortfolioResult, csv_path: &str) {
        let mut lines = vec![csv_row(["risk", "expected_return", "weights"])];
        for point in &result.pareto_front {
            lines.push(csv_row([
                js_number(point.risk),
                js_number(point.expected_return),
                json_num_array(&point.weights),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 7. hinfinity-robust-control
// =============================================================================

pub struct HInfinityRobustControlAdapter;

pub fn hinfinity_robust_control_adapter() -> HInfinityRobustControlAdapter {
    HInfinityRobustControlAdapter
}

impl DESModelRegistration<HInfinityRobustControlParams, HInfinityRobustControlResult>
    for HInfinityRobustControlAdapter
{
    fn id(&self) -> &str {
        "hinfinity-robust-control"
    }
    fn description(&self) -> &str {
        "H-infinity-style robust control against a worst-case bounded disturbance station."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("x0", num(None, None, None, Some(2.0))),
                ("a", num(None, None, None, Some(0.25))),
                ("b", num(None, None, None, Some(1.0))),
                ("gain", num(Some(1e-12), None, None, Some(3.2))),
                ("disturbanceMax", num(Some(0.0), None, None, Some(0.45))),
                ("controlMax", num(Some(1e-12), None, None, Some(5.0))),
                ("gamma", num(Some(1e-12), None, None, Some(2.5))),
                ("dt", num(Some(1e-12), None, None, Some(0.03))),
                ("numSteps", num(Some(1.0), None, Some(true), Some(260.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: HInfinityRobustControlParams,
        _runtime: &DESRuntimeConfig,
    ) -> HInfinityRobustControlResult {
        run_h_infinity_robust_control(params).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(
        &self,
        result: &HInfinityRobustControlResult,
        _params: &HInfinityRobustControlParams,
    ) -> String {
        [
            "H-INFINITY ROBUST CONTROL (DES)".to_string(),
            "-------------------------------".to_string(),
            format!("  Final state:    {:.6}", result.final_state),
            format!("  Peak |state|:   {:.6}", result.peak_abs_state),
            format!(
                "  L2 gain est.:   {:.6} <= gamma {}",
                result.l2_gain_estimate,
                js_number(result.gamma)
            ),
            format!(
                "  Bounded:        {}",
                if result.bounded_by_gamma { "yes" } else { "no" }
            ),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &HInfinityRobustControlResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "tick",
            "time",
            "state",
            "control",
            "disturbance",
            "cost",
        ])];
        for row in &result.trace {
            lines.push(csv_row([
                row.tick.to_string(),
                js_number(row.time),
                js_number(row.state[0]),
                js_number(row.control[0]),
                js_number(row.disturbance[0]),
                js_number(row.cost),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// 8. pursuit-evasion-game
// =============================================================================

pub struct PursuitEvasionGameAdapter;

pub fn pursuit_evasion_game_adapter() -> PursuitEvasionGameAdapter {
    PursuitEvasionGameAdapter
}

impl DESModelRegistration<PursuitEvasionGameParams, PursuitEvasionGameResult>
    for PursuitEvasionGameAdapter
{
    fn id(&self) -> &str {
        "pursuit-evasion-game"
    }
    fn description(&self) -> &str {
        "Differential game: pursuit/evasion as plant, pursuer policy, and evader policy stations."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                (
                    "pursuer",
                    arr(num(None, None, None, None), Some(2), Some(2)),
                ),
                ("evader", arr(num(None, None, None, None), Some(2), Some(2))),
                ("pursuerSpeed", num(Some(1e-12), None, None, Some(1.25))),
                ("evaderSpeed", num(Some(0.0), None, None, Some(0.6))),
                ("captureRadius", num(Some(1e-12), None, None, Some(0.25))),
                ("dt", num(Some(1e-12), None, None, Some(0.1))),
                ("numSteps", num(Some(1.0), None, Some(true), Some(120.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: PursuitEvasionGameParams,
        _runtime: &DESRuntimeConfig,
    ) -> PursuitEvasionGameResult {
        run_pursuit_evasion_game(params).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(
        &self,
        result: &PursuitEvasionGameResult,
        _params: &PursuitEvasionGameParams,
    ) -> String {
        let capture = match result.capture_tick {
            None => "not captured".to_string(),
            Some(t) => t.to_string(),
        };
        [
            "PURSUIT/EVASION DIFFERENTIAL GAME (DES)".to_string(),
            "---------------------------------------".to_string(),
            format!("  Capture tick:   {capture}"),
            format!("  Final distance: {:.6}", result.final_distance),
            format!("  Steps recorded: {}", result.distance_history.len()),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &PursuitEvasionGameResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "tick", "time", "px", "py", "ex", "ey", "ux", "uy", "wx", "wy", "distance",
        ])];
        for row in &result.trace {
            let s = &row.state;
            lines.push(csv_row([
                row.tick.to_string(),
                js_number(row.time),
                js_number(s[0]),
                js_number(s[1]),
                js_number(s[2]),
                js_number(s[3]),
                js_number(row.control[0]),
                js_number(row.control[1]),
                js_number(row.disturbance[0]),
                js_number(row.disturbance[1]),
                js_number((s[2] - s[0]).hypot(s[3] - s[1])),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}
