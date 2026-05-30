//! Port of `src/des/general/adapters/classical-optimization-adapter.ts`
//! (module `des::general::adapters::classical_optimization_adapter`).
//!
//! Registers QP / assignment / VRP / job-shop / flow-shop JSON adapters
//! (8 models) over classical-optimization station graphs.
//!
//! ## Conversion notes
//!
//!   * `Q` / `cost` `number[][]` matrices -> `Vec<Vec<f64>>`.
//!   * `rule: 'fifo'|'spt'|'edd'` -> [`DispatchRule`].
//!   * `AssignmentParams & {epsilon?; maxIter?}` for the auction model maps to
//!     the engine's dedicated [`AuctionAssignmentParams`].
//!   * `result.assignment.forEach((job, worker) => ...)` -> `enumerate()` over
//!     `assignment[worker] = job` (here `job` is an `i64`, `-1` if unassigned).
//!   * `JSON.stringify(row.x)` in the QP CSV -> [`json_num_array`].
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; each adapter is
//! exposed via the `adapter_*()` constructors.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::classical_optimization_models::{
    run_auction_assignment, run_flow_shop_neh, run_hungarian_assignment, run_job_shop_dispatch,
    run_qp_coordinate_descent, run_qp_projected_gradient, run_vrp_nearest_neighbor, run_vrp_savings,
    AssignmentParams, AssignmentResult, AuctionAssignmentParams, DispatchRule, FlowShopNEHParams,
    FlowShopNEHResult, JobShopDispatchParams, JobShopDispatchResult, QPProjectedGradientParams,
    QPProjectedGradientResult, VRPSavingsParams, VRPSavingsResult,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};

// =============================================================================
// Formatting helpers (JS parity).
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

/// `JSON.stringify(numbers)` — a JSON array of numbers (`NaN`/`Infinity` -> `null`).
fn json_num_array(values: &[f64]) -> String {
    let inner: Vec<String> = values
        .iter()
        .map(|v| if v.is_finite() { js_number(*v) } else { "null".to_string() })
        .collect();
    format!("[{}]", inner.join(","))
}

/// `numbers.map(v => v.toFixed(n)).join(', ')`.
fn fixed_join(values: &[f64], digits: usize) -> String {
    values.iter().map(|v| format!("{:.*}", digits, v)).collect::<Vec<_>>().join(", ")
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn number_vector_schema() -> ParamSchema {
    arr(num(None, None, None, None), Some(1))
}

fn number_matrix_schema() -> ParamSchema {
    arr(number_vector_schema(), Some(1))
}

fn customer_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("x", num(None, None, None, None)),
            ("y", num(None, None, None, None)),
            ("demand", num(Some(0.0), None, None, None)),
        ],
        vec!["id", "x", "y", "demand"],
    )
}

fn operation_schema() -> ParamSchema {
    obj(
        vec![("machine", string_field()), ("duration", num(Some(0.0), None, None, None))],
        vec!["machine", "duration"],
    )
}

fn job_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("due", num(None, None, None, None)),
            ("operations", arr(operation_schema(), Some(1))),
        ],
        vec!["id", "operations"],
    )
}

fn flow_shop_job_schema() -> ParamSchema {
    obj(
        vec![
            ("id", string_field()),
            ("processingTimes", arr(num(Some(0.0), None, None, None), Some(1))),
            ("due", num(None, None, None, None)),
        ],
        vec!["id", "processingTimes"],
    )
}

// =============================================================================
// Generic example helper
// =============================================================================

fn example<P>(name: &str, model: &str, description: &str, parameters: P) -> RegistrationExample<P> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: model.to_string(),
            description: Some(description.to_string()),
            parameters,
            runtime: None,
            metadata: None,
        },
    }
}

// Shared CSV emitter for the two QP variants.
fn qp_write_csv(result: &QPProjectedGradientResult, csv_path: &str) {
    let mut lines = vec![csv_row(["iter", "objective", "gradient_norm", "x"])];
    for row in &result.trace {
        lines.push(csv_row([
            row.iter.to_string(),
            js_number(row.objective),
            js_number(row.gradient_norm),
            json_num_array(&row.x),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// Shared CSV emitter for the two assignment variants.
fn assignment_write_csv(result: &AssignmentResult, csv_path: &str) {
    let mut lines = vec![csv_row(["worker", "job", "objective"])];
    for (worker, job) in result.assignment.iter().enumerate() {
        lines.push(csv_row([worker.to_string(), job.to_string(), js_number(result.objective)]));
    }
    write_csv_lines(csv_path, &lines);
}

// Shared CSV emitter for the two VRP variants.
fn vrp_write_csv(result: &VRPSavingsResult, csv_path: &str) {
    let mut lines = vec![csv_row(["route", "customers", "load", "distance"])];
    for (i, r) in result.routes.iter().enumerate() {
        lines.push(csv_row([
            i.to_string(),
            r.customers.join("|"),
            js_number(r.load),
            js_number(r.distance),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// Shared CSV emitter for the scheduling variants.
fn schedule_write_csv(schedule: &[crate::des::general::classical_optimization_models::ScheduledOperation], csv_path: &str) {
    let mut lines = vec![csv_row(["job", "operation", "machine", "start", "finish"])];
    for op in schedule {
        lines.push(csv_row([
            op.job_id.clone(),
            op.op_index.to_string(),
            op.machine.clone(),
            js_number(op.start),
            js_number(op.finish),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// =============================================================================
// qp-projected-gradient
// =============================================================================

pub struct QPProjectedGradientAdapter;
pub fn adapter_qp_projected_gradient() -> QPProjectedGradientAdapter {
    QPProjectedGradientAdapter
}

impl DESModelRegistration<QPProjectedGradientParams, QPProjectedGradientResult>
    for QPProjectedGradientAdapter
{
    fn id(&self) -> &str {
        "qp-projected-gradient"
    }
    fn description(&self) -> &str {
        "Box-constrained quadratic programming via projected-gradient state tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("Q", number_matrix_schema()),
                ("c", number_vector_schema()),
                ("lower", number_vector_schema()),
                ("upper", number_vector_schema()),
                ("x0", number_vector_schema()),
                ("stepSize", num(Some(1e-12), None, None, Some(0.12))),
                ("maxIter", num(Some(1.0), None, Some(true), Some(200.0))),
                ("tol", num(Some(0.0), None, None, Some(1e-8))),
            ],
            vec![],
        )
    }
    fn run(&self, params: QPProjectedGradientParams, _runtime: &DESRuntimeConfig) -> QPProjectedGradientResult {
        run_qp_projected_gradient(params)
    }
    fn summarize(&self, result: &QPProjectedGradientResult, _params: &QPProjectedGradientParams) -> String {
        [
            "QP PROJECTED GRADIENT (DES)".to_string(),
            "---------------------------".to_string(),
            format!("  Objective:      {:.8}", result.objective),
            format!("  x*:             [{}]", fixed_join(&result.x, 6)),
            format!("  Iterations:     {}", result.iterations),
            format!("  Gradient norm:  {}", to_exponential(result.gradient_norm, 3)),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &QPProjectedGradientResult, csv_path: &str) {
        qp_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<QPProjectedGradientParams>> {
        vec![example(
            "box-constrained-q2",
            "qp-projected-gradient",
            "Small box-constrained quadratic program solved by movable state tokens.",
            QPProjectedGradientParams::default(),
        )]
    }
}

// =============================================================================
// qp-coordinate-descent
// =============================================================================

pub struct QPCoordinateDescentAdapter;
pub fn adapter_qp_coordinate_descent() -> QPCoordinateDescentAdapter {
    QPCoordinateDescentAdapter
}

impl DESModelRegistration<QPProjectedGradientParams, QPProjectedGradientResult>
    for QPCoordinateDescentAdapter
{
    fn id(&self) -> &str {
        "qp-coordinate-descent"
    }
    fn description(&self) -> &str {
        "Box-constrained quadratic programming via coordinate-descent state tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("Q", number_matrix_schema()),
                ("c", number_vector_schema()),
                ("lower", number_vector_schema()),
                ("upper", number_vector_schema()),
                ("x0", number_vector_schema()),
                ("maxIter", num(Some(1.0), None, Some(true), Some(100.0))),
                ("tol", num(Some(0.0), None, None, Some(1e-8))),
            ],
            vec![],
        )
    }
    fn run(&self, params: QPProjectedGradientParams, _runtime: &DESRuntimeConfig) -> QPProjectedGradientResult {
        run_qp_coordinate_descent(params)
    }
    fn summarize(&self, result: &QPProjectedGradientResult, _params: &QPProjectedGradientParams) -> String {
        [
            "QP COORDINATE DESCENT (DES)".to_string(),
            "---------------------------".to_string(),
            format!("  Objective:      {:.8}", result.objective),
            format!("  x*:             [{}]", fixed_join(&result.x, 6)),
            format!("  Iterations:     {}", result.iterations),
            format!("  Gradient norm:  {}", to_exponential(result.gradient_norm, 3)),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &QPProjectedGradientResult, csv_path: &str) {
        qp_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<QPProjectedGradientParams>> {
        vec![example(
            "box-constrained-coordinate",
            "qp-coordinate-descent",
            "Small box-constrained quadratic program solved by coordinate-descent state tokens.",
            QPProjectedGradientParams::default(),
        )]
    }
}

// =============================================================================
// hungarian-assignment
// =============================================================================

pub struct HungarianAssignmentAdapter;
pub fn adapter_hungarian_assignment() -> HungarianAssignmentAdapter {
    HungarianAssignmentAdapter
}

impl DESModelRegistration<AssignmentParams, AssignmentResult> for HungarianAssignmentAdapter {
    fn id(&self) -> &str {
        "hungarian-assignment"
    }
    fn description(&self) -> &str {
        "Assignment problem with row/column reduction stations and assignment-result tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(vec![("cost", number_matrix_schema())], vec![])
    }
    fn run(&self, params: AssignmentParams, _runtime: &DESRuntimeConfig) -> AssignmentResult {
        run_hungarian_assignment(params)
    }
    fn summarize(&self, result: &AssignmentResult, _params: &AssignmentParams) -> String {
        [
            "HUNGARIAN-STYLE ASSIGNMENT (DES)".to_string(),
            "--------------------------------".to_string(),
            format!("  Objective:      {:.6}", result.objective),
            format!(
                "  Assignment:     [{}]",
                result.assignment.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
            ),
            format!("  Row reductions: [{}]", fixed_join(&result.row_reductions, 2)),
            format!("  Col reductions: [{}]", fixed_join(&result.col_reductions, 2)),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &AssignmentResult, csv_path: &str) {
        assignment_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<AssignmentParams>> {
        vec![example(
            "three-by-three",
            "hungarian-assignment",
            "3x3 assignment through row reduction, column reduction, and assignment builder stations.",
            AssignmentParams::default(),
        )]
    }
}

// =============================================================================
// auction-assignment
// =============================================================================

pub struct AuctionAssignmentAdapter;
pub fn adapter_auction_assignment() -> AuctionAssignmentAdapter {
    AuctionAssignmentAdapter
}

impl DESModelRegistration<AuctionAssignmentParams, AssignmentResult> for AuctionAssignmentAdapter {
    fn id(&self) -> &str {
        "auction-assignment"
    }
    fn description(&self) -> &str {
        "Assignment problem using movable auction price/assignment state tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("cost", number_matrix_schema()),
                ("epsilon", num(Some(1e-12), None, None, Some(0.01))),
                ("maxIter", num(Some(1.0), None, Some(true), None)),
            ],
            vec![],
        )
    }
    fn run(&self, params: AuctionAssignmentParams, _runtime: &DESRuntimeConfig) -> AssignmentResult {
        run_auction_assignment(params)
    }
    fn summarize(&self, result: &AssignmentResult, _params: &AuctionAssignmentParams) -> String {
        [
            "AUCTION ASSIGNMENT (DES)".to_string(),
            "------------------------".to_string(),
            format!("  Objective:      {:.6}", result.objective),
            format!(
                "  Assignment:     [{}]",
                result.assignment.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
            ),
            format!("  Price vector:   [{}]", fixed_join(&result.col_reductions, 3)),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &AssignmentResult, csv_path: &str) {
        assignment_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<AuctionAssignmentParams>> {
        vec![example(
            "three-by-three-auction",
            "auction-assignment",
            "3x3 assignment through movable auction state tokens.",
            AuctionAssignmentParams { epsilon: Some(0.01), ..Default::default() },
        )]
    }
}

// =============================================================================
// vrp-savings
// =============================================================================

pub struct VRPSavingsAdapter;
pub fn adapter_vrp_savings() -> VRPSavingsAdapter {
    VRPSavingsAdapter
}

fn vrp_schema() -> ParamSchema {
    obj(
        vec![
            (
                "depot",
                obj(vec![("x", num(None, None, None, None)), ("y", num(None, None, None, None))], vec!["x", "y"]),
            ),
            ("customers", arr(customer_schema(), Some(1))),
            ("vehicleCapacity", num(Some(1e-12), None, None, Some(5.0))),
        ],
        vec![],
    )
}

impl DESModelRegistration<VRPSavingsParams, VRPSavingsResult> for VRPSavingsAdapter {
    fn id(&self) -> &str {
        "vrp-savings"
    }
    fn description(&self) -> &str {
        "Capacitated vehicle routing with Clarke-Wright savings and route-merge tokens."
    }
    fn schema(&self) -> ParamSchema {
        vrp_schema()
    }
    fn run(&self, params: VRPSavingsParams, _runtime: &DESRuntimeConfig) -> VRPSavingsResult {
        run_vrp_savings(params)
    }
    fn summarize(&self, result: &VRPSavingsResult, _params: &VRPSavingsParams) -> String {
        [
            "VRP SAVINGS HEURISTIC (DES)".to_string(),
            "---------------------------".to_string(),
            format!("  Routes:         {}", result.routes.len()),
            format!("  Total distance: {:.6}", result.total_distance),
            format!("  Savings pairs:  {}", result.savings_considered),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &VRPSavingsResult, csv_path: &str) {
        vrp_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<VRPSavingsParams>> {
        vec![example(
            "small-cvrp",
            "vrp-savings",
            "Small capacitated VRP using savings and route-merge stations.",
            VRPSavingsParams { vehicle_capacity: Some(5.0), ..Default::default() },
        )]
    }
}

// =============================================================================
// vrp-nearest-neighbor
// =============================================================================

pub struct VRPNearestNeighborAdapter;
pub fn adapter_vrp_nearest_neighbor() -> VRPNearestNeighborAdapter {
    VRPNearestNeighborAdapter
}

impl DESModelRegistration<VRPSavingsParams, VRPSavingsResult> for VRPNearestNeighborAdapter {
    fn id(&self) -> &str {
        "vrp-nearest-neighbor"
    }
    fn description(&self) -> &str {
        "Capacitated vehicle routing with nearest-neighbor route-construction tokens."
    }
    fn schema(&self) -> ParamSchema {
        vrp_schema()
    }
    fn run(&self, params: VRPSavingsParams, _runtime: &DESRuntimeConfig) -> VRPSavingsResult {
        run_vrp_nearest_neighbor(params)
    }
    fn summarize(&self, result: &VRPSavingsResult, _params: &VRPSavingsParams) -> String {
        [
            "VRP NEAREST NEIGHBOR (DES)".to_string(),
            "--------------------------".to_string(),
            format!("  Routes:         {}", result.routes.len()),
            format!("  Total distance: {:.6}", result.total_distance),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &VRPSavingsResult, csv_path: &str) {
        vrp_write_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<VRPSavingsParams>> {
        vec![example(
            "small-cvrp-nearest",
            "vrp-nearest-neighbor",
            "Small capacitated VRP using a nearest-neighbor route-construction station.",
            VRPSavingsParams { vehicle_capacity: Some(5.0), ..Default::default() },
        )]
    }
}

// =============================================================================
// job-shop-dispatch
// =============================================================================

pub struct JobShopDispatchAdapter;
pub fn adapter_job_shop_dispatch() -> JobShopDispatchAdapter {
    JobShopDispatchAdapter
}

impl DESModelRegistration<JobShopDispatchParams, JobShopDispatchResult> for JobShopDispatchAdapter {
    fn id(&self) -> &str {
        "job-shop-dispatch"
    }
    fn description(&self) -> &str {
        "Job-shop scheduling via job tokens and a dispatch-rule scheduler station."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("jobs", arr(job_schema(), Some(1))),
                ("rule", str_enum(&["fifo", "spt", "edd"], "spt")),
            ],
            vec![],
        )
    }
    fn run(&self, params: JobShopDispatchParams, _runtime: &DESRuntimeConfig) -> JobShopDispatchResult {
        run_job_shop_dispatch(params)
    }
    fn summarize(&self, result: &JobShopDispatchResult, _params: &JobShopDispatchParams) -> String {
        [
            "JOB-SHOP DISPATCH (DES)".to_string(),
            "-----------------------".to_string(),
            format!("  Operations:     {}", result.schedule.len()),
            format!("  Makespan:       {:.3}", result.makespan),
            format!("  Total flow:     {:.3}", result.total_flow_time),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &JobShopDispatchResult, csv_path: &str) {
        schedule_write_csv(&result.schedule, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<JobShopDispatchParams>> {
        vec![example(
            "three-job-spt",
            "job-shop-dispatch",
            "Three-job two-machine schedule using a shortest-processing-time dispatch station.",
            JobShopDispatchParams { rule: Some(DispatchRule::Spt), ..Default::default() },
        )]
    }
}

// =============================================================================
// flow-shop-neh
// =============================================================================

pub struct FlowShopNEHAdapter;
pub fn adapter_flow_shop_neh() -> FlowShopNEHAdapter {
    FlowShopNEHAdapter
}

impl DESModelRegistration<FlowShopNEHParams, FlowShopNEHResult> for FlowShopNEHAdapter {
    fn id(&self) -> &str {
        "flow-shop-neh"
    }
    fn description(&self) -> &str {
        "Flow-shop scheduling with NEH sequence tokens and a schedule-builder station."
    }
    fn schema(&self) -> ParamSchema {
        obj(vec![("jobs", arr(flow_shop_job_schema(), Some(1)))], vec![])
    }
    fn run(&self, params: FlowShopNEHParams, _runtime: &DESRuntimeConfig) -> FlowShopNEHResult {
        run_flow_shop_neh(params)
    }
    fn summarize(&self, result: &FlowShopNEHResult, _params: &FlowShopNEHParams) -> String {
        [
            "FLOW-SHOP NEH (DES)".to_string(),
            "-------------------".to_string(),
            format!("  Sequence:       {}", result.sequence.join(" -> ")),
            format!("  Operations:     {}", result.schedule.len()),
            format!("  Makespan:       {:.3}", result.makespan),
            format!("  Total flow:     {:.3}", result.total_flow_time),
            format!("  Stations:       {}", result.topology.stations.join(" -> ")),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &FlowShopNEHResult, csv_path: &str) {
        schedule_write_csv(&result.schedule, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<FlowShopNEHParams>> {
        vec![example(
            "four-job-flow-shop",
            "flow-shop-neh",
            "Four-job flow-shop schedule using NEH sequence and schedule stations.",
            FlowShopNEHParams::default(),
        )]
    }
}
