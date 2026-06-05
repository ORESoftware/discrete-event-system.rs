//! Rust-facing bridge for external/reference vehicle-routing solvers.
//!
//! The native Rust reference computes an exact small CVRP route-cover check
//! without Python startup. Explicit OR-Tools Routing validation is launched
//! from Rust through a tiny inline Python adapter, so the checked-in Python
//! script stays compatibility glue instead of carrying reusable modeling logic.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::classical_optimization_models::{
    run_vrp_exact, Point, VRPCustomer, VRPRoute, VRPSavingsParams,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalRoutingReferenceSolver {
    Auto,
    RustExact,
    OrTools,
    Fallback,
}

impl ExternalRoutingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalRoutingReferenceSolver::Auto => "auto",
            ExternalRoutingReferenceSolver::RustExact => "rust-exact",
            ExternalRoutingReferenceSolver::OrTools => "ortools",
            ExternalRoutingReferenceSolver::Fallback => "fallback",
        }
    }
}

fn registered_routing_rust_fallback_enabled() -> bool {
    [
        "ROUTING_REFERENCE_REGISTERED_FALLBACK",
        "ROUTING_REFERENCE_EXTERNAL_FALLBACK",
        "ROUTING_REFERENCE_RUST_FIRST",
        "ORES_EXTERNAL_REFERENCE_RUST_FIRST",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
                )
            })
            .unwrap_or(false)
    })
}

fn should_use_rust_routing_reference(opts: &ExternalRoutingReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalRoutingReferenceSolver::Auto
            | ExternalRoutingReferenceSolver::RustExact
            | ExternalRoutingReferenceSolver::Fallback
    )
}

fn should_use_registered_routing_fallback(opts: &ExternalRoutingReferenceOptions) -> bool {
    registered_routing_rust_fallback_enabled()
        && matches!(opts.solver, ExternalRoutingReferenceSolver::OrTools)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalRoutingReferenceOptions {
    pub solver: ExternalRoutingReferenceSolver,
}

impl Default for ExternalRoutingReferenceOptions {
    fn default() -> Self {
        ExternalRoutingReferenceOptions {
            solver: ExternalRoutingReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalRoutingReferenceStatus {
    Optimal,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalRoutingReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalRoutingReferenceStatus::Optimal => "optimal",
            ExternalRoutingReferenceStatus::Infeasible => "infeasible",
            ExternalRoutingReferenceStatus::Unsupported => "unsupported",
            ExternalRoutingReferenceStatus::NumericalError => "numerical-error",
            ExternalRoutingReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalRoutingReferenceSolution {
    pub status: ExternalRoutingReferenceStatus,
    pub solver: String,
    pub routes: Vec<VRPRoute>,
    pub objective: Option<f64>,
    pub feasible_route_masks: Option<usize>,
    pub ortools_status: Option<String>,
    pub ortools_routes: Vec<VRPRoute>,
    pub ortools_objective: Option<f64>,
    pub message: String,
    pub ortools_message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct RoutingReferencePayload {
    status: String,
    solver: Option<String>,
    routes: Option<Vec<RoutingReferenceRoutePayload>>,
    objective: Option<f64>,
    #[serde(rename = "feasibleRouteMasks")]
    feasible_route_masks: Option<usize>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsRoutes")]
    ortools_routes: Option<Vec<RoutingReferenceRoutePayload>>,
    message: Option<String>,
    #[serde(rename = "ortoolsMessage")]
    ortools_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RoutingReferenceRoutePayload {
    customers: Vec<String>,
    load: f64,
    distance: f64,
}

impl From<RoutingReferenceRoutePayload> for VRPRoute {
    fn from(value: RoutingReferenceRoutePayload) -> Self {
        VRPRoute {
            customers: value.customers,
            load: value.load,
            distance: value.distance,
        }
    }
}

fn status_from_str(status: &str) -> ExternalRoutingReferenceStatus {
    match status {
        "optimal" => ExternalRoutingReferenceStatus::Optimal,
        "infeasible" => ExternalRoutingReferenceStatus::Infeasible,
        "unsupported" => ExternalRoutingReferenceStatus::Unsupported,
        "unavailable" => ExternalRoutingReferenceStatus::Unavailable,
        _ => ExternalRoutingReferenceStatus::NumericalError,
    }
}

const RUST_CVRP_MAX_EXACT_CUSTOMERS: usize = 16;
const ORTOOLS_ROUTING_DISTANCE_SCALE: i64 = 1_000_000;
const ORTOOLS_ROUTING_DEMAND_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_ROUTING_ADAPTER: &str = r#"
import json
import math
import sys

SOLVER = "ortools:routing"

def emit(status, message, routes=None, objective=None, ortools_status=None):
    routes = routes or []
    print(json.dumps({
        "status": status,
        "solver": SOLVER,
        "routes": routes,
        "objective": objective,
        "ortoolsStatus": ortools_status,
        "ortoolsRoutes": routes,
        "ortoolsObjective": objective,
        "message": message,
        "ortoolsMessage": message,
    }))

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception as exc:
    emit("unavailable", f"OR-Tools Routing unavailable: {exc}", ortools_status="unavailable")
    raise SystemExit(0)

def distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

def route_distance(depot, route):
    if not route:
        return 0.0
    total = distance(depot, route[0])
    for left, right in zip(route, route[1:]):
        total += distance(left, right)
    return total + distance(route[-1], depot)

try:
    payload = json.load(sys.stdin)
    depot = payload["depot"]
    customers = payload.get("customers") or []
    scaled_capacity = int(payload["scaledCapacity"])
    distance_scale = int(payload.get("distanceScale", 1000000))
    n = len(customers)
    if n == 0:
        emit("optimal", "empty instance", routes=[], objective=0.0, ortools_status="optimal")
        raise SystemExit(0)

    points = [depot] + customers
    manager = pywrapcp.RoutingIndexManager(n + 1, n, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(distance(points[from_node], points[to_node]) * distance_scale))

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    scaled_demands = [0] + [int(customer["scaledDemand"]) for customer in customers]

    def demand_callback(index):
        return scaled_demands[manager.IndexToNode(index)]

    demand = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand,
        0,
        [scaled_capacity for _ in range(n)],
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(payload.get("timeLimitSeconds", 5)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        emit("infeasible", "OR-Tools Routing found no solution", ortools_status="infeasible")
        raise SystemExit(0)

    routes = []
    for vehicle in range(n):
        index = routing.Start(vehicle)
        ids = []
        route_customers = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                customer = customers[node - 1]
                ids.append(customer["id"])
                route_customers.append(customer)
            index = solution.Value(routing.NextVar(index))
        if ids:
            routes.append({
                "customers": ids,
                "load": sum(float(customer["demand"]) for customer in route_customers),
                "distance": route_distance(depot, route_customers),
            })
    routes.sort(key=lambda route: route["customers"])
    emit(
        "optimal",
        "OR-Tools Routing local-search solution",
        routes=routes,
        objective=sum(float(route["distance"]) for route in routes),
        ortools_status="optimal",
    )
except Exception as exc:
    emit("numerical-error", str(exc), routes=[], objective=None, ortools_status="error")
    raise SystemExit(1)
"#;

fn rust_routing_empty_solution(
    status: ExternalRoutingReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalRoutingReferenceSolution {
    ExternalRoutingReferenceSolution {
        status,
        solver: solver.into(),
        routes: Vec::new(),
        objective: None,
        feasible_route_masks: None,
        ortools_status: None,
        ortools_routes: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        ortools_message: String::new(),
        elapsed_ms,
    }
}

fn relabel_registered_routing_fallback(
    mut solution: ExternalRoutingReferenceSolution,
    opts: &ExternalRoutingReferenceOptions,
) -> ExternalRoutingReferenceSolution {
    if should_use_registered_routing_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-routing-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn validate_rust_cvrp_inputs(
    depot: Point,
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
) -> Result<(), String> {
    if !depot.x.is_finite() || !depot.y.is_finite() {
        return Err("depot coordinates must be finite".to_string());
    }
    if !vehicle_capacity.is_finite() || vehicle_capacity <= 0.0 {
        return Err("vehicle_capacity must be finite and positive".to_string());
    }
    let mut ids = std::collections::HashSet::new();
    for (index, customer) in customers.iter().enumerate() {
        if customer.id.trim().is_empty() {
            return Err(format!("customers[{index}].id must be non-empty"));
        }
        if !ids.insert(customer.id.clone()) {
            return Err(format!("duplicate customer id {:?}", customer.id));
        }
        if !customer.x.is_finite() || !customer.y.is_finite() {
            return Err(format!("customers[{index}] coordinates must be finite"));
        }
        if !customer.demand.is_finite() || customer.demand < 0.0 {
            return Err(format!(
                "customers[{index}].demand must be finite and non-negative"
            ));
        }
    }
    Ok(())
}

fn solve_cvrp_with_rust_reference(
    depot: Point,
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
) -> ExternalRoutingReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_cvrp_inputs(depot, customers, vehicle_capacity) {
        return rust_routing_empty_solution(
            ExternalRoutingReferenceStatus::NumericalError,
            "rust:exact-cvrp",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if customers.len() > RUST_CVRP_MAX_EXACT_CUSTOMERS {
        return rust_routing_empty_solution(
            ExternalRoutingReferenceStatus::Unsupported,
            "rust:exact-cvrp",
            format!(
                "exact CVRP only practical for n <= {RUST_CVRP_MAX_EXACT_CUSTOMERS}, got {}",
                customers.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if customers
        .iter()
        .any(|customer| customer.demand > vehicle_capacity + 1e-9)
    {
        return rust_routing_empty_solution(
            ExternalRoutingReferenceStatus::Infeasible,
            "rust:exact-cvrp",
            "customer demand exceeds vehicle capacity",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if customers.is_empty() {
        return ExternalRoutingReferenceSolution {
            status: ExternalRoutingReferenceStatus::Optimal,
            solver: "rust:exact-cvrp".to_string(),
            routes: Vec::new(),
            objective: Some(0.0),
            feasible_route_masks: Some(0),
            ortools_status: None,
            ortools_routes: Vec::new(),
            ortools_objective: None,
            message: "empty instance".to_string(),
            ortools_message: String::new(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let result = run_vrp_exact(VRPSavingsParams {
        depot: Some(depot),
        customers: Some(customers.to_vec()),
        vehicle_capacity: Some(vehicle_capacity),
    });
    ExternalRoutingReferenceSolution {
        status: ExternalRoutingReferenceStatus::Optimal,
        solver: "rust:exact-cvrp".to_string(),
        routes: result.routes,
        objective: Some(result.total_distance),
        feasible_route_masks: Some(result.savings_considered),
        ortools_status: None,
        ortools_routes: Vec::new(),
        ortools_objective: None,
        message: "exact CVRP route-cover dynamic program".to_string(),
        ortools_message: String::new(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalRoutingReferenceSolution {
    ExternalRoutingReferenceSolution {
        status: ExternalRoutingReferenceStatus::Unavailable,
        solver: "ortools:routing".to_string(),
        routes: Vec::new(),
        objective: None,
        feasible_route_masks: None,
        ortools_status: None,
        ortools_routes: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        ortools_message: String::new(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalRoutingReferenceSolution {
    ExternalRoutingReferenceSolution {
        status: ExternalRoutingReferenceStatus::NumericalError,
        solver: "ortools:routing".to_string(),
        routes: Vec::new(),
        objective: None,
        feasible_route_masks: None,
        ortools_status: None,
        ortools_routes: Vec::new(),
        ortools_objective: None,
        message: message.into(),
        ortools_message: String::new(),
        elapsed_ms,
    }
}

fn routing_reference_timeout_ms() -> u64 {
    std::env::var("ROUTING_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_routing_reference_output(
    mut child: std::process::Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let started = Instant::now();
    let timeout = Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if timeout_ms > 0 && started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(err) => return Err(format!("failed to poll OR-Tools Routing adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools Routing adapter: {err}"))
}

fn scaled_ortools_routing_value(value: f64, scale: i64, name: &str) -> Result<i64, String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled.abs() > i64::MAX as f64 {
        return Err(format!("{name} is too large for OR-Tools integer scaling"));
    }
    let rounded = scaled.round();
    if (scaled - rounded).abs() > 1e-6 {
        return Err(format!(
            "{name}={value} cannot be represented with OR-Tools demand scale {scale}"
        ));
    }
    Ok(rounded as i64)
}

fn choose_ortools_routing_demand_scale(
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
) -> Result<i64, String> {
    for scale in ORTOOLS_ROUTING_DEMAND_SCALES {
        if scaled_ortools_routing_value(vehicle_capacity, scale, "vehicle_capacity").is_ok()
            && customers.iter().enumerate().all(|(index, customer)| {
                scaled_ortools_routing_value(
                    customer.demand,
                    scale,
                    &format!("customers[{index}].demand"),
                )
                .is_ok()
            })
        {
            return Ok(scale);
        }
    }
    Err(format!(
        "OR-Tools Routing integer demand scaling supports at most {} decimal places",
        ORTOOLS_ROUTING_DEMAND_SCALES
            .last()
            .copied()
            .unwrap_or(1)
            .ilog10()
    ))
}

fn ortools_routing_payload(
    depot: Point,
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
) -> Result<Value, String> {
    validate_rust_cvrp_inputs(depot, customers, vehicle_capacity)?;
    let demand_scale = choose_ortools_routing_demand_scale(customers, vehicle_capacity)?;
    let scaled_capacity =
        scaled_ortools_routing_value(vehicle_capacity, demand_scale, "vehicle_capacity")?;
    let customers = customers
        .iter()
        .enumerate()
        .map(|(index, customer)| {
            Ok(json!({
                "id": &customer.id,
                "x": customer.x,
                "y": customer.y,
                "demand": customer.demand,
                "scaledDemand": scaled_ortools_routing_value(
                    customer.demand,
                    demand_scale,
                    &format!("customers[{index}].demand"),
                )?,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!({
        "depot": {
            "x": depot.x,
            "y": depot.y,
        },
        "customers": customers,
        "scaledCapacity": scaled_capacity,
        "demandScale": demand_scale,
        "distanceScale": ORTOOLS_ROUTING_DISTANCE_SCALE,
    }))
}

fn run_ortools_routing_reference(
    depot: Point,
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
) -> ExternalRoutingReferenceSolution {
    let started = Instant::now();
    let payload = match ortools_routing_payload(depot, customers, vehicle_capacity) {
        Ok(payload) => payload,
        Err(message) => {
            return rust_routing_empty_solution(
                ExternalRoutingReferenceStatus::NumericalError,
                "ortools:routing",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_ROUTING_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start OR-Tools Routing adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write OR-Tools Routing adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = routing_reference_timeout_ms();
    let (output, timed_out) = match wait_for_routing_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools Routing adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools Routing adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<RoutingReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalRoutingReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-routing-reference".to_string()),
            routes: parsed
                .routes
                .unwrap_or_default()
                .into_iter()
                .map(VRPRoute::from)
                .collect(),
            objective: parsed.objective,
            feasible_route_masks: parsed.feasible_route_masks,
            ortools_status: parsed.ortools_status,
            ortools_routes: parsed
                .ortools_routes
                .unwrap_or_default()
                .into_iter()
                .map(VRPRoute::from)
                .collect(),
            ortools_objective: parsed.ortools_objective,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    stderr.clone()
                }
            }),
            ortools_message: parsed.ortools_message.unwrap_or_default(),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse OR-Tools Routing adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_cvrp_with_external_reference(
    depot: Point,
    customers: &[VRPCustomer],
    vehicle_capacity: f64,
    opts: &ExternalRoutingReferenceOptions,
) -> ExternalRoutingReferenceSolution {
    if should_use_rust_routing_reference(opts) || should_use_registered_routing_fallback(opts) {
        return relabel_registered_routing_fallback(
            solve_cvrp_with_rust_reference(depot, customers, vehicle_capacity),
            opts,
        );
    }

    run_ortools_routing_reference(depot, customers, vehicle_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ROUTING_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.key, previous),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn sample_customers() -> Vec<VRPCustomer> {
        vec![
            VRPCustomer {
                id: "A".to_string(),
                x: 1.0,
                y: 2.0,
                demand: 2.0,
            },
            VRPCustomer {
                id: "B".to_string(),
                x: 2.0,
                y: 1.0,
                demand: 2.0,
            },
            VRPCustomer {
                id: "C".to_string(),
                x: 4.0,
                y: 1.0,
                demand: 2.0,
            },
            VRPCustomer {
                id: "D".to_string(),
                x: 5.0,
                y: 2.0,
                demand: 1.0,
            },
            VRPCustomer {
                id: "E".to_string(),
                x: 3.0,
                y: 4.0,
                demand: 2.0,
            },
        ]
    }

    #[test]
    fn rust_reference_solves_sample_cvrp() {
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &sample_customers(),
            5.0,
            &ExternalRoutingReferenceOptions {
                solver: ExternalRoutingReferenceSolver::RustExact,
            },
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-cvrp");
        assert!(solution.objective.is_some());
        assert_eq!(
            solution
                .routes
                .iter()
                .map(|route| route.customers.len())
                .sum::<usize>(),
            5
        );
        assert!(solution.feasible_route_masks.is_some());
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_infeasible_capacity() {
        let customers = vec![VRPCustomer {
            id: "A".to_string(),
            x: 1.0,
            y: 0.0,
            demand: 2.0,
        }];
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &customers,
            1.0,
            &ExternalRoutingReferenceOptions {
                solver: ExternalRoutingReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Infeasible);
        assert_eq!(solution.solver, "rust:exact-cvrp");
        assert!(solution.objective.is_none());
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &sample_customers(),
            5.0,
            &ExternalRoutingReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-cvrp");
        assert_eq!(
            solution
                .routes
                .iter()
                .map(|route| route.customers.len())
                .sum::<usize>(),
            5
        );
        assert!(solution.objective.is_some());
    }

    #[test]
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = ROUTING_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("ROUTING_REFERENCE_REGISTERED_FALLBACK", "rust");
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &sample_customers(),
            5.0,
            &ExternalRoutingReferenceOptions {
                solver: ExternalRoutingReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-routing-fallback-for-ortools"
        );
        assert_eq!(
            solution
                .routes
                .iter()
                .map(|route| route.customers.len())
                .sum::<usize>(),
            5
        );
        assert!(solution.objective.is_some());
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn rust_first_env_forces_ortools_to_rust_reference_without_python() {
        let _lock = ROUTING_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("ORES_EXTERNAL_REFERENCE_RUST_FIRST", "true");
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &sample_customers(),
            5.0,
            &ExternalRoutingReferenceOptions {
                solver: ExternalRoutingReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-routing-fallback-for-ortools"
        );
        assert!(solution.objective.is_some());
    }

    #[test]
    fn routing_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_routing_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = ROUTING_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not-a-python-for-routing-ortools");
        let solution = solve_cvrp_with_external_reference(
            Point { x: 0.0, y: 0.0 },
            &sample_customers(),
            5.0,
            &ExternalRoutingReferenceOptions {
                solver: ExternalRoutingReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalRoutingReferenceStatus::Unavailable);
        assert_eq!(solution.solver, "ortools:routing");
        assert!(solution.message.contains("OR-Tools Routing adapter"));
        assert!(!solution.message.contains("routing_reference.py"));
    }
}
