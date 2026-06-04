//! Rust-facing bridge for external/reference vehicle-routing solvers.
//!
//! The native Rust reference computes an exact small CVRP route-cover check
//! without Python startup. The checked-in Python bridge
//! (`scripts/routing_reference.py`) remains available for OR-Tools Routing on
//! the same input.

use std::io::Write;
use std::path::PathBuf;
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
        solver: "external-routing-reference".to_string(),
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
        solver: "external-routing-reference".to_string(),
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

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("routing_reference.py")
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
            Err(err) => return Err(format!("failed to poll routing_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for routing_reference.py: {err}"))
}

fn run_routing_reference_json(
    payload: Value,
    opts: &ExternalRoutingReferenceOptions,
) -> ExternalRoutingReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command
        .arg(reference_script())
        .arg("--solver")
        .arg(opts.solver.as_arg());
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start routing_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write routing_reference.py stdin: {err}"),
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
            format!("routing_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; routing_reference.py timed out after {timeout_ms}ms")
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
                "failed to parse routing_reference.py output: {err}; stderr={}",
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
    if matches!(
        opts.solver,
        ExternalRoutingReferenceSolver::Auto
            | ExternalRoutingReferenceSolver::RustExact
            | ExternalRoutingReferenceSolver::Fallback
    ) {
        return solve_cvrp_with_rust_reference(depot, customers, vehicle_capacity);
    }

    run_routing_reference_json(
        json!({
            "depot": {
                "x": depot.x,
                "y": depot.y,
            },
            "customers": customers.iter().map(|customer| json!({
                "id": &customer.id,
                "x": customer.x,
                "y": customer.y,
                "demand": customer.demand,
            })).collect::<Vec<_>>(),
            "vehicle_capacity": vehicle_capacity,
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn routing_python_bridge_wait_enforces_timeout() {
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
}
