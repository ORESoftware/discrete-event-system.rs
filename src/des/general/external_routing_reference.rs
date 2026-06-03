//! Rust-facing bridge for external/reference vehicle-routing solvers.
//!
//! The checked-in Python bridge (`scripts/routing_reference.py`) computes an
//! exact small CVRP route-cover reference and, when installed, calls OR-Tools
//! Routing on the same input. This module owns typed model serialization and
//! status mapping so callers do not need to shell out manually.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::classical_optimization_models::{Point, VRPCustomer, VRPRoute};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalRoutingReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalRoutingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalRoutingReferenceSolver::Auto => "auto",
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
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for routing_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
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
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            ortools_message: parsed.ortools_message.unwrap_or_default(),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse routing_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
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
