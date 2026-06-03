//! Rust-facing bridge for external/reference min-cost-flow solvers.
//!
//! The checked-in Python bridge (`scripts/min_cost_flow_reference.py`) computes
//! a deterministic successive-shortest-path reference and, when installed,
//! calls OR-Tools SimpleMinCostFlow on an integer-scaled, lower-bound-normalized
//! copy of the same input. This module owns typed serialization and status
//! mapping for those same-input network-flow cross-checks.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::min_cost_flow::{MinCostFlowArcResult, MinCostFlowProblem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinCostFlowReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalMinCostFlowReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMinCostFlowReferenceSolver::Auto => "auto",
            ExternalMinCostFlowReferenceSolver::OrTools => "ortools",
            ExternalMinCostFlowReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinCostFlowReferenceOptions {
    pub solver: ExternalMinCostFlowReferenceSolver,
}

impl Default for ExternalMinCostFlowReferenceOptions {
    fn default() -> Self {
        ExternalMinCostFlowReferenceOptions {
            solver: ExternalMinCostFlowReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinCostFlowReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalMinCostFlowReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalMinCostFlowReferenceStatus::Optimal => "optimal",
            ExternalMinCostFlowReferenceStatus::Feasible => "feasible",
            ExternalMinCostFlowReferenceStatus::Infeasible => "infeasible",
            ExternalMinCostFlowReferenceStatus::Unsupported => "unsupported",
            ExternalMinCostFlowReferenceStatus::NumericalError => "numerical-error",
            ExternalMinCostFlowReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinCostFlowReferenceSolution {
    pub status: ExternalMinCostFlowReferenceStatus,
    pub solver: String,
    pub objective: Option<f64>,
    pub flows: Vec<MinCostFlowArcResult>,
    pub node_balance: Vec<f64>,
    pub iterations: Option<u64>,
    pub ortools_status: Option<String>,
    pub ortools_objective: Option<f64>,
    pub ortools_flows: Vec<MinCostFlowArcResult>,
    pub ortools_node_balance: Vec<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct MinCostFlowReferencePayload {
    status: String,
    solver: Option<String>,
    objective: Option<f64>,
    flows: Option<Vec<MinCostFlowArcResultPayload>>,
    #[serde(rename = "nodeBalance")]
    node_balance: Option<Vec<f64>>,
    iterations: Option<u64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsFlows")]
    ortools_flows: Option<Vec<MinCostFlowArcResultPayload>>,
    #[serde(rename = "ortoolsNodeBalance")]
    ortools_node_balance: Option<Vec<f64>>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MinCostFlowArcResultPayload {
    from: usize,
    to: usize,
    #[serde(rename = "lowerBound")]
    lower_bound: f64,
    capacity: f64,
    cost: f64,
    flow: f64,
    name: Option<String>,
}

impl From<MinCostFlowArcResultPayload> for MinCostFlowArcResult {
    fn from(value: MinCostFlowArcResultPayload) -> Self {
        MinCostFlowArcResult {
            from: value.from,
            to: value.to,
            lower_bound: value.lower_bound,
            capacity: value.capacity,
            cost: value.cost,
            flow: value.flow,
            name: value.name,
        }
    }
}

fn status_from_str(status: &str) -> ExternalMinCostFlowReferenceStatus {
    match status {
        "optimal" => ExternalMinCostFlowReferenceStatus::Optimal,
        "feasible" => ExternalMinCostFlowReferenceStatus::Feasible,
        "infeasible" => ExternalMinCostFlowReferenceStatus::Infeasible,
        "unsupported" => ExternalMinCostFlowReferenceStatus::Unsupported,
        "unavailable" => ExternalMinCostFlowReferenceStatus::Unavailable,
        _ => ExternalMinCostFlowReferenceStatus::NumericalError,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinCostFlowReferenceSolution {
    ExternalMinCostFlowReferenceSolution {
        status: ExternalMinCostFlowReferenceStatus::Unavailable,
        solver: "external-min-cost-flow-reference".to_string(),
        objective: None,
        flows: Vec::new(),
        node_balance: Vec::new(),
        iterations: None,
        ortools_status: None,
        ortools_objective: None,
        ortools_flows: Vec::new(),
        ortools_node_balance: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinCostFlowReferenceSolution {
    ExternalMinCostFlowReferenceSolution {
        status: ExternalMinCostFlowReferenceStatus::NumericalError,
        solver: "external-min-cost-flow-reference".to_string(),
        objective: None,
        flows: Vec::new(),
        node_balance: Vec::new(),
        iterations: None,
        ortools_status: None,
        ortools_objective: None,
        ortools_flows: Vec::new(),
        ortools_node_balance: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("min_cost_flow_reference.py")
}

fn run_min_cost_flow_reference_json(
    payload: Value,
    opts: &ExternalMinCostFlowReferenceOptions,
) -> ExternalMinCostFlowReferenceSolution {
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
                format!("failed to start min_cost_flow_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write min_cost_flow_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for min_cost_flow_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<MinCostFlowReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalMinCostFlowReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-min-cost-flow-reference".to_string()),
            objective: parsed.objective,
            flows: parsed
                .flows
                .unwrap_or_default()
                .into_iter()
                .map(MinCostFlowArcResult::from)
                .collect(),
            node_balance: parsed.node_balance.unwrap_or_default(),
            iterations: parsed.iterations,
            ortools_status: parsed.ortools_status,
            ortools_objective: parsed.ortools_objective,
            ortools_flows: parsed
                .ortools_flows
                .unwrap_or_default()
                .into_iter()
                .map(MinCostFlowArcResult::from)
                .collect(),
            ortools_node_balance: parsed.ortools_node_balance.unwrap_or_default(),
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse min_cost_flow_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_min_cost_flow_with_external_reference(
    problem: &MinCostFlowProblem,
    opts: &ExternalMinCostFlowReferenceOptions,
) -> ExternalMinCostFlowReferenceSolution {
    run_min_cost_flow_reference_json(
        json!({
            "num_nodes": problem.num_nodes,
            "supplies": &problem.supplies,
            "arcs": problem.arcs.iter().map(|arc| json!({
                "from": arc.from,
                "to": arc.to,
                "lower_bound": arc.lower_bound,
                "capacity": arc.capacity,
                "cost": arc.cost,
                "name": &arc.name,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
