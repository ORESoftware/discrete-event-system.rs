//! Rust-facing bridge for external/reference min-cost-flow solvers.
//!
//! The native Rust reference computes a deterministic successive-shortest-path
//! check without Python startup. The checked-in Python bridge
//! (`scripts/min_cost_flow_reference.py`) remains available for OR-Tools
//! SimpleMinCostFlow on an integer-scaled, lower-bound-normalized copy of the
//! same input.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::min_cost_flow::{
    solve_min_cost_flow, MinCostFlowArcResult, MinCostFlowProblem, MinCostFlowStatus,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinCostFlowReferenceSolver {
    Auto,
    RustSuccessiveShortestPath,
    OrTools,
    Fallback,
}

impl ExternalMinCostFlowReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMinCostFlowReferenceSolver::Auto => "auto",
            ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath => "rust-ssp",
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

fn status_from_min_cost_flow_status(
    status: MinCostFlowStatus,
) -> ExternalMinCostFlowReferenceStatus {
    match status {
        MinCostFlowStatus::Optimal => ExternalMinCostFlowReferenceStatus::Optimal,
        MinCostFlowStatus::Infeasible => ExternalMinCostFlowReferenceStatus::Infeasible,
    }
}

const RUST_MIN_COST_FLOW_EPS: f64 = 1e-9;

fn validate_rust_min_cost_flow_problem(problem: &MinCostFlowProblem) -> Result<(), String> {
    if problem.num_nodes == 0 {
        return Err("num_nodes must be positive".to_string());
    }
    if problem.supplies.len() != problem.num_nodes {
        return Err(format!(
            "supplies length {} != num_nodes {}",
            problem.supplies.len(),
            problem.num_nodes
        ));
    }
    if problem.supplies.iter().any(|value| !value.is_finite()) {
        return Err("supplies must be finite".to_string());
    }
    let total_supply = problem.supplies.iter().sum::<f64>();
    if total_supply.abs() > 1e-7 {
        return Err(format!("supplies must sum to zero, got {total_supply:.3e}"));
    }
    if problem.arcs.is_empty() {
        return Err("arcs must be non-empty".to_string());
    }
    for (index, arc) in problem.arcs.iter().enumerate() {
        if arc.from >= problem.num_nodes || arc.to >= problem.num_nodes {
            return Err(format!("arc {index} endpoint out of range"));
        }
        if arc.from == arc.to {
            return Err(format!("arc {index} is a self-loop"));
        }
        if !arc.lower_bound.is_finite() || !arc.capacity.is_finite() || !arc.cost.is_finite() {
            return Err(format!("arc {index} fields must be finite"));
        }
        if arc.lower_bound < -RUST_MIN_COST_FLOW_EPS {
            return Err(format!("arc {index} lower_bound must be non-negative"));
        }
        if arc.capacity + RUST_MIN_COST_FLOW_EPS < arc.lower_bound {
            return Err(format!(
                "arc {index} capacity {} < lower_bound {}",
                arc.capacity, arc.lower_bound
            ));
        }
    }
    Ok(())
}

fn rust_min_cost_flow_empty_solution(
    status: ExternalMinCostFlowReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinCostFlowReferenceSolution {
    ExternalMinCostFlowReferenceSolution {
        status,
        solver: solver.into(),
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

fn solve_min_cost_flow_with_rust_reference(
    problem: &MinCostFlowProblem,
) -> ExternalMinCostFlowReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_min_cost_flow_problem(problem) {
        return rust_min_cost_flow_empty_solution(
            ExternalMinCostFlowReferenceStatus::NumericalError,
            "rust:ssp-min-cost-flow",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let solution = solve_min_cost_flow(problem.clone());
    let status = status_from_min_cost_flow_status(solution.status);
    ExternalMinCostFlowReferenceSolution {
        status,
        solver: "rust:ssp-min-cost-flow".to_string(),
        objective: if status == ExternalMinCostFlowReferenceStatus::Optimal {
            Some(solution.total_cost)
        } else {
            None
        },
        flows: solution.arc_flows,
        node_balance: solution.node_balance,
        iterations: Some(solution.iterations as u64),
        ortools_status: None,
        ortools_objective: None,
        ortools_flows: Vec::new(),
        ortools_node_balance: Vec::new(),
        message: solution
            .message
            .unwrap_or_else(|| "successive shortest augmenting path reference".to_string()),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
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
    if matches!(
        opts.solver,
        ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath
            | ExternalMinCostFlowReferenceSolver::Fallback
    ) {
        return solve_min_cost_flow_with_rust_reference(problem);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::min_cost_flow::MinCostFlowArc;

    fn transportation_problem() -> MinCostFlowProblem {
        MinCostFlowProblem {
            num_nodes: 4,
            supplies: vec![5.0, 7.0, -6.0, -6.0],
            arcs: vec![
                MinCostFlowArc {
                    from: 0,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 2.0,
                    name: Some("s0_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 0,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 5.0,
                    cost: 4.0,
                    name: Some("s0_d1".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 2,
                    lower_bound: 0.0,
                    capacity: 6.0,
                    cost: 5.0,
                    name: Some("s1_d0".to_string()),
                },
                MinCostFlowArc {
                    from: 1,
                    to: 3,
                    lower_bound: 0.0,
                    capacity: 8.0,
                    cost: 1.0,
                    name: Some("s1_d1".to_string()),
                },
            ],
        }
    }

    #[test]
    fn rust_reference_solves_transportation_problem() {
        let solution = solve_min_cost_flow_with_external_reference(
            &transportation_problem(),
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath,
            },
        );

        assert_eq!(solution.status, ExternalMinCostFlowReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:ssp-min-cost-flow");
        assert_eq!(solution.objective, Some(21.0));
        assert_eq!(solution.flows.len(), 4);
        assert_eq!(solution.node_balance, vec![5.0, 7.0, -6.0, -6.0]);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_infeasible_problem() {
        let problem = MinCostFlowProblem {
            num_nodes: 2,
            supplies: vec![1.0, -1.0],
            arcs: vec![MinCostFlowArc {
                from: 0,
                to: 1,
                lower_bound: 0.0,
                capacity: 0.0,
                cost: 1.0,
                name: Some("blocked".to_string()),
            }],
        };

        let solution = solve_min_cost_flow_with_external_reference(
            &problem,
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinCostFlowReferenceStatus::Infeasible
        );
        assert_eq!(solution.solver, "rust:ssp-min-cost-flow");
        assert!(solution.objective.is_none());
    }
}
