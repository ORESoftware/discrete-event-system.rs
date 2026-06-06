//! Rust-facing bridge for external/reference min-cost-flow solvers.
//!
//! The native Rust reference computes a deterministic successive-shortest-path
//! check without Python startup. OR-Tools SimpleMinCostFlow compatibility
//! validation remains available through explicit force-Python switches and a
//! tiny inline Python adapter over an integer-scaled, lower-bound-normalized
//! copy of the same input.

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn min_cost_flow_reference_force_python_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "1" | "true"
            | "yes"
            | "on"
            | "python"
            | "py"
            | "legacy-python"
            | "python-reference"
            | "python-bridge"
    )
}

fn min_cost_flow_python_reference_forced() -> bool {
    [
        "MIN_COST_FLOW_REFERENCE_FORCE_PYTHON",
        "MIN_COST_FLOW_REFERENCE_ORTOOLS_FORCE_PYTHON",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| min_cost_flow_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_min_cost_flow_reference(opts: &ExternalMinCostFlowReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalMinCostFlowReferenceSolver::Auto
            | ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath
            | ExternalMinCostFlowReferenceSolver::Fallback
    )
}

fn should_use_registered_min_cost_flow_fallback(
    opts: &ExternalMinCostFlowReferenceOptions,
) -> bool {
    matches!(opts.solver, ExternalMinCostFlowReferenceSolver::OrTools)
        && !min_cost_flow_python_reference_forced()
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
const ORTOOLS_INTEGER_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_MIN_COST_FLOW_SOLVER: &str = "ortools:simple-min-cost-flow";

const ORTOOLS_MIN_COST_FLOW_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:simple-min-cost-flow"


def result(status, objective=None, flows=None, node_balance=None, message=""):
    return {
        "status": status,
        "solver": SOLVER,
        "objective": objective,
        "flows": [] if flows is None else flows,
        "nodeBalance": [] if node_balance is None else node_balance,
        "iterations": None,
        "message": message,
    }


try:
    from ortools.graph.python import min_cost_flow
except Exception as exc:
    print(json.dumps(result(
        "unavailable",
        message=f"OR-Tools SimpleMinCostFlow unavailable: {exc}",
    )))
    sys.exit(0)


def status_name(status):
    return str(status).split(".")[-1].lower()


try:
    problem = json.load(sys.stdin)
    flow_scale = float(problem["flowScale"])
    cost_scale = float(problem["costScale"])
    solver = min_cost_flow.SimpleMinCostFlow()
    for arc in problem["arcs"]:
        solver.add_arc_with_capacity_and_unit_cost(
            int(arc["from"]),
            int(arc["to"]),
            int(arc["scaledCapacity"]),
            int(arc["scaledCost"]),
        )
    for node, supply in enumerate(problem["scaledSupplies"]):
        solver.set_node_supply(node, int(supply))

    status = solver.solve()
    mapped = status_name(status)
    if status != solver.OPTIMAL:
        print(json.dumps(result(
            mapped,
            message=f"OR-Tools SimpleMinCostFlow status {mapped}",
        )))
        sys.exit(0)

    flows = []
    for index, arc in enumerate(problem["arcs"]):
        flow = float(arc["lowerBound"]) + solver.flow(index) / flow_scale
        flows.append({
            "from": int(arc["from"]),
            "to": int(arc["to"]),
            "lowerBound": float(arc["lowerBound"]),
            "capacity": float(arc["capacity"]),
            "cost": float(arc["cost"]),
            "flow": flow,
            "name": arc.get("name"),
        })
    node_balance = [0.0 for _ in range(int(problem["numNodes"]))]
    for arc, flow in zip(problem["arcs"], flows):
        node_balance[int(arc["from"])] += flow["flow"]
        node_balance[int(arc["to"])] -= flow["flow"]
    objective = float(problem["baseCost"]) + solver.optimal_cost() / (
        flow_scale * cost_scale
    )
    print(json.dumps(result(
        "optimal",
        objective=objective,
        flows=flows,
        node_balance=node_balance,
        message="OR-Tools SimpleMinCostFlow",
    )))
except Exception as exc:
    print(json.dumps(result("error", message=str(exc))))
    sys.exit(1)
"#;

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

fn relabel_registered_min_cost_flow_fallback(
    mut solution: ExternalMinCostFlowReferenceSolution,
    opts: &ExternalMinCostFlowReferenceOptions,
) -> ExternalMinCostFlowReferenceSolution {
    if should_use_registered_min_cost_flow_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-min-cost-flow-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn ortools_empty_solution(
    status: ExternalMinCostFlowReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinCostFlowReferenceSolution {
    rust_min_cost_flow_empty_solution(status, ORTOOLS_MIN_COST_FLOW_SOLVER, message, elapsed_ms)
}

fn scaled_ortools_value(value: f64, scale: i64) -> Option<i64> {
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled.abs() > i64::MAX as f64 {
        return None;
    }
    let rounded = scaled.round();
    if (rounded - scaled).abs() <= 1e-6 {
        Some(rounded as i64)
    } else {
        None
    }
}

fn choose_ortools_flow_scale(problem: &MinCostFlowProblem) -> Option<i64> {
    ORTOOLS_INTEGER_SCALES.into_iter().find(|scale| {
        problem
            .supplies
            .iter()
            .all(|value| scaled_ortools_value(*value, *scale).is_some())
            && problem.arcs.iter().all(|arc| {
                scaled_ortools_value(arc.lower_bound, *scale).is_some()
                    && scaled_ortools_value(arc.capacity, *scale).is_some()
                    && scaled_ortools_value(arc.capacity - arc.lower_bound, *scale).is_some()
            })
    })
}

fn choose_ortools_cost_scale(problem: &MinCostFlowProblem) -> Option<i64> {
    ORTOOLS_INTEGER_SCALES.into_iter().find(|scale| {
        problem
            .arcs
            .iter()
            .all(|arc| scaled_ortools_value(arc.cost, *scale).is_some())
    })
}

fn ortools_min_cost_flow_payload(
    problem: &MinCostFlowProblem,
    flow_scale: i64,
    cost_scale: i64,
) -> Value {
    let mut adjusted_supply = problem.supplies.clone();
    let mut base_cost = 0.0;
    for arc in &problem.arcs {
        adjusted_supply[arc.from] -= arc.lower_bound;
        adjusted_supply[arc.to] += arc.lower_bound;
        base_cost += arc.lower_bound * arc.cost;
    }
    json!({
        "numNodes": problem.num_nodes,
        "flowScale": flow_scale,
        "costScale": cost_scale,
        "baseCost": base_cost,
        "scaledSupplies": adjusted_supply.iter().map(|supply| {
            scaled_ortools_value(*supply, flow_scale)
                .expect("flow scale chosen for adjusted supplies")
        }).collect::<Vec<_>>(),
        "arcs": problem.arcs.iter().map(|arc| {
            json!({
                "from": arc.from,
                "to": arc.to,
                "lowerBound": arc.lower_bound,
                "capacity": arc.capacity,
                "cost": arc.cost,
                "scaledCapacity": scaled_ortools_value(
                    arc.capacity - arc.lower_bound,
                    flow_scale,
                ).expect("flow scale chosen for residual arc capacity"),
                "scaledCost": scaled_ortools_value(arc.cost, cost_scale)
                    .expect("cost scale chosen for arc cost"),
                "name": &arc.name,
            })
        }).collect::<Vec<_>>(),
    })
}

fn min_cost_flow_reference_timeout_ms() -> u64 {
    std::env::var("MIN_COST_FLOW_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_min_cost_flow_reference_output(
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
            Err(err) => {
                return Err(format!(
                    "failed to poll OR-Tools min-cost-flow adapter: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools min-cost-flow adapter: {err}"))
}

fn run_ortools_min_cost_flow_reference(
    problem: &MinCostFlowProblem,
) -> ExternalMinCostFlowReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_min_cost_flow_problem(problem) {
        return ortools_empty_solution(
            ExternalMinCostFlowReferenceStatus::NumericalError,
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let Some(flow_scale) = choose_ortools_flow_scale(problem) else {
        return ortools_empty_solution(
            ExternalMinCostFlowReferenceStatus::Unsupported,
            "OR-Tools SimpleMinCostFlow requires integer-scalable supplies/capacities/costs",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let Some(cost_scale) = choose_ortools_cost_scale(problem) else {
        return ortools_empty_solution(
            ExternalMinCostFlowReferenceStatus::Unsupported,
            "OR-Tools SimpleMinCostFlow requires integer-scalable supplies/capacities/costs",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let payload = ortools_min_cost_flow_payload(problem, flow_scale, cost_scale);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_MIN_COST_FLOW_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ortools_empty_solution(
                ExternalMinCostFlowReferenceStatus::Unavailable,
                format!("failed to start OR-Tools min-cost-flow adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return ortools_empty_solution(
                ExternalMinCostFlowReferenceStatus::NumericalError,
                format!("failed to write OR-Tools min-cost-flow adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = min_cost_flow_reference_timeout_ms();
    let (output, timed_out) = match wait_for_min_cost_flow_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return ortools_empty_solution(
                ExternalMinCostFlowReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools min-cost-flow adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools min-cost-flow adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => ortools_empty_solution(
            ExternalMinCostFlowReferenceStatus::NumericalError,
            format!(
                "failed to parse OR-Tools min-cost-flow adapter output: {err}; stderr={stderr}"
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_min_cost_flow_with_external_reference(
    problem: &MinCostFlowProblem,
    opts: &ExternalMinCostFlowReferenceOptions,
) -> ExternalMinCostFlowReferenceSolution {
    if should_use_rust_min_cost_flow_reference(opts)
        || should_use_registered_min_cost_flow_fallback(opts)
    {
        return relabel_registered_min_cost_flow_fallback(
            solve_min_cost_flow_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_min_cost_flow_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::min_cost_flow::MinCostFlowArc;
    use std::sync::Mutex;

    static MIN_COST_FLOW_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn min_cost_flow_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "MIN_COST_FLOW_REFERENCE_FORCE_PYTHON",
            "MIN_COST_FLOW_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

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

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let solution = solve_min_cost_flow_with_external_reference(
            &transportation_problem(),
            &ExternalMinCostFlowReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalMinCostFlowReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:ssp-min-cost-flow");
        assert_eq!(solution.objective, Some(21.0));
        assert_eq!(solution.node_balance, vec![5.0, 7.0, -6.0, -6.0]);
    }

    #[test]
    fn registered_ortools_alias_defaults_to_rust_reference_without_python() {
        let _lock = MIN_COST_FLOW_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = min_cost_flow_force_python_off_guards();
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-min-cost-flow-alias",
        );

        let solution = solve_min_cost_flow_with_external_reference(
            &transportation_problem(),
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalMinCostFlowReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-min-cost-flow-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(21.0));
        assert_eq!(solution.node_balance, vec![5.0, 7.0, -6.0, -6.0]);
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn min_cost_flow_force_python_keeps_ortools_bridge_available() {
        let _lock = MIN_COST_FLOW_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("MIN_COST_FLOW_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-forced-min-cost-flow",
        );

        let solution = solve_min_cost_flow_with_external_reference(
            &transportation_problem(),
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinCostFlowReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, "ortools:simple-min-cost-flow");
        assert!(solution.message.contains("OR-Tools min-cost-flow adapter"));
    }

    #[test]
    fn ortools_adapter_rejects_unscaled_values_without_python() {
        let _lock = MIN_COST_FLOW_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("MIN_COST_FLOW_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = MinCostFlowProblem {
            num_nodes: 2,
            supplies: vec![1.0 / 3.0, -1.0 / 3.0],
            arcs: vec![MinCostFlowArc {
                from: 0,
                to: 1,
                lower_bound: 0.0,
                capacity: 1.0 / 3.0,
                cost: 1.0,
                name: None,
            }],
        };

        let solution = solve_min_cost_flow_with_external_reference(
            &problem,
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinCostFlowReferenceStatus::Unsupported
        );
        assert_eq!(solution.solver, "ortools:simple-min-cost-flow");
        assert!(solution
            .message
            .contains("requires integer-scalable supplies/capacities/costs"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = MIN_COST_FLOW_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("MIN_COST_FLOW_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");

        let solution = solve_min_cost_flow_with_external_reference(
            &transportation_problem(),
            &ExternalMinCostFlowReferenceOptions {
                solver: ExternalMinCostFlowReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinCostFlowReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, "ortools:simple-min-cost-flow");
        assert!(solution.message.contains("OR-Tools min-cost-flow adapter"));
        assert!(!solution.message.contains("min_cost_flow_reference.py"));
    }

    #[test]
    fn min_cost_flow_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_min_cost_flow_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
