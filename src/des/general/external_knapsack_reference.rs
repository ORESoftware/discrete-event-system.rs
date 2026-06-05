//! Rust-facing bridge for external/reference 0/1 knapsack solvers.
//!
//! The native Rust reference computes an independent exact branch-and-bound
//! check without Python startup. Explicit OR-Tools CP-SAT validation is launched
//! from Rust with a tiny Python adapter over an integer-scaled copy of the same
//! input.

use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::knapsack::KnapsackProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalKnapsackReferenceSolver {
    Auto,
    RustBranchAndBound,
    OrTools,
    Fallback,
}

impl ExternalKnapsackReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalKnapsackReferenceSolver::Auto => "auto",
            ExternalKnapsackReferenceSolver::RustBranchAndBound => "rust-branch-and-bound",
            ExternalKnapsackReferenceSolver::OrTools => "ortools",
            ExternalKnapsackReferenceSolver::Fallback => "fallback",
        }
    }
}

fn registered_knapsack_rust_fallback_enabled() -> bool {
    std::env::var("KNAPSACK_REFERENCE_REGISTERED_FALLBACK")
        .or_else(|_| std::env::var("KNAPSACK_REFERENCE_EXTERNAL_FALLBACK"))
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
            )
        })
        .unwrap_or(false)
}

fn should_use_rust_knapsack_reference(opts: &ExternalKnapsackReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalKnapsackReferenceSolver::Auto
            | ExternalKnapsackReferenceSolver::RustBranchAndBound
            | ExternalKnapsackReferenceSolver::Fallback
    )
}

fn should_use_registered_knapsack_fallback(opts: &ExternalKnapsackReferenceOptions) -> bool {
    registered_knapsack_rust_fallback_enabled()
        && matches!(opts.solver, ExternalKnapsackReferenceSolver::OrTools)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalKnapsackReferenceOptions {
    pub solver: ExternalKnapsackReferenceSolver,
}

impl Default for ExternalKnapsackReferenceOptions {
    fn default() -> Self {
        ExternalKnapsackReferenceOptions {
            solver: ExternalKnapsackReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalKnapsackReferenceStatus {
    Optimal,
    Feasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalKnapsackReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalKnapsackReferenceStatus::Optimal => "optimal",
            ExternalKnapsackReferenceStatus::Feasible => "feasible",
            ExternalKnapsackReferenceStatus::Unsupported => "unsupported",
            ExternalKnapsackReferenceStatus::NumericalError => "numerical-error",
            ExternalKnapsackReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalKnapsackReferenceSolution {
    pub status: ExternalKnapsackReferenceStatus,
    pub solver: String,
    pub selected_item_indices: Vec<usize>,
    pub selected_item_ids: Vec<String>,
    pub total_weight: Option<f64>,
    pub total_value: Option<f64>,
    pub objective: Option<f64>,
    pub upper_bound: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_item_indices: Vec<usize>,
    pub ortools_selected_item_ids: Vec<String>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_total_value: Option<f64>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct KnapsackReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedItemIndices")]
    selected_item_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedItemIds")]
    selected_item_ids: Option<Vec<String>>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "totalValue")]
    total_value: Option<f64>,
    objective: Option<f64>,
    #[serde(rename = "upperBound")]
    upper_bound: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedItemIndices")]
    ortools_selected_item_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedItemIds")]
    ortools_selected_item_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsTotalValue")]
    ortools_total_value: Option<f64>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

#[derive(Clone, Debug)]
struct RustKnapsackSearchItem {
    index: usize,
    weight: f64,
    value: f64,
    density: f64,
}

const RUST_KNAPSACK_EPS: f64 = 1e-9;
const RUST_KNAPSACK_MAX_EXACT_ITEMS: usize = 64;
const ORTOOLS_INTEGER_SCALES: [i64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
const ORTOOLS_KNAPSACK_SOLVER: &str = "ortools:cp-sat-knapsack";

const ORTOOLS_KNAPSACK_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:cp-sat-knapsack"


def solution(status, problem, selected_indices=None, upper_bound=None, message=""):
    indices = [] if selected_indices is None else sorted(int(index) for index in selected_indices)
    items_by_index = {int(item["index"]): item for item in problem["items"]}
    selected_ids = [items_by_index[index]["id"] for index in indices]
    total_weight = sum(float(items_by_index[index]["weight"]) for index in indices)
    total_value = sum(float(items_by_index[index]["value"]) for index in indices)
    output = {
        "status": status,
        "solver": SOLVER,
        "selectedItemIndices": indices,
        "selectedItemIds": selected_ids,
        "totalWeight": total_weight,
        "totalValue": total_value,
        "objective": total_value,
        "upperBound": upper_bound,
        "message": message,
    }
    if upper_bound is not None:
        output["objectiveBound"] = upper_bound
    return output


try:
    from ortools.sat.python import cp_model
except Exception as exc:
    fallback_problem = {
        "items": [],
    }
    print(json.dumps(solution("unavailable", fallback_problem, None, None, str(exc))))
    sys.exit(0)


try:
    problem = json.load(sys.stdin)
    model = cp_model.CpModel()
    x = [
        model.NewBoolVar(f"x_{item['id']}")
        for item in problem["items"]
    ]
    model.Add(sum(
        int(item["scaledWeight"]) * x[index]
        for index, item in enumerate(problem["items"])
    ) <= int(problem["scaledCapacity"]))
    model.Maximize(sum(
        int(item["scaledValue"]) * x[index]
        for index, item in enumerate(problem["items"])
    ))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(json.dumps(solution(
            "infeasible" if status_name == "infeasible" else status_name,
            problem,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )))
        sys.exit(0)
    selected = [
        index for index, var in enumerate(x) if solver.Value(var)
    ]
    print(json.dumps(solution(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        problem,
        selected,
        solver.BestObjectiveBound() / float(problem["valueScale"]),
        f"OR-Tools CP-SAT status {status_name}",
    )))
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "solver": SOLVER,
        "selectedItemIndices": [],
        "selectedItemIds": [],
        "totalWeight": 0.0,
        "totalValue": 0.0,
        "objective": None,
        "upperBound": None,
        "message": str(exc),
    }))
    sys.exit(1)
"#;

fn status_from_str(status: &str) -> ExternalKnapsackReferenceStatus {
    match status {
        "optimal" => ExternalKnapsackReferenceStatus::Optimal,
        "feasible" => ExternalKnapsackReferenceStatus::Feasible,
        "unsupported" => ExternalKnapsackReferenceStatus::Unsupported,
        "unavailable" => ExternalKnapsackReferenceStatus::Unavailable,
        _ => ExternalKnapsackReferenceStatus::NumericalError,
    }
}

fn validate_rust_knapsack_problem(problem: &KnapsackProblem) -> Result<(), String> {
    if !problem.capacity.is_finite() || problem.capacity <= 0.0 {
        return Err("capacity must be finite and > 0".to_string());
    }
    if problem.items.is_empty() {
        return Err("items must be non-empty".to_string());
    }
    let mut seen = HashSet::new();
    for (index, item) in problem.items.iter().enumerate() {
        if item.id.trim().is_empty() {
            return Err(format!("items[{index}].id must be non-empty"));
        }
        if !seen.insert(item.id.clone()) {
            return Err(format!("duplicate item id {:?}", item.id));
        }
        if !item.weight.is_finite() || item.weight <= 0.0 {
            return Err(format!("items[{index}].weight must be finite and > 0"));
        }
        if !item.value.is_finite() || item.value < 0.0 {
            return Err(format!(
                "items[{index}].value must be finite and non-negative"
            ));
        }
    }
    Ok(())
}

fn rust_knapsack_empty_solution(
    status: ExternalKnapsackReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalKnapsackReferenceSolution {
    ExternalKnapsackReferenceSolution {
        status,
        solver: solver.into(),
        selected_item_indices: Vec::new(),
        selected_item_ids: Vec::new(),
        total_weight: None,
        total_value: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_item_indices: Vec::new(),
        ortools_selected_item_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_total_value: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn relabel_registered_knapsack_fallback(
    mut solution: ExternalKnapsackReferenceSolution,
    opts: &ExternalKnapsackReferenceOptions,
) -> ExternalKnapsackReferenceSolution {
    if should_use_registered_knapsack_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-knapsack-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn rust_knapsack_sorted_items(problem: &KnapsackProblem) -> Vec<RustKnapsackSearchItem> {
    let mut items = problem
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| RustKnapsackSearchItem {
            index,
            weight: item.weight,
            value: item.value,
            density: item.value / item.weight,
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .density
            .total_cmp(&left.density)
            .then_with(|| right.value.total_cmp(&left.value))
            .then_with(|| left.weight.total_cmp(&right.weight))
            .then_with(|| left.index.cmp(&right.index))
    });
    items
}

fn rust_knapsack_fractional_upper_bound(
    capacity: f64,
    order: &[RustKnapsackSearchItem],
    pos: usize,
    current_weight: f64,
    current_value: f64,
) -> f64 {
    if current_weight > capacity + RUST_KNAPSACK_EPS {
        return f64::NEG_INFINITY;
    }
    let mut bound = current_value;
    let mut remaining = capacity - current_weight;
    for item in &order[pos..] {
        if item.weight <= remaining + RUST_KNAPSACK_EPS {
            bound += item.value;
            remaining -= item.weight;
        } else if remaining > RUST_KNAPSACK_EPS {
            bound += item.value * (remaining / item.weight);
            break;
        } else {
            break;
        }
    }
    bound
}

fn rust_knapsack_candidate_better(
    value: f64,
    weight: f64,
    indices: &[usize],
    best_value: f64,
    best_weight: f64,
    best_indices: &[usize],
) -> bool {
    if value > best_value + RUST_KNAPSACK_EPS {
        return true;
    }
    if (value - best_value).abs() <= RUST_KNAPSACK_EPS && weight < best_weight - RUST_KNAPSACK_EPS {
        return true;
    }
    if (value - best_value).abs() <= RUST_KNAPSACK_EPS
        && (weight - best_weight).abs() <= RUST_KNAPSACK_EPS
    {
        let mut left = indices.to_vec();
        let mut right = best_indices.to_vec();
        left.sort_unstable();
        right.sort_unstable();
        return left < right;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn rust_knapsack_search_branch_and_bound(
    capacity: f64,
    order: &[RustKnapsackSearchItem],
    pos: usize,
    current_weight: f64,
    current_value: f64,
    current: &mut Vec<usize>,
    best_indices: &mut Vec<usize>,
    best_weight: &mut f64,
    best_value: &mut f64,
) {
    if current_weight > capacity + RUST_KNAPSACK_EPS {
        return;
    }
    if pos == order.len() {
        if rust_knapsack_candidate_better(
            current_value,
            current_weight,
            current,
            *best_value,
            *best_weight,
            best_indices,
        ) {
            *best_indices = current.clone();
            *best_weight = current_weight;
            *best_value = current_value;
        }
        return;
    }

    let bound =
        rust_knapsack_fractional_upper_bound(capacity, order, pos, current_weight, current_value);
    if bound + RUST_KNAPSACK_EPS < *best_value {
        return;
    }

    let item = &order[pos];
    current.push(item.index);
    rust_knapsack_search_branch_and_bound(
        capacity,
        order,
        pos + 1,
        current_weight + item.weight,
        current_value + item.value,
        current,
        best_indices,
        best_weight,
        best_value,
    );
    current.pop();
    rust_knapsack_search_branch_and_bound(
        capacity,
        order,
        pos + 1,
        current_weight,
        current_value,
        current,
        best_indices,
        best_weight,
        best_value,
    );
}

fn rust_knapsack_solution(
    problem: &KnapsackProblem,
    status: ExternalKnapsackReferenceStatus,
    mut selected_item_indices: Vec<usize>,
    upper_bound: Option<f64>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalKnapsackReferenceSolution {
    selected_item_indices.sort_unstable();
    let selected_item_ids = selected_item_indices
        .iter()
        .map(|&index| problem.items[index].id.clone())
        .collect::<Vec<_>>();
    let total_weight = selected_item_indices
        .iter()
        .map(|&index| problem.items[index].weight)
        .sum::<f64>();
    let total_value = selected_item_indices
        .iter()
        .map(|&index| problem.items[index].value)
        .sum::<f64>();
    ExternalKnapsackReferenceSolution {
        status,
        solver: "rust:branch-and-bound-knapsack".to_string(),
        selected_item_indices,
        selected_item_ids,
        total_weight: Some(total_weight),
        total_value: Some(total_value),
        objective: Some(total_value),
        upper_bound,
        ortools_status: None,
        ortools_selected_item_indices: Vec::new(),
        ortools_selected_item_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_total_value: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn solve_knapsack_with_rust_reference(
    problem: &KnapsackProblem,
) -> ExternalKnapsackReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_knapsack_problem(problem) {
        return rust_knapsack_empty_solution(
            ExternalKnapsackReferenceStatus::NumericalError,
            "rust:branch-and-bound-knapsack",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if problem.items.len() > RUST_KNAPSACK_MAX_EXACT_ITEMS {
        return rust_knapsack_solution(
            problem,
            ExternalKnapsackReferenceStatus::Unsupported,
            Vec::new(),
            None,
            format!(
                "exact knapsack only practical for <= {RUST_KNAPSACK_MAX_EXACT_ITEMS} items, got {}",
                problem.items.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let order = rust_knapsack_sorted_items(problem);
    let root_bound = rust_knapsack_fractional_upper_bound(problem.capacity, &order, 0, 0.0, 0.0);
    let mut best_indices = Vec::new();
    let mut best_weight = 0.0;
    let mut best_value = 0.0;
    for item in &order {
        if best_weight + item.weight <= problem.capacity + RUST_KNAPSACK_EPS {
            best_indices.push(item.index);
            best_weight += item.weight;
            best_value += item.value;
        }
    }

    let mut current = Vec::new();
    rust_knapsack_search_branch_and_bound(
        problem.capacity,
        &order,
        0,
        0.0,
        0.0,
        &mut current,
        &mut best_indices,
        &mut best_weight,
        &mut best_value,
    );

    rust_knapsack_solution(
        problem,
        ExternalKnapsackReferenceStatus::Optimal,
        best_indices,
        Some(root_bound),
        "exact branch-and-bound with fractional-relaxation bound",
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn ortools_empty_solution(
    status: ExternalKnapsackReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalKnapsackReferenceSolution {
    rust_knapsack_empty_solution(status, ORTOOLS_KNAPSACK_SOLVER, message, elapsed_ms)
}

fn scaled_ortools_value(value: f64, scale: i64) -> Option<i64> {
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return None;
    }
    let rounded = scaled.round();
    if (rounded - scaled).abs() <= 1e-6 {
        Some(rounded as i64)
    } else {
        None
    }
}

fn choose_ortools_weight_scale(problem: &KnapsackProblem) -> Option<i64> {
    ORTOOLS_INTEGER_SCALES.into_iter().find(|scale| {
        scaled_ortools_value(problem.capacity, *scale).is_some()
            && problem
                .items
                .iter()
                .all(|item| scaled_ortools_value(item.weight, *scale).is_some())
    })
}

fn choose_ortools_value_scale(problem: &KnapsackProblem) -> Option<i64> {
    ORTOOLS_INTEGER_SCALES.into_iter().find(|scale| {
        problem
            .items
            .iter()
            .all(|item| scaled_ortools_value(item.value, *scale).is_some())
    })
}

fn ortools_knapsack_payload(
    problem: &KnapsackProblem,
    weight_scale: i64,
    value_scale: i64,
) -> Value {
    json!({
        "scaledCapacity": scaled_ortools_value(problem.capacity, weight_scale)
            .expect("weight scale chosen for capacity"),
        "valueScale": value_scale,
        "items": problem.items.iter().enumerate().map(|(index, item)| {
            json!({
                "id": item.id,
                "index": index,
                "weight": item.weight,
                "value": item.value,
                "scaledWeight": scaled_ortools_value(item.weight, weight_scale)
                    .expect("weight scale chosen for item weights"),
                "scaledValue": scaled_ortools_value(item.value, value_scale)
                    .expect("value scale chosen for item values"),
            })
        }).collect::<Vec<_>>(),
    })
}

fn knapsack_reference_timeout_ms() -> u64 {
    std::env::var("KNAPSACK_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_knapsack_reference_output(
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
            Err(err) => return Err(format!("failed to poll OR-Tools knapsack adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools knapsack adapter: {err}"))
}

fn run_ortools_knapsack_reference(problem: &KnapsackProblem) -> ExternalKnapsackReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_knapsack_problem(problem) {
        return ortools_empty_solution(
            ExternalKnapsackReferenceStatus::NumericalError,
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let Some(weight_scale) = choose_ortools_weight_scale(problem) else {
        return ortools_empty_solution(
            ExternalKnapsackReferenceStatus::Unsupported,
            "OR-Tools CP-SAT bridge requires integer-scalable weights/capacity and values",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let Some(value_scale) = choose_ortools_value_scale(problem) else {
        return ortools_empty_solution(
            ExternalKnapsackReferenceStatus::Unsupported,
            "OR-Tools CP-SAT bridge requires integer-scalable weights/capacity and values",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    };
    let payload = ortools_knapsack_payload(problem, weight_scale, value_scale);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_KNAPSACK_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ortools_empty_solution(
                ExternalKnapsackReferenceStatus::Unavailable,
                format!("failed to start OR-Tools knapsack adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return ortools_empty_solution(
                ExternalKnapsackReferenceStatus::NumericalError,
                format!("failed to write OR-Tools knapsack adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    drop(child.stdin.take());
    let timeout_ms = knapsack_reference_timeout_ms();
    let (mut output, timed_out) = match wait_for_knapsack_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return ortools_empty_solution(
                ExternalKnapsackReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    };
    if timed_out {
        let timeout_message = format!("OR-Tools knapsack adapter timed out after {timeout_ms}ms");
        if output.stderr.is_empty() {
            output.stderr = timeout_message.into_bytes();
        } else {
            let mut stderr = timeout_message.into_bytes();
            stderr.push(b'\n');
            stderr.extend(output.stderr);
            output.stderr = stderr;
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<KnapsackReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalKnapsackReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-knapsack-reference".to_string()),
            selected_item_indices: parsed.selected_item_indices.unwrap_or_default(),
            selected_item_ids: parsed.selected_item_ids.unwrap_or_default(),
            total_weight: parsed.total_weight,
            total_value: parsed.total_value,
            objective: parsed.objective,
            upper_bound: parsed.upper_bound,
            ortools_status: parsed.ortools_status,
            ortools_selected_item_indices: parsed.ortools_selected_item_indices.unwrap_or_default(),
            ortools_selected_item_ids: parsed.ortools_selected_item_ids.unwrap_or_default(),
            ortools_total_weight: parsed.ortools_total_weight,
            ortools_total_value: parsed.ortools_total_value,
            ortools_objective: parsed.ortools_objective,
            ortools_objective_bound: parsed.ortools_objective_bound,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => ortools_empty_solution(
            ExternalKnapsackReferenceStatus::NumericalError,
            format!(
                "failed to parse OR-Tools knapsack adapter output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_knapsack_with_external_reference(
    problem: &KnapsackProblem,
    opts: &ExternalKnapsackReferenceOptions,
) -> ExternalKnapsackReferenceSolution {
    if should_use_rust_knapsack_reference(opts) || should_use_registered_knapsack_fallback(opts) {
        return relabel_registered_knapsack_fallback(
            solve_knapsack_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_knapsack_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::knapsack::{
        build_sample_knapsack_problem, KnapsackItem, KnapsackProblem,
    };
    use std::sync::Mutex;

    static KNAPSACK_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn rust_reference_solves_sample_knapsack() {
        let problem = build_sample_knapsack_problem();
        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions {
                solver: ExternalKnapsackReferenceSolver::RustBranchAndBound,
            },
        );

        assert_eq!(solution.status, ExternalKnapsackReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:branch-and-bound-knapsack");
        assert_eq!(solution.selected_item_ids, vec!["B", "C", "D"]);
        assert_eq!(solution.total_weight, Some(26.0));
        assert_eq!(solution.total_value, Some(51.0));
        assert_eq!(solution.objective, Some(51.0));
        assert!(solution.upper_bound.is_some());
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_with_tie_breaking() {
        let problem = KnapsackProblem {
            capacity: 5.0,
            items: vec![
                KnapsackItem {
                    id: "A".to_string(),
                    weight: 5.0,
                    value: 10.0,
                },
                KnapsackItem {
                    id: "B".to_string(),
                    weight: 4.0,
                    value: 10.0,
                },
                KnapsackItem {
                    id: "C".to_string(),
                    weight: 1.0,
                    value: 0.0,
                },
            ],
        };

        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions {
                solver: ExternalKnapsackReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalKnapsackReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:branch-and-bound-knapsack");
        assert_eq!(solution.selected_item_ids, vec!["B"]);
        assert_eq!(solution.total_weight, Some(4.0));
        assert_eq!(solution.total_value, Some(10.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_knapsack_problem();

        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalKnapsackReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:branch-and-bound-knapsack");
        assert_eq!(solution.selected_item_ids, vec!["B", "C", "D"]);
        assert_eq!(solution.objective, Some(51.0));
    }

    #[test]
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = KNAPSACK_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _guard = EnvVarGuard::set("KNAPSACK_REFERENCE_REGISTERED_FALLBACK", "rust");
        let problem = build_sample_knapsack_problem();

        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions {
                solver: ExternalKnapsackReferenceSolver::OrTools,
            },
        );

        assert_eq!(solution.status, ExternalKnapsackReferenceStatus::Optimal);
        assert_eq!(
            solution.solver,
            "rust:registered-knapsack-fallback-for-ortools"
        );
        assert_eq!(solution.selected_item_ids, vec!["B", "C", "D"]);
        assert_eq!(solution.objective, Some(51.0));
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn ortools_adapter_rejects_unscaled_values_without_python() {
        let _lock = KNAPSACK_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _fallback_guard = EnvVarGuard::set("KNAPSACK_REFERENCE_REGISTERED_FALLBACK", "0");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = KnapsackProblem {
            capacity: 1.0 / 3.0,
            items: vec![KnapsackItem {
                id: "A".to_string(),
                weight: 1.0 / 3.0,
                value: 1.0,
            }],
        };

        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions {
                solver: ExternalKnapsackReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalKnapsackReferenceStatus::Unsupported
        );
        assert_eq!(solution.solver, "ortools:cp-sat-knapsack");
        assert!(solution
            .message
            .contains("requires integer-scalable weights/capacity and values"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = KNAPSACK_REFERENCE_ENV_LOCK.lock().expect("lock env guard");
        let _fallback_guard = EnvVarGuard::set("KNAPSACK_REFERENCE_REGISTERED_FALLBACK", "0");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = build_sample_knapsack_problem();

        let solution = solve_knapsack_with_external_reference(
            &problem,
            &ExternalKnapsackReferenceOptions {
                solver: ExternalKnapsackReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalKnapsackReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, "ortools:cp-sat-knapsack");
        assert!(solution.message.contains("OR-Tools knapsack adapter"));
        assert!(!solution.message.contains("knapsack_reference.py"));
    }

    #[test]
    fn knapsack_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_knapsack_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }

    #[test]
    fn knapsack_adapter_wait_observes_closed_stdin() {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("cat >/dev/null; printf done")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn stdin reader");
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(b"{\"capacity\":1,\"items\":[{\"id\":\"A\",\"weight\":1,\"value\":1}]}")
            .expect("write stdin");
        drop(child.stdin.take());

        let (output, timed_out) =
            wait_for_knapsack_reference_output(child, 1_000).expect("closed stdin output");

        assert!(!timed_out);
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout), "done");
    }
}
