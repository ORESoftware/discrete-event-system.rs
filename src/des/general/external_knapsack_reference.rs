//! Rust-facing bridge for external/reference 0/1 knapsack solvers.
//!
//! The native Rust reference computes an independent exact branch-and-bound
//! check without Python startup. The Python bridge (`scripts/knapsack_reference.py`)
//! remains available for OR-Tools CP-SAT.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
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

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalKnapsackReferenceSolution {
    ExternalKnapsackReferenceSolution {
        status: ExternalKnapsackReferenceStatus::Unavailable,
        solver: "external-knapsack-reference".to_string(),
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

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalKnapsackReferenceSolution {
    ExternalKnapsackReferenceSolution {
        status: ExternalKnapsackReferenceStatus::NumericalError,
        solver: "external-knapsack-reference".to_string(),
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

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("knapsack_reference.py")
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
            Err(err) => return Err(format!("failed to poll knapsack_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for knapsack_reference.py: {err}"))
}

fn run_knapsack_reference_json(
    payload: Value,
    opts: &ExternalKnapsackReferenceOptions,
) -> ExternalKnapsackReferenceSolution {
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
                format!("failed to start knapsack_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write knapsack_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = knapsack_reference_timeout_ms();
    let (mut output, timed_out) = match wait_for_knapsack_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0);
        }
    };
    if timed_out {
        let timeout_message = format!("knapsack_reference.py timed out after {timeout_ms}ms");
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
        Err(err) => numerical_error(
            format!(
                "failed to parse knapsack_reference.py output: {err}; stderr={}",
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
    if matches!(
        opts.solver,
        ExternalKnapsackReferenceSolver::Auto
            | ExternalKnapsackReferenceSolver::RustBranchAndBound
            | ExternalKnapsackReferenceSolver::Fallback
    ) {
        return solve_knapsack_with_rust_reference(problem);
    }

    run_knapsack_reference_json(
        json!({
            "capacity": problem.capacity,
            "items": problem.items.iter().map(|item| json!({
                "id": &item.id,
                "weight": item.weight,
                "value": item.value,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::knapsack::{
        build_sample_knapsack_problem, KnapsackItem, KnapsackProblem,
    };

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
    fn knapsack_python_bridge_wait_enforces_timeout() {
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
}
