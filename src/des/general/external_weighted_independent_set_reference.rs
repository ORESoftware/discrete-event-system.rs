//! Rust-facing bridge for external/reference weighted independent-set solvers.
//!
//! The native Rust reference computes a deterministic exact branch-and-bound
//! check without Python startup. The Python bridge
//! (`scripts/weighted_independent_set_reference.py`) remains available for
//! OR-Tools CP-SAT.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::weighted_independent_set::WeightedIndependentSetProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedIndependentSetReferenceSolver {
    Auto,
    RustBranchAndBound,
    OrTools,
    Fallback,
}

impl ExternalWeightedIndependentSetReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalWeightedIndependentSetReferenceSolver::Auto => "auto",
            ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound => {
                "rust-branch-and-bound"
            }
            ExternalWeightedIndependentSetReferenceSolver::OrTools => "ortools",
            ExternalWeightedIndependentSetReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedIndependentSetReferenceOptions {
    pub solver: ExternalWeightedIndependentSetReferenceSolver,
}

impl Default for ExternalWeightedIndependentSetReferenceOptions {
    fn default() -> Self {
        ExternalWeightedIndependentSetReferenceOptions {
            solver: ExternalWeightedIndependentSetReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalWeightedIndependentSetReferenceStatus {
    Optimal,
    Feasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalWeightedIndependentSetReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalWeightedIndependentSetReferenceStatus::Optimal => "optimal",
            ExternalWeightedIndependentSetReferenceStatus::Feasible => "feasible",
            ExternalWeightedIndependentSetReferenceStatus::Unsupported => "unsupported",
            ExternalWeightedIndependentSetReferenceStatus::NumericalError => "numerical-error",
            ExternalWeightedIndependentSetReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalWeightedIndependentSetReferenceSolution {
    pub status: ExternalWeightedIndependentSetReferenceStatus,
    pub solver: String,
    pub selected_vertex_indices: Vec<usize>,
    pub selected_vertex_ids: Vec<String>,
    pub total_weight: Option<f64>,
    pub objective: Option<f64>,
    pub upper_bound: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_vertex_indices: Vec<usize>,
    pub ortools_selected_vertex_ids: Vec<String>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct WeightedIndependentSetReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedVertexIndices")]
    selected_vertex_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedVertexIds")]
    selected_vertex_ids: Option<Vec<String>>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    objective: Option<f64>,
    #[serde(rename = "upperBound")]
    upper_bound: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedVertexIndices")]
    ortools_selected_vertex_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedVertexIds")]
    ortools_selected_vertex_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalWeightedIndependentSetReferenceStatus {
    match status {
        "optimal" => ExternalWeightedIndependentSetReferenceStatus::Optimal,
        "feasible" => ExternalWeightedIndependentSetReferenceStatus::Feasible,
        "unsupported" => ExternalWeightedIndependentSetReferenceStatus::Unsupported,
        "unavailable" => ExternalWeightedIndependentSetReferenceStatus::Unavailable,
        _ => ExternalWeightedIndependentSetReferenceStatus::NumericalError,
    }
}

#[derive(Clone, Debug)]
struct RustWisSearchVertex {
    index: usize,
    weight: f64,
}

const RUST_WIS_EPS: f64 = 1e-9;
const RUST_WIS_MAX_EXACT_VERTICES: usize = 64;

fn validate_rust_weighted_independent_set_problem(
    problem: &WeightedIndependentSetProblem,
) -> Result<HashMap<String, usize>, String> {
    if problem.vertices.is_empty() {
        return Err("vertices must be non-empty".to_string());
    }
    let mut vertex_index = HashMap::with_capacity(problem.vertices.len());
    for (index, vertex) in problem.vertices.iter().enumerate() {
        if vertex.id.trim().is_empty() {
            return Err(format!("vertices[{index}].id must be non-empty"));
        }
        if vertex_index.insert(vertex.id.clone(), index).is_some() {
            return Err(format!("duplicate vertex id {:?}", vertex.id));
        }
        if !vertex.weight.is_finite() || vertex.weight < 0.0 {
            return Err(format!(
                "vertices[{index}].weight must be finite and non-negative"
            ));
        }
    }

    let mut seen_edges = HashSet::new();
    for (edge_index, (from, to)) in problem.edges.iter().enumerate() {
        let Some(&from_index) = vertex_index.get(from) else {
            return Err(format!(
                "edges[{edge_index}] endpoints must belong to vertices"
            ));
        };
        let Some(&to_index) = vertex_index.get(to) else {
            return Err(format!(
                "edges[{edge_index}] endpoints must belong to vertices"
            ));
        };
        if from_index == to_index {
            return Err(format!("edges[{edge_index}] must not be a self-loop"));
        }
        let key = if from_index < to_index {
            (from_index, to_index)
        } else {
            (to_index, from_index)
        };
        if !seen_edges.insert(key) {
            return Err(format!("duplicate undirected edge {from:?}-{to:?}"));
        }
    }

    Ok(vertex_index)
}

fn rust_wis_empty_solution(
    status: ExternalWeightedIndependentSetReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    ExternalWeightedIndependentSetReferenceSolution {
        status,
        solver: solver.into(),
        selected_vertex_indices: Vec::new(),
        selected_vertex_ids: Vec::new(),
        total_weight: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_wis_adjacency(
    problem: &WeightedIndependentSetProblem,
    vertex_index: &HashMap<String, usize>,
) -> Vec<Vec<bool>> {
    let mut adjacency = vec![vec![false; problem.vertices.len()]; problem.vertices.len()];
    for (from, to) in &problem.edges {
        let from_index = vertex_index[from];
        let to_index = vertex_index[to];
        adjacency[from_index][to_index] = true;
        adjacency[to_index][from_index] = true;
    }
    adjacency
}

fn rust_wis_sorted_vertices(problem: &WeightedIndependentSetProblem) -> Vec<RustWisSearchVertex> {
    let mut vertices = problem
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| RustWisSearchVertex {
            index,
            weight: vertex.weight,
        })
        .collect::<Vec<_>>();
    vertices.sort_by(|left, right| {
        right.weight.total_cmp(&left.weight).then_with(|| {
            problem.vertices[left.index]
                .id
                .cmp(&problem.vertices[right.index].id)
        })
    });
    vertices
}

fn rust_wis_compatible(adjacency: &[Vec<bool>], vertex: usize, selected: &[usize]) -> bool {
    selected.iter().all(|&other| !adjacency[vertex][other])
}

fn rust_wis_candidate_better(
    problem: &WeightedIndependentSetProblem,
    weight: f64,
    indices: &[usize],
    best_weight: f64,
    best_indices: &[usize],
) -> bool {
    if weight > best_weight + RUST_WIS_EPS {
        return true;
    }
    if (weight - best_weight).abs() <= RUST_WIS_EPS && indices.len() < best_indices.len() {
        return true;
    }
    if (weight - best_weight).abs() <= RUST_WIS_EPS && indices.len() == best_indices.len() {
        let mut left = indices
            .iter()
            .map(|&index| problem.vertices[index].id.clone())
            .collect::<Vec<_>>();
        let mut right = best_indices
            .iter()
            .map(|&index| problem.vertices[index].id.clone())
            .collect::<Vec<_>>();
        left.sort();
        right.sort();
        return left < right;
    }
    false
}

fn rust_wis_solution(
    problem: &WeightedIndependentSetProblem,
    status: ExternalWeightedIndependentSetReferenceStatus,
    mut selected_vertex_indices: Vec<usize>,
    upper_bound: Option<f64>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    selected_vertex_indices.sort_unstable();
    let selected_vertex_ids = selected_vertex_indices
        .iter()
        .map(|&index| problem.vertices[index].id.clone())
        .collect::<Vec<_>>();
    let total_weight = selected_vertex_indices
        .iter()
        .map(|&index| problem.vertices[index].weight)
        .sum::<f64>();
    ExternalWeightedIndependentSetReferenceSolution {
        status,
        solver: "rust:branch-and-bound-weighted-independent-set".to_string(),
        selected_vertex_indices,
        selected_vertex_ids,
        total_weight: Some(total_weight),
        objective: Some(total_weight),
        upper_bound,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_wis_greedy(adjacency: &[Vec<bool>], order: &[RustWisSearchVertex]) -> Vec<usize> {
    let mut selected = Vec::new();
    for vertex in order {
        if rust_wis_compatible(adjacency, vertex.index, &selected) {
            selected.push(vertex.index);
        }
    }
    selected
}

#[allow(clippy::too_many_arguments)]
fn rust_wis_exact_search(
    problem: &WeightedIndependentSetProblem,
    adjacency: &[Vec<bool>],
    order: &[RustWisSearchVertex],
    suffix_weight: &[f64],
    pos: usize,
    current_weight: f64,
    current: &mut Vec<usize>,
    best_indices: &mut Vec<usize>,
    best_weight: &mut f64,
) {
    if pos == order.len() {
        if rust_wis_candidate_better(problem, current_weight, current, *best_weight, best_indices) {
            *best_indices = current.clone();
            *best_weight = current_weight;
        }
        return;
    }
    if current_weight + suffix_weight[pos] + RUST_WIS_EPS < *best_weight {
        return;
    }

    let vertex = &order[pos];
    if rust_wis_compatible(adjacency, vertex.index, current) {
        current.push(vertex.index);
        rust_wis_exact_search(
            problem,
            adjacency,
            order,
            suffix_weight,
            pos + 1,
            current_weight + vertex.weight,
            current,
            best_indices,
            best_weight,
        );
        current.pop();
    }
    rust_wis_exact_search(
        problem,
        adjacency,
        order,
        suffix_weight,
        pos + 1,
        current_weight,
        current,
        best_indices,
        best_weight,
    );
}

fn solve_weighted_independent_set_with_rust_reference(
    problem: &WeightedIndependentSetProblem,
) -> ExternalWeightedIndependentSetReferenceSolution {
    let started = Instant::now();
    let vertex_index = match validate_rust_weighted_independent_set_problem(problem) {
        Ok(vertex_index) => vertex_index,
        Err(message) => {
            return rust_wis_empty_solution(
                ExternalWeightedIndependentSetReferenceStatus::NumericalError,
                "rust:branch-and-bound-weighted-independent-set",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    if problem.vertices.len() > RUST_WIS_MAX_EXACT_VERTICES {
        return rust_wis_empty_solution(
            ExternalWeightedIndependentSetReferenceStatus::Unsupported,
            "rust:branch-and-bound-weighted-independent-set",
            format!(
                "exact weighted independent set only practical for <= {RUST_WIS_MAX_EXACT_VERTICES} vertices, got {}",
                problem.vertices.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let adjacency = rust_wis_adjacency(problem, &vertex_index);
    let order = rust_wis_sorted_vertices(problem);
    let mut suffix_weight = vec![0.0; order.len() + 1];
    for index in (0..order.len()).rev() {
        suffix_weight[index] = suffix_weight[index + 1] + order[index].weight;
    }
    let mut best_indices = rust_wis_greedy(&adjacency, &order);
    let mut best_weight = best_indices
        .iter()
        .map(|&index| problem.vertices[index].weight)
        .sum::<f64>();
    let mut current = Vec::new();
    rust_wis_exact_search(
        problem,
        &adjacency,
        &order,
        &suffix_weight,
        0,
        0.0,
        &mut current,
        &mut best_indices,
        &mut best_weight,
    );

    rust_wis_solution(
        problem,
        ExternalWeightedIndependentSetReferenceStatus::Optimal,
        best_indices,
        Some(suffix_weight[0]),
        "exact branch-and-bound weighted independent set",
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    ExternalWeightedIndependentSetReferenceSolution {
        status: ExternalWeightedIndependentSetReferenceStatus::Unavailable,
        solver: "external-weighted-independent-set-reference".to_string(),
        selected_vertex_indices: Vec::new(),
        selected_vertex_ids: Vec::new(),
        total_weight: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalWeightedIndependentSetReferenceSolution {
    ExternalWeightedIndependentSetReferenceSolution {
        status: ExternalWeightedIndependentSetReferenceStatus::NumericalError,
        solver: "external-weighted-independent-set-reference".to_string(),
        selected_vertex_indices: Vec::new(),
        selected_vertex_ids: Vec::new(),
        total_weight: None,
        objective: None,
        upper_bound: None,
        ortools_status: None,
        ortools_selected_vertex_indices: Vec::new(),
        ortools_selected_vertex_ids: Vec::new(),
        ortools_total_weight: None,
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
    root.join("scripts")
        .join("weighted_independent_set_reference.py")
}

fn weighted_independent_set_reference_timeout_ms() -> u64 {
    std::env::var("WEIGHTED_INDEPENDENT_SET_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_weighted_independent_set_reference_output(
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
                    "failed to poll weighted_independent_set_reference.py: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for weighted_independent_set_reference.py: {err}"))
}

fn run_weighted_independent_set_reference_json(
    payload: Value,
    opts: &ExternalWeightedIndependentSetReferenceOptions,
) -> ExternalWeightedIndependentSetReferenceSolution {
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
                format!(
                    "failed to start weighted_independent_set_reference.py with {python}: {err}"
                ),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write weighted_independent_set_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = weighted_independent_set_reference_timeout_ms();
    let (output, timed_out) =
        match wait_for_weighted_independent_set_reference_output(child, timeout_ms) {
            Ok(output) => output,
            Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
        };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("weighted_independent_set_reference.py timed out after {timeout_ms}ms")
        } else {
            format!(
                "{stderr}; weighted_independent_set_reference.py timed out after {timeout_ms}ms"
            )
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<WeightedIndependentSetReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalWeightedIndependentSetReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-weighted-independent-set-reference".to_string()),
            selected_vertex_indices: parsed.selected_vertex_indices.unwrap_or_default(),
            selected_vertex_ids: parsed.selected_vertex_ids.unwrap_or_default(),
            total_weight: parsed.total_weight,
            objective: parsed.objective,
            upper_bound: parsed.upper_bound,
            ortools_status: parsed.ortools_status,
            ortools_selected_vertex_indices: parsed
                .ortools_selected_vertex_indices
                .unwrap_or_default(),
            ortools_selected_vertex_ids: parsed.ortools_selected_vertex_ids.unwrap_or_default(),
            ortools_total_weight: parsed.ortools_total_weight,
            ortools_objective: parsed.ortools_objective,
            ortools_objective_bound: parsed.ortools_objective_bound,
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse weighted_independent_set_reference.py output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_weighted_independent_set_with_external_reference(
    problem: &WeightedIndependentSetProblem,
    opts: &ExternalWeightedIndependentSetReferenceOptions,
) -> ExternalWeightedIndependentSetReferenceSolution {
    if matches!(
        opts.solver,
        ExternalWeightedIndependentSetReferenceSolver::Auto
            | ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound
            | ExternalWeightedIndependentSetReferenceSolver::Fallback
    ) {
        return solve_weighted_independent_set_with_rust_reference(problem);
    }

    run_weighted_independent_set_reference_json(
        json!({
            "vertices": problem.vertices.iter().map(|vertex| json!({
                "id": &vertex.id,
                "weight": vertex.weight,
            })).collect::<Vec<_>>(),
            "edges": problem.edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::weighted_independent_set::{
        build_sample_weighted_independent_set_problem, WeightedIndependentSetProblem,
        WeightedIndependentSetVertex,
    };

    #[test]
    fn rust_reference_solves_sample_weighted_independent_set() {
        let problem = build_sample_weighted_independent_set_problem();
        let solution = solve_weighted_independent_set_with_external_reference(
            &problem,
            &ExternalWeightedIndependentSetReferenceOptions {
                solver: ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedIndependentSetReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:branch-and-bound-weighted-independent-set"
        );
        assert_eq!(solution.selected_vertex_ids, vec!["B", "D", "G"]);
        assert_eq!(solution.total_weight, Some(16.0));
        assert_eq!(solution.objective, Some(16.0));
        assert!(solution.upper_bound.is_some());
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_with_tie_breaking() {
        let problem = WeightedIndependentSetProblem {
            vertices: vec![
                WeightedIndependentSetVertex {
                    id: "A".to_string(),
                    weight: 5.0,
                },
                WeightedIndependentSetVertex {
                    id: "B".to_string(),
                    weight: 5.0,
                },
                WeightedIndependentSetVertex {
                    id: "C".to_string(),
                    weight: 0.0,
                },
            ],
            edges: vec![("A".to_string(), "B".to_string())],
        };

        let solution = solve_weighted_independent_set_with_external_reference(
            &problem,
            &ExternalWeightedIndependentSetReferenceOptions {
                solver: ExternalWeightedIndependentSetReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalWeightedIndependentSetReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:branch-and-bound-weighted-independent-set"
        );
        assert_eq!(solution.selected_vertex_ids, vec!["A"]);
        assert_eq!(solution.total_weight, Some(5.0));
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_weighted_independent_set_problem();

        let solution = solve_weighted_independent_set_with_external_reference(
            &problem,
            &ExternalWeightedIndependentSetReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalWeightedIndependentSetReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:branch-and-bound-weighted-independent-set"
        );
        assert_eq!(solution.selected_vertex_ids, vec!["B", "D", "G"]);
        assert_eq!(solution.objective, Some(16.0));
    }

    #[test]
    fn weighted_independent_set_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_weighted_independent_set_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
