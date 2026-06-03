//! Rust-facing bridge for external/reference graph-coloring solvers.
//!
//! The native Rust reference computes an exact DSATUR check without Python
//! startup. The Python bridge (`scripts/graph_coloring_reference.py`) remains
//! available for OR-Tools CP-SAT.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::graph_coloring::GraphColoringProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalGraphColoringReferenceSolver {
    Auto,
    RustDsatur,
    OrTools,
    Fallback,
}

impl ExternalGraphColoringReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalGraphColoringReferenceSolver::Auto => "auto",
            ExternalGraphColoringReferenceSolver::RustDsatur => "rust-dsatur",
            ExternalGraphColoringReferenceSolver::OrTools => "ortools",
            ExternalGraphColoringReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalGraphColoringReferenceOptions {
    pub solver: ExternalGraphColoringReferenceSolver,
}

impl Default for ExternalGraphColoringReferenceOptions {
    fn default() -> Self {
        ExternalGraphColoringReferenceOptions {
            solver: ExternalGraphColoringReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalGraphColoringReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalGraphColoringReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalGraphColoringReferenceStatus::Optimal => "optimal",
            ExternalGraphColoringReferenceStatus::Feasible => "feasible",
            ExternalGraphColoringReferenceStatus::Infeasible => "infeasible",
            ExternalGraphColoringReferenceStatus::Unsupported => "unsupported",
            ExternalGraphColoringReferenceStatus::NumericalError => "numerical-error",
            ExternalGraphColoringReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalGraphColoringReferenceSolution {
    pub status: ExternalGraphColoringReferenceStatus,
    pub solver: String,
    pub color_indices: Vec<usize>,
    pub color_names: Vec<String>,
    pub used_color_count: Option<usize>,
    pub objective: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_color_indices: Vec<usize>,
    pub ortools_color_names: Vec<String>,
    pub ortools_used_color_count: Option<usize>,
    pub ortools_objective: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct GraphColoringReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "colorIndices")]
    color_indices: Option<Vec<usize>>,
    #[serde(rename = "colorNames")]
    color_names: Option<Vec<String>>,
    #[serde(rename = "usedColorCount")]
    used_color_count: Option<usize>,
    objective: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsColorIndices")]
    ortools_color_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsColorNames")]
    ortools_color_names: Option<Vec<String>>,
    #[serde(rename = "ortoolsUsedColorCount")]
    ortools_used_color_count: Option<usize>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalGraphColoringReferenceStatus {
    match status {
        "optimal" => ExternalGraphColoringReferenceStatus::Optimal,
        "feasible" => ExternalGraphColoringReferenceStatus::Feasible,
        "infeasible" => ExternalGraphColoringReferenceStatus::Infeasible,
        "unsupported" => ExternalGraphColoringReferenceStatus::Unsupported,
        "unavailable" => ExternalGraphColoringReferenceStatus::Unavailable,
        _ => ExternalGraphColoringReferenceStatus::NumericalError,
    }
}

const RUST_GRAPH_COLORING_UNCOLORED: usize = usize::MAX;
const RUST_GRAPH_COLORING_MAX_EXACT_VERTICES: usize = 40;

fn validate_rust_graph_coloring_problem(
    problem: &GraphColoringProblem,
) -> Result<HashMap<String, usize>, String> {
    if problem.vertices.is_empty() {
        return Err("vertices must be non-empty".to_string());
    }
    let mut vertex_index = HashMap::with_capacity(problem.vertices.len());
    for (index, vertex) in problem.vertices.iter().enumerate() {
        if vertex.trim().is_empty() {
            return Err("vertices must be non-empty strings".to_string());
        }
        if vertex_index.insert(vertex.clone(), index).is_some() {
            return Err("vertices must be unique".to_string());
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

fn rust_graph_coloring_empty_solution(
    status: ExternalGraphColoringReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    ExternalGraphColoringReferenceSolution {
        status,
        solver: solver.into(),
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        objective: None,
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_graph_coloring_adjacency(
    problem: &GraphColoringProblem,
    vertex_index: &HashMap<String, usize>,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); problem.vertices.len()];
    for (from, to) in &problem.edges {
        let from_index = vertex_index[from];
        let to_index = vertex_index[to];
        adjacency[from_index].push(to_index);
        adjacency[to_index].push(from_index);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn rust_graph_coloring_color_names(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("C{}", index + 1)).collect()
}

fn rust_graph_coloring_solution(
    status: ExternalGraphColoringReferenceStatus,
    colors: Vec<usize>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    let used_color_count = colors
        .iter()
        .copied()
        .filter(|&color| color != RUST_GRAPH_COLORING_UNCOLORED)
        .max()
        .map(|color| color + 1)
        .unwrap_or(0);
    ExternalGraphColoringReferenceSolution {
        status,
        solver: "rust:dsatur-graph-coloring".to_string(),
        color_indices: colors,
        color_names: rust_graph_coloring_color_names(used_color_count),
        used_color_count: Some(used_color_count),
        objective: Some(used_color_count as f64),
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn rust_graph_coloring_greedy(
    problem: &GraphColoringProblem,
    adjacency: &[Vec<usize>],
) -> Vec<usize> {
    let mut order = (0..problem.vertices.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        adjacency[right]
            .len()
            .cmp(&adjacency[left].len())
            .then_with(|| problem.vertices[left].cmp(&problem.vertices[right]))
    });

    let mut colors = vec![RUST_GRAPH_COLORING_UNCOLORED; problem.vertices.len()];
    for vertex in order {
        let unavailable = adjacency[vertex]
            .iter()
            .filter_map(|&neighbor| {
                let color = colors[neighbor];
                (color != RUST_GRAPH_COLORING_UNCOLORED).then_some(color)
            })
            .collect::<HashSet<_>>();
        let mut color = 0;
        while unavailable.contains(&color) {
            color += 1;
        }
        colors[vertex] = color;
    }
    colors
}

fn rust_graph_coloring_select_dsatur_vertex(
    adjacency: &[Vec<usize>],
    colors: &[usize],
) -> Option<usize> {
    let mut best: Option<(usize, usize, usize)> = None;
    for vertex in 0..colors.len() {
        if colors[vertex] != RUST_GRAPH_COLORING_UNCOLORED {
            continue;
        }
        let saturation = adjacency[vertex]
            .iter()
            .filter_map(|&neighbor| {
                let color = colors[neighbor];
                (color != RUST_GRAPH_COLORING_UNCOLORED).then_some(color)
            })
            .collect::<HashSet<_>>()
            .len();
        let degree = adjacency[vertex].len();
        if best.is_none_or(|(_, best_saturation, best_degree)| {
            saturation > best_saturation || (saturation == best_saturation && degree > best_degree)
        }) {
            best = Some((vertex, saturation, degree));
        }
    }
    best.map(|(vertex, _, _)| vertex)
}

fn rust_graph_coloring_can_use_color(
    adjacency: &[Vec<usize>],
    colors: &[usize],
    vertex: usize,
    color: usize,
) -> bool {
    adjacency[vertex]
        .iter()
        .all(|&neighbor| colors[neighbor] != color)
}

fn rust_graph_coloring_dsatur(
    adjacency: &[Vec<usize>],
    max_colors: usize,
    colors: &mut [usize],
    used_colors: usize,
) -> bool {
    let Some(vertex) = rust_graph_coloring_select_dsatur_vertex(adjacency, colors) else {
        return true;
    };
    for color in 0..used_colors.saturating_add(1).min(max_colors) {
        if !rust_graph_coloring_can_use_color(adjacency, colors, vertex, color) {
            continue;
        }
        colors[vertex] = color;
        if rust_graph_coloring_dsatur(adjacency, max_colors, colors, used_colors.max(color + 1)) {
            return true;
        }
        colors[vertex] = RUST_GRAPH_COLORING_UNCOLORED;
    }
    false
}

fn solve_graph_coloring_with_rust_reference(
    problem: &GraphColoringProblem,
) -> ExternalGraphColoringReferenceSolution {
    let started = Instant::now();
    let vertex_index = match validate_rust_graph_coloring_problem(problem) {
        Ok(vertex_index) => vertex_index,
        Err(message) => {
            return rust_graph_coloring_empty_solution(
                ExternalGraphColoringReferenceStatus::NumericalError,
                "rust:dsatur-graph-coloring",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    if problem.vertices.len() > RUST_GRAPH_COLORING_MAX_EXACT_VERTICES {
        return rust_graph_coloring_empty_solution(
            ExternalGraphColoringReferenceStatus::Unsupported,
            "rust:dsatur-graph-coloring",
            format!(
                "exact graph-coloring only practical for <= {RUST_GRAPH_COLORING_MAX_EXACT_VERTICES} vertices, got {}",
                problem.vertices.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let adjacency = rust_graph_coloring_adjacency(problem, &vertex_index);
    let greedy = rust_graph_coloring_greedy(problem, &adjacency);
    let upper = greedy
        .iter()
        .copied()
        .filter(|&color| color != RUST_GRAPH_COLORING_UNCOLORED)
        .max()
        .map(|color| color + 1)
        .unwrap_or_else(|| problem.vertices.len().max(1));
    let lower = if problem.edges.is_empty() { 1 } else { 2 };
    for color_count in lower..=upper {
        let mut colors = vec![RUST_GRAPH_COLORING_UNCOLORED; problem.vertices.len()];
        if rust_graph_coloring_dsatur(&adjacency, color_count, &mut colors, 0) {
            return rust_graph_coloring_solution(
                ExternalGraphColoringReferenceStatus::Optimal,
                colors,
                "exact DSATUR-style chromatic search",
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    rust_graph_coloring_empty_solution(
        ExternalGraphColoringReferenceStatus::Infeasible,
        "rust:dsatur-graph-coloring",
        "no coloring found",
        started.elapsed().as_secs_f64() * 1000.0,
    )
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    ExternalGraphColoringReferenceSolution {
        status: ExternalGraphColoringReferenceStatus::Unavailable,
        solver: "external-graph-coloring-reference".to_string(),
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        objective: None,
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    ExternalGraphColoringReferenceSolution {
        status: ExternalGraphColoringReferenceStatus::NumericalError,
        solver: "external-graph-coloring-reference".to_string(),
        color_indices: Vec::new(),
        color_names: Vec::new(),
        used_color_count: None,
        objective: None,
        ortools_status: None,
        ortools_color_indices: Vec::new(),
        ortools_color_names: Vec::new(),
        ortools_used_color_count: None,
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
    root.join("scripts").join("graph_coloring_reference.py")
}

fn run_graph_coloring_reference_json(
    payload: Value,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
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
                format!("failed to start graph_coloring_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write graph_coloring_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for graph_coloring_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<GraphColoringReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalGraphColoringReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-graph-coloring-reference".to_string()),
            color_indices: parsed.color_indices.unwrap_or_default(),
            color_names: parsed.color_names.unwrap_or_default(),
            used_color_count: parsed.used_color_count,
            objective: parsed.objective,
            ortools_status: parsed.ortools_status,
            ortools_color_indices: parsed.ortools_color_indices.unwrap_or_default(),
            ortools_color_names: parsed.ortools_color_names.unwrap_or_default(),
            ortools_used_color_count: parsed.ortools_used_color_count,
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
                "failed to parse graph_coloring_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_graph_coloring_with_external_reference(
    problem: &GraphColoringProblem,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
    if matches!(
        opts.solver,
        ExternalGraphColoringReferenceSolver::RustDsatur
            | ExternalGraphColoringReferenceSolver::Fallback
    ) {
        return solve_graph_coloring_with_rust_reference(problem);
    }

    run_graph_coloring_reference_json(
        json!({
            "vertices": &problem.vertices,
            "edges": problem.edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::graph_coloring::{
        build_sample_graph_coloring_problem, GraphColoringProblem,
    };

    #[test]
    fn rust_reference_solves_sample_graph_coloring() {
        let problem = build_sample_graph_coloring_problem();
        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions {
                solver: ExternalGraphColoringReferenceSolver::RustDsatur,
            },
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:dsatur-graph-coloring");
        assert_eq!(solution.used_color_count, Some(3));
        assert_eq!(solution.objective, Some(3.0));
        assert_eq!(solution.color_names, vec!["C1", "C2", "C3"]);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_triangle() {
        let problem = GraphColoringProblem {
            vertices: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            edges: vec![
                ("A".to_string(), "B".to_string()),
                ("B".to_string(), "C".to_string()),
                ("A".to_string(), "C".to_string()),
            ],
        };

        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions {
                solver: ExternalGraphColoringReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:dsatur-graph-coloring");
        assert_eq!(solution.used_color_count, Some(3));
        assert_eq!(solution.color_indices.len(), 3);
    }
}
