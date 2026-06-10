//! Rust-facing bridge for external/reference graph-coloring solvers.
//!
//! The native Rust reference computes an exact DSATUR check without Python
//! startup. Registered OR-Tools aliases default to that Rust reference;
//! explicit force-Python switches keep the inline OR-Tools adapter available
//! for compatibility validation.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn graph_coloring_reference_force_python_value(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        normalized.as_str(),
        "1" | "true"
            | "yes"
            | "y"
            | "on"
            | "bridge"
            | "legacy-python"
            | "python-reference"
            | "python-bridge"
            | "legacy"
            | "compat"
            | "compatibility"
    )
}

fn graph_coloring_python_reference_forced() -> bool {
    [
        "GRAPH_COLORING_REFERENCE_FORCE_PYTHON",
        "GRAPH_COLORING_REFERENCE_ORTOOLS_FORCE_PYTHON",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| graph_coloring_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_graph_coloring_reference(opts: &ExternalGraphColoringReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalGraphColoringReferenceSolver::Auto
            | ExternalGraphColoringReferenceSolver::RustDsatur
            | ExternalGraphColoringReferenceSolver::Fallback
    )
}

fn should_use_registered_graph_coloring_fallback(
    opts: &ExternalGraphColoringReferenceOptions,
) -> bool {
    matches!(opts.solver, ExternalGraphColoringReferenceSolver::OrTools)
        && !graph_coloring_python_reference_forced()
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
    #[serde(rename = "objectiveBound")]
    objective_bound: Option<f64>,
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
const ORTOOLS_GRAPH_COLORING_SOLVER: &str = "ortools:cp-sat-graph-coloring";

const ORTOOLS_GRAPH_COLORING_ADAPTER: &str = r#"
import json
import sys

SOLVER = "ortools:cp-sat-graph-coloring"


def color_names(count):
    return [f"C{index + 1}" for index in range(count)]


def output(status, problem, colors=None, objective_bound=None, message=""):
    if colors is None:
        used = None
        names = []
        objective = None
        color_indices = []
    else:
        color_indices = [int(color) for color in colors]
        used = max(color_indices) + 1 if color_indices else 0
        names = color_names(used)
        objective = float(used)
    result = {
        "status": status,
        "solver": SOLVER,
        "colorIndices": color_indices,
        "colorNames": names,
        "usedColorCount": used,
        "objective": objective,
        "message": message,
    }
    if objective_bound is not None:
        result["objectiveBound"] = objective_bound
    return result


try:
    from ortools.sat.python import cp_model
except Exception as exc:
    print(json.dumps(output("unavailable", {"numVertices": 0}, None, None, str(exc))))
    sys.exit(0)


try:
    problem = json.load(sys.stdin)
    n = int(problem["numVertices"])
    model = cp_model.CpModel()
    colors = [model.NewIntVar(0, max(0, n - 1), f"color_v{index}") for index in range(n)]
    for ai, bi in problem["edges"]:
        model.Add(colors[int(ai)] != colors[int(bi)])
    max_color = model.NewIntVar(0, max(0, n - 1), "max_color")
    model.AddMaxEquality(max_color, colors)
    model.Minimize(max_color)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(json.dumps(output(
            "infeasible" if status_name == "infeasible" else status_name,
            problem,
            None,
            None,
            f"OR-Tools CP-SAT status {status_name}",
        )))
        sys.exit(0)
    assignment = [int(solver.Value(var)) for var in colors]
    print(json.dumps(output(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        problem,
        assignment,
        solver.BestObjectiveBound() + 1.0,
        f"OR-Tools CP-SAT status {status_name}",
    )))
except Exception as exc:
    print(json.dumps({
        "status": "error",
        "solver": SOLVER,
        "colorIndices": [],
        "colorNames": [],
        "usedColorCount": None,
        "objective": None,
        "message": str(exc),
    }))
    sys.exit(1)
"#;

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

fn ortools_graph_coloring_empty_solution(
    status: ExternalGraphColoringReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalGraphColoringReferenceSolution {
    rust_graph_coloring_empty_solution(status, ORTOOLS_GRAPH_COLORING_SOLVER, message, elapsed_ms)
}

fn ortools_graph_coloring_payload(
    problem: &GraphColoringProblem,
    vertex_index: &HashMap<String, usize>,
) -> Value {
    json!({
        "numVertices": problem.vertices.len(),
        "vertices": &problem.vertices,
        "edges": problem.edges.iter().map(|(a, b)| json!([
            vertex_index[a],
            vertex_index[b],
        ])).collect::<Vec<_>>(),
    })
}

fn relabel_registered_graph_coloring_fallback(
    mut solution: ExternalGraphColoringReferenceSolution,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
    if should_use_registered_graph_coloring_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-graph-coloring-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn graph_coloring_reference_timeout_ms() -> u64 {
    std::env::var("GRAPH_COLORING_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_graph_coloring_reference_output(
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
                    "failed to poll OR-Tools graph-coloring adapter: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools graph-coloring adapter: {err}"))
}

fn run_ortools_graph_coloring_reference(
    problem: &GraphColoringProblem,
) -> ExternalGraphColoringReferenceSolution {
    let started = Instant::now();
    let vertex_index = match validate_rust_graph_coloring_problem(problem) {
        Ok(vertex_index) => vertex_index,
        Err(message) => {
            return ortools_graph_coloring_empty_solution(
                ExternalGraphColoringReferenceStatus::NumericalError,
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let payload = ortools_graph_coloring_payload(problem, &vertex_index);
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_GRAPH_COLORING_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return ortools_graph_coloring_empty_solution(
                ExternalGraphColoringReferenceStatus::Unavailable,
                format!("failed to start OR-Tools graph-coloring adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return ortools_graph_coloring_empty_solution(
                ExternalGraphColoringReferenceStatus::NumericalError,
                format!("failed to write OR-Tools graph-coloring adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = graph_coloring_reference_timeout_ms();
    let (output, timed_out) = match wait_for_graph_coloring_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return ortools_graph_coloring_empty_solution(
                ExternalGraphColoringReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools graph-coloring adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools graph-coloring adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
            ortools_objective_bound: parsed.ortools_objective_bound.or(parsed.objective_bound),
            message: parsed.message.unwrap_or_else(|| {
                if output.status.success() {
                    "ok".to_string()
                } else {
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => ortools_graph_coloring_empty_solution(
            ExternalGraphColoringReferenceStatus::NumericalError,
            format!(
                "failed to parse OR-Tools graph-coloring adapter output: {err}; stderr={stderr}"
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_graph_coloring_with_external_reference(
    problem: &GraphColoringProblem,
    opts: &ExternalGraphColoringReferenceOptions,
) -> ExternalGraphColoringReferenceSolution {
    if should_use_rust_graph_coloring_reference(opts)
        || should_use_registered_graph_coloring_fallback(opts)
    {
        return relabel_registered_graph_coloring_fallback(
            solve_graph_coloring_with_rust_reference(problem),
            opts,
        );
    }

    run_ortools_graph_coloring_reference(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::graph_coloring::{
        build_sample_graph_coloring_problem, GraphColoringProblem,
    };

    use crate::des::shared::test_support::ENV_LOCK as GRAPH_COLORING_REFERENCE_ENV_LOCK;

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

    fn graph_coloring_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "GRAPH_COLORING_REFERENCE_FORCE_PYTHON",
            "GRAPH_COLORING_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    #[test]
    fn graph_coloring_force_python_requires_explicit_compatibility_value() {
        for value in [
            "1",
            "true",
            " yes ",
            "ON",
            "bridge",
            "python_reference",
            "python-bridge",
            "legacy-python",
            "legacy",
            "compatibility",
        ] {
            assert!(
                graph_coloring_reference_force_python_value(value),
                "{value:?} should enable the graph-coloring compatibility bridge"
            );
        }

        for value in [
            "", "0", "false", "off", "python", "py", "auto", "rust", "native",
        ] {
            assert!(
                !graph_coloring_reference_force_python_value(value),
                "{value:?} should keep Rust graph-coloring fallback active"
            );
        }
    }

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

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_graph_coloring_problem();

        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:dsatur-graph-coloring");
        assert_eq!(solution.used_color_count, Some(3));
        assert_eq!(solution.objective, Some(3.0));
    }

    #[test]
    fn registered_ortools_alias_defaults_to_rust_reference_without_python() {
        let _lock = GRAPH_COLORING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = graph_coloring_force_python_off_guards();
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-graph-coloring-alias",
        );
        let problem = build_sample_graph_coloring_problem();

        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions {
                solver: ExternalGraphColoringReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:registered-graph-coloring-fallback-for-ortools"
        );
        assert_eq!(solution.used_color_count, Some(3));
        assert_eq!(solution.objective, Some(3.0));
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn graph_coloring_force_python_keeps_ortools_bridge_available() {
        let _lock = GRAPH_COLORING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("GRAPH_COLORING_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-forced-graph-coloring",
        );
        let problem = build_sample_graph_coloring_problem();

        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions {
                solver: ExternalGraphColoringReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, ORTOOLS_GRAPH_COLORING_SOLVER);
        assert!(solution.message.contains("OR-Tools graph-coloring adapter"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = GRAPH_COLORING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("GRAPH_COLORING_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard = EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python");
        let problem = build_sample_graph_coloring_problem();

        let solution = solve_graph_coloring_with_external_reference(
            &problem,
            &ExternalGraphColoringReferenceOptions {
                solver: ExternalGraphColoringReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalGraphColoringReferenceStatus::Unavailable
        );
        assert_eq!(solution.solver, ORTOOLS_GRAPH_COLORING_SOLVER);
        assert!(solution.message.contains("OR-Tools graph-coloring adapter"));
        assert!(!solution.message.contains("graph_coloring_reference.py"));
    }

    #[test]
    fn graph_coloring_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_graph_coloring_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
