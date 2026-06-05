//! Rust-facing bridge for external/reference minimum spanning tree solvers.
//!
//! The native Rust reference computes an independent Kruskal check without
//! Python startup. The Python bridge (`scripts/minimum_spanning_tree_reference.py`)
//! remains available for OR-Tools CP-SAT using a root-flow connectivity
//! formulation.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::minimum_spanning_tree::MinimumSpanningTreeProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinimumSpanningTreeReferenceSolver {
    Auto,
    RustKruskal,
    OrTools,
    Fallback,
}

impl ExternalMinimumSpanningTreeReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMinimumSpanningTreeReferenceSolver::Auto => "auto",
            ExternalMinimumSpanningTreeReferenceSolver::RustKruskal => "rust-kruskal",
            ExternalMinimumSpanningTreeReferenceSolver::OrTools => "ortools",
            ExternalMinimumSpanningTreeReferenceSolver::Fallback => "fallback",
        }
    }
}

fn registered_minimum_spanning_tree_rust_fallback_enabled() -> bool {
    std::env::var("MINIMUM_SPANNING_TREE_REFERENCE_REGISTERED_FALLBACK")
        .or_else(|_| std::env::var("MINIMUM_SPANNING_TREE_REFERENCE_EXTERNAL_FALLBACK"))
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "rust" | "fallback" | "rust-fallback"
            )
        })
        .unwrap_or(false)
}

fn should_use_rust_minimum_spanning_tree_reference(
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> bool {
    matches!(
        opts.solver,
        ExternalMinimumSpanningTreeReferenceSolver::Auto
            | ExternalMinimumSpanningTreeReferenceSolver::RustKruskal
            | ExternalMinimumSpanningTreeReferenceSolver::Fallback
    )
}

fn should_use_registered_minimum_spanning_tree_fallback(
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> bool {
    registered_minimum_spanning_tree_rust_fallback_enabled()
        && matches!(
            opts.solver,
            ExternalMinimumSpanningTreeReferenceSolver::OrTools
        )
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinimumSpanningTreeReferenceOptions {
    pub solver: ExternalMinimumSpanningTreeReferenceSolver,
}

impl Default for ExternalMinimumSpanningTreeReferenceOptions {
    fn default() -> Self {
        ExternalMinimumSpanningTreeReferenceOptions {
            solver: ExternalMinimumSpanningTreeReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMinimumSpanningTreeReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    NumericalError,
    Unavailable,
}

impl ExternalMinimumSpanningTreeReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalMinimumSpanningTreeReferenceStatus::Optimal => "optimal",
            ExternalMinimumSpanningTreeReferenceStatus::Feasible => "feasible",
            ExternalMinimumSpanningTreeReferenceStatus::Infeasible => "infeasible",
            ExternalMinimumSpanningTreeReferenceStatus::NumericalError => "numerical-error",
            ExternalMinimumSpanningTreeReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMinimumSpanningTreeReferenceSolution {
    pub status: ExternalMinimumSpanningTreeReferenceStatus,
    pub solver: String,
    pub selected_edge_indices: Vec<usize>,
    pub selected_edge_ids: Vec<String>,
    pub objective: Option<f64>,
    pub total_weight: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_selected_edge_indices: Vec<usize>,
    pub ortools_selected_edge_ids: Vec<String>,
    pub ortools_objective: Option<f64>,
    pub ortools_total_weight: Option<f64>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct MinimumSpanningTreeReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "selectedEdgeIndices")]
    selected_edge_indices: Option<Vec<usize>>,
    #[serde(rename = "selectedEdgeIds")]
    selected_edge_ids: Option<Vec<String>>,
    objective: Option<f64>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSelectedEdgeIndices")]
    ortools_selected_edge_indices: Option<Vec<usize>>,
    #[serde(rename = "ortoolsSelectedEdgeIds")]
    ortools_selected_edge_ids: Option<Vec<String>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<f64>,
    #[serde(rename = "ortoolsTotalWeight")]
    ortools_total_weight: Option<f64>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

fn status_from_str(status: &str) -> ExternalMinimumSpanningTreeReferenceStatus {
    match status {
        "optimal" => ExternalMinimumSpanningTreeReferenceStatus::Optimal,
        "feasible" => ExternalMinimumSpanningTreeReferenceStatus::Feasible,
        "infeasible" => ExternalMinimumSpanningTreeReferenceStatus::Infeasible,
        "unavailable" => ExternalMinimumSpanningTreeReferenceStatus::Unavailable,
        _ => ExternalMinimumSpanningTreeReferenceStatus::NumericalError,
    }
}

#[derive(Clone, Debug)]
struct RustDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl RustDisjointSet {
    fn new(size: usize) -> Self {
        RustDisjointSet {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return false;
        }
        if self.rank[root_a] < self.rank[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
        true
    }
}

fn validate_rust_minimum_spanning_tree_problem(
    problem: &MinimumSpanningTreeProblem,
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

    let mut seen_ids = HashSet::new();
    let mut seen_edges = HashSet::new();
    for (edge_index, edge) in problem.edges.iter().enumerate() {
        if edge.id.trim().is_empty() {
            return Err(format!("edges[{edge_index}].id must be non-empty"));
        }
        if !seen_ids.insert(edge.id.clone()) {
            return Err(format!("duplicate edge id {:?}", edge.id));
        }
        let Some(&from) = vertex_index.get(&edge.from) else {
            return Err(format!(
                "edges[{edge_index}] endpoints must belong to vertices"
            ));
        };
        let Some(&to) = vertex_index.get(&edge.to) else {
            return Err(format!(
                "edges[{edge_index}] endpoints must belong to vertices"
            ));
        };
        if from == to {
            return Err(format!("edges[{edge_index}] must not be a self-loop"));
        }
        if !edge.weight.is_finite() {
            return Err(format!("edges[{edge_index}].weight must be finite"));
        }
        let key = if from < to { (from, to) } else { (to, from) };
        if !seen_edges.insert(key) {
            return Err(format!(
                "duplicate undirected edge {:?}-{:?}",
                edge.from, edge.to
            ));
        }
    }

    Ok(vertex_index)
}

fn minimum_spanning_tree_empty_solution(
    status: ExternalMinimumSpanningTreeReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    ExternalMinimumSpanningTreeReferenceSolution {
        status,
        solver: solver.into(),
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        objective: None,
        total_weight: None,
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn relabel_registered_minimum_spanning_tree_fallback(
    mut solution: ExternalMinimumSpanningTreeReferenceSolution,
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    if should_use_registered_minimum_spanning_tree_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-minimum-spanning-tree-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
}

fn solve_minimum_spanning_tree_with_rust_reference(
    problem: &MinimumSpanningTreeProblem,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    let started = Instant::now();
    let vertex_index = match validate_rust_minimum_spanning_tree_problem(problem) {
        Ok(vertex_index) => vertex_index,
        Err(message) => {
            return minimum_spanning_tree_empty_solution(
                ExternalMinimumSpanningTreeReferenceStatus::NumericalError,
                "rust:kruskal-mst",
                message,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    if problem.vertices.len() == 1 {
        return ExternalMinimumSpanningTreeReferenceSolution {
            status: ExternalMinimumSpanningTreeReferenceStatus::Optimal,
            solver: "rust:kruskal-mst".to_string(),
            selected_edge_indices: Vec::new(),
            selected_edge_ids: Vec::new(),
            objective: Some(0.0),
            total_weight: Some(0.0),
            ortools_status: None,
            ortools_selected_edge_indices: Vec::new(),
            ortools_selected_edge_ids: Vec::new(),
            ortools_objective: None,
            ortools_total_weight: None,
            ortools_objective_bound: None,
            message: "single-vertex MST".to_string(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    let mut order = (0..problem.edges.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        problem.edges[left]
            .weight
            .total_cmp(&problem.edges[right].weight)
            .then_with(|| problem.edges[left].id.cmp(&problem.edges[right].id))
    });

    let mut dsu = RustDisjointSet::new(problem.vertices.len());
    let mut selected = Vec::new();
    for edge_index in order {
        let edge = &problem.edges[edge_index];
        let from = vertex_index[&edge.from];
        let to = vertex_index[&edge.to];
        if dsu.union(from, to) {
            selected.push(edge_index);
            if selected.len() + 1 == problem.vertices.len() {
                break;
            }
        }
    }

    if selected.len() + 1 != problem.vertices.len() {
        return minimum_spanning_tree_empty_solution(
            ExternalMinimumSpanningTreeReferenceStatus::Infeasible,
            "rust:kruskal-mst",
            "graph is disconnected",
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    selected.sort_unstable();
    let total_weight = selected
        .iter()
        .map(|&edge_index| problem.edges[edge_index].weight)
        .sum::<f64>();
    let selected_edge_ids = selected
        .iter()
        .map(|&edge_index| problem.edges[edge_index].id.clone())
        .collect();

    ExternalMinimumSpanningTreeReferenceSolution {
        status: ExternalMinimumSpanningTreeReferenceStatus::Optimal,
        solver: "rust:kruskal-mst".to_string(),
        selected_edge_indices: selected,
        selected_edge_ids,
        objective: Some(total_weight),
        total_weight: Some(total_weight),
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
        ortools_objective_bound: None,
        message: "Kruskal minimum spanning tree".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn unavailable(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    ExternalMinimumSpanningTreeReferenceSolution {
        status: ExternalMinimumSpanningTreeReferenceStatus::Unavailable,
        solver: "external-minimum-spanning-tree-reference".to_string(),
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        objective: None,
        total_weight: None,
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    ExternalMinimumSpanningTreeReferenceSolution {
        status: ExternalMinimumSpanningTreeReferenceStatus::NumericalError,
        solver: "external-minimum-spanning-tree-reference".to_string(),
        selected_edge_indices: Vec::new(),
        selected_edge_ids: Vec::new(),
        objective: None,
        total_weight: None,
        ortools_status: None,
        ortools_selected_edge_indices: Vec::new(),
        ortools_selected_edge_ids: Vec::new(),
        ortools_objective: None,
        ortools_total_weight: None,
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
        .join("minimum_spanning_tree_reference.py")
}

fn minimum_spanning_tree_reference_timeout_ms() -> u64 {
    std::env::var("MINIMUM_SPANNING_TREE_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_minimum_spanning_tree_reference_output(
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
                    "failed to poll minimum_spanning_tree_reference.py: {err}"
                ))
            }
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for minimum_spanning_tree_reference.py: {err}"))
}

fn run_minimum_spanning_tree_reference_json(
    payload: Value,
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> ExternalMinimumSpanningTreeReferenceSolution {
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
                format!("failed to start minimum_spanning_tree_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write minimum_spanning_tree_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = minimum_spanning_tree_reference_timeout_ms();
    let (output, timed_out) =
        match wait_for_minimum_spanning_tree_reference_output(child, timeout_ms) {
            Ok(output) => output,
            Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
        };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("minimum_spanning_tree_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; minimum_spanning_tree_reference.py timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<MinimumSpanningTreeReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalMinimumSpanningTreeReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-minimum-spanning-tree-reference".to_string()),
            selected_edge_indices: parsed.selected_edge_indices.unwrap_or_default(),
            selected_edge_ids: parsed.selected_edge_ids.unwrap_or_default(),
            objective: parsed.objective,
            total_weight: parsed.total_weight,
            ortools_status: parsed.ortools_status,
            ortools_selected_edge_indices: parsed.ortools_selected_edge_indices.unwrap_or_default(),
            ortools_selected_edge_ids: parsed.ortools_selected_edge_ids.unwrap_or_default(),
            ortools_objective: parsed.ortools_objective,
            ortools_total_weight: parsed.ortools_total_weight,
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
                "failed to parse minimum_spanning_tree_reference.py output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_minimum_spanning_tree_with_external_reference(
    problem: &MinimumSpanningTreeProblem,
    opts: &ExternalMinimumSpanningTreeReferenceOptions,
) -> ExternalMinimumSpanningTreeReferenceSolution {
    if should_use_rust_minimum_spanning_tree_reference(opts)
        || should_use_registered_minimum_spanning_tree_fallback(opts)
    {
        return relabel_registered_minimum_spanning_tree_fallback(
            solve_minimum_spanning_tree_with_rust_reference(problem),
            opts,
        );
    }

    run_minimum_spanning_tree_reference_json(
        json!({
            "vertices": &problem.vertices,
            "edges": problem.edges.iter().map(|edge| json!({
                "id": edge.id,
                "from": edge.from,
                "to": edge.to,
                "weight": edge.weight,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::minimum_spanning_tree::{
        build_sample_minimum_spanning_tree_problem, MinimumSpanningTreeEdge,
    };
    use std::sync::Mutex;

    static MINIMUM_SPANNING_TREE_REFERENCE_ENV_LOCK: Mutex<()> = Mutex::new(());

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
    fn rust_reference_solves_sample_mst() {
        let problem = build_sample_minimum_spanning_tree_problem();
        let solution = solve_minimum_spanning_tree_with_external_reference(
            &problem,
            &ExternalMinimumSpanningTreeReferenceOptions {
                solver: ExternalMinimumSpanningTreeReferenceSolver::RustKruskal,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinimumSpanningTreeReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:kruskal-mst");
        assert_eq!(solution.objective, Some(6.0));
        assert_eq!(solution.total_weight, Some(6.0));
        assert_eq!(solution.selected_edge_ids, vec!["AB", "BC", "CD", "DE"]);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_disconnected_graph() {
        let problem = MinimumSpanningTreeProblem {
            vertices: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            edges: vec![MinimumSpanningTreeEdge {
                id: "AB".to_string(),
                from: "A".to_string(),
                to: "B".to_string(),
                weight: 1.0,
            }],
        };

        let solution = solve_minimum_spanning_tree_with_external_reference(
            &problem,
            &ExternalMinimumSpanningTreeReferenceOptions {
                solver: ExternalMinimumSpanningTreeReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinimumSpanningTreeReferenceStatus::Infeasible
        );
        assert_eq!(solution.solver, "rust:kruskal-mst");
        assert!(solution.selected_edge_indices.is_empty());
        assert!(solution.objective.is_none());
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_sample_minimum_spanning_tree_problem();

        let solution = solve_minimum_spanning_tree_with_external_reference(
            &problem,
            &ExternalMinimumSpanningTreeReferenceOptions::default(),
        );

        assert_eq!(
            solution.status,
            ExternalMinimumSpanningTreeReferenceStatus::Optimal
        );
        assert_eq!(solution.solver, "rust:kruskal-mst");
        assert_eq!(solution.objective, Some(6.0));
        assert_eq!(solution.selected_edge_ids, vec!["AB", "BC", "CD", "DE"]);
    }

    #[test]
    fn registered_ortools_alias_can_use_rust_reference_without_python() {
        let _lock = MINIMUM_SPANNING_TREE_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _guard = EnvVarGuard::set(
            "MINIMUM_SPANNING_TREE_REFERENCE_REGISTERED_FALLBACK",
            "rust",
        );
        let problem = build_sample_minimum_spanning_tree_problem();

        let solution = solve_minimum_spanning_tree_with_external_reference(
            &problem,
            &ExternalMinimumSpanningTreeReferenceOptions {
                solver: ExternalMinimumSpanningTreeReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalMinimumSpanningTreeReferenceStatus::Optimal
        );
        assert_eq!(
            solution.solver,
            "rust:registered-minimum-spanning-tree-fallback-for-ortools"
        );
        assert_eq!(solution.objective, Some(6.0));
        assert_eq!(solution.selected_edge_ids, vec!["AB", "BC", "CD", "DE"]);
        assert!(solution
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
    }

    #[test]
    fn minimum_spanning_tree_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_minimum_spanning_tree_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
