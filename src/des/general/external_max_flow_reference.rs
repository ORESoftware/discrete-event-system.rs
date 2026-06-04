//! Rust-facing bridge for external/reference max-flow solvers.
//!
//! The native Rust reference computes an independent Edmonds-Karp check without
//! Python startup. The checked-in Python bridge (`scripts/max_flow_reference.py`)
//! remains available for OR-Tools SimpleMaxFlow when installed and when
//! capacities can be integer-scaled.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::max_flow::{MaxFlowEdgeFlow, MaxFlowProblem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMaxFlowReferenceSolver {
    Auto,
    RustEdmondsKarp,
    OrTools,
    Fallback,
}

impl ExternalMaxFlowReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMaxFlowReferenceSolver::Auto => "auto",
            ExternalMaxFlowReferenceSolver::RustEdmondsKarp => "rust-edmonds-karp",
            ExternalMaxFlowReferenceSolver::OrTools => "ortools",
            ExternalMaxFlowReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMaxFlowReferenceOptions {
    pub solver: ExternalMaxFlowReferenceSolver,
}

impl Default for ExternalMaxFlowReferenceOptions {
    fn default() -> Self {
        ExternalMaxFlowReferenceOptions {
            solver: ExternalMaxFlowReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMaxFlowReferenceStatus {
    Optimal,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalMaxFlowReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalMaxFlowReferenceStatus::Optimal => "optimal",
            ExternalMaxFlowReferenceStatus::Infeasible => "infeasible",
            ExternalMaxFlowReferenceStatus::Unsupported => "unsupported",
            ExternalMaxFlowReferenceStatus::NumericalError => "numerical-error",
            ExternalMaxFlowReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExternalMaxFlowReferenceCut {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
    pub cut_edges: Vec<MaxFlowEdgeFlow>,
    pub capacity: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalMaxFlowReferenceSolution {
    pub status: ExternalMaxFlowReferenceStatus,
    pub solver: String,
    pub max_flow: Option<f64>,
    pub edge_flows: Vec<MaxFlowEdgeFlow>,
    pub min_cut: ExternalMaxFlowReferenceCut,
    pub node_balance: Vec<f64>,
    pub iterations: Option<u64>,
    pub ortools_status: Option<String>,
    pub ortools_max_flow: Option<f64>,
    pub ortools_edge_flows: Vec<MaxFlowEdgeFlow>,
    pub ortools_min_cut: ExternalMaxFlowReferenceCut,
    pub ortools_node_balance: Vec<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct MaxFlowReferencePayload {
    status: String,
    solver: Option<String>,
    #[serde(rename = "maxFlow")]
    max_flow: Option<f64>,
    #[serde(rename = "edgeFlows")]
    edge_flows: Option<Vec<MaxFlowEdgeFlowPayload>>,
    #[serde(rename = "minCut")]
    min_cut: Option<MaxFlowCutPayload>,
    #[serde(rename = "nodeBalance")]
    node_balance: Option<Vec<f64>>,
    iterations: Option<u64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsMaxFlow")]
    ortools_max_flow: Option<f64>,
    #[serde(rename = "ortoolsEdgeFlows")]
    ortools_edge_flows: Option<Vec<MaxFlowEdgeFlowPayload>>,
    #[serde(rename = "ortoolsMinCut")]
    ortools_min_cut: Option<MaxFlowCutPayload>,
    #[serde(rename = "ortoolsNodeBalance")]
    ortools_node_balance: Option<Vec<f64>>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MaxFlowEdgeFlowPayload {
    from: usize,
    to: usize,
    capacity: f64,
    name: Option<String>,
    flow: f64,
}

impl From<MaxFlowEdgeFlowPayload> for MaxFlowEdgeFlow {
    fn from(value: MaxFlowEdgeFlowPayload) -> Self {
        MaxFlowEdgeFlow {
            from: value.from,
            to: value.to,
            capacity: value.capacity,
            name: value.name,
            flow: value.flow,
        }
    }
}

#[derive(Debug, Deserialize)]
struct MaxFlowCutPayload {
    #[serde(rename = "sourceSide")]
    source_side: Option<Vec<usize>>,
    #[serde(rename = "sinkSide")]
    sink_side: Option<Vec<usize>>,
    #[serde(rename = "cutEdges")]
    cut_edges: Option<Vec<MaxFlowEdgeFlowPayload>>,
    capacity: Option<f64>,
}

impl From<MaxFlowCutPayload> for ExternalMaxFlowReferenceCut {
    fn from(value: MaxFlowCutPayload) -> Self {
        ExternalMaxFlowReferenceCut {
            source_side: value.source_side.unwrap_or_default(),
            sink_side: value.sink_side.unwrap_or_default(),
            cut_edges: value
                .cut_edges
                .unwrap_or_default()
                .into_iter()
                .map(MaxFlowEdgeFlow::from)
                .collect(),
            capacity: value.capacity.unwrap_or(f64::NAN),
        }
    }
}

#[derive(Clone, Debug)]
struct RustResidualEdge {
    to: usize,
    rev: usize,
    cap: f64,
}

#[derive(Clone, Copy, Debug)]
struct RustForwardRef {
    from: usize,
    edge_index: usize,
}

const RUST_MAX_FLOW_EPS: f64 = 1e-12;

fn status_from_str(status: &str) -> ExternalMaxFlowReferenceStatus {
    match status {
        "optimal" => ExternalMaxFlowReferenceStatus::Optimal,
        "infeasible" => ExternalMaxFlowReferenceStatus::Infeasible,
        "unsupported" => ExternalMaxFlowReferenceStatus::Unsupported,
        "unavailable" => ExternalMaxFlowReferenceStatus::Unavailable,
        _ => ExternalMaxFlowReferenceStatus::NumericalError,
    }
}

fn validate_rust_max_flow_problem(problem: &MaxFlowProblem) -> Result<(), String> {
    if problem.num_nodes < 2 {
        return Err("numNodes must be at least 2".to_string());
    }
    if problem.source >= problem.num_nodes {
        return Err("source is outside node range".to_string());
    }
    if problem.sink >= problem.num_nodes {
        return Err("sink is outside node range".to_string());
    }
    if problem.source == problem.sink {
        return Err("source and sink must differ".to_string());
    }
    if problem.edges.is_empty() {
        return Err("edges must be non-empty".to_string());
    }
    for (index, edge) in problem.edges.iter().enumerate() {
        if edge.from >= problem.num_nodes || edge.to >= problem.num_nodes {
            return Err(format!("edge {index} endpoint is outside node range"));
        }
        if !edge.capacity.is_finite() || edge.capacity < 0.0 {
            return Err(format!(
                "edge {index} capacity must be finite and non-negative"
            ));
        }
    }
    Ok(())
}

fn max_flow_node_balance(num_nodes: usize, edge_flows: &[MaxFlowEdgeFlow]) -> Vec<f64> {
    let mut balance = vec![0.0; num_nodes];
    for edge in edge_flows {
        balance[edge.from] -= edge.flow;
        balance[edge.to] += edge.flow;
    }
    balance
}

fn rust_max_flow_cut(
    problem: &MaxFlowProblem,
    residual: &[Vec<RustResidualEdge>],
    edge_flows: &[MaxFlowEdgeFlow],
) -> ExternalMaxFlowReferenceCut {
    let mut seen = vec![false; problem.num_nodes];
    let mut queue = VecDeque::from([problem.source]);
    seen[problem.source] = true;
    while let Some(node) = queue.pop_front() {
        for edge in &residual[node] {
            if edge.cap > RUST_MAX_FLOW_EPS && !seen[edge.to] {
                seen[edge.to] = true;
                queue.push_back(edge.to);
            }
        }
    }

    let source_side: Vec<usize> = seen
        .iter()
        .enumerate()
        .filter_map(|(node, is_seen)| is_seen.then_some(node))
        .collect();
    let sink_side: Vec<usize> = seen
        .iter()
        .enumerate()
        .filter_map(|(node, is_seen)| (!is_seen).then_some(node))
        .collect();
    let cut_edges: Vec<MaxFlowEdgeFlow> = edge_flows
        .iter()
        .filter(|edge| seen[edge.from] && !seen[edge.to])
        .cloned()
        .collect();
    let capacity = cut_edges.iter().map(|edge| edge.capacity).sum();

    ExternalMaxFlowReferenceCut {
        source_side,
        sink_side,
        cut_edges,
        capacity,
    }
}

fn solve_max_flow_with_rust_reference(
    problem: &MaxFlowProblem,
) -> ExternalMaxFlowReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_max_flow_problem(problem) {
        return empty_solution(
            ExternalMaxFlowReferenceStatus::NumericalError,
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut residual = vec![Vec::<RustResidualEdge>::new(); problem.num_nodes];
    let mut forward_refs = Vec::<RustForwardRef>::with_capacity(problem.edges.len());
    for edge in &problem.edges {
        let forward_rev = residual[edge.to].len();
        let reverse_rev = residual[edge.from].len();
        residual[edge.from].push(RustResidualEdge {
            to: edge.to,
            rev: forward_rev,
            cap: edge.capacity,
        });
        residual[edge.to].push(RustResidualEdge {
            to: edge.from,
            rev: reverse_rev,
            cap: 0.0,
        });
        forward_refs.push(RustForwardRef {
            from: edge.from,
            edge_index: residual[edge.from].len() - 1,
        });
    }

    let mut max_flow = 0.0;
    let mut iterations = 0_u64;
    loop {
        let mut parent_node = vec![None; problem.num_nodes];
        let mut parent_edge = vec![None; problem.num_nodes];
        let mut queue = VecDeque::from([problem.source]);
        parent_node[problem.source] = Some(problem.source);
        let mut found_sink = false;

        while let Some(node) = queue.pop_front() {
            for (edge_index, edge) in residual[node].iter().enumerate() {
                if edge.cap <= RUST_MAX_FLOW_EPS || parent_node[edge.to].is_some() {
                    continue;
                }
                parent_node[edge.to] = Some(node);
                parent_edge[edge.to] = Some(edge_index);
                if edge.to == problem.sink {
                    found_sink = true;
                    break;
                }
                queue.push_back(edge.to);
            }
            if found_sink {
                break;
            }
        }

        if !found_sink {
            break;
        }

        let mut bottleneck = f64::INFINITY;
        let mut node = problem.sink;
        while node != problem.source {
            let prev = parent_node[node].expect("max-flow parent node missing after BFS");
            let edge_index = parent_edge[node].expect("max-flow parent edge missing after BFS");
            bottleneck = bottleneck.min(residual[prev][edge_index].cap);
            node = prev;
        }

        let mut node = problem.sink;
        while node != problem.source {
            let prev = parent_node[node].expect("max-flow parent node missing after BFS");
            let edge_index = parent_edge[node].expect("max-flow parent edge missing after BFS");
            let to = residual[prev][edge_index].to;
            let rev = residual[prev][edge_index].rev;
            residual[prev][edge_index].cap -= bottleneck;
            residual[to][rev].cap += bottleneck;
            node = prev;
        }

        iterations += 1;
        max_flow += bottleneck;
    }

    let edge_flows: Vec<MaxFlowEdgeFlow> = problem
        .edges
        .iter()
        .zip(forward_refs.iter())
        .map(|(edge, reference)| {
            let residual_capacity = residual[reference.from][reference.edge_index].cap;
            MaxFlowEdgeFlow {
                from: edge.from,
                to: edge.to,
                capacity: edge.capacity,
                name: edge.name.clone(),
                flow: edge.capacity - residual_capacity,
            }
        })
        .collect();
    let min_cut = rust_max_flow_cut(problem, &residual, &edge_flows);
    let node_balance = max_flow_node_balance(problem.num_nodes, &edge_flows);

    ExternalMaxFlowReferenceSolution {
        status: ExternalMaxFlowReferenceStatus::Optimal,
        solver: "rust:edmonds-karp-max-flow".to_string(),
        max_flow: Some(max_flow),
        edge_flows,
        min_cut,
        node_balance,
        iterations: Some(iterations),
        ortools_status: None,
        ortools_max_flow: None,
        ortools_edge_flows: Vec::new(),
        ortools_min_cut: ExternalMaxFlowReferenceCut::default(),
        ortools_node_balance: Vec::new(),
        message: "Rust Edmonds-Karp augmenting-path reference".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn empty_solution(
    status: ExternalMaxFlowReferenceStatus,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalMaxFlowReferenceSolution {
    ExternalMaxFlowReferenceSolution {
        status,
        solver: "external-max-flow-reference".to_string(),
        max_flow: None,
        edge_flows: Vec::new(),
        min_cut: ExternalMaxFlowReferenceCut::default(),
        node_balance: Vec::new(),
        iterations: None,
        ortools_status: None,
        ortools_max_flow: None,
        ortools_edge_flows: Vec::new(),
        ortools_min_cut: ExternalMaxFlowReferenceCut::default(),
        ortools_node_balance: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("max_flow_reference.py")
}

fn max_flow_reference_timeout_ms() -> u64 {
    std::env::var("MAX_FLOW_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_max_flow_reference_output(
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
            Err(err) => return Err(format!("failed to poll max_flow_reference.py: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for max_flow_reference.py: {err}"))
}

fn run_max_flow_reference_json(
    payload: Value,
    opts: &ExternalMaxFlowReferenceOptions,
) -> ExternalMaxFlowReferenceSolution {
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
            return empty_solution(
                ExternalMaxFlowReferenceStatus::Unavailable,
                format!("failed to start max_flow_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return empty_solution(
                ExternalMaxFlowReferenceStatus::NumericalError,
                format!("failed to write max_flow_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = max_flow_reference_timeout_ms();
    let (output, timed_out) = match wait_for_max_flow_reference_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => {
            return empty_solution(
                ExternalMaxFlowReferenceStatus::NumericalError,
                err,
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("max_flow_reference.py timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; max_flow_reference.py timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
    match serde_json::from_slice::<MaxFlowReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalMaxFlowReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-max-flow-reference".to_string()),
            max_flow: parsed.max_flow,
            edge_flows: parsed
                .edge_flows
                .unwrap_or_default()
                .into_iter()
                .map(MaxFlowEdgeFlow::from)
                .collect(),
            min_cut: parsed.min_cut.map(Into::into).unwrap_or_default(),
            node_balance: parsed.node_balance.unwrap_or_default(),
            iterations: parsed.iterations,
            ortools_status: parsed.ortools_status,
            ortools_max_flow: parsed.ortools_max_flow,
            ortools_edge_flows: parsed
                .ortools_edge_flows
                .unwrap_or_default()
                .into_iter()
                .map(MaxFlowEdgeFlow::from)
                .collect(),
            ortools_min_cut: parsed.ortools_min_cut.map(Into::into).unwrap_or_default(),
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
        Err(err) => empty_solution(
            ExternalMaxFlowReferenceStatus::NumericalError,
            format!(
                "failed to parse max_flow_reference.py output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_max_flow_with_external_reference(
    problem: &MaxFlowProblem,
    opts: &ExternalMaxFlowReferenceOptions,
) -> ExternalMaxFlowReferenceSolution {
    if matches!(
        opts.solver,
        ExternalMaxFlowReferenceSolver::Auto
            | ExternalMaxFlowReferenceSolver::RustEdmondsKarp
            | ExternalMaxFlowReferenceSolver::Fallback
    ) {
        return solve_max_flow_with_rust_reference(problem);
    }

    run_max_flow_reference_json(
        json!({
            "numNodes": problem.num_nodes,
            "source": problem.source,
            "sink": problem.sink,
            "edges": problem.edges.iter().map(|edge| {
                json!({
                    "from": edge.from,
                    "to": edge.to,
                    "capacity": edge.capacity,
                    "name": edge.name,
                })
            }).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::max_flow::{build_textbook_max_flow_problem, MaxFlowEdge};

    #[test]
    fn rust_reference_solves_textbook_max_flow() {
        let problem = build_textbook_max_flow_problem();
        let solution = solve_max_flow_with_external_reference(
            &problem,
            &ExternalMaxFlowReferenceOptions {
                solver: ExternalMaxFlowReferenceSolver::RustEdmondsKarp,
            },
        );

        assert_eq!(solution.status, ExternalMaxFlowReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:edmonds-karp-max-flow");
        assert!((solution.max_flow.unwrap() - 23.0).abs() <= 1e-9);
        assert!((solution.min_cut.capacity - 23.0).abs() <= 1e-9);
        assert_eq!(solution.node_balance.len(), problem.num_nodes);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_without_python() {
        let problem = MaxFlowProblem {
            num_nodes: 4,
            source: 0,
            sink: 3,
            edges: vec![
                MaxFlowEdge {
                    from: 0,
                    to: 1,
                    capacity: 3.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 0,
                    to: 2,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 3,
                    capacity: 2.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 2,
                    to: 3,
                    capacity: 3.0,
                    name: None,
                },
                MaxFlowEdge {
                    from: 1,
                    to: 2,
                    capacity: 1.0,
                    name: None,
                },
            ],
        };

        let solution = solve_max_flow_with_external_reference(
            &problem,
            &ExternalMaxFlowReferenceOptions {
                solver: ExternalMaxFlowReferenceSolver::Fallback,
            },
        );

        assert_eq!(solution.status, ExternalMaxFlowReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:edmonds-karp-max-flow");
        assert!((solution.max_flow.unwrap() - 5.0).abs() <= 1e-9);
        assert!((solution.min_cut.capacity - 5.0).abs() <= 1e-9);
    }

    #[test]
    fn auto_prefers_rust_reference_without_python() {
        let problem = build_textbook_max_flow_problem();

        let solution = solve_max_flow_with_external_reference(
            &problem,
            &ExternalMaxFlowReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalMaxFlowReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:edmonds-karp-max-flow");
        assert!((solution.max_flow.unwrap() - 23.0).abs() <= 1e-9);
    }

    #[test]
    fn max_flow_python_bridge_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_max_flow_reference_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
