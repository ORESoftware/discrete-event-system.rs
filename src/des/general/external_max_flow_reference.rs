//! Rust-facing bridge for external/reference max-flow solvers.
//!
//! The checked-in Python bridge (`scripts/max_flow_reference.py`) computes an
//! Edmonds-Karp reference and calls OR-Tools SimpleMaxFlow when installed and
//! when capacities can be integer-scaled.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::max_flow::{MaxFlowEdgeFlow, MaxFlowProblem};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMaxFlowReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalMaxFlowReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalMaxFlowReferenceSolver::Auto => "auto",
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

fn status_from_str(status: &str) -> ExternalMaxFlowReferenceStatus {
    match status {
        "optimal" => ExternalMaxFlowReferenceStatus::Optimal,
        "infeasible" => ExternalMaxFlowReferenceStatus::Infeasible,
        "unsupported" => ExternalMaxFlowReferenceStatus::Unsupported,
        "unavailable" => ExternalMaxFlowReferenceStatus::Unavailable,
        _ => ExternalMaxFlowReferenceStatus::NumericalError,
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
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return empty_solution(
                ExternalMaxFlowReferenceStatus::NumericalError,
                format!("failed to write max_flow_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return empty_solution(
                ExternalMaxFlowReferenceStatus::NumericalError,
                format!("failed to wait for max_flow_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
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
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                }
            }),
            elapsed_ms,
        },
        Err(err) => empty_solution(
            ExternalMaxFlowReferenceStatus::NumericalError,
            format!(
                "failed to parse max_flow_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_max_flow_with_external_reference(
    problem: &MaxFlowProblem,
    opts: &ExternalMaxFlowReferenceOptions,
) -> ExternalMaxFlowReferenceSolution {
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
