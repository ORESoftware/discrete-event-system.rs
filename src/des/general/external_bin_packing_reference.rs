//! Rust-facing bridge for external/reference bin-packing solvers.
//!
//! The checked-in Python bridge (`scripts/bin_packing_reference.py`) computes a
//! deterministic exact small-instance reference and, when installed, calls
//! OR-Tools CP-SAT on the same item/capacity input.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::bin_packing::BinPackingProblem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalBinPackingReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalBinPackingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalBinPackingReferenceSolver::Auto => "auto",
            ExternalBinPackingReferenceSolver::OrTools => "ortools",
            ExternalBinPackingReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceOptions {
    pub solver: ExternalBinPackingReferenceSolver,
}

impl Default for ExternalBinPackingReferenceOptions {
    fn default() -> Self {
        ExternalBinPackingReferenceOptions {
            solver: ExternalBinPackingReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalBinPackingReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalBinPackingReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalBinPackingReferenceStatus::Optimal => "optimal",
            ExternalBinPackingReferenceStatus::Feasible => "feasible",
            ExternalBinPackingReferenceStatus::Infeasible => "infeasible",
            ExternalBinPackingReferenceStatus::Unsupported => "unsupported",
            ExternalBinPackingReferenceStatus::NumericalError => "numerical-error",
            ExternalBinPackingReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceBin {
    pub item_ids: Vec<String>,
    pub load: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalBinPackingReferenceSolution {
    pub status: ExternalBinPackingReferenceStatus,
    pub solver: String,
    pub bins: Vec<ExternalBinPackingReferenceBin>,
    pub objective: Option<usize>,
    pub total_weight: Option<f64>,
    pub lower_bound_bins: Option<usize>,
    pub ortools_status: Option<String>,
    pub ortools_bins: Vec<ExternalBinPackingReferenceBin>,
    pub ortools_objective: Option<usize>,
    pub ortools_objective_bound: Option<f64>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct BinPackingReferencePayload {
    status: String,
    solver: Option<String>,
    bins: Option<Vec<BinPackingReferenceBinPayload>>,
    objective: Option<usize>,
    #[serde(rename = "totalWeight")]
    total_weight: Option<f64>,
    #[serde(rename = "lowerBoundBins")]
    lower_bound_bins: Option<usize>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsBins")]
    ortools_bins: Option<Vec<BinPackingReferenceBinPayload>>,
    #[serde(rename = "ortoolsObjective")]
    ortools_objective: Option<usize>,
    #[serde(rename = "ortoolsObjectiveBound")]
    ortools_objective_bound: Option<f64>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BinPackingReferenceBinPayload {
    items: Vec<String>,
    load: f64,
}

impl From<BinPackingReferenceBinPayload> for ExternalBinPackingReferenceBin {
    fn from(value: BinPackingReferenceBinPayload) -> Self {
        ExternalBinPackingReferenceBin {
            item_ids: value.items,
            load: value.load,
        }
    }
}

fn status_from_str(status: &str) -> ExternalBinPackingReferenceStatus {
    match status {
        "optimal" => ExternalBinPackingReferenceStatus::Optimal,
        "feasible" => ExternalBinPackingReferenceStatus::Feasible,
        "infeasible" => ExternalBinPackingReferenceStatus::Infeasible,
        "unsupported" => ExternalBinPackingReferenceStatus::Unsupported,
        "unavailable" => ExternalBinPackingReferenceStatus::Unavailable,
        _ => ExternalBinPackingReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status: ExternalBinPackingReferenceStatus::Unavailable,
        solver: "external-bin-packing-reference".to_string(),
        bins: Vec::new(),
        objective: None,
        total_weight: None,
        lower_bound_bins: None,
        ortools_status: None,
        ortools_bins: Vec::new(),
        ortools_objective: None,
        ortools_objective_bound: None,
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalBinPackingReferenceSolution {
    ExternalBinPackingReferenceSolution {
        status: ExternalBinPackingReferenceStatus::NumericalError,
        solver: "external-bin-packing-reference".to_string(),
        bins: Vec::new(),
        objective: None,
        total_weight: None,
        lower_bound_bins: None,
        ortools_status: None,
        ortools_bins: Vec::new(),
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
    root.join("scripts").join("bin_packing_reference.py")
}

fn run_bin_packing_reference_json(
    payload: Value,
    opts: &ExternalBinPackingReferenceOptions,
) -> ExternalBinPackingReferenceSolution {
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
                format!("failed to start bin_packing_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write bin_packing_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for bin_packing_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<BinPackingReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalBinPackingReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-bin-packing-reference".to_string()),
            bins: parsed
                .bins
                .unwrap_or_default()
                .into_iter()
                .map(ExternalBinPackingReferenceBin::from)
                .collect(),
            objective: parsed.objective,
            total_weight: parsed.total_weight,
            lower_bound_bins: parsed.lower_bound_bins,
            ortools_status: parsed.ortools_status,
            ortools_bins: parsed
                .ortools_bins
                .unwrap_or_default()
                .into_iter()
                .map(ExternalBinPackingReferenceBin::from)
                .collect(),
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
                "failed to parse bin_packing_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_bin_packing_with_external_reference(
    problem: &BinPackingProblem,
    opts: &ExternalBinPackingReferenceOptions,
) -> ExternalBinPackingReferenceSolution {
    run_bin_packing_reference_json(
        json!({
            "capacity": problem.capacity,
            "items": problem.items.iter().map(|item| json!({
                "id": &item.id,
                "weight": item.weight,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
