//! Rust-facing bridge for external/reference scheduling solvers.
//!
//! The checked-in Python bridge (`scripts/scheduling_reference.py`) computes a
//! deterministic exact small job-shop reference and, when installed, calls
//! OR-Tools CP-SAT using interval variables plus no-overlap machine resources.
//! This module owns typed model serialization and status mapping for those
//! same-input cross-checks.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::classical_optimization_models::{JobShopJob, ScheduledOperation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSchedulingReferenceSolver {
    Auto,
    OrTools,
    Fallback,
}

impl ExternalSchedulingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalSchedulingReferenceSolver::Auto => "auto",
            ExternalSchedulingReferenceSolver::OrTools => "ortools",
            ExternalSchedulingReferenceSolver::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSchedulingReferenceOptions {
    pub solver: ExternalSchedulingReferenceSolver,
}

impl Default for ExternalSchedulingReferenceOptions {
    fn default() -> Self {
        ExternalSchedulingReferenceOptions {
            solver: ExternalSchedulingReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSchedulingReferenceStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unsupported,
    NumericalError,
    Unavailable,
}

impl ExternalSchedulingReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalSchedulingReferenceStatus::Optimal => "optimal",
            ExternalSchedulingReferenceStatus::Feasible => "feasible",
            ExternalSchedulingReferenceStatus::Infeasible => "infeasible",
            ExternalSchedulingReferenceStatus::Unsupported => "unsupported",
            ExternalSchedulingReferenceStatus::NumericalError => "numerical-error",
            ExternalSchedulingReferenceStatus::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalJobShopReferenceSolution {
    pub status: ExternalSchedulingReferenceStatus,
    pub solver: String,
    pub schedule: Vec<ScheduledOperation>,
    pub makespan: Option<f64>,
    pub total_flow_time: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_makespan: Option<f64>,
    pub ortools_total_flow_time: Option<f64>,
    pub ortools_schedule: Vec<ScheduledOperation>,
    pub message: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Deserialize)]
struct SchedulingReferencePayload {
    status: String,
    solver: Option<String>,
    schedule: Option<Vec<ScheduledOperationPayload>>,
    makespan: Option<f64>,
    #[serde(rename = "totalFlowTime")]
    total_flow_time: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsMakespan")]
    ortools_makespan: Option<f64>,
    #[serde(rename = "ortoolsTotalFlowTime")]
    ortools_total_flow_time: Option<f64>,
    #[serde(rename = "ortoolsSchedule")]
    ortools_schedule: Option<Vec<ScheduledOperationPayload>>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScheduledOperationPayload {
    #[serde(rename = "jobId")]
    job_id: String,
    #[serde(rename = "opIndex")]
    op_index: usize,
    machine: String,
    start: f64,
    finish: f64,
}

impl From<ScheduledOperationPayload> for ScheduledOperation {
    fn from(value: ScheduledOperationPayload) -> Self {
        ScheduledOperation {
            job_id: value.job_id,
            op_index: value.op_index,
            machine: value.machine,
            start: value.start,
            finish: value.finish,
        }
    }
}

fn status_from_str(status: &str) -> ExternalSchedulingReferenceStatus {
    match status {
        "optimal" => ExternalSchedulingReferenceStatus::Optimal,
        "feasible" => ExternalSchedulingReferenceStatus::Feasible,
        "infeasible" => ExternalSchedulingReferenceStatus::Infeasible,
        "unsupported" => ExternalSchedulingReferenceStatus::Unsupported,
        "unavailable" => ExternalSchedulingReferenceStatus::Unavailable,
        _ => ExternalSchedulingReferenceStatus::NumericalError,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalJobShopReferenceSolution {
    ExternalJobShopReferenceSolution {
        status: ExternalSchedulingReferenceStatus::Unavailable,
        solver: "external-scheduling-reference".to_string(),
        schedule: Vec::new(),
        makespan: None,
        total_flow_time: None,
        ortools_status: None,
        ortools_makespan: None,
        ortools_total_flow_time: None,
        ortools_schedule: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn numerical_error(
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalJobShopReferenceSolution {
    ExternalJobShopReferenceSolution {
        status: ExternalSchedulingReferenceStatus::NumericalError,
        solver: "external-scheduling-reference".to_string(),
        schedule: Vec::new(),
        makespan: None,
        total_flow_time: None,
        ortools_status: None,
        ortools_makespan: None,
        ortools_total_flow_time: None,
        ortools_schedule: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn reference_script() -> PathBuf {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    root.join("scripts").join("scheduling_reference.py")
}

fn run_scheduling_reference_json(
    payload: Value,
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
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
                format!("failed to start scheduling_reference.py with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write scheduling_reference.py stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            return numerical_error(
                format!("failed to wait for scheduling_reference.py: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match serde_json::from_slice::<SchedulingReferencePayload>(&output.stdout) {
        Ok(parsed) => ExternalJobShopReferenceSolution {
            status: status_from_str(&parsed.status),
            solver: parsed
                .solver
                .unwrap_or_else(|| "external-scheduling-reference".to_string()),
            schedule: parsed
                .schedule
                .unwrap_or_default()
                .into_iter()
                .map(ScheduledOperation::from)
                .collect(),
            makespan: parsed.makespan,
            total_flow_time: parsed.total_flow_time,
            ortools_status: parsed.ortools_status,
            ortools_makespan: parsed.ortools_makespan,
            ortools_total_flow_time: parsed.ortools_total_flow_time,
            ortools_schedule: parsed
                .ortools_schedule
                .unwrap_or_default()
                .into_iter()
                .map(ScheduledOperation::from)
                .collect(),
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
                "failed to parse scheduling_reference.py output: {err}; stderr={}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            elapsed_ms,
        ),
    }
}

pub fn solve_job_shop_with_external_reference(
    jobs: &[JobShopJob],
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
    run_scheduling_reference_json(
        json!({
            "jobs": jobs.iter().map(|job| json!({
                "id": &job.id,
                "due": job.due,
                "operations": job.operations.iter().map(|op| json!({
                    "machine": &op.machine,
                    "duration": op.duration,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}
