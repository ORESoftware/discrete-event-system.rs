//! Rust-facing bridge for external/reference scheduling solvers.
//!
//! The native Rust reference computes deterministic exact small scheduling
//! checks without Python startup. The checked-in Python bridge
//! (`scripts/scheduling_reference.py`) remains available for OR-Tools CP-SAT
//! using interval variables plus no-overlap machine resources.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::des::general::classical_optimization_models::{
    run_flow_shop_exact, run_job_shop_exact, FlowShopJob, FlowShopNEHParams, JobShopDispatchParams,
    JobShopJob, ScheduledOperation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalSchedulingReferenceSolver {
    Auto,
    RustExact,
    OrTools,
    Fallback,
}

impl ExternalSchedulingReferenceSolver {
    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalSchedulingReferenceSolver::Auto => "auto",
            ExternalSchedulingReferenceSolver::RustExact => "rust-exact",
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
    pub sequence: Vec<String>,
    pub schedule: Vec<ScheduledOperation>,
    pub makespan: Option<f64>,
    pub total_flow_time: Option<f64>,
    pub ortools_status: Option<String>,
    pub ortools_sequence: Vec<String>,
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
    sequence: Option<Vec<String>>,
    schedule: Option<Vec<ScheduledOperationPayload>>,
    makespan: Option<f64>,
    #[serde(rename = "totalFlowTime")]
    total_flow_time: Option<f64>,
    #[serde(rename = "ortoolsStatus")]
    ortools_status: Option<String>,
    #[serde(rename = "ortoolsSequence")]
    ortools_sequence: Option<Vec<String>>,
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

const RUST_JOB_SHOP_MAX_EXACT_OPS: usize = 20;
const RUST_FLOW_SHOP_MAX_EXACT_JOBS: usize = 10;

fn validate_rust_job_shop_jobs(jobs: &[JobShopJob]) -> Result<(), String> {
    if jobs.is_empty() {
        return Err("jobs must be non-empty".to_string());
    }
    let mut ids = HashSet::new();
    for (job_index, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() {
            return Err(format!("jobs[{job_index}].id must be non-empty"));
        }
        if !ids.insert(job.id.clone()) {
            return Err(format!("duplicate job id {:?}", job.id));
        }
        if job.operations.is_empty() {
            return Err(format!("jobs[{job_index}].operations must be non-empty"));
        }
        if job.due.is_some_and(|due| !due.is_finite()) {
            return Err(format!("jobs[{job_index}].due must be finite"));
        }
        for (op_index, operation) in job.operations.iter().enumerate() {
            if operation.machine.trim().is_empty() {
                return Err(format!(
                    "jobs[{job_index}].operations[{op_index}].machine must be non-empty"
                ));
            }
            if !operation.duration.is_finite() || operation.duration < 0.0 {
                return Err(format!(
                    "jobs[{job_index}].operations[{op_index}].duration must be finite and non-negative"
                ));
            }
        }
    }
    Ok(())
}

fn validate_rust_flow_shop_jobs(jobs: &[FlowShopJob]) -> Result<(), String> {
    if jobs.is_empty() {
        return Err("jobs must be non-empty".to_string());
    }
    let machine_count = jobs[0].processing_times.len();
    if machine_count == 0 {
        return Err("jobs[0].processingTimes must be non-empty".to_string());
    }
    let mut ids = HashSet::new();
    for (job_index, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() {
            return Err(format!("jobs[{job_index}].id must be non-empty"));
        }
        if !ids.insert(job.id.clone()) {
            return Err(format!("duplicate job id {:?}", job.id));
        }
        if job.due.is_some_and(|due| !due.is_finite()) {
            return Err(format!("jobs[{job_index}].due must be finite"));
        }
        if job.processing_times.len() != machine_count {
            return Err(format!(
                "jobs[{job_index}].processingTimes length {} != {machine_count}",
                job.processing_times.len()
            ));
        }
        for (machine_index, &duration) in job.processing_times.iter().enumerate() {
            if !duration.is_finite() || duration < 0.0 {
                return Err(format!(
                    "jobs[{job_index}].processingTimes[{machine_index}] must be finite and non-negative"
                ));
            }
        }
    }
    Ok(())
}

fn rust_scheduling_empty_solution(
    status: ExternalSchedulingReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalJobShopReferenceSolution {
    ExternalJobShopReferenceSolution {
        status,
        solver: solver.into(),
        sequence: Vec::new(),
        schedule: Vec::new(),
        makespan: None,
        total_flow_time: None,
        ortools_status: None,
        ortools_sequence: Vec::new(),
        ortools_makespan: None,
        ortools_total_flow_time: None,
        ortools_schedule: Vec::new(),
        message: message.into(),
        elapsed_ms,
    }
}

fn solve_job_shop_with_rust_reference(jobs: &[JobShopJob]) -> ExternalJobShopReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_job_shop_jobs(jobs) {
        return rust_scheduling_empty_solution(
            ExternalSchedulingReferenceStatus::NumericalError,
            "rust:exact-job-shop",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let total_ops = jobs.iter().map(|job| job.operations.len()).sum::<usize>();
    if total_ops > RUST_JOB_SHOP_MAX_EXACT_OPS {
        return rust_scheduling_empty_solution(
            ExternalSchedulingReferenceStatus::Unsupported,
            "rust:exact-job-shop",
            format!(
                "exact job-shop only practical for <= {RUST_JOB_SHOP_MAX_EXACT_OPS} operations, got {total_ops}"
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let result = run_job_shop_exact(JobShopDispatchParams {
        jobs: Some(jobs.to_vec()),
        rule: None,
    });
    ExternalJobShopReferenceSolution {
        status: ExternalSchedulingReferenceStatus::Optimal,
        solver: "rust:exact-job-shop".to_string(),
        sequence: Vec::new(),
        schedule: result.schedule,
        makespan: Some(result.makespan),
        total_flow_time: Some(result.total_flow_time),
        ortools_status: None,
        ortools_sequence: Vec::new(),
        ortools_makespan: None,
        ortools_total_flow_time: None,
        ortools_schedule: Vec::new(),
        message: "exact job-shop branch-and-bound".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn solve_flow_shop_with_rust_reference(jobs: &[FlowShopJob]) -> ExternalJobShopReferenceSolution {
    let started = Instant::now();
    if let Err(message) = validate_rust_flow_shop_jobs(jobs) {
        return rust_scheduling_empty_solution(
            ExternalSchedulingReferenceStatus::NumericalError,
            "rust:exact-flow-shop",
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    if jobs.len() > RUST_FLOW_SHOP_MAX_EXACT_JOBS {
        return rust_scheduling_empty_solution(
            ExternalSchedulingReferenceStatus::Unsupported,
            "rust:exact-flow-shop",
            format!(
                "exact flow-shop only practical for <= {RUST_FLOW_SHOP_MAX_EXACT_JOBS} jobs, got {}",
                jobs.len()
            ),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let result = run_flow_shop_exact(FlowShopNEHParams {
        jobs: Some(jobs.to_vec()),
    });
    ExternalJobShopReferenceSolution {
        status: ExternalSchedulingReferenceStatus::Optimal,
        solver: "rust:exact-flow-shop".to_string(),
        sequence: result.sequence,
        schedule: result.schedule,
        makespan: Some(result.makespan),
        total_flow_time: Some(result.total_flow_time),
        ortools_status: None,
        ortools_sequence: Vec::new(),
        ortools_makespan: None,
        ortools_total_flow_time: None,
        ortools_schedule: Vec::new(),
        message: "exact permutation flow-shop enumeration".to_string(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn unavailable(message: impl Into<String>, elapsed_ms: f64) -> ExternalJobShopReferenceSolution {
    ExternalJobShopReferenceSolution {
        status: ExternalSchedulingReferenceStatus::Unavailable,
        solver: "external-scheduling-reference".to_string(),
        sequence: Vec::new(),
        schedule: Vec::new(),
        makespan: None,
        total_flow_time: None,
        ortools_status: None,
        ortools_sequence: Vec::new(),
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
        sequence: Vec::new(),
        schedule: Vec::new(),
        makespan: None,
        total_flow_time: None,
        ortools_status: None,
        ortools_sequence: Vec::new(),
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
            sequence: parsed.sequence.unwrap_or_default(),
            schedule: parsed
                .schedule
                .unwrap_or_default()
                .into_iter()
                .map(ScheduledOperation::from)
                .collect(),
            makespan: parsed.makespan,
            total_flow_time: parsed.total_flow_time,
            ortools_status: parsed.ortools_status,
            ortools_sequence: parsed.ortools_sequence.unwrap_or_default(),
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
    if matches!(
        opts.solver,
        ExternalSchedulingReferenceSolver::RustExact | ExternalSchedulingReferenceSolver::Fallback
    ) {
        return solve_job_shop_with_rust_reference(jobs);
    }

    run_scheduling_reference_json(
        json!({
            "kind": "job-shop",
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

pub fn solve_flow_shop_with_external_reference(
    jobs: &[FlowShopJob],
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
    if matches!(
        opts.solver,
        ExternalSchedulingReferenceSolver::RustExact | ExternalSchedulingReferenceSolver::Fallback
    ) {
        return solve_flow_shop_with_rust_reference(jobs);
    }

    run_scheduling_reference_json(
        json!({
            "kind": "flow-shop",
            "jobs": jobs.iter().map(|job| json!({
                "id": &job.id,
                "due": job.due,
                "processingTimes": &job.processing_times,
            })).collect::<Vec<_>>(),
        }),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::classical_optimization_models::JobOperation;

    fn sample_job_shop_jobs() -> Vec<JobShopJob> {
        vec![
            JobShopJob {
                id: "J1".to_string(),
                due: Some(10.0),
                operations: vec![
                    JobOperation {
                        machine: "M1".to_string(),
                        duration: 3.0,
                    },
                    JobOperation {
                        machine: "M2".to_string(),
                        duration: 2.0,
                    },
                ],
            },
            JobShopJob {
                id: "J2".to_string(),
                due: Some(8.0),
                operations: vec![
                    JobOperation {
                        machine: "M2".to_string(),
                        duration: 2.0,
                    },
                    JobOperation {
                        machine: "M1".to_string(),
                        duration: 4.0,
                    },
                ],
            },
            JobShopJob {
                id: "J3".to_string(),
                due: Some(12.0),
                operations: vec![
                    JobOperation {
                        machine: "M1".to_string(),
                        duration: 2.0,
                    },
                    JobOperation {
                        machine: "M2".to_string(),
                        duration: 3.0,
                    },
                ],
            },
        ]
    }

    fn sample_flow_shop_jobs() -> Vec<FlowShopJob> {
        vec![
            FlowShopJob {
                id: "F1".to_string(),
                processing_times: vec![2.0, 3.0, 2.0],
                due: None,
            },
            FlowShopJob {
                id: "F2".to_string(),
                processing_times: vec![4.0, 1.0, 3.0],
                due: None,
            },
            FlowShopJob {
                id: "F3".to_string(),
                processing_times: vec![3.0, 2.0, 4.0],
                due: None,
            },
            FlowShopJob {
                id: "F4".to_string(),
                processing_times: vec![2.0, 5.0, 1.0],
                due: None,
            },
        ]
    }

    #[test]
    fn rust_reference_solves_sample_job_shop() {
        let solution = solve_job_shop_with_external_reference(
            &sample_job_shop_jobs(),
            &ExternalSchedulingReferenceOptions {
                solver: ExternalSchedulingReferenceSolver::RustExact,
            },
        );

        assert_eq!(solution.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-job-shop");
        assert_eq!(solution.makespan, Some(9.0));
        assert_eq!(solution.schedule.len(), 6);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn rust_reference_solves_sample_flow_shop() {
        let solution = solve_flow_shop_with_external_reference(
            &sample_flow_shop_jobs(),
            &ExternalSchedulingReferenceOptions {
                solver: ExternalSchedulingReferenceSolver::RustExact,
            },
        );

        assert_eq!(solution.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-flow-shop");
        assert!(solution.makespan.is_some());
        assert_eq!(solution.sequence.len(), 4);
        assert_eq!(solution.schedule.len(), 12);
        assert!(solution.ortools_status.is_none());
    }

    #[test]
    fn fallback_alias_uses_rust_reference_for_job_shop_validation_error() {
        let jobs = vec![JobShopJob {
            id: "".to_string(),
            due: None,
            operations: vec![JobOperation {
                machine: "M1".to_string(),
                duration: 1.0,
            }],
        }];

        let solution = solve_job_shop_with_external_reference(
            &jobs,
            &ExternalSchedulingReferenceOptions {
                solver: ExternalSchedulingReferenceSolver::Fallback,
            },
        );

        assert_eq!(
            solution.status,
            ExternalSchedulingReferenceStatus::NumericalError
        );
        assert_eq!(solution.solver, "rust:exact-job-shop");
    }
}
