//! Rust-facing bridge for external/reference scheduling solvers.
//!
//! The native Rust reference computes deterministic exact small scheduling
//! checks without Python startup. Registered OR-Tools aliases default to that
//! Rust reference; explicit force-Python switches keep the inline OR-Tools
//! adapter available for compatibility validation.

use std::collections::HashSet;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

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

fn scheduling_reference_force_python_value(value: &str) -> bool {
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

fn scheduling_python_reference_forced() -> bool {
    [
        "SCHEDULING_REFERENCE_FORCE_PYTHON",
        "SCHEDULING_REFERENCE_ORTOOLS_FORCE_PYTHON",
        "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
    ]
    .into_iter()
    .any(|key| {
        std::env::var(key)
            .map(|value| scheduling_reference_force_python_value(&value))
            .unwrap_or(false)
    })
}

fn should_use_rust_scheduling_reference(opts: &ExternalSchedulingReferenceOptions) -> bool {
    matches!(
        opts.solver,
        ExternalSchedulingReferenceSolver::Auto
            | ExternalSchedulingReferenceSolver::RustExact
            | ExternalSchedulingReferenceSolver::Fallback
    )
}

fn should_use_registered_scheduling_fallback(opts: &ExternalSchedulingReferenceOptions) -> bool {
    matches!(opts.solver, ExternalSchedulingReferenceSolver::OrTools)
        && !scheduling_python_reference_forced()
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

fn relabel_registered_scheduling_fallback(
    mut solution: ExternalJobShopReferenceSolution,
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
    if should_use_registered_scheduling_fallback(opts) {
        let requested = opts.solver.as_arg();
        let rust_solver = solution.solver;
        solution.solver = format!("rust:registered-scheduling-fallback-for-{requested}");
        solution.message = format!(
            "{}; requested solver '{requested}' was validated with Rust fallback '{rust_solver}'",
            solution.message
        );
    }
    solution
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

fn scheduling_reference_timeout_ms() -> u64 {
    std::env::var("SCHEDULING_REFERENCE_TIMEOUT_MS")
        .or_else(|_| std::env::var("EXTERNAL_REFERENCE_TIMEOUT_MS"))
        .or_else(|_| std::env::var("EXTERNAL_TIMEOUT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_scheduling_adapter_output(
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
            Err(err) => return Err(format!("failed to poll OR-Tools scheduling adapter: {err}")),
        }
    }
    child
        .wait_with_output()
        .map(|output| (output, timed_out))
        .map_err(|err| format!("failed to wait for OR-Tools scheduling adapter: {err}"))
}

const ORTOOLS_SCHEDULING_SCALE: i64 = 1_000;

const ORTOOLS_SCHEDULING_ADAPTER: &str = r#"
import json
import sys

JOB_SHOP_SOLVER = "ortools:cp-sat"
FLOW_SHOP_SOLVER = "ortools:cp-sat-flow-shop"

def emit(status, solver, message, schedule=None, sequence=None, makespan=None, total_flow_time=None, ortools_status=None):
    schedule = [] if schedule is None else schedule
    sequence = [] if sequence is None else sequence
    payload = {
        "status": status,
        "solver": solver,
        "sequence": sequence,
        "schedule": schedule,
        "makespan": makespan,
        "totalFlowTime": total_flow_time,
        "message": message,
        "ortoolsStatus": ortools_status,
        "ortoolsSequence": sequence,
        "ortoolsMakespan": makespan,
        "ortoolsTotalFlowTime": total_flow_time,
        "ortoolsSchedule": schedule,
    }
    print(json.dumps(payload))

try:
    from ortools.sat.python import cp_model
except Exception as exc:
    emit("unavailable", JOB_SHOP_SOLVER, f"OR-Tools CP-SAT unavailable: {exc}", ortools_status="unavailable")
    raise SystemExit(0)

def schedule_result(schedule):
    makespan = max((operation["finish"] for operation in schedule), default=0.0)
    completions = {}
    for operation in schedule:
        job_id = operation["jobId"]
        completions[job_id] = max(completions.get(job_id, 0.0), operation["finish"])
    return float(makespan), float(sum(completions.values()))

def solve_job_shop(data):
    jobs = data["jobs"]
    scale = int(data["scale"])
    horizon = int(data["horizon"])
    model = cp_model.CpModel()
    operations = {}
    machine_intervals = {}
    last_ends = []

    for job_index, job in enumerate(jobs):
        previous_end = None
        for op_index, operation in enumerate(job["operations"]):
            duration = int(operation["scaledDuration"])
            suffix = f"j{job_index}_o{op_index}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")
            operations[(job_index, op_index)] = (start, end)
            machine_intervals.setdefault(operation["machine"], []).append(interval)
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end
        last_ends.append(previous_end)

    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        mapped = "infeasible" if status_name == "infeasible" else status_name
        emit(mapped, JOB_SHOP_SOLVER, f"OR-Tools CP-SAT status {status_name}", ortools_status=status_name)
        return

    schedule = []
    for job_index, job in enumerate(jobs):
        for op_index, operation in enumerate(job["operations"]):
            start_var, end_var = operations[(job_index, op_index)]
            schedule.append(
                {
                    "jobId": job["id"],
                    "opIndex": op_index,
                    "machine": operation["machine"],
                    "start": float(solver.Value(start_var)) / scale,
                    "finish": float(solver.Value(end_var)) / scale,
                }
            )
    schedule.sort(key=lambda op: (op["start"], op["finish"], op["machine"], op["jobId"], op["opIndex"]))
    makespan, total_flow_time = schedule_result(schedule)
    emit(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        JOB_SHOP_SOLVER,
        f"OR-Tools CP-SAT status {status_name}",
        schedule=schedule,
        makespan=makespan,
        total_flow_time=total_flow_time,
        ortools_status=status_name,
    )

def flow_shop_schedule(sequence):
    if not sequence:
        return []
    machine_count = len(sequence[0]["processingTimes"])
    machine_ready = [0.0 for _ in range(machine_count)]
    schedule = []
    for job in sequence:
        job_ready = 0.0
        for machine_index, duration in enumerate(job["processingTimes"]):
            start = max(machine_ready[machine_index], job_ready)
            finish = start + float(duration)
            schedule.append(
                {
                    "jobId": job["id"],
                    "opIndex": machine_index,
                    "machine": f"M{machine_index + 1}",
                    "start": float(start),
                    "finish": float(finish),
                }
            )
            machine_ready[machine_index] = finish
            job_ready = finish
    return schedule

def solve_flow_shop(data):
    jobs = data["jobs"]
    scale = int(data["scale"])
    n = len(jobs)
    machine_count = int(data["machineCount"])
    scaled = [[int(duration) for duration in job["scaledProcessingTimes"]] for job in jobs]
    horizon = int(data["horizon"])
    model = cp_model.CpModel()
    assigned = {
        (job, pos): model.NewBoolVar(f"assign_j{job}_p{pos}")
        for job in range(n)
        for pos in range(n)
    }
    for job in range(n):
        model.AddExactlyOne(assigned[(job, pos)] for pos in range(n))
    for pos in range(n):
        model.AddExactlyOne(assigned[(job, pos)] for job in range(n))

    completion = [
        [model.NewIntVar(0, horizon, f"c_p{pos}_m{machine}") for machine in range(machine_count)]
        for pos in range(n)
    ]
    for pos in range(n):
        for machine in range(machine_count):
            duration_expr = sum(assigned[(job, pos)] * scaled[job][machine] for job in range(n))
            if pos == 0 and machine == 0:
                model.Add(completion[pos][machine] >= duration_expr)
            elif pos == 0:
                model.Add(completion[pos][machine] >= completion[pos][machine - 1] + duration_expr)
            elif machine == 0:
                model.Add(completion[pos][machine] >= completion[pos - 1][machine] + duration_expr)
            else:
                model.Add(completion[pos][machine] >= completion[pos - 1][machine] + duration_expr)
                model.Add(completion[pos][machine] >= completion[pos][machine - 1] + duration_expr)

    makespan = completion[n - 1][machine_count - 1]
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        mapped = "infeasible" if status_name == "infeasible" else status_name
        emit(mapped, FLOW_SHOP_SOLVER, f"OR-Tools CP-SAT status {status_name}", ortools_status=status_name)
        return

    sequence = []
    for pos in range(n):
        job_index = next(job for job in range(n) if solver.BooleanValue(assigned[(job, pos)]))
        sequence.append(jobs[job_index])
    schedule = flow_shop_schedule(sequence)
    makespan, total_flow_time = schedule_result(schedule)
    emit(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        FLOW_SHOP_SOLVER,
        f"OR-Tools CP-SAT status {status_name}",
        schedule=schedule,
        sequence=[job["id"] for job in sequence],
        makespan=makespan,
        total_flow_time=total_flow_time,
        ortools_status=status_name,
    )

try:
    data = json.load(sys.stdin)
    if data.get("kind") == "flow-shop":
        solve_flow_shop(data)
    else:
        solve_job_shop(data)
except Exception as exc:
    solver = FLOW_SHOP_SOLVER
    try:
        if data.get("kind") != "flow-shop":
            solver = JOB_SHOP_SOLVER
    except Exception:
        solver = JOB_SHOP_SOLVER
    emit("numerical-error", solver, str(exc), ortools_status="error")
    raise SystemExit(1)
"#;

fn scaled_ortools_scheduling_duration(duration: f64) -> Option<i64> {
    if !duration.is_finite() || duration < 0.0 {
        return None;
    }
    let scaled = (duration * ORTOOLS_SCHEDULING_SCALE as f64).round();
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return None;
    }
    Some(scaled as i64)
}

fn checked_scaled_duration_sum(total: &mut i64, duration: i64) -> Result<(), String> {
    *total = total
        .checked_add(duration)
        .ok_or_else(|| "OR-Tools CP-SAT bridge duration scaling overflow".to_string())?;
    Ok(())
}

fn ortools_job_shop_payload(jobs: &[JobShopJob]) -> Result<Value, String> {
    validate_rust_job_shop_jobs(jobs)?;
    let mut horizon = 0_i64;
    let mut job_payloads = Vec::with_capacity(jobs.len());
    for job in jobs {
        let mut operation_payloads = Vec::with_capacity(job.operations.len());
        for operation in &job.operations {
            let scaled_duration = scaled_ortools_scheduling_duration(operation.duration)
                .ok_or_else(|| {
                    "OR-Tools CP-SAT bridge requires finite non-negative durations".to_string()
                })?;
            checked_scaled_duration_sum(&mut horizon, scaled_duration)?;
            operation_payloads.push(json!({
                "machine": &operation.machine,
                "duration": operation.duration,
                "scaledDuration": scaled_duration,
            }));
        }
        job_payloads.push(json!({
            "id": &job.id,
            "due": job.due,
            "operations": operation_payloads,
        }));
    }
    Ok(json!({
        "kind": "job-shop",
        "scale": ORTOOLS_SCHEDULING_SCALE,
        "horizon": horizon,
        "jobs": job_payloads,
    }))
}

fn ortools_flow_shop_payload(jobs: &[FlowShopJob]) -> Result<Value, String> {
    validate_rust_flow_shop_jobs(jobs)?;
    let machine_count = jobs[0].processing_times.len();
    let mut horizon = 0_i64;
    let mut job_payloads = Vec::with_capacity(jobs.len());
    for job in jobs {
        let mut scaled_processing_times = Vec::with_capacity(job.processing_times.len());
        for &duration in &job.processing_times {
            let scaled_duration =
                scaled_ortools_scheduling_duration(duration).ok_or_else(|| {
                    "OR-Tools CP-SAT bridge requires finite non-negative durations".to_string()
                })?;
            checked_scaled_duration_sum(&mut horizon, scaled_duration)?;
            scaled_processing_times.push(scaled_duration);
        }
        job_payloads.push(json!({
            "id": &job.id,
            "due": job.due,
            "processingTimes": &job.processing_times,
            "scaledProcessingTimes": scaled_processing_times,
        }));
    }
    Ok(json!({
        "kind": "flow-shop",
        "scale": ORTOOLS_SCHEDULING_SCALE,
        "horizon": horizon,
        "machineCount": machine_count,
        "jobs": job_payloads,
    }))
}

fn run_ortools_scheduling_reference(payload: Value) -> ExternalJobShopReferenceSolution {
    let started = Instant::now();
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(&python);
    command.arg("-c").arg(ORTOOLS_SCHEDULING_ADAPTER);
    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            return unavailable(
                format!("failed to start OR-Tools scheduling adapter with {python}: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        if let Err(err) = stdin.write_all(payload.to_string().as_bytes()) {
            return numerical_error(
                format!("failed to write OR-Tools scheduling adapter stdin: {err}"),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    let timeout_ms = scheduling_reference_timeout_ms();
    let (output, timed_out) = match wait_for_scheduling_adapter_output(child, timeout_ms) {
        Ok(output) => output,
        Err(err) => return numerical_error(err, started.elapsed().as_secs_f64() * 1000.0),
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stderr = if timed_out {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            format!("OR-Tools scheduling adapter timed out after {timeout_ms}ms")
        } else {
            format!("{stderr}; OR-Tools scheduling adapter timed out after {timeout_ms}ms")
        }
    } else {
        String::from_utf8_lossy(&output.stderr).trim().to_string()
    };
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
                    stderr.clone()
                }
            }),
            elapsed_ms,
        },
        Err(err) => numerical_error(
            format!(
                "failed to parse OR-Tools scheduling adapter output: {err}; stderr={}",
                stderr
            ),
            elapsed_ms,
        ),
    }
}

fn run_ortools_job_shop_reference(jobs: &[JobShopJob]) -> ExternalJobShopReferenceSolution {
    let started = Instant::now();
    match ortools_job_shop_payload(jobs) {
        Ok(payload) => run_ortools_scheduling_reference(payload),
        Err(message) => numerical_error(message, started.elapsed().as_secs_f64() * 1000.0),
    }
}

fn run_ortools_flow_shop_reference(jobs: &[FlowShopJob]) -> ExternalJobShopReferenceSolution {
    let started = Instant::now();
    match ortools_flow_shop_payload(jobs) {
        Ok(payload) => run_ortools_scheduling_reference(payload),
        Err(message) => numerical_error(message, started.elapsed().as_secs_f64() * 1000.0),
    }
}

pub fn solve_job_shop_with_external_reference(
    jobs: &[JobShopJob],
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
    if should_use_rust_scheduling_reference(opts) || should_use_registered_scheduling_fallback(opts)
    {
        return relabel_registered_scheduling_fallback(
            solve_job_shop_with_rust_reference(jobs),
            opts,
        );
    }

    run_ortools_job_shop_reference(jobs)
}

pub fn solve_flow_shop_with_external_reference(
    jobs: &[FlowShopJob],
    opts: &ExternalSchedulingReferenceOptions,
) -> ExternalJobShopReferenceSolution {
    if should_use_rust_scheduling_reference(opts) || should_use_registered_scheduling_fallback(opts)
    {
        return relabel_registered_scheduling_fallback(
            solve_flow_shop_with_rust_reference(jobs),
            opts,
        );
    }

    run_ortools_flow_shop_reference(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::classical_optimization_models::JobOperation;

    use crate::des::shared::test_support::ENV_LOCK as SCHEDULING_REFERENCE_ENV_LOCK;

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

    fn scheduling_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "SCHEDULING_REFERENCE_FORCE_PYTHON",
            "SCHEDULING_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    #[test]
    fn scheduling_force_python_requires_explicit_compatibility_value() {
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
                scheduling_reference_force_python_value(value),
                "{value:?} should enable the scheduling compatibility bridge"
            );
        }

        for value in [
            "", "0", "false", "off", "python", "py", "auto", "rust", "native",
        ] {
            assert!(
                !scheduling_reference_force_python_value(value),
                "{value:?} should keep Rust scheduling fallback active"
            );
        }
    }

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

    #[test]
    fn auto_prefers_rust_job_shop_reference_without_python() {
        let solution = solve_job_shop_with_external_reference(
            &sample_job_shop_jobs(),
            &ExternalSchedulingReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-job-shop");
        assert_eq!(solution.makespan, Some(9.0));
        assert_eq!(solution.schedule.len(), 6);
    }

    #[test]
    fn auto_prefers_rust_flow_shop_reference_without_python() {
        let solution = solve_flow_shop_with_external_reference(
            &sample_flow_shop_jobs(),
            &ExternalSchedulingReferenceOptions::default(),
        );

        assert_eq!(solution.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(solution.solver, "rust:exact-flow-shop");
        assert!(solution.makespan.is_some());
        assert_eq!(solution.sequence.len(), 4);
    }

    #[test]
    fn registered_ortools_alias_defaults_to_rust_reference_without_python() {
        let _lock = SCHEDULING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guards = scheduling_force_python_off_guards();
        let _python_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-scheduling-alias");
        let opts = ExternalSchedulingReferenceOptions {
            solver: ExternalSchedulingReferenceSolver::OrTools,
        };

        let job_shop = solve_job_shop_with_external_reference(&sample_job_shop_jobs(), &opts);
        assert_eq!(job_shop.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(
            job_shop.solver,
            "rust:registered-scheduling-fallback-for-ortools"
        );
        assert!(job_shop
            .message
            .contains("requested solver 'ortools' was validated with Rust fallback"));
        assert_eq!(job_shop.makespan, Some(9.0));

        let flow_shop = solve_flow_shop_with_external_reference(&sample_flow_shop_jobs(), &opts);
        assert_eq!(flow_shop.status, ExternalSchedulingReferenceStatus::Optimal);
        assert_eq!(
            flow_shop.solver,
            "rust:registered-scheduling-fallback-for-ortools"
        );
        assert_eq!(flow_shop.sequence.len(), 4);
        assert!(flow_shop.makespan.is_some());
    }

    #[test]
    fn scheduling_force_python_keeps_ortools_bridge_available() {
        let _lock = SCHEDULING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("SCHEDULING_REFERENCE_FORCE_PYTHON", "1");
        let _python_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-forced-scheduling");
        let opts = ExternalSchedulingReferenceOptions {
            solver: ExternalSchedulingReferenceSolver::OrTools,
        };

        let job_shop = solve_job_shop_with_external_reference(&sample_job_shop_jobs(), &opts);
        assert_eq!(
            job_shop.status,
            ExternalSchedulingReferenceStatus::Unavailable
        );
        assert!(job_shop.message.contains("OR-Tools scheduling adapter"));

        let flow_shop = solve_flow_shop_with_external_reference(&sample_flow_shop_jobs(), &opts);
        assert_eq!(
            flow_shop.status,
            ExternalSchedulingReferenceStatus::Unavailable
        );
        assert!(flow_shop.message.contains("OR-Tools scheduling adapter"));
    }

    #[test]
    fn ortools_adapter_reports_startup_without_repo_script() {
        let _lock = SCHEDULING_REFERENCE_ENV_LOCK
            .lock()
            .expect("lock env guard");
        let _force_python_guard = EnvVarGuard::set("SCHEDULING_REFERENCE_FORCE_PYTHON", "1");
        let _guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-scheduling-ortools",
        );

        let solution = solve_job_shop_with_external_reference(
            &sample_job_shop_jobs(),
            &ExternalSchedulingReferenceOptions {
                solver: ExternalSchedulingReferenceSolver::OrTools,
            },
        );

        assert_eq!(
            solution.status,
            ExternalSchedulingReferenceStatus::Unavailable
        );
        assert!(solution.message.contains("OR-Tools scheduling adapter"));
        assert!(!solution.message.contains("scheduling_reference.py"));
    }

    #[test]
    fn scheduling_adapter_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_scheduling_adapter_output(child, 10).expect("timeout output");

        assert!(timed_out);
        assert!(!output.status.success());
    }
}
