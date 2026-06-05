use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::classical_optimization_models::{
    FlowShopJob, JobOperation, JobShopJob, ScheduledOperation,
};
use des_engine::des::general::external_scheduling_reference::{
    solve_flow_shop_with_external_reference, solve_job_shop_with_external_reference,
    ExternalJobShopReferenceSolution, ExternalSchedulingReferenceOptions,
    ExternalSchedulingReferenceSolver,
};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchedulingKind {
    Auto,
    JobShop,
    FlowShop,
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage(program: &str) -> String {
    format!("usage: {program} [--solver auto|fallback|rust-exact|ortools] [--kind auto|job-shop|flow-shop]")
}

fn parse_args(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<(ExternalSchedulingReferenceSolver, SchedulingKind), CliError> {
    let mut solver = ExternalSchedulingReferenceSolver::Auto;
    let mut kind = SchedulingKind::Auto;
    let mut values = args.into_iter().peekable();
    while let Some(raw) = values.next() {
        if raw == "-h" || raw == "--help" {
            return Err(CliError(usage(program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        match key.as_str() {
            "--solver" => {
                let value = next_option_value(program, "--solver", inline_value, &mut values)?;
                solver = match value.as_str() {
                    "auto" => ExternalSchedulingReferenceSolver::Auto,
                    "fallback" => ExternalSchedulingReferenceSolver::Fallback,
                    "rust-exact" | "rust_exact" => ExternalSchedulingReferenceSolver::RustExact,
                    "ortools" => ExternalSchedulingReferenceSolver::OrTools,
                    _ => {
                        return Err(CliError(format!(
                            "unknown solver {value:?}\n{}",
                            usage(program)
                        )))
                    }
                };
            }
            "--kind" => {
                let value = next_option_value(program, "--kind", inline_value, &mut values)?;
                kind = match value.as_str() {
                    "auto" => SchedulingKind::Auto,
                    "job-shop" | "job_shop" | "jobshop" => SchedulingKind::JobShop,
                    "flow-shop" | "flow_shop" | "flowshop" | "permutation-flow-shop" => {
                        SchedulingKind::FlowShop
                    }
                    _ => {
                        return Err(CliError(format!(
                            "unknown kind {value:?}\n{}",
                            usage(program)
                        )))
                    }
                };
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok((solver, kind))
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{option} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_optional_number(raw: Option<&Value>, field: &str) -> Result<Option<f64>, String> {
    raw.map(|value| parse_number(value, format!("{field} must be numeric")))
        .transpose()
}

fn parse_job_shop_jobs(raw: &Value) -> Result<Vec<JobShopJob>, String> {
    let jobs = raw
        .get("jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| "jobs must be non-empty".to_string())?;
    if jobs.is_empty() {
        return Err("jobs must be non-empty".to_string());
    }
    jobs.iter()
        .enumerate()
        .map(|(job_index, raw_job)| {
            let object = raw_job
                .as_object()
                .ok_or_else(|| format!("jobs[{job_index}] must be an object"))?;
            let id = object
                .get("id")
                .map(parse_string)
                .unwrap_or_else(|| format!("J{}", job_index + 1));
            let raw_ops = object
                .get("operations")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("jobs[{job_index}].operations must be non-empty"))?;
            if raw_ops.is_empty() {
                return Err(format!("jobs[{job_index}].operations must be non-empty"));
            }
            let operations = raw_ops
                .iter()
                .enumerate()
                .map(|(op_index, raw_op)| {
                    let op = raw_op.as_object().ok_or_else(|| {
                        format!("jobs[{job_index}].operations[{op_index}] must be an object")
                    })?;
                    Ok(JobOperation {
                        machine: op
                            .get("machine")
                            .map(parse_string)
                            .ok_or_else(|| {
                                format!(
                                    "jobs[{job_index}].operations[{op_index}].machine must be non-empty"
                                )
                            })?,
                        duration: parse_number(
                            op.get("duration").ok_or_else(|| {
                                format!(
                                    "jobs[{job_index}].operations[{op_index}].duration is required"
                                )
                            })?,
                            format!(
                                "jobs[{job_index}].operations[{op_index}].duration must be numeric"
                            ),
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(JobShopJob {
                id,
                due: parse_optional_number(object.get("due"), &format!("jobs[{job_index}].due"))?,
                operations,
            })
        })
        .collect()
}

fn parse_flow_shop_jobs(raw: &Value) -> Result<Vec<FlowShopJob>, String> {
    let jobs = raw
        .get("jobs")
        .and_then(Value::as_array)
        .ok_or_else(|| "jobs must be non-empty".to_string())?;
    if jobs.is_empty() {
        return Err("jobs must be non-empty".to_string());
    }
    jobs.iter()
        .enumerate()
        .map(|(job_index, raw_job)| {
            let object = raw_job
                .as_object()
                .ok_or_else(|| format!("jobs[{job_index}] must be an object"))?;
            let id = object
                .get("id")
                .map(parse_string)
                .unwrap_or_else(|| format!("F{}", job_index + 1));
            let raw_times = object
                .get("processingTimes")
                .or_else(|| object.get("processing_times"))
                .and_then(Value::as_array)
                .ok_or_else(|| format!("jobs[{job_index}].processingTimes must be non-empty"))?;
            if raw_times.is_empty() {
                return Err(format!(
                    "jobs[{job_index}].processingTimes must be non-empty"
                ));
            }
            let processing_times = raw_times
                .iter()
                .enumerate()
                .map(|(time_index, value)| {
                    parse_number(
                        value,
                        format!("jobs[{job_index}].processingTimes[{time_index}] must be numeric"),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(FlowShopJob {
                id,
                processing_times,
                due: parse_optional_number(object.get("due"), &format!("jobs[{job_index}].due"))?,
            })
        })
        .collect()
}

fn infer_kind(raw: &Value, requested: SchedulingKind) -> SchedulingKind {
    if requested != SchedulingKind::Auto {
        return requested;
    }
    let raw_kind = raw
        .get("kind")
        .or_else(|| raw.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase()
        .replace('_', "-");
    if matches!(
        raw_kind.as_str(),
        "flow-shop" | "flowshop" | "permutation-flow-shop"
    ) {
        return SchedulingKind::FlowShop;
    }
    if matches!(raw_kind.as_str(), "job-shop" | "jobshop") {
        return SchedulingKind::JobShop;
    }
    if raw
        .get("jobs")
        .and_then(Value::as_array)
        .is_some_and(|jobs| {
            jobs.iter().any(|job| {
                job.get("processingTimes").is_some() || job.get("processing_times").is_some()
            })
        })
    {
        SchedulingKind::FlowShop
    } else {
        SchedulingKind::JobShop
    }
}

fn operation_json(operation: &ScheduledOperation) -> Value {
    json!({
        "jobId": operation.job_id,
        "opIndex": operation.op_index,
        "machine": operation.machine,
        "start": operation.start,
        "finish": operation.finish,
    })
}

fn solution_json(solution: &ExternalJobShopReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "schedule": solution.schedule.iter().map(operation_json).collect::<Vec<_>>(),
        "sequence": solution.sequence,
        "makespan": solution.makespan,
        "totalFlowTime": solution.total_flow_time,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_sequence.is_empty()
        || !solution.ortools_schedule.is_empty()
        || solution.ortools_makespan.is_some()
        || solution.ortools_total_flow_time.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsSequence"] = json!(solution.ortools_sequence);
        output["ortoolsMakespan"] = json!(solution.ortools_makespan);
        output["ortoolsTotalFlowTime"] = json!(solution.ortools_total_flow_time);
        output["ortoolsSchedule"] = json!(solution
            .ortools_schedule
            .iter()
            .map(operation_json)
            .collect::<Vec<_>>());
    }
    output
}

fn explicit_ortools_json(
    ortools: &ExternalJobShopReferenceSolution,
    reference: &ExternalJobShopReferenceSolution,
) -> Value {
    let mut output = solution_json(ortools);
    output["sequence"] = json!(ortools.sequence);
    output["schedule"] = json!(ortools
        .schedule
        .iter()
        .map(operation_json)
        .collect::<Vec<_>>());
    output["referenceStatus"] = json!(reference.status.as_str());
    output["referenceMakespan"] = json!(reference.makespan);
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:scheduling-reference",
        "schedule": [],
        "sequence": [],
        "makespan": null,
        "totalFlowTime": null,
        "message": message.into(),
    })
}

fn solve_job_shop(
    jobs: &[JobShopJob],
    solver: ExternalSchedulingReferenceSolver,
) -> ExternalJobShopReferenceSolution {
    solve_job_shop_with_external_reference(jobs, &ExternalSchedulingReferenceOptions { solver })
}

fn solve_flow_shop(
    jobs: &[FlowShopJob],
    solver: ExternalSchedulingReferenceSolver,
) -> ExternalJobShopReferenceSolution {
    solve_flow_shop_with_external_reference(jobs, &ExternalSchedulingReferenceOptions { solver })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "scheduling_reference".to_string());
    let (solver, requested_kind) = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    match infer_kind(&payload, requested_kind) {
        SchedulingKind::JobShop | SchedulingKind::Auto => {
            let jobs = parse_job_shop_jobs(&payload).map_err(CliError)?;
            if solver == ExternalSchedulingReferenceSolver::OrTools {
                let reference = solve_job_shop(&jobs, ExternalSchedulingReferenceSolver::RustExact);
                let ortools = solve_job_shop(&jobs, solver);
                Ok(explicit_ortools_json(&ortools, &reference))
            } else {
                Ok(solution_json(&solve_job_shop(&jobs, solver)))
            }
        }
        SchedulingKind::FlowShop => {
            let jobs = parse_flow_shop_jobs(&payload).map_err(CliError)?;
            if solver == ExternalSchedulingReferenceSolver::OrTools {
                let reference =
                    solve_flow_shop(&jobs, ExternalSchedulingReferenceSolver::RustExact);
                let ortools = solve_flow_shop(&jobs, solver);
                Ok(explicit_ortools_json(&ortools, &reference))
            } else {
                Ok(solution_json(&solve_flow_shop(&jobs, solver)))
            }
        }
    }
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                args.first()
                    .map(String::as_str)
                    .unwrap_or("scheduling_reference")
            )
        );
        return;
    }
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!("{}", error_json(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(args, &stdin) {
        Ok(output) => {
            println!(
                "{}",
                serde_json::to_string(&output).expect("serialize scheduling output")
            );
        }
        Err(error) => {
            println!("{}", error_json(error.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOB_SHOP_SAMPLE: &str = r#"{
        "kind": "job-shop",
        "jobs": [
            {"id": "J1", "due": 10, "operations": [
                {"machine": "M1", "duration": 3},
                {"machine": "M2", "duration": 2}
            ]},
            {"id": "J2", "due": 8, "operations": [
                {"machine": "M2", "duration": 2},
                {"machine": "M1", "duration": 4}
            ]},
            {"id": "J3", "due": 12, "operations": [
                {"machine": "M1", "duration": 2},
                {"machine": "M2", "duration": 3}
            ]}
        ]
    }"#;

    const FLOW_SHOP_SAMPLE: &str = r#"{
        "kind": "flow-shop",
        "jobs": [
            {"id": "F1", "processingTimes": [2, 3, 2]},
            {"id": "F2", "processingTimes": [4, 1, 3]},
            {"id": "F3", "processingTimes": [3, 2, 4]},
            {"id": "F4", "processingTimes": [2, 5, 1]}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_exact_job_shop() {
        let output = run(
            vec![
                "scheduling_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            JOB_SHOP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-job-shop");
        assert_eq!(output["makespan"], 9.0);
        assert_eq!(output["schedule"].as_array().expect("schedule").len(), 6);
    }

    #[test]
    fn accepts_flow_shop_and_rust_exact_alias() {
        let output = run(
            vec![
                "scheduling_reference".to_string(),
                "--solver=rust-exact".to_string(),
                "--kind=flow-shop".to_string(),
            ],
            FLOW_SHOP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-flow-shop");
        assert_eq!(output["sequence"].as_array().expect("sequence").len(), 4);
        assert_eq!(output["schedule"].as_array().expect("schedule").len(), 12);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["scheduling_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("jobs"));
    }
}
