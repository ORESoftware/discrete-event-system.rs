use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_validation_tools::{
    run_simulation_validation_json_with_external_reference,
    ExternalSimulationValidationReferenceOptions,
};
use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage(program: &str) -> String {
    format!("usage: {program} [--engine ENGINE]")
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut impl Iterator<Item = String>,
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

fn parse_args(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<Option<String>, CliError> {
    let mut engine = None::<String>;
    let mut values = args.into_iter();
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
            "--engine" => {
                engine = Some(next_option_value(
                    program,
                    "--engine",
                    inline_value,
                    &mut values,
                )?);
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(engine)
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "failed",
        "verdict": "failure",
        "simulator": "rust:simulation-validation-reference",
        "message": message.into(),
        "metrics": {},
        "checks": [],
        "trace": [],
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "simulation_validation_reference".to_string());
    let engine_id = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    let run = run_simulation_validation_json_with_external_reference(
        &payload,
        &ExternalSimulationValidationReferenceOptions { engine_id },
    );
    Ok(run.raw)
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                raw_args
                    .first()
                    .map(String::as_str)
                    .unwrap_or("simulation_validation_reference")
            )
        );
        return;
    }
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!("{}", error_json(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(raw_args, &stdin) {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize simulation validation output")
        ),
        Err(err) => {
            println!("{}", error_json(err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_queue_cli_runs_rust_reference() {
        let output = run(
            vec![
                "simulation_validation_reference".to_string(),
                "--engine".to_string(),
                "simpy".to_string(),
            ],
            r#"{
                "kind": "simulation-validation",
                "model_format": "json-event-network",
                "model": {
                    "servers": 1,
                    "arrival_times": [0.0, 1.0],
                    "service_times": [1.0, 1.0]
                },
                "expected_trace_properties": [
                    "queue_length_never_negative",
                    "departures_after_arrivals"
                ],
                "metric_expectations": [
                    {
                        "name": "jobs_completed",
                        "comparison": "==",
                        "target": 2.0,
                        "tolerance": 0.0
                    }
                ]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["simulator"], "rust:single-station-des-for-simpy");
        assert_eq!(output["metrics"]["jobs_completed"], 2.0);
    }

    #[test]
    fn mobility_cli_uses_engine_alias_label() {
        let output = run(
            vec![
                "simulation_validation_reference".to_string(),
                "--engine=sumo".to_string(),
            ],
            r#"{
                "kind": "simulation-validation",
                "model_format": "json-mobility-network",
                "model": {
                    "routes": [
                        {"depart": 0.0, "segments": [1.5, 2.5]}
                    ]
                },
                "expected_trace_properties": ["vehicles_complete"]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["simulator"], "rust:mobility-network-for-sumo");
        assert_eq!(output["metrics"]["vehicles_completed"], 1.0);
    }
}
