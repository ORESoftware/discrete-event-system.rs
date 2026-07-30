use std::env;
use std::error::Error;
use std::fmt;
use std::io;

use des_engine::des::general::external_validation_tools::run_formal_benchmark_validation_json_with_rust_reference;
use serde_json::{json, Value};

mod common;

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage(program: &str) -> String {
    format!("usage: {program} [--tool TOOL]")
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
    let mut tool = None::<String>;
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
            "--tool" => {
                tool = Some(
                    common::validate_tool_id(next_option_value(
                        program,
                        "--tool",
                        inline_value,
                        &mut values,
                    )?)
                    .map_err(CliError)?,
                );
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(tool)
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "failed",
        "verdict": "failure",
        "validator": "rust:formal-benchmark-validation-reference",
        "message": message.into(),
        "checks": [],
        "stdout": "",
        "stderr": "",
    })
}

fn selected_tool(payload: &Value, tool_override: Option<String>) -> String {
    tool_override
        .or_else(|| {
            payload
                .get("tool")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "auto".to_string())
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "formal_benchmark_validation_reference".to_string());
    let tool_override = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    let tool =
        common::validate_tool_id(selected_tool(&payload, tool_override)).map_err(CliError)?;
    Ok(run_formal_benchmark_validation_json_with_rust_reference(
        &payload, &tool,
    ))
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
                    .unwrap_or("formal_benchmark_validation_reference")
            )
        );
        return;
    }
    let stdin = match common::read_validation_input(io::stdin().lock()) {
        Ok(stdin) => stdin,
        Err(err) => {
            println!("{}", error_json(err));
            std::process::exit(1);
        }
    };
    match run(raw_args, &stdin) {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize formal validation output")
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
    fn tla_cli_runs_rust_structural_validator() {
        let output = run(
            vec![
                "formal_benchmark_validation_reference".to_string(),
                "--tool".to_string(),
                "tla".to_string(),
            ],
            r#"{
                "kind": "tla-validation",
                "module": "---- MODULE Queue ----\nInit == TRUE\nNext == TRUE\nSpec == Init /\\ [][Next]_vars\n====\n"
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:tla-structural");
    }

    #[test]
    fn benchmark_manifest_cli_validates_entries() {
        let output = run(
            vec!["formal_benchmark_validation_reference".to_string()],
            r#"{
                "kind": "external-benchmark-manifest",
                "suite": "smoke",
                "entries": [
                    {"name": "tiny-lp", "family": "lp", "format": "lp", "path": "tiny.lp"}
                ]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:benchmark-manifest");
    }

    #[test]
    fn security_protocol_alias_uses_rust_structural_validator() {
        let output = run(
            vec![
                "formal_benchmark_validation_reference".to_string(),
                "--tool=tamarin".to_string(),
            ],
            r#"{
                "kind": "security-protocol-validation",
                "model": "theory Handshake begin\nrule Send: [ ] --[ Secret(x) ]-> [ ]\nlemma secrecy: \"All x #i. Secret(x) @ i ==> not False\"\nend\n",
                "properties": ["lemma secrecy"]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:security-protocol-structural");
    }

    #[test]
    fn external_formal_tool_aliases_use_rust_structural_validators() {
        let alloy = run(
            vec![
                "formal_benchmark_validation_reference".to_string(),
                "--tool".to_string(),
                "alloy".to_string(),
            ],
            r#"{
                "kind": "alloy-validation",
                "model": "module soccer\nsig Team {}\npred show {}\n",
                "commands": ["run show"]
            }"#,
        )
        .expect("run alloy validation");
        assert_eq!(alloy["status"], "ok");
        assert_eq!(alloy["verdict"], "valid");
        assert_eq!(alloy["validator"], "builtin:alloy-structural");

        let promela = run(
            vec![
                "formal_benchmark_validation_reference".to_string(),
                "--tool=spin".to_string(),
            ],
            r#"{
                "kind": "spin-validation",
                "model": "init { skip; }\n",
                "properties": ["ltl eventually_done { <> true }"]
            }"#,
        )
        .expect("run spin validation");
        assert_eq!(promela["status"], "ok");
        assert_eq!(promela["verdict"], "valid");
        assert_eq!(promela["validator"], "builtin:promela-structural");

        let uppaal = run(
            vec![
                "formal_benchmark_validation_reference".to_string(),
                "--tool".to_string(),
                "uppaal".to_string(),
            ],
            r#"{
                "kind": "uppaal-validation",
                "model": "<nta><template><name>T</name><location id=\"l0\"/><init ref=\"l0\"/><transition><source ref=\"l0\"/><target ref=\"l0\"/></transition></template></nta>",
                "queries": ["A[] not deadlock"]
            }"#,
        )
        .expect("run uppaal validation");
        assert_eq!(uppaal["status"], "ok");
        assert_eq!(uppaal["verdict"], "valid");
        assert_eq!(uppaal["validator"], "builtin:uppaal-structural");
    }
}
