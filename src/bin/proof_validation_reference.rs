use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_validation_tools::run_proof_validation_json_with_rust_reference;
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

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<String, CliError> {
    let mut tool = "drat".to_string();
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
                tool = next_option_value(program, "--tool", inline_value, &mut values)?;
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
        "kind": "proof-validation-result",
        "tool": "rust:proof-validation-reference",
        "validator": "rust:proof-validation-reference",
        "status": "ok",
        "verdict": "invalid",
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "proof_validation_reference".to_string());
    let tool = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    Ok(run_proof_validation_json_with_rust_reference(
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
                    .unwrap_or("proof_validation_reference")
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
            serde_json::to_string(&output).expect("serialize proof validation output")
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
    fn drat_cli_accepts_empty_clause_proof() {
        let output = run(
            vec![
                "proof_validation_reference".to_string(),
                "--tool".to_string(),
                "drat".to_string(),
            ],
            r#"{
                "cnf": "p cnf 1 2\n1 0\n-1 0\n",
                "proof": "0\n"
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:small-cnf-proof-for-drat");
        assert_eq!(output["cnf_status"], "unsat");
    }

    #[test]
    fn artifact_payloads_are_supported_in_rust() {
        let output = run(
            vec![
                "proof_validation_reference".to_string(),
                "--tool=lrat".to_string(),
            ],
            r#"{
                "artifacts": [
                    {"name": "model", "content": "p cnf 1 2\n1 0\n-1 0\n"},
                    {"name": "lrat", "content": "1 0 0\n"}
                ]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:small-cnf-proof-for-lrat");
        assert_eq!(output["cnf_status"], "unsat");
    }

    #[test]
    fn veripb_cli_checks_unsat_opb_derivation() {
        let output = run(
            vec![
                "proof_validation_reference".to_string(),
                "--tool".to_string(),
                "veripb".to_string(),
            ],
            r#"{
                "kind": "opb-proof-validation",
                "opb": "1 x >= 1;\n1 x <= 0;\n",
                "proof": "u 1 x >= 1;\n"
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:small-opb-proof-for-veripb");
        assert_eq!(output["pb_status"], "unsat");
    }

    #[test]
    fn cli_tool_normalization_is_owned_by_rust_dispatch() {
        let lrat = run(
            vec![
                "proof_validation_reference".to_string(),
                "--tool".to_string(),
                "CaKe_LpR".to_string(),
            ],
            r#"{
                "cnf": "p cnf 1 2\n1 0\n-1 0\n",
                "proof": "1 0 0\n"
            }"#,
        )
        .expect("run");

        assert_eq!(lrat["status"], "ok");
        assert_eq!(lrat["verdict"], "valid");
        assert_eq!(lrat["validator"], "builtin:small-cnf-proof-for-cake-lpr");

        let veripb = run(
            vec![
                "proof_validation_reference".to_string(),
                "--tool=VeRiPb_Checker".to_string(),
            ],
            r#"{
                "kind": "opb-proof-validation",
                "opb": "1 x >= 1;\n1 x <= 0;\n",
                "proof": "u 1 x >= 1;\n"
            }"#,
        )
        .expect("run");

        assert_eq!(veripb["status"], "ok");
        assert_eq!(veripb["verdict"], "valid");
        assert_eq!(
            veripb["validator"],
            "builtin:small-opb-proof-for-veripb-checker"
        );
    }
}
