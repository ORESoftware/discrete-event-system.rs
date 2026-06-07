use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_validation_tools::run_output_validation_json_with_rust_reference;
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
    let mut tool = "json-schema".to_string();
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
        "status": "failed",
        "verdict": "failure",
        "validator": "rust:output-validation-reference",
        "message": message.into(),
        "errors": [],
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "output_validation_reference".to_string());
    let tool = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    Ok(run_output_validation_json_with_rust_reference(
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
                    .unwrap_or("output_validation_reference")
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
            serde_json::to_string(&output).expect("serialize output-validation output")
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
    fn json_schema_cli_uses_rust_reference() {
        let output = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool".to_string(),
                "json-schema".to_string(),
            ],
            r#"{
                "schema": {
                    "type": "object",
                    "properties": {"score": {"type": "integer", "minimum": 0}},
                    "required": ["score"]
                },
                "instance": {"score": 3}
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(output["validator"], "builtin:json-schema-subset");
    }

    #[test]
    fn table_alias_preserves_registered_validator_label() {
        let output = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool=pandera".to_string(),
            ],
            r#"{
                "kind": "table-validation",
                "schema": {
                    "columns": {"score": {"type": "number", "required": true}},
                    "minRows": 1
                },
                "rows": [{"score": 4.0}]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(
            output["validator"],
            "builtin:table-schema-subset-for-pandera"
        );
    }

    #[test]
    fn profile_aliases_reach_rust_only_structural_checks() {
        let output = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool".to_string(),
                "whylogs".to_string(),
            ],
            r#"{
                "kind": "profile-validation",
                "profile": {
                    "row_count": 8,
                    "features": {
                        "score": {
                            "type": "number",
                            "count": 8,
                            "missing": 0,
                            "min": 0.0,
                            "max": 3.0,
                            "mean": 1.5
                        }
                    }
                },
                "constraints": [
                    {"feature": "score", "metric": "mean", "comparison": "<=", "target": 2.0}
                ]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "valid");
        assert_eq!(
            output["validator"],
            "builtin:data-profile-structural-for-whylogs"
        );
    }

    #[test]
    fn structural_validator_aliases_run_through_rust_cli() {
        let yaml = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool=yamllint".to_string(),
            ],
            r#"{
                "kind": "yaml-validation",
                "yaml": "---\nname: soccer-learning\nsteps:\n  - simulate\n  - validate\n"
            }"#,
        )
        .expect("run yaml validation");
        assert_eq!(yaml["status"], "ok");
        assert_eq!(yaml["verdict"], "valid");
        assert_eq!(yaml["validator"], "builtin:yaml-structural");

        let graphql = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool".to_string(),
                "graphql-inspector".to_string(),
            ],
            r#"{
                "kind": "graphql-schema-validation",
                "schema": "type Query { score: Int }\n"
            }"#,
        )
        .expect("run graphql validation");
        assert_eq!(graphql["status"], "ok");
        assert_eq!(graphql["verdict"], "valid");
        assert_eq!(graphql["validator"], "builtin:graphql-schema-structural");

        let sql = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool".to_string(),
                "sqlfluff".to_string(),
            ],
            r#"{
                "sql": "select id, score from where score > 0"
            }"#,
        )
        .expect("run sql validation");
        assert_eq!(sql["status"], "ok");
        assert_eq!(sql["verdict"], "invalid");
        assert_eq!(sql["validator"], "builtin:sql-structural-for-sqlfluff");

        let csv = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool=csvlint".to_string(),
            ],
            r#"{
                "schema": {
                    "columns": {"episode": {"type": "integer", "required": true}},
                    "minRows": 1
                },
                "csv": "episode\n1\n"
            }"#,
        )
        .expect("run csv validation");
        assert_eq!(csv["status"], "ok");
        assert_eq!(csv["verdict"], "valid");
        assert_eq!(csv["validator"], "builtin:table-schema-subset");

        let frictionless = run(
            vec![
                "output_validation_reference".to_string(),
                "--tool".to_string(),
                "frictionless".to_string(),
            ],
            r#"{
                "kind": "data-package-validation",
                "package": {
                    "profile": "tabular-data-package",
                    "resources": [
                        {
                            "name": "episodes",
                            "path": "episodes.csv",
                            "schema": {
                                "fields": [
                                    {"name": "episode", "type": "integer", "constraints": {"required": true, "minimum": 1}},
                                    {"name": "score", "type": "number", "constraints": {"minimum": 0}}
                                ],
                                "primaryKey": "episode"
                            },
                            "rows": [
                                {"episode": 1, "score": 3.5},
                                {"episode": 2, "score": 2.0}
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("run frictionless validation");
        assert_eq!(frictionless["status"], "ok");
        assert_eq!(frictionless["verdict"], "valid");
        assert_eq!(
            frictionless["validator"],
            "builtin:frictionless-data-package-structural"
        );
    }
}
