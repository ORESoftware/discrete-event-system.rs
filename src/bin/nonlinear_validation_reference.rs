use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_nonlinear_validation_reference::{
    solve_nonlinear_validation_json_with_external_reference,
    ExternalNonlinearValidationReferenceOptions, ExternalNonlinearValidationReferenceSolver,
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
    format!(
        "usage: {program} [--solver auto|fallback|scipy|ipopt|bonmin|minotaur|couenne|symphony|knitro|mosek|baron|copt|casadi|nlopt|nlopt-cli]"
    )
}

fn parse_solver(value: &str) -> Result<ExternalNonlinearValidationReferenceSolver, CliError> {
    match value {
        "auto" => Ok(ExternalNonlinearValidationReferenceSolver::Auto),
        "fallback" | "rust" | "rust-fallback" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Fallback)
        }
        "scipy" => Ok(ExternalNonlinearValidationReferenceSolver::Scipy),
        "ipopt" => Ok(ExternalNonlinearValidationReferenceSolver::Ipopt),
        "bonmin" => Ok(ExternalNonlinearValidationReferenceSolver::Bonmin),
        "minotaur" => Ok(ExternalNonlinearValidationReferenceSolver::Minotaur),
        "couenne" => Ok(ExternalNonlinearValidationReferenceSolver::Couenne),
        "symphony" => Ok(ExternalNonlinearValidationReferenceSolver::Symphony),
        "knitro" => Ok(ExternalNonlinearValidationReferenceSolver::Knitro),
        "mosek" => Ok(ExternalNonlinearValidationReferenceSolver::Mosek),
        "baron" => Ok(ExternalNonlinearValidationReferenceSolver::Baron),
        "copt" => Ok(ExternalNonlinearValidationReferenceSolver::Copt),
        "casadi" => Ok(ExternalNonlinearValidationReferenceSolver::Casadi),
        "nlopt" => Ok(ExternalNonlinearValidationReferenceSolver::Nlopt),
        "nlopt-cli" => Ok(ExternalNonlinearValidationReferenceSolver::NloptCli),
        other => Err(CliError(format!("unknown solver {other:?}"))),
    }
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
) -> Result<ExternalNonlinearValidationReferenceSolver, CliError> {
    let mut solver = ExternalNonlinearValidationReferenceSolver::Auto;
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
            "--solver" => {
                let value = next_option_value(program, "--solver", inline_value, &mut values)?;
                solver = parse_solver(&value)?;
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(solver)
}

fn solution_json(
    solution: &des_engine::des::general::external_nonlinear_validation_reference::ExternalNonlinearValidationReferenceSolution,
) -> Value {
    json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "x": solution.x,
        "objective": solution.objective,
        "message": solution.message,
        "iterations": solution.iterations.unwrap_or(0),
    })
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "failed",
        "solver": "rust:nonlinear-validation-reference",
        "x": [],
        "objective": null,
        "message": message.into(),
        "iterations": 0,
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "nonlinear_validation_reference".to_string());
    let solver = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    let solution = solve_nonlinear_validation_json_with_external_reference(
        payload,
        &ExternalNonlinearValidationReferenceOptions { solver },
    );
    Ok(solution_json(&solution))
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
                    .unwrap_or("nonlinear_validation_reference")
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
            serde_json::to_string(&output).expect("serialize nonlinear validation output")
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

    const SAMPLE: &str = r#"{
        "kind": "nonlinear-validation",
        "variables": [
            {"name": "x", "lb": 0.0, "ub": 3.0, "start": 0.2},
            {"name": "y", "lb": 0.0, "ub": 3.0, "start": 0.2}
        ],
        "objective": "(x - 1)**2 + (y - 2)**2",
        "constraints": [
            {"name": "demand", "expr": "x + y", "sense": ">=", "rhs": 1.0}
        ],
        "sense": "min"
    }"#;

    #[test]
    fn rust_fallback_solves_small_expression_model() {
        let output = run(
            vec![
                "nonlinear_validation_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "builtin:nlp-pattern-search");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
        assert!(output["objective"]
            .as_f64()
            .is_some_and(|value| value <= 1e-6));
    }

    #[test]
    fn registered_solver_label_is_preserved_for_fallback_validation() {
        let output = run(
            vec![
                "nonlinear_validation_reference".to_string(),
                "--solver".to_string(),
                "nlopt".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "builtin:nlp-pattern-search-for-nlopt");
    }
}
