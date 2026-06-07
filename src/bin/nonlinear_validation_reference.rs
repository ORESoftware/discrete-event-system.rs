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
        "usage: {program} [--solver auto|fallback|rust:nonlinear-validation-reference|scipy|scipy:slsqp|ipopt|bonmin|minotaur|couenne|symphony|knitro|mosek|mosek-cli|baron|copt|copt-cli|casadi|nlopt|nlopt:bobyqa|nlopt-cli]"
    )
}

fn parse_solver(value: &str) -> Result<ExternalNonlinearValidationReferenceSolver, CliError> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "auto" => Ok(ExternalNonlinearValidationReferenceSolver::Auto),
        "fallback"
        | "builtin"
        | "rust"
        | "rust-fallback"
        | "rust:fallback"
        | "rust-nonlinear-validation-reference"
        | "rust:nonlinear-validation-reference"
        | "builtin-nlp-pattern-search"
        | "builtin:nlp-pattern-search" => Ok(ExternalNonlinearValidationReferenceSolver::Fallback),
        "scipy"
        | "scipy-minimize"
        | "scipy:minimize"
        | "scipy-slsqp"
        | "scipy:slsqp"
        | "builtin-nlp-pattern-search-for-scipy"
        | "builtin:nlp-pattern-search-for-scipy" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Scipy)
        }
        "ipopt"
        | "ipopt-default"
        | "ipopt:default"
        | "builtin-nlp-pattern-search-for-ipopt"
        | "builtin:nlp-pattern-search-for-ipopt" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Ipopt)
        }
        "bonmin"
        | "builtin-nlp-pattern-search-for-bonmin"
        | "builtin:nlp-pattern-search-for-bonmin" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Bonmin)
        }
        "minotaur"
        | "builtin-nlp-pattern-search-for-minotaur"
        | "builtin:nlp-pattern-search-for-minotaur" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Minotaur)
        }
        "couenne"
        | "builtin-nlp-pattern-search-for-couenne"
        | "builtin:nlp-pattern-search-for-couenne" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Couenne)
        }
        "symphony"
        | "builtin-nlp-pattern-search-for-symphony"
        | "builtin:nlp-pattern-search-for-symphony" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Symphony)
        }
        "knitro"
        | "artelys-knitro"
        | "builtin-nlp-pattern-search-for-knitro"
        | "builtin:nlp-pattern-search-for-knitro" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Knitro)
        }
        "mosek"
        | "mosek-cli"
        | "mosek:cli"
        | "builtin-nlp-pattern-search-for-mosek"
        | "builtin:nlp-pattern-search-for-mosek" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Mosek)
        }
        "baron"
        | "builtin-nlp-pattern-search-for-baron"
        | "builtin:nlp-pattern-search-for-baron" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Baron)
        }
        "copt"
        | "copt-cli"
        | "copt-cmd-cli"
        | "copt:cli"
        | "copt-cmd:cli"
        | "builtin-nlp-pattern-search-for-copt"
        | "builtin:nlp-pattern-search-for-copt" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Copt)
        }
        "casadi"
        | "casadi-ipopt"
        | "casadi:ipopt"
        | "builtin-nlp-pattern-search-for-casadi"
        | "builtin:nlp-pattern-search-for-casadi" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Casadi)
        }
        "nlopt"
        | "nlopt:bobyqa"
        | "nlopt-cobyla"
        | "nlopt:cobyla"
        | "nlopt-direct"
        | "nlopt:direct"
        | "nlopt-nelder-mead"
        | "nlopt:nelder-mead"
        | "builtin-nlp-pattern-search-for-nlopt"
        | "builtin:nlp-pattern-search-for-nlopt" => {
            Ok(ExternalNonlinearValidationReferenceSolver::Nlopt)
        }
        "nlopt-cli"
        | "nlopt:cli"
        | "builtin-nlp-pattern-search-for-nlopt-cli"
        | "builtin:nlp-pattern-search-for-nlopt-cli" => {
            Ok(ExternalNonlinearValidationReferenceSolver::NloptCli)
        }
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
    use std::sync::Mutex;

    static NONLINEAR_VALIDATION_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
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

    fn nonlinear_validation_python_off_guards() -> Vec<EnvVarGuard> {
        vec![
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-nonlinear-cli"),
            EnvVarGuard::set("PYTHON", "/definitely/not-python-for-nonlinear-cli"),
            EnvVarGuard::set(
                "SCIPY_OPTIMIZE_PYTHON",
                "/definitely/not-python-for-nonlinear-cli",
            ),
            EnvVarGuard::set(
                "ORES_SCIPY_OPTIMIZE_PYTHON",
                "/definitely/not-python-for-nonlinear-cli",
            ),
            EnvVarGuard::clear("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON"),
            EnvVarGuard::clear("ORES_EXTERNAL_VALIDATION_PYTHON_IMPORT_PROBES"),
            EnvVarGuard::clear("EXTERNAL_VALIDATION_PYTHON_IMPORT_PROBES"),
            EnvVarGuard::clear("EXTERNAL_VALIDATION_PROBE_PYTHON_IMPORTS"),
        ]
    }

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
    fn parser_accepts_rust_and_external_solver_labels_used_by_validation_tools() {
        for raw in [
            "rust:nonlinear-validation-reference",
            "builtin:nlp-pattern-search",
            "RuSt_FaLlBaCk",
        ] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalNonlinearValidationReferenceSolver::Fallback,
                "{raw}"
            );
        }
        for raw in [
            "scipy:slsqp",
            "ipopt:default",
            "mosek-cli",
            "copt-cmd-cli",
            "casadi:ipopt",
            "nlopt:bobyqa",
            "nlopt:cli",
            "NlOpT_NeLdEr_MeAd",
        ] {
            assert!(parse_solver(raw).is_ok(), "{raw}");
        }
        for (raw, expected) in [
            (
                "builtin:nlp-pattern-search-for-scipy",
                ExternalNonlinearValidationReferenceSolver::Scipy,
            ),
            (
                "builtin:nlp-pattern-search-for-ipopt",
                ExternalNonlinearValidationReferenceSolver::Ipopt,
            ),
            (
                "builtin:nlp-pattern-search-for-casadi",
                ExternalNonlinearValidationReferenceSolver::Casadi,
            ),
            (
                "mosek:cli",
                ExternalNonlinearValidationReferenceSolver::Mosek,
            ),
            ("copt:cli", ExternalNonlinearValidationReferenceSolver::Copt),
            (
                "builtin:nlp-pattern-search-for-nlopt-cli",
                ExternalNonlinearValidationReferenceSolver::NloptCli,
            ),
        ] {
            assert_eq!(parse_solver(raw).unwrap(), expected, "{raw}");
        }
    }

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

    #[test]
    fn scipy_cli_alias_uses_rust_pattern_search_without_python() {
        let _lock = NONLINEAR_VALIDATION_CLI_ENV_LOCK
            .lock()
            .expect("nonlinear validation CLI env lock poisoned");
        let _guards = nonlinear_validation_python_off_guards();
        let output = run(
            vec![
                "nonlinear_validation_reference".to_string(),
                "--solver=scipy:slsqp".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "builtin:nlp-pattern-search-for-scipy");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
        assert!(output["objective"]
            .as_f64()
            .is_some_and(|value| value <= 1e-6));
    }

    #[test]
    fn commercial_cli_labels_use_rust_pattern_search_fallback() {
        for (solver, expected_label) in [
            ("mosek-cli", "builtin:nlp-pattern-search-for-mosek"),
            ("copt-cli", "builtin:nlp-pattern-search-for-copt"),
        ] {
            let output = run(
                vec![
                    "nonlinear_validation_reference".to_string(),
                    "--solver".to_string(),
                    solver.to_string(),
                ],
                SAMPLE,
            )
            .expect("run");

            assert_eq!(output["status"], "optimal", "{solver}: {output:?}");
            assert_eq!(output["solver"], expected_label, "{solver}: {output:?}");
            assert_eq!(output["x"].as_array().expect("x").len(), 2);
            assert!(output["objective"]
                .as_f64()
                .is_some_and(|value| value <= 1e-6));
        }
    }
}
