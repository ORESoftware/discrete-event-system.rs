use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_assignment_reference::{
    solve_assignment_with_external_reference, ExternalAssignmentReferenceOptions,
    ExternalAssignmentReferenceSolution, ExternalAssignmentReferenceSolver,
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
        "usage: {program} [--solver auto|fallback|rust-dp|rust-exact|ortools|linear-sum-assignment|scipy]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalAssignmentReferenceSolver, CliError> {
    let mut solver = ExternalAssignmentReferenceSolver::Auto;
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
                let value = if let Some(value) = inline_value {
                    value
                } else {
                    let value = values.next().ok_or_else(|| {
                        CliError(format!("--solver requires a value\n{}", usage(program)))
                    })?;
                    if value.starts_with("--") {
                        return Err(CliError(format!(
                            "--solver requires a value\n{}",
                            usage(program)
                        )));
                    }
                    value
                };
                let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
                solver = match normalized.as_str() {
                    "auto" => ExternalAssignmentReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalAssignmentReferenceSolver::Fallback
                    }
                    "rust" | "native" | "rust-native" | "rust-dp" | "rust:dp" | "dp"
                    | "rust-exact" | "rust:exact" | "exact" => {
                        ExternalAssignmentReferenceSolver::RustDp
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-assignment"
                    | "ortools:simple-linear-sum-assignment"
                    | "ortools-simple-linear-sum-assignment" => {
                        ExternalAssignmentReferenceSolver::OrTools
                    }
                    "scipy"
                    | "linear-sum-assignment"
                    | "scipy-linear-sum-assignment"
                    | "scipy:linear-sum-assignment"
                    | "scipy:linear-sum-assignment-validation"
                    | "scipy-linear-sum-assignment-validation" => {
                        ExternalAssignmentReferenceSolver::Scipy
                    }
                    _ => {
                        return Err(CliError(format!(
                            "unknown solver {normalized:?}\n{}",
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
    Ok(solver)
}

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_cost(raw: &Value) -> Result<Vec<Vec<f64>>, String> {
    let rows = raw
        .get("cost")
        .and_then(Value::as_array)
        .ok_or_else(|| "cost matrix must be non-empty".to_string())?;
    if rows.is_empty() {
        return Err("cost matrix must be non-empty".to_string());
    }
    rows.iter()
        .enumerate()
        .map(|(row_index, row)| {
            let values = row
                .as_array()
                .ok_or_else(|| format!("cost row {row_index} must be an array"))?;
            values
                .iter()
                .enumerate()
                .map(|(col_index, value)| {
                    parse_number(
                        value,
                        format!("cost[{row_index}][{col_index}] must be numeric"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn solution_json(solution: &ExternalAssignmentReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "assignment": solution.assignment,
        "objective": solution.objective,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_assignment.is_empty()
        || solution.ortools_objective.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsAssignment"] = json!(solution.ortools_assignment);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
    }
    if solution.scipy_status.is_some()
        || !solution.scipy_assignment.is_empty()
        || solution.scipy_objective.is_some()
    {
        output["scipyStatus"] = json!(solution.scipy_status);
        output["scipyAssignment"] = json!(solution.scipy_assignment);
        output["scipyObjective"] = json!(solution.scipy_objective);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:assignment-reference",
        "assignment": [],
        "objective": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "assignment_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let cost = parse_cost(&payload).map_err(CliError)?;
    let solution = solve_assignment_with_external_reference(
        &cost,
        &ExternalAssignmentReferenceOptions { solver },
    );
    Ok(solution_json(&solution))
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                args.first()
                    .map(String::as_str)
                    .unwrap_or("assignment_reference")
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
                serde_json::to_string(&output).expect("serialize assignment output")
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
    use std::sync::Mutex;

    static ASSIGNMENT_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn assignment_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "ASSIGNMENT_REFERENCE_FORCE_PYTHON",
            "ASSIGNMENT_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ASSIGNMENT_REFERENCE_SCIPY_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "cost": [
            [8.0, 2.0, 5.0, 9.0],
            [6.0, 4.0, 7.0, 3.0],
            [5.0, 8.0, 1.0, 6.0],
            [7.0, 3.0, 4.0, 2.0]
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_assignment_dp() {
        let output = run(
            vec![
                "assignment_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:assignment-dp");
        assert_eq!(output["assignment"], json!([1, 0, 2, 3]));
        assert_eq!(output["objective"], 11.0);
    }

    #[test]
    fn accepts_rust_exact_alias() {
        let output = run(
            vec![
                "assignment_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            r#"{"cost": [[1, 1, 4], [1, 1, 2]]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["assignment"], json!([0, 1]));
        assert_eq!(output["objective"], 2.0);
    }

    #[test]
    fn external_cli_aliases_default_to_rust_reference_without_python() {
        let _lock = ASSIGNMENT_CLI_ENV_LOCK
            .lock()
            .expect("lock assignment CLI env guard");
        let _force_python_guards = assignment_force_python_off_guards();
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-assignment-cli");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-assignment-cli");

        for (alias, expected_solver) in [
            (
                "ortools:simple-linear-sum-assignment",
                "rust:registered-assignment-fallback-for-ortools",
            ),
            (
                "scipy:linear-sum-assignment",
                "rust:registered-assignment-fallback-for-scipy",
            ),
        ] {
            let output = run(
                vec![
                    "assignment_reference".to_string(),
                    format!("--solver={alias}"),
                ],
                SAMPLE,
            )
            .expect(alias);

            assert_eq!(output["status"], "optimal", "{alias}");
            assert_eq!(output["solver"], expected_solver, "{alias}");
            assert_eq!(output["assignment"], json!([1, 0, 2, 3]), "{alias}");
            assert!(output["message"]
                .as_str()
                .expect("message")
                .contains("validated with Rust fallback"));
        }
    }

    #[test]
    fn parses_assignment_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "rust_dp",
            "rust:dp",
            "rust-exact",
            "rust:exact",
            "native",
        ] {
            assert_eq!(
                parse_solver(
                    "assignment_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalAssignmentReferenceSolver::RustDp
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools:simple-linear-sum-assignment",
        ] {
            assert_eq!(
                parse_solver(
                    "assignment_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalAssignmentReferenceSolver::OrTools
            );
        }

        for alias in [
            "scipy",
            "linear-sum-assignment",
            "scipy_linear_sum_assignment",
            "scipy:linear-sum-assignment-validation",
        ] {
            assert_eq!(
                parse_solver(
                    "assignment_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalAssignmentReferenceSolver::Scipy
            );
        }

        assert_eq!(
            parse_solver(
                "assignment_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalAssignmentReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["assignment_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("cost matrix must be non-empty"));
    }
}
