use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_cp_sat_reference::{
    solve_cp_sat_json_with_external_reference, ExternalCpSatReferenceOptions,
    ExternalCpSatReferenceSolver, ExternalCpSatReferenceStatus,
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

#[derive(Clone, Debug)]
struct CliArgs {
    solver: ExternalCpSatReferenceSolver,
    enumerate_solutions: Option<usize>,
    max_nodes: Option<usize>,
    assumption_core: bool,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--solver auto|rust-enumeration|python-enumeration-legacy|ortools|ortools-cp-sat] [--enumerate-solutions N] [--max-nodes N] [--assumption-core]"
    )
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

fn parse_solver(value: &str) -> Result<ExternalCpSatReferenceSolver, CliError> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "auto" => Ok(ExternalCpSatReferenceSolver::Auto),
        "rust" | "native" | "exact" | "rust-enumeration" | "rust-exact" | "rust:exact"
        | "rust:enumeration" => Ok(ExternalCpSatReferenceSolver::RustEnumeration),
        "python-enumeration" | "python-enumeration-legacy" => {
            Ok(ExternalCpSatReferenceSolver::PythonEnumeration)
        }
        "ortools" | "or-tools" | "google-ortools" | "google-or-tools" | "ortools-cp-sat"
        | "ortools:cp-sat" | "or-tools-cp-sat" => Ok(ExternalCpSatReferenceSolver::OrToolsCpSat),
        other => Err(CliError(format!("unknown solver {other:?}"))),
    }
}

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<CliArgs, CliError> {
    let mut parsed = CliArgs {
        solver: ExternalCpSatReferenceSolver::Auto,
        enumerate_solutions: None,
        max_nodes: None,
        assumption_core: false,
    };
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
                parsed.solver = parse_solver(&value)?;
            }
            "--enumerate-solutions" => {
                let value =
                    next_option_value(program, "--enumerate-solutions", inline_value, &mut values)?;
                let limit = value.parse::<usize>().map_err(|err| {
                    CliError(format!("--enumerate-solutions must be an integer: {err}"))
                })?;
                if limit == 0 {
                    return Err(CliError(
                        "--enumerate-solutions must be positive".to_string(),
                    ));
                }
                parsed.enumerate_solutions = Some(limit);
            }
            "--max-nodes" => {
                let value = next_option_value(program, "--max-nodes", inline_value, &mut values)?;
                let limit = value
                    .parse::<usize>()
                    .map_err(|err| CliError(format!("--max-nodes must be an integer: {err}")))?;
                if limit == 0 {
                    return Err(CliError("--max-nodes must be positive".to_string()));
                }
                parsed.max_nodes = Some(limit);
            }
            "--assumption-core" => {
                if inline_value.is_some() {
                    return Err(CliError(format!(
                        "--assumption-core does not take a value\n{}",
                        usage(program)
                    )));
                }
                parsed.assumption_core = true;
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(parsed)
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:cp-sat-reference",
        "assignment": [],
        "objective": null,
        "nodes": 0,
        "message": message.into(),
    })
}

fn run(
    raw_args: Vec<String>,
    stdin: &str,
) -> Result<(Value, ExternalCpSatReferenceStatus), CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "cp_sat_reference".to_string());
    let args = parse_args(&program, raw_args.into_iter().skip(1))?;
    let model = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    let run = solve_cp_sat_json_with_external_reference(
        &model,
        &ExternalCpSatReferenceOptions {
            solver: args.solver,
            enumerate_solutions: args.enumerate_solutions,
            max_nodes: args.max_nodes,
            assumption_core: args.assumption_core,
        },
    );
    Ok((run.raw, run.status))
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
                    .unwrap_or("cp_sat_reference")
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
        Ok((output, status)) => {
            println!(
                "{}",
                serde_json::to_string(&output).expect("serialize CP-SAT reference output")
            );
            if status == ExternalCpSatReferenceStatus::Unavailable {
                std::process::exit(2);
            }
        }
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

    static CP_SAT_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn cp_sat_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "CP_SAT_REFERENCE_FORCE_PYTHON",
            "CP_SAT_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "variables": [
            {"name": "x", "domain": [0, 1]},
            {"name": "y", "domain": [0, 1]}
        ],
        "constraints": [
            {
                "kind": "exactly_one",
                "literals": [
                    {"var": 0, "positive": true},
                    {"var": 1, "positive": true}
                ]
            }
        ],
        "objective": {
            "sense": "max",
            "terms": [
                {"var": 0, "coeff": 2},
                {"var": 1, "coeff": 1}
            ]
        }
    }"#;

    #[test]
    fn rust_enumeration_solves_reference_model() {
        let (output, status) = run(
            vec![
                "cp_sat_reference".to_string(),
                "--solver".to_string(),
                "rust-enumeration".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(output["solver"], "rust:cp-native-enumeration");
        assert_eq!(output["assignment"], json!([1, 0]));
        assert_eq!(output["objective"], 2);
    }

    #[test]
    fn rust_solution_enumeration_returns_pool() {
        let (output, status) = run(
            vec![
                "cp_sat_reference".to_string(),
                "--solver".to_string(),
                "rust-enumeration".to_string(),
                "--enumerate-solutions".to_string(),
                "2".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(output["solver"], "rust:cp-native-solution-enumeration");
        assert_eq!(output["solutions"].as_array().expect("solutions").len(), 2);
    }

    #[test]
    fn rust_solution_enumeration_accepts_max_nodes() {
        let (output, status) = run(
            vec![
                "cp_sat_reference".to_string(),
                "--solver".to_string(),
                "rust-enumeration".to_string(),
                "--enumerate-solutions".to_string(),
                "2".to_string(),
                "--max-nodes".to_string(),
                "1".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(status, ExternalCpSatReferenceStatus::Exhausted);
        assert_eq!(output["solver"], "rust:cp-native-solution-enumeration");
        assert_eq!(output["exhausted"], false);
        assert!(output["message"]
            .as_str()
            .is_some_and(|message| message.contains("node limit")));
    }

    #[test]
    fn ortools_cp_sat_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = CP_SAT_CLI_ENV_LOCK
            .lock()
            .expect("lock CP-SAT CLI env guard");
        let _force_python_guards = cp_sat_force_python_off_guards();
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-cp-sat-cli");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-cp-sat-cli");

        let (output, status) = run(
            vec![
                "cp_sat_reference".to_string(),
                "--solver=ortools:cp-sat".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(
            output["solver"],
            "rust:registered-cp-sat-fallback-for-ortools-cp-sat"
        );
        assert_eq!(output["assignment"], json!([1, 0]));
        assert_eq!(output["objective"], 2);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn python_enumeration_cli_alias_uses_rust_backend() {
        let (output, status) = run(
            vec![
                "cp_sat_reference".to_string(),
                "--solver".to_string(),
                "python-enumeration".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(status, ExternalCpSatReferenceStatus::Optimal);
        assert_eq!(output["solver"], "rust:cp-native-enumeration");
        assert_eq!(output["assignment"], json!([1, 0]));
        assert_eq!(output["objective"], 2);
    }
}
