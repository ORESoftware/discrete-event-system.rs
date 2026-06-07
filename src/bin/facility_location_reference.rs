use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_facility_location_reference::{
    solve_facility_location_with_external_reference, ExternalFacilityLocationReferenceOptions,
    ExternalFacilityLocationReferenceSolution, ExternalFacilityLocationReferenceSolver,
};
use des_engine::des::general::facility_location::FacilityLocationProblem;
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
        "usage: {program} [--solver auto|fallback|rust-exact|rust:exact-facility-location|ortools|ortools:cp-sat-facility-location]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalFacilityLocationReferenceSolver, CliError> {
    let mut solver = ExternalFacilityLocationReferenceSolver::Auto;
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
                    "auto" | "default" => ExternalFacilityLocationReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalFacilityLocationReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "facility-location"
                    | "exact-facility-location"
                    | "rust-facility-location"
                    | "rust:facility-location"
                    | "rust-exact-facility-location"
                    | "rust:exact-facility-location" => {
                        ExternalFacilityLocationReferenceSolver::RustExact
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-cp-sat"
                    | "ortools:cp-sat"
                    | "or-tools-cp-sat"
                    | "cp-sat-facility-location"
                    | "ortools-facility-location"
                    | "ortools:facility-location"
                    | "ortools-cp-sat-facility-location"
                    | "ortools:cp-sat-facility-location"
                    | "or-tools-cp-sat-facility-location" => {
                        ExternalFacilityLocationReferenceSolver::OrTools
                    }
                    _ => {
                        return Err(CliError(format!(
                            "unknown solver {value:?}\n{}",
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

fn parse_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn parse_string_array(raw: &Value, primary: &str, alias: &str) -> Result<Vec<String>, String> {
    raw.get(primary)
        .or_else(|| raw.get(alias))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{primary} must be non-empty"))
        .map(|values| values.iter().map(parse_string).collect())
}

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_number_array(raw: &Value, field: &str) -> Result<Vec<f64>, String> {
    raw.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| parse_number(value, format!("{field}[{index}] must be numeric")))
        .collect()
}

fn parse_number_matrix(raw: &Value, field: &str) -> Result<Vec<Vec<f64>>, String> {
    raw.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.as_array()
                .ok_or_else(|| format!("{field}[{row_index}] must be an array"))?
                .iter()
                .enumerate()
                .map(|(column_index, value)| {
                    parse_number(
                        value,
                        format!("{field}[{row_index}][{column_index}] must be numeric"),
                    )
                })
                .collect()
        })
        .collect()
}

fn parse_facility_location_problem(raw: &Value) -> Result<FacilityLocationProblem, String> {
    Ok(FacilityLocationProblem {
        facility_ids: parse_string_array(raw, "facilities", "facilityIds")?,
        customer_ids: parse_string_array(raw, "customers", "customerIds")?,
        fixed_costs: parse_number_array(raw, "fixedCosts")?,
        service_costs: parse_number_matrix(raw, "serviceCosts")?,
    })
}

fn solution_json(solution: &ExternalFacilityLocationReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "openFacilityIndices": solution.open_facility_indices,
        "openFacilities": solution.open_facility_ids,
        "assignments": solution.assignments.iter().map(|assignment| json!({
            "customerIndex": assignment.customer_index,
            "customer": assignment.customer_id,
            "facilityIndex": assignment.facility_index,
            "facility": assignment.facility_id,
            "cost": assignment.cost,
        })).collect::<Vec<_>>(),
        "objective": solution.objective,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_open_facility_indices.is_empty()
        || !solution.ortools_assignments.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsOpenFacilityIndices"] = json!(solution.ortools_open_facility_indices);
        output["ortoolsOpenFacilities"] = json!(solution.ortools_open_facility_ids);
        output["ortoolsAssignments"] = json!(solution
            .ortools_assignments
            .iter()
            .map(|assignment| json!({
                "customerIndex": assignment.customer_index,
                "customer": assignment.customer_id,
                "facilityIndex": assignment.facility_index,
                "facility": assignment.facility_id,
                "cost": assignment.cost,
            }))
            .collect::<Vec<_>>());
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:facility-location-reference",
        "openFacilityIndices": [],
        "openFacilities": [],
        "assignments": [],
        "objective": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "facility_location_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_facility_location_problem(&payload).map_err(CliError)?;
    let solution = solve_facility_location_with_external_reference(
        &problem,
        &ExternalFacilityLocationReferenceOptions { solver },
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
                    .unwrap_or("facility_location_reference")
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
                serde_json::to_string(&output).expect("serialize facility-location output")
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

    static FACILITY_LOCATION_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn facility_location_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "FACILITY_LOCATION_REFERENCE_FORCE_PYTHON",
            "FACILITY_LOCATION_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "facilities": ["North", "Central", "South"],
        "customers": ["A", "B", "C", "D", "E"],
        "fixedCosts": [6, 10, 6],
        "serviceCosts": [
            [2, 4, 7, 9, 8],
            [5, 3, 4, 4, 6],
            [9, 7, 5, 3, 2]
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_exact() {
        let output = run(
            vec![
                "facility_location_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-facility-location");
        assert_eq!(output["openFacilities"], json!(["North", "South"]));
        assert_eq!(output["objective"], 28.0);
    }

    #[test]
    fn accepts_alias_ids_and_rust_exact_alias() {
        let output = run(
            vec![
                "facility_location_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            r#"{
                "facilityIds": ["A", "B"],
                "customerIds": ["C"],
                "fixedCosts": [1, 1],
                "serviceCosts": [[1], [1]]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["openFacilities"], json!(["A"]));
        assert_eq!(output["objective"], 2.0);
    }

    #[test]
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = FACILITY_LOCATION_CLI_ENV_LOCK
            .lock()
            .expect("lock facility-location CLI env guard");
        let _force_python_guards = facility_location_force_python_off_guards();
        let _python_bin_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-facility-location-cli",
        );
        let _python_guard =
            EnvVarGuard::set("PYTHON", "/definitely/not-python-for-facility-location-cli");

        let output = run(
            vec![
                "facility_location_reference".to_string(),
                "--solver=ortools:cp-sat-facility-location".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:registered-facility-location-fallback-for-ortools"
        );
        assert_eq!(output["openFacilities"], json!(["North", "South"]));
        assert_eq!(output["objective"], 28.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_facility_location_solver_aliases_used_by_validation_tools() {
        let rust_aliases = [
            "rust",
            "native",
            "exact",
            "rust:exact",
            "rust_exact_facility_location",
            "rust:exact-facility-location",
        ];
        for alias in rust_aliases {
            let solver = parse_solver(
                "facility_location_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(solver, ExternalFacilityLocationReferenceSolver::RustExact);
        }

        let ortools_aliases = [
            "or-tools",
            "google-ortools",
            "ortools:cp-sat",
            "ortools_cp_sat_facility_location",
            "ortools:cp-sat-facility-location",
        ];
        for alias in ortools_aliases {
            let solver = parse_solver(
                "facility_location_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(solver, ExternalFacilityLocationReferenceSolver::OrTools);
        }

        let fallback = parse_solver(
            "facility_location_reference",
            ["--solver=rust:fallback".to_string()],
        )
        .expect("fallback alias");
        assert_eq!(fallback, ExternalFacilityLocationReferenceSolver::Fallback);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["facility_location_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("facilities"));
    }
}
