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
    format!("usage: {program} [--solver auto|fallback|rust-exact|ortools]")
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
                solver = match value.as_str() {
                    "auto" => ExternalFacilityLocationReferenceSolver::Auto,
                    "fallback" => ExternalFacilityLocationReferenceSolver::Fallback,
                    "rust-exact" | "rust_exact" => {
                        ExternalFacilityLocationReferenceSolver::RustExact
                    }
                    "ortools" => ExternalFacilityLocationReferenceSolver::OrTools,
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
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["facility_location_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("facilities"));
    }
}
