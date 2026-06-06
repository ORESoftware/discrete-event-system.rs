use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_set_cover_reference::{
    solve_set_cover_with_external_reference, ExternalSetCoverReferenceOptions,
    ExternalSetCoverReferenceSolution, ExternalSetCoverReferenceSolver,
};
use des_engine::des::general::set_cover::{SetCoverProblem, SetCoverSet};
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
        "usage: {program} [--solver auto|fallback|rust-exact|rust-exact-set-cover|ortools|ortools-cp-sat-set-cover]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalSetCoverReferenceSolver, CliError> {
    let mut solver = ExternalSetCoverReferenceSolver::Auto;
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
                    "auto" => ExternalSetCoverReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalSetCoverReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "rust-exact-set-cover"
                    | "rust:exact-set-cover"
                    | "exact-set-cover"
                    | "set-cover-exact" => ExternalSetCoverReferenceSolver::RustExact,
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-set-cover"
                    | "ortools:set-cover"
                    | "ortools-cp-sat"
                    | "ortools:cp-sat"
                    | "cp-sat-set-cover"
                    | "ortools-cp-sat-set-cover"
                    | "ortools:cp-sat-set-cover" => ExternalSetCoverReferenceSolver::OrTools,
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

fn value_array<'a>(value: &'a Value, field: &str, message: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| message.to_string())
}

fn parse_string_array(value: &Value, field: &str, message: &str) -> Result<Vec<String>, String> {
    value_array(value, field, message).map(|items| {
        items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
    })
}

fn parse_set_cover_problem(raw: &Value) -> Result<SetCoverProblem, String> {
    let universe = parse_string_array(raw, "universe", "universe must be non-empty")?;
    let raw_sets = value_array(raw, "sets", "sets must be non-empty")?;
    let mut sets = Vec::with_capacity(raw_sets.len());
    for (index, raw_set) in raw_sets.iter().enumerate() {
        let Some(object) = raw_set.as_object() else {
            return Err(format!("sets[{index}] must be an object"));
        };
        let id = object
            .get("id")
            .map(|value| match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| format!("S{}", index + 1));
        let cost = object.get("cost").and_then(Value::as_f64).unwrap_or(0.0);
        let elements = parse_string_array(raw_set, "elements", "set elements must be non-empty")?;
        sets.push(SetCoverSet { id, cost, elements });
    }
    Ok(SetCoverProblem { universe, sets })
}

fn solution_json(solution: &ExternalSetCoverReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "selectedSetIndices": solution.selected_set_indices,
        "selectedSets": solution.selected_set_ids,
        "objective": solution.objective,
        "coveredElements": solution.covered_elements,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_selected_set_indices.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsSelectedSetIndices"] = json!(solution.ortools_selected_set_indices);
        output["ortoolsSelectedSets"] = json!(solution.ortools_selected_set_ids);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsCoveredElements"] = json!(solution.ortools_covered_elements);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:set-cover-reference",
        "selectedSetIndices": [],
        "selectedSets": [],
        "objective": null,
        "coveredElements": [],
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "set_cover_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_set_cover_problem(&payload).map_err(CliError)?;
    let solution = solve_set_cover_with_external_reference(
        &problem,
        &ExternalSetCoverReferenceOptions { solver },
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
                    .unwrap_or("set_cover_reference")
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
                serde_json::to_string(&output).expect("serialize set-cover output")
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
        "universe": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "sets": [
            {"id": "A", "cost": 3.0, "elements": ["E1", "E2", "E3"]},
            {"id": "B", "cost": 2.0, "elements": ["E2", "E4"]},
            {"id": "C", "cost": 4.0, "elements": ["E3", "E4", "E5"]},
            {"id": "D", "cost": 2.0, "elements": ["E5", "E6"]},
            {"id": "E", "cost": 5.0, "elements": ["E1", "E4", "E6"]},
            {"id": "F", "cost": 1.0, "elements": ["E6"]}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_exact_reference() {
        let output = run(
            vec![
                "set_cover_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-set-cover");
        assert_eq!(output["objective"], 7.0);
        assert_eq!(output["selectedSets"], json!(["A", "B", "D"]));
    }

    #[test]
    fn parses_set_cover_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "native",
            "rust_exact",
            "rust:exact",
            "rust-exact-set-cover",
            "rust:exact-set-cover",
            "exact-set-cover",
            "set-cover-exact",
        ] {
            assert_eq!(
                parse_solver(
                    "set_cover_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalSetCoverReferenceSolver::RustExact
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools:set-cover",
            "ortools:cp-sat",
            "cp-sat-set-cover",
            "ortools-cp-sat-set-cover",
            "ortools:cp-sat-set-cover",
        ] {
            assert_eq!(
                parse_solver(
                    "set_cover_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalSetCoverReferenceSolver::OrTools
            );
        }

        assert_eq!(
            parse_solver(
                "set_cover_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalSetCoverReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["set_cover_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("universe must be non-empty"));
    }
}
