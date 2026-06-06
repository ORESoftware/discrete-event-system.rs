use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_weighted_max_sat_reference::{
    solve_weighted_max_sat_with_external_reference, ExternalWeightedMaxSatReferenceOptions,
    ExternalWeightedMaxSatReferenceSolution, ExternalWeightedMaxSatReferenceSolver,
};
use des_engine::des::general::weighted_max_sat::{WeightedMaxSatClause, WeightedMaxSatProblem};
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
        "usage: {program} [--solver auto|fallback|rust-enumeration|rust:exact-weighted-max-sat|ortools|ortools:cp-sat-weighted-max-sat]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalWeightedMaxSatReferenceSolver, CliError> {
    let mut solver = ExternalWeightedMaxSatReferenceSolver::Auto;
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
                    "auto" | "default" => ExternalWeightedMaxSatReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalWeightedMaxSatReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-enumeration"
                    | "rust:enumeration"
                    | "rust-exact"
                    | "rust:exact"
                    | "weighted-max-sat"
                    | "max-sat"
                    | "weighted-maxsat"
                    | "maxsat"
                    | "exact-weighted-max-sat"
                    | "rust-weighted-max-sat"
                    | "rust:weighted-max-sat"
                    | "rust-exact-weighted-max-sat"
                    | "rust:exact-weighted-max-sat" => {
                        ExternalWeightedMaxSatReferenceSolver::RustEnumeration
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-cp-sat"
                    | "ortools:cp-sat"
                    | "or-tools-cp-sat"
                    | "cp-sat-weighted-max-sat"
                    | "ortools-weighted-max-sat"
                    | "ortools:weighted-max-sat"
                    | "ortools-cp-sat-weighted-max-sat"
                    | "ortools:cp-sat-weighted-max-sat"
                    | "or-tools-cp-sat-weighted-max-sat" => {
                        ExternalWeightedMaxSatReferenceSolver::OrTools
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

fn parse_num_vars(raw: &Value) -> Result<usize, String> {
    let value = raw
        .get("numVars")
        .or_else(|| raw.get("num_vars"))
        .ok_or_else(|| "numVars must be positive".to_string())?;
    if let Some(number) = value.as_u64() {
        return usize::try_from(number).map_err(|_| "numVars is too large".to_string());
    }
    if let Some(number) = value.as_i64() {
        if number > 0 {
            return usize::try_from(number).map_err(|_| "numVars is too large".to_string());
        }
    }
    if let Some(text) = value.as_str() {
        return text
            .parse::<usize>()
            .map_err(|_| "numVars must be positive".to_string());
    }
    Err("numVars must be positive".to_string())
}

fn value_array<'a>(value: &'a Value, field: &str, message: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| message.to_string())
}

fn value_as_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn parse_literals(raw_clause: &Value, clause_index: usize) -> Result<Vec<i64>, String> {
    value_array(
        raw_clause,
        "literals",
        &format!("clauses[{clause_index}].literals must be non-empty"),
    )?
    .iter()
    .map(|literal| {
        literal
            .as_i64()
            .or_else(|| literal.as_str().and_then(|text| text.parse::<i64>().ok()))
            .ok_or_else(|| format!("clauses[{clause_index}].literals must be integers"))
    })
    .collect()
}

fn parse_problem(raw: &Value) -> Result<WeightedMaxSatProblem, String> {
    let num_vars = parse_num_vars(raw)?;
    let raw_clauses = value_array(raw, "clauses", "clauses must be non-empty")?;
    let clauses = raw_clauses
        .iter()
        .enumerate()
        .map(|(index, raw_clause)| {
            let Some(object) = raw_clause.as_object() else {
                return Err(format!("clauses[{index}] must be an object"));
            };
            let id = object
                .get("id")
                .map(value_as_string)
                .unwrap_or_else(|| format!("C{}", index + 1));
            let weight = object.get("weight").and_then(Value::as_f64).unwrap_or(0.0);
            let hard = object.get("hard").and_then(Value::as_bool).unwrap_or(false);
            let literals = parse_literals(raw_clause, index)?;
            Ok(WeightedMaxSatClause {
                id,
                literals,
                weight,
                hard,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(WeightedMaxSatProblem { num_vars, clauses })
}

fn solution_json(solution: &ExternalWeightedMaxSatReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "assignment": solution.assignment,
        "objective": solution.objective,
        "satisfiedSoftWeight": solution.satisfied_soft_weight,
        "unsatisfiedSoftWeight": solution.unsatisfied_soft_weight,
        "satisfiedClauseIds": solution.satisfied_clause_ids,
        "violatedHardClauseIds": solution.violated_hard_clause_ids,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_assignment.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsAssignment"] = json!(solution.ortools_assignment);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsSatisfiedSoftWeight"] = json!(solution.ortools_satisfied_soft_weight);
        output["ortoolsUnsatisfiedSoftWeight"] = json!(solution.ortools_unsatisfied_soft_weight);
        output["ortoolsSatisfiedClauseIds"] = json!(solution.ortools_satisfied_clause_ids);
        output["ortoolsViolatedHardClauseIds"] = json!(solution.ortools_violated_hard_clause_ids);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:weighted-max-sat-reference",
        "assignment": [],
        "objective": null,
        "satisfiedSoftWeight": null,
        "unsatisfiedSoftWeight": null,
        "satisfiedClauseIds": [],
        "violatedHardClauseIds": [],
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "weighted_max_sat_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_problem(&payload).map_err(CliError)?;
    let solution = solve_weighted_max_sat_with_external_reference(
        &problem,
        &ExternalWeightedMaxSatReferenceOptions { solver },
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
                    .unwrap_or("weighted_max_sat_reference")
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
                serde_json::to_string(&output).expect("serialize weighted-max-sat output")
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
        "numVars": 3,
        "clauses": [
            {"id": "H_cover", "literals": [1, 2], "weight": 0.0, "hard": true},
            {"id": "H_implication", "literals": [-2, 3], "weight": 0.0, "hard": true},
            {"id": "S_pick_x1", "literals": [1], "weight": 6.0, "hard": false},
            {"id": "S_pick_x2", "literals": [2], "weight": 6.0, "hard": false},
            {"id": "S_not_both_x1_x2", "literals": [-1, -2], "weight": 5.0, "hard": false},
            {"id": "S_pick_x3", "literals": [3], "weight": 4.0, "hard": false},
            {"id": "S_skip_x3", "literals": [-3], "weight": 3.0, "hard": false}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_enumeration_reference() {
        let output = run(
            vec![
                "weighted_max_sat_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-weighted-max-sat");
        assert_eq!(output["assignment"], json!([true, true, true]));
        assert_eq!(output["objective"], 16.0);
        assert_eq!(output["violatedHardClauseIds"], json!([]));
    }

    #[test]
    fn accepts_num_vars_alias_and_string_literals() {
        let output = run(
            vec!["weighted_max_sat_reference".to_string()],
            r#"{"num_vars":"1","clauses":[{"id":"C1","literals":["1"],"weight":2.0}]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["assignment"], json!([true]));
        assert_eq!(output["objective"], 2.0);
    }

    #[test]
    fn parses_weighted_max_sat_solver_aliases_used_by_validation_tools() {
        let rust_aliases = [
            "rust",
            "native",
            "exact",
            "rust:exact",
            "rust_enumeration",
            "rust:enumeration",
            "rust:exact-weighted-max-sat",
        ];
        for alias in rust_aliases {
            let solver = parse_solver(
                "weighted_max_sat_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(
                solver,
                ExternalWeightedMaxSatReferenceSolver::RustEnumeration
            );
        }

        let ortools_aliases = [
            "or-tools",
            "google-ortools",
            "ortools:cp-sat",
            "ortools_cp_sat_weighted_max_sat",
            "ortools:cp-sat-weighted-max-sat",
        ];
        for alias in ortools_aliases {
            let solver = parse_solver(
                "weighted_max_sat_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(solver, ExternalWeightedMaxSatReferenceSolver::OrTools);
        }

        let fallback = parse_solver(
            "weighted_max_sat_reference",
            ["--solver=rust:fallback".to_string()],
        )
        .expect("fallback alias");
        assert_eq!(fallback, ExternalWeightedMaxSatReferenceSolver::Fallback);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["weighted_max_sat_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("numVars must be positive"));
    }
}
