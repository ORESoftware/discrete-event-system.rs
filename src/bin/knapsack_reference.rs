use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_knapsack_reference::{
    solve_knapsack_with_external_reference, ExternalKnapsackReferenceOptions,
    ExternalKnapsackReferenceSolution, ExternalKnapsackReferenceSolver,
};
use des_engine::des::general::knapsack::{KnapsackItem, KnapsackProblem};
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
    format!("usage: {program} [--solver auto|fallback|rust-branch-and-bound|rust-exact|ortools]")
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalKnapsackReferenceSolver, CliError> {
    let mut solver = ExternalKnapsackReferenceSolver::Auto;
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
                    "auto" => ExternalKnapsackReferenceSolver::Auto,
                    "fallback" => ExternalKnapsackReferenceSolver::Fallback,
                    "rust-branch-and-bound"
                    | "rust_branch_and_bound"
                    | "rust-exact"
                    | "rust_exact" => ExternalKnapsackReferenceSolver::RustBranchAndBound,
                    "ortools" => ExternalKnapsackReferenceSolver::OrTools,
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

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_item(raw: &Value, index: usize) -> Result<KnapsackItem, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| format!("items[{index}] must be an object"))?;
    let id = object
        .get("id")
        .map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| format!("I{}", index + 1));
    Ok(KnapsackItem {
        id,
        weight: parse_number(
            object
                .get("weight")
                .ok_or_else(|| format!("items[{index}].weight is required"))?,
            format!("items[{index}].weight must be numeric"),
        )?,
        value: parse_number(
            object
                .get("value")
                .ok_or_else(|| format!("items[{index}].value is required"))?,
            format!("items[{index}].value must be numeric"),
        )?,
    })
}

fn parse_number_array(value: &Value, field: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| parse_number(value, format!("{field}[{index}] must be numeric")))
        .collect()
}

fn parse_items(raw: &Value) -> Result<Vec<KnapsackItem>, String> {
    if let Some(items) = raw.get("items") {
        return items
            .as_array()
            .ok_or_else(|| "items must be an array".to_string())?
            .iter()
            .enumerate()
            .map(|(index, item)| parse_item(item, index))
            .collect();
    }
    let weights = raw
        .get("weights")
        .ok_or_else(|| "items must be non-empty".to_string())
        .and_then(|value| parse_number_array(value, "weights"))?;
    let values = raw
        .get("values")
        .ok_or_else(|| "values must be an array".to_string())
        .and_then(|value| parse_number_array(value, "values"))?;
    if weights.len() != values.len() {
        return Err("weights and values must have the same length".to_string());
    }
    Ok(weights
        .into_iter()
        .zip(values)
        .enumerate()
        .map(|(index, (weight, value))| KnapsackItem {
            id: format!("I{}", index + 1),
            weight,
            value,
        })
        .collect())
}

fn parse_knapsack_problem(raw: &Value) -> Result<KnapsackProblem, String> {
    let capacity = raw
        .get("capacity")
        .ok_or_else(|| "capacity must be finite and > 0".to_string())
        .and_then(|value| parse_number(value, "capacity must be numeric"))?;
    Ok(KnapsackProblem {
        capacity,
        items: parse_items(raw)?,
    })
}

fn solution_json(solution: &ExternalKnapsackReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "selectedItemIndices": solution.selected_item_indices,
        "selectedItemIds": solution.selected_item_ids,
        "totalWeight": solution.total_weight,
        "totalValue": solution.total_value,
        "objective": solution.objective,
        "upperBound": solution.upper_bound,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_selected_item_indices.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsSelectedItemIndices"] = json!(solution.ortools_selected_item_indices);
        output["ortoolsSelectedItemIds"] = json!(solution.ortools_selected_item_ids);
        output["ortoolsTotalWeight"] = json!(solution.ortools_total_weight);
        output["ortoolsTotalValue"] = json!(solution.ortools_total_value);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:knapsack-reference",
        "selectedItemIndices": [],
        "selectedItemIds": [],
        "totalWeight": 0.0,
        "totalValue": 0.0,
        "objective": null,
        "upperBound": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "knapsack_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_knapsack_problem(&payload).map_err(CliError)?;
    let solution = solve_knapsack_with_external_reference(
        &problem,
        &ExternalKnapsackReferenceOptions { solver },
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
                    .unwrap_or("knapsack_reference")
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
                serde_json::to_string(&output).expect("serialize knapsack output")
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
        "capacity": 26.0,
        "items": [
            {"id": "A", "weight": 12.0, "value": 24.0},
            {"id": "B", "weight": 7.0, "value": 13.0},
            {"id": "C", "weight": 11.0, "value": 23.0},
            {"id": "D", "weight": 8.0, "value": 15.0}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_branch_and_bound() {
        let output = run(
            vec![
                "knapsack_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:branch-and-bound-knapsack");
        assert_eq!(output["selectedItemIds"], json!(["B", "C", "D"]));
        assert_eq!(output["objective"], 51.0);
    }

    #[test]
    fn accepts_weights_values_and_rust_exact_alias() {
        let output = run(
            vec![
                "knapsack_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            r#"{"capacity": 5, "weights": [5, 4, 1], "values": [10, 10, 0]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["selectedItemIds"], json!(["I2"]));
        assert_eq!(output["objective"], 10.0);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["knapsack_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("capacity"));
    }
}
