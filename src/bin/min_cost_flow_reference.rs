use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_min_cost_flow_reference::{
    solve_min_cost_flow_with_external_reference, ExternalMinCostFlowReferenceOptions,
    ExternalMinCostFlowReferenceSolution, ExternalMinCostFlowReferenceSolver,
};
use des_engine::des::general::min_cost_flow::{
    MinCostFlowArc, MinCostFlowArcResult, MinCostFlowProblem,
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
        "usage: {program} [--solver auto|fallback|rust-ssp|rust-ssp-min-cost-flow|ortools|ortools-simple-min-cost-flow]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalMinCostFlowReferenceSolver, CliError> {
    let mut solver = ExternalMinCostFlowReferenceSolver::Auto;
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
                    "auto" => ExternalMinCostFlowReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalMinCostFlowReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "ssp"
                    | "rust-ssp"
                    | "rust:ssp"
                    | "rust-ssp-min-cost-flow"
                    | "rust:ssp-min-cost-flow"
                    | "ssp-min-cost-flow"
                    | "successive-shortest-path"
                    | "successive-shortest-augmenting-path"
                    | "min-cost-flow-ssp"
                    | "min-cost-flow-exact" => {
                        ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-min-cost-flow"
                    | "ortools:min-cost-flow"
                    | "simple-min-cost-flow"
                    | "ortools-simple-min-cost-flow"
                    | "ortools:simple-min-cost-flow" => ExternalMinCostFlowReferenceSolver::OrTools,
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

fn parse_usize(value: &Value, message: impl Into<String>) -> Result<usize, String> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| value.as_i64().and_then(|value| usize::try_from(value).ok()))
        .or_else(|| value.as_str().and_then(|text| text.parse::<usize>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_number_array(value: &Value, message: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| message.to_string())?
        .iter()
        .enumerate()
        .map(|(index, value)| parse_number(value, format!("{message}[{index}] must be numeric")))
        .collect()
}

fn parse_arc(raw: &Value, index: usize) -> Result<MinCostFlowArc, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| format!("arc {index} must be an object"))?;
    let name = object.get("name").and_then(|value| {
        if value.is_null() {
            None
        } else if let Some(text) = value.as_str() {
            Some(text.to_string())
        } else {
            Some(value.to_string())
        }
    });
    Ok(MinCostFlowArc {
        from: parse_usize(
            object
                .get("from")
                .ok_or_else(|| format!("arc {index}.from is required"))?,
            format!("arc {index}.from must be a non-negative integer"),
        )?,
        to: parse_usize(
            object
                .get("to")
                .ok_or_else(|| format!("arc {index}.to is required"))?,
            format!("arc {index}.to must be a non-negative integer"),
        )?,
        lower_bound: object
            .get("lower_bound")
            .or_else(|| object.get("lowerBound"))
            .map(|value| parse_number(value, format!("arc {index}.lowerBound must be numeric")))
            .transpose()?
            .unwrap_or(0.0),
        capacity: parse_number(
            object
                .get("capacity")
                .ok_or_else(|| format!("arc {index}.capacity is required"))?,
            format!("arc {index}.capacity must be numeric"),
        )?,
        cost: parse_number(
            object
                .get("cost")
                .ok_or_else(|| format!("arc {index}.cost is required"))?,
            format!("arc {index}.cost must be numeric"),
        )?,
        name,
    })
}

fn parse_min_cost_flow_problem(raw: &Value) -> Result<MinCostFlowProblem, String> {
    let num_nodes = raw
        .get("num_nodes")
        .or_else(|| raw.get("numNodes"))
        .ok_or_else(|| "num_nodes must be positive".to_string())
        .and_then(|value| parse_usize(value, "num_nodes must be a non-negative integer"))?;
    let supplies = raw
        .get("supplies")
        .ok_or_else(|| "supplies must be an array".to_string())
        .and_then(|value| parse_number_array(value, "supplies"))?;
    let raw_arcs = raw
        .get("arcs")
        .and_then(Value::as_array)
        .ok_or_else(|| "arcs must be non-empty".to_string())?;
    let arcs = raw_arcs
        .iter()
        .enumerate()
        .map(|(index, arc)| parse_arc(arc, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MinCostFlowProblem {
        num_nodes,
        supplies,
        arcs,
    })
}

fn arc_result_json(arc: &MinCostFlowArcResult) -> Value {
    json!({
        "from": arc.from,
        "to": arc.to,
        "lowerBound": arc.lower_bound,
        "capacity": arc.capacity,
        "cost": arc.cost,
        "flow": arc.flow,
        "name": arc.name,
    })
}

fn solution_json(solution: &ExternalMinCostFlowReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "objective": solution.objective,
        "flows": solution.flows.iter().map(arc_result_json).collect::<Vec<_>>(),
        "nodeBalance": solution.node_balance,
        "iterations": solution.iterations,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || solution.ortools_objective.is_some()
        || !solution.ortools_flows.is_empty()
        || !solution.ortools_node_balance.is_empty()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsFlows"] = json!(solution
            .ortools_flows
            .iter()
            .map(arc_result_json)
            .collect::<Vec<_>>());
        output["ortoolsNodeBalance"] = json!(solution.ortools_node_balance);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:min-cost-flow-reference",
        "objective": null,
        "flows": [],
        "nodeBalance": [],
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "min_cost_flow_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_min_cost_flow_problem(&payload).map_err(CliError)?;
    let solution = solve_min_cost_flow_with_external_reference(
        &problem,
        &ExternalMinCostFlowReferenceOptions { solver },
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
                    .unwrap_or("min_cost_flow_reference")
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
                serde_json::to_string(&output).expect("serialize min-cost-flow output")
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
        "numNodes": 4,
        "supplies": [5.0, 7.0, -6.0, -6.0],
        "arcs": [
            {"from": 0, "to": 2, "lowerBound": 0.0, "capacity": 5.0, "cost": 2.0, "name": "s0_d0"},
            {"from": 0, "to": 3, "lowerBound": 0.0, "capacity": 5.0, "cost": 4.0, "name": "s0_d1"},
            {"from": 1, "to": 2, "lowerBound": 0.0, "capacity": 6.0, "cost": 5.0, "name": "s1_d0"},
            {"from": 1, "to": 3, "lowerBound": 0.0, "capacity": 8.0, "cost": 1.0, "name": "s1_d1"}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_successive_shortest_path() {
        let output = run(
            vec![
                "min_cost_flow_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:ssp-min-cost-flow");
        assert_eq!(output["objective"], 21.0);
        assert_eq!(output["flows"].as_array().expect("flows").len(), 4);
        assert_eq!(output["nodeBalance"], json!([5.0, 7.0, -6.0, -6.0]));
    }

    #[test]
    fn accepts_rust_exact_alias_and_snake_case_num_nodes() {
        let output = run(
            vec![
                "min_cost_flow_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["objective"], 21.0);
    }

    #[test]
    fn parses_min_cost_flow_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "native",
            "rust_exact",
            "ssp",
            "rust:ssp",
            "rust-ssp-min-cost-flow",
            "rust:ssp-min-cost-flow",
            "successive-shortest-path",
            "successive-shortest-augmenting-path",
            "min-cost-flow-ssp",
            "min-cost-flow-exact",
        ] {
            assert_eq!(
                parse_solver(
                    "min_cost_flow_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalMinCostFlowReferenceSolver::RustSuccessiveShortestPath
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools:min-cost-flow",
            "simple-min-cost-flow",
            "ortools-simple-min-cost-flow",
            "ortools:simple-min-cost-flow",
        ] {
            assert_eq!(
                parse_solver(
                    "min_cost_flow_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalMinCostFlowReferenceSolver::OrTools
            );
        }

        assert_eq!(
            parse_solver(
                "min_cost_flow_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalMinCostFlowReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["min_cost_flow_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("num_nodes"));
    }
}
