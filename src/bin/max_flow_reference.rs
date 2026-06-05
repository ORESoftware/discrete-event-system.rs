use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_max_flow_reference::{
    solve_max_flow_with_external_reference, ExternalMaxFlowReferenceCut,
    ExternalMaxFlowReferenceOptions, ExternalMaxFlowReferenceSolution,
    ExternalMaxFlowReferenceSolver,
};
use des_engine::des::general::max_flow::{MaxFlowEdge, MaxFlowEdgeFlow, MaxFlowProblem};
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
    format!("usage: {program} [--solver auto|fallback|rust-edmonds-karp|rust-exact|ortools]")
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalMaxFlowReferenceSolver, CliError> {
    let mut solver = ExternalMaxFlowReferenceSolver::Auto;
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
                    "auto" => ExternalMaxFlowReferenceSolver::Auto,
                    "fallback" => ExternalMaxFlowReferenceSolver::Fallback,
                    "rust-edmonds-karp" | "rust_edmonds_karp" | "rust-exact" | "rust_exact" => {
                        ExternalMaxFlowReferenceSolver::RustEdmondsKarp
                    }
                    "ortools" => ExternalMaxFlowReferenceSolver::OrTools,
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

fn parse_edge(raw: &Value, index: usize) -> Result<MaxFlowEdge, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| format!("edge {index} must be an object"))?;
    let name = object.get("name").and_then(|value| {
        if value.is_null() {
            None
        } else if let Some(text) = value.as_str() {
            Some(text.to_string())
        } else {
            Some(value.to_string())
        }
    });
    Ok(MaxFlowEdge {
        from: parse_usize(
            object
                .get("from")
                .ok_or_else(|| format!("edge {index}.from is required"))?,
            format!("edge {index}.from must be a non-negative integer"),
        )?,
        to: parse_usize(
            object
                .get("to")
                .ok_or_else(|| format!("edge {index}.to is required"))?,
            format!("edge {index}.to must be a non-negative integer"),
        )?,
        capacity: parse_number(
            object
                .get("capacity")
                .ok_or_else(|| format!("edge {index}.capacity is required"))?,
            format!("edge {index}.capacity must be numeric"),
        )?,
        name,
    })
}

fn parse_max_flow_problem(raw: &Value) -> Result<MaxFlowProblem, String> {
    let num_nodes = raw
        .get("numNodes")
        .or_else(|| raw.get("num_nodes"))
        .ok_or_else(|| "numNodes must be at least 2".to_string())
        .and_then(|value| parse_usize(value, "numNodes must be a non-negative integer"))?;
    let source = raw
        .get("source")
        .ok_or_else(|| "source is required".to_string())
        .and_then(|value| parse_usize(value, "source must be a non-negative integer"))?;
    let sink = raw
        .get("sink")
        .ok_or_else(|| "sink is required".to_string())
        .and_then(|value| parse_usize(value, "sink must be a non-negative integer"))?;
    let raw_edges = raw
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "edges must be non-empty".to_string())?;
    let edges = raw_edges
        .iter()
        .enumerate()
        .map(|(index, edge)| parse_edge(edge, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MaxFlowProblem {
        num_nodes,
        source,
        sink,
        edges,
    })
}

fn edge_flow_json(edge: &MaxFlowEdgeFlow) -> Value {
    json!({
        "from": edge.from,
        "to": edge.to,
        "capacity": edge.capacity,
        "name": edge.name,
        "flow": edge.flow,
    })
}

fn cut_json(cut: &ExternalMaxFlowReferenceCut) -> Value {
    json!({
        "sourceSide": cut.source_side,
        "sinkSide": cut.sink_side,
        "cutEdges": cut.cut_edges.iter().map(edge_flow_json).collect::<Vec<_>>(),
        "capacity": cut.capacity,
    })
}

fn solution_json(solution: &ExternalMaxFlowReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "maxFlow": solution.max_flow,
        "edgeFlows": solution.edge_flows.iter().map(edge_flow_json).collect::<Vec<_>>(),
        "minCut": cut_json(&solution.min_cut),
        "nodeBalance": solution.node_balance,
        "iterations": solution.iterations,
        "trace": [],
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || solution.ortools_max_flow.is_some()
        || !solution.ortools_edge_flows.is_empty()
        || !solution.ortools_node_balance.is_empty()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsMaxFlow"] = json!(solution.ortools_max_flow);
        output["ortoolsEdgeFlows"] = json!(solution
            .ortools_edge_flows
            .iter()
            .map(edge_flow_json)
            .collect::<Vec<_>>());
        output["ortoolsMinCut"] = cut_json(&solution.ortools_min_cut);
        output["ortoolsNodeBalance"] = json!(solution.ortools_node_balance);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:max-flow-reference",
        "maxFlow": null,
        "edgeFlows": [],
        "minCut": {},
        "nodeBalance": [],
        "trace": [],
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "max_flow_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_max_flow_problem(&payload).map_err(CliError)?;
    let solution = solve_max_flow_with_external_reference(
        &problem,
        &ExternalMaxFlowReferenceOptions { solver },
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
                    .unwrap_or("max_flow_reference")
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
                serde_json::to_string(&output).expect("serialize max-flow output")
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
        "source": 0,
        "sink": 3,
        "edges": [
            {"from": 0, "to": 1, "capacity": 3.0, "name": "s-a"},
            {"from": 0, "to": 2, "capacity": 2.0, "name": "s-b"},
            {"from": 1, "to": 3, "capacity": 2.0, "name": "a-t"},
            {"from": 2, "to": 3, "capacity": 3.0, "name": "b-t"},
            {"from": 1, "to": 2, "capacity": 1.0, "name": "a-b"}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_edmonds_karp() {
        let output = run(
            vec![
                "max_flow_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:edmonds-karp-max-flow");
        assert_eq!(output["maxFlow"], 5.0);
        assert_eq!(output["minCut"]["capacity"], 5.0);
        assert_eq!(output["edgeFlows"].as_array().expect("edge flows").len(), 5);
    }

    #[test]
    fn accepts_rust_exact_alias() {
        let output = run(
            vec![
                "max_flow_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["maxFlow"], 5.0);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["max_flow_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("numNodes"));
    }
}
