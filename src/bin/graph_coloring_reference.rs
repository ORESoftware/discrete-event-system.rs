use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_graph_coloring_reference::{
    solve_graph_coloring_with_external_reference, ExternalGraphColoringReferenceOptions,
    ExternalGraphColoringReferenceSolution, ExternalGraphColoringReferenceSolver,
};
use des_engine::des::general::graph_coloring::GraphColoringProblem;
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
    format!("usage: {program} [--solver auto|fallback|rust-dsatur|rust-exact|ortools]")
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalGraphColoringReferenceSolver, CliError> {
    let mut solver = ExternalGraphColoringReferenceSolver::Auto;
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
                    "auto" => ExternalGraphColoringReferenceSolver::Auto,
                    "fallback" => ExternalGraphColoringReferenceSolver::Fallback,
                    "rust-dsatur" | "rust_dsatur" | "rust-exact" | "rust_exact" => {
                        ExternalGraphColoringReferenceSolver::RustDsatur
                    }
                    "ortools" => ExternalGraphColoringReferenceSolver::OrTools,
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

fn parse_vertices(raw: &Value) -> Result<Vec<String>, String> {
    raw.get("vertices")
        .or_else(|| raw.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| "vertices must be non-empty".to_string())
        .map(|vertices| vertices.iter().map(parse_string).collect())
}

fn parse_edge(raw: &Value, index: usize) -> Result<(String, String), String> {
    if let Some(items) = raw.as_array() {
        if items.len() == 2 {
            return Ok((parse_string(&items[0]), parse_string(&items[1])));
        }
        return Err(format!("edges[{index}] must be a two-item list"));
    }
    let object = raw
        .as_object()
        .ok_or_else(|| format!("edges[{index}] must be a two-item list"))?;
    Ok((
        object
            .get("from")
            .ok_or_else(|| format!("edges[{index}].from is required"))
            .map(parse_string)?,
        object
            .get("to")
            .ok_or_else(|| format!("edges[{index}].to is required"))
            .map(parse_string)?,
    ))
}

fn parse_graph_coloring_problem(raw: &Value) -> Result<GraphColoringProblem, String> {
    let vertices = parse_vertices(raw)?;
    let raw_edges = raw
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "edges must be an array".to_string())?;
    let edges = raw_edges
        .iter()
        .enumerate()
        .map(|(index, edge)| parse_edge(edge, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GraphColoringProblem { vertices, edges })
}

fn solution_json(solution: &ExternalGraphColoringReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "colorIndices": solution.color_indices,
        "colorNames": solution.color_names,
        "usedColorCount": solution.used_color_count,
        "objective": solution.objective,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_color_indices.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsColorIndices"] = json!(solution.ortools_color_indices);
        output["ortoolsColorNames"] = json!(solution.ortools_color_names);
        output["ortoolsUsedColorCount"] = json!(solution.ortools_used_color_count);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:graph-coloring-reference",
        "colorIndices": [],
        "colorNames": [],
        "usedColorCount": null,
        "objective": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "graph_coloring_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_graph_coloring_problem(&payload).map_err(CliError)?;
    let solution = solve_graph_coloring_with_external_reference(
        &problem,
        &ExternalGraphColoringReferenceOptions { solver },
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
                    .unwrap_or("graph_coloring_reference")
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
                serde_json::to_string(&output).expect("serialize graph-coloring output")
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
        "vertices": ["A", "B", "C", "D", "E", "F"],
        "edges": [
            ["A", "B"],
            ["B", "C"],
            ["C", "D"],
            ["D", "E"],
            ["E", "A"],
            ["A", "F"],
            ["C", "F"]
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_dsatur() {
        let output = run(
            vec![
                "graph_coloring_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:dsatur-graph-coloring");
        assert_eq!(output["usedColorCount"], 3);
        assert_eq!(output["objective"], 3.0);
    }

    #[test]
    fn accepts_object_edges_and_rust_exact_alias() {
        let output = run(
            vec![
                "graph_coloring_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            r#"{"nodes": ["A", "B"], "edges": [{"from": "A", "to": "B"}]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["usedColorCount"], 2);
        assert_eq!(output["objective"], 2.0);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["graph_coloring_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("vertices"));
    }
}
