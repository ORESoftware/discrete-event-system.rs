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
    format!(
        "usage: {program} [--solver auto|fallback|rust-dsatur|rust-dsatur-graph-coloring|ortools|ortools-cp-sat-graph-coloring]"
    )
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
                let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
                solver = match normalized.as_str() {
                    "auto" => ExternalGraphColoringReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalGraphColoringReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "dsatur"
                    | "rust-dsatur"
                    | "rust:dsatur"
                    | "rust-dsatur-graph-coloring"
                    | "rust:dsatur-graph-coloring"
                    | "dsatur-graph-coloring"
                    | "graph-coloring-dsatur"
                    | "graph-coloring-exact" => ExternalGraphColoringReferenceSolver::RustDsatur,
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-graph-coloring"
                    | "ortools:graph-coloring"
                    | "ortools-cp-sat"
                    | "ortools:cp-sat"
                    | "cp-sat-graph-coloring"
                    | "ortools-cp-sat-graph-coloring"
                    | "ortools:cp-sat-graph-coloring" => {
                        ExternalGraphColoringReferenceSolver::OrTools
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
    use std::sync::Mutex;

    static GRAPH_COLORING_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn graph_coloring_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "GRAPH_COLORING_REFERENCE_FORCE_PYTHON",
            "GRAPH_COLORING_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

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
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = GRAPH_COLORING_CLI_ENV_LOCK
            .lock()
            .expect("lock graph-coloring CLI env guard");
        let _force_python_guards = graph_coloring_force_python_off_guards();
        let _python_bin_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-graph-coloring-cli",
        );
        let _python_guard =
            EnvVarGuard::set("PYTHON", "/definitely/not-python-for-graph-coloring-cli");

        let output = run(
            vec![
                "graph_coloring_reference".to_string(),
                "--solver=ortools:cp-sat-graph-coloring".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:registered-graph-coloring-fallback-for-ortools"
        );
        assert_eq!(output["usedColorCount"], 3);
        assert_eq!(output["objective"], 3.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_graph_coloring_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "native",
            "rust_exact",
            "dsatur",
            "rust:dsatur",
            "rust-dsatur-graph-coloring",
            "rust:dsatur-graph-coloring",
            "graph-coloring-dsatur",
            "graph-coloring-exact",
        ] {
            assert_eq!(
                parse_solver(
                    "graph_coloring_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalGraphColoringReferenceSolver::RustDsatur
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools:graph-coloring",
            "ortools:cp-sat",
            "cp-sat-graph-coloring",
            "ortools-cp-sat-graph-coloring",
            "ortools:cp-sat-graph-coloring",
        ] {
            assert_eq!(
                parse_solver(
                    "graph_coloring_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalGraphColoringReferenceSolver::OrTools
            );
        }

        assert_eq!(
            parse_solver(
                "graph_coloring_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalGraphColoringReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["graph_coloring_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("vertices"));
    }
}
