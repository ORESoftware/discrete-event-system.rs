use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_minimum_spanning_tree_reference::{
    solve_minimum_spanning_tree_with_external_reference,
    ExternalMinimumSpanningTreeReferenceOptions, ExternalMinimumSpanningTreeReferenceSolution,
    ExternalMinimumSpanningTreeReferenceSolver,
};
use des_engine::des::general::minimum_spanning_tree::{
    MinimumSpanningTreeEdge, MinimumSpanningTreeProblem,
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
        "usage: {program} [--solver auto|fallback|rust-kruskal|rust-kruskal-mst|ortools|ortools-cp-sat-mst]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalMinimumSpanningTreeReferenceSolver, CliError> {
    let mut solver = ExternalMinimumSpanningTreeReferenceSolver::Auto;
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
                    "auto" => ExternalMinimumSpanningTreeReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalMinimumSpanningTreeReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "kruskal"
                    | "rust-kruskal"
                    | "rust:kruskal"
                    | "rust-kruskal-mst"
                    | "rust:kruskal-mst"
                    | "kruskal-mst"
                    | "mst-kruskal"
                    | "minimum-spanning-tree-kruskal"
                    | "minimum-spanning-tree-exact" => {
                        ExternalMinimumSpanningTreeReferenceSolver::RustKruskal
                    }
                    "ortools" | "or-tools" | "google-ortools" | "google-or-tools"
                    | "ortools-mst" | "ortools:mst" | "ortools-cp-sat" | "ortools:cp-sat"
                    | "cp-sat-mst" | "ortools-cp-sat-mst" | "ortools:cp-sat-mst" => {
                        ExternalMinimumSpanningTreeReferenceSolver::OrTools
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

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
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

fn parse_edge(raw: &Value, index: usize) -> Result<MinimumSpanningTreeEdge, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| format!("edges[{index}] must be an object"))?;
    Ok(MinimumSpanningTreeEdge {
        id: object
            .get("id")
            .map(parse_string)
            .unwrap_or_else(|| format!("E{}", index + 1)),
        from: object
            .get("from")
            .ok_or_else(|| format!("edges[{index}].from is required"))
            .map(parse_string)?,
        to: object
            .get("to")
            .ok_or_else(|| format!("edges[{index}].to is required"))
            .map(parse_string)?,
        weight: parse_number(
            object
                .get("weight")
                .ok_or_else(|| format!("edges[{index}].weight is required"))?,
            format!("edges[{index}].weight must be numeric"),
        )?,
    })
}

fn parse_minimum_spanning_tree_problem(raw: &Value) -> Result<MinimumSpanningTreeProblem, String> {
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
    Ok(MinimumSpanningTreeProblem { vertices, edges })
}

fn solution_json(solution: &ExternalMinimumSpanningTreeReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "selectedEdgeIndices": solution.selected_edge_indices,
        "selectedEdgeIds": solution.selected_edge_ids,
        "objective": solution.objective,
        "totalWeight": solution.total_weight,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_selected_edge_indices.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsSelectedEdgeIndices"] = json!(solution.ortools_selected_edge_indices);
        output["ortoolsSelectedEdgeIds"] = json!(solution.ortools_selected_edge_ids);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsTotalWeight"] = json!(solution.ortools_total_weight);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:minimum-spanning-tree-reference",
        "selectedEdgeIndices": [],
        "selectedEdgeIds": [],
        "objective": null,
        "totalWeight": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "minimum_spanning_tree_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_minimum_spanning_tree_problem(&payload).map_err(CliError)?;
    let solution = solve_minimum_spanning_tree_with_external_reference(
        &problem,
        &ExternalMinimumSpanningTreeReferenceOptions { solver },
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
                    .unwrap_or("minimum_spanning_tree_reference")
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
                serde_json::to_string(&output).expect("serialize MST output")
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

    static MINIMUM_SPANNING_TREE_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn minimum_spanning_tree_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "MINIMUM_SPANNING_TREE_REFERENCE_FORCE_PYTHON",
            "MINIMUM_SPANNING_TREE_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "vertices": ["A", "B", "C", "D", "E"],
        "edges": [
            {"id": "AB", "from": "A", "to": "B", "weight": 1.0},
            {"id": "AC", "from": "A", "to": "C", "weight": 4.0},
            {"id": "AE", "from": "A", "to": "E", "weight": 7.0},
            {"id": "BC", "from": "B", "to": "C", "weight": 2.0},
            {"id": "BD", "from": "B", "to": "D", "weight": 5.0},
            {"id": "CD", "from": "C", "to": "D", "weight": 1.0},
            {"id": "CE", "from": "C", "to": "E", "weight": 3.0},
            {"id": "DE", "from": "D", "to": "E", "weight": 2.0}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_kruskal() {
        let output = run(
            vec![
                "minimum_spanning_tree_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:kruskal-mst");
        assert_eq!(output["selectedEdgeIds"], json!(["AB", "BC", "CD", "DE"]));
        assert_eq!(output["objective"], 6.0);
    }

    #[test]
    fn accepts_nodes_alias_and_rust_exact_alias() {
        let output = run(
            vec![
                "minimum_spanning_tree_reference".to_string(),
                "--solver=rust-exact".to_string(),
            ],
            r#"{"nodes": ["A", "B"], "edges": [{"id": "AB", "from": "A", "to": "B", "weight": 2}]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["selectedEdgeIds"], json!(["AB"]));
        assert_eq!(output["objective"], 2.0);
    }

    #[test]
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = MINIMUM_SPANNING_TREE_CLI_ENV_LOCK
            .lock()
            .expect("lock minimum-spanning-tree CLI env guard");
        let _force_python_guards = minimum_spanning_tree_force_python_off_guards();
        let _python_bin_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-minimum-spanning-tree-cli",
        );
        let _python_guard = EnvVarGuard::set(
            "PYTHON",
            "/definitely/not-python-for-minimum-spanning-tree-cli",
        );

        let output = run(
            vec![
                "minimum_spanning_tree_reference".to_string(),
                "--solver=ortools:cp-sat-mst".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:registered-minimum-spanning-tree-fallback-for-ortools"
        );
        assert_eq!(output["selectedEdgeIds"], json!(["AB", "BC", "CD", "DE"]));
        assert_eq!(output["objective"], 6.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_minimum_spanning_tree_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "native",
            "rust_exact",
            "kruskal",
            "rust:kruskal",
            "rust-kruskal-mst",
            "rust:kruskal-mst",
            "mst-kruskal",
            "minimum-spanning-tree-kruskal",
            "minimum-spanning-tree-exact",
        ] {
            assert_eq!(
                parse_solver(
                    "minimum_spanning_tree_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalMinimumSpanningTreeReferenceSolver::RustKruskal
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools:mst",
            "ortools:cp-sat",
            "cp-sat-mst",
            "ortools-cp-sat-mst",
            "ortools:cp-sat-mst",
        ] {
            assert_eq!(
                parse_solver(
                    "minimum_spanning_tree_reference",
                    ["--solver".to_string(), alias.to_string()]
                )
                .expect(alias),
                ExternalMinimumSpanningTreeReferenceSolver::OrTools
            );
        }

        assert_eq!(
            parse_solver(
                "minimum_spanning_tree_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalMinimumSpanningTreeReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error =
            run(vec!["minimum_spanning_tree_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("vertices"));
    }
}
