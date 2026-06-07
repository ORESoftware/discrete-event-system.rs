use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_weighted_independent_set_reference::{
    solve_weighted_independent_set_with_external_reference,
    ExternalWeightedIndependentSetReferenceOptions,
    ExternalWeightedIndependentSetReferenceSolution, ExternalWeightedIndependentSetReferenceSolver,
};
use des_engine::des::general::weighted_independent_set::{
    WeightedIndependentSetProblem, WeightedIndependentSetVertex,
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
        "usage: {program} [--solver auto|fallback|rust-branch-and-bound|rust:branch-and-bound-weighted-independent-set|ortools|ortools:cp-sat-weighted-independent-set]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalWeightedIndependentSetReferenceSolver, CliError> {
    let mut solver = ExternalWeightedIndependentSetReferenceSolver::Auto;
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
                    "auto" | "default" => ExternalWeightedIndependentSetReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalWeightedIndependentSetReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-branch-and-bound"
                    | "rust:branch-and-bound"
                    | "rust-exact"
                    | "rust:exact"
                    | "weighted-independent-set"
                    | "independent-set"
                    | "maximum-weight-independent-set"
                    | "branch-and-bound-weighted-independent-set"
                    | "rust-weighted-independent-set"
                    | "rust:weighted-independent-set"
                    | "rust-branch-and-bound-weighted-independent-set"
                    | "rust:branch-and-bound-weighted-independent-set"
                    | "rust-exact-weighted-independent-set"
                    | "rust:exact-weighted-independent-set" => {
                        ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-cp-sat"
                    | "ortools:cp-sat"
                    | "or-tools-cp-sat"
                    | "cp-sat-weighted-independent-set"
                    | "ortools-weighted-independent-set"
                    | "ortools:weighted-independent-set"
                    | "ortools-cp-sat-weighted-independent-set"
                    | "ortools:cp-sat-weighted-independent-set"
                    | "or-tools-cp-sat-weighted-independent-set" => {
                        ExternalWeightedIndependentSetReferenceSolver::OrTools
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

fn parse_vertices(raw: &Value) -> Result<Vec<WeightedIndependentSetVertex>, String> {
    value_array(raw, "vertices", "vertices must be non-empty").map(|items| {
        items
            .iter()
            .enumerate()
            .map(|(index, raw_vertex)| {
                if let Some(object) = raw_vertex.as_object() {
                    let id = object
                        .get("id")
                        .map(value_as_string)
                        .unwrap_or_else(|| format!("V{}", index + 1));
                    let weight = object.get("weight").and_then(Value::as_f64).unwrap_or(0.0);
                    WeightedIndependentSetVertex { id, weight }
                } else {
                    WeightedIndependentSetVertex {
                        id: value_as_string(raw_vertex),
                        weight: 1.0,
                    }
                }
            })
            .collect::<Vec<_>>()
    })
}

fn parse_edges(raw: &Value) -> Result<Vec<(String, String)>, String> {
    let raw_edges = value_array(raw, "edges", "edges must be an array")?;
    raw_edges
        .iter()
        .enumerate()
        .map(|(index, raw_edge)| {
            let edge = raw_edge
                .as_array()
                .ok_or_else(|| format!("edges[{index}] must be a two-item list"))?;
            if edge.len() != 2 {
                return Err(format!("edges[{index}] must be a two-item list"));
            }
            Ok((value_as_string(&edge[0]), value_as_string(&edge[1])))
        })
        .collect()
}

fn parse_problem(raw: &Value) -> Result<WeightedIndependentSetProblem, String> {
    Ok(WeightedIndependentSetProblem {
        vertices: parse_vertices(raw)?,
        edges: parse_edges(raw)?,
    })
}

fn solution_json(solution: &ExternalWeightedIndependentSetReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "selectedVertexIndices": solution.selected_vertex_indices,
        "selectedVertexIds": solution.selected_vertex_ids,
        "totalWeight": solution.total_weight,
        "objective": solution.objective,
        "upperBound": solution.upper_bound,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_selected_vertex_indices.is_empty()
        || solution.ortools_objective.is_some()
        || solution.ortools_objective_bound.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsSelectedVertexIndices"] = json!(solution.ortools_selected_vertex_indices);
        output["ortoolsSelectedVertexIds"] = json!(solution.ortools_selected_vertex_ids);
        output["ortoolsTotalWeight"] = json!(solution.ortools_total_weight);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsObjectiveBound"] = json!(solution.ortools_objective_bound);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:weighted-independent-set-reference",
        "selectedVertexIndices": [],
        "selectedVertexIds": [],
        "totalWeight": 0.0,
        "objective": null,
        "upperBound": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "weighted_independent_set_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let problem = parse_problem(&payload).map_err(CliError)?;
    let solution = solve_weighted_independent_set_with_external_reference(
        &problem,
        &ExternalWeightedIndependentSetReferenceOptions { solver },
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
                    .unwrap_or("weighted_independent_set_reference")
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
                serde_json::to_string(&output).expect("serialize weighted-independent-set output")
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

    static WEIGHTED_INDEPENDENT_SET_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn weighted_independent_set_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "WEIGHTED_INDEPENDENT_SET_REFERENCE_FORCE_PYTHON",
            "WEIGHTED_INDEPENDENT_SET_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "vertices": [
            {"id": "A", "weight": 8.0},
            {"id": "B", "weight": 7.0},
            {"id": "C", "weight": 6.0},
            {"id": "D", "weight": 6.0},
            {"id": "E", "weight": 5.0},
            {"id": "F", "weight": 4.0},
            {"id": "G", "weight": 3.0}
        ],
        "edges": [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "E"], ["C", "D"], ["C", "F"], ["D", "E"], ["D", "F"], ["E", "F"], ["E", "G"], ["F", "G"]]
    }"#;

    #[test]
    fn fallback_uses_rust_branch_and_bound_reference() {
        let output = run(
            vec![
                "weighted_independent_set_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:branch-and-bound-weighted-independent-set"
        );
        assert_eq!(output["selectedVertexIds"], json!(["B", "D", "G"]));
        assert_eq!(output["totalWeight"], 16.0);
    }

    #[test]
    fn string_vertices_default_to_unit_weight() {
        let output = run(
            vec!["weighted_independent_set_reference".to_string()],
            r#"{"vertices":["A","B","C"],"edges":[["A","B"]]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["totalWeight"], 2.0);
        assert_eq!(output["selectedVertexIds"], json!(["A", "C"]));
    }

    #[test]
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = WEIGHTED_INDEPENDENT_SET_CLI_ENV_LOCK
            .lock()
            .expect("lock weighted-independent-set CLI env guard");
        let _force_python_guards = weighted_independent_set_force_python_off_guards();
        let _python_bin_guard = EnvVarGuard::set(
            "PYTHON_BIN",
            "/definitely/not-python-for-weighted-independent-set-cli",
        );
        let _python_guard = EnvVarGuard::set(
            "PYTHON",
            "/definitely/not-python-for-weighted-independent-set-cli",
        );

        let output = run(
            vec![
                "weighted_independent_set_reference".to_string(),
                "--solver=ortools:cp-sat-weighted-independent-set".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:registered-weighted-independent-set-fallback-for-ortools"
        );
        assert_eq!(output["selectedVertexIds"], json!(["B", "D", "G"]));
        assert_eq!(output["totalWeight"], 16.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_weighted_independent_set_solver_aliases_used_by_validation_tools() {
        let rust_aliases = [
            "rust",
            "native",
            "exact",
            "rust:exact",
            "rust_branch_and_bound_weighted_independent_set",
            "rust:branch-and-bound-weighted-independent-set",
            "rust:exact-weighted-independent-set",
        ];
        for alias in rust_aliases {
            let solver = parse_solver(
                "weighted_independent_set_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(
                solver,
                ExternalWeightedIndependentSetReferenceSolver::RustBranchAndBound
            );
        }

        let ortools_aliases = [
            "or-tools",
            "google-ortools",
            "ortools:cp-sat",
            "ortools_cp_sat_weighted_independent_set",
            "ortools:cp-sat-weighted-independent-set",
        ];
        for alias in ortools_aliases {
            let solver = parse_solver(
                "weighted_independent_set_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(
                solver,
                ExternalWeightedIndependentSetReferenceSolver::OrTools
            );
        }

        let fallback = parse_solver(
            "weighted_independent_set_reference",
            ["--solver=rust:fallback".to_string()],
        )
        .expect("fallback alias");
        assert_eq!(
            fallback,
            ExternalWeightedIndependentSetReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error =
            run(vec!["weighted_independent_set_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("vertices must be non-empty"));
    }
}
