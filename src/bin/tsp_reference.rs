use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_tsp_reference::{
    solve_euclidean_tsp_with_external_reference, solve_tsp_with_external_reference,
    ExternalTspPoint, ExternalTspReferenceOptions, ExternalTspReferenceSolution,
    ExternalTspReferenceSolver,
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

enum TspInput {
    DistanceMatrix(Vec<Vec<f64>>),
    Points(Vec<ExternalTspPoint>),
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--solver auto|fallback|rust-held-karp|rust-held-karp-tsp|ortools|ortools-routing-tsp]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalTspReferenceSolver, CliError> {
    let mut solver = ExternalTspReferenceSolver::Auto;
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
                    "auto" => ExternalTspReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalTspReferenceSolver::Fallback
                    }
                    "rust"
                    | "native"
                    | "rust-native"
                    | "exact"
                    | "rust-exact"
                    | "rust:exact"
                    | "held-karp"
                    | "rust-held-karp"
                    | "rust:held-karp"
                    | "rust-held-karp-tsp"
                    | "rust:held-karp-tsp"
                    | "tsp-held-karp"
                    | "tsp-held-karp-solver" => ExternalTspReferenceSolver::RustHeldKarp,
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-tsp"
                    | "ortools-routing-tsp"
                    | "ortools:routing-tsp"
                    | "routing-tsp" => ExternalTspReferenceSolver::OrTools,
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

fn value_array<'a>(value: &'a Value, field: &str) -> Option<&'a Vec<Value>> {
    value.get(field).and_then(Value::as_array)
}

fn parse_number(value: &Value, message: impl Into<String>) -> Result<f64, String> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| message.into())
}

fn parse_point(raw: &Value, index: usize) -> Result<ExternalTspPoint, String> {
    if let Some(object) = raw.as_object() {
        let id = object.get("id").map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        });
        let x = object
            .get("x")
            .ok_or_else(|| format!("point {index} must include x"))?;
        let y = object
            .get("y")
            .ok_or_else(|| format!("point {index} must include y"))?;
        return Ok(ExternalTspPoint {
            id,
            x: parse_number(x, format!("point {index}.x must be numeric"))?,
            y: parse_number(y, format!("point {index}.y must be numeric"))?,
        });
    }
    if let Some(items) = raw.as_array() {
        if items.len() >= 2 {
            return Ok(ExternalTspPoint {
                id: Some(index.to_string()),
                x: parse_number(&items[0], format!("point {index}[0] must be numeric"))?,
                y: parse_number(&items[1], format!("point {index}[1] must be numeric"))?,
            });
        }
    }
    Err(format!(
        "point {index} must be an object with x/y or a length-2 array"
    ))
}

fn parse_points(raw: &Value) -> Result<Option<Vec<ExternalTspPoint>>, String> {
    let Some(raw_points) = value_array(raw, "points").or_else(|| value_array(raw, "cities")) else {
        return Ok(None);
    };
    let points = raw_points
        .iter()
        .enumerate()
        .map(|(index, point)| parse_point(point, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(points))
}

fn parse_distance_matrix(raw: &Value) -> Result<Option<Vec<Vec<f64>>>, String> {
    let Some(matrix) = raw
        .get("distanceMatrix")
        .or_else(|| raw.get("distance_matrix"))
    else {
        return Ok(None);
    };
    let rows = matrix
        .as_array()
        .ok_or_else(|| "distanceMatrix must be an array of rows".to_string())?;
    rows.iter()
        .enumerate()
        .map(|(row_index, row)| {
            let row = row
                .as_array()
                .ok_or_else(|| format!("distance row {row_index} must be an array"))?;
            row.iter()
                .enumerate()
                .map(|(column_index, value)| {
                    parse_number(
                        value,
                        format!("distance[{row_index}][{column_index}] must be numeric"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn parse_input(raw: &Value) -> Result<TspInput, String> {
    let points = parse_points(raw)?;
    let distance_matrix = parse_distance_matrix(raw)?;
    if let Some(matrix) = distance_matrix {
        if let Some(points) = points {
            if points.len() != matrix.len() {
                return Err(format!(
                    "points length {} != distance matrix size {}",
                    points.len(),
                    matrix.len()
                ));
            }
        }
        return Ok(TspInput::DistanceMatrix(matrix));
    }
    let Some(points) = points else {
        return Err("points or distanceMatrix is required".to_string());
    };
    Ok(TspInput::Points(points))
}

fn solution_json(solution: &ExternalTspReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "tour": solution.tour,
        "objective": solution.objective,
        "message": solution.message,
    });
    if solution.ortools_status.is_some()
        || !solution.ortools_tour.is_empty()
        || solution.ortools_objective.is_some()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsTour"] = json!(solution.ortools_tour);
        output["ortoolsObjective"] = json!(solution.ortools_objective);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:tsp-reference",
        "tour": [],
        "objective": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "tsp_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let opts = ExternalTspReferenceOptions { solver };
    let solution = match parse_input(&payload).map_err(CliError)? {
        TspInput::DistanceMatrix(matrix) => solve_tsp_with_external_reference(&matrix, &opts),
        TspInput::Points(points) => solve_euclidean_tsp_with_external_reference(&points, &opts),
    };
    Ok(solution_json(&solution))
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(args.first().map(String::as_str).unwrap_or("tsp_reference"))
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
                serde_json::to_string(&output).expect("serialize TSP output")
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

    static TSP_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn tsp_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "TSP_REFERENCE_FORCE_PYTHON",
            "TSP_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const UNIT_SQUARE_POINTS: &str = r#"{
        "points": [
            {"id": "A", "x": 0.0, "y": 0.0},
            {"id": "B", "x": 1.0, "y": 0.0},
            {"id": "C", "x": 1.0, "y": 1.0},
            {"id": "D", "x": 0.0, "y": 1.0}
        ]
    }"#;

    const UNIT_SQUARE_MATRIX: &str = r#"{
        "distanceMatrix": [
            [0.0, 1.0, 1.4142135623730951, 1.0],
            [1.0, 0.0, 1.0, 1.4142135623730951],
            [1.4142135623730951, 1.0, 0.0, 1.0],
            [1.0, 1.4142135623730951, 1.0, 0.0]
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_held_karp_for_points() {
        let output = run(
            vec![
                "tsp_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            UNIT_SQUARE_POINTS,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:held-karp-tsp");
        assert_eq!(output["tour"], json!([0, 1, 2, 3]));
        assert_eq!(output["objective"], 4.0);
    }

    #[test]
    fn accepts_distance_matrix_alias() {
        let output = run(vec!["tsp_reference".to_string()], UNIT_SQUARE_MATRIX).expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:held-karp-tsp");
        assert_eq!(output["tour"], json!([0, 1, 2, 3]));
        assert_eq!(output["objective"], 4.0);
    }

    #[test]
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = TSP_CLI_ENV_LOCK.lock().expect("lock TSP CLI env guard");
        let _force_python_guards = tsp_force_python_off_guards();
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-tsp-cli");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-tsp-cli");

        let output = run(
            vec![
                "tsp_reference".to_string(),
                "--solver=ortools:routing-tsp".to_string(),
            ],
            UNIT_SQUARE_POINTS,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:registered-tsp-fallback-for-ortools");
        assert_eq!(output["tour"], json!([0, 1, 2, 3]));
        assert_eq!(output["objective"], 4.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_tsp_solver_aliases_used_by_validation_tools() {
        for alias in [
            "rust",
            "native",
            "rust_exact",
            "held-karp",
            "rust:held-karp",
            "rust-held-karp-tsp",
            "rust:held-karp-tsp",
            "tsp-held-karp",
            "tsp-held-karp-solver",
        ] {
            assert_eq!(
                parse_solver("tsp_reference", ["--solver".to_string(), alias.to_string()])
                    .expect(alias),
                ExternalTspReferenceSolver::RustHeldKarp
            );
        }

        for alias in [
            "ortools",
            "or-tools",
            "google-or-tools",
            "ortools-routing-tsp",
            "ortools:routing-tsp",
            "routing-tsp",
        ] {
            assert_eq!(
                parse_solver("tsp_reference", ["--solver".to_string(), alias.to_string()])
                    .expect(alias),
                ExternalTspReferenceSolver::OrTools
            );
        }

        assert_eq!(
            parse_solver(
                "tsp_reference",
                ["--solver".to_string(), "rust:fallback".to_string()]
            )
            .expect("rust:fallback"),
            ExternalTspReferenceSolver::Fallback
        );
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["tsp_reference".to_string()], "{}").expect_err("error");
        assert!(error
            .to_string()
            .contains("points or distanceMatrix is required"));
    }
}
