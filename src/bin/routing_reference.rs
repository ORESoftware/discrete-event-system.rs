use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::classical_optimization_models::{Point, VRPCustomer};
use des_engine::des::general::external_routing_reference::{
    solve_cvrp_with_external_reference, ExternalRoutingReferenceOptions,
    ExternalRoutingReferenceSolution, ExternalRoutingReferenceSolver,
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

struct RoutingInput {
    depot: Point,
    customers: Vec<VRPCustomer>,
    vehicle_capacity: f64,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--solver auto|fallback|rust-exact|rust:exact-cvrp|ortools|ortools:routing]"
    )
}

fn parse_solver(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalRoutingReferenceSolver, CliError> {
    let mut solver = ExternalRoutingReferenceSolver::Auto;
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
                    "auto" | "default" => ExternalRoutingReferenceSolver::Auto,
                    "fallback" | "rust-fallback" | "rust:fallback" => {
                        ExternalRoutingReferenceSolver::Fallback
                    }
                    "rust" | "native" | "rust-native" | "exact" | "rust-exact" | "rust:exact"
                    | "routing" | "cvrp" | "exact-cvrp" | "rust-routing" | "rust:routing"
                    | "rust-cvrp" | "rust:cvrp" | "rust-exact-cvrp" | "rust:exact-cvrp" => {
                        ExternalRoutingReferenceSolver::RustExact
                    }
                    "ortools"
                    | "or-tools"
                    | "google-ortools"
                    | "google-or-tools"
                    | "ortools-routing"
                    | "ortools:routing"
                    | "or-tools-routing"
                    | "ortools-cvrp"
                    | "ortools:cvrp"
                    | "ortools-routing-cvrp"
                    | "ortools:routing-cvrp" => ExternalRoutingReferenceSolver::OrTools,
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

fn parse_point(raw: Option<&Value>) -> Result<Point, String> {
    let Some(raw) = raw else {
        return Ok(Point { x: 0.0, y: 0.0 });
    };
    let object = raw
        .as_object()
        .ok_or_else(|| "depot must be an object with x/y".to_string())?;
    Ok(Point {
        x: parse_number(
            object
                .get("x")
                .ok_or_else(|| "depot.x is required".to_string())?,
            "depot.x must be numeric",
        )?,
        y: parse_number(
            object
                .get("y")
                .ok_or_else(|| "depot.y is required".to_string())?,
            "depot.y must be numeric",
        )?,
    })
}

fn parse_id(value: Option<&Value>, index: usize) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => format!("c{index}"),
    }
}

fn parse_customers(raw: Option<&Value>) -> Result<Vec<VRPCustomer>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let customers = raw
        .as_array()
        .ok_or_else(|| "customers must be an array".to_string())?;
    customers
        .iter()
        .enumerate()
        .map(|(index, customer)| {
            let object = customer
                .as_object()
                .ok_or_else(|| format!("customers[{index}] must be an object"))?;
            Ok(VRPCustomer {
                id: parse_id(object.get("id"), index),
                x: parse_number(
                    object
                        .get("x")
                        .ok_or_else(|| format!("customers[{index}].x is required"))?,
                    format!("customers[{index}].x must be numeric"),
                )?,
                y: parse_number(
                    object
                        .get("y")
                        .ok_or_else(|| format!("customers[{index}].y is required"))?,
                    format!("customers[{index}].y must be numeric"),
                )?,
                demand: parse_number(
                    object
                        .get("demand")
                        .ok_or_else(|| format!("customers[{index}].demand is required"))?,
                    format!("customers[{index}].demand must be numeric"),
                )?,
            })
        })
        .collect()
}

fn parse_routing_input(raw: &Value) -> Result<RoutingInput, String> {
    let vehicle_capacity = raw
        .get("vehicle_capacity")
        .or_else(|| raw.get("capacity"))
        .ok_or_else(|| "vehicle_capacity is required".to_string())
        .and_then(|value| parse_number(value, "vehicle_capacity must be numeric"))?;
    Ok(RoutingInput {
        depot: parse_point(raw.get("depot"))?,
        customers: parse_customers(raw.get("customers"))?,
        vehicle_capacity,
    })
}

fn route_json(route: &des_engine::des::general::classical_optimization_models::VRPRoute) -> Value {
    json!({
        "customers": route.customers,
        "load": route.load,
        "distance": route.distance,
    })
}

fn solution_json(solution: &ExternalRoutingReferenceSolution) -> Value {
    let mut output = json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "routes": solution.routes.iter().map(route_json).collect::<Vec<_>>(),
        "objective": solution.objective,
        "message": solution.message,
    });
    if let Some(count) = solution.feasible_route_masks {
        output["feasibleRouteMasks"] = json!(count);
    }
    if solution.ortools_status.is_some()
        || !solution.ortools_routes.is_empty()
        || solution.ortools_objective.is_some()
        || !solution.ortools_message.is_empty()
    {
        output["ortoolsStatus"] = json!(solution.ortools_status);
        output["ortoolsRoutes"] = json!(solution
            .ortools_routes
            .iter()
            .map(route_json)
            .collect::<Vec<_>>());
        output["ortoolsObjective"] = json!(solution.ortools_objective);
        output["ortoolsMessage"] = json!(solution.ortools_message);
    }
    output
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "error",
        "solver": "rust:routing-reference",
        "routes": [],
        "objective": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "routing_reference".to_string());
    let solver = parse_solver(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON stdin: {err}")))?;
    let input = parse_routing_input(&payload).map_err(CliError)?;
    let solution = solve_cvrp_with_external_reference(
        input.depot,
        &input.customers,
        input.vehicle_capacity,
        &ExternalRoutingReferenceOptions { solver },
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
                    .unwrap_or("routing_reference")
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
                serde_json::to_string(&output).expect("serialize routing output")
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

    static ROUTING_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn routing_force_python_off_guards() -> Vec<EnvVarGuard> {
        [
            "ROUTING_REFERENCE_FORCE_PYTHON",
            "ROUTING_REFERENCE_ORTOOLS_FORCE_PYTHON",
            "ORES_EXTERNAL_REFERENCE_FORCE_PYTHON",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, "0"))
        .collect()
    }

    const SAMPLE: &str = r#"{
        "depot": {"x": 0.0, "y": 0.0},
        "vehicle_capacity": 5.0,
        "customers": [
            {"id": "A", "x": 1.0, "y": 2.0, "demand": 2.0},
            {"id": "B", "x": 2.0, "y": 1.0, "demand": 2.0},
            {"id": "C", "x": 4.0, "y": 1.0, "demand": 2.0},
            {"id": "D", "x": 5.0, "y": 2.0, "demand": 1.0},
            {"id": "E", "x": 3.0, "y": 4.0, "demand": 2.0}
        ]
    }"#;

    #[test]
    fn fallback_uses_rust_exact_reference() {
        let output = run(
            vec![
                "routing_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:exact-cvrp");
        assert_eq!(
            output["routes"]
                .as_array()
                .expect("routes")
                .iter()
                .map(|route| route["customers"].as_array().expect("customers").len())
                .sum::<usize>(),
            5
        );
        assert!(output["objective"].as_f64().expect("objective") > 0.0);
        assert!(output["feasibleRouteMasks"].as_u64().expect("masks") > 0);
    }

    #[test]
    fn accepts_capacity_alias() {
        let output = run(
            vec!["routing_reference".to_string()],
            r#"{"capacity": 1.0, "customers": []}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["objective"], 0.0);
        assert_eq!(output["routes"], json!([]));
    }

    #[test]
    fn ortools_cli_alias_defaults_to_rust_reference_without_python() {
        let _lock = ROUTING_CLI_ENV_LOCK
            .lock()
            .expect("lock routing CLI env guard");
        let _force_python_guards = routing_force_python_off_guards();
        let _python_bin_guard =
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not-python-for-routing-cli");
        let _python_guard = EnvVarGuard::set("PYTHON", "/definitely/not-python-for-routing-cli");

        let output = run(
            vec![
                "routing_reference".to_string(),
                "--solver=ortools:routing-cvrp".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(
            output["solver"],
            "rust:registered-routing-fallback-for-ortools"
        );
        assert_eq!(
            output["routes"]
                .as_array()
                .expect("routes")
                .iter()
                .map(|route| route["customers"].as_array().expect("customers").len())
                .sum::<usize>(),
            5
        );
        assert!(output["objective"].as_f64().expect("objective") > 0.0);
        assert!(output["message"]
            .as_str()
            .expect("message")
            .contains("validated with Rust fallback"));
    }

    #[test]
    fn parses_routing_solver_aliases_used_by_validation_tools() {
        let rust_aliases = [
            "rust",
            "native",
            "exact",
            "rust:exact",
            "rust_exact_cvrp",
            "rust:exact-cvrp",
            "rust:routing",
        ];
        for alias in rust_aliases {
            let solver = parse_solver(
                "routing_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(solver, ExternalRoutingReferenceSolver::RustExact);
        }

        let ortools_aliases = [
            "or-tools",
            "google-ortools",
            "ortools:routing",
            "ortools_cvrp",
            "ortools:routing-cvrp",
        ];
        for alias in ortools_aliases {
            let solver = parse_solver(
                "routing_reference",
                ["--solver".to_string(), alias.to_string()],
            )
            .expect(alias);
            assert_eq!(solver, ExternalRoutingReferenceSolver::OrTools);
        }

        let fallback = parse_solver("routing_reference", ["--solver=rust:fallback".to_string()])
            .expect("fallback alias");
        assert_eq!(fallback, ExternalRoutingReferenceSolver::Fallback);
    }

    #[test]
    fn invalid_payload_returns_error_to_caller() {
        let error = run(vec!["routing_reference".to_string()], "{}").expect_err("error");
        assert!(error.to_string().contains("vehicle_capacity is required"));
    }
}
