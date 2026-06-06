use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::advanced_optimization_models::PortfolioAsset;
use des_engine::des::general::external_nonlinear_reference::{
    solve_exponential_fit_with_external_reference, solve_global_benchmark_with_external_reference,
    solve_pareto_portfolio_with_external_reference, solve_rosenbrock_with_external_reference,
    ExternalNonlinearBenchmarkObjective, ExternalNonlinearReferenceOptions,
    ExternalNonlinearReferenceSolution, ExternalNonlinearReferenceSolver,
    ExternalNonlinearReferenceStatus, ExternalParetoPortfolioReferenceSolution,
};
use des_engine::des::general::nonlinear_optimization_models::CurveFitPoint;
use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Clone, Debug)]
struct Args {
    solver: ExternalNonlinearReferenceSolver,
    max_iterations: Option<usize>,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--solver auto|rust|rust:nonlinear-reference|fallback|scipy|scipy:slsqp|nlopt|nlopt:bobyqa] [--max-iterations N]"
    )
}

fn parse_solver(value: &str) -> Result<ExternalNonlinearReferenceSolver, CliError> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "auto" => Ok(ExternalNonlinearReferenceSolver::Auto),
        "rust"
        | "rust-fallback"
        | "rust:fallback"
        | "rust-reference"
        | "rust:reference"
        | "rust-nonlinear-reference"
        | "rust:nonlinear-reference"
        | "rust-known-rosenbrock-minimum"
        | "rust:known-rosenbrock-minimum"
        | "rust-gauss-newton"
        | "rust:gauss-newton"
        | "rust-analytic-global-benchmark"
        | "rust:analytic-global-benchmark"
        | "rust-bounded-center"
        | "rust:bounded-center"
        | "rust-pareto-portfolio-enumeration"
        | "rust:pareto-portfolio-enumeration" => Ok(ExternalNonlinearReferenceSolver::RustFallback),
        "fallback" | "builtin" | "builtin-nonlinear-reference" | "builtin:nonlinear-reference" => {
            Ok(ExternalNonlinearReferenceSolver::Fallback)
        }
        "scipy"
        | "scipy-minimize"
        | "scipy:minimize"
        | "scipy-slsqp"
        | "scipy:slsqp"
        | "scipy-lbfgsb"
        | "scipy:lbfgsb"
        | "rust-registered-nonlinear-fallback-for-scipy"
        | "rust:registered-nonlinear-fallback-for-scipy" => {
            Ok(ExternalNonlinearReferenceSolver::Scipy)
        }
        "nlopt"
        | "nlopt-cli"
        | "nlopt:bobyqa"
        | "nlopt-cobyla"
        | "nlopt:cobyla"
        | "nlopt-nelder-mead"
        | "nlopt:nelder-mead"
        | "nlopt-lbfgs"
        | "nlopt:lbfgs"
        | "rust-registered-nonlinear-fallback-for-nlopt"
        | "rust:registered-nonlinear-fallback-for-nlopt" => {
            Ok(ExternalNonlinearReferenceSolver::Nlopt)
        }
        other => Err(CliError(format!("unknown solver {other:?}"))),
    }
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{option} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<Args, CliError> {
    let mut parsed = Args {
        solver: ExternalNonlinearReferenceSolver::Auto,
        max_iterations: None,
    };
    let mut values = args.into_iter();
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
                let value = next_option_value(program, "--solver", inline_value, &mut values)?;
                parsed.solver = parse_solver(&value)?;
            }
            "--max-iterations" => {
                let value =
                    next_option_value(program, "--max-iterations", inline_value, &mut values)?;
                parsed.max_iterations = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| CliError(format!("invalid --max-iterations: {err}")))?,
                );
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(parsed)
}

fn get_any<'a>(raw: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| raw.get(*key))
}

fn finite_number(value: &Value, name: impl Into<String>) -> Result<f64, CliError> {
    let name = name.into();
    let number = value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| CliError(format!("{name} must be numeric")))?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(CliError(format!("{name} must be finite")))
    }
}

fn numbers(value: Option<&Value>, name: &str) -> Result<Vec<f64>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?
        .iter()
        .enumerate()
        .map(|(idx, value)| finite_number(value, format!("{name}[{idx}]")))
        .collect()
}

fn default_curve_fit_points() -> Vec<CurveFitPoint> {
    vec![
        CurveFitPoint { x: 0.0, y: 2.00 },
        CurveFitPoint { x: 1.0, y: 1.22 },
        CurveFitPoint { x: 2.0, y: 0.74 },
        CurveFitPoint { x: 3.0, y: 0.45 },
        CurveFitPoint { x: 4.0, y: 0.27 },
    ]
}

fn parse_points(raw: &Value) -> Result<Vec<CurveFitPoint>, CliError> {
    let Some(points) = get_any(raw, &["points"]) else {
        return Ok(default_curve_fit_points());
    };
    let points = points
        .as_array()
        .ok_or_else(|| CliError("points must be a list".to_string()))?;
    if points.is_empty() {
        return Err(CliError("points must be non-empty".to_string()));
    }
    points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            Ok(CurveFitPoint {
                x: finite_number(
                    get_any(point, &["x"])
                        .ok_or_else(|| CliError(format!("points[{idx}].x missing")))?,
                    format!("points[{idx}].x"),
                )?,
                y: finite_number(
                    get_any(point, &["y"])
                        .ok_or_else(|| CliError(format!("points[{idx}].y missing")))?,
                    format!("points[{idx}].y"),
                )?,
            })
        })
        .collect()
}

fn parse_assets(raw: &Value) -> Result<Vec<PortfolioAsset>, CliError> {
    let Some(assets) = get_any(raw, &["assets"]) else {
        return Ok(Vec::new());
    };
    let assets = assets
        .as_array()
        .ok_or_else(|| CliError("assets must be a list".to_string()))?;
    let mut out = Vec::with_capacity(assets.len());
    for (idx, asset) in assets.iter().enumerate() {
        out.push(PortfolioAsset {
            name: get_any(asset, &["name"])
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("asset-{idx}")),
            expected_return: finite_number(
                get_any(asset, &["expectedReturn", "expected_return"])
                    .ok_or_else(|| CliError(format!("assets[{idx}].expectedReturn missing")))?,
                format!("assets[{idx}].expectedReturn"),
            )?,
            risk: finite_number(
                get_any(asset, &["risk"])
                    .ok_or_else(|| CliError(format!("assets[{idx}].risk missing")))?,
                format!("assets[{idx}].risk"),
            )?,
        });
    }
    Ok(out)
}

fn benchmark_objective(raw: &Value) -> Result<ExternalNonlinearBenchmarkObjective, CliError> {
    match get_any(raw, &["objective"])
        .and_then(Value::as_str)
        .unwrap_or("sphere")
        .to_ascii_lowercase()
        .replace('_', "-")
        .as_str()
    {
        "sphere" => Ok(ExternalNonlinearBenchmarkObjective::Sphere),
        "rastrigin" => Ok(ExternalNonlinearBenchmarkObjective::Rastrigin),
        "rosenbrock" => Ok(ExternalNonlinearBenchmarkObjective::Rosenbrock),
        other => Err(CliError(format!("unknown objective {other:?}"))),
    }
}

fn solution_json(solution: &ExternalNonlinearReferenceSolution) -> Value {
    json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "x": solution.x,
        "objective": solution.objective,
        "gradientNorm": solution.gradient_norm,
        "residualNorm": solution.residual_norm,
        "iterations": solution.iterations,
        "evaluations": solution.evaluations,
        "message": solution.message,
        "elapsedMs": solution.elapsed_ms,
    })
}

fn pareto_solution_json(solution: &ExternalParetoPortfolioReferenceSolution) -> Value {
    json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "paretoFront": solution.pareto_front.iter().map(|point| json!({
            "weights": point.weights,
            "expectedReturn": point.expected_return,
            "risk": point.risk,
        })).collect::<Vec<_>>(),
        "candidateCount": solution.candidate_count,
        "hypervolume": solution.hypervolume,
        "message": solution.message,
        "elapsedMs": solution.elapsed_ms,
    })
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": ExternalNonlinearReferenceStatus::NumericalError.as_str(),
        "solver": "rust:nonlinear-reference",
        "x": [],
        "objective": null,
        "gradientNorm": null,
        "residualNorm": null,
        "iterations": null,
        "evaluations": null,
        "message": message.into(),
    })
}

fn unsupported_json(kind: &str) -> Value {
    json!({
        "status": ExternalNonlinearReferenceStatus::Unsupported.as_str(),
        "solver": "nonlinear-reference",
        "x": [],
        "objective": null,
        "gradientNorm": null,
        "residualNorm": null,
        "iterations": null,
        "evaluations": null,
        "message": format!("unknown kind: {kind}"),
    })
}

fn solve(raw: &Value, args: &Args) -> Result<Value, CliError> {
    let opts = ExternalNonlinearReferenceOptions {
        solver: args.solver,
        max_iterations: args.max_iterations,
    };
    let kind = get_any(raw, &["kind"])
        .and_then(Value::as_str)
        .unwrap_or("rosenbrock")
        .to_ascii_lowercase()
        .replace('-', "_");
    match kind.as_str() {
        "rosenbrock" => {
            let mut x0 = numbers(get_any(raw, &["x0", "initial"]), "x0")?;
            if x0.is_empty() {
                x0 = vec![-1.2, 1.0];
            }
            Ok(solution_json(&solve_rosenbrock_with_external_reference(
                &x0, &opts,
            )))
        }
        "least_squares" | "least_squares_fit" | "curve_fit" => {
            let points = parse_points(raw)?;
            let mut initial = numbers(get_any(raw, &["initial", "x0"]), "initial")?;
            if initial.is_empty() {
                initial = vec![1.0, -0.2];
            }
            Ok(solution_json(
                &solve_exponential_fit_with_external_reference(&points, &initial, &opts),
            ))
        }
        "global_benchmark" | "global" => {
            let objective = benchmark_objective(raw)?;
            let dimension = get_any(raw, &["dimension"])
                .map(|value| finite_number(value, "dimension"))
                .transpose()?
                .unwrap_or(3.0) as usize;
            let lower = get_any(raw, &["lower"])
                .map(|value| finite_number(value, "lower"))
                .transpose()?
                .unwrap_or(-5.0);
            let upper = get_any(raw, &["upper"])
                .map(|value| finite_number(value, "upper"))
                .transpose()?
                .unwrap_or(5.0);
            Ok(solution_json(
                &solve_global_benchmark_with_external_reference(
                    objective, dimension, lower, upper, &opts,
                ),
            ))
        }
        "pareto_portfolio" => {
            let assets = parse_assets(raw)?;
            let samples = get_any(raw, &["samples"])
                .map(|value| finite_number(value, "samples"))
                .transpose()?
                .unwrap_or(240.0) as usize;
            let seed = get_any(raw, &["seed"])
                .map(|value| finite_number(value, "seed"))
                .transpose()?
                .unwrap_or(19.0) as u32;
            Ok(pareto_solution_json(
                &solve_pareto_portfolio_with_external_reference(&assets, samples, seed, &opts),
            ))
        }
        _ => Ok(unsupported_json(&kind)),
    }
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "nonlinear_reference".to_string());
    let args = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    solve(&payload, &args)
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                raw_args
                    .first()
                    .map(String::as_str)
                    .unwrap_or("nonlinear_reference")
            )
        );
        return;
    }
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!("{}", error_json(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(raw_args, &stdin) {
        Ok(output) => {
            println!(
                "{}",
                serde_json::to_string(&output).expect("serialize nonlinear reference output")
            );
            if output["status"] == ExternalNonlinearReferenceStatus::Unavailable.as_str() {
                std::process::exit(2);
            }
        }
        Err(err) => {
            println!("{}", error_json(err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_rust_and_external_solver_labels_used_by_validation_tools() {
        for raw in [
            "rust:nonlinear-reference",
            "rust:known-rosenbrock-minimum",
            "rust:gauss-newton",
            "rust:analytic-global-benchmark",
            "rust:pareto-portfolio-enumeration",
        ] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalNonlinearReferenceSolver::RustFallback,
                "{raw}"
            );
        }
        for raw in ["fallback", "builtin:nonlinear-reference"] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalNonlinearReferenceSolver::Fallback,
                "{raw}"
            );
        }
        for raw in [
            "scipy:minimize",
            "scipy:slsqp",
            "ScIpY_LbFgSb",
            "rust:registered-nonlinear-fallback-for-scipy",
        ] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalNonlinearReferenceSolver::Scipy,
                "{raw}"
            );
        }
        for raw in [
            "nlopt:bobyqa",
            "nlopt:cobyla",
            "NlOpT_NeLdEr_MeAd",
            "rust:registered-nonlinear-fallback-for-nlopt",
        ] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalNonlinearReferenceSolver::Nlopt,
                "{raw}"
            );
        }
    }

    #[test]
    fn rust_rosenbrock_cli_uses_known_minimum() {
        let output = run(
            vec![
                "nonlinear_reference".to_string(),
                "--solver=fallback".to_string(),
            ],
            r#"{"kind":"rosenbrock","x0":[-1.2,1.0,0.8]}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:known-rosenbrock-minimum");
        assert_eq!(output["x"].as_array().expect("x").len(), 3);
    }

    #[test]
    fn rust_least_squares_cli_uses_gauss_newton() {
        let output = run(
            vec![
                "nonlinear_reference".to_string(),
                "--solver=auto".to_string(),
            ],
            r#"{"kind":"least_squares"}"#,
        )
        .expect("run");

        assert!(matches!(
            output["status"].as_str(),
            Some("optimal" | "feasible")
        ));
        assert_eq!(output["solver"], "rust:gauss-newton");
        assert!(output["objective"].as_f64().is_some());
    }

    #[test]
    fn rust_global_cli_solves_known_rastrigin_minimum() {
        let output = run(
            vec![
                "nonlinear_reference".to_string(),
                "--solver=rust".to_string(),
            ],
            r#"{"kind":"global_benchmark","objective":"rastrigin","dimension":3,"lower":-5.12,"upper":5.12}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:known-rastrigin-minimum");
        assert!(output["objective"]
            .as_f64()
            .is_some_and(|value| value <= 1e-12));
    }

    #[test]
    fn rust_pareto_cli_returns_front() {
        let output = run(
            vec![
                "nonlinear_reference".to_string(),
                "--solver=fallback".to_string(),
            ],
            r#"{"kind":"pareto_portfolio","samples":32,"seed":7}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:pareto-portfolio-enumeration");
        assert!(output["paretoFront"]
            .as_array()
            .is_some_and(|front| !front.is_empty()));
    }
}
