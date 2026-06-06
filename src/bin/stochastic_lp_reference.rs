use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_stochastic_lp_reference::{
    solve_stochastic_lp_with_external_reference, ExternalStochasticLpReferenceOptions,
    ExternalStochasticLpReferenceSolution, ExternalStochasticLpReferenceSolver,
    ExternalStochasticLpReferenceStatus,
};
use des_engine::des::general::stochastic_lp::{SLPProblem, Scenario};
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
        "usage: {program} [--solver auto|rust-monolithic|highs|scipy|scipy-highs|scipy-highs-ds|scipy-highs-ipm|fallback]"
    )
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

fn parse_solver(value: &str) -> Result<ExternalStochasticLpReferenceSolver, CliError> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "auto" => Ok(ExternalStochasticLpReferenceSolver::Auto),
        "rust"
        | "native"
        | "rust-native"
        | "rust-monolithic"
        | "rust-monolithic-slp"
        | "rust:monolithic"
        | "rust:monolithic-slp"
        | "monolithic"
        | "monolithic-slp"
        | "slp"
        | "highs"
        | "highs-cli"
        | "highs:cli"
        | "rust-highs"
        | "rust:highs" => Ok(ExternalStochasticLpReferenceSolver::RustMonolithic),
        "scipy"
        | "scipy-highs"
        | "scipy:highs"
        | "scipy-highs-ds"
        | "scipy:highs-ds"
        | "scipy-highs-simplex"
        | "scipy:highs-simplex"
        | "scipy-highs-dual-simplex"
        | "scipy:highs-dual-simplex"
        | "scipy-highs-ipm"
        | "scipy:highs-ipm" => Ok(ExternalStochasticLpReferenceSolver::Scipy),
        "fallback" | "rust-fallback" | "rust:fallback" => {
            Ok(ExternalStochasticLpReferenceSolver::Fallback)
        }
        other => Err(CliError(format!("unknown solver {other:?}"))),
    }
}

fn parse_args(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<ExternalStochasticLpReferenceSolver, CliError> {
    let mut solver = ExternalStochasticLpReferenceSolver::Auto;
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
                solver = parse_solver(&value)?;
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
    let value = value.ok_or_else(|| CliError(format!("{name} must be a list")))?;
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?
        .iter()
        .enumerate()
        .map(|(idx, value)| finite_number(value, format!("{name}[{idx}]")))
        .collect()
}

fn matrix(
    value: Option<&Value>,
    name: &str,
    cols: Option<usize>,
) -> Result<Vec<Vec<f64>>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let rows = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list of rows")))?;
    let mut out = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        let row_values = numbers(Some(row), &format!("{name}[{row_idx}]"))?;
        if let Some(cols) = cols {
            if row_values.len() != cols {
                return Err(CliError(format!(
                    "{name}[{row_idx}] length {} != {cols}",
                    row_values.len()
                )));
            }
        }
        out.push(row_values);
    }
    Ok(out)
}

fn normalize(raw: &Value) -> Result<(SLPProblem, Vec<Scenario>), CliError> {
    let source = raw.get("problem").unwrap_or(raw);
    let c_first = numbers(get_any(source, &["cFirst", "c_first", "c"]), "cFirst")?;
    let q_second = numbers(get_any(source, &["qSecond", "q_second", "q"]), "qSecond")?;
    let n_first = c_first.len();
    let n_second = q_second.len();
    if n_first == 0 || n_second == 0 {
        return Err(CliError("cFirst and qSecond must be non-empty".to_string()));
    }

    let a_first = matrix(
        get_any(source, &["aFirst", "a_first", "A"]),
        "aFirst",
        Some(n_first),
    )?;
    let b_first = numbers(get_any(source, &["bFirst", "b_first", "b"]), "bFirst")
        .unwrap_or_else(|_| Vec::new());
    if a_first.len() != b_first.len() {
        return Err(CliError(format!(
            "aFirst rows {} != bFirst length {}",
            a_first.len(),
            b_first.len()
        )));
    }

    let w_second = matrix(
        get_any(source, &["wSecond", "w_second", "W"]),
        "wSecond",
        Some(n_second),
    )?;
    if w_second.is_empty() {
        return Err(CliError("wSecond must be non-empty".to_string()));
    }

    let raw_scenarios = get_any(source, &["scenarios", "scenarioSet"])
        .and_then(Value::as_array)
        .ok_or_else(|| CliError("scenarios must be a non-empty list".to_string()))?;
    if raw_scenarios.is_empty() {
        return Err(CliError("scenarios must be a non-empty list".to_string()));
    }

    let default_prob = 1.0 / raw_scenarios.len() as f64;
    let mut scenarios = Vec::with_capacity(raw_scenarios.len());
    for (scenario_idx, raw_scenario) in raw_scenarios.iter().enumerate() {
        let raw_scenario = raw_scenario
            .as_object()
            .ok_or_else(|| CliError(format!("scenarios[{scenario_idx}] must be an object")))?;
        let scenario_value = Value::Object(raw_scenario.clone());
        let t = matrix(
            get_any(&scenario_value, &["t", "T"]),
            &format!("scenarios[{scenario_idx}].t"),
            Some(n_first),
        )?;
        let h = numbers(
            get_any(&scenario_value, &["h"]),
            &format!("scenarios[{scenario_idx}].h"),
        )?;
        if t.len() != w_second.len() || h.len() != w_second.len() {
            return Err(CliError(format!(
                "scenarios[{scenario_idx}] must have {} recourse rows; got T={} h={}",
                w_second.len(),
                t.len(),
                h.len()
            )));
        }
        let prob = get_any(&scenario_value, &["prob", "probability"])
            .map(|value| finite_number(value, format!("scenarios[{scenario_idx}].prob")))
            .transpose()?
            .unwrap_or(default_prob);
        if prob < 0.0 {
            return Err(CliError(format!(
                "scenarios[{scenario_idx}].prob must be non-negative"
            )));
        }
        scenarios.push(Scenario {
            t,
            h,
            prob: Some(prob),
            meta: None,
        });
    }

    let theta_lower_bound = get_any(source, &["thetaLowerBound", "theta_lower_bound"])
        .map(|value| finite_number(value, "thetaLowerBound"))
        .transpose()?
        .unwrap_or(0.0);
    let theta_upper_bound = get_any(source, &["thetaUpperBound", "theta_upper_bound"])
        .map(|value| finite_number(value, "thetaUpperBound"))
        .transpose()?
        .unwrap_or(1.0e9);

    Ok((
        SLPProblem {
            c_first,
            a_first,
            b_first,
            q_second,
            w_second,
            theta_lower_bound,
            theta_upper_bound,
            var_names: None,
        },
        scenarios,
    ))
}

fn solution_json(solution: &ExternalStochasticLpReferenceSolution) -> Value {
    json!({
        "status": solution.status.as_str(),
        "solver": solution.solver,
        "x": solution.x,
        "objective": solution.objective,
        "cFirstX": solution.c_first_x,
        "expectedQ": solution.expected_q,
        "yByScenario": solution.y_by_scenario,
        "scenarioValues": solution.scenario_values,
        "iterations": solution.iterations,
        "message": solution.message,
    })
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": ExternalStochasticLpReferenceStatus::NumericalError.as_str(),
        "solver": "rust:stochastic-lp-reference",
        "x": [],
        "objective": null,
        "cFirstX": null,
        "expectedQ": null,
        "yByScenario": [],
        "scenarioValues": [],
        "iterations": null,
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "stochastic_lp_reference".to_string());
    let solver = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    let (problem, scenarios) = normalize(&payload)?;
    let solution = solve_stochastic_lp_with_external_reference(
        &problem,
        &scenarios,
        &ExternalStochasticLpReferenceOptions { solver },
    );
    Ok(solution_json(&solution))
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
                    .unwrap_or("stochastic_lp_reference")
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
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize stochastic LP output")
        ),
        Err(err) => {
            println!("{}", error_json(err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "cFirst": [1.0],
        "aFirst": [[1.0]],
        "bFirst": [5.0],
        "qSecond": [3.0],
        "wSecond": [[1.0]],
        "scenarios": [
            {"t": [[-1.0]], "h": [2.0], "prob": 0.5},
            {"t": [[-1.0]], "h": [4.0], "prob": 0.5}
        ]
    }"#;

    #[test]
    fn rust_cli_solves_small_stochastic_lp() {
        let output = run(
            vec![
                "stochastic_lp_reference".to_string(),
                "--solver".to_string(),
                "rust-monolithic".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:monolithic-slp");
        assert_eq!(output["x"].as_array().expect("x").len(), 1);
        assert_eq!(output["yByScenario"].as_array().expect("y").len(), 2);
        assert!(output["objective"].as_f64().is_some());
    }

    #[test]
    fn fallback_alias_uses_rust_monolithic_reference() {
        let output = run(
            vec![
                "stochastic_lp_reference".to_string(),
                "--solver=fallback".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:monolithic-slp");
    }

    #[test]
    fn generic_highs_alias_uses_rust_monolithic_reference() {
        let output = run(
            vec![
                "stochastic_lp_reference".to_string(),
                "--solver=highs".to_string(),
            ],
            SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:monolithic-slp");
    }

    #[test]
    fn parses_stochastic_solver_aliases_used_by_rust_facades() {
        for alias in [
            "rust",
            "rust_monolithic",
            "rust:monolithic-slp",
            "monolithic-slp",
            "highs-cli",
            "highs:cli",
            "rust:highs",
        ] {
            assert_eq!(
                parse_solver(alias).expect(alias),
                ExternalStochasticLpReferenceSolver::RustMonolithic
            );
        }

        for alias in [
            "scipy",
            "scipy-highs",
            "scipy:highs",
            "scipy_highs_ds",
            "scipy-highs-simplex",
            "scipy-highs-dual-simplex",
            "scipy:highs-ipm",
        ] {
            assert_eq!(
                parse_solver(alias).expect(alias),
                ExternalStochasticLpReferenceSolver::Scipy
            );
        }

        assert_eq!(
            parse_solver("rust:fallback").expect("rust:fallback"),
            ExternalStochasticLpReferenceSolver::Fallback
        );
    }
}
