use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use des_engine::des::general::external_quadratic_reference::{
    solve_miqcp_with_external_reference, solve_miqp_with_external_reference,
    solve_misocp_with_external_reference, solve_qcp_with_external_reference,
    solve_qp_with_external_reference, solve_socp_with_external_reference,
    ExternalQuadraticReferenceOptions, ExternalQuadraticReferenceSolution,
    ExternalQuadraticReferenceSolver, ExternalQuadraticReferenceStatus,
};
use des_engine::des::general::qp::{
    MixedIntegerQuadraticProgram, MixedIntegerQuadraticallyConstrainedProgram,
    MixedIntegerSecondOrderConeProgram, QuadraticConstraint, QuadraticProgram,
    QuadraticallyConstrainedProgram, SecondOrderCone, SecondOrderConeProgram,
};
use serde_json::{json, Map, Value};

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
    solver: ExternalQuadraticReferenceSolver,
    max_enumerations: Option<usize>,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--solver auto|rust|fallback|highs|scipy|osqp|cvxpy|scs|clarabel|ecos|mosek|copt|qpoases|proxqp|cosmo|sdpa|csdp] [--max-enumerations N]"
    )
}

fn parse_solver(value: &str) -> Result<ExternalQuadraticReferenceSolver, CliError> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "auto" => Ok(ExternalQuadraticReferenceSolver::Auto),
        "rust" | "rust-internal" | "rust-active-set" => {
            Ok(ExternalQuadraticReferenceSolver::RustInternal)
        }
        "fallback" | "rust-fallback" | "builtin" => Ok(ExternalQuadraticReferenceSolver::Fallback),
        "highs" | "highspy" | "highs-qp" => Ok(ExternalQuadraticReferenceSolver::Highs),
        "scipy" | "scipy-slsqp" => Ok(ExternalQuadraticReferenceSolver::Scipy),
        "osqp" => Ok(ExternalQuadraticReferenceSolver::Osqp),
        "cvxpy" => Ok(ExternalQuadraticReferenceSolver::Cvxpy),
        "scs" => Ok(ExternalQuadraticReferenceSolver::Scs),
        "clarabel" => Ok(ExternalQuadraticReferenceSolver::Clarabel),
        "ecos" => Ok(ExternalQuadraticReferenceSolver::Ecos),
        "mosek" => Ok(ExternalQuadraticReferenceSolver::Mosek),
        "copt" => Ok(ExternalQuadraticReferenceSolver::Copt),
        "qpoases" => Ok(ExternalQuadraticReferenceSolver::Qpoases),
        "proxqp" => Ok(ExternalQuadraticReferenceSolver::Proxqp),
        "cosmo" => Ok(ExternalQuadraticReferenceSolver::Cosmo),
        "sdpa" => Ok(ExternalQuadraticReferenceSolver::Sdpa),
        "csdp" => Ok(ExternalQuadraticReferenceSolver::Csdp),
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
        solver: ExternalQuadraticReferenceSolver::Auto,
        max_enumerations: None,
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
            "--max-enumerations" => {
                let value =
                    next_option_value(program, "--max-enumerations", inline_value, &mut values)?;
                parsed.max_enumerations = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| CliError(format!("invalid --max-enumerations: {err}")))?,
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

fn optional_numbers(value: Option<&Value>, name: &str) -> Result<Vec<f64>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?
        .iter()
        .enumerate()
        .map(|(idx, value)| finite_number(value, format!("{name}[{idx}]")))
        .collect()
}

fn numbers(value: Option<&Value>, name: &str) -> Result<Vec<f64>, CliError> {
    let value = value.ok_or_else(|| CliError(format!("{name} must be a list")))?;
    optional_numbers(Some(value), name)
}

fn matrix(
    value: Option<&Value>,
    name: &str,
    cols: Option<usize>,
) -> Result<Vec<Vec<f64>>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
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

fn square_matrix_or_zero(
    value: Option<&Value>,
    name: &str,
    n: usize,
) -> Result<Vec<Vec<f64>>, CliError> {
    let rows = matrix(value, name, Some(n))?;
    if rows.is_empty() {
        Ok(vec![vec![0.0; n]; n])
    } else if rows.len() == n {
        Ok(rows)
    } else {
        Err(CliError(format!("{name} row count {} != {n}", rows.len())))
    }
}

fn optional_bounds(
    value: Option<&Value>,
    name: &str,
    n: usize,
    default: Option<Option<f64>>,
) -> Result<Option<Vec<Option<f64>>>, CliError> {
    let Some(value) = value else {
        return Ok(default.map(|item| vec![item; n]));
    };
    if value.is_null() {
        return Ok(default.map(|item| vec![item; n]));
    }
    let items = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?;
    if items.len() != n {
        return Err(CliError(format!("{name} length {} != {n}", items.len())));
    }
    let mut out = Vec::with_capacity(n);
    for (idx, item) in items.iter().enumerate() {
        if item.is_null() {
            out.push(None);
        } else {
            out.push(Some(finite_number(item, format!("{name}[{idx}]"))?));
        }
    }
    Ok(Some(out))
}

fn bools(value: Option<&Value>, name: &str) -> Result<Vec<bool>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            item.as_bool()
                .or_else(|| item.as_i64().map(|value| value != 0))
                .ok_or_else(|| CliError(format!("{name}[{idx}] must be boolean")))
        })
        .collect()
}

fn string_list(value: Option<&Value>, name: &str) -> Result<Option<Vec<String>>, CliError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let items = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be a list")))?;
    Ok(Some(
        items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| item.to_string())
            })
            .collect(),
    ))
}

fn parse_qp(raw: &Value) -> Result<QuadraticProgram, CliError> {
    let c = numbers(get_any(raw, &["c"]), "c")?;
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    Ok(QuadraticProgram {
        q: square_matrix_or_zero(get_any(raw, &["Q", "q"]), "Q", n)?,
        c,
        a_ub: Some(matrix(get_any(raw, &["A_ub", "a_ub"]), "A_ub", Some(n))?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub"]), "b_ub")?),
        a_eq: Some(matrix(get_any(raw, &["A_eq", "a_eq"]), "A_eq", Some(n))?),
        b_eq: Some(optional_numbers(get_any(raw, &["b_eq"]), "b_eq")?),
        lb: optional_bounds(get_any(raw, &["lb"]), "lb", n, Some(Some(0.0)))?,
        ub: optional_bounds(get_any(raw, &["ub"]), "ub", n, Some(None))?,
        var_names: string_list(get_any(raw, &["var_names", "varNames"]), "var_names")?,
    })
}

fn parse_socp(raw: &Value) -> Result<SecondOrderConeProgram, CliError> {
    let c = numbers(get_any(raw, &["c"]), "c")?;
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    let cone_items = get_any(raw, &["cones"])
        .and_then(Value::as_array)
        .ok_or_else(|| CliError("cones must be a list".to_string()))?;
    let mut cones = Vec::with_capacity(cone_items.len());
    for (idx, item) in cone_items.iter().enumerate() {
        let a = matrix(
            get_any(item, &["A", "a"]),
            &format!("cones[{idx}].A"),
            Some(n),
        )?;
        let b = optional_numbers(get_any(item, &["b"]), &format!("cones[{idx}].b"))?;
        let cone_c = numbers(get_any(item, &["c"]), &format!("cones[{idx}].c"))?;
        let d = finite_number(
            get_any(item, &["d"]).ok_or_else(|| CliError(format!("cones[{idx}].d missing")))?,
            format!("cones[{idx}].d"),
        )?;
        cones.push(SecondOrderCone {
            a,
            b,
            c: cone_c,
            d,
            name: get_any(item, &["name"])
                .and_then(Value::as_str)
                .map(str::to_string),
        });
    }
    Ok(SecondOrderConeProgram {
        c,
        a_ub: Some(matrix(get_any(raw, &["A_ub", "a_ub"]), "A_ub", Some(n))?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub"]), "b_ub")?),
        a_eq: Some(matrix(get_any(raw, &["A_eq", "a_eq"]), "A_eq", Some(n))?),
        b_eq: Some(optional_numbers(get_any(raw, &["b_eq"]), "b_eq")?),
        lb: optional_bounds(get_any(raw, &["lb"]), "lb", n, Some(None))?,
        ub: optional_bounds(get_any(raw, &["ub"]), "ub", n, Some(None))?,
        cones,
        var_names: string_list(get_any(raw, &["var_names", "varNames"]), "var_names")?,
    })
}

fn parse_qcp(raw: &Value) -> Result<QuadraticallyConstrainedProgram, CliError> {
    let c = numbers(get_any(raw, &["c"]), "c")?;
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    let raw_constraints = get_any(raw, &["quadratic_constraints", "q_constraints"])
        .and_then(Value::as_array)
        .ok_or_else(|| CliError("quadratic_constraints must be a list".to_string()))?;
    let mut quadratic_constraints = Vec::with_capacity(raw_constraints.len());
    for (idx, item) in raw_constraints.iter().enumerate() {
        quadratic_constraints.push(QuadraticConstraint {
            q: square_matrix_or_zero(
                get_any(item, &["Q", "q"]),
                &format!("quadratic_constraints[{idx}].Q"),
                n,
            )?,
            c: numbers(
                get_any(item, &["c"]),
                &format!("quadratic_constraints[{idx}].c"),
            )?,
            rhs: finite_number(
                get_any(item, &["rhs"])
                    .ok_or_else(|| CliError(format!("quadratic_constraints[{idx}].rhs missing")))?,
                format!("quadratic_constraints[{idx}].rhs"),
            )?,
            name: get_any(item, &["name"])
                .and_then(Value::as_str)
                .map(str::to_string),
        });
    }
    Ok(QuadraticallyConstrainedProgram {
        q: square_matrix_or_zero(get_any(raw, &["Q", "q"]), "Q", n)?,
        c,
        a_ub: Some(matrix(get_any(raw, &["A_ub", "a_ub"]), "A_ub", Some(n))?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub"]), "b_ub")?),
        a_eq: Some(matrix(get_any(raw, &["A_eq", "a_eq"]), "A_eq", Some(n))?),
        b_eq: Some(optional_numbers(get_any(raw, &["b_eq"]), "b_eq")?),
        lb: optional_bounds(get_any(raw, &["lb"]), "lb", n, Some(None))?,
        ub: optional_bounds(get_any(raw, &["ub"]), "ub", n, Some(None))?,
        quadratic_constraints,
        var_names: string_list(get_any(raw, &["var_names", "varNames"]), "var_names")?,
    })
}

fn solution_json(solution: &ExternalQuadraticReferenceSolution) -> Value {
    let mut out = Map::new();
    out.insert("status".to_string(), json!(solution.status.as_str()));
    out.insert("solver".to_string(), json!(solution.solver));
    out.insert("x".to_string(), json!(solution.x));
    out.insert("objective".to_string(), json!(solution.objective));
    out.insert("message".to_string(), json!(solution.message));
    out.insert("elapsedMs".to_string(), json!(solution.elapsed_ms));
    if let Some(value) = &solution.dual_ub {
        out.insert("dualUB".to_string(), json!(value));
    }
    if let Some(value) = &solution.dual_eq {
        out.insert("dualEQ".to_string(), json!(value));
    }
    if let Some(value) = &solution.dual_lower_bounds {
        out.insert("dualLowerBounds".to_string(), json!(value));
    }
    if let Some(value) = &solution.dual_upper_bounds {
        out.insert("dualUpperBounds".to_string(), json!(value));
    }
    if let Some(value) = &solution.reduced_gradient {
        out.insert("reducedGradient".to_string(), json!(value));
    }
    if let Some(value) = solution.iterations {
        out.insert("iterations".to_string(), json!(value));
    }
    if let Some(value) = solution.enumerated {
        out.insert("enumerated".to_string(), json!(value));
    }
    Value::Object(out)
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": ExternalQuadraticReferenceStatus::NumericalError.as_str(),
        "solver": "rust:quadratic-reference",
        "x": [],
        "objective": null,
        "message": message.into(),
    })
}

fn solve(raw: &Value, args: &Args) -> Result<ExternalQuadraticReferenceSolution, CliError> {
    let source = raw.get("problem").unwrap_or(raw);
    let opts = ExternalQuadraticReferenceOptions {
        solver: args.solver,
        max_enumerations: args.max_enumerations,
    };
    let has_integer = get_any(source, &["integer_vars"])
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    let has_cones = get_any(source, &["cones"])
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    let has_qcp = get_any(source, &["quadratic_constraints", "q_constraints"])
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());

    if has_integer && has_cones {
        let socp = parse_socp(source)?;
        let integer_vars = bools(get_any(source, &["integer_vars"]), "integer_vars")?;
        return Ok(solve_misocp_with_external_reference(
            &MixedIntegerSecondOrderConeProgram { socp, integer_vars },
            &opts,
        ));
    }
    if has_integer && has_qcp {
        let qcp = parse_qcp(source)?;
        let integer_vars = bools(get_any(source, &["integer_vars"]), "integer_vars")?;
        return Ok(solve_miqcp_with_external_reference(
            &MixedIntegerQuadraticallyConstrainedProgram { qcp, integer_vars },
            &opts,
        ));
    }
    if has_integer {
        let qp = parse_qp(source)?;
        let integer_vars = bools(get_any(source, &["integer_vars"]), "integer_vars")?;
        return Ok(solve_miqp_with_external_reference(
            &MixedIntegerQuadraticProgram { qp, integer_vars },
            &opts,
        ));
    }
    if has_cones {
        return Ok(solve_socp_with_external_reference(
            &parse_socp(source)?,
            &opts,
        ));
    }
    if has_qcp {
        return Ok(solve_qcp_with_external_reference(
            &parse_qcp(source)?,
            &opts,
        ));
    }
    Ok(solve_qp_with_external_reference(&parse_qp(source)?, &opts))
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "qp_reference".to_string());
    let args = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    Ok(solution_json(&solve(&payload, &args)?))
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
                    .unwrap_or("qp_reference")
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
                serde_json::to_string(&output).expect("serialize quadratic reference output")
            );
            if output["status"] == ExternalQuadraticReferenceStatus::Unavailable.as_str() {
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

    const QP_SAMPLE: &str = r#"{
        "Q": [[2.0, 0.0], [0.0, 2.0]],
        "c": [-2.0, -4.0],
        "A_eq": [[1.0, 1.0]],
        "b_eq": [1.0],
        "lb": [0.0, 0.0]
    }"#;

    const MIQP_SAMPLE: &str = r#"{
        "Q": [[2.0, 0.0], [0.0, 2.0]],
        "c": [-2.8, -1.2],
        "A_ub": [[-1.0, -1.0]],
        "b_ub": [-1.5],
        "lb": [0.0, 0.0],
        "ub": [3.0, 3.0],
        "integer_vars": [true, false]
    }"#;

    const SOCP_SAMPLE: &str = r#"{
        "c": [1.0, 0.0],
        "lb": [-2.0, -2.0],
        "ub": [2.0, 2.0],
        "cones": [
            {"A": [[0.0, 1.0]], "b": [0.0], "c": [0.0, 0.0], "d": 1.0}
        ]
    }"#;

    const QCP_SAMPLE: &str = r#"{
        "Q": [[0.0, 0.0], [0.0, 0.0]],
        "c": [-1.0, 0.0],
        "lb": [-2.0, -2.0],
        "ub": [2.0, 2.0],
        "quadratic_constraints": [
            {"Q": [[1.0, 0.0], [0.0, 1.0]], "c": [0.0, 0.0], "rhs": 1.0}
        ]
    }"#;

    #[test]
    fn rust_qp_cli_solves_active_set_reference() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver".to_string(),
                "fallback".to_string(),
            ],
            QP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:qp-active-set");
        assert!(output["objective"].as_f64().is_some());
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }

    #[test]
    fn rust_qp_cli_treats_null_optional_blocks_as_empty() {
        let output = run(
            vec!["qp_reference".to_string(), "--solver=fallback".to_string()],
            r#"{
                "Q": [[2.0, 0.5], [0.5, 2.0]],
                "c": [-5.0, -6.0],
                "A_ub": [[1.0, 1.0]],
                "b_ub": [3.0],
                "A_eq": null,
                "b_eq": null,
                "lb": [0.0, 0.0],
                "ub": [4.0, 4.0],
                "var_names": null
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:qp-active-set");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }

    #[test]
    fn rust_miqp_cli_enumerates_integer_domain() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver=rust".to_string(),
                "--max-enumerations=100".to_string(),
            ],
            MIQP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:miqp-enumeration");
        assert_eq!(output["enumerated"], 4);
    }

    #[test]
    fn rust_socp_cli_uses_pattern_search() {
        let output = run(
            vec!["qp_reference".to_string(), "--solver=fallback".to_string()],
            SOCP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:socp-pattern-search");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }

    #[test]
    fn rust_socp_cli_treats_null_optional_blocks_as_empty() {
        let output = run(
            vec!["qp_reference".to_string(), "--solver=fallback".to_string()],
            r#"{
                "c": [1.0, 0.0],
                "A_ub": null,
                "b_ub": null,
                "A_eq": null,
                "b_eq": null,
                "lb": [-2.0, -2.0],
                "ub": [2.0, 2.0],
                "cones": [
                    {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0], "c": [0.0, 0.0], "d": 1.0}
                ],
                "var_names": null
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:socp-pattern-search");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }

    #[test]
    fn rust_qcp_cli_uses_pattern_search() {
        let output = run(
            vec!["qp_reference".to_string(), "--solver=fallback".to_string()],
            QCP_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:qcp-pattern-search");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }
}
