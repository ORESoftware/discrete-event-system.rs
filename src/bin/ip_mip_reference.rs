use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use des_engine::des::general::lp::{
    solve_lp_internal, InternalSimplexOptions, LPProblem, LPStatus, Sense,
};
use serde_json::{json, Value};

const EPS: f64 = 1.0e-9;

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
    problem: Option<PathBuf>,
    out: Option<PathBuf>,
    solver: String,
    max_enumerations: usize,
    pool_size: Option<usize>,
}

#[derive(Clone, Debug)]
struct Problem {
    sense: Sense,
    c: Vec<f64>,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    integer_vars: Vec<bool>,
    lb: Vec<f64>,
    ub: Vec<Option<f64>>,
}

#[derive(Clone, Debug)]
struct RemainderSolution {
    x: Vec<f64>,
    objective: f64,
}

#[derive(Clone, Debug)]
struct EnumerationResult {
    status: &'static str,
    solver: &'static str,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    message: String,
    enumerated: usize,
    solutions: Option<Vec<RemainderSolution>>,
    exhausted: Option<bool>,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--problem PATH] [--out PATH] [--solver auto|brute-force|enumeration|rust-enumeration] [--max-enumerations N] [--pool-size N]"
    )
}

fn parse_args(program: &str, args: impl IntoIterator<Item = String>) -> Result<Args, CliError> {
    let mut parsed = Args {
        problem: None,
        out: None,
        solver: "auto".to_string(),
        max_enumerations: 1_000_000,
        pool_size: None,
    };
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
            "--problem" => {
                parsed.problem = Some(PathBuf::from(next_option_value(
                    program,
                    "--problem",
                    inline_value,
                    &mut values,
                )?));
            }
            "--out" => {
                parsed.out = Some(PathBuf::from(next_option_value(
                    program,
                    "--out",
                    inline_value,
                    &mut values,
                )?));
            }
            "--solver" => {
                parsed.solver = next_option_value(program, "--solver", inline_value, &mut values)?;
            }
            "--max-enumerations" => {
                let value =
                    next_option_value(program, "--max-enumerations", inline_value, &mut values)?;
                parsed.max_enumerations = value.parse::<usize>().map_err(|err| {
                    CliError(format!("--max-enumerations must be an integer: {err}"))
                })?;
            }
            "--pool-size" => {
                let value = next_option_value(program, "--pool-size", inline_value, &mut values)?;
                parsed.pool_size = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| CliError(format!("--pool-size must be an integer: {err}")))?
                        .max(1),
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

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut std::iter::Peekable<impl Iterator<Item = String>>,
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

fn normalized_solver_label(solver: &str) -> String {
    let mut normalized = solver.trim().to_ascii_lowercase().replace('_', "-");
    if normalized.ends_with(":default") {
        normalized.truncate(normalized.len() - ":default".len());
    }
    normalized
}

fn rust_reference_solver_alias_supported(solver: &str) -> bool {
    matches!(
        normalized_solver_label(solver).as_str(),
        "auto"
            | "brute-force"
            | "enumeration"
            | "fallback"
            | "rust"
            | "rust-enumeration"
            | "rust-fallback"
            | "rust:fallback"
            | "rust-internal"
            | "rust:internal"
            | "milp"
            | "highs"
            | "highs-cli"
            | "highs:mip"
            | "highs:cli"
            | "cbc"
            | "cbc-cli"
            | "cbc:mip"
            | "cbc:cli"
            | "glpk"
            | "glpk-cli"
            | "glpk:mip"
            | "glpk:cli"
            | "scip"
            | "scip-cli"
            | "scip:mip"
            | "scip:cli"
            | "gurobi"
            | "gurobi-cli"
            | "gurobi:mip"
            | "gurobi:cli"
            | "cplex"
            | "cplex-cli"
            | "cplex:mip"
            | "cplex:cli"
            | "xpress"
            | "xpress-cli"
            | "xpress:mip"
            | "xpress:cli"
            | "fico-xpress"
            | "fico:xpress"
            | "lindo"
            | "lindo-cli"
            | "lindo:mip"
            | "lindo:cli"
            | "mosek"
            | "mosek-cli"
            | "mosek:mip"
            | "mosek:cli"
            | "copt"
            | "copt-cli"
            | "copt:mip"
            | "copt:cli"
            | "scipy"
            | "scipy-milp"
            | "scipy:milp"
            | "ortools"
            | "ortools-scip"
            | "ortools:scip"
            | "ortools-cp-sat"
            | "ortools:cp-sat"
            | "ortools:cpsat"
            | "cp-sat"
            | "cpsat"
    )
}

fn number(value: &Value, message: impl Into<String>) -> Result<f64, CliError> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        .ok_or_else(|| CliError(message.into()))
}

fn usize_value(value: &Value, message: impl Into<String>) -> Result<usize, CliError> {
    let message = message.into();
    if let Some(raw) = value.as_u64() {
        return usize::try_from(raw).map_err(|_| CliError(message));
    }
    if let Some(raw) = value.as_i64() {
        return usize::try_from(raw).map_err(|_| CliError(message));
    }
    value
        .as_str()
        .and_then(|text| text.parse::<usize>().ok())
        .ok_or_else(|| CliError(message))
}

fn number_array(value: &Value, field: &str) -> Result<Vec<f64>, CliError> {
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{field} must be an array")))?
        .iter()
        .enumerate()
        .map(|(idx, value)| number(value, format!("{field}[{idx}] must be numeric")))
        .collect()
}

fn number_matrix(value: &Value, field: &str) -> Result<Vec<Vec<f64>>, CliError> {
    value
        .as_array()
        .ok_or_else(|| CliError(format!("{field} must be an array")))?
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            row.as_array()
                .ok_or_else(|| CliError(format!("{field}[{row_idx}] must be an array")))?
                .iter()
                .enumerate()
                .map(|(col_idx, value)| {
                    number(
                        value,
                        format!("{field}[{row_idx}][{col_idx}] must be numeric"),
                    )
                })
                .collect()
        })
        .collect()
}

fn optional_bound_array(
    raw: Option<&Value>,
    n: usize,
    default_lb: bool,
) -> Result<Vec<Option<f64>>, CliError> {
    let Some(raw) = raw else {
        return Ok(if default_lb {
            vec![Some(0.0); n]
        } else {
            vec![None; n]
        });
    };
    let values = raw
        .as_array()
        .ok_or_else(|| CliError("bound vector must be an array".to_string()))?;
    if values.len() != n {
        return Err(CliError("bound vector length mismatch".to_string()));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if value.is_null() {
                Ok(None)
            } else {
                Ok(Some(number(
                    value,
                    format!("bound[{idx}] must be numeric"),
                )?))
            }
        })
        .collect()
}

fn append_linear_constraints(
    p: &Value,
    n: usize,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("linear_constraints") else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("linear_constraints must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let row = number_array(
            constraint.get("coefs").ok_or_else(|| {
                CliError(format!("linear constraint {idx} coefs must be an array"))
            })?,
            &format!("linear_constraints[{idx}].coefs"),
        )?;
        if row.len() != n {
            return Err(CliError(format!(
                "linear constraint {idx} coefficient length does not match variable count"
            )));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(CliError(format!(
                "linear constraint {idx} coefficients must be finite"
            )));
        }
        let lower = constraint
            .get("lower")
            .filter(|value| !value.is_null())
            .map(|value| {
                number(
                    value,
                    format!("linear constraint {idx} lower bound must be numeric"),
                )
            })
            .transpose()?;
        let upper = constraint
            .get("upper")
            .filter(|value| !value.is_null())
            .map(|value| {
                number(
                    value,
                    format!("linear constraint {idx} upper bound must be numeric"),
                )
            })
            .transpose()?;
        if lower.is_none() && upper.is_none() {
            return Err(CliError(format!(
                "linear constraint {idx} needs at least one bound"
            )));
        }
        if lower.is_some_and(|value| !value.is_finite()) {
            return Err(CliError(format!(
                "linear constraint {idx} lower bound must be finite"
            )));
        }
        if upper.is_some_and(|value| !value.is_finite()) {
            return Err(CliError(format!(
                "linear constraint {idx} upper bound must be finite"
            )));
        }
        if let (Some(lower), Some(upper)) = (lower, upper) {
            if lower > upper + EPS {
                return Err(CliError(format!(
                    "linear constraint {idx} lower bound exceeds upper bound"
                )));
            }
        }
        if let Some(upper) = upper {
            a.push(row.clone());
            b.push(upper);
        }
        if let Some(lower) = lower {
            a.push(row.iter().map(|value| -value).collect());
            b.push(-lower);
        }
    }
    Ok(())
}

fn append_dense_equalities(
    p: &Value,
    n: usize,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_a_eq) = p.get("A_eq").or_else(|| p.get("a_eq")) else {
        return Ok(());
    };
    let rows = number_matrix(raw_a_eq, "A_eq")?;
    let rhs = number_array(
        p.get("b_eq")
            .ok_or_else(|| CliError("b_eq must be an array when A_eq is provided".to_string()))?,
        "b_eq",
    )?;
    if rows.len() != rhs.len() {
        return Err(CliError("A_eq/b_eq length mismatch".to_string()));
    }
    for (idx, (row, rhs)) in rows.into_iter().zip(rhs).enumerate() {
        if row.len() != n {
            return Err(CliError(format!(
                "A_eq row {idx} coefficient length does not match variable count"
            )));
        }
        if row.iter().any(|value| !value.is_finite()) || !rhs.is_finite() {
            return Err(CliError(format!(
                "A_eq row {idx} coefficients and rhs must be finite"
            )));
        }
        append_equality(a, b, row, rhs);
    }
    Ok(())
}

fn append_lazy_constraints(
    p: &Value,
    n: usize,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p
        .get("lazy_constraints")
        .or_else(|| p.get("lazyConstraints"))
    else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("lazy_constraints must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let row = number_array(
            constraint
                .get("coefs")
                .or_else(|| constraint.get("coefficients"))
                .ok_or_else(|| CliError(format!("lazy constraint {idx} coefs must be an array")))?,
            &format!("lazy_constraints[{idx}].coefs"),
        )?;
        if row.len() != n {
            return Err(CliError(format!(
                "lazy constraint {idx} coefficient length does not match variable count"
            )));
        }
        let rhs = number(
            constraint
                .get("rhs")
                .or_else(|| constraint.get("upper"))
                .ok_or_else(|| CliError(format!("lazy constraint {idx} rhs must be numeric")))?,
            format!("lazy constraint {idx} rhs must be numeric"),
        )?;
        if row.iter().any(|value| !value.is_finite()) || !rhs.is_finite() {
            return Err(CliError(format!(
                "lazy constraint {idx} coefficients and rhs must be finite"
            )));
        }
        a.push(row);
        b.push(rhs);
    }
    Ok(())
}

fn variable_is_binary(idx: usize, integer_vars: &[bool], lb: &[f64], ub: &[Option<f64>]) -> bool {
    integer_vars[idx] && lb[idx].abs() <= 1.0e-12 && ub[idx].is_some_and(|upper| upper <= 1.0 + EPS)
}

fn finite_product_factor_bounds(
    var: usize,
    idx: usize,
    lb: &[f64],
    ub: &[Option<f64>],
) -> Result<(f64, f64), CliError> {
    let lower = lb[var];
    if !lower.is_finite() {
        return Err(CliError(format!(
            "product {idx} continuous factor {var} lower bound must be finite"
        )));
    }
    let upper = ub[var].ok_or_else(|| {
        CliError(format!(
            "product {idx} continuous factor {var} needs a finite upper bound"
        ))
    })?;
    if !upper.is_finite() {
        return Err(CliError(format!(
            "product {idx} continuous factor {var} needs a finite upper bound"
        )));
    }
    if upper + EPS < lower {
        return Err(CliError(format!(
            "product {idx} continuous factor {var} lower bound exceeds upper bound"
        )));
    }
    Ok((lower, upper))
}

fn append_product_linearization(
    idx: usize,
    target: usize,
    x_var: usize,
    y_var: usize,
    integer_vars: &[bool],
    lb: &[f64],
    ub: &[Option<f64>],
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let n = integer_vars.len();
    if target == x_var || target == y_var {
        return Err(CliError(format!(
            "product {idx} target_var must be distinct from factor variables"
        )));
    }
    if x_var == y_var {
        return Err(CliError(format!(
            "product {idx} x_var and y_var must be distinct"
        )));
    }
    if !lb[target].is_finite() {
        return Err(CliError(format!(
            "product {idx} target lower bound must be finite"
        )));
    }

    let x_binary = variable_is_binary(x_var, integer_vars, lb, ub);
    let y_binary = variable_is_binary(y_var, integer_vars, lb, ub);
    if !x_binary && !y_binary {
        return Err(CliError(format!(
            "product {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
        )));
    }

    if x_binary && y_binary {
        let mut row = vec![0.0; n];
        row[target] = 1.0;
        row[x_var] = -1.0;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; n];
        row[target] = 1.0;
        row[y_var] = -1.0;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; n];
        row[target] = -1.0;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; n];
        row[target] = -1.0;
        row[x_var] = 1.0;
        row[y_var] = 1.0;
        a.push(row);
        b.push(1.0);
        return Ok(());
    }

    let binary = if x_binary { x_var } else { y_var };
    let continuous = if x_binary { y_var } else { x_var };
    let (lower, upper) = finite_product_factor_bounds(continuous, idx, lb, ub)?;

    let mut row = vec![0.0; n];
    row[target] = 1.0;
    row[binary] = -upper;
    a.push(row);
    b.push(0.0);

    let mut row = vec![0.0; n];
    row[target] = -1.0;
    row[binary] = lower;
    a.push(row);
    b.push(0.0);

    let mut row = vec![0.0; n];
    row[target] = 1.0;
    row[continuous] = -1.0;
    row[binary] = -lower;
    a.push(row);
    b.push(-lower);

    let mut row = vec![0.0; n];
    row[target] = -1.0;
    row[continuous] = 1.0;
    row[binary] = upper;
    a.push(row);
    b.push(upper);
    Ok(())
}

fn append_product_constraints(
    p: &Value,
    integer_vars: &[bool],
    lb: &[f64],
    ub: &[Option<f64>],
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("products").or_else(|| p.get("product_constraints")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("products must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("product {idx} target_var is required")))?,
            format!("product {idx} target_var must be a nonnegative integer"),
        )?;
        let x_var = usize_value(
            constraint
                .get("x_var")
                .ok_or_else(|| CliError(format!("product {idx} x_var is required")))?,
            format!("product {idx} x_var must be a nonnegative integer"),
        )?;
        let y_var = usize_value(
            constraint
                .get("y_var")
                .ok_or_else(|| CliError(format!("product {idx} y_var is required")))?,
            format!("product {idx} y_var must be a nonnegative integer"),
        )?;
        for (role, var) in [("target_var", target), ("x_var", x_var), ("y_var", y_var)] {
            if var >= integer_vars.len() {
                return Err(CliError(format!("product {idx} {role} out of range")));
            }
        }
        append_product_linearization(idx, target, x_var, y_var, integer_vars, lb, ub, a, b)?;
    }
    Ok(())
}

fn max_lhs_over_bounds(
    row: &[f64],
    rhs: f64,
    lb: &[f64],
    ub: &[Option<f64>],
    name: &str,
) -> Result<f64, CliError> {
    let mut max_lhs = 0.0;
    for (idx, &coeff) in row.iter().enumerate() {
        if !coeff.is_finite() {
            return Err(CliError(format!(
                "indicator {name} coefficients must be finite"
            )));
        }
        if coeff > 0.0 {
            let upper = ub[idx].ok_or_else(|| {
                CliError(format!(
                    "indicator {name} needs a finite upper bound for variable x{idx}"
                ))
            })?;
            if !upper.is_finite() {
                return Err(CliError(format!(
                    "indicator {name} needs a finite upper bound for variable x{idx}"
                )));
            }
            max_lhs += coeff * upper;
        } else if coeff < 0.0 {
            if !lb[idx].is_finite() {
                return Err(CliError(format!(
                    "indicator {name} needs a finite lower bound for variable x{idx}"
                )));
            }
            max_lhs += coeff * lb[idx];
        }
    }
    Ok((max_lhs - rhs).max(0.0))
}

struct IndicatorLeRow<'a> {
    indicator_idx: usize,
    row: Vec<f64>,
    rhs: f64,
    binary_var: usize,
    active_value: bool,
    name: &'a str,
}

fn append_indicator_le_row(
    spec: IndicatorLeRow<'_>,
    lb: &[f64],
    ub: &[Option<f64>],
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    if !spec.rhs.is_finite() {
        return Err(CliError(format!(
            "indicator {} rhs must be finite",
            spec.indicator_idx
        )));
    }
    let m = max_lhs_over_bounds(&spec.row, spec.rhs, lb, ub, spec.name)?;
    let mut compiled = spec.row;
    let compiled_rhs = if spec.active_value {
        compiled[spec.binary_var] += m;
        spec.rhs + m
    } else {
        compiled[spec.binary_var] -= m;
        spec.rhs
    };
    a.push(compiled);
    b.push(compiled_rhs);
    Ok(())
}

fn append_indicators(
    p: &Value,
    n: usize,
    integer_vars: &[bool],
    lb: &[f64],
    ub: &[Option<f64>],
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_indicators) = p.get("indicators") else {
        return Ok(());
    };
    let indicators = raw_indicators
        .as_array()
        .ok_or_else(|| CliError("indicators must be an array".to_string()))?;
    for (idx, indicator) in indicators.iter().enumerate() {
        let binary_var = usize_value(
            indicator
                .get("binary_var")
                .ok_or_else(|| CliError(format!("indicator {idx} binary_var is required")))?,
            format!("indicator {idx} binary_var must be a nonnegative integer"),
        )?;
        if binary_var >= n {
            return Err(CliError(format!("indicator {idx} binary_var out of range")));
        }
        if !integer_vars[binary_var] {
            return Err(CliError(format!(
                "indicator {idx} trigger variable must be integer/binary"
            )));
        }
        let trigger_upper = ub[binary_var].ok_or_else(|| {
            CliError(format!(
                "indicator {idx} trigger variable must have finite binary upper bound <= 1"
            ))
        })?;
        if !trigger_upper.is_finite() || trigger_upper > 1.0 + EPS {
            return Err(CliError(format!(
                "indicator {idx} trigger variable must have finite binary upper bound <= 1"
            )));
        }
        let row = number_array(
            indicator
                .get("coefs")
                .ok_or_else(|| CliError(format!("indicator {idx} coefs must be an array")))?,
            &format!("indicators[{idx}].coefs"),
        )?;
        if row.len() != n {
            return Err(CliError(format!(
                "indicator {idx} coefficient length does not match variable count"
            )));
        }
        let rhs = number(
            indicator
                .get("rhs")
                .ok_or_else(|| CliError(format!("indicator {idx} rhs is required")))?,
            format!("indicator {idx} rhs must be numeric"),
        )?;
        let active_value = indicator
            .get("active_value")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let name = indicator
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("indicator");
        match indicator
            .get("sense")
            .and_then(Value::as_str)
            .unwrap_or("le")
        {
            "le" => append_indicator_le_row(
                IndicatorLeRow {
                    indicator_idx: idx,
                    row,
                    rhs,
                    binary_var,
                    active_value,
                    name,
                },
                lb,
                ub,
                a,
                b,
            )?,
            "ge" => append_indicator_le_row(
                IndicatorLeRow {
                    indicator_idx: idx,
                    row: row.iter().map(|value| -value).collect(),
                    rhs: -rhs,
                    binary_var,
                    active_value,
                    name,
                },
                lb,
                ub,
                a,
                b,
            )?,
            "eq" => {
                append_indicator_le_row(
                    IndicatorLeRow {
                        indicator_idx: idx,
                        row: row.clone(),
                        rhs,
                        binary_var,
                        active_value,
                        name,
                    },
                    lb,
                    ub,
                    a,
                    b,
                )?;
                append_indicator_le_row(
                    IndicatorLeRow {
                        indicator_idx: idx,
                        row: row.iter().map(|value| -value).collect(),
                        rhs: -rhs,
                        binary_var,
                        active_value,
                        name,
                    },
                    lb,
                    ub,
                    a,
                    b,
                )?;
            }
            other => {
                return Err(CliError(format!(
                    "indicator {idx} has unknown sense {other}"
                )));
            }
        }
    }
    Ok(())
}

fn append_equality(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, row: Vec<f64>, rhs: f64) {
    a.push(row.clone());
    b.push(rhs);
    a.push(row.iter().map(|value| -value).collect());
    b.push(-rhs);
}

fn append_binary_helper(
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
) -> usize {
    for row in a.iter_mut() {
        row.push(0.0);
    }
    let idx = c.len();
    c.push(0.0);
    integer_vars.push(true);
    lb.push(0.0);
    ub.push(Some(1.0));
    idx
}

fn append_continuous_helper(
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    upper: Option<f64>,
) -> usize {
    for row in a.iter_mut() {
        row.push(0.0);
    }
    let idx = c.len();
    c.push(0.0);
    integer_vars.push(false);
    lb.push(0.0);
    ub.push(upper.filter(|value| value.is_finite()));
    idx
}

fn append_bounded_continuous_helper(
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    lower: f64,
    upper: f64,
) -> Result<usize, CliError> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(CliError(
            "bounded helper lower and upper bounds must be finite".to_string(),
        ));
    }
    if upper + EPS < lower {
        return Err(CliError(
            "bounded helper lower bound exceeds upper bound".to_string(),
        ));
    }
    for row in a.iter_mut() {
        row.push(0.0);
    }
    let idx = c.len();
    c.push(0.0);
    integer_vars.push(false);
    lb.push(lower);
    ub.push(Some(upper));
    Ok(idx)
}

fn append_quadratic_objective_terms(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_terms) = p
        .get("quadratic_objective")
        .or_else(|| p.get("quadratic_objective_terms"))
    else {
        return Ok(());
    };
    let terms = raw_terms
        .as_array()
        .ok_or_else(|| CliError("quadratic objective terms must be an array".to_string()))?;
    for (idx, term) in terms.iter().enumerate() {
        let x_var = usize_value(
            term.get("x_var").ok_or_else(|| {
                CliError(format!("quadratic objective term {idx} x_var is required"))
            })?,
            format!("quadratic objective term {idx} x_var must be a nonnegative integer"),
        )?;
        let y_var = usize_value(
            term.get("y_var").ok_or_else(|| {
                CliError(format!("quadratic objective term {idx} y_var is required"))
            })?,
            format!("quadratic objective term {idx} y_var must be a nonnegative integer"),
        )?;
        for (role, var) in [("x_var", x_var), ("y_var", y_var)] {
            if var >= c.len() {
                return Err(CliError(format!(
                    "quadratic objective term {idx} {role} out of range"
                )));
            }
        }
        let coeff = number(
            term.get("coeff").ok_or_else(|| {
                CliError(format!("quadratic objective term {idx} coeff is required"))
            })?,
            format!("quadratic objective term {idx} coeff must be numeric"),
        )?;
        if !coeff.is_finite() {
            return Err(CliError(format!(
                "quadratic objective term {idx} coeff must be finite"
            )));
        }

        let x_binary = variable_is_binary(x_var, integer_vars, lb, ub);
        let y_binary = variable_is_binary(y_var, integer_vars, lb, ub);
        if x_var == y_var {
            if !x_binary {
                return Err(CliError(format!(
                    "quadratic objective term {idx} square is exact only for binary variables"
                )));
            }
            c[x_var] += coeff;
            continue;
        }
        if !x_binary && !y_binary {
            return Err(CliError(format!(
                "quadratic objective term {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
            )));
        }

        let (product_lb, product_ub) = if x_binary && y_binary {
            (0.0, 1.0)
        } else {
            let continuous = if x_binary { y_var } else { x_var };
            let (lower, upper) = finite_product_factor_bounds(continuous, idx, lb, ub)?;
            (0.0_f64.min(lower), 0.0_f64.max(upper))
        };
        let helper =
            append_bounded_continuous_helper(c, integer_vars, lb, ub, a, product_lb, product_ub)?;
        c[helper] = coeff;
        append_product_linearization(idx, helper, x_var, y_var, integer_vars, lb, ub, a, b)?;
    }
    Ok(())
}

fn finite_sos_upper(ub: &[Option<f64>], var: usize, idx: usize) -> Result<f64, CliError> {
    let upper = ub[var].filter(|value| value.is_finite()).ok_or_else(|| {
        CliError(format!(
            "sos {idx} variable x{var} needs a finite upper bound"
        ))
    })?;
    if upper < 0.0 {
        return Err(CliError(format!(
            "sos {idx} variable x{var} has a negative upper bound"
        )));
    }
    Ok(upper)
}

fn sos_ordered_vars(sos: &Value, idx: usize) -> Result<Vec<usize>, CliError> {
    let raw_vars = sos
        .get("vars")
        .ok_or_else(|| CliError(format!("sos {idx} vars must be an array")))?
        .as_array()
        .ok_or_else(|| CliError(format!("sos {idx} vars must be an array")))?;
    let vars = raw_vars
        .iter()
        .map(|value| {
            usize_value(
                value,
                format!("sos {idx} variable must be a nonnegative integer"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some(raw_weights) = sos.get("weights") else {
        return Ok(vars);
    };
    let weights = number_array(raw_weights, &format!("sos {idx} weights"))?;
    if weights.len() != vars.len() {
        return Err(CliError(
            "sos weight length does not match variable count".to_string(),
        ));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(CliError(format!("sos {idx} weights must be finite")));
    }
    let mut pairs = weights.into_iter().zip(vars).collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for pair in pairs.windows(2) {
        if (pair[0].0 - pair[1].0).abs() <= 1.0e-12 {
            return Err(CliError("sos weights must be unique".to_string()));
        }
    }
    Ok(pairs.into_iter().map(|(_, var)| var).collect())
}

fn append_sos2_adjacency_constraints(
    idx: usize,
    ordered: &[usize],
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    for &var in ordered {
        finite_sos_upper(ub, var, idx)?;
    }
    if ordered.len() <= 2 {
        return Ok(());
    }
    let segments = (0..ordered.len() - 1)
        .map(|_| append_binary_helper(c, integer_vars, lb, ub, a))
        .collect::<Vec<_>>();
    let mut row = vec![0.0; c.len()];
    for &segment in &segments {
        row[segment] = 1.0;
    }
    a.push(row);
    b.push(1.0);
    for (pos, &var) in ordered.iter().enumerate() {
        let upper = finite_sos_upper(ub, var, idx)?;
        let mut row = vec![0.0; c.len()];
        row[var] = 1.0;
        if pos > 0 {
            row[segments[pos - 1]] -= upper;
        }
        if pos + 1 < ordered.len() {
            row[segments[pos]] -= upper;
        }
        a.push(row);
        b.push(0.0);
    }
    Ok(())
}

fn append_sos_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_sets) = p.get("sos") else {
        return Ok(());
    };
    let sets = raw_sets
        .as_array()
        .ok_or_else(|| CliError("sos must be an array".to_string()))?;
    for (idx, sos) in sets.iter().enumerate() {
        let ordered = sos_ordered_vars(sos, idx)?;
        if ordered.is_empty() {
            return Err(CliError(format!("sos {idx} has no variables")));
        }
        for (pos, &var) in ordered.iter().enumerate() {
            if var >= c.len() {
                return Err(CliError(format!("sos {idx} variable {var} out of range")));
            }
            if ordered[..pos].contains(&var) {
                return Err(CliError(format!("sos {idx} contains duplicate variables")));
            }
            if lb[var].abs() > 1.0e-12 {
                return Err(CliError(format!(
                    "sos {idx} variable x{var} must have lower bound 0"
                )));
            }
        }
        match sos.get("kind").and_then(Value::as_str).unwrap_or("sos1") {
            "sos1" => {
                let selectors = ordered
                    .iter()
                    .map(|_| append_binary_helper(c, integer_vars, lb, ub, a))
                    .collect::<Vec<_>>();
                for (pos, &var) in ordered.iter().enumerate() {
                    let upper = finite_sos_upper(ub, var, idx)?;
                    let mut row = vec![0.0; c.len()];
                    row[var] = 1.0;
                    row[selectors[pos]] = -upper;
                    a.push(row);
                    b.push(0.0);
                }
                let mut row = vec![0.0; c.len()];
                for selector in selectors {
                    row[selector] = 1.0;
                }
                a.push(row);
                b.push(1.0);
            }
            "sos2" => {
                append_sos2_adjacency_constraints(idx, &ordered, c, integer_vars, lb, ub, a, b)?;
            }
            other => return Err(CliError(format!("sos {idx} has unknown kind {other}"))),
        }
    }
    Ok(())
}

fn append_pwl_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("pwl") else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("pwl must be an array".to_string()))?;
    for (idx, pwl) in constraints.iter().enumerate() {
        let x_var = usize_value(
            pwl.get("x_var")
                .ok_or_else(|| CliError(format!("pwl {idx} x_var is required")))?,
            format!("pwl {idx} x_var must be a nonnegative integer"),
        )?;
        let y_var = usize_value(
            pwl.get("y_var")
                .ok_or_else(|| CliError(format!("pwl {idx} y_var is required")))?,
            format!("pwl {idx} y_var must be a nonnegative integer"),
        )?;
        if x_var >= c.len() {
            return Err(CliError(format!("pwl {idx} x_var out of range")));
        }
        if y_var >= c.len() {
            return Err(CliError(format!("pwl {idx} y_var out of range")));
        }
        if x_var == y_var {
            return Err(CliError(format!(
                "pwl {idx} x_var and y_var must be distinct"
            )));
        }
        if integer_vars[x_var] || integer_vars[y_var] {
            return Err(CliError(format!(
                "pwl {idx} x_var and y_var must be continuous"
            )));
        }
        let raw_points = pwl
            .get("points")
            .ok_or_else(|| CliError(format!("pwl {idx} points must be an array")))?
            .as_array()
            .ok_or_else(|| CliError(format!("pwl {idx} points must be an array")))?;
        if raw_points.len() < 2 {
            return Err(CliError(format!(
                "pwl {idx} needs at least two breakpoints"
            )));
        }
        let mut points = Vec::with_capacity(raw_points.len());
        for (pos, point) in raw_points.iter().enumerate() {
            let px = number(
                point
                    .get("x")
                    .ok_or_else(|| CliError(format!("pwl {idx} breakpoint {pos} x is required")))?,
                format!("pwl {idx} breakpoint {pos} x must be numeric"),
            )?;
            let py = number(
                point
                    .get("y")
                    .ok_or_else(|| CliError(format!("pwl {idx} breakpoint {pos} y is required")))?,
                format!("pwl {idx} breakpoint {pos} y must be numeric"),
            )?;
            if !px.is_finite() || !py.is_finite() {
                return Err(CliError(format!(
                    "pwl {idx} breakpoint {pos} must be finite"
                )));
            }
            if px < -1.0e-12 || py < -1.0e-12 {
                return Err(CliError(format!(
                    "pwl {idx} breakpoint {pos} must be non-negative"
                )));
            }
            if let Some((previous_x, _)) = points.last() {
                if px <= *previous_x + 1.0e-12 {
                    return Err(CliError(format!(
                        "pwl {idx} breakpoint x values must be strictly increasing"
                    )));
                }
            }
            points.push((px, py));
        }

        let lambdas = points
            .iter()
            .map(|_| append_continuous_helper(c, integer_vars, lb, ub, a, Some(1.0)))
            .collect::<Vec<_>>();

        let mut row = vec![0.0; c.len()];
        for &lambda in &lambdas {
            row[lambda] = 1.0;
        }
        append_equality(a, b, row, 1.0);

        let mut row = vec![0.0; c.len()];
        row[x_var] = 1.0;
        for (&lambda, &(px, _)) in lambdas.iter().zip(&points) {
            row[lambda] -= px;
        }
        append_equality(a, b, row, 0.0);

        let mut row = vec![0.0; c.len()];
        row[y_var] = 1.0;
        for (&lambda, &(_, py)) in lambdas.iter().zip(&points) {
            row[lambda] -= py;
        }
        append_equality(a, b, row, 0.0);

        append_sos2_adjacency_constraints(idx, &lambdas, c, integer_vars, lb, ub, a, b)?;
    }
    Ok(())
}

fn append_semi_variables(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_semis) = p.get("semi_variables") else {
        return Ok(());
    };
    let semis = raw_semis
        .as_array()
        .ok_or_else(|| CliError("semi_variables must be an array".to_string()))?;
    for (idx, semi) in semis.iter().enumerate() {
        let var = usize_value(
            semi.get("var")
                .ok_or_else(|| CliError(format!("semi variable {idx} var is required")))?,
            format!("semi variable {idx} var must be a nonnegative integer"),
        )?;
        if var >= c.len() {
            return Err(CliError(format!("semi variable {idx} index out of range")));
        }
        if lb[var].abs() > 1.0e-12 {
            return Err(CliError(format!(
                "semi variable {idx} expects ordinary lower bound 0"
            )));
        }
        let lower = number(
            semi.get("lower")
                .ok_or_else(|| CliError(format!("semi variable {idx} lower is required")))?,
            format!("semi variable {idx} lower bound must be numeric"),
        )?;
        if !lower.is_finite() || lower <= 0.0 {
            return Err(CliError(format!(
                "semi variable {idx} lower bound must be finite and positive"
            )));
        }
        let upper = ub[var]
            .ok_or_else(|| CliError(format!("semi variable {idx} needs a finite upper bound")))?;
        if !upper.is_finite() {
            return Err(CliError(format!(
                "semi variable {idx} needs a finite upper bound"
            )));
        }
        if upper + EPS < lower {
            return Err(CliError(format!(
                "semi variable {idx} lower bound exceeds upper bound"
            )));
        }
        match semi
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("semi_continuous")
        {
            "semi_integer" => integer_vars[var] = true,
            "semi_continuous" => integer_vars[var] = false,
            other => {
                return Err(CliError(format!(
                    "semi variable {idx} has unknown kind {other}"
                )));
            }
        }
        let active = append_binary_helper(c, integer_vars, lb, ub, a);

        let mut row = vec![0.0; c.len()];
        row[var] = 1.0;
        row[active] = -upper;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; c.len()];
        row[var] = -1.0;
        row[active] = lower;
        a.push(row);
        b.push(0.0);
    }
    Ok(())
}

fn append_abs_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("abs").or_else(|| p.get("absolute_values")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("abs constraints must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let arg = usize_value(
            constraint
                .get("arg_var")
                .ok_or_else(|| CliError(format!("abs {idx} arg_var is required")))?,
            format!("abs {idx} arg_var must be a nonnegative integer"),
        )?;
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("abs {idx} target_var is required")))?,
            format!("abs {idx} target_var must be a nonnegative integer"),
        )?;
        if arg >= c.len() {
            return Err(CliError(format!("abs {idx} arg_var out of range")));
        }
        if target >= c.len() {
            return Err(CliError(format!("abs {idx} target_var out of range")));
        }
        if arg == target {
            return Err(CliError(format!(
                "abs {idx} arg_var and target_var must be distinct"
            )));
        }
        let lower = lb[arg];
        if !lower.is_finite() {
            return Err(CliError(format!(
                "abs {idx} argument lower bound must be finite"
            )));
        }
        let upper = ub[arg].filter(|value| value.is_finite());
        if upper.is_some_and(|value| value + EPS < lower) {
            return Err(CliError(format!(
                "abs {idx} argument lower bound exceeds upper bound"
            )));
        }

        if lower >= -1.0e-12 {
            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            row[arg] = -1.0;
            append_equality(a, b, row, 0.0);
            continue;
        }
        if upper.is_some_and(|value| value <= 1.0e-12) {
            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            row[arg] = 1.0;
            append_equality(a, b, row, 0.0);
            continue;
        }
        let Some(upper) = upper else {
            return Err(CliError(format!(
                "abs {idx} mixed-sign argument needs a finite upper bound"
            )));
        };

        let selector = append_binary_helper(c, integer_vars, lb, ub, a);

        let mut row = vec![0.0; c.len()];
        row[arg] = 1.0;
        row[target] = -1.0;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; c.len()];
        row[arg] = -1.0;
        row[target] = -1.0;
        a.push(row);
        b.push(0.0);

        let mut row = vec![0.0; c.len()];
        row[target] = 1.0;
        row[arg] = -1.0;
        row[selector] = -2.0 * lower;
        a.push(row);
        b.push(-2.0 * lower);

        let mut row = vec![0.0; c.len()];
        row[target] = 1.0;
        row[arg] = 1.0;
        row[selector] = -2.0 * upper;
        a.push(row);
        b.push(0.0);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ExtremumCandidate {
    Var(usize),
    Constant(f64),
}

fn extremum_candidates(
    constraint: &Value,
    idx: usize,
    label: &str,
) -> Result<Vec<ExtremumCandidate>, CliError> {
    let mut candidates = Vec::new();
    if let Some(arg_vars) = constraint.get("arg_vars") {
        let args = arg_vars
            .as_array()
            .ok_or_else(|| CliError(format!("{label} {idx} arg_vars must be an array")))?;
        for (pos, value) in args.iter().enumerate() {
            candidates.push(ExtremumCandidate::Var(usize_value(
                value,
                format!("{label} {idx} argument variable {pos} must be a nonnegative integer"),
            )?));
        }
    }
    if let Some(value) = constraint.get("constant").filter(|value| !value.is_null()) {
        let constant = number(value, format!("{label} {idx} constant must be numeric"))?;
        if !constant.is_finite() {
            return Err(CliError(format!("{label} {idx} constant must be finite")));
        }
        candidates.push(ExtremumCandidate::Constant(constant));
    }
    if candidates.is_empty() {
        return Err(CliError(format!(
            "{label} {idx} needs at least one argument or a constant"
        )));
    }
    Ok(candidates)
}

fn validate_extremum_candidates(
    candidates: &[ExtremumCandidate],
    target: usize,
    variable_count: usize,
    label: &str,
    idx: usize,
    lb: &[f64],
) -> Result<(), CliError> {
    let mut seen = Vec::new();
    for candidate in candidates {
        if let ExtremumCandidate::Var(var) = *candidate {
            if var >= variable_count {
                return Err(CliError(format!(
                    "{label} {idx} argument variable out of range"
                )));
            }
            if var == target {
                return Err(CliError(format!(
                    "{label} {idx} target_var must be distinct from argument variables"
                )));
            }
            if seen.contains(&var) {
                return Err(CliError(format!(
                    "{label} {idx} duplicate argument variable {var}"
                )));
            }
            if !lb[var].is_finite() {
                return Err(CliError(format!(
                    "{label} {idx} argument variable {var} lower bound must be finite"
                )));
            }
            seen.push(var);
        }
    }
    Ok(())
}

fn candidate_lower(candidate: ExtremumCandidate, lb: &[f64]) -> f64 {
    match candidate {
        ExtremumCandidate::Var(var) => lb[var],
        ExtremumCandidate::Constant(value) => value,
    }
}

fn candidate_upper(candidate: ExtremumCandidate, ub: &[Option<f64>]) -> Option<f64> {
    match candidate {
        ExtremumCandidate::Var(var) => ub[var].filter(|value| value.is_finite()),
        ExtremumCandidate::Constant(value) => Some(value),
    }
}

fn append_extremum_single_candidate(
    candidate: ExtremumCandidate,
    target: usize,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    variable_count: usize,
) {
    let mut row = vec![0.0; variable_count];
    row[target] = 1.0;
    let rhs = match candidate {
        ExtremumCandidate::Var(var) => {
            row[var] = -1.0;
            0.0
        }
        ExtremumCandidate::Constant(value) => value,
    };
    append_equality(a, b, row, rhs);
}

fn append_maximum_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("maximums").or_else(|| p.get("max_constraints")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("maximums must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("maximum {idx} target_var is required")))?,
            format!("maximum {idx} target_var must be a nonnegative integer"),
        )?;
        if target >= c.len() {
            return Err(CliError(format!("maximum {idx} target_var out of range")));
        }
        let candidates = extremum_candidates(constraint, idx, "maximum")?;
        validate_extremum_candidates(&candidates, target, c.len(), "maximum", idx, lb)?;
        if candidates.len() == 1 {
            append_extremum_single_candidate(candidates[0], target, a, b, c.len());
            continue;
        }

        let max_upper = if let Some(target_upper) = ub[target].filter(|value| value.is_finite()) {
            target_upper
        } else {
            let mut max_upper = f64::NEG_INFINITY;
            for candidate in &candidates {
                let upper = candidate_upper(*candidate, ub).ok_or_else(|| {
                    CliError(format!(
                        "maximum {idx} needs finite argument uppers or a finite target upper bound"
                    ))
                })?;
                max_upper = max_upper.max(upper);
            }
            max_upper
        };
        if !max_upper.is_finite() {
            return Err(CliError(format!(
                "maximum {idx} upper bound must be finite"
            )));
        }

        let selectors = candidates
            .iter()
            .map(|_| append_binary_helper(c, integer_vars, lb, ub, a))
            .collect::<Vec<_>>();
        for (pos, candidate) in candidates.iter().enumerate() {
            let mut row = vec![0.0; c.len()];
            row[target] = -1.0;
            match *candidate {
                ExtremumCandidate::Var(var) => row[var] = 1.0,
                ExtremumCandidate::Constant(value) => {
                    a.push(row);
                    b.push(-value);
                    let big_m = (max_upper - value).max(0.0);
                    let mut row = vec![0.0; c.len()];
                    row[target] = 1.0;
                    row[selectors[pos]] = big_m;
                    a.push(row);
                    b.push(value + big_m);
                    continue;
                }
            }
            a.push(row);
            b.push(0.0);

            let big_m = (max_upper - candidate_lower(*candidate, lb)).max(0.0);
            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            row[selectors[pos]] = big_m;
            if let ExtremumCandidate::Var(var) = *candidate {
                row[var] = -1.0;
            }
            a.push(row);
            b.push(big_m);
        }

        let mut row = vec![0.0; c.len()];
        for selector in selectors {
            row[selector] = 1.0;
        }
        append_equality(a, b, row, 1.0);
    }
    Ok(())
}

fn append_minimum_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("minimums").or_else(|| p.get("min_constraints")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("minimums must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("minimum {idx} target_var is required")))?,
            format!("minimum {idx} target_var must be a nonnegative integer"),
        )?;
        if target >= c.len() {
            return Err(CliError(format!("minimum {idx} target_var out of range")));
        }
        if !lb[target].is_finite() {
            return Err(CliError(format!(
                "minimum {idx} target lower bound must be finite"
            )));
        }
        let candidates = extremum_candidates(constraint, idx, "minimum")?;
        validate_extremum_candidates(&candidates, target, c.len(), "minimum", idx, lb)?;
        if candidates.len() == 1 {
            append_extremum_single_candidate(candidates[0], target, a, b, c.len());
            continue;
        }

        let selectors = candidates
            .iter()
            .map(|_| append_binary_helper(c, integer_vars, lb, ub, a))
            .collect::<Vec<_>>();
        for (pos, candidate) in candidates.iter().enumerate() {
            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            match *candidate {
                ExtremumCandidate::Var(var) => {
                    row[var] = -1.0;
                    a.push(row);
                    b.push(0.0);
                }
                ExtremumCandidate::Constant(value) => {
                    a.push(row);
                    b.push(value);
                }
            }

            let upper = candidate_upper(*candidate, ub).ok_or_else(|| {
                CliError(format!(
                    "minimum {idx} argument variable needs a finite upper bound"
                ))
            })?;
            let big_m = (upper - lb[target]).max(0.0);
            let mut row = vec![0.0; c.len()];
            row[target] = -1.0;
            row[selectors[pos]] = big_m;
            let rhs = match *candidate {
                ExtremumCandidate::Var(var) => {
                    row[var] = 1.0;
                    big_m
                }
                ExtremumCandidate::Constant(value) => -value + big_m,
            };
            a.push(row);
            b.push(rhs);
        }

        let mut row = vec![0.0; c.len()];
        for selector in selectors {
            row[selector] = 1.0;
        }
        append_equality(a, b, row, 1.0);
    }
    Ok(())
}

fn validate_logical_binary_var(
    var: usize,
    idx: usize,
    role: &str,
    integer_vars: &[bool],
    lb: &[f64],
    ub: &[Option<f64>],
    n: usize,
) -> Result<(), CliError> {
    if var >= n {
        return Err(CliError(format!(
            "logical {idx} {role} variable out of range"
        )));
    }
    if lb[var].abs() > 1.0e-12 {
        return Err(CliError(format!(
            "logical {idx} {role} variable must have lower bound 0"
        )));
    }
    if !integer_vars[var] {
        return Err(CliError(format!(
            "logical {idx} {role} variable must be integer/binary"
        )));
    }
    let upper = ub[var].ok_or_else(|| {
        CliError(format!(
            "logical {idx} {role} variable must have finite binary upper bound <= 1"
        ))
    })?;
    if !upper.is_finite() || upper > 1.0 + EPS {
        return Err(CliError(format!(
            "logical {idx} {role} variable must have finite binary upper bound <= 1"
        )));
    }
    Ok(())
}

fn append_logical_constraints(
    p: &Value,
    n: usize,
    integer_vars: &[bool],
    lb: &[f64],
    ub: &[Option<f64>],
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("logical").or_else(|| p.get("logic_constraints")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("logical constraints must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("logical {idx} target_var is required")))?,
            format!("logical {idx} target_var must be a nonnegative integer"),
        )?;
        validate_logical_binary_var(target, idx, "target", integer_vars, lb, ub, n)?;
        let raw_args = constraint
            .get("arg_vars")
            .ok_or_else(|| CliError(format!("logical {idx} needs at least one argument")))?;
        let raw_args = raw_args
            .as_array()
            .ok_or_else(|| CliError(format!("logical {idx} arg_vars must be an array")))?;
        if raw_args.is_empty() {
            return Err(CliError(format!(
                "logical {idx} needs at least one argument"
            )));
        }
        let mut args = Vec::with_capacity(raw_args.len());
        for value in raw_args {
            let var = usize_value(
                value,
                format!("logical {idx} argument variable must be a nonnegative integer"),
            )?;
            validate_logical_binary_var(var, idx, "argument", integer_vars, lb, ub, n)?;
            if var == target {
                return Err(CliError(format!(
                    "logical {idx} target_var must be distinct from argument variables"
                )));
            }
            if args.contains(&var) {
                return Err(CliError(format!(
                    "logical {idx} duplicate argument variable {var}"
                )));
            }
            args.push(var);
        }

        match constraint
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("and")
        {
            "and" => {
                for &var in &args {
                    let mut row = vec![0.0; n];
                    row[target] = 1.0;
                    row[var] = -1.0;
                    a.push(row);
                    b.push(0.0);
                }
                let mut row = vec![0.0; n];
                for &var in &args {
                    row[var] = 1.0;
                }
                row[target] = -1.0;
                a.push(row);
                b.push(args.len() as f64 - 1.0);
            }
            "or" => {
                for &var in &args {
                    let mut row = vec![0.0; n];
                    row[var] = 1.0;
                    row[target] = -1.0;
                    a.push(row);
                    b.push(0.0);
                }
                let mut row = vec![0.0; n];
                row[target] = 1.0;
                for &var in &args {
                    row[var] = -1.0;
                }
                a.push(row);
                b.push(0.0);
            }
            other => return Err(CliError(format!("logical {idx} has unknown kind {other}"))),
        }
    }
    Ok(())
}

fn l1_abs_helper_upper(
    lower: f64,
    upper: Option<f64>,
    idx: usize,
    pos: usize,
) -> Result<Option<f64>, CliError> {
    if !lower.is_finite() {
        return Err(CliError(format!(
            "l1_norm {idx} argument {pos} lower bound must be finite"
        )));
    }
    if upper.is_some_and(|value| value + EPS < lower) {
        return Err(CliError(format!(
            "l1_norm {idx} argument {pos} lower bound exceeds upper bound"
        )));
    }
    if lower >= -1.0e-12 {
        return Ok(upper.map(|value| value.max(0.0)));
    }
    if upper.is_some_and(|value| value <= 1.0e-12) {
        return Ok(Some((-lower).max(0.0)));
    }
    let Some(upper) = upper else {
        return Err(CliError(format!(
            "l1_norm {idx} mixed-sign argument {pos} needs a finite upper bound"
        )));
    };
    Ok(Some((-lower).max(upper).max(0.0)))
}

fn append_l1_norm_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p.get("l1_norms").or_else(|| p.get("l1_norm_constraints")) else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("l1_norms must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("l1_norm {idx} target_var is required")))?,
            format!("l1_norm {idx} target_var must be a nonnegative integer"),
        )?;
        if target >= c.len() {
            return Err(CliError(format!("l1_norm {idx} target_var out of range")));
        }
        if !lb[target].is_finite() {
            return Err(CliError(format!(
                "l1_norm {idx} target lower bound must be finite"
            )));
        }
        let raw_args = constraint
            .get("arg_vars")
            .ok_or_else(|| CliError(format!("l1_norm {idx} needs at least one argument")))?;
        let raw_args = raw_args
            .as_array()
            .ok_or_else(|| CliError(format!("l1_norm {idx} arg_vars must be an array")))?;
        if raw_args.is_empty() {
            return Err(CliError(format!(
                "l1_norm {idx} needs at least one argument"
            )));
        }

        let mut args = Vec::with_capacity(raw_args.len());
        for (pos, value) in raw_args.iter().enumerate() {
            let var = usize_value(
                value,
                format!("l1_norm {idx} argument variable must be a nonnegative integer"),
            )?;
            if var >= c.len() {
                return Err(CliError(format!(
                    "l1_norm {idx} argument variable {var} out of range"
                )));
            }
            if var == target {
                return Err(CliError(format!(
                    "l1_norm {idx} target_var must be distinct from argument variables"
                )));
            }
            if args.contains(&var) {
                return Err(CliError(format!(
                    "l1_norm {idx} duplicate argument variable {var}"
                )));
            }
            let upper = ub[var].filter(|value| value.is_finite());
            l1_abs_helper_upper(lb[var], upper, idx, pos)?;
            args.push(var);
        }

        let mut helpers = Vec::with_capacity(args.len());
        for (pos, &arg) in args.iter().enumerate() {
            let lower = lb[arg];
            let upper = ub[arg].filter(|value| value.is_finite());
            let helper_upper = l1_abs_helper_upper(lower, upper, idx, pos)?;
            let helper = append_continuous_helper(c, integer_vars, lb, ub, a, helper_upper);
            helpers.push(helper);

            if lower >= -1.0e-12 {
                let mut row = vec![0.0; c.len()];
                row[helper] = 1.0;
                row[arg] = -1.0;
                append_equality(a, b, row, 0.0);
                continue;
            }
            if upper.is_some_and(|value| value <= 1.0e-12) {
                let mut row = vec![0.0; c.len()];
                row[helper] = 1.0;
                row[arg] = 1.0;
                append_equality(a, b, row, 0.0);
                continue;
            }
            let Some(upper) = upper else {
                return Err(CliError(format!(
                    "l1_norm {idx} mixed-sign argument {pos} needs a finite upper bound"
                )));
            };
            let selector = append_binary_helper(c, integer_vars, lb, ub, a);

            let mut row = vec![0.0; c.len()];
            row[arg] = 1.0;
            row[helper] = -1.0;
            a.push(row);
            b.push(0.0);

            let mut row = vec![0.0; c.len()];
            row[arg] = -1.0;
            row[helper] = -1.0;
            a.push(row);
            b.push(0.0);

            let mut row = vec![0.0; c.len()];
            row[helper] = 1.0;
            row[arg] = -1.0;
            row[selector] = -2.0 * lower;
            a.push(row);
            b.push(-2.0 * lower);

            let mut row = vec![0.0; c.len()];
            row[helper] = 1.0;
            row[arg] = 1.0;
            row[selector] = -2.0 * upper;
            a.push(row);
            b.push(0.0);
        }

        let mut row = vec![0.0; c.len()];
        row[target] = 1.0;
        for helper in helpers {
            row[helper] = -1.0;
        }
        append_equality(a, b, row, 0.0);
    }
    Ok(())
}

fn linf_abs_helper_upper(
    lower: f64,
    upper: Option<f64>,
    idx: usize,
    pos: usize,
) -> Result<Option<f64>, CliError> {
    if !lower.is_finite() {
        return Err(CliError(format!(
            "linf_norm {idx} argument {pos} lower bound must be finite"
        )));
    }
    if upper.is_some_and(|value| value + EPS < lower) {
        return Err(CliError(format!(
            "linf_norm {idx} argument {pos} lower bound exceeds upper bound"
        )));
    }
    if lower >= -1.0e-12 {
        return Ok(upper.map(|value| value.max(0.0)));
    }
    if upper.is_some_and(|value| value <= 1.0e-12) {
        return Ok(Some((-lower).max(0.0)));
    }
    let Some(upper) = upper else {
        return Err(CliError(format!(
            "linf_norm {idx} mixed-sign argument {pos} needs a finite upper bound"
        )));
    };
    Ok(Some((-lower).max(upper).max(0.0)))
}

fn append_linf_norm_constraints(
    p: &Value,
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    lb: &mut Vec<f64>,
    ub: &mut Vec<Option<f64>>,
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = p
        .get("linf_norms")
        .or_else(|| p.get("linf_norm_constraints"))
    else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("linf_norms must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let target = usize_value(
            constraint
                .get("target_var")
                .ok_or_else(|| CliError(format!("linf_norm {idx} target_var is required")))?,
            format!("linf_norm {idx} target_var must be a nonnegative integer"),
        )?;
        if target >= c.len() {
            return Err(CliError(format!("linf_norm {idx} target_var out of range")));
        }
        if !lb[target].is_finite() {
            return Err(CliError(format!(
                "linf_norm {idx} target lower bound must be finite"
            )));
        }
        let raw_args = constraint
            .get("arg_vars")
            .ok_or_else(|| CliError(format!("linf_norm {idx} needs at least one argument")))?;
        let raw_args = raw_args
            .as_array()
            .ok_or_else(|| CliError(format!("linf_norm {idx} arg_vars must be an array")))?;
        if raw_args.is_empty() {
            return Err(CliError(format!(
                "linf_norm {idx} needs at least one argument"
            )));
        }

        let mut args = Vec::with_capacity(raw_args.len());
        for (pos, value) in raw_args.iter().enumerate() {
            let var = usize_value(
                value,
                format!("linf_norm {idx} argument variable must be a nonnegative integer"),
            )?;
            if var >= c.len() {
                return Err(CliError(format!(
                    "linf_norm {idx} argument variable {var} out of range"
                )));
            }
            if var == target {
                return Err(CliError(format!(
                    "linf_norm {idx} target_var must be distinct from argument variables"
                )));
            }
            if args.contains(&var) {
                return Err(CliError(format!(
                    "linf_norm {idx} duplicate argument variable {var}"
                )));
            }
            let upper = ub[var].filter(|value| value.is_finite());
            linf_abs_helper_upper(lb[var], upper, idx, pos)?;
            args.push(var);
        }

        let mut helpers = Vec::with_capacity(args.len());
        for (pos, &arg) in args.iter().enumerate() {
            let lower = lb[arg];
            let upper = ub[arg].filter(|value| value.is_finite());
            let helper_upper = linf_abs_helper_upper(lower, upper, idx, pos)?;
            let helper = append_continuous_helper(c, integer_vars, lb, ub, a, helper_upper);
            helpers.push(helper);

            if lower >= -1.0e-12 {
                let mut row = vec![0.0; c.len()];
                row[helper] = 1.0;
                row[arg] = -1.0;
                append_equality(a, b, row, 0.0);
                continue;
            }
            if upper.is_some_and(|value| value <= 1.0e-12) {
                let mut row = vec![0.0; c.len()];
                row[helper] = 1.0;
                row[arg] = 1.0;
                append_equality(a, b, row, 0.0);
                continue;
            }
            let Some(upper) = upper else {
                return Err(CliError(format!(
                    "linf_norm {idx} mixed-sign argument {pos} needs a finite upper bound"
                )));
            };
            let selector = append_binary_helper(c, integer_vars, lb, ub, a);

            let mut row = vec![0.0; c.len()];
            row[arg] = 1.0;
            row[helper] = -1.0;
            a.push(row);
            b.push(0.0);

            let mut row = vec![0.0; c.len()];
            row[arg] = -1.0;
            row[helper] = -1.0;
            a.push(row);
            b.push(0.0);

            let mut row = vec![0.0; c.len()];
            row[helper] = 1.0;
            row[arg] = -1.0;
            row[selector] = -2.0 * lower;
            a.push(row);
            b.push(-2.0 * lower);

            let mut row = vec![0.0; c.len()];
            row[helper] = 1.0;
            row[arg] = 1.0;
            row[selector] = -2.0 * upper;
            a.push(row);
            b.push(0.0);
        }

        if helpers.len() == 1 {
            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            row[helpers[0]] = -1.0;
            append_equality(a, b, row, 0.0);
            continue;
        }

        let max_upper = if let Some(target_upper) = ub[target].filter(|value| value.is_finite()) {
            target_upper
        } else {
            let mut max_upper = f64::NEG_INFINITY;
            for &helper in &helpers {
                let upper = ub[helper]
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        CliError(format!(
                        "linf_norm {idx} needs finite helper uppers or a finite target upper bound"
                    ))
                    })?;
                max_upper = max_upper.max(upper);
            }
            max_upper
        };
        if !max_upper.is_finite() {
            return Err(CliError(format!(
                "linf_norm {idx} upper bound must be finite"
            )));
        }

        let selectors = helpers
            .iter()
            .map(|_| append_binary_helper(c, integer_vars, lb, ub, a))
            .collect::<Vec<_>>();
        for (pos, &helper) in helpers.iter().enumerate() {
            let mut row = vec![0.0; c.len()];
            row[target] = -1.0;
            row[helper] = 1.0;
            a.push(row);
            b.push(0.0);

            let mut row = vec![0.0; c.len()];
            row[target] = 1.0;
            row[helper] = -1.0;
            row[selectors[pos]] = max_upper;
            a.push(row);
            b.push(max_upper);
        }

        let mut row = vec![0.0; c.len()];
        for selector in selectors {
            row[selector] = 1.0;
        }
        append_equality(a, b, row, 1.0);
    }
    Ok(())
}

fn parse_problem(raw: &Value) -> Result<Problem, CliError> {
    let p = raw.get("problem").unwrap_or(raw);
    let mut c = number_array(
        p.get("c")
            .ok_or_else(|| CliError("c must be an array".to_string()))?,
        "c",
    )?;
    let n = c.len();
    let mut a = if let Some(a) = p
        .get("a")
        .or_else(|| p.get("A"))
        .or_else(|| p.get("A_ub"))
        .or_else(|| p.get("a_ub"))
    {
        number_matrix(a, "a")?
    } else {
        Vec::new()
    };
    let mut b = if let Some(b) = p.get("b").or_else(|| p.get("b_ub")) {
        number_array(b, "b")?
    } else {
        Vec::new()
    };
    if a.len() != b.len() {
        return Err(CliError(
            "constraint matrix/vector length mismatch".to_string(),
        ));
    }
    for row in &a {
        if row.len() != n {
            return Err(CliError("constraint row length mismatch".to_string()));
        }
    }
    append_linear_constraints(p, n, &mut a, &mut b)?;
    append_dense_equalities(p, n, &mut a, &mut b)?;
    append_lazy_constraints(p, n, &mut a, &mut b)?;
    let mut integer_vars = p
        .get("integer_vars")
        .or_else(|| p.get("integerVars"))
        .and_then(Value::as_array)
        .ok_or_else(|| CliError("integer_vars must be an array".to_string()))?
        .iter()
        .map(|value| value.as_bool().unwrap_or(false))
        .collect::<Vec<_>>();
    if integer_vars.len() != n {
        return Err(CliError("integer_vars length mismatch".to_string()));
    }
    let lbs = optional_bound_array(p.get("lb"), n, true)?;
    let mut ubs = optional_bound_array(p.get("ub"), n, false)?;
    let mut lb = lbs
        .into_iter()
        .map(|value| value.unwrap_or(0.0))
        .collect::<Vec<_>>();
    append_indicators(p, n, &integer_vars, &lb, &ubs, &mut a, &mut b)?;
    append_abs_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_maximum_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_minimum_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_logical_constraints(p, c.len(), &integer_vars, &lb, &ubs, &mut a, &mut b)?;
    append_l1_norm_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_linf_norm_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_quadratic_objective_terms(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_product_constraints(p, &integer_vars, &lb, &ubs, &mut a, &mut b)?;
    append_pwl_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_sos_constraints(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    append_semi_variables(
        p,
        &mut c,
        &mut integer_vars,
        &mut lb,
        &mut ubs,
        &mut a,
        &mut b,
    )?;
    let sense = match p.get("sense").and_then(Value::as_str).unwrap_or("max") {
        "max" => Sense::Max,
        "min" => Sense::Min,
        other => return Err(CliError(format!("unknown sense {other:?}"))),
    };
    Ok(Problem {
        sense,
        c,
        a,
        b,
        integer_vars,
        lb,
        ub: ubs,
    })
}

fn objective(problem: &Problem, x: &[f64]) -> f64 {
    problem
        .c
        .iter()
        .zip(x)
        .map(|(coef, value)| coef * value)
        .sum()
}

fn feasible(problem: &Problem, x: &[f64]) -> bool {
    if x.len() != problem.c.len() {
        return false;
    }
    for (idx, &value) in x.iter().enumerate() {
        if value < problem.lb[idx] - 1.0e-7 {
            return false;
        }
        if problem.ub[idx].is_some_and(|upper| value > upper + 1.0e-7) {
            return false;
        }
        if problem.integer_vars[idx] && (value - value.round()).abs() > 1.0e-7 {
            return false;
        }
    }
    problem.a.iter().zip(&problem.b).all(|(row, &rhs)| {
        row.iter()
            .zip(x)
            .map(|(coef, value)| coef * value)
            .sum::<f64>()
            <= rhs + 1.0e-7
    })
}

fn solve_continuous_remainder(
    problem: &Problem,
    fixed: &[(usize, f64)],
) -> Result<Option<RemainderSolution>, &'static str> {
    let n = problem.c.len();
    let mut x = vec![0.0; n];
    for &(idx, value) in fixed {
        x[idx] = value;
    }
    let continuous = (0..n)
        .filter(|idx| !fixed.iter().any(|(fixed_idx, _)| fixed_idx == idx))
        .collect::<Vec<_>>();
    if continuous.is_empty() {
        return Ok(feasible(problem, &x).then(|| RemainderSolution {
            objective: objective(problem, &x),
            x,
        }));
    }

    let c = continuous
        .iter()
        .map(|&idx| problem.c[idx])
        .collect::<Vec<_>>();
    let mut a_ub = Vec::with_capacity(problem.a.len());
    let mut b_ub = Vec::with_capacity(problem.b.len());
    for (row, &rhs) in problem.a.iter().zip(&problem.b) {
        let fixed_lhs = fixed
            .iter()
            .map(|&(idx, value)| row[idx] * value)
            .sum::<f64>();
        a_ub.push(continuous.iter().map(|&idx| row[idx]).collect::<Vec<_>>());
        b_ub.push(rhs - fixed_lhs);
    }
    let lp = LPProblem {
        sense: problem.sense,
        c,
        a_ub: (!a_ub.is_empty()).then_some(a_ub),
        b_ub: (!b_ub.is_empty()).then_some(b_ub),
        a_eq: None,
        b_eq: None,
        lb: Some(
            continuous
                .iter()
                .map(|&idx| Some(problem.lb[idx]))
                .collect(),
        ),
        ub: Some(continuous.iter().map(|&idx| problem.ub[idx]).collect()),
        var_names: None,
        con_names: None,
    };
    let solution = solve_lp_internal(&lp, &InternalSimplexOptions::default());
    match solution.status {
        LPStatus::Optimal => {
            for (local_idx, &global_idx) in continuous.iter().enumerate() {
                x[global_idx] = solution.x[local_idx];
            }
            if feasible(problem, &x) {
                Ok(Some(RemainderSolution {
                    objective: objective(problem, &x),
                    x,
                }))
            } else {
                Ok(None)
            }
        }
        LPStatus::Unbounded => Err("unbounded"),
        LPStatus::Infeasible | LPStatus::IterLimit | LPStatus::NumericalError => Ok(None),
    }
}

fn integer_domains(
    problem: &Problem,
    solver: &'static str,
) -> Result<Vec<(usize, Vec<i64>)>, EnumerationResult> {
    let mut domains = Vec::new();
    for (idx, &is_integer) in problem.integer_vars.iter().enumerate() {
        if !is_integer {
            continue;
        }
        let Some(upper) = problem.ub[idx] else {
            return Err(EnumerationResult {
                status: "unavailable",
                solver,
                x: None,
                objective: None,
                message: format!("x{idx} has no finite upper bound"),
                enumerated: 0,
                solutions: None,
                exhausted: None,
            });
        };
        let lo = problem.lb[idx].ceil() as i64;
        let hi = upper.floor() as i64;
        if hi < lo {
            return Err(EnumerationResult {
                status: "infeasible",
                solver,
                x: None,
                objective: None,
                message: String::new(),
                enumerated: 0,
                solutions: None,
                exhausted: Some(true),
            });
        }
        domains.push((idx, (lo..=hi).collect()));
    }
    Ok(domains)
}

fn enumerate_domains<F>(
    domains: &[(usize, Vec<i64>)],
    current: &mut Vec<(usize, f64)>,
    max_enumerations: usize,
    enumerated: &mut usize,
    callback: &mut F,
) -> Result<(), ()>
where
    F: FnMut(&[(usize, f64)]) -> Result<(), ()>,
{
    if current.len() == domains.len() {
        *enumerated += 1;
        if *enumerated > max_enumerations {
            return Err(());
        }
        return callback(current);
    }
    let (idx, domain) = &domains[current.len()];
    for value in domain {
        current.push((*idx, *value as f64));
        enumerate_domains(domains, current, max_enumerations, enumerated, callback)?;
        current.pop();
    }
    Ok(())
}

fn better(problem: &Problem, candidate: f64, incumbent: f64) -> bool {
    match problem.sense {
        Sense::Max => candidate > incumbent + EPS,
        Sense::Min => candidate < incumbent - EPS,
    }
}

fn brute_force(problem: &Problem, max_enumerations: usize) -> EnumerationResult {
    let solver = "rust:bounded-enumeration";
    let domains = match integer_domains(problem, solver) {
        Ok(domains) => domains,
        Err(result) => return result,
    };
    let mut best = None::<RemainderSolution>;
    let mut best_obj = match problem.sense {
        Sense::Max => f64::NEG_INFINITY,
        Sense::Min => f64::INFINITY,
    };
    let mut enumerated = 0usize;
    let mut current = Vec::new();
    let mut saw_unbounded = false;
    let cap_hit = enumerate_domains(
        &domains,
        &mut current,
        max_enumerations,
        &mut enumerated,
        &mut |fixed| match solve_continuous_remainder(problem, fixed) {
            Ok(Some(solution)) => {
                if best.is_none() || better(problem, solution.objective, best_obj) {
                    best_obj = solution.objective;
                    best = Some(solution);
                }
                Ok(())
            }
            Ok(None) => Ok(()),
            Err("unbounded") => {
                saw_unbounded = true;
                Err(())
            }
            Err(_) => Ok(()),
        },
    )
    .is_err();
    if saw_unbounded {
        return EnumerationResult {
            status: "unbounded",
            solver,
            x: None,
            objective: None,
            message: String::new(),
            enumerated,
            solutions: None,
            exhausted: None,
        };
    }
    if cap_hit {
        return EnumerationResult {
            status: "unavailable",
            solver,
            x: None,
            objective: None,
            message: "enumeration cap reached".to_string(),
            enumerated,
            solutions: None,
            exhausted: None,
        };
    }
    let Some(best) = best else {
        return EnumerationResult {
            status: "infeasible",
            solver,
            x: None,
            objective: None,
            message: String::new(),
            enumerated,
            solutions: None,
            exhausted: None,
        };
    };
    EnumerationResult {
        status: "optimal",
        solver,
        x: Some(best.x),
        objective: Some(best.objective),
        message: "exact bounded enumeration".to_string(),
        enumerated,
        solutions: None,
        exhausted: None,
    }
}

fn brute_force_pool(
    problem: &Problem,
    pool_size: usize,
    max_enumerations: usize,
) -> EnumerationResult {
    let solver = "rust:bounded-enumeration-pool";
    if !problem.integer_vars.iter().any(|&is_integer| is_integer) {
        return EnumerationResult {
            status: "unavailable",
            solver,
            x: None,
            objective: None,
            message: "solution pool requires integer variables".to_string(),
            enumerated: 0,
            solutions: Some(Vec::new()),
            exhausted: Some(false),
        };
    }
    let domains = match integer_domains(problem, solver) {
        Ok(domains) => domains,
        Err(mut result) => {
            result.solutions = Some(Vec::new());
            result.exhausted.get_or_insert(false);
            return result;
        }
    };
    let mut candidates = Vec::<RemainderSolution>::new();
    let mut enumerated = 0usize;
    let mut current = Vec::new();
    let mut saw_unbounded = false;
    let cap_hit = enumerate_domains(
        &domains,
        &mut current,
        max_enumerations,
        &mut enumerated,
        &mut |fixed| match solve_continuous_remainder(problem, fixed) {
            Ok(Some(solution)) => {
                candidates.push(solution);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err("unbounded") => {
                saw_unbounded = true;
                Err(())
            }
            Err(_) => Ok(()),
        },
    )
    .is_err();
    if saw_unbounded {
        return EnumerationResult {
            status: "unbounded",
            solver,
            x: None,
            objective: None,
            message: String::new(),
            enumerated,
            solutions: Some(Vec::new()),
            exhausted: Some(false),
        };
    }
    if cap_hit {
        return EnumerationResult {
            status: "unavailable",
            solver,
            x: None,
            objective: None,
            message: "enumeration cap reached".to_string(),
            enumerated,
            solutions: None,
            exhausted: None,
        };
    }
    if candidates.is_empty() {
        return EnumerationResult {
            status: "infeasible",
            solver,
            x: None,
            objective: None,
            message: String::new(),
            enumerated,
            solutions: Some(Vec::new()),
            exhausted: Some(true),
        };
    }
    candidates.sort_by(|left, right| {
        let ordering = left
            .objective
            .partial_cmp(&right.objective)
            .unwrap_or(std::cmp::Ordering::Equal);
        match problem.sense {
            Sense::Max => ordering.reverse(),
            Sense::Min => ordering,
        }
    });
    let chosen = candidates
        .iter()
        .take(pool_size)
        .cloned()
        .collect::<Vec<_>>();
    let best = chosen[0].clone();
    EnumerationResult {
        status: "optimal",
        solver,
        x: Some(best.x),
        objective: Some(best.objective),
        message: "exact bounded solution-pool enumeration".to_string(),
        enumerated,
        solutions: Some(chosen),
        exhausted: Some(pool_size >= candidates.len()),
    }
}

fn result_payload_json(result: &EnumerationResult) -> Value {
    let mut payload = json!({
        "status": result.status,
        "solver": result.solver,
        "x": result.x,
        "objective": result.objective,
        "message": result.message,
        "enumerated": result.enumerated,
    });
    if let Some(solutions) = &result.solutions {
        payload["solutions"] = json!(solutions
            .iter()
            .map(|solution| json!({
                "x": solution.x,
                "objective": solution.objective,
            }))
            .collect::<Vec<_>>());
    }
    if let Some(exhausted) = result.exhausted {
        payload["exhausted"] = json!(exhausted);
    }
    payload
}

fn result_json(result: &EnumerationResult) -> Value {
    let payload = json!({
        "result": result_payload_json(result),
    });
    payload
}

fn error_json(message: impl Into<String>) -> Value {
    unavailable_json("rust:bounded-enumeration", message)
}

fn unavailable_json(solver: &'static str, message: impl Into<String>) -> Value {
    json!({
        "result": {
            "status": "unavailable",
            "solver": solver,
            "x": null,
            "objective": null,
            "message": message.into(),
            "enumerated": 0,
        }
    })
}

fn objective_specs(raw: &Value) -> Result<Option<&Vec<Value>>, CliError> {
    let p = raw.get("problem").unwrap_or(raw);
    match p.get("multi_objectives") {
        None => Ok(None),
        Some(Value::Array(objectives)) if objectives.is_empty() => Ok(None),
        Some(Value::Array(objectives)) => Ok(Some(objectives)),
        Some(_) => Err(CliError("multi_objectives must be an array".to_string())),
    }
}

fn objective_coefficients(
    objective_spec: &Value,
    idx: usize,
    variable_count: usize,
) -> Result<Vec<f64>, CliError> {
    let mut coeffs = number_array(
        objective_spec
            .get("c")
            .ok_or_else(|| CliError(format!("objective {idx} c must be an array")))?,
        &format!("multi_objectives[{idx}].c"),
    )?;
    if coeffs.len() > variable_count {
        return Err(CliError(format!(
            "objective {idx} coefficient length does not match variable count"
        )));
    }
    coeffs.resize(variable_count, 0.0);
    Ok(coeffs)
}

fn objective_sense(objective_spec: &Value, idx: usize) -> Result<Sense, CliError> {
    match objective_spec
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("max")
    {
        "max" => Ok(Sense::Max),
        "min" => Ok(Sense::Min),
        other => Err(CliError(format!(
            "objective {idx} has unknown sense {other:?}"
        ))),
    }
}

fn objective_from_coefficients(coeffs: &[f64], x: &[f64]) -> f64 {
    coeffs.iter().zip(x).map(|(coef, value)| coef * value).sum()
}

fn solve_multi_objective(
    mut problem: Problem,
    objectives: &[Value],
    max_enumerations: usize,
) -> Result<Value, CliError> {
    let variable_count = problem.c.len();
    let mut stages = Vec::<Value>::with_capacity(objectives.len());
    let mut objective_rows = Vec::<Vec<f64>>::with_capacity(objectives.len());
    let mut total_enumerated = 0usize;
    let mut final_x = None::<Vec<f64>>;

    for (idx, objective_spec) in objectives.iter().enumerate() {
        let coeffs = objective_coefficients(objective_spec, idx, variable_count)?;
        let sense = objective_sense(objective_spec, idx)?;
        let mut stage_problem = problem.clone();
        stage_problem.c = coeffs.clone();
        stage_problem.sense = sense;
        let result = brute_force(&stage_problem, max_enumerations);
        total_enumerated = total_enumerated.saturating_add(result.enumerated);
        stages.push(result_payload_json(&result));
        if result.status != "optimal" {
            return Ok(json!({
                "result": {
                    "status": result.status,
                    "solver": result.solver,
                    "x": result.x,
                    "objective": result.objective,
                    "message": result.message,
                    "enumerated": result.enumerated,
                    "objective_values": [],
                    "stages": stages,
                }
            }));
        }
        let x = result.x.ok_or_else(|| {
            CliError("optimal multi-objective stage did not return x".to_string())
        })?;
        let optimum = objective_from_coefficients(&coeffs, &x);
        append_equality(&mut problem.a, &mut problem.b, coeffs.clone(), optimum);
        objective_rows.push(coeffs);
        final_x = Some(x);
    }

    let final_x =
        final_x.ok_or_else(|| CliError("multi_objectives must be non-empty".to_string()))?;
    let objective_values = objective_rows
        .iter()
        .map(|coeffs| objective_from_coefficients(coeffs, &final_x))
        .collect::<Vec<_>>();
    Ok(json!({
        "result": {
            "status": "optimal",
            "solver": "rust:lexicographic-multi-objective",
            "x": final_x,
            "objective": objective_values.last().copied(),
            "message": "sequential lexicographic optimization",
            "enumerated": total_enumerated,
            "objective_values": objective_values,
            "stages": stages,
        }
    }))
}

fn run(args: &Args, input: &str) -> Result<Value, CliError> {
    let raw = serde_json::from_str::<Value>(input)
        .map_err(|err| CliError(format!("failed to parse JSON: {err}")))?;
    let problem = parse_problem(&raw)?;
    if !rust_reference_solver_alias_supported(&args.solver) {
        return Ok(error_json(format!(
            "unknown or unavailable solver '{}'",
            args.solver
        )));
    }
    if let Some(objectives) = objective_specs(&raw)? {
        if args.pool_size.is_some() {
            return Ok(unavailable_json(
                "rust:bounded-enumeration-pool",
                "solution pools for multi-objective MIPs are not supported",
            ));
        }
        return solve_multi_objective(problem, objectives, args.max_enumerations);
    }
    let result = if let Some(pool_size) = args.pool_size {
        brute_force_pool(&problem, pool_size, args.max_enumerations)
    } else {
        brute_force(&problem, args.max_enumerations)
    };
    Ok(result_json(&result))
}

fn read_input(args: &Args) -> Result<String, CliError> {
    if let Some(path) = &args.problem {
        return fs::read_to_string(path)
            .map_err(|err| CliError(format!("failed to read {}: {err}", path.display())));
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|err| CliError(format!("failed to read stdin: {err}")))?;
    Ok(input)
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
                    .unwrap_or("ip_mip_reference")
            )
        );
        return;
    }
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "ip_mip_reference".to_string());
    let output = match parse_args(&program, raw_args.into_iter().skip(1)).and_then(|args| {
        let input = read_input(&args)?;
        let output = run(&args, &input)?;
        if let Some(path) = &args.out {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|err| {
                    CliError(format!("failed to create {}: {err}", parent.display()))
                })?;
            }
            fs::write(
                path,
                format!(
                    "{}\n",
                    serde_json::to_string_pretty(&output).expect("serialize pretty output")
                ),
            )
            .map_err(|err| CliError(format!("failed to write {}: {err}", path.display())))?;
        }
        Ok(output)
    }) {
        Ok(output) => output,
        Err(err) => error_json(err.to_string()),
    };
    println!(
        "{}",
        serde_json::to_string(&output["result"]).expect("serialize result")
    );
    if output["result"]["status"] == "unavailable" {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static IP_MIP_REFERENCE_CLI_ENV_LOCK: Mutex<()> = Mutex::new(());

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

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            std::env::remove_var(key);
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

    fn ip_mip_reference_python_off_guards() -> Vec<EnvVarGuard> {
        vec![
            EnvVarGuard::set("PYTHON_BIN", "/definitely/not/python"),
            EnvVarGuard::set("PYTHON", "/definitely/not/python"),
            EnvVarGuard::set("SCIPY_MILP_PYTHON", "/definitely/not/python"),
            EnvVarGuard::set("ORES_SCIPY_MILP_PYTHON", "/definitely/not/python"),
            EnvVarGuard::unset("IP_MIP_REFERENCE_FORCE_PYTHON"),
            EnvVarGuard::unset("IP_MIP_REFERENCE_SCIPY_FORCE_PYTHON"),
            EnvVarGuard::unset("ORES_EXTERNAL_REFERENCE_FORCE_PYTHON"),
        ]
    }

    const BINARY_SAMPLE: &str = r#"{
        "sense": "max",
        "c": [3, 2],
        "a": [[2, 1]],
        "b": [2],
        "integer_vars": [true, true],
        "lb": [0, 0],
        "ub": [1, 1]
    }"#;

    #[test]
    fn bounded_enumeration_solves_binary_mip() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            BINARY_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
        assert_eq!(output["result"]["objective"], 3.0);
    }

    #[test]
    fn external_solver_names_alias_to_rust_enumeration() {
        for solver in [
            "ortools",
            "ortools:cp-sat",
            "ortools_scip",
            "scipy:milp",
            "rust:fallback",
            "rust:internal",
        ] {
            let output = run(
                &Args {
                    problem: None,
                    out: None,
                    solver: solver.to_string(),
                    max_enumerations: 100,
                    pool_size: None,
                },
                BINARY_SAMPLE,
            )
            .expect("run");

            assert_eq!(output["result"]["status"], "optimal", "{solver}: {output}");
            assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
            assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
        }
    }

    #[test]
    fn major_external_mip_cli_aliases_use_rust_enumeration_without_python() {
        let _env_lock = IP_MIP_REFERENCE_CLI_ENV_LOCK.lock().expect("env lock");
        let _guards = ip_mip_reference_python_off_guards();

        for solver in [
            "highs",
            "highs:cli",
            "cbc",
            "glpk",
            "scip",
            "gurobi",
            "cplex",
            "xpress",
            "fico-xpress",
            "lindo",
            "mosek",
            "copt",
            "scipy:milp",
        ] {
            let output = run(
                &Args {
                    problem: None,
                    out: None,
                    solver: solver.to_string(),
                    max_enumerations: 100,
                    pool_size: None,
                },
                BINARY_SAMPLE,
            )
            .expect("run");

            assert_eq!(output["result"]["status"], "optimal", "{solver}: {output}");
            assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
            assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
            assert_eq!(output["result"]["objective"], 3.0);
        }
    }

    #[test]
    fn bounded_enumeration_expands_linear_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2],
                "linear_constraints": [
                    {"coefs": [2, 1], "upper": 2},
                    {"coefs": [1, 1], "lower": 1}
                ],
                "integer_vars": [true, true],
                "lb": [0, 0],
                "ub": [1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
        assert_eq!(output["result"]["objective"], 3.0);
    }

    #[test]
    fn bounded_enumeration_expands_dense_equalities() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2],
                "A_eq": [[1, 1]],
                "b_eq": [4],
                "integerVars": [true, true],
                "lb": [0, 0],
                "ub": [4, 4]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([4.0, 0.0]));
        assert_eq!(output["result"]["objective"], 12.0);
    }

    #[test]
    fn bounded_enumeration_accepts_dense_a_alias() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2],
                "A": [[1, 1]],
                "b": [4],
                "integerVars": [true, true],
                "lb": [0, 0],
                "ub": [4, 4]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([4.0, 0.0]));
        assert_eq!(output["result"]["objective"], 12.0);
    }

    #[test]
    fn bounded_enumeration_accepts_lazy_constraints_alias() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2],
                "lazyConstraints": [
                    {"coefs": [1, 1], "rhs": 4}
                ],
                "integerVars": [true, true],
                "lb": [0, 0],
                "ub": [4, 4]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([4.0, 0.0]));
        assert_eq!(output["result"]["objective"], 12.0);
    }

    #[test]
    fn bounded_enumeration_expands_indicator_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2],
                "indicators": [
                    {
                        "binary_var": 0,
                        "active_value": true,
                        "coefs": [0, 1],
                        "rhs": 0,
                        "sense": "le"
                    }
                ],
                "integer_vars": [true, true],
                "lb": [0, 0],
                "ub": [1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
        assert_eq!(output["result"]["objective"], 3.0);
    }

    #[test]
    fn bounded_enumeration_expands_binary_product_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 0, 1],
                "products": [
                    {"target_var": 2, "x_var": 0, "y_var": 1}
                ],
                "integer_vars": [true, true, true],
                "lb": [0, 0, 0],
                "ub": [1, 1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 1.0);
        assert_eq!(output["result"]["x"], json!([1.0, 1.0, 1.0]));
    }

    #[test]
    fn bounded_enumeration_expands_binary_continuous_product_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 0, 1],
                "product_constraints": [
                    {"target_var": 2, "x_var": 0, "y_var": 1}
                ],
                "integer_vars": [true, false, false],
                "lb": [0, -2, -2],
                "ub": [1, 3, 3]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 3.0);
        assert_eq!(output["result"]["x"], json!([1.0, 3.0, 3.0]));
    }

    #[test]
    fn bounded_enumeration_expands_binary_square_quadratic_objective() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [1],
                "quadratic_objective": [
                    {"x_var": 0, "y_var": 0, "coeff": 4}
                ],
                "integer_vars": [true],
                "lb": [0],
                "ub": [1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 5.0);
        assert_eq!(output["result"]["x"], json!([1.0]));
    }

    #[test]
    fn bounded_enumeration_expands_binary_continuous_quadratic_objective() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 0],
                "quadratic_objective_terms": [
                    {"x_var": 0, "y_var": 1, "coeff": 1}
                ],
                "integer_vars": [true, false],
                "lb": [0, -2],
                "ub": [1, 3]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 3.0);
        assert_eq!(output["result"]["x"], json!([1.0, 3.0, 3.0]));
    }

    #[test]
    fn bounded_enumeration_expands_semi_continuous_variables() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [1],
                "linear_constraints": [
                    {"coefs": [1], "lower": 1}
                ],
                "semi_variables": [
                    {"var": 0, "lower": 2, "kind": "semi_continuous"}
                ],
                "integer_vars": [false],
                "lb": [0],
                "ub": [5]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 2.0);
        assert_eq!(output["result"]["x"], json!([2.0, 1.0]));
    }

    #[test]
    fn bounded_enumeration_expands_semi_integer_variables() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [1],
                "linear_constraints": [
                    {"coefs": [1], "lower": 2.5}
                ],
                "semi_variables": [
                    {"var": 0, "lower": 2, "kind": "semi_integer"}
                ],
                "integer_vars": [false],
                "lb": [0],
                "ub": [5]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 3.0);
        assert_eq!(output["result"]["x"], json!([3.0, 1.0]));
    }

    #[test]
    fn bounded_enumeration_expands_sos1_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [1, 2, 3],
                "sos": [
                    {"kind": "sos1", "vars": [0, 1, 2]}
                ],
                "integer_vars": [false, false, false],
                "lb": [0, 0, 0],
                "ub": [1, 1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 3.0);
        assert_eq!(output["result"]["x"], json!([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn bounded_enumeration_expands_weighted_sos2_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [1, 0, 1],
                "linear_constraints": [
                    {"coefs": [1, 1, 1], "lower": 2}
                ],
                "sos": [
                    {"kind": "sos2", "vars": [2, 0, 1], "weights": [3, 1, 2]}
                ],
                "integer_vars": [false, false, false],
                "lb": [0, 0, 0],
                "ub": [1, 1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 1.0);
        let x = output["result"]["x"].as_array().expect("solution vector");
        assert_eq!(x.len(), 5);
        let original_sum = x[0].as_f64().unwrap() + x[1].as_f64().unwrap() + x[2].as_f64().unwrap();
        assert!((original_sum - 2.0).abs() < 1.0e-7);
        assert!(
            (x[0].as_f64().unwrap() - 1.0).abs() < 1.0e-7
                || (x[2].as_f64().unwrap() - 1.0).abs() < 1.0e-7
        );
    }

    #[test]
    fn bounded_enumeration_expands_pwl_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 1],
                "linear_constraints": [
                    {"coefs": [1, 0], "lower": 1.5, "upper": 1.5}
                ],
                "pwl": [
                    {
                        "x_var": 0,
                        "y_var": 1,
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 1, "y": 2},
                            {"x": 2, "y": 2}
                        ]
                    }
                ],
                "integer_vars": [false, false],
                "lb": [0, 0],
                "ub": [2, 3]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert!((output["result"]["objective"].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() - 1.5).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            7
        );
    }

    #[test]
    fn bounded_enumeration_expands_abs_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "brute-force".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 1],
                "abs": [
                    {"arg_var": 0, "target_var": 1}
                ],
                "integer_vars": [true, true],
                "lb": [-2, 0],
                "ub": [2, 2]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["objective"], 2.0);
        assert_eq!(output["result"]["x"][1], 2.0);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            3
        );
    }

    #[test]
    fn bounded_enumeration_expands_maximum_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [0.01, 0, 1],
                "a": [[-1, 0, 0], [0, -1, 0]],
                "b": [1, -2],
                "maximums": [
                    {"target_var": 2, "arg_vars": [0, 1], "constant": 1.5}
                ],
                "integer_vars": [false, false, false],
                "lb": [-2, 0, 0],
                "ub": [4, 5, 5]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert!((output["result"]["objective"].as_f64().unwrap() - 1.99).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() + 1.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][2].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            6
        );
    }

    #[test]
    fn bounded_enumeration_expands_minimum_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [-0.01, -0.001, 1],
                "a": [[-1, 0, 0], [0, -1, 0]],
                "b": [-3, -2.5],
                "minimums": [
                    {"target_var": 2, "arg_vars": [0, 1], "constant": 2.5}
                ],
                "integer_vars": [false, false, false],
                "lb": [-2, 0, 0],
                "ub": [4, 5, 3]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert!((output["result"]["objective"].as_f64().unwrap() - 2.4675).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() - 3.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() - 2.5).abs() < 1.0e-7);
        assert!((output["result"]["x"][2].as_f64().unwrap() - 2.5).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            6
        );
    }

    #[test]
    fn bounded_enumeration_expands_logical_and_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [-3, -2, 10],
                "logical": [
                    {"kind": "and", "target_var": 2, "arg_vars": [0, 1]}
                ],
                "integer_vars": [true, true, true],
                "lb": [0, 0, 0],
                "ub": [1, 1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([1.0, 1.0, 1.0]));
        assert_eq!(output["result"]["objective"], 5.0);
    }

    #[test]
    fn bounded_enumeration_expands_logical_or_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [3, 2, -1],
                "a": [[1, 1, 0]],
                "b": [1],
                "logic_constraints": [
                    {"kind": "or", "target_var": 2, "arg_vars": [0, 1]}
                ],
                "integer_vars": [true, true, true],
                "lb": [0, 0, 0],
                "ub": [1, 1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert_eq!(output["result"]["x"], json!([1.0, 0.0, 1.0]));
        assert_eq!(output["result"]["objective"], 2.0);
    }

    #[test]
    fn bounded_enumeration_expands_l1_norm_mixed_sign_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [0, 0, 1],
                "a": [[-1, 0, 0], [0, 1, 0]],
                "b": [-1, -2],
                "l1_norms": [
                    {"target_var": 2, "arg_vars": [0, 1]}
                ],
                "integer_vars": [false, false, false],
                "lb": [-3, -4, 0],
                "ub": [3, 4, 10]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert!((output["result"]["objective"].as_f64().unwrap() - 3.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() - 1.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() + 2.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][2].as_f64().unwrap() - 3.0).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            7
        );
    }

    #[test]
    fn bounded_enumeration_expands_l1_norm_one_sided_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [0, 0, 1],
                "a": [[-1, 0, 0], [0, 1, 0]],
                "b": [-2, -1],
                "l1_norm_constraints": [
                    {"target_var": 2, "arg_vars": [0, 1]}
                ],
                "integer_vars": [false, false, false],
                "lb": [0, -3, 0],
                "ub": [5, 0, 10]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert!((output["result"]["objective"].as_f64().unwrap() - 3.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() + 1.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][2].as_f64().unwrap() - 3.0).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            5
        );
    }

    #[test]
    fn bounded_enumeration_expands_linf_norm_mixed_sign_constraints() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [0, 0, 1],
                "a": [[-1, 0, 0], [0, 1, 0]],
                "b": [-1.5, -2],
                "linf_norms": [
                    {"target_var": 2, "arg_vars": [0, 1]}
                ],
                "integer_vars": [false, false, false],
                "lb": [-3, -4, 0],
                "ub": [3, 4, 10]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert!((output["result"]["objective"].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        let x0 = output["result"]["x"][0].as_f64().unwrap();
        assert!((1.5 - 1.0e-7..=2.0 + 1.0e-7).contains(&x0));
        assert!((output["result"]["x"][1].as_f64().unwrap() + 2.0).abs() < 1.0e-7);
        assert!((output["result"]["x"][2].as_f64().unwrap() - 2.0).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            9
        );
    }

    #[test]
    fn bounded_enumeration_expands_linf_norm_single_argument_constraint() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "min",
                "c": [0, 1],
                "a": [[1, 0]],
                "b": [-1.5],
                "linf_norm_constraints": [
                    {"target_var": 1, "arg_vars": [0]}
                ],
                "integer_vars": [false, false],
                "lb": [-4, 0],
                "ub": [0, 10]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration");
        assert!((output["result"]["objective"].as_f64().unwrap() - 1.5).abs() < 1.0e-7);
        assert!((output["result"]["x"][0].as_f64().unwrap() + 1.5).abs() < 1.0e-7);
        assert!((output["result"]["x"][1].as_f64().unwrap() - 1.5).abs() < 1.0e-7);
        assert_eq!(
            output["result"]["x"]
                .as_array()
                .expect("solution vector")
                .len(),
            3
        );
    }

    #[test]
    fn bounded_enumeration_handles_continuous_remainder() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [1, 2],
                "a": [[1, 1]],
                "b": [3],
                "integer_vars": [true, false],
                "lb": [0, 0],
                "ub": [2, null]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["x"], json!([0.0, 3.0]));
        assert_eq!(output["result"]["objective"], 6.0);
    }

    #[test]
    fn lexicographic_multi_objective_locks_prior_optimum() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: None,
            },
            r#"{
                "sense": "max",
                "c": [0, 0],
                "linear_constraints": [
                    {"coefs": [1, 1], "upper": 1}
                ],
                "multi_objectives": [
                    {"name": "prefer_home", "c": [1, 0], "sense": "max"},
                    {"name": "prefer_away", "c": [0, 1], "sense": "max"}
                ],
                "integer_vars": [true, true],
                "lb": [0, 0],
                "ub": [1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(
            output["result"]["solver"],
            "rust:lexicographic-multi-objective"
        );
        assert_eq!(output["result"]["x"], json!([1.0, 0.0]));
        assert_eq!(output["result"]["objective"], 0.0);
        assert_eq!(output["result"]["objective_values"], json!([1.0, 0.0]));
        assert_eq!(
            output["result"]["stages"]
                .as_array()
                .expect("stage results")
                .len(),
            2
        );
    }

    #[test]
    fn lexicographic_multi_objective_rejects_solution_pool() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: Some(2),
            },
            r#"{
                "sense": "max",
                "c": [0, 0],
                "multi_objectives": [
                    {"c": [1, 0], "sense": "max"},
                    {"c": [0, 1], "sense": "max"}
                ],
                "integer_vars": [true, true],
                "lb": [0, 0],
                "ub": [1, 1]
            }"#,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "unavailable");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration-pool");
        assert_eq!(
            output["result"]["message"],
            "solution pools for multi-objective MIPs are not supported"
        );
    }

    #[test]
    fn solution_pool_enumerates_top_assignments() {
        let output = run(
            &Args {
                problem: None,
                out: None,
                solver: "enumeration".to_string(),
                max_enumerations: 100,
                pool_size: Some(3),
            },
            BINARY_SAMPLE,
        )
        .expect("run");

        assert_eq!(output["result"]["status"], "optimal");
        assert_eq!(output["result"]["solver"], "rust:bounded-enumeration-pool");
        assert_eq!(
            output["result"]["solutions"]
                .as_array()
                .expect("solutions")
                .len(),
            3
        );
        assert_eq!(output["result"]["solutions"][0]["objective"], 3.0);
    }
}
