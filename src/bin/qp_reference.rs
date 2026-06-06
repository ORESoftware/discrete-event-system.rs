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
        "usage: {program} [--solver auto|rust|rust:qp-active-set|fallback|highs|highs:qp|scipy|scipy:slsqp|osqp|cvxpy|scs|clarabel|ecos|mosek|copt|qpoases|proxqp|cosmo|sdpa|csdp] [--max-enumerations N]"
    )
}

fn parse_solver(value: &str) -> Result<ExternalQuadraticReferenceSolver, CliError> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "auto" => Ok(ExternalQuadraticReferenceSolver::Auto),
        "rust"
        | "rust-internal"
        | "rust:internal"
        | "rust-active-set"
        | "rust:active-set"
        | "rust-qp-active-set"
        | "rust:qp-active-set"
        | "rust-quadratic-reference"
        | "rust:quadratic-reference"
        | "rust-miqp-enumeration"
        | "rust:miqp-enumeration"
        | "rust-socp-pattern-search"
        | "rust:socp-pattern-search"
        | "rust-qcp-pattern-search"
        | "rust:qcp-pattern-search" => Ok(ExternalQuadraticReferenceSolver::RustInternal),
        "fallback"
        | "rust-fallback"
        | "rust:fallback"
        | "builtin"
        | "builtin-qp-active-set"
        | "builtin:qp-active-set" => Ok(ExternalQuadraticReferenceSolver::Fallback),
        "highs" | "highspy" | "highs-qp" | "highs:qp" | "highs-quadratic" | "highs:quadratic" => {
            Ok(ExternalQuadraticReferenceSolver::Highs)
        }
        "scipy" | "scipy-slsqp" | "scipy:slsqp" => Ok(ExternalQuadraticReferenceSolver::Scipy),
        "osqp" | "osqp-qp" | "osqp:qp" | "cvxpy-osqp" | "cvxpy:osqp" => {
            Ok(ExternalQuadraticReferenceSolver::Osqp)
        }
        "cvxpy" | "cvxpy-default" | "cvxpy:default" => Ok(ExternalQuadraticReferenceSolver::Cvxpy),
        "scs" | "cvxpy-scs" | "cvxpy:scs" => Ok(ExternalQuadraticReferenceSolver::Scs),
        "clarabel" | "cvxpy-clarabel" | "cvxpy:clarabel" => {
            Ok(ExternalQuadraticReferenceSolver::Clarabel)
        }
        "ecos" | "cvxpy-ecos" | "cvxpy:ecos" => Ok(ExternalQuadraticReferenceSolver::Ecos),
        "mosek" | "cvxpy-mosek" | "cvxpy:mosek" => Ok(ExternalQuadraticReferenceSolver::Mosek),
        "copt" | "cvxpy-copt" | "cvxpy:copt" => Ok(ExternalQuadraticReferenceSolver::Copt),
        "qpoases" | "cvxpy-qpoases" | "cvxpy:qpoases" => {
            Ok(ExternalQuadraticReferenceSolver::Qpoases)
        }
        "proxqp" | "cvxpy-proxqp" | "cvxpy:proxqp" => Ok(ExternalQuadraticReferenceSolver::Proxqp),
        "cosmo" | "cvxpy-cosmo" | "cvxpy:cosmo" => Ok(ExternalQuadraticReferenceSolver::Cosmo),
        "sdpa" | "cvxpy-sdpa" | "cvxpy:sdpa" => Ok(ExternalQuadraticReferenceSolver::Sdpa),
        "csdp" | "cvxpy-csdp" | "cvxpy:csdp" => Ok(ExternalQuadraticReferenceSolver::Csdp),
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

fn first_array<'a>(
    raw: &'a Value,
    keys: &[&'static str],
) -> Option<(&'static str, &'a Vec<Value>)> {
    keys.iter().find_map(|key| {
        raw.get(*key)
            .and_then(Value::as_array)
            .map(|items| (*key, items))
    })
}

fn has_non_empty_array(raw: &Value, keys: &[&'static str]) -> bool {
    first_array(raw, keys).is_some_and(|(_, items)| !items.is_empty())
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

fn optional_finite_number(
    value: Option<&Value>,
    name: impl Into<String>,
    default: f64,
) -> Result<f64, CliError> {
    let Some(value) = value else {
        return Ok(default);
    };
    if value.is_null() {
        return Ok(default);
    }
    finite_number(value, name)
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

fn is_number_like(value: &Value) -> bool {
    value.as_f64().is_some()
        || value
            .as_str()
            .and_then(|text| text.parse::<f64>().ok())
            .is_some()
}

fn sparse_index(value: &Value, name: impl Into<String>, n: usize) -> Result<usize, CliError> {
    let name = name.into();
    let raw = value
        .as_u64()
        .or_else(|| value.as_str().and_then(|text| text.parse::<u64>().ok()))
        .ok_or_else(|| CliError(format!("{name} must be a non-negative integer")))?;
    let idx = usize::try_from(raw).map_err(|_| CliError(format!("{name} is too large")))?;
    if idx >= n {
        return Err(CliError(format!("{name} index {idx} is outside 0..{n}")));
    }
    Ok(idx)
}

fn dense_or_sparse_row(value: Option<&Value>, name: &str, n: usize) -> Result<Vec<f64>, CliError> {
    let Some(value) = value else {
        return Ok(vec![0.0; n]);
    };
    if value.is_null() {
        return Ok(vec![0.0; n]);
    }
    let items = value.as_array().ok_or_else(|| {
        CliError(format!(
            "{name} must be a dense row or sparse coefficient list"
        ))
    })?;
    if items.len() == n && items.iter().all(is_number_like) {
        return numbers(Some(value), name);
    }

    let mut row = vec![0.0; n];
    for (entry_idx, entry) in items.iter().enumerate() {
        let (idx, coef) = if let Some(pair) = entry.as_array() {
            if pair.len() != 2 {
                return Err(CliError(format!(
                    "{name}[{entry_idx}] sparse tuple must be [index, coefficient]"
                )));
            }
            (
                sparse_index(&pair[0], format!("{name}[{entry_idx}][0]"), n)?,
                finite_number(&pair[1], format!("{name}[{entry_idx}][1]"))?,
            )
        } else {
            let idx_value = get_any(entry, &["i", "idx", "index", "var"]).ok_or_else(|| {
                CliError(format!("{name}[{entry_idx}] sparse object missing index"))
            })?;
            let coef_value = get_any(entry, &["coeff", "coef", "coefficient", "value"])
                .ok_or_else(|| {
                    CliError(format!(
                        "{name}[{entry_idx}] sparse object missing coefficient"
                    ))
                })?;
            (
                sparse_index(idx_value, format!("{name}[{entry_idx}].index"), n)?,
                finite_number(coef_value, format!("{name}[{entry_idx}].coefficient"))?,
            )
        };
        row[idx] += coef;
    }
    Ok(row)
}

fn objective_sign(raw: &Value) -> f64 {
    match get_any(raw, &["sense"])
        .and_then(Value::as_str)
        .unwrap_or("min")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "max" | "maximize" | "maximise" => -1.0,
        _ => 1.0,
    }
}

fn scale_vector(values: Vec<f64>, scale: f64) -> Vec<f64> {
    values.into_iter().map(|value| value * scale).collect()
}

fn scale_matrix(values: Vec<Vec<f64>>, scale: f64) -> Vec<Vec<f64>> {
    values
        .into_iter()
        .map(|row| scale_vector(row, scale))
        .collect()
}

fn quadratic_matrix(
    raw: &Value,
    name: &str,
    n: usize,
    objective: bool,
    scale: f64,
) -> Result<Vec<Vec<f64>>, CliError> {
    let mut matrix = scale_matrix(
        square_matrix_or_zero(get_any(raw, &["Q", "q"]), name, n)?,
        scale,
    );
    let Some(terms) = get_any(raw, &["quadratic"]) else {
        return Ok(matrix);
    };
    if terms.is_null() {
        return Ok(matrix);
    }
    let terms = terms
        .as_array()
        .ok_or_else(|| CliError(format!("{name}.quadratic must be a list")))?;
    for (term_idx, term) in terms.iter().enumerate() {
        let i = sparse_index(
            get_any(term, &["i", "var_i", "varI"])
                .ok_or_else(|| CliError(format!("{name}.quadratic[{term_idx}].i missing")))?,
            format!("{name}.quadratic[{term_idx}].i"),
            n,
        )?;
        let j = sparse_index(
            get_any(term, &["j", "var_j", "varJ"])
                .ok_or_else(|| CliError(format!("{name}.quadratic[{term_idx}].j missing")))?,
            format!("{name}.quadratic[{term_idx}].j"),
            n,
        )?;
        let coeff = scale
            * finite_number(
                get_any(term, &["coeff", "coef", "coefficient", "value"]).ok_or_else(|| {
                    CliError(format!("{name}.quadratic[{term_idx}].coeff missing"))
                })?,
                format!("{name}.quadratic[{term_idx}].coeff"),
            )?;
        if objective {
            if i == j {
                matrix[i][i] += 2.0 * coeff;
            } else {
                matrix[i][j] += coeff;
                matrix[j][i] += coeff;
            }
        } else if i == j {
            matrix[i][i] += coeff;
        } else {
            let half = 0.5 * coeff;
            matrix[i][j] += half;
            matrix[j][i] += half;
        }
    }
    Ok(matrix)
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
    let sign = objective_sign(raw);
    let c = scale_vector(numbers(get_any(raw, &["c"]), "c")?, sign);
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    Ok(QuadraticProgram {
        q: quadratic_matrix(raw, "Q", n, true, sign)?,
        c,
        a_ub: Some(matrix(
            get_any(raw, &["A_ub", "a_ub", "A", "a"]),
            "A_ub",
            Some(n),
        )?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub", "b"]), "b_ub")?),
        a_eq: Some(matrix(get_any(raw, &["A_eq", "a_eq"]), "A_eq", Some(n))?),
        b_eq: Some(optional_numbers(get_any(raw, &["b_eq"]), "b_eq")?),
        lb: optional_bounds(get_any(raw, &["lb"]), "lb", n, Some(Some(0.0)))?,
        ub: optional_bounds(get_any(raw, &["ub"]), "ub", n, Some(None))?,
        var_names: string_list(get_any(raw, &["var_names", "varNames"]), "var_names")?,
    })
}

fn parse_soc_cone(
    item: &Value,
    n: usize,
    idx: usize,
    cone_key: &str,
) -> Result<SecondOrderCone, CliError> {
    let cone_name = format!("{cone_key}[{idx}]");
    let name = get_any(item, &["name"])
        .and_then(Value::as_str)
        .map(str::to_string);
    if get_any(item, &["A", "a"]).is_some() {
        return Ok(SecondOrderCone {
            a: matrix(
                get_any(item, &["A", "a"]),
                &format!("{cone_name}.A"),
                Some(n),
            )?,
            b: optional_numbers(get_any(item, &["b"]), &format!("{cone_name}.b"))?,
            c: dense_or_sparse_row(
                get_any(item, &["c", "rhsCoeffs", "rhs_coeffs"]),
                &format!("{cone_name}.c"),
                n,
            )?,
            d: optional_finite_number(
                get_any(item, &["d", "rhsConstant", "rhs_constant"]),
                format!("{cone_name}.d"),
                0.0,
            )?,
            name,
        });
    }

    let terms = get_any(item, &["terms"])
        .and_then(Value::as_array)
        .ok_or_else(|| CliError(format!("{cone_name}.terms must be a list")))?;
    let mut a = Vec::with_capacity(terms.len());
    let mut b = Vec::with_capacity(terms.len());
    for (term_idx, term) in terms.iter().enumerate() {
        a.push(dense_or_sparse_row(
            get_any(term, &["coeffs", "coefficients", "row", "a"]),
            &format!("{cone_name}.terms[{term_idx}].coeffs"),
            n,
        )?);
        b.push(optional_finite_number(
            get_any(term, &["constant", "b", "offset"]),
            format!("{cone_name}.terms[{term_idx}].constant"),
            0.0,
        )?);
    }
    Ok(SecondOrderCone {
        a,
        b,
        c: dense_or_sparse_row(
            get_any(item, &["rhsCoeffs", "rhs_coeffs", "c"]),
            &format!("{cone_name}.rhsCoeffs"),
            n,
        )?,
        d: optional_finite_number(
            get_any(item, &["rhsConstant", "rhs_constant", "d"]),
            format!("{cone_name}.rhsConstant"),
            0.0,
        )?,
        name,
    })
}

fn parse_socp(raw: &Value) -> Result<SecondOrderConeProgram, CliError> {
    let sign = objective_sign(raw);
    let c = scale_vector(numbers(get_any(raw, &["c"]), "c")?, sign);
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    let (cone_key, cone_items) = first_array(raw, &["cones", "soc"])
        .ok_or_else(|| CliError("cones/soc must be a list".to_string()))?;
    let mut cones = Vec::with_capacity(cone_items.len());
    for (idx, item) in cone_items.iter().enumerate() {
        cones.push(parse_soc_cone(item, n, idx, cone_key)?);
    }
    Ok(SecondOrderConeProgram {
        c,
        a_ub: Some(matrix(
            get_any(raw, &["A_ub", "a_ub", "A", "a"]),
            "A_ub",
            Some(n),
        )?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub", "b"]), "b_ub")?),
        a_eq: Some(matrix(get_any(raw, &["A_eq", "a_eq"]), "A_eq", Some(n))?),
        b_eq: Some(optional_numbers(get_any(raw, &["b_eq"]), "b_eq")?),
        lb: optional_bounds(get_any(raw, &["lb"]), "lb", n, Some(None))?,
        ub: optional_bounds(get_any(raw, &["ub"]), "ub", n, Some(None))?,
        cones,
        var_names: string_list(get_any(raw, &["var_names", "varNames"]), "var_names")?,
    })
}

fn parse_qcp(raw: &Value) -> Result<QuadraticallyConstrainedProgram, CliError> {
    let sign = objective_sign(raw);
    let c = scale_vector(numbers(get_any(raw, &["c"]), "c")?, sign);
    let n = c.len();
    if n == 0 {
        return Err(CliError("c must be non-empty".to_string()));
    }
    let (_, raw_constraints) = first_array(
        raw,
        &[
            "quadratic_constraints",
            "q_constraints",
            "quadraticConstraints",
        ],
    )
    .ok_or_else(|| CliError("quadratic_constraints must be a list".to_string()))?;
    let mut quadratic_constraints = Vec::with_capacity(raw_constraints.len());
    for (idx, item) in raw_constraints.iter().enumerate() {
        let row_name = format!("quadratic_constraints[{idx}]");
        let q = quadratic_matrix(item, &format!("{row_name}.Q"), n, false, 1.0)?;
        let c = if get_any(item, &["c"]).is_some() {
            numbers(get_any(item, &["c"]), &format!("{row_name}.c"))?
        } else {
            dense_or_sparse_row(get_any(item, &["linear"]), &format!("{row_name}.linear"), n)?
        };
        let rhs = finite_number(
            get_any(item, &["rhs"]).ok_or_else(|| CliError(format!("{row_name}.rhs missing")))?,
            format!("{row_name}.rhs"),
        )?;
        let sense = get_any(item, &["sense"])
            .and_then(Value::as_str)
            .unwrap_or("<=")
            .trim();
        let name = get_any(item, &["name"])
            .and_then(Value::as_str)
            .map(str::to_string);
        if sense == ">=" {
            quadratic_constraints.push(QuadraticConstraint {
                q: scale_matrix(q, -1.0),
                c: scale_vector(c, -1.0),
                rhs: -rhs,
                name,
            });
        } else if sense == "=" || sense == "==" {
            quadratic_constraints.push(QuadraticConstraint {
                q: q.clone(),
                c: c.clone(),
                rhs,
                name: name.clone(),
            });
            quadratic_constraints.push(QuadraticConstraint {
                q: scale_matrix(q, -1.0),
                c: scale_vector(c, -1.0),
                rhs: -rhs,
                name: name.map(|value| format!("{value}__eq_lower")),
            });
        } else {
            quadratic_constraints.push(QuadraticConstraint { q, c, rhs, name });
        }
    }
    Ok(QuadraticallyConstrainedProgram {
        q: quadratic_matrix(raw, "Q", n, true, sign)?,
        c,
        a_ub: Some(matrix(
            get_any(raw, &["A_ub", "a_ub", "A", "a"]),
            "A_ub",
            Some(n),
        )?),
        b_ub: Some(optional_numbers(get_any(raw, &["b_ub", "b"]), "b_ub")?),
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

fn solution_with_original_sense_objective(
    source: &Value,
    mut solution: ExternalQuadraticReferenceSolution,
) -> ExternalQuadraticReferenceSolution {
    let sign = objective_sign(source);
    if sign != 1.0 {
        if let Some(objective) = solution.objective {
            if objective.is_finite() {
                solution.objective = Some(objective / sign);
            }
        }
    }
    solution
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
    let finish = |solution| Ok(solution_with_original_sense_objective(source, solution));
    let opts = ExternalQuadraticReferenceOptions {
        solver: args.solver,
        max_enumerations: args.max_enumerations,
    };
    let has_integer = get_any(source, &["integer_vars", "integerVars"])
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    let has_cones = has_non_empty_array(source, &["cones", "soc"]);
    let has_qcp = has_non_empty_array(
        source,
        &[
            "quadratic_constraints",
            "q_constraints",
            "quadraticConstraints",
        ],
    );

    if has_integer && has_cones {
        let socp = parse_socp(source)?;
        let integer_vars = bools(
            get_any(source, &["integer_vars", "integerVars"]),
            "integer_vars",
        )?;
        return finish(solve_misocp_with_external_reference(
            &MixedIntegerSecondOrderConeProgram { socp, integer_vars },
            &opts,
        ));
    }
    if has_integer && has_qcp {
        let qcp = parse_qcp(source)?;
        let integer_vars = bools(
            get_any(source, &["integer_vars", "integerVars"]),
            "integer_vars",
        )?;
        return finish(solve_miqcp_with_external_reference(
            &MixedIntegerQuadraticallyConstrainedProgram { qcp, integer_vars },
            &opts,
        ));
    }
    if has_integer {
        let qp = parse_qp(source)?;
        let integer_vars = bools(
            get_any(source, &["integer_vars", "integerVars"]),
            "integer_vars",
        )?;
        return finish(solve_miqp_with_external_reference(
            &MixedIntegerQuadraticProgram { qp, integer_vars },
            &opts,
        ));
    }
    if has_cones {
        return finish(solve_socp_with_external_reference(
            &parse_socp(source)?,
            &opts,
        ));
    }
    if has_qcp {
        return finish(solve_qcp_with_external_reference(
            &parse_qcp(source)?,
            &opts,
        ));
    }
    finish(solve_qp_with_external_reference(&parse_qp(source)?, &opts))
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
    fn parser_accepts_cvxpy_prefixed_solver_aliases() {
        for (raw, expected) in [
            ("cvxpy-osqp", ExternalQuadraticReferenceSolver::Osqp),
            ("cvxpy:osqp", ExternalQuadraticReferenceSolver::Osqp),
            ("cvxpy-scs", ExternalQuadraticReferenceSolver::Scs),
            ("cvxpy:clarabel", ExternalQuadraticReferenceSolver::Clarabel),
            ("cvxpy-ecos", ExternalQuadraticReferenceSolver::Ecos),
            ("cvxpy:mosek", ExternalQuadraticReferenceSolver::Mosek),
            ("cvxpy-copt", ExternalQuadraticReferenceSolver::Copt),
            ("cvxpy:qpoases", ExternalQuadraticReferenceSolver::Qpoases),
            ("cvxpy-proxqp", ExternalQuadraticReferenceSolver::Proxqp),
            ("cvxpy:cosmo", ExternalQuadraticReferenceSolver::Cosmo),
            ("cvxpy-sdpa", ExternalQuadraticReferenceSolver::Sdpa),
            ("cvxpy:csdp", ExternalQuadraticReferenceSolver::Csdp),
        ] {
            assert_eq!(parse_solver(raw).unwrap(), expected, "{raw}");
        }
    }

    #[test]
    fn parser_accepts_rust_and_external_solver_labels_used_by_validation_tools() {
        for raw in [
            "rust:internal",
            "rust:active-set",
            "rust:qp-active-set",
            "rust:quadratic-reference",
            "rust:miqp-enumeration",
            "rust:socp-pattern-search",
            "rust:qcp-pattern-search",
        ] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalQuadraticReferenceSolver::RustInternal,
                "{raw}"
            );
        }
        for raw in ["rust:fallback", "builtin:qp-active-set"] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalQuadraticReferenceSolver::Fallback,
                "{raw}"
            );
        }
        for raw in ["highs:qp", "highs:quadratic"] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalQuadraticReferenceSolver::Highs,
                "{raw}"
            );
        }
        for raw in ["scipy:slsqp", "ScIpY_SlSqP"] {
            assert_eq!(
                parse_solver(raw).unwrap(),
                ExternalQuadraticReferenceSolver::Scipy,
                "{raw}"
            );
        }
        for raw in ["osqp:qp", "cvxpy:default"] {
            assert!(parse_solver(raw).is_ok(), "{raw}");
        }
    }

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
    fn rust_qp_cli_accepts_sparse_quadratic_objective_terms() {
        let output = run(
            vec!["qp_reference".to_string(), "--solver=fallback".to_string()],
            r#"{
                "c": [-2.0, 0.0],
                "quadratic": [
                    {"i": 0, "j": 0, "coeff": 1.0}
                ],
                "lb": [0.0, 0.0],
                "ub": [2.0, 2.0]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:qp-active-set");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }

    #[test]
    fn rust_qp_cli_reports_original_objective_for_max_sense_sparse_terms() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver=fallback".to_string(),
                "--max-enumerations=100".to_string(),
            ],
            r#"{
                "sense": "max",
                "c": [2.0, 0.0],
                "quadratic": [
                    {"i": 0, "j": 0, "coeff": -1.0}
                ],
                "lb": [0.0, 0.0],
                "ub": [2.0, 2.0]
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:qp-active-set");
        let objective = output["objective"].as_f64().expect("objective");
        assert!((objective - 1.0).abs() < 1e-9);
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
    fn rust_miqp_cli_accepts_dense_a_b_and_camel_case_integer_aliases() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver=rust".to_string(),
                "--max-enumerations=100".to_string(),
            ],
            r#"{
                "Q": [[2.0, 0.0], [0.0, 2.0]],
                "c": [-2.8, -1.2],
                "A": [[-1.0, -1.0]],
                "b": [-1.5],
                "lb": [0.0, 0.0],
                "ub": [3.0, 3.0],
                "integerVars": [true, false]
            }"#,
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
    fn rust_misocp_cli_accepts_wrapped_sparse_soc_aliases() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver=fallback".to_string(),
                "--max-enumerations=100".to_string(),
            ],
            r#"{
                "problem": {
                    "c": [1.0, 0.0],
                    "lb": [0.0, 0.0],
                    "ub": [2.0, 2.0],
                    "integerVars": [true, false],
                    "soc": [
                        {
                            "terms": [
                                {"coeffs": [[1, 1.0]], "constant": 0.0}
                            ],
                            "rhsCoeffs": [],
                            "rhsConstant": 1.0
                        }
                    ]
                }
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:misocp-enumeration");
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

    #[test]
    fn rust_miqcp_cli_accepts_wrapped_sparse_quadratic_constraints_alias() {
        let output = run(
            vec![
                "qp_reference".to_string(),
                "--solver=fallback".to_string(),
                "--max-enumerations=100".to_string(),
            ],
            r#"{
                "problem": {
                    "Q": [[0.0, 0.0], [0.0, 0.0]],
                    "c": [-1.0, 0.0],
                    "lb": [0.0, 0.0],
                    "ub": [2.0, 2.0],
                    "integerVars": [true, false],
                    "quadraticConstraints": [
                        {
                            "quadratic": [
                                {"i": 0, "j": 0, "coeff": 1.0},
                                {"i": 1, "j": 1, "coeff": 1.0}
                            ],
                            "linear": [],
                            "sense": "<=",
                            "rhs": 1.0
                        }
                    ]
                }
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:miqcp-enumeration");
        assert_eq!(output["x"].as_array().expect("x").len(), 2);
    }
}
