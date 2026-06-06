use des_engine::des::general::lp::{
    solve_lp_internal, solve_lp_internal_ipm, InternalInteriorPointOptions, InternalSimplexOptions,
    LPProblem, LPStatus, Sense,
};
use serde_json::{json, Value};
use std::io::{self, Read};

#[derive(Debug)]
struct CliError(String);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceLpMethod {
    InternalSimplex,
    InternalIpm,
}

impl ReferenceLpMethod {
    fn from_cli(method: &str) -> Result<Self, CliError> {
        let normalized = method.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "" | "rust" | "fallback" | "internal" | "internal-simplex" | "simplex"
            | "revised-simplex" | "highs" | "highs-ds" | "highs-simplex" | "scipy:highs"
            | "scipy:highs-ds" | "scipy:simplex" | "scipy:revised-simplex" | "glop"
            | "ortools:glop" | "pdlp" | "ortools:pdlp" => Ok(Self::InternalSimplex),
            "internal-ipm" | "internal-interior-point" | "ipm" | "interior-point"
            | "highs-ipm" | "scipy:highs-ipm" | "scipy:interior-point" => {
                Ok(Self::InternalIpm)
            }
            other => Err(CliError(format!(
                "unsupported Rust LP reference method {other:?}; use rust/internal-simplex/internal-ipm"
            ))),
        }
    }

    fn solver_label(self) -> &'static str {
        match self {
            Self::InternalSimplex => "rust:internal-simplex",
            Self::InternalIpm => "rust:internal-ipm",
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn usage(program: &str) -> String {
    format!("usage: {program} [--method rust|fallback|internal|internal-simplex|internal-ipm]")
}

fn next_option_value(
    program: &str,
    flag: &str,
    inline: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    if let Some(value) = inline {
        if value.is_empty() {
            return Err(CliError(format!(
                "{flag} requires a value\n{}",
                usage(program)
            )));
        }
        return Ok(value.to_string());
    }
    args.next()
        .ok_or_else(|| CliError(format!("{flag} requires a value\n{}", usage(program))))
}

fn parse_args(
    program: &str,
    mut args: impl Iterator<Item = String>,
) -> Result<ReferenceLpMethod, CliError> {
    let mut method = "rust".to_string();
    while let Some(arg) = args.next() {
        let (flag, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(key, value)| (key, Some(value)));
        match flag {
            "--method" => method = next_option_value(program, "--method", inline, &mut args)?,
            "-h" | "--help" => return Err(CliError(usage(program))),
            other => {
                return Err(CliError(format!(
                    "unknown option {other:?}\n{}",
                    usage(program)
                )))
            }
        }
    }
    ReferenceLpMethod::from_cli(&method)
}

fn get_any<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value.get(*key))
}

fn finite_number(value: &Value, name: &str) -> Result<f64, CliError> {
    let number = value
        .as_f64()
        .ok_or_else(|| CliError(format!("{name} must be a finite number")))?;
    if !number.is_finite() {
        return Err(CliError(format!("{name} must be finite")));
    }
    Ok(number)
}

fn number_array(value: Option<&Value>, name: &str) -> Result<Vec<f64>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be an array")))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, entry)| finite_number(entry, &format!("{name}[{idx}]")))
        .collect()
}

fn optional_bound_array(
    value: Option<&Value>,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<Option<f64>>>, CliError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be an array")))?;
    if values.len() != expected_len {
        return Err(CliError(format!(
            "{name} length {} != variable count {expected_len}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            if entry.is_null() {
                Ok(None)
            } else {
                finite_number(entry, &format!("{name}[{idx}]")).map(Some)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn matrix(value: Option<&Value>, name: &str, cols: usize) -> Result<Vec<Vec<f64>>, CliError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let rows = value
        .as_array()
        .ok_or_else(|| CliError(format!("{name} must be an array of rows")))?;
    rows.iter()
        .enumerate()
        .map(|(row_idx, row_value)| {
            let row = row_value
                .as_array()
                .ok_or_else(|| CliError(format!("{name}[{row_idx}] must be an array")))?;
            if row.len() != cols {
                return Err(CliError(format!(
                    "{name}[{row_idx}] length {} != variable count {cols}",
                    row.len()
                )));
            }
            row.iter()
                .enumerate()
                .map(|(col_idx, entry)| {
                    finite_number(entry, &format!("{name}[{row_idx}][{col_idx}]"))
                })
                .collect()
        })
        .collect()
}

fn parse_linear_constraints(
    raw: &Value,
    cols: usize,
    a_ub: &mut Vec<Vec<f64>>,
    b_ub: &mut Vec<f64>,
    a_eq: &mut Vec<Vec<f64>>,
    b_eq: &mut Vec<f64>,
) -> Result<(), CliError> {
    let Some(raw_constraints) = raw.get("linear_constraints") else {
        return Ok(());
    };
    let constraints = raw_constraints
        .as_array()
        .ok_or_else(|| CliError("linear_constraints must be an array".to_string()))?;
    for (idx, constraint) in constraints.iter().enumerate() {
        let coefs = get_any(constraint, &["coefs", "coefficients", "row"])
            .ok_or_else(|| CliError(format!("linear_constraints[{idx}] needs coefs")))?;
        let row = number_array(Some(coefs), &format!("linear_constraints[{idx}].coefs"))?;
        if row.len() != cols {
            return Err(CliError(format!(
                "linear_constraints[{idx}] row length {} != variable count {cols}",
                row.len()
            )));
        }
        let lower = constraint
            .get("lower")
            .filter(|value| !value.is_null())
            .map(|value| finite_number(value, &format!("linear_constraints[{idx}].lower")))
            .transpose()?;
        let upper = constraint
            .get("upper")
            .filter(|value| !value.is_null())
            .map(|value| finite_number(value, &format!("linear_constraints[{idx}].upper")))
            .transpose()?;
        if lower.is_none() && upper.is_none() {
            return Err(CliError(format!(
                "linear_constraints[{idx}] needs lower or upper"
            )));
        }
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if lo > hi + 1e-9 {
                return Err(CliError(format!(
                    "linear_constraints[{idx}] lower exceeds upper"
                )));
            }
            if (lo - hi).abs() <= 1e-9 {
                a_eq.push(row);
                b_eq.push(hi);
                continue;
            }
        }
        if let Some(hi) = upper {
            a_ub.push(row.clone());
            b_ub.push(hi);
        }
        if let Some(lo) = lower {
            a_ub.push(row.iter().map(|coef| -coef).collect());
            b_ub.push(-lo);
        }
    }
    Ok(())
}

fn parse_lp(payload: &Value) -> Result<(LPProblem, f64), CliError> {
    let raw = payload.get("lp").unwrap_or(payload);
    let c = number_array(raw.get("c"), "c")?;
    if c.is_empty() {
        return Err(CliError(
            "c must be a non-empty objective vector".to_string(),
        ));
    }
    let n = c.len();
    let sense = match raw
        .get("sense")
        .and_then(Value::as_str)
        .unwrap_or("max")
        .to_ascii_lowercase()
        .as_str()
    {
        "min" | "minimize" | "minimise" => Sense::Min,
        "max" | "maximize" | "maximise" => Sense::Max,
        other => return Err(CliError(format!("unknown LP sense {other:?}"))),
    };

    let mut a_ub = matrix(get_any(raw, &["A_ub", "a_ub", "Aub", "aUb"]), "A_ub", n)?;
    let mut b_ub = number_array(get_any(raw, &["b_ub", "bUb", "bUB"]), "b_ub")?;
    let mut a_eq = matrix(get_any(raw, &["A_eq", "a_eq", "Aeq", "aEq"]), "A_eq", n)?;
    let mut b_eq = number_array(get_any(raw, &["b_eq", "bEq", "bEQ"]), "b_eq")?;
    if a_ub.len() != b_ub.len() {
        return Err(CliError(format!(
            "A_ub rows {} != b_ub length {}",
            a_ub.len(),
            b_ub.len()
        )));
    }
    if a_eq.len() != b_eq.len() {
        return Err(CliError(format!(
            "A_eq rows {} != b_eq length {}",
            a_eq.len(),
            b_eq.len()
        )));
    }
    parse_linear_constraints(raw, n, &mut a_ub, &mut b_ub, &mut a_eq, &mut b_eq)?;

    let objective_offset = get_any(raw, &["objective_offset", "objectiveOffset"])
        .map(|value| finite_number(value, "objective_offset"))
        .transpose()?
        .unwrap_or(0.0);

    Ok((
        LPProblem {
            sense,
            c,
            a_ub: (!a_ub.is_empty()).then_some(a_ub),
            b_ub: (!b_ub.is_empty()).then_some(b_ub),
            a_eq: (!a_eq.is_empty()).then_some(a_eq),
            b_eq: (!b_eq.is_empty()).then_some(b_eq),
            lb: optional_bound_array(raw.get("lb"), "lb", n)?,
            ub: optional_bound_array(raw.get("ub"), "ub", n)?,
            var_names: raw
                .get("var_names")
                .or_else(|| raw.get("varNames"))
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_str().unwrap_or("").to_string())
                        .collect()
                }),
            con_names: None,
        },
        objective_offset,
    ))
}

fn optional_vec(value: &Option<Vec<f64>>) -> Value {
    value.as_ref().map_or(Value::Null, |items| json!(items))
}

fn optional_string_vec(value: &Option<Vec<String>>) -> Value {
    value.as_ref().map_or(Value::Null, |items| json!(items))
}

fn solution_json(
    solution: des_engine::des::general::lp::LPSolution,
    offset: f64,
    method: ReferenceLpMethod,
) -> Value {
    let objective = if solution.status == LPStatus::Optimal && solution.objective.is_finite() {
        json!(solution.objective + offset)
    } else {
        Value::Null
    };
    let mut out = json!({
        "status": solution.status.as_str(),
        "x": if solution.status == LPStatus::Optimal { json!(solution.x) } else { json!([]) },
        "objective": objective,
        "iters": solution.iters,
        "solver": method.solver_label(),
        "message": solution.message.unwrap_or_else(|| format!("{} reference", method.solver_label())),
        "elapsedMs": solution.elapsed_ms,
    });
    if let Some(object) = out.as_object_mut() {
        object.insert("dualUB".to_string(), optional_vec(&solution.dual_ub));
        object.insert("dualEQ".to_string(), optional_vec(&solution.dual_eq));
        object.insert(
            "reducedCosts".to_string(),
            optional_vec(&solution.reduced_costs),
        );
        object.insert(
            "varBasis".to_string(),
            optional_string_vec(&solution.var_basis),
        );
        object.insert(
            "rowBasis".to_string(),
            optional_string_vec(&solution.row_basis),
        );
        if let Some(ray) = solution.unbounded_ray {
            object.insert("unboundedRay".to_string(), json!(ray));
        }
    }
    out
}

fn error_json(message: impl Into<String>) -> Value {
    json!({
        "status": "numerical-error",
        "x": [],
        "objective": null,
        "iters": null,
        "solver": "rust:internal-simplex",
        "message": message.into(),
    })
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "lp_solve_reference".to_string());
    let method = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload: Value = serde_json::from_str(stdin)
        .map_err(|err| CliError(format!("failed to parse JSON input: {err}")))?;
    let (lp, offset) = parse_lp(&payload)?;
    let solution = match method {
        ReferenceLpMethod::InternalSimplex => {
            solve_lp_internal(&lp, &InternalSimplexOptions::default())
        }
        ReferenceLpMethod::InternalIpm => {
            solve_lp_internal_ipm(&lp, &InternalInteriorPointOptions::default())
        }
    };
    Ok(solution_json(solution, offset, method))
}

fn main() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        println!("{}", error_json(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(std::env::args().collect(), &input) {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize LP output")
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

    #[test]
    fn rust_internal_simplex_solves_small_lp() {
        let output = run(
            vec!["lp_solve_reference".to_string()],
            r#"{"c":[3,2],"A_ub":[[1,1],[1,3]],"b_ub":[4,6],"sense":"max"}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:internal-simplex");
        assert_eq!(output["objective"], 12.0);
    }

    #[test]
    fn accepts_lp_payload_and_linear_row_bounds() {
        let output = run(
            vec![
                "lp_solve_reference".to_string(),
                "--method=fallback".to_string(),
            ],
            r#"{
                "lp": {
                    "sense": "min",
                    "c": [1],
                    "linear_constraints": [{"coefs": [1], "lower": 2}],
                    "objectiveOffset": 5.5
                }
            }"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["objective"], 7.5);
    }

    #[test]
    fn method_alias_can_select_native_interior_point() {
        let output = run(
            vec![
                "lp_solve_reference".to_string(),
                "--method=scipy:interior-point".to_string(),
            ],
            r#"{"c":[1,1],"A_ub":[[1,0],[0,1]],"b_ub":[4,3],"sense":"max"}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "optimal");
        assert_eq!(output["solver"], "rust:internal-ipm");
        assert!((output["objective"].as_f64().expect("objective") - 7.0).abs() < 1e-5);
    }

    #[test]
    fn unsupported_method_fails_before_python_is_needed() {
        let err = run(
            vec![
                "lp_solve_reference".to_string(),
                "--method=python-only".to_string(),
            ],
            r#"{"c":[1],"sense":"max"}"#,
        )
        .expect_err("unsupported method should fail");

        assert!(err
            .to_string()
            .contains("unsupported Rust LP reference method"));
    }

    #[test]
    fn reports_unbounded_ray() {
        let output = run(
            vec!["lp_solve_reference".to_string()],
            r#"{"c":[1],"A_ub":[[-1]],"b_ub":[1],"sense":"max"}"#,
        )
        .expect("run");

        assert_eq!(output["status"], "unbounded");
        assert!(output["unboundedRay"].as_array().is_some());
    }
}
