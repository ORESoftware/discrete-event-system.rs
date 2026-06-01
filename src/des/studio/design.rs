//! OpenMDAO-style design studies for studio JSON specs.
//!
//! A studio diagram already gives us connected components and runnable signal
//! flow. This module adds the next small layer: declare tunable block
//! parameters, objective targets, and a simple driver. The implementation is
//! intentionally finite-difference and dependency-free so it works in the SDK,
//! the static workbench, and tests without pulling in an optimizer stack.

use serde_json::{json, Value};

use super::run::run;
use super::spec::{demo_from_spec, StudioSpecError};

const DEFAULT_ITERATIONS: usize = 24;
const DEFAULT_STEP: f64 = 0.2;
const DEFAULT_EPS: f64 = 1.0e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct StudioDesignVariable {
    pub id: String,
    pub block: String,
    pub op: usize,
    pub field: String,
    pub lower: f64,
    pub upper: f64,
    pub initial: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudioDesignObjective {
    pub id: String,
    pub block: String,
    pub target: f64,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudioDesignDriver {
    pub iterations: usize,
    pub step: f64,
    pub eps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudioDesignStudy {
    pub variables: Vec<StudioDesignVariable>,
    pub objectives: Vec<StudioDesignObjective>,
    pub driver: StudioDesignDriver,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StudioDesignRun {
    pub study: StudioDesignStudy,
    pub initial_objective: f64,
    pub final_objective: f64,
    pub trace: Vec<Value>,
    pub final_spec: Value,
}

impl StudioDesignRun {
    pub fn to_json(&self) -> Value {
        json!({
            "driver": {
                "kind": "finite-difference-gradient-descent",
                "iterations": self.study.driver.iterations,
                "step": self.study.driver.step,
                "eps": self.study.driver.eps,
            },
            "variables": self.study.variables.iter().map(|v| json!({
                "id": v.id,
                "block": v.block,
                "op": v.op,
                "field": v.field,
                "lower": v.lower,
                "upper": v.upper,
                "value": read_variable(&self.final_spec, v).unwrap_or(0.0),
            })).collect::<Vec<_>>(),
            "objectives": self.study.objectives.iter().map(|o| json!({
                "id": o.id,
                "block": o.block,
                "target": o.target,
                "weight": o.weight,
            })).collect::<Vec<_>>(),
            "initialObjective": self.initial_objective,
            "finalObjective": self.final_objective,
            "trace": self.trace,
        })
    }
}

/// Run the declared design study, if the spec has a `design` block.
///
/// The returned `final_spec` is a clone of the input with optimized parameter
/// values written back into the declared block cells.
pub fn run_design_study(spec: &Value) -> Result<Option<StudioDesignRun>, StudioSpecError> {
    let study = match parse_design_study(spec)? {
        Some(study) => study,
        None => return Ok(None),
    };

    let mut current = spec.clone();
    for variable in &study.variables {
        let initial = variable
            .initial
            .or_else(|| read_variable(&current, variable))
            .ok_or_else(|| {
                StudioSpecError::new(format!(
                    "design variable `{}` points at a missing numeric parameter",
                    variable.id
                ))
            })?;
        write_variable(
            &mut current,
            variable,
            clamp(initial, variable.lower, variable.upper),
        )?;
    }

    let initial_objective = evaluate_objective(&current, &study.objectives)?;
    let mut trace = vec![trace_row(0, initial_objective, &current, &study.variables)?];
    let mut objective = initial_objective;

    for iter in 1..=study.driver.iterations {
        let mut gradients = Vec::with_capacity(study.variables.len());
        for variable in &study.variables {
            gradients.push(finite_difference_gradient(
                &current,
                variable,
                &study.objectives,
                study.driver.eps,
            )?);
        }

        for (variable, gradient) in study.variables.iter().zip(gradients.iter()) {
            let x = read_variable(&current, variable).ok_or_else(|| {
                StudioSpecError::new(format!(
                    "design variable `{}` points at a missing numeric parameter",
                    variable.id
                ))
            })?;
            let next = clamp(
                x - study.driver.step * gradient,
                variable.lower,
                variable.upper,
            );
            write_variable(&mut current, variable, next)?;
        }

        objective = evaluate_objective(&current, &study.objectives)?;
        trace.push(trace_row(iter, objective, &current, &study.variables)?);
    }

    Ok(Some(StudioDesignRun {
        study,
        initial_objective,
        final_objective: objective,
        trace,
        final_spec: current,
    }))
}

fn parse_design_study(spec: &Value) -> Result<Option<StudioDesignStudy>, StudioSpecError> {
    let Some(design) = spec.get("design") else {
        return Ok(None);
    };
    let obj = design
        .as_object()
        .ok_or_else(|| StudioSpecError::new("studio design block must be an object"))?;
    let variables_value = obj
        .get("variables")
        .and_then(Value::as_array)
        .ok_or_else(|| StudioSpecError::new("studio design requires `variables` array"))?;
    let objectives_value = obj
        .get("objectives")
        .and_then(Value::as_array)
        .ok_or_else(|| StudioSpecError::new("studio design requires `objectives` array"))?;

    if variables_value.is_empty() {
        return Err(StudioSpecError::new(
            "studio design requires at least one variable",
        ));
    }
    if objectives_value.is_empty() {
        return Err(StudioSpecError::new(
            "studio design requires at least one objective",
        ));
    }

    let mut variables = Vec::with_capacity(variables_value.len());
    for (idx, value) in variables_value.iter().enumerate() {
        variables.push(parse_variable(value, idx)?);
    }

    let mut objectives = Vec::with_capacity(objectives_value.len());
    for (idx, value) in objectives_value.iter().enumerate() {
        objectives.push(parse_objective(value, idx)?);
    }

    let driver = parse_driver(obj.get("driver"))?;

    Ok(Some(StudioDesignStudy {
        variables,
        objectives,
        driver,
    }))
}

fn parse_variable(value: &Value, idx: usize) -> Result<StudioDesignVariable, StudioSpecError> {
    let obj = value.as_object().ok_or_else(|| {
        StudioSpecError::new(format!("design.variables[{idx}] must be an object"))
    })?;
    let id = read_str(obj, "id").unwrap_or("var").to_string();
    let block = read_str(obj, "block")
        .ok_or_else(|| {
            StudioSpecError::new(format!("design.variables[{idx}] requires string `block`"))
        })?
        .to_string();
    let op = read_usize_obj(obj, "op").unwrap_or(0);
    let field = read_str(obj, "field")
        .ok_or_else(|| {
            StudioSpecError::new(format!("design.variables[{idx}] requires string `field`"))
        })?
        .to_string();
    let lower = read_f64_obj(obj, "lower")
        .or_else(|| read_f64_obj(obj, "lo"))
        .unwrap_or(f64::NEG_INFINITY);
    let upper = read_f64_obj(obj, "upper")
        .or_else(|| read_f64_obj(obj, "hi"))
        .unwrap_or(f64::INFINITY);
    if lower > upper {
        return Err(StudioSpecError::new(format!(
            "design variable `{id}` has lower bound greater than upper bound"
        )));
    }

    Ok(StudioDesignVariable {
        id,
        block,
        op,
        field,
        lower,
        upper,
        initial: read_f64_obj(obj, "initial"),
    })
}

fn parse_objective(value: &Value, idx: usize) -> Result<StudioDesignObjective, StudioSpecError> {
    let obj = value.as_object().ok_or_else(|| {
        StudioSpecError::new(format!("design.objectives[{idx}] must be an object"))
    })?;
    let block = read_str(obj, "block")
        .ok_or_else(|| {
            StudioSpecError::new(format!("design.objectives[{idx}] requires string `block`"))
        })?
        .to_string();
    let id = read_str(obj, "id").unwrap_or(&block).to_string();
    Ok(StudioDesignObjective {
        id,
        block,
        target: read_f64_obj(obj, "target").unwrap_or(0.0),
        weight: read_f64_obj(obj, "weight").unwrap_or(1.0),
    })
}

fn parse_driver(value: Option<&Value>) -> Result<StudioDesignDriver, StudioSpecError> {
    let Some(value) = value else {
        return Ok(default_driver());
    };
    let obj = value
        .as_object()
        .ok_or_else(|| StudioSpecError::new("design.driver must be an object"))?;
    Ok(StudioDesignDriver {
        iterations: read_usize_obj(obj, "iterations").unwrap_or(DEFAULT_ITERATIONS),
        step: read_f64_obj(obj, "step").unwrap_or(DEFAULT_STEP).max(0.0),
        eps: read_f64_obj(obj, "eps")
            .unwrap_or(DEFAULT_EPS)
            .abs()
            .max(f64::EPSILON),
    })
}

fn default_driver() -> StudioDesignDriver {
    StudioDesignDriver {
        iterations: DEFAULT_ITERATIONS,
        step: DEFAULT_STEP,
        eps: DEFAULT_EPS,
    }
}

fn evaluate_objective(
    spec: &Value,
    objectives: &[StudioDesignObjective],
) -> Result<f64, StudioSpecError> {
    let mut demo = demo_from_spec(spec)?;
    let steps = demo.steps;
    let dt = demo.dt;
    let run_out = run(&mut demo.compiled, steps, dt);
    let mut score = 0.0;
    for objective in objectives {
        let value = run_out.final_value(&objective.block).ok_or_else(|| {
            StudioSpecError::new(format!(
                "design objective `{}` references unknown block `{}`",
                objective.id, objective.block
            ))
        })?;
        let err = value - objective.target;
        score += objective.weight * err * err;
    }
    Ok(score)
}

fn finite_difference_gradient(
    spec: &Value,
    variable: &StudioDesignVariable,
    objectives: &[StudioDesignObjective],
    eps: f64,
) -> Result<f64, StudioSpecError> {
    let x = read_variable(spec, variable).ok_or_else(|| {
        StudioSpecError::new(format!(
            "design variable `{}` points at a missing numeric parameter",
            variable.id
        ))
    })?;
    let lo = variable.lower;
    let hi = variable.upper;
    let xp = clamp(x + eps, lo, hi);
    let xm = clamp(x - eps, lo, hi);

    if (xp - xm).abs() <= f64::EPSILON {
        return Ok(0.0);
    }

    let mut plus = spec.clone();
    let mut minus = spec.clone();
    write_variable(&mut plus, variable, xp)?;
    write_variable(&mut minus, variable, xm)?;
    let fp = evaluate_objective(&plus, objectives)?;
    let fm = evaluate_objective(&minus, objectives)?;
    Ok((fp - fm) / (xp - xm))
}

fn trace_row(
    iteration: usize,
    objective: f64,
    spec: &Value,
    variables: &[StudioDesignVariable],
) -> Result<Value, StudioSpecError> {
    let mut values = serde_json::Map::new();
    for variable in variables {
        values.insert(
            variable.id.clone(),
            json!(read_variable(spec, variable).ok_or_else(|| {
                StudioSpecError::new(format!(
                    "design variable `{}` points at a missing numeric parameter",
                    variable.id
                ))
            })?),
        );
    }
    Ok(json!({
        "iteration": iteration,
        "objective": objective,
        "variables": values,
    }))
}

fn read_variable(spec: &Value, variable: &StudioDesignVariable) -> Option<f64> {
    spec.get("blocks")?
        .as_array()?
        .iter()
        .find(|block| block.get("id").and_then(Value::as_str) == Some(variable.block.as_str()))?
        .get("cell")?
        .as_array()?
        .get(variable.op)?
        .get(&variable.field)?
        .as_f64()
}

fn write_variable(
    spec: &mut Value,
    variable: &StudioDesignVariable,
    value: f64,
) -> Result<(), StudioSpecError> {
    let blocks = spec
        .get_mut("blocks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| StudioSpecError::new("studio design requires mutable `blocks` array"))?;
    let block = blocks
        .iter_mut()
        .find(|block| block.get("id").and_then(Value::as_str) == Some(variable.block.as_str()))
        .ok_or_else(|| {
            StudioSpecError::new(format!(
                "design variable `{}` references unknown block `{}`",
                variable.id, variable.block
            ))
        })?;
    let cell = block
        .get_mut("cell")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            StudioSpecError::new(format!(
                "design variable `{}` block `{}` has no cell array",
                variable.id, variable.block
            ))
        })?;
    let op = cell.get_mut(variable.op).ok_or_else(|| {
        StudioSpecError::new(format!(
            "design variable `{}` references missing op {} on block `{}`",
            variable.id, variable.op, variable.block
        ))
    })?;
    let op_obj = op.as_object_mut().ok_or_else(|| {
        StudioSpecError::new(format!(
            "design variable `{}` op {} on block `{}` is not an object",
            variable.id, variable.op, variable.block
        ))
    })?;
    match op_obj.get_mut(&variable.field) {
        Some(slot) if slot.as_f64().is_some() => {
            *slot = json!(value);
            Ok(())
        }
        Some(_) => Err(StudioSpecError::new(format!(
            "design variable `{}` field `{}` is not numeric",
            variable.id, variable.field
        ))),
        None => Err(StudioSpecError::new(format!(
            "design variable `{}` field `{}` does not exist",
            variable.id, variable.field
        ))),
    }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn read_str<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(Value::as_str)
}

fn read_f64_obj(obj: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(Value::as_f64)
}

fn read_usize_obj(obj: &serde_json::Map<String, Value>, key: &str) -> Option<usize> {
    obj.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    use super::super::spec::example_spec;
    use super::*;

    #[test]
    fn design_study_optimizes_example_gain() {
        let run = run_design_study(&example_spec()).unwrap().unwrap();
        assert!(run.final_objective < run.initial_objective);
        assert_eq!(run.trace.len(), run.study.driver.iterations + 1);
        let gain = read_variable(&run.final_spec, &run.study.variables[0]).unwrap();
        assert!((0.0..=6.0).contains(&gain));
    }

    #[test]
    fn design_study_rejects_bad_variable_field() {
        let mut spec = example_spec();
        spec["design"]["variables"][0]["field"] = json!("missing");
        let err = run_design_study(&spec).unwrap_err();
        assert!(err.to_string().contains("field `missing` does not exist"));
    }
}
