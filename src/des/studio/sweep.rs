//! Driver-style parameter sweeps for Studio models.
//!
//! This is a small OpenMDAO-inspired driver surface: take a declared design
//! variable, run the model over sampled values, and record objective/constraint
//! metrics from final block signals.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::run::run;
use super::spec::{
    compile_model_spec, StudioConstraintSpec, StudioDesignVariableSpec, StudioModelSpec,
    StudioObjectiveSense, StudioObjectiveSpec, StudioSpecError,
};

/// User-facing sweep errors.
#[derive(Clone, Debug, PartialEq)]
pub enum StudioSweepError {
    MissingDesignVariable(String),
    InvalidDesignVariable(String),
    InvalidParam { block: String, param: String },
    Compile(StudioSpecError),
}

impl std::fmt::Display for StudioSweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StudioSweepError::MissingDesignVariable(name) => {
                write!(f, "unknown design variable `{name}`")
            }
            StudioSweepError::InvalidDesignVariable(msg) => write!(f, "{msg}"),
            StudioSweepError::InvalidParam { block, param } => {
                write!(f, "block `{block}` has no numeric parameter `{param}`")
            }
            StudioSweepError::Compile(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StudioSweepError {}

impl From<StudioSpecError> for StudioSweepError {
    fn from(value: StudioSpecError) -> Self {
        StudioSweepError::Compile(value)
    }
}

/// One objective value sampled during a sweep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioObjectiveValue {
    pub name: String,
    pub block: String,
    pub sense: StudioObjectiveSense,
    pub value: f64,
    pub target: Option<f64>,
    pub error: Option<f64>,
}

/// One constraint evaluation sampled during a sweep.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioConstraintValue {
    pub name: String,
    pub block: String,
    pub value: f64,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub satisfied: bool,
}

/// One sampled model run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioSweepCase {
    pub value: f64,
    pub final_signals: BTreeMap<String, f64>,
    pub objectives: Vec<StudioObjectiveValue>,
    pub constraints: Vec<StudioConstraintValue>,
}

/// Sweep result suitable for UI plots, reports, or JSON APIs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioSweepResult {
    pub design_variable: StudioDesignVariableSpec,
    pub cases: Vec<StudioSweepCase>,
    pub best_case_index: Option<usize>,
}

fn sampled_values(dv: &StudioDesignVariableSpec) -> Result<Vec<f64>, StudioSweepError> {
    if !dv.lower.is_finite() || !dv.upper.is_finite() {
        return Err(StudioSweepError::InvalidDesignVariable(format!(
            "design variable `{}` bounds must be finite",
            dv.name
        )));
    }
    if dv.samples == 0 {
        return Err(StudioSweepError::InvalidDesignVariable(format!(
            "design variable `{}` must request at least one sample",
            dv.name
        )));
    }
    if dv.samples == 1 {
        return Ok(vec![dv.lower]);
    }
    let denom = (dv.samples - 1) as f64;
    Ok((0..dv.samples)
        .map(|i| dv.lower + (dv.upper - dv.lower) * (i as f64 / denom))
        .collect())
}

fn set_design_value(
    spec: &mut StudioModelSpec,
    dv: &StudioDesignVariableSpec,
    value: f64,
) -> Result<(), StudioSweepError> {
    let Some(block) = spec.blocks.iter_mut().find(|b| b.id == dv.block) else {
        return Err(StudioSweepError::InvalidParam {
            block: dv.block.clone(),
            param: dv.param.clone(),
        });
    };
    match block.params.get(&dv.param) {
        Some(Value::Number(_)) | None => {
            block.params.insert(dv.param.clone(), Value::from(value));
            Ok(())
        }
        _ => Err(StudioSweepError::InvalidParam {
            block: dv.block.clone(),
            param: dv.param.clone(),
        }),
    }
}

fn objective_value(
    objective: &StudioObjectiveSpec,
    signals: &BTreeMap<String, f64>,
) -> Option<StudioObjectiveValue> {
    let value = *signals.get(&objective.block)?;
    Some(StudioObjectiveValue {
        name: objective.name.clone(),
        block: objective.block.clone(),
        sense: objective.sense,
        value,
        target: objective.target,
        error: objective.target.map(|target| value - target),
    })
}

fn constraint_value(
    constraint: &StudioConstraintSpec,
    signals: &BTreeMap<String, f64>,
) -> Option<StudioConstraintValue> {
    let value = *signals.get(&constraint.block)?;
    let above_lower = constraint.lower.map(|lo| value >= lo).unwrap_or(true);
    let below_upper = constraint.upper.map(|hi| value <= hi).unwrap_or(true);
    Some(StudioConstraintValue {
        name: constraint.name.clone(),
        block: constraint.block.clone(),
        value,
        lower: constraint.lower,
        upper: constraint.upper,
        satisfied: above_lower && below_upper,
    })
}

fn objective_score(case: &StudioSweepCase) -> Option<f64> {
    let first = case.objectives.first()?;
    Some(match first.sense {
        StudioObjectiveSense::Minimize => first.value,
        StudioObjectiveSense::Maximize => -first.value,
        StudioObjectiveSense::Track => first.error.unwrap_or(first.value).abs(),
    })
}

/// Run a sweep for a named design variable.
pub fn run_design_sweep(
    spec: &StudioModelSpec,
    design_variable_name: &str,
) -> Result<StudioSweepResult, StudioSweepError> {
    let dv = spec
        .design_variables
        .iter()
        .find(|v| v.name == design_variable_name)
        .cloned()
        .ok_or_else(|| StudioSweepError::MissingDesignVariable(design_variable_name.to_string()))?;

    let mut cases = Vec::new();
    for value in sampled_values(&dv)? {
        let mut case_spec = spec.clone();
        set_design_value(&mut case_spec, &dv, value)?;
        let mut compiled = compile_model_spec(&case_spec)?;
        let run_out = run(&mut compiled, case_spec.steps, case_spec.dt);
        let final_signals: BTreeMap<String, f64> = run_out
            .node_ids
            .iter()
            .filter_map(|id| run_out.final_value(id).map(|v| (id.clone(), v)))
            .collect();
        let objectives = spec
            .objectives
            .iter()
            .filter_map(|objective| objective_value(objective, &final_signals))
            .collect();
        let constraints = spec
            .constraints
            .iter()
            .filter_map(|constraint| constraint_value(constraint, &final_signals))
            .collect();
        cases.push(StudioSweepCase {
            value,
            final_signals,
            objectives,
            constraints,
        });
    }

    let best_case_index = cases
        .iter()
        .enumerate()
        .filter(|(_, case)| case.constraints.iter().all(|c| c.satisfied))
        .filter_map(|(idx, case)| objective_score(case).map(|score| (idx, score)))
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx);

    Ok(StudioSweepResult {
        design_variable: dv,
        cases,
        best_case_index,
    })
}

/// Run the first declared design variable, if one exists.
pub fn run_first_design_sweep(
    spec: &StudioModelSpec,
) -> Result<Option<StudioSweepResult>, StudioSweepError> {
    let Some(dv) = spec.design_variables.first() else {
        return Ok(None);
    };
    run_design_sweep(spec, &dv.name).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::studio::starter_model_spec;

    #[test]
    fn sweep_runs_declared_design_variable_and_selects_track_target() {
        let spec = starter_model_spec();
        let sweep = run_design_sweep(&spec, "gain.k").unwrap();
        assert_eq!(sweep.cases.len(), 9);
        assert_eq!(sweep.best_case_index, Some(2));
        assert!((sweep.cases[2].value - 0.5).abs() < 1e-9);
        assert!((sweep.cases[2].objectives[0].value - 3.95).abs() < 1e-9);
    }
}
