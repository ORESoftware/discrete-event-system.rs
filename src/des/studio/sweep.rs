//! Driver-style parameter sweeps for Studio models.
//!
//! This is a small OpenMDAO-inspired driver surface: take a declared design
//! variable, run the model over sampled values, and record objective/constraint
//! metrics from final block signals.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

use super::run::run;
use super::spec::{
    compile_model_spec, StudioConstraintSpec, StudioDesignVariableSpec, StudioModelSpec,
    StudioObjectiveSense, StudioObjectiveSpec, StudioSpecError, MAX_SWEEP_SAMPLES,
};

/// User-facing sweep errors.
#[derive(Clone, Debug, PartialEq)]
pub enum StudioSweepError {
    MissingDesignVariable(String),
    InvalidDesignVariable(String),
    InvalidParam {
        block: String,
        param: String,
    },
    MissingMetric {
        kind: &'static str,
        name: String,
        block: String,
    },
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
            StudioSweepError::MissingMetric { kind, name, block } => {
                write!(
                    f,
                    "{kind} `{name}` could not read a final signal from block `{block}`"
                )
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
    if dv.samples > MAX_SWEEP_SAMPLES {
        return Err(StudioSweepError::InvalidDesignVariable(format!(
            "design variable `{}` may request at most {MAX_SWEEP_SAMPLES} samples",
            dv.name
        )));
    }
    if dv.lower > dv.upper {
        return Err(StudioSweepError::InvalidDesignVariable(format!(
            "design variable `{}` lower bound exceeds upper bound",
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
    let sample_values = sampled_values(&dv)?;
    let _ = compile_model_spec(spec)?;

    let mut cases = Vec::new();
    for value in sample_values {
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
            .map(|objective| {
                objective_value(objective, &final_signals).ok_or_else(|| {
                    StudioSweepError::MissingMetric {
                        kind: "objective",
                        name: objective.name.clone(),
                        block: objective.block.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let constraints = spec
            .constraints
            .iter()
            .map(|constraint| {
                constraint_value(constraint, &final_signals).ok_or_else(|| {
                    StudioSweepError::MissingMetric {
                        kind: "constraint",
                        name: constraint.name.clone(),
                        block: constraint.block.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
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

fn sweep_metric(case: &StudioSweepCase) -> (String, f64) {
    if let Some(objective) = case.objectives.first() {
        return (objective.name.clone(), objective.value);
    }
    case.final_signals
        .iter()
        .next()
        .map(|(name, value)| (name.clone(), *value))
        .unwrap_or_else(|| ("metric".to_string(), 0.0))
}

fn scale_x(value: f64, min: f64, span: f64) -> f64 {
    54.0 + ((value - min) / span) * 560.0
}

fn scale_y(value: f64, min: f64, span: f64) -> f64 {
    206.0 - ((value - min) / span) * 154.0
}

fn path_from_points(points: &[(f64, f64)]) -> String {
    points
        .iter()
        .enumerate()
        .map(|(idx, (x, y))| {
            let op = if idx == 0 { "M" } else { "L" };
            format!("{op} {x:.3} {y:.3}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn sweep_frames(sweep: &StudioSweepResult) -> Vec<Value> {
    if sweep.cases.is_empty() {
        return vec![json!({
            "t": 0.0,
            "case": 0.0,
            "shapes": [
                {"kind":"text","x":320.0,"y":120.0,"text":"No sweep cases","anchor":"middle","fontSize":18.0,"fill":"#64748b"}
            ],
            "caption": "No sweep cases"
        })];
    }

    let xs: Vec<f64> = sweep.cases.iter().map(|case| case.value).collect();
    let ys: Vec<f64> = sweep
        .cases
        .iter()
        .map(|case| sweep_metric(case).1)
        .collect();
    let min_x = xs.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    let max_x = xs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let min_y = ys.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    let max_y = ys
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let span_x = (max_x - min_x).abs().max(1e-9);
    let span_y = (max_y - min_y).abs().max(1e-9);
    let metric_name = sweep_metric(&sweep.cases[0]).0;

    sweep
        .cases
        .iter()
        .enumerate()
        .map(|(idx, case)| {
            let (_, metric) = sweep_metric(case);
            let feasible = case.constraints.iter().all(|constraint| constraint.satisfied);
            let points: Vec<(f64, f64)> = sweep.cases[..=idx]
                .iter()
                .map(|sample| {
                    let (_, y) = sweep_metric(sample);
                    (
                        scale_x(sample.value, min_x, span_x),
                        scale_y(y, min_y, span_y),
                    )
                })
                .collect();
            let mut shapes = vec![
                json!({"kind":"rect","x":0.0,"y":0.0,"w":680.0,"h":260.0,"fill":"#ffffff","stroke":"#e2e8f0","strokeWidth":1.0}),
                json!({"kind":"line","x1":54.0,"y1":206.0,"x2":614.0,"y2":206.0,"stroke":"#cad4df","strokeWidth":1.0}),
                json!({"kind":"line","x1":54.0,"y1":52.0,"x2":54.0,"y2":206.0,"stroke":"#cad4df","strokeWidth":1.0}),
                json!({"kind":"text","x":54.0,"y":34.0,"text":metric_name,"fontSize":12.0,"fill":"#475569"}),
                json!({"kind":"text","x":614.0,"y":232.0,"text":sweep.design_variable.name,"anchor":"end","fontSize":12.0,"fill":"#475569"}),
                json!({"kind":"path","d":path_from_points(&points),"fill":"none","stroke":"#2563eb","strokeWidth":2.4}),
            ];
            for (sample_idx, sample) in sweep.cases[..=idx].iter().enumerate() {
                let (_, y) = sweep_metric(sample);
                let is_best = sweep.best_case_index == Some(sample_idx);
                shapes.push(json!({
                    "kind":"circle",
                    "x": scale_x(sample.value, min_x, span_x),
                    "y": scale_y(y, min_y, span_y),
                    "r": if is_best { 6.0 } else if sample_idx == idx { 5.0 } else { 3.5 },
                    "fill": if is_best { "#16a34a" } else if sample_idx == idx { "#2563eb" } else { "#93c5fd" },
                    "stroke": if sample.constraints.iter().all(|constraint| constraint.satisfied) { "#14532d" } else { "#991b1b" },
                    "strokeWidth": if sample.constraints.iter().all(|constraint| constraint.satisfied) { 1.0 } else { 2.0 }
                }));
            }
            shapes.push(json!({
                "kind":"text",
                "x":334.0,
                "y":22.0,
                "text":format!("case {}: {} = {:.3}, {} = {:.3}", idx + 1, sweep.design_variable.name, case.value, metric_name, metric),
                "anchor":"middle",
                "fontSize":13.0,
                "fill":"#0f172a"
            }));
            json!({
                "t": idx as f64,
                "case": idx as f64,
                "designValue": case.value,
                "metric": metric,
                "feasible": if feasible { 1.0 } else { 0.0 },
                "shapes": shapes,
                "caption": format!(
                    "case {} / {}: {} = {:.3}, {} = {:.3}",
                    idx + 1,
                    sweep.cases.len(),
                    sweep.design_variable.name,
                    case.value,
                    metric_name,
                    metric
                )
            })
        })
        .collect()
}

impl StudioSweepResult {
    /// Render this sweep through the standard animated player.
    pub fn to_artifact(&self, title: &str) -> RunArtifact {
        let results = serde_json::to_value(self).unwrap_or(Value::Null);
        let summary = format!(
            "Swept `{}` across {} case(s){}.",
            self.design_variable.name,
            self.cases.len(),
            self.best_case_index
                .map(|idx| format!("; best feasible case index {idx}"))
                .unwrap_or_default()
        );
        RunArtifact::sim(
            "studio-sweep",
            title,
            "Studio design-variable sweep rendered as an animated driver trace.",
            sweep_frames(self),
            results,
            vec![UiControl::range(
                "speed",
                "Speed (fps)",
                1.0,
                30.0,
                1.0,
                8.0,
            )],
            &summary,
        )
    }
}

/// Run a named design-variable sweep and return an immediately renderable player artifact.
pub fn run_design_sweep_artifact(
    spec: &StudioModelSpec,
    design_variable_name: &str,
) -> Result<RunArtifact, StudioSweepError> {
    let sweep = run_design_sweep(spec, design_variable_name)?;
    Ok(sweep.to_artifact(&format!("{} Sweep: {}", spec.name, design_variable_name)))
}

/// Run the first declared design-variable sweep as a player artifact, if present.
pub fn run_first_design_sweep_artifact(
    spec: &StudioModelSpec,
) -> Result<Option<RunArtifact>, StudioSweepError> {
    run_first_design_sweep(spec).map(|sweep| {
        sweep.map(|result| {
            result.to_artifact(&format!(
                "{} Sweep: {}",
                spec.name, result.design_variable.name
            ))
        })
    })
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

    #[test]
    fn sweep_rejects_reversed_bounds() {
        let mut spec = starter_model_spec();
        spec.design_variables[0].lower = 2.0;
        spec.design_variables[0].upper = 0.0;
        let err = run_design_sweep(&spec, "gain.k").unwrap_err();
        assert!(err.to_string().contains("lower bound exceeds upper bound"));
    }

    #[test]
    fn sweep_rejects_oversized_sample_count() {
        let mut spec = starter_model_spec();
        spec.design_variables[0].samples = MAX_SWEEP_SAMPLES + 1;
        let err = run_design_sweep(&spec, "gain.k").unwrap_err();
        assert!(err.to_string().contains("may request at most"));
    }

    #[test]
    fn sweep_rejects_unknown_objective_block() {
        let mut spec = starter_model_spec();
        spec.objectives[0].block = "missing".to_string();
        let err = run_design_sweep(&spec, "gain.k").unwrap_err();
        assert!(err.to_string().contains("objective"));
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn sweep_renders_as_animated_player_artifact() {
        let artifact = run_design_sweep_artifact(&starter_model_spec(), "gain.k").unwrap();
        assert_eq!(artifact.kind, "studio-sweep");
        assert_eq!(artifact.frames.len(), 9);
        assert_eq!(artifact.results["designVariable"]["name"], "gain.k");
        assert!(artifact
            .to_player_html()
            .contains("ramp-gain-sink Sweep: gain.k"));
    }
}
