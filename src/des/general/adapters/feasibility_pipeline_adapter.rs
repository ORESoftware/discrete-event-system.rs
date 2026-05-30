//! Port of `src/des/general/adapters/feasibility-pipeline-adapter.ts`
//! (module `des::general::adapters::feasibility_pipeline_adapter`).
//!
//! JSON adapter for the feasibility-checker / improver pipeline.
//!
//! ## Conversion notes
//!
//!   * `problem` / `candidate` / `improvement` reuse the engine structs
//!     (`FeasibilityPipelineParams` and friends); the schema still mirrors the
//!     TS validator metadata.
//!   * `formatNumber` / `valuesSummary` are ported as free helpers.
//!   * `withLogger(runtime, fn)` -> [`with_logger`]; the structured log events
//!     are emitted via the ported [`JsonlLogger`].
//!   * The CSV uses `jsonCsvRow` (each cell `JSON.stringify`-d unless already a
//!     string): numeric cells go through [`json_num`], the `values` map and the
//!     `violations` array are serialised by [`json_values_map`] /
//!     [`json_violations`].
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the
//! adapter is exposed via [`adapter()`].
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and the `drawPipeline` scene builder) is not ported, so
//! `animate` is a no-op here.
//!
//! PORT NOTE: `FeasibilityEvaluation::values` is a `HashMap` (the TS used an
//! insertion-ordered object). `valuesSummary` / `json_values_map` sort keys for
//! a deterministic output, which can differ in ordering from the TS. Likewise
//! `json_violations` emits fields in the struct's declaration order, omitting
//! `None`s (matching `JSON.stringify`'s `undefined` omission), but exact key
//! order / number spelling may differ from the TS `JSON.stringify`.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::adapters::adapter_utils::{
    json_csv_row, validation_line, with_logger, write_csv_lines,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::feasibility_pipeline::{
    run_feasibility_pipeline, CandidateSolutionInput, ConstraintSense, FeasibilityEvaluation,
    FeasibilityImprovementOptions, FeasibilityPipelineParams, FeasibilityPipelineResult,
    FeasibilityStatus, FeasibilityViolation, LinearConstraint, LinearObjective, ObjectiveSense,
    OptimizationVariable, StructuredOptimizationProblem, VariableKind, ViolationKind,
};
use crate::des::observability::logger::JsonValue as LogJson;

// =============================================================================
// Number-formatting helpers (JS parity).
// =============================================================================

/// `String(v)` for a JS number.
fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `JSON.stringify(v)` for a number (non-finite -> `null`).
fn json_num(v: f64) -> String {
    if v.is_finite() {
        js_number(v)
    } else {
        "null".to_string()
    }
}

/// `v.toExponential(digits)` (mantissa with `digits` fraction digits).
fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

/// `function formatNumber(x)`.
fn format_number(x: f64) -> String {
    if !x.is_finite() {
        return js_number(x);
    }
    if x.abs() >= 1e9 || (x.abs() < 1e-3 && x != 0.0) {
        return to_exponential(x, 3);
    }
    format!("{x:.4}")
}

/// JSON-quote a string.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn violation_kind_str(k: ViolationKind) -> &'static str {
    match k {
        ViolationKind::Domain => "domain",
        ViolationKind::Constraint => "constraint",
    }
}

/// `JSON.stringify(values)` for the variable-name -> value map (keys sorted).
fn json_values_map(values: &HashMap<String, f64>) -> String {
    let mut keys: Vec<&String> = values.keys().collect();
    keys.sort();
    let inner: Vec<String> = keys
        .iter()
        .map(|k| format!("{}:{}", json_str(k), json_num(values[*k])))
        .collect();
    format!("{{{}}}", inner.join(","))
}

/// `JSON.stringify(violation)` (fields in declaration order, `None` omitted).
fn json_violation(v: &FeasibilityViolation) -> String {
    let mut parts: Vec<String> = vec![
        format!("\"kind\":{}", json_str(violation_kind_str(v.kind))),
        format!("\"name\":{}", json_str(&v.name)),
        format!("\"violation\":{}", json_num(v.violation)),
        format!("\"message\":{}", json_str(&v.message)),
    ];
    if let Some(x) = &v.variable {
        parts.push(format!("\"variable\":{}", json_str(x)));
    }
    if let Some(x) = &v.constraint {
        parts.push(format!("\"constraint\":{}", json_str(x)));
    }
    if let Some(x) = v.activity {
        parts.push(format!("\"activity\":{}", json_num(x)));
    }
    if let Some(x) = v.rhs {
        parts.push(format!("\"rhs\":{}", json_num(x)));
    }
    format!("{{{}}}", parts.join(","))
}

fn json_violations(vs: &[FeasibilityViolation]) -> String {
    format!("[{}]", vs.iter().map(json_violation).collect::<Vec<_>>().join(","))
}

/// `function valuesSummary(e)`.
fn values_summary(e: &FeasibilityEvaluation) -> String {
    let mut keys: Vec<&String> = e.values.keys().collect();
    keys.sort();
    keys.iter()
        .take(6)
        .map(|k| format!("{}={}", k, format_number(e.values[*k])))
        .collect::<Vec<_>>()
        .join("  ")
}

fn feasibility_status_str(s: FeasibilityStatus) -> &'static str {
    match s {
        FeasibilityStatus::Feasible => "feasible",
        FeasibilityStatus::Infeasible => "infeasible",
        FeasibilityStatus::Improved => "improved",
        FeasibilityStatus::InfeasibleImproved => "infeasible-improved",
        FeasibilityStatus::TimeLimit => "time-limit",
        FeasibilityStatus::TickLimit => "tick-limit",
    }
}

// =============================================================================
// Schema
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn str_enum(allowed: &[&str], default: Option<&str>) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: default.map(|s| s.to_string()),
        description: None,
    }
}

fn boolean(default: Option<bool>) -> ParamSchema {
    ParamSchema::Boolean { default, description: None }
}

fn array(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>, description: Option<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: description.map(|s| s.to_string()),
    }
}

fn coefficient_map_schema() -> ParamSchema {
    obj(vec![], vec![], Some("Map from variable name to finite coefficient."))
}

fn variable_schema() -> ParamSchema {
    obj(
        vec![
            ("name", string_field()),
            ("type", str_enum(&["continuous", "integer", "binary"], Some("continuous"))),
            ("lb", num(None, None, None, None)),
            ("ub", num(None, None, None, None)),
            ("step", num(Some(0.0), None, None, None)),
        ],
        vec!["name"],
        None,
    )
}

fn objective_schema() -> ParamSchema {
    obj(
        vec![
            ("constant", num(None, None, None, Some(0.0))),
            ("coefficients", coefficient_map_schema()),
        ],
        vec!["coefficients"],
        None,
    )
}

fn constraint_schema() -> ParamSchema {
    obj(
        vec![
            ("name", string_field()),
            ("coefficients", coefficient_map_schema()),
            ("sense", str_enum(&["<=", ">=", "="], None)),
            ("rhs", num(None, None, None, None)),
            ("tolerance", num(Some(0.0), None, None, None)),
        ],
        vec!["coefficients", "sense", "rhs"],
        None,
    )
}

fn problem_schema() -> ParamSchema {
    obj(
        vec![
            ("sense", str_enum(&["min", "max"], None)),
            ("variables", array(variable_schema(), Some(1))),
            ("objective", objective_schema()),
            ("constraints", array(constraint_schema(), None)),
            ("tolerance", num(Some(0.0), None, None, Some(1e-8))),
        ],
        vec!["sense", "variables", "objective"],
        None,
    )
}

fn candidate_schema() -> ParamSchema {
    obj(
        vec![
            ("id", str_enum(&[], Some("user-candidate"))),
            ("values", obj(vec![], vec![], None)),
            ("vector", array(num(None, None, None, None), None)),
        ],
        vec![],
        None,
    )
}

fn improvement_schema() -> ParamSchema {
    obj(
        vec![
            ("enabled", boolean(Some(true))),
            ("maxIterations", num(Some(0.0), None, Some(true), Some(200.0))),
            ("seed", num(None, None, Some(true), Some(1.0))),
            ("continuousStep", num(Some(0.0), None, None, Some(1.0))),
            ("integerStep", num(Some(0.0), None, None, Some(1.0))),
            ("penalty", num(Some(0.0), None, None, Some(1_000_000.0))),
            ("allowRepair", boolean(Some(true))),
        ],
        vec![],
        None,
    )
}

/// `const feasibilitySchema`.
pub fn feasibility_schema() -> ParamSchema {
    obj(
        vec![
            ("problem", problem_schema()),
            ("candidate", candidate_schema()),
            ("improvement", improvement_schema()),
            ("timeLimitMs", num(Some(0.0), None, None, Some(180000.0))),
            ("maxTicks", num(Some(1.0), None, Some(true), None)),
            ("checkEveryTicks", num(Some(1.0), None, Some(true), Some(1.0))),
        ],
        vec!["problem", "candidate"],
        Some("Check a user candidate for a structured optimization problem and optionally improve it internally."),
    )
}

/// `const adapter`.
pub struct FeasibilityPipelineAdapter;

/// Construct the adapter (see the module PORT NOTE on registration).
pub fn adapter() -> FeasibilityPipelineAdapter {
    FeasibilityPipelineAdapter
}

impl DESModelRegistration<FeasibilityPipelineParams, FeasibilityPipelineResult>
    for FeasibilityPipelineAdapter
{
    fn id(&self) -> &str {
        "feasibility-pipeline"
    }

    fn description(&self) -> &str {
        "General optimization feasibility checker and internal improvement pipeline."
    }

    fn schema(&self) -> ParamSchema {
        feasibility_schema()
    }

    fn run(
        &self,
        params: FeasibilityPipelineParams,
        runtime: &DESRuntimeConfig,
    ) -> FeasibilityPipelineResult {
        let variables = params.problem.variables.len();
        let constraints = params.problem.constraints.as_ref().map(|c| c.len()).unwrap_or(0);
        with_logger(runtime, move |mut logger| {
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("feasibility-pipeline-start".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("variables".to_string(), LogJson::Number(variables as f64)),
                    ("constraints".to_string(), LogJson::Number(constraints as f64)),
                ]));
            }
            let result = run_feasibility_pipeline(params);
            let stride = (result.trace.len() / 50).max(1);
            let mut i = 0;
            while i < result.trace.len() {
                let row = &result.trace[i];
                if let Some(l) = logger.as_deref_mut() {
                    l.log(LogJson::Object(vec![
                        ("kind".to_string(), LogJson::String("feasibility-pipeline-trace".to_string())),
                        ("level".to_string(), LogJson::String("debug".to_string())),
                        ("candidateId".to_string(), LogJson::String(row.candidate_id.clone())),
                        ("objective".to_string(), LogJson::Number(row.objective_value)),
                        ("feasible".to_string(), LogJson::Bool(row.feasible)),
                        ("totalViolation".to_string(), LogJson::Number(row.total_violation)),
                    ]));
                }
                i += stride;
            }
            if let Some(l) = logger {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("feasibility-pipeline-finish".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("status".to_string(), LogJson::String(feasibility_status_str(result.status).to_string())),
                    ("bestCandidate".to_string(), LogJson::String(result.best.candidate_id.clone())),
                    ("feasible".to_string(), LogJson::Bool(result.best.feasible)),
                    ("objective".to_string(), LogJson::Number(result.best.objective_value)),
                    ("totalViolation".to_string(), LogJson::Number(result.best.total_violation)),
                ]));
            }
            result
        })
    }

    fn summarize(
        &self,
        result: &FeasibilityPipelineResult,
        _params: &FeasibilityPipelineParams,
    ) -> String {
        [
            "FEASIBILITY PIPELINE".to_string(),
            "------------------------".to_string(),
            format!(
                "  status={} trace={} improvements={}",
                feasibility_status_str(result.status),
                result.trace.len(),
                result.improvements.len()
            ),
            format!(
                "  initial feasible={} objective={} violation={}",
                result.initial.feasible,
                format_number(result.initial.objective_value),
                format_number(result.initial.total_violation)
            ),
            format!(
                "  best    feasible={} objective={} violation={} candidate={}",
                result.best.feasible,
                format_number(result.best.objective_value),
                format_number(result.best.total_violation),
                result.best.candidate_id
            ),
            format!(
                "  wall-clock={} / {} ms checks={}",
                format_number(result.wall_clock.elapsed_ms),
                format_number(result.wall_clock.budget_ms),
                result.wall_clock.checks
            ),
            format!(
                "  network stationary={} moving={} edges={}",
                result.network.stationary_entities.len(),
                result.network.moving_entities.len(),
                result.network.edges.len()
            ),
            format!("  validation: {}", validation_line(&result.validation)),
            format!("  values: {}", values_summary(&result.best)),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &FeasibilityPipelineResult, csv_path: &str) {
        let mut lines = vec![
            "candidate_id,parent_id,iteration,origin,objective,comparable_objective,feasible,total_violation,max_violation,values,violations"
                .to_string(),
        ];
        for row in &result.trace {
            lines.push(json_csv_row([
                row.candidate_id.clone(),
                row.parent_id.clone().unwrap_or_default(),
                row.iteration.to_string(),
                row.origin.as_str().to_string(),
                json_num(row.objective_value),
                json_num(row.comparable_objective),
                row.feasible.to_string(),
                json_num(row.total_violation),
                json_num(row.max_violation),
                json_values_map(&row.values),
                json_violations(&row.violations),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn animate(
        &self,
        _result: &FeasibilityPipelineResult,
        _params: &FeasibilityPipelineParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }

    fn examples(&self) -> Vec<RegistrationExample<FeasibilityPipelineParams>> {
        let mut candidate_values: HashMap<String, f64> = HashMap::new();
        candidate_values.insert("x0".to_string(), 1.0);
        candidate_values.insert("x1".to_string(), 1.0);
        candidate_values.insert("x2".to_string(), 0.0);

        let problem = StructuredOptimizationProblem {
            sense: ObjectiveSense::Max,
            variables: vec![
                OptimizationVariable { name: "x0".to_string(), kind: Some(VariableKind::Binary), lb: None, ub: None, step: None },
                OptimizationVariable { name: "x1".to_string(), kind: Some(VariableKind::Binary), lb: None, ub: None, step: None },
                OptimizationVariable { name: "x2".to_string(), kind: Some(VariableKind::Binary), lb: None, ub: None, step: None },
            ],
            objective: LinearObjective {
                constant: None,
                coefficients: vec![("x0".to_string(), 60.0), ("x1".to_string(), 100.0), ("x2".to_string(), 120.0)],
            },
            constraints: Some(vec![LinearConstraint {
                name: Some("capacity".to_string()),
                coefficients: vec![("x0".to_string(), 10.0), ("x1".to_string(), 20.0), ("x2".to_string(), 30.0)],
                sense: ConstraintSense::Le,
                rhs: 50.0,
                tolerance: None,
            }]),
            tolerance: None,
        };

        vec![RegistrationExample {
            name: "repair and improve a binary knapsack candidate".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "feasibility-pipeline".to_string(),
                description: None,
                parameters: FeasibilityPipelineParams {
                    problem,
                    candidate: CandidateSolutionInput { id: None, values: Some(candidate_values), vector: None },
                    improvement: Some(FeasibilityImprovementOptions {
                        enabled: Some(true),
                        max_iterations: Some(60),
                        seed: Some(4),
                        continuous_step: None,
                        integer_step: Some(1.0),
                        penalty: None,
                        allow_repair: None,
                    }),
                    time_limit_ms: None,
                    max_ticks: None,
                    check_every_ticks: None,
                },
                runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
                metadata: None,
            },
        }]
    }
}
