//! Port of `src/des/general/adapters/math-blocks-adapter.ts`
//! (module `des::general::adapters::math_blocks_adapter`).
//!
//! Registers `math-ode-blocks` / `math-heat1d-blocks` / `math-equation` JSON
//! adapters (block-diagram numerics).
//!
//! ## Conversion notes
//!
//!   * `format: 'json'|'latex'|'xml'`, `kind: 'ode'|'heat1d'`, `method:
//!     'euler'|'trapezoid'` literal unions reuse the ported
//!     [`EquationInputFormat`] / [`EquationProblemKind`] / [`IntegratorMethod`].
//!   * `state`/`derivatives` name-keyed maps -> `HashMap<String, f64>`.
//!   * The `math-equation` result is a tagged union over ode/heat1d, matched in
//!     summarize/writeCsv (`r.ode!`/`r.heat1d?` -> `Option` match).
//!
//! PORT NOTE: the numerical `des/general/math-blocks` engine is NOT ported. The
//! partially-ported `des/general/math_equation_input` exists but its
//! `ODEBlockSystemResult`/`Heat1DBlockResult`/`MathEquationResult` are TRIMMED
//! (only block-graph + validation), lacking `finalState`/`trace`/`steps`/`dx`/
//! `cfl`/`x`/`finalValues` that this adapter reads. To keep the adapter glue
//! faithful and self-contained, this file defines FULL local stub result types
//! and `unimplemented!()` `run_*` kernels. When `math-blocks` (and the full
//! `math-equation-input` result) are ported, replace these stubs and delete the
//! placeholders. The parameter types are reused from `math_equation_input`.
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; each adapter is
//! exposed via the `adapter_*()` constructors.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and the `palette`/`finiteRange`/`heatColor`/frame
//! builders) is not ported, so `animate` is a no-op here.
//!
//! PORT NOTE: the TS `run` bodies wrap the kernel in `withLogger`; the logging
//! wrapper is dropped here because the kernels are unimplemented placeholders.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::adapters::adapter_utils::{csv_row, validation_line, write_csv_lines};
use crate::des::general::des_base::validation::ValidationCheck;
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, JsonObject, JsonValue, ParamSchema,
    RegistrationExample, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::math_equation_input::{
    EquationInputFormat, EquationProblemKind, Heat1DBlockParams, IntegratorMethod,
    MathEquationInputParams, MathEquationNetwork, ODEBlockSystemParams, ODEStateSpec,
};

// =============================================================================
// PORT NOTE: full local stub result types for the unported `math-blocks` engine.
// =============================================================================

#[derive(Clone, Debug)]
pub struct ODETraceRow {
    pub tick: u64,
    pub time: f64,
    pub state: HashMap<String, f64>,
    pub derivatives: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct ODEBlockSystemResult {
    /// Insertion-ordered final state (TS `Object.entries(finalState)`).
    pub final_state: Vec<(String, f64)>,
    pub steps: u64,
    pub params: ODEBlockSystemParams,
    pub trace: Vec<ODETraceRow>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
pub struct Heat1DTraceRow {
    pub tick: u64,
    pub time: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Heat1DBlockResult {
    pub trace: Vec<Heat1DTraceRow>,
    pub steps: u64,
    pub params: Heat1DBlockParams,
    pub dx: f64,
    pub cfl: f64,
    pub x: Vec<f64>,
    pub final_values: Vec<f64>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
pub struct MathEquationResult {
    pub input_format: EquationInputFormat,
    pub kind: EquationProblemKind,
    pub network: MathEquationNetwork,
    pub ode: Option<ODEBlockSystemResult>,
    pub heat1d: Option<Heat1DBlockResult>,
    pub validation: Vec<ValidationCheck>,
}

const ENGINE_MISSING: &str =
    "math-blocks numerical engine is not ported yet (see module PORT NOTE)";

pub fn run_ode_block_system(_params: ODEBlockSystemParams) -> ODEBlockSystemResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_heat1d_block_grid(_params: Heat1DBlockParams) -> Heat1DBlockResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_math_equation_problem(_params: MathEquationInputParams) -> MathEquationResult {
    unimplemented!("{ENGINE_MISSING}")
}

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        let s = v.to_string();
        if s == "-0" {
            "0".to_string()
        } else {
            s
        }
    }
}

/// `Number.prototype.toPrecision(digits)` (display-only).
fn to_precision(x: f64, digits: usize) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    if x == 0.0 {
        return if digits <= 1 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(digits - 1))
        };
    }
    let neg = x < 0.0;
    let ax = x.abs();
    let e = ax.log10().floor() as i32;
    let body = if e < -6 || e >= digits as i32 {
        let raw = format!("{:.*e}", digits.saturating_sub(1), ax);
        match raw.split_once('e') {
            Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
            _ => raw,
        }
    } else {
        let decimals = (digits as i32 - 1 - e).max(0) as usize;
        format!("{:.*}", decimals, ax)
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

fn input_format_str(f: EquationInputFormat) -> &'static str {
    match f {
        EquationInputFormat::Json => "json",
        EquationInputFormat::Latex => "latex",
        EquationInputFormat::Xml => "xml",
    }
}

fn problem_kind_str(k: EquationProblemKind) -> &'static str {
    match k {
        EquationProblemKind::Ode => "ode",
        EquationProblemKind::Heat1d => "heat1d",
    }
}

// =============================================================================
// Shared summarize / CSV bodies.
// =============================================================================

fn ode_state_names(params: &ODEBlockSystemParams) -> Vec<String> {
    params.states.iter().map(|s| s.name.clone()).collect()
}

fn summarize_ode(r: &ODEBlockSystemResult) -> String {
    let final_state = r
        .final_state
        .iter()
        .map(|(k, v)| format!("{k}={}", to_precision(*v, 6)))
        .collect::<Vec<_>>()
        .join(", ");
    [
        "MATH ODE BLOCKS".to_string(),
        "------------------------".to_string(),
        format!(
            "  states={} steps={} dt={}",
            ode_state_names(&r.params).join(", "),
            r.steps,
            js_number(r.params.dt)
        ),
        format!("  final state: {final_state}"),
        format!("  validation: {}", validation_line(&r.validation)),
    ]
    .join("\n")
}

fn write_ode_csv(r: &ODEBlockSystemResult, csv_path: &str) {
    let names = ode_state_names(&r.params);
    let mut header = vec!["tick".to_string(), "time".to_string()];
    for n in &names {
        header.push(n.clone());
    }
    for n in &names {
        header.push(format!("d_{n}"));
    }
    let mut lines = vec![csv_row(header)];
    for row in &r.trace {
        let mut cells = vec![row.tick.to_string(), js_number(row.time)];
        for n in &names {
            cells.push(js_number(row.state[n.as_str()]));
        }
        for n in &names {
            cells.push(js_number(row.derivatives[n.as_str()]));
        }
        lines.push(csv_row(cells));
    }
    write_csv_lines(csv_path, &lines);
}

fn heat_cell_header(x: &[f64]) -> Vec<String> {
    let mut header = vec![
        "tick".to_string(),
        "time".to_string(),
        "min".to_string(),
        "max".to_string(),
        "mean".to_string(),
    ];
    for i in 0..x.len() {
        header.push(format!("cell_{i}"));
    }
    header
}

fn write_heat_csv(r: &Heat1DBlockResult, csv_path: &str) {
    let mut lines = vec![csv_row(heat_cell_header(&r.x))];
    for row in &r.trace {
        let mut cells = vec![
            row.tick.to_string(),
            js_number(row.time),
            js_number(row.min),
            js_number(row.max),
            js_number(row.mean),
        ];
        for v in &row.values {
            cells.push(js_number(*v));
        }
        lines.push(csv_row(cells));
    }
    write_csv_lines(csv_path, &lines);
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: None,
    }
}

fn string_field() -> ParamSchema {
    ParamSchema::String {
        allowed: None,
        default: None,
        description: None,
    }
}

fn string_default(default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: None,
        default: Some(default.to_string()),
        description: None,
    }
}

fn str_enum(allowed: &[&str], default: Option<&str>) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: default.map(|s| s.to_string()),
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length,
        max_length,
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn numeric_map_schema() -> ParamSchema {
    obj(vec![], vec![])
}

fn ode_state_schema() -> ParamSchema {
    obj(
        vec![
            ("name", string_field()),
            ("initial", num(None, None, None, None)),
            ("derivative", string_field()),
        ],
        vec!["name", "initial", "derivative"],
    )
}

fn ode_schema() -> ParamSchema {
    obj(
        vec![
            ("states", arr(ode_state_schema(), Some(1), Some(100))),
            ("t0", num(None, None, None, Some(0.0))),
            ("t1", num(None, None, None, None)),
            ("dt", num(Some(1e-12), None, None, None)),
            ("method", str_enum(&["euler", "trapezoid"], Some("euler"))),
            ("constants", numeric_map_schema()),
        ],
        vec!["states", "t1", "dt"],
    )
}

fn heat_schema() -> ParamSchema {
    obj(
        vec![
            ("cells", num(Some(3.0), Some(1000.0), Some(true), None)),
            ("length", num(Some(1e-12), None, None, None)),
            ("alpha", num(Some(0.0), None, None, None)),
            ("t0", num(None, None, None, Some(0.0))),
            ("t1", num(None, None, None, None)),
            ("dt", num(Some(1e-12), None, None, None)),
            ("initialExpression", string_default("sin(pi*x/length)")),
            (
                "initialValues",
                arr(num(None, None, None, None), Some(3), None),
            ),
            ("leftBoundary", num(None, None, None, None)),
            ("rightBoundary", num(None, None, None, None)),
            ("constants", numeric_map_schema()),
        ],
        vec!["cells", "length", "alpha", "t1", "dt"],
    )
}

fn equation_schema() -> ParamSchema {
    obj(
        vec![
            ("format", str_enum(&["json", "latex", "xml"], None)),
            ("kind", str_enum(&["ode", "heat1d"], None)),
            ("equation", string_field()),
            ("ode", obj(vec![], vec![])),
            ("heat1d", obj(vec![], vec![])),
            ("states", arr(ode_state_schema(), Some(1), None)),
            ("constants", numeric_map_schema()),
            ("initial", numeric_map_schema()),
            ("t0", num(None, None, None, Some(0.0))),
            ("t1", num(None, None, None, Some(1.0))),
            ("dt", num(Some(1e-12), None, None, None)),
            ("method", str_enum(&["euler", "trapezoid"], Some("euler"))),
            ("cells", num(Some(3.0), Some(1000.0), Some(true), None)),
            ("length", num(Some(1e-12), None, None, None)),
            ("alpha", num(Some(0.0), None, None, None)),
            ("initialExpression", string_field()),
            (
                "initialValues",
                arr(num(None, None, None, None), Some(3), None),
            ),
            ("leftBoundary", num(None, None, None, None)),
            ("rightBoundary", num(None, None, None, None)),
        ],
        vec!["format"],
    )
}

// =============================================================================
// Examples
// =============================================================================

fn example<P>(
    name: &str,
    model: &str,
    parameters: P,
    runtime: Option<DESRuntimeConfig>,
) -> RegistrationExample<P> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: model.to_string(),
            description: None,
            parameters,
            runtime,
            metadata: None,
        },
    }
}

fn animate_runtime() -> DESRuntimeConfig {
    DESRuntimeConfig {
        animate: Some(true),
        ..Default::default()
    }
}

// =============================================================================
// math-ode-blocks
// =============================================================================

pub struct MathODEBlocksAdapter;
pub fn adapter_math_ode_blocks() -> MathODEBlocksAdapter {
    MathODEBlocksAdapter
}

impl DESModelRegistration<ODEBlockSystemParams, ODEBlockSystemResult> for MathODEBlocksAdapter {
    fn id(&self) -> &str {
        "math-ode-blocks"
    }
    fn description(&self) -> &str {
        "ODE system assembled from stationary math blocks, integrators, expression RHS blocks, sources, and sinks."
    }
    fn schema(&self) -> ParamSchema {
        ode_schema()
    }
    fn run(
        &self,
        params: ODEBlockSystemParams,
        _runtime: &DESRuntimeConfig,
    ) -> ODEBlockSystemResult {
        // PORT NOTE: TS wraps this in `withLogger`; kernel is unimplemented.
        run_ode_block_system(params)
    }
    fn summarize(&self, result: &ODEBlockSystemResult, _params: &ODEBlockSystemParams) -> String {
        summarize_ode(result)
    }
    fn write_csv(&self, result: &ODEBlockSystemResult, csv_path: &str) {
        write_ode_csv(result, csv_path);
    }
    fn animate(
        &self,
        _result: &ODEBlockSystemResult,
        _params: &ODEBlockSystemParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<ODEBlockSystemParams>> {
        vec![example(
            "exponential decay",
            "math-ode-blocks",
            ODEBlockSystemParams {
                states: vec![ODEStateSpec {
                    name: "y".to_string(),
                    initial: 1.0,
                    derivative: "-k*y".to_string(),
                }],
                constants: HashMap::from([("k".to_string(), 1.0)]),
                t0: 0.0,
                t1: 1.0,
                dt: 0.01,
                method: IntegratorMethod::Euler,
            },
            Some(animate_runtime()),
        )]
    }
}

// =============================================================================
// math-heat1d-blocks
// =============================================================================

pub struct MathHeat1DBlocksAdapter;
pub fn adapter_math_heat1d_blocks() -> MathHeat1DBlocksAdapter {
    MathHeat1DBlocksAdapter
}

impl DESModelRegistration<Heat1DBlockParams, Heat1DBlockResult> for MathHeat1DBlocksAdapter {
    fn id(&self) -> &str {
        "math-heat1d-blocks"
    }
    fn description(&self) -> &str {
        "1D heat equation PDE as a stationary grid of cell integrators and Laplacian blocks."
    }
    fn schema(&self) -> ParamSchema {
        heat_schema()
    }
    fn run(&self, params: Heat1DBlockParams, _runtime: &DESRuntimeConfig) -> Heat1DBlockResult {
        // `initialValues` empty-array normalization to `None`, mirroring the TS.
        let normalized = Heat1DBlockParams {
            initial_values: params.initial_values.filter(|v| !v.is_empty()),
            ..params
        };
        // PORT NOTE: TS wraps this in `withLogger`; kernel is unimplemented.
        run_heat1d_block_grid(normalized)
    }
    fn summarize(&self, result: &Heat1DBlockResult, _params: &Heat1DBlockParams) -> String {
        let last = result.trace.last().expect("heat trace is non-empty");
        [
            "MATH HEAT1D BLOCKS".to_string(),
            "------------------------".to_string(),
            format!(
                "  cells={} steps={} dt={} dx={} cfl={}",
                js_number(result.params.cells),
                result.steps,
                js_number(result.params.dt),
                to_precision(result.dx, 5),
                to_precision(result.cfl, 5)
            ),
            format!(
                "  final min={} max={} mean={}",
                to_precision(last.min, 6),
                to_precision(last.max, 6),
                to_precision(last.mean, 6)
            ),
            format!("  validation: {}", validation_line(&result.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &Heat1DBlockResult, csv_path: &str) {
        write_heat_csv(result, csv_path);
    }
    fn animate(
        &self,
        _result: &Heat1DBlockResult,
        _params: &Heat1DBlockParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<Heat1DBlockParams>> {
        vec![example(
            "cooling sine pulse",
            "math-heat1d-blocks",
            Heat1DBlockParams {
                cells: 31.0,
                length: 1.0,
                alpha: 0.02,
                t0: 0.0,
                t1: 0.5,
                dt: 0.005,
                constants: HashMap::new(),
                initial_expression: "sin(pi*x/length)".to_string(),
                initial_values: None,
                left_boundary: Some(0.0),
                right_boundary: Some(0.0),
            },
            Some(animate_runtime()),
        )]
    }
}

// =============================================================================
// math-equation
// =============================================================================

pub struct MathEquationAdapter;
pub fn adapter_math_equation() -> MathEquationAdapter {
    MathEquationAdapter
}

impl DESModelRegistration<MathEquationInputParams, MathEquationResult> for MathEquationAdapter {
    fn id(&self) -> &str {
        "math-equation"
    }
    fn description(&self) -> &str {
        "Parse a math equation supplied as LaTeX, XML, or structured JSON, generate a stationary/moving block network, and solve it numerically."
    }
    fn schema(&self) -> ParamSchema {
        equation_schema()
    }
    fn run(
        &self,
        params: MathEquationInputParams,
        _runtime: &DESRuntimeConfig,
    ) -> MathEquationResult {
        // PORT NOTE: TS wraps this in `withLogger`; kernel is unimplemented.
        run_math_equation_problem(params)
    }
    fn summarize(&self, r: &MathEquationResult, _params: &MathEquationInputParams) -> String {
        let model_line = if let (EquationProblemKind::Ode, Some(ode)) = (r.kind, r.ode.as_ref()) {
            let entries = ode
                .final_state
                .iter()
                .map(|(k, v)| format!("{k}={}", to_precision(*v, 6)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("  ODE final state: {entries}")
        } else {
            let max = r
                .heat1d
                .as_ref()
                .and_then(|h| h.trace.last())
                .map(|row| to_precision(row.max, 6))
                .unwrap_or_else(|| "n/a".to_string());
            format!("  heat final max: {max}")
        };
        [
            "MATH EQUATION INPUT".to_string(),
            "------------------------".to_string(),
            format!(
                "  format={} kind={}",
                input_format_str(r.input_format),
                problem_kind_str(r.kind)
            ),
            format!(
                "  generated network: nodes={} edges={}",
                r.network.nodes.len(),
                r.network.edges.len()
            ),
            model_line,
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &MathEquationResult, csv_path: &str) {
        if matches!(r.kind, EquationProblemKind::Ode) {
            if let Some(ode) = &r.ode {
                write_ode_csv(ode, csv_path);
                return;
            }
        }
        if let Some(heat) = &r.heat1d {
            write_heat_csv(heat, csv_path);
        }
    }
    fn animate(
        &self,
        _result: &MathEquationResult,
        _params: &MathEquationInputParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<MathEquationInputParams>> {
        let mut constants = JsonObject::new();
        constants.insert("k".to_string(), JsonValue::Number(1.0));
        vec![example(
            "latex ODE decay",
            "math-equation",
            MathEquationInputParams {
                format: EquationInputFormat::Latex,
                kind: Some(EquationProblemKind::Ode),
                equation: Some("\\frac{dy}{dt} = -k y; y(0)=1".to_string()),
                constants: Some(constants),
                t1: Some(1.0),
                dt: Some(0.01),
                ..Default::default()
            },
            Some(animate_runtime()),
        )]
    }
}
