//! Port of `src/des/general/adapters/signal-transforms-adapter.ts`
//! (module `des::general::adapters::signal_transforms_adapter`).
//!
//! Registers Z / Laplace / Fourier transform JSON adapters with a shared
//! complex-plane accumulation animation.
//!
//! ## Conversion notes
//!
//!   * The TS file carries BOTH a `ParamSchema` and a `zod` schema; the Rust
//!     [`DESModelRegistration`] trait dropped `zodSchema` (see `des_spec` module
//!     docs), so only the `ParamSchema` is ported. The zod `.refine` cross-field
//!     rule (sequence/samples XOR expression) lives in the engine validator.
//!   * `ComplexValue {re, im}` reuses the engine struct; `zValues`/`sValues`
//!     inputs reuse [`ComplexPointInput`].
//!   * `quadrature: 'rectangular'|'trapezoid'` -> [`QuadratureRule`].
//!   * `formatComplex(z)` uses the engine default of 6 significant digits;
//!     `magnitude.toPrecision(6)` -> [`to_precision`] (copied from the engine,
//!     whose helper is private).
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; each adapter is
//! exposed via the `adapter_*()` constructors.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and `transformFrame`/`animateTransform` and their
//! helpers) is not ported, so `animate` is a no-op here.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::adapters::adapter_utils::{csv_row, validation_line, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::signal_transforms::{
    format_complex, run_fourier_transform, run_laplace_transform, run_z_transform,
    ComplexPointInput, FourierTransformParams, LaplaceTransformParams, QuadratureRule,
    TransformRunResult, ZTransformParams,
};

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

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

/// `Number.prototype.toPrecision(digits)` (display-only; copied from the engine
/// where the helper is private).
fn to_precision(x: f64, digits: usize) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    if x == 0.0 {
        return if digits <= 1 { "0".to_string() } else { format!("0.{}", "0".repeat(digits - 1)) };
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

// =============================================================================
// Shared summarize / CSV.
// =============================================================================

fn summarize_transform(title: &str, result: &TransformRunResult) -> String {
    let mut lines = vec![
        title.to_string(),
        "-".repeat(title.chars().count()),
        format!("  convention: {}", result.convention),
        format!("  samples:    {}", result.samples.len()),
        format!("  points:     {}", result.outputs.len()),
        format!(
            "  entities:   sources={} stations={} sinks={}",
            result.entity_framework.sources.len(),
            result.entity_framework.stations.len(),
            result.entity_framework.sinks.len()
        ),
        format!("  movables:   {}", result.entity_framework.movable_entities.join(", ")),
        format!("  validation: {}", validation_line(&result.validation)),
    ];
    for output in result.outputs.iter().take(6) {
        lines.push(format!(
            "  {}: {}  |.|={}",
            output.label,
            format_complex(output.value, 6),
            to_precision(output.magnitude, 6)
        ));
    }
    if result.outputs.len() > 6 {
        lines.push(format!("  ... {} more point(s)", result.outputs.len() - 6));
    }
    lines.join("\n")
}

fn write_transform_csv(result: &TransformRunResult, csv_path: &str) {
    let mut lines = vec![csv_row([
        "point_index",
        "label",
        "point_re",
        "point_im",
        "value_re",
        "value_im",
        "magnitude",
        "phase",
        "direct_reference_re",
        "direct_reference_im",
        "absolute_error",
        "samples_used",
    ])];
    for output in &result.outputs {
        lines.push(csv_row([
            output.point_index.to_string(),
            output.label.clone(),
            js_number(output.point.re),
            js_number(output.point.im),
            js_number(output.value.re),
            js_number(output.value.im),
            js_number(output.magnitude),
            js_number(output.phase),
            js_number(output.direct_reference.re),
            js_number(output.direct_reference.im),
            js_number(output.absolute_error),
            output.samples_used.to_string(),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn string_field() -> ParamSchema {
    ParamSchema::String { allowed: None, default: None, description: None }
}

fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn number_vector_schema() -> ParamSchema {
    arr(num(None, None, None, None), Some(1))
}

fn numeric_map_schema() -> ParamSchema {
    obj(vec![], vec![])
}

fn complex_point_schema() -> ParamSchema {
    obj(
        vec![
            ("label", string_field()),
            ("re", num(None, None, None, None)),
            ("im", num(None, None, None, Some(0.0))),
        ],
        vec!["re"],
    )
}

fn complex_point_array_schema() -> ParamSchema {
    arr(complex_point_schema(), Some(1))
}

/// Shared continuous-transform fields (`continuousTransformFields`).
fn continuous_transform_fields() -> Vec<(&'static str, ParamSchema)> {
    vec![
        ("samples", number_vector_schema()),
        ("expression", string_field()),
        ("constants", numeric_map_schema()),
        ("t0", num(None, None, None, Some(0.0))),
        ("t1", num(None, None, None, Some(1.0))),
        ("dt", num(Some(1e-12), None, None, Some(0.01))),
        ("quadrature", str_enum(&["rectangular", "trapezoid"], "trapezoid")),
        ("tolerance", num(Some(0.0), None, None, Some(1e-9))),
    ]
}

fn z_transform_schema() -> ParamSchema {
    obj(
        vec![
            ("sequence", number_vector_schema()),
            ("expression", string_field()),
            ("constants", numeric_map_schema()),
            ("terms", num(Some(1.0), Some(1_000_000.0), Some(true), Some(8.0))),
            ("startIndex", num(None, None, Some(true), Some(0.0))),
            ("zValues", complex_point_array_schema()),
            ("tolerance", num(Some(0.0), None, None, Some(1e-9))),
        ],
        vec!["zValues"],
    )
}

fn laplace_transform_schema() -> ParamSchema {
    let mut fields = continuous_transform_fields();
    fields.push(("sValues", complex_point_array_schema()));
    obj(fields, vec!["sValues"])
}

fn fourier_transform_schema() -> ParamSchema {
    let mut fields = continuous_transform_fields();
    fields.push(("omegaValues", number_vector_schema()));
    obj(fields, vec!["omegaValues"])
}

// =============================================================================
// Examples
// =============================================================================

fn example<P>(name: &str, model: &str, description: &str, parameters: P) -> RegistrationExample<P> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: model.to_string(),
            description: Some(description.to_string()),
            parameters,
            runtime: None,
            metadata: None,
        },
    }
}

fn cp(label: &str, re: f64, im: Option<f64>) -> ComplexPointInput {
    ComplexPointInput { label: Some(label.to_string()), re, im }
}

// =============================================================================
// z-transform
// =============================================================================

pub struct ZTransformAdapter;
pub fn adapter_z_transform() -> ZTransformAdapter {
    ZTransformAdapter
}

impl DESModelRegistration<ZTransformParams, TransformRunResult> for ZTransformAdapter {
    fn id(&self) -> &str {
        "z-transform"
    }
    fn description(&self) -> &str {
        "Finite Z-transform as source, kernel, accumulator, and sink stations exchanging movable contribution tokens."
    }
    fn schema(&self) -> ParamSchema {
        z_transform_schema()
    }
    fn run(&self, params: ZTransformParams, _runtime: &DESRuntimeConfig) -> TransformRunResult {
        run_z_transform(params)
    }
    fn summarize(&self, result: &TransformRunResult, _params: &ZTransformParams) -> String {
        summarize_transform("Z-TRANSFORM (DES)", result)
    }
    fn write_csv(&self, result: &TransformRunResult, csv_path: &str) {
        write_transform_csv(result, csv_path);
    }
    fn animate(&self, _result: &TransformRunResult, _params: &ZTransformParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<ZTransformParams>> {
        vec![example(
            "finite geometric sequence",
            "z-transform",
            "Finite geometric sequence evaluated at several z-plane points.",
            ZTransformParams {
                sequence: Some(vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]),
                start_index: Some(0),
                z_values: Some(vec![cp("z=2", 2.0, None), cp("z=1", 1.0, None), cp("z=-1", -1.0, None)]),
                ..Default::default()
            },
        )]
    }
}

// =============================================================================
// laplace-transform
// =============================================================================

pub struct LaplaceTransformAdapter;
pub fn adapter_laplace_transform() -> LaplaceTransformAdapter {
    LaplaceTransformAdapter
}

impl DESModelRegistration<LaplaceTransformParams, TransformRunResult> for LaplaceTransformAdapter {
    fn id(&self) -> &str {
        "laplace-transform"
    }
    fn description(&self) -> &str {
        "Numerical Laplace transform with function samples moving through transform kernel stations."
    }
    fn schema(&self) -> ParamSchema {
        laplace_transform_schema()
    }
    fn run(&self, params: LaplaceTransformParams, _runtime: &DESRuntimeConfig) -> TransformRunResult {
        run_laplace_transform(params)
    }
    fn summarize(&self, result: &TransformRunResult, _params: &LaplaceTransformParams) -> String {
        summarize_transform("LAPLACE TRANSFORM (DES)", result)
    }
    fn write_csv(&self, result: &TransformRunResult, csv_path: &str) {
        write_transform_csv(result, csv_path);
    }
    fn animate(&self, _result: &TransformRunResult, _params: &LaplaceTransformParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<LaplaceTransformParams>> {
        vec![example(
            "decaying exponential",
            "laplace-transform",
            "Laplace transform of exp(-a t) over a finite integration window.",
            LaplaceTransformParams {
                expression: Some("exp(-a*t)".to_string()),
                constants: Some(HashMap::from([("a".to_string(), 2.0)])),
                t0: Some(0.0),
                t1: Some(8.0),
                dt: Some(0.01),
                quadrature: Some(QuadratureRule::Trapezoid),
                s_values: Some(vec![cp("s=1", 1.0, None), cp("s=0.5+i", 0.5, Some(1.0))]),
                ..Default::default()
            },
        )]
    }
}

// =============================================================================
// fourier-transform
// =============================================================================

pub struct FourierTransformAdapter;
pub fn adapter_fourier_transform() -> FourierTransformAdapter {
    FourierTransformAdapter
}

impl DESModelRegistration<FourierTransformParams, TransformRunResult> for FourierTransformAdapter {
    fn id(&self) -> &str {
        "fourier-transform"
    }
    fn description(&self) -> &str {
        "Numerical Fourier transform using angular frequencies and movable weighted sample tokens."
    }
    fn schema(&self) -> ParamSchema {
        fourier_transform_schema()
    }
    fn run(&self, params: FourierTransformParams, _runtime: &DESRuntimeConfig) -> TransformRunResult {
        run_fourier_transform(params)
    }
    fn summarize(&self, result: &TransformRunResult, _params: &FourierTransformParams) -> String {
        summarize_transform("FOURIER TRANSFORM (DES)", result)
    }
    fn write_csv(&self, result: &TransformRunResult, csv_path: &str) {
        write_transform_csv(result, csv_path);
    }
    fn animate(&self, _result: &TransformRunResult, _params: &FourierTransformParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<FourierTransformParams>> {
        vec![example(
            "windowed sinusoid",
            "fourier-transform",
            "Fourier transform of sin(2t) on one period with angular frequency probes.",
            FourierTransformParams {
                expression: Some("sin(omega0*t)".to_string()),
                constants: Some(HashMap::from([("omega0".to_string(), 2.0)])),
                t0: Some(0.0),
                t1: Some(6.283185307179586),
                dt: Some(0.0031415926535897933),
                quadrature: Some(QuadratureRule::Trapezoid),
                omega_values: Some(vec![0.0, 1.0, 2.0, 3.0, -2.0]),
                ..Default::default()
            },
        )]
    }
}
