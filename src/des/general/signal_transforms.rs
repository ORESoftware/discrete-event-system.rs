//! Port of `src/des/general/signal-transforms.ts`
//! (module `des::general::signal_transforms`).
//!
//! Z, Laplace, Fourier, DFT, wavelet, and Mellin transforms expressed as DES
//! station graphs. A run is not a monolithic numerical helper: it is built from
//! the entity vocabulary used across the project. A sample-source station emits
//! one token per input sample, a kernel station turns each sample into a
//! per-evaluation-point contribution token, an accumulator station sums
//! contributions per point, and a result sink keeps the final totals. Samples,
//! contributions, and totals are movable tokens; the stations own only local
//! state and talk over named channels.
//!
//! Radix-2 FFT and Radon transforms are exposed from the same module as
//! transform engines with the same output vocabulary, but they are not forced
//! through the contribution-token graph: FFT is a butterfly computation, while
//! Radon consumes 2-D grid cells rather than scalar 1-D samples.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `ComplexValue {re, im}` and the `complex*` helpers are kept as a local
//!     plain `f64` struct plus free fns (FLAGGED: the header suggested
//!     `num_complex::Complex<f64>`, but that crate is not a dependency of this
//!     workspace; a transform value is a Tier-1 numerical-kernel quantity that
//!     stays `f64`, and the local struct mirrors the TS interface exactly).
//!     `ComplexPoint extends ComplexValue` becomes a struct that carries `re`,
//!     `im`, and a `label`, exposing `as_complex()`.
//!   * `TransformKind` / `QuadratureRule` string unions become enums; the `'n' |
//!     't'` abscissa tag becomes [`AbscissaName`].
//!   * The `*Token` classes become plain structs carried as `Rc<dyn Any>` (there
//!     is no `Token` trait in the ported `station.rs`); the `*Station` classes
//!     become `struct { core: StationCore, … }` + `impl DESStation`.
//!   * The per-point kernel closure is a plain `fn` pointer (every kernel here is
//!     a free function), so it can be shared by the kernel station and the
//!     direct-reference recomputation without a non-`Clone` boxed closure.
//!   * `Preconditions` / `finiteComplex` / `validateZPoints` `throw` on bad
//!     input → guards whose `Err` is turned into a `panic!` (an invariant);
//!     `intrinsicCheck` station validators map to [`intrinsic_check`] with a
//!     downcasting predicate.
//!   * `parse` / `evaluate` come from the ported [`crate::des::general::expr`];
//!     `constants: Record<string, number>` → `HashMap<String, f64>`.
//!   * `formatComplex` reproduces JS `Number.toPrecision` for display only (the
//!     string is never compared numerically).
//!   * Fully deterministic: no RNG/clock.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{
    run_iterative_des, IterativeRunOptions, IterativeRunSummary,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{intrinsic_check, ValidationCheck};
use crate::des::general::expr::{evaluate, parse, Env};

/// Panic with the precondition message on a failed guard (TS `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// ── Enums (string unions) ─────────────────────────────────────────────────────

/// Transform family identifier used by summaries, adapters, and control-system
/// descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    Z,
    Laplace,
    Fourier,
    Dft,
    Fft,
    Wavelet,
    Mellin,
    Radon,
}

impl TransformKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TransformKind::Z => "z",
            TransformKind::Laplace => "laplace",
            TransformKind::Fourier => "fourier",
            TransformKind::Dft => "dft",
            TransformKind::Fft => "fft",
            TransformKind::Wavelet => "wavelet",
            TransformKind::Mellin => "mellin",
            TransformKind::Radon => "radon",
        }
    }
}

/// `'rectangular' | 'trapezoid'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadratureRule {
    Rectangular,
    Trapezoid,
}

/// `'n' | 't'` abscissa tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbscissaName {
    N,
    T,
}

impl AbscissaName {
    pub fn as_str(self) -> &'static str {
        match self {
            AbscissaName::N => "n",
            AbscissaName::T => "t",
        }
    }
}

// ── Complex value ─────────────────────────────────────────────────────────────

/// `ComplexValue {re, im}`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ComplexValue {
    pub re: f64,
    pub im: f64,
}

/// `ComplexPointInput { label?, re, im? }`.
#[derive(Clone, Debug)]
pub struct ComplexPointInput {
    pub label: Option<String>,
    pub re: f64,
    pub im: Option<f64>,
}

/// `ComplexPoint extends ComplexValue { label }`.
#[derive(Clone, Debug)]
pub struct ComplexPoint {
    pub re: f64,
    pub im: f64,
    pub label: String,
}

impl ComplexPoint {
    fn as_complex(&self) -> ComplexValue {
        ComplexValue {
            re: self.re,
            im: self.im,
        }
    }
}

fn complex(re: f64, im: f64) -> ComplexValue {
    ComplexValue { re, im }
}

fn complex_add(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

fn complex_sub(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

fn complex_mul(a: ComplexValue, b: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

fn complex_scale(a: ComplexValue, k: f64) -> ComplexValue {
    ComplexValue {
        re: a.re * k,
        im: a.im * k,
    }
}

fn complex_exp(re: f64, im: f64) -> ComplexValue {
    let mag = re.exp();
    ComplexValue {
        re: mag * im.cos(),
        im: mag * im.sin(),
    }
}

fn complex_magnitude(a: ComplexValue) -> f64 {
    a.re.hypot(a.im)
}

fn complex_abs_diff(a: ComplexValue, b: ComplexValue) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}

fn complex_pow_integer(base: ComplexValue, exponent: f64) -> ComplexValue {
    require(Preconditions::integer(
        "signal-transform",
        "integer power exponent",
        exponent,
    ));
    if exponent == 0.0 {
        return complex(1.0, 0.0);
    }
    let r = complex_magnitude(base);
    if r == 0.0 {
        if exponent < 0.0 {
            panic!("z-transform is undefined at z=0 for positive sequence indices");
        }
        return complex(0.0, 0.0);
    }
    let theta = base.im.atan2(base.re);
    let mag = r.powf(exponent);
    ComplexValue {
        re: mag * (exponent * theta).cos(),
        im: mag * (exponent * theta).sin(),
    }
}

/// JS `Number.prototype.toPrecision` (display-only; never compared).
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
        // Exponential notation with `digits-1` fractional digits. Rust's `{:e}`
        // omits the leading `+` on a non-negative exponent that JS includes, so
        // patch it in.
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

/// `formatComplex(z, digits=6)` → `"re ± imAbs i"`.
pub fn format_complex(z: ComplexValue, digits: usize) -> String {
    let re = to_precision(z.re, digits);
    let im_abs = to_precision(z.im.abs(), digits);
    let sign = if z.im < 0.0 { "-" } else { "+" };
    format!("{re} {sign} {im_abs}i")
}

fn finite_complex(model: &str, param: &str, z: ComplexValue) {
    require(Preconditions::finite(model, &format!("{param}.re"), z.re));
    require(Preconditions::finite(model, &format!("{param}.im"), z.im));
}

fn is_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn finite_constants(constants: Option<&HashMap<String, f64>>) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    out.insert("pi".to_string(), std::f64::consts::PI);
    out.insert("e".to_string(), std::f64::consts::E);
    let Some(constants) = constants else {
        return out;
    };
    for (key, &value) in constants {
        if !is_identifier(key) {
            panic!("constant name must be an identifier: {key}");
        }
        require(Preconditions::finite(
            "signal-transform",
            &format!("constants.{key}"),
            value,
        ));
        out.insert(key.clone(), value);
    }
    out
}

fn normalize_complex_points(
    values: Option<&[ComplexPointInput]>,
    fallback: &[ComplexPointInput],
    point_name: &str,
) -> Vec<ComplexPoint> {
    let raw: &[ComplexPointInput] = match values {
        Some(v) if !v.is_empty() => v,
        _ => fallback,
    };
    require(Preconditions::non_empty(
        "signal-transform",
        point_name,
        raw,
    ));
    raw.iter()
        .enumerate()
        .map(|(i, p)| {
            let point = ComplexPoint {
                re: p.re,
                im: p.im.unwrap_or(0.0),
                label: p
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("{point_name}[{i}]")),
            };
            finite_complex(
                "signal-transform",
                &format!("{point_name}[{i}]"),
                point.as_complex(),
            );
            point
        })
        .collect()
}

fn normalize_omega_points(values: Option<&[f64]>) -> Vec<ComplexPoint> {
    let default = [0.0];
    let raw: &[f64] = match values {
        Some(v) if !v.is_empty() => v,
        _ => &default,
    };
    require(Preconditions::non_empty(
        "fourier-transform",
        "omegaValues",
        raw,
    ));
    raw.iter()
        .enumerate()
        .map(|(i, &omega)| {
            require(Preconditions::finite(
                "fourier-transform",
                &format!("omegaValues[{i}]"),
                omega,
            ));
            ComplexPoint {
                re: omega,
                im: 0.0,
                label: format!("omega={omega}"),
            }
        })
        .collect()
}

fn normalize_bin_points(values: Option<&[usize]>, n: usize, inverse: bool) -> Vec<ComplexPoint> {
    require(Preconditions::integer_in_range(
        "dft-transform",
        "sample count",
        n as f64,
        1.0,
        1_000_000.0,
    ));
    let default: Vec<usize> = (0..n).collect();
    let raw: &[usize] = match values {
        Some(v) if !v.is_empty() => v,
        _ => &default,
    };
    raw.iter()
        .map(|&k| {
            require(Preconditions::integer_in_range(
                "dft-transform",
                "kValues",
                k as f64,
                0.0,
                (n - 1) as f64,
            ));
            ComplexPoint {
                re: k as f64,
                im: if inverse { -(n as f64) } else { n as f64 },
                label: format!("k={k}"),
            }
        })
        .collect()
}

fn normalize_wavelet_points(values: Option<&[WaveletPointInput]>) -> Vec<ComplexPoint> {
    let default = [WaveletPointInput {
        label: None,
        scale: 1.0,
        shift: 0.0,
    }];
    let raw: &[WaveletPointInput] = match values {
        Some(v) if !v.is_empty() => v,
        _ => &default,
    };
    require(Preconditions::non_empty(
        "wavelet-transform",
        "scaleShiftValues",
        raw,
    ));
    raw.iter()
        .enumerate()
        .map(|(i, p)| {
            require(Preconditions::positive(
                "wavelet-transform",
                &format!("scaleShiftValues[{i}].scale"),
                p.scale,
            ));
            require(Preconditions::finite(
                "wavelet-transform",
                &format!("scaleShiftValues[{i}].shift"),
                p.shift,
            ));
            ComplexPoint {
                re: p.scale,
                im: p.shift,
                label: p
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("a={},b={}", p.scale, p.shift)),
            }
        })
        .collect()
}

fn normalize_wavelet_grid_points(
    scales: Option<&[f64]>,
    translations: Option<&[f64]>,
) -> Vec<ComplexPoint> {
    let default_scales = [1.0];
    let default_translations = [0.0];
    let scale_values = match scales {
        Some(v) if !v.is_empty() => v,
        _ => &default_scales,
    };
    let translation_values = match translations {
        Some(v) if !v.is_empty() => v,
        _ => &default_translations,
    };
    let mut points = Vec::new();
    for &scale in scale_values {
        require(Preconditions::positive("wavelet-transform", "scale", scale));
        for &translation in translation_values {
            require(Preconditions::finite(
                "wavelet-transform",
                "translation",
                translation,
            ));
            points.push(ComplexPoint {
                re: scale,
                im: translation,
                label: format!("a={scale},b={translation}"),
            });
        }
    }
    points
}

fn wavelet_points_from_params(params: &WaveletTransformParams) -> Vec<ComplexPoint> {
    if let Some(points) = params.scale_shift_values.as_deref() {
        if !points.is_empty() {
            return normalize_wavelet_points(Some(points));
        }
    }
    normalize_wavelet_grid_points(params.scales.as_deref(), params.translations.as_deref())
}

// ── Records / result shapes ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TransformSampleRecord {
    pub sample_index: usize,
    pub abscissa_name: AbscissaName,
    pub abscissa: f64,
    pub ordinate: Option<f64>,
    pub value: f64,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct TransformContributionRecord {
    pub sample_index: usize,
    pub abscissa: f64,
    pub point_index: usize,
    pub point_label: String,
    pub contribution: ComplexValue,
    pub cumulative: ComplexValue,
}

/// The accumulator's per-point output, before the direct-reference comparison
/// (`Omit<TransformOutputPoint, 'directReference' | 'absoluteError'>`).
#[derive(Clone, Debug)]
pub struct PartialOutputPoint {
    pub point_index: usize,
    pub label: String,
    pub point: ComplexValue,
    pub value: ComplexValue,
    pub magnitude: f64,
    pub phase: f64,
    pub samples_used: usize,
}

#[derive(Clone, Debug)]
pub struct TransformOutputPoint {
    pub point_index: usize,
    pub label: String,
    pub point: ComplexValue,
    pub value: ComplexValue,
    pub magnitude: f64,
    pub phase: f64,
    pub samples_used: usize,
    pub direct_reference: ComplexValue,
    pub absolute_error: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TransformEntityFrameworkSummary {
    pub sources: Vec<String>,
    pub stations: Vec<String>,
    pub sinks: Vec<String>,
    pub movable_entities: Vec<String>,
    pub edges: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TransformRunResult {
    pub kind: TransformKind,
    pub convention: String,
    pub samples: Vec<TransformSampleRecord>,
    pub outputs: Vec<TransformOutputPoint>,
    pub trace: Vec<TransformContributionRecord>,
    pub topology: StationGraphSummary,
    pub entity_framework: TransformEntityFrameworkSummary,
    pub run_summary: IterativeRunSummary,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug, Default)]
pub struct ZTransformParams {
    pub sequence: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub terms: Option<usize>,
    pub start_index: Option<i64>,
    pub z_values: Option<Vec<ComplexPointInput>>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct LaplaceTransformParams {
    pub samples: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub quadrature: Option<QuadratureRule>,
    pub s_values: Option<Vec<ComplexPointInput>>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct FourierTransformParams {
    pub samples: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub quadrature: Option<QuadratureRule>,
    pub omega_values: Option<Vec<f64>>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct DiscreteFourierTransformParams {
    pub sequence: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub terms: Option<usize>,
    pub k_values: Option<Vec<usize>>,
    pub inverse: Option<bool>,
    pub normalize: Option<bool>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct FastFourierTransformParams {
    pub sequence: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub terms: Option<usize>,
    pub k_values: Option<Vec<usize>>,
    pub inverse: Option<bool>,
    pub normalize: Option<bool>,
    pub tolerance: Option<f64>,
}

pub type DftTransformParams = DiscreteFourierTransformParams;
pub type FftTransformParams = FastFourierTransformParams;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveletMother {
    Haar,
    MexicanHat,
    Morlet,
    MorletReal,
}

impl Default for WaveletMother {
    fn default() -> Self {
        WaveletMother::Haar
    }
}

impl WaveletMother {
    pub fn as_str(self) -> &'static str {
        match self {
            WaveletMother::Haar => "haar",
            WaveletMother::MexicanHat => "mexican-hat",
            WaveletMother::Morlet => "morlet",
            WaveletMother::MorletReal => "morlet-real",
        }
    }
}

pub type WaveletKind = WaveletMother;

#[derive(Clone, Debug)]
pub struct WaveletPointInput {
    pub label: Option<String>,
    pub scale: f64,
    pub shift: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WaveletTransformParams {
    pub samples: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub quadrature: Option<QuadratureRule>,
    pub scale_shift_values: Option<Vec<WaveletPointInput>>,
    pub scales: Option<Vec<f64>>,
    pub translations: Option<Vec<f64>>,
    pub mother: Option<WaveletMother>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct MellinTransformParams {
    pub samples: Option<Vec<f64>>,
    pub expression: Option<String>,
    pub constants: Option<HashMap<String, f64>>,
    pub t0: Option<f64>,
    pub t1: Option<f64>,
    pub dt: Option<f64>,
    pub x0: Option<f64>,
    pub x1: Option<f64>,
    pub dx: Option<f64>,
    pub quadrature: Option<QuadratureRule>,
    pub s_values: Option<Vec<ComplexPointInput>>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RadonProjectionInput {
    pub label: Option<String>,
    pub theta: f64,
    pub rho: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RadonTransformParams {
    pub grid: Vec<Vec<f64>>,
    pub image: Option<Vec<Vec<f64>>>,
    /// x-coordinate of the first cell center. Defaults to a centered grid.
    pub x0: Option<f64>,
    /// y-coordinate of the first cell center. Defaults to a centered grid.
    pub y0: Option<f64>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub projections: Option<Vec<RadonProjectionInput>>,
    pub theta_values: Option<Vec<f64>>,
    pub rho_values: Option<Vec<f64>>,
    /// Full acceptance width around a line. Defaults to one grid-cell diagonal.
    pub line_width: Option<f64>,
    pub tolerance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RadonOutputPoint {
    pub point_index: usize,
    pub label: String,
    pub theta: f64,
    pub rho: f64,
    pub value: ComplexValue,
    pub direct_reference: ComplexValue,
    pub absolute_error: f64,
    pub cells_used: usize,
}

#[derive(Clone, Debug)]
pub struct RadonRunResult {
    pub kind: TransformKind,
    pub convention: String,
    pub width: usize,
    pub height: usize,
    pub outputs: Vec<RadonOutputPoint>,
    pub validation: Vec<ValidationCheck>,
}

const SAMPLE_CHANNEL: &str = "transform-sample";
const CONTRIBUTION_CHANNEL: &str = "transform-contribution";
const RESULT_CHANNEL: &str = "transform-result";

/// A per-point kernel: `(sample, point) -> contribution`. Every kernel here is
/// a free fn, so a plain `fn` pointer (which is `Copy`) suffices.
type KernelFn = fn(&TransformSampleRecord, &ComplexPoint) -> ComplexValue;

// ── Tokens ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct TransformSampleToken {
    sample: TransformSampleRecord,
}

#[derive(Clone, Debug)]
struct TransformContributionToken {
    sample: TransformSampleRecord,
    point_index: usize,
    point: ComplexPoint,
    contribution: ComplexValue,
}

#[derive(Clone, Debug)]
struct TransformTotalsToken {
    outputs: Vec<PartialOutputPoint>,
    trace: Vec<TransformContributionRecord>,
}

// ── Stations ──────────────────────────────────────────────────────────────────

/// Emits one [`TransformSampleToken`] per configured sample, in order.
pub struct TransformSampleSourceStation {
    core: StationCore,
    samples: Vec<TransformSampleRecord>,
    index: usize,
}

impl TransformSampleSourceStation {
    pub const CH_SAMPLE: &'static str = SAMPLE_CHANNEL;

    pub fn new(id: impl Into<String>, samples: Vec<TransformSampleRecord>) -> Self {
        let mut st = TransformSampleSourceStation {
            core: StationCore::new(id),
            samples,
            index: 0,
        };
        let id_str = st.core.id.clone();
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                format!("{id_str}.emitted-all-samples"),
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformSampleSourceStation>()
                        .unwrap();
                    st.index == st.samples.len()
                },
                Some("every input sample emitted exactly once".to_string()),
                Some(Box::new(|s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformSampleSourceStation>()
                        .unwrap();
                    format!("{}/{}", st.index, st.samples.len())
                })),
                Some("signal-transform".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }
}

impl DESStation for TransformSampleSourceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn assert_preconditions(&mut self) {
        let model = "TransformSampleSourceStation";
        require(Preconditions::non_empty(
            model,
            &format!("{}.samples", self.core.id),
            &self.samples,
        ));
        for sample in &self.samples {
            require(Preconditions::integer_in_range(
                model,
                "sampleIndex",
                sample.sample_index as f64,
                0.0,
                1e9,
            ));
            require(Preconditions::finite(model, "abscissa", sample.abscissa));
            require(Preconditions::finite(model, "value", sample.value));
            require(Preconditions::finite(model, "weight", sample.weight));
        }
    }
    fn has_work(&self) -> bool {
        self.index < self.samples.len()
    }
    fn run_time_step(&mut self) {
        if self.index >= self.samples.len() {
            return;
        }
        let token = TransformSampleToken {
            sample: self.samples[self.index].clone(),
        };
        self.core.emit(Rc::new(token), Self::CH_SAMPLE);
        self.index += 1;
    }
}

/// Turns each incoming sample into one contribution token per evaluation point.
pub struct TransformKernelStation {
    core: StationCore,
    points: Vec<ComplexPoint>,
    kernel: KernelFn,
}

impl TransformKernelStation {
    pub const CH_SAMPLE: &'static str = SAMPLE_CHANNEL;
    pub const CH_CONTRIBUTION: &'static str = CONTRIBUTION_CHANNEL;

    pub fn new(id: impl Into<String>, points: Vec<ComplexPoint>, kernel: KernelFn) -> Self {
        TransformKernelStation {
            core: StationCore::new(id),
            points,
            kernel,
        }
    }
}

impl DESStation for TransformKernelStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_SAMPLE) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self.core.drain::<TransformSampleToken>(Self::CH_SAMPLE);
        let id = self.core.id.clone();
        for token in tokens {
            for (point_index, point) in self.points.iter().enumerate() {
                let contribution = (self.kernel)(&token.sample, point);
                finite_complex(
                    "TransformKernelStation",
                    &format!("{id}.contribution"),
                    contribution,
                );
                let out = TransformContributionToken {
                    sample: token.sample.clone(),
                    point_index,
                    point: point.clone(),
                    contribution,
                };
                self.core.emit(Rc::new(out), Self::CH_CONTRIBUTION);
            }
        }
    }
}

/// Accumulates contributions per point and emits the totals once complete.
pub struct TransformAccumulatorStation {
    core: StationCore,
    points: Vec<ComplexPoint>,
    expected_samples: usize,
    sums: Vec<ComplexValue>,
    counts: Vec<usize>,
    trace: Vec<TransformContributionRecord>,
    total_contributions: usize,
    emitted: bool,
}

impl TransformAccumulatorStation {
    pub const CH_CONTRIBUTION: &'static str = CONTRIBUTION_CHANNEL;
    pub const CH_RESULT: &'static str = RESULT_CHANNEL;

    pub fn new(id: impl Into<String>, points: Vec<ComplexPoint>, expected_samples: usize) -> Self {
        let n = points.len();
        let mut st = TransformAccumulatorStation {
            core: StationCore::new(id),
            points,
            expected_samples,
            sums: vec![complex(0.0, 0.0); n],
            counts: vec![0; n],
            trace: Vec::new(),
            total_contributions: 0,
            emitted: false,
        };
        let id_str = st.core.id.clone();
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                format!("{id_str}.complete-contribution-count"),
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformAccumulatorStation>()
                        .unwrap();
                    st.counts.iter().all(|&count| count == st.expected_samples)
                },
                Some("one contribution per sample per evaluation point".to_string()),
                Some(Box::new(|s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformAccumulatorStation>()
                        .unwrap();
                    st.counts
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })),
                Some("signal-transform".to_string()),
                None,
            )
            .boxed(),
        );
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                format!("{id_str}.finite-sums"),
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformAccumulatorStation>()
                        .unwrap();
                    st.sums
                        .iter()
                        .all(|sum| sum.re.is_finite() && sum.im.is_finite())
                },
                Some("all accumulated complex sums finite".to_string()),
                None,
                Some("signal-transform".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn outputs(&self) -> Vec<PartialOutputPoint> {
        self.points
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                let value = self.sums[point_index];
                PartialOutputPoint {
                    point_index,
                    label: point.label.clone(),
                    point: ComplexValue {
                        re: point.re,
                        im: point.im,
                    },
                    value: ComplexValue {
                        re: value.re,
                        im: value.im,
                    },
                    magnitude: complex_magnitude(value),
                    phase: value.im.atan2(value.re),
                    samples_used: self.counts[point_index],
                }
            })
            .collect()
    }
}

impl DESStation for TransformAccumulatorStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_CONTRIBUTION) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self
            .core
            .drain::<TransformContributionToken>(Self::CH_CONTRIBUTION);
        for token in tokens {
            let point_index = token.point_index;
            self.sums[point_index] = complex_add(self.sums[point_index], token.contribution);
            self.counts[point_index] += 1;
            self.total_contributions += 1;
            self.trace.push(TransformContributionRecord {
                sample_index: token.sample.sample_index,
                abscissa: token.sample.abscissa,
                point_index,
                point_label: token.point.label.clone(),
                contribution: token.contribution,
                cumulative: self.sums[point_index],
            });
        }
        if !self.emitted && self.total_contributions == self.expected_samples * self.points.len() {
            let totals = TransformTotalsToken {
                outputs: self.outputs(),
                trace: self.trace.clone(),
            };
            self.core.emit(Rc::new(totals), Self::CH_RESULT);
            self.emitted = true;
        }
    }
}

/// Keeps the most recent [`TransformTotalsToken`].
pub struct TransformResultSinkStation {
    core: StationCore,
    latest: Option<TransformTotalsToken>,
}

impl TransformResultSinkStation {
    pub const CH_RESULT: &'static str = RESULT_CHANNEL;

    pub fn new(id: impl Into<String>) -> Self {
        let mut st = TransformResultSinkStation {
            core: StationCore::new(id),
            latest: None,
        };
        let id_str = st.core.id.clone();
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                format!("{id_str}.received-result"),
                |s| {
                    let st = s
                        .as_any()
                        .downcast_ref::<TransformResultSinkStation>()
                        .unwrap();
                    st.latest.is_some()
                },
                Some("one transform result token reaches the sink".to_string()),
                None,
                Some("signal-transform".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }
}

impl DESStation for TransformResultSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(Self::CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self.core.drain::<TransformTotalsToken>(Self::CH_RESULT);
        if let Some(last) = tokens.last() {
            self.latest = Some((**last).clone());
        }
    }
}

// ── Pipeline assembly ─────────────────────────────────────────────────────────

fn build_entity_framework(
    source_id: &str,
    kernel_id: &str,
    accumulator_id: &str,
    sink_id: &str,
) -> TransformEntityFrameworkSummary {
    let s = StationOrId::Id(source_id.to_string());
    let k = StationOrId::Id(kernel_id.to_string());
    let a = StationOrId::Id(accumulator_id.to_string());
    let si = StationOrId::Id(sink_id.to_string());
    let edges = vec![
        channel_edge(&s, SAMPLE_CHANNEL, &k, Some(SAMPLE_CHANNEL)),
        channel_edge(&k, CONTRIBUTION_CHANNEL, &a, Some(CONTRIBUTION_CHANNEL)),
        channel_edge(&a, RESULT_CHANNEL, &si, Some(RESULT_CHANNEL)),
    ];
    TransformEntityFrameworkSummary {
        sources: vec![source_id.to_string()],
        stations: vec![kernel_id.to_string(), accumulator_id.to_string()],
        sinks: vec![sink_id.to_string()],
        movable_entities: vec![
            "TransformSampleToken".to_string(),
            "TransformContributionToken".to_string(),
            "TransformTotalsToken".to_string(),
        ],
        edges,
    }
}

fn direct_transform(
    samples: &[TransformSampleRecord],
    points: &[ComplexPoint],
    kernel: KernelFn,
) -> Vec<ComplexValue> {
    let mut sums = vec![complex(0.0, 0.0); points.len()];
    for sample in samples {
        for (i, point) in points.iter().enumerate() {
            sums[i] = complex_add(sums[i], kernel(sample, point));
        }
    }
    sums
}

fn reference_checks(
    outputs: &[TransformOutputPoint],
    tolerance: f64,
    kind: TransformKind,
) -> Vec<ValidationCheck> {
    outputs
        .iter()
        .map(|output| {
            let passed = output.absolute_error <= tolerance;
            ValidationCheck {
                name: format!("{}-transform.reference.{}", kind.as_str(), output.label),
                group: Some("signal-transform-reference".to_string()),
                passed,
                observed: Some(format_complex(output.value, 6)),
                expected: Some(format_complex(output.direct_reference, 6)),
                details: if passed {
                    None
                } else {
                    Some(format!(
                        "abs-error={:.3e} > tolerance={tolerance}",
                        output.absolute_error
                    ))
                },
            }
        })
        .collect()
}

struct TransformPipelineArgs {
    kind: TransformKind,
    convention: String,
    samples: Vec<TransformSampleRecord>,
    points: Vec<ComplexPoint>,
    kernel: KernelFn,
    tolerance: f64,
}

fn run_transform_pipeline(args: TransformPipelineArgs) -> TransformRunResult {
    let kind_str = args.kind.as_str();
    let source_id = format!("{kind_str}-sample-source");
    let kernel_id = format!("{kind_str}-kernel-station");
    let accumulator_id = format!("{kind_str}-accumulator-station");
    let sink_id = format!("{kind_str}-result-sink");

    let source = Rc::new(RefCell::new(TransformSampleSourceStation::new(
        source_id.clone(),
        args.samples.clone(),
    )));
    let kernel_station = Rc::new(RefCell::new(TransformKernelStation::new(
        kernel_id.clone(),
        args.points.clone(),
        args.kernel,
    )));
    let accumulator = Rc::new(RefCell::new(TransformAccumulatorStation::new(
        accumulator_id.clone(),
        args.points.clone(),
        args.samples.len(),
    )));
    let sink = Rc::new(RefCell::new(TransformResultSinkStation::new(
        sink_id.clone(),
    )));

    source.borrow_mut().core_mut().pipe(
        kernel_station.clone() as StationRef,
        SAMPLE_CHANNEL,
        SAMPLE_CHANNEL,
    );
    kernel_station.borrow_mut().core_mut().pipe(
        accumulator.clone() as StationRef,
        CONTRIBUTION_CHANNEL,
        CONTRIBUTION_CHANNEL,
    );
    accumulator.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        RESULT_CHANNEL,
        RESULT_CHANNEL,
    );

    let run_summary = run_iterative_des(
        vec![
            source.clone() as StationRef,
            kernel_station.clone() as StationRef,
            accumulator.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            max_ticks: Some(args.samples.len() + 10),
            shuffle: false,
            ..Default::default()
        },
    );

    let latest = sink
        .borrow()
        .latest
        .clone()
        .unwrap_or_else(|| panic!("{kind_str}-transform did not produce a result"));

    let direct = direct_transform(&args.samples, &args.points, args.kernel);
    let outputs: Vec<TransformOutputPoint> = latest
        .outputs
        .iter()
        .map(|output| {
            let direct_reference = direct[output.point_index];
            let absolute_error = complex_abs_diff(output.value, direct_reference);
            TransformOutputPoint {
                point_index: output.point_index,
                label: output.label.clone(),
                point: output.point,
                value: output.value,
                magnitude: output.magnitude,
                phase: output.phase,
                samples_used: output.samples_used,
                direct_reference,
                absolute_error,
            }
        })
        .collect();

    let mut validation: Vec<ValidationCheck> = run_summary.validation.clone().unwrap_or_default();
    validation.extend(reference_checks(&outputs, args.tolerance, args.kind));

    let entity_framework =
        build_entity_framework(&source_id, &kernel_id, &accumulator_id, &sink_id);
    let s = StationOrId::Id(source_id.clone());
    let k = StationOrId::Id(kernel_id.clone());
    let a = StationOrId::Id(accumulator_id.clone());
    let si = StationOrId::Id(sink_id.clone());
    let topology = station_graph(
        &[s, k, a, si],
        &entity_framework.movable_entities,
        &entity_framework.edges,
    );

    TransformRunResult {
        kind: args.kind,
        convention: args.convention,
        samples: args.samples.to_vec(),
        outputs,
        trace: latest.trace,
        topology,
        entity_framework,
        run_summary,
        validation,
    }
}

// ── Sample builders ───────────────────────────────────────────────────────────

fn build_z_samples(params: &ZTransformParams) -> Vec<TransformSampleRecord> {
    let start_index = params.start_index.unwrap_or(0);
    require(Preconditions::integer(
        "z-transform",
        "startIndex",
        start_index as f64,
    ));
    let sequence = match &params.sequence {
        Some(s) if !s.is_empty() => s.clone(),
        _ => build_expression_sequence(params),
    };
    require(Preconditions::non_empty(
        "z-transform",
        "sequence",
        &sequence,
    ));
    require(Preconditions::all_finite(
        "z-transform",
        "sequence",
        &sequence,
    ));
    sequence
        .iter()
        .enumerate()
        .map(|(sample_index, &value)| TransformSampleRecord {
            sample_index,
            abscissa_name: AbscissaName::N,
            abscissa: (start_index + sample_index as i64) as f64,
            ordinate: None,
            value,
            weight: 1.0,
        })
        .collect()
}

fn build_expression_sequence(params: &ZTransformParams) -> Vec<f64> {
    build_discrete_expression_sequence(
        "z-transform",
        params.expression.as_deref(),
        params.constants.as_ref(),
        params.terms,
        params.start_index.unwrap_or(0),
    )
}

fn build_dft_samples(params: &DiscreteFourierTransformParams) -> Vec<TransformSampleRecord> {
    let sequence = match &params.sequence {
        Some(s) if !s.is_empty() => s.clone(),
        _ => build_discrete_expression_sequence(
            "dft-transform",
            params.expression.as_deref(),
            params.constants.as_ref(),
            params.terms,
            0,
        ),
    };
    require(Preconditions::non_empty(
        "dft-transform",
        "sequence",
        &sequence,
    ));
    require(Preconditions::all_finite(
        "dft-transform",
        "sequence",
        &sequence,
    ));
    sequence
        .iter()
        .enumerate()
        .map(|(sample_index, &value)| TransformSampleRecord {
            sample_index,
            abscissa_name: AbscissaName::N,
            abscissa: sample_index as f64,
            ordinate: None,
            value,
            weight: if params.normalize.unwrap_or(false) {
                1.0 / sequence.len() as f64
            } else {
                1.0
            },
        })
        .collect()
}

fn build_fft_samples(params: &FastFourierTransformParams) -> Vec<TransformSampleRecord> {
    build_dft_samples(&DiscreteFourierTransformParams {
        sequence: params.sequence.clone(),
        expression: params.expression.clone(),
        constants: params.constants.clone(),
        terms: params.terms,
        k_values: None,
        inverse: params.inverse,
        normalize: params.normalize,
        tolerance: params.tolerance,
    })
}

fn build_discrete_expression_sequence(
    model: &str,
    expression: Option<&str>,
    constants: Option<&HashMap<String, f64>>,
    terms: Option<usize>,
    start_index: i64,
) -> Vec<f64> {
    let Some(expression) = expression else {
        panic!("{model} requires either a finite sequence or a sequence expression");
    };
    let terms = terms.unwrap_or(8);
    require(Preconditions::integer_in_range(
        model,
        "terms",
        terms as f64,
        1.0,
        1000000.0,
    ));
    let ast = parse(expression);
    let constants = finite_constants(constants);
    let mut values: Vec<f64> = Vec::new();
    for i in 0..terms {
        let n = start_index + i as i64;
        let mut env: Env = constants.clone();
        env.insert("n".to_string(), n as f64);
        env.insert("index".to_string(), i as f64);
        env.insert("tick".to_string(), i as f64);
        let value = evaluate(&ast, &env);
        require(Preconditions::finite(
            model,
            &format!("expression[{i}]"),
            value,
        ));
        values.push(value);
    }
    values
}

fn build_continuous_samples(model: &str, params: &ContinuousParams) -> Vec<TransformSampleRecord> {
    let t0 = params.t0.unwrap_or(0.0);
    let dt = params.dt.unwrap_or(0.01);
    let quadrature = params.quadrature.unwrap_or(QuadratureRule::Trapezoid);
    require(Preconditions::finite(model, "t0", t0));
    require(Preconditions::positive(model, "dt", dt));

    let values = match &params.samples {
        Some(s) if !s.is_empty() => s.clone(),
        _ => build_expression_samples(model, params, t0, dt, quadrature),
    };
    require(Preconditions::non_empty(model, "samples", &values));
    require(Preconditions::all_finite(model, "samples", &values));
    if quadrature == QuadratureRule::Trapezoid {
        require(Preconditions::check(
            model,
            "samples.length",
            "be at least 2 for trapezoid quadrature",
            values.len() >= 2,
            Some(values.len().to_string()),
        ));
    }

    let n = values.len();
    values
        .iter()
        .enumerate()
        .map(|(sample_index, &value)| TransformSampleRecord {
            sample_index,
            abscissa_name: AbscissaName::T,
            abscissa: t0 + sample_index as f64 * dt,
            ordinate: None,
            value,
            weight: quadrature_weight(sample_index, n, dt, quadrature),
        })
        .collect()
}

fn build_expression_samples(
    model: &str,
    params: &ContinuousParams,
    t0: f64,
    dt: f64,
    quadrature: QuadratureRule,
) -> Vec<f64> {
    let Some(expression) = &params.expression else {
        panic!("{model} requires either samples or an expression");
    };
    let t1 = params.t1.unwrap_or(1.0);
    require(Preconditions::finite(model, "t1", t1));
    require(Preconditions::check(
        model,
        "t1",
        "be greater than t0",
        t1 > t0,
        Some(format!("{{t0: {t0}, t1: {t1}}}")),
    ));
    let exact_steps = (t1 - t0) / dt;
    let steps = exact_steps.round();
    require(Preconditions::check(
        model,
        "(t1 - t0) / dt",
        "be an integer number of steps",
        (exact_steps - steps).abs() <= 1e-9 * 1.0_f64.max(exact_steps.abs()),
        Some(exact_steps.to_string()),
    ));
    require(Preconditions::integer_in_range(
        model, "steps", steps, 1.0, 1000000.0,
    ));
    let steps = steps as usize;
    let sample_count = if quadrature == QuadratureRule::Trapezoid {
        steps + 1
    } else {
        steps
    };
    let ast = parse(expression);
    let constants = finite_constants(params.constants.as_ref());
    let mut values: Vec<f64> = Vec::new();
    for i in 0..sample_count {
        let t = t0 + i as f64 * dt;
        let mut env: Env = constants.clone();
        env.insert("t".to_string(), t);
        env.insert("x".to_string(), t);
        env.insert("time".to_string(), t);
        env.insert("tick".to_string(), i as f64);
        let value = evaluate(&ast, &env);
        require(Preconditions::finite(
            model,
            &format!("expression[{i}]"),
            value,
        ));
        values.push(value);
    }
    values
}

fn quadrature_weight(i: usize, n: usize, dt: f64, rule: QuadratureRule) -> f64 {
    if rule == QuadratureRule::Rectangular {
        return dt;
    }
    if i == 0 || i == n - 1 {
        0.5 * dt
    } else {
        dt
    }
}

// ── Kernels ───────────────────────────────────────────────────────────────────

fn z_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    let z_to_minus_n = complex_pow_integer(point.as_complex(), -sample.abscissa);
    complex_scale(z_to_minus_n, sample.value)
}

fn laplace_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    complex_scale(
        complex_exp(-point.re * sample.abscissa, -point.im * sample.abscissa),
        sample.value * sample.weight,
    )
}

fn fourier_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    complex_scale(
        complex_exp(0.0, -point.re * sample.abscissa),
        sample.value * sample.weight,
    )
}

fn dft_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    let n = point.im;
    let k = point.re;
    let angle = -std::f64::consts::TAU * k * sample.sample_index as f64 / n;
    complex_scale(complex_exp(0.0, angle), sample.value * sample.weight)
}

fn haar_wavelet_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    let scale = point.re;
    let shift = point.im;
    let u = (sample.abscissa - shift) / scale;
    let psi = if (0.0..0.5).contains(&u) {
        1.0
    } else if (0.5..1.0).contains(&u) {
        -1.0
    } else {
        0.0
    };
    complex(sample.value * sample.weight * psi / scale.sqrt(), 0.0)
}

fn mexican_hat_wavelet_kernel(
    sample: &TransformSampleRecord,
    point: &ComplexPoint,
) -> ComplexValue {
    let scale = point.re;
    let shift = point.im;
    let u = (sample.abscissa - shift) / scale;
    let psi = (1.0 - u * u) * (-0.5 * u * u).exp();
    complex(sample.value * sample.weight * psi / scale.sqrt(), 0.0)
}

fn morlet_real_wavelet_kernel(
    sample: &TransformSampleRecord,
    point: &ComplexPoint,
) -> ComplexValue {
    let scale = point.re;
    let shift = point.im;
    let u = (sample.abscissa - shift) / scale;
    let psi = (5.0 * u).cos() * (-0.5 * u * u).exp();
    complex(sample.value * sample.weight * psi / scale.sqrt(), 0.0)
}

fn mellin_kernel(sample: &TransformSampleRecord, point: &ComplexPoint) -> ComplexValue {
    let t = sample.abscissa;
    require(Preconditions::positive("mellin-transform", "sample.t", t));
    let mag = t.powf(point.re - 1.0);
    let phase = point.im * t.ln();
    complex_scale(complex_exp(0.0, phase), sample.value * sample.weight * mag)
}

fn validate_z_points(samples: &[TransformSampleRecord], points: &[ComplexPoint]) {
    let has_positive_index = samples.iter().any(|sample| sample.abscissa > 0.0);
    if !has_positive_index {
        return;
    }
    for point in points {
        require(Preconditions::check(
            "z-transform",
            &format!("z={}", point.label),
            "be nonzero when any n > 0",
            complex_magnitude(point.as_complex()) > 0.0,
            Some(format!("{{re: {}, im: {}}}", point.re, point.im)),
        ));
    }
}

fn validate_positive_time_samples(model: &str, samples: &[TransformSampleRecord]) {
    for sample in samples {
        require(Preconditions::positive(
            model,
            "sample abscissa",
            sample.abscissa,
        ));
    }
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn fft_radix2(input: &[ComplexValue], inverse: bool, normalize: bool) -> Vec<ComplexValue> {
    let n = input.len();
    if !is_power_of_two(n) {
        panic!("fft-transform requires a non-empty power-of-two sample count");
    }
    let bits = n.trailing_zeros() as usize;
    let mut out = vec![complex(0.0, 0.0); n];
    for (i, &value) in input.iter().enumerate() {
        out[bit_reverse(i, bits)] = value;
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = if inverse { 1.0 } else { -1.0 } * std::f64::consts::TAU / len as f64;
        let w_len = complex_exp(0.0, theta);
        let mut start = 0usize;
        while start < n {
            let mut w = complex(1.0, 0.0);
            for j in 0..half {
                let u = out[start + j];
                let v = complex_mul(out[start + j + half], w);
                out[start + j] = complex_add(u, v);
                out[start + j + half] = complex_sub(u, v);
                w = complex_mul(w, w_len);
            }
            start += len;
        }
        len *= 2;
    }
    if normalize {
        let scale = 1.0 / n as f64;
        out.iter_mut().for_each(|z| *z = complex_scale(*z, scale));
    }
    out
}

fn fft_reference_outputs(
    kind: TransformKind,
    samples: &[TransformSampleRecord],
    values: &[ComplexValue],
    inverse: bool,
    tolerance: f64,
) -> (Vec<TransformOutputPoint>, Vec<ValidationCheck>) {
    let points = normalize_bin_points(None, samples.len(), inverse);
    let direct = direct_transform(samples, &points, dft_kernel);
    let outputs: Vec<TransformOutputPoint> = values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let direct_reference = direct[i];
            let absolute_error = complex_abs_diff(value, direct_reference);
            TransformOutputPoint {
                point_index: i,
                label: points[i].label.clone(),
                point: points[i].as_complex(),
                value,
                magnitude: complex_magnitude(value),
                phase: value.im.atan2(value.re),
                samples_used: samples.len(),
                direct_reference,
                absolute_error,
            }
        })
        .collect();
    let validation = reference_checks(&outputs, tolerance, kind);
    (outputs, validation)
}

fn radon_grid(params: &RadonTransformParams) -> &[Vec<f64>] {
    if !params.grid.is_empty() {
        &params.grid
    } else {
        params.image.as_deref().unwrap_or(&[])
    }
}

fn validate_grid(params: &RadonTransformParams) -> (usize, usize, f64, f64, f64, f64, f64) {
    let grid = radon_grid(params);
    require(Preconditions::non_empty("radon-transform", "grid", grid));
    let height = grid.len();
    let width = grid[0].len();
    require(Preconditions::non_empty(
        "radon-transform",
        "grid[0]",
        &grid[0],
    ));
    for (row, values) in grid.iter().enumerate() {
        require(Preconditions::length_eq(
            "radon-transform",
            &format!("grid[{row}]"),
            values,
            width,
        ));
        require(Preconditions::all_finite(
            "radon-transform",
            &format!("grid[{row}]"),
            values,
        ));
    }
    let dx = params.dx.unwrap_or(1.0);
    let dy = params.dy.unwrap_or(1.0);
    require(Preconditions::positive("radon-transform", "dx", dx));
    require(Preconditions::positive("radon-transform", "dy", dy));
    let x0 = params
        .x0
        .unwrap_or_else(|| -0.5 * (width.saturating_sub(1)) as f64 * dx);
    let y0 = params
        .y0
        .unwrap_or_else(|| -0.5 * (height.saturating_sub(1)) as f64 * dy);
    require(Preconditions::finite("radon-transform", "x0", x0));
    require(Preconditions::finite("radon-transform", "y0", y0));
    let line_width = params
        .line_width
        .unwrap_or_else(|| (dx * dx + dy * dy).sqrt());
    require(Preconditions::positive(
        "radon-transform",
        "lineWidth",
        line_width,
    ));
    (width, height, dx, dy, x0, y0, line_width)
}

fn normalize_radon_projections(
    values: Option<&[RadonProjectionInput]>,
) -> Vec<RadonProjectionInput> {
    let default = [RadonProjectionInput {
        label: Some("theta=0,rho=0".to_string()),
        theta: 0.0,
        rho: 0.0,
    }];
    let raw: &[RadonProjectionInput] = match values {
        Some(v) if !v.is_empty() => v,
        _ => &default,
    };
    require(Preconditions::non_empty(
        "radon-transform",
        "projections",
        raw,
    ));
    raw.iter()
        .enumerate()
        .map(|(i, p)| {
            require(Preconditions::finite(
                "radon-transform",
                &format!("projections[{i}].theta"),
                p.theta,
            ));
            require(Preconditions::finite(
                "radon-transform",
                &format!("projections[{i}].rho"),
                p.rho,
            ));
            RadonProjectionInput {
                label: p
                    .label
                    .clone()
                    .or_else(|| Some(format!("theta={},rho={}", p.theta, p.rho))),
                theta: p.theta,
                rho: p.rho,
            }
        })
        .collect()
}

fn normalize_radon_projection_grid(
    theta_values: Option<&[f64]>,
    rho_values: Option<&[f64]>,
) -> Vec<RadonProjectionInput> {
    let default_theta = [0.0];
    let default_rho = [0.0];
    let theta_values = match theta_values {
        Some(v) if !v.is_empty() => v,
        _ => &default_theta,
    };
    let rho_values = match rho_values {
        Some(v) if !v.is_empty() => v,
        _ => &default_rho,
    };
    let mut projections = Vec::new();
    for &theta in theta_values {
        require(Preconditions::finite("radon-transform", "theta", theta));
        for &rho in rho_values {
            require(Preconditions::finite("radon-transform", "rho", rho));
            projections.push(RadonProjectionInput {
                label: Some(format!("theta={theta},rho={rho}")),
                theta,
                rho,
            });
        }
    }
    projections
}

fn radon_projections_from_params(params: &RadonTransformParams) -> Vec<RadonProjectionInput> {
    if let Some(projections) = params.projections.as_deref() {
        if !projections.is_empty() {
            return normalize_radon_projections(Some(projections));
        }
    }
    normalize_radon_projection_grid(params.theta_values.as_deref(), params.rho_values.as_deref())
}

fn radon_projection_value(
    grid: &[Vec<f64>],
    theta: f64,
    rho: f64,
    dx: f64,
    dy: f64,
    x0: f64,
    y0: f64,
    line_width: f64,
) -> (f64, usize) {
    let c = theta.cos();
    let s = theta.sin();
    let half_width = 0.5 * line_width;
    let mut value = 0.0;
    let mut used = 0usize;
    for (row, values) in grid.iter().enumerate() {
        let y = y0 + row as f64 * dy;
        for (col, &cell) in values.iter().enumerate() {
            let x = x0 + col as f64 * dx;
            let distance = (x * c + y * s - rho).abs();
            if distance <= half_width {
                value += cell * dx * dy;
                used += 1;
            }
        }
    }
    (value, used)
}

// ── A small shared view over the continuous-transform params ──────────────────

/// `LaplaceTransformParams` and `FourierTransformParams` share every field the
/// continuous-sample builders read; this borrowed view avoids duplicating the
/// builder per param type (the TS used a structural union).
struct ContinuousParams<'a> {
    samples: &'a Option<Vec<f64>>,
    expression: &'a Option<String>,
    constants: &'a Option<HashMap<String, f64>>,
    t0: Option<f64>,
    t1: Option<f64>,
    dt: Option<f64>,
    quadrature: Option<QuadratureRule>,
}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn run_z_transform(params: ZTransformParams) -> TransformRunResult {
    let samples = build_z_samples(&params);
    let points = normalize_complex_points(
        params.z_values.as_deref(),
        &[ComplexPointInput {
            label: Some("z=1".to_string()),
            re: 1.0,
            im: Some(0.0),
        }],
        "zValues",
    );
    validate_z_points(&samples, &points);
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Z,
        convention: "X(z) = sum_n x[n] z^(-n), evaluated over the supplied finite sequence."
            .to_string(),
        samples,
        points,
        kernel: z_kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_laplace_transform(params: LaplaceTransformParams) -> TransformRunResult {
    let continuous = ContinuousParams {
        samples: &params.samples,
        expression: &params.expression,
        constants: &params.constants,
        t0: params.t0,
        t1: params.t1,
        dt: params.dt,
        quadrature: params.quadrature,
    };
    let samples = build_continuous_samples("laplace-transform", &continuous);
    let points = normalize_complex_points(
        params.s_values.as_deref(),
        &[ComplexPointInput {
            label: Some("s=1".to_string()),
            re: 1.0,
            im: Some(0.0),
        }],
        "sValues",
    );
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Laplace,
        convention: "F(s) = integral f(t) exp(-s t) dt, evaluated by weighted sample tokens."
            .to_string(),
        samples,
        points,
        kernel: laplace_kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_fourier_transform(params: FourierTransformParams) -> TransformRunResult {
    let continuous = ContinuousParams {
        samples: &params.samples,
        expression: &params.expression,
        constants: &params.constants,
        t0: params.t0,
        t1: params.t1,
        dt: params.dt,
        quadrature: params.quadrature,
    };
    let samples = build_continuous_samples("fourier-transform", &continuous);
    let points = normalize_omega_points(params.omega_values.as_deref());
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Fourier,
        convention:
            "F(omega) = integral f(t) exp(-i omega t) dt, evaluated by weighted sample tokens."
                .to_string(),
        samples,
        points,
        kernel: fourier_kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_dft_transform(params: DiscreteFourierTransformParams) -> TransformRunResult {
    let samples = build_dft_samples(&params);
    let inverse = params.inverse.unwrap_or(false);
    let normalize = params.normalize.unwrap_or(false);
    let points = normalize_bin_points(params.k_values.as_deref(), samples.len(), inverse);
    let convention = if inverse {
        "X[k] = sum_n x[n] exp(+i 2*pi*k*n/N), with optional 1/N normalization."
    } else if normalize {
        "X[k] = (1/N) sum_n x[n] exp(-i 2*pi*k*n/N), evaluated over supplied DFT bins."
    } else {
        "X[k] = sum_n x[n] exp(-i 2*pi*k*n/N), evaluated over supplied DFT bins."
    };
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Dft,
        convention: convention.to_string(),
        samples,
        points,
        kernel: dft_kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_discrete_fourier_transform(
    params: DiscreteFourierTransformParams,
) -> TransformRunResult {
    run_dft_transform(params)
}

pub fn run_fft_transform(params: FastFourierTransformParams) -> TransformRunResult {
    let samples = build_fft_samples(&params);
    let values: Vec<ComplexValue> = samples
        .iter()
        .map(|sample| complex(sample.value, 0.0))
        .collect();
    let inverse = params.inverse.unwrap_or(false);
    let normalize = params.normalize.unwrap_or(false);
    let fft_values = fft_radix2(&values, inverse, normalize);
    let tolerance = params.tolerance.unwrap_or(1e-9);
    let (outputs, validation) = fft_reference_outputs(
        TransformKind::Fft,
        &samples,
        &fft_values,
        inverse,
        tolerance,
    );
    let n = samples.len();
    let stage_count = n.trailing_zeros() as usize;
    let stations = vec!["fft-butterfly-network".to_string()];
    let movables = vec!["FftButterflyStage".to_string()];
    let topology = station_graph(
        &[StationOrId::Id(stations[0].clone())],
        &movables,
        &Vec::<String>::new(),
    );
    TransformRunResult {
        kind: TransformKind::Fft,
        convention: "Radix-2 Cooley-Tukey FFT computing the DFT bins X[k].".to_string(),
        samples,
        outputs,
        trace: Vec::new(),
        topology,
        entity_framework: TransformEntityFrameworkSummary {
            sources: vec!["fft-input-vector".to_string()],
            stations,
            sinks: vec!["fft-output-vector".to_string()],
            movable_entities: movables,
            edges: Vec::new(),
        },
        run_summary: IterativeRunSummary {
            ticks: stage_count,
            ..Default::default()
        },
        validation,
    }
}

pub fn run_wavelet_transform(params: WaveletTransformParams) -> TransformRunResult {
    let continuous = ContinuousParams {
        samples: &params.samples,
        expression: &params.expression,
        constants: &params.constants,
        t0: params.t0,
        t1: params.t1,
        dt: params.dt,
        quadrature: params.quadrature,
    };
    let samples = build_continuous_samples("wavelet-transform", &continuous);
    let points = wavelet_points_from_params(&params);
    let mother = params.mother.unwrap_or_default();
    let kernel: KernelFn = match mother {
        WaveletMother::Haar => haar_wavelet_kernel,
        WaveletMother::MexicanHat => mexican_hat_wavelet_kernel,
        WaveletMother::Morlet => morlet_real_wavelet_kernel,
        WaveletMother::MorletReal => morlet_real_wavelet_kernel,
    };
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Wavelet,
        convention: format!(
            "W(a,b) = integral f(t) psi((t-b)/a) dt / sqrt(a), mother={}.",
            mother.as_str()
        ),
        samples,
        points,
        kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_mellin_transform(params: MellinTransformParams) -> TransformRunResult {
    let continuous = ContinuousParams {
        samples: &params.samples,
        expression: &params.expression,
        constants: &params.constants,
        t0: params.x0.or(params.t0),
        t1: params.x1.or(params.t1),
        dt: params.dx.or(params.dt),
        quadrature: params.quadrature,
    };
    let samples = build_continuous_samples("mellin-transform", &continuous);
    validate_positive_time_samples("mellin-transform", &samples);
    let points = normalize_complex_points(
        params.s_values.as_deref(),
        &[ComplexPointInput {
            label: Some("s=1".to_string()),
            re: 1.0,
            im: Some(0.0),
        }],
        "sValues",
    );
    run_transform_pipeline(TransformPipelineArgs {
        kind: TransformKind::Mellin,
        convention:
            "M(s) = integral_0^infinity t^(s-1) f(t) dt, evaluated on a positive finite window."
                .to_string(),
        samples,
        points,
        kernel: mellin_kernel,
        tolerance: params.tolerance.unwrap_or(1e-9),
    })
}

pub fn run_radon_transform(params: RadonTransformParams) -> RadonRunResult {
    let (width, height, dx, dy, x0, y0, line_width) = validate_grid(&params);
    let grid = radon_grid(&params);
    let projections = radon_projections_from_params(&params);
    let tolerance = params.tolerance.unwrap_or(1e-12);
    let outputs: Vec<RadonOutputPoint> = projections
        .iter()
        .enumerate()
        .map(|(point_index, p)| {
            let (value, cells_used) =
                radon_projection_value(grid, p.theta, p.rho, dx, dy, x0, y0, line_width);
            let value = complex(value, 0.0);
            let direct_reference = value;
            RadonOutputPoint {
                point_index,
                label: p
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("projection[{point_index}]")),
                theta: p.theta,
                rho: p.rho,
                value,
                direct_reference,
                absolute_error: complex_abs_diff(value, direct_reference),
                cells_used,
            }
        })
        .collect();
    let validation = outputs
        .iter()
        .map(|output| ValidationCheck {
            name: format!("radon-transform.reference.{}", output.label),
            group: Some("signal-transform-reference".to_string()),
            passed: output.absolute_error <= tolerance,
            observed: Some(format_complex(output.value, 6)),
            expected: Some(format_complex(output.direct_reference, 6)),
            details: None,
        })
        .collect();
    RadonRunResult {
        kind: TransformKind::Radon,
        convention: "R(theta,rho) = integral over x*cos(theta)+y*sin(theta)=rho; approximated by finite grid-line accumulation.".to_string(),
        width,
        height,
        outputs,
        validation,
    }
}
