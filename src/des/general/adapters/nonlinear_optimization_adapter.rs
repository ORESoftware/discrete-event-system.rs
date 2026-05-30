//! Port of `src/des/general/adapters/nonlinear-optimization-adapter.ts`
//! (module `des::general::adapters::nonlinear_optimization_adapter`).
//!
//! Registers the Newton / BFGS / Gauss–Newton / Levenberg–Marquardt JSON
//! adapters (4 models) over nonlinear-optimization station graphs.
//!
//! ## Conversion notes
//!
//!   * The four `registerModel(...)` calls collapse into two reusable adapter
//!     structs — [`UnconstrainedAdapter`] and [`NlsAdapter`] — each carrying the
//!     per-model `id` / `description` / summary `title` / `run` fn pointer and
//!     example, mirroring the shared `unconstrainedSummary` / `nlsSummary` /
//!     `writeUnconstrainedCsv` / `writeNLSCsv` helpers in the TS source.
//!   * `JSON.stringify(row.x)` / `JSON.stringify(row.params)` -> a JSON-style
//!     number array string ([`json_num_array`]).
//!   * `toExponential(3)` -> [`js_to_exponential`] (display only).
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the four
//! adapters are exposed via [`adapter_newton_rosenbrock`],
//! [`adapter_bfgs_rosenbrock`], [`adapter_gauss_newton_curve_fit`], and
//! [`adapter_levenberg_marquardt_curve_fit`] for the integrator to wire in.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::nonlinear_optimization_models::{
    run_bfgs_rosenbrock, run_gauss_newton_curve_fit, run_levenberg_marquardt_curve_fit,
    run_newton_rosenbrock, NonlinearLeastSquaresParams, NonlinearLeastSquaresResult,
    UnconstrainedOptParams, UnconstrainedOptResult,
};

// ── Shared schema fragments ───────────────────────────────────────────────────

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

fn vector_schema() -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(num(None, None, None, None)),
        min_length: Some(1),
        max_length: None,
        description: None,
    }
}

fn unconstrained_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("x0".to_string(), vector_schema()),
            (
                "maxIter".to_string(),
                num(Some(1.0), None, Some(true), Some(100.0)),
            ),
            ("tol".to_string(), num(Some(0.0), None, None, Some(1e-8))),
        ],
        required: Some(vec![]),
        description: None,
    }
}

fn curve_point_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("x".to_string(), num(None, None, None, None)),
            ("y".to_string(), num(None, None, None, None)),
        ],
        required: Some(vec!["x".to_string(), "y".to_string()]),
        description: None,
    }
}

fn nls_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "points".to_string(),
                ParamSchema::Array {
                    items: Box::new(curve_point_schema()),
                    min_length: Some(2),
                    max_length: None,
                    description: None,
                },
            ),
            ("initial".to_string(), vector_schema()),
            (
                "maxIter".to_string(),
                num(Some(1.0), None, Some(true), Some(30.0)),
            ),
            ("tol".to_string(), num(Some(0.0), None, None, Some(1e-8))),
            ("lambda".to_string(), num(Some(0.0), None, None, Some(0.1))),
        ],
        required: Some(vec![]),
        description: None,
    }
}

// ── Display helpers ───────────────────────────────────────────────────────────

/// JS `Number.prototype.toExponential(digits)` (display only).
fn js_to_exponential(x: f64, digits: usize) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    let raw = format!("{:.*e}", digits, x);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

/// `JSON.stringify(numbers)` for a number array.
fn json_num_array(v: &[f64]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn unconstrained_summary(title: &str, result: &UnconstrainedOptResult) -> String {
    let x = result
        .x
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    [
        title.to_string(),
        "----------------------------------------".to_string(),
        format!(
            "  Objective:      {}",
            js_to_exponential(result.objective, 3)
        ),
        format!("  x*:             [{x}]"),
        format!(
            "  Gradient norm:  {}",
            js_to_exponential(result.gradient_norm, 3)
        ),
        format!("  Iterations:     {}", result.iterations),
        format!(
            "  Stations:       {}",
            result.topology.stations.join(" -> ")
        ),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

fn nls_summary(title: &str, result: &NonlinearLeastSquaresResult) -> String {
    let params = result
        .params
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    [
        title.to_string(),
        "----------------------------------------".to_string(),
        format!("  SSE:            {}", js_to_exponential(result.sse, 3)),
        format!("  Params:         [{params}]"),
        format!(
            "  Gradient norm:  {}",
            js_to_exponential(result.gradient_norm, 3)
        ),
        format!("  Iterations:     {}", result.iterations),
        format!(
            "  Stations:       {}",
            result.topology.stations.join(" -> ")
        ),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

fn write_unconstrained_csv(result: &UnconstrainedOptResult, csv_path: &str) {
    let mut lines = vec![csv_row(["iter", "objective", "gradient_norm", "x"])];
    for row in &result.trace {
        lines.push(csv_row([
            row.iter.to_string(),
            row.objective.to_string(),
            row.gradient_norm.to_string(),
            json_num_array(&row.x),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

fn write_nls_csv(result: &NonlinearLeastSquaresResult, csv_path: &str) {
    let mut lines = vec![csv_row(["iter", "sse", "gradient_norm", "params"])];
    for row in &result.trace {
        lines.push(csv_row([
            row.iter.to_string(),
            row.sse.to_string(),
            row.gradient_norm.to_string(),
            json_num_array(&row.params),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

// ── Unconstrained adapter (Newton / BFGS) ─────────────────────────────────────

/// One example spec attached to an unconstrained adapter.
struct UncExample {
    description: &'static str,
    x0: Vec<f64>,
    max_iter: f64,
    tol: f64,
}

/// Generic adapter for the Newton / BFGS Rosenbrock models.
pub struct UnconstrainedAdapter {
    id: &'static str,
    description: &'static str,
    title: &'static str,
    run_fn: fn(UnconstrainedOptParams) -> UnconstrainedOptResult,
    example: UncExample,
}

impl DESModelRegistration<UnconstrainedOptParams, UnconstrainedOptResult> for UnconstrainedAdapter {
    fn id(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        self.description
    }
    fn schema(&self) -> ParamSchema {
        unconstrained_schema()
    }
    fn run(
        &self,
        params: UnconstrainedOptParams,
        _runtime: &DESRuntimeConfig,
    ) -> UnconstrainedOptResult {
        (self.run_fn)(params)
    }
    fn summarize(
        &self,
        result: &UnconstrainedOptResult,
        _params: &UnconstrainedOptParams,
    ) -> String {
        unconstrained_summary(self.title, result)
    }
    fn write_csv(&self, result: &UnconstrainedOptResult, csv_path: &str) {
        write_unconstrained_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<UnconstrainedOptParams>> {
        vec![RegistrationExample {
            name: "rosenbrock".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: self.id.to_string(),
                description: Some(self.example.description.to_string()),
                parameters: UnconstrainedOptParams {
                    x0: Some(self.example.x0.clone()),
                    max_iter: Some(self.example.max_iter as usize),
                    tol: Some(self.example.tol),
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}

pub fn adapter_newton_rosenbrock() -> UnconstrainedAdapter {
    UnconstrainedAdapter {
        id: "newton-rosenbrock",
        description: "Newton minimization of Rosenbrock through movable optimization state tokens.",
        title: "NEWTON ROSENBROCK (DES)",
        run_fn: run_newton_rosenbrock,
        example: UncExample {
            description: "Newton method on Rosenbrock as a DES state-token loop.",
            x0: vec![-1.2, 1.0],
            max_iter: 50.0,
            tol: 1e-8,
        },
    }
}

pub fn adapter_bfgs_rosenbrock() -> UnconstrainedAdapter {
    UnconstrainedAdapter {
        id: "bfgs-rosenbrock",
        description: "BFGS quasi-Newton minimization of Rosenbrock through movable optimization state tokens.",
        title: "BFGS ROSENBROCK (DES)",
        run_fn: run_bfgs_rosenbrock,
        example: UncExample {
            description: "BFGS on Rosenbrock as a DES state-token loop.",
            x0: vec![-1.2, 1.0],
            max_iter: 100.0,
            tol: 1e-6,
        },
    }
}

// ── Nonlinear least squares adapter (Gauss–Newton / LM) ───────────────────────

/// One example spec attached to an NLS adapter.
struct NlsExample {
    description: &'static str,
    initial: Vec<f64>,
    max_iter: Option<f64>,
    lambda: Option<f64>,
}

/// Generic adapter for the Gauss–Newton / Levenberg–Marquardt curve-fit models.
pub struct NlsAdapter {
    id: &'static str,
    description: &'static str,
    title: &'static str,
    run_fn: fn(NonlinearLeastSquaresParams) -> NonlinearLeastSquaresResult,
    example: NlsExample,
}

impl DESModelRegistration<NonlinearLeastSquaresParams, NonlinearLeastSquaresResult> for NlsAdapter {
    fn id(&self) -> &str {
        self.id
    }
    fn description(&self) -> &str {
        self.description
    }
    fn schema(&self) -> ParamSchema {
        nls_schema()
    }
    fn run(
        &self,
        params: NonlinearLeastSquaresParams,
        _runtime: &DESRuntimeConfig,
    ) -> NonlinearLeastSquaresResult {
        (self.run_fn)(params)
    }
    fn summarize(
        &self,
        result: &NonlinearLeastSquaresResult,
        _params: &NonlinearLeastSquaresParams,
    ) -> String {
        nls_summary(self.title, result)
    }
    fn write_csv(&self, result: &NonlinearLeastSquaresResult, csv_path: &str) {
        write_nls_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<NonlinearLeastSquaresParams>> {
        vec![RegistrationExample {
            name: "exp-decay".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: self.id.to_string(),
                description: Some(self.example.description.to_string()),
                parameters: NonlinearLeastSquaresParams {
                    points: None,
                    initial: Some(self.example.initial.clone()),
                    max_iter: self.example.max_iter.map(|v| v as usize),
                    tol: None,
                    lambda: self.example.lambda,
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}

pub fn adapter_gauss_newton_curve_fit() -> NlsAdapter {
    NlsAdapter {
        id: "gauss-newton-curve-fit",
        description: "Nonlinear exponential curve fitting with Gauss-Newton state-token updates.",
        title: "GAUSS-NEWTON CURVE FIT (DES)",
        run_fn: run_gauss_newton_curve_fit,
        example: NlsExample {
            description: "Fit y = a exp(bx) with Gauss-Newton state tokens.",
            initial: vec![1.0, -0.2],
            max_iter: Some(20.0),
            lambda: None,
        },
    }
}

pub fn adapter_levenberg_marquardt_curve_fit() -> NlsAdapter {
    NlsAdapter {
        id: "levenberg-marquardt-curve-fit",
        description: "Nonlinear exponential curve fitting with Levenberg-Marquardt damped state-token updates.",
        title: "LEVENBERG-MARQUARDT CURVE FIT (DES)",
        run_fn: run_levenberg_marquardt_curve_fit,
        example: NlsExample {
            description: "Fit y = a exp(bx) with LM damped state tokens.",
            initial: vec![1.0, -0.2],
            max_iter: Some(30.0),
            lambda: Some(0.1),
        },
    }
}
