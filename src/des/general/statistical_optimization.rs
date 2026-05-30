//! Port of `src/des/general/statistical-optimization.ts`
//! (module `des::general::statistical_optimization`).
//!
//! Statistical + stochastic optimisation extensions layered above the two-stage
//! SLP: distribution fitting (MLE vs method of moments), CVaR / chance /
//! DRO-lite scenario optimisation, a multi-stage SDDP-style capacity-expansion
//! station, and an adaptive simulation-optimisation station.
//!
//! Conversion notes (per the TS "RUST MIGRATION" header):
//!   * `class … extends PureTransform<I,O>` → struct + `impl Transform`.
//!   * `class … extends FixedPointIterationStation<S>` → struct + `impl
//!     FixedPoint` (local template-method machinery, see below).
//!   * Every TS `rng: () => number` closure → an injected `&mut impl
//!     RandomSource`; the file-local `sampleNormal`/`sampleExponential` are
//!     reimplemented threading the RNG, while `sampleGamma`/`samplePoisson` come
//!     from the already-ported `random_variables` (note the `(rng, …)` argument
//!     order there vs the TS `(…, rng)`).
//!   * `@deprecated` free-fn shims (`fitDistribution`, `sampleFittedDistribution`,
//!     `sampleDemandVector`, `buildDemandScenarios`, `capacityProfit`) are
//!     DROPPED; the public API is the `…Transform` structs, with the underlying
//!     logic kept as private helpers used internally.
//!   * `interface Cut` (private in TS but referenced by the exported `SDDPResult`)
//!     is made `pub` here so the public result struct is well-formed.
//!   * `logger?: OptimizationLogger` → `Option<Box<dyn OptimizationLogger>>`
//!     (rather than the header's suggested `Option<&dyn …>`, to avoid threading a
//!     lifetime through the station structs + trait impls).
//!   * special-function helpers (digamma/trigamma/logGamma) are plain f64 numerics;
//!     family-specific fit failures are recoverable (`Result<_, String>`, caught
//!     by the fitting station), while bad params surface as `Preconditions`
//!     (`Result`) / `panic!`.
//!
//! FLAGGED — NOT in the provided dependency list (defined minimally here):
//!   * The DES base machinery `FixedPointIterationStation`, `runResultStation`,
//!     `ValidationCheck`, `intrinsicCheck`, `monotonicityValidator` (imported in
//!     TS from `./des-base`) are reproduced locally as the `FixedPoint` /
//!     `ResultStation` traits, `run_result_station`, the `ValidationCheck`
//!     struct, and inline validator construction.

#![allow(dead_code)]

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::des::general::des_base::preconditions::{Check, PreconditionError, Preconditions};
use crate::des::general::prng::mulberry32;
use crate::des::general::random_variables::{sample_gamma, sample_poisson};
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::transform::Transform;

// =============================================================================
// Locally-reproduced DES base machinery (FLAGGED above).
// =============================================================================

/// Local equivalent of `des-base/validation.ts`'s `ValidationCheck`.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub observed: Option<String>,
    pub expected: Option<String>,
    pub group: Option<String>,
    pub details: Option<String>,
}

/// Mirror of `intrinsicCheck`: turn a boolean predicate into a check.
fn intrinsic_check(
    name: &str,
    passed: bool,
    expected: &str,
    observed: Option<String>,
    group: &str,
) -> ValidationCheck {
    ValidationCheck {
        name: name.to_string(),
        passed,
        observed,
        expected: Some(expected.to_string()),
        group: Some(group.to_string()),
        details: None,
    }
}

/// Mirror of `monotonicityValidator` for the non-increasing direction.
fn monotonicity_non_increasing_check(
    name: &str,
    xs: &[f64],
    tol: f64,
    group: &str,
) -> ValidationCheck {
    let mut first_violation: i64 = -1;
    for i in 1..xs.len() {
        let d = xs[i] - xs[i - 1];
        if !(d <= tol) {
            first_violation = i as i64;
            break;
        }
    }
    let passed = first_violation == -1;
    let observed = Some(if passed {
        format!("non-increasing (n={})", xs.len())
    } else {
        format!("breaks at i={first_violation}")
    });
    let details = if passed {
        None
    } else {
        let i = first_violation as usize;
        Some(format!("xs[{}]={}  xs[{}]={}", i - 1, xs[i - 1], i, xs[i]))
    };
    ValidationCheck {
        name: name.to_string(),
        passed,
        observed,
        expected: Some("non-increasing".to_string()),
        group: Some(group.to_string()),
        details,
    }
}

/// `convergenceReason: 'converged' | 'maxiter' | 'running'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergenceReason {
    Converged,
    MaxIter,
    Running,
}

/// The bookkeeping that `FixedPointIterationStation` holds in its base class.
#[derive(Clone, Debug)]
pub struct FixedPointCore<S> {
    current: Option<S>,
    iteration: usize,
    last_delta: f64,
    finished: bool,
    convergence_reason: ConvergenceReason,
    tol: f64,
    max_iter: usize,
    delta_history: Vec<f64>,
}

impl<S> FixedPointCore<S> {
    fn new(tol: f64, max_iter: usize) -> Self {
        FixedPointCore {
            current: None,
            iteration: 0,
            last_delta: f64::INFINITY,
            finished: false,
            convergence_reason: ConvergenceReason::Running,
            tol,
            max_iter,
            delta_history: Vec::new(),
        }
    }
}

/// Local equivalent of the `FixedPointIterationStation<S>` template method.
trait FixedPoint {
    type State: Clone;

    fn core(&self) -> &FixedPointCore<Self::State>;
    fn core_mut(&mut self) -> &mut FixedPointCore<Self::State>;

    fn initial_state(&mut self) -> Self::State;
    fn apply_operator(&mut self, prev: &Self::State) -> Self::State;
    fn delta(&self, prev: &Self::State, next: &Self::State) -> f64;

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        self.default_should_stop(iter, last_delta)
    }

    fn default_should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let c = self.core_mut();
        if iter >= c.max_iter {
            c.convergence_reason = ConvergenceReason::MaxIter;
            return true;
        }
        if iter > 0 && last_delta < c.tol {
            c.convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        false
    }

    fn bootstrap(&mut self) {
        let s = self.initial_state();
        self.core_mut().current = Some(s);
    }

    fn get_current(&self) -> &Self::State {
        self.core().current.as_ref().expect("station bootstrapped")
    }

    fn run_time_step(&mut self) {
        if self.core().finished {
            return;
        }
        let (iter, last_delta) = {
            let c = self.core();
            (c.iteration, c.last_delta)
        };
        if self.should_stop(iter, last_delta) {
            self.core_mut().finished = true;
            return;
        }
        let cur = self.core().current.clone().expect("station bootstrapped");
        let next = self.apply_operator(&cur);
        let d = self.delta(&cur, &next);
        let c = self.core_mut();
        c.last_delta = d;
        c.current = Some(next);
        c.iteration += 1;
        c.delta_history.push(d);
    }

    fn run_to_completion(&mut self) {
        while !self.core().finished {
            self.run_time_step();
        }
    }
}

/// Local equivalent of `DESResultStation<R>` + `runResultStation`.
trait ResultStation: FixedPoint {
    type Output;
    fn collect_validation(&self) -> Vec<ValidationCheck>;
    fn result(&self, validation: Vec<ValidationCheck>) -> Self::Output;
}

fn run_result_station<S: ResultStation>(mut station: S) -> S::Output {
    station.run_to_completion();
    let validation = station.collect_validation();
    station.result(validation)
}

// =============================================================================
// Logging
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub enum LogValue {
    Int(i64),
    Num(f64),
    Str(String),
    Ints(Vec<i64>),
    Nums(Vec<f64>),
}

/// A structured log event, mirroring the TS `{kind, level?, …}` object.
#[derive(Clone, Debug)]
pub struct LogEvent {
    pub kind: String,
    pub level: Option<LogLevel>,
    pub fields: Vec<(String, LogValue)>,
}

/// `interface OptimizationLogger` → trait.
pub trait OptimizationLogger {
    fn log(&self, event: &LogEvent);
}

// =============================================================================
// Numeric helpers
// =============================================================================

const MAX_GRID_CANDIDATES: usize = 200_000;
const MAX_SDDP_GRID_POINTS: usize = 2_000;
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

fn mean(xs: &[f64]) -> f64 {
    let s: f64 = xs.iter().sum();
    s / xs.len() as f64
}

fn variance_n(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

fn variance_unbiased(xs: &[f64]) -> f64 {
    if xs.len() <= 1 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

fn stddev(xs: &[f64]) -> f64 {
    variance_unbiased(xs).max(0.0).sqrt()
}

fn round10(x: f64) -> f64 {
    (x * 1e10).round() / 1e10
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = (q * (sorted.len() as f64 - 1.0))
        .max(0.0)
        .min(sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

fn sample_normal(mu: f64, sigma: f64, rng: &mut impl RandomSource) -> f64 {
    let u1 = 1.0 - rng.next_float();
    let u2 = rng.next_float();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mu + sigma * z
}

fn sample_exponential(rate: f64, rng: &mut impl RandomSource) -> f64 {
    let u = 1.0 - rng.next_float();
    -u.ln() / rate
}

fn digamma(x0: f64) -> f64 {
    let mut x = x0;
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

fn trigamma(x0: f64) -> f64 {
    let mut x = x0;
    let mut result = 0.0;
    while x < 7.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + inv2 / 2.0 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
}

fn log_gamma(z: f64) -> f64 {
    const P: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - log_gamma(1.0 - z);
    }
    let mut x = 0.999_999_999_999_809_9;
    let zz = z - 1.0;
    for (i, &p) in P.iter().enumerate() {
        x += p / (zz + i as f64 + 1.0);
    }
    let t = zz + P.len() as f64 - 0.5;
    0.5 * (2.0 * PI).ln() + (zz + 0.5) * t.ln() - t + x.ln()
}

fn log_factorial(n: f64) -> f64 {
    if n < 2.0 {
        return 0.0;
    }
    log_gamma(n + 1.0)
}

fn err_to_string(e: PreconditionError) -> String {
    e.to_string()
}

fn param(map: &HashMap<String, f64>, key: &str) -> f64 {
    map.get(key).copied().unwrap_or(f64::NAN)
}

fn params_of(entries: &[(&str, f64)]) -> HashMap<String, f64> {
    entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

// =============================================================================
// Distribution fitting
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionFamily {
    Normal,
    Lognormal,
    Exponential,
    Gamma,
    Poisson,
    Empirical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMethod {
    Mle,
    Moments,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Real,
    Positive,
    NonnegativeInteger,
    Empirical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalPoint {
    pub value: f64,
    pub prob: f64,
}

#[derive(Clone, Debug)]
pub struct FittedDistribution {
    pub family: DistributionFamily,
    pub method: FitMethod,
    pub params: HashMap<String, f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub mean: f64,
    pub variance: f64,
    pub support: Support,
    pub empirical: Option<Vec<EmpiricalPoint>>,
}

#[derive(Clone, Debug)]
pub struct DistributionFitParams {
    pub samples: Vec<f64>,
    pub families: Option<Vec<DistributionFamily>>,
    pub methods: Option<Vec<FitMethod>>,
}

#[derive(Clone, Debug)]
pub struct DistributionFitResult {
    pub samples: Vec<f64>,
    pub sample_mean: f64,
    pub sample_variance: f64,
    pub fits: Vec<FittedDistribution>,
    pub best_by_aic: FittedDistribution,
    pub validation: Vec<ValidationCheck>,
}

/// Fit ONE `(family, method)` to a sample (the `transform` input).
pub struct DistributionFitter {
    family: DistributionFamily,
    method: FitMethod,
}

impl DistributionFitter {
    pub fn new(family: DistributionFamily, method: FitMethod) -> Self {
        DistributionFitter { family, method }
    }
}

impl<'a> Transform<&'a [f64], FittedDistribution> for DistributionFitter {
    fn transform(&self, samples: &'a [f64]) -> FittedDistribution {
        fit_distribution_impl(samples, self.family, self.method).unwrap_or_else(|e| panic!("{e}"))
    }
}

fn fit_distribution_impl(
    samples: &[f64],
    family: DistributionFamily,
    method: FitMethod,
) -> Result<FittedDistribution, String> {
    let cls = "fitDistribution";
    Preconditions::non_empty(cls, "samples", samples).map_err(err_to_string)?;
    Preconditions::all_finite(cls, "samples", samples).map_err(err_to_string)?;
    let n = samples.len();
    let m = mean(samples);
    let v_n = variance_n(samples).max(1e-12);
    let v_u = variance_unbiased(samples).max(1e-12);

    match family {
        DistributionFamily::Normal => {
            let sigma2 = if method == FitMethod::Mle { v_n } else { v_u };
            let sigma = sigma2.sqrt();
            let ll: f64 = samples
                .iter()
                .map(|&x| -0.5 * (2.0 * PI * sigma2).ln() - (x - m) * (x - m) / (2.0 * sigma2))
                .sum();
            Ok(FittedDistribution {
                family,
                method,
                params: params_of(&[("mu", m), ("sigma", sigma)]),
                log_likelihood: ll,
                aic: 2.0 * 2.0 - 2.0 * ll,
                mean: m,
                variance: sigma2,
                support: Support::Real,
                empirical: None,
            })
        }
        DistributionFamily::Lognormal => {
            let positive = samples.iter().filter(|&&x| x > 0.0).count();
            if positive != n {
                return Err("lognormal fit requires all samples > 0".to_string());
            }
            if method == FitMethod::Mle {
                let logs: Vec<f64> = samples.iter().map(|&x| x.ln()).collect();
                let mu = mean(&logs);
                let sigma2 = variance_n(&logs).max(1e-12);
                let sigma = sigma2.sqrt();
                let ll: f64 = samples
                    .iter()
                    .map(|&x| {
                        -x.ln()
                            - sigma.ln()
                            - 0.5 * (2.0 * PI).ln()
                            - (x.ln() - mu) * (x.ln() - mu) / (2.0 * sigma2)
                    })
                    .sum();
                let mn = (mu + sigma2 / 2.0).exp();
                let vv = (sigma2.exp() - 1.0) * (2.0 * mu + sigma2).exp();
                Ok(FittedDistribution {
                    family,
                    method,
                    params: params_of(&[("muLog", mu), ("sigmaLog", sigma)]),
                    log_likelihood: ll,
                    aic: 2.0 * 2.0 - 2.0 * ll,
                    mean: mn,
                    variance: vv,
                    support: Support::Positive,
                    empirical: None,
                })
            } else {
                let sigma2 = (1.0 + v_u / (m * m).max(1e-12)).ln();
                let mu = m.max(1e-12).ln() - sigma2 / 2.0;
                let sigma = sigma2.max(1e-12).sqrt();
                let ll: f64 = samples
                    .iter()
                    .map(|&x| {
                        -x.ln()
                            - sigma.ln()
                            - 0.5 * (2.0 * PI).ln()
                            - (x.ln() - mu) * (x.ln() - mu) / (2.0 * sigma2)
                    })
                    .sum();
                Ok(FittedDistribution {
                    family,
                    method,
                    params: params_of(&[("muLog", mu), ("sigmaLog", sigma)]),
                    log_likelihood: ll,
                    aic: 2.0 * 2.0 - 2.0 * ll,
                    mean: m,
                    variance: v_u,
                    support: Support::Positive,
                    empirical: None,
                })
            }
        }
        DistributionFamily::Exponential => {
            if samples.iter().any(|&x| x < 0.0) {
                return Err("exponential fit requires samples >= 0".to_string());
            }
            let rate = 1.0 / m.max(1e-12);
            let ll: f64 = samples.iter().map(|&x| rate.ln() - rate * x).sum();
            Ok(FittedDistribution {
                family,
                method,
                params: params_of(&[("rate", rate)]),
                log_likelihood: ll,
                aic: 2.0 - 2.0 * ll,
                mean: 1.0 / rate,
                variance: 1.0 / (rate * rate),
                support: Support::Positive,
                empirical: None,
            })
        }
        DistributionFamily::Gamma => {
            if samples.iter().any(|&x| x <= 0.0) {
                return Err("gamma fit requires samples > 0".to_string());
            }
            let mut shape = (m * m / if method == FitMethod::Mle { v_n } else { v_u }).max(1e-6);
            if method == FitMethod::Mle {
                let log_mean: f64 = samples.iter().map(|&x| x.ln()).sum::<f64>() / n as f64;
                let s = m.ln() - log_mean;
                for _ in 0..25 {
                    let f = shape.ln() - digamma(shape) - s;
                    let fp = 1.0 / shape - trigamma(shape);
                    let next = shape - f / fp;
                    if !next.is_finite() || next <= 0.0 {
                        break;
                    }
                    if (next - shape).abs() < 1e-10 {
                        shape = next;
                        break;
                    }
                    shape = next;
                }
            }
            let scale = m / shape;
            let ll: f64 = samples
                .iter()
                .map(|&x| {
                    (shape - 1.0) * x.ln() - x / scale - shape * scale.ln() - log_gamma(shape)
                })
                .sum();
            Ok(FittedDistribution {
                family,
                method,
                params: params_of(&[("shape", shape), ("scale", scale)]),
                log_likelihood: ll,
                aic: 2.0 * 2.0 - 2.0 * ll,
                mean: shape * scale,
                variance: shape * scale * scale,
                support: Support::Positive,
                empirical: None,
            })
        }
        DistributionFamily::Poisson => {
            if samples.iter().any(|&x| x < 0.0 || x.fract() != 0.0) {
                return Err("poisson fit requires non-negative integer samples".to_string());
            }
            let lambda = m.max(1e-12);
            let ll: f64 = samples
                .iter()
                .map(|&x| x * lambda.ln() - lambda - log_factorial(x))
                .sum();
            Ok(FittedDistribution {
                family,
                method,
                params: params_of(&[("lambda", lambda)]),
                log_likelihood: ll,
                aic: 2.0 - 2.0 * ll,
                mean: lambda,
                variance: lambda,
                support: Support::NonnegativeInteger,
                empirical: None,
            })
        }
        DistributionFamily::Empirical => {
            let mut sorted = samples.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut empirical: Vec<EmpiricalPoint> = Vec::new();
            let mut i = 0;
            while i < sorted.len() {
                let v = sorted[i];
                let mut count = 0usize;
                while i < sorted.len() && sorted[i] == v {
                    count += 1;
                    i += 1;
                }
                empirical.push(EmpiricalPoint {
                    value: v,
                    prob: count as f64 / n as f64,
                });
            }
            let ll: f64 = samples
                .iter()
                .map(|&x| {
                    let prob = empirical
                        .iter()
                        .find(|p| p.value == x)
                        .map(|p| p.prob)
                        .unwrap_or(0.0);
                    prob.max(1e-12).ln()
                })
                .sum();
            let aic = 2.0 * empirical.len() as f64 - 2.0 * ll;
            Ok(FittedDistribution {
                family,
                method,
                params: HashMap::new(),
                log_likelihood: ll,
                aic,
                mean: m,
                variance: v_n,
                support: Support::Empirical,
                empirical: Some(empirical),
            })
        }
    }
}

fn sample_fitted_distribution_unchecked(
    fit: &FittedDistribution,
    rng: &mut impl RandomSource,
) -> f64 {
    match fit.family {
        DistributionFamily::Normal => {
            sample_normal(param(&fit.params, "mu"), param(&fit.params, "sigma"), rng)
        }
        DistributionFamily::Lognormal => sample_normal(
            param(&fit.params, "muLog"),
            param(&fit.params, "sigmaLog"),
            rng,
        )
        .exp(),
        DistributionFamily::Exponential => sample_exponential(param(&fit.params, "rate"), rng),
        DistributionFamily::Gamma => sample_gamma(
            rng,
            param(&fit.params, "shape"),
            param(&fit.params, "scale"),
        ),
        DistributionFamily::Poisson => sample_poisson(rng, param(&fit.params, "lambda")),
        DistributionFamily::Empirical => {
            let empty = Vec::new();
            let points = fit.empirical.as_ref().unwrap_or(&empty);
            let mut u = rng.next_float();
            for p in points {
                u -= p.prob;
                if u <= 0.0 {
                    return p.value;
                }
            }
            points.last().map(|p| p.value).unwrap_or(0.0)
        }
    }
}

/// Draw one sample from a fitted distribution (the `rng` is the `transform` input).
pub struct FittedDistributionSampler {
    fit: FittedDistribution,
}

impl FittedDistributionSampler {
    pub fn new(fit: FittedDistribution) -> Self {
        FittedDistributionSampler { fit }
    }
}

impl<'a, R: RandomSource> Transform<&'a mut R, f64> for FittedDistributionSampler {
    fn transform(&self, rng: &'a mut R) -> f64 {
        validate_fitted_distribution("sampleFittedDistribution", "fit", &self.fit)
            .unwrap_or_else(|e| panic!("{e}"));
        sample_fitted_distribution_unchecked(&self.fit, rng)
    }
}

// =============================================================================
// Distribution fitting station
// =============================================================================

#[derive(Clone, Debug)]
struct DistFitState {
    idx: usize,
    fits: Vec<FittedDistribution>,
}

pub struct DistributionFitStation {
    params: DistributionFitParams,
    families: Vec<DistributionFamily>,
    methods: Vec<FitMethod>,
    errors: Vec<String>,
    core: FixedPointCore<DistFitState>,
}

fn default_families() -> Vec<DistributionFamily> {
    vec![
        DistributionFamily::Normal,
        DistributionFamily::Lognormal,
        DistributionFamily::Exponential,
        DistributionFamily::Gamma,
        DistributionFamily::Poisson,
        DistributionFamily::Empirical,
    ]
}

fn default_methods() -> Vec<FitMethod> {
    vec![FitMethod::Mle, FitMethod::Moments]
}

impl DistributionFitStation {
    pub fn new(params: DistributionFitParams) -> Result<Self, PreconditionError> {
        let families = params.families.clone().unwrap_or_else(default_families);
        let methods = params.methods.clone().unwrap_or_else(default_methods);
        let max_iter = families.len() * methods.len() + 1;
        Self::assert_preconditions(&params, &families, &methods)?;
        let mut st = DistributionFitStation {
            params,
            families,
            methods,
            errors: Vec::new(),
            core: FixedPointCore::new(0.0, max_iter),
        };
        st.bootstrap();
        Ok(st)
    }

    fn assert_preconditions(
        params: &DistributionFitParams,
        families: &[DistributionFamily],
        methods: &[FitMethod],
    ) -> Check {
        Preconditions::non_empty("DistributionFitStation", "samples", &params.samples)?;
        Preconditions::check(
            "DistributionFitStation",
            "samples.length",
            "be at least 2",
            params.samples.len() >= 2,
            Some(params.samples.len().to_string()),
        )?;
        Preconditions::all_finite("DistributionFitStation", "samples", &params.samples)?;
        Preconditions::non_empty("DistributionFitStation", "families", families)?;
        Preconditions::non_empty("DistributionFitStation", "methods", methods)?;
        Ok(())
    }
}

impl FixedPoint for DistributionFitStation {
    type State = DistFitState;

    fn core(&self) -> &FixedPointCore<Self::State> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut FixedPointCore<Self::State> {
        &mut self.core
    }

    fn initial_state(&mut self) -> Self::State {
        DistFitState {
            idx: 0,
            fits: Vec::new(),
        }
    }

    fn apply_operator(&mut self, prev: &Self::State) -> Self::State {
        let mut pairs: Vec<(DistributionFamily, FitMethod)> = Vec::new();
        for &f in &self.families {
            for &m in &self.methods {
                pairs.push((f, m));
            }
        }
        if prev.idx >= pairs.len() {
            return prev.clone();
        }
        let (family, method) = pairs[prev.idx];
        let mut fits = prev.fits.clone();
        match fit_distribution_impl(&self.params.samples, family, method) {
            Ok(f) => fits.push(f),
            Err(e) => self.errors.push(format!("{family:?}/{method:?}: {e}")),
        }
        DistFitState {
            idx: prev.idx + 1,
            fits,
        }
    }

    fn delta(&self, prev: &Self::State, next: &Self::State) -> f64 {
        if next.idx == prev.idx {
            0.0
        } else {
            1.0
        }
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let total = self.families.len() * self.methods.len();
        let idx = self.get_current().idx;
        if idx >= total {
            self.core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        self.default_should_stop(iter, last_delta)
    }
}

impl ResultStation for DistributionFitStation {
    type Output = DistributionFitResult;

    fn collect_validation(&self) -> Vec<ValidationCheck> {
        let fits = &self.get_current().fits;
        vec![
            intrinsic_check(
                "distribution-fit-has-at-least-one-fit",
                !fits.is_empty(),
                "at least one admissible family/method",
                None,
                "distribution-fit",
            ),
            intrinsic_check(
                "distribution-fit-aic-finite",
                fits.iter().all(|f| f.aic.is_finite()),
                "finite AIC for all fits",
                None,
                "distribution-fit",
            ),
        ]
    }

    fn result(&self, validation: Vec<ValidationCheck>) -> DistributionFitResult {
        let mut fits = self.get_current().fits.clone();
        fits.sort_by(|a, b| {
            a.aic
                .partial_cmp(&b.aic)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if fits.is_empty() {
            panic!("no distribution fit succeeded: {}", self.errors.join("; "));
        }
        DistributionFitResult {
            samples: self.params.samples.clone(),
            sample_mean: mean(&self.params.samples),
            sample_variance: variance_unbiased(&self.params.samples),
            best_by_aic: fits[0].clone(),
            fits,
            validation,
        }
    }
}

pub fn run_distribution_fit(
    params: DistributionFitParams,
) -> Result<DistributionFitResult, PreconditionError> {
    Ok(run_result_station(DistributionFitStation::new(params)?))
}

// =============================================================================
// Shared capacity-planning scenario utilities.
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandRange {
    pub low: f64,
    pub high: f64,
}

/// `interface DemandSpec` (discriminated by `kind`) → enum.
#[derive(Clone, Debug)]
pub enum DemandSpec {
    Uniform(Vec<DemandRange>),
    Fitted(Vec<FittedDistribution>),
    Empirical(Vec<Vec<EmpiricalPoint>>),
}

#[derive(Clone, Debug)]
pub struct DemandScenario {
    pub demand: Vec<f64>,
    pub prob: f64,
}

fn validate_fitted_distribution(model: &str, param_name: &str, fit: &FittedDistribution) -> Check {
    match fit.family {
        DistributionFamily::Normal => {
            Preconditions::finite(
                model,
                &format!("{param_name}.params.mu"),
                param(&fit.params, "mu"),
            )?;
            Preconditions::positive(
                model,
                &format!("{param_name}.params.sigma"),
                param(&fit.params, "sigma"),
            )?;
        }
        DistributionFamily::Lognormal => {
            Preconditions::finite(
                model,
                &format!("{param_name}.params.muLog"),
                param(&fit.params, "muLog"),
            )?;
            Preconditions::positive(
                model,
                &format!("{param_name}.params.sigmaLog"),
                param(&fit.params, "sigmaLog"),
            )?;
        }
        DistributionFamily::Exponential => {
            Preconditions::positive(
                model,
                &format!("{param_name}.params.rate"),
                param(&fit.params, "rate"),
            )?;
        }
        DistributionFamily::Gamma => {
            Preconditions::positive(
                model,
                &format!("{param_name}.params.shape"),
                param(&fit.params, "shape"),
            )?;
            Preconditions::positive(
                model,
                &format!("{param_name}.params.scale"),
                param(&fit.params, "scale"),
            )?;
        }
        DistributionFamily::Poisson => {
            Preconditions::non_negative(
                model,
                &format!("{param_name}.params.lambda"),
                param(&fit.params, "lambda"),
            )?;
        }
        DistributionFamily::Empirical => {
            let empty = Vec::new();
            let points = fit.empirical.as_ref().unwrap_or(&empty);
            Preconditions::non_empty(model, &format!("{param_name}.empirical"), points)?;
            let values: Vec<f64> = points.iter().map(|p| p.value).collect();
            Preconditions::all_finite(model, &format!("{param_name}.empirical.values"), &values)?;
            let probs: Vec<f64> = points.iter().map(|p| p.prob).collect();
            Preconditions::probability_vector(
                model,
                &format!("{param_name}.empirical.prob"),
                &probs,
                1e-6,
            )?;
        }
    }
    Ok(())
}

fn validate_demand_spec(
    model: &str,
    param_name: &str,
    spec: &DemandSpec,
    n_products: usize,
) -> Check {
    Preconditions::integer_in_range(model, "nProducts", n_products as f64, 1.0, MAX_SAFE_INTEGER)?;
    match spec {
        DemandSpec::Uniform(ranges) => {
            Preconditions::length_eq(model, &format!("{param_name}.ranges"), ranges, n_products)?;
            for (i, r) in ranges.iter().enumerate() {
                Preconditions::non_negative(
                    model,
                    &format!("{param_name}.ranges[{i}].low"),
                    r.low,
                )?;
                Preconditions::non_negative(
                    model,
                    &format!("{param_name}.ranges[{i}].high"),
                    r.high,
                )?;
                Preconditions::check(
                    model,
                    &format!("{param_name}.ranges[{i}].high"),
                    "be >= low",
                    r.high >= r.low,
                    Some(format!("low={}, high={}", r.low, r.high)),
                )?;
            }
        }
        DemandSpec::Fitted(fitted) => {
            Preconditions::length_eq(model, &format!("{param_name}.fitted"), fitted, n_products)?;
            for (i, f) in fitted.iter().enumerate() {
                validate_fitted_distribution(model, &format!("{param_name}.fitted[{i}]"), f)?;
            }
        }
        DemandSpec::Empirical(empirical) => {
            Preconditions::length_eq(
                model,
                &format!("{param_name}.empirical"),
                empirical,
                n_products,
            )?;
            for (i, points) in empirical.iter().enumerate() {
                Preconditions::non_empty(model, &format!("{param_name}.empirical[{i}]"), points)?;
                let values: Vec<f64> = points.iter().map(|p| p.value).collect();
                Preconditions::all_finite(
                    model,
                    &format!("{param_name}.empirical[{i}].values"),
                    &values,
                )?;
                let probs: Vec<f64> = points.iter().map(|p| p.prob).collect();
                Preconditions::probability_vector(
                    model,
                    &format!("{param_name}.empirical[{i}].prob"),
                    &probs,
                    1e-6,
                )?;
            }
        }
    }
    Ok(())
}

fn sample_demand_vector_unchecked(
    spec: &DemandSpec,
    _n_products: usize,
    rng: &mut impl RandomSource,
) -> Vec<f64> {
    match spec {
        DemandSpec::Uniform(ranges) => ranges
            .iter()
            .map(|r| r.low + rng.next_float() * (r.high - r.low))
            .collect(),
        DemandSpec::Fitted(fitted) => fitted
            .iter()
            .map(|f| sample_fitted_distribution_unchecked(f, rng).max(0.0))
            .collect(),
        DemandSpec::Empirical(empirical) => empirical
            .iter()
            .map(|points| {
                let mut u = rng.next_float();
                for p in points {
                    u -= p.prob;
                    if u <= 0.0 {
                        return p.value;
                    }
                }
                points.last().map(|p| p.value).unwrap_or(0.0)
            })
            .collect(),
    }
}

/// Sample one demand vector (the `rng` is the `transform` input).
pub struct DemandVectorSampler {
    spec: DemandSpec,
    n_products: usize,
}

impl DemandVectorSampler {
    pub fn new(spec: DemandSpec, n_products: usize) -> Self {
        DemandVectorSampler { spec, n_products }
    }
}

impl<'a, R: RandomSource> Transform<&'a mut R, Vec<f64>> for DemandVectorSampler {
    fn transform(&self, rng: &'a mut R) -> Vec<f64> {
        validate_demand_spec("sampleDemandVector", "spec", &self.spec, self.n_products)
            .unwrap_or_else(|e| panic!("{e}"));
        sample_demand_vector_unchecked(&self.spec, self.n_products, rng)
    }
}

fn build_demand_scenarios(
    spec: &DemandSpec,
    n_products: usize,
    n: usize,
    seed: u32,
) -> Vec<DemandScenario> {
    let mut rng = mulberry32(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(DemandScenario {
            demand: sample_demand_vector_unchecked(spec, n_products, &mut rng),
            prob: 1.0 / n as f64,
        });
    }
    out
}

/// Build `N` equiprobable demand scenarios from a seeded RNG (the spec is the input).
pub struct DemandScenarioBuilder {
    n_products: usize,
    n: usize,
    seed: u32,
}

impl DemandScenarioBuilder {
    pub fn new(n_products: usize, n: usize, seed: u32) -> Self {
        DemandScenarioBuilder {
            n_products,
            n,
            seed,
        }
    }
}

impl<'a> Transform<&'a DemandSpec, Vec<DemandScenario>> for DemandScenarioBuilder {
    fn transform(&self, spec: &'a DemandSpec) -> Vec<DemandScenario> {
        Preconditions::integer_in_range(
            "buildDemandScenarios",
            "N",
            self.n as f64,
            1.0,
            MAX_SAFE_INTEGER,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        validate_demand_spec("buildDemandScenarios", "spec", spec, self.n_products)
            .unwrap_or_else(|e| panic!("{e}"));
        build_demand_scenarios(spec, self.n_products, self.n, self.seed)
    }
}

/// Capacity decision `x` and a realised `demand` vector for [`CapacityProfit`].
#[derive(Clone, Debug)]
pub struct CapacityProfitInput {
    pub x: Vec<f64>,
    pub demand: Vec<f64>,
}

fn capacity_profit(x: &[f64], demand: &[f64], cost: &[f64], price: &[f64]) -> f64 {
    let mut z = 0.0;
    for i in 0..x.len() {
        z += -cost[i] * x[i] + price[i] * x[i].min(demand[i]);
    }
    z
}

/// Newsvendor-style profit of a capacity decision under one demand realisation.
pub struct CapacityProfit {
    cost: Vec<f64>,
    price: Vec<f64>,
}

impl CapacityProfit {
    pub fn new(cost: Vec<f64>, price: Vec<f64>) -> Self {
        CapacityProfit { cost, price }
    }
}

impl Transform<CapacityProfitInput, f64> for CapacityProfit {
    fn transform(&self, input: CapacityProfitInput) -> f64 {
        capacity_profit(&input.x, &input.demand, &self.cost, &self.price)
    }
}

fn total_shortfall(x: &[f64], demand: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        s += (demand[i] - x[i]).max(0.0);
    }
    s
}

fn enumerate_grid(n: usize, x_max: f64, step: f64) -> Vec<Vec<f64>> {
    let mut levels: Vec<f64> = Vec::new();
    let mut x = 0.0;
    while x <= x_max + 1e-9 {
        levels.push(round10(x));
        x += step;
    }
    let mut out: Vec<Vec<f64>> = Vec::new();
    let mut cur = vec![0.0; n];
    fill_grid(0, n, &levels, &mut cur, &mut out);
    out
}

fn fill_grid(i: usize, n: usize, levels: &[f64], cur: &mut [f64], out: &mut Vec<Vec<f64>>) {
    if i == n {
        out.push(cur.to_vec());
        return;
    }
    for &v in levels {
        cur[i] = v;
        fill_grid(i + 1, n, levels, cur, out);
    }
}

fn grid_point_count(n: usize, x_max: f64, step: f64) -> f64 {
    ((x_max / step).floor() + 1.0).powi(n as i32)
}

fn risk_grid_size(params: &RiskCapacityParams) -> Result<usize, PreconditionError> {
    Preconditions::non_empty("RiskCapacityStation", "cost", &params.cost)?;
    Preconditions::positive("RiskCapacityStation", "xMax", params.x_max)?;
    Preconditions::positive("RiskCapacityStation", "step", params.step)?;
    let count = grid_point_count(params.cost.len(), params.x_max, params.step);
    Preconditions::integer_in_range(
        "RiskCapacityStation",
        "grid candidate count",
        count,
        1.0,
        MAX_GRID_CANDIDATES as f64,
    )?;
    Ok(count as usize)
}

fn sddp_grid_size(params: &SDDPParams) -> Result<usize, PreconditionError> {
    Preconditions::positive("CapacityExpansionSDDPStation", "xMax", params.x_max)?;
    Preconditions::positive("CapacityExpansionSDDPStation", "step", params.step)?;
    let count = (params.x_max / params.step).floor() + 1.0;
    Preconditions::integer_in_range(
        "CapacityExpansionSDDPStation",
        "grid point count",
        count,
        1.0,
        MAX_SDDP_GRID_POINTS as f64,
    )?;
    Ok(count as usize)
}

fn adaptive_max_iter(params: &AdaptiveSimOptParams) -> Result<usize, PreconditionError> {
    Preconditions::non_empty(
        "AdaptiveSimulationOptimizerStation",
        "alternatives",
        &params.alternatives,
    )?;
    Preconditions::integer_in_range(
        "AdaptiveSimulationOptimizerStation",
        "batchSize",
        params.batch_size as f64,
        1.0,
        MAX_SAFE_INTEGER,
    )?;
    Preconditions::integer_in_range(
        "AdaptiveSimulationOptimizerStation",
        "budget",
        params.budget as f64,
        1.0,
        MAX_SAFE_INTEGER,
    )?;
    let ceil_div = params.budget.div_ceil(params.batch_size);
    Ok(ceil_div + params.alternatives.len() + 2)
}

// =============================================================================
// CVaR / chance / DRO-lite scenario optimisation.
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskKind {
    Expectation,
    Cvar,
    Chance,
    Dro,
}

#[derive(Clone, Copy, Debug)]
pub struct RiskConfig {
    pub kind: RiskKind,
    pub alpha: Option<f64>,
    pub lambda: Option<f64>,
    pub min_service_level: Option<f64>,
    pub shortfall_limit: Option<f64>,
    pub radius: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RiskCapacityParams {
    pub cost: Vec<f64>,
    pub price: Vec<f64>,
    pub demand: DemandSpec,
    pub num_scenarios: usize,
    pub seed: u32,
    pub x_max: f64,
    pub step: f64,
    pub risk: RiskConfig,
}

#[derive(Clone, Debug)]
pub struct RiskCandidateResult {
    pub x: Vec<f64>,
    pub mean_profit: f64,
    pub sd_profit: f64,
    pub cvar_loss: f64,
    pub service_level: f64,
    pub robust_objective: f64,
    pub feasible: bool,
}

#[derive(Clone, Debug)]
pub struct RiskCapacityResult {
    pub params: RiskCapacityParams,
    pub scenarios: Vec<DemandScenario>,
    pub candidates: Vec<RiskCandidateResult>,
    pub best: RiskCandidateResult,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
struct RiskState {
    idx: usize,
    candidates: Vec<RiskCandidateResult>,
}

pub struct RiskCapacityStation {
    params: RiskCapacityParams,
    scenarios: Vec<DemandScenario>,
    grid: Vec<Vec<f64>>,
    core: FixedPointCore<RiskState>,
}

impl RiskCapacityStation {
    pub fn new(params: RiskCapacityParams) -> Result<Self, PreconditionError> {
        let grid_count = risk_grid_size(&params)?;
        let max_iter = grid_count + 1;
        Self::assert_preconditions(&params, grid_count)?;
        let scenarios = build_demand_scenarios(
            &params.demand,
            params.cost.len(),
            params.num_scenarios,
            params.seed,
        );
        let grid = enumerate_grid(params.cost.len(), params.x_max, params.step);
        let mut st = RiskCapacityStation {
            params,
            scenarios,
            grid,
            core: FixedPointCore::new(0.0, max_iter),
        };
        st.bootstrap();
        Ok(st)
    }

    fn assert_preconditions(p: &RiskCapacityParams, grid_count: usize) -> Check {
        Preconditions::non_empty("RiskCapacityStation", "cost", &p.cost)?;
        Preconditions::length_eq("RiskCapacityStation", "price", &p.price, p.cost.len())?;
        Preconditions::all_finite("RiskCapacityStation", "cost", &p.cost)?;
        Preconditions::all_finite("RiskCapacityStation", "price", &p.price)?;
        Preconditions::arr_non_negative("RiskCapacityStation", "cost", &p.cost)?;
        Preconditions::arr_non_negative("RiskCapacityStation", "price", &p.price)?;
        Preconditions::integer_in_range(
            "RiskCapacityStation",
            "numScenarios",
            p.num_scenarios as f64,
            1.0,
            MAX_SAFE_INTEGER,
        )?;
        Preconditions::positive("RiskCapacityStation", "xMax", p.x_max)?;
        Preconditions::positive("RiskCapacityStation", "step", p.step)?;
        Preconditions::check(
            "RiskCapacityStation",
            "step",
            "be <= xMax",
            p.step <= p.x_max,
            Some(p.step.to_string()),
        )?;
        Preconditions::integer_in_range(
            "RiskCapacityStation",
            "grid candidate count",
            grid_count as f64,
            1.0,
            MAX_GRID_CANDIDATES as f64,
        )?;
        validate_demand_spec("RiskCapacityStation", "demand", &p.demand, p.cost.len())?;
        if let Some(alpha) = p.risk.alpha {
            Preconditions::in_range("RiskCapacityStation", "risk.alpha", alpha, 0.5, 0.999)?;
        }
        if let Some(min_sl) = p.risk.min_service_level {
            Preconditions::in_range(
                "RiskCapacityStation",
                "risk.minServiceLevel",
                min_sl,
                0.0,
                1.0,
            )?;
        }
        if let Some(sl) = p.risk.shortfall_limit {
            Preconditions::non_negative("RiskCapacityStation", "risk.shortfallLimit", sl)?;
        }
        if let Some(l) = p.risk.lambda {
            Preconditions::non_negative("RiskCapacityStation", "risk.lambda", l)?;
        }
        if let Some(r) = p.risk.radius {
            Preconditions::non_negative("RiskCapacityStation", "risk.radius", r)?;
        }
        Ok(())
    }

    fn best(&self) -> RiskCandidateResult {
        let cs = &self.get_current().candidates;
        let feasible: Vec<&RiskCandidateResult> = cs.iter().filter(|c| c.feasible).collect();
        let pool: Vec<&RiskCandidateResult> = if !feasible.is_empty() {
            feasible
        } else {
            cs.iter().collect()
        };
        let mut best = pool[0];
        for &c in pool.iter().skip(1) {
            if c.robust_objective > best.robust_objective {
                best = c;
            }
        }
        best.clone()
    }

    fn evaluate(&self, x: &[f64]) -> RiskCandidateResult {
        let profits: Vec<f64> = self
            .scenarios
            .iter()
            .map(|s| capacity_profit(x, &s.demand, &self.params.cost, &self.params.price))
            .collect();
        let shortfalls: Vec<f64> = self
            .scenarios
            .iter()
            .map(|s| total_shortfall(x, &s.demand))
            .collect();
        let mut losses = shortfalls.clone();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = self.params.risk.alpha.unwrap_or(0.9);
        let var_loss = quantile_sorted(&losses, alpha);
        let tail: Vec<f64> = losses
            .iter()
            .copied()
            .filter(|&l| l >= var_loss - 1e-12)
            .collect();
        let cvar_loss = if tail.is_empty() {
            var_loss
        } else {
            mean(&tail)
        };
        let mean_profit = mean(&profits);
        let sd_profit = stddev(&profits);
        let shortfall_limit = self.params.risk.shortfall_limit.unwrap_or(0.0);
        let service_level = shortfalls
            .iter()
            .filter(|&&s| s <= shortfall_limit + 1e-12)
            .count() as f64
            / shortfalls.len() as f64;
        let min_sl = self.params.risk.min_service_level.unwrap_or(0.0);
        let feasible = self.params.risk.kind != RiskKind::Chance || service_level >= min_sl - 1e-12;
        let mut robust_objective = mean_profit;
        match self.params.risk.kind {
            RiskKind::Cvar => {
                robust_objective = mean_profit - self.params.risk.lambda.unwrap_or(1.0) * cvar_loss
            }
            RiskKind::Dro => {
                robust_objective = mean_profit - self.params.risk.radius.unwrap_or(1.0) * sd_profit
            }
            RiskKind::Chance if !feasible => {
                robust_objective = mean_profit - 1e6 * (min_sl - service_level)
            }
            _ => {}
        }
        RiskCandidateResult {
            x: x.to_vec(),
            mean_profit,
            sd_profit,
            cvar_loss,
            service_level,
            robust_objective,
            feasible,
        }
    }
}

impl FixedPoint for RiskCapacityStation {
    type State = RiskState;

    fn core(&self) -> &FixedPointCore<Self::State> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut FixedPointCore<Self::State> {
        &mut self.core
    }

    fn initial_state(&mut self) -> Self::State {
        RiskState {
            idx: 0,
            candidates: Vec::new(),
        }
    }

    fn apply_operator(&mut self, prev: &Self::State) -> Self::State {
        if prev.idx >= self.grid.len() {
            return prev.clone();
        }
        let mut candidates = prev.candidates.clone();
        let cand = self.evaluate(&self.grid[prev.idx]);
        candidates.push(cand);
        RiskState {
            idx: prev.idx + 1,
            candidates,
        }
    }

    fn delta(&self, prev: &Self::State, next: &Self::State) -> f64 {
        if next.idx == prev.idx {
            0.0
        } else {
            1.0
        }
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let idx = self.get_current().idx;
        if idx >= self.grid.len() {
            self.core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        self.default_should_stop(iter, last_delta)
    }
}

impl ResultStation for RiskCapacityStation {
    type Output = RiskCapacityResult;

    fn collect_validation(&self) -> Vec<ValidationCheck> {
        let candidates = &self.get_current().candidates;
        let any_feasible = candidates.iter().any(|c| c.feasible);
        let best_feasible = self.best().feasible;
        vec![
            intrinsic_check(
                "risk-capacity-evaluated-entire-grid",
                candidates.len() == self.grid.len(),
                "all grid candidates evaluated",
                Some(format!("{}/{}", candidates.len(), self.grid.len())),
                "risk-capacity",
            ),
            intrinsic_check(
                "risk-capacity-best-feasible-if-feasible-exists",
                !any_feasible || best_feasible,
                "best candidate is feasible when any feasible candidate exists",
                None,
                "risk-capacity",
            ),
        ]
    }

    fn result(&self, validation: Vec<ValidationCheck>) -> RiskCapacityResult {
        RiskCapacityResult {
            params: self.params.clone(),
            scenarios: self.scenarios.clone(),
            candidates: self.get_current().candidates.clone(),
            best: self.best(),
            validation,
        }
    }
}

pub fn run_risk_capacity(
    params: RiskCapacityParams,
) -> Result<RiskCapacityResult, PreconditionError> {
    Ok(run_result_station(RiskCapacityStation::new(params)?))
}

// =============================================================================
// Multi-stage SDDP-style capacity expansion.
// =============================================================================

/// Optimality cut (private in TS, but referenced by the public `SDDPResult`).
#[derive(Clone, Debug)]
pub struct Cut {
    pub slope: f64,
    pub intercept: f64,
    pub stage: usize,
    pub at: f64,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct SDDPParams {
    pub horizon: usize,
    pub demand: Vec<DemandRange>,
    pub price: Vec<f64>,
    pub expansion_cost: Vec<f64>,
    pub initial_capacity: f64,
    pub x_max: f64,
    pub step: f64,
    pub samples_per_stage: usize,
    pub seed: u32,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SDDPIteration {
    pub iter: usize,
    pub upper_bound: f64,
    pub lower_bound: f64,
    pub exact_objective: f64,
    pub gap: f64,
    pub cut_counts: Vec<usize>,
    pub forward_states: Vec<f64>,
    pub forward_return: f64,
}

#[derive(Clone, Debug)]
pub struct SDDPResult {
    pub params: SDDPParams,
    pub exact_objective: f64,
    pub exact_policy: Vec<Vec<f64>>,
    pub final_upper_bound: f64,
    pub final_lower_bound: f64,
    pub gap: f64,
    pub cuts: Vec<Vec<Cut>>,
    pub trace: Vec<SDDPIteration>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
struct SDDPState {
    iter: usize,
    cuts: Vec<Vec<Cut>>,
    upper_bound: f64,
    lower_bound: f64,
    forward_states: Vec<f64>,
    forward_return: f64,
}

#[derive(Clone, Debug)]
struct ExactDp {
    objective: f64,
    policy: Vec<Vec<f64>>,
}

fn clone_cuts(cuts: &[Vec<Cut>]) -> Vec<Vec<Cut>> {
    cuts.to_vec()
}

fn build_sddp_grid(params: &SDDPParams) -> Vec<f64> {
    let mut grid = Vec::new();
    let mut x = 0.0;
    while x <= params.x_max + 1e-9 {
        grid.push(round10(x));
        x += params.step;
    }
    grid
}

fn build_stage_scenarios(params: &SDDPParams) -> Vec<Vec<f64>> {
    let mut rng = mulberry32(params.seed);
    params
        .demand
        .iter()
        .map(|r| {
            let mut xs = Vec::with_capacity(params.samples_per_stage);
            for _ in 0..params.samples_per_stage {
                xs.push(r.low + rng.next_float() * (r.high - r.low));
            }
            xs
        })
        .collect()
}

pub struct CapacityExpansionSDDPStation {
    params: SDDPParams,
    logger: Option<Box<dyn OptimizationLogger>>,
    grid: Vec<f64>,
    scenarios: Vec<Vec<f64>>,
    exact: ExactDp,
    trace: Vec<SDDPIteration>,
    upper_history: Vec<f64>,
    core: FixedPointCore<SDDPState>,
}

impl CapacityExpansionSDDPStation {
    pub fn new(
        params: SDDPParams,
        logger: Option<Box<dyn OptimizationLogger>>,
    ) -> Result<Self, PreconditionError> {
        let tol = params.tol.unwrap_or(1e-4);
        let max_iter = params.max_iter.unwrap_or(60);
        Self::assert_preconditions(&params)?;
        let grid = build_sddp_grid(&params);
        let scenarios = build_stage_scenarios(&params);
        let exact = solve_exact_dp(&params, &grid, &scenarios);
        let mut st = CapacityExpansionSDDPStation {
            params,
            logger,
            grid,
            scenarios,
            exact,
            trace: Vec::new(),
            upper_history: Vec::new(),
            core: FixedPointCore::new(tol, max_iter),
        };
        st.bootstrap();
        Ok(st)
    }

    fn assert_preconditions(p: &SDDPParams) -> Check {
        Preconditions::integer_in_range(
            "CapacityExpansionSDDPStation",
            "horizon",
            p.horizon as f64,
            1.0,
            200.0,
        )?;
        Preconditions::length_eq(
            "CapacityExpansionSDDPStation",
            "demand",
            &p.demand,
            p.horizon,
        )?;
        Preconditions::length_eq("CapacityExpansionSDDPStation", "price", &p.price, p.horizon)?;
        Preconditions::length_eq(
            "CapacityExpansionSDDPStation",
            "expansionCost",
            &p.expansion_cost,
            p.horizon,
        )?;
        Preconditions::non_negative(
            "CapacityExpansionSDDPStation",
            "initialCapacity",
            p.initial_capacity,
        )?;
        Preconditions::positive("CapacityExpansionSDDPStation", "xMax", p.x_max)?;
        Preconditions::positive("CapacityExpansionSDDPStation", "step", p.step)?;
        sddp_grid_size(p)?;
        Preconditions::integer_in_range(
            "CapacityExpansionSDDPStation",
            "samplesPerStage",
            p.samples_per_stage as f64,
            1.0,
            1_000_000.0,
        )?;
        if let Some(mi) = p.max_iter {
            Preconditions::integer_in_range(
                "CapacityExpansionSDDPStation",
                "maxIter",
                mi as f64,
                1.0,
                MAX_SAFE_INTEGER,
            )?;
        }
        if let Some(t) = p.tol {
            Preconditions::non_negative("CapacityExpansionSDDPStation", "tol", t)?;
        }
        Preconditions::check(
            "CapacityExpansionSDDPStation",
            "initialCapacity",
            "be <= xMax",
            p.initial_capacity <= p.x_max,
            Some(p.initial_capacity.to_string()),
        )?;
        for t in 0..p.horizon {
            Preconditions::non_negative(
                "CapacityExpansionSDDPStation",
                &format!("demand[{t}].low"),
                p.demand[t].low,
            )?;
            Preconditions::non_negative(
                "CapacityExpansionSDDPStation",
                &format!("demand[{t}].high"),
                p.demand[t].high,
            )?;
            Preconditions::check(
                "CapacityExpansionSDDPStation",
                &format!("demand[{t}].high"),
                "be >= low",
                p.demand[t].high >= p.demand[t].low,
                Some(format!(
                    "low={}, high={}",
                    p.demand[t].low, p.demand[t].high
                )),
            )?;
            Preconditions::non_negative(
                "CapacityExpansionSDDPStation",
                &format!("price[{t}]"),
                p.price[t],
            )?;
            Preconditions::non_negative(
                "CapacityExpansionSDDPStation",
                &format!("expansionCost[{t}]"),
                p.expansion_cost[t],
            )?;
        }
        Ok(())
    }

    fn remaining_revenue_upper(&self, t0: usize) -> f64 {
        let mut s = 0.0;
        for t in t0..self.params.horizon {
            s += self.params.price[t] * self.params.demand[t].high;
        }
        s
    }

    fn idx(&self, x: f64) -> usize {
        let k = (x / self.params.step).round() as i64;
        let max = self.grid.len() as i64 - 1;
        k.max(0).min(max) as usize
    }

    fn vhat(&self, cuts: &[Vec<Cut>], stage: usize, x: f64) -> f64 {
        if stage >= self.params.horizon {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        for c in &cuts[stage] {
            let v = c.slope * x + c.intercept;
            if v < best {
                best = v;
            }
        }
        best
    }

    fn bellman_approx(&self, cuts: &[Vec<Cut>], stage: usize, x: f64) -> f64 {
        let p = &self.params;
        let mut best = f64::NEG_INFINITY;
        for &x_next in &self.grid {
            if x_next + 1e-9 < x {
                continue;
            }
            let mut q = -p.expansion_cost[stage] * (x_next - x);
            let mut rev = 0.0;
            for &d in &self.scenarios[stage] {
                rev += p.price[stage] * x_next.min(d);
            }
            q += rev / self.scenarios[stage].len() as f64 + self.vhat(cuts, stage + 1, x_next);
            if q > best {
                best = q;
            }
        }
        best
    }

    fn make_cut(&self, cuts: &[Vec<Cut>], stage: usize, x: f64) -> Cut {
        let h = self.params.step;
        let x_lo = (x - h).max(0.0);
        let x_hi = (x + h).min(self.params.x_max);
        let val = self.bellman_approx(cuts, stage, x);
        let lo = self.bellman_approx(cuts, stage, x_lo);
        let hi = self.bellman_approx(cuts, stage, x_hi);
        let slope = if x_hi == x_lo {
            0.0
        } else {
            (hi - lo) / (x_hi - x_lo)
        };
        Cut {
            slope,
            intercept: val - slope * x,
            stage,
            at: x,
            value: val,
        }
    }

    fn choose_next(&self, cuts: &[Vec<Cut>], stage: usize, x: f64, demand: Option<f64>) -> f64 {
        let p = &self.params;
        let mut best_x = x;
        let mut best = f64::NEG_INFINITY;
        for &x_next in &self.grid {
            if x_next + 1e-9 < x {
                continue;
            }
            let revenue = match demand {
                None => {
                    self.scenarios[stage]
                        .iter()
                        .map(|&d| p.price[stage] * x_next.min(d))
                        .sum::<f64>()
                        / self.scenarios[stage].len() as f64
                }
                Some(d) => p.price[stage] * x_next.min(d),
            };
            let q = -p.expansion_cost[stage] * (x_next - x)
                + revenue
                + self.vhat(cuts, stage + 1, x_next);
            if q > best + 1e-12 {
                best = q;
                best_x = x_next;
            }
        }
        best_x
    }

    fn forward_pass(&self, cuts: &[Vec<Cut>], iter: usize) -> (Vec<f64>, f64) {
        let mut rng = mulberry32(
            self.params
                .seed
                .wrapping_add(1_000_003u32.wrapping_mul((iter as u32).wrapping_add(1))),
        );
        let mut states = vec![self.params.initial_capacity];
        let mut x = self.params.initial_capacity;
        let mut total = 0.0;
        for t in 0..self.params.horizon {
            let r = self.params.demand[t];
            let d = r.low + rng.next_float() * (r.high - r.low);
            let x_next = self.choose_next(cuts, t, x, Some(d));
            total += -self.params.expansion_cost[t] * (x_next - x)
                + self.params.price[t] * x_next.min(d);
            x = x_next;
            states.push(x);
        }
        (states, total)
    }

    fn evaluate_greedy_policy(&self, cuts: &[Vec<Cut>]) -> f64 {
        let t_horizon = self.params.horizon;
        let mut next = vec![0.0; self.grid.len()];
        for t in (0..t_horizon).rev() {
            let mut cur = vec![0.0; self.grid.len()];
            for i in 0..self.grid.len() {
                let x = self.grid[i];
                let x_next = self.choose_next(cuts, t, x, None);
                let j = self.idx(x_next);
                let revenue = self.scenarios[t]
                    .iter()
                    .map(|&d| self.params.price[t] * x_next.min(d))
                    .sum::<f64>()
                    / self.scenarios[t].len() as f64;
                cur[i] = -self.params.expansion_cost[t] * (x_next - x) + revenue + next[j];
            }
            next = cur;
        }
        next[self.idx(self.params.initial_capacity)]
    }
}

fn solve_exact_dp(params: &SDDPParams, grid: &[f64], scenarios: &[Vec<f64>]) -> ExactDp {
    let t_horizon = params.horizon;
    let mut next = vec![0.0; grid.len()];
    let mut policy: Vec<Vec<f64>> = vec![vec![0.0; grid.len()]; t_horizon];
    let idx = |x: f64| -> usize {
        let k = (x / params.step).round() as i64;
        let max = grid.len() as i64 - 1;
        k.max(0).min(max) as usize
    };
    for t in (0..t_horizon).rev() {
        let mut cur = vec![0.0; grid.len()];
        for i in 0..grid.len() {
            let x = grid[i];
            let mut best = f64::NEG_INFINITY;
            let mut best_x = x;
            for &x_next in grid {
                if x_next + 1e-9 < x {
                    continue;
                }
                let revenue = scenarios[t]
                    .iter()
                    .map(|&d| params.price[t] * x_next.min(d))
                    .sum::<f64>()
                    / scenarios[t].len() as f64;
                let q = -params.expansion_cost[t] * (x_next - x) + revenue + next[idx(x_next)];
                if q > best {
                    best = q;
                    best_x = x_next;
                }
            }
            cur[i] = best;
            policy[t][i] = best_x;
        }
        next = cur;
    }
    ExactDp {
        objective: next[idx(params.initial_capacity)],
        policy,
    }
}

impl FixedPoint for CapacityExpansionSDDPStation {
    type State = SDDPState;

    fn core(&self) -> &FixedPointCore<Self::State> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut FixedPointCore<Self::State> {
        &mut self.core
    }

    fn initial_state(&mut self) -> Self::State {
        let mut cuts: Vec<Vec<Cut>> = Vec::new();
        for t in 0..self.params.horizon {
            let upper = self.remaining_revenue_upper(t);
            cuts.push(vec![Cut {
                slope: 0.0,
                intercept: upper,
                stage: t,
                at: 0.0,
                value: upper,
            }]);
        }
        let upper_bound = self.vhat(&cuts, 0, self.params.initial_capacity);
        let lower_bound = self.evaluate_greedy_policy(&cuts);
        SDDPState {
            iter: 0,
            cuts,
            upper_bound,
            lower_bound,
            forward_states: vec![self.params.initial_capacity],
            forward_return: 0.0,
        }
    }

    fn apply_operator(&mut self, prev: &Self::State) -> Self::State {
        let mut cuts = clone_cuts(&prev.cuts);
        let (states, total) = self.forward_pass(&cuts, prev.iter);
        for t in (0..self.params.horizon).rev() {
            let x = states[t];
            let cut = self.make_cut(&cuts, t, x);
            cuts[t].push(cut.clone());
            if cuts[t].len() > 80 {
                let remove = cuts[t].len() - 80;
                cuts[t].drain(1..1 + remove);
            }
            if let Some(l) = &self.logger {
                l.log(&LogEvent {
                    kind: "sddp-cut".to_string(),
                    level: Some(LogLevel::Debug),
                    fields: vec![
                        ("iter".to_string(), LogValue::Int((prev.iter + 1) as i64)),
                        ("stage".to_string(), LogValue::Int(t as i64)),
                        ("at".to_string(), LogValue::Num(x)),
                        ("slope".to_string(), LogValue::Num(cut.slope)),
                        ("intercept".to_string(), LogValue::Num(cut.intercept)),
                    ],
                });
            }
        }
        let upper_bound = self.vhat(&cuts, 0, self.params.initial_capacity);
        let lower_bound = self.evaluate_greedy_policy(&cuts);
        let cut_counts: Vec<usize> = cuts.iter().map(|c| c.len()).collect();
        let trace_row = SDDPIteration {
            iter: prev.iter + 1,
            upper_bound,
            lower_bound,
            exact_objective: self.exact.objective,
            gap: upper_bound - lower_bound,
            cut_counts: cut_counts.clone(),
            forward_states: states.clone(),
            forward_return: total,
        };
        self.trace.push(trace_row.clone());
        self.upper_history.push(upper_bound);
        if let Some(l) = &self.logger {
            l.log(&LogEvent {
                kind: "sddp-iteration".to_string(),
                level: Some(LogLevel::Info),
                fields: vec![
                    ("iter".to_string(), LogValue::Int(trace_row.iter as i64)),
                    ("upperBound".to_string(), LogValue::Num(upper_bound)),
                    ("lowerBound".to_string(), LogValue::Num(lower_bound)),
                    (
                        "exactObjective".to_string(),
                        LogValue::Num(self.exact.objective),
                    ),
                    ("gap".to_string(), LogValue::Num(upper_bound - lower_bound)),
                    (
                        "cutCounts".to_string(),
                        LogValue::Ints(cut_counts.iter().map(|&c| c as i64).collect()),
                    ),
                    ("forwardStates".to_string(), LogValue::Nums(states.clone())),
                    ("forwardReturn".to_string(), LogValue::Num(total)),
                ],
            });
        }
        SDDPState {
            iter: prev.iter + 1,
            cuts,
            upper_bound,
            lower_bound,
            forward_states: states,
            forward_return: total,
        }
    }

    fn delta(&self, prev: &Self::State, next: &Self::State) -> f64 {
        (prev.upper_bound - next.upper_bound).abs()
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let (ub, lb) = {
            let c = self.get_current();
            (c.upper_bound, c.lower_bound)
        };
        let tol = self.core().tol;
        if iter > 0 && (ub - lb) <= tol {
            self.core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        self.default_should_stop(iter, last_delta)
    }
}

impl ResultStation for CapacityExpansionSDDPStation {
    type Output = SDDPResult;

    fn collect_validation(&self) -> Vec<ValidationCheck> {
        let cur = self.get_current();
        vec![
            intrinsic_check(
                "sddp-upper-bound-dominates-exact",
                cur.upper_bound + 1e-6 >= self.exact.objective,
                "upper bound >= exact sampled-grid objective",
                Some(format!(
                    "{:.6} vs exact {:.6}",
                    cur.upper_bound, self.exact.objective
                )),
                "sddp",
            ),
            intrinsic_check(
                "sddp-lower-bound-no-better-than-exact",
                cur.lower_bound <= self.exact.objective + 1e-6,
                "policy lower bound <= exact objective",
                Some(format!(
                    "{:.6} vs exact {:.6}",
                    cur.lower_bound, self.exact.objective
                )),
                "sddp",
            ),
            monotonicity_non_increasing_check(
                "sddp-upper-history-non-increasing",
                &self.upper_history,
                1e-8,
                "sddp",
            ),
        ]
    }

    fn result(&self, validation: Vec<ValidationCheck>) -> SDDPResult {
        let cur = self.get_current();
        SDDPResult {
            params: self.params.clone(),
            exact_objective: self.exact.objective,
            exact_policy: self.exact.policy.clone(),
            final_upper_bound: cur.upper_bound,
            final_lower_bound: cur.lower_bound,
            gap: cur.upper_bound - cur.lower_bound,
            cuts: cur.cuts.clone(),
            trace: self.trace.clone(),
            validation,
        }
    }
}

pub fn run_capacity_expansion_sddp(
    params: SDDPParams,
    logger: Option<Box<dyn OptimizationLogger>>,
) -> Result<SDDPResult, PreconditionError> {
    Ok(run_result_station(CapacityExpansionSDDPStation::new(
        params, logger,
    )?))
}

// =============================================================================
// Adaptive simulation optimisation.
// =============================================================================

#[derive(Clone, Debug)]
pub struct AdaptiveAlternative {
    pub name: String,
    pub x: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSimOptParams {
    pub cost: Vec<f64>,
    pub price: Vec<f64>,
    pub demand: DemandSpec,
    pub alternatives: Vec<AdaptiveAlternative>,
    pub seed: u32,
    pub initial_samples: usize,
    pub budget: usize,
    pub batch_size: usize,
    pub exploration: f64,
}

#[derive(Clone, Debug)]
pub struct AlternativeStats {
    pub name: String,
    pub x: Vec<f64>,
    pub n: f64,
    pub mean: f64,
    pub m2: f64,
    pub sd: f64,
    pub stderr: f64,
    pub ucb: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveTraceRow {
    pub iter: usize,
    pub sampled: String,
    pub total_samples: usize,
    pub best_name: String,
    pub best_mean: f64,
    pub max_stderr: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSimOptResult {
    pub params: AdaptiveSimOptParams,
    pub stats: Vec<AlternativeStats>,
    pub best: AlternativeStats,
    pub trace: Vec<AdaptiveTraceRow>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug)]
struct AdaptiveState {
    iter: usize,
    stats: Vec<AlternativeStats>,
    total_samples: usize,
    trace: Vec<AdaptiveTraceRow>,
}

fn sample_into(
    rng: &mut impl RandomSource,
    spec: &DemandSpec,
    cost: &[f64],
    price: &[f64],
    exploration: f64,
    st: &mut AlternativeStats,
) {
    let d = sample_demand_vector_unchecked(spec, cost.len(), rng);
    let z = capacity_profit(&st.x, &d, cost, price);
    st.n += 1.0;
    let delta = z - st.mean;
    st.mean += delta / st.n;
    st.m2 += delta * (z - st.mean);
    st.sd = if st.n > 1.0 {
        (st.m2 / (st.n - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    st.stderr = if st.n > 1.0 {
        st.sd / st.n.sqrt()
    } else {
        f64::INFINITY
    };
    st.ucb = st.mean
        + exploration
            * (if st.stderr.is_finite() {
                st.stderr
            } else {
                1e9
            });
}

pub struct AdaptiveSimulationOptimizerStation {
    params: AdaptiveSimOptParams,
    logger: Option<Box<dyn OptimizationLogger>>,
    rng: crate::des::shared::capabilities::SeededRandom,
    core: FixedPointCore<AdaptiveState>,
}

impl AdaptiveSimulationOptimizerStation {
    pub fn new(
        params: AdaptiveSimOptParams,
        logger: Option<Box<dyn OptimizationLogger>>,
    ) -> Result<Self, PreconditionError> {
        let max_iter = adaptive_max_iter(&params)?;
        Self::assert_preconditions(&params)?;
        let rng = mulberry32(params.seed);
        let mut st = AdaptiveSimulationOptimizerStation {
            params,
            logger,
            rng,
            core: FixedPointCore::new(0.0, max_iter),
        };
        st.bootstrap();
        Ok(st)
    }

    fn assert_preconditions(p: &AdaptiveSimOptParams) -> Check {
        Preconditions::non_empty(
            "AdaptiveSimulationOptimizerStation",
            "alternatives",
            &p.alternatives,
        )?;
        Preconditions::integer_in_range(
            "AdaptiveSimulationOptimizerStation",
            "alternatives.length",
            p.alternatives.len() as f64,
            2.0,
            MAX_SAFE_INTEGER,
        )?;
        Preconditions::non_empty("AdaptiveSimulationOptimizerStation", "cost", &p.cost)?;
        Preconditions::length_eq(
            "AdaptiveSimulationOptimizerStation",
            "price",
            &p.price,
            p.cost.len(),
        )?;
        Preconditions::arr_non_negative("AdaptiveSimulationOptimizerStation", "cost", &p.cost)?;
        Preconditions::arr_non_negative("AdaptiveSimulationOptimizerStation", "price", &p.price)?;
        validate_demand_spec(
            "AdaptiveSimulationOptimizerStation",
            "demand",
            &p.demand,
            p.cost.len(),
        )?;
        Preconditions::integer_in_range(
            "AdaptiveSimulationOptimizerStation",
            "initialSamples",
            p.initial_samples as f64,
            1.0,
            MAX_SAFE_INTEGER,
        )?;
        Preconditions::integer_in_range(
            "AdaptiveSimulationOptimizerStation",
            "budget",
            p.budget as f64,
            (p.alternatives.len() * p.initial_samples) as f64,
            MAX_SAFE_INTEGER,
        )?;
        Preconditions::integer_in_range(
            "AdaptiveSimulationOptimizerStation",
            "batchSize",
            p.batch_size as f64,
            1.0,
            MAX_SAFE_INTEGER,
        )?;
        Preconditions::non_negative(
            "AdaptiveSimulationOptimizerStation",
            "exploration",
            p.exploration,
        )?;
        let mut names: Vec<&str> = Vec::new();
        for (i, a) in p.alternatives.iter().enumerate() {
            Preconditions::check(
                "AdaptiveSimulationOptimizerStation",
                &format!("alternatives[{i}].name"),
                "be a non-empty string",
                !a.name.trim().is_empty(),
                Some(a.name.clone()),
            )?;
            Preconditions::check(
                "AdaptiveSimulationOptimizerStation",
                &format!("alternatives[{i}].name"),
                "be unique",
                !names.contains(&a.name.as_str()),
                Some(a.name.clone()),
            )?;
            names.push(a.name.as_str());
            Preconditions::length_eq(
                "AdaptiveSimulationOptimizerStation",
                &format!("alternative.{}.x", a.name),
                &a.x,
                p.cost.len(),
            )?;
            Preconditions::arr_non_negative(
                "AdaptiveSimulationOptimizerStation",
                &format!("alternative.{}.x", a.name),
                &a.x,
            )?;
        }
        Ok(())
    }
}

impl FixedPoint for AdaptiveSimulationOptimizerStation {
    type State = AdaptiveState;

    fn core(&self) -> &FixedPointCore<Self::State> {
        &self.core
    }
    fn core_mut(&mut self) -> &mut FixedPointCore<Self::State> {
        &mut self.core
    }

    fn initial_state(&mut self) -> Self::State {
        let mut stats: Vec<AlternativeStats> = self
            .params
            .alternatives
            .iter()
            .map(|a| AlternativeStats {
                name: a.name.clone(),
                x: a.x.clone(),
                n: 0.0,
                mean: 0.0,
                m2: 0.0,
                sd: 0.0,
                stderr: f64::INFINITY,
                ucb: f64::INFINITY,
            })
            .collect();
        let mut total_samples = 0usize;
        for st in &mut stats {
            for _ in 0..self.params.initial_samples {
                sample_into(
                    &mut self.rng,
                    &self.params.demand,
                    &self.params.cost,
                    &self.params.price,
                    self.params.exploration,
                    st,
                );
                total_samples += 1;
            }
        }
        AdaptiveState {
            iter: 0,
            stats,
            total_samples,
            trace: Vec::new(),
        }
    }

    fn apply_operator(&mut self, prev: &Self::State) -> Self::State {
        let mut stats: Vec<AlternativeStats> = prev.stats.clone();
        let mut chosen_idx = 0usize;
        for i in 1..stats.len() {
            if stats[i].ucb > stats[chosen_idx].ucb {
                chosen_idx = i;
            }
        }
        let mut total_samples = prev.total_samples;
        let reps = self
            .params
            .batch_size
            .min(self.params.budget.saturating_sub(total_samples));
        for _ in 0..reps {
            sample_into(
                &mut self.rng,
                &self.params.demand,
                &self.params.cost,
                &self.params.price,
                self.params.exploration,
                &mut stats[chosen_idx],
            );
            total_samples += 1;
        }
        let mut best_idx = 0usize;
        for i in 1..stats.len() {
            if stats[i].mean > stats[best_idx].mean {
                best_idx = i;
            }
        }
        let max_stderr = stats
            .iter()
            .map(|s| if s.stderr.is_finite() { s.stderr } else { 0.0 })
            .fold(f64::NEG_INFINITY, f64::max);
        let row = AdaptiveTraceRow {
            iter: prev.iter + 1,
            sampled: stats[chosen_idx].name.clone(),
            total_samples,
            best_name: stats[best_idx].name.clone(),
            best_mean: stats[best_idx].mean,
            max_stderr,
        };
        if let Some(l) = &self.logger {
            l.log(&LogEvent {
                kind: "adaptive-simopt-iteration".to_string(),
                level: Some(LogLevel::Info),
                fields: vec![
                    ("iter".to_string(), LogValue::Int(row.iter as i64)),
                    ("sampled".to_string(), LogValue::Str(row.sampled.clone())),
                    (
                        "totalSamples".to_string(),
                        LogValue::Int(row.total_samples as i64),
                    ),
                    ("bestName".to_string(), LogValue::Str(row.best_name.clone())),
                    ("bestMean".to_string(), LogValue::Num(row.best_mean)),
                    ("maxStderr".to_string(), LogValue::Num(row.max_stderr)),
                ],
            });
        }
        let mut trace = prev.trace.clone();
        trace.push(row);
        AdaptiveState {
            iter: prev.iter + 1,
            stats,
            total_samples,
            trace,
        }
    }

    fn delta(&self, prev: &Self::State, next: &Self::State) -> f64 {
        (prev.total_samples as f64 - next.total_samples as f64).abs()
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let total = self.get_current().total_samples;
        if total >= self.params.budget {
            self.core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        self.default_should_stop(iter, last_delta)
    }
}

impl ResultStation for AdaptiveSimulationOptimizerStation {
    type Output = AdaptiveSimOptResult;

    fn collect_validation(&self) -> Vec<ValidationCheck> {
        let cur = self.get_current();
        vec![
            intrinsic_check(
                "adaptive-budget-respected",
                cur.total_samples >= self.params.budget,
                "totalSamples >= budget",
                Some(cur.total_samples.to_string()),
                "adaptive-simopt",
            ),
            intrinsic_check(
                "adaptive-all-alternatives-sampled",
                cur.stats
                    .iter()
                    .all(|a| a.n >= self.params.initial_samples as f64),
                "each alternative has initialSamples",
                None,
                "adaptive-simopt",
            ),
        ]
    }

    fn result(&self, validation: Vec<ValidationCheck>) -> AdaptiveSimOptResult {
        let stats = self.get_current().stats.clone();
        let mut best_idx = 0usize;
        for i in 1..stats.len() {
            if stats[i].mean > stats[best_idx].mean {
                best_idx = i;
            }
        }
        let best = stats[best_idx].clone();
        AdaptiveSimOptResult {
            params: self.params.clone(),
            stats,
            best,
            trace: self.get_current().trace.clone(),
            validation,
        }
    }
}

pub fn run_adaptive_sim_opt(
    params: AdaptiveSimOptParams,
    logger: Option<Box<dyn OptimizationLogger>>,
) -> Result<AdaptiveSimOptResult, PreconditionError> {
    Ok(run_result_station(AdaptiveSimulationOptimizerStation::new(
        params, logger,
    )?))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_fit_recovers_normal_and_is_deterministic() {
        // Synthetic Gaussian-ish sample (fixed, no RNG): mean ~ 5, spread ~ 2.
        let samples: Vec<f64> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 4.5, 5.5, 5.0, 4.0, 6.0];
        let params = DistributionFitParams {
            samples: samples.clone(),
            families: Some(vec![
                DistributionFamily::Normal,
                DistributionFamily::Exponential,
            ]),
            methods: Some(vec![FitMethod::Mle]),
        };
        let r1 = run_distribution_fit(params.clone()).unwrap();
        let r2 = run_distribution_fit(params).unwrap();

        // Determinism: identical sample mean / variance and number of fits.
        assert_eq!(r1.fits.len(), r2.fits.len());
        assert!((r1.sample_mean - r2.sample_mean).abs() < 1e-12);

        // Sample mean recovered.
        assert!((r1.sample_mean - mean(&samples)).abs() < 1e-12);
        // Normal should win on AIC for this symmetric sample.
        assert_eq!(r1.best_by_aic.family, DistributionFamily::Normal);
        // The fitted normal mean matches the sample mean.
        assert!((param(&r1.best_by_aic.params, "mu") - mean(&samples)).abs() < 1e-9);
        // Validators pass.
        assert!(r1.validation.iter().all(|c| c.passed));
    }

    #[test]
    fn sddp_bounds_bracket_the_exact_objective() {
        let params = SDDPParams {
            horizon: 3,
            demand: vec![
                DemandRange {
                    low: 2.0,
                    high: 6.0,
                },
                DemandRange {
                    low: 3.0,
                    high: 7.0,
                },
                DemandRange {
                    low: 1.0,
                    high: 5.0,
                },
            ],
            price: vec![2.0, 2.0, 2.0],
            expansion_cost: vec![1.0, 1.0, 1.0],
            initial_capacity: 0.0,
            x_max: 8.0,
            step: 2.0,
            samples_per_stage: 8,
            seed: 12345,
            max_iter: Some(40),
            tol: Some(1e-4),
        };
        let res = run_capacity_expansion_sddp(params, None).unwrap();

        // The SDDP upper bound dominates the exact sampled-grid objective and the
        // greedy policy lower bound never exceeds it.
        assert!(res.final_upper_bound + 1e-6 >= res.exact_objective);
        assert!(res.final_lower_bound <= res.exact_objective + 1e-6);
        // Gap is non-negative.
        assert!(res.gap >= -1e-6);
        // All registered invariants hold.
        assert!(
            res.validation.iter().all(|c| c.passed),
            "validators: {:?}",
            res.validation
        );
    }

    #[test]
    fn adaptive_simopt_respects_budget_and_is_deterministic() {
        let make_params = || AdaptiveSimOptParams {
            cost: vec![1.0],
            price: vec![3.0],
            demand: DemandSpec::Uniform(vec![DemandRange {
                low: 0.0,
                high: 10.0,
            }]),
            alternatives: vec![
                AdaptiveAlternative {
                    name: "low".to_string(),
                    x: vec![2.0],
                },
                AdaptiveAlternative {
                    name: "mid".to_string(),
                    x: vec![5.0],
                },
                AdaptiveAlternative {
                    name: "high".to_string(),
                    x: vec![9.0],
                },
            ],
            seed: 777,
            initial_samples: 10,
            budget: 400,
            batch_size: 20,
            exploration: 1.0,
        };
        let r1 = run_adaptive_sim_opt(make_params(), None).unwrap();
        let r2 = run_adaptive_sim_opt(make_params(), None).unwrap();

        // Determinism under a fixed seed.
        assert_eq!(r1.best.name, r2.best.name);
        assert!((r1.best.mean - r2.best.mean).abs() < 1e-12);

        // Budget respected and every alternative got at least initialSamples.
        let total: f64 = r1.stats.iter().map(|s| s.n).sum();
        assert!(total >= r1.params.budget as f64);
        assert!(r1
            .stats
            .iter()
            .all(|s| s.n >= r1.params.initial_samples as f64));
        assert!(r1.validation.iter().all(|c| c.passed));
    }
}
