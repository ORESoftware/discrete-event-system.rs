//! Port of `src/des/general/adapters/statistical-optimization-adapter.ts`
//! (module `des::general::adapters::statistical_optimization_adapter`).
//!
//! Registers the stochastic-LP / distribution-fit / risk-capacity /
//! sddp-capacity / adaptive-simopt JSON adapters (5 models).
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * The dual param spelling (`cost`/`price` vs `c`/`p`, `numScenarios` vs
//!     `N`) is reconciled in [`normalize_stochastic_lp_params`] via `??`-style
//!     fallbacks. The TS `throw`s in normalize/assert are uncaught validation
//!     errors that propagate out of `run`, so they map to `panic!`.
//!   * `fitted.params` (an open `string -> number` map) -> `HashMap<String, f64>`.
//!   * The `evalX` closure capturing `oos`/`actual` -> a local closure.
//!   * `JSON.stringify` in CSV cells -> the local JSON helpers
//!     ([`json_num_array`], [`json_usize_array`], [`json_str_num_map`]).
//!   * `withLogger(runtime, fn)` -> [`with_logger`]; structured events go through
//!     the ported [`JsonlLogger`] ([`LogJson`]).
//!
//! PORT NOTE: `registerModel` / the registry is not ported yet; each adapter is
//! exposed via the `adapter_*()` constructors for explicit registration later.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, `addBar`/`lineChartSeries` and each `animate` body) is not
//! ported, so `animate` is a no-op here.
//!
//! PORT NOTE: `runCapacityExpansionSDDP` / `runAdaptiveSimOpt` take a logger in
//! TS (the `withLogger` `JsonlLogger`). The Rust engine accepts an owned
//! `Option<Box<dyn OptimizationLogger>>` (`'static`), which cannot borrow the
//! `with_logger` `JsonlLogger`, and `JsonlLogger` does not implement
//! `OptimizationLogger`. Those two `run`s therefore pass `None`; the structured
//! per-iteration log is omitted (it is a side effect only; results are
//! unaffected). The `risk-capacity` and `stochastic-lp` models log around the
//! call, which is preserved.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::des::general::adapters::adapter_utils::{
    csv_row, validation_line, with_logger, write_csv_lines,
};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, OneOfVariant, ParamSchema,
    RegistrationExample, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::statistical_optimization::{
    run_adaptive_sim_opt, run_capacity_expansion_sddp, run_distribution_fit, run_risk_capacity,
    AdaptiveAlternative, AdaptiveSimOptParams, AdaptiveSimOptResult, DemandRange, DemandSpec,
    DistributionFamily, DistributionFitParams, DistributionFitResult, FitMethod,
    RiskCapacityParams, RiskCapacityResult, RiskConfig, RiskKind, SDDPParams, SDDPResult,
};
use crate::des::general::stochastic_lp::{
    build_production_scenarios, build_production_slp, solve_production_closed_form,
    solve_slp_benders, solve_slp_monolithic, BendersOpts, SLPSolveResult, SLPStatus,
    UniformDemandSpec,
};
use crate::des::observability::logger::JsonValue as LogJson;

// =============================================================================
// Formatting helpers (JS parity)
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

/// `Number.prototype.toExponential(digits)`.
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

/// `numbers.map(String).join(', ')` — the JS `Array.prototype.join`.
fn join_nums(values: &[f64]) -> String {
    values.iter().map(|v| js_number(*v)).collect::<Vec<_>>().join(", ")
}

/// `lengths.join(', ')` for an integer list.
fn join_usize(values: &[usize]) -> String {
    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
}

/// `JSON.stringify(numbers)` — a JSON array (`NaN`/`Infinity` -> `null`).
fn json_num_array(values: &[f64]) -> String {
    let inner: Vec<String> = values
        .iter()
        .map(|v| if v.is_finite() { js_number(*v) } else { "null".to_string() })
        .collect();
    format!("[{}]", inner.join(","))
}

/// `JSON.stringify(integers)`.
fn json_usize_array(values: &[usize]) -> String {
    let inner: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", inner.join(","))
}

fn json_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `JSON.stringify(params)` for a `string -> number` map.
///
/// PORT NOTE: the TS object preserves insertion order; the Rust source map is a
/// `HashMap`, so keys are sorted for a deterministic (but possibly reordered)
/// rendering.
fn json_str_num_map(map: &HashMap<String, f64>) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let inner: Vec<String> = keys
        .iter()
        .map(|k| {
            let v = map[*k];
            let val = if v.is_finite() { js_number(v) } else { "null".to_string() };
            format!("{}:{}", json_quote(k), val)
        })
        .collect();
    format!("{{{}}}", inner.join(","))
}

// =============================================================================
// Enum -> TS string conversions
// =============================================================================

fn family_str(f: DistributionFamily) -> &'static str {
    match f {
        DistributionFamily::Normal => "normal",
        DistributionFamily::Lognormal => "lognormal",
        DistributionFamily::Exponential => "exponential",
        DistributionFamily::Gamma => "gamma",
        DistributionFamily::Poisson => "poisson",
        DistributionFamily::Empirical => "empirical",
    }
}

fn method_str(m: FitMethod) -> &'static str {
    match m {
        FitMethod::Mle => "mle",
        FitMethod::Moments => "moments",
    }
}

fn risk_kind_str(k: RiskKind) -> &'static str {
    match k {
        RiskKind::Expectation => "expectation",
        RiskKind::Cvar => "cvar",
        RiskKind::Chance => "chance",
        RiskKind::Dro => "dro",
    }
}

fn slp_status_str(s: SLPStatus) -> &'static str {
    match s {
        SLPStatus::Optimal => "optimal",
        SLPStatus::Unbounded => "unbounded",
        SLPStatus::Infeasible => "infeasible",
        SLPStatus::IterLimit => "iter-limit",
    }
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

/// A string-enum field with no default (the TS `{kind:'string', enum:[...]}`).
fn str_enum_nd(allowed: &[&str]) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: None,
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length: None, description: None }
}

fn arr_mm(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

const FAMILY_VALUES: [&str; 6] =
    ["normal", "lognormal", "exponential", "gamma", "poisson", "empirical"];
const METHOD_VALUES: [&str; 2] = ["mle", "moments"];

fn range_schema() -> ParamSchema {
    obj(
        vec![("low", num(Some(0.0), None, None, None)), ("high", num(Some(0.0), None, None, None))],
        vec!["low", "high"],
    )
}

fn fitted_distribution_schema() -> ParamSchema {
    obj(
        vec![
            ("family", str_enum_nd(&FAMILY_VALUES)),
            ("method", str_enum(&METHOD_VALUES, "mle")),
            ("params", obj(vec![], vec![])),
            ("logLikelihood", num(None, None, None, Some(0.0))),
            ("aic", num(None, None, None, Some(0.0))),
            ("mean", num(None, None, None, Some(0.0))),
            ("variance", num(None, None, None, Some(0.0))),
            ("support", str_enum(&["real", "positive", "nonnegative-integer", "empirical"], "positive")),
        ],
        vec!["family", "params"],
    )
}

fn empirical_point_schema() -> ParamSchema {
    obj(
        vec![("value", num(None, None, None, None)), ("prob", num(Some(0.0), Some(1.0), None, None))],
        vec!["value", "prob"],
    )
}

fn demand_schema() -> ParamSchema {
    ParamSchema::OneOf {
        variants: vec![
            OneOfVariant {
                tag: "uniform".to_string(),
                tag_field: None,
                description: None,
                schema: obj(
                    vec![
                        ("kind", str_enum(&["uniform"], "uniform")),
                        ("ranges", arr(range_schema(), Some(1))),
                    ],
                    vec!["kind", "ranges"],
                ),
            },
            OneOfVariant {
                tag: "fitted".to_string(),
                tag_field: None,
                description: None,
                schema: obj(
                    vec![
                        ("kind", str_enum_nd(&["fitted"])),
                        ("fitted", arr(fitted_distribution_schema(), Some(1))),
                    ],
                    vec!["kind", "fitted"],
                ),
            },
            OneOfVariant {
                tag: "empirical".to_string(),
                tag_field: None,
                description: None,
                schema: obj(
                    vec![
                        ("kind", str_enum_nd(&["empirical"])),
                        ("empirical", arr(arr(empirical_point_schema(), Some(1)), Some(1))),
                    ],
                    vec!["kind", "empirical"],
                ),
            },
        ],
        description: None,
    }
}

fn fit_params_schema() -> ParamSchema {
    obj(
        vec![
            ("samples", arr(num(None, None, None, None), Some(2))),
            ("families", arr(str_enum_nd(&FAMILY_VALUES), None)),
            ("methods", arr(str_enum_nd(&METHOD_VALUES), None)),
        ],
        vec!["samples"],
    )
}

fn risk_params_schema() -> ParamSchema {
    obj(
        vec![
            ("cost", arr(num(Some(0.0), None, None, None), Some(1))),
            ("price", arr(num(Some(0.0), None, None, None), Some(1))),
            ("demand", demand_schema()),
            ("numScenarios", num(Some(1.0), None, Some(true), Some(200.0))),
            ("seed", num(None, None, Some(true), Some(42.0))),
            ("xMax", num(Some(0.0), None, None, None)),
            ("step", num(Some(0.0), None, None, None)),
            (
                "risk",
                obj(
                    vec![
                        ("kind", str_enum(&["expectation", "cvar", "chance", "dro"], "expectation")),
                        ("alpha", num(Some(0.5), Some(0.999), None, Some(0.9))),
                        ("lambda", num(Some(0.0), None, None, Some(1.0))),
                        ("minServiceLevel", num(Some(0.0), Some(1.0), None, Some(0.9))),
                        ("shortfallLimit", num(Some(0.0), None, None, Some(0.0))),
                        ("radius", num(Some(0.0), None, None, Some(1.0))),
                    ],
                    vec!["kind"],
                ),
            ),
        ],
        vec!["cost", "price", "demand", "xMax", "step", "risk"],
    )
}

fn sddp_params_schema() -> ParamSchema {
    obj(
        vec![
            ("horizon", num(Some(1.0), None, Some(true), None)),
            ("demand", arr(range_schema(), Some(1))),
            ("price", arr(num(Some(0.0), None, None, None), Some(1))),
            ("expansionCost", arr(num(Some(0.0), None, None, None), Some(1))),
            ("initialCapacity", num(Some(0.0), None, None, Some(0.0))),
            ("xMax", num(Some(0.0), None, None, None)),
            ("step", num(Some(0.0), None, None, None)),
            ("samplesPerStage", num(Some(1.0), None, Some(true), Some(40.0))),
            ("seed", num(None, None, Some(true), Some(7.0))),
            ("maxIter", num(Some(1.0), None, Some(true), Some(40.0))),
            ("tol", num(Some(0.0), None, None, Some(1e-3))),
        ],
        vec!["horizon", "demand", "price", "expansionCost", "xMax", "step"],
    )
}

fn alt_schema() -> ParamSchema {
    obj(
        vec![("name", string_field()), ("x", arr(num(Some(0.0), None, None, None), Some(1)))],
        vec!["name", "x"],
    )
}

fn adaptive_params_schema() -> ParamSchema {
    obj(
        vec![
            ("cost", arr(num(Some(0.0), None, None, None), Some(1))),
            ("price", arr(num(Some(0.0), None, None, None), Some(1))),
            ("demand", demand_schema()),
            ("alternatives", arr(alt_schema(), Some(2))),
            ("seed", num(None, None, Some(true), Some(11.0))),
            ("initialSamples", num(Some(1.0), None, Some(true), Some(5.0))),
            ("budget", num(Some(1.0), None, Some(true), Some(120.0))),
            ("batchSize", num(Some(1.0), None, Some(true), Some(5.0))),
            ("exploration", num(Some(0.0), None, None, Some(1.5))),
        ],
        vec!["cost", "price", "demand", "alternatives"],
    )
}

fn stochastic_lp_schema() -> ParamSchema {
    obj(
        vec![
            ("cost", arr(num(Some(0.0), None, None, None), Some(1))),
            ("price", arr(num(Some(0.0), None, None, None), Some(1))),
            ("c", arr(num(Some(0.0), None, None, None), Some(1))),
            ("p", arr(num(Some(0.0), None, None, None), Some(1))),
            (
                "ranges",
                arr(arr_mm(num(Some(0.0), None, None, None), Some(2), Some(2)), Some(1)),
            ),
            ("numScenarios", num(Some(1.0), None, Some(true), Some(200.0))),
            ("N", num(Some(1.0), None, Some(true), Some(200.0))),
            ("seed", num(None, None, Some(true), Some(42.0))),
            ("budget", num(Some(0.0), None, None, None)),
            ("maxIter", num(Some(1.0), None, Some(true), Some(200.0))),
            ("tol", num(Some(0.0), None, None, Some(1e-7))),
            ("oosN", num(Some(0.0), None, Some(true), Some(0.0))),
        ],
        vec!["ranges"],
    )
}

// =============================================================================
// stochastic-lp param / result types (local — the TS `interface`s)
// =============================================================================

/// `interface StochasticLPParams`.
#[derive(Clone, Debug, Default)]
pub struct StochasticLPParams {
    pub cost: Option<Vec<f64>>,
    pub price: Option<Vec<f64>>,
    pub c: Option<Vec<f64>>,
    pub p: Option<Vec<f64>>,
    pub ranges: Vec<(f64, f64)>,
    pub num_scenarios: Option<usize>,
    pub n: Option<usize>,
    pub seed: Option<u32>,
    pub budget: Option<f64>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub oos_n: Option<usize>,
}

/// `interface NormalizedStochasticLPParams`.
#[derive(Clone, Debug)]
struct NormalizedStochasticLPParams {
    cost: Vec<f64>,
    price: Vec<f64>,
    ranges: Vec<(f64, f64)>,
    num_scenarios: usize,
    seed: u32,
    budget: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    oos_n: Option<usize>,
}

/// `StochasticLPAdapterResult['outOfSample']`.
#[derive(Clone, Debug)]
pub struct StochasticLPOutOfSample {
    pub n: usize,
    pub monolithic: f64,
    pub benders: f64,
    pub closed_form: Option<f64>,
}

/// `interface StochasticLPAdapterResult`.
#[derive(Clone, Debug)]
pub struct StochasticLPAdapterResult {
    pub closed_form: Option<SLPSolveResult>,
    pub monolithic: SLPSolveResult,
    pub benders: SLPSolveResult,
    pub out_of_sample: Option<StochasticLPOutOfSample>,
}

/// `normalizeStochasticLPParams` — the `throw` becomes `panic!` (an uncaught
/// validation error that propagates out of `run`).
fn normalize_stochastic_lp_params(params: StochasticLPParams) -> NormalizedStochasticLPParams {
    let cost = match &params.cost {
        Some(c) if !c.is_empty() => Some(c.clone()),
        _ => params.c.clone(),
    };
    let price = match &params.price {
        Some(p) if !p.is_empty() => Some(p.clone()),
        _ => params.p.clone(),
    };
    let (cost, price) = match (cost, price) {
        (Some(c), Some(p)) => (c, p),
        _ => panic!("stochastic-lp: provide cost/price or c/p arrays"),
    };
    NormalizedStochasticLPParams {
        cost,
        price,
        ranges: params.ranges,
        num_scenarios: params.num_scenarios.or(params.n).unwrap_or(200),
        seed: params.seed.unwrap_or(42),
        budget: params.budget,
        max_iter: params.max_iter,
        tol: params.tol,
        oos_n: params.oos_n,
    }
}

/// `assertStochasticLPParams` — invariant violations -> `panic!`.
fn assert_stochastic_lp_params(params: &NormalizedStochasticLPParams) {
    if params.cost.len() != params.price.len() || params.cost.len() != params.ranges.len() {
        panic!("stochastic-lp: cost, price, and ranges must have the same length");
    }
    for (i, &(lo, hi)) in params.ranges.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < lo {
            panic!("stochastic-lp: ranges[{i}] must satisfy 0 <= low <= high");
        }
    }
}

// =============================================================================
// Examples
// =============================================================================

fn example<P>(name: &str, model: &str, parameters: P) -> RegistrationExample<P> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: model.to_string(),
            description: None,
            parameters,
            runtime: Some(DESRuntimeConfig { animate: Some(true), ..Default::default() }),
            metadata: None,
        },
    }
}

// =============================================================================
// stochastic-lp
// =============================================================================

pub struct StochasticLPAdapter;
pub fn adapter_stochastic_lp() -> StochasticLPAdapter {
    StochasticLPAdapter
}

impl DESModelRegistration<StochasticLPParams, StochasticLPAdapterResult> for StochasticLPAdapter {
    fn id(&self) -> &str {
        "stochastic-lp"
    }
    fn description(&self) -> &str {
        "Two-stage stochastic LP via SAA monolithic solve and Benders/L-shaped DES cuts."
    }
    fn schema(&self) -> ParamSchema {
        stochastic_lp_schema()
    }
    fn run(&self, params: StochasticLPParams, runtime: &DESRuntimeConfig) -> StochasticLPAdapterResult {
        let actual = normalize_stochastic_lp_params(params);
        with_logger(runtime, move |mut logger| {
            assert_stochastic_lp_params(&actual);
            let slp = build_production_slp(actual.cost.clone(), actual.price.clone(), actual.budget);
            let scenarios = build_production_scenarios(
                UniformDemandSpec { ranges: actual.ranges.clone(), seed: actual.seed },
                actual.num_scenarios,
            );
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("stochastic-lp-start".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("numScenarios".to_string(), LogJson::Number(actual.num_scenarios as f64)),
                    (
                        "budget".to_string(),
                        match actual.budget {
                            Some(b) => LogJson::Number(b),
                            None => LogJson::Null,
                        },
                    ),
                ]));
            }
            let closed_form = if actual.budget.is_none() {
                Some(solve_production_closed_form(
                    actual.cost.clone(),
                    actual.price.clone(),
                    actual.ranges.clone(),
                ))
            } else {
                None
            };
            let monolithic = solve_slp_monolithic(slp.clone(), scenarios.clone());
            let benders = solve_slp_benders(
                slp,
                scenarios,
                BendersOpts {
                    max_iter: Some(actual.max_iter.unwrap_or(200)),
                    tol: Some(actual.tol.unwrap_or(1e-7)),
                    verbose: None,
                    reference_path: None,
                    reference_tol: None,
                    silent_if_missing: None,
                },
            );
            let out_of_sample = if actual.oos_n.unwrap_or(0) > 0 {
                let oos = build_production_scenarios(
                    UniformDemandSpec {
                        ranges: actual.ranges.clone(),
                        seed: actual.seed.wrapping_add(99_991),
                    },
                    actual.oos_n.unwrap(),
                );
                let eval_x = |x: &[f64]| -> f64 {
                    let mut z = 0.0;
                    for i in 0..actual.cost.len() {
                        z += -actual.cost[i] * x[i];
                    }
                    let mut q = 0.0;
                    for sc in &oos {
                        let d = &sc.meta.as_ref().expect("scenario meta").d;
                        for i in 0..actual.price.len() {
                            q += actual.price[i] * x[i].min(d[i]);
                        }
                    }
                    z + q / (oos.len() as f64)
                };
                Some(StochasticLPOutOfSample {
                    n: actual.oos_n.unwrap(),
                    monolithic: eval_x(&monolithic.x),
                    benders: eval_x(&benders.x),
                    closed_form: closed_form.as_ref().map(|cf| eval_x(&cf.x)),
                })
            } else {
                None
            };
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("stochastic-lp-finish".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("monoObjective".to_string(), LogJson::Number(monolithic.objective)),
                    ("bendersObjective".to_string(), LogJson::Number(benders.objective)),
                    ("iterations".to_string(), LogJson::Number(benders.iterations as f64)),
                ]));
            }
            StochasticLPAdapterResult { closed_form, monolithic, benders, out_of_sample }
        })
    }
    fn summarize(&self, r: &StochasticLPAdapterResult, _params: &StochasticLPParams) -> String {
        let cuts = r
            .benders
            .benders_trace
            .as_ref()
            .map(|t| t.iter().filter(|it| it.cut_added.is_some()).count())
            .unwrap_or(0);
        let mut lines = vec![
            "STOCHASTIC LP".to_string(),
            "------------------------".to_string(),
            format!(
                "  Monolithic: status={} z={:.4} iters={}",
                slp_status_str(r.monolithic.status),
                r.monolithic.objective,
                r.monolithic.iterations
            ),
            format!(
                "  Benders:    status={} z={:.4} cuts={}",
                slp_status_str(r.benders.status),
                r.benders.objective,
                cuts
            ),
            format!(
                "  |Delta z|:  {}",
                to_exponential((r.monolithic.objective - r.benders.objective).abs(), 3)
            ),
        ];
        if let Some(cf) = &r.closed_form {
            lines.push(format!("  Closed form z*: {:.4}", cf.objective));
        }
        if let Some(oos) = &r.out_of_sample {
            lines.push(format!(
                "  OOS N={}: monolithic={:.4} benders={:.4}",
                oos.n, oos.monolithic, oos.benders
            ));
        }
        lines.join("\n")
    }
    fn write_csv(&self, r: &StochasticLPAdapterResult, csv_path: &str) {
        let mut lines = vec!["iter,upper_bound,lower_bound,gap,theta,expected_q".to_string()];
        if let Some(trace) = &r.benders.benders_trace {
            for it in trace {
                lines.push(csv_row([
                    format!("{:.8}", it.iter as f64),
                    format!("{:.8}", it.upper_bound),
                    format!("{:.8}", it.lower_bound),
                    format!("{:.8}", it.gap),
                    format!("{:.8}", it.theta_master),
                    format!("{:.8}", it.expected_q),
                ]));
            }
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _r: &StochasticLPAdapterResult,
        _params: &StochasticLPParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<StochasticLPParams>> {
        vec![example(
            "2-product capacity planning",
            "stochastic-lp",
            StochasticLPParams {
                cost: Some(vec![10.0, 12.0]),
                price: Some(vec![25.0, 28.0]),
                ranges: vec![(50.0, 100.0), (40.0, 80.0)],
                num_scenarios: Some(200),
                seed: Some(42),
                ..Default::default()
            },
        )]
    }
}

// =============================================================================
// distribution-fit
// =============================================================================

pub struct DistributionFitAdapter;
pub fn adapter_distribution_fit() -> DistributionFitAdapter {
    DistributionFitAdapter
}

impl DESModelRegistration<DistributionFitParams, DistributionFitResult> for DistributionFitAdapter {
    fn id(&self) -> &str {
        "distribution-fit"
    }
    fn description(&self) -> &str {
        "Fit demand/service samples by MLE and method of moments, then rank by AIC."
    }
    fn schema(&self) -> ParamSchema {
        fit_params_schema()
    }
    fn run(&self, params: DistributionFitParams, _runtime: &DESRuntimeConfig) -> DistributionFitResult {
        run_distribution_fit(params).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &DistributionFitResult, _params: &DistributionFitParams) -> String {
        [
            "DISTRIBUTION FIT".to_string(),
            "------------------------".to_string(),
            format!(
                "  n={} mean={:.4} var={:.4}",
                r.samples.len(),
                r.sample_mean,
                r.sample_variance
            ),
            format!(
                "  best={}/{} AIC={:.3}",
                family_str(r.best_by_aic.family),
                method_str(r.best_by_aic.method),
                r.best_by_aic.aic
            ),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &DistributionFitResult, csv_path: &str) {
        let mut lines = vec!["rank,family,method,aic,log_likelihood,mean,variance,params".to_string()];
        for (i, f) in r.fits.iter().enumerate() {
            lines.push(csv_row([
                js_number((i + 1) as f64),
                family_str(f.family).to_string(),
                method_str(f.method).to_string(),
                format!("{:.8}", f.aic),
                format!("{:.8}", f.log_likelihood),
                format!("{:.8}", f.mean),
                format!("{:.8}", f.variance),
                json_str_num_map(&f.params),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _r: &DistributionFitResult,
        _params: &DistributionFitParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<DistributionFitParams>> {
        vec![example(
            "positive service times",
            "distribution-fit",
            DistributionFitParams {
                samples: vec![8.2, 9.1, 10.4, 7.6, 12.3, 9.9, 11.1, 8.7, 10.8, 9.4],
                families: Some(vec![
                    DistributionFamily::Normal,
                    DistributionFamily::Lognormal,
                    DistributionFamily::Gamma,
                    DistributionFamily::Exponential,
                ]),
                methods: Some(vec![FitMethod::Mle, FitMethod::Moments]),
            },
        )]
    }
}

// =============================================================================
// risk-capacity
// =============================================================================

pub struct RiskCapacityAdapter;
pub fn adapter_risk_capacity() -> RiskCapacityAdapter {
    RiskCapacityAdapter
}

impl DESModelRegistration<RiskCapacityParams, RiskCapacityResult> for RiskCapacityAdapter {
    fn id(&self) -> &str {
        "risk-capacity"
    }
    fn description(&self) -> &str {
        "Scenario capacity planning with expectation, CVaR, chance, or DRO-lite objectives."
    }
    fn schema(&self) -> ParamSchema {
        risk_params_schema()
    }
    fn run(&self, params: RiskCapacityParams, runtime: &DESRuntimeConfig) -> RiskCapacityResult {
        let risk_kind = params.risk.kind;
        let scenarios = params.num_scenarios;
        with_logger(runtime, move |mut logger| {
            if let Some(l) = logger.as_deref_mut() {
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("risk-capacity-start".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    ("risk".to_string(), LogJson::String(risk_kind_str(risk_kind).to_string())),
                    ("scenarios".to_string(), LogJson::Number(scenarios as f64)),
                ]));
            }
            let result = run_risk_capacity(params).unwrap_or_else(|e| panic!("{e}"));
            if let Some(l) = logger.as_deref_mut() {
                let best = &result.best;
                l.log(LogJson::Object(vec![
                    ("kind".to_string(), LogJson::String("risk-capacity-finish".to_string())),
                    ("level".to_string(), LogJson::String("info".to_string())),
                    (
                        "best".to_string(),
                        LogJson::Object(vec![
                            (
                                "x".to_string(),
                                LogJson::Array(best.x.iter().map(|v| LogJson::Number(*v)).collect()),
                            ),
                            ("meanProfit".to_string(), LogJson::Number(best.mean_profit)),
                            ("sdProfit".to_string(), LogJson::Number(best.sd_profit)),
                            ("cvarLoss".to_string(), LogJson::Number(best.cvar_loss)),
                            ("serviceLevel".to_string(), LogJson::Number(best.service_level)),
                            ("robustObjective".to_string(), LogJson::Number(best.robust_objective)),
                            ("feasible".to_string(), LogJson::Bool(best.feasible)),
                        ]),
                    ),
                ]));
            }
            result
        })
    }
    fn summarize(&self, r: &RiskCapacityResult, _params: &RiskCapacityParams) -> String {
        [
            "RISK CAPACITY".to_string(),
            "------------------------".to_string(),
            format!(
                "  risk={} scenarios={}",
                risk_kind_str(r.params.risk.kind),
                r.scenarios.len()
            ),
            format!(
                "  best x=[{}] objective={:.3}",
                join_nums(&r.best.x),
                r.best.robust_objective
            ),
            format!(
                "  mean={:.3} sd={:.3} service={:.1}% CVaR(loss)={:.3}",
                r.best.mean_profit,
                r.best.sd_profit,
                100.0 * r.best.service_level,
                r.best.cvar_loss
            ),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &RiskCapacityResult, csv_path: &str) {
        let mut lines =
            vec!["x,mean_profit,sd_profit,cvar_loss,service_level,robust_objective,feasible".to_string()];
        for c in &r.candidates {
            lines.push(csv_row([
                json_num_array(&c.x),
                js_number(c.mean_profit),
                js_number(c.sd_profit),
                js_number(c.cvar_loss),
                js_number(c.service_level),
                js_number(c.robust_objective),
                if c.feasible { "1".to_string() } else { "0".to_string() },
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _r: &RiskCapacityResult,
        _params: &RiskCapacityParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<RiskCapacityParams>> {
        vec![example(
            "CVaR capacity",
            "risk-capacity",
            RiskCapacityParams {
                cost: vec![10.0, 12.0],
                price: vec![25.0, 28.0],
                demand: DemandSpec::Uniform(vec![
                    DemandRange { low: 50.0, high: 100.0 },
                    DemandRange { low: 40.0, high: 80.0 },
                ]),
                num_scenarios: 250,
                seed: 5,
                x_max: 120.0,
                step: 10.0,
                risk: RiskConfig {
                    kind: RiskKind::Cvar,
                    alpha: Some(0.9),
                    lambda: Some(0.2),
                    min_service_level: None,
                    shortfall_limit: None,
                    radius: None,
                },
            },
        )]
    }
}

// =============================================================================
// sddp-capacity
// =============================================================================

pub struct SddpCapacityAdapter;
pub fn adapter_sddp_capacity() -> SddpCapacityAdapter {
    SddpCapacityAdapter
}

impl DESModelRegistration<SDDPParams, SDDPResult> for SddpCapacityAdapter {
    fn id(&self) -> &str {
        "sddp-capacity"
    }
    fn description(&self) -> &str {
        "Multi-stage stochastic capacity expansion via SDDP-style value-function cuts."
    }
    fn schema(&self) -> ParamSchema {
        sddp_params_schema()
    }
    fn run(&self, params: SDDPParams, _runtime: &DESRuntimeConfig) -> SDDPResult {
        // PORT NOTE: TS passes the withLogger JsonlLogger into runCapacityExpansionSDDP;
        // the Rust engine takes an owned `Option<Box<dyn OptimizationLogger>>` ('static)
        // which cannot borrow the with_logger JsonlLogger, so logging is omitted (None).
        run_capacity_expansion_sddp(params, None).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &SDDPResult, _params: &SDDPParams) -> String {
        let cut_lengths: Vec<usize> = r.cuts.iter().map(|c| c.len()).collect();
        [
            "SDDP CAPACITY".to_string(),
            "------------------------".to_string(),
            format!("  horizon={} iterations={}", r.params.horizon, r.trace.len()),
            format!("  exact sampled-grid objective={:.4}", r.exact_objective),
            format!(
                "  upper={:.4} lower={:.4} gap={:.4}",
                r.final_upper_bound, r.final_lower_bound, r.gap
            ),
            format!("  cuts by stage=[{}]", join_usize(&cut_lengths)),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &SDDPResult, csv_path: &str) {
        let mut lines = vec![
            "iter,upper_bound,lower_bound,exact_objective,gap,cut_counts,forward_states,forward_return"
                .to_string(),
        ];
        for t in &r.trace {
            lines.push(csv_row([
                js_number(t.iter as f64),
                js_number(t.upper_bound),
                js_number(t.lower_bound),
                js_number(t.exact_objective),
                js_number(t.gap),
                json_usize_array(&t.cut_counts),
                json_num_array(&t.forward_states),
                js_number(t.forward_return),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(&self, _r: &SDDPResult, _params: &SDDPParams, _runtime: &DESRuntimeConfig) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<SDDPParams>> {
        vec![example(
            "3-stage capacity expansion",
            "sddp-capacity",
            SDDPParams {
                horizon: 3,
                demand: vec![
                    DemandRange { low: 20.0, high: 50.0 },
                    DemandRange { low: 30.0, high: 70.0 },
                    DemandRange { low: 40.0, high: 90.0 },
                ],
                price: vec![25.0, 24.0, 23.0],
                expansion_cost: vec![12.0, 10.0, 8.0],
                initial_capacity: 0.0,
                x_max: 100.0,
                step: 10.0,
                samples_per_stage: 50,
                seed: 7,
                max_iter: Some(35),
                tol: Some(0.01),
            },
        )]
    }
}

// =============================================================================
// adaptive-simopt
// =============================================================================

pub struct AdaptiveSimOptAdapter;
pub fn adapter_adaptive_simopt() -> AdaptiveSimOptAdapter {
    AdaptiveSimOptAdapter
}

impl DESModelRegistration<AdaptiveSimOptParams, AdaptiveSimOptResult> for AdaptiveSimOptAdapter {
    fn id(&self) -> &str {
        "adaptive-simopt"
    }
    fn description(&self) -> &str {
        "Adaptive simulation optimisation with sequential UCB allocation across candidate policies."
    }
    fn schema(&self) -> ParamSchema {
        adaptive_params_schema()
    }
    fn run(&self, params: AdaptiveSimOptParams, _runtime: &DESRuntimeConfig) -> AdaptiveSimOptResult {
        // PORT NOTE: TS passes the withLogger JsonlLogger into runAdaptiveSimOpt; the Rust
        // engine takes an owned `Option<Box<dyn OptimizationLogger>>` ('static) which cannot
        // borrow the with_logger JsonlLogger, so logging is omitted (None).
        run_adaptive_sim_opt(params, None).unwrap_or_else(|e| panic!("{e}"))
    }
    fn summarize(&self, r: &AdaptiveSimOptResult, _params: &AdaptiveSimOptParams) -> String {
        let total: f64 = r.stats.iter().map(|a| a.n).sum();
        [
            "ADAPTIVE SIMOPT".to_string(),
            "------------------------".to_string(),
            format!(
                "  best={} x=[{}] mean={:.3} stderr={:.3} n={}",
                r.best.name,
                join_nums(&r.best.x),
                r.best.mean,
                r.best.stderr,
                js_number(r.best.n)
            ),
            format!("  total samples={} alternatives={}", js_number(total), r.stats.len()),
            format!("  validation: {}", validation_line(&r.validation)),
        ]
        .join("\n")
    }
    fn write_csv(&self, r: &AdaptiveSimOptResult, csv_path: &str) {
        let mut lines = vec!["name,x,n,mean,sd,stderr,ucb".to_string()];
        for s in &r.stats {
            lines.push(csv_row([
                s.name.clone(),
                json_num_array(&s.x),
                js_number(s.n),
                js_number(s.mean),
                js_number(s.sd),
                js_number(s.stderr),
                js_number(s.ucb),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
    fn animate(
        &self,
        _r: &AdaptiveSimOptResult,
        _params: &AdaptiveSimOptParams,
        _runtime: &DESRuntimeConfig,
    ) {
        // PORT NOTE: animation subsystem not ported (see module docs). No-op.
    }
    fn examples(&self) -> Vec<RegistrationExample<AdaptiveSimOptParams>> {
        vec![example(
            "adaptive capacity candidates",
            "adaptive-simopt",
            AdaptiveSimOptParams {
                cost: vec![10.0, 12.0],
                price: vec![25.0, 28.0],
                demand: DemandSpec::Uniform(vec![
                    DemandRange { low: 50.0, high: 100.0 },
                    DemandRange { low: 40.0, high: 80.0 },
                ]),
                alternatives: vec![
                    AdaptiveAlternative { name: "lean".to_string(), x: vec![60.0, 50.0] },
                    AdaptiveAlternative { name: "balanced".to_string(), x: vec![80.0, 65.0] },
                    AdaptiveAlternative { name: "buffered".to_string(), x: vec![100.0, 80.0] },
                ],
                seed: 11,
                initial_samples: 5,
                budget: 120,
                batch_size: 5,
                exploration: 1.5,
            },
        )]
    }
}
