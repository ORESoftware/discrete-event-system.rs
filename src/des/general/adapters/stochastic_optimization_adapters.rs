//! Port of `src/des/general/adapters/stochastic-optimization-adapters.ts`
//! (module `des::general::adapters::stochastic_optimization_adapters`).
//!
//! Two JSON adapters: the two-stage stochastic LP (`stochastic-lp`, SAA +
//! Benders) and a multi-stage SDDP inventory model (`multistage-sddp`).
//!
//! ## Conversion notes
//!
//!   * `closedForm?: ReturnType<typeof solveProductionClosedForm>` -> an
//!     explicit `Option<SLPSolveResult>`.
//!   * `ranges: Array<[number, number]>` -> `Vec<(f64, f64)>`; the equal-length
//!     `c`/`p`/`ranges` invariant `throw` -> `panic!`.
//!   * The `evalX` closure becomes a borrowing `eval_x` closure over the
//!     out-of-sample scenarios.
//!   * `solveSLPMonolithic` / `solveSLPBenders` take the problem + scenarios by
//!     value here, so both are cloned (the TS reused one object reference).
//!   * `gapToExact?.toExponential(3) ?? 'n/a'`, `params.budget ?? 'none'`,
//!     `tr.policyValue ?? ''` -> `Option` formatting helpers.
//!
//! PORT NOTE: this multistage block is a near-duplicate of
//! `multistage_sddp_adapter.rs` (both register `multistage-sddp`), mirroring the
//! TS sources. The integrator should register only one.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the
//! adapters are exposed via [`stochastic_lp_adapter`] / [`multi_stage_adapter`].

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, json_csv_row, write_csv_lines};
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};
use crate::des::general::multistage_stochastic::{
    build_default_multi_stage_inventory_problem, run_multi_stage_inventory_demo,
    MultiStageInventoryProblem, MultiStageRunResult, SDDPOptions, SDDPStatus,
};
use crate::des::general::stochastic_lp::{
    build_production_scenarios, build_production_slp, solve_production_closed_form,
    solve_slp_benders, solve_slp_monolithic, BendersOpts, SLPMethod, SLPSolveResult, SLPStatus,
    UniformDemandSpec,
};

// =============================================================================
// Number-formatting helpers (JS parity).
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

fn json_num(v: f64) -> String {
    if v.is_finite() {
        js_number(v)
    } else {
        "null".to_string()
    }
}

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

/// `JSON.stringify(numbers)` for a number array.
fn json_num_array(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| json_num(*x)).collect::<Vec<_>>().join(","))
}

/// `function fmtVec(x)` — `x.map(v => v.toFixed(4)).join(', ')`.
fn fmt_vec(x: &[f64]) -> String {
    x.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", ")
}

fn slp_status_str(s: SLPStatus) -> &'static str {
    match s {
        SLPStatus::Optimal => "optimal",
        SLPStatus::Unbounded => "unbounded",
        SLPStatus::Infeasible => "infeasible",
        SLPStatus::IterLimit => "iter-limit",
    }
}

fn slp_method_str(m: SLPMethod) -> &'static str {
    match m {
        SLPMethod::Monolithic => "monolithic",
        SLPMethod::Benders => "benders",
        SLPMethod::ClosedForm => "closed-form",
    }
}

fn sddp_status_str(s: SDDPStatus) -> &'static str {
    match s {
        SDDPStatus::Optimal => "optimal",
        SDDPStatus::IterLimit => "iter-limit",
    }
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default, description: None }
}

fn arr(items: ParamSchema, min_length: Option<usize>, max_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length, max_length, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>, description: Option<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: description.map(|s| s.to_string()),
    }
}

// =============================================================================
// Two-stage stochastic LP adapter
// =============================================================================

/// `interface StochasticLPParams`.
#[derive(Clone, Debug)]
pub struct StochasticLPParams {
    pub c: Vec<f64>,
    pub p: Vec<f64>,
    pub ranges: Vec<(f64, f64)>,
    pub n: Option<usize>,
    pub seed: Option<u32>,
    pub budget: Option<f64>,
    pub tol: Option<f64>,
    pub max_iter: Option<usize>,
    pub oos_n: Option<usize>,
}

/// `StochasticLPRunResult['outOfSample']`.
#[derive(Clone, Debug)]
pub struct OutOfSample {
    pub n: usize,
    pub monolithic: f64,
    pub benders: f64,
    pub closed_form: Option<f64>,
}

/// `interface StochasticLPRunResult`.
#[derive(Clone, Debug)]
pub struct StochasticLPRunResult {
    pub closed_form: Option<SLPSolveResult>,
    pub monolithic: SLPSolveResult,
    pub benders: SLPSolveResult,
    pub out_of_sample: Option<OutOfSample>,
}

fn pair_schema() -> ParamSchema {
    arr(num(None, None, None, None), Some(2), Some(2))
}

/// `const stochasticLPSchema`.
pub fn stochastic_lp_schema() -> ParamSchema {
    obj(
        vec![
            ("c", arr(num(None, None, None, None), None, None)),
            ("p", arr(num(None, None, None, None), None, None)),
            ("ranges", arr(pair_schema(), None, None)),
            ("N", num(Some(1.0), None, Some(true), Some(200.0))),
            ("seed", num(None, None, Some(true), Some(42.0))),
            ("budget", num(Some(0.0), None, None, None)),
            ("tol", num(Some(0.0), None, None, Some(1e-7))),
            ("maxIter", num(Some(1.0), None, Some(true), Some(200.0))),
            ("oosN", num(Some(0.0), None, Some(true), Some(0.0))),
        ],
        vec!["c", "p", "ranges"],
        Some("Two-stage stochastic LP: production capacity under demand uncertainty."),
    )
}

/// `const stochasticLPAdapter`.
pub struct StochasticLpAdapter;

/// Construct the stochastic-LP adapter (see the module PORT NOTE on registration).
pub fn stochastic_lp_adapter() -> StochasticLpAdapter {
    StochasticLpAdapter
}

impl DESModelRegistration<StochasticLPParams, StochasticLPRunResult> for StochasticLpAdapter {
    fn id(&self) -> &str {
        "stochastic-lp"
    }

    fn description(&self) -> &str {
        "Two-stage stochastic LP via SAA and Benders/L-shaped decomposition."
    }

    fn schema(&self) -> ParamSchema {
        stochastic_lp_schema()
    }

    fn run(&self, params: StochasticLPParams, _runtime: &DESRuntimeConfig) -> StochasticLPRunResult {
        if params.c.len() != params.p.len() || params.c.len() != params.ranges.len() {
            panic!("stochastic-lp: c, p, and ranges must have the same length");
        }
        let n = params.n.unwrap_or(200);
        let seed = params.seed.unwrap_or(42);
        let slp = build_production_slp(params.c.clone(), params.p.clone(), params.budget);
        let scenarios =
            build_production_scenarios(UniformDemandSpec { ranges: params.ranges.clone(), seed }, n);
        let closed_form = if params.budget.is_none() {
            Some(solve_production_closed_form(
                params.c.clone(),
                params.p.clone(),
                params.ranges.clone(),
            ))
        } else {
            None
        };
        let monolithic = solve_slp_monolithic(slp.clone(), scenarios.clone());
        let benders = solve_slp_benders(
            slp,
            scenarios,
            BendersOpts {
                max_iter: Some(params.max_iter.unwrap_or(200)),
                tol: Some(params.tol.unwrap_or(1e-7)),
                verbose: None,
                reference_path: None,
                reference_tol: None,
                silent_if_missing: None,
            },
        );

        let oos_n = params.oos_n.unwrap_or(0);
        let out_of_sample = if oos_n > 0 {
            let oos = build_production_scenarios(
                UniformDemandSpec { ranges: params.ranges.clone(), seed: seed + 99991 },
                oos_n,
            );
            let eval_x = |x: &[f64]| -> f64 {
                let mut z = 0.0;
                for i in 0..params.c.len() {
                    z += -params.c[i] * x[i];
                }
                let mut q = 0.0;
                for sc in &oos {
                    let d = sc.meta.as_ref().expect("production scenario carries demand meta");
                    for i in 0..params.p.len() {
                        q += params.p[i] * x[i].min(d.d[i]);
                    }
                }
                z + q / oos.len() as f64
            };
            Some(OutOfSample {
                n: oos_n,
                monolithic: eval_x(&monolithic.x),
                benders: eval_x(&benders.x),
                closed_form: closed_form.as_ref().map(|cf| eval_x(&cf.x)),
            })
        } else {
            None
        };

        StochasticLPRunResult { closed_form, monolithic, benders, out_of_sample }
    }

    fn summarize(&self, result: &StochasticLPRunResult, params: &StochasticLPParams) -> String {
        let mut lines = vec![
            "STOCHASTIC LP (two-stage SAA + Benders)".to_string(),
            "---------------------------------------".to_string(),
            format!("  Scenarios:       {}", params.n.unwrap_or(200)),
            format!(
                "  Budget:          {}",
                params.budget.map(js_number).unwrap_or_else(|| "none".to_string())
            ),
            format!(
                "  Monolithic:      {}  z={:.6}  x=[{}]",
                slp_status_str(result.monolithic.status),
                result.monolithic.objective,
                fmt_vec(&result.monolithic.x)
            ),
            format!(
                "  Benders:         {}  z={:.6}  x=[{}]",
                slp_status_str(result.benders.status),
                result.benders.objective,
                fmt_vec(&result.benders.x)
            ),
            format!(
                "  |Delta z|:       {}",
                to_exponential((result.benders.objective - result.monolithic.objective).abs(), 2)
            ),
            format!("  Benders iters:   {}", result.benders.iterations),
        ];
        if let Some(cf) = &result.closed_form {
            lines.push(format!(
                "  Closed-form:     z={:.6}  x=[{}]",
                cf.objective,
                fmt_vec(&cf.x)
            ));
        }
        if let Some(oos) = &result.out_of_sample {
            lines.push(format!(
                "  OOS N={}: monolithic={:.4}  benders={:.4}",
                oos.n, oos.monolithic, oos.benders
            ));
        }
        lines.join("\n")
    }

    fn write_csv(&self, result: &StochasticLPRunResult, csv_path: &str) {
        let mut lines = vec!["method,status,objective,iterations,x".to_string()];
        for row in [&result.monolithic, &result.benders] {
            lines.push(json_csv_row([
                slp_method_str(row.method).to_string(),
                slp_status_str(row.status).to_string(),
                json_num(row.objective),
                row.iterations.to_string(),
                json_num_array(&row.x),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}

// =============================================================================
// Multi-stage SDDP adapter
// =============================================================================

/// `options` block for the multistage adapter.
#[derive(Clone, Debug, Default)]
pub struct SddpOptionsParams {
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub seed: Option<u32>,
    pub evaluate_policy_every: Option<usize>,
    pub finite_diff_step: Option<f64>,
    pub cut_grid_size: Option<usize>,
}

/// `interface MultiStageParams`.
#[derive(Clone, Debug, Default)]
pub struct MultiStageParams {
    pub problem: Option<MultiStageInventoryProblem>,
    pub options: Option<SddpOptionsParams>,
}

fn demand_outcome_schema() -> ParamSchema {
    obj(
        vec![
            ("demand", num(Some(0.0), None, None, None)),
            ("prob", num(Some(0.0), Some(1.0), None, None)),
        ],
        vec!["demand", "prob"],
        None,
    )
}

fn multi_stage_problem_schema() -> ParamSchema {
    obj(
        vec![
            ("horizon", num(Some(1.0), None, Some(true), None)),
            ("initialInventory", num(Some(0.0), None, None, None)),
            ("capacity", num(Some(1e-9), None, None, None)),
            ("maxOrder", arr(num(Some(0.0), None, None, None), None, None)),
            ("price", arr(num(Some(0.0), None, None, None), None, None)),
            ("orderCost", arr(num(Some(0.0), None, None, None), None, None)),
            ("holdCost", arr(num(Some(0.0), None, None, None), None, None)),
            ("stockoutCost", arr(num(Some(0.0), None, None, None), None, None)),
            ("salvageValue", num(Some(0.0), None, None, None)),
            ("demands", arr(arr(demand_outcome_schema(), None, None), None, None)),
        ],
        vec![
            "horizon",
            "initialInventory",
            "capacity",
            "maxOrder",
            "price",
            "orderCost",
            "holdCost",
            "stockoutCost",
            "salvageValue",
            "demands",
        ],
        Some("Multi-stage inventory/storage stochastic program."),
    )
}

/// `const multiStageSchema`.
pub fn multi_stage_schema() -> ParamSchema {
    obj(
        vec![
            ("problem", multi_stage_problem_schema()),
            (
                "options",
                obj(
                    vec![
                        ("maxIter", num(Some(1.0), None, Some(true), Some(80.0))),
                        ("tol", num(Some(0.0), None, None, Some(1e-4))),
                        ("seed", num(None, None, Some(true), Some(1.0))),
                        ("evaluatePolicyEvery", num(Some(1.0), None, Some(true), Some(80.0))),
                        ("finiteDiffStep", num(Some(1e-9), None, None, None)),
                        ("cutGridSize", num(Some(2.0), None, Some(true), Some(21.0))),
                    ],
                    vec![],
                    None,
                ),
            ),
        ],
        vec![],
        Some("Multi-stage stochastic inventory solved by SDDP and exact scenario tree validation."),
    )
}

/// `const multiStageAdapter`.
pub struct MultiStageStochasticAdapter;

/// Construct the multistage adapter (see the module PORT NOTE on registration).
pub fn multi_stage_adapter() -> MultiStageStochasticAdapter {
    MultiStageStochasticAdapter
}

impl DESModelRegistration<MultiStageParams, MultiStageRunResult> for MultiStageStochasticAdapter {
    fn id(&self) -> &str {
        "multistage-sddp"
    }

    fn description(&self) -> &str {
        "Multi-stage stochastic inventory/storage optimisation via SDDP cut recursion."
    }

    fn schema(&self) -> ParamSchema {
        multi_stage_schema()
    }

    fn run(&self, params: MultiStageParams, _runtime: &DESRuntimeConfig) -> MultiStageRunResult {
        let problem = params.problem.unwrap_or_else(build_default_multi_stage_inventory_problem);
        let options = params
            .options
            .map(|o| SDDPOptions {
                max_iter: o.max_iter,
                tol: o.tol,
                seed: o.seed,
                exact_objective: None,
                evaluate_policy_every: o.evaluate_policy_every,
                finite_diff_step: o.finite_diff_step,
                cut_grid_size: o.cut_grid_size,
            })
            .unwrap_or_default();
        run_multi_stage_inventory_demo(problem, options)
    }

    fn summarize(&self, result: &MultiStageRunResult, _params: &MultiStageParams) -> String {
        let cuts_per_stage = result
            .sddp
            .cuts_per_stage
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        [
            "MULTI-STAGE STOCHASTIC PROGRAM (SDDP)".to_string(),
            "-------------------------------------".to_string(),
            format!(
                "  Exact tree:      {}  z={:.6}  nodes={}",
                result.exact.status, result.exact.objective, result.exact.node_count
            ),
            format!("  SDDP status:     {}", sddp_status_str(result.sddp.status)),
            format!("  SDDP iters:      {}", result.sddp.iterations),
            format!("  Upper bound:     {:.6}", result.sddp.upper_bound),
            format!("  Policy value:    {:.6}", result.sddp.policy_value),
            format!(
                "  Gap to exact:    {}",
                result.sddp.gap_to_exact.map(|g| to_exponential(g, 3)).unwrap_or_else(|| "n/a".to_string())
            ),
            format!("  Cuts/stage:      [{cuts_per_stage}]"),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &MultiStageRunResult, csv_path: &str) {
        let mut lines =
            vec!["iter,upper_bound,policy_value,gap_to_exact,terminal_inventory,cuts_added".to_string()];
        for tr in &result.sddp.trace {
            lines.push(csv_row([
                tr.iter.to_string(),
                js_number(tr.upper_bound),
                tr.policy_value.map(js_number).unwrap_or_default(),
                tr.gap_to_exact.map(js_number).unwrap_or_default(),
                js_number(tr.terminal_inventory),
                tr.cuts_added.len().to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}
