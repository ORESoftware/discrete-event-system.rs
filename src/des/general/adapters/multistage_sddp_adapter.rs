//! Port of `src/des/general/adapters/multistage-sddp-adapter.ts`
//! (module `des::general::adapters::multistage_sddp_adapter`).
//!
//! JSON adapter registering the multistage-SDDP inventory model.
//!
//! ## Conversion notes
//!
//!   * `params.problem ?? buildDefault…()`, `params.options ?? {}` ->
//!     `Option::unwrap_or_else` / `unwrap_or_default`.
//!   * `gapToExact?.toExponential(3) ?? 'n/a'` -> [`js_to_exponential`] +
//!     `Option`-formatting; `tr.policyValue ?? ''` / `tr.gapToExact ?? ''` ->
//!     `Option`-to-cell formatting.
//!   * `SDDPStatus` is printed via a `match` to its TS string form.
//!   * NOTE (as in the TS source): this is a near-duplicate of the multistage
//!     block in `stochastic-optimization-adapters.ts`; both register id
//!     `multistage-sddp`.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet (Rust has
//! no module-load side effects); the adapter is exposed via [`adapter()`] for
//! the integrator to wire in.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};
use crate::des::general::multistage_stochastic::{
    build_default_multi_stage_inventory_problem, run_multi_stage_inventory_demo,
    MultiStageInventoryProblem, MultiStageRunResult, SDDPOptions, SDDPStatus,
};

/// `interface MultiStageParams`.
#[derive(Clone, Debug, Default)]
pub struct MultiStageParams {
    pub problem: Option<MultiStageInventoryProblem>,
    pub options: Option<SDDPOptions>,
}

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

fn arr(items: ParamSchema) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length: None,
        max_length: None,
        description: None,
    }
}

fn demand_outcome_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("demand".to_string(), num(Some(0.0), None, None, None)),
            ("prob".to_string(), num(Some(0.0), Some(1.0), None, None)),
        ],
        required: Some(vec!["demand".to_string(), "prob".to_string()]),
        description: None,
    }
}

fn multi_stage_problem_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "horizon".to_string(),
                num(Some(1.0), None, Some(true), None),
            ),
            (
                "initialInventory".to_string(),
                num(Some(0.0), None, None, None),
            ),
            ("capacity".to_string(), num(Some(1e-9), None, None, None)),
            (
                "maxOrder".to_string(),
                arr(num(Some(0.0), None, None, None)),
            ),
            ("price".to_string(), arr(num(Some(0.0), None, None, None))),
            (
                "orderCost".to_string(),
                arr(num(Some(0.0), None, None, None)),
            ),
            (
                "holdCost".to_string(),
                arr(num(Some(0.0), None, None, None)),
            ),
            (
                "stockoutCost".to_string(),
                arr(num(Some(0.0), None, None, None)),
            ),
            ("salvageValue".to_string(), num(Some(0.0), None, None, None)),
            ("demands".to_string(), arr(arr(demand_outcome_schema()))),
        ],
        required: Some(vec![
            "horizon".to_string(),
            "initialInventory".to_string(),
            "capacity".to_string(),
            "maxOrder".to_string(),
            "price".to_string(),
            "orderCost".to_string(),
            "holdCost".to_string(),
            "stockoutCost".to_string(),
            "salvageValue".to_string(),
            "demands".to_string(),
        ]),
        description: Some("Multi-stage inventory/storage stochastic program.".to_string()),
    }
}

/// `const multiStageSchema`.
pub fn multi_stage_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("problem".to_string(), multi_stage_problem_schema()),
            (
                "options".to_string(),
                ParamSchema::Object {
                    fields: vec![
                        (
                            "maxIter".to_string(),
                            num(Some(1.0), None, Some(true), Some(80.0)),
                        ),
                        ("tol".to_string(), num(Some(0.0), None, None, Some(1e-4))),
                        ("seed".to_string(), num(None, None, Some(true), Some(1.0))),
                        (
                            "evaluatePolicyEvery".to_string(),
                            num(Some(1.0), None, Some(true), Some(80.0)),
                        ),
                        (
                            "finiteDiffStep".to_string(),
                            num(Some(1e-9), None, None, None),
                        ),
                        (
                            "cutGridSize".to_string(),
                            num(Some(2.0), None, Some(true), Some(21.0)),
                        ),
                    ],
                    required: Some(vec![]),
                    description: None,
                },
            ),
        ],
        required: Some(vec![]),
        description: Some(
            "Multi-stage stochastic inventory solved by SDDP and exact scenario tree validation."
                .to_string(),
        ),
    }
}

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

fn sddp_status_str(status: SDDPStatus) -> &'static str {
    match status {
        SDDPStatus::Optimal => "optimal",
        SDDPStatus::IterLimit => "iter-limit",
    }
}

/// `const multiStageAdapter`.
pub struct MultiStageAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> MultiStageAdapter {
    MultiStageAdapter
}

impl DESModelRegistration<MultiStageParams, MultiStageRunResult> for MultiStageAdapter {
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
        let problem = params
            .problem
            .unwrap_or_else(build_default_multi_stage_inventory_problem);
        run_multi_stage_inventory_demo(problem, params.options.unwrap_or_default())
    }

    fn summarize(&self, result: &MultiStageRunResult, _params: &MultiStageParams) -> String {
        let gap = match result.sddp.gap_to_exact {
            Some(g) => js_to_exponential(g, 3),
            None => "n/a".to_string(),
        };
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
            format!("  Gap to exact:    {gap}"),
            format!("  Cuts/stage:      [{cuts_per_stage}]"),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &MultiStageRunResult, csv_path: &str) {
        let mut lines = vec![
            "iter,upper_bound,policy_value,gap_to_exact,terminal_inventory,cuts_added".to_string(),
        ];
        for tr in &result.sddp.trace {
            lines.push(csv_row([
                tr.iter.to_string(),
                tr.upper_bound.to_string(),
                tr.policy_value.map(|v| v.to_string()).unwrap_or_default(),
                tr.gap_to_exact.map(|v| v.to_string()).unwrap_or_default(),
                tr.terminal_inventory.to_string(),
                tr.cuts_added.len().to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }
}
