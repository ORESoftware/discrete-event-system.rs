//! Port of `src/des/general/adapters/domain-application-adapter.ts`
//! (module `des::general::adapters::domain_application_adapter`).
//!
//! Registers 11 domain-application JSON adapters (fuzzy control, logistics,
//! manufacturing, supply chain, ops, finance, pricing, energy, active learning,
//! decision science) sharing one summarize / CSV factory set.
//!
//! ## Conversion notes
//!
//!   * The TS `domainSummary(title)` / `animateDomainModel(title)` higher-order
//!     factories become a single shared `domain_summary(title, result)` /
//!     `write_domain_csv` plus one adapter struct per model carrying its title.
//!   * `row.metrics: Record<string, number|string|boolean>` -> ordered
//!     `Vec<(String, MetricValue)>` (insertion order, like a JS object).
//!   * `row.plan: unknown` -> [`JsonValue`]; `jsonCsvCell(...)` receives the
//!     pre-stringified JSON (matching the `adapter_utils` convention).
//!
//! PORT NOTE: the `des/general/domain-application-models` engine is now ported
//! (`crate::des::general::domain_application_models`). Each `run_*` kernel
//! delegates to the real engine and erases the typed plan `P` to a `JsonValue`
//! (the TS `DomainModelResult<unknown>` erasure) so the shared summarize/CSV
//! glue stays plan-agnostic. The parameter types are re-exported from the
//! engine; the erased result/candidate/trace/metric view types are kept local.
//!
//! PORT NOTE: the animation subsystem (`animation/frame-recorder`,
//! `animation/types`, and `buildDomainFrame`/`drawCandidateBars`/
//! `drawMetricPanel`/`domainCharts`/`seriesIfPresent`) is not ported, so
//! `animate` is a no-op here.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, json_csv_cell, write_csv_lines};
use crate::des::general::des_base::learning_optimization::StationGraphSummary;
use crate::des::general::des_spec::{
    DESModelRegistration, DESRuntimeConfig, JsonValue, ParamSchema,
};
use crate::des::general::domain_application_models as engine;

// =============================================================================
// PORT NOTE: local stubs for the unported `domain-application-models` engine.
// =============================================================================

/// `number | string | boolean` metric value (TS `Record` value union).
#[derive(Clone, Debug)]
pub enum MetricValue {
    Number(f64),
    Text(String),
    Flag(bool),
}

/// Insertion-ordered metric map (mirrors a JS object's `Object.entries`).
pub type MetricMap = Vec<(String, MetricValue)>;

/// `DomainTrace` — time base plus named series. (Used only by the unported
/// animation path; retained for the integrator.)
#[derive(Clone, Debug, Default)]
pub struct DomainTrace {
    pub t: Vec<f64>,
    pub series: std::collections::HashMap<String, Vec<f64>>,
}

/// One evaluated candidate plan.
#[derive(Clone, Debug)]
pub struct DomainCandidate {
    pub candidate_id: String,
    pub objective: f64,
    pub feasible: bool,
    pub metrics: MetricMap,
    pub plan: JsonValue,
    pub trace: Option<DomainTrace>,
}

/// `DomainModelResult<unknown>` (plan type erased to [`JsonValue`]).
#[derive(Clone, Debug)]
pub struct DomainModelResult {
    pub model_id: String,
    pub category: String,
    pub best: DomainCandidate,
    pub candidates: Vec<DomainCandidate>,
    pub topology: StationGraphSummary,
}

// Per-model parameters are the real engine structs (the `domain-application-
// models` engine is now ported); re-exported so the adapter and its callers
// keep the original type names.
pub use engine::DecisionScienceParams;
pub use engine::{
    ActiveLearningParams, AdaptiveFuzzyControlParams, BuyerAwareDynamicPricingParams, EnergyParams,
    FinancialControlParams, LogisticsRoutingParams, ManufacturingParams, OperationsParams,
    RevenueManagementParams, SupplyChainParams,
};

// All result aliases collapse onto the erased `DomainModelResult`.
pub type AdaptiveFuzzyControlResult = DomainModelResult;
pub type LogisticsRoutingResult = DomainModelResult;
pub type ManufacturingResult = DomainModelResult;
pub type SupplyChainResult = DomainModelResult;
pub type OperationsResult = DomainModelResult;
pub type FinancialControlResult = DomainModelResult;
pub type RevenueManagementResult = DomainModelResult;
pub type BuyerAwareDynamicPricingResult = DomainModelResult;
pub type EnergyResult = DomainModelResult;
pub type ActiveLearningResult = DomainModelResult;
pub type DecisionScienceResult = DomainModelResult;

// =============================================================================
// Engine bridge: call the real `domain_application_models` kernels and erase the
// typed plan `P` to a `JsonValue` (via serde) so the shared summarize/CSV glue
// stays plan-agnostic, mirroring the TS `DomainModelResult<unknown>` erasure.
// =============================================================================

fn erase_metric(v: engine::MetricValue) -> MetricValue {
    match v {
        engine::MetricValue::Num(n) => MetricValue::Number(n),
        engine::MetricValue::Str(s) => MetricValue::Text(s),
        engine::MetricValue::Bool(b) => MetricValue::Flag(b),
    }
}

fn erase_trace(t: engine::DomainTrace) -> DomainTrace {
    DomainTrace {
        t: t.t,
        series: t.series.into_iter().collect(),
    }
}

fn erase_eval<P: serde::Serialize>(e: engine::DomainEvaluation<P>) -> DomainCandidate {
    DomainCandidate {
        candidate_id: e.candidate_id,
        objective: e.objective,
        feasible: e.feasible,
        metrics: e
            .metrics
            .into_iter()
            .map(|(k, v)| (k, erase_metric(v)))
            .collect(),
        // TS `JSON.stringify(plan)`; serde -> `serde_json::Value` -> `JsonValue`.
        plan: serde_json::to_value(&e.plan)
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
        trace: e.trace.map(erase_trace),
    }
}

fn erase<P: serde::Serialize>(r: engine::DomainModelResult<P>) -> DomainModelResult {
    DomainModelResult {
        model_id: r.model_id,
        category: r.category,
        best: erase_eval(r.best),
        candidates: r.candidates.into_iter().map(erase_eval).collect(),
        topology: r.topology,
    }
}

pub fn run_adaptive_fuzzy_control(
    params: AdaptiveFuzzyControlParams,
) -> AdaptiveFuzzyControlResult {
    erase(engine::run_adaptive_fuzzy_control(params))
}
pub fn run_logistics_routing_heuristics(params: LogisticsRoutingParams) -> LogisticsRoutingResult {
    erase(engine::run_logistics_routing_heuristics(params))
}
pub fn run_bottleneck_production_control(params: ManufacturingParams) -> ManufacturingResult {
    erase(engine::run_bottleneck_production_control(params))
}
pub fn run_supply_chain_risk_pooling(params: SupplyChainParams) -> SupplyChainResult {
    erase(engine::run_supply_chain_risk_pooling(params))
}
pub fn run_workforce_service_operations(params: OperationsParams) -> OperationsResult {
    erase(engine::run_workforce_service_operations(params))
}
pub fn run_portfolio_drawdown_control(params: FinancialControlParams) -> FinancialControlResult {
    erase(engine::run_portfolio_drawdown_control(params))
}
pub fn run_dynamic_pricing_revenue(params: RevenueManagementParams) -> RevenueManagementResult {
    erase(engine::run_dynamic_pricing_revenue(params))
}
pub fn run_buyer_aware_dynamic_pricing(
    params: BuyerAwareDynamicPricingParams,
) -> BuyerAwareDynamicPricingResult {
    erase(engine::run_buyer_aware_dynamic_pricing(params))
}
pub fn run_energy_storage_dispatch(params: EnergyParams) -> EnergyResult {
    erase(engine::run_energy_storage_dispatch(params))
}
pub fn run_active_learning_acquisition(params: ActiveLearningParams) -> ActiveLearningResult {
    erase(engine::run_active_learning_acquisition(params))
}
pub fn run_visual_decision_frontier(params: DecisionScienceParams) -> DecisionScienceResult {
    erase(engine::run_visual_decision_frontier(params))
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

/// `formatMetric(v)` — magnitude-dependent fixed/exponential formatting.
fn format_metric(v: f64) -> String {
    let a = v.abs();
    if a >= 1000.0 {
        format!("{v:.1}")
    } else if a >= 10.0 {
        format!("{v:.2}")
    } else if a >= 0.01 {
        format!("{v:.4}")
    } else {
        to_exponential(v, 2)
    }
}

/// `metricsLine(metrics)` — first four entries, `k=v` joined by `, `.
fn metrics_line(metrics: &MetricMap) -> String {
    metrics
        .iter()
        .take(4)
        .map(|(k, v)| match v {
            MetricValue::Number(n) => format!("{k}={}", format_metric(*n)),
            MetricValue::Text(s) => format!("{k}={s}"),
            MetricValue::Flag(b) => format!("{k}={b}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// `JSON.stringify` of a string.
fn json_quote(s: &str) -> String {
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

fn metric_value_json(v: &MetricValue) -> String {
    match v {
        MetricValue::Number(n) => {
            if n.is_finite() {
                js_number(*n)
            } else {
                "null".to_string()
            }
        }
        MetricValue::Text(s) => json_quote(s),
        MetricValue::Flag(b) => b.to_string(),
    }
}

/// `JSON.stringify(metrics)` for an ordered metric map.
fn metrics_json(metrics: &MetricMap) -> String {
    let inner: Vec<String> = metrics
        .iter()
        .map(|(k, v)| format!("{}:{}", json_quote(k), metric_value_json(v)))
        .collect();
    format!("{{{}}}", inner.join(","))
}

/// `JSON.stringify(plan)` for an erased plan value.
fn json_value_string(v: &JsonValue) -> String {
    match v {
        JsonValue::Undefined | JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if n.is_finite() {
                js_number(*n)
            } else {
                "null".to_string()
            }
        }
        JsonValue::String(s) => json_quote(s),
        JsonValue::Array(items) => {
            let inner: Vec<String> = items.iter().map(json_value_string).collect();
            format!("[{}]", inner.join(","))
        }
        JsonValue::Object(obj) => {
            let inner: Vec<String> = obj
                .keys()
                .map(|k| {
                    format!(
                        "{}:{}",
                        json_quote(k),
                        json_value_string(obj.get(k).unwrap())
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

// =============================================================================
// Shared summarize / CSV.
// =============================================================================

fn domain_summary(title: &str, result: &DomainModelResult) -> String {
    [
        title.to_string(),
        "----------------------------------------".to_string(),
        format!("  Category:       {}", result.category),
        format!("  Best plan:      {}", result.best.candidate_id),
        format!("  Objective:      {:.6}", result.best.objective),
        format!("  Metrics:        {}", metrics_line(&result.best.metrics)),
        format!("  Candidates:     {}", result.candidates.len()),
        format!(
            "  Stations:       {}",
            result.topology.stations.join(" -> ")
        ),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

fn write_domain_csv(result: &DomainModelResult, csv_path: &str) {
    let mut lines = vec![csv_row([
        "candidate_id",
        "objective",
        "feasible",
        "metrics",
        "plan",
    ])];
    for row in &result.candidates {
        let prefix = csv_row([
            row.candidate_id.clone(),
            js_number(row.objective),
            row.feasible.to_string(),
        ]);
        lines.push(format!(
            "{},{},{}",
            prefix,
            json_csv_cell(&metrics_json(&row.metrics)),
            json_csv_cell(&json_value_string(&row.plan))
        ));
    }
    write_csv_lines(csv_path, &lines);
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(min: Option<f64>, integer: Option<bool>, default: Option<f64>) -> ParamSchema {
    ParamSchema::Number {
        min,
        max: None,
        integer,
        default,
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        required: Some(Vec::new()),
        description: None,
    }
}

// =============================================================================
// Adapters (one struct per model). `examples()` defaults to empty (the TS file
// registers no examples for these models).
// =============================================================================

macro_rules! domain_adapter {
    (
        $adapter:ident, $ctor:ident, $params:ty, $result:ty,
        $id:literal, $title:literal, $desc:literal, $run:path, $schema:expr
    ) => {
        pub struct $adapter;
        pub fn $ctor() -> $adapter {
            $adapter
        }
        impl DESModelRegistration<$params, $result> for $adapter {
            fn id(&self) -> &str {
                $id
            }
            fn description(&self) -> &str {
                $desc
            }
            fn schema(&self) -> ParamSchema {
                $schema
            }
            fn run(&self, params: $params, _runtime: &DESRuntimeConfig) -> $result {
                $run(params)
            }
            fn summarize(&self, result: &$result, _params: &$params) -> String {
                domain_summary($title, result)
            }
            fn write_csv(&self, result: &$result, csv_path: &str) {
                write_domain_csv(result, csv_path);
            }
            fn animate(&self, _result: &$result, _params: &$params, _runtime: &DESRuntimeConfig) {
                // PORT NOTE: animation subsystem not ported (see module docs). No-op.
            }
        }
    };
}

domain_adapter!(
    AdaptiveFuzzyControlAdapter,
    adapter_adaptive_fuzzy_control,
    AdaptiveFuzzyControlParams,
    AdaptiveFuzzyControlResult,
    "adaptive-fuzzy-control",
    "ADAPTIVE FUZZY CONTROL",
    "Adaptive fuzzy control: tune fuzzy controller candidates over a first-order plant station graph.",
    run_adaptive_fuzzy_control,
    obj(vec![
        ("steps", num(Some(1.0), Some(true), Some(140.0))),
        ("dt", num(Some(1e-9), None, Some(0.1))),
        ("setpoint", num(None, None, Some(22.0))),
        ("initialTemp", num(None, None, Some(16.0))),
        ("outsideTemp", num(None, None, Some(8.0))),
        ("disturbance", num(Some(0.0), None, Some(0.15))),
    ])
);

domain_adapter!(
    LogisticsRoutingHeuristicsAdapter,
    adapter_logistics_routing_heuristics,
    LogisticsRoutingParams,
    LogisticsRoutingResult,
    "logistics-routing-heuristics",
    "LOGISTICS ROUTING HEURISTICS",
    "Logistics routing: compare nearest-neighbor, sweep, and savings heuristics as movable candidate plans.",
    run_logistics_routing_heuristics,
    obj(vec![("vehicleCapacity", num(Some(1e-9), None, Some(7.0)))])
);

domain_adapter!(
    BottleneckProductionControlAdapter,
    adapter_bottleneck_production_control,
    ManufacturingParams,
    ManufacturingResult,
    "bottleneck-production-control",
    "BOTTLENECK PRODUCTION CONTROL",
    "Manufacturing production control: bottleneck-buffer-rope and adaptive expedite policies.",
    run_bottleneck_production_control,
    obj(vec![
        ("horizon", num(Some(1.0), Some(true), Some(18.0))),
        ("dailyDemand", num(Some(0.0), None, Some(8.0))),
    ])
);

domain_adapter!(
    SupplyChainRiskPoolingAdapter,
    adapter_supply_chain_risk_pooling,
    SupplyChainParams,
    SupplyChainResult,
    "supply-chain-risk-pooling",
    "SUPPLY CHAIN RISK POOLING",
    "Supply chain management: multi-echelon risk-pooling reorder policy candidates.",
    run_supply_chain_risk_pooling,
    obj(vec![("horizon", num(Some(1.0), Some(true), Some(20.0)))])
);

domain_adapter!(
    WorkforceServiceOperationsAdapter,
    adapter_workforce_service_operations,
    OperationsParams,
    OperationsResult,
    "workforce-service-operations",
    "WORKFORCE SERVICE OPERATIONS",
    "Operations management: service-risk workforce roster heuristics with flex-pool control.",
    run_workforce_service_operations,
    obj(vec![("overtimeCost", num(Some(1e-9), None, Some(18.0)))])
);

domain_adapter!(
    PortfolioDrawdownControlAdapter,
    adapter_portfolio_drawdown_control,
    FinancialControlParams,
    FinancialControlResult,
    "portfolio-drawdown-control",
    "PORTFOLIO DRAWDOWN CONTROL",
    "Financial engineering: CPPI-style portfolio drawdown control candidates.",
    run_portfolio_drawdown_control,
    obj(vec![("initialWealth", num(Some(1e-9), None, Some(100.0)))])
);

domain_adapter!(
    DynamicPricingRevenueAdapter,
    adapter_dynamic_pricing_revenue,
    RevenueManagementParams,
    RevenueManagementResult,
    "dynamic-pricing-revenue",
    "DYNAMIC PRICING REVENUE MANAGEMENT",
    "Revenue management: dynamic pricing policies using scarcity and demand smoothing.",
    run_dynamic_pricing_revenue,
    obj(vec![("capacity", num(Some(1e-9), None, Some(120.0)))])
);

domain_adapter!(
    BuyerAwareDynamicPricingAdapter,
    adapter_buyer_aware_dynamic_pricing,
    BuyerAwareDynamicPricingParams,
    BuyerAwareDynamicPricingResult,
    "buyer-aware-dynamic-pricing",
    "BUYER-AWARE DYNAMIC PRICING",
    "Revenue management: buyer-aware dynamic pricing with privacy, fairness, inventory, and retention guardrails.",
    run_buyer_aware_dynamic_pricing,
    obj(vec![
        ("horizon", num(Some(1.0), Some(true), Some(12.0))),
        ("initialInventory", num(Some(1e-9), None, Some(160.0))),
        ("privacyBudget", num(Some(0.0), None, Some(0.0))),
        ("fairnessTolerance", num(Some(0.0), None, Some(0.18))),
        ("sustainabilityWeight", num(Some(0.0), None, Some(120.0))),
    ])
);

domain_adapter!(
    EnergyStorageDispatchAdapter,
    adapter_energy_storage_dispatch,
    EnergyParams,
    EnergyResult,
    "energy-storage-dispatch",
    "ENERGY STORAGE DISPATCH",
    "Energy optimization: storage dispatch candidates for renewable integration and price arbitrage.",
    run_energy_storage_dispatch,
    obj(vec![("batteryCapacity", num(Some(1e-9), None, Some(50.0)))])
);

domain_adapter!(
    ActiveLearningAcquisitionAdapter,
    adapter_active_learning_acquisition,
    ActiveLearningParams,
    ActiveLearningResult,
    "active-learning-acquisition",
    "ACTIVE LEARNING ACQUISITION",
    "Machine/statistical learning: active-learning acquisition policies over unlabeled data movables.",
    run_active_learning_acquisition,
    obj(vec![("budget", num(Some(1e-9), None, Some(9.0)))])
);

domain_adapter!(
    VisualDecisionFrontierAdapter,
    adapter_visual_decision_frontier,
    DecisionScienceParams,
    DecisionScienceResult,
    "visual-decision-frontier",
    "VISUAL DECISION FRONTIER",
    "Decision science: MCDA frontier scoring with visualization-ready alternatives and weights.",
    run_visual_decision_frontier,
    obj(vec![("riskWeight", num(Some(0.0), None, Some(0.35)))])
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the formerly-`unimplemented!()` fuzzy-control kernel: it
    /// now drives the real engine, producing evaluated candidates with metrics
    /// and a serialized (camelCase) plan — not a panic.
    #[test]
    fn fuzzy_control_adapter_runs_and_serializes_plan() {
        let adapter = adapter_adaptive_fuzzy_control();
        let result = adapter.run(
            AdaptiveFuzzyControlParams::default(),
            &DESRuntimeConfig::default(),
        );
        assert!(!result.candidates.is_empty(), "engine produced candidates");
        assert!(!result.best.metrics.is_empty(), "best has metrics");
        // The erased plan must be a non-null JSON object with camelCase keys
        // (serde `rename_all`), proving real plan serialization.
        match &result.best.plan {
            JsonValue::Object(o) => {
                assert!(o.contains_key("errorGain"), "camelCase plan key present");
            }
            other => panic!("expected object plan, got {other:?}"),
        }
        // CSV emission exercises the plan stringify path end-to-end.
        let summary = domain_summary("ADAPTIVE FUZZY CONTROL", &result);
        assert!(summary.contains("Candidates:"));
    }

    /// The routing kernel exercises a nested plan (enum + Vec<Vec<usize>>): the
    /// `RoutingHeuristic` serializes to its kebab-case wire name.
    #[test]
    fn routing_adapter_serializes_nested_plan() {
        let adapter = adapter_logistics_routing_heuristics();
        let result = adapter.run(
            LogisticsRoutingParams::default(),
            &DESRuntimeConfig::default(),
        );
        assert!(!result.candidates.is_empty());
        match &result.best.plan {
            JsonValue::Object(o) => {
                assert!(o.contains_key("heuristic") && o.contains_key("routes"));
            }
            other => panic!("expected object plan, got {other:?}"),
        }
    }
}
