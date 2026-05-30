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
//! PORT NOTE: the entire `des/general/domain-application-models` engine module
//! is NOT yet ported. The result/candidate/trace/metric types below are MINIMAL
//! local stubs, and every `run_*` kernel is a `unimplemented!()` placeholder.
//! When the engine is ported, replace these stubs with the real types/functions
//! and delete the placeholders. The adapter glue (schemas, summaries, CSV,
//! examples) is faithful and should carry over unchanged.
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

// Per-model parameter stubs. Field shapes live in each adapter's `schema()`;
// the integrator should replace these with the real engine structs.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveFuzzyControlParams {}
#[derive(Clone, Debug, Default)]
pub struct LogisticsRoutingParams {}
#[derive(Clone, Debug, Default)]
pub struct ManufacturingParams {}
#[derive(Clone, Debug, Default)]
pub struct SupplyChainParams {}
#[derive(Clone, Debug, Default)]
pub struct OperationsParams {}
#[derive(Clone, Debug, Default)]
pub struct FinancialControlParams {}
#[derive(Clone, Debug, Default)]
pub struct RevenueManagementParams {}
#[derive(Clone, Debug, Default)]
pub struct BuyerAwareDynamicPricingParams {}
#[derive(Clone, Debug, Default)]
pub struct EnergyParams {}
#[derive(Clone, Debug, Default)]
pub struct ActiveLearningParams {}
#[derive(Clone, Debug, Default)]
pub struct DecisionScienceParams {}

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

const ENGINE_MISSING: &str =
    "domain-application-models engine is not ported yet (see module PORT NOTE)";

pub fn run_adaptive_fuzzy_control(_params: AdaptiveFuzzyControlParams) -> AdaptiveFuzzyControlResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_logistics_routing_heuristics(_params: LogisticsRoutingParams) -> LogisticsRoutingResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_bottleneck_production_control(_params: ManufacturingParams) -> ManufacturingResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_supply_chain_risk_pooling(_params: SupplyChainParams) -> SupplyChainResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_workforce_service_operations(_params: OperationsParams) -> OperationsResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_portfolio_drawdown_control(_params: FinancialControlParams) -> FinancialControlResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_dynamic_pricing_revenue(_params: RevenueManagementParams) -> RevenueManagementResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_buyer_aware_dynamic_pricing(
    _params: BuyerAwareDynamicPricingParams,
) -> BuyerAwareDynamicPricingResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_energy_storage_dispatch(_params: EnergyParams) -> EnergyResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_active_learning_acquisition(_params: ActiveLearningParams) -> ActiveLearningResult {
    unimplemented!("{ENGINE_MISSING}")
}
pub fn run_visual_decision_frontier(_params: DecisionScienceParams) -> DecisionScienceResult {
    unimplemented!("{ENGINE_MISSING}")
}

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
                .map(|k| format!("{}:{}", json_quote(k), json_value_string(obj.get(k).unwrap())))
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
        format!("  Stations:       {}", result.topology.stations.join(" -> ")),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

fn write_domain_csv(result: &DomainModelResult, csv_path: &str) {
    let mut lines = vec![csv_row(["candidate_id", "objective", "feasible", "metrics", "plan"])];
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
    ParamSchema::Number { min, max: None, integer, default, description: None }
}

fn obj(fields: Vec<(&str, ParamSchema)>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
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
