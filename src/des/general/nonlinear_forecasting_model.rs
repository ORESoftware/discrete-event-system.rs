//! Port of `src/des/general/nonlinear-forecasting-model.ts` — module
//! `des::general::nonlinear_forecasting_model`.
//!
//! Nonlinear prediction / forecasting expressed as an explicit DES station
//! graph:
//!
//! `ForecastDataSource -> POMDPLatentVariable -> MDPVariableDiscovery ->
//! NonlinearEquationTuning -> ForecastProjection -> ResultSink`.
//!
//! The POMDP station infers hidden regime beliefs from noisy observations. The
//! MDP station treats candidate variables as discovery actions and uses value
//! iteration to decide which observed, nonlinear, lagged, and latent-belief
//! variables are worth adding. The equation station fine-tunes a nonlinear
//! basis expansion (ridge-regularised normal equations), and the projection
//! station rolls it forward for forecasting with a belief-modulated band.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * String unions `RegimeId` / `RegimeObservation` / `VariableSource` become
//!     [`RegimeId`] / [`RegimeObservation`] / [`VariableSource`] enums.
//!   * Tokens (`class X implements Token`) become plain structs flowed as
//!     `Rc<dyn Any>`; downstream stations downcast with `drain::<T>()`. The
//!     scenario (which owns non-`Clone` boxed closures in its `POMDPSpec` and
//!     feature candidates) is shared via `Rc<ForecastScenario>` instead of being
//!     copied into each token.
//!   * `*Station extends DESStation` become structs embedding a [`StationCore`]
//!     and implementing the [`DESStation`] trait.
//!   * Feature masks packed in a JS `number` become a `u32` bitset using
//!     `count_ones()` and bit iteration; the `evalCache: Map<number, ...>`
//!     becomes an `Rc<RefCell<HashMap<u32, FitEvaluation>>>` so the (`Fn`,
//!     not `FnMut`) reward closure handed to the MDP solver can memoise through
//!     interior mutability.
//!   * `Preconditions.*` throw in TS -> here they return `Result`; the
//!     `require` helper turns a failed guard into a `panic!` (invariant).
//!   * The synthetic series is deterministic, so no RNG injection is needed.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::general::belief::DiscreteBelief;
use crate::des::general::des_base::learning_optimization::{
    channel_edge, station_graph, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::pomdp::{belief_update, mdp_value_iteration, MDPVIOptions, POMDPSpec};

/// Panic on a failed precondition (the TS guards `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Enums (string unions)
// =============================================================================

/// Hidden macro regime (`'baseline' | 'expansion' | 'contraction' | 'shock'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeId {
    Baseline,
    Expansion,
    Contraction,
    Shock,
}

impl RegimeId {
    pub fn as_str(self) -> &'static str {
        match self {
            RegimeId::Baseline => "baseline",
            RegimeId::Expansion => "expansion",
            RegimeId::Contraction => "contraction",
            RegimeId::Shock => "shock",
        }
    }
}

/// Discretised regime observation (`'low' | 'flat' | 'high' | 'volatile'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeObservation {
    Low,
    Flat,
    High,
    Volatile,
}

impl RegimeObservation {
    pub fn as_str(self) -> &'static str {
        match self {
            RegimeObservation::Low => "low",
            RegimeObservation::Flat => "flat",
            RegimeObservation::High => "high",
            RegimeObservation::Volatile => "volatile",
        }
    }
}

/// Provenance of a discovered feature (`'observed' | 'lagged' | 'nonlinear' | 'pomdp'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableSource {
    Observed,
    Lagged,
    Nonlinear,
    Pomdp,
}

impl VariableSource {
    pub fn as_str(self) -> &'static str {
        match self {
            VariableSource::Observed => "observed",
            VariableSource::Lagged => "lagged",
            VariableSource::Nonlinear => "nonlinear",
            VariableSource::Pomdp => "pomdp",
        }
    }
}

/// `REGIMES` — canonical order parallel to the transition/observation matrices.
fn regimes() -> Vec<RegimeId> {
    vec![RegimeId::Baseline, RegimeId::Expansion, RegimeId::Contraction, RegimeId::Shock]
}

/// `REGIME_OBSERVATIONS` — canonical observation order.
fn regime_observations() -> Vec<RegimeObservation> {
    vec![RegimeObservation::Low, RegimeObservation::Flat, RegimeObservation::High, RegimeObservation::Volatile]
}

/// Index of an observation in [`regime_observations`] (TS `indexOf`).
fn regime_observation_index(obs: RegimeObservation) -> usize {
    match obs {
        RegimeObservation::Low => 0,
        RegimeObservation::Flat => 1,
        RegimeObservation::High => 2,
        RegimeObservation::Volatile => 3,
    }
}

const CH_DATA: &str = "forecast-data";
const CH_BELIEF: &str = "latent-belief-trace";
const CH_VARIABLES: &str = "discovered-variables";
const CH_EQUATION: &str = "fine-tuned-equation";
const CH_PROJECTION: &str = "forecast-projection";

// =============================================================================
// Parameter / data shapes
// =============================================================================

/// Public knobs (all optional; defaults applied in [`normalize_params`]).
#[derive(Clone, Debug, Default)]
pub struct NonlinearMDPPOMDPForecastParams {
    pub training_periods: Option<usize>,
    pub forecast_horizon: Option<usize>,
    pub mdp_budget: Option<usize>,
    pub ridge: Option<f64>,
    pub fine_tune_iterations: Option<usize>,
    pub validation_share: Option<f64>,
}

#[derive(Clone, Debug)]
struct NormalizedForecastParams {
    training_periods: usize,
    forecast_horizon: usize,
    mdp_budget: usize,
    ridge: f64,
    fine_tune_iterations: usize,
    validation_share: f64,
}

#[derive(Clone, Copy, Debug)]
struct ForecastObservation {
    t: usize,
    demand: f64,
    supply: f64,
    price: f64,
    y: f64,
    hidden_regime: RegimeId,
}

/// The whole problem instance. Holds non-`Clone` boxed closures, hence shared
/// via `Rc` rather than copied.
struct ForecastScenario {
    params: NormalizedForecastParams,
    observations: Vec<ForecastObservation>,
    feature_candidates: Vec<FeatureCandidate>,
    pomdp_spec: POMDPSpec<RegimeId, String, RegimeObservation>,
}

#[derive(Clone, Copy, Debug)]
struct FeatureContext {
    t: f64,
    demand: f64,
    supply_gap: f64,
    price: f64,
    lag_y: f64,
    momentum: f64,
    trend: f64,
    belief_baseline: f64,
    belief_expansion: f64,
    belief_contraction: f64,
    belief_shock: f64,
}

/// A candidate explanatory variable: metadata plus a pure feature extractor.
struct FeatureCandidate {
    id: String,
    label: String,
    source: VariableSource,
    cost: f64,
    compute: Box<dyn Fn(&FeatureContext) -> f64>,
}

/// Train / validation split tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Validation,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ForecastRow {
    t: usize,
    target: f64,
    context: FeatureContext,
    split: Split,
}

// =============================================================================
// Public result shapes
// =============================================================================

/// One step of the latent-regime belief trace.
#[derive(Clone, Debug)]
pub struct LatentBeliefPoint {
    pub t: usize,
    pub observation: RegimeObservation,
    pub prior: Vec<f64>,
    pub posterior: Vec<f64>,
    pub mode: RegimeId,
    pub entropy: f64,
}

/// The full latent-regime belief trace.
#[derive(Clone, Debug)]
pub struct LatentBeliefTrace {
    pub states: Vec<RegimeId>,
    pub points: Vec<LatentBeliefPoint>,
    pub final_belief: Vec<f64>,
    pub transition_matrix: Vec<Vec<f64>>,
}

/// One action of the MDP variable-discovery search.
#[derive(Clone, Debug)]
pub struct MDPDiscoveryStep {
    pub step: usize,
    pub state_mask: u32,
    pub action: String,
    pub reward: f64,
    pub validation_mse_before: f64,
    pub validation_mse_after: f64,
    pub selected_after: Vec<String>,
}

/// A discovered variable's metadata (the TS inline `{id, label, source, cost}`).
#[derive(Clone, Debug)]
pub struct SelectedVariable {
    pub id: String,
    pub label: String,
    pub source: VariableSource,
    pub cost: f64,
}

/// Outcome of the MDP variable-discovery station.
#[derive(Clone, Debug)]
pub struct VariableDiscoveryResult {
    pub selected_feature_indices: Vec<usize>,
    pub selected_variables: Vec<SelectedVariable>,
    pub rows: Vec<ForecastRow>,
    pub train_rows: Vec<ForecastRow>,
    pub validation_rows: Vec<ForecastRow>,
    pub baseline_validation_mse: f64,
    pub validation_mse: f64,
    pub train_mse: f64,
    pub mdp_states: usize,
    pub mdp_actions: usize,
    pub mdp_iterations: usize,
    pub mdp_final_delta: f64,
    pub action_trace: Vec<MDPDiscoveryStep>,
}

/// One row of the fine-tuning trace.
#[derive(Clone, Debug)]
pub struct FineTuneTraceRow {
    pub iter: usize,
    pub mse: f64,
    pub validation_mse: f64,
    pub coefficients: Vec<f64>,
}

/// A fitted-equation diagnostic point.
#[derive(Clone, Debug)]
pub struct FittedPoint {
    pub t: usize,
    pub actual: f64,
    pub predicted: f64,
    pub split: Split,
}

/// The tuned nonlinear equation.
#[derive(Clone, Debug)]
pub struct TunedEquation {
    pub feature_indices: Vec<usize>,
    pub feature_ids: Vec<String>,
    pub feature_labels: Vec<String>,
    pub coefficients: Vec<f64>,
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    pub intercept: f64,
    pub equation_text: String,
    pub in_sample_mse: f64,
    pub validation_mse: f64,
    pub trace: Vec<FineTuneTraceRow>,
    pub fitted: Vec<FittedPoint>,
}

/// One projected forecast period.
#[derive(Clone, Debug)]
pub struct ForecastProjectionPoint {
    pub t: usize,
    pub horizon_step: usize,
    pub forecast: f64,
    pub actual: f64,
    pub lower: f64,
    pub upper: f64,
    pub belief_mode: RegimeId,
    pub belief_entropy: f64,
}

/// MDP search summary embedded in the final result.
#[derive(Clone, Debug)]
pub struct MdpSummary {
    pub states: usize,
    pub actions: usize,
    pub iterations: usize,
    pub final_delta: f64,
    pub action_trace: Vec<MDPDiscoveryStep>,
}

/// Aggregate forecast metrics.
#[derive(Clone, Debug)]
pub struct ForecastMetrics {
    pub baseline_validation_mse: f64,
    pub validation_mse: f64,
    pub train_mse: f64,
    pub in_sample_mse: f64,
    pub forecast_mse: f64,
    pub baseline_forecast_mse: f64,
    pub final_belief_entropy: f64,
    pub selected_variable_count: usize,
}

/// The top-level result of [`run_nonlinear_mdp_pomdp_forecast`].
#[derive(Clone, Debug)]
pub struct NonlinearMDPPOMDPForecastResult {
    pub model_id: &'static str,
    pub selected_variables: Vec<String>,
    pub discovered_variables: Vec<SelectedVariable>,
    pub equation: TunedEquation,
    pub pomdp: LatentBeliefTrace,
    pub mdp: MdpSummary,
    pub projection: Vec<ForecastProjectionPoint>,
    pub metrics: ForecastMetrics,
    pub topology: StationGraphSummary,
}

// =============================================================================
// Tokens
// =============================================================================

struct ForecastDataToken {
    scenario: Rc<ForecastScenario>,
}

struct LatentBeliefTraceToken {
    scenario: Rc<ForecastScenario>,
    belief_trace: LatentBeliefTrace,
}

struct DiscoveredVariablesToken {
    scenario: Rc<ForecastScenario>,
    belief_trace: LatentBeliefTrace,
    discovery: VariableDiscoveryResult,
}

struct FineTunedEquationToken {
    scenario: Rc<ForecastScenario>,
    belief_trace: LatentBeliefTrace,
    discovery: VariableDiscoveryResult,
    equation: TunedEquation,
}

struct ForecastProjectionToken {
    result: NonlinearMDPPOMDPForecastResult,
}

// =============================================================================
// Stations
// =============================================================================

struct ForecastDataSourceStation {
    core: StationCore,
    scenario: Rc<ForecastScenario>,
    emitted: bool,
}

impl ForecastDataSourceStation {
    fn new(id: &str, scenario: Rc<ForecastScenario>) -> Self {
        ForecastDataSourceStation { core: StationCore::new(id), scenario, emitted: false }
    }
}

impl DESStation for ForecastDataSourceStation {
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
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        self.core.emit(Rc::new(ForecastDataToken { scenario: self.scenario.clone() }), CH_DATA);
        self.emitted = true;
    }
}

struct POMDPLatentVariableStation {
    core: StationCore,
}

impl POMDPLatentVariableStation {
    fn new(id: &str) -> Self {
        POMDPLatentVariableStation { core: StationCore::new(id) }
    }
}

impl DESStation for POMDPLatentVariableStation {
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
        self.core.inbox_size(CH_DATA) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<ForecastDataToken>(CH_DATA) {
            let belief_trace = infer_latent_regime_beliefs(token.scenario.as_ref());
            self.core.emit(
                Rc::new(LatentBeliefTraceToken { scenario: token.scenario.clone(), belief_trace }),
                CH_BELIEF,
            );
        }
    }
}

struct MDPVariableDiscoveryStation {
    core: StationCore,
}

impl MDPVariableDiscoveryStation {
    fn new(id: &str) -> Self {
        MDPVariableDiscoveryStation { core: StationCore::new(id) }
    }
}

impl DESStation for MDPVariableDiscoveryStation {
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
        self.core.inbox_size(CH_BELIEF) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<LatentBeliefTraceToken>(CH_BELIEF) {
            let discovery = discover_variables_by_mdp(&token.scenario, &token.belief_trace);
            self.core.emit(
                Rc::new(DiscoveredVariablesToken {
                    scenario: token.scenario.clone(),
                    belief_trace: token.belief_trace.clone(),
                    discovery,
                }),
                CH_VARIABLES,
            );
        }
    }
}

struct NonlinearEquationTuningStation {
    core: StationCore,
}

impl NonlinearEquationTuningStation {
    fn new(id: &str) -> Self {
        NonlinearEquationTuningStation { core: StationCore::new(id) }
    }
}

impl DESStation for NonlinearEquationTuningStation {
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
        self.core.inbox_size(CH_VARIABLES) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<DiscoveredVariablesToken>(CH_VARIABLES) {
            let equation = fine_tune_equation(token.scenario.as_ref(), &token.discovery);
            self.core.emit(
                Rc::new(FineTunedEquationToken {
                    scenario: token.scenario.clone(),
                    belief_trace: token.belief_trace.clone(),
                    discovery: token.discovery.clone(),
                    equation,
                }),
                CH_EQUATION,
            );
        }
    }
}

struct ForecastProjectionStation {
    core: StationCore,
}

impl ForecastProjectionStation {
    fn new(id: &str) -> Self {
        ForecastProjectionStation { core: StationCore::new(id) }
    }
}

impl DESStation for ForecastProjectionStation {
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
        self.core.inbox_size(CH_EQUATION) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<FineTunedEquationToken>(CH_EQUATION) {
            let scenario = token.scenario.as_ref();
            let projection = project_forecast(scenario, &token.belief_trace, &token.equation);
            let actual: Vec<f64> = projection.iter().map(|row| row.actual).collect();
            let predicted: Vec<f64> = projection.iter().map(|row| row.forecast).collect();
            let last_training_y = scenario.observations[scenario.params.training_periods - 1].y;
            let baseline_forecast_mse = mse(&actual, &vec![last_training_y; actual.len()]);
            let result = NonlinearMDPPOMDPForecastResult {
                model_id: "nonlinear-mdp-pomdp-forecast",
                selected_variables: token.discovery.selected_variables.iter().map(|v| v.id.clone()).collect(),
                discovered_variables: token.discovery.selected_variables.clone(),
                equation: token.equation.clone(),
                pomdp: token.belief_trace.clone(),
                mdp: MdpSummary {
                    states: token.discovery.mdp_states,
                    actions: token.discovery.mdp_actions,
                    iterations: token.discovery.mdp_iterations,
                    final_delta: token.discovery.mdp_final_delta,
                    action_trace: token.discovery.action_trace.clone(),
                },
                metrics: ForecastMetrics {
                    baseline_validation_mse: token.discovery.baseline_validation_mse,
                    validation_mse: token.discovery.validation_mse,
                    train_mse: token.discovery.train_mse,
                    in_sample_mse: token.equation.in_sample_mse,
                    forecast_mse: mse(&actual, &predicted),
                    baseline_forecast_mse,
                    final_belief_entropy: entropy(&token.belief_trace.final_belief),
                    selected_variable_count: token.discovery.selected_variables.len(),
                },
                projection,
                topology: station_graph(&[], &[], &[]),
            };
            self.core.emit(Rc::new(ForecastProjectionToken { result }), CH_PROJECTION);
        }
    }
}

struct ForecastResultSinkStation {
    core: StationCore,
    result: Option<NonlinearMDPPOMDPForecastResult>,
}

impl ForecastResultSinkStation {
    fn new(id: &str) -> Self {
        ForecastResultSinkStation { core: StationCore::new(id), result: None }
    }
}

impl DESStation for ForecastResultSinkStation {
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
        self.core.inbox_size(CH_PROJECTION) > 0
    }
    fn run_time_step(&mut self) {
        let tokens = self.core.drain::<ForecastProjectionToken>(CH_PROJECTION);
        if let Some(last) = tokens.last() {
            self.result = Some(last.result.clone());
        }
    }
}

// =============================================================================
// Entry point
// =============================================================================

/// Run the full nonlinear MDP/POMDP forecasting pipeline.
pub fn run_nonlinear_mdp_pomdp_forecast(
    params: NonlinearMDPPOMDPForecastParams,
) -> NonlinearMDPPOMDPForecastResult {
    let scenario = Rc::new(build_forecast_scenario(&params));
    let source = Rc::new(RefCell::new(ForecastDataSourceStation::new("nonlinear-forecast-data-source", scenario.clone())));
    let pomdp = Rc::new(RefCell::new(POMDPLatentVariableStation::new("pomdp-latent-variable-station")));
    let mdp = Rc::new(RefCell::new(MDPVariableDiscoveryStation::new("mdp-variable-discovery-station")));
    let tuning = Rc::new(RefCell::new(NonlinearEquationTuningStation::new("nonlinear-equation-tuning-station")));
    let projection = Rc::new(RefCell::new(ForecastProjectionStation::new("forecast-projection-station")));
    let sink = Rc::new(RefCell::new(ForecastResultSinkStation::new("forecast-result-sink")));

    source.borrow_mut().core_mut().pipe(pomdp.clone() as StationRef, CH_DATA, CH_DATA);
    pomdp.borrow_mut().core_mut().pipe(mdp.clone() as StationRef, CH_BELIEF, CH_BELIEF);
    mdp.borrow_mut().core_mut().pipe(tuning.clone() as StationRef, CH_VARIABLES, CH_VARIABLES);
    tuning.borrow_mut().core_mut().pipe(projection.clone() as StationRef, CH_EQUATION, CH_EQUATION);
    projection.borrow_mut().core_mut().pipe(sink.clone() as StationRef, CH_PROJECTION, CH_PROJECTION);

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            pomdp.clone() as StationRef,
            mdp.clone() as StationRef,
            tuning.clone() as StationRef,
            projection.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions { shuffle: false, max_ticks: Some(12), run_validators: false, ..Default::default() },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("nonlinear-mdp-pomdp-forecast did not produce a result"));

    let s = StationOrId::Id("nonlinear-forecast-data-source".to_string());
    let p = StationOrId::Id("pomdp-latent-variable-station".to_string());
    let m = StationOrId::Id("mdp-variable-discovery-station".to_string());
    let tu = StationOrId::Id("nonlinear-equation-tuning-station".to_string());
    let pr = StationOrId::Id("forecast-projection-station".to_string());
    let si = StationOrId::Id("forecast-result-sink".to_string());
    let movables: Vec<String> = vec![
        "ForecastDataToken".to_string(),
        "LatentBeliefTraceToken".to_string(),
        "DiscoveredVariablesToken".to_string(),
        "FineTunedEquationToken".to_string(),
        "ForecastProjectionToken".to_string(),
    ];
    let edges = vec![
        channel_edge(&s, CH_DATA, &p, Some(CH_DATA)),
        channel_edge(&p, CH_BELIEF, &m, Some(CH_BELIEF)),
        channel_edge(&m, CH_VARIABLES, &tu, Some(CH_VARIABLES)),
        channel_edge(&tu, CH_EQUATION, &pr, Some(CH_EQUATION)),
        channel_edge(&pr, CH_PROJECTION, &si, Some(CH_PROJECTION)),
    ];
    result.topology = station_graph(&[s, p, m, tu, pr, si], &movables, &edges);
    result
}

// =============================================================================
// Scenario construction
// =============================================================================

fn build_forecast_scenario(params: &NonlinearMDPPOMDPForecastParams) -> ForecastScenario {
    let actual = normalize_params(params);
    let observations = synthetic_forecast_series(actual.training_periods, actual.forecast_horizon);
    ForecastScenario {
        params: actual,
        observations,
        feature_candidates: feature_candidates(),
        pomdp_spec: build_regime_pomdp(),
    }
}

fn normalize_params(params: &NonlinearMDPPOMDPForecastParams) -> NormalizedForecastParams {
    let actual = NormalizedForecastParams {
        training_periods: params.training_periods.unwrap_or(42),
        forecast_horizon: params.forecast_horizon.unwrap_or(8),
        mdp_budget: params.mdp_budget.unwrap_or(6),
        ridge: params.ridge.unwrap_or(0.03),
        fine_tune_iterations: params.fine_tune_iterations.unwrap_or(18),
        validation_share: params.validation_share.unwrap_or(0.25),
    };
    require(Preconditions::integer_in_range(
        "runNonlinearMDPPOMDPForecast",
        "trainingPeriods",
        actual.training_periods as f64,
        18.0,
        200.0,
    ));
    require(Preconditions::integer_in_range(
        "runNonlinearMDPPOMDPForecast",
        "forecastHorizon",
        actual.forecast_horizon as f64,
        1.0,
        80.0,
    ));
    require(Preconditions::integer_in_range(
        "runNonlinearMDPPOMDPForecast",
        "mdpBudget",
        actual.mdp_budget as f64,
        1.0,
        10.0,
    ));
    require(Preconditions::non_negative("runNonlinearMDPPOMDPForecast", "ridge", actual.ridge));
    require(Preconditions::integer_in_range(
        "runNonlinearMDPPOMDPForecast",
        "fineTuneIterations",
        actual.fine_tune_iterations as f64,
        1.0,
        200.0,
    ));
    require(Preconditions::in_range(
        "runNonlinearMDPPOMDPForecast",
        "validationShare",
        actual.validation_share,
        0.1,
        0.5,
    ));
    actual
}

fn synthetic_forecast_series(training_periods: usize, forecast_horizon: usize) -> Vec<ForecastObservation> {
    let total = training_periods + forecast_horizon;
    let mut out: Vec<ForecastObservation> = Vec::with_capacity(total);
    for t in 0..total {
        let hidden_regime = hidden_regime_at(t, training_periods);
        let shock = if hidden_regime == RegimeId::Shock { 1.0 } else { 0.0 };
        let contraction = if hidden_regime == RegimeId::Contraction { 1.0 } else { 0.0 };
        let expansion = if hidden_regime == RegimeId::Expansion { 1.0 } else { 0.0 };
        let tf = t as f64;
        let demand = 1.18 + 0.018 * tf + 0.22 * (tf / 3.2).sin() + 0.12 * expansion - 0.13 * shock;
        let supply = 1.02 + 0.17 * (tf / 4.6).cos() + 0.04 * expansion - 0.11 * contraction - 0.23 * shock;
        let price = 1.00 + 0.07 * (tf / 5.1).sin() + (0.94 - supply).max(0.0) * 0.24 + 0.10 * shock;
        let regime_lift = match hidden_regime {
            RegimeId::Expansion => 5.8,
            RegimeId::Contraction => -5.2,
            RegimeId::Shock => -11.5,
            RegimeId::Baseline => 0.0,
        };
        let deterministic_noise = 1.1 * (1.7 * tf).sin() + 0.55 * (0.61 * tf).cos();
        let y = if t == 0 {
            56.0 + 8.0 * (0.9 * demand - 0.65 * price).tanh() + regime_lift + deterministic_noise
        } else {
            let lag_y = out[t - 1].y;
            17.5 + 0.64 * lag_y
                + 14.5 * (0.92 * demand - 0.68 * price).tanh()
                + 6.4 * demand * (1.12 - supply)
                + 2.2 * (tf / 5.7).sin()
                + regime_lift
                + deterministic_noise
        };
        out.push(ForecastObservation { t, demand, supply, price, y, hidden_regime });
    }
    out
}

fn hidden_regime_at(t: usize, training_periods: usize) -> RegimeId {
    let tp = training_periods as f64;
    if t < (tp * 0.28).floor() as usize {
        return RegimeId::Baseline;
    }
    if t < (tp * 0.52).floor() as usize {
        return RegimeId::Expansion;
    }
    if t < (tp * 0.64).floor() as usize {
        return RegimeId::Shock;
    }
    if t < training_periods {
        return RegimeId::Contraction;
    }
    if t < training_periods + 3 {
        RegimeId::Baseline
    } else {
        RegimeId::Expansion
    }
}

fn build_regime_pomdp() -> POMDPSpec<RegimeId, String, RegimeObservation> {
    let transition_matrix = regime_transition_matrix();
    let regs = regimes();
    POMDPSpec {
        states: regimes(),
        actions: vec!["observe".to_string()],
        observations: regime_observations(),
        transition: Box::new(move |s_idx, _a_idx| transition_matrix[s_idx].clone()),
        observation: Box::new(move |s_next_idx, _a_idx| observation_likelihood(regs[s_next_idx])),
        reward: Box::new(|_s_idx, _a_idx| -0.01),
        discount: 0.94,
        initial_belief: Some(vec![0.64, 0.14, 0.14, 0.08]),
        is_terminal: None,
    }
}

fn regime_transition_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![0.72, 0.16, 0.08, 0.04],
        vec![0.18, 0.70, 0.04, 0.08],
        vec![0.24, 0.04, 0.66, 0.06],
        vec![0.20, 0.10, 0.25, 0.45],
    ]
}

fn observation_likelihood(regime: RegimeId) -> Vec<f64> {
    match regime {
        RegimeId::Baseline => vec![0.17, 0.56, 0.18, 0.09],
        RegimeId::Expansion => vec![0.07, 0.20, 0.62, 0.11],
        RegimeId::Contraction => vec![0.60, 0.21, 0.07, 0.12],
        RegimeId::Shock => vec![0.25, 0.10, 0.10, 0.55],
    }
}

// =============================================================================
// POMDP latent regime inference
// =============================================================================

fn infer_latent_regime_beliefs(scenario: &ForecastScenario) -> LatentBeliefTrace {
    let spec = &scenario.pomdp_spec;
    let mut posterior = DiscreteBelief::new(spec.states.clone(), spec.initial_belief.as_deref());
    let mut points: Vec<LatentBeliefPoint> = vec![LatentBeliefPoint {
        t: 0,
        observation: RegimeObservation::Flat,
        prior: posterior.as_array(),
        posterior: posterior.as_array(),
        mode: posterior.mode(),
        entropy: posterior.entropy(),
    }];
    for t in 1..scenario.params.training_periods {
        let obs = classify_regime_observation(&scenario.observations[t - 1], &scenario.observations[t]);
        let mut prior = posterior.clone();
        prior.propagate(|_state, index| (spec.transition)(index, 0));
        posterior = belief_update(spec, &posterior, 0, regime_observation_index(obs));
        points.push(LatentBeliefPoint {
            t,
            observation: obs,
            prior: prior.as_array(),
            posterior: posterior.as_array(),
            mode: posterior.mode(),
            entropy: posterior.entropy(),
        });
    }
    LatentBeliefTrace {
        states: regimes(),
        points,
        final_belief: posterior.as_array(),
        transition_matrix: regime_transition_matrix(),
    }
}

fn classify_regime_observation(prev: &ForecastObservation, cur: &ForecastObservation) -> RegimeObservation {
    let dy = cur.y - prev.y;
    let expected = 8.0 * (cur.demand - prev.demand) - 5.0 * (cur.price - prev.price) + 3.0 * (cur.supply - prev.supply);
    let residual = dy - expected;
    if residual.abs() > 7.0 || (cur.supply - prev.supply).abs() > 0.18 {
        return RegimeObservation::Volatile;
    }
    if residual > 2.1 {
        return RegimeObservation::High;
    }
    if residual < -2.1 {
        return RegimeObservation::Low;
    }
    RegimeObservation::Flat
}

// =============================================================================
// Feature candidates
// =============================================================================

fn feature_candidates() -> Vec<FeatureCandidate> {
    vec![
        FeatureCandidate {
            id: "observed-demand-index".to_string(),
            label: "observed demand index".to_string(),
            source: VariableSource::Observed,
            cost: 0.25,
            compute: Box::new(|ctx| ctx.demand),
        },
        FeatureCandidate {
            id: "observed-supply-gap".to_string(),
            label: "observed supply gap".to_string(),
            source: VariableSource::Observed,
            cost: 0.22,
            compute: Box::new(|ctx| ctx.supply_gap),
        },
        FeatureCandidate {
            id: "observed-price-pressure".to_string(),
            label: "observed price pressure".to_string(),
            source: VariableSource::Observed,
            cost: 0.28,
            compute: Box::new(|ctx| ctx.price),
        },
        FeatureCandidate {
            id: "lagged-outcome".to_string(),
            label: "lagged outcome".to_string(),
            source: VariableSource::Lagged,
            cost: 0.30,
            compute: Box::new(|ctx| ctx.lag_y),
        },
        FeatureCandidate {
            id: "lagged-momentum".to_string(),
            label: "lagged momentum".to_string(),
            source: VariableSource::Lagged,
            cost: 0.34,
            compute: Box::new(|ctx| ctx.momentum),
        },
        FeatureCandidate {
            id: "nonlinear-demand-saturation".to_string(),
            label: "nonlinear demand saturation".to_string(),
            source: VariableSource::Nonlinear,
            cost: 0.38,
            compute: Box::new(|ctx| (0.92 * ctx.demand - 0.68 * ctx.price).tanh()),
        },
        FeatureCandidate {
            id: "nonlinear-demand-supply-coupling".to_string(),
            label: "nonlinear demand/supply coupling".to_string(),
            source: VariableSource::Nonlinear,
            cost: 0.42,
            compute: Box::new(|ctx| ctx.demand * ctx.supply_gap),
        },
        FeatureCandidate {
            id: "latent-expansion-belief".to_string(),
            label: "POMDP expansion belief".to_string(),
            source: VariableSource::Pomdp,
            cost: 0.18,
            compute: Box::new(|ctx| ctx.belief_expansion),
        },
        FeatureCandidate {
            id: "latent-contraction-belief".to_string(),
            label: "POMDP contraction belief".to_string(),
            source: VariableSource::Pomdp,
            cost: 0.20,
            compute: Box::new(|ctx| ctx.belief_contraction),
        },
        FeatureCandidate {
            id: "latent-shock-belief".to_string(),
            label: "POMDP shock belief".to_string(),
            source: VariableSource::Pomdp,
            cost: 0.20,
            compute: Box::new(|ctx| ctx.belief_shock),
        },
    ]
}

// =============================================================================
// MDP variable discovery
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct FitEvaluation {
    train_mse: f64,
    validation_mse: f64,
}

struct SplitRows {
    train_rows: Vec<ForecastRow>,
    validation_rows: Vec<ForecastRow>,
}

/// Next feature-mask after taking `action` from `mask`.
fn transition_to(mask: u32, action: usize, stop_action: usize, mdp_budget: usize) -> u32 {
    if action == stop_action {
        return mask;
    }
    if (mask.count_ones() as usize) >= mdp_budget {
        return mask;
    }
    if mask & (1u32 << action) != 0 {
        return mask;
    }
    mask | (1u32 << action)
}

/// Memoised fit evaluation for a feature mask.
fn evaluate_mask_cached(
    cache: &RefCell<HashMap<u32, FitEvaluation>>,
    mask: u32,
    num_features: usize,
    scenario: &ForecastScenario,
    train_rows: &[ForecastRow],
    validation_rows: &[ForecastRow],
) -> FitEvaluation {
    let cached = cache.borrow().get(&mask).copied();
    if let Some(v) = cached {
        return v;
    }
    let v = evaluate_feature_mask(&mask_to_indices(mask, num_features), scenario, train_rows, validation_rows);
    cache.borrow_mut().insert(mask, v);
    v
}

/// Immediate reward of selecting `action` from `mask`.
#[allow(clippy::too_many_arguments)]
fn reward_of(
    cache: &RefCell<HashMap<u32, FitEvaluation>>,
    mask: u32,
    action: usize,
    num_features: usize,
    scenario: &ForecastScenario,
    train_rows: &[ForecastRow],
    validation_rows: &[ForecastRow],
    stop_action: usize,
    mdp_budget: usize,
) -> f64 {
    if action == stop_action {
        return 0.0;
    }
    let next = transition_to(mask, action, stop_action, mdp_budget);
    if next == mask {
        return -5.0;
    }
    let before = evaluate_mask_cached(cache, mask, num_features, scenario, train_rows, validation_rows);
    let after = evaluate_mask_cached(cache, next, num_features, scenario, train_rows, validation_rows);
    let feature_cost = scenario.feature_candidates[action].cost * 0.55;
    let overfit_penalty = 0.05 * (after.validation_mse - after.train_mse).max(0.0);
    before.validation_mse - after.validation_mse - feature_cost - overfit_penalty
}

fn discover_variables_by_mdp(
    scenario: &Rc<ForecastScenario>,
    belief_trace: &LatentBeliefTrace,
) -> VariableDiscoveryResult {
    let rows = build_forecast_rows(scenario, belief_trace);
    let split = split_rows(&rows, scenario.params.validation_share);
    let num_features = scenario.feature_candidates.len();
    let num_states = 1usize << num_features;
    let mut actions: Vec<String> = scenario.feature_candidates.iter().map(|f| f.id.clone()).collect();
    actions.push("stop".to_string());
    let stop_action = actions.len() - 1;
    let mdp_budget = scenario.params.mdp_budget;

    let eval_cache: Rc<RefCell<HashMap<u32, FitEvaluation>>> = Rc::new(RefCell::new(HashMap::new()));

    let transition: Box<dyn Fn(usize, usize) -> Vec<f64>> = {
        Box::new(move |s_idx, a_idx| {
            let mut row = vec![0.0; num_states];
            let next = transition_to(s_idx as u32, a_idx, stop_action, mdp_budget);
            row[next as usize] = 1.0;
            row
        })
    };
    let reward: Box<dyn Fn(usize, usize) -> f64> = {
        let cache = Rc::clone(&eval_cache);
        let scen = Rc::clone(scenario);
        let train_rows = split.train_rows.clone();
        let validation_rows = split.validation_rows.clone();
        Box::new(move |s_idx, a_idx| {
            reward_of(
                &cache,
                s_idx as u32,
                a_idx,
                num_features,
                scen.as_ref(),
                &train_rows,
                &validation_rows,
                stop_action,
                mdp_budget,
            )
        })
    };

    let mdp_spec: POMDPSpec<usize, String, String> = POMDPSpec {
        states: (0..num_states).collect(),
        actions: actions.clone(),
        observations: vec!["none".to_string()],
        transition,
        observation: Box::new(|_s_next_idx, _a_idx| vec![1.0]),
        reward,
        discount: 0.92,
        initial_belief: None,
        is_terminal: None,
    };
    let vi = mdp_value_iteration(&mdp_spec, &MDPVIOptions { tol: 1e-7, max_iter: 250 });

    let mut mask: u32 = 0;
    let mut action_trace: Vec<MDPDiscoveryStep> = Vec::new();
    for step in 0..(mdp_budget + 2) {
        let action = vi.policy[mask as usize];
        let next = transition_to(mask, action, stop_action, mdp_budget);
        let before = evaluate_mask_cached(&eval_cache, mask, num_features, scenario, &split.train_rows, &split.validation_rows);
        let after = evaluate_mask_cached(&eval_cache, next, num_features, scenario, &split.train_rows, &split.validation_rows);
        action_trace.push(MDPDiscoveryStep {
            step,
            state_mask: mask,
            action: actions[action].clone(),
            reward: reward_of(
                &eval_cache,
                mask,
                action,
                num_features,
                scenario,
                &split.train_rows,
                &split.validation_rows,
                stop_action,
                mdp_budget,
            ),
            validation_mse_before: before.validation_mse,
            validation_mse_after: after.validation_mse,
            selected_after: mask_to_indices(next, num_features)
                .iter()
                .map(|&i| scenario.feature_candidates[i].id.clone())
                .collect(),
        });
        if action == stop_action || next == mask {
            break;
        }
        mask = next;
    }

    let selected = mask_to_indices(mask, num_features);
    let final_eval = evaluate_mask_cached(&eval_cache, mask, num_features, scenario, &split.train_rows, &split.validation_rows);
    let baseline = evaluate_mask_cached(&eval_cache, 0, num_features, scenario, &split.train_rows, &split.validation_rows);
    let selected_variables: Vec<SelectedVariable> = selected
        .iter()
        .map(|&i| {
            let f = &scenario.feature_candidates[i];
            SelectedVariable { id: f.id.clone(), label: f.label.clone(), source: f.source, cost: f.cost }
        })
        .collect();
    VariableDiscoveryResult {
        selected_feature_indices: selected,
        selected_variables,
        rows,
        train_rows: split.train_rows,
        validation_rows: split.validation_rows,
        baseline_validation_mse: baseline.validation_mse,
        validation_mse: final_eval.validation_mse,
        train_mse: final_eval.train_mse,
        mdp_states: num_states,
        mdp_actions: actions.len(),
        mdp_iterations: vi.iterations,
        mdp_final_delta: vi.final_delta,
        action_trace,
    }
}

fn build_forecast_rows(scenario: &ForecastScenario, belief_trace: &LatentBeliefTrace) -> Vec<ForecastRow> {
    let mut rows: Vec<ForecastRow> = Vec::new();
    let train_n = scenario.params.training_periods;
    let validation_count = (((train_n - 2) as f64) * scenario.params.validation_share).floor().max(2.0) as usize;
    let validation_start = train_n - validation_count;
    for t in 2..train_n {
        rows.push(ForecastRow {
            t,
            target: scenario.observations[t].y,
            context: feature_context_for_training(scenario, belief_trace, t),
            split: if t >= validation_start { Split::Validation } else { Split::Train },
        });
    }
    rows
}

fn feature_context_for_training(scenario: &ForecastScenario, belief_trace: &LatentBeliefTrace, t: usize) -> FeatureContext {
    let cur = &scenario.observations[t];
    let lag = scenario.observations[t - 1].y;
    let prev = scenario.observations[t - 2].y;
    let point = belief_trace
        .points
        .get(t)
        .unwrap_or_else(|| belief_trace.points.last().expect("belief trace has at least one point"));
    feature_context(cur, lag, prev, &point.prior, scenario.params.training_periods)
}

fn feature_context(
    obs: &ForecastObservation,
    lag_y: f64,
    prev_y: f64,
    belief: &[f64],
    training_periods: usize,
) -> FeatureContext {
    FeatureContext {
        t: obs.t as f64,
        demand: obs.demand,
        supply_gap: 1.12 - obs.supply,
        price: obs.price,
        lag_y,
        momentum: lag_y - prev_y,
        trend: obs.t as f64 / (training_periods as f64 - 1.0).max(1.0),
        belief_baseline: belief[0],
        belief_expansion: belief[1],
        belief_contraction: belief[2],
        belief_shock: belief[3],
    }
}

fn split_rows(rows: &[ForecastRow], validation_share: f64) -> SplitRows {
    let validation_count = ((rows.len() as f64) * validation_share).floor().max(2.0) as usize;
    let cut = rows.len().saturating_sub(validation_count);
    SplitRows {
        train_rows: rows[..cut].to_vec(),
        validation_rows: rows[cut..].to_vec(),
    }
}

fn evaluate_feature_mask(
    feature_indices: &[usize],
    scenario: &ForecastScenario,
    train_rows: &[ForecastRow],
    validation_rows: &[ForecastRow],
) -> FitEvaluation {
    let fit = ridge_fit(feature_indices, &scenario.feature_candidates, train_rows, scenario.params.ridge);
    FitEvaluation {
        train_mse: prediction_mse(&fit, &scenario.feature_candidates, train_rows),
        validation_mse: prediction_mse(&fit, &scenario.feature_candidates, validation_rows),
    }
}

// =============================================================================
// Ridge fit + fine-tuning
// =============================================================================

#[derive(Clone, Debug)]
struct RidgeFit {
    feature_indices: Vec<usize>,
    coefficients: Vec<f64>,
    means: Vec<f64>,
    scales: Vec<f64>,
}

fn fine_tune_equation(scenario: &ForecastScenario, discovery: &VariableDiscoveryResult) -> TunedEquation {
    let all_rows = &discovery.rows;
    let target = ridge_fit(&discovery.selected_feature_indices, &scenario.feature_candidates, all_rows, scenario.params.ridge);
    let mut start_coefficients = vec![mean(&all_rows.iter().map(|row| row.target).collect::<Vec<_>>())];
    start_coefficients.extend(std::iter::repeat_n(0.0, discovery.selected_feature_indices.len()));
    let mut trace: Vec<FineTuneTraceRow> = Vec::new();
    for iter in 0..=scenario.params.fine_tune_iterations {
        let alpha = if iter == scenario.params.fine_tune_iterations {
            1.0
        } else {
            1.0 - (-0.32 * iter as f64).exp()
        };
        let coeffs: Vec<f64> = target
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, &v)| start_coefficients[i] + alpha * (v - start_coefficients[i]))
            .collect();
        let iter_fit = RidgeFit {
            feature_indices: target.feature_indices.clone(),
            coefficients: coeffs.clone(),
            means: target.means.clone(),
            scales: target.scales.clone(),
        };
        trace.push(FineTuneTraceRow {
            iter,
            mse: prediction_mse(&iter_fit, &scenario.feature_candidates, all_rows),
            validation_mse: prediction_mse(&iter_fit, &scenario.feature_candidates, &discovery.validation_rows),
            coefficients: coeffs,
        });
    }
    let fitted: Vec<FittedPoint> = all_rows
        .iter()
        .map(|row| FittedPoint {
            t: row.t,
            actual: row.target,
            predicted: predict_with_fit(&target, &scenario.feature_candidates, &row.context),
            split: row.split,
        })
        .collect();
    let feature_ids: Vec<String> = discovery.selected_variables.iter().map(|v| v.id.clone()).collect();
    TunedEquation {
        feature_indices: discovery.selected_feature_indices.clone(),
        feature_ids: feature_ids.clone(),
        feature_labels: discovery.selected_variables.iter().map(|v| v.label.clone()).collect(),
        coefficients: target.coefficients[1..].to_vec(),
        means: target.means.clone(),
        scales: target.scales.clone(),
        intercept: target.coefficients[0],
        equation_text: equation_text(&feature_ids, &target.coefficients),
        in_sample_mse: prediction_mse(&target, &scenario.feature_candidates, all_rows),
        validation_mse: prediction_mse(&target, &scenario.feature_candidates, &discovery.validation_rows),
        trace,
        fitted,
    }
}

fn ridge_fit(
    feature_indices: &[usize],
    candidates: &[FeatureCandidate],
    rows: &[ForecastRow],
    ridge: f64,
) -> RidgeFit {
    let p = feature_indices.len();
    let raw: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| feature_indices.iter().map(|&i| (candidates[i].compute)(&row.context)).collect())
        .collect();
    let mut means = vec![0.0; p];
    let mut scales = vec![1.0; p];
    for j in 0..p {
        means[j] = mean(&raw.iter().map(|row| row[j]).collect::<Vec<_>>());
        let variance = mean(&raw.iter().map(|row| (row[j] - means[j]).powi(2)).collect::<Vec<_>>());
        scales[j] = variance.max(1e-10).sqrt();
    }
    let dim = p + 1;
    let mut a = vec![vec![0.0; dim]; dim];
    let mut b = vec![0.0; dim];
    for r in 0..rows.len() {
        let mut x = vec![1.0];
        for j in 0..p {
            x.push((raw[r][j] - means[j]) / scales[j]);
        }
        for i in 0..dim {
            b[i] += x[i] * rows[r].target;
            for j in 0..dim {
                a[i][j] += x[i] * x[j];
            }
        }
    }
    for i in 1..dim {
        a[i][i] += ridge;
    }
    RidgeFit {
        feature_indices: feature_indices.to_vec(),
        coefficients: solve_linear_system(&a, &b),
        means,
        scales,
    }
}

fn prediction_mse(fit: &RidgeFit, candidates: &[FeatureCandidate], rows: &[ForecastRow]) -> f64 {
    let actual: Vec<f64> = rows.iter().map(|row| row.target).collect();
    let predicted: Vec<f64> = rows.iter().map(|row| predict_with_fit(fit, candidates, &row.context)).collect();
    mse(&actual, &predicted)
}

fn predict_with_fit(fit: &RidgeFit, candidates: &[FeatureCandidate], context: &FeatureContext) -> f64 {
    let mut y = fit.coefficients[0];
    for j in 0..fit.feature_indices.len() {
        let raw = (candidates[fit.feature_indices[j]].compute)(context);
        y += fit.coefficients[j + 1] * (raw - fit.means[j]) / fit.scales[j];
    }
    y
}

// =============================================================================
// Projection
// =============================================================================

fn project_forecast(
    scenario: &ForecastScenario,
    belief_trace: &LatentBeliefTrace,
    equation: &TunedEquation,
) -> Vec<ForecastProjectionPoint> {
    let mut coefficients = vec![equation.intercept];
    coefficients.extend(equation.coefficients.iter().copied());
    let feature_fit = RidgeFit {
        feature_indices: equation.feature_indices.clone(),
        coefficients,
        means: equation.means.clone(),
        scales: equation.scales.clone(),
    };
    let mut out: Vec<ForecastProjectionPoint> = Vec::new();
    let mut belief = DiscreteBelief::new(regimes(), Some(&belief_trace.final_belief));
    let mut lag_y = scenario.observations[scenario.params.training_periods - 1].y;
    let mut prev_y = scenario.observations[scenario.params.training_periods - 2].y;
    let residual_scale = equation.in_sample_mse.max(1e-9).sqrt();
    for h in 1..=scenario.params.forecast_horizon {
        belief.propagate(|_state, index| (scenario.pomdp_spec.transition)(index, 0));
        let t = scenario.params.training_periods + h - 1;
        let obs = &scenario.observations[t];
        let belief_array = belief.as_array();
        let ctx = feature_context(obs, lag_y, prev_y, &belief_array, scenario.params.training_periods);
        let forecast = predict_with_fit(&feature_fit, &scenario.feature_candidates, &ctx);
        let band = residual_scale * (1.1 + 0.07 * h as f64 + 0.22 * belief.entropy());
        out.push(ForecastProjectionPoint {
            t,
            horizon_step: h,
            forecast,
            actual: obs.y,
            lower: forecast - 1.96 * band,
            upper: forecast + 1.96 * band,
            belief_mode: belief.mode(),
            belief_entropy: belief.entropy(),
        });
        prev_y = lag_y;
        lag_y = forecast;
    }
    out
}

// =============================================================================
// Numeric helpers
// =============================================================================

fn mask_to_indices(mask: u32, num_features: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for i in 0..num_features {
        if mask & (1u32 << i) != 0 {
            out.push(i);
        }
    }
    out
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / (xs.len().max(1) as f64)
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() {
        panic!("mse: length mismatch");
    }
    if actual.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..actual.len() {
        total += (actual[i] - predicted[i]).powi(2);
    }
    total / actual.len() as f64
}

fn entropy(weights: &[f64]) -> f64 {
    let mut h = 0.0;
    for &w in weights {
        if w > 0.0 {
            h -= w * w.ln();
        }
    }
    h
}

fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < 1e-10 {
            m[col][col] += 1e-8;
            pivot = col;
        }
        m.swap(col, pivot);
        let div = m[col][col];
        for c in col..=n {
            m[col][c] /= div;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            for c in col..=n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }
    m.iter().map(|row| row[n]).collect()
}

fn equation_text(feature_ids: &[String], coefficients: &[f64]) -> String {
    let mut terms = vec![format!("{:.3}", coefficients[0])];
    for i in 0..feature_ids.len() {
        let sign = if coefficients[i + 1] >= 0.0 { "+" } else { "-" };
        terms.push(format!("{} {:.3}*z({})", sign, coefficients[i + 1].abs(), feature_ids[i]));
    }
    format!("y_hat = {}", terms.join(" "))
}

#[cfg(test)]
mod tests {
    //! Unit tests for the deterministic numeric kernels. The full pipeline runs
    //! an MDP value iteration over 2^|features| states, which is exercised by the
    //! integration suite rather than here to keep unit tests fast.

    use super::*;

    #[test]
    fn mse_matches_manual_and_rejects_mismatch() {
        assert_eq!(mse(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!((mse(&[0.0, 0.0], &[1.0, 3.0]) - 5.0).abs() < 1e-12);
        assert_eq!(mse(&[], &[]), 0.0);
    }

    #[test]
    fn entropy_of_uniform_is_log_n() {
        let h = entropy(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - 4.0_f64.ln()).abs() < 1e-12);
        assert_eq!(entropy(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn mask_indices_and_transitions() {
        assert_eq!(mask_to_indices(0b0101, 4), vec![0, 2]);
        // stop action leaves the mask unchanged.
        assert_eq!(transition_to(0b0001, 3, 3, 4), 0b0001);
        // adding a new feature sets its bit.
        assert_eq!(transition_to(0b0001, 1, 3, 4), 0b0011);
        // re-selecting a set feature is a no-op.
        assert_eq!(transition_to(0b0011, 1, 3, 4), 0b0011);
        // budget exhausted -> no change.
        assert_eq!(transition_to(0b0011, 2, 3, 2), 0b0011);
    }

    #[test]
    fn solve_linear_system_2x2() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4].
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![3.0, 5.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 0.8).abs() < 1e-9, "x0={}", x[0]);
        assert!((x[1] - 1.4).abs() < 1e-9, "x1={}", x[1]);
    }

    #[test]
    fn synthetic_series_length_and_regime_progression() {
        let series = synthetic_forecast_series(50, 8);
        assert_eq!(series.len(), 58);
        assert_eq!(hidden_regime_at(0, 50), RegimeId::Baseline);
        assert_eq!(hidden_regime_at(20, 50), RegimeId::Expansion);
        assert_eq!(hidden_regime_at(30, 50), RegimeId::Shock);
        assert_eq!(hidden_regime_at(40, 50), RegimeId::Contraction);
        assert_eq!(hidden_regime_at(60, 50), RegimeId::Expansion);
    }

    #[test]
    fn equation_text_formats_signs() {
        let txt = equation_text(&["f0".to_string(), "f1".to_string()], &[1.0, 2.0, -3.0]);
        assert_eq!(txt, "y_hat = 1.000 + 2.000*z(f0) - 3.000*z(f1)");
    }
}
