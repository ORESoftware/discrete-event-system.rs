//! Port of `src/des/general/domain-application-models.ts`.
//!
//! Applied operations / control / data-science models, each expressed as the
//! same explicit DES topology:
//!
//! ```text
//! ScenarioSource -> CandidateGenerator -> PlanEvaluator -> ResultSink
//! ```
//!
//! Scenarios, candidate plans, and evaluated plans are moving tokens; the
//! generator / evaluator / sink are stationary entities.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//! * The generic Scenario/Plan station pipeline carries over as Rust generics
//!   `<S, P>` (both `Clone + 'static`). Tokens are stored as `Rc<dyn Any>` and
//!   recovered with `drain::<T>()`.
//! * `class FooStation extends DESStation` → a `struct` embedding [`StationCore`]
//!   and `impl DESStation` (Rust has no class inheritance).
//! * `interface FooScenario extends Required<FooParams>` → a fully-populated
//!   struct (Option defaults resolved up front in the `run*` entry points).
//! * Generator / evaluator logic → boxed closures
//!   (`Box<dyn Fn(&S) -> Vec<DomainCandidate<P>>>` /
//!   `Box<dyn Fn(&S, &P, &str) -> DomainEvaluation<P>>`).
//! * `DomainModelResult<P = unknown>` → generic with a concrete payload type per
//!   domain (no `unknown`).
//! * `metrics: Record<string, number | string | boolean>` → an ordered
//!   `Vec<(String, MetricValue)>` (insertion order preserved, like the TS
//!   object literal).
//! * `Preconditions` `throw` → `panic!` via the local [`require`] helper.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::des_base::learning_optimization::{channel_edge, station_graph, StationGraphSummary, StationOrId};
use super::des_base::preconditions::{Check, Preconditions};
use super::des_base::runner::{run_iterative_des, IterativeRunOptions};
use super::des_base::station::{AnyToken, DESStation, StationCore, StationRef};

/// Convert a recoverable [`Check`] into a `panic!` (the TS guards `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Shared result / evaluation types
// =============================================================================

/// A metric value (`number | string | boolean`).
#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    Num(f64),
    Str(String),
    Bool(bool),
}

fn m_num(key: &str, v: f64) -> (String, MetricValue) {
    (key.to_string(), MetricValue::Num(v))
}
fn m_str(key: &str, v: String) -> (String, MetricValue) {
    (key.to_string(), MetricValue::Str(v))
}
fn m_bool(key: &str, v: bool) -> (String, MetricValue) {
    (key.to_string(), MetricValue::Bool(v))
}

/// `interface DomainTrace`.
#[derive(Clone, Debug, Default)]
pub struct DomainTrace {
    pub t: Vec<f64>,
    /// Ordered named series (TS `Record<string, number[]>`).
    pub series: Vec<(String, Vec<f64>)>,
    pub captions: Option<Vec<String>>,
}

/// `interface DomainEvaluation<P>`.
#[derive(Clone, Debug)]
pub struct DomainEvaluation<P> {
    pub candidate_id: String,
    pub plan: P,
    pub objective: f64,
    pub feasible: bool,
    pub metrics: Vec<(String, MetricValue)>,
    pub trace: Option<DomainTrace>,
}

/// `interface DomainModelResult<P>`.
#[derive(Clone, Debug)]
pub struct DomainModelResult<P> {
    pub model_id: String,
    pub category: String,
    pub best: DomainEvaluation<P>,
    pub candidates: Vec<DomainEvaluation<P>>,
    pub topology: StationGraphSummary,
}

/// A generated candidate (`{candidateId, plan}`).
#[derive(Clone, Debug)]
pub struct DomainCandidate<P> {
    pub candidate_id: String,
    pub plan: P,
}

fn cand<P>(candidate_id: &str, plan: P) -> DomainCandidate<P> {
    DomainCandidate { candidate_id: candidate_id.to_string(), plan }
}

#[derive(Clone, Debug)]
struct DomainScenario<S> {
    model_id: String,
    category: String,
    scenario: S,
}

// =============================================================================
// Tokens (stored as `Rc<dyn Any>`)
// =============================================================================

struct DomainScenarioToken<S> {
    payload: DomainScenario<S>,
}

struct DomainPlanToken<S, P> {
    // Carried for parity with the TS token shape; not read by the Rust pipeline.
    #[allow(dead_code)]
    model_id: String,
    #[allow(dead_code)]
    category: String,
    scenario: S,
    candidate_id: String,
    plan: P,
}

struct DomainEvaluationToken<P> {
    evaluation: DomainEvaluation<P>,
}

// =============================================================================
// Pipeline stations
// =============================================================================

const CH_SCENARIO: &str = "scenario";
const CH_PLAN: &str = "candidate-plan";
const CH_EVALUATION: &str = "evaluation";

struct DomainScenarioSourceStation<S> {
    core: StationCore,
    payload: DomainScenario<S>,
    emitted: bool,
}

impl<S: Clone + 'static> DomainScenarioSourceStation<S> {
    fn new(id: &str, payload: DomainScenario<S>) -> Self {
        DomainScenarioSourceStation { core: StationCore::new(id), payload, emitted: false }
    }
}

impl<S: Clone + 'static> DESStation for DomainScenarioSourceStation<S> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let token = DomainScenarioToken { payload: self.payload.clone() };
        let any: AnyToken = Rc::new(token);
        self.core.emit(any, CH_SCENARIO);
        self.emitted = true;
    }
}

type GenerateFn<S, P> = Box<dyn Fn(&S) -> Vec<DomainCandidate<P>>>;

struct DomainCandidateGeneratorStation<S, P> {
    core: StationCore,
    generate: GenerateFn<S, P>,
}

impl<S: Clone + 'static, P: 'static> DomainCandidateGeneratorStation<S, P> {
    fn new(id: &str, generate: GenerateFn<S, P>) -> Self {
        DomainCandidateGeneratorStation { core: StationCore::new(id), generate }
    }
}

impl<S: Clone + 'static, P: 'static> DESStation for DomainCandidateGeneratorStation<S, P> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_SCENARIO) > 0
    }
    fn run_time_step(&mut self) {
        let scenarios = self.core.drain::<DomainScenarioToken<S>>(CH_SCENARIO);
        let mut out: Vec<DomainPlanToken<S, P>> = Vec::new();
        for token in scenarios {
            for candidate in (self.generate)(&token.payload.scenario) {
                out.push(DomainPlanToken {
                    model_id: token.payload.model_id.clone(),
                    category: token.payload.category.clone(),
                    scenario: token.payload.scenario.clone(),
                    candidate_id: candidate.candidate_id,
                    plan: candidate.plan,
                });
            }
        }
        for token in out {
            let any: AnyToken = Rc::new(token);
            self.core.emit(any, CH_PLAN);
        }
    }
}

type EvaluateFn<S, P> = Box<dyn Fn(&S, &P, &str) -> DomainEvaluation<P>>;

struct DomainPlanEvaluatorStation<S, P> {
    core: StationCore,
    evaluate: EvaluateFn<S, P>,
}

impl<S: 'static, P: 'static> DomainPlanEvaluatorStation<S, P> {
    fn new(id: &str, evaluate: EvaluateFn<S, P>) -> Self {
        DomainPlanEvaluatorStation { core: StationCore::new(id), evaluate }
    }
}

impl<S: 'static, P: 'static> DESStation for DomainPlanEvaluatorStation<S, P> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_PLAN) > 0
    }
    fn run_time_step(&mut self) {
        let plans = self.core.drain::<DomainPlanToken<S, P>>(CH_PLAN);
        let mut out: Vec<DomainEvaluation<P>> = Vec::new();
        for token in plans {
            out.push((self.evaluate)(&token.scenario, &token.plan, &token.candidate_id));
        }
        for evaluation in out {
            let any: AnyToken = Rc::new(DomainEvaluationToken { evaluation });
            self.core.emit(any, CH_EVALUATION);
        }
    }
}

struct DomainResultSinkStation<P> {
    core: StationCore,
    evaluations: Vec<DomainEvaluation<P>>,
}

impl<P: Clone + 'static> DomainResultSinkStation<P> {
    fn new(id: &str) -> Self {
        DomainResultSinkStation { core: StationCore::new(id), evaluations: Vec::new() }
    }

    fn best(&self) -> DomainEvaluation<P> {
        let feasible: Vec<&DomainEvaluation<P>> = self.evaluations.iter().filter(|row| row.feasible).collect();
        if feasible.is_empty() {
            panic!("{}: no feasible domain plans were evaluated", self.core.id);
        }
        let mut best = feasible[0];
        for &row in &feasible {
            if row.objective > best.objective {
                best = row;
            }
        }
        best.clone()
    }
}

impl<P: Clone + 'static> DESStation for DomainResultSinkStation<P> {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CH_EVALUATION) > 0
    }
    fn run_time_step(&mut self) {
        let incoming = self.core.drain::<DomainEvaluationToken<P>>(CH_EVALUATION);
        for token in incoming {
            self.evaluations.push(token.evaluation.clone());
        }
    }
}

fn run_domain_pipeline<S, P>(
    model_id: &str,
    category: &str,
    scenario: S,
    generate: GenerateFn<S, P>,
    evaluate: EvaluateFn<S, P>,
) -> DomainModelResult<P>
where
    S: Clone + 'static,
    P: Clone + 'static,
{
    let source = Rc::new(RefCell::new(DomainScenarioSourceStation::new(
        &format!("{model_id}-scenario-source"),
        DomainScenario { model_id: model_id.to_string(), category: category.to_string(), scenario },
    )));
    let generator = Rc::new(RefCell::new(DomainCandidateGeneratorStation::new(
        &format!("{model_id}-candidate-generator"),
        generate,
    )));
    let evaluator = Rc::new(RefCell::new(DomainPlanEvaluatorStation::new(
        &format!("{model_id}-plan-evaluator"),
        evaluate,
    )));
    let sink = Rc::new(RefCell::new(DomainResultSinkStation::<P>::new(&format!("{model_id}-result-sink"))));

    {
        let target: StationRef = generator.clone();
        source.borrow_mut().core_mut().pipe(target, CH_SCENARIO, CH_SCENARIO);
    }
    {
        let target: StationRef = evaluator.clone();
        generator.borrow_mut().core_mut().pipe(target, CH_PLAN, CH_PLAN);
    }
    {
        let target: StationRef = sink.clone();
        evaluator.borrow_mut().core_mut().pipe(target, CH_EVALUATION, CH_EVALUATION);
    }

    let stations: Vec<StationRef> = vec![source.clone(), generator.clone(), evaluator.clone(), sink.clone()];
    run_iterative_des(
        stations,
        IterativeRunOptions { shuffle: false, max_ticks: Some(8), run_validators: false, ..Default::default() },
    );

    let mut candidates = sink.borrow().evaluations.clone();
    candidates.sort_by(|a, b| b.objective.partial_cmp(&a.objective).unwrap_or(std::cmp::Ordering::Equal));
    let best = sink.borrow().best();

    let s_oi = StationOrId::from(format!("{model_id}-scenario-source"));
    let g_oi = StationOrId::from(format!("{model_id}-candidate-generator"));
    let e_oi = StationOrId::from(format!("{model_id}-plan-evaluator"));
    let k_oi = StationOrId::from(format!("{model_id}-result-sink"));
    let movables = vec![
        "DomainScenarioToken".to_string(),
        "DomainPlanToken".to_string(),
        "DomainEvaluationToken".to_string(),
    ];
    let edges = vec![
        channel_edge(&s_oi, CH_SCENARIO, &g_oi, Some(CH_SCENARIO)),
        channel_edge(&g_oi, CH_PLAN, &e_oi, Some(CH_PLAN)),
        channel_edge(&e_oi, CH_EVALUATION, &k_oi, Some(CH_EVALUATION)),
    ];
    let topology = station_graph(&[s_oi, g_oi, e_oi, k_oi], &movables, &edges);

    DomainModelResult { model_id: model_id.to_string(), category: category.to_string(), best, candidates, topology }
}

// =============================================================================
// Shared math helpers
// =============================================================================

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    lo.max(hi.min(x))
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn check_positive_int(model: &str, param: &str, value: f64) {
    require(Preconditions::integer(model, param, value));
    require(Preconditions::check(model, param, "be >= 1", value >= 1.0, Some(value.to_string())));
}

// =============================================================================
// 1. Control systems: adaptive/fuzzy/intelligent control
// =============================================================================

/// `interface AdaptiveFuzzyControlParams`.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveFuzzyControlParams {
    pub steps: Option<usize>,
    pub dt: Option<f64>,
    pub setpoint: Option<f64>,
    pub initial_temp: Option<f64>,
    pub outside_temp: Option<f64>,
    pub disturbance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FuzzyControlScenario {
    pub steps: usize,
    pub dt: f64,
    pub setpoint: f64,
    pub initial_temp: f64,
    pub outside_temp: f64,
    pub disturbance: f64,
    pub plant_loss: f64,
    pub plant_gain: f64,
    pub control_max: f64,
}

#[derive(Clone, Debug)]
pub struct FuzzyControlPlan {
    pub error_gain: f64,
    pub derivative_gain: f64,
    pub output_gain: f64,
    pub adaptive_boost: f64,
}

pub type AdaptiveFuzzyControlResult = DomainModelResult<FuzzyControlPlan>;

pub fn run_adaptive_fuzzy_control(params: AdaptiveFuzzyControlParams) -> AdaptiveFuzzyControlResult {
    let scenario = FuzzyControlScenario {
        steps: params.steps.unwrap_or(140),
        dt: params.dt.unwrap_or(0.1),
        setpoint: params.setpoint.unwrap_or(22.0),
        initial_temp: params.initial_temp.unwrap_or(16.0),
        outside_temp: params.outside_temp.unwrap_or(8.0),
        disturbance: params.disturbance.unwrap_or(0.15),
        plant_loss: 0.06,
        plant_gain: 0.42,
        control_max: 6.0,
    };
    check_positive_int("runAdaptiveFuzzyControl", "steps", scenario.steps as f64);
    require(Preconditions::positive("runAdaptiveFuzzyControl", "dt", scenario.dt));
    run_domain_pipeline(
        "adaptive-fuzzy-control",
        "Control systems (adaptive; fuzzy; intelligent)",
        scenario,
        Box::new(fuzzy_control_candidates),
        Box::new(evaluate_fuzzy_control),
    )
}

fn fuzzy_control_candidates(_scenario: &FuzzyControlScenario) -> Vec<DomainCandidate<FuzzyControlPlan>> {
    vec![
        cand("calm-fuzzy", FuzzyControlPlan { error_gain: 0.35, derivative_gain: 0.10, output_gain: 2.8, adaptive_boost: 0.0 }),
        cand("balanced-adaptive-fuzzy", FuzzyControlPlan { error_gain: 0.55, derivative_gain: 0.20, output_gain: 4.2, adaptive_boost: 0.8 }),
        cand("aggressive-fuzzy", FuzzyControlPlan { error_gain: 0.85, derivative_gain: 0.25, output_gain: 5.8, adaptive_boost: 0.4 }),
        cand("energy-saver-fuzzy", FuzzyControlPlan { error_gain: 0.45, derivative_gain: 0.35, output_gain: 3.3, adaptive_boost: 0.2 }),
    ]
}

fn evaluate_fuzzy_control(scenario: &FuzzyControlScenario, plan: &FuzzyControlPlan, candidate_id: &str) -> DomainEvaluation<FuzzyControlPlan> {
    let mut temp = scenario.initial_temp;
    let mut prev_error = scenario.setpoint - temp;
    let mut energy = 0.0;
    let mut sq_err = 0.0;
    let mut settling_tick = scenario.steps;
    for k in 0..scenario.steps {
        let error = scenario.setpoint - temp;
        let d_error = error - prev_error;
        let boost = if error.abs() > 1.5 { plan.adaptive_boost * 1.5_f64.min(error.abs() / 4.0) } else { 0.0 };
        let fuzzy_signal = (plan.error_gain * error + plan.derivative_gain * d_error).tanh();
        let control = clamp(plan.output_gain * (fuzzy_signal + boost), 0.0, scenario.control_max);
        let outdoor_leak = scenario.plant_loss * (scenario.outside_temp - temp);
        let seasonal_disturbance = scenario.disturbance * (0.15 * k as f64).sin();
        temp += scenario.dt * (outdoor_leak + scenario.plant_gain * control + seasonal_disturbance);
        energy += control * scenario.dt;
        sq_err += error * error;
        if settling_tick == scenario.steps && error.abs() < 0.25 {
            settling_tick = k;
        }
        prev_error = error;
    }
    let rms_error = (sq_err / scenario.steps as f64).sqrt();
    let objective = -rms_error - 0.025 * energy - 0.001 * settling_tick as f64;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: true,
        metrics: vec![m_num("rmsError", rms_error), m_num("energy", energy), m_num("settlingTick", settling_tick as f64), m_num("finalTemp", temp)],
        trace: None,
    }
}

// =============================================================================
// 2. Logistics/transportation: routing heuristics
// =============================================================================

/// `interface LogisticsRoutingParams`.
#[derive(Clone, Debug, Default)]
pub struct LogisticsRoutingParams {
    pub vehicle_capacity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Customer {
    pub id: usize,
    pub x: f64,
    pub y: f64,
    pub demand: f64,
}

#[derive(Clone, Debug)]
pub struct RoutingScenario {
    pub depot: (f64, f64),
    pub customers: Vec<Customer>,
    pub vehicle_capacity: f64,
}

/// `type ... = 'nearest-neighbor' | 'sweep' | 'savings' | 'balanced-savings'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingHeuristic {
    NearestNeighbor,
    Sweep,
    Savings,
    BalancedSavings,
}

#[derive(Clone, Debug)]
pub struct RoutingPlan {
    pub heuristic: RoutingHeuristic,
    pub routes: Vec<Vec<usize>>,
}

pub type LogisticsRoutingResult = DomainModelResult<RoutingPlan>;

pub fn run_logistics_routing_heuristics(params: LogisticsRoutingParams) -> LogisticsRoutingResult {
    let scenario = RoutingScenario {
        depot: (0.0, 0.0),
        vehicle_capacity: params.vehicle_capacity.unwrap_or(7.0),
        customers: vec![
            Customer { id: 1, x: 2.0, y: 1.0, demand: 2.0 },
            Customer { id: 2, x: 3.0, y: 4.0, demand: 2.0 },
            Customer { id: 3, x: -1.0, y: 3.0, demand: 1.0 },
            Customer { id: 4, x: -3.0, y: 2.0, demand: 3.0 },
            Customer { id: 5, x: -2.0, y: -2.0, demand: 2.0 },
            Customer { id: 6, x: 3.0, y: -2.0, demand: 2.0 },
            Customer { id: 7, x: 5.0, y: 1.0, demand: 1.0 },
        ],
    };
    require(Preconditions::positive("runLogisticsRoutingHeuristics", "vehicleCapacity", scenario.vehicle_capacity));
    run_domain_pipeline(
        "logistics-routing-heuristics",
        "Logistics/transportation (optimal routing, heuristics, scheduling)",
        scenario,
        Box::new(routing_candidates),
        Box::new(evaluate_routing_plan),
    )
}

fn routing_candidates(scenario: &RoutingScenario) -> Vec<DomainCandidate<RoutingPlan>> {
    let nearest = build_nearest_neighbor_routes(scenario);
    // polar sweep: sort customers by angle atan2(y, x), then split by capacity.
    let mut sorted_customers = scenario.customers.clone();
    sorted_customers.sort_by(|a, b| {
        a.y.atan2(a.x).partial_cmp(&b.y.atan2(b.x)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let sweep_seq: Vec<usize> = sorted_customers.iter().map(|c| c.id).collect();
    let sweep = split_sequence_by_capacity(&sweep_seq, scenario);
    let savings = build_savings_routes(scenario, false);
    let balanced_savings = build_savings_routes(scenario, true);
    vec![
        cand("nearest-neighbor", RoutingPlan { heuristic: RoutingHeuristic::NearestNeighbor, routes: nearest }),
        cand("polar-sweep", RoutingPlan { heuristic: RoutingHeuristic::Sweep, routes: sweep }),
        cand("clarke-wright-savings", RoutingPlan { heuristic: RoutingHeuristic::Savings, routes: savings }),
        cand("balanced-savings", RoutingPlan { heuristic: RoutingHeuristic::BalancedSavings, routes: balanced_savings }),
    ]
}

fn build_nearest_neighbor_routes(scenario: &RoutingScenario) -> Vec<Vec<usize>> {
    let mut remaining: Vec<usize> = scenario.customers.iter().map(|c| c.id).collect();
    let mut routes: Vec<Vec<usize>> = Vec::new();
    while !remaining.is_empty() {
        let mut route: Vec<usize> = Vec::new();
        let mut load = 0.0;
        let mut cur = scenario.depot;
        loop {
            let mut best: Option<usize> = None;
            let mut best_d = f64::INFINITY;
            for &id in &remaining {
                let c = customer_by_id(scenario, id);
                if load + c.demand > scenario.vehicle_capacity {
                    continue;
                }
                let d = dist(cur, (c.x, c.y));
                if d < best_d {
                    best_d = d;
                    best = Some(id);
                }
            }
            let Some(best_id) = best else {
                break;
            };
            let bc = customer_by_id(scenario, best_id).clone();
            route.push(best_id);
            load += bc.demand;
            remaining.retain(|&x| x != best_id);
            cur = (bc.x, bc.y);
        }
        routes.push(route);
    }
    routes
}

struct Saving {
    a: usize,
    b: usize,
    saving: f64,
}

fn build_savings_routes(scenario: &RoutingScenario, balance: bool) -> Vec<Vec<usize>> {
    let mut routes: Vec<Vec<usize>> = scenario.customers.iter().map(|c| vec![c.id]).collect();
    let mut savings: Vec<Saving> = Vec::new();
    for a in &scenario.customers {
        for b in &scenario.customers {
            if a.id >= b.id {
                continue;
            }
            let base_saving = dist(scenario.depot, (a.x, a.y)) + dist(scenario.depot, (b.x, b.y)) - dist((a.x, a.y), (b.x, b.y));
            let balance_penalty = if balance { 0.04 * (a.demand - b.demand).abs() } else { 0.0 };
            savings.push(Saving { a: a.id, b: b.id, saving: base_saving - balance_penalty });
        }
    }
    savings.sort_by(|x, y| y.saving.partial_cmp(&x.saving).unwrap_or(std::cmp::Ordering::Equal));
    for s in &savings {
        let ia = routes.iter().position(|r| r[0] == s.a || *r.last().unwrap() == s.a);
        let ib = routes.iter().position(|r| r[0] == s.b || *r.last().unwrap() == s.b);
        let (Some(ia), Some(ib)) = (ia, ib) else {
            continue;
        };
        if ia == ib {
            continue;
        }
        if route_load(scenario, &routes[ia]) + route_load(scenario, &routes[ib]) > scenario.vehicle_capacity {
            continue;
        }
        let merged = merge_route_ends(&routes[ia], &routes[ib], s.a, s.b);
        let (hi, lo) = if ia > ib { (ia, ib) } else { (ib, ia) };
        routes.remove(hi);
        routes.remove(lo);
        routes.push(merged);
    }
    routes
}

fn merge_route_ends(a: &[usize], b: &[usize], a_id: usize, b_id: usize) -> Vec<usize> {
    let aa: Vec<usize> = if a[0] == a_id { a.iter().rev().cloned().collect() } else { a.to_vec() };
    let bb: Vec<usize> = if *b.last().unwrap() == b_id { b.iter().rev().cloned().collect() } else { b.to_vec() };
    let mut out = aa;
    out.extend(bb);
    out
}

fn split_sequence_by_capacity(sequence: &[usize], scenario: &RoutingScenario) -> Vec<Vec<usize>> {
    let mut routes: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut load = 0.0;
    for &id in sequence {
        let d = customer_by_id(scenario, id).demand;
        if !cur.is_empty() && load + d > scenario.vehicle_capacity {
            routes.push(std::mem::take(&mut cur));
            load = 0.0;
        }
        cur.push(id);
        load += d;
    }
    if !cur.is_empty() {
        routes.push(cur);
    }
    routes
}

fn evaluate_routing_plan(scenario: &RoutingScenario, plan: &RoutingPlan, candidate_id: &str) -> DomainEvaluation<RoutingPlan> {
    let route_distance: f64 = plan.routes.iter().map(|route| route_length(scenario, route)).sum();
    let capacity_violation: f64 = plan.routes.iter().map(|route| (route_load(scenario, route) - scenario.vehicle_capacity).max(0.0)).sum();
    let objective = -route_distance - 1000.0 * capacity_violation - 0.2 * plan.routes.len() as f64;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: capacity_violation == 0.0,
        metrics: vec![m_num("routeDistance", route_distance), m_num("vehicles", plan.routes.len() as f64), m_num("capacityViolation", capacity_violation)],
        trace: None,
    }
}

fn customer_by_id(scenario: &RoutingScenario, id: usize) -> &Customer {
    scenario.customers.iter().find(|c| c.id == id).unwrap_or_else(|| panic!("unknown customer {id}"))
}

fn route_load(scenario: &RoutingScenario, route: &[usize]) -> f64 {
    route.iter().map(|&id| customer_by_id(scenario, id).demand).sum()
}

fn route_length(scenario: &RoutingScenario, route: &[usize]) -> f64 {
    let mut total = 0.0;
    let mut cur = scenario.depot;
    for &id in route {
        let c = customer_by_id(scenario, id);
        total += dist(cur, (c.x, c.y));
        cur = (c.x, c.y);
    }
    total + dist(cur, scenario.depot)
}

// =============================================================================
// 3. Manufacturing: production planning/control
// =============================================================================

/// `interface ManufacturingParams`.
#[derive(Clone, Debug, Default)]
pub struct ManufacturingParams {
    pub horizon: Option<usize>,
    pub daily_demand: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ManufacturingScenario {
    pub horizon: usize,
    pub daily_demand: f64,
    pub stage1_rate: f64,
    pub stage2_rate: f64,
}

#[derive(Clone, Debug)]
pub struct ManufacturingPlan {
    pub release_lot: f64,
    pub wip_cap: f64,
    pub expedite_threshold: f64,
}

pub type ManufacturingResult = DomainModelResult<ManufacturingPlan>;

pub fn run_bottleneck_production_control(params: ManufacturingParams) -> ManufacturingResult {
    let scenario = ManufacturingScenario {
        horizon: params.horizon.unwrap_or(18),
        daily_demand: params.daily_demand.unwrap_or(8.0),
        stage1_rate: 12.0,
        stage2_rate: 9.0,
    };
    check_positive_int("runBottleneckProductionControl", "horizon", scenario.horizon as f64);
    require(Preconditions::positive("runBottleneckProductionControl", "dailyDemand", scenario.daily_demand));
    run_domain_pipeline(
        "bottleneck-production-control",
        "Manufacturing (production planning and control, novel algorithms)",
        scenario,
        Box::new(manufacturing_candidates),
        Box::new(evaluate_manufacturing_plan),
    )
}

fn manufacturing_candidates(_scenario: &ManufacturingScenario) -> Vec<DomainCandidate<ManufacturingPlan>> {
    vec![
        cand("push-large-lots", ManufacturingPlan { release_lot: 16.0, wip_cap: 50.0, expedite_threshold: 30.0 }),
        cand("lean-kanban", ManufacturingPlan { release_lot: 8.0, wip_cap: 20.0, expedite_threshold: 14.0 }),
        cand("bottleneck-buffer-rope", ManufacturingPlan { release_lot: 10.0, wip_cap: 28.0, expedite_threshold: 18.0 }),
        cand("adaptive-expedite-control", ManufacturingPlan { release_lot: 12.0, wip_cap: 32.0, expedite_threshold: 10.0 }),
    ]
}

fn evaluate_manufacturing_plan(scenario: &ManufacturingScenario, plan: &ManufacturingPlan, candidate_id: &str) -> DomainEvaluation<ManufacturingPlan> {
    let mut raw = 0.0;
    let mut buffer = 0.0;
    let mut finished = 0.0;
    let mut backlog = 0.0;
    let mut wip_area = 0.0;
    let mut shipped = 0.0;
    let mut expedites = 0.0;
    for _t in 0..scenario.horizon {
        let wip = raw + buffer;
        let release = if wip < plan.wip_cap { plan.release_lot.min(plan.wip_cap - wip) } else { 0.0 };
        raw += release;
        if backlog > plan.expedite_threshold {
            raw += 2.0;
            expedites += 1.0;
        }
        let m1 = raw.min(scenario.stage1_rate);
        raw -= m1;
        buffer += m1;
        let m2 = buffer.min(scenario.stage2_rate);
        buffer -= m2;
        finished += m2;
        let demand = scenario.daily_demand + backlog;
        let ship = finished.min(demand);
        finished -= ship;
        backlog = demand - ship;
        shipped += ship;
        wip_area += raw + buffer + finished;
    }
    let avg_wip = wip_area / scenario.horizon as f64;
    let service = shipped / (scenario.horizon as f64 * scenario.daily_demand);
    let objective = 15.0 * shipped - 3.5 * backlog - 0.25 * avg_wip - 1.2 * expedites;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: true,
        metrics: vec![m_num("shipped", shipped), m_num("service", service), m_num("backlog", backlog), m_num("avgWip", avg_wip), m_num("expedites", expedites)],
        trace: None,
    }
}

// =============================================================================
// 4. Supply chain management: risk pooling
// =============================================================================

/// `interface SupplyChainParams`.
#[derive(Clone, Debug, Default)]
pub struct SupplyChainParams {
    pub horizon: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct SupplyChainScenario {
    pub horizon: usize,
    pub demand: Vec<f64>,
    pub lead_time: usize,
}

#[derive(Clone, Debug)]
pub struct SupplyChainPlan {
    pub base_stock: f64,
    pub review_period: usize,
    pub risk_pooling: f64,
}

pub type SupplyChainResult = DomainModelResult<SupplyChainPlan>;

pub fn run_supply_chain_risk_pooling(params: SupplyChainParams) -> SupplyChainResult {
    let horizon = params.horizon.unwrap_or(20);
    let demand: Vec<f64> = (0..horizon)
        .map(|t| 12.0 + 4.0 * (0.65 * t as f64).sin() + if t % 5 == 0 { 5.0 } else { 0.0 })
        .collect();
    let scenario = SupplyChainScenario { horizon, demand, lead_time: 2 };
    check_positive_int("runSupplyChainRiskPooling", "horizon", horizon as f64);
    run_domain_pipeline(
        "supply-chain-risk-pooling",
        "Supply chain management (novel algorithms)",
        scenario,
        Box::new(supply_chain_candidates),
        Box::new(evaluate_supply_chain_plan),
    )
}

fn supply_chain_candidates(_scenario: &SupplyChainScenario) -> Vec<DomainCandidate<SupplyChainPlan>> {
    vec![
        cand("local-minmax", SupplyChainPlan { base_stock: 28.0, review_period: 1, risk_pooling: 0.0 }),
        cand("pooled-safety-stock", SupplyChainPlan { base_stock: 36.0, review_period: 2, risk_pooling: 0.45 }),
        cand("service-first-pooling", SupplyChainPlan { base_stock: 44.0, review_period: 1, risk_pooling: 0.7 }),
        cand("inventory-lean-pooling", SupplyChainPlan { base_stock: 32.0, review_period: 3, risk_pooling: 0.55 }),
    ]
}

struct PipelineOrder {
    t: usize,
    qty_a: f64,
    qty_b: f64,
}

fn evaluate_supply_chain_plan(scenario: &SupplyChainScenario, plan: &SupplyChainPlan, candidate_id: &str) -> DomainEvaluation<SupplyChainPlan> {
    let mut inv_a = plan.base_stock;
    let mut inv_b = plan.base_stock;
    let mut pipeline: Vec<PipelineOrder> = Vec::new();
    let mut served = 0.0;
    let mut demand_total = 0.0;
    let mut holding = 0.0;
    let mut stockout = 0.0;
    for t in 0..scenario.horizon {
        for order in pipeline.iter().filter(|o| o.t == t) {
            inv_a += order.qty_a;
            inv_b += order.qty_b;
        }
        let d_a = scenario.demand[t] * (0.9 + 0.1 * (t as f64).sin());
        let d_b = scenario.demand[t] * (1.1 - 0.1 * (t as f64).sin());
        let transfer = plan.risk_pooling * (inv_a - inv_b).max(0.0) / 2.0;
        inv_a -= transfer;
        inv_b += transfer;
        let s_a = inv_a.min(d_a);
        let s_b = inv_b.min(d_b);
        inv_a -= s_a;
        inv_b -= s_b;
        served += s_a + s_b;
        demand_total += d_a + d_b;
        stockout += (d_a - s_a) + (d_b - s_b);
        holding += inv_a + inv_b;
        if t % plan.review_period == 0 {
            pipeline.push(PipelineOrder {
                t: t + scenario.lead_time,
                qty_a: (plan.base_stock - inv_a).max(0.0),
                qty_b: (plan.base_stock - inv_b).max(0.0),
            });
        }
    }
    let fill_rate = served / demand_total;
    let objective = 1000.0 * fill_rate - 0.18 * holding - 5.0 * stockout;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: true,
        metrics: vec![m_num("fillRate", fill_rate), m_num("holding", holding), m_num("stockout", stockout), m_num("served", served)],
        trace: None,
    }
}

// =============================================================================
// 5. Operations management: workforce service operations
// =============================================================================

/// `interface OperationsParams`.
#[derive(Clone, Debug, Default)]
pub struct OperationsParams {
    pub overtime_cost: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct OperationsScenario {
    pub demand: Vec<f64>,
    pub overtime_cost: f64,
}

#[derive(Clone, Debug)]
pub struct OperationsPlan {
    pub staffing: Vec<f64>,
    pub flex_pool: f64,
}

pub type OperationsResult = DomainModelResult<OperationsPlan>;

pub fn run_workforce_service_operations(params: OperationsParams) -> OperationsResult {
    let scenario = OperationsScenario {
        demand: vec![7.0, 11.0, 15.0, 12.0, 9.0, 6.0],
        overtime_cost: params.overtime_cost.unwrap_or(18.0),
    };
    require(Preconditions::positive("runWorkforceServiceOperations", "overtimeCost", scenario.overtime_cost));
    run_domain_pipeline(
        "workforce-service-operations",
        "Operations management (novel algorithms)",
        scenario,
        Box::new(operations_candidates),
        Box::new(evaluate_operations_plan),
    )
}

fn operations_candidates(_scenario: &OperationsScenario) -> Vec<DomainCandidate<OperationsPlan>> {
    vec![
        cand("lean-fixed-roster", OperationsPlan { staffing: vec![7.0, 9.0, 11.0, 10.0, 8.0, 6.0], flex_pool: 1.0 }),
        cand("service-buffer-roster", OperationsPlan { staffing: vec![8.0, 11.0, 14.0, 12.0, 10.0, 7.0], flex_pool: 2.0 }),
        cand("risk-pooled-flex-roster", OperationsPlan { staffing: vec![7.0, 10.0, 13.0, 11.0, 9.0, 6.0], flex_pool: 4.0 }),
        cand("overlap-wave-roster", OperationsPlan { staffing: vec![8.0, 12.0, 13.0, 13.0, 9.0, 6.0], flex_pool: 1.0 }),
    ]
}

fn evaluate_operations_plan(scenario: &OperationsScenario, plan: &OperationsPlan, candidate_id: &str) -> DomainEvaluation<OperationsPlan> {
    let mut covered = 0.0;
    let mut demand = 0.0;
    let mut idle = 0.0;
    let mut overtime = 0.0;
    for i in 0..scenario.demand.len() {
        let available = plan.staffing[i] + plan.flex_pool * (if scenario.demand[i] > plan.staffing[i] { 0.85 } else { 0.25 });
        covered += available.min(scenario.demand[i]);
        demand += scenario.demand[i];
        idle += (available - scenario.demand[i]).max(0.0);
        overtime += (scenario.demand[i] - available).max(0.0);
    }
    let service_level = covered / demand;
    let labor_cost = plan.staffing.iter().sum::<f64>() * 12.0 + plan.flex_pool * 20.0 + overtime * scenario.overtime_cost;
    let objective = 900.0 * service_level - labor_cost - 2.0 * idle;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: service_level >= 0.9,
        metrics: vec![m_num("serviceLevel", service_level), m_num("laborCost", labor_cost), m_num("overtime", overtime), m_num("idle", idle)],
        trace: None,
    }
}

// =============================================================================
// 6. Financial engineering: portfolio drawdown control
// =============================================================================

/// `interface FinancialControlParams`.
#[derive(Clone, Debug, Default)]
pub struct FinancialControlParams {
    pub initial_wealth: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FinancialScenario {
    pub initial_wealth: f64,
    pub returns: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct FinancialPlan {
    pub floor_fraction: f64,
    pub multiplier: f64,
    pub vol_target: f64,
}

pub type FinancialControlResult = DomainModelResult<FinancialPlan>;

pub fn run_portfolio_drawdown_control(params: FinancialControlParams) -> FinancialControlResult {
    let scenario = FinancialScenario {
        initial_wealth: params.initial_wealth.unwrap_or(100.0),
        returns: vec![0.012, 0.008, -0.018, 0.015, -0.025, 0.010, 0.006, -0.010, 0.020, -0.012, 0.011, 0.007],
    };
    require(Preconditions::positive("runPortfolioDrawdownControl", "initialWealth", scenario.initial_wealth));
    run_domain_pipeline(
        "portfolio-drawdown-control",
        "Financial engineering (applied control theory, novel algorithms)",
        scenario,
        Box::new(financial_candidates),
        Box::new(evaluate_financial_plan),
    )
}

fn financial_candidates(_scenario: &FinancialScenario) -> Vec<DomainCandidate<FinancialPlan>> {
    vec![
        cand("buy-and-hold", FinancialPlan { floor_fraction: 0.0, multiplier: 1.0, vol_target: 1.0 }),
        cand("conservative-cppi", FinancialPlan { floor_fraction: 0.88, multiplier: 2.2, vol_target: 0.7 }),
        cand("adaptive-drawdown-control", FinancialPlan { floor_fraction: 0.9, multiplier: 3.4, vol_target: 0.55 }),
        cand("growth-cppi", FinancialPlan { floor_fraction: 0.82, multiplier: 4.1, vol_target: 0.9 }),
    ]
}

fn evaluate_financial_plan(scenario: &FinancialScenario, plan: &FinancialPlan, candidate_id: &str) -> DomainEvaluation<FinancialPlan> {
    let mut wealth = scenario.initial_wealth;
    let mut peak = wealth;
    let mut max_drawdown = 0.0_f64;
    let mut turnover = 0.0;
    let mut prev_risk = 0.0;
    for &r in &scenario.returns {
        let floor = scenario.initial_wealth * plan.floor_fraction;
        let cushion = (wealth - floor).max(0.0);
        let risky_weight = clamp(plan.multiplier * cushion / wealth.max(1e-12), 0.0, plan.vol_target);
        wealth *= 1.0 + risky_weight * r + (1.0 - risky_weight) * 0.001;
        peak = peak.max(wealth);
        max_drawdown = max_drawdown.max((peak - wealth) / peak);
        turnover += (risky_weight - prev_risk).abs();
        prev_risk = risky_weight;
    }
    let objective = wealth - 85.0 * max_drawdown - 0.8 * turnover;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: wealth > 0.0,
        metrics: vec![m_num("finalWealth", wealth), m_num("maxDrawdown", max_drawdown), m_num("turnover", turnover)],
        trace: None,
    }
}

// =============================================================================
// 7. Revenue management: dynamic pricing
// =============================================================================

/// `interface RevenueManagementParams`.
#[derive(Clone, Debug, Default)]
pub struct RevenueManagementParams {
    pub capacity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RevenueScenario {
    pub capacity: f64,
    pub periods: usize,
    pub base_price: f64,
    pub base_demand: f64,
    pub elasticity: f64,
}

#[derive(Clone, Debug)]
pub struct PricingPlan {
    pub price_floor: f64,
    pub price_ceiling: f64,
    pub scarcity_gain: f64,
    pub smoothing: f64,
}

pub type RevenueManagementResult = DomainModelResult<PricingPlan>;

pub fn run_dynamic_pricing_revenue(params: RevenueManagementParams) -> RevenueManagementResult {
    let scenario = RevenueScenario {
        capacity: params.capacity.unwrap_or(120.0),
        periods: 16,
        base_price: 100.0,
        base_demand: 10.0,
        elasticity: 1.35,
    };
    require(Preconditions::positive("runDynamicPricingRevenue", "capacity", scenario.capacity));
    run_domain_pipeline(
        "dynamic-pricing-revenue",
        "Revenue management (novel dynamic pricing algorithms)",
        scenario,
        Box::new(pricing_candidates),
        Box::new(evaluate_pricing_plan),
    )
}

fn pricing_candidates(_scenario: &RevenueScenario) -> Vec<DomainCandidate<PricingPlan>> {
    vec![
        cand("fixed-reference-price", PricingPlan { price_floor: 100.0, price_ceiling: 100.0, scarcity_gain: 0.0, smoothing: 1.0 }),
        cand("scarcity-surge", PricingPlan { price_floor: 82.0, price_ceiling: 150.0, scarcity_gain: 0.45, smoothing: 0.55 }),
        cand("bayesian-demand-smoothing", PricingPlan { price_floor: 88.0, price_ceiling: 140.0, scarcity_gain: 0.32, smoothing: 0.78 }),
        cand("sellout-protection-pricing", PricingPlan { price_floor: 90.0, price_ceiling: 170.0, scarcity_gain: 0.70, smoothing: 0.45 }),
    ]
}

fn evaluate_pricing_plan(scenario: &RevenueScenario, plan: &PricingPlan, candidate_id: &str) -> DomainEvaluation<PricingPlan> {
    let mut inventory = scenario.capacity;
    let mut price = scenario.base_price;
    let mut revenue = 0.0;
    let mut sold = 0.0;
    for t in 0..scenario.periods {
        let scarcity = 1.0 - inventory / scenario.capacity;
        let target_price = clamp(scenario.base_price * (1.0 + plan.scarcity_gain * scarcity), plan.price_floor, plan.price_ceiling);
        price = plan.smoothing * price + (1.0 - plan.smoothing) * target_price;
        let season = 1.0 + 0.35 * (std::f64::consts::PI * t as f64 / (1.0_f64).max(scenario.periods as f64 - 1.0)).sin();
        let demand = scenario.base_demand * season * (-scenario.elasticity * (price / scenario.base_price - 1.0)).exp();
        let qty = inventory.min(demand);
        inventory -= qty;
        sold += qty;
        revenue += qty * price;
    }
    let sell_through = sold / scenario.capacity;
    let objective = revenue - 8.0 * inventory - 250.0 * (sell_through - 0.995).max(0.0);
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: true,
        metrics: vec![m_num("revenue", revenue), m_num("sold", sold), m_num("inventory", inventory), m_num("sellThrough", sell_through), m_num("finalPrice", price)],
        trace: None,
    }
}

// =============================================================================
// 8. Revenue management: buyer-aware dynamic pricing
// =============================================================================

/// `interface BuyerAwareDynamicPricingParams`.
#[derive(Clone, Debug, Default)]
pub struct BuyerAwareDynamicPricingParams {
    pub horizon: Option<usize>,
    pub initial_inventory: Option<f64>,
    pub privacy_budget: Option<f64>,
    pub fairness_tolerance: Option<f64>,
    pub sustainability_weight: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct BuyerSegment {
    pub id: String,
    pub size: f64,
    pub willingness_to_pay: f64,
    pub price_sensitivity: f64,
    pub online_signal: f64,
    pub consent_rate: f64,
    pub fairness_expectation: f64,
    pub retention_value: f64,
    pub sustainability_preference: f64,
}

#[derive(Clone, Debug)]
pub struct BuyerAwarePricingScenario {
    pub horizon: usize,
    pub initial_inventory: f64,
    pub privacy_budget: f64,
    pub fairness_tolerance: f64,
    pub sustainability_weight: f64,
    pub base_price: f64,
    pub unit_cost: f64,
    pub replenishment: Vec<f64>,
    pub demand_pulse: Vec<f64>,
    pub segments: Vec<BuyerSegment>,
}

#[derive(Clone, Debug)]
pub struct BuyerAwarePricingPlan {
    pub price_floor: f64,
    pub price_ceiling: f64,
    pub scarcity_gain: f64,
    pub demand_signal_gain: f64,
    pub personalization_gain: f64,
    pub consent_gate: bool,
    pub fairness_clamp: f64,
    pub smoothing: f64,
    pub max_price_changes: usize,
    pub retention_care: f64,
    pub waste_penalty: f64,
    pub sustainability_credit: f64,
}

struct PeriodPricingState {
    t: usize,
    public_price: f64,
    average_price: f64,
    inventory: f64,
    sold: f64,
    revenue: f64,
    fairness_spread: f64,
    retention_index: f64,
}

pub type BuyerAwareDynamicPricingResult = DomainModelResult<BuyerAwarePricingPlan>;

pub fn run_buyer_aware_dynamic_pricing(params: BuyerAwareDynamicPricingParams) -> BuyerAwareDynamicPricingResult {
    let horizon = params.horizon.unwrap_or(12);
    check_positive_int("runBuyerAwareDynamicPricing", "horizon", horizon as f64);
    let scenario = BuyerAwarePricingScenario {
        horizon,
        initial_inventory: params.initial_inventory.unwrap_or(160.0),
        privacy_budget: params.privacy_budget.unwrap_or(0.0),
        fairness_tolerance: params.fairness_tolerance.unwrap_or(0.18),
        sustainability_weight: params.sustainability_weight.unwrap_or(120.0),
        base_price: 100.0,
        unit_cost: 42.0,
        replenishment: (0..horizon).map(|t| if t == horizon / 2 { 34.0 } else { 0.0 }).collect(),
        demand_pulse: (0..horizon)
            .map(|t| 1.0 + 0.18 * (std::f64::consts::PI * t as f64 / (1.0_f64).max(horizon as f64 - 1.0)).sin() + if (t as f64) > horizon as f64 * 0.58 { 0.06 } else { 0.0 })
            .collect(),
        segments: vec![
            BuyerSegment { id: "value-seekers".to_string(), size: 18.0, willingness_to_pay: 82.0, price_sensitivity: 1.70, online_signal: 0.45, consent_rate: 0.40, fairness_expectation: 0.86, retention_value: 8.0, sustainability_preference: 0.55 },
            BuyerSegment { id: "convenience-buyers".to_string(), size: 14.0, willingness_to_pay: 118.0, price_sensitivity: 1.10, online_signal: 0.65, consent_rate: 0.64, fairness_expectation: 0.58, retention_value: 12.0, sustainability_preference: 0.35 },
            BuyerSegment { id: "premium-loyalists".to_string(), size: 8.0, willingness_to_pay: 148.0, price_sensitivity: 0.76, online_signal: 0.72, consent_rate: 0.82, fairness_expectation: 0.46, retention_value: 18.0, sustainability_preference: 0.42 },
            BuyerSegment { id: "privacy-protective".to_string(), size: 10.0, willingness_to_pay: 105.0, price_sensitivity: 1.30, online_signal: 0.50, consent_rate: 0.18, fairness_expectation: 0.92, retention_value: 15.0, sustainability_preference: 0.65 },
            BuyerSegment { id: "sustainability-led".to_string(), size: 7.0, willingness_to_pay: 126.0, price_sensitivity: 0.95, online_signal: 0.58, consent_rate: 0.55, fairness_expectation: 0.70, retention_value: 16.0, sustainability_preference: 0.95 },
        ],
    };
    require(Preconditions::positive("runBuyerAwareDynamicPricing", "initialInventory", scenario.initial_inventory));
    require(Preconditions::non_negative("runBuyerAwareDynamicPricing", "privacyBudget", scenario.privacy_budget));
    require(Preconditions::non_negative("runBuyerAwareDynamicPricing", "fairnessTolerance", scenario.fairness_tolerance));
    require(Preconditions::non_negative("runBuyerAwareDynamicPricing", "sustainabilityWeight", scenario.sustainability_weight));
    run_domain_pipeline(
        "buyer-aware-dynamic-pricing",
        "Revenue management (novel dynamic pricing algorithms)",
        scenario,
        Box::new(buyer_aware_pricing_candidates),
        Box::new(evaluate_buyer_aware_pricing_plan),
    )
}

fn buyer_aware_pricing_candidates(_scenario: &BuyerAwarePricingScenario) -> Vec<DomainCandidate<BuyerAwarePricingPlan>> {
    vec![
        cand("static-reference-price", BuyerAwarePricingPlan { price_floor: 100.0, price_ceiling: 100.0, scarcity_gain: 0.0, demand_signal_gain: 0.0, personalization_gain: 0.0, consent_gate: true, fairness_clamp: 0.0, smoothing: 1.0, max_price_changes: 0, retention_care: 0.55, waste_penalty: 8.0, sustainability_credit: 0.30 }),
        cand("limited-inventory-public-price", BuyerAwarePricingPlan { price_floor: 82.0, price_ceiling: 138.0, scarcity_gain: 0.38, demand_signal_gain: 0.22, personalization_gain: 0.0, consent_gate: true, fairness_clamp: 0.05, smoothing: 0.72, max_price_changes: 2, retention_care: 0.66, waste_penalty: 9.0, sustainability_credit: 0.38 }),
        cand("consent-aware-buyer-signals", BuyerAwarePricingPlan { price_floor: 80.0, price_ceiling: 145.0, scarcity_gain: 0.34, demand_signal_gain: 0.30, personalization_gain: 0.22, consent_gate: true, fairness_clamp: 0.13, smoothing: 0.62, max_price_changes: 3, retention_care: 0.78, waste_penalty: 8.0, sustainability_credit: 0.48 }),
        cand("aggressive-personalized-yield", BuyerAwarePricingPlan { price_floor: 75.0, price_ceiling: 185.0, scarcity_gain: 0.58, demand_signal_gain: 0.42, personalization_gain: 0.55, consent_gate: false, fairness_clamp: 0.36, smoothing: 0.35, max_price_changes: 8, retention_care: 0.25, waste_penalty: 5.0, sustainability_credit: 0.10 }),
        cand("fair-sustainable-lifecycle", BuyerAwarePricingPlan { price_floor: 86.0, price_ceiling: 132.0, scarcity_gain: 0.28, demand_signal_gain: 0.18, personalization_gain: 0.12, consent_gate: true, fairness_clamp: 0.09, smoothing: 0.78, max_price_changes: 2, retention_care: 0.95, waste_penalty: 13.0, sustainability_credit: 0.85 }),
    ]
}

fn segment_price(scenario: &BuyerAwarePricingScenario, plan: &BuyerAwarePricingPlan, public_price: f64, segment: &BuyerSegment) -> f64 {
    let consent_share = if plan.consent_gate { segment.consent_rate } else { 1.0 };
    let personal_component = scenario.base_price * plan.personalization_gain * (segment.willingness_to_pay / scenario.base_price - 1.0) * consent_share;
    let signal_component = scenario.base_price * plan.demand_signal_gain * 0.12 * (segment.online_signal - 0.55) * consent_share;
    let raw = public_price + personal_component + signal_component;
    let lo = plan.price_floor.max(public_price * (1.0 - plan.fairness_clamp));
    let hi = plan.price_ceiling.min(public_price * (1.0 + plan.fairness_clamp));
    clamp(raw, lo, hi)
}

fn evaluate_buyer_aware_pricing_plan(scenario: &BuyerAwarePricingScenario, plan: &BuyerAwarePricingPlan, candidate_id: &str) -> DomainEvaluation<BuyerAwarePricingPlan> {
    let mut inventory = scenario.initial_inventory;
    let max_inventory = scenario.initial_inventory + scenario.replenishment.iter().sum::<f64>();
    let mut public_price = scenario.base_price;
    let mut price_changes: usize = 0;
    let mut revenue = 0.0;
    let mut gross_margin = 0.0;
    let mut sold_total = 0.0;
    let mut price_weighted_sold = 0.0;
    let mut privacy_violations = 0.0;
    let mut fairness_spread_sum = 0.0;
    let mut fairness_penalty = 0.0;
    let mut retention_numerator = 0.0;
    let mut retention_denominator = 0.0;
    let mut trace: Vec<PeriodPricingState> = Vec::new();

    for t in 0..scenario.horizon {
        inventory += scenario.replenishment[t];
        let demand_pulse = scenario.demand_pulse[t];
        let scarcity = 1.0 - inventory / max_inventory.max(1e-12);
        let target = clamp(
            scenario.base_price * (1.0 + plan.scarcity_gain * scarcity + plan.demand_signal_gain * (demand_pulse - 1.0)),
            plan.price_floor,
            plan.price_ceiling,
        );
        let would_change = (target - public_price).abs() / public_price.max(1e-12) > 0.025;
        let change_allowed = would_change && price_changes < plan.max_price_changes;
        let effective_target = if change_allowed { target } else { public_price };
        if change_allowed {
            price_changes += 1;
        }
        public_price = plan.smoothing * public_price + (1.0 - plan.smoothing) * effective_target;

        let prices: Vec<f64> = scenario.segments.iter().map(|s| segment_price(scenario, plan, public_price, s)).collect();
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
        let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let fairness_spread = (max_price - min_price) / avg_price.max(1e-12);
        let fairness_excess = (fairness_spread - scenario.fairness_tolerance).max(0.0);
        let fairness_weight = scenario.segments.iter().map(|s| s.fairness_expectation).sum::<f64>() / scenario.segments.len() as f64;
        fairness_spread_sum += fairness_spread;
        fairness_penalty += fairness_excess * fairness_weight * 950.0;

        let expected_demand: Vec<f64> = scenario.segments.iter().enumerate().map(|(i, segment)| {
            let price = prices[i];
            let affordability = segment.willingness_to_pay / price.max(1e-12);
            let response = (segment.price_sensitivity * (affordability - 1.0)).exp();
            let privacy_exposure = if plan.consent_gate { 0.0 } else { (1.0 - segment.consent_rate) * plan.personalization_gain };
            let fairness_drag = 1.0 - clamp(fairness_excess * segment.fairness_expectation * 1.7, 0.0, 0.60);
            let privacy_drag = 1.0 - clamp(privacy_exposure * 0.32, 0.0, 0.45);
            let sustainability_lift = 1.0 + 0.07 * plan.sustainability_credit * segment.sustainability_preference;
            segment.size * demand_pulse * response * fairness_drag * privacy_drag * sustainability_lift
        }).collect();
        let period_demand = expected_demand.iter().sum::<f64>();
        let period_sold = inventory.min(period_demand);

        if period_demand > 0.0 {
            for (i, segment) in scenario.segments.iter().enumerate() {
                let sold = period_sold * expected_demand[i] / period_demand;
                let price = prices[i];
                let privacy_exposure = if plan.consent_gate { 0.0 } else { (1.0 - segment.consent_rate) * plan.personalization_gain };
                let retention_factor = clamp(
                    0.92
                        + 0.08 * plan.retention_care
                        + 0.04 * plan.sustainability_credit * segment.sustainability_preference
                        - fairness_excess * segment.fairness_expectation * 1.15
                        - privacy_exposure * 0.38,
                    0.0,
                    1.08,
                );
                let segment_revenue = sold * price;
                revenue += segment_revenue;
                gross_margin += sold * (price - scenario.unit_cost);
                sold_total += sold;
                price_weighted_sold += sold * price;
                retention_numerator += sold * segment.retention_value * retention_factor;
                retention_denominator += sold * segment.retention_value;
                privacy_violations += if plan.consent_gate { 0.0 } else { sold * (1.0 - segment.consent_rate) * plan.personalization_gain };
            }
        }
        inventory -= period_sold;
        trace.push(PeriodPricingState {
            t,
            public_price,
            average_price: avg_price,
            inventory,
            sold: period_sold,
            revenue,
            fairness_spread,
            retention_index: if retention_denominator > 0.0 { retention_numerator / retention_denominator } else { 1.0 },
        });
    }

    let avg_fairness_spread = fairness_spread_sum / scenario.horizon as f64;
    let avg_price = price_weighted_sold / sold_total.max(1e-12);
    let sell_through = sold_total / max_inventory.max(1e-12);
    let final_inventory = inventory;
    let retention_index = if retention_denominator > 0.0 { retention_numerator / retention_denominator } else { 1.0 };
    let waste_share = final_inventory / max_inventory.max(1e-12);
    let sustainability_score = clamp(1.0 - waste_share + 0.08 * plan.sustainability_credit - 0.03 * price_changes as f64, 0.0, 1.1);
    let privacy_cost = privacy_violations * 18.0 * (0.6 + plan.personalization_gain);
    let waste_cost = plan.waste_penalty * final_inventory;
    let objective = gross_margin
        + 0.20 * retention_numerator
        + scenario.sustainability_weight * sustainability_score
        - privacy_cost
        - fairness_penalty
        - waste_cost
        - 35.0 * price_changes as f64;
    let feasible = privacy_violations <= scenario.privacy_budget + 1e-9
        && avg_fairness_spread <= scenario.fairness_tolerance + 0.025
        && retention_index >= 0.78;

    let domain_trace = DomainTrace {
        t: trace.iter().map(|row| row.t as f64).collect(),
        series: vec![
            ("publicPrice".to_string(), trace.iter().map(|row| row.public_price).collect()),
            ("averagePrice".to_string(), trace.iter().map(|row| row.average_price).collect()),
            ("inventory".to_string(), trace.iter().map(|row| row.inventory).collect()),
            ("sold".to_string(), trace.iter().map(|row| row.sold).collect()),
            ("cumulativeRevenue".to_string(), trace.iter().map(|row| row.revenue).collect()),
            ("fairnessSpread".to_string(), trace.iter().map(|row| row.fairness_spread).collect()),
            ("retentionIndex".to_string(), trace.iter().map(|row| row.retention_index).collect()),
        ],
        captions: Some(
            trace
                .iter()
                .map(|row| format!("t={}: price={:.2} inventory={:.1} fairness={:.3}", row.t, row.average_price, row.inventory, row.fairness_spread))
                .collect(),
        ),
    };

    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible,
        metrics: vec![
            m_num("revenue", revenue),
            m_num("grossMargin", gross_margin),
            m_num("unitsSold", sold_total),
            m_num("finalInventory", final_inventory),
            m_num("sellThrough", sell_through),
            m_num("avgPrice", avg_price),
            m_num("avgFairnessSpread", avg_fairness_spread),
            m_num("privacyViolations", privacy_violations),
            m_num("retentionIndex", retention_index),
            m_num("sustainabilityScore", sustainability_score),
            m_num("priceChanges", price_changes as f64),
        ],
        trace: Some(domain_trace),
    }
}

// =============================================================================
// 9. Energy: optimization of power systems
// =============================================================================

/// `interface EnergyParams`.
#[derive(Clone, Debug, Default)]
pub struct EnergyParams {
    pub battery_capacity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EnergyScenario {
    pub demand: Vec<f64>,
    pub renewable: Vec<f64>,
    pub price: Vec<f64>,
    pub battery_capacity: f64,
    pub max_charge: f64,
}

#[derive(Clone, Debug)]
pub struct EnergyPlan {
    pub charge_below: f64,
    pub discharge_above: f64,
    pub reserve: f64,
}

pub type EnergyResult = DomainModelResult<EnergyPlan>;

pub fn run_energy_storage_dispatch(params: EnergyParams) -> EnergyResult {
    let scenario = EnergyScenario {
        demand: vec![42.0, 40.0, 38.0, 36.0, 45.0, 58.0, 67.0, 72.0, 68.0, 54.0, 48.0, 44.0],
        renewable: vec![8.0, 9.0, 12.0, 20.0, 30.0, 34.0, 28.0, 18.0, 12.0, 9.0, 8.0, 7.0],
        price: vec![36.0, 32.0, 28.0, 24.0, 18.0, 22.0, 42.0, 68.0, 74.0, 55.0, 44.0, 38.0],
        battery_capacity: params.battery_capacity.unwrap_or(50.0),
        max_charge: 12.0,
    };
    require(Preconditions::positive("runEnergyStorageDispatch", "batteryCapacity", scenario.battery_capacity));
    run_domain_pipeline(
        "energy-storage-dispatch",
        "Energy (optimization of power systems)",
        scenario,
        Box::new(energy_candidates),
        Box::new(evaluate_energy_plan),
    )
}

fn energy_candidates(_scenario: &EnergyScenario) -> Vec<DomainCandidate<EnergyPlan>> {
    vec![
        cand("no-storage-reference", EnergyPlan { charge_below: f64::NEG_INFINITY, discharge_above: f64::INFINITY, reserve: 0.0 }),
        cand("price-arbitrage-dispatch", EnergyPlan { charge_below: 30.0, discharge_above: 55.0, reserve: 8.0 }),
        cand("renewable-first-dispatch", EnergyPlan { charge_below: 42.0, discharge_above: 62.0, reserve: 15.0 }),
        cand("reliability-reserve-dispatch", EnergyPlan { charge_below: 34.0, discharge_above: 48.0, reserve: 22.0 }),
    ]
}

fn evaluate_energy_plan(scenario: &EnergyScenario, plan: &EnergyPlan, candidate_id: &str) -> DomainEvaluation<EnergyPlan> {
    let mut soc = scenario.battery_capacity / 2.0;
    let mut cost = 0.0;
    let mut curtailment = 0.0;
    let mut unserved = 0.0;
    let mut emissions = 0.0;
    for t in 0..scenario.demand.len() {
        let mut net_load = scenario.demand[t] - scenario.renewable[t];
        if scenario.price[t] < plan.charge_below {
            let charge = scenario.max_charge.min(scenario.battery_capacity - soc);
            soc += charge;
            net_load += charge / 0.92;
        }
        if scenario.price[t] > plan.discharge_above && soc > plan.reserve {
            let discharge = scenario.max_charge.min(soc - plan.reserve).min(net_load.max(0.0));
            soc -= discharge;
            net_load -= 0.92 * discharge;
        }
        if net_load < 0.0 {
            curtailment += -net_load;
        }
        let thermal = net_load.max(0.0);
        cost += thermal * scenario.price[t] + 0.08 * thermal * thermal;
        emissions += 0.45 * thermal;
        unserved += (net_load - 75.0).max(0.0);
    }
    let objective = -cost - 1000.0 * unserved - 8.0 * curtailment - 2.0 * emissions;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: unserved < 1e-9,
        metrics: vec![m_num("cost", cost), m_num("curtailment", curtailment), m_num("unserved", unserved), m_num("emissions", emissions), m_num("finalSoc", soc)],
        trace: None,
    }
}

// =============================================================================
// 10. Machine learning: active-learning acquisition
// =============================================================================

/// `interface ActiveLearningParams`.
#[derive(Clone, Debug, Default)]
pub struct ActiveLearningParams {
    pub budget: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct PoolItem {
    pub id: usize,
    pub uncertainty: f64,
    pub diversity: f64,
    pub cost: f64,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct ActiveLearningScenario {
    pub budget: f64,
    pub pool: Vec<PoolItem>,
}

#[derive(Clone, Debug)]
pub struct ActiveLearningPlan {
    pub uncertainty_weight: f64,
    pub diversity_weight: f64,
    pub cost_weight: f64,
}

pub type ActiveLearningResult = DomainModelResult<ActiveLearningPlan>;

pub fn run_active_learning_acquisition(params: ActiveLearningParams) -> ActiveLearningResult {
    let scenario = ActiveLearningScenario {
        budget: params.budget.unwrap_or(9.0),
        pool: vec![
            PoolItem { id: 1, uncertainty: 0.92, diversity: 0.35, cost: 2.0, value: 0.9 },
            PoolItem { id: 2, uncertainty: 0.65, diversity: 0.80, cost: 3.0, value: 0.85 },
            PoolItem { id: 3, uncertainty: 0.74, diversity: 0.72, cost: 2.0, value: 0.78 },
            PoolItem { id: 4, uncertainty: 0.40, diversity: 0.95, cost: 2.0, value: 0.66 },
            PoolItem { id: 5, uncertainty: 0.88, diversity: 0.45, cost: 4.0, value: 0.95 },
            PoolItem { id: 6, uncertainty: 0.55, diversity: 0.60, cost: 1.0, value: 0.60 },
        ],
    };
    require(Preconditions::positive("runActiveLearningAcquisition", "budget", scenario.budget));
    run_domain_pipeline(
        "active-learning-acquisition",
        "Machine learning and statistical learning (novel algorithms and novel use cases)",
        scenario,
        Box::new(active_learning_candidates),
        Box::new(evaluate_active_learning_plan),
    )
}

fn active_learning_candidates(_scenario: &ActiveLearningScenario) -> Vec<DomainCandidate<ActiveLearningPlan>> {
    vec![
        cand("uncertainty-sampling", ActiveLearningPlan { uncertainty_weight: 1.0, diversity_weight: 0.0, cost_weight: 0.0 }),
        cand("diversity-regularized-active-learning", ActiveLearningPlan { uncertainty_weight: 0.7, diversity_weight: 0.55, cost_weight: 0.1 }),
        cand("cost-aware-information-gain", ActiveLearningPlan { uncertainty_weight: 0.75, diversity_weight: 0.35, cost_weight: 0.45 }),
        cand("balanced-portfolio-acquisition", ActiveLearningPlan { uncertainty_weight: 0.55, diversity_weight: 0.65, cost_weight: 0.25 }),
    ]
}

fn score_active(item: &PoolItem, plan: &ActiveLearningPlan) -> f64 {
    plan.uncertainty_weight * item.uncertainty + plan.diversity_weight * item.diversity - plan.cost_weight * item.cost
}

fn evaluate_active_learning_plan(scenario: &ActiveLearningScenario, plan: &ActiveLearningPlan, candidate_id: &str) -> DomainEvaluation<ActiveLearningPlan> {
    let mut ranked = scenario.pool.clone();
    ranked.sort_by(|a, b| score_active(b, plan).partial_cmp(&score_active(a, plan)).unwrap_or(std::cmp::Ordering::Equal));
    let mut cost = 0.0;
    let mut info_gain = 0.0;
    let mut selected: Vec<usize> = Vec::new();
    for item in &ranked {
        if cost + item.cost > scenario.budget {
            continue;
        }
        selected.push(item.id);
        cost += item.cost;
        info_gain += item.value * (0.65 * item.uncertainty + 0.35 * item.diversity);
    }
    let expected_error_reduction = 1.0 - (-info_gain / 2.8).exp();
    let objective = 100.0 * expected_error_reduction - 0.8 * cost + 2.0 * selected.len() as f64;
    let selected_str = selected.iter().map(|id| id.to_string()).collect::<Vec<_>>().join("|");
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: !selected.is_empty(),
        metrics: vec![m_str("selected", selected_str), m_num("cost", cost), m_num("infoGain", info_gain), m_num("expectedErrorReduction", expected_error_reduction)],
        trace: None,
    }
}

// =============================================================================
// 11. Decision science: visual decision frontier
// =============================================================================

/// `interface DecisionScienceParams`.
#[derive(Clone, Debug, Default)]
pub struct DecisionScienceParams {
    pub risk_weight: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct DecisionAlternative {
    pub name: String,
    pub cost: f64,
    pub impact: f64,
    pub risk: f64,
    pub adoption: f64,
}

#[derive(Clone, Debug)]
pub struct DecisionScenario {
    pub alternatives: Vec<DecisionAlternative>,
    pub risk_weight: f64,
}

#[derive(Clone, Debug)]
pub struct DecisionPlan {
    pub impact_weight: f64,
    pub adoption_weight: f64,
    pub cost_weight: f64,
    pub risk_weight: f64,
}

pub type DecisionScienceResult = DomainModelResult<DecisionPlan>;

pub fn run_visual_decision_frontier(params: DecisionScienceParams) -> DecisionScienceResult {
    let scenario = DecisionScenario {
        risk_weight: params.risk_weight.unwrap_or(0.35),
        alternatives: vec![
            DecisionAlternative { name: "pilot automation".to_string(), cost: 42.0, impact: 78.0, risk: 22.0, adoption: 74.0 },
            DecisionAlternative { name: "full platform rebuild".to_string(), cost: 88.0, impact: 96.0, risk: 65.0, adoption: 58.0 },
            DecisionAlternative { name: "targeted workflow redesign".to_string(), cost: 35.0, impact: 70.0, risk: 18.0, adoption: 82.0 },
            DecisionAlternative { name: "analytics copilot".to_string(), cost: 54.0, impact: 86.0, risk: 35.0, adoption: 76.0 },
            DecisionAlternative { name: "status quo plus training".to_string(), cost: 18.0, impact: 42.0, risk: 9.0, adoption: 90.0 },
        ],
    };
    require(Preconditions::non_negative("runVisualDecisionFrontier", "riskWeight", scenario.risk_weight));
    run_domain_pipeline(
        "visual-decision-frontier",
        "Decision science (using data science combined with visualization)",
        scenario,
        Box::new(decision_candidates),
        Box::new(evaluate_decision_plan),
    )
}

fn decision_candidates(scenario: &DecisionScenario) -> Vec<DomainCandidate<DecisionPlan>> {
    vec![
        cand("impact-led-view", DecisionPlan { impact_weight: 0.60, adoption_weight: 0.20, cost_weight: 0.12, risk_weight: scenario.risk_weight }),
        cand("adoption-led-view", DecisionPlan { impact_weight: 0.38, adoption_weight: 0.42, cost_weight: 0.12, risk_weight: scenario.risk_weight }),
        cand("risk-adjusted-frontier", DecisionPlan { impact_weight: 0.48, adoption_weight: 0.28, cost_weight: 0.08, risk_weight: scenario.risk_weight + 0.2 }),
        cand("lean-value-frontier", DecisionPlan { impact_weight: 0.42, adoption_weight: 0.25, cost_weight: 0.25, risk_weight: scenario.risk_weight }),
    ]
}

fn evaluate_decision_plan(scenario: &DecisionScenario, plan: &DecisionPlan, candidate_id: &str) -> DomainEvaluation<DecisionPlan> {
    let mut scored: Vec<(usize, f64)> = scenario
        .alternatives
        .iter()
        .enumerate()
        .map(|(i, alt)| {
            let score = plan.impact_weight * alt.impact + plan.adoption_weight * alt.adoption - plan.cost_weight * alt.cost - plan.risk_weight * alt.risk;
            (i, score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = &scored[0];
    let top_alt = &scenario.alternatives[top.0];
    let separation = if scored.len() > 1 { top.1 - scored[1].1 } else { top.1 };
    let frontier_count = scenario.alternatives.iter().filter(|alt| alt.impact >= 70.0 && alt.risk <= 40.0).count();
    let objective = top.1 + 0.15 * separation + frontier_count as f64;
    DomainEvaluation {
        candidate_id: candidate_id.to_string(),
        plan: plan.clone(),
        objective,
        feasible: true,
        metrics: vec![
            m_str("topAlternative", top_alt.name.clone()),
            m_num("topScore", top.1),
            m_num("separation", separation),
            m_num("frontierCount", frontier_count as f64),
            m_bool("visualizationReady", true),
        ],
        trace: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_control_picks_a_feasible_best() {
        let result = run_adaptive_fuzzy_control(AdaptiveFuzzyControlParams::default());
        assert_eq!(result.model_id, "adaptive-fuzzy-control");
        assert!(result.best.feasible);
        assert_eq!(result.candidates.len(), 4);
        // candidates are sorted by descending objective.
        assert!(result.candidates[0].objective >= result.candidates[result.candidates.len() - 1].objective);
        assert_eq!(result.topology.stations.len(), 4);
    }

    #[test]
    fn routing_best_respects_capacity() {
        let result = run_logistics_routing_heuristics(LogisticsRoutingParams::default());
        assert!(result.best.feasible);
        // every route in the best plan must be within capacity.
        let cap = 7.0;
        for (_k, v) in &result.best.metrics {
            if let MetricValue::Num(n) = v {
                assert!(n.is_finite());
            }
        }
        // No capacity violation in the best plan.
        let violation = result
            .best
            .metrics
            .iter()
            .find(|(k, _)| k == "capacityViolation")
            .map(|(_, v)| matches!(v, MetricValue::Num(x) if *x == 0.0))
            .unwrap_or(false);
        assert!(violation);
        let _ = cap;
    }

    #[test]
    fn buyer_aware_pricing_produces_trace() {
        let result = run_buyer_aware_dynamic_pricing(BuyerAwareDynamicPricingParams::default());
        assert_eq!(result.candidates.len(), 5);
        let with_trace = result.candidates.iter().find(|c| c.trace.is_some()).unwrap();
        let trace = with_trace.trace.as_ref().unwrap();
        assert_eq!(trace.series.len(), 7);
        assert_eq!(trace.t.len(), 12);
    }

    #[test]
    fn energy_dispatch_runs_all_domains() {
        let _ = run_bottleneck_production_control(ManufacturingParams::default());
        let _ = run_supply_chain_risk_pooling(SupplyChainParams::default());
        let _ = run_workforce_service_operations(OperationsParams::default());
        let _ = run_portfolio_drawdown_control(FinancialControlParams::default());
        let _ = run_dynamic_pricing_revenue(RevenueManagementParams::default());
        let _ = run_energy_storage_dispatch(EnergyParams::default());
        let _ = run_active_learning_acquisition(ActiveLearningParams::default());
        let result = run_visual_decision_frontier(DecisionScienceParams::default());
        assert!(result.best.feasible);
    }
}
