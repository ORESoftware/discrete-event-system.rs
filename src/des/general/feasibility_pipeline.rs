//! Port of `src/des/general/feasibility-pipeline.ts` — a general optimization
//! feasibility checker pipeline, modelled as a runnable discrete-event system.
//!
//! A user supplies a structured optimization problem plus their incumbent
//! solution. The pipeline checks variable domains, linear constraints, and the
//! objective value, then optionally feeds evaluated candidates into a local
//! internal improver. The topology is:
//!
//! candidate-source and improver feed a domain checker, which feeds a
//! constraint checker, which feeds an objective evaluator, which feeds both the
//! solution sink and (back) the improver. A wall-clock checker station caps the
//! run.
//!
//! ## TS to Rust mapping
//!
//!   * `VariableKind` / `ConstraintSense` / `ObjectiveSense` string unions
//!     become enums; the `'user' | 'repair' | 'neighbor'` origin union becomes
//!     [`CandidateOrigin`]; the `kind: 'domain' | 'constraint'` violation tag
//!     becomes [`ViolationKind`]; the result `status` union becomes
//!     [`FeasibilityStatus`].
//!   * `Record<string, number>` for a candidate's variable assignment becomes a
//!     `HashMap<String, f64>` (only ever key-accessed or cloned). The objective
//!     / constraint `coefficients` (`Record<string, number>`) become an ordered
//!     `Vec<(String, f64)>` so the linear-combination summation order is
//!     deterministic and matches the TS object insertion order (a `HashMap`
//!     would give a nondeterministic summation order). (Flagged representation
//!     choice.)
//!   * `class *Token` / `class *Station` become structs (tokens carried as
//!     `Rc<dyn Any>`; stations implement the `DESStation` trait).
//!   * `mulberry32(seed)` in the local improver becomes an injected
//!     `RandomSource`.
//!   * `Math.round` (round half toward +infinity) is reproduced via
//!     [`js_round`] so integer repair / proposals match the TS exactly (Rust's
//!     `f64::round` rounds half away from zero, which differs for negative
//!     half-integers).
//!   * `Preconditions.*` throws become `panic!` (fatal invariant violations).
//!   * The wall-clock checker, stop channel, and stop-signal token are reused
//!     from [`internal_solver_network`](crate::des::general::internal_solver_network).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{
    run_iterative_des, IterativeRunOptions, IterativeRunSummary, RunReason,
};
use crate::des::general::des_base::station::{AnyToken, DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::{intrinsic_check, ValidationCheck};
use crate::des::general::internal_solver_network::{
    StopSignalPayload, StopSignalToken, WallClockCheckerStation, STOP_CHANNEL,
};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

// =============================================================================
// Channels
// =============================================================================

pub const CANDIDATE_CHANNEL: &str = "candidate";
pub const DOMAIN_CHANNEL: &str = "domain-checked";
pub const CONSTRAINT_CHANNEL: &str = "constraint-checked";
pub const EVALUATION_CHANNEL: &str = "evaluation";

const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

// =============================================================================
// Enums (TS string unions)
// =============================================================================

/// Variable domain kind. (TS `type VariableKind`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableKind {
    Continuous,
    Integer,
    Binary,
}

/// Constraint sense. (TS `type ConstraintSense = '<=' | '>=' | '='`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintSense {
    Le,
    Ge,
    Eq,
}

impl ConstraintSense {
    pub fn as_str(self) -> &'static str {
        match self {
            ConstraintSense::Le => "<=",
            ConstraintSense::Ge => ">=",
            ConstraintSense::Eq => "=",
        }
    }
}

/// Objective optimization direction. (TS `type ObjectiveSense = 'min' | 'max'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Min,
    Max,
}

/// How a candidate was produced. (TS `'user' | 'repair' | 'neighbor'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateOrigin {
    User,
    Repair,
    Neighbor,
}

impl CandidateOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            CandidateOrigin::User => "user",
            CandidateOrigin::Repair => "repair",
            CandidateOrigin::Neighbor => "neighbor",
        }
    }
}

/// Whether a violation is a domain or constraint violation. (TS `kind`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    Domain,
    Constraint,
}

/// Pipeline-node role. (TS `'source' | 'checker' | 'evaluator' | 'improver' |
/// 'sink'`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeasibilityNodeRole {
    Source,
    Checker,
    Evaluator,
    Improver,
    Sink,
}

/// Final pipeline status. (TS result `status` union.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeasibilityStatus {
    Feasible,
    Infeasible,
    Improved,
    InfeasibleImproved,
    TimeLimit,
    TickLimit,
}

// =============================================================================
// Problem / candidate structs
// =============================================================================

/// A decision variable. (TS `interface OptimizationVariable`.)
#[derive(Clone, Debug)]
pub struct OptimizationVariable {
    pub name: String,
    pub kind: Option<VariableKind>,
    pub lb: Option<f64>,
    pub ub: Option<f64>,
    pub step: Option<f64>,
}

/// Linear objective. (TS `interface LinearObjective`.) `coefficients` keeps
/// insertion order (see module docs).
#[derive(Clone, Debug)]
pub struct LinearObjective {
    pub constant: Option<f64>,
    pub coefficients: Vec<(String, f64)>,
}

/// Linear constraint. (TS `interface LinearConstraint`.)
#[derive(Clone, Debug)]
pub struct LinearConstraint {
    pub name: Option<String>,
    pub coefficients: Vec<(String, f64)>,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub tolerance: Option<f64>,
}

/// The structured problem. (TS `interface StructuredOptimizationProblem`.)
#[derive(Clone, Debug)]
pub struct StructuredOptimizationProblem {
    pub sense: ObjectiveSense,
    pub variables: Vec<OptimizationVariable>,
    pub objective: LinearObjective,
    pub constraints: Option<Vec<LinearConstraint>>,
    pub tolerance: Option<f64>,
}

/// The incumbent solution input. (TS `interface CandidateSolutionInput`.)
#[derive(Clone, Debug, Default)]
pub struct CandidateSolutionInput {
    pub id: Option<String>,
    pub values: Option<HashMap<String, f64>>,
    pub vector: Option<Vec<f64>>,
}

/// Local-improver options. (TS `interface FeasibilityImprovementOptions`.)
#[derive(Clone, Debug, Default)]
pub struct FeasibilityImprovementOptions {
    pub enabled: Option<bool>,
    pub max_iterations: Option<usize>,
    pub seed: Option<u32>,
    pub continuous_step: Option<f64>,
    pub integer_step: Option<f64>,
    pub penalty: Option<f64>,
    pub allow_repair: Option<bool>,
}

/// Top-level pipeline params. (TS `interface FeasibilityPipelineParams`.)
#[derive(Clone, Debug)]
pub struct FeasibilityPipelineParams {
    pub problem: StructuredOptimizationProblem,
    pub candidate: CandidateSolutionInput,
    pub improvement: Option<FeasibilityImprovementOptions>,
    pub time_limit_ms: Option<f64>,
    pub max_ticks: Option<usize>,
    pub check_every_ticks: Option<usize>,
}

/// A candidate flowing through the pipeline. (TS `interface CandidatePayload`.)
#[derive(Clone, Debug)]
pub struct CandidatePayload {
    pub id: String,
    pub parent_id: Option<String>,
    pub iteration: usize,
    pub origin: CandidateOrigin,
    pub values: HashMap<String, f64>,
}

/// A single domain or constraint violation. (TS `interface
/// FeasibilityViolation`.)
#[derive(Clone, Debug)]
pub struct FeasibilityViolation {
    pub kind: ViolationKind,
    pub name: String,
    pub violation: f64,
    pub message: String,
    pub variable: Option<String>,
    pub constraint: Option<String>,
    pub activity: Option<f64>,
    pub rhs: Option<f64>,
}

/// A fully evaluated candidate. (TS `interface FeasibilityEvaluation`.)
#[derive(Clone, Debug)]
pub struct FeasibilityEvaluation {
    pub candidate_id: String,
    pub parent_id: Option<String>,
    pub iteration: usize,
    pub origin: CandidateOrigin,
    pub values: HashMap<String, f64>,
    pub objective_value: f64,
    pub comparable_objective: f64,
    pub total_violation: f64,
    pub max_violation: f64,
    pub feasible: bool,
    pub domain_violations: Vec<FeasibilityViolation>,
    pub constraint_violations: Vec<FeasibilityViolation>,
    pub violations: Vec<FeasibilityViolation>,
    pub merit: f64,
}

// =============================================================================
// Network description structs
// =============================================================================

#[derive(Clone, Debug)]
pub struct FeasibilityPipelineNode {
    pub id: String,
    pub kind: String,
    pub role: FeasibilityNodeRole,
}

#[derive(Clone, Debug)]
pub struct FeasibilityPipelineMovingEntity {
    pub id: String,
    pub kind: String,
    pub token_type: String,
}

#[derive(Clone, Debug)]
pub struct FeasibilityPipelineEdge {
    pub from: String,
    pub to: String,
    pub moving_entity: String,
    pub channel: String,
}

#[derive(Clone, Debug)]
pub struct FeasibilityPipelineNetwork {
    pub stationary_entities: Vec<FeasibilityPipelineNode>,
    pub moving_entities: Vec<FeasibilityPipelineMovingEntity>,
    pub edges: Vec<FeasibilityPipelineEdge>,
}

/// Wall-clock accounting block.
#[derive(Clone, Debug)]
pub struct WallClockReport {
    pub budget_ms: f64,
    pub elapsed_ms: f64,
    pub checks: usize,
    pub expired: bool,
}

/// Final pipeline result. (TS `interface FeasibilityPipelineResult`.)
#[derive(Clone, Debug)]
pub struct FeasibilityPipelineResult {
    pub status: FeasibilityStatus,
    pub initial: FeasibilityEvaluation,
    pub best: FeasibilityEvaluation,
    pub trace: Vec<FeasibilityEvaluation>,
    pub improvements: Vec<FeasibilityEvaluation>,
    pub stop_signals: Vec<StopSignalPayload>,
    pub wall_clock: WallClockReport,
    pub run_summary: IterativeRunSummary,
    pub network: FeasibilityPipelineNetwork,
    pub validation: Vec<ValidationCheck>,
}

// =============================================================================
// Tokens
// =============================================================================

pub struct CandidateToken {
    pub payload: CandidatePayload,
}

impl CandidateToken {
    pub fn new(payload: CandidatePayload) -> Self {
        CandidateToken { payload }
    }
}

pub struct DomainCheckedToken {
    pub candidate: CandidatePayload,
    pub domain_violations: Vec<FeasibilityViolation>,
}

impl DomainCheckedToken {
    pub fn new(candidate: CandidatePayload, domain_violations: Vec<FeasibilityViolation>) -> Self {
        DomainCheckedToken { candidate, domain_violations }
    }
}

pub struct ConstraintCheckedToken {
    pub candidate: CandidatePayload,
    pub domain_violations: Vec<FeasibilityViolation>,
    pub constraint_violations: Vec<FeasibilityViolation>,
}

impl ConstraintCheckedToken {
    pub fn new(
        candidate: CandidatePayload,
        domain_violations: Vec<FeasibilityViolation>,
        constraint_violations: Vec<FeasibilityViolation>,
    ) -> Self {
        ConstraintCheckedToken { candidate, domain_violations, constraint_violations }
    }
}

pub struct FeasibilityEvaluationToken {
    pub payload: FeasibilityEvaluation,
}

impl FeasibilityEvaluationToken {
    pub fn new(payload: FeasibilityEvaluation) -> Self {
        FeasibilityEvaluationToken { payload }
    }
}

// =============================================================================
// Stations
// =============================================================================

/// Emits the user's candidate exactly once. (TS `class
/// CandidateSourceStation`.)
pub struct CandidateSourceStation {
    core: StationCore,
    candidate: CandidatePayload,
    emitted: bool,
}

impl CandidateSourceStation {
    pub fn new(id: impl Into<String>, candidate: CandidatePayload) -> Self {
        CandidateSourceStation { core: StationCore::new(id), candidate, emitted: false }
    }
}

impl DESStation for CandidateSourceStation {
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
        let token: AnyToken = Rc::new(CandidateToken::new(self.candidate.clone()));
        self.core.emit(token, CANDIDATE_CHANNEL);
        self.emitted = true;
    }
}

/// Checks variable domains. (TS `class DomainCheckerStation`.)
pub struct DomainCheckerStation {
    core: StationCore,
    problem: StructuredOptimizationProblem,
}

impl DomainCheckerStation {
    pub fn new(id: impl Into<String>, problem: StructuredOptimizationProblem) -> Self {
        let mut st = DomainCheckerStation { core: StationCore::new(id), problem };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "domain-checker.problem-has-variables",
                |s| !downcast::<DomainCheckerStation>(s).problem.variables.is_empty(),
                Some("at least one variable".to_string()),
                Some(Box::new(|s| {
                    downcast::<DomainCheckerStation>(s).problem.variables.len().to_string()
                })),
                Some("feasibility-pipeline".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }
}

impl DESStation for DomainCheckerStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        validate_problem(&self.problem);
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CANDIDATE_CHANNEL) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<CandidateToken>(CANDIDATE_CHANNEL) {
            let domain = check_domain(&self.problem, &token.payload.values);
            let out: AnyToken = Rc::new(DomainCheckedToken::new(token.payload.clone(), domain));
            self.core.emit(out, DOMAIN_CHANNEL);
        }
    }
}

/// Checks linear constraints. (TS `class ConstraintCheckerStation`.)
pub struct ConstraintCheckerStation {
    core: StationCore,
    problem: StructuredOptimizationProblem,
}

impl ConstraintCheckerStation {
    pub fn new(id: impl Into<String>, problem: StructuredOptimizationProblem) -> Self {
        ConstraintCheckerStation { core: StationCore::new(id), problem }
    }
}

impl DESStation for ConstraintCheckerStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        validate_problem(&self.problem);
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(DOMAIN_CHANNEL) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<DomainCheckedToken>(DOMAIN_CHANNEL) {
            let constraint = check_constraints(&self.problem, &token.candidate.values);
            let out: AnyToken = Rc::new(ConstraintCheckedToken::new(
                token.candidate.clone(),
                token.domain_violations.clone(),
                constraint,
            ));
            self.core.emit(out, CONSTRAINT_CHANNEL);
        }
    }
}

/// Computes objective + merit, producing an evaluation. (TS `class
/// ObjectiveEvaluatorStation`.)
pub struct ObjectiveEvaluatorStation {
    core: StationCore,
    problem: StructuredOptimizationProblem,
    penalty: f64,
}

impl ObjectiveEvaluatorStation {
    pub fn new(id: impl Into<String>, problem: StructuredOptimizationProblem, penalty: f64) -> Self {
        let mut st = ObjectiveEvaluatorStation { core: StationCore::new(id), problem, penalty };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "objective-evaluator.penalty-positive",
                |s| downcast::<ObjectiveEvaluatorStation>(s).penalty > 0.0,
                Some("penalty > 0".to_string()),
                Some(Box::new(|s| downcast::<ObjectiveEvaluatorStation>(s).penalty.to_string())),
                Some("feasibility-pipeline".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }
}

impl DESStation for ObjectiveEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        validate_problem(&self.problem);
        Preconditions::positive("ObjectiveEvaluatorStation", "penalty", self.penalty)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(CONSTRAINT_CHANNEL) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<ConstraintCheckedToken>(CONSTRAINT_CHANNEL) {
            let eval = finalize_evaluation(
                &self.problem,
                &token.candidate,
                token.domain_violations.clone(),
                token.constraint_violations.clone(),
                self.penalty,
            );
            let out: AnyToken = Rc::new(FeasibilityEvaluationToken::new(eval));
            self.core.emit(out, EVALUATION_CHANNEL);
        }
    }
}

/// Local improver: repairs, then proposes neighbours. (TS `class
/// ImprovementStation`.)
pub struct ImprovementStation {
    core: StationCore,
    problem: StructuredOptimizationProblem,
    enabled: bool,
    max_iterations: usize,
    continuous_step: f64,
    integer_step: f64,
    allow_repair: bool,
    rng: Box<dyn RandomSource>,
    initialized: bool,
    done: bool,
    waiting: bool,
    repair_tried: bool,
    proposal_count: usize,
    best_eval: Option<FeasibilityEvaluation>,
}

impl ImprovementStation {
    pub fn new(
        id: impl Into<String>,
        problem: StructuredOptimizationProblem,
        opts: FeasibilityImprovementOptions,
    ) -> Self {
        let mut st = ImprovementStation {
            core: StationCore::new(id),
            problem,
            enabled: opts.enabled.unwrap_or(true),
            max_iterations: opts.max_iterations.unwrap_or(200),
            continuous_step: opts.continuous_step.unwrap_or(1.0),
            integer_step: opts.integer_step.unwrap_or(1.0),
            allow_repair: opts.allow_repair.unwrap_or(true),
            rng: Box::new(mulberry32(opts.seed.unwrap_or(1))),
            initialized: false,
            done: false,
            waiting: false,
            repair_tried: false,
            proposal_count: 0,
            best_eval: None,
        };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "improvement-station.best-evaluation-exists",
                |s| downcast::<ImprovementStation>(s).best_eval.is_some(),
                Some("at least one evaluated candidate".to_string()),
                Some(Box::new(|s| {
                    let st = downcast::<ImprovementStation>(s);
                    match &st.best_eval {
                        Some(b) => b.candidate_id.clone(),
                        None => "missing".to_string(),
                    }
                })),
                Some("feasibility-pipeline".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    fn emit_candidate(&mut self, values: HashMap<String, f64>, origin: CandidateOrigin) {
        let candidate = CandidatePayload {
            id: format!("{}-{}", origin.as_str(), self.proposal_count + 1),
            parent_id: self.best_eval.as_ref().map(|b| b.candidate_id.clone()),
            iteration: self.proposal_count + 1,
            origin,
            values,
        };
        self.proposal_count += 1;
        self.waiting = true;
        let token: AnyToken = Rc::new(CandidateToken::new(candidate));
        self.core.emit(token, CANDIDATE_CHANNEL);
    }
}

impl DESStation for ImprovementStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn assert_preconditions(&mut self) {
        validate_problem(&self.problem);
        Preconditions::integer_in_range(
            "ImprovementStation",
            "maxIterations",
            self.max_iterations as f64,
            0.0,
            MAX_SAFE_INTEGER,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        Preconditions::positive("ImprovementStation", "continuousStep", self.continuous_step)
            .unwrap_or_else(|e| panic!("{e}"));
        Preconditions::positive("ImprovementStation", "integerStep", self.integer_step)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    fn has_work(&self) -> bool {
        if self.core.inbox_size(EVALUATION_CHANNEL) > 0 {
            return true;
        }
        if !self.enabled || self.done || !self.initialized || self.waiting {
            return false;
        }
        self.proposal_count < self.max_iterations || (self.allow_repair && !self.repair_tried)
    }

    fn run_time_step(&mut self) {
        for token in self.core.drain::<FeasibilityEvaluationToken>(EVALUATION_CHANNEL) {
            self.waiting = false;
            self.initialized = true;
            let better = match &self.best_eval {
                None => true,
                Some(b) => evaluation_better(&token.payload, b, &self.problem),
            };
            if better {
                self.best_eval = Some(token.payload.clone());
            }
        }
        if !self.enabled || self.done || self.best_eval.is_none() || self.waiting {
            return;
        }
        if self.allow_repair && !self.repair_tried {
            self.repair_tried = true;
            let best_values = self.best_eval.as_ref().unwrap().values.clone();
            let repaired = repair_values(&self.problem, &best_values);
            if !same_values(&self.problem, &repaired, &best_values) {
                self.emit_candidate(repaired, CandidateOrigin::Repair);
                return;
            }
        }
        if self.proposal_count >= self.max_iterations {
            self.done = true;
            return;
        }
        let best_values = self.best_eval.as_ref().unwrap().values.clone();
        let neighbor = propose_neighbor(
            &self.problem,
            &best_values,
            &mut *self.rng,
            self.continuous_step,
            self.integer_step,
        );
        self.emit_candidate(neighbor, CandidateOrigin::Neighbor);
    }
}

/// Collects evaluations and stop signals. (TS `class FeasibilitySinkStation`.)
pub struct FeasibilitySinkStation {
    core: StationCore,
    problem: StructuredOptimizationProblem,
    pub trace: Vec<FeasibilityEvaluation>,
    pub stops: Vec<StopSignalPayload>,
}

impl FeasibilitySinkStation {
    pub fn new(id: impl Into<String>, problem: StructuredOptimizationProblem) -> Self {
        let mut st = FeasibilitySinkStation {
            core: StationCore::new(id),
            problem,
            trace: Vec::new(),
            stops: Vec::new(),
        };
        st.add_validator(
            intrinsic_check::<dyn DESStation>(
                "feasibility-sink.trace-nonempty",
                |s| !downcast::<FeasibilitySinkStation>(s).trace.is_empty(),
                Some("at least one evaluation".to_string()),
                Some(Box::new(|s| downcast::<FeasibilitySinkStation>(s).trace.len().to_string())),
                Some("feasibility-pipeline".to_string()),
                None,
            )
            .boxed(),
        );
        st
    }

    pub fn best(&self) -> Option<FeasibilityEvaluation> {
        let mut best: Option<&FeasibilityEvaluation> = None;
        for row in &self.trace {
            match best {
                None => best = Some(row),
                Some(b) if evaluation_better(row, b, &self.problem) => best = Some(row),
                _ => {}
            }
        }
        best.cloned()
    }
}

impl DESStation for FeasibilitySinkStation {
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
        false
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<FeasibilityEvaluationToken>(EVALUATION_CHANNEL) {
            self.trace.push(token.payload.clone());
        }
        for token in self.core.drain::<StopSignalToken>(STOP_CHANNEL) {
            self.stops.push(token.payload.clone());
        }
    }
}

// =============================================================================
// Driver
// =============================================================================

/// Run the full feasibility pipeline. (TS `runFeasibilityPipeline`.)
pub fn run_feasibility_pipeline(params: FeasibilityPipelineParams) -> FeasibilityPipelineResult {
    validate_problem(&params.problem);
    let improvement = params.improvement.clone().unwrap_or_default();
    let penalty = improvement.penalty.unwrap_or(1_000_000.0);

    let source = Rc::new(RefCell::new(CandidateSourceStation::new(
        "candidate-source",
        candidate_payload_from_input(&params.problem, &params.candidate),
    )));
    let domain = Rc::new(RefCell::new(DomainCheckerStation::new("domain-checker", params.problem.clone())));
    let constraints = Rc::new(RefCell::new(ConstraintCheckerStation::new(
        "constraint-checker",
        params.problem.clone(),
    )));
    let objective = Rc::new(RefCell::new(ObjectiveEvaluatorStation::new(
        "objective-evaluator",
        params.problem.clone(),
        penalty,
    )));
    let improver = Rc::new(RefCell::new(ImprovementStation::new(
        "improvement-station",
        params.problem.clone(),
        improvement.clone(),
    )));
    let sink = Rc::new(RefCell::new(FeasibilitySinkStation::new("feasibility-sink", params.problem.clone())));
    let budget_ms = params.time_limit_ms.unwrap_or(180000.0);
    let checker = Rc::new(RefCell::new(WallClockCheckerStation::new(
        "wall-clock-checker",
        budget_ms,
        params.check_every_ticks.unwrap_or(1),
        None,
    )));

    source.borrow_mut().core_mut().pipe(domain.clone() as StationRef, CANDIDATE_CHANNEL, CANDIDATE_CHANNEL);
    improver.borrow_mut().core_mut().pipe(domain.clone() as StationRef, CANDIDATE_CHANNEL, CANDIDATE_CHANNEL);
    domain.borrow_mut().core_mut().pipe(constraints.clone() as StationRef, DOMAIN_CHANNEL, DOMAIN_CHANNEL);
    constraints.borrow_mut().core_mut().pipe(objective.clone() as StationRef, CONSTRAINT_CHANNEL, CONSTRAINT_CHANNEL);
    objective.borrow_mut().core_mut().pipe(sink.clone() as StationRef, EVALUATION_CHANNEL, EVALUATION_CHANNEL);
    objective.borrow_mut().core_mut().pipe(improver.clone() as StationRef, EVALUATION_CHANNEL, EVALUATION_CHANNEL);
    checker.borrow_mut().core_mut().pipe(sink.clone() as StationRef, STOP_CHANNEL, STOP_CHANNEL);

    let max_ticks = params.max_ticks.unwrap_or_else(|| default_max_ticks(&improvement));
    let stations: Vec<StationRef> = vec![
        source as StationRef,
        domain as StationRef,
        constraints as StationRef,
        objective as StationRef,
        improver as StationRef,
        checker.clone() as StationRef,
        sink.clone() as StationRef,
    ];
    let checker_for_stop = checker.clone();
    let summary = run_iterative_des(
        stations,
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            stop_when: Some(Box::new(move |_, _| checker_for_stop.borrow().expired())),
            ..Default::default()
        },
    );

    let initial = sink
        .borrow()
        .trace
        .first()
        .cloned()
        .unwrap_or_else(|| evaluate_candidate(&params.problem, &params.candidate, penalty));
    let best = sink.borrow().best().unwrap_or_else(|| initial.clone());
    let improvements: Vec<FeasibilityEvaluation> = sink
        .borrow()
        .trace
        .iter()
        .filter(|row| row.candidate_id != initial.candidate_id && evaluation_better(row, &initial, &params.problem))
        .cloned()
        .collect();

    let expired = checker.borrow().expired();
    let status = pipeline_status(&params.problem, summary.reason, expired, &initial, &best);
    let wall_clock = WallClockReport {
        budget_ms,
        elapsed_ms: checker.borrow().elapsed_ms(),
        checks: checker.borrow().num_checks(),
        expired,
    };
    let validation = summary.validation.clone().unwrap_or_default();
    let trace = sink.borrow().trace.clone();
    let stop_signals = sink.borrow().stops.clone();

    FeasibilityPipelineResult {
        status,
        initial,
        best,
        trace,
        improvements,
        stop_signals,
        wall_clock,
        run_summary: summary,
        network: describe_feasibility_pipeline_network(),
        validation,
    }
}

/// Evaluate a single candidate without running the pipeline. (TS
/// `evaluateCandidate`.)
pub fn evaluate_candidate(
    problem: &StructuredOptimizationProblem,
    candidate: &CandidateSolutionInput,
    penalty: f64,
) -> FeasibilityEvaluation {
    validate_problem(problem);
    let payload = candidate_payload_from_input(problem, candidate);
    let domain = check_domain(problem, &payload.values);
    let constraint = check_constraints(problem, &payload.values);
    finalize_evaluation(problem, &payload, domain, constraint, penalty)
}

// =============================================================================
// Free helpers
// =============================================================================

fn candidate_payload_from_input(
    problem: &StructuredOptimizationProblem,
    input: &CandidateSolutionInput,
) -> CandidatePayload {
    let mut values: HashMap<String, f64> = HashMap::new();
    if let Some(iv) = &input.values {
        for v in &problem.variables {
            values.insert(v.name.clone(), iv.get(&v.name).copied().unwrap_or(f64::NAN));
        }
    } else if let Some(vec) = &input.vector {
        for (i, v) in problem.variables.iter().enumerate() {
            values.insert(v.name.clone(), vec.get(i).copied().unwrap_or(f64::NAN));
        }
    } else {
        for v in &problem.variables {
            values.insert(v.name.clone(), f64::NAN);
        }
    }
    CandidatePayload {
        id: input.id.clone().unwrap_or_else(|| "user-candidate".to_string()),
        parent_id: None,
        iteration: 0,
        origin: CandidateOrigin::User,
        values,
    }
}

fn check_domain(problem: &StructuredOptimizationProblem, values: &HashMap<String, f64>) -> Vec<FeasibilityViolation> {
    let tol = problem.tolerance.unwrap_or(1e-8);
    let mut out = Vec::new();
    for v in &problem.variables {
        let x = values.get(&v.name).copied().unwrap_or(f64::NAN);
        let lb = lower_bound(v);
        let ub = upper_bound(v);
        if !x.is_finite() {
            out.push(FeasibilityViolation {
                kind: ViolationKind::Domain,
                name: format!("{}.finite", v.name),
                variable: Some(v.name.clone()),
                violation: f64::INFINITY,
                message: format!("{} is missing or not finite", v.name),
                constraint: None,
                activity: None,
                rhs: None,
            });
            continue;
        }
        if x < lb - tol {
            out.push(FeasibilityViolation {
                kind: ViolationKind::Domain,
                name: format!("{}.lb", v.name),
                variable: Some(v.name.clone()),
                violation: lb - x,
                message: format!("{}={} below lower bound {}", v.name, x, lb),
                constraint: None,
                activity: None,
                rhs: None,
            });
        }
        if x > ub + tol {
            out.push(FeasibilityViolation {
                kind: ViolationKind::Domain,
                name: format!("{}.ub", v.name),
                variable: Some(v.name.clone()),
                violation: x - ub,
                message: format!("{}={} above upper bound {}", v.name, x, ub),
                constraint: None,
                activity: None,
                rhs: None,
            });
        }
        if matches!(v.kind, Some(VariableKind::Integer) | Some(VariableKind::Binary))
            && (x - js_round(x)).abs() > tol
        {
            out.push(FeasibilityViolation {
                kind: ViolationKind::Domain,
                name: format!("{}.integer", v.name),
                variable: Some(v.name.clone()),
                violation: (x - js_round(x)).abs(),
                message: format!("{}={} is not integral", v.name, x),
                constraint: None,
                activity: None,
                rhs: None,
            });
        }
        if v.kind == Some(VariableKind::Binary) && x.abs().min((x - 1.0).abs()) > tol {
            out.push(FeasibilityViolation {
                kind: ViolationKind::Domain,
                name: format!("{}.binary", v.name),
                variable: Some(v.name.clone()),
                violation: x.abs().min((x - 1.0).abs()),
                message: format!("{}={} is not binary", v.name, x),
                constraint: None,
                activity: None,
                rhs: None,
            });
        }
    }
    out
}

fn check_constraints(
    problem: &StructuredOptimizationProblem,
    values: &HashMap<String, f64>,
) -> Vec<FeasibilityViolation> {
    let empty: Vec<LinearConstraint> = Vec::new();
    let constraints = problem.constraints.as_ref().unwrap_or(&empty);
    let mut out = Vec::new();
    for (i, c) in constraints.iter().enumerate() {
        let tol = c.tolerance.or(problem.tolerance).unwrap_or(1e-8);
        let activity = evaluate_linear(&c.coefficients, values, 0.0);
        let violation = match c.sense {
            ConstraintSense::Le => (activity - c.rhs - tol).max(0.0),
            ConstraintSense::Ge => (c.rhs - activity - tol).max(0.0),
            ConstraintSense::Eq => ((activity - c.rhs).abs() - tol).max(0.0),
        };
        if violation > 0.0 {
            let name = c.name.clone().unwrap_or_else(|| format!("constraint-{i}"));
            out.push(FeasibilityViolation {
                kind: ViolationKind::Constraint,
                name: name.clone(),
                constraint: Some(name.clone()),
                activity: Some(activity),
                rhs: Some(c.rhs),
                violation,
                message: format!(
                    "{}: activity {} {} {} violated by {}",
                    name,
                    activity,
                    c.sense.as_str(),
                    c.rhs,
                    violation
                ),
                variable: None,
            });
        }
    }
    out
}

fn finalize_evaluation(
    problem: &StructuredOptimizationProblem,
    candidate: &CandidatePayload,
    domain_violations: Vec<FeasibilityViolation>,
    constraint_violations: Vec<FeasibilityViolation>,
    penalty: f64,
) -> FeasibilityEvaluation {
    let objective_value =
        evaluate_linear(&problem.objective.coefficients, &candidate.values, problem.objective.constant.unwrap_or(0.0));
    let comparable_objective =
        if problem.sense == ObjectiveSense::Min { objective_value } else { -objective_value };
    let mut violations = domain_violations.clone();
    violations.extend(constraint_violations.iter().cloned());
    let total_violation: f64 = violations.iter().map(|v| safe_violation(v.violation)).sum();
    let max_violation = violations.iter().map(|v| safe_violation(v.violation)).fold(0.0, f64::max);
    FeasibilityEvaluation {
        candidate_id: candidate.id.clone(),
        parent_id: candidate.parent_id.clone(),
        iteration: candidate.iteration,
        origin: candidate.origin,
        values: candidate.values.clone(),
        objective_value,
        comparable_objective,
        total_violation,
        max_violation,
        feasible: violations.is_empty(),
        domain_violations,
        constraint_violations,
        violations,
        merit: total_violation * penalty + comparable_objective,
    }
}

fn evaluate_linear(coefficients: &[(String, f64)], values: &HashMap<String, f64>, constant: f64) -> f64 {
    let mut out = constant;
    for (name, coeff) in coefficients {
        out += coeff * values.get(name).copied().unwrap_or(f64::NAN);
    }
    out
}

fn safe_violation(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        1e12
    }
}

fn evaluation_better(
    a: &FeasibilityEvaluation,
    b: &FeasibilityEvaluation,
    problem: &StructuredOptimizationProblem,
) -> bool {
    let tol = problem.tolerance.unwrap_or(1e-8);
    if a.feasible && !b.feasible {
        return true;
    }
    if !a.feasible && b.feasible {
        return false;
    }
    if a.feasible && b.feasible {
        return a.comparable_objective < b.comparable_objective - tol;
    }
    if a.total_violation < b.total_violation - tol {
        return true;
    }
    if (a.total_violation - b.total_violation).abs() <= tol && a.merit < b.merit - tol {
        return true;
    }
    false
}

fn repair_values(
    problem: &StructuredOptimizationProblem,
    input: &HashMap<String, f64>,
) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for v in &problem.variables {
        let mut x = input.get(&v.name).copied().unwrap_or(f64::NAN);
        let lb = lower_bound(v);
        let ub = upper_bound(v);
        if !x.is_finite() {
            if lb.is_finite() && ub.is_finite() {
                x = (lb + ub) / 2.0;
            } else if lb.is_finite() {
                x = lb;
            } else if ub.is_finite() {
                x = ub;
            } else {
                x = 0.0;
            }
        }
        x = clamp(x, lb, ub);
        if v.kind == Some(VariableKind::Binary) {
            x = if x >= 0.5 { 1.0 } else { 0.0 };
        } else if v.kind == Some(VariableKind::Integer) {
            x = js_round(x);
        }
        out.insert(v.name.clone(), clamp(x, lb, ub));
    }
    out
}

fn propose_neighbor(
    problem: &StructuredOptimizationProblem,
    input: &HashMap<String, f64>,
    rng: &mut dyn RandomSource,
    continuous_step: f64,
    integer_step: f64,
) -> HashMap<String, f64> {
    let base = repair_values(problem, input);
    let variables = &problem.variables;
    let binary: Vec<&OptimizationVariable> =
        variables.iter().filter(|v| v.kind == Some(VariableKind::Binary)).collect();
    if binary.len() >= 2 && rng.next_float() < 0.5 {
        let ones: Vec<&OptimizationVariable> =
            binary.iter().copied().filter(|v| base[&v.name] >= 0.5).collect();
        let zeros: Vec<&OptimizationVariable> =
            binary.iter().copied().filter(|v| base[&v.name] < 0.5).collect();
        if !ones.is_empty() && !zeros.is_empty() {
            let mut out = base.clone();
            let drop = ones[(rng.next_float() * ones.len() as f64).floor() as usize];
            let add = zeros[(rng.next_float() * zeros.len() as f64).floor() as usize];
            out.insert(drop.name.clone(), 0.0);
            out.insert(add.name.clone(), 1.0);
            return out;
        }
    }
    for _attempt in 0..variables.len() {
        let v = &variables[(rng.next_float() * variables.len() as f64).floor() as usize];
        let mut out = base.clone();
        let lb = lower_bound(v);
        let ub = upper_bound(v);
        if v.kind == Some(VariableKind::Binary) {
            let nv = if base[&v.name] >= 0.5 { 0.0 } else { 1.0 };
            out.insert(v.name.clone(), nv);
        } else {
            let sign = if rng.next_float() < 0.5 { -1.0 } else { 1.0 };
            let step = v.step.unwrap_or(if v.kind == Some(VariableKind::Integer) {
                integer_step
            } else {
                continuous_step
            });
            let mut nv = base[&v.name] + sign * step;
            if v.kind == Some(VariableKind::Integer) {
                nv = js_round(nv);
            }
            out.insert(v.name.clone(), nv);
        }
        let clamped = clamp(out[&v.name], lb, ub);
        out.insert(v.name.clone(), clamped);
        if !same_values(problem, &out, &base) {
            return out;
        }
    }
    base
}

fn same_values(
    problem: &StructuredOptimizationProblem,
    a: &HashMap<String, f64>,
    b: &HashMap<String, f64>,
) -> bool {
    let tol = problem.tolerance.unwrap_or(1e-8);
    problem.variables.iter().all(|v| {
        let av = a.get(&v.name).copied().unwrap_or(f64::NAN);
        let bv = b.get(&v.name).copied().unwrap_or(f64::NAN);
        (av - bv).abs() <= tol
    })
}

fn lower_bound(v: &OptimizationVariable) -> f64 {
    if v.kind == Some(VariableKind::Binary) {
        return v.lb.unwrap_or(0.0).max(0.0);
    }
    v.lb.unwrap_or(f64::NEG_INFINITY)
}

fn upper_bound(v: &OptimizationVariable) -> f64 {
    if v.kind == Some(VariableKind::Binary) {
        return v.ub.unwrap_or(1.0).min(1.0);
    }
    v.ub.unwrap_or(f64::INFINITY)
}

fn clamp(x: f64, lb: f64, ub: f64) -> f64 {
    ub.min(x.max(lb))
}

/// JavaScript `Math.round` semantics (round half toward +infinity).
fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

fn validate_problem(problem: &StructuredOptimizationProblem) {
    Preconditions::non_empty("FeasibilityPipeline", "variables", &problem.variables)
        .unwrap_or_else(|e| panic!("{e}"));
    // `sense` is a typed enum (always min or max), so the TS sense check is a
    // no-op here.
    let mut names: HashSet<String> = HashSet::new();
    for v in &problem.variables {
        Preconditions::check(
            "FeasibilityPipeline",
            "variable.name",
            "be non-empty",
            !v.name.is_empty(),
            Some(v.name.clone()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        Preconditions::check(
            "FeasibilityPipeline",
            &format!("variable.{}.unique", v.name),
            "be unique",
            !names.contains(&v.name),
            Some(v.name.clone()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        names.insert(v.name.clone());
        if let Some(lb) = v.lb {
            Preconditions::finite("FeasibilityPipeline", &format!("{}.lb", v.name), lb)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        if let Some(ub) = v.ub {
            Preconditions::finite("FeasibilityPipeline", &format!("{}.ub", v.name), ub)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        Preconditions::check(
            "FeasibilityPipeline",
            &format!("{}.bounds", v.name),
            "satisfy lb <= ub",
            lower_bound(v) <= upper_bound(v),
            None,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        if let Some(step) = v.step {
            Preconditions::positive("FeasibilityPipeline", &format!("{}.step", v.name), step)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // `kind` is a typed enum, so the TS type-validity check is a no-op.
    }
    validate_coefficients("objective.coefficients", &problem.objective.coefficients, &names);
    if let Some(c) = problem.objective.constant {
        Preconditions::finite("FeasibilityPipeline", "objective.constant", c).unwrap_or_else(|e| panic!("{e}"));
    }
    let empty: Vec<LinearConstraint> = Vec::new();
    let constraints = problem.constraints.as_ref().unwrap_or(&empty);
    for (i, c) in constraints.iter().enumerate() {
        // `sense` is a typed enum, so the TS sense check is a no-op.
        Preconditions::finite("FeasibilityPipeline", &format!("constraints[{i}].rhs"), c.rhs)
            .unwrap_or_else(|e| panic!("{e}"));
        if let Some(t) = c.tolerance {
            Preconditions::non_negative("FeasibilityPipeline", &format!("constraints[{i}].tolerance"), t)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        validate_coefficients(&format!("constraints[{i}].coefficients"), &c.coefficients, &names);
    }
    if let Some(t) = problem.tolerance {
        Preconditions::non_negative("FeasibilityPipeline", "tolerance", t).unwrap_or_else(|e| panic!("{e}"));
    }
}

fn validate_coefficients(param: &str, coeffs: &[(String, f64)], variable_names: &HashSet<String>) {
    for (name, coeff) in coeffs {
        Preconditions::check(
            "FeasibilityPipeline",
            &format!("{param}.{name}"),
            "reference a declared variable",
            variable_names.contains(name),
            Some(name.clone()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        Preconditions::finite("FeasibilityPipeline", &format!("{param}.{name}"), *coeff)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

fn default_max_ticks(opts: &FeasibilityImprovementOptions) -> usize {
    if opts.enabled.unwrap_or(true) {
        opts.max_iterations.unwrap_or(200) * 4 + 16
    } else {
        16
    }
}

fn pipeline_status(
    problem: &StructuredOptimizationProblem,
    reason: Option<RunReason>,
    expired: bool,
    initial: &FeasibilityEvaluation,
    best: &FeasibilityEvaluation,
) -> FeasibilityStatus {
    if expired {
        return FeasibilityStatus::TimeLimit;
    }
    if reason == Some(RunReason::MaxTicks) {
        return FeasibilityStatus::TickLimit;
    }
    let improved = best.candidate_id != initial.candidate_id && evaluation_better(best, initial, problem);
    if best.feasible && improved {
        return FeasibilityStatus::Improved;
    }
    if best.feasible {
        return FeasibilityStatus::Feasible;
    }
    if improved {
        FeasibilityStatus::InfeasibleImproved
    } else {
        FeasibilityStatus::Infeasible
    }
}

fn describe_feasibility_pipeline_network() -> FeasibilityPipelineNetwork {
    let node = |id: &str, kind: &str, role: FeasibilityNodeRole| FeasibilityPipelineNode {
        id: id.to_string(),
        kind: kind.to_string(),
        role,
    };
    let moving = |id: &str, kind: &str, token_type: &str| FeasibilityPipelineMovingEntity {
        id: id.to_string(),
        kind: kind.to_string(),
        token_type: token_type.to_string(),
    };
    let edge = |from: &str, to: &str, moving_entity: &str, channel: &str| FeasibilityPipelineEdge {
        from: from.to_string(),
        to: to.to_string(),
        moving_entity: moving_entity.to_string(),
        channel: channel.to_string(),
    };
    FeasibilityPipelineNetwork {
        stationary_entities: vec![
            node("candidate-source", "candidate-source", FeasibilityNodeRole::Source),
            node("domain-checker", "domain-checker", FeasibilityNodeRole::Checker),
            node("constraint-checker", "constraint-checker", FeasibilityNodeRole::Checker),
            node("objective-evaluator", "objective-evaluator", FeasibilityNodeRole::Evaluator),
            node("improvement-station", "local-improver", FeasibilityNodeRole::Improver),
            node("wall-clock-checker", "wall-clock-checker", FeasibilityNodeRole::Checker),
            node("feasibility-sink", "feasibility-sink", FeasibilityNodeRole::Sink),
        ],
        moving_entities: vec![
            moving("CandidateToken", "candidate-solution", "CandidateToken"),
            moving("DomainCheckedToken", "domain-checked-candidate", "DomainCheckedToken"),
            moving("ConstraintCheckedToken", "constraint-checked-candidate", "ConstraintCheckedToken"),
            moving("FeasibilityEvaluationToken", "evaluation", "FeasibilityEvaluationToken"),
            moving("StopSignalToken", "stop-signal", "StopSignalToken"),
        ],
        edges: vec![
            edge("candidate-source", "domain-checker", "CandidateToken", CANDIDATE_CHANNEL),
            edge("improvement-station", "domain-checker", "CandidateToken", CANDIDATE_CHANNEL),
            edge("domain-checker", "constraint-checker", "DomainCheckedToken", DOMAIN_CHANNEL),
            edge("constraint-checker", "objective-evaluator", "ConstraintCheckedToken", CONSTRAINT_CHANNEL),
            edge("objective-evaluator", "feasibility-sink", "FeasibilityEvaluationToken", EVALUATION_CHANNEL),
            edge("objective-evaluator", "improvement-station", "FeasibilityEvaluationToken", EVALUATION_CHANNEL),
            edge("wall-clock-checker", "feasibility-sink", "StopSignalToken", STOP_CHANNEL),
        ],
    }
}

/// Downcast a `&dyn DESStation` to a concrete station type for validators.
fn downcast<T: 'static>(s: &dyn DESStation) -> &T {
    s.as_any().downcast_ref::<T>().expect("validator received an unexpected station type")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Smoke tests on a tiny binary-knapsack-style feasibility problem with an
    //! infeasible starting incumbent, verifying the improver repairs / improves
    //! it and the pipeline classifies the result.

    use super::*;

    fn problem() -> StructuredOptimizationProblem {
        StructuredOptimizationProblem {
            sense: ObjectiveSense::Max,
            variables: vec![
                OptimizationVariable {
                    name: "x".to_string(),
                    kind: Some(VariableKind::Binary),
                    lb: None,
                    ub: None,
                    step: None,
                },
                OptimizationVariable {
                    name: "y".to_string(),
                    kind: Some(VariableKind::Binary),
                    lb: None,
                    ub: None,
                    step: None,
                },
            ],
            objective: LinearObjective {
                constant: Some(0.0),
                coefficients: vec![("x".to_string(), 3.0), ("y".to_string(), 2.0)],
            },
            constraints: Some(vec![LinearConstraint {
                name: Some("budget".to_string()),
                coefficients: vec![("x".to_string(), 1.0), ("y".to_string(), 1.0)],
                sense: ConstraintSense::Le,
                rhs: 1.0,
                tolerance: None,
            }]),
            tolerance: Some(1e-8),
        }
    }

    fn values(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn evaluate_candidate_flags_infeasibility() {
        let p = problem();
        let cand = CandidateSolutionInput {
            id: Some("start".to_string()),
            values: Some(values(&[("x", 1.0), ("y", 1.0)])),
            vector: None,
        };
        let eval = evaluate_candidate(&p, &cand, 1_000_000.0);
        // x=y=1 violates the budget x+y<=1, so it is infeasible.
        assert!(!eval.feasible);
        assert!(eval.total_violation > 0.0);
        // Maximisation: comparable objective is negated objective value (5).
        assert!((eval.comparable_objective + 5.0).abs() < 1e-9);
    }

    #[test]
    fn pipeline_improves_infeasible_start() {
        let p = problem();
        let params = FeasibilityPipelineParams {
            problem: p,
            candidate: CandidateSolutionInput {
                id: Some("start".to_string()),
                values: Some(values(&[("x", 1.0), ("y", 1.0)])),
                vector: None,
            },
            improvement: Some(FeasibilityImprovementOptions {
                enabled: Some(true),
                max_iterations: Some(50),
                seed: Some(1),
                ..Default::default()
            }),
            time_limit_ms: Some(180000.0),
            max_ticks: None,
            check_every_ticks: None,
        };
        let result = run_feasibility_pipeline(params);
        assert!(!result.trace.is_empty());
        // The improver should discover a feasible incumbent (x=1,y=0 giving
        // objective 3, the optimum under the budget).
        assert!(result.best.feasible, "expected a feasible best, got {:?}", result.best.violations);
        assert!(result.best.total_violation.abs() < 1e-9);
    }
}
