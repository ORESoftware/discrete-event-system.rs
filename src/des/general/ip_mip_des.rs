//! Port of `src/des/general/ip-mip-des.ts` — module `des::general::ip_mip_des`.
//!
//! Integer / mixed-integer programming expressed as a branch-and-cut DES station
//! graph. Instead of a single branch-and-bound station, this builds a graph of
//! stationary solver roles and lets movable tokens carry subproblems, relaxation
//! results, cuts, and integer candidates:
//!
//! `SearchController → LPRelaxation → {RoundingRepair, CutGenerator,
//! NodeDecision}`, with `NodeDecision` feeding child nodes / completions back to
//! the controller and integer candidates to the incumbent. The LP relaxation
//! station is pluggable (incremental primal/dual simplex, DES simplex, internal
//! two-phase simplex, or a SciPy/HiGHS bridge).
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `LPRelaxationAlgorithm` / `ConcreteLPRelaxationAlgorithm` string unions →
//!     [`LpRelaxationAlgorithm`] / [`ConcreteLpRelaxationAlgorithm`] enums.
//!   * `IPMIPTokenState` discriminated union → [`IpmipTokenState`] enum.
//!   * `*Station` classes (extending `DESStation`) → structs embedding a
//!     `StationCore` + `impl DESStation`; `BranchAndCutSolverStation` (a
//!     `CompositeDESStation`) embeds a [`CompositeDESStation`] and delegates.
//!   * Pluggable LP backend → enum dispatch in [`solve_node_relaxation`].
//!   * Stateful tokens (`PayloadStatefulToken`) flow through the graph; because
//!     tokens travel as `Rc<dyn Any>` (immutable), each station clones the
//!     incoming token to apply a `transition_token`, then re-tracks it in the
//!     shared [`StatefulTokenRegistry`] (dedup-by-id keeps the accumulated
//!     history) — see the FLAG below.
//!   * `throw` / `Preconditions` on bad problems → `panic!` (invariants).
//!
//! FLAGGED divergences (documented limitations of the ported `des_base`):
//!   * The shared token registry is an `Rc<RefCell<StatefulTokenRegistry>>`;
//!     `registry.track` stores a clone of each token's `StatefulToken` base, so
//!     the per-id dedup keeps the latest (fullest-history) copy. This reproduces
//!     the TS stats but is a clone rather than a shared mutable reference.
//!   * `CompositeDESStation` validator aggregation does NOT recurse into children
//!     during `run_iterative_des` (per `composite_station.rs`), so the
//!     `IncumbentStation` intrinsic validator is registered (for explicit
//!     introspection) but not auto-run by the runner; `assert_no_validation_failures`
//!     therefore sees no child checks.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

use crate::des::general::des_base::composite_station::CompositeDESStation;
use crate::des::general::des_base::runner::{
    assert_no_validation_failures, run_iterative_des, IterativeRunOptions, RunReason,
};
use crate::des::general::des_base::stateful_token::{
    transition_token, PayloadStatefulToken, PayloadStatefulTokenOpts, StatefulToken,
    StatefulTokenRegistry, StatefulTokenRegistryStats, TokenStateMode, TransitionTokenOpts,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::general::incremental_lp::{
    IncrementalLP, IncrementalLPInit, IncrementalPivotRule, PivotMode, Sense as IncSense,
    SolverStatus,
};
use crate::des::general::lp::{
    solve_lp_external, solve_lp_internal, solve_lp_internal_ipm, ExternalSolverOptions,
    InternalInteriorPointOptions, InternalSimplexOptions, LPProblem, LPStatus, Sense,
};
use crate::des::general::lp_des::{solve_lp_via_des, DESSimplexOptions, PivotRule};

// -----------------------------------------------------------------------------
// Public problem / result types
// -----------------------------------------------------------------------------

/// Concrete LP relaxation backend (the TS `ConcreteLPRelaxationAlgorithm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConcreteLpRelaxationAlgorithm {
    IncrementalPrimalDual,
    DesSimplexDantzig,
    DesSimplexBland,
    InternalSimplex,
    InternalInteriorPoint,
    ExternalHighs,
    ExternalHighsDs,
    ExternalHighsIpm,
}

impl ConcreteLpRelaxationAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual => "incremental-primal-dual",
            ConcreteLpRelaxationAlgorithm::DesSimplexDantzig => "des-simplex-dantzig",
            ConcreteLpRelaxationAlgorithm::DesSimplexBland => "des-simplex-bland",
            ConcreteLpRelaxationAlgorithm::InternalSimplex => "internal-simplex",
            ConcreteLpRelaxationAlgorithm::InternalInteriorPoint => "internal-ipm",
            ConcreteLpRelaxationAlgorithm::ExternalHighs => "external-highs",
            ConcreteLpRelaxationAlgorithm::ExternalHighsDs => "external-highs-ds",
            ConcreteLpRelaxationAlgorithm::ExternalHighsIpm => "external-highs-ipm",
        }
    }
}

/// Requested LP relaxation backend, which may be `auto` (the TS
/// `LPRelaxationAlgorithm = ConcreteLPRelaxationAlgorithm | 'auto'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LpRelaxationAlgorithm {
    Auto,
    Concrete(ConcreteLpRelaxationAlgorithm),
}

impl LpRelaxationAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            LpRelaxationAlgorithm::Auto => "auto",
            LpRelaxationAlgorithm::Concrete(c) => c.as_str(),
        }
    }
}

/// Structural features of an IP/MIP problem (the TS `IPMIPProblemFeatures`).
#[derive(Clone, Debug)]
pub struct IPMIPProblemFeatures {
    pub variable_count: usize,
    pub constraint_count: usize,
    pub integer_count: usize,
    pub continuous_count: usize,
    pub binary_count: usize,
    pub finite_upper_bounds: usize,
    pub nonzeros: usize,
    pub density: f64,
    pub all_integer: bool,
    pub all_binary: bool,
    pub constraint_variable_components: usize,
}

/// The technique plan computed by [`build_ipmip_solver_technique_plan`].
#[derive(Clone, Debug)]
pub struct IPMIPSolverTechniquePlan {
    pub requested_lp_algorithm: LpRelaxationAlgorithm,
    pub root_lp_algorithm: ConcreteLpRelaxationAlgorithm,
    pub external_solvers_allowed: bool,
    pub uses_external_solvers: bool,
    pub external_candidate: bool,
    pub primal_dual_dynamic: bool,
    pub decomposition_candidate: bool,
    pub decomposition_reason: Option<String>,
    pub rationale: Vec<String>,
    pub features: IPMIPProblemFeatures,
}

/// Optional graph metadata: a variable interpreted as a movable entity.
#[derive(Clone, Debug)]
pub struct VariableNode {
    pub var_index: usize,
    pub node_id: String,
    pub label: Option<String>,
}

/// Optional graph metadata: a stationary constraint anchor.
#[derive(Clone, Debug)]
pub struct ConstraintNode {
    pub row_index: usize,
    pub node_id: String,
    pub label: Option<String>,
}

/// An integer / mixed-integer program.
#[derive(Clone, Debug)]
pub struct IPMIPProblem {
    pub sense: Sense,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub integer_vars: Vec<bool>,
    pub ub: Option<Vec<f64>>,
    pub var_names: Option<Vec<String>>,
    pub con_names: Option<Vec<String>>,
    pub lazy_constraints: Option<Vec<BranchOrCutConstraint>>,
    pub variable_nodes: Option<Vec<VariableNode>>,
    pub constraint_nodes: Option<Vec<ConstraintNode>>,
}

/// One member of a MIP infeasibility conflict.
///
/// The base MIP surface has ordinary `<=` rows, optional finite upper bounds,
/// and per-variable integrality requirements. Lower bounds are implicit
/// non-negativity in this backend, so they are structural rather than removable
/// conflict members.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IPMIPConflictMember {
    LinearRow(usize),
    UpperBound(usize),
    Integrality(usize),
}

impl IPMIPConflictMember {
    pub fn kind(self) -> &'static str {
        match self {
            IPMIPConflictMember::LinearRow(_) => "linear-row",
            IPMIPConflictMember::UpperBound(_) => "upper-bound",
            IPMIPConflictMember::Integrality(_) => "integrality",
        }
    }

    pub fn index(self) -> usize {
        match self {
            IPMIPConflictMember::LinearRow(idx)
            | IPMIPConflictMember::UpperBound(idx)
            | IPMIPConflictMember::Integrality(idx) => idx,
        }
    }
}

/// Options for MIP conflict refinement.
#[derive(Clone, Debug)]
pub struct IPMIPConflictOptions {
    pub solve_options: IPMIPSolveOptions,
}

impl Default for IPMIPConflictOptions {
    fn default() -> Self {
        IPMIPConflictOptions {
            solve_options: IPMIPSolveOptions {
                max_nodes: Some(10_000),
                max_ticks: Some(100_000),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                allow_external_solvers: Some(false),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        }
    }
}

/// A row/bound/integrality-level MIP infeasibility conflict.
#[derive(Clone, Debug, PartialEq)]
pub struct IPMIPInfeasibilityConflict {
    pub infeasible: bool,
    pub members: Vec<IPMIPConflictMember>,
    pub minimal: bool,
    pub checks: usize,
    pub solver: String,
    pub message: Option<String>,
}

/// One relaxable row/bound member in a weighted MIP feasibility relaxation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IPMIPFeasRelaxMember {
    LinearRow(usize),
    UpperBound(usize),
}

impl IPMIPFeasRelaxMember {
    pub fn kind(self) -> &'static str {
        match self {
            IPMIPFeasRelaxMember::LinearRow(_) => "linear-row",
            IPMIPFeasRelaxMember::UpperBound(_) => "upper-bound",
        }
    }

    pub fn index(self) -> usize {
        match self {
            IPMIPFeasRelaxMember::LinearRow(idx) | IPMIPFeasRelaxMember::UpperBound(idx) => idx,
        }
    }
}

/// Options for weighted L1 MIP feasibility relaxation.
///
/// Missing penalty vectors default to unit penalties. Original integrality is
/// preserved; only linear rows and finite upper bounds receive nonnegative
/// continuous relaxation slacks.
#[derive(Clone, Debug, Default)]
pub struct IPMIPFeasRelaxOptions {
    pub row_penalties: Option<Vec<f64>>,
    pub upper_bound_penalties: Option<Vec<f64>>,
    pub solve_options: IPMIPSolveOptions,
}

/// One slack variable created by [`build_ipmip_feasibility_relaxation_problem`].
#[derive(Clone, Debug, PartialEq)]
pub struct IPMIPFeasRelaxSlack {
    pub member: IPMIPFeasRelaxMember,
    pub slack_var: usize,
    pub penalty: f64,
}

/// The ordinary MIP generated for a weighted feasibility relaxation.
#[derive(Clone, Debug)]
pub struct IPMIPFeasRelaxModel {
    pub problem: IPMIPProblem,
    pub original_var_count: usize,
    pub slacks: Vec<IPMIPFeasRelaxSlack>,
}

/// One positive row/bound violation in a weighted MIP feasibility relaxation.
#[derive(Clone, Debug, PartialEq)]
pub struct IPMIPFeasRelaxViolation {
    pub member: IPMIPFeasRelaxMember,
    pub amount: f64,
    pub penalty: f64,
    pub cost: f64,
}

/// Result of solving a weighted MIP feasibility relaxation.
#[derive(Clone, Debug)]
pub struct IPMIPFeasRelaxResult {
    pub status: IPMIPStatus,
    pub x: Vec<f64>,
    pub relaxation_cost: f64,
    pub violations: Vec<IPMIPFeasRelaxViolation>,
    pub relaxation_solution: IPMIPSolution,
    pub solver: String,
    pub message: Option<String>,
}

/// MIP model with source-level variable lower bounds. The underlying branch-and
/// cut solver still receives a non-negative model by substituting `x = y + lb`.
#[derive(Clone, Debug)]
pub struct LowerBoundedIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Vec<f64>,
}

/// A source-level linear row bound, matching the row-bound surface exposed by
/// full-featured MIP solvers. `lower <= coefs·x <= upper`; omit one side for a
/// one-sided row, or set both sides equal for equality.
#[derive(Clone, Debug)]
pub struct LinearRowConstraint {
    pub coefs: Vec<f64>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub name: Option<String>,
}

/// MIP model with source-level linear rows using arbitrary lower/upper row
/// bounds.
#[derive(Clone, Debug)]
pub struct GeneralLinearIPMIPProblem {
    pub base: IPMIPProblem,
    pub linear_constraints: Vec<LinearRowConstraint>,
}

/// Linear row sense used by source-level indicator constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndicatorSense {
    Le,
    Ge,
    Eq,
}

impl IndicatorSense {
    pub fn as_str(self) -> &'static str {
        match self {
            IndicatorSense::Le => "le",
            IndicatorSense::Ge => "ge",
            IndicatorSense::Eq => "eq",
        }
    }
}

/// A source-level indicator constraint:
/// `binary_var == active_value => coefs·x sense rhs`.
#[derive(Clone, Debug)]
pub struct IndicatorConstraint {
    pub binary_var: usize,
    pub active_value: bool,
    pub coefs: Vec<f64>,
    pub sense: IndicatorSense,
    pub rhs: f64,
    pub name: Option<String>,
}

/// MIP model with source-level indicator constraints.
#[derive(Clone, Debug)]
pub struct IndicatorIPMIPProblem {
    pub base: IPMIPProblem,
    pub indicators: Vec<IndicatorConstraint>,
}

/// Special ordered set kind, matching common MIP solver modelling surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialOrderedSetKind {
    Sos1,
    Sos2,
}

impl SpecialOrderedSetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpecialOrderedSetKind::Sos1 => "sos1",
            SpecialOrderedSetKind::Sos2 => "sos2",
        }
    }
}

/// A source-level special ordered set. If weights are present, they define the
/// order; otherwise the given variable order is used.
#[derive(Clone, Debug)]
pub struct SpecialOrderedSet {
    pub kind: SpecialOrderedSetKind,
    pub vars: Vec<usize>,
    pub weights: Option<Vec<f64>>,
    pub name: Option<String>,
}

/// MIP model with source-level special ordered sets.
#[derive(Clone, Debug)]
pub struct SosIPMIPProblem {
    pub base: IPMIPProblem,
    pub sos: Vec<SpecialOrderedSet>,
}

/// Semi-variable kind, matching the common MIP solver domains:
/// semi-continuous `x in {0} U [L, U]` and semi-integer
/// `x in {0} U ([L, U] cap Z)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemiVariableKind {
    SemiContinuous,
    SemiInteger,
}

impl SemiVariableKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SemiVariableKind::SemiContinuous => "semi_continuous",
            SemiVariableKind::SemiInteger => "semi_integer",
        }
    }
}

/// A source-level semi-continuous or semi-integer variable. The upper bound is
/// read from [`IPMIPProblem::ub`], so source models keep one authoritative upper
/// bound.
#[derive(Clone, Debug)]
pub struct SemiVariable {
    pub kind: SemiVariableKind,
    pub var: usize,
    pub lower: f64,
    pub name: Option<String>,
}

/// MIP model with source-level semi-continuous / semi-integer variables.
#[derive(Clone, Debug)]
pub struct SemiIPMIPProblem {
    pub base: IPMIPProblem,
    pub semi_variables: Vec<SemiVariable>,
}

/// A breakpoint in a piecewise-linear function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiecewiseLinearPoint {
    pub x: f64,
    pub y: f64,
}

/// A source-level piecewise-linear constraint `y_var = f(x_var)`.
/// The compiler uses the standard SOS2 convex-combination formulation, so this
/// supports non-convex PWL functions as long as the breakpoint x-values are
/// strictly increasing.
#[derive(Clone, Debug)]
pub struct PiecewiseLinearConstraint {
    pub x_var: usize,
    pub y_var: usize,
    pub points: Vec<PiecewiseLinearPoint>,
    pub name: Option<String>,
}

/// MIP model with source-level piecewise-linear constraints.
#[derive(Clone, Debug)]
pub struct PwlIPMIPProblem {
    pub base: IPMIPProblem,
    pub pwl: Vec<PiecewiseLinearConstraint>,
}

/// A source-level absolute-value constraint `target_var = |arg_var|`.
///
/// The compiler uses an exact bounded big-M formulation when the argument can
/// change sign, and simpler linear equalities when its declared source bounds
/// are one-sided.
#[derive(Clone, Debug)]
pub struct AbsoluteValueConstraint {
    pub arg_var: usize,
    pub target_var: usize,
    pub name: Option<String>,
}

/// MIP model with source-level absolute-value constraints. Optional source
/// lower bounds are compiled before the absolute-value rows are generated.
#[derive(Clone, Debug)]
pub struct AbsIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub abs: Vec<AbsoluteValueConstraint>,
}

/// A source-level maximum constraint:
/// `target_var = max(arg_vars..., constant?)`.
#[derive(Clone, Debug)]
pub struct MaximumConstraint {
    pub target_var: usize,
    pub arg_vars: Vec<usize>,
    pub constant: Option<f64>,
    pub name: Option<String>,
}

/// MIP model with source-level maximum general constraints. Optional source
/// lower bounds are compiled before maximum rows are generated.
#[derive(Clone, Debug)]
pub struct MaximumIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub maximums: Vec<MaximumConstraint>,
}

/// A source-level minimum constraint:
/// `target_var = min(arg_vars..., constant?)`.
#[derive(Clone, Debug)]
pub struct MinimumConstraint {
    pub target_var: usize,
    pub arg_vars: Vec<usize>,
    pub constant: Option<f64>,
    pub name: Option<String>,
}

/// MIP model with source-level minimum general constraints. Optional source
/// lower bounds are compiled before minimum rows are generated.
#[derive(Clone, Debug)]
pub struct MinimumIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub minimums: Vec<MinimumConstraint>,
}

/// Source-level binary logical constraint kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalConstraintKind {
    And,
    Or,
}

impl LogicalConstraintKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LogicalConstraintKind::And => "and",
            LogicalConstraintKind::Or => "or",
        }
    }
}

/// A source-level binary logical constraint:
/// `target_var = and(arg_vars...)` or `target_var = or(arg_vars...)`.
#[derive(Clone, Debug)]
pub struct LogicalConstraint {
    pub kind: LogicalConstraintKind,
    pub target_var: usize,
    pub arg_vars: Vec<usize>,
    pub name: Option<String>,
}

/// MIP model with source-level binary logical constraints.
#[derive(Clone, Debug)]
pub struct LogicalIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub logical: Vec<LogicalConstraint>,
}

/// A source-level L1 norm constraint:
/// `target_var = sum(abs(arg_vars...))`.
#[derive(Clone, Debug)]
pub struct L1NormConstraint {
    pub target_var: usize,
    pub arg_vars: Vec<usize>,
    pub name: Option<String>,
}

/// MIP model with source-level L1 norm general constraints. Optional source
/// lower bounds are compiled before norm rows are generated.
#[derive(Clone, Debug)]
pub struct L1NormIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub l1_norms: Vec<L1NormConstraint>,
}

/// A source-level infinity norm constraint:
/// `target_var = max(abs(arg_vars...))`.
#[derive(Clone, Debug)]
pub struct LInfNormConstraint {
    pub target_var: usize,
    pub arg_vars: Vec<usize>,
    pub name: Option<String>,
}

/// MIP model with source-level infinity norm general constraints. Optional
/// source lower bounds are compiled before norm rows are generated.
#[derive(Clone, Debug)]
pub struct LInfNormIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub linf_norms: Vec<LInfNormConstraint>,
}

/// A source-level product constraint:
/// `target_var = x_var * y_var`.
///
/// The MIP compiler enforces this exactly for binary-binary and
/// binary-continuous bounded products.
#[derive(Clone, Debug)]
pub struct ProductConstraint {
    pub target_var: usize,
    pub x_var: usize,
    pub y_var: usize,
    pub name: Option<String>,
}

/// MIP model with source-level product general constraints. Optional source
/// lower bounds are compiled before product rows are generated.
#[derive(Clone, Debug)]
pub struct ProductIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub products: Vec<ProductConstraint>,
}

/// A source-level quadratic objective term `coeff * x_var * y_var`.
#[derive(Clone, Debug)]
pub struct QuadraticObjectiveTerm {
    pub x_var: usize,
    pub y_var: usize,
    pub coeff: f64,
    pub name: Option<String>,
}

/// MIP model with source-level quadratic objective terms. Terms are compiled
/// exactly when they are binary squares, binary-binary products, or
/// binary-continuous products with finite bounds.
#[derive(Clone, Debug)]
pub struct QuadraticObjectiveIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub quadratic_objective: Vec<QuadraticObjectiveTerm>,
}

/// MIP model that can combine the source-level modeling features exposed by
/// full-featured MIP solvers before compiling them into the ordinary
/// non-negative branch-and-cut backend.
#[derive(Clone, Debug)]
pub struct SourceIPMIPProblem {
    pub base: IPMIPProblem,
    pub lb: Option<Vec<f64>>,
    pub linear_constraints: Vec<LinearRowConstraint>,
    pub indicators: Vec<IndicatorConstraint>,
    pub sos: Vec<SpecialOrderedSet>,
    pub semi_variables: Vec<SemiVariable>,
    pub pwl: Vec<PiecewiseLinearConstraint>,
    pub abs: Vec<AbsoluteValueConstraint>,
    pub maximums: Vec<MaximumConstraint>,
    pub minimums: Vec<MinimumConstraint>,
    pub logical: Vec<LogicalConstraint>,
    pub l1_norms: Vec<L1NormConstraint>,
    pub linf_norms: Vec<LInfNormConstraint>,
    pub products: Vec<ProductConstraint>,
}

/// One objective in a lexicographic multi-objective MIP.
#[derive(Clone, Debug)]
pub struct LexicographicObjective {
    pub sense: Sense,
    pub c: Vec<f64>,
    pub name: Option<String>,
}

/// MIP model with source-level lexicographic objectives.
#[derive(Clone, Debug)]
pub struct MultiObjectiveIPMIPProblem {
    pub base: IPMIPProblem,
    pub objectives: Vec<LexicographicObjective>,
}

/// Options for [`solve_ipmip_with_des`]. `None` fields take the TS defaults.
#[derive(Clone, Debug, Default)]
pub struct IPMIPSolveOptions {
    pub max_nodes: Option<usize>,
    pub max_ticks: Option<usize>,
    pub time_limit_ms: Option<f64>,
    pub lp_max_iters: Option<usize>,
    pub int_tol: Option<f64>,
    /// Optional full feasible incumbent in the solver's current variable
    /// coordinates. The start is validated and used to seed the incumbent before
    /// the root LP is processed.
    pub mip_start: Option<Vec<f64>>,
    pub branch_rule: Option<BranchRule>,
    /// Optional per-variable branching priorities. Larger values are branched
    /// before smaller values; `branch_rule` breaks ties. Missing priorities
    /// default to 0.
    pub branch_priorities: Option<Vec<i32>>,
    /// Stop once the incumbent is within this relative gap of the best known
    /// search bound. `0.0` requires a closed gap; `None` disables the limit.
    pub mip_gap_rel: Option<f64>,
    /// Stop once the incumbent is within this absolute gap of the best known
    /// search bound. `0.0` requires a closed gap; `None` disables the limit.
    pub mip_gap_abs: Option<f64>,
    /// Stop once this many feasible incumbent solutions have been accepted.
    /// The optional MIP start counts as the first incumbent when feasible.
    pub solution_limit: Option<usize>,
    /// Stop once the incumbent reaches this objective target. For maximization
    /// this means `z >= objective_limit`; for minimization it means
    /// `z <= objective_limit`.
    pub objective_limit: Option<f64>,
    pub node_selection: Option<NodeSelection>,
    pub lp_algorithm: Option<LpRelaxationAlgorithm>,
    pub allow_external_solvers: Option<bool>,
    pub max_cut_rounds: Option<usize>,
    pub max_cuts_per_node: Option<usize>,
    pub heuristic_passes: Option<usize>,
    pub verbose: Option<bool>,
    /// Cooperative cancellation flag checked between solver DES ticks.
    pub cancel_flag: Option<Arc<AtomicBool>>,
    /// Monotonic signal. Each increment asks the solver to stop after the next
    /// newly accepted incumbent after the signal is observed.
    pub take_next_incumbent_signal: Option<Arc<AtomicU64>>,
}

/// Branching rule (TS `'most-fractional' | 'first-fractional'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchRule {
    MostFractional,
    FirstFractional,
}

/// Node-selection rule (TS `'dfs' | 'best-bound'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeSelection {
    Dfs,
    BestBound,
}

#[derive(Clone, Debug)]
struct FilledIPMIPSolveOptions {
    max_nodes: usize,
    max_ticks: usize,
    time_limit_ms: f64,
    lp_max_iters: usize,
    int_tol: f64,
    mip_start: Option<Vec<f64>>,
    branch_rule: BranchRule,
    branch_priorities: Vec<i32>,
    mip_gap_rel: Option<f64>,
    mip_gap_abs: Option<f64>,
    solution_limit: Option<usize>,
    objective_limit: Option<f64>,
    node_selection: NodeSelection,
    lp_algorithm: LpRelaxationAlgorithm,
    allow_external_solvers: bool,
    max_cut_rounds: usize,
    max_cuts_per_node: usize,
    heuristic_passes: usize,
    verbose: bool,
    cancel_flag: Option<Arc<AtomicBool>>,
    take_next_incumbent_signal: Option<Arc<AtomicU64>>,
}

/// Overall solve status (TS `IPMIPSolution['status']`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IPMIPStatus {
    Optimal,
    Infeasible,
    Unbounded,
    MaxNodes,
    TickLimit,
    TimeLimit,
    GapLimit,
    SolutionLimit,
    ObjectiveLimit,
    Cancelled,
    TakeNextIncumbent,
}

impl IPMIPStatus {
    /// The TS string spelling of this status.
    pub fn as_str(self) -> &'static str {
        match self {
            IPMIPStatus::Optimal => "optimal",
            IPMIPStatus::Infeasible => "infeasible",
            IPMIPStatus::Unbounded => "unbounded",
            IPMIPStatus::MaxNodes => "maxnodes",
            IPMIPStatus::TickLimit => "tick-limit",
            IPMIPStatus::TimeLimit => "time-limit",
            IPMIPStatus::GapLimit => "gap-limit",
            IPMIPStatus::SolutionLimit => "solution-limit",
            IPMIPStatus::ObjectiveLimit => "objective-limit",
            IPMIPStatus::Cancelled => "cancelled",
            IPMIPStatus::TakeNextIncumbent => "take-next",
        }
    }
}

/// Aggregate performance counters.
#[derive(Clone, Debug)]
pub struct IPMIPPerformanceStats {
    pub elapsed_ms: f64,
    pub ticks: usize,
    pub nodes_per_second: f64,
    pub lp_solves_per_second: f64,
    pub ms_per_node: f64,
    pub total_lp_solver_ms: f64,
    pub avg_lp_solver_ms: f64,
    pub lp_solver_time_share: f64,
    pub avg_lp_iterations_per_solve: f64,
    pub cuts_per_node: f64,
    pub candidates_per_node: f64,
    pub tokens_created: u64,
}

/// Action recorded for one decision-station event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceAction {
    Branch,
    Cut,
    Prune,
    Incumbent,
    Unbounded,
}

/// One node-decision trace event.
#[derive(Clone, Debug)]
pub struct IPMIPTraceEvent {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub lp_status: LPStatus,
    pub lp_z: Option<f64>,
    pub solver: String,
    pub fractional: Vec<usize>,
    pub action: TraceAction,
    pub reason: Option<String>,
    pub branch_var: Option<usize>,
    pub children: Option<Vec<usize>>,
    pub cuts_added: Option<usize>,
    pub node_token_id: Option<String>,
    pub lineage_root: Option<String>,
    pub token_generation: Option<usize>,
    pub state_mode: Option<TokenStateMode>,
}

/// A node in the solver's topology summary.
#[derive(Clone, Debug)]
pub struct SolverTopologyNode {
    pub id: String,
    pub role: String,
    pub emits: Vec<String>,
    pub parent_id: Option<String>,
}

/// The token-registry snapshot type (TS `SolverTokenStats`).
pub type SolverTokenStats = StatefulTokenRegistryStats;

/// The full IP/MIP solution.
#[derive(Clone, Debug)]
pub struct IPMIPSolution {
    pub status: IPMIPStatus,
    pub x: Vec<f64>,
    pub z: f64,
    pub best_bound: f64,
    pub gap: f64,
    pub nodes_explored: usize,
    pub lp_solves: usize,
    pub total_lp_iterations: usize,
    pub cuts_added: usize,
    pub candidates_tried: usize,
    pub lp_algorithm: LpRelaxationAlgorithm,
    pub lp_algorithm_usage: HashMap<ConcreteLpRelaxationAlgorithm, u64>,
    pub technique_plan: IPMIPSolverTechniquePlan,
    pub incumbent_source: Option<String>,
    pub elapsed_ms: f64,
    pub in_house_only: bool,
    pub uses_external_solvers: bool,
    pub performance: IPMIPPerformanceStats,
    pub solver_kind: &'static str,
    pub execution_mode: &'static str,
    pub composite_station_id: String,
    pub token_stats: SolverTokenStats,
    pub trace: Vec<IPMIPTraceEvent>,
    pub topology: Vec<SolverTopologyNode>,
}

/// Result of a lexicographic multi-objective MIP solve.
#[derive(Clone, Debug)]
pub struct MultiObjectiveIPMIPSolution {
    pub status: IPMIPStatus,
    pub x: Vec<f64>,
    pub objective_values: Vec<f64>,
    pub stage_solutions: Vec<IPMIPSolution>,
    pub elapsed_ms: f64,
    pub solver_kind: &'static str,
}

/// Options for enumerating a native MIP solution pool.
#[derive(Clone, Debug)]
pub struct IPMIPSolutionPoolOptions {
    /// Maximum number of distinct integer assignments to return. Default 10.
    pub max_solutions: Option<usize>,
    /// Options passed to each repeated branch-and-cut solve.
    pub solve_options: IPMIPSolveOptions,
    /// Integer comparison tolerance for excluding assignments. Default follows
    /// `solve_options.int_tol` and then 1e-6.
    pub int_tol: Option<f64>,
}

impl Default for IPMIPSolutionPoolOptions {
    fn default() -> Self {
        Self {
            max_solutions: None,
            solve_options: IPMIPSolveOptions::default(),
            int_tol: None,
        }
    }
}

/// Native MIP solution-pool result.
///
/// Solutions are returned in repeated-optimum order by distinct assignments of
/// the original integer variables. Continuous variables are re-optimized for
/// each integer assignment, so the pool intentionally does not enumerate
/// alternate continuous optima under the same integer pattern.
#[derive(Clone, Debug)]
pub struct IPMIPSolutionPool {
    pub status: IPMIPStatus,
    pub solutions: Vec<IPMIPSolution>,
    pub exhausted: bool,
    pub elapsed_ms: f64,
    pub solver_kind: &'static str,
}

/// A branching or cutting constraint `coefs·x <= rhs`.
#[derive(Clone, Debug)]
pub struct BranchOrCutConstraint {
    pub coefs: Vec<f64>,
    pub rhs: f64,
    pub name: String,
    pub kind: ConstraintKind,
}

/// Whether a constraint came from branching or cutting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    Branch,
    Cut,
    Lazy,
}

/// Branch direction (TS `'le' | 'ge'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchType {
    Le,
    Ge,
}

/// A branch-and-cut subproblem.
#[derive(Clone, Debug)]
pub struct IpNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub depth: usize,
    pub constraints: Vec<BranchOrCutConstraint>,
    pub cut_rounds: usize,
    pub branch_var: Option<usize>,
    pub branch_type: Option<BranchType>,
    pub branch_value: Option<f64>,
    pub bound_guess: Option<f64>,
}

/// The LP relaxation result payload.
#[derive(Clone, Debug)]
pub struct RelaxationPayload {
    pub node: IpNode,
    pub status: LPStatus,
    pub x: Vec<f64>,
    pub z: f64,
    pub solver: String,
    pub selected_algorithm: ConcreteLpRelaxationAlgorithm,
    pub iters: usize,
    pub fractional: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct CandidatePayload {
    pub node_id: usize,
    pub x: Vec<f64>,
    pub z: f64,
    pub source: String,
}

#[derive(Clone, Debug)]
pub struct CutPayload {
    pub node_id: usize,
    pub cut: BranchOrCutConstraint,
}

#[derive(Clone, Debug)]
pub struct CompletePayload {
    pub node_id: usize,
}

/// Token state machine (TS `IPMIPTokenState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpmipTokenState {
    Queued,
    RelaxationQueued,
    Relaxed,
    Candidate,
    Cut,
    Complete,
}

// Token aliases — distinct concrete types so `drain::<T>` disambiguates.
type NodeToken = PayloadStatefulToken<IpmipTokenState, IpNode>;
type CompleteToken = PayloadStatefulToken<IpmipTokenState, CompletePayload>;
type RelaxationToken = PayloadStatefulToken<IpmipTokenState, RelaxationPayload>;
type CandidateToken = PayloadStatefulToken<IpmipTokenState, CandidatePayload>;
type CutToken = PayloadStatefulToken<IpmipTokenState, CutPayload>;

/// Per-token construction parameters (the TS token-class `opts` object).
struct TokenOpts {
    token_id: String,
    tick: f64,
    station_id: String,
    parent: Option<StatefulToken<IpmipTokenState>>,
    event: Option<String>,
    detail: Option<String>,
}

fn new_node_token(node: IpNode, opts: TokenOpts) -> NodeToken {
    PayloadStatefulToken::new(PayloadStatefulTokenOpts {
        kind: "ip-node".to_string(),
        token_id: opts.token_id,
        payload: node,
        initial_state: IpmipTokenState::Queued,
        tick: opts.tick,
        station_id: opts.station_id,
        event: opts.event,
        detail: opts.detail,
        parent: opts.parent.map(|p| p.lineage),
        causation_token_id: None,
        state_mode: None,
    })
}

fn new_complete_token(node_id: usize, opts: TokenOpts) -> CompleteToken {
    PayloadStatefulToken::new(PayloadStatefulTokenOpts {
        kind: "ip-complete".to_string(),
        token_id: opts.token_id,
        payload: CompletePayload { node_id },
        initial_state: IpmipTokenState::Complete,
        tick: opts.tick,
        station_id: opts.station_id,
        event: Some("node-complete".to_string()),
        detail: None,
        parent: opts.parent.map(|p| p.lineage),
        causation_token_id: None,
        state_mode: Some(TokenStateMode::Stateless),
    })
}

fn new_relaxation_token(
    payload: RelaxationPayload,
    parent: &StatefulToken<IpmipTokenState>,
    opts: TokenOpts,
) -> RelaxationToken {
    PayloadStatefulToken::new(PayloadStatefulTokenOpts {
        kind: "ip-relaxation".to_string(),
        token_id: opts.token_id,
        payload,
        initial_state: IpmipTokenState::Relaxed,
        tick: opts.tick,
        station_id: opts.station_id,
        event: Some("lp-relaxed".to_string()),
        detail: None,
        parent: Some(parent.lineage.clone()),
        causation_token_id: None,
        state_mode: None,
    })
}

fn new_candidate_token(
    payload: CandidatePayload,
    parent: &StatefulToken<IpmipTokenState>,
    opts: TokenOpts,
) -> CandidateToken {
    PayloadStatefulToken::new(PayloadStatefulTokenOpts {
        kind: "ip-candidate".to_string(),
        token_id: opts.token_id,
        payload,
        initial_state: IpmipTokenState::Candidate,
        tick: opts.tick,
        station_id: opts.station_id,
        event: Some("candidate-generated".to_string()),
        detail: None,
        parent: Some(parent.lineage.clone()),
        causation_token_id: None,
        state_mode: None,
    })
}

fn new_cut_token(
    payload: CutPayload,
    parent: &StatefulToken<IpmipTokenState>,
    opts: TokenOpts,
) -> CutToken {
    PayloadStatefulToken::new(PayloadStatefulTokenOpts {
        kind: "ip-cut".to_string(),
        token_id: opts.token_id,
        payload,
        initial_state: IpmipTokenState::Cut,
        tick: opts.tick,
        station_id: opts.station_id,
        event: Some("cut-generated".to_string()),
        detail: None,
        parent: Some(parent.lineage.clone()),
        causation_token_id: None,
        state_mode: None,
    })
}

const MODEL: &str = "ip-mip-des";
const EPS: f64 = 1e-9;

type Registry = Rc<RefCell<StatefulTokenRegistry>>;

// -----------------------------------------------------------------------------
// Station graph
// -----------------------------------------------------------------------------

/// Maintains the frontier of branch/cut subproblems and dispatches them.
pub struct SearchControllerStation {
    core: StationCore,
    frontier: Vec<NodeToken>,
    in_flight: usize,
    done: bool,
    max_nodes_hit: bool,
    pub nodes_dispatched: usize,
    next_node_id: usize,
    tick: usize,
    sense: Sense,
    max_nodes: usize,
    node_selection: NodeSelection,
    registry: Registry,
}

impl SearchControllerStation {
    fn new(p: &IPMIPProblem, opts: &FilledIPMIPSolveOptions, registry: Registry) -> Self {
        let mut station = SearchControllerStation {
            core: StationCore::new("ip-search-controller"),
            frontier: Vec::new(),
            in_flight: 0,
            done: false,
            max_nodes_hit: false,
            nodes_dispatched: 0,
            next_node_id: 1,
            tick: 0,
            sense: p.sense,
            max_nodes: opts.max_nodes,
            node_selection: opts.node_selection,
            registry,
        };
        let root = IpNode {
            node_id: 0,
            parent_id: None,
            depth: 0,
            constraints: Vec::new(),
            cut_rounds: 0,
            branch_var: None,
            branch_type: None,
            branch_value: None,
            bound_guess: None,
        };
        let tok = new_node_token(
            root,
            TokenOpts {
                token_id: "ip-node-0".to_string(),
                tick: station.tick as f64,
                station_id: station.core.id.clone(),
                parent: None,
                event: Some("root-created".to_string()),
                detail: None,
            },
        );
        station.registry.borrow_mut().track(tok.base.clone());
        station.frontier.push(tok);
        station
    }

    pub fn allocate_node_id(&mut self) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }
    pub fn hit_node_limit(&self) -> bool {
        self.max_nodes_hit
    }
    pub fn frontier_size(&self) -> usize {
        self.frontier.len()
    }

    pub fn best_frontier_bound(&self) -> Option<f64> {
        let finite: Vec<f64> = self
            .frontier
            .iter()
            .filter_map(|t| t.payload.bound_guess)
            .filter(|x| x.is_finite())
            .collect();
        if finite.is_empty() {
            return None;
        }
        Some(if self.sense == Sense::Max {
            finite.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        } else {
            finite.iter().copied().fold(f64::INFINITY, f64::min)
        })
    }

    fn push_node(&mut self, token: NodeToken) {
        self.registry.borrow_mut().track(token.base.clone());
        self.frontier.push(token);
        if self.node_selection == NodeSelection::Dfs {
            return;
        }
        let sense = self.sense;
        let default = if sense == Sense::Max {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        self.frontier.sort_by(|a, b| {
            let ba = a.payload.bound_guess.unwrap_or(default);
            let bb = b.payload.bound_guess.unwrap_or(default);
            let key = if sense == Sense::Max {
                ba - bb
            } else {
                bb - ba
            };
            key.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn pop_node(&mut self) -> Option<NodeToken> {
        self.frontier.pop()
    }
}

impl DESStation for SearchControllerStation {
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
        !self.done || self.core.has_work()
    }
    fn run_time_step(&mut self) {
        let incoming = self.core.drain::<NodeToken>("nodes");
        for t in incoming {
            self.push_node((*t).clone());
        }
        for t in self.core.drain::<CompleteToken>("complete") {
            self.registry.borrow_mut().track(t.base.clone());
            self.in_flight = self.in_flight.saturating_sub(1);
        }

        if self.done {
            return;
        }
        if self.nodes_dispatched >= self.max_nodes {
            if !self.frontier.is_empty() {
                self.max_nodes_hit = true;
            }
            if self.in_flight == 0 {
                self.done = true;
            }
            self.tick += 1;
            return;
        }
        let tok = self.pop_node();
        let mut tok = match tok {
            Some(t) => t,
            None => {
                if self.in_flight == 0 {
                    self.done = true;
                }
                self.tick += 1;
                return;
            }
        };
        self.nodes_dispatched += 1;
        self.in_flight += 1;
        let sid = self.core.id.clone();
        transition_token(
            &mut tok.base,
            IpmipTokenState::RelaxationQueued,
            TransitionTokenOpts {
                tick: self.tick as f64,
                station_id: sid,
                event: "dispatch-to-relaxation".to_string(),
                detail: None,
            },
        );
        self.registry.borrow_mut().track(tok.base.clone());
        self.core.emit(Rc::new(tok), "relax");
        self.tick += 1;
    }
}

/// Stationary LP solver block with a selectable backend.
pub struct LPRelaxationStation {
    core: StationCore,
    p: Rc<IPMIPProblem>,
    algorithm: LpRelaxationAlgorithm,
    technique_plan: IPMIPSolverTechniquePlan,
    lp_max_iters: usize,
    int_tol: f64,
    pub lp_solves: usize,
    pub total_iterations: usize,
    pub total_solver_elapsed_ms: f64,
    pub algorithm_usage: HashMap<ConcreteLpRelaxationAlgorithm, u64>,
    tick: usize,
    registry: Registry,
}

impl LPRelaxationStation {
    fn new(
        p: Rc<IPMIPProblem>,
        algorithm: LpRelaxationAlgorithm,
        technique_plan: IPMIPSolverTechniquePlan,
        lp_max_iters: usize,
        int_tol: f64,
        registry: Registry,
    ) -> Self {
        LPRelaxationStation {
            core: StationCore::new("ip-lp-relaxation"),
            p,
            algorithm,
            technique_plan,
            lp_max_iters,
            int_tol,
            lp_solves: 0,
            total_iterations: 0,
            total_solver_elapsed_ms: 0.0,
            algorithm_usage: HashMap::new(),
            tick: 0,
            registry,
        }
    }
}

impl DESStation for LPRelaxationStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        let nodes = self.core.drain::<NodeToken>("nodes");
        let sid = self.core.id.clone();
        for rc in nodes {
            let mut tok = (*rc).clone();
            self.registry.borrow_mut().track(tok.base.clone());
            let selected = select_lp_relaxation_algorithm(
                &self.p,
                &tok.payload,
                self.algorithm,
                &self.technique_plan,
            );
            let mut used = selected;
            let mut r = solve_node_relaxation(&self.p, &tok.payload, selected, self.lp_max_iters);
            if self.algorithm == LpRelaxationAlgorithm::Auto
                && is_external_lp_algorithm(selected)
                && r.status == LPStatus::NumericalError
            {
                let fallback_message = r
                    .message
                    .clone()
                    .unwrap_or_else(|| "external solver unavailable".to_string());
                used = if has_negative_root_rhs(&self.p) {
                    ConcreteLpRelaxationAlgorithm::InternalSimplex
                } else {
                    ConcreteLpRelaxationAlgorithm::IncrementalPrimalDual
                };
                r = solve_node_relaxation(&self.p, &tok.payload, used, self.lp_max_iters);
                let prev = r.message.clone().unwrap_or_default();
                let sep = if r.message.is_some() { " | " } else { "" };
                r.message = Some(format!(
                    "{prev}{sep}auto fallback from {}: {fallback_message}",
                    selected.as_str()
                ));
                r.solver = format!("{} (auto fallback from {})", r.solver, selected.as_str());
            }
            self.lp_solves += 1;
            self.total_iterations += r.iters.unwrap_or(0);
            self.total_solver_elapsed_ms += r.elapsed_ms;
            *self.algorithm_usage.entry(used).or_insert(0) += 1;
            let fractional = if r.status == LPStatus::Optimal {
                list_fractionals(&r.x, &self.p.integer_vars, self.int_tol)
            } else {
                Vec::new()
            };
            let payload = RelaxationPayload {
                node: tok.payload.clone(),
                status: r.status,
                x: r.x.clone(),
                z: r.objective,
                solver: r.solver.clone(),
                selected_algorithm: used,
                iters: r.iters.unwrap_or(0),
                fractional,
            };
            transition_token(
                &mut tok.base,
                IpmipTokenState::Relaxed,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: sid.clone(),
                    event: "lp-relaxation-solved".to_string(),
                    detail: Some(r.status.as_str().to_string()),
                },
            );
            let out = new_relaxation_token(
                payload,
                &tok.base,
                TokenOpts {
                    token_id: format!("ip-relax-{}-{}", tok.payload.node_id, self.lp_solves),
                    tick: self.tick as f64,
                    station_id: sid.clone(),
                    parent: None,
                    event: None,
                    detail: None,
                },
            );
            {
                let mut reg = self.registry.borrow_mut();
                reg.track(tok.base.clone());
                reg.track(out.base.clone());
            }
            self.core.emit(Rc::new(out), "relaxed");
        }
        self.tick += 1;
    }
}

/// Movable-variable rounding, repair, and local search.
pub struct RoundingRepairStation {
    core: StationCore,
    p: Rc<IPMIPProblem>,
    int_tol: f64,
    heuristic_passes: usize,
    pub candidates_tried: usize,
    tick: usize,
    registry: Registry,
}

impl RoundingRepairStation {
    fn new(p: Rc<IPMIPProblem>, int_tol: f64, heuristic_passes: usize, registry: Registry) -> Self {
        RoundingRepairStation {
            core: StationCore::new("ip-rounding-repair"),
            p,
            int_tol,
            heuristic_passes,
            candidates_tried: 0,
            tick: 0,
            registry,
        }
    }
}

impl DESStation for RoundingRepairStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        let sid = self.core.id.clone();
        for rc in self.core.drain::<RelaxationToken>("relaxed") {
            let tok = (*rc).clone();
            self.registry.borrow_mut().track(tok.base.clone());
            let r = &tok.payload;
            if r.status != LPStatus::Optimal || r.x.is_empty() {
                continue;
            }
            for cand in
                generate_integer_candidates(&self.p, &r.x, self.int_tol, self.heuristic_passes)
            {
                self.candidates_tried += 1;
                let z = objective(&self.p, &cand.x);
                let out = new_candidate_token(
                    CandidatePayload {
                        node_id: r.node.node_id,
                        x: cand.x,
                        z,
                        source: cand.source,
                    },
                    &tok.base,
                    TokenOpts {
                        token_id: format!(
                            "ip-candidate-{}-{}",
                            r.node.node_id, self.candidates_tried
                        ),
                        tick: self.tick as f64,
                        station_id: sid.clone(),
                        parent: None,
                        event: None,
                        detail: None,
                    },
                );
                self.registry.borrow_mut().track(out.base.clone());
                self.core.emit(Rc::new(out), "candidate");
            }
        }
        self.tick += 1;
    }
}

/// Best feasible integer solution anchor.
pub struct IncumbentStation {
    core: StationCore,
    p: Rc<IPMIPProblem>,
    int_tol: f64,
    pub best_x: Vec<f64>,
    pub best_z: f64,
    pub source: Option<String>,
    pub updates: usize,
    pub candidates_seen: usize,
    registry: Registry,
}

fn downcast_incumbent(s: &dyn DESStation) -> &IncumbentStation {
    s.as_any()
        .downcast_ref::<IncumbentStation>()
        .expect("validator received a non-IncumbentStation station")
}

impl IncumbentStation {
    fn new(p: Rc<IPMIPProblem>, int_tol: f64, registry: Registry) -> Self {
        let best_z = if p.sense == Sense::Max {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut station = IncumbentStation {
            core: StationCore::new("ip-incumbent"),
            p,
            int_tol,
            best_x: Vec::new(),
            best_z,
            source: None,
            updates: 0,
            candidates_seen: 0,
            registry,
        };
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "ip.incumbent-feasible",
                |s: &dyn DESStation| {
                    let st = downcast_incumbent(s);
                    st.best_x.is_empty() || is_integer_feasible(&st.p, &st.best_x, st.int_tol)
                },
                Some("incumbent satisfies Ax <= b, bounds, and integrality".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_incumbent(s);
                    format!(
                        "z={}, source={}",
                        st.best_z,
                        st.source.as_deref().unwrap_or("none")
                    )
                })),
                Some("ip-mip-des-intrinsic".to_string()),
                None,
            )
            .boxed(),
        );
        station
    }

    pub fn has_incumbent(&self) -> bool {
        !self.best_x.is_empty()
    }

    pub fn is_improvement(&self, z: f64) -> bool {
        if self.p.sense == Sense::Max {
            z > self.best_z + 1e-9
        } else {
            z < self.best_z - 1e-9
        }
    }

    fn seed_incumbent(&mut self, x: Vec<f64>, source: impl Into<String>) {
        if !is_integer_feasible(&self.p, &x, self.int_tol) {
            panic!("{MODEL}: mip_start must satisfy bounds, rows, and integrality");
        }
        let z = objective(&self.p, &x);
        self.best_x = x;
        self.best_z = z;
        self.source = Some(source.into());
        self.updates += 1;
        self.candidates_seen += 1;
    }
}

impl DESStation for IncumbentStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        for rc in self.core.drain::<CandidateToken>("candidate") {
            self.registry.borrow_mut().track(rc.base.clone());
            let c = &rc.payload;
            self.candidates_seen += 1;
            if !is_integer_feasible(&self.p, &c.x, self.int_tol) {
                continue;
            }
            if !self.is_improvement(c.z) {
                continue;
            }
            self.best_x = c.x.clone();
            self.best_z = c.z;
            self.source = Some(format!("{}@node{}", c.source, c.node_id));
            self.updates += 1;
        }
    }
}

/// Valid-inequality station, currently binary cover cuts.
pub struct CutGeneratorStation {
    core: StationCore,
    p: Rc<IPMIPProblem>,
    int_tol: f64,
    max_cuts_per_node: usize,
    pub cuts_generated: usize,
    tick: usize,
    registry: Registry,
}

impl CutGeneratorStation {
    fn new(
        p: Rc<IPMIPProblem>,
        int_tol: f64,
        max_cuts_per_node: usize,
        registry: Registry,
    ) -> Self {
        CutGeneratorStation {
            core: StationCore::new("ip-cut-generator"),
            p,
            int_tol,
            max_cuts_per_node,
            cuts_generated: 0,
            tick: 0,
            registry,
        }
    }
}

impl DESStation for CutGeneratorStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        let sid = self.core.id.clone();
        for rc in self.core.drain::<RelaxationToken>("relaxed") {
            let tok = (*rc).clone();
            self.registry.borrow_mut().track(tok.base.clone());
            let r = &tok.payload;
            if r.status != LPStatus::Optimal || r.fractional.is_empty() {
                continue;
            }
            let cuts = generate_binary_cover_cuts(
                &self.p,
                &r.x,
                self.int_tol,
                self.max_cuts_per_node,
                &r.node,
            );
            for cut in cuts {
                self.cuts_generated += 1;
                let out = new_cut_token(
                    CutPayload {
                        node_id: r.node.node_id,
                        cut,
                    },
                    &tok.base,
                    TokenOpts {
                        token_id: format!("ip-cut-{}-{}", r.node.node_id, self.cuts_generated),
                        tick: self.tick as f64,
                        station_id: sid.clone(),
                        parent: None,
                        event: None,
                        detail: None,
                    },
                );
                self.registry.borrow_mut().track(out.base.clone());
                self.core.emit(Rc::new(out), "cut");
            }
        }
        self.tick += 1;
    }
}

/// Prune, strengthen (cut), or branch on each relaxed node.
pub struct NodeDecisionStation {
    core: StationCore,
    p: Rc<IPMIPProblem>,
    controller: Rc<RefCell<SearchControllerStation>>,
    incumbent: Rc<RefCell<IncumbentStation>>,
    int_tol: f64,
    branch_rule: BranchRule,
    branch_priorities: Vec<i32>,
    max_cut_rounds: usize,
    verbose: bool,
    pub trace: Vec<IPMIPTraceEvent>,
    cuts_by_node: HashMap<usize, Vec<BranchOrCutConstraint>>,
    pub lazy_cuts_added: usize,
    pub saw_unbounded: bool,
    tick: usize,
    registry: Registry,
}

impl NodeDecisionStation {
    fn new(
        p: Rc<IPMIPProblem>,
        controller: Rc<RefCell<SearchControllerStation>>,
        incumbent: Rc<RefCell<IncumbentStation>>,
        int_tol: f64,
        opts: &FilledIPMIPSolveOptions,
        registry: Registry,
    ) -> Self {
        NodeDecisionStation {
            core: StationCore::new("ip-node-decision"),
            p,
            controller,
            incumbent,
            int_tol,
            branch_rule: opts.branch_rule,
            branch_priorities: opts.branch_priorities.clone(),
            max_cut_rounds: opts.max_cut_rounds,
            verbose: opts.verbose,
            trace: Vec::new(),
            cuts_by_node: HashMap::new(),
            lazy_cuts_added: 0,
            saw_unbounded: false,
            tick: 0,
            registry,
        }
    }

    fn decide(&mut self, tok: &RelaxationToken) {
        let r = tok.payload.clone();
        let node = r.node.clone();
        if r.status == LPStatus::Infeasible {
            self.record(
                tok,
                TraceAction::Prune,
                Some("LP infeasible".to_string()),
                None,
                None,
                None,
            );
            return;
        }
        if r.status == LPStatus::Unbounded {
            self.saw_unbounded = true;
            self.record(
                tok,
                TraceAction::Unbounded,
                Some("LP relaxation unbounded".to_string()),
                None,
                None,
                None,
            );
            return;
        }
        if r.status != LPStatus::Optimal {
            self.record(
                tok,
                TraceAction::Prune,
                Some(r.status.as_str().to_string()),
                None,
                None,
                None,
            );
            return;
        }
        let (best_z, has_inc) = {
            let inc = self.incumbent.borrow();
            (inc.best_z, inc.has_incumbent())
        };
        if bound_dominated(&self.p, r.z, best_z, has_inc) {
            self.record(
                tok,
                TraceAction::Prune,
                Some("bound dominated by incumbent".to_string()),
                None,
                None,
                None,
            );
            return;
        }
        if r.fractional.is_empty() && is_integer_base_feasible(&self.p, &r.x, self.int_tol) {
            let lazy_cuts =
                violated_lazy_constraints(&self.p, &r.x, self.int_tol, &node.constraints);
            if !lazy_cuts.is_empty() {
                let child_id = self.controller.borrow_mut().allocate_node_id();
                let mut constraints = node.constraints.clone();
                constraints.extend(lazy_cuts.iter().cloned());
                let child = IpNode {
                    node_id: child_id,
                    parent_id: Some(node.node_id),
                    depth: node.depth,
                    constraints,
                    cut_rounds: node.cut_rounds,
                    branch_var: node.branch_var,
                    branch_type: node.branch_type,
                    branch_value: node.branch_value,
                    bound_guess: Some(r.z),
                };
                let child_tok = new_node_token(
                    child,
                    TokenOpts {
                        token_id: format!("ip-node-{child_id}"),
                        tick: self.tick as f64,
                        station_id: self.core.id.clone(),
                        parent: Some(tok.base.clone()),
                        event: Some("lazy-child-created".to_string()),
                        detail: Some(format!("{} lazy constraint(s)", lazy_cuts.len())),
                    },
                );
                self.registry.borrow_mut().track(child_tok.base.clone());
                self.core.emit(Rc::new(child_tok), "nodes");
                self.lazy_cuts_added += lazy_cuts.len();
                self.record(
                    tok,
                    TraceAction::Cut,
                    Some(format!("added {} lazy constraint(s)", lazy_cuts.len())),
                    None,
                    Some(vec![child_id]),
                    Some(lazy_cuts.len()),
                );
                return;
            }
            let cand = new_candidate_token(
                CandidatePayload {
                    node_id: node.node_id,
                    x: r.x.clone(),
                    z: r.z,
                    source: "lp-integer".to_string(),
                },
                &tok.base,
                TokenOpts {
                    token_id: format!("ip-candidate-lp-{}-{}", node.node_id, self.tick),
                    tick: self.tick as f64,
                    station_id: self.core.id.clone(),
                    parent: None,
                    event: None,
                    detail: None,
                },
            );
            self.registry.borrow_mut().track(cand.base.clone());
            self.incumbent
                .borrow_mut()
                .core_mut()
                .take(Rc::new(cand), "candidate");
            self.incumbent.borrow_mut().run_time_step();
            self.record(
                tok,
                TraceAction::Incumbent,
                Some("LP relaxation is integer-feasible".to_string()),
                None,
                None,
                None,
            );
            return;
        }

        let pending_cuts = self
            .cuts_by_node
            .get(&node.node_id)
            .cloned()
            .unwrap_or_default();
        if !pending_cuts.is_empty() && node.cut_rounds < self.max_cut_rounds {
            let child_id = self.controller.borrow_mut().allocate_node_id();
            let mut constraints = node.constraints.clone();
            constraints.extend(pending_cuts.iter().cloned());
            let child = IpNode {
                node_id: child_id,
                parent_id: Some(node.node_id),
                depth: node.depth,
                constraints,
                cut_rounds: node.cut_rounds + 1,
                branch_var: node.branch_var,
                branch_type: node.branch_type,
                branch_value: node.branch_value,
                bound_guess: Some(r.z),
            };
            let child_tok = new_node_token(
                child,
                TokenOpts {
                    token_id: format!("ip-node-{child_id}"),
                    tick: self.tick as f64,
                    station_id: self.core.id.clone(),
                    parent: Some(tok.base.clone()),
                    event: Some("cut-child-created".to_string()),
                    detail: Some(format!("{} cuts", pending_cuts.len())),
                },
            );
            self.registry.borrow_mut().track(child_tok.base.clone());
            self.core.emit(Rc::new(child_tok), "nodes");
            self.record(
                tok,
                TraceAction::Cut,
                Some(format!("added {} valid cut(s)", pending_cuts.len())),
                None,
                Some(vec![child_id]),
                Some(pending_cuts.len()),
            );
            return;
        }

        let j = pick_branch_var(
            &r.x,
            &r.fractional,
            self.branch_rule,
            &self.branch_priorities,
        );
        let xj = r.x[j];
        let lo = xj.floor();
        let hi = xj.ceil();
        let mut le = vec![0.0; self.p.c.len()];
        le[j] = 1.0;
        let mut ge = vec![0.0; self.p.c.len()];
        ge[j] = -1.0;
        let left_id = self.controller.borrow_mut().allocate_node_id();
        let mut left_constraints = node.constraints.clone();
        left_constraints.push(BranchOrCutConstraint {
            coefs: le,
            rhs: lo,
            name: format!("{}<={lo}", var_name(&self.p, j)),
            kind: ConstraintKind::Branch,
        });
        let left = IpNode {
            node_id: left_id,
            parent_id: Some(node.node_id),
            depth: node.depth + 1,
            constraints: left_constraints,
            cut_rounds: 0,
            branch_var: Some(j),
            branch_type: Some(BranchType::Le),
            branch_value: Some(lo),
            bound_guess: Some(r.z),
        };
        let right_id = self.controller.borrow_mut().allocate_node_id();
        let mut right_constraints = node.constraints.clone();
        right_constraints.push(BranchOrCutConstraint {
            coefs: ge,
            rhs: -hi,
            name: format!("{}>={hi}", var_name(&self.p, j)),
            kind: ConstraintKind::Branch,
        });
        let right = IpNode {
            node_id: right_id,
            parent_id: Some(node.node_id),
            depth: node.depth + 1,
            constraints: right_constraints,
            cut_rounds: 0,
            branch_var: Some(j),
            branch_type: Some(BranchType::Ge),
            branch_value: Some(hi),
            bound_guess: Some(r.z),
        };
        let left_tok = new_node_token(
            left,
            TokenOpts {
                token_id: format!("ip-node-{left_id}"),
                tick: self.tick as f64,
                station_id: self.core.id.clone(),
                parent: Some(tok.base.clone()),
                event: Some("branch-left-created".to_string()),
                detail: Some(format!("{}<={lo}", var_name(&self.p, j))),
            },
        );
        let right_tok = new_node_token(
            right,
            TokenOpts {
                token_id: format!("ip-node-{right_id}"),
                tick: self.tick as f64,
                station_id: self.core.id.clone(),
                parent: Some(tok.base.clone()),
                event: Some("branch-right-created".to_string()),
                detail: Some(format!("{}>={hi}", var_name(&self.p, j))),
            },
        );
        {
            let mut reg = self.registry.borrow_mut();
            reg.track(left_tok.base.clone());
            reg.track(right_tok.base.clone());
        }
        self.core.emit(Rc::new(left_tok), "nodes");
        self.core.emit(Rc::new(right_tok), "nodes");
        self.record(
            tok,
            TraceAction::Branch,
            Some(format!("branch on {}={:.6}", var_name(&self.p, j), xj)),
            Some(j),
            Some(vec![left_id, right_id]),
            None,
        );
    }

    fn record(
        &mut self,
        tok: &RelaxationToken,
        action: TraceAction,
        reason: Option<String>,
        branch_var: Option<usize>,
        children: Option<Vec<usize>>,
        cuts_added: Option<usize>,
    ) {
        let r = &tok.payload;
        let mut fractional = r.fractional.clone();
        fractional.truncate(16);
        let reason_for_log = reason.clone();
        let (node_id, depth, z, action_dbg) = (r.node.node_id, r.node.depth, r.z, action);
        self.trace.push(IPMIPTraceEvent {
            node_id: r.node.node_id,
            parent_id: r.node.parent_id,
            depth: r.node.depth,
            lp_status: r.status,
            lp_z: if r.z.is_finite() { Some(r.z) } else { None },
            solver: r.solver.clone(),
            fractional,
            action,
            reason,
            branch_var,
            children,
            cuts_added,
            node_token_id: Some(
                tok.base
                    .lineage
                    .parent_token_id
                    .clone()
                    .unwrap_or_else(|| tok.base.lineage.token_id.clone()),
            ),
            lineage_root: Some(tok.base.lineage.root_token_id.clone()),
            token_generation: Some(tok.base.lineage.generation),
            state_mode: Some(tok.base.state_mode),
        });
        if self.verbose {
            let reason_part = reason_for_log
                .map(|x| format!(" ({x})"))
                .unwrap_or_default();
            eprintln!("node {node_id} d={depth} z={z} {action_dbg:?}{reason_part}");
        }
    }
}

impl DESStation for NodeDecisionStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        for rc in self.core.drain::<CutToken>("cuts") {
            self.registry.borrow_mut().track(rc.base.clone());
            self.cuts_by_node
                .entry(rc.payload.node_id)
                .or_default()
                .push(rc.payload.cut.clone());
        }
        let relaxed = self.core.drain::<RelaxationToken>("relaxed");
        let sid = self.core.id.clone();
        for rc in relaxed {
            let tok = (*rc).clone();
            self.registry.borrow_mut().track(tok.base.clone());
            self.decide(&tok);
            let done = new_complete_token(
                tok.payload.node.node_id,
                TokenOpts {
                    token_id: format!("ip-complete-{}-{}", tok.payload.node.node_id, self.tick),
                    tick: self.tick as f64,
                    station_id: sid.clone(),
                    parent: Some(tok.base.clone()),
                    event: None,
                    detail: None,
                },
            );
            self.registry.borrow_mut().track(done.base.clone());
            self.core.emit(Rc::new(done), "complete");
        }
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// Public solver
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct MIPGapValues {
    best_bound: f64,
    abs_gap: f64,
    rel_gap: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IPMIPStopCondition {
    TimeLimit,
    GapLimit,
    SolutionLimit,
    ObjectiveLimit,
    Cancelled,
    TakeNextIncumbent,
}

/// Composite single-threaded in-house branch-and-cut solver.
pub struct BranchAndCutSolverStation {
    composite: CompositeDESStation,
    pub token_registry: Registry,
    pub technique_plan: IPMIPSolverTechniquePlan,
    pub controller: Rc<RefCell<SearchControllerStation>>,
    pub lp: Rc<RefCell<LPRelaxationStation>>,
    pub heuristic: Rc<RefCell<RoundingRepairStation>>,
    pub incumbent: Rc<RefCell<IncumbentStation>>,
    pub cuts: Rc<RefCell<CutGeneratorStation>>,
    pub decision: Rc<RefCell<NodeDecisionStation>>,
    p: Rc<IPMIPProblem>,
}

impl BranchAndCutSolverStation {
    fn new(id: &str, p: Rc<IPMIPProblem>, opts: &FilledIPMIPSolveOptions) -> Self {
        let mut composite = CompositeDESStation::new(id);
        let registry: Registry = Rc::new(RefCell::new(StatefulTokenRegistry::new()));
        let technique_plan =
            build_ipmip_solver_technique_plan(&p, opts.lp_algorithm, opts.allow_external_solvers);

        let controller = Rc::new(RefCell::new(SearchControllerStation::new(
            &p,
            opts,
            registry.clone(),
        )));
        composite.add_substation(controller.clone());
        let lp = Rc::new(RefCell::new(LPRelaxationStation::new(
            p.clone(),
            opts.lp_algorithm,
            technique_plan.clone(),
            opts.lp_max_iters,
            opts.int_tol,
            registry.clone(),
        )));
        composite.add_substation(lp.clone());
        let heuristic = Rc::new(RefCell::new(RoundingRepairStation::new(
            p.clone(),
            opts.int_tol,
            opts.heuristic_passes,
            registry.clone(),
        )));
        composite.add_substation(heuristic.clone());
        let incumbent = Rc::new(RefCell::new(IncumbentStation::new(
            p.clone(),
            opts.int_tol,
            registry.clone(),
        )));
        if let Some(start) = opts.mip_start.clone() {
            incumbent
                .borrow_mut()
                .seed_incumbent(start, "user-mip-start");
        }
        composite.add_substation(incumbent.clone());
        let cuts = Rc::new(RefCell::new(CutGeneratorStation::new(
            p.clone(),
            opts.int_tol,
            opts.max_cuts_per_node,
            registry.clone(),
        )));
        composite.add_substation(cuts.clone());
        let decision = Rc::new(RefCell::new(NodeDecisionStation::new(
            p.clone(),
            controller.clone(),
            incumbent.clone(),
            opts.int_tol,
            opts,
            registry.clone(),
        )));
        composite.add_substation(decision.clone());

        controller
            .borrow_mut()
            .core_mut()
            .pipe(lp.clone() as StationRef, "relax", "nodes");
        lp.borrow_mut()
            .core_mut()
            .pipe(heuristic.clone() as StationRef, "relaxed", "relaxed");
        lp.borrow_mut()
            .core_mut()
            .pipe(cuts.clone() as StationRef, "relaxed", "relaxed");
        lp.borrow_mut()
            .core_mut()
            .pipe(decision.clone() as StationRef, "relaxed", "relaxed");
        heuristic.borrow_mut().core_mut().pipe(
            incumbent.clone() as StationRef,
            "candidate",
            "candidate",
        );
        cuts.borrow_mut()
            .core_mut()
            .pipe(decision.clone() as StationRef, "cut", "cuts");
        decision
            .borrow_mut()
            .core_mut()
            .pipe(controller.clone() as StationRef, "nodes", "nodes");
        decision.borrow_mut().core_mut().pipe(
            controller.clone() as StationRef,
            "complete",
            "complete",
        );

        BranchAndCutSolverStation {
            composite,
            token_registry: registry,
            technique_plan,
            controller,
            lp,
            heuristic,
            incumbent,
            cuts,
            decision,
            p,
        }
    }

    pub fn best_bound(&self) -> f64 {
        compute_best_bound(&self.p, &self.incumbent.borrow(), &self.controller.borrow())
    }

    fn active_gap_values(&self) -> Option<MIPGapValues> {
        let incumbent = self.incumbent.borrow();
        if !incumbent.has_incumbent() || !incumbent.best_z.is_finite() {
            return None;
        }
        let best_bound = self.controller.borrow().best_frontier_bound()?;
        if !best_bound.is_finite() {
            return None;
        }
        let abs_gap = (best_bound - incumbent.best_z).abs();
        let rel_gap = abs_gap / 1.0_f64.max(incumbent.best_z.abs());
        Some(MIPGapValues {
            best_bound,
            abs_gap,
            rel_gap,
        })
    }

    fn gap_limit_satisfied(&self, rel_limit: Option<f64>, abs_limit: Option<f64>) -> bool {
        let Some(values) = self.active_gap_values() else {
            return false;
        };
        rel_limit.is_some_and(|limit| values.rel_gap <= limit + EPS)
            || abs_limit.is_some_and(|limit| values.abs_gap <= limit + EPS)
    }

    fn solution_limit_satisfied(&self, solution_limit: Option<usize>) -> bool {
        let Some(solution_limit) = solution_limit else {
            return false;
        };
        self.incumbent.borrow().updates >= solution_limit
    }

    fn objective_limit_satisfied(&self, objective_limit: Option<f64>) -> bool {
        let Some(objective_limit) = objective_limit else {
            return false;
        };
        let incumbent = self.incumbent.borrow();
        if !incumbent.has_incumbent() || !incumbent.best_z.is_finite() {
            return false;
        }
        if self.p.sense == Sense::Max {
            incumbent.best_z >= objective_limit - EPS
        } else {
            incumbent.best_z <= objective_limit + EPS
        }
    }

    pub fn has_incumbent(&self) -> bool {
        self.incumbent.borrow().has_incumbent()
    }

    pub fn token_stats(&self) -> SolverTokenStats {
        self.token_registry.borrow().snapshot()
    }

    pub fn topology(&self) -> Vec<SolverTopologyNode> {
        solver_topology(&self.composite.core().id)
    }
}

impl DESStation for BranchAndCutSolverStation {
    fn core(&self) -> &StationCore {
        self.composite.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.composite.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        self.composite.assert_preconditions();
    }
    fn has_work(&self) -> bool {
        self.composite.has_work()
    }
    fn run_time_step(&mut self) {
        self.composite.run_time_step();
    }
    fn on_finalize(&mut self) {
        self.composite.on_finalize();
    }
    fn num_validators(&self) -> usize {
        self.composite.num_validators()
    }
}

fn fill_ipmip_options(opts: &IPMIPSolveOptions) -> FilledIPMIPSolveOptions {
    let max_nodes = opts.max_nodes.unwrap_or(10000);
    FilledIPMIPSolveOptions {
        max_nodes,
        max_ticks: opts.max_ticks.unwrap_or_else(|| 100.max(max_nodes * 8)),
        time_limit_ms: opts.time_limit_ms.unwrap_or(f64::INFINITY),
        lp_max_iters: opts.lp_max_iters.unwrap_or(2000),
        int_tol: opts.int_tol.unwrap_or(1e-6),
        mip_start: opts.mip_start.clone(),
        branch_rule: opts.branch_rule.unwrap_or(BranchRule::MostFractional),
        branch_priorities: opts.branch_priorities.clone().unwrap_or_default(),
        mip_gap_rel: opts.mip_gap_rel,
        mip_gap_abs: opts.mip_gap_abs,
        solution_limit: opts.solution_limit,
        objective_limit: opts.objective_limit,
        node_selection: opts.node_selection.unwrap_or(NodeSelection::Dfs),
        lp_algorithm: opts.lp_algorithm.unwrap_or(LpRelaxationAlgorithm::Auto),
        allow_external_solvers: opts.allow_external_solvers.unwrap_or(false),
        max_cut_rounds: opts.max_cut_rounds.unwrap_or(1),
        max_cuts_per_node: opts.max_cuts_per_node.unwrap_or(2),
        heuristic_passes: opts.heuristic_passes.unwrap_or(60),
        verbose: opts.verbose.unwrap_or(false),
        cancel_flag: opts.cancel_flag.clone(),
        take_next_incumbent_signal: opts.take_next_incumbent_signal.clone(),
    }
}

fn validate_branch_priorities(p: &IPMIPProblem, priorities: Option<&[i32]>) {
    if let Some(priorities) = priorities {
        if priorities.len() != p.c.len() {
            panic!(
                "{MODEL}: branch_priorities length {} does not match variable count {}",
                priorities.len(),
                p.c.len()
            );
        }
    }
}

fn validate_mip_gap_limit(name: &str, value: Option<f64>) {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            panic!("{MODEL}: {name} must be a finite non-negative value");
        }
    }
}

fn validate_solution_limit(value: Option<usize>) {
    if value == Some(0) {
        panic!("{MODEL}: solution_limit must be positive");
    }
}

fn validate_objective_limit(value: Option<f64>) {
    if let Some(value) = value {
        if !value.is_finite() {
            panic!("{MODEL}: objective_limit must be finite");
        }
    }
}

/// Solve an IP/MIP using the in-house branch-and-cut DES.
pub fn solve_ipmip_with_des(p: IPMIPProblem, opts: IPMIPSolveOptions) -> IPMIPSolution {
    validate_ipmip_problem(&p);
    validate_mip_start(&p, opts.mip_start.as_deref(), opts.int_tol.unwrap_or(1e-6));
    validate_branch_priorities(&p, opts.branch_priorities.as_deref());
    validate_mip_gap_limit("mip_gap_rel", opts.mip_gap_rel);
    validate_mip_gap_limit("mip_gap_abs", opts.mip_gap_abs);
    validate_solution_limit(opts.solution_limit);
    validate_objective_limit(opts.objective_limit);
    let filled = fill_ipmip_options(&opts);
    let t0 = Instant::now();
    let p_rc = Rc::new(p);
    let solver = Rc::new(RefCell::new(BranchAndCutSolverStation::new(
        "ip-branch-and-cut",
        p_rc.clone(),
        &filled,
    )));

    let time_limit = filled.time_limit_ms;
    let mip_gap_rel = filled.mip_gap_rel;
    let mip_gap_abs = filled.mip_gap_abs;
    let solution_limit = filled.solution_limit;
    let objective_limit = filled.objective_limit;
    let cancel_flag = filled.cancel_flag.clone();
    let take_next_incumbent_signal = filled.take_next_incumbent_signal.clone();
    let mut seen_take_next_signal = take_next_incumbent_signal
        .as_ref()
        .map(|signal| signal.load(Ordering::Relaxed))
        .unwrap_or(0);
    let mut take_next_target_updates: Option<usize> = None;
    let stop_clock = t0;
    let stop_condition: Rc<RefCell<Option<IPMIPStopCondition>>> = Rc::new(RefCell::new(None));
    let stop_condition_for_loop = stop_condition.clone();
    let solver_for_stop = solver.clone();
    let summary = run_iterative_des(
        vec![solver.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(filled.max_ticks),
            stop_when: Some(Box::new(move |_tick, _ents| {
                if cancel_flag
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    *stop_condition_for_loop.borrow_mut() = Some(IPMIPStopCondition::Cancelled);
                    return true;
                }
                if let Some(signal) = take_next_incumbent_signal.as_ref() {
                    let current_signal = signal.load(Ordering::Relaxed);
                    if current_signal > seen_take_next_signal {
                        seen_take_next_signal = current_signal;
                        let current_updates = solver_for_stop.borrow().incumbent.borrow().updates;
                        take_next_target_updates = Some(current_updates.saturating_add(1));
                    }
                    if let Some(target_updates) = take_next_target_updates {
                        if solver_for_stop.borrow().incumbent.borrow().updates >= target_updates {
                            *stop_condition_for_loop.borrow_mut() =
                                Some(IPMIPStopCondition::TakeNextIncumbent);
                            return true;
                        }
                    }
                }
                if time_limit.is_finite()
                    && stop_clock.elapsed().as_secs_f64() * 1000.0 >= time_limit
                {
                    *stop_condition_for_loop.borrow_mut() = Some(IPMIPStopCondition::TimeLimit);
                    return true;
                }
                if solver_for_stop
                    .borrow()
                    .objective_limit_satisfied(objective_limit)
                {
                    *stop_condition_for_loop.borrow_mut() =
                        Some(IPMIPStopCondition::ObjectiveLimit);
                    return true;
                }
                if solver_for_stop
                    .borrow()
                    .solution_limit_satisfied(solution_limit)
                {
                    *stop_condition_for_loop.borrow_mut() = Some(IPMIPStopCondition::SolutionLimit);
                    return true;
                }
                if solver_for_stop
                    .borrow()
                    .gap_limit_satisfied(mip_gap_rel, mip_gap_abs)
                {
                    *stop_condition_for_loop.borrow_mut() = Some(IPMIPStopCondition::GapLimit);
                    return true;
                }
                false
            })),
            ..Default::default()
        },
    );
    if let Err(e) = assert_no_validation_failures(&summary, MODEL) {
        panic!("{e}");
    }

    let solver_ref = solver.borrow();
    let has_inc = solver_ref.has_incumbent();
    let best_bound = solver_ref.best_bound();
    let z = if has_inc {
        solver_ref.incumbent.borrow().best_z
    } else if p_rc.sense == Sense::Max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let hit_node_limit = solver_ref.controller.borrow().hit_node_limit();
    let optimal = summary.reason == Some(RunReason::Done) && !hit_node_limit && has_inc;
    let stop_condition = *stop_condition.borrow();
    let saw_unbounded = solver_ref.decision.borrow().saw_unbounded;
    let status = if summary.reason == Some(RunReason::MaxTicks) {
        IPMIPStatus::TickLimit
    } else if summary.reason == Some(RunReason::StopWhen)
        && stop_condition == Some(IPMIPStopCondition::Cancelled)
    {
        IPMIPStatus::Cancelled
    } else if summary.reason == Some(RunReason::StopWhen)
        && stop_condition == Some(IPMIPStopCondition::TakeNextIncumbent)
    {
        IPMIPStatus::TakeNextIncumbent
    } else if summary.reason == Some(RunReason::StopWhen)
        && stop_condition == Some(IPMIPStopCondition::ObjectiveLimit)
    {
        IPMIPStatus::ObjectiveLimit
    } else if summary.reason == Some(RunReason::StopWhen)
        && stop_condition == Some(IPMIPStopCondition::SolutionLimit)
    {
        IPMIPStatus::SolutionLimit
    } else if summary.reason == Some(RunReason::StopWhen)
        && stop_condition == Some(IPMIPStopCondition::GapLimit)
    {
        IPMIPStatus::GapLimit
    } else if summary.reason == Some(RunReason::StopWhen) {
        IPMIPStatus::TimeLimit
    } else if hit_node_limit {
        IPMIPStatus::MaxNodes
    } else if optimal {
        IPMIPStatus::Optimal
    } else if !has_inc && saw_unbounded {
        IPMIPStatus::Unbounded
    } else {
        IPMIPStatus::Infeasible
    };
    let final_bound = if optimal { z } else { best_bound };
    let gap = if !has_inc || !final_bound.is_finite() {
        f64::INFINITY
    } else {
        (final_bound - z).abs() / 1.0_f64.max(z.abs())
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let token_stats = solver_ref.token_stats();
    let lp_solves = solver_ref.lp.borrow().lp_solves;
    let total_lp_iterations = solver_ref.lp.borrow().total_iterations;
    let total_lp_solver_ms = solver_ref.lp.borrow().total_solver_elapsed_ms;
    let cuts_added =
        solver_ref.cuts.borrow().cuts_generated + solver_ref.decision.borrow().lazy_cuts_added;
    let candidates_tried = solver_ref.heuristic.borrow().candidates_tried;
    let algorithm_usage = solver_ref.lp.borrow().algorithm_usage.clone();
    let uses_external_solvers = did_use_external_lp(&algorithm_usage);
    let performance = build_ipmip_performance(BuildPerfArgs {
        elapsed_ms,
        ticks: summary.ticks,
        nodes_explored: lp_solves,
        lp_solves,
        total_lp_iterations,
        total_lp_solver_ms,
        cuts_added,
        candidates_tried,
        tokens_created: token_stats.created,
    });

    let solution = IPMIPSolution {
        status,
        x: if has_inc {
            solver_ref.incumbent.borrow().best_x.clone()
        } else {
            Vec::new()
        },
        z,
        best_bound: final_bound,
        gap: if optimal { 0.0 } else { gap },
        nodes_explored: lp_solves,
        lp_solves,
        total_lp_iterations,
        cuts_added,
        candidates_tried,
        lp_algorithm: filled.lp_algorithm,
        lp_algorithm_usage: algorithm_usage,
        technique_plan: solver_ref.technique_plan.clone(),
        incumbent_source: solver_ref.incumbent.borrow().source.clone(),
        elapsed_ms,
        in_house_only: !uses_external_solvers,
        uses_external_solvers,
        performance,
        solver_kind: "in-house-branch-and-cut",
        execution_mode: "single-threaded",
        composite_station_id: solver_ref.core().id.clone(),
        token_stats,
        trace: solver_ref.decision.borrow().trace.clone(),
        topology: solver_ref.topology(),
    };
    solution
}

/// Enumerate a solution pool by repeatedly solving the MIP and excluding the
/// previous assignment of the original integer variables.
pub fn solve_ipmip_solution_pool_with_des(
    p: IPMIPProblem,
    opts: IPMIPSolutionPoolOptions,
) -> IPMIPSolutionPool {
    validate_ipmip_problem(&p);
    let max_solutions = opts.max_solutions.unwrap_or(10);
    if max_solutions == 0 {
        panic!("{MODEL}: solution pool max_solutions must be positive");
    }
    let int_tol = opts.int_tol.or(opts.solve_options.int_tol).unwrap_or(1e-6);
    let original_n = p.c.len();
    let integer_vars: Vec<usize> = p
        .integer_vars
        .iter()
        .enumerate()
        .filter_map(|(j, &is_int)| is_int.then_some(j))
        .collect();
    if integer_vars.is_empty() {
        panic!("{MODEL}: solution pool requires at least one integer variable");
    }
    validate_solution_pool_bounds(&p, &integer_vars);

    let t0 = Instant::now();
    let mut working = p.clone();
    let mut solutions = Vec::new();
    let mut status = IPMIPStatus::Optimal;
    let mut exhausted = false;

    for iteration in 0..max_solutions {
        let mut solve_options = opts.solve_options.clone();
        if iteration > 0 {
            solve_options.mip_start = None;
        }
        let mut sol = solve_ipmip_with_des(working.clone(), solve_options);
        match sol.status {
            IPMIPStatus::Optimal => {
                let original_x = sol.x[..original_n].to_vec();
                append_integer_assignment_exclusion(
                    &mut working,
                    &integer_vars,
                    &original_x,
                    int_tol,
                );
                sol.x = original_x;
                sol.solver_kind = "in-house-branch-and-cut-solution-pool-member";
                solutions.push(sol);
            }
            IPMIPStatus::Infeasible => {
                exhausted = true;
                status = if solutions.is_empty() {
                    IPMIPStatus::Infeasible
                } else {
                    IPMIPStatus::Optimal
                };
                break;
            }
            other => {
                status = other;
                break;
            }
        }
    }

    IPMIPSolutionPool {
        status,
        solutions,
        exhausted,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        solver_kind: "in-house-branch-and-cut-solution-pool",
    }
}

fn validate_ipmip_feas_relax_penalties(name: &str, penalties: &Option<Vec<f64>>, expected: usize) {
    if let Some(penalties) = penalties {
        if penalties.len() != expected {
            panic!(
                "{MODEL}: {name} length {} does not match expected length {expected}",
                penalties.len()
            );
        }
        for (idx, &penalty) in penalties.iter().enumerate() {
            if !penalty.is_finite() || penalty < 0.0 {
                panic!("{MODEL}: {name}[{idx}] must be a finite non-negative penalty");
            }
        }
    }
}

fn ipmip_feas_relax_penalty(penalties: &Option<Vec<f64>>, idx: usize) -> f64 {
    penalties
        .as_ref()
        .and_then(|values| values.get(idx))
        .copied()
        .unwrap_or(1.0)
}

fn ipmip_row_name(p: &IPMIPProblem, row: usize) -> String {
    p.con_names
        .as_ref()
        .and_then(|names| names.get(row))
        .cloned()
        .unwrap_or_else(|| format!("row_{row}"))
}

fn add_ipmip_feas_relax_slack(
    c: &mut Vec<f64>,
    integer_vars: &mut Vec<bool>,
    ub: &mut Vec<f64>,
    var_names: &mut Vec<String>,
    rows: &mut [Vec<f64>],
    slacks: &mut Vec<IPMIPFeasRelaxSlack>,
    member: IPMIPFeasRelaxMember,
    penalty: f64,
    name: String,
) -> usize {
    let idx = c.len();
    c.push(penalty);
    integer_vars.push(false);
    ub.push(f64::INFINITY);
    var_names.push(name);
    for row in rows {
        row.push(0.0);
    }
    slacks.push(IPMIPFeasRelaxSlack {
        member,
        slack_var: idx,
        penalty,
    });
    idx
}

/// Build an ordinary MIP whose optimum is the weighted L1 feasibility relaxation
/// of `p`, preserving original integrality and relaxing linear rows plus finite
/// upper bounds with nonnegative continuous slacks.
pub fn build_ipmip_feasibility_relaxation_problem(
    p: &IPMIPProblem,
    opts: &IPMIPFeasRelaxOptions,
) -> IPMIPFeasRelaxModel {
    validate_ipmip_problem(p);
    let n = p.c.len();
    validate_ipmip_feas_relax_penalties("row_penalties", &opts.row_penalties, p.a.len());
    validate_ipmip_feas_relax_penalties("upper_bound_penalties", &opts.upper_bound_penalties, n);

    let source_ub = p.ub.clone().unwrap_or_else(|| vec![f64::INFINITY; n]);
    let mut c = vec![0.0; n];
    let mut integer_vars = p.integer_vars.clone();
    let mut ub = source_ub.clone();
    let mut var_names: Vec<String> = (0..n).map(|j| var_name(p, j)).collect();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    let mut con_names = Vec::new();
    let mut slacks = Vec::new();

    for (row_idx, (coefs, &bound)) in p.a.iter().zip(&p.b).enumerate() {
        let penalty = ipmip_feas_relax_penalty(&opts.row_penalties, row_idx);
        let slack = add_ipmip_feas_relax_slack(
            &mut c,
            &mut integer_vars,
            &mut ub,
            &mut var_names,
            &mut rows,
            &mut slacks,
            IPMIPFeasRelaxMember::LinearRow(row_idx),
            penalty,
            format!("fr_row_{row_idx}"),
        );
        let mut row = coefs.clone();
        row.resize(c.len(), 0.0);
        row[slack] = -1.0;
        rows.push(row);
        rhs.push(bound);
        con_names.push(format!("feasrelax_{}", ipmip_row_name(p, row_idx)));
    }

    for j in 0..n {
        let upper = source_ub[j];
        if !upper.is_finite() {
            continue;
        }
        let penalty = ipmip_feas_relax_penalty(&opts.upper_bound_penalties, j);
        let slack = add_ipmip_feas_relax_slack(
            &mut c,
            &mut integer_vars,
            &mut ub,
            &mut var_names,
            &mut rows,
            &mut slacks,
            IPMIPFeasRelaxMember::UpperBound(j),
            penalty,
            format!("fr_ub_{}", var_name(p, j)),
        );
        ub[j] = f64::INFINITY;
        let mut row = vec![0.0; c.len()];
        row[j] = 1.0;
        row[slack] = -1.0;
        rows.push(row);
        rhs.push(upper);
        con_names.push(format!("feasrelax_ub_{}", var_name(p, j)));
    }

    IPMIPFeasRelaxModel {
        problem: IPMIPProblem {
            sense: Sense::Min,
            c,
            a: rows,
            b: rhs,
            integer_vars,
            ub: Some(ub),
            var_names: Some(var_names),
            con_names: Some(con_names),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        original_var_count: n,
        slacks,
    }
}

/// Solve a weighted L1 MIP feasibility relaxation and decode positive row/bound
/// violations.
pub fn solve_ipmip_feasibility_relaxation_with_des(
    p: &IPMIPProblem,
    opts: &IPMIPFeasRelaxOptions,
) -> IPMIPFeasRelaxResult {
    let model = build_ipmip_feasibility_relaxation_problem(p, opts);
    let solution = solve_ipmip_with_des(model.problem.clone(), opts.solve_options.clone());
    let status = solution.status;
    let solver = solution.solver_kind.to_string();
    if status != IPMIPStatus::Optimal {
        return IPMIPFeasRelaxResult {
            status,
            x: Vec::new(),
            relaxation_cost: f64::NAN,
            violations: Vec::new(),
            relaxation_solution: solution,
            solver,
            message: Some(format!(
                "relaxation solve ended with status {}",
                status.as_str()
            )),
        };
    }

    let tol = opts.solve_options.int_tol.unwrap_or(1e-6);
    let x = solution.x[..model.original_var_count].to_vec();
    let mut violations = Vec::new();
    for slack in &model.slacks {
        let amount = solution.x.get(slack.slack_var).copied().unwrap_or(0.0);
        if amount > 10.0 * tol {
            violations.push(IPMIPFeasRelaxViolation {
                member: slack.member,
                amount,
                penalty: slack.penalty,
                cost: amount * slack.penalty,
            });
        }
    }

    IPMIPFeasRelaxResult {
        status,
        x,
        relaxation_cost: solution.z,
        violations,
        relaxation_solution: solution,
        solver,
        message: None,
    }
}

/// Return every row, finite upper bound, and integrality marker that may
/// participate in a MIP conflict.
pub fn collect_ipmip_conflict_members(p: &IPMIPProblem) -> Vec<IPMIPConflictMember> {
    validate_ipmip_problem(p);
    let mut members = Vec::new();
    for r in 0..p.a.len() {
        members.push(IPMIPConflictMember::LinearRow(r));
    }
    if let Some(ub) = &p.ub {
        for (j, &upper) in ub.iter().enumerate() {
            if upper.is_finite() {
                members.push(IPMIPConflictMember::UpperBound(j));
            }
        }
    }
    for (j, &is_integer) in p.integer_vars.iter().enumerate() {
        if is_integer {
            members.push(IPMIPConflictMember::Integrality(j));
        }
    }
    members
}

/// Build the feasibility MIP induced by a selected conflict subset.
///
/// Objective coefficients are zeroed; only feasibility status is meaningful.
pub fn ipmip_feasibility_problem_from_conflict_members(
    p: &IPMIPProblem,
    members: &[IPMIPConflictMember],
) -> IPMIPProblem {
    validate_ipmip_problem(p);
    let n = p.c.len();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    let mut row_names = Vec::new();
    let mut integer_vars = vec![false; n];
    let mut ub = vec![f64::INFINITY; n];
    let mut has_upper_bound = false;

    for &member in members {
        match member {
            IPMIPConflictMember::LinearRow(row) => {
                if row >= p.a.len() {
                    panic!("{MODEL}: conflict row {row} out of range");
                }
                rows.push(p.a[row].clone());
                rhs.push(p.b[row]);
                row_names.push(
                    p.con_names
                        .as_ref()
                        .and_then(|names| names.get(row))
                        .cloned()
                        .unwrap_or_else(|| format!("row_{row}")),
                );
            }
            IPMIPConflictMember::UpperBound(var) => {
                if var >= n {
                    panic!("{MODEL}: conflict upper bound {var} out of range");
                }
                let Some(source_ub) = &p.ub else {
                    panic!("{MODEL}: conflict upper bound {var} requested but model has no ub");
                };
                if !source_ub[var].is_finite() {
                    panic!("{MODEL}: conflict upper bound {var} is not finite");
                }
                ub[var] = source_ub[var];
                has_upper_bound = true;
            }
            IPMIPConflictMember::Integrality(var) => {
                if var >= n {
                    panic!("{MODEL}: conflict integrality {var} out of range");
                }
                integer_vars[var] = true;
            }
        }
    }

    if rows.is_empty() {
        rows.push(vec![0.0; n]);
        rhs.push(0.0);
        row_names.push("conflict_dummy_feasible_row".to_string());
    }

    IPMIPProblem {
        sense: Sense::Min,
        c: vec![0.0; n],
        a: rows,
        b: rhs,
        integer_vars,
        ub: if has_upper_bound { Some(ub) } else { None },
        var_names: p.var_names.clone(),
        con_names: Some(row_names),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    }
}

fn ipmip_conflict_subset_status(
    p: &IPMIPProblem,
    members: &[IPMIPConflictMember],
    opts: &IPMIPConflictOptions,
) -> IPMIPStatus {
    let subproblem = ipmip_feasibility_problem_from_conflict_members(p, members);
    solve_ipmip_with_des(subproblem, opts.solve_options.clone()).status
}

/// Find a minimal row/bound/integrality infeasibility conflict for a small MIP.
///
/// This is a deletion-filter conflict refiner: it removes a member whenever the
/// remaining subsystem is still infeasible. The result is minimal by single
/// deletion, not guaranteed minimum-cardinality.
pub fn find_ipmip_infeasibility_conflict(
    p: &IPMIPProblem,
    opts: &IPMIPConflictOptions,
) -> IPMIPInfeasibilityConflict {
    let mut members = collect_ipmip_conflict_members(p);
    let mut checks = 1usize;
    let full_status = ipmip_conflict_subset_status(p, &members, opts);
    if full_status != IPMIPStatus::Infeasible {
        return IPMIPInfeasibilityConflict {
            infeasible: false,
            members: Vec::new(),
            minimal: false,
            checks,
            solver: "ipmip-des-conflict-deletion-filter".to_string(),
            message: Some(format!(
                "model is not infeasible; feasibility status is {}",
                full_status.as_str()
            )),
        };
    }

    let mut idx = 0usize;
    while idx < members.len() {
        let mut trial = members.clone();
        trial.remove(idx);
        checks += 1;
        if ipmip_conflict_subset_status(p, &trial, opts) == IPMIPStatus::Infeasible {
            members = trial;
        } else {
            idx += 1;
        }
    }

    let mut minimal = true;
    for idx in 0..members.len() {
        let mut trial = members.clone();
        trial.remove(idx);
        checks += 1;
        if ipmip_conflict_subset_status(p, &trial, opts) == IPMIPStatus::Infeasible {
            minimal = false;
            break;
        }
    }

    IPMIPInfeasibilityConflict {
        infeasible: true,
        members,
        minimal,
        checks,
        solver: "ipmip-des-conflict-deletion-filter".to_string(),
        message: Some("minimal infeasible MIP row/bound/integrality subsystem".to_string()),
    }
}

/// Compile source-level lower bounds into an ordinary non-negative MIP.
///
/// If `x` is the source variable and `lb` is its lower bound, the compiled
/// variable is `y = x - lb`. Linear rows become `A y <= b - A lb`, upper bounds
/// become `ub - lb`, and the source objective is the compiled objective plus
/// `c·lb`.
pub fn linearize_lower_bounds_problem(problem: &LowerBoundedIPMIPProblem) -> (IPMIPProblem, f64) {
    validate_lower_bounded_ipmip_problem(problem);
    let n = problem.base.c.len();
    let lb = &problem.lb;
    let mut shifted = problem.base.clone();

    for (row, rhs) in shifted.a.iter().zip(shifted.b.iter_mut()) {
        let row_shift: f64 = row.iter().zip(lb).map(|(a, l)| a * l).sum();
        *rhs -= row_shift;
    }

    let source_ub = problem.base.ub.as_ref();
    shifted.ub = Some(
        (0..n)
            .map(|j| {
                let upper = source_ub
                    .and_then(|ub| ub.get(j))
                    .copied()
                    .unwrap_or(f64::INFINITY);
                if upper.is_finite() {
                    upper - lb[j]
                } else {
                    f64::INFINITY
                }
            })
            .collect(),
    );
    shifted.variable_nodes = None;
    shifted.constraint_nodes = None;
    let objective_offset = linear_objective_value(&problem.base.c, lb);
    validate_ipmip_problem(&shifted);
    (shifted, objective_offset)
}

/// Solve a source-level lower-bounded MIP and map the incumbent and objective
/// back into the original source coordinates.
pub fn solve_lower_bounded_ipmip_with_des(
    problem: LowerBoundedIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let lb = problem.lb.clone();
    let (shifted, objective_offset) = linearize_lower_bounds_problem(&problem);
    let mut sol = solve_ipmip_with_des(shifted, shift_mip_start_options(opts, &lb));
    if !sol.x.is_empty() {
        for (x, lower) in sol.x.iter_mut().zip(lb.iter()) {
            *x += lower;
        }
    }
    sol.z = add_objective_offset(sol.z, objective_offset);
    sol.best_bound = add_objective_offset(sol.best_bound, objective_offset);
    for event in &mut sol.trace {
        if let Some(z) = event.lp_z.as_mut() {
            *z = add_objective_offset(*z, objective_offset);
        }
    }
    sol.gap = if sol.status == IPMIPStatus::Optimal {
        0.0
    } else if sol.x.is_empty() || !sol.best_bound.is_finite() || !sol.z.is_finite() {
        f64::INFINITY
    } else {
        (sol.best_bound - sol.z).abs() / 1.0_f64.max(sol.z.abs())
    };
    sol.solver_kind = "in-house-lower-bound-branch-and-cut";
    sol
}

fn shift_mip_start_options(mut opts: IPMIPSolveOptions, lb: &[f64]) -> IPMIPSolveOptions {
    if let Some(start) = opts.mip_start.as_mut() {
        if start.len() != lb.len() {
            panic!(
                "{MODEL}: mip_start length {} does not match lower-bound vector length {}",
                start.len(),
                lb.len()
            );
        }
        for (value, lower) in start.iter_mut().zip(lb.iter()) {
            *value -= lower;
        }
    }
    opts
}

/// Compile source-level row bounds into ordinary `<=` rows. A row upper bound
/// appends `a·x <= upper`; a row lower bound appends `-a·x <= -lower`.
pub fn linearize_general_linear_constraints(
    base: &IPMIPProblem,
    constraints: &[LinearRowConstraint],
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_linear_row_constraint(base, constraint, idx);
        let name = constraint
            .name
            .clone()
            .unwrap_or_else(|| format!("linear_row_{idx}"));
        if let Some(upper) = constraint.upper {
            append_compiled_row(
                &mut out,
                constraint.coefs.clone(),
                upper,
                format!("{name}_upper"),
            );
        }
        if let Some(lower) = constraint.lower {
            append_compiled_row(
                &mut out,
                constraint.coefs.iter().map(|v| -v).collect(),
                -lower,
                format!("{name}_lower"),
            );
        }
    }
    out
}

/// Linearise a [`GeneralLinearIPMIPProblem`] into an ordinary `<=`-row MIP.
pub fn linearize_general_linear_problem(problem: &GeneralLinearIPMIPProblem) -> IPMIPProblem {
    linearize_general_linear_constraints(&problem.base, &problem.linear_constraints)
}

/// Solve a source-level MIP with arbitrary lower/upper linear row bounds.
pub fn solve_general_linear_ipmip_with_des(
    problem: GeneralLinearIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let linearized = linearize_general_linear_problem(&problem);
    solve_ipmip_with_des(linearized, opts)
}

/// Linearise source-level indicator constraints with finite-bound big-M rows.
///
/// The base solver assumes non-negative variables. For a row `a·x <= b`, the
/// compiler computes the smallest M implied by the declared upper bounds and
/// adds either `a·x + M z <= b + M` for `z = 1 => a·x <= b`, or
/// `a·x - M z <= b` for `z = 0 => a·x <= b`.
pub fn linearize_indicator_constraints(
    base: &IPMIPProblem,
    indicators: &[IndicatorConstraint],
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    for (idx, indicator) in indicators.iter().enumerate() {
        validate_indicator_constraint(base, indicator, idx);
        match indicator.sense {
            IndicatorSense::Le => append_indicator_le_row(
                base,
                &mut out,
                indicator,
                indicator.coefs.clone(),
                indicator.rhs,
                None,
            ),
            IndicatorSense::Ge => append_indicator_le_row(
                base,
                &mut out,
                indicator,
                indicator.coefs.iter().map(|v| -v).collect(),
                -indicator.rhs,
                None,
            ),
            IndicatorSense::Eq => {
                append_indicator_le_row(
                    base,
                    &mut out,
                    indicator,
                    indicator.coefs.clone(),
                    indicator.rhs,
                    Some("le"),
                );
                append_indicator_le_row(
                    base,
                    &mut out,
                    indicator,
                    indicator.coefs.iter().map(|v| -v).collect(),
                    -indicator.rhs,
                    Some("ge"),
                );
            }
        }
    }
    out
}

/// Linearise an [`IndicatorIPMIPProblem`] into an ordinary MIP.
pub fn linearize_indicator_problem(problem: &IndicatorIPMIPProblem) -> IPMIPProblem {
    linearize_indicator_constraints(&problem.base, &problem.indicators)
}

/// Solve a source-level indicator MIP through the existing branch-and-cut DES.
pub fn solve_indicator_ipmip_with_des(
    problem: IndicatorIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let linearized = linearize_indicator_problem(&problem);
    solve_ipmip_with_des(linearized, opts)
}

/// Linearise source-level SOS1/SOS2 constraints with finite-bound helper
/// binaries.
pub fn linearize_sos_constraints(base: &IPMIPProblem, sets: &[SpecialOrderedSet]) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, set) in sets.iter().enumerate() {
        let ordered = ordered_sos_vars(base, set, idx);
        match set.kind {
            SpecialOrderedSetKind::Sos1 => append_sos1_rows(base, &mut out, set, idx, &ordered),
            SpecialOrderedSetKind::Sos2 => append_sos2_rows(base, &mut out, set, idx, &ordered),
        }
    }
    out
}

/// Linearise an [`SosIPMIPProblem`] into an ordinary MIP.
pub fn linearize_sos_problem(problem: &SosIPMIPProblem) -> IPMIPProblem {
    linearize_sos_constraints(&problem.base, &problem.sos)
}

/// Solve a source-level SOS MIP through the existing branch-and-cut DES.
pub fn solve_sos_ipmip_with_des(
    problem: SosIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let linearized = linearize_sos_problem(&problem);
    solve_ipmip_with_des(linearized, opts)
}

/// Linearise source-level semi-continuous / semi-integer variables with a
/// binary activation variable per semi variable.
pub fn linearize_semi_variables(base: &IPMIPProblem, semis: &[SemiVariable]) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, semi) in semis.iter().enumerate() {
        validate_semi_variable(base, semi, idx);
        append_semi_variable_rows(base, &mut out, semi, idx);
    }
    out
}

/// Linearise a [`SemiIPMIPProblem`] into an ordinary MIP.
pub fn linearize_semi_problem(problem: &SemiIPMIPProblem) -> IPMIPProblem {
    linearize_semi_variables(&problem.base, &problem.semi_variables)
}

/// Solve a source-level semi-variable MIP through the existing branch-and-cut
/// DES.
pub fn solve_semi_ipmip_with_des(
    problem: SemiIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let linearized = linearize_semi_problem(&problem);
    solve_ipmip_with_des(linearized, opts)
}

/// Linearise source-level piecewise-linear constraints with SOS2 lambda
/// variables.
pub fn linearize_pwl_constraints(
    base: &IPMIPProblem,
    pwl: &[PiecewiseLinearConstraint],
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in pwl.iter().enumerate() {
        validate_pwl_constraint(base, constraint, idx);
        append_pwl_constraint_rows(&mut out, constraint, idx);
    }
    out
}

/// Linearise a [`PwlIPMIPProblem`] into an ordinary MIP.
pub fn linearize_pwl_problem(problem: &PwlIPMIPProblem) -> IPMIPProblem {
    linearize_pwl_constraints(&problem.base, &problem.pwl)
}

/// Solve a source-level PWL MIP through the existing branch-and-cut DES.
pub fn solve_pwl_ipmip_with_des(
    problem: PwlIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let linearized = linearize_pwl_problem(&problem);
    solve_ipmip_with_des(linearized, opts)
}

/// Linearise source-level absolute-value constraints into ordinary MIP rows.
///
/// `base` is the current compiled-coordinate model. `source_lb` and
/// `source_ub` describe the original source-coordinate bounds before any
/// lower-bound shift, so rows can preserve `target = |arg|` exactly.
pub fn linearize_abs_constraints(
    base: &IPMIPProblem,
    constraints: &[AbsoluteValueConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_abs_constraint(base, source_lb, source_ub, constraint, idx);
        append_abs_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise an [`AbsIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_abs_problem(problem: &AbsIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: problem.abs.clone(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    })
}

/// Solve a source-level absolute-value MIP through the existing branch-and-cut
/// DES and map the original variables back into source coordinates.
pub fn solve_abs_ipmip_with_des(
    problem: AbsIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: problem.abs,
            maximums: Vec::new(),
            minimums: Vec::new(),
            logical: Vec::new(),
            l1_norms: Vec::new(),
            linf_norms: Vec::new(),
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-absolute-value-branch-and-cut";
    sol
}

/// Linearise source-level maximum constraints into ordinary MIP rows.
pub fn linearize_maximum_constraints(
    base: &IPMIPProblem,
    constraints: &[MaximumConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_maximum_constraint(base, source_lb, source_ub, constraint, idx);
        append_maximum_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise a [`MaximumIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_maximum_problem(problem: &MaximumIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: problem.maximums.clone(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    })
}

/// Solve a source-level maximum-constraint MIP through the existing
/// branch-and-cut DES and map original variables back into source coordinates.
pub fn solve_maximum_ipmip_with_des(
    problem: MaximumIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: problem.maximums,
            minimums: Vec::new(),
            logical: Vec::new(),
            l1_norms: Vec::new(),
            linf_norms: Vec::new(),
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-maximum-constraint-branch-and-cut";
    sol
}

/// Linearise source-level minimum constraints into ordinary MIP rows.
pub fn linearize_minimum_constraints(
    base: &IPMIPProblem,
    constraints: &[MinimumConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_minimum_constraint(base, source_lb, source_ub, constraint, idx);
        append_minimum_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise a [`MinimumIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_minimum_problem(problem: &MinimumIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: problem.minimums.clone(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    })
}

/// Solve a source-level minimum-constraint MIP through the existing
/// branch-and-cut DES and map original variables back into source coordinates.
pub fn solve_minimum_ipmip_with_des(
    problem: MinimumIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: Vec::new(),
            minimums: problem.minimums,
            logical: Vec::new(),
            l1_norms: Vec::new(),
            linf_norms: Vec::new(),
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-minimum-constraint-branch-and-cut";
    sol
}

/// Linearise source-level binary logical constraints into ordinary MIP rows.
pub fn linearize_logical_constraints(
    base: &IPMIPProblem,
    constraints: &[LogicalConstraint],
    source_lb: &[f64],
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_logical_constraint(base, source_lb, constraint, idx);
        append_logical_constraint_rows(&mut out, constraint, idx);
    }
    out
}

/// Linearise a [`LogicalIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_logical_problem(problem: &LogicalIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: problem.logical.clone(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    })
}

/// Solve a source-level binary logical MIP through the existing branch-and-cut
/// DES and map original variables back into source coordinates.
pub fn solve_logical_ipmip_with_des(
    problem: LogicalIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: Vec::new(),
            minimums: Vec::new(),
            logical: problem.logical,
            l1_norms: Vec::new(),
            linf_norms: Vec::new(),
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-logical-constraint-branch-and-cut";
    sol
}

/// Linearise source-level L1 norm constraints into ordinary MIP rows.
pub fn linearize_l1_norm_constraints(
    base: &IPMIPProblem,
    constraints: &[L1NormConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_l1_norm_constraint(base, source_lb, source_ub, constraint, idx);
        append_l1_norm_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise an [`L1NormIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_l1_norm_problem(problem: &L1NormIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: problem.l1_norms.clone(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    })
}

/// Solve a source-level L1 norm MIP through the existing branch-and-cut DES and
/// map original variables back into source coordinates.
pub fn solve_l1_norm_ipmip_with_des(
    problem: L1NormIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: Vec::new(),
            minimums: Vec::new(),
            logical: Vec::new(),
            l1_norms: problem.l1_norms,
            linf_norms: Vec::new(),
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-l1-norm-constraint-branch-and-cut";
    sol
}

/// Linearise source-level infinity norm constraints into ordinary MIP rows.
pub fn linearize_linf_norm_constraints(
    base: &IPMIPProblem,
    constraints: &[LInfNormConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_linf_norm_constraint(base, source_lb, source_ub, constraint, idx);
        append_linf_norm_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise an [`LInfNormIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_linf_norm_problem(problem: &LInfNormIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: problem.linf_norms.clone(),
        products: Vec::new(),
    })
}

/// Solve a source-level infinity norm MIP through the existing branch-and-cut
/// DES and map original variables back into source coordinates.
pub fn solve_linf_norm_ipmip_with_des(
    problem: LInfNormIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: Vec::new(),
            minimums: Vec::new(),
            logical: Vec::new(),
            l1_norms: Vec::new(),
            linf_norms: problem.linf_norms,
            products: Vec::new(),
        },
        opts,
    );
    sol.solver_kind = "in-house-linf-norm-constraint-branch-and-cut";
    sol
}

/// Linearise exact source-level product constraints into ordinary MIP rows.
///
/// This supports binary-binary and binary-continuous bounded products. General
/// continuous-continuous products are nonconvex and are intentionally rejected
/// by this linear compiler.
pub fn linearize_product_constraints(
    base: &IPMIPProblem,
    constraints: &[ProductConstraint],
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> IPMIPProblem {
    validate_ipmip_problem(base);
    let mut out = base.clone();
    ensure_compilation_metadata(&mut out);
    for (idx, constraint) in constraints.iter().enumerate() {
        validate_product_constraint(base, source_lb, source_ub, constraint, idx);
        append_product_constraint_rows(&mut out, constraint, source_lb, source_ub, idx);
    }
    out
}

/// Linearise a [`ProductIPMIPProblem`] into an ordinary MIP, returning the
/// objective offset and original source variable count alongside the compiled
/// model.
pub fn linearize_product_problem(problem: &ProductIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    linearize_source_ipmip_problem(&SourceIPMIPProblem {
        base: problem.base.clone(),
        lb: problem.lb.clone(),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: problem.products.clone(),
    })
}

/// Solve a source-level product MIP through the existing branch-and-cut DES and
/// map original variables back into source coordinates.
pub fn solve_product_ipmip_with_des(
    problem: ProductIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let mut sol = solve_source_ipmip_with_des(
        SourceIPMIPProblem {
            base: problem.base,
            lb: problem.lb,
            linear_constraints: Vec::new(),
            indicators: Vec::new(),
            sos: Vec::new(),
            semi_variables: Vec::new(),
            pwl: Vec::new(),
            abs: Vec::new(),
            maximums: Vec::new(),
            minimums: Vec::new(),
            logical: Vec::new(),
            l1_norms: Vec::new(),
            linf_norms: Vec::new(),
            products: problem.products,
        },
        opts,
    );
    sol.solver_kind = "in-house-product-constraint-branch-and-cut";
    sol
}

/// Compile source-level quadratic objective terms into ordinary linear MIP
/// objective coefficients plus exact product helper rows.
pub fn linearize_quadratic_objective_problem(
    problem: &QuadraticObjectiveIPMIPProblem,
) -> (IPMIPProblem, f64, usize) {
    validate_quadratic_objective_problem(problem);
    let original_var_count = problem.base.c.len();
    let source_lb = problem
        .lb
        .clone()
        .unwrap_or_else(|| vec![0.0; original_var_count]);
    let mut source_ub = problem
        .base
        .ub
        .clone()
        .unwrap_or_else(|| vec![f64::INFINITY; original_var_count]);

    let (mut working, mut objective_offset) = if problem.lb.is_some() {
        linearize_lower_bounds_problem(&LowerBoundedIPMIPProblem {
            base: problem.base.clone(),
            lb: source_lb.clone(),
        })
    } else {
        (problem.base.clone(), 0.0)
    };
    ensure_compilation_metadata(&mut working);
    let mut compiled_lb = source_lb;

    for (idx, term) in problem.quadratic_objective.iter().enumerate() {
        if term.x_var == term.y_var {
            if !product_var_is_binary(&working, &compiled_lb, Some(&source_ub), term.x_var) {
                panic!(
                    "{MODEL}: quadratic objective term {idx} square is exact only for binary variables"
                );
            }
            working.c[term.x_var] += term.coeff;
            continue;
        }

        let (product_lb, product_ub) =
            quadratic_product_bounds(&working, &compiled_lb, &source_ub, term, idx);
        let helper_name = term
            .name
            .clone()
            .unwrap_or_else(|| format!("quadratic_objective_{idx}"));
        let helper = append_continuous_helper_var(
            &mut working,
            helper_name.clone(),
            product_ub - product_lb,
        );
        working.c[helper] = term.coeff;
        objective_offset += term.coeff * product_lb;
        compiled_lb.push(product_lb);
        source_ub.push(product_ub);
        let constraint = ProductConstraint {
            target_var: helper,
            x_var: term.x_var,
            y_var: term.y_var,
            name: Some(helper_name),
        };
        validate_product_constraint(&working, &compiled_lb, Some(&source_ub), &constraint, idx);
        append_product_constraint_rows(
            &mut working,
            &constraint,
            &compiled_lb,
            Some(&source_ub),
            idx,
        );
    }

    (working, objective_offset, original_var_count)
}

/// Solve a MIP with source-level quadratic objective terms through the ordinary
/// branch-and-cut backend after exact product linearization.
pub fn solve_quadratic_objective_ipmip_with_des(
    problem: QuadraticObjectiveIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let lb = problem
        .lb
        .clone()
        .unwrap_or_else(|| vec![0.0; problem.base.c.len()]);
    let (linearized, objective_offset, original_var_count) =
        linearize_quadratic_objective_problem(&problem);
    let mut sol = solve_ipmip_with_des(linearized, opts);
    if !sol.x.is_empty() {
        for (x, lower) in sol.x.iter_mut().take(original_var_count).zip(lb.iter()) {
            *x += lower;
        }
    }
    sol.z = add_objective_offset(sol.z, objective_offset);
    sol.best_bound = add_objective_offset(sol.best_bound, objective_offset);
    for event in &mut sol.trace {
        if let Some(z) = event.lp_z.as_mut() {
            *z = add_objective_offset(*z, objective_offset);
        }
    }
    sol.gap = if sol.status == IPMIPStatus::Optimal {
        0.0
    } else if sol.x.is_empty() || !sol.best_bound.is_finite() || !sol.z.is_finite() {
        f64::INFINITY
    } else {
        (sol.best_bound - sol.z).abs() / 1.0_f64.max(sol.z.abs())
    };
    sol.solver_kind = "in-house-quadratic-objective-branch-and-cut";
    sol
}

/// Compile a feature-rich source MIP into the ordinary non-negative MIP
/// accepted by the branch-and-cut backend. Source lower bounds are shifted
/// first, and every source-level row that references original variables is
/// translated into those shifted coordinates before feature linearization.
pub fn linearize_source_ipmip_problem(problem: &SourceIPMIPProblem) -> (IPMIPProblem, f64, usize) {
    validate_source_ipmip_problem(problem);
    let original_var_count = problem.base.c.len();
    let lb = source_lower_bounds(problem);
    validate_zero_source_lbs_for_zero_sensitive_features(problem, &lb);

    let (mut working, objective_offset) = if problem.lb.is_some() {
        linearize_lower_bounds_problem(&LowerBoundedIPMIPProblem {
            base: problem.base.clone(),
            lb: lb.clone(),
        })
    } else {
        (problem.base.clone(), 0.0)
    };

    let shifted_rows = shift_linear_row_constraints(&problem.linear_constraints, &lb);
    if !shifted_rows.is_empty() {
        working = linearize_general_linear_constraints(&working, &shifted_rows);
    }

    let shifted_indicators = shift_indicator_constraints(&problem.indicators, &lb);
    if !shifted_indicators.is_empty() {
        working = linearize_indicator_constraints(&working, &shifted_indicators);
    }

    if !problem.abs.is_empty() {
        working =
            linearize_abs_constraints(&working, &problem.abs, &lb, problem.base.ub.as_deref());
    }

    if !problem.maximums.is_empty() {
        working = linearize_maximum_constraints(
            &working,
            &problem.maximums,
            &lb,
            problem.base.ub.as_deref(),
        );
    }

    if !problem.minimums.is_empty() {
        working = linearize_minimum_constraints(
            &working,
            &problem.minimums,
            &lb,
            problem.base.ub.as_deref(),
        );
    }

    if !problem.logical.is_empty() {
        working = linearize_logical_constraints(&working, &problem.logical, &lb);
    }

    if !problem.l1_norms.is_empty() {
        working = linearize_l1_norm_constraints(
            &working,
            &problem.l1_norms,
            &lb,
            problem.base.ub.as_deref(),
        );
    }

    if !problem.linf_norms.is_empty() {
        working = linearize_linf_norm_constraints(
            &working,
            &problem.linf_norms,
            &lb,
            problem.base.ub.as_deref(),
        );
    }

    if !problem.products.is_empty() {
        working = linearize_product_constraints(
            &working,
            &problem.products,
            &lb,
            problem.base.ub.as_deref(),
        );
    }

    if !problem.sos.is_empty() {
        working = linearize_sos_constraints(&working, &problem.sos);
    }

    if !problem.semi_variables.is_empty() {
        working = linearize_semi_variables(&working, &problem.semi_variables);
    }

    let shifted_pwl = shift_pwl_constraints(&problem.pwl, &lb);
    if !shifted_pwl.is_empty() {
        working = linearize_pwl_constraints(&working, &shifted_pwl);
    }

    (working, objective_offset, original_var_count)
}

/// Solve a feature-rich source MIP and map the original variables and objective
/// back into source coordinates. Helper variables introduced by linearization
/// remain in the returned `x` vector after the original variables.
pub fn solve_source_ipmip_with_des(
    problem: SourceIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> IPMIPSolution {
    let lb = source_lower_bounds(&problem);
    let (linearized, objective_offset, original_var_count) =
        linearize_source_ipmip_problem(&problem);
    let mut sol = solve_ipmip_with_des(linearized, opts);
    if !sol.x.is_empty() {
        for (x, lower) in sol.x.iter_mut().take(original_var_count).zip(lb.iter()) {
            *x += lower;
        }
    }
    sol.z = add_objective_offset(sol.z, objective_offset);
    sol.best_bound = add_objective_offset(sol.best_bound, objective_offset);
    for event in &mut sol.trace {
        if let Some(z) = event.lp_z.as_mut() {
            *z = add_objective_offset(*z, objective_offset);
        }
    }
    sol.gap = if sol.status == IPMIPStatus::Optimal {
        0.0
    } else if sol.x.is_empty() || !sol.best_bound.is_finite() || !sol.z.is_finite() {
        f64::INFINITY
    } else {
        (sol.best_bound - sol.z).abs() / 1.0_f64.max(sol.z.abs())
    };
    sol.solver_kind = "in-house-source-feature-branch-and-cut";
    sol
}

/// Solve lexicographic multi-objective MIPs by optimizing each objective in
/// priority order and fixing each proven optimum before moving to the next.
pub fn solve_multi_objective_ipmip_with_des(
    problem: MultiObjectiveIPMIPProblem,
    opts: IPMIPSolveOptions,
) -> MultiObjectiveIPMIPSolution {
    validate_multi_objective_problem(&problem);
    let t0 = Instant::now();
    let mut working = problem.base.clone();
    ensure_compilation_metadata(&mut working);
    let mut stage_solutions = Vec::with_capacity(problem.objectives.len());

    for (idx, objective) in problem.objectives.iter().enumerate() {
        working.sense = objective.sense;
        working.c = objective.c.clone();
        let sol = solve_ipmip_with_des(working.clone(), opts.clone());
        let status = sol.status;
        stage_solutions.push(sol.clone());
        if status != IPMIPStatus::Optimal {
            return MultiObjectiveIPMIPSolution {
                status,
                x: sol.x,
                objective_values: Vec::new(),
                stage_solutions,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                solver_kind: "in-house-lexicographic-branch-and-cut",
            };
        }
        let optimum = linear_objective_value(&objective.c, &sol.x);
        append_compiled_equality(
            &mut working,
            objective.c.clone(),
            optimum,
            objective
                .name
                .clone()
                .unwrap_or_else(|| format!("multi_objective_{idx}")),
        );
    }

    let x = stage_solutions
        .last()
        .map(|sol| sol.x.clone())
        .unwrap_or_default();
    let objective_values = problem
        .objectives
        .iter()
        .map(|objective| linear_objective_value(&objective.c, &x))
        .collect();
    MultiObjectiveIPMIPSolution {
        status: IPMIPStatus::Optimal,
        x,
        objective_values,
        stage_solutions,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        solver_kind: "in-house-lexicographic-branch-and-cut",
    }
}

fn validate_indicator_constraint(base: &IPMIPProblem, indicator: &IndicatorConstraint, idx: usize) {
    let n = base.c.len();
    if indicator.binary_var >= n {
        panic!(
            "{MODEL}: indicator {idx} binary_var {} out of range {n}",
            indicator.binary_var
        );
    }
    if !base.integer_vars[indicator.binary_var] {
        panic!(
            "{MODEL}: indicator {idx} trigger variable {} must be integer/binary",
            indicator.binary_var
        );
    }
    let trigger_ub = base
        .ub
        .as_ref()
        .and_then(|ub| ub.get(indicator.binary_var))
        .copied()
        .unwrap_or(f64::INFINITY);
    if !trigger_ub.is_finite() || trigger_ub > 1.0 + 1e-9 {
        panic!(
            "{MODEL}: indicator {idx} trigger variable {} must have finite binary upper bound <= 1",
            indicator.binary_var
        );
    }
    if indicator.coefs.len() != n {
        panic!(
            "{MODEL}: indicator {idx} coefficient length {} does not match variable count {n}",
            indicator.coefs.len()
        );
    }
    if !indicator.rhs.is_finite() {
        panic!("{MODEL}: indicator {idx} rhs must be finite");
    }
    for (j, &coef) in indicator.coefs.iter().enumerate() {
        if !coef.is_finite() {
            panic!("{MODEL}: indicator {idx} coefficient {j} must be finite");
        }
    }
}

fn validate_source_ipmip_problem(problem: &SourceIPMIPProblem) {
    if let Some(lb) = &problem.lb {
        validate_lower_bounded_ipmip_problem(&LowerBoundedIPMIPProblem {
            base: problem.base.clone(),
            lb: lb.clone(),
        });
    } else {
        validate_ipmip_problem(&problem.base);
    }
}

fn validate_quadratic_objective_problem(problem: &QuadraticObjectiveIPMIPProblem) {
    if let Some(lb) = &problem.lb {
        validate_lower_bounded_ipmip_problem(&LowerBoundedIPMIPProblem {
            base: problem.base.clone(),
            lb: lb.clone(),
        });
    } else {
        validate_ipmip_problem(&problem.base);
    }
    if problem.quadratic_objective.is_empty() {
        panic!("{MODEL}: quadratic objective needs at least one term");
    }
    let n = problem.base.c.len();
    let source_lb = problem.lb.clone().unwrap_or_else(|| vec![0.0; n]);
    let source_ub = problem
        .base
        .ub
        .clone()
        .unwrap_or_else(|| vec![f64::INFINITY; n]);
    for (idx, term) in problem.quadratic_objective.iter().enumerate() {
        if term.x_var >= n {
            panic!(
                "{MODEL}: quadratic objective term {idx} x_var {} out of range {n}",
                term.x_var
            );
        }
        if term.y_var >= n {
            panic!(
                "{MODEL}: quadratic objective term {idx} y_var {} out of range {n}",
                term.y_var
            );
        }
        if !term.coeff.is_finite() {
            panic!("{MODEL}: quadratic objective term {idx} coeff must be finite");
        }
        if term.x_var == term.y_var {
            if !product_var_is_binary(&problem.base, &source_lb, Some(&source_ub), term.x_var) {
                panic!(
                    "{MODEL}: quadratic objective term {idx} square is exact only for binary variables"
                );
            }
        } else {
            quadratic_product_bounds(&problem.base, &source_lb, &source_ub, term, idx);
        }
    }
}

fn source_lower_bounds(problem: &SourceIPMIPProblem) -> Vec<f64> {
    problem
        .lb
        .clone()
        .unwrap_or_else(|| vec![0.0; problem.base.c.len()])
}

fn validate_zero_source_lbs_for_zero_sensitive_features(problem: &SourceIPMIPProblem, lb: &[f64]) {
    for (idx, indicator) in problem.indicators.iter().enumerate() {
        validate_zero_source_lb(lb, indicator.binary_var, "indicator trigger", idx);
    }
    for (idx, constraint) in problem.logical.iter().enumerate() {
        validate_zero_source_lb(lb, constraint.target_var, "logical target", idx);
        for &var in &constraint.arg_vars {
            validate_zero_source_lb(lb, var, "logical argument", idx);
        }
    }
    for (idx, set) in problem.sos.iter().enumerate() {
        for &var in &set.vars {
            validate_zero_source_lb(lb, var, "sos variable", idx);
        }
    }
    for (idx, semi) in problem.semi_variables.iter().enumerate() {
        validate_zero_source_lb(lb, semi.var, "semi variable", idx);
    }
}

fn validate_zero_source_lb(lb: &[f64], var: usize, kind: &str, idx: usize) {
    if var >= lb.len() {
        panic!("{MODEL}: {kind} {idx} variable {var} out of range");
    }
    if lb[var].abs() > 1e-12 {
        panic!(
            "{MODEL}: {kind} {idx} variable {var} must have source lower bound 0 before compilation"
        );
    }
}

fn shift_linear_row_constraints(
    constraints: &[LinearRowConstraint],
    lb: &[f64],
) -> Vec<LinearRowConstraint> {
    constraints
        .iter()
        .enumerate()
        .map(|(idx, constraint)| {
            if constraint.coefs.len() != lb.len() {
                panic!(
                    "{MODEL}: linear constraint {idx} coefficient length {} does not match variable count {}",
                    constraint.coefs.len(),
                    lb.len()
                );
            }
            let shift = linear_objective_value(&constraint.coefs, lb);
            LinearRowConstraint {
                coefs: constraint.coefs.clone(),
                lower: constraint.lower.map(|v| v - shift),
                upper: constraint.upper.map(|v| v - shift),
                name: constraint.name.clone(),
            }
        })
        .collect()
}

fn shift_indicator_constraints(
    indicators: &[IndicatorConstraint],
    lb: &[f64],
) -> Vec<IndicatorConstraint> {
    indicators
        .iter()
        .enumerate()
        .map(|(idx, indicator)| {
            if indicator.coefs.len() != lb.len() {
                panic!(
                    "{MODEL}: indicator {idx} coefficient length {} does not match variable count {}",
                    indicator.coefs.len(),
                    lb.len()
                );
            }
            let shift = linear_objective_value(&indicator.coefs, lb);
            IndicatorConstraint {
                binary_var: indicator.binary_var,
                active_value: indicator.active_value,
                coefs: indicator.coefs.clone(),
                sense: indicator.sense,
                rhs: indicator.rhs - shift,
                name: indicator.name.clone(),
            }
        })
        .collect()
}

fn shift_pwl_constraints(
    constraints: &[PiecewiseLinearConstraint],
    lb: &[f64],
) -> Vec<PiecewiseLinearConstraint> {
    constraints
        .iter()
        .enumerate()
        .map(|(idx, pwl)| {
            if pwl.x_var >= lb.len() {
                panic!("{MODEL}: pwl {idx} x_var {} out of range", pwl.x_var);
            }
            if pwl.y_var >= lb.len() {
                panic!("{MODEL}: pwl {idx} y_var {} out of range", pwl.y_var);
            }
            PiecewiseLinearConstraint {
                x_var: pwl.x_var,
                y_var: pwl.y_var,
                points: pwl
                    .points
                    .iter()
                    .map(|point| PiecewiseLinearPoint {
                        x: point.x - lb[pwl.x_var],
                        y: point.y - lb[pwl.y_var],
                    })
                    .collect(),
                name: pwl.name.clone(),
            }
        })
        .collect()
}

fn validate_abs_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &AbsoluteValueConstraint,
    idx: usize,
) {
    let n = base.c.len();
    if source_lb.len() != n {
        panic!(
            "{MODEL}: abs {idx} source lower-bound length {} does not match variable count {n}",
            source_lb.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: abs {idx} source upper-bound length {} does not match variable count {n}",
                ub.len()
            );
        }
    }
    if constraint.arg_var >= n {
        panic!(
            "{MODEL}: abs {idx} arg_var {} out of range {n}",
            constraint.arg_var
        );
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: abs {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    if constraint.arg_var == constraint.target_var {
        panic!("{MODEL}: abs {idx} arg_var and target_var must be distinct");
    }

    let lower = source_lb[constraint.arg_var];
    let target_lower = source_lb[constraint.target_var];
    if !lower.is_finite() {
        panic!("{MODEL}: abs {idx} argument lower bound must be finite");
    }
    if !target_lower.is_finite() {
        panic!("{MODEL}: abs {idx} target lower bound must be finite");
    }
    let upper = source_ub
        .and_then(|ub| ub.get(constraint.arg_var))
        .copied()
        .unwrap_or(f64::INFINITY);
    if upper.is_finite() && upper + 1e-9 < lower {
        panic!("{MODEL}: abs {idx} argument lower bound {lower} exceeds upper bound {upper}");
    }
    if lower < -1e-12 && !upper.is_finite() {
        panic!("{MODEL}: abs {idx} mixed-sign argument needs a finite upper bound");
    }
}

fn append_abs_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &AbsoluteValueConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("abs_{idx}"));
    let arg = constraint.arg_var;
    let target = constraint.target_var;
    let arg_lb = source_lb[arg];
    let target_lb = source_lb[target];
    let arg_ub = source_ub
        .and_then(|ub| ub.get(arg))
        .copied()
        .unwrap_or(f64::INFINITY);

    if arg_lb >= -1e-12 {
        let mut row = vec![0.0; out.c.len()];
        row[target] = 1.0;
        row[arg] = -1.0;
        append_compiled_equality(out, row, arg_lb - target_lb, format!("{name}_nonnegative"));
        return;
    }
    if arg_ub <= 1e-12 {
        let mut row = vec![0.0; out.c.len()];
        row[target] = 1.0;
        row[arg] = 1.0;
        append_compiled_equality(out, row, -arg_lb - target_lb, format!("{name}_nonpositive"));
        return;
    }

    let selector = append_binary_helper_var(out, format!("{name}_sign"));

    let mut row = vec![0.0; out.c.len()];
    row[arg] = 1.0;
    row[target] = -1.0;
    append_compiled_row(
        out,
        row,
        target_lb - arg_lb,
        format!("{name}_target_ge_arg"),
    );

    let mut row = vec![0.0; out.c.len()];
    row[arg] = -1.0;
    row[target] = -1.0;
    append_compiled_row(
        out,
        row,
        arg_lb + target_lb,
        format!("{name}_target_ge_neg_arg"),
    );

    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[arg] = -1.0;
    row[selector] = -2.0 * arg_lb;
    append_compiled_row(
        out,
        row,
        -arg_lb - target_lb,
        format!("{name}_target_le_positive_branch"),
    );

    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[arg] = 1.0;
    row[selector] = -2.0 * arg_ub;
    append_compiled_row(
        out,
        row,
        -arg_lb - target_lb,
        format!("{name}_target_le_negative_branch"),
    );
}

#[derive(Clone, Copy, Debug)]
enum MaximumCandidate {
    Var(usize),
    Constant(f64),
}

fn validate_maximum_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &MaximumConstraint,
    idx: usize,
) {
    let n = base.c.len();
    if source_lb.len() != n {
        panic!(
            "{MODEL}: maximum {idx} source lower-bound length {} does not match variable count {n}",
            source_lb.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: maximum {idx} source upper-bound length {} does not match variable count {n}",
                ub.len()
            );
        }
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: maximum {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    let target_lb = source_lb[constraint.target_var];
    if !target_lb.is_finite() {
        panic!("{MODEL}: maximum {idx} target lower bound must be finite");
    }
    if constraint.arg_vars.is_empty() && constraint.constant.is_none() {
        panic!("{MODEL}: maximum {idx} needs at least one argument or a constant");
    }
    let mut seen = HashSet::new();
    for &var in &constraint.arg_vars {
        if var >= n {
            panic!("{MODEL}: maximum {idx} argument variable {var} out of range {n}");
        }
        if var == constraint.target_var {
            panic!("{MODEL}: maximum {idx} target_var must be distinct from argument variables");
        }
        if !seen.insert(var) {
            panic!("{MODEL}: maximum {idx} duplicate argument variable {var}");
        }
        if !source_lb[var].is_finite() {
            panic!("{MODEL}: maximum {idx} argument variable {var} lower bound must be finite");
        }
    }
    if let Some(constant) = constraint.constant {
        if !constant.is_finite() {
            panic!("{MODEL}: maximum {idx} constant must be finite");
        }
    }
    maximum_constraint_upper_bound(source_lb, source_ub, constraint, idx);
}

fn maximum_constraint_candidates(constraint: &MaximumConstraint) -> Vec<MaximumCandidate> {
    let mut candidates: Vec<MaximumCandidate> = constraint
        .arg_vars
        .iter()
        .copied()
        .map(MaximumCandidate::Var)
        .collect();
    if let Some(constant) = constraint.constant {
        candidates.push(MaximumCandidate::Constant(constant));
    }
    candidates
}

fn maximum_constraint_upper_bound(
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &MaximumConstraint,
    idx: usize,
) -> f64 {
    let target_upper = source_ub
        .and_then(|ub| ub.get(constraint.target_var))
        .copied()
        .unwrap_or(f64::INFINITY);
    if target_upper.is_finite() {
        return target_upper;
    }

    let mut upper = f64::NEG_INFINITY;
    for candidate in maximum_constraint_candidates(constraint) {
        let candidate_upper = match candidate {
            MaximumCandidate::Var(var) => source_ub
                .and_then(|ub| ub.get(var))
                .copied()
                .unwrap_or(f64::INFINITY),
            MaximumCandidate::Constant(value) => value,
        };
        upper = upper.max(candidate_upper);
    }
    if !upper.is_finite() {
        panic!(
            "{MODEL}: maximum {idx} needs finite argument uppers or a finite target upper bound"
        );
    }
    if !source_lb[constraint.target_var].is_finite() {
        panic!("{MODEL}: maximum {idx} target lower bound must be finite");
    }
    upper
}

fn maximum_candidate_lower(candidate: MaximumCandidate, source_lb: &[f64]) -> f64 {
    match candidate {
        MaximumCandidate::Var(var) => source_lb[var],
        MaximumCandidate::Constant(value) => value,
    }
}

fn append_maximum_candidate_lower_row(
    out: &mut IPMIPProblem,
    candidate: MaximumCandidate,
    source_lb: &[f64],
    target: usize,
    name: &str,
    pos: usize,
) {
    let target_lb = source_lb[target];
    match candidate {
        MaximumCandidate::Var(var) => {
            let mut row = vec![0.0; out.c.len()];
            row[var] = 1.0;
            row[target] = -1.0;
            append_compiled_row(
                out,
                row,
                target_lb - source_lb[var],
                format!("{name}_target_ge_arg_{pos}"),
            );
        }
        MaximumCandidate::Constant(value) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = -1.0;
            append_compiled_row(
                out,
                row,
                target_lb - value,
                format!("{name}_target_ge_constant"),
            );
        }
    }
}

fn append_maximum_candidate_equality(
    out: &mut IPMIPProblem,
    candidate: MaximumCandidate,
    source_lb: &[f64],
    target: usize,
    name: &str,
) {
    let target_lb = source_lb[target];
    match candidate {
        MaximumCandidate::Var(var) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            row[var] = -1.0;
            append_compiled_equality(
                out,
                row,
                source_lb[var] - target_lb,
                format!("{name}_single_arg"),
            );
        }
        MaximumCandidate::Constant(value) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            append_compiled_equality(out, row, value - target_lb, format!("{name}_constant"));
        }
    }
}

fn append_maximum_candidate_upper_row(
    out: &mut IPMIPProblem,
    candidate: MaximumCandidate,
    source_lb: &[f64],
    target: usize,
    selector: usize,
    max_upper: f64,
    name: &str,
    pos: usize,
) {
    let target_lb = source_lb[target];
    let candidate_lower = maximum_candidate_lower(candidate, source_lb);
    let big_m = (max_upper - candidate_lower).max(0.0);
    if !big_m.is_finite() {
        panic!("{MODEL}: maximum {name} candidate {pos} has non-finite big-M");
    }
    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[selector] = big_m;
    let rhs = match candidate {
        MaximumCandidate::Var(var) => {
            row[var] = -1.0;
            source_lb[var] + big_m - target_lb
        }
        MaximumCandidate::Constant(value) => value + big_m - target_lb,
    };
    append_compiled_row(out, row, rhs, format!("{name}_target_le_candidate_{pos}"));
}

fn append_maximum_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &MaximumConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("maximum_{idx}"));
    let target = constraint.target_var;
    let candidates = maximum_constraint_candidates(constraint);
    if candidates.len() == 1 {
        append_maximum_candidate_equality(out, candidates[0], source_lb, target, &name);
        return;
    }

    let max_upper = maximum_constraint_upper_bound(source_lb, source_ub, constraint, idx);
    let selectors: Vec<usize> = candidates
        .iter()
        .enumerate()
        .map(|(pos, _)| append_binary_helper_var(out, format!("{name}_select_{pos}")))
        .collect();

    for (pos, &candidate) in candidates.iter().enumerate() {
        append_maximum_candidate_lower_row(out, candidate, source_lb, target, &name, pos);
        append_maximum_candidate_upper_row(
            out,
            candidate,
            source_lb,
            target,
            selectors[pos],
            max_upper,
            &name,
            pos,
        );
    }

    let mut row = vec![0.0; out.c.len()];
    for selector in selectors {
        row[selector] = 1.0;
    }
    append_compiled_equality(out, row, 1.0, format!("{name}_one_active"));
}

#[derive(Clone, Copy, Debug)]
enum MinimumCandidate {
    Var(usize),
    Constant(f64),
}

fn validate_minimum_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &MinimumConstraint,
    idx: usize,
) {
    let n = base.c.len();
    if source_lb.len() != n {
        panic!(
            "{MODEL}: minimum {idx} source lower-bound length {} does not match variable count {n}",
            source_lb.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: minimum {idx} source upper-bound length {} does not match variable count {n}",
                ub.len()
            );
        }
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: minimum {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    if !source_lb[constraint.target_var].is_finite() {
        panic!("{MODEL}: minimum {idx} target lower bound must be finite");
    }
    if constraint.arg_vars.is_empty() && constraint.constant.is_none() {
        panic!("{MODEL}: minimum {idx} needs at least one argument or a constant");
    }
    let mut seen = HashSet::new();
    for &var in &constraint.arg_vars {
        if var >= n {
            panic!("{MODEL}: minimum {idx} argument variable {var} out of range {n}");
        }
        if var == constraint.target_var {
            panic!("{MODEL}: minimum {idx} target_var must be distinct from argument variables");
        }
        if !seen.insert(var) {
            panic!("{MODEL}: minimum {idx} duplicate argument variable {var}");
        }
        if !source_lb[var].is_finite() {
            panic!("{MODEL}: minimum {idx} argument variable {var} lower bound must be finite");
        }
    }
    if let Some(constant) = constraint.constant {
        if !constant.is_finite() {
            panic!("{MODEL}: minimum {idx} constant must be finite");
        }
    }
    for candidate in minimum_constraint_candidates(constraint) {
        minimum_candidate_upper(candidate, source_ub, idx);
    }
}

fn minimum_constraint_candidates(constraint: &MinimumConstraint) -> Vec<MinimumCandidate> {
    let mut candidates: Vec<MinimumCandidate> = constraint
        .arg_vars
        .iter()
        .copied()
        .map(MinimumCandidate::Var)
        .collect();
    if let Some(constant) = constraint.constant {
        candidates.push(MinimumCandidate::Constant(constant));
    }
    candidates
}

fn minimum_candidate_upper(
    candidate: MinimumCandidate,
    source_ub: Option<&[f64]>,
    idx: usize,
) -> f64 {
    match candidate {
        MinimumCandidate::Var(var) => {
            let upper = source_ub
                .and_then(|ub| ub.get(var))
                .copied()
                .unwrap_or(f64::INFINITY);
            if !upper.is_finite() {
                panic!("{MODEL}: minimum {idx} argument variable {var} needs a finite upper bound");
            }
            upper
        }
        MinimumCandidate::Constant(value) => value,
    }
}

fn minimum_candidate_lower(candidate: MinimumCandidate, source_lb: &[f64]) -> f64 {
    match candidate {
        MinimumCandidate::Var(var) => source_lb[var],
        MinimumCandidate::Constant(value) => value,
    }
}

fn append_minimum_candidate_upper_row(
    out: &mut IPMIPProblem,
    candidate: MinimumCandidate,
    source_lb: &[f64],
    target: usize,
    name: &str,
    pos: usize,
) {
    let target_lb = source_lb[target];
    match candidate {
        MinimumCandidate::Var(var) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            row[var] = -1.0;
            append_compiled_row(
                out,
                row,
                source_lb[var] - target_lb,
                format!("{name}_target_le_arg_{pos}"),
            );
        }
        MinimumCandidate::Constant(value) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            append_compiled_row(
                out,
                row,
                value - target_lb,
                format!("{name}_target_le_constant"),
            );
        }
    }
}

fn append_minimum_candidate_equality(
    out: &mut IPMIPProblem,
    candidate: MinimumCandidate,
    source_lb: &[f64],
    target: usize,
    name: &str,
) {
    let target_lb = source_lb[target];
    match candidate {
        MinimumCandidate::Var(var) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            row[var] = -1.0;
            append_compiled_equality(
                out,
                row,
                source_lb[var] - target_lb,
                format!("{name}_single_arg"),
            );
        }
        MinimumCandidate::Constant(value) => {
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            append_compiled_equality(out, row, value - target_lb, format!("{name}_constant"));
        }
    }
}

fn append_minimum_candidate_lower_row(
    out: &mut IPMIPProblem,
    candidate: MinimumCandidate,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    target: usize,
    selector: usize,
    name: &str,
    pos: usize,
) {
    let target_lb = source_lb[target];
    let candidate_upper = minimum_candidate_upper(candidate, source_ub, pos);
    let big_m = (candidate_upper - target_lb).max(0.0);
    if !big_m.is_finite() {
        panic!("{MODEL}: minimum {name} candidate {pos} has non-finite big-M");
    }
    let mut row = vec![0.0; out.c.len()];
    row[target] = -1.0;
    row[selector] = big_m;
    let rhs = match candidate {
        MinimumCandidate::Var(var) => {
            row[var] = 1.0;
            target_lb - source_lb[var] + big_m
        }
        MinimumCandidate::Constant(value) => target_lb - value + big_m,
    };
    append_compiled_row(out, row, rhs, format!("{name}_target_ge_candidate_{pos}"));
}

fn append_minimum_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &MinimumConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("minimum_{idx}"));
    let target = constraint.target_var;
    let candidates = minimum_constraint_candidates(constraint);
    if candidates.len() == 1 {
        append_minimum_candidate_equality(out, candidates[0], source_lb, target, &name);
        return;
    }

    let selectors: Vec<usize> = candidates
        .iter()
        .enumerate()
        .map(|(pos, _)| append_binary_helper_var(out, format!("{name}_select_{pos}")))
        .collect();

    for (pos, &candidate) in candidates.iter().enumerate() {
        append_minimum_candidate_upper_row(out, candidate, source_lb, target, &name, pos);
        append_minimum_candidate_lower_row(
            out,
            candidate,
            source_lb,
            source_ub,
            target,
            selectors[pos],
            &name,
            pos,
        );
    }

    let mut row = vec![0.0; out.c.len()];
    for selector in selectors {
        row[selector] = 1.0;
    }
    append_compiled_equality(out, row, 1.0, format!("{name}_one_active"));
}

fn validate_logical_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    constraint: &LogicalConstraint,
    idx: usize,
) {
    let n = base.c.len();
    if source_lb.len() != n {
        panic!(
            "{MODEL}: logical {idx} source lower-bound length {} does not match variable count {n}",
            source_lb.len()
        );
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: logical {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    if constraint.arg_vars.is_empty() {
        panic!("{MODEL}: logical {idx} needs at least one argument");
    }
    validate_binary_source_var(
        base,
        source_lb,
        constraint.target_var,
        "logical target",
        idx,
    );
    let mut seen = HashSet::new();
    for &var in &constraint.arg_vars {
        if var >= n {
            panic!("{MODEL}: logical {idx} argument variable {var} out of range {n}");
        }
        if var == constraint.target_var {
            panic!("{MODEL}: logical {idx} target_var must be distinct from argument variables");
        }
        if !seen.insert(var) {
            panic!("{MODEL}: logical {idx} duplicate argument variable {var}");
        }
        validate_binary_source_var(base, source_lb, var, "logical argument", idx);
    }
}

fn validate_binary_source_var(
    base: &IPMIPProblem,
    source_lb: &[f64],
    var: usize,
    kind: &str,
    idx: usize,
) {
    if source_lb[var].abs() > 1e-12 {
        panic!("{MODEL}: {kind} {idx} variable {var} must have source lower bound 0");
    }
    if !base.integer_vars[var] {
        panic!("{MODEL}: {kind} {idx} variable {var} must be integer/binary");
    }
    let upper = base
        .ub
        .as_ref()
        .and_then(|ub| ub.get(var))
        .copied()
        .unwrap_or(f64::INFINITY);
    if !upper.is_finite() || upper > 1.0 + 1e-9 {
        panic!("{MODEL}: {kind} {idx} variable {var} must have finite binary upper bound <= 1");
    }
}

fn append_logical_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &LogicalConstraint,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("logical_{idx}"));
    let target = constraint.target_var;
    match constraint.kind {
        LogicalConstraintKind::And => {
            for (pos, &var) in constraint.arg_vars.iter().enumerate() {
                let mut row = vec![0.0; out.c.len()];
                row[target] = 1.0;
                row[var] = -1.0;
                append_compiled_row(out, row, 0.0, format!("{name}_target_le_arg_{pos}"));
            }
            let mut row = vec![0.0; out.c.len()];
            for &var in &constraint.arg_vars {
                row[var] = 1.0;
            }
            row[target] = -1.0;
            append_compiled_row(
                out,
                row,
                constraint.arg_vars.len() as f64 - 1.0,
                format!("{name}_target_ge_all_args"),
            );
        }
        LogicalConstraintKind::Or => {
            for (pos, &var) in constraint.arg_vars.iter().enumerate() {
                let mut row = vec![0.0; out.c.len()];
                row[var] = 1.0;
                row[target] = -1.0;
                append_compiled_row(out, row, 0.0, format!("{name}_target_ge_arg_{pos}"));
            }
            let mut row = vec![0.0; out.c.len()];
            row[target] = 1.0;
            for &var in &constraint.arg_vars {
                row[var] = -1.0;
            }
            append_compiled_row(out, row, 0.0, format!("{name}_target_le_any_arg"));
        }
    }
}

fn validate_l1_norm_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &L1NormConstraint,
    idx: usize,
) {
    let n = source_lb.len();
    if n > base.c.len() {
        panic!(
            "{MODEL}: l1_norm {idx} source lower-bound length {n} exceeds compiled variable count {}",
            base.c.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: l1_norm {idx} source upper-bound length {} does not match source variable count {n}",
                ub.len()
            );
        }
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: l1_norm {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    if constraint.arg_vars.is_empty() {
        panic!("{MODEL}: l1_norm {idx} needs at least one argument");
    }
    if !source_lb[constraint.target_var].is_finite() {
        panic!("{MODEL}: l1_norm {idx} target lower bound must be finite");
    }

    let mut seen = HashSet::new();
    for (pos, &var) in constraint.arg_vars.iter().enumerate() {
        if var >= n {
            panic!("{MODEL}: l1_norm {idx} argument variable {var} out of range {n}");
        }
        if var == constraint.target_var {
            panic!("{MODEL}: l1_norm {idx} target_var must be distinct from argument variables");
        }
        if !seen.insert(var) {
            panic!("{MODEL}: l1_norm {idx} duplicate argument variable {var}");
        }
        let lower = source_lb[var];
        let upper = source_ub
            .and_then(|ub| ub.get(var))
            .copied()
            .unwrap_or(f64::INFINITY);
        l1_abs_helper_upper(lower, upper, idx, pos);
    }
}

fn l1_abs_helper_upper(lower: f64, upper: f64, idx: usize, pos: usize) -> f64 {
    if !lower.is_finite() {
        panic!("{MODEL}: l1_norm {idx} argument {pos} lower bound must be finite");
    }
    if upper.is_finite() && upper + 1e-9 < lower {
        panic!(
            "{MODEL}: l1_norm {idx} argument {pos} lower bound {lower} exceeds upper bound {upper}"
        );
    }
    if lower >= -1e-12 {
        return upper.max(0.0);
    }
    if upper <= 1e-12 {
        return (-lower).max(0.0);
    }
    if !upper.is_finite() {
        panic!("{MODEL}: l1_norm {idx} mixed-sign argument {pos} needs a finite upper bound");
    }
    (-lower).max(upper).max(0.0)
}

fn source_bounds_for_compiled_model(
    out: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let mut compiled_lb = vec![0.0; out.c.len()];
    for (j, &lower) in source_lb.iter().enumerate() {
        compiled_lb[j] = lower;
    }

    let mut compiled_ub = out
        .ub
        .as_ref()
        .cloned()
        .unwrap_or_else(|| vec![f64::INFINITY; out.c.len()]);
    if compiled_ub.len() < out.c.len() {
        compiled_ub.resize(out.c.len(), f64::INFINITY);
    }
    if let Some(source_ub) = source_ub {
        for (j, &upper) in source_ub.iter().enumerate() {
            compiled_ub[j] = upper;
        }
    }

    (compiled_lb, compiled_ub)
}

fn sync_helper_bounds(out: &IPMIPProblem, compiled_lb: &mut Vec<f64>, compiled_ub: &mut Vec<f64>) {
    while compiled_lb.len() < out.c.len() {
        let idx = compiled_lb.len();
        compiled_lb.push(0.0);
        compiled_ub.push(
            out.ub
                .as_ref()
                .and_then(|ub| ub.get(idx))
                .copied()
                .unwrap_or(f64::INFINITY),
        );
    }
}

fn append_l1_norm_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &L1NormConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("l1_norm_{idx}"));
    let target = constraint.target_var;
    let (mut compiled_lb, mut compiled_ub) =
        source_bounds_for_compiled_model(out, source_lb, source_ub);
    let mut helper_vars = Vec::with_capacity(constraint.arg_vars.len());

    for (pos, &arg) in constraint.arg_vars.iter().enumerate() {
        sync_helper_bounds(out, &mut compiled_lb, &mut compiled_ub);
        let helper_upper = l1_abs_helper_upper(compiled_lb[arg], compiled_ub[arg], idx, pos);
        let helper = append_continuous_helper_var(out, format!("{name}_abs_{pos}"), helper_upper);
        compiled_lb.push(0.0);
        compiled_ub.push(helper_upper);
        let abs_constraint = AbsoluteValueConstraint {
            arg_var: arg,
            target_var: helper,
            name: Some(format!("{name}_abs_{pos}")),
        };
        validate_abs_constraint(out, &compiled_lb, Some(&compiled_ub), &abs_constraint, idx);
        append_abs_constraint_rows(out, &abs_constraint, &compiled_lb, Some(&compiled_ub), idx);
        sync_helper_bounds(out, &mut compiled_lb, &mut compiled_ub);
        helper_vars.push(helper);
    }

    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    for helper in helper_vars {
        row[helper] = -1.0;
    }
    append_compiled_equality(out, row, -source_lb[target], format!("{name}_target_sum"));
}

fn validate_linf_norm_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &LInfNormConstraint,
    idx: usize,
) {
    let n = source_lb.len();
    if n > base.c.len() {
        panic!(
            "{MODEL}: linf_norm {idx} source lower-bound length {n} exceeds compiled variable count {}",
            base.c.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: linf_norm {idx} source upper-bound length {} does not match source variable count {n}",
                ub.len()
            );
        }
    }
    if constraint.target_var >= n {
        panic!(
            "{MODEL}: linf_norm {idx} target_var {} out of range {n}",
            constraint.target_var
        );
    }
    if constraint.arg_vars.is_empty() {
        panic!("{MODEL}: linf_norm {idx} needs at least one argument");
    }
    if !source_lb[constraint.target_var].is_finite() {
        panic!("{MODEL}: linf_norm {idx} target lower bound must be finite");
    }

    let mut seen = HashSet::new();
    for (pos, &var) in constraint.arg_vars.iter().enumerate() {
        if var >= n {
            panic!("{MODEL}: linf_norm {idx} argument variable {var} out of range {n}");
        }
        if var == constraint.target_var {
            panic!("{MODEL}: linf_norm {idx} target_var must be distinct from argument variables");
        }
        if !seen.insert(var) {
            panic!("{MODEL}: linf_norm {idx} duplicate argument variable {var}");
        }
        let lower = source_lb[var];
        let upper = source_ub
            .and_then(|ub| ub.get(var))
            .copied()
            .unwrap_or(f64::INFINITY);
        linf_abs_helper_upper(lower, upper, idx, pos);
    }
}

fn linf_abs_helper_upper(lower: f64, upper: f64, idx: usize, pos: usize) -> f64 {
    if !lower.is_finite() {
        panic!("{MODEL}: linf_norm {idx} argument {pos} lower bound must be finite");
    }
    if upper.is_finite() && upper + 1e-9 < lower {
        panic!(
            "{MODEL}: linf_norm {idx} argument {pos} lower bound {lower} exceeds upper bound {upper}"
        );
    }
    if lower >= -1e-12 {
        return upper.max(0.0);
    }
    if upper <= 1e-12 {
        return (-lower).max(0.0);
    }
    if !upper.is_finite() {
        panic!("{MODEL}: linf_norm {idx} mixed-sign argument {pos} needs a finite upper bound");
    }
    (-lower).max(upper).max(0.0)
}

fn append_linf_norm_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &LInfNormConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("linf_norm_{idx}"));
    let (mut compiled_lb, mut compiled_ub) =
        source_bounds_for_compiled_model(out, source_lb, source_ub);
    let mut helper_vars = Vec::with_capacity(constraint.arg_vars.len());

    for (pos, &arg) in constraint.arg_vars.iter().enumerate() {
        sync_helper_bounds(out, &mut compiled_lb, &mut compiled_ub);
        let helper_upper = linf_abs_helper_upper(compiled_lb[arg], compiled_ub[arg], idx, pos);
        let helper = append_continuous_helper_var(out, format!("{name}_abs_{pos}"), helper_upper);
        compiled_lb.push(0.0);
        compiled_ub.push(helper_upper);
        let abs_constraint = AbsoluteValueConstraint {
            arg_var: arg,
            target_var: helper,
            name: Some(format!("{name}_abs_{pos}")),
        };
        validate_abs_constraint(out, &compiled_lb, Some(&compiled_ub), &abs_constraint, idx);
        append_abs_constraint_rows(out, &abs_constraint, &compiled_lb, Some(&compiled_ub), idx);
        sync_helper_bounds(out, &mut compiled_lb, &mut compiled_ub);
        helper_vars.push(helper);
    }

    let max_constraint = MaximumConstraint {
        target_var: constraint.target_var,
        arg_vars: helper_vars,
        constant: None,
        name: Some(format!("{name}_max_abs")),
    };
    validate_maximum_constraint(out, &compiled_lb, Some(&compiled_ub), &max_constraint, idx);
    append_maximum_constraint_rows(out, &max_constraint, &compiled_lb, Some(&compiled_ub), idx);
}

fn validate_product_constraint(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    constraint: &ProductConstraint,
    idx: usize,
) {
    let n = source_lb.len();
    if n > base.c.len() {
        panic!(
            "{MODEL}: product {idx} source lower-bound length {n} exceeds compiled variable count {}",
            base.c.len()
        );
    }
    if let Some(ub) = source_ub {
        if ub.len() != n {
            panic!(
                "{MODEL}: product {idx} source upper-bound length {} does not match source variable count {n}",
                ub.len()
            );
        }
    }
    for (role, var) in [
        ("target_var", constraint.target_var),
        ("x_var", constraint.x_var),
        ("y_var", constraint.y_var),
    ] {
        if var >= n {
            panic!("{MODEL}: product {idx} {role} {var} out of range {n}");
        }
    }
    if constraint.target_var == constraint.x_var || constraint.target_var == constraint.y_var {
        panic!("{MODEL}: product {idx} target_var must be distinct from factor variables");
    }
    if constraint.x_var == constraint.y_var {
        panic!("{MODEL}: product {idx} x_var and y_var must be distinct");
    }
    if !source_lb[constraint.target_var].is_finite() {
        panic!("{MODEL}: product {idx} target lower bound must be finite");
    }

    let x_binary = product_var_is_binary(base, source_lb, source_ub, constraint.x_var);
    let y_binary = product_var_is_binary(base, source_lb, source_ub, constraint.y_var);
    if !x_binary && !y_binary {
        panic!(
            "{MODEL}: product {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
        );
    }
    if !x_binary {
        validate_product_continuous_factor(source_lb, source_ub, constraint.x_var, idx);
    }
    if !y_binary {
        validate_product_continuous_factor(source_lb, source_ub, constraint.y_var, idx);
    }
}

fn product_source_upper(source_ub: Option<&[f64]>, var: usize) -> f64 {
    source_ub
        .and_then(|ub| ub.get(var))
        .copied()
        .unwrap_or(f64::INFINITY)
}

fn product_var_is_binary(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    var: usize,
) -> bool {
    let upper = product_source_upper(source_ub, var);
    source_lb[var].abs() <= 1e-12
        && base.integer_vars[var]
        && upper.is_finite()
        && upper <= 1.0 + 1e-9
}

fn validate_product_continuous_factor(
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    var: usize,
    idx: usize,
) {
    let lower = source_lb[var];
    let upper = product_source_upper(source_ub, var);
    if !lower.is_finite() {
        panic!("{MODEL}: product {idx} continuous factor {var} lower bound must be finite");
    }
    if !upper.is_finite() {
        panic!("{MODEL}: product {idx} continuous factor {var} needs a finite upper bound");
    }
    if upper + 1e-9 < lower {
        panic!(
            "{MODEL}: product {idx} continuous factor {var} lower bound {lower} exceeds upper bound {upper}"
        );
    }
}

fn quadratic_product_bounds(
    base: &IPMIPProblem,
    source_lb: &[f64],
    source_ub: &[f64],
    term: &QuadraticObjectiveTerm,
    idx: usize,
) -> (f64, f64) {
    let x_binary = product_var_is_binary(base, source_lb, Some(source_ub), term.x_var);
    let y_binary = product_var_is_binary(base, source_lb, Some(source_ub), term.y_var);
    if !x_binary && !y_binary {
        panic!(
            "{MODEL}: quadratic objective term {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
        );
    }
    if x_binary && y_binary {
        return (0.0, 1.0);
    }

    let continuous = if x_binary { term.y_var } else { term.x_var };
    let lower = source_lb[continuous];
    let upper = source_ub[continuous];
    if !lower.is_finite() {
        panic!(
            "{MODEL}: quadratic objective term {idx} continuous factor {continuous} lower bound must be finite"
        );
    }
    if !upper.is_finite() {
        panic!(
            "{MODEL}: quadratic objective term {idx} continuous factor {continuous} needs a finite upper bound"
        );
    }
    if upper + 1e-9 < lower {
        panic!(
            "{MODEL}: quadratic objective term {idx} continuous factor {continuous} lower bound {lower} exceeds upper bound {upper}"
        );
    }
    (0.0_f64.min(lower), 0.0_f64.max(upper))
}

fn append_product_constraint_rows(
    out: &mut IPMIPProblem,
    constraint: &ProductConstraint,
    source_lb: &[f64],
    source_ub: Option<&[f64]>,
    idx: usize,
) {
    let name = constraint
        .name
        .clone()
        .unwrap_or_else(|| format!("product_{idx}"));
    let x_binary = product_var_is_binary(out, source_lb, source_ub, constraint.x_var);
    let y_binary = product_var_is_binary(out, source_lb, source_ub, constraint.y_var);

    if x_binary && y_binary {
        append_binary_binary_product_rows(
            out,
            constraint.target_var,
            constraint.x_var,
            constraint.y_var,
            source_lb[constraint.target_var],
            &name,
        );
    } else if x_binary {
        append_binary_continuous_product_rows(
            out,
            constraint.target_var,
            constraint.x_var,
            constraint.y_var,
            source_lb[constraint.y_var],
            product_source_upper(source_ub, constraint.y_var),
            source_lb[constraint.target_var],
            &name,
        );
    } else {
        append_binary_continuous_product_rows(
            out,
            constraint.target_var,
            constraint.y_var,
            constraint.x_var,
            source_lb[constraint.x_var],
            product_source_upper(source_ub, constraint.x_var),
            source_lb[constraint.target_var],
            &name,
        );
    }
}

fn append_binary_binary_product_rows(
    out: &mut IPMIPProblem,
    target: usize,
    x: usize,
    y: usize,
    target_lb: f64,
    name: &str,
) {
    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[x] = -1.0;
    append_compiled_row(out, row, -target_lb, format!("{name}_target_le_x"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[y] = -1.0;
    append_compiled_row(out, row, -target_lb, format!("{name}_target_le_y"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = -1.0;
    append_compiled_row(out, row, target_lb, format!("{name}_target_ge_zero"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = -1.0;
    row[x] = 1.0;
    row[y] = 1.0;
    append_compiled_row(out, row, 1.0 + target_lb, format!("{name}_target_ge_xy"));
}

fn append_binary_continuous_product_rows(
    out: &mut IPMIPProblem,
    target: usize,
    binary: usize,
    continuous: usize,
    continuous_lb: f64,
    continuous_ub: f64,
    target_lb: f64,
    name: &str,
) {
    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[binary] = -continuous_ub;
    append_compiled_row(out, row, -target_lb, format!("{name}_target_le_ub_binary"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = -1.0;
    row[binary] = continuous_lb;
    append_compiled_row(out, row, target_lb, format!("{name}_target_ge_lb_binary"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = 1.0;
    row[continuous] = -1.0;
    row[binary] = -continuous_lb;
    append_compiled_row(out, row, -target_lb, format!("{name}_target_le_active"));

    let mut row = vec![0.0; out.c.len()];
    row[target] = -1.0;
    row[continuous] = 1.0;
    row[binary] = continuous_ub;
    append_compiled_row(
        out,
        row,
        continuous_ub - continuous_lb + target_lb,
        format!("{name}_target_ge_active"),
    );
}

fn append_indicator_le_row(
    base: &IPMIPProblem,
    out: &mut IPMIPProblem,
    indicator: &IndicatorConstraint,
    mut row: Vec<f64>,
    rhs: f64,
    suffix: Option<&str>,
) {
    let m = indicator_big_m(base, &row, rhs, indicator.name.as_deref());
    let compiled_rhs = if indicator.active_value {
        row[indicator.binary_var] += m;
        rhs + m
    } else {
        row[indicator.binary_var] -= m;
        rhs
    };
    out.a.push(row);
    out.b.push(compiled_rhs);
    if let Some(names) = &mut out.con_names {
        let mut name = indicator
            .name
            .clone()
            .unwrap_or_else(|| format!("indicator_{}", out.a.len() - 1));
        if let Some(suffix) = suffix {
            name.push('_');
            name.push_str(suffix);
        }
        names.push(name);
    }
}

fn indicator_big_m(base: &IPMIPProblem, row: &[f64], rhs: f64, name: Option<&str>) -> f64 {
    let ub = base.ub.as_ref();
    let mut max_lhs = 0.0;
    for (j, &coef) in row.iter().enumerate() {
        if coef > 0.0 {
            let Some(upper) = ub.and_then(|u| u.get(j)).copied() else {
                panic!(
                    "{MODEL}: indicator {} needs a finite upper bound for positive coefficient variable {}",
                    name.unwrap_or("<unnamed>"),
                    j
                );
            };
            if !upper.is_finite() {
                panic!(
                    "{MODEL}: indicator {} needs a finite upper bound for positive coefficient variable {}",
                    name.unwrap_or("<unnamed>"),
                    j
                );
            }
            max_lhs += coef * upper;
        }
    }
    0.0_f64.max(max_lhs - rhs)
}

fn ensure_compilation_metadata(p: &mut IPMIPProblem) {
    let n = p.c.len();
    if p.ub.is_none() {
        p.ub = Some(vec![f64::INFINITY; n]);
    }
    if p.var_names.is_none() {
        p.var_names = Some((0..n).map(|j| format!("x{j}")).collect());
    }
    if p.con_names.is_none() {
        p.con_names = Some((0..p.a.len()).map(|i| format!("c{i}")).collect());
    }
    p.variable_nodes = None;
    p.constraint_nodes = None;
}

fn ordered_sos_vars(base: &IPMIPProblem, set: &SpecialOrderedSet, idx: usize) -> Vec<usize> {
    if set.vars.is_empty() {
        panic!("{MODEL}: sos {idx} has no variables");
    }
    if let Some(weights) = &set.weights {
        if weights.len() != set.vars.len() {
            panic!(
                "{MODEL}: sos {idx} weight length {} does not match variable count {}",
                weights.len(),
                set.vars.len()
            );
        }
        let mut pairs: Vec<(f64, usize)> = weights
            .iter()
            .copied()
            .zip(set.vars.iter().copied())
            .collect();
        for (w, _) in &pairs {
            if !w.is_finite() {
                panic!("{MODEL}: sos {idx} weights must be finite");
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for w in pairs.windows(2) {
            if (w[0].0 - w[1].0).abs() <= 1e-12 {
                panic!("{MODEL}: sos {idx} weights must be unique");
            }
        }
        validate_sos_vars(base, idx, pairs.iter().map(|(_, v)| *v));
        pairs.into_iter().map(|(_, v)| v).collect()
    } else {
        validate_sos_vars(base, idx, set.vars.iter().copied());
        set.vars.clone()
    }
}

fn validate_sos_vars<I>(base: &IPMIPProblem, idx: usize, vars: I)
where
    I: IntoIterator<Item = usize>,
{
    let n = base.c.len();
    let mut seen = HashSet::new();
    for v in vars {
        if v >= n {
            panic!("{MODEL}: sos {idx} variable {v} out of range {n}");
        }
        if !seen.insert(v) {
            panic!("{MODEL}: sos {idx} contains duplicate variable {v}");
        }
        finite_upper_bound(base, v, "sos", idx);
    }
}

fn append_sos1_rows(
    base: &IPMIPProblem,
    out: &mut IPMIPProblem,
    set: &SpecialOrderedSet,
    idx: usize,
    ordered: &[usize],
) {
    let name = set.name.as_deref().unwrap_or("sos1");
    let mut selectors = Vec::with_capacity(ordered.len());
    for (pos, &var) in ordered.iter().enumerate() {
        let y = append_binary_helper_var(out, format!("{name}_sel_{pos}"));
        selectors.push(y);
        let ub = finite_upper_bound(base, var, "sos", idx);
        let mut row = vec![0.0; out.c.len()];
        row[var] = 1.0;
        row[y] = -ub;
        append_compiled_row(out, row, 0.0, format!("{name}_link_{pos}"));
    }
    let mut row = vec![0.0; out.c.len()];
    for y in selectors {
        row[y] = 1.0;
    }
    append_compiled_row(out, row, 1.0, format!("{name}_at_most_one"));
}

fn append_sos2_rows(
    base: &IPMIPProblem,
    out: &mut IPMIPProblem,
    set: &SpecialOrderedSet,
    idx: usize,
    ordered: &[usize],
) {
    if ordered.len() <= 2 {
        return;
    }
    let name = set.name.as_deref().unwrap_or("sos2");
    let mut segments = Vec::with_capacity(ordered.len() - 1);
    for pos in 0..(ordered.len() - 1) {
        segments.push(append_binary_helper_var(out, format!("{name}_seg_{pos}")));
    }
    let mut sum_row = vec![0.0; out.c.len()];
    for &segment in &segments {
        sum_row[segment] = 1.0;
    }
    append_compiled_row(out, sum_row, 1.0, format!("{name}_one_segment"));

    for (pos, &var) in ordered.iter().enumerate() {
        let ub = finite_upper_bound(base, var, "sos", idx);
        let mut row = vec![0.0; out.c.len()];
        row[var] = 1.0;
        if pos > 0 {
            row[segments[pos - 1]] -= ub;
        }
        if pos + 1 < ordered.len() {
            row[segments[pos]] -= ub;
        }
        append_compiled_row(out, row, 0.0, format!("{name}_link_{pos}"));
    }
}

fn append_binary_helper_var(out: &mut IPMIPProblem, name: String) -> usize {
    for row in &mut out.a {
        row.push(0.0);
    }
    let idx = out.c.len();
    out.c.push(0.0);
    out.integer_vars.push(true);
    out.ub
        .as_mut()
        .expect("compiler metadata initialized")
        .push(1.0);
    out.var_names
        .as_mut()
        .expect("compiler metadata initialized")
        .push(name);
    idx
}

fn append_continuous_helper_var(out: &mut IPMIPProblem, name: String, upper: f64) -> usize {
    for row in &mut out.a {
        row.push(0.0);
    }
    let idx = out.c.len();
    out.c.push(0.0);
    out.integer_vars.push(false);
    out.ub
        .as_mut()
        .expect("compiler metadata initialized")
        .push(upper);
    out.var_names
        .as_mut()
        .expect("compiler metadata initialized")
        .push(name);
    idx
}

fn append_compiled_row(out: &mut IPMIPProblem, row: Vec<f64>, rhs: f64, name: String) {
    out.a.push(row);
    out.b.push(rhs);
    out.con_names
        .as_mut()
        .expect("compiler metadata initialized")
        .push(name);
}

fn append_compiled_equality(out: &mut IPMIPProblem, row: Vec<f64>, rhs: f64, name: String) {
    append_compiled_row(out, row.clone(), rhs, format!("{name}_le"));
    append_compiled_row(
        out,
        row.into_iter().map(|v| -v).collect(),
        -rhs,
        format!("{name}_ge"),
    );
}

fn finite_upper_bound(base: &IPMIPProblem, var: usize, kind: &str, idx: usize) -> f64 {
    let upper = base
        .ub
        .as_ref()
        .and_then(|ub| ub.get(var))
        .copied()
        .unwrap_or(f64::INFINITY);
    if !upper.is_finite() {
        panic!("{MODEL}: {kind} {idx} variable {var} needs a finite upper bound");
    }
    if upper < 0.0 {
        panic!("{MODEL}: {kind} {idx} variable {var} has a negative upper bound");
    }
    upper
}

fn validate_semi_variable(base: &IPMIPProblem, semi: &SemiVariable, idx: usize) {
    let n = base.c.len();
    if semi.var >= n {
        panic!(
            "{MODEL}: semi variable {idx} index {} out of range {n}",
            semi.var
        );
    }
    if !semi.lower.is_finite() || semi.lower <= 0.0 {
        panic!("{MODEL}: semi variable {idx} lower bound must be finite and positive");
    }
    let upper = finite_upper_bound(base, semi.var, "semi", idx);
    if upper + 1e-9 < semi.lower {
        panic!(
            "{MODEL}: semi variable {idx} lower bound {} exceeds upper bound {}",
            semi.lower, upper
        );
    }
    if semi.kind == SemiVariableKind::SemiContinuous && base.integer_vars[semi.var] {
        panic!(
            "{MODEL}: semi-continuous variable {} must not already be marked integer",
            semi.var
        );
    }
}

fn append_semi_variable_rows(
    base: &IPMIPProblem,
    out: &mut IPMIPProblem,
    semi: &SemiVariable,
    idx: usize,
) {
    let name = semi
        .name
        .clone()
        .unwrap_or_else(|| format!("semi_{}", semi.var));
    let upper = finite_upper_bound(base, semi.var, "semi", idx);
    out.integer_vars[semi.var] = semi.kind == SemiVariableKind::SemiInteger;
    let y = append_binary_helper_var(out, format!("{name}_active"));

    let mut upper_row = vec![0.0; out.c.len()];
    upper_row[semi.var] = 1.0;
    upper_row[y] = -upper;
    append_compiled_row(out, upper_row, 0.0, format!("{name}_upper_link"));

    let mut lower_row = vec![0.0; out.c.len()];
    lower_row[semi.var] = -1.0;
    lower_row[y] = semi.lower;
    append_compiled_row(out, lower_row, 0.0, format!("{name}_lower_link"));
}

fn validate_pwl_constraint(base: &IPMIPProblem, pwl: &PiecewiseLinearConstraint, idx: usize) {
    let n = base.c.len();
    if pwl.x_var >= n {
        panic!("{MODEL}: pwl {idx} x_var {} out of range {n}", pwl.x_var);
    }
    if pwl.y_var >= n {
        panic!("{MODEL}: pwl {idx} y_var {} out of range {n}", pwl.y_var);
    }
    if pwl.x_var == pwl.y_var {
        panic!("{MODEL}: pwl {idx} x_var and y_var must be distinct");
    }
    if base.integer_vars[pwl.x_var] || base.integer_vars[pwl.y_var] {
        panic!("{MODEL}: pwl {idx} x_var and y_var must be continuous variables");
    }
    if pwl.points.len() < 2 {
        panic!("{MODEL}: pwl {idx} needs at least two breakpoints");
    }
    for (pos, point) in pwl.points.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() {
            panic!("{MODEL}: pwl {idx} breakpoint {pos} must be finite");
        }
        if point.x < -1e-12 || point.y < -1e-12 {
            panic!("{MODEL}: pwl {idx} breakpoint {pos} must be non-negative for this MIP backend");
        }
        if pos > 0 && point.x <= pwl.points[pos - 1].x + 1e-12 {
            panic!("{MODEL}: pwl {idx} breakpoint x values must be strictly increasing");
        }
    }
}

fn append_pwl_constraint_rows(out: &mut IPMIPProblem, pwl: &PiecewiseLinearConstraint, idx: usize) {
    let name = pwl.name.clone().unwrap_or_else(|| format!("pwl_{idx}"));
    let lambdas: Vec<usize> = pwl
        .points
        .iter()
        .enumerate()
        .map(|(pos, _)| append_continuous_helper_var(out, format!("{name}_lambda_{pos}"), 1.0))
        .collect();

    let mut sum_row = vec![0.0; out.c.len()];
    for &lambda in &lambdas {
        sum_row[lambda] = 1.0;
    }
    append_compiled_equality(out, sum_row, 1.0, format!("{name}_lambda_sum"));

    let mut x_row = vec![0.0; out.c.len()];
    x_row[pwl.x_var] = 1.0;
    for (&lambda, point) in lambdas.iter().zip(&pwl.points) {
        x_row[lambda] -= point.x;
    }
    append_compiled_equality(out, x_row, 0.0, format!("{name}_x_interp"));

    let mut y_row = vec![0.0; out.c.len()];
    y_row[pwl.y_var] = 1.0;
    for (&lambda, point) in lambdas.iter().zip(&pwl.points) {
        y_row[lambda] -= point.y;
    }
    append_compiled_equality(out, y_row, 0.0, format!("{name}_y_interp"));

    let set = SpecialOrderedSet {
        kind: SpecialOrderedSetKind::Sos2,
        vars: lambdas.clone(),
        weights: Some(pwl.points.iter().map(|p| p.x).collect()),
        name: Some(format!("{name}_sos2")),
    };
    let sos_base = out.clone();
    append_sos2_rows(&sos_base, out, &set, idx, &lambdas);
}

fn validate_multi_objective_problem(problem: &MultiObjectiveIPMIPProblem) {
    validate_ipmip_problem(&problem.base);
    if problem.objectives.is_empty() {
        panic!("{MODEL}: multi-objective problem needs at least one objective");
    }
    let n = problem.base.c.len();
    for (idx, objective) in problem.objectives.iter().enumerate() {
        if objective.c.len() != n {
            panic!(
                "{MODEL}: multi-objective {idx} coefficient length {} does not match variable count {n}",
                objective.c.len()
            );
        }
        for (j, &coef) in objective.c.iter().enumerate() {
            if !coef.is_finite() {
                panic!("{MODEL}: multi-objective {idx} coefficient {j} must be finite");
            }
        }
    }
}

// -----------------------------------------------------------------------------
// LP backend selection
// -----------------------------------------------------------------------------

/// The normalised LP-solve result shape used internally (the TS `LPSolution`
/// subset consumed by this module).
struct NodeLPResult {
    status: LPStatus,
    x: Vec<f64>,
    objective: f64,
    solver: String,
    elapsed_ms: f64,
    iters: Option<usize>,
    message: Option<String>,
}

fn solve_node_relaxation(
    p: &IPMIPProblem,
    node: &IpNode,
    algorithm: ConcreteLpRelaxationAlgorithm,
    lp_max_iters: usize,
) -> NodeLPResult {
    use ConcreteLpRelaxationAlgorithm::*;
    if algorithm == IncrementalPrimalDual {
        return solve_incremental_relaxation(p, node, lp_max_iters);
    }
    let lp = node_to_lp_problem(p, node);
    match algorithm {
        InternalSimplex => {
            let s = solve_lp_internal(
                &lp,
                &InternalSimplexOptions {
                    max_iter: Some(lp_max_iters),
                    tol: None,
                    basis_start: None,
                },
            );
            NodeLPResult {
                status: s.status,
                x: s.x,
                objective: s.objective,
                solver: s.solver,
                elapsed_ms: s.elapsed_ms,
                iters: s.iters,
                message: s.message,
            }
        }
        InternalInteriorPoint => {
            let s = solve_lp_internal_ipm(
                &lp,
                &InternalInteriorPointOptions {
                    max_iter: Some(lp_max_iters),
                    tol: None,
                    step_fraction: None,
                    regularization: None,
                },
            );
            NodeLPResult {
                status: s.status,
                x: s.x,
                objective: s.objective,
                solver: s.solver,
                elapsed_ms: s.elapsed_ms,
                iters: s.iters,
                message: s.message,
            }
        }
        DesSimplexDantzig => {
            let s = solve_lp_via_des(
                &lp,
                &DESSimplexOptions {
                    max_iter: Some(lp_max_iters),
                    pivot_rule: Some(PivotRule::Dantzig),
                    tol: None,
                },
            );
            NodeLPResult {
                status: s.status,
                x: s.x,
                objective: s.objective,
                solver: s.solver,
                elapsed_ms: s.elapsed_ms,
                iters: s.iters,
                message: s.message,
            }
        }
        DesSimplexBland => {
            let s = solve_lp_via_des(
                &lp,
                &DESSimplexOptions {
                    max_iter: Some(lp_max_iters),
                    pivot_rule: Some(PivotRule::Bland),
                    tol: None,
                },
            );
            NodeLPResult {
                status: s.status,
                x: s.x,
                objective: s.objective,
                solver: s.solver,
                elapsed_ms: s.elapsed_ms,
                iters: s.iters,
                message: s.message,
            }
        }
        ExternalHighs | ExternalHighsDs | ExternalHighsIpm => {
            let method = match algorithm {
                ExternalHighsDs => "highs-ds",
                ExternalHighsIpm => "highs-ipm",
                _ => "highs",
            };
            let s = solve_lp_external(
                &lp,
                &ExternalSolverOptions {
                    method: Some(method.to_string()),
                    ..Default::default()
                },
            );
            NodeLPResult {
                status: s.status,
                x: s.x,
                objective: s.objective,
                solver: s.solver,
                elapsed_ms: s.elapsed_ms,
                iters: s.iters,
                message: s.message,
            }
        }
        IncrementalPrimalDual => {
            unreachable!("IncrementalPrimalDual is dispatched by the early return above")
        }
    }
}

/// Build the heuristic technique plan for an IP/MIP problem.
pub fn build_ipmip_solver_technique_plan(
    p: &IPMIPProblem,
    requested_lp_algorithm: LpRelaxationAlgorithm,
    allow_external_solvers: bool,
) -> IPMIPSolverTechniquePlan {
    use ConcreteLpRelaxationAlgorithm::*;
    validate_ipmip_problem(p);
    let features = analyze_ipmip_problem(p);
    let mut rationale: Vec<String> = Vec::new();
    let negative_root_rhs = has_negative_root_rhs(p);
    let root_lp_algorithm: ConcreteLpRelaxationAlgorithm;

    match requested_lp_algorithm {
        LpRelaxationAlgorithm::Concrete(req) => {
            if is_external_lp_algorithm(req) && !allow_external_solvers {
                panic!("{MODEL}: external LP backend \"{}\" requested, but allowExternalSolvers is false", req.as_str());
            }
            root_lp_algorithm = req;
            rationale.push(format!(
                "fixed LP relaxation backend requested: {}",
                req.as_str()
            ));
            if req == IncrementalPrimalDual && negative_root_rhs {
                rationale.push("warning: root has negative RHS rows; incremental LP requires a non-negative initial RHS".to_string());
            }
        }
        LpRelaxationAlgorithm::Auto => {
            if negative_root_rhs {
                root_lp_algorithm = if allow_external_solvers
                    && features.variable_count * features.constraint_count >= 2500
                {
                    ExternalHighs
                } else {
                    InternalSimplex
                };
                rationale.push(
                    if root_lp_algorithm == ExternalHighs {
                        "root relaxation has lower-bound rows with negative RHS, so auto uses an external Phase-1-capable LP backend"
                    } else {
                        "root relaxation has lower-bound rows with negative RHS, so auto uses the in-house Phase-1 simplex backend"
                    }
                    .to_string(),
                );
            } else if features.variable_count * features.constraint_count >= 2500 {
                root_lp_algorithm = if allow_external_solvers {
                    if features.density > 0.35
                        || features.variable_count > features.constraint_count * 3
                    {
                        ExternalHighsIpm
                    } else if features.constraint_count > features.variable_count * 2 {
                        ExternalHighsDs
                    } else {
                        ExternalHighs
                    }
                } else {
                    IncrementalPrimalDual
                };
                rationale.push(if allow_external_solvers {
                    format!("large relaxation ({} vars x {} rows) is an external-solver candidate", features.variable_count, features.constraint_count)
                } else {
                    format!(
                        "large relaxation ({} vars x {} rows) stays in-house because external solvers are disabled",
                        features.variable_count, features.constraint_count
                    )
                });
            } else {
                root_lp_algorithm = IncrementalPrimalDual;
                rationale.push("small/medium branch-cut relaxation uses in-engine incremental primal-dual simplex".to_string());
            }
        }
    }

    if features.all_binary {
        rationale.push(
            "all integer variables are binary, so cover cuts and rounding/repair are active"
                .to_string(),
        );
    } else if features.continuous_count > 0 {
        rationale.push(
            "mixed integer/continuous model keeps continuous variables in the LP relaxation"
                .to_string(),
        );
    }

    let decomposition_candidate = features.constraint_variable_components > 1;
    let decomposition_reason = if decomposition_candidate {
        Some(format!(
            "constraint-variable graph has {} disconnected components",
            features.constraint_variable_components
        ))
    } else {
        None
    };
    if decomposition_candidate {
        rationale.push(format!(
            "{}; separable decomposition is structurally valid",
            decomposition_reason.as_ref().unwrap()
        ));
    }

    IPMIPSolverTechniquePlan {
        requested_lp_algorithm,
        root_lp_algorithm,
        external_solvers_allowed: allow_external_solvers,
        uses_external_solvers: is_external_lp_algorithm(root_lp_algorithm),
        external_candidate: allow_external_solvers && is_external_lp_algorithm(root_lp_algorithm),
        primal_dual_dynamic: root_lp_algorithm == IncrementalPrimalDual
            || requested_lp_algorithm == LpRelaxationAlgorithm::Auto,
        decomposition_candidate,
        decomposition_reason,
        rationale,
        features,
    }
}

fn select_lp_relaxation_algorithm(
    p: &IPMIPProblem,
    node: &IpNode,
    requested: LpRelaxationAlgorithm,
    plan: &IPMIPSolverTechniquePlan,
) -> ConcreteLpRelaxationAlgorithm {
    use ConcreteLpRelaxationAlgorithm::*;
    if let LpRelaxationAlgorithm::Concrete(c) = requested {
        return c;
    }
    if has_negative_root_rhs(p) {
        return plan.root_lp_algorithm;
    }
    if !plan.external_solvers_allowed {
        return plan.root_lp_algorithm;
    }
    if !node.constraints.is_empty() || node.cut_rounds > 0 || node.depth > 0 {
        return IncrementalPrimalDual;
    }
    let f = &plan.features;
    if f.variable_count * f.constraint_count >= 2500 {
        return plan.root_lp_algorithm;
    }
    if f.constraint_count > f.variable_count * 3 && f.constraint_count > 40 {
        return ExternalHighsDs;
    }
    if f.variable_count > f.constraint_count * 4 && f.variable_count > 80 {
        return ExternalHighsIpm;
    }
    if p.sense == Sense::Min && f.density > 0.5 && f.variable_count > 40 {
        return ExternalHighs;
    }
    plan.root_lp_algorithm
}

fn is_external_lp_algorithm(a: ConcreteLpRelaxationAlgorithm) -> bool {
    use ConcreteLpRelaxationAlgorithm::*;
    matches!(a, ExternalHighs | ExternalHighsDs | ExternalHighsIpm)
}

fn did_use_external_lp(usage: &HashMap<ConcreteLpRelaxationAlgorithm, u64>) -> bool {
    usage
        .iter()
        .any(|(&k, &v)| v > 0 && is_external_lp_algorithm(k))
}

struct BuildPerfArgs {
    elapsed_ms: f64,
    ticks: usize,
    nodes_explored: usize,
    lp_solves: usize,
    total_lp_iterations: usize,
    total_lp_solver_ms: f64,
    cuts_added: usize,
    candidates_tried: usize,
    tokens_created: u64,
}

fn build_ipmip_performance(o: BuildPerfArgs) -> IPMIPPerformanceStats {
    let seconds = (o.elapsed_ms / 1000.0).max(1e-9);
    let node_denom = o.nodes_explored.max(1) as f64;
    let lp_denom = o.lp_solves.max(1) as f64;
    IPMIPPerformanceStats {
        elapsed_ms: o.elapsed_ms,
        ticks: o.ticks,
        nodes_per_second: o.nodes_explored as f64 / seconds,
        lp_solves_per_second: o.lp_solves as f64 / seconds,
        ms_per_node: o.elapsed_ms / node_denom,
        total_lp_solver_ms: o.total_lp_solver_ms,
        avg_lp_solver_ms: o.total_lp_solver_ms / lp_denom,
        lp_solver_time_share: if o.elapsed_ms > 0.0 {
            o.total_lp_solver_ms / o.elapsed_ms
        } else {
            0.0
        },
        avg_lp_iterations_per_solve: o.total_lp_iterations as f64 / lp_denom,
        cuts_per_node: o.cuts_added as f64 / node_denom,
        candidates_per_node: o.candidates_tried as f64 / node_denom,
        tokens_created: o.tokens_created,
    }
}

fn has_negative_root_rhs(p: &IPMIPProblem) -> bool {
    p.b.iter().any(|&v| v < -1e-9)
}

/// Compute the structural features of an IP/MIP problem.
pub fn analyze_ipmip_problem(p: &IPMIPProblem) -> IPMIPProblemFeatures {
    let variable_count = p.c.len();
    let constraint_count = p.a.len();
    let integer_count = p.integer_vars.iter().filter(|&&b| b).count();
    let finite_upper_bounds =
        p.ub.as_ref()
            .map_or(0, |u| u.iter().filter(|v| v.is_finite()).count());
    let mut binary_count = 0;
    for j in 0..variable_count {
        let ubj =
            p.ub.as_ref()
                .and_then(|u| u.get(j))
                .copied()
                .unwrap_or(f64::INFINITY);
        if p.integer_vars[j] && ubj <= 1.0 + 1e-9 {
            binary_count += 1;
        }
    }
    let mut nonzeros = 0;
    for row in &p.a {
        for &a in row {
            if a.abs() > 1e-12 {
                nonzeros += 1;
            }
        }
    }
    IPMIPProblemFeatures {
        variable_count,
        constraint_count,
        integer_count,
        continuous_count: variable_count - integer_count,
        binary_count,
        finite_upper_bounds,
        nonzeros,
        density: nonzeros as f64 / (1.0_f64).max((variable_count * constraint_count) as f64),
        all_integer: integer_count == variable_count,
        all_binary: binary_count == variable_count,
        constraint_variable_components: count_constraint_variable_components(p),
    }
}

fn count_constraint_variable_components(p: &IPMIPProblem) -> usize {
    let n = p.c.len();
    let m = p.a.len();
    let total = n + m;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total];
    for i in 0..m {
        for j in 0..n {
            if p.a[i][j].abs() <= 1e-12 {
                continue;
            }
            let row_node = n + i;
            adj[j].push(row_node);
            adj[row_node].push(j);
        }
    }
    let mut seen = vec![false; total];
    let mut components = 0;
    for k in 0..total {
        if seen[k] || adj[k].is_empty() {
            continue;
        }
        components += 1;
        let mut stack = vec![k];
        seen[k] = true;
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if seen[v] {
                    continue;
                }
                seen[v] = true;
                stack.push(v);
            }
        }
    }
    components
}

fn solve_incremental_relaxation(
    p: &IPMIPProblem,
    node: &IpNode,
    lp_max_iters: usize,
) -> NodeLPResult {
    let t0 = Instant::now();
    let root = root_incremental_rows(p);
    let has_negative_rhs = root.b.iter().any(|&rhs| rhs < -1e-9)
        || node
            .constraints
            .iter()
            .any(|constraint| constraint.rhs < -1e-9);
    if has_negative_rhs {
        let lp_problem = node_to_lp_problem(p, node);
        let s = solve_lp_internal(
            &lp_problem,
            &InternalSimplexOptions {
                max_iter: Some(lp_max_iters),
                tol: None,
                basis_start: None,
            },
        );
        return NodeLPResult {
            status: s.status,
            x: s.x,
            objective: s.objective,
            solver: "incremental-primal-dual:phase1-simplex-fallback".to_string(),
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
            iters: s.iters,
            message: s.message,
        };
    }
    let mut lp = IncrementalLP::new(IncrementalLPInit {
        sense: to_inc_sense(p.sense),
        c: p.c.clone(),
        a: root.a,
        b: root.b,
        var_names: p.var_names.clone(),
        con_names: Some(root.names),
    });
    lp.set_pivot_rule(IncrementalPivotRule::Bland);
    for c in &node.constraints {
        lp.apply_add_constraint(&c.coefs, c.rhs, Some(c.name.clone()));
    }
    let trace = lp.solve_to_optimum(lp_max_iters);
    let status = match lp.status {
        SolverStatus::Optimal => LPStatus::Optimal,
        SolverStatus::Infeasible => LPStatus::Infeasible,
        SolverStatus::Unbounded => LPStatus::Unbounded,
        _ => LPStatus::IterLimit,
    };
    let iters = trace
        .iter()
        .filter(|e| e.mode == PivotMode::Primal || e.mode == PivotMode::Dual)
        .count();
    NodeLPResult {
        status,
        x: if status == LPStatus::Optimal {
            lp.get_x()
        } else {
            Vec::new()
        },
        objective: if status == LPStatus::Optimal {
            lp.get_z()
        } else {
            f64::NAN
        },
        solver: "incremental-primal-dual".to_string(),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        iters: Some(iters),
        message: None,
    }
}

fn node_to_lp_problem(p: &IPMIPProblem, node: &IpNode) -> LPProblem {
    let mut a: Vec<Vec<f64>> = p.a.iter().cloned().collect();
    let mut b = p.b.clone();
    for c in &node.constraints {
        a.push(c.coefs.clone());
        b.push(c.rhs);
    }
    LPProblem {
        sense: p.sense,
        c: p.c.clone(),
        a_ub: Some(a),
        b_ub: Some(b),
        lb: Some(vec![Some(0.0); p.c.len()]),
        ub: p.ub.as_ref().map(|u| {
            u.iter()
                .map(|&v| if v.is_finite() { Some(v) } else { None })
                .collect()
        }),
        var_names: p.var_names.clone(),
        con_names: p.con_names.clone(),
        ..Default::default()
    }
}

struct RootRows {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    names: Vec<String>,
}

fn root_incremental_rows(p: &IPMIPProblem) -> RootRows {
    let mut a: Vec<Vec<f64>> = p.a.iter().cloned().collect();
    let mut b = p.b.clone();
    let mut names: Vec<String> = match &p.con_names {
        Some(n) => n.clone(),
        None => (0..p.a.len()).map(|i| format!("c{i}")).collect(),
    };
    if let Some(ub) = &p.ub {
        for j in 0..p.c.len() {
            if !ub[j].is_finite() {
                continue;
            }
            let mut row = vec![0.0; p.c.len()];
            row[j] = 1.0;
            a.push(row);
            b.push(ub[j]);
            names.push(format!("ub_{}", var_name(p, j)));
        }
    }
    RootRows { a, b, names }
}

// -----------------------------------------------------------------------------
// Heuristics, cuts, and branching helpers
// -----------------------------------------------------------------------------

struct Candidate {
    x: Vec<f64>,
    source: String,
}

fn generate_integer_candidates(
    p: &IPMIPProblem,
    x_lp: &[f64],
    tol: f64,
    passes: usize,
) -> Vec<Candidate> {
    let mut seeds: Vec<Candidate> = Vec::new();
    for mode in ["round", "floor", "ceil"] {
        let mut x = x_lp.to_vec();
        for j in 0..x.len() {
            if !p.integer_vars[j] {
                continue;
            }
            x[j] = match mode {
                "round" => x[j].round(),
                "floor" => x[j].floor(),
                _ => x[j].ceil(),
            };
        }
        seeds.push(Candidate {
            x: clamp_bounds(p, &x),
            source: mode.to_string(),
        });
    }
    let frac: Vec<usize> = list_fractionals(x_lp, &p.integer_vars, tol)
        .into_iter()
        .take(4)
        .collect();
    for j in frac {
        for val in [x_lp[j].floor(), x_lp[j].ceil()] {
            let mut x = x_lp.to_vec();
            for k in 0..x.len() {
                if p.integer_vars[k] {
                    x[k] = x[k].round();
                }
            }
            x[j] = val;
            seeds.push(Candidate {
                x: clamp_bounds(p, &x),
                source: format!("one-flip-{}", var_name(p, j)),
            });
        }
    }

    let mut out: Vec<Candidate> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in seeds {
        let repaired = match repair_and_improve_candidate(p, &s.x, tol, passes) {
            Some(r) => r,
            None => continue,
        };
        let key = repaired
            .iter()
            .map(|v| format!("{v:.9}"))
            .collect::<Vec<_>>()
            .join(",");
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        out.push(Candidate {
            x: repaired,
            source: format!("round-repair:{}", s.source),
        });
    }
    out
}

fn repair_and_improve_candidate(
    p: &IPMIPProblem,
    x0: &[f64],
    tol: f64,
    passes: usize,
) -> Option<Vec<f64>> {
    let mut x = clamp_bounds(p, x0);
    let mut pass = 0;
    while pass < passes && !satisfies_linear_rows(p, &x, tol) {
        let base_violation = total_violation(p, &x);
        let mut best: Option<(usize, f64, f64)> = None; // (j, dir, score)
        for j in 0..x.len() {
            if !p.integer_vars[j] {
                continue;
            }
            for dir in [-1.0, 1.0] {
                let mut y = x.clone();
                y[j] += dir;
                if !bounds_ok(p, &y, tol) {
                    continue;
                }
                let new_violation = total_violation(p, &y);
                let reduction = base_violation - new_violation;
                if reduction <= 1e-12 {
                    continue;
                }
                let obj_loss = if p.sense == Sense::Max {
                    -dir * p.c[j]
                } else {
                    dir * p.c[j]
                };
                let score = reduction / 1e-9_f64.max(1.0 + 0.0_f64.max(obj_loss));
                if best.is_none() || score > best.unwrap().2 {
                    best = Some((j, dir, score));
                }
            }
        }
        let (bj, bdir, _) = best?;
        x[bj] += bdir;
        pass += 1;
    }
    if !is_integer_feasible(p, &x, tol) {
        return None;
    }

    for _pass in 0..passes {
        let mut improved = false;
        let mut best_x = x.clone();
        let mut best_z = objective(p, &x);
        for j in 0..x.len() {
            if !p.integer_vars[j] {
                continue;
            }
            for dir in [-1.0, 1.0] {
                let mut y = x.clone();
                y[j] += dir;
                if !is_integer_feasible(p, &y, tol) {
                    continue;
                }
                let z = objective(p, &y);
                if (p.sense == Sense::Max && z > best_z + 1e-9)
                    || (p.sense == Sense::Min && z < best_z - 1e-9)
                {
                    best_z = z;
                    best_x = y;
                    improved = true;
                }
            }
        }
        x = best_x;
        if !improved {
            break;
        }
    }
    Some(x)
}

fn generate_binary_cover_cuts(
    p: &IPMIPProblem,
    x: &[f64],
    tol: f64,
    max_cuts: usize,
    node: &IpNode,
) -> Vec<BranchOrCutConstraint> {
    let mut out: Vec<BranchOrCutConstraint> = Vec::new();
    let existing: HashSet<String> = node.constraints.iter().map(|c| c.name.clone()).collect();
    let mut r = 0;
    while r < p.a.len() && out.len() < max_cuts {
        let row = &p.a[r];
        if row.iter().any(|&a| a < -tol) {
            r += 1;
            continue;
        }
        let mut binary: Vec<(f64, usize, f64)> = row
            .iter()
            .enumerate()
            .map(|(j, &a)| (a, j, x[j]))
            .filter(|&(a, j, _)| {
                a > tol
                    && p.integer_vars[j]
                    && p.ub
                        .as_ref()
                        .and_then(|u| u.get(j))
                        .copied()
                        .unwrap_or(f64::INFINITY)
                        <= 1.0 + tol
            })
            .collect();
        binary.sort_by(|u, v| v.2.partial_cmp(&u.2).unwrap_or(std::cmp::Ordering::Equal));
        if binary.len() < 2 {
            r += 1;
            continue;
        }
        let mut sum = 0.0;
        let mut cover: Vec<usize> = Vec::new();
        for item in &binary {
            sum += item.0;
            cover.push(item.1);
            if sum > p.b[r] + tol {
                break;
            }
        }
        if sum <= p.b[r] + tol || cover.len() < 2 {
            r += 1;
            continue;
        }
        let mut coefs = vec![0.0; p.c.len()];
        for &j in &cover {
            coefs[j] = 1.0;
        }
        let rhs = (cover.len() - 1) as f64;
        let lhs: f64 = cover.iter().map(|&j| x[j]).sum();
        if lhs <= rhs + 1e-7 {
            r += 1;
            continue;
        }
        let name = format!(
            "cover_r{r}_{}",
            cover
                .iter()
                .map(|j| j.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        if existing.contains(&name) {
            r += 1;
            continue;
        }
        out.push(BranchOrCutConstraint {
            coefs,
            rhs,
            name,
            kind: ConstraintKind::Cut,
        });
        r += 1;
    }
    out
}

fn list_fractionals(x: &[f64], integer_vars: &[bool], tol: f64) -> Vec<usize> {
    let mut out = Vec::new();
    for j in 0..x.len() {
        if !integer_vars[j] {
            continue;
        }
        let f = x[j] - x[j].floor();
        if f > tol && f < 1.0 - tol {
            out.push(j);
        }
    }
    out
}

fn branch_priority(priorities: &[i32], var: usize) -> i32 {
    priorities.get(var).copied().unwrap_or(0)
}

fn pick_branch_var(
    x: &[f64],
    fractionals: &[usize],
    rule: BranchRule,
    branch_priorities: &[i32],
) -> usize {
    let highest_priority = fractionals
        .iter()
        .map(|&j| branch_priority(branch_priorities, j))
        .max()
        .unwrap_or(0);
    let priority_fractionals: Vec<usize> = fractionals
        .iter()
        .copied()
        .filter(|&j| branch_priority(branch_priorities, j) == highest_priority)
        .collect();
    let candidates = priority_fractionals.as_slice();
    if rule == BranchRule::FirstFractional {
        return candidates[0];
    }
    let mut best = candidates[0];
    let mut best_score = f64::NEG_INFINITY;
    for &j in candidates {
        let f = x[j] - x[j].floor();
        let score = f * (1.0 - f);
        if score > best_score {
            best = j;
            best_score = score;
        }
    }
    best
}

// -----------------------------------------------------------------------------
// Validation and common math
// -----------------------------------------------------------------------------

/// Validate an IP/MIP problem (TS `throw`s become `panic!`).
pub fn validate_ipmip_problem(p: &IPMIPProblem) {
    let model = MODEL;
    let req = |c: crate::des::general::des_base::preconditions::Check| {
        if let Err(e) = c {
            panic!("{e}");
        }
    };
    use crate::des::general::des_base::preconditions::Preconditions as P;
    req(P::check(
        model,
        "sense",
        "be max or min",
        true,
        Some(p.sense.as_str().to_string()),
    ));
    req(P::non_empty(model, "c", &p.c));
    req(P::non_empty(model, "A", &p.a));
    req(P::length_eq(model, "b", &p.b, p.a.len()));
    req(P::length_eq(
        model,
        "integerVars",
        &p.integer_vars,
        p.c.len(),
    ));
    req(P::all_finite(model, "c", &p.c));
    req(P::all_finite(model, "b", &p.b));
    for i in 0..p.a.len() {
        req(P::length_eq(model, &format!("A[{i}]"), &p.a[i], p.c.len()));
        req(P::all_finite(model, &format!("A[{i}]"), &p.a[i]));
    }
    if let Some(ub) = &p.ub {
        req(P::length_eq(model, "ub", ub, p.c.len()));
        for j in 0..ub.len() {
            if !ub[j].is_finite() {
                continue;
            }
            req(P::non_negative(model, &format!("ub[{j}]"), ub[j]));
        }
    }
    if let Some(vn) = &p.var_names {
        req(P::length_eq(model, "varNames", vn, p.c.len()));
    }
    if let Some(cn) = &p.con_names {
        req(P::length_eq(model, "conNames", cn, p.a.len()));
    }
    if let Some(rows) = &p.lazy_constraints {
        for (i, row) in rows.iter().enumerate() {
            req(P::length_eq(
                model,
                &format!("lazyConstraints[{i}].coefs"),
                &row.coefs,
                p.c.len(),
            ));
            req(P::all_finite(
                model,
                &format!("lazyConstraints[{i}].coefs"),
                &row.coefs,
            ));
            req(P::finite(
                model,
                &format!("lazyConstraints[{i}].rhs"),
                row.rhs,
            ));
        }
    }
}

fn validate_lower_bounded_ipmip_problem(problem: &LowerBoundedIPMIPProblem) {
    validate_ipmip_problem(&problem.base);
    let n = problem.base.c.len();
    if problem.lb.len() != n {
        panic!(
            "{MODEL}: lb length {} does not match variable count {n}",
            problem.lb.len()
        );
    }
    for (j, &lower) in problem.lb.iter().enumerate() {
        if !lower.is_finite() {
            panic!("{MODEL}: lb[{j}] must be finite");
        }
        if problem.base.integer_vars[j] && (lower - lower.round()).abs() > 1e-9 {
            panic!("{MODEL}: lb[{j}] for integer variable must be integral");
        }
        if let Some(ub) = &problem.base.ub {
            let upper = ub[j];
            if upper.is_finite() && upper + 1e-9 < lower {
                panic!("{MODEL}: lb[{j}] {lower} exceeds ub[{j}] {upper}");
            }
        }
    }
}

fn validate_linear_row_constraint(
    base: &IPMIPProblem,
    constraint: &LinearRowConstraint,
    idx: usize,
) {
    let n = base.c.len();
    if constraint.coefs.len() != n {
        panic!(
            "{MODEL}: linear row {idx} coefficient length {} does not match variable count {n}",
            constraint.coefs.len()
        );
    }
    for (j, &coef) in constraint.coefs.iter().enumerate() {
        if !coef.is_finite() {
            panic!("{MODEL}: linear row {idx} coefficient {j} must be finite");
        }
    }
    if constraint.lower.is_none() && constraint.upper.is_none() {
        panic!("{MODEL}: linear row {idx} needs at least one finite bound");
    }
    if let Some(lower) = constraint.lower {
        if !lower.is_finite() {
            panic!("{MODEL}: linear row {idx} lower bound must be finite");
        }
    }
    if let Some(upper) = constraint.upper {
        if !upper.is_finite() {
            panic!("{MODEL}: linear row {idx} upper bound must be finite");
        }
    }
    if let (Some(lower), Some(upper)) = (constraint.lower, constraint.upper) {
        if lower > upper + 1e-9 {
            panic!("{MODEL}: linear row {idx} lower bound {lower} exceeds upper bound {upper}");
        }
    }
}

fn validate_mip_start(p: &IPMIPProblem, start: Option<&[f64]>, tol: f64) {
    let Some(start) = start else {
        return;
    };
    if start.len() != p.c.len() {
        panic!(
            "{MODEL}: mip_start length {} does not match variable count {}",
            start.len(),
            p.c.len()
        );
    }
    for (j, &value) in start.iter().enumerate() {
        if !value.is_finite() {
            panic!("{MODEL}: mip_start[{j}] must be finite");
        }
    }
    if !is_integer_feasible(p, start, tol) {
        panic!("{MODEL}: mip_start must satisfy bounds, rows, and integrality");
    }
}

fn validate_solution_pool_bounds(p: &IPMIPProblem, integer_vars: &[usize]) {
    let Some(ub) = &p.ub else {
        panic!("{MODEL}: solution pool requires finite upper bounds for integer variables");
    };
    for &j in integer_vars {
        let upper = ub[j];
        if !upper.is_finite() {
            panic!("{MODEL}: solution pool requires finite ub[{j}] for integer variable");
        }
        if (upper - upper.floor()).abs() > 1e-9 {
            panic!("{MODEL}: solution pool requires integral ub[{j}] for integer variable");
        }
    }
}

fn append_integer_assignment_exclusion(
    p: &mut IPMIPProblem,
    integer_vars: &[usize],
    assignment: &[f64],
    tol: f64,
) {
    if assignment.len() > p.c.len() {
        panic!(
            "{MODEL}: solution pool assignment length {} exceeds problem variable count {}",
            assignment.len(),
            p.c.len()
        );
    }
    let mut ub =
        p.ub.clone()
            .unwrap_or_else(|| vec![f64::INFINITY; p.c.len()]);
    if ub.len() != p.c.len() {
        panic!(
            "{MODEL}: solution pool ub length {} does not match variable count {}",
            ub.len(),
            p.c.len()
        );
    }

    let mut selector_vars = Vec::new();
    for &j in integer_vars {
        let upper = ub[j];
        if !upper.is_finite() {
            panic!("{MODEL}: solution pool requires finite ub[{j}] for integer variable");
        }
        let value = assignment[j].round();
        if (assignment[j] - value).abs() > tol {
            panic!(
                "{MODEL}: solution pool assignment for integer variable {j} is not integral: {}",
                assignment[j]
            );
        }
        if value < -tol || value > upper + tol {
            panic!("{MODEL}: solution pool assignment for integer variable {j} is outside bounds");
        }

        let le_var = add_solution_pool_binary(p, &mut ub, format!("pool_excl_x{j}_lt_{value:.0}"));
        let ge_var = add_solution_pool_binary(p, &mut ub, format!("pool_excl_x{j}_gt_{value:.0}"));
        selector_vars.push(le_var);
        selector_vars.push(ge_var);

        let mut le_row = vec![0.0; p.c.len()];
        le_row[j] = 1.0;
        le_row[le_var] = upper - value + 1.0;
        p.a.push(le_row);
        p.b.push(upper);
        if let Some(names) = p.con_names.as_mut() {
            names.push(format!("pool_exclude_x{j}_le_{}", value - 1.0));
        }

        let mut ge_row = vec![0.0; p.c.len()];
        ge_row[j] = -1.0;
        ge_row[ge_var] = value + 1.0;
        p.a.push(ge_row);
        p.b.push(0.0);
        if let Some(names) = p.con_names.as_mut() {
            names.push(format!("pool_exclude_x{j}_ge_{}", value + 1.0));
        }
    }

    let mut exclude_row = vec![0.0; p.c.len()];
    for var in selector_vars {
        exclude_row[var] = -1.0;
    }
    p.a.push(exclude_row);
    p.b.push(-1.0);
    if let Some(names) = p.con_names.as_mut() {
        names.push("pool_exclude_previous_integer_assignment".to_string());
    }
    p.ub = Some(ub);
    p.constraint_nodes = None;
    validate_ipmip_problem(p);
}

fn add_solution_pool_binary(p: &mut IPMIPProblem, ub: &mut Vec<f64>, name: String) -> usize {
    for row in &mut p.a {
        row.push(0.0);
    }
    let idx = p.c.len();
    p.c.push(0.0);
    p.integer_vars.push(true);
    ub.push(1.0);
    if let Some(names) = p.var_names.as_mut() {
        names.push(name);
    }
    p.variable_nodes = None;
    idx
}

fn is_integer_feasible(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    is_integer_base_feasible(p, x, tol) && violated_lazy_constraints(p, x, tol, &[]).is_empty()
}

fn is_integer_base_feasible(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    if !bounds_ok(p, x, tol) || !satisfies_linear_rows(p, x, tol) {
        return false;
    }
    for j in 0..x.len() {
        if p.integer_vars[j] && (x[j] - x[j].round()).abs() > tol {
            return false;
        }
    }
    true
}

fn violated_lazy_constraints(
    p: &IPMIPProblem,
    x: &[f64],
    tol: f64,
    existing: &[BranchOrCutConstraint],
) -> Vec<BranchOrCutConstraint> {
    let Some(rows) = &p.lazy_constraints else {
        return Vec::new();
    };
    let existing_names = existing
        .iter()
        .map(|row| row.name.as_str())
        .collect::<HashSet<_>>();
    rows.iter()
        .filter(|row| {
            !existing_names.contains(row.name.as_str()) && branch_or_cut_lhs(row, x) > row.rhs + tol
        })
        .cloned()
        .collect()
}

fn branch_or_cut_lhs(row: &BranchOrCutConstraint, x: &[f64]) -> f64 {
    row.coefs
        .iter()
        .zip(x)
        .map(|(coef, value)| coef * value)
        .sum()
}

fn add_objective_offset(value: f64, offset: f64) -> f64 {
    if value.is_finite() {
        value + offset
    } else {
        value
    }
}

fn satisfies_linear_rows(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    for i in 0..p.a.len() {
        let mut lhs = 0.0;
        for j in 0..x.len() {
            lhs += p.a[i][j] * x[j];
        }
        if lhs > p.b[i] + tol {
            return false;
        }
    }
    true
}

fn bounds_ok(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    for j in 0..x.len() {
        if x[j] < -tol {
            return false;
        }
        if let Some(ub) = p.ub.as_ref().and_then(|u| u.get(j)).copied() {
            if ub.is_finite() && x[j] > ub + tol {
                return false;
            }
        }
    }
    true
}

fn total_violation(p: &IPMIPProblem, x: &[f64]) -> f64 {
    let mut v = 0.0;
    for i in 0..p.a.len() {
        let mut lhs = 0.0;
        for j in 0..x.len() {
            lhs += p.a[i][j] * x[j];
        }
        v += 0.0_f64.max(lhs - p.b[i]);
    }
    if let Some(rows) = &p.lazy_constraints {
        for row in rows {
            v += 0.0_f64.max(branch_or_cut_lhs(row, x) - row.rhs);
        }
    }
    for j in 0..x.len() {
        v += 0.0_f64.max(-x[j]);
        if let Some(ub) = p.ub.as_ref().and_then(|u| u.get(j)).copied() {
            if ub.is_finite() {
                v += 0.0_f64.max(x[j] - ub);
            }
        }
    }
    v
}

fn clamp_bounds(p: &IPMIPProblem, x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for j in 0..y.len() {
        y[j] = 0.0_f64.max(y[j]);
        if let Some(ub) = p.ub.as_ref().and_then(|u| u.get(j)).copied() {
            if ub.is_finite() {
                y[j] = ub.min(y[j]);
            }
        }
    }
    y
}

fn linear_objective_value(c: &[f64], x: &[f64]) -> f64 {
    c.iter().zip(x).map(|(&cj, &xj)| cj * xj).sum()
}

fn objective(p: &IPMIPProblem, x: &[f64]) -> f64 {
    linear_objective_value(&p.c, x)
}

fn bound_dominated(p: &IPMIPProblem, bound: f64, incumbent: f64, has_incumbent: bool) -> bool {
    if !has_incumbent {
        return false;
    }
    if p.sense == Sense::Max {
        bound <= incumbent + 1e-9
    } else {
        bound >= incumbent - 1e-9
    }
}

fn compute_best_bound(
    p: &IPMIPProblem,
    inc: &IncumbentStation,
    ctrl: &SearchControllerStation,
) -> f64 {
    if let Some(frontier) = ctrl.best_frontier_bound() {
        return frontier;
    }
    if inc.has_incumbent() {
        return inc.best_z;
    }
    if p.sense == Sense::Max {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    }
}

fn var_name(p: &IPMIPProblem, j: usize) -> String {
    p.var_names
        .as_ref()
        .and_then(|v| v.get(j))
        .cloned()
        .unwrap_or_else(|| format!("x{j}"))
}

fn to_inc_sense(s: Sense) -> IncSense {
    match s {
        Sense::Max => IncSense::Max,
        Sense::Min => IncSense::Min,
    }
}

fn solver_topology(parent_id: &str) -> Vec<SolverTopologyNode> {
    let pid = parent_id.to_string();
    vec![
        SolverTopologyNode {
            id: pid.clone(),
            role: "composite single-threaded in-house branch-and-cut solver".to_string(),
            emits: vec![],
            parent_id: None,
        },
        SolverTopologyNode {
            id: "ip-search-controller".to_string(),
            parent_id: Some(pid.clone()),
            role: "frontier of branch/cut subproblems".to_string(),
            emits: vec!["node".to_string()],
        },
        SolverTopologyNode {
            id: "ip-lp-relaxation".to_string(),
            parent_id: Some(pid.clone()),
            role: "stationary LP solver block with selectable backend".to_string(),
            emits: vec!["relaxation".to_string()],
        },
        SolverTopologyNode {
            id: "ip-rounding-repair".to_string(),
            parent_id: Some(pid.clone()),
            role: "movable-variable rounding, repair, and local search".to_string(),
            emits: vec!["candidate".to_string()],
        },
        SolverTopologyNode {
            id: "ip-incumbent".to_string(),
            parent_id: Some(pid.clone()),
            role: "best feasible integer solution anchor".to_string(),
            emits: vec![],
        },
        SolverTopologyNode {
            id: "ip-cut-generator".to_string(),
            parent_id: Some(pid.clone()),
            role: "valid-inequality station, currently binary cover cuts".to_string(),
            emits: vec!["cut".to_string()],
        },
        SolverTopologyNode {
            id: "ip-node-decision".to_string(),
            parent_id: Some(pid),
            role: "prune, strengthen, or branch".to_string(),
            emits: vec!["node".to_string(), "complete".to_string()],
        },
    ]
}

// -----------------------------------------------------------------------------
// Convenience builders
// -----------------------------------------------------------------------------

/// Build a binary knapsack IP `max v·x s.t. w·x <= capacity, x in {0,1}`.
pub fn build_binary_knapsack_ip(
    values: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
) -> IPMIPProblem {
    if let Err(e) = crate::des::general::des_base::preconditions::Preconditions::length_eq(
        MODEL,
        "weights",
        &weights,
        values.len(),
    ) {
        panic!("{e}");
    }
    let n = values.len();
    IPMIPProblem {
        sense: Sense::Max,
        c: values.clone(),
        a: vec![weights],
        b: vec![capacity],
        integer_vars: vec![true; n],
        ub: Some(vec![1.0; n]),
        var_names: Some((0..n).map(|i| format!("item_{i}")).collect()),
        con_names: Some(vec!["capacity".to_string()]),
        lazy_constraints: None,
        variable_nodes: Some(
            (0..n)
                .map(|i| VariableNode {
                    var_index: i,
                    node_id: format!("item_{i}"),
                    label: Some(format!("item {i}")),
                })
                .collect(),
        ),
        constraint_nodes: Some(vec![ConstraintNode {
            row_index: 0,
            node_id: "capacity".to_string(),
            label: Some("capacity anchor".to_string()),
        }]),
    }
}

/// Build a small mixed-integer program.
pub fn build_small_mixed_ip() -> IPMIPProblem {
    IPMIPProblem {
        sense: Sense::Max,
        c: vec![1.0, 1.0, 1.0],
        a: vec![vec![1.0, 1.0, 0.0]],
        b: vec![3.0],
        integer_vars: vec![true, true, false],
        ub: Some(vec![10.0, 10.0, 10.0]),
        var_names: Some(vec![
            "x_int_a".to_string(),
            "x_int_b".to_string(),
            "y_cont".to_string(),
        ]),
        con_names: Some(vec!["integer_sum".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    }
}

/// Build a lower-bounded production mix model:
/// `max x + 2y`, `x + y <= 8`, `3 <= x <= 6`, `0 <= y <= 10`.
/// The lower bound forces the unique optimum to `x = 3, y = 5`.
pub fn build_lower_bounded_production_ip() -> LowerBoundedIPMIPProblem {
    LowerBoundedIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![8.0],
            integer_vars: vec![false, false],
            ub: Some(vec![6.0, 10.0]),
            var_names: Some(vec!["base_load".to_string(), "premium_load".to_string()]),
            con_names: Some(vec!["total_capacity".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: vec![3.0, 0.0],
    }
}

/// Build a model with source-level equality, greater-than, and ranged rows:
/// `max 3x + 2y`, `x + 2y = 8`, `x - y >= 1`, `5 <= x + y <= 7`.
/// The unique optimum is `x = 6, y = 1`.
pub fn build_general_linear_rows_ip() -> GeneralLinearIPMIPProblem {
    GeneralLinearIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 0.0]],
            b: vec![10.0],
            integer_vars: vec![false, false],
            ub: Some(vec![10.0, 10.0]),
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            con_names: Some(vec!["x_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        linear_constraints: vec![
            LinearRowConstraint {
                coefs: vec![1.0, 2.0],
                lower: Some(8.0),
                upper: Some(8.0),
                name: Some("balance_eq".to_string()),
            },
            LinearRowConstraint {
                coefs: vec![1.0, -1.0],
                lower: Some(1.0),
                upper: None,
                name: Some("dominance_ge".to_string()),
            },
            LinearRowConstraint {
                coefs: vec![1.0, 1.0],
                lower: Some(5.0),
                upper: Some(7.0),
                name: Some("throughput_range".to_string()),
            },
        ],
    }
}

/// Build a fixed-charge indicator MIP:
/// `use = 0 => production <= 0`, max `5 production - 3 use`.
pub fn build_fixed_charge_indicator_ip() -> IndicatorIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![-3.0, 5.0],
        a: vec![vec![0.0, 1.0]],
        b: vec![4.0],
        integer_vars: vec![true, false],
        ub: Some(vec![1.0, 4.0]),
        var_names: Some(vec!["use_line".to_string(), "production".to_string()]),
        con_names: Some(vec!["production_cap".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    IndicatorIPMIPProblem {
        base,
        indicators: vec![IndicatorConstraint {
            binary_var: 0,
            active_value: false,
            coefs: vec![0.0, 1.0],
            sense: IndicatorSense::Le,
            rhs: 0.0,
            name: Some("closed_line_no_production".to_string()),
        }],
    }
}

/// Build an SOS1 choice model:
/// at most one activity in the set may be positive.
pub fn build_sos1_choice_ip() -> SosIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![10.0, 9.0, 1.0],
        a: vec![vec![1.0, 1.0, 1.0]],
        b: vec![8.0],
        integer_vars: vec![false, false, false],
        ub: Some(vec![4.0, 4.0, 4.0]),
        var_names: Some(vec![
            "activity_a".to_string(),
            "activity_b".to_string(),
            "activity_c".to_string(),
        ]),
        con_names: Some(vec!["budget".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    SosIPMIPProblem {
        base,
        sos: vec![SpecialOrderedSet {
            kind: SpecialOrderedSetKind::Sos1,
            vars: vec![0, 1, 2],
            weights: Some(vec![1.0, 2.0, 3.0]),
            name: Some("activity_choice".to_string()),
        }],
    }
}

/// Build an SOS2 adjacency model where the two best variables are nonadjacent.
pub fn build_sos2_adjacency_ip() -> SosIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![10.0, 0.0, 0.0, 10.0],
        a: vec![vec![1.0, 1.0, 1.0, 1.0]],
        b: vec![2.0],
        integer_vars: vec![false, false, false, false],
        ub: Some(vec![1.0, 1.0, 1.0, 1.0]),
        var_names: Some(vec![
            "lambda_0".to_string(),
            "lambda_1".to_string(),
            "lambda_2".to_string(),
            "lambda_3".to_string(),
        ]),
        con_names: Some(vec!["mass".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    SosIPMIPProblem {
        base,
        sos: vec![SpecialOrderedSet {
            kind: SpecialOrderedSetKind::Sos2,
            vars: vec![0, 1, 2, 3],
            weights: Some(vec![0.0, 1.0, 2.0, 3.0]),
            name: Some("piecewise_lambdas".to_string()),
        }],
    }
}

/// Build a semi-continuous model where the ordinary LP optimum would choose
/// `0 < x < lower`, so the semi domain forces the machine off.
pub fn build_semi_continuous_gate_ip() -> SemiIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![1.0],
        a: vec![vec![1.0]],
        b: vec![2.0],
        integer_vars: vec![false],
        ub: Some(vec![5.0]),
        var_names: Some(vec!["production".to_string()]),
        con_names: Some(vec!["small_order_cap".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    SemiIPMIPProblem {
        base,
        semi_variables: vec![SemiVariable {
            kind: SemiVariableKind::SemiContinuous,
            var: 0,
            lower: 3.0,
            name: Some("production".to_string()),
        }],
    }
}

/// Build a semi-integer lot-size model with a fractional resource cap. The
/// source variable starts continuous, and the semi-integer compiler makes it
/// integral in the ordinary MIP.
pub fn build_semi_integer_lot_ip() -> SemiIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![2.0],
        a: vec![vec![1.0]],
        b: vec![4.5],
        integer_vars: vec![false],
        ub: Some(vec![5.0]),
        var_names: Some(vec!["lot_size".to_string()]),
        con_names: Some(vec!["resource_cap".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    SemiIPMIPProblem {
        base,
        semi_variables: vec![SemiVariable {
            kind: SemiVariableKind::SemiInteger,
            var: 0,
            lower: 3.0,
            name: Some("lot_size".to_string()),
        }],
    }
}

/// Build a non-convex PWL reward model:
/// `reward = f(activity)`, max reward, with the best breakpoint at
/// `activity = 1`.
pub fn build_piecewise_linear_reward_ip() -> PwlIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![0.0, 1.0],
        a: vec![vec![1.0, 0.0]],
        b: vec![3.0],
        integer_vars: vec![false, false],
        ub: Some(vec![3.0, 4.0]),
        var_names: Some(vec!["activity".to_string(), "reward".to_string()]),
        con_names: Some(vec!["activity_cap".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    PwlIPMIPProblem {
        base,
        pwl: vec![PiecewiseLinearConstraint {
            x_var: 0,
            y_var: 1,
            points: vec![
                PiecewiseLinearPoint { x: 0.0, y: 0.0 },
                PiecewiseLinearPoint { x: 1.0, y: 4.0 },
                PiecewiseLinearPoint { x: 2.0, y: 1.0 },
                PiecewiseLinearPoint { x: 3.0, y: 3.0 },
            ],
            name: Some("activity_reward".to_string()),
        }],
    }
}

/// Build a bounded absolute-value penalty model:
/// `deviation <= -2`, `penalty = |deviation|`, minimize `penalty`, with
/// `-3 <= deviation <= 4`. The exact optimum is `deviation = -2`, penalty 2.
pub fn build_absolute_value_penalty_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0, 1.0],
            a: vec![vec![1.0, 0.0]],
            b: vec![-2.0],
            integer_vars: vec![false, false],
            ub: Some(vec![4.0, 4.0]),
            var_names: Some(vec!["deviation".to_string(), "penalty".to_string()]),
            con_names: Some(vec!["negative_deviation_required".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![-3.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: vec![AbsoluteValueConstraint {
            arg_var: 0,
            target_var: 1,
            name: Some("absolute_deviation".to_string()),
        }],
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a bounded maximum general-constraint model:
/// `peak = max(load_a, load_b, 1.5)`, minimize `peak + 0.01 load_a`, with
/// `-1 <= load_a`, `2 <= load_b`. The unique optimum is
/// `load_a = -1`, `load_b = peak = 2`, objective 1.99.
pub fn build_maximum_peak_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.01, 0.0, 1.0],
            a: vec![vec![-1.0, 0.0, 0.0], vec![0.0, -1.0, 0.0]],
            b: vec![1.0, -2.0],
            integer_vars: vec![false, false, false],
            ub: Some(vec![4.0, 5.0, 5.0]),
            var_names: Some(vec![
                "load_a".to_string(),
                "load_b".to_string(),
                "peak".to_string(),
            ]),
            con_names: Some(vec!["load_a_floor".to_string(), "load_b_floor".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![-2.0, 0.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: vec![MaximumConstraint {
            target_var: 2,
            arg_vars: vec![0, 1],
            constant: Some(1.5),
            name: Some("peak_load".to_string()),
        }],
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a bounded minimum general-constraint model:
/// `floor = min(load_a, load_b, 2.5)`, maximize
/// `floor - 0.01 load_a - 0.001 load_b`, with `load_a >= 3`,
/// `load_b >= 2.5`, and `-2 <= load_a`. The unique optimum is
/// `load_a = 3`, `load_b = floor = 2.5`, objective 2.4675.
pub fn build_minimum_floor_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![-0.01, -0.001, 1.0],
            a: vec![vec![-1.0, 0.0, 0.0], vec![0.0, -1.0, 0.0]],
            b: vec![-3.0, -2.5],
            integer_vars: vec![false, false, false],
            ub: Some(vec![4.0, 5.0, 3.0]),
            var_names: Some(vec![
                "load_a".to_string(),
                "load_b".to_string(),
                "floor".to_string(),
            ]),
            con_names: Some(vec!["load_a_floor".to_string(), "load_b_floor".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![-2.0, 0.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: vec![MinimumConstraint {
            target_var: 2,
            arg_vars: vec![0, 1],
            constant: Some(2.5),
            name: Some("service_floor".to_string()),
        }],
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a binary logical-gate model:
/// `both = and(choice_a, choice_b)`, `either = or(choice_a, choice_b)`,
/// `choice_a + choice_b <= 1`, maximize `either + 0.1 choice_a
/// + 0.01 choice_b + 2 both`. The unique optimum is
/// `choice_a = 1`, `choice_b = 0`, `both = 0`, `either = 1`.
pub fn build_logical_gate_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![0.1, 0.01, 2.0, 1.0],
            a: vec![vec![1.0, 1.0, 0.0, 0.0]],
            b: vec![1.0],
            integer_vars: vec![true, true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0, 1.0]),
            var_names: Some(vec![
                "choice_a".to_string(),
                "choice_b".to_string(),
                "both".to_string(),
                "either".to_string(),
            ]),
            con_names: Some(vec!["choose_at_most_one".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: None,
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: vec![
            LogicalConstraint {
                kind: LogicalConstraintKind::And,
                target_var: 2,
                arg_vars: vec![0, 1],
                name: Some("both_choices".to_string()),
            },
            LogicalConstraint {
                kind: LogicalConstraintKind::Or,
                target_var: 3,
                arg_vars: vec![0, 1],
                name: Some("either_choice".to_string()),
            },
        ],
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a bounded L1 norm model:
/// `norm = |dev_a| + |dev_b|`, `dev_a >= 1`, `dev_b <= -2`, minimize
/// `norm + 0.01 dev_a - 0.02 dev_b`, with signed bounded deviations. The
/// unique optimum is `dev_a = 1`, `dev_b = -2`, `norm = 3`, objective 3.05.
pub fn build_l1_norm_deviation_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.01, -0.02, 1.0],
            a: vec![vec![-1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            b: vec![-1.0, -2.0],
            integer_vars: vec![false, false, false],
            ub: Some(vec![3.0, 2.0, 8.0]),
            var_names: Some(vec![
                "dev_a".to_string(),
                "dev_b".to_string(),
                "norm".to_string(),
            ]),
            con_names: Some(vec!["dev_a_floor".to_string(), "dev_b_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![-3.0, -4.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: vec![L1NormConstraint {
            target_var: 2,
            arg_vars: vec![0, 1],
            name: Some("deviation_l1".to_string()),
        }],
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a bounded infinity norm model:
/// `radius = max(|dev_a|, |dev_b|)`, `dev_a >= 1.5`, `dev_b <= -2`,
/// minimize `radius + 0.01 dev_a - 0.02 dev_b`, with signed bounded
/// deviations. The unique optimum is `dev_a = 1.5`, `dev_b = -2`,
/// `radius = 2`, objective 2.055.
pub fn build_linf_norm_deviation_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.01, -0.02, 1.0],
            a: vec![vec![-1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            b: vec![-1.5, -2.0],
            integer_vars: vec![false, false, false],
            ub: Some(vec![3.0, 2.0, 5.0]),
            var_names: Some(vec![
                "dev_a".to_string(),
                "dev_b".to_string(),
                "radius".to_string(),
            ]),
            con_names: Some(vec!["dev_a_floor".to_string(), "dev_b_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![-3.0, -4.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: vec![LInfNormConstraint {
            target_var: 2,
            arg_vars: vec![0, 1],
            name: Some("deviation_linf".to_string()),
        }],
        products: Vec::new(),
    }
}

/// Build a bounded binary-continuous product model:
/// `served = use_machine * activity`, maximize
/// `served - 0.1 use_machine - 0.01 activity`, with
/// `1 <= activity <= 4`. The unique optimum is `use_machine = 1`,
/// `activity = served = 4`, objective 3.86.
pub fn build_product_activation_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![-0.1, -0.01, 1.0],
            a: vec![vec![0.0, 1.0, 0.0]],
            b: vec![4.0],
            integer_vars: vec![true, false, false],
            ub: Some(vec![1.0, 4.0, 4.0]),
            var_names: Some(vec![
                "use_machine".to_string(),
                "activity".to_string(),
                "served".to_string(),
            ]),
            con_names: Some(vec!["activity_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![0.0, 1.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: vec![ProductConstraint {
            target_var: 2,
            x_var: 0,
            y_var: 1,
            name: Some("machine_activity_product".to_string()),
        }],
    }
}

/// Build a binary-binary product model:
/// `accepted = choose_a * choose_b`, maximize
/// `accepted - 0.1 choose_a - 0.2 choose_b`. The unique optimum is
/// `choose_a = choose_b = accepted = 1`, objective 0.7.
pub fn build_binary_product_gate_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![-0.1, -0.2, 1.0],
            a: vec![vec![1.0, 1.0, 0.0]],
            b: vec![2.0],
            integer_vars: vec![true, true, true],
            ub: Some(vec![1.0, 1.0, 1.0]),
            var_names: Some(vec![
                "choose_a".to_string(),
                "choose_b".to_string(),
                "accepted".to_string(),
            ]),
            con_names: Some(vec!["binary_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![0.0, 0.0, 0.0]),
        linear_constraints: Vec::new(),
        indicators: Vec::new(),
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: Vec::new(),
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: vec![ProductConstraint {
            target_var: 2,
            x_var: 0,
            y_var: 1,
            name: Some("binary_product_gate".to_string()),
        }],
    }
}

/// Build a source-level quadratic-objective MIP:
/// maximize
/// `2 use_machine * premium + 1.2 use_machine * activity
///  - 0.1 use_machine - 0.2 premium - 0.01 activity`,
/// with `1 <= activity <= 4`. The unique optimum is `use_machine = 1`,
/// `premium = 1`, `activity = 4`, objective 6.46.
pub fn build_quadratic_objective_mix_ip() -> QuadraticObjectiveIPMIPProblem {
    QuadraticObjectiveIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![-0.1, -0.2, -0.01],
            a: vec![vec![0.0, 0.0, 1.0]],
            b: vec![4.0],
            integer_vars: vec![true, true, false],
            ub: Some(vec![1.0, 1.0, 4.0]),
            var_names: Some(vec![
                "use_machine".to_string(),
                "premium".to_string(),
                "activity".to_string(),
            ]),
            con_names: Some(vec!["activity_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![0.0, 0.0, 1.0]),
        quadratic_objective: vec![
            QuadraticObjectiveTerm {
                x_var: 0,
                y_var: 1,
                coeff: 2.0,
                name: Some("machine_premium_bonus".to_string()),
            },
            QuadraticObjectiveTerm {
                x_var: 0,
                y_var: 2,
                coeff: 1.2,
                name: Some("machine_activity_bonus".to_string()),
            },
        ],
    }
}

/// Build a source-level model that combines several commercial-solver-style
/// features before compilation: nonzero variable lower bounds, a ranged linear
/// row, an indicator row, and a non-convex PWL reward curve.
pub fn build_source_feature_mix_ip() -> SourceIPMIPProblem {
    SourceIPMIPProblem {
        base: IPMIPProblem {
            sense: Sense::Max,
            c: vec![0.0, 1.0, -1.0],
            a: vec![vec![1.0, 0.0, 0.0]],
            b: vec![4.0],
            integer_vars: vec![false, false, true],
            ub: Some(vec![4.0, 6.0, 1.0]),
            var_names: Some(vec![
                "activity".to_string(),
                "reward".to_string(),
                "use_upgrade".to_string(),
            ]),
            con_names: Some(vec!["activity_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        lb: Some(vec![1.0, 0.0, 0.0]),
        linear_constraints: vec![LinearRowConstraint {
            coefs: vec![1.0, 1.0, 0.0],
            lower: None,
            upper: Some(7.0),
            name: Some("activity_reward_budget".to_string()),
        }],
        indicators: vec![IndicatorConstraint {
            binary_var: 2,
            active_value: false,
            coefs: vec![1.0, 0.0, 0.0],
            sense: IndicatorSense::Le,
            rhs: 1.0,
            name: Some("upgrade_needed_above_baseline".to_string()),
        }],
        sos: Vec::new(),
        semi_variables: Vec::new(),
        pwl: vec![PiecewiseLinearConstraint {
            x_var: 0,
            y_var: 1,
            points: vec![
                PiecewiseLinearPoint { x: 1.0, y: 0.0 },
                PiecewiseLinearPoint { x: 2.0, y: 5.0 },
                PiecewiseLinearPoint { x: 3.0, y: 1.0 },
                PiecewiseLinearPoint { x: 4.0, y: 3.0 },
            ],
            name: Some("upgrade_reward".to_string()),
        }],
        abs: Vec::new(),
        maximums: Vec::new(),
        minimums: Vec::new(),
        logical: Vec::new(),
        l1_norms: Vec::new(),
        linf_norms: Vec::new(),
        products: Vec::new(),
    }
}

/// Build a lexicographic MIP:
/// first select as many choices as possible, then prefer choice A over B among
/// equal-cardinality selections.
pub fn build_lexicographic_choice_ip() -> MultiObjectiveIPMIPProblem {
    let base = IPMIPProblem {
        sense: Sense::Max,
        c: vec![0.0, 0.0],
        a: vec![vec![1.0, 1.0]],
        b: vec![1.0],
        integer_vars: vec![true, true],
        ub: Some(vec![1.0, 1.0]),
        var_names: Some(vec!["choice_a".to_string(), "choice_b".to_string()]),
        con_names: Some(vec!["choose_at_most_one".to_string()]),
        lazy_constraints: None,
        variable_nodes: None,
        constraint_nodes: None,
    };
    MultiObjectiveIPMIPProblem {
        base,
        objectives: vec![
            LexicographicObjective {
                sense: Sense::Max,
                c: vec![1.0, 1.0],
                name: Some("maximize_cardinality".to_string()),
            },
            LexicographicObjective {
                sense: Sense::Max,
                c: vec![3.0, 1.0],
                name: Some("prefer_choice_a".to_string()),
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    //! The branch-and-cut DES recovers the known optima of a small binary
    //! knapsack and a mixed-integer toy, staying entirely in-house.

    use super::*;

    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    #[test]
    fn solves_binary_knapsack() {
        // values [60,100,120], weights [10,20,30], capacity 50 -> take items 1,2 = 220.
        let p = build_binary_knapsack_ip(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let sol = solve_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal, "status={:?}", sol.status);
        assert!((sol.z - 220.0).abs() < 1e-6, "z={}", sol.z);
        assert!(sol.in_house_only);
        assert!(sol.lp_solves >= 1);
    }

    #[test]
    fn external_lp_backend_requires_explicit_opt_in() {
        let p = build_binary_knapsack_ip(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let result = std::panic::catch_unwind(|| {
            solve_ipmip_with_des(
                p,
                IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::ExternalHighs,
                    )),
                    allow_external_solvers: Some(false),
                    ..Default::default()
                },
            )
        });

        assert!(
            result.is_err(),
            "external LP backend should be rejected unless allow_external_solvers is true"
        );
        let message = panic_message(result.as_ref().err().expect("panic payload").as_ref());
        assert!(
            message.contains("allowExternalSolvers is false"),
            "unexpected panic: {message}"
        );
    }

    #[test]
    fn mip_start_seeds_incumbent_when_feasible() {
        let p = build_binary_knapsack_ip(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let sol = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                mip_start: Some(vec![0.0, 1.0, 1.0]),
                ..Default::default()
            },
        );

        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert_eq!(sol.incumbent_source.as_deref(), Some("user-mip-start"));
        assert_eq!(
            sol.x.iter().map(|v| v.round() as i32).collect::<Vec<_>>(),
            vec![0, 1, 1]
        );
    }

    #[test]
    fn lazy_constraint_cuts_integer_candidate_before_incumbent() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![2.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec!["x".to_string(), "y".to_string()]),
            con_names: Some(vec!["loose-capacity".to_string()]),
            lazy_constraints: Some(vec![BranchOrCutConstraint {
                coefs: vec![1.0, 1.0],
                rhs: 1.0,
                name: "lazy-at-most-one".to_string(),
                kind: ConstraintKind::Lazy,
            }]),
            variable_nodes: None,
            constraint_nodes: None,
        };

        let sol = solve_ipmip_with_des(p, IPMIPSolveOptions::default());

        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 1.0).abs() < 1e-6, "z={}", sol.z);
        assert!(sol.cuts_added >= 1);
        assert!(sol.trace.iter().any(|event| {
            event.action == TraceAction::Cut
                && event
                    .reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("lazy constraint"))
        }));
    }

    #[test]
    fn solves_small_mixed_ip() {
        let p = build_small_mixed_ip();
        let sol = solve_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        // max x_a + x_b + y with x_a + x_b <= 3, all <= 10. Optimum = 3 + 10 = 13.
        assert!((sol.z - 13.0).abs() < 1e-6, "z={}", sol.z);
    }

    #[test]
    fn branch_priorities_select_first_branch_variable() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            b: vec![0.5, 0.5],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec![
                "low_priority".to_string(),
                "high_priority".to_string(),
            ]),
            con_names: Some(vec!["cap_low".to_string(), "cap_high".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };

        let sol = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                branch_rule: Some(BranchRule::FirstFractional),
                branch_priorities: Some(vec![0, 10]),
                max_cut_rounds: Some(0),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                ..Default::default()
            },
        );
        let first_branch = sol
            .trace
            .iter()
            .find(|event| event.action == TraceAction::Branch)
            .expect("branch event");

        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert_eq!(first_branch.branch_var, Some(1));
    }

    #[test]
    fn mip_gap_limit_stops_with_accepted_incumbent() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![1.0, 10.0],
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            b: vec![0.5, 1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec!["fractional_bonus".to_string(), "accepted".to_string()]),
            con_names: Some(vec!["bonus_cap".to_string(), "accepted_cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };

        let sol = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                mip_start: Some(vec![0.0, 1.0]),
                mip_gap_rel: Some(0.051),
                max_cut_rounds: Some(0),
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                ..Default::default()
            },
        );

        assert_eq!(sol.status, IPMIPStatus::GapLimit);
        assert!((sol.z - 10.0).abs() < 1e-6, "z={}", sol.z);
        assert!(
            (sol.best_bound - 10.5).abs() < 1e-6,
            "bound={}",
            sol.best_bound
        );
        assert!(sol.gap <= 0.051 + 1e-9, "gap={}", sol.gap);
    }

    #[test]
    fn enumerates_binary_solution_pool() {
        let p = IPMIPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0]],
            b: vec![1.0],
            integer_vars: vec![true, true],
            ub: Some(vec![1.0, 1.0]),
            var_names: Some(vec!["choose_a".to_string(), "choose_b".to_string()]),
            con_names: Some(vec!["choose_at_most_one".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let pool = solve_ipmip_solution_pool_with_des(
            p,
            IPMIPSolutionPoolOptions {
                max_solutions: Some(4),
                solve_options: IPMIPSolveOptions {
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    max_cut_rounds: Some(0),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert_eq!(pool.status, IPMIPStatus::Optimal);
        assert!(pool.exhausted);
        assert_eq!(pool.solutions.len(), 3);
        assert_eq!(pool.solutions[0].x, vec![1.0, 0.0]);
        assert_eq!(pool.solutions[1].x, vec![0.0, 1.0]);
        assert_eq!(pool.solutions[2].x, vec![0.0, 0.0]);
        assert!((pool.solutions[0].z - 3.0).abs() < 1e-6);
        assert!((pool.solutions[1].z - 2.0).abs() < 1e-6);
        assert!((pool.solutions[2].z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mip_feasibility_relaxation_preserves_integrality_and_repairs_rows() {
        let p = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a: vec![vec![1.0], vec![-1.0]],
            b: vec![0.5, -0.5],
            integer_vars: vec![true],
            ub: None,
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec!["x_le_half".to_string(), "x_ge_half".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };
        let relax = solve_ipmip_feasibility_relaxation_with_des(
            &p,
            &IPMIPFeasRelaxOptions {
                row_penalties: Some(vec![3.0, 1.0]),
                solve_options: IPMIPSolveOptions {
                    max_cut_rounds: Some(0),
                    lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                        ConcreteLpRelaxationAlgorithm::InternalSimplex,
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        assert_eq!(relax.status, IPMIPStatus::Optimal, "{:?}", relax.message);
        assert!((relax.relaxation_cost - 0.5).abs() < 1e-6);
        assert!((relax.x[0] - 0.0).abs() < 1e-6, "x={:?}", relax.x);
        assert_eq!(relax.violations.len(), 1);
        assert_eq!(
            relax.violations[0].member,
            IPMIPFeasRelaxMember::LinearRow(1)
        );
        assert!((relax.violations[0].amount - 0.5).abs() < 1e-6);
    }

    #[test]
    fn finds_integrality_mip_infeasibility_conflict() {
        let p = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a: vec![vec![1.0], vec![-1.0], vec![1.0]],
            b: vec![0.5, -0.5, 10.0],
            integer_vars: vec![true],
            ub: None,
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_le_half".to_string(),
                "x_ge_half".to_string(),
                "redundant_cap".to_string(),
            ]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };

        let conflict = find_ipmip_infeasibility_conflict(&p, &IPMIPConflictOptions::default());

        assert!(conflict.infeasible, "{:?}", conflict.message);
        assert!(conflict.minimal, "{:?}", conflict.message);
        assert_eq!(
            conflict.members,
            vec![
                IPMIPConflictMember::LinearRow(0),
                IPMIPConflictMember::LinearRow(1),
                IPMIPConflictMember::Integrality(0),
            ]
        );
        let subsystem = ipmip_feasibility_problem_from_conflict_members(&p, &conflict.members);
        assert_eq!(
            solve_ipmip_with_des(subsystem, IPMIPConflictOptions::default().solve_options).status,
            IPMIPStatus::Infeasible
        );

        for idx in 0..conflict.members.len() {
            let mut trial = conflict.members.clone();
            trial.remove(idx);
            let subsystem = ipmip_feasibility_problem_from_conflict_members(&p, &trial);
            assert_eq!(
                solve_ipmip_with_des(subsystem, IPMIPConflictOptions::default().solve_options)
                    .status,
                IPMIPStatus::Optimal
            );
        }
    }

    #[test]
    fn mip_conflict_finder_reports_feasible_models() {
        let p = IPMIPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a: vec![vec![1.0]],
            b: vec![1.0],
            integer_vars: vec![true],
            ub: Some(vec![1.0]),
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec!["cap".to_string()]),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        };

        let conflict = find_ipmip_infeasibility_conflict(&p, &IPMIPConflictOptions::default());

        assert!(!conflict.infeasible);
        assert!(!conflict.minimal);
        assert!(conflict.members.is_empty());
    }

    #[test]
    fn uses_mip_start_as_initial_incumbent() {
        let p = build_binary_knapsack_ip(vec![60.0, 100.0, 120.0], vec![10.0, 20.0, 30.0], 50.0);
        let sol = solve_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                max_nodes: Some(0),
                mip_start: Some(vec![1.0, 0.0, 0.0]),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::MaxNodes);
        assert_eq!(sol.incumbent_source.as_deref(), Some("user-mip-start"));
        assert_eq!(sol.x, vec![1.0, 0.0, 0.0]);
        assert!((sol.z - 60.0).abs() < 1e-6, "z={}", sol.z);
        assert_eq!(sol.nodes_explored, 0);
    }

    #[test]
    fn solves_lower_bounded_ip() {
        let p = build_lower_bounded_production_ip();
        let (linearized, offset) = linearize_lower_bounds_problem(&p);
        assert_eq!(linearized.b, vec![5.0]);
        assert_eq!(linearized.ub.as_ref().unwrap(), &vec![3.0, 10.0]);
        assert!((offset - 3.0).abs() < 1e-9, "offset={offset}");
        let sol = solve_lower_bounded_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 13.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 3.0).abs() < 1e-6, "x={}", sol.x[0]);
        assert!((sol.x[1] - 5.0).abs() < 1e-6, "y={}", sol.x[1]);
    }

    #[test]
    fn lower_bounded_ip_shifts_mip_start() {
        let p = build_lower_bounded_production_ip();
        let sol = solve_lower_bounded_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                max_nodes: Some(0),
                mip_start: Some(vec![3.0, 5.0]),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::MaxNodes);
        assert_eq!(sol.incumbent_source.as_deref(), Some("user-mip-start"));
        assert!((sol.z - 13.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 3.0).abs() < 1e-6, "x={}", sol.x[0]);
        assert!((sol.x[1] - 5.0).abs() < 1e-6, "y={}", sol.x[1]);
        assert_eq!(sol.nodes_explored, 0);
    }

    #[test]
    fn solves_general_linear_rows_ip() {
        let p = build_general_linear_rows_ip();
        let linearized = linearize_general_linear_problem(&p);
        assert_eq!(linearized.a.len(), p.base.a.len() + 5);
        let sol = solve_general_linear_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 20.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 6.0).abs() < 1e-6, "x={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "y={}", sol.x[1]);
    }

    #[test]
    fn solves_fixed_charge_indicator_ip() {
        let p = build_fixed_charge_indicator_ip();
        let linearized = linearize_indicator_problem(&p);
        assert_eq!(linearized.a.len(), p.base.a.len() + 1);
        let sol = solve_indicator_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 17.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "use={}", sol.x[0]);
        assert!((sol.x[1] - 4.0).abs() < 1e-6, "prod={}", sol.x[1]);
    }

    #[test]
    fn solves_sos1_choice_ip() {
        let p = build_sos1_choice_ip();
        let linearized = linearize_sos_problem(&p);
        assert!(linearized.c.len() > p.base.c.len());
        let sol = solve_sos_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 40.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 4.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!(sol.x[1].abs() < 1e-6, "x1={}", sol.x[1]);
        assert!(sol.x[2].abs() < 1e-6, "x2={}", sol.x[2]);
    }

    #[test]
    fn solves_sos2_adjacency_ip() {
        let p = build_sos2_adjacency_ip();
        let linearized = linearize_sos_problem(&p);
        assert_eq!(linearized.c.len(), p.base.c.len() + p.base.c.len() - 1);
        let sol = solve_sos_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 10.0).abs() < 1e-6, "z={}", sol.z);
        let positive: Vec<usize> = sol
            .x
            .iter()
            .take(4)
            .enumerate()
            .filter_map(|(i, &v)| if v > 1e-6 { Some(i) } else { None })
            .collect();
        assert!(
            positive.len() <= 2 && positive.windows(2).all(|window| window[1] == window[0] + 1),
            "positive={positive:?} x={:?}",
            sol.x
        );
    }

    #[test]
    fn solves_semi_continuous_gate_ip() {
        let p = build_semi_continuous_gate_ip();
        let linearized = linearize_semi_problem(&p);
        assert_eq!(linearized.c.len(), p.base.c.len() + 1);
        assert_eq!(linearized.a.len(), p.base.a.len() + 2);
        assert!(!linearized.integer_vars[0]);
        assert!(linearized.integer_vars[1]);
        let sol = solve_semi_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!(sol.z.abs() < 1e-6, "z={}", sol.z);
        assert!(sol.x[0].abs() < 1e-6, "x={}", sol.x[0]);
    }

    #[test]
    fn solves_semi_integer_lot_ip() {
        let p = build_semi_integer_lot_ip();
        let linearized = linearize_semi_problem(&p);
        assert_eq!(linearized.c.len(), p.base.c.len() + 1);
        assert!(linearized.integer_vars[0]);
        assert!(linearized.integer_vars[1]);
        let sol = solve_semi_ipmip_with_des(p, IPMIPSolveOptions::default());
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 8.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 4.0).abs() < 1e-6, "lot={}", sol.x[0]);
    }

    #[test]
    fn solves_piecewise_linear_reward_ip() {
        let p = build_piecewise_linear_reward_ip();
        let linearized = linearize_pwl_problem(&p);
        assert_eq!(linearized.c.len(), p.base.c.len() + 7);
        assert_eq!(linearized.a.len(), p.base.a.len() + 11);
        let sol = solve_pwl_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 4.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "activity={}", sol.x[0]);
        assert!((sol.x[1] - 4.0).abs() < 1e-6, "reward={}", sol.x[1]);
    }

    #[test]
    fn solves_absolute_value_penalty_ip() {
        let source = build_absolute_value_penalty_ip();
        let p = AbsIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            abs: source.abs.clone(),
        };
        let (linearized, offset, original_vars) = linearize_abs_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len() + 1);
        assert_eq!(linearized.a.len(), p.base.a.len() + 4);
        assert!((offset - 0.0).abs() < 1e-9, "offset={offset}");
        let sol = solve_abs_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 2.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] + 2.0).abs() < 1e-6, "deviation={}", sol.x[0]);
        assert!((sol.x[1] - 2.0).abs() < 1e-6, "penalty={}", sol.x[1]);
    }

    #[test]
    fn solves_maximum_peak_ip() {
        let source = build_maximum_peak_ip();
        let p = MaximumIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            maximums: source.maximums.clone(),
        };
        let (linearized, offset, original_vars) = linearize_maximum_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(
            linearized.c.len(),
            p.base.c.len() + p.maximums[0].arg_vars.len() + 1
        );
        assert_eq!(linearized.a.len(), p.base.a.len() + 8);
        assert!((offset + 0.02).abs() < 1e-9, "offset={offset}");
        let sol = solve_maximum_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 1.99).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] + 1.0).abs() < 1e-6, "load_a={}", sol.x[0]);
        assert!((sol.x[1] - 2.0).abs() < 1e-6, "load_b={}", sol.x[1]);
        assert!((sol.x[2] - 2.0).abs() < 1e-6, "peak={}", sol.x[2]);
    }

    #[test]
    fn solves_minimum_floor_ip() {
        let source = build_minimum_floor_ip();
        let p = MinimumIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            minimums: source.minimums.clone(),
        };
        let (linearized, offset, original_vars) = linearize_minimum_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(
            linearized.c.len(),
            p.base.c.len() + p.minimums[0].arg_vars.len() + 1
        );
        assert_eq!(linearized.a.len(), p.base.a.len() + 8);
        assert!((offset - 0.02).abs() < 1e-9, "offset={offset}");
        let sol = solve_minimum_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 2.4675).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 3.0).abs() < 1e-6, "load_a={}", sol.x[0]);
        assert!((sol.x[1] - 2.5).abs() < 1e-6, "load_b={}", sol.x[1]);
        assert!((sol.x[2] - 2.5).abs() < 1e-6, "floor={}", sol.x[2]);
    }

    #[test]
    fn solves_logical_gate_ip() {
        let source = build_logical_gate_ip();
        let p = LogicalIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            logical: source.logical.clone(),
        };
        let (linearized, offset, original_vars) = linearize_logical_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len());
        assert_eq!(linearized.a.len(), p.base.a.len() + 6);
        assert!(offset.abs() < 1e-9, "offset={offset}");
        let sol = solve_logical_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 1.1).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "choice_a={}", sol.x[0]);
        assert!(sol.x[1].abs() < 1e-6, "choice_b={}", sol.x[1]);
        assert!(sol.x[2].abs() < 1e-6, "both={}", sol.x[2]);
        assert!((sol.x[3] - 1.0).abs() < 1e-6, "either={}", sol.x[3]);
    }

    #[test]
    fn solves_l1_norm_deviation_ip() {
        let source = build_l1_norm_deviation_ip();
        let p = L1NormIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            l1_norms: source.l1_norms.clone(),
        };
        let (linearized, offset, original_vars) = linearize_l1_norm_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len() + 4);
        assert_eq!(linearized.a.len(), p.base.a.len() + 10);
        assert!((offset - 0.05).abs() < 1e-9, "offset={offset}");
        let sol = solve_l1_norm_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 3.05).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "dev_a={}", sol.x[0]);
        assert!((sol.x[1] + 2.0).abs() < 1e-6, "dev_b={}", sol.x[1]);
        assert!((sol.x[2] - 3.0).abs() < 1e-6, "norm={}", sol.x[2]);
    }

    #[test]
    fn solves_linf_norm_deviation_ip() {
        let source = build_linf_norm_deviation_ip();
        let p = LInfNormIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            linf_norms: source.linf_norms.clone(),
        };
        let (linearized, offset, original_vars) = linearize_linf_norm_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len() + 6);
        assert_eq!(linearized.a.len(), p.base.a.len() + 14);
        assert!((offset - 0.05).abs() < 1e-9, "offset={offset}");
        let sol = solve_linf_norm_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 2.055).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.5).abs() < 1e-6, "dev_a={}", sol.x[0]);
        assert!((sol.x[1] + 2.0).abs() < 1e-6, "dev_b={}", sol.x[1]);
        assert!((sol.x[2] - 2.0).abs() < 1e-6, "radius={}", sol.x[2]);
    }

    #[test]
    fn solves_product_activation_ip() {
        let source = build_product_activation_ip();
        let p = ProductIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            products: source.products.clone(),
        };
        let (linearized, offset, original_vars) = linearize_product_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len());
        assert_eq!(linearized.a.len(), p.base.a.len() + 4);
        assert!((offset + 0.01).abs() < 1e-9, "offset={offset}");
        let sol = solve_product_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 3.86).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "use={}", sol.x[0]);
        assert!((sol.x[1] - 4.0).abs() < 1e-6, "activity={}", sol.x[1]);
        assert!((sol.x[2] - 4.0).abs() < 1e-6, "served={}", sol.x[2]);
    }

    #[test]
    fn solves_binary_product_gate_ip() {
        let source = build_binary_product_gate_ip();
        let p = ProductIPMIPProblem {
            base: source.base.clone(),
            lb: source.lb.clone(),
            products: source.products.clone(),
        };
        let (linearized, offset, original_vars) = linearize_product_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len());
        assert_eq!(linearized.a.len(), p.base.a.len() + 4);
        assert!(offset.abs() < 1e-9, "offset={offset}");
        let sol = solve_product_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 0.7).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "choose_a={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "choose_b={}", sol.x[1]);
        assert!((sol.x[2] - 1.0).abs() < 1e-6, "accepted={}", sol.x[2]);
    }

    #[test]
    fn solves_quadratic_objective_mix_ip() {
        let p = build_quadratic_objective_mix_ip();
        let (linearized, offset, original_vars) = linearize_quadratic_objective_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len() + 2);
        assert_eq!(linearized.a.len(), p.base.a.len() + 8);
        assert!((offset + 0.01).abs() < 1e-9, "offset={offset}");
        let sol = solve_quadratic_objective_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 6.46).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "use={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "premium={}", sol.x[1]);
        assert!((sol.x[2] - 4.0).abs() < 1e-6, "activity={}", sol.x[2]);
    }

    #[test]
    fn solves_source_feature_mix_ip() {
        let p = build_source_feature_mix_ip();
        let (linearized, offset, original_vars) = linearize_source_ipmip_problem(&p);
        assert_eq!(original_vars, p.base.c.len());
        assert_eq!(linearized.c.len(), p.base.c.len() + 7);
        assert!((offset - 0.0).abs() < 1e-9, "offset={offset}");
        let sol = solve_source_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert!((sol.z - 4.0).abs() < 1e-6, "z={}", sol.z);
        assert!((sol.x[0] - 2.0).abs() < 1e-6, "activity={}", sol.x[0]);
        assert!((sol.x[1] - 5.0).abs() < 1e-6, "reward={}", sol.x[1]);
        assert!((sol.x[2] - 1.0).abs() < 1e-6, "use={}", sol.x[2]);
    }

    #[test]
    fn solves_lexicographic_choice_ip() {
        let p = build_lexicographic_choice_ip();
        let sol = solve_multi_objective_ipmip_with_des(
            p,
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(sol.status, IPMIPStatus::Optimal);
        assert_eq!(sol.stage_solutions.len(), 2);
        assert_eq!(sol.objective_values, vec![1.0, 3.0]);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "choice_a={}", sol.x[0]);
        assert!(sol.x[1].abs() < 1e-6, "choice_b={}", sol.x[1]);
    }

    #[test]
    fn analyze_reports_binary() {
        let p = build_binary_knapsack_ip(vec![1.0, 2.0], vec![1.0, 1.0], 1.0);
        let f = analyze_ipmip_problem(&p);
        assert_eq!(f.variable_count, 2);
        assert_eq!(f.constraint_count, 1);
        assert!(f.all_binary);
        assert!(f.all_integer);
    }
}
