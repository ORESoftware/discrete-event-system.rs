//! Port of `src/des/general/stochastic-lp.ts` — module `des::general::stochastic_lp`.
//!
//! Two-stage stochastic linear programming expressed as a discrete-event SYSTEM
//! that blends the simulation half of DES (scenario sampling — Monte-Carlo over
//! omega) with the algorithmic half (piecewise-linear value-function
//! approximation via Benders / L-shaped cuts, executed on the same tick clock).
//! The master is a long-lived station owning an `IncrementalLP` instance; cuts
//! arrive as movables; one Benders iteration is one tick.
//!
//! The problem, in maximisation form, is `max c·x + E_w[Q(x, w)]` subject to
//! `A x <= b, x >= 0`, where the recourse function `Q(x, w) = max q·y` subject
//! to `T(w) x + W y <= h(w), y >= 0`. We provide three solvers:
//!
//!   1. [`SLPMonolithicSolver`] — Sample Average Approximation (SAA): sample N
//!      scenarios, build ONE giant LP, solve from scratch with
//!      [`solve_lp_internal`].
//!   2. [`solve_slp_benders`] — Benders / L-shaped decomposition as a DES. The
//!      master owns an [`IncrementalLP`] (warm-started); each Benders iteration
//!      adds exactly one optimality cut and is repaired by a few dual-simplex
//!      pivots.
//!   3. [`ProductionClosedFormSolver`] — analytical newsvendor-style oracle for
//!      the simple production-planning case (used for validation).
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * The various `interface`s become structs; the `BendersStation` (a
//!     `FixedPointIterationStation<BendersIterState>`) becomes a struct + the
//!     [`FixedPointIterationStation`] trait impl.
//!   * INJECTED RNG: the TS file re-declared its own `mulberry32`; here the
//!     single [`SeededRandom`] from `shared::capabilities` is used instead (its
//!     `next_float` reproduces the TS mulberry32 sequence bit-for-bit), so no
//!     second copy is ported.
//!   * The master LP is a long-lived [`IncrementalLP`] (warm-started) held as a
//!     struct field; cuts are added via `apply_add_constraint`.
//!   * Constraint matrices `number[][]` → `Vec<Vec<f64>>`; duals are `Vec<f64>`.
//!   * Infeasible / unbounded subproblem outcomes become a status enum, not a
//!     bare throw; the warm-start precondition (`rhs >= 0`) stays a `panic!`.
//!   * FLAGGED: the TS `externalReferenceValidator` imported from `./des-base`
//!     has NO counterpart in the ported `des_base::validation` (only
//!     `intrinsic_check` exists, mirroring the `value_iteration.rs` precedent
//!     that dropped it). It is reproduced here as a LOCAL closure-backed
//!     validator ([`external_reference_validator`]) reading a tiny JSON file —
//!     not a new shared API.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use crate::des::general::des_base::fixed_point::{
    ConvergenceReason, FixedPointCore, FixedPointIterationStation, FixedPointOptions,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::validation::{intrinsic_check, FnValidator, ValidationCheck};
use crate::des::general::incremental_lp::{
    IncrementalLP, IncrementalLPInit, Sense as IncSense, SolverStatus,
};
use crate::des::general::lp::{
    self, solve_lp_internal, InternalSimplexOptions, LPProblem, LPStatus,
};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// PROBLEM TYPES
// -----------------------------------------------------------------------------

/// A two-stage stochastic LP in maximisation form.
#[derive(Clone, Debug, Default)]
pub struct SLPProblem {
    /// First-stage objective coefficients (length `n_first`).
    pub c_first: Vec<f64>,
    /// First-stage constraint matrix `A` (`m_first × n_first`). May be empty.
    pub a_first: Vec<Vec<f64>>,
    /// First-stage RHS (length `m_first`).
    pub b_first: Vec<f64>,
    /// Second-stage objective coefficients (length `n_second`).
    pub q_second: Vec<f64>,
    /// Second-stage technology matrix `W` (`m_second × n_second`).
    pub w_second: Vec<Vec<f64>>,
    /// Lower bound on `Q(x, w)` — used to translate theta to a non-negative variable.
    pub theta_lower_bound: f64,
    /// Upper bound on `Q(x, w)` — keeps the master bounded before any cuts arrive.
    pub theta_upper_bound: f64,
    /// Optional names.
    pub var_names: Option<Vec<String>>,
}

/// Free-form scenario metadata (the TS `meta?: any`; the production builder
/// stores the sampled demand vector here).
#[derive(Clone, Debug, Default)]
pub struct ScenarioMeta {
    pub d: Vec<f64>,
}

/// One realisation of the random data `w`.
#[derive(Clone, Debug, Default)]
pub struct Scenario {
    /// Scenario-specific `T` matrix (`m_second × n_first`).
    pub t: Vec<Vec<f64>>,
    /// Scenario-specific RHS `h(w)` (length `m_second`).
    pub h: Vec<f64>,
    /// Probability mass (default `1/N`).
    pub prob: Option<f64>,
    pub meta: Option<ScenarioMeta>,
}

/// Stop reason recorded on the iteration that ended Benders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BendersStopReason {
    Converged,
    IterLimit,
    SubproblemError,
}

/// The optimality cut added to the master at the end of an iteration.
#[derive(Clone, Debug)]
pub struct CutAdded {
    pub coefs: Vec<f64>,
    pub rhs: f64,
    pub pi_h: f64,
    pub pi_t: Vec<f64>,
}

/// One Benders iteration's trace entry.
#[derive(Clone, Debug)]
pub struct BendersIteration {
    pub iter: usize,
    /// First-stage decision the master proposed this iteration.
    pub x_master: Vec<f64>,
    /// Master's belief about theta (an upper bound on `E[Q(x*, w)]`).
    pub theta_master: f64,
    /// Per-scenario subproblem objective `Q(x*, w_s)`.
    pub scenario_values: Vec<f64>,
    /// Per-scenario subproblem dual `pi_s*` (length `m_second`).
    pub scenario_duals: Vec<Vec<f64>>,
    /// Empirical `E[Q]` from this iteration's subproblem solves.
    pub expected_q: f64,
    /// Cut added to the master at the end of this iteration.
    pub cut_added: Option<CutAdded>,
    /// Master objective at `x*`: `c·x + theta` — a valid UPPER bound.
    pub upper_bound: f64,
    /// Feasible objective `c·x + E[Q]` — a valid LOWER bound.
    pub lower_bound: f64,
    /// `UB − LB`.
    pub gap: f64,
    /// Stop reason if this iteration ended Benders.
    pub stop_reason: Option<BendersStopReason>,
}

/// Solve status (TS `'optimal' | 'unbounded' | 'infeasible' | 'iter-limit'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SLPStatus {
    Optimal,
    Unbounded,
    Infeasible,
    IterLimit,
}

/// Solution method (TS `'monolithic' | 'benders' | 'closed-form'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SLPMethod {
    Monolithic,
    Benders,
    ClosedForm,
}

/// The result of solving a two-stage SLP.
#[derive(Clone, Debug)]
pub struct SLPSolveResult {
    pub status: SLPStatus,
    /// First-stage decision.
    pub x: Vec<f64>,
    /// Total objective: `c·x + E[Q]` (empirical, on these scenarios).
    pub objective: f64,
    pub c_first_x: f64,
    pub expected_q: f64,
    /// Per-scenario second-stage decisions.
    pub y_by_scenario: Vec<Vec<f64>>,
    /// Per-scenario `Q` values.
    pub scenario_values: Vec<f64>,
    /// Pivots / iterations.
    pub iterations: usize,
    /// Solution method.
    pub method: SLPMethod,
    /// Per-iteration trace (Benders only).
    pub benders_trace: Option<Vec<BendersIteration>>,
    /// Wall-clock ms.
    pub elapsed_ms: f64,
}

// -----------------------------------------------------------------------------
// SUBPROBLEM SOLVER
// -----------------------------------------------------------------------------

/// A recourse subproblem `max q·y s.t. W y <= rhs, y >= 0`.
#[derive(Clone, Debug)]
pub struct SubproblemDualsInput {
    pub q: Vec<f64>,
    pub w: Vec<Vec<f64>>,
    pub rhs: Vec<f64>,
}

/// Status of a recourse subproblem solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubproblemStatus {
    Optimal,
    Unbounded,
    Infeasible,
}

/// Result of solving a recourse subproblem, with duals recovered from the slack
/// reduced costs at the optimum.
#[derive(Clone, Debug)]
pub struct SubproblemDualsResult {
    pub status: SubproblemStatus,
    pub y: Vec<f64>,
    pub obj: f64,
    pub duals: Vec<f64>,
}

/// Solve a recourse subproblem and recover its dual prices from the slack
/// reduced costs at the optimum.
pub struct SubproblemWithDualsSolver;

impl Transform<SubproblemDualsInput, SubproblemDualsResult> for SubproblemWithDualsSolver {
    fn transform(&self, input: SubproblemDualsInput) -> SubproblemDualsResult {
        let SubproblemDualsInput { q, w, rhs } = input;
        // Validate the warm-start precondition.
        for (i, &ri) in rhs.iter().enumerate() {
            if ri < -1e-9 {
                // Negative RHS would require Phase-1 simplex in IncrementalLP.
                panic!(
                    "solveSubproblemWithDuals: rhs[{i}] = {ri} < 0; would require Phase-1 simplex"
                );
            }
        }
        let mut lp = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: q.clone(),
            a: w.clone(),
            b: rhs.clone(),
            var_names: None,
            con_names: None,
        });
        lp.solve_to_optimum(1000);
        if lp.status == SolverStatus::Unbounded {
            return SubproblemDualsResult {
                status: SubproblemStatus::Unbounded,
                y: vec![],
                obj: f64::NAN,
                duals: vec![],
            };
        }
        if lp.status == SolverStatus::Infeasible {
            return SubproblemDualsResult {
                status: SubproblemStatus::Infeasible,
                y: vec![],
                obj: f64::NAN,
                duals: vec![],
            };
        }
        let y = lp.get_x();
        let obj = lp.get_z();
        // Duals = reduced costs of slack columns at the optimum.
        let rc = lp.get_reduced_costs();
        let duals = rc[q.len()..q.len() + w.len()].to_vec();
        SubproblemDualsResult {
            status: SubproblemStatus::Optimal,
            y,
            obj,
            duals,
        }
    }
}

/// Deprecated shim: prefer `SubproblemWithDualsSolver.transform({q, w, rhs})`.
pub fn solve_subproblem_with_duals(
    q: &[f64],
    w: &[Vec<f64>],
    rhs: &[f64],
) -> SubproblemDualsResult {
    SubproblemWithDualsSolver.transform(SubproblemDualsInput {
        q: q.to_vec(),
        w: w.to_vec(),
        rhs: rhs.to_vec(),
    })
}

// -----------------------------------------------------------------------------
// METHOD 1: Sample Average Approximation — monolithic LP via solve_lp_internal
// -----------------------------------------------------------------------------

/// Convert an [`LPStatus`] into the SLP-level [`SLPStatus`] union. The TS source
/// cast the raw string through `as any`; `NumericalError` (not in the SLP union)
/// maps to `IterLimit` (a non-infeasible "did not solve" outcome).
fn slp_status_from_lp(s: LPStatus) -> SLPStatus {
    match s {
        LPStatus::Optimal => SLPStatus::Optimal,
        LPStatus::Infeasible => SLPStatus::Infeasible,
        LPStatus::Unbounded => SLPStatus::Unbounded,
        LPStatus::IterLimit | LPStatus::NumericalError => SLPStatus::IterLimit,
    }
}

/// Sample Average Approximation: build and solve ONE monolithic LP over all
/// scenarios. The problem is configuration; the scenario set is the input.
pub struct SLPMonolithicSolver {
    p: SLPProblem,
}

impl SLPMonolithicSolver {
    pub fn new(p: SLPProblem) -> Self {
        SLPMonolithicSolver { p }
    }
}

impl Transform<Vec<Scenario>, SLPSolveResult> for SLPMonolithicSolver {
    fn transform(&self, scenarios: Vec<Scenario>) -> SLPSolveResult {
        let p = &self.p;
        let t0 = Instant::now();
        let n = scenarios.len();
        let n_first = p.c_first.len();
        let n_second = p.q_second.len();
        let m_first = p.a_first.len();
        let m_second = p.w_second.len();
        let total_vars = n_first + n * n_second;

        let mut c = vec![0.0; total_vars];
        for j in 0..n_first {
            c[j] = p.c_first[j];
        }
        for s in 0..n {
            let w = scenarios[s].prob.unwrap_or(1.0 / n as f64);
            for j in 0..n_second {
                c[n_first + s * n_second + j] = w * p.q_second[j];
            }
        }
        let mut a_ub: Vec<Vec<f64>> = Vec::new();
        let mut b_ub: Vec<f64> = Vec::new();
        // First-stage constraints.
        for i in 0..m_first {
            let mut row = vec![0.0; total_vars];
            for j in 0..n_first {
                row[j] = p.a_first[i][j];
            }
            a_ub.push(row);
            b_ub.push(p.b_first[i]);
        }
        // Second-stage constraints, scenario-by-scenario.
        for s in 0..n {
            for i in 0..m_second {
                let mut row = vec![0.0; total_vars];
                for j in 0..n_first {
                    row[j] = scenarios[s].t[i][j];
                }
                for j in 0..n_second {
                    row[n_first + s * n_second + j] = p.w_second[i][j];
                }
                a_ub.push(row);
                b_ub.push(scenarios[s].h[i]);
            }
        }
        let lp = LPProblem {
            sense: lp::Sense::Max,
            c,
            a_ub: Some(a_ub),
            b_ub: Some(b_ub),
            ..Default::default()
        };
        let sol = solve_lp_internal(
            &lp,
            &InternalSimplexOptions {
                max_iter: Some(50000),
                tol: None,
            },
        );

        let x = sol
            .x
            .get(0..n_first)
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let mut y_by_scenario: Vec<Vec<f64>> = Vec::new();
        let mut scenario_values: Vec<f64> = Vec::new();
        for s in 0..n {
            let lo = n_first + s * n_second;
            let hi = n_first + (s + 1) * n_second;
            let y_s = sol
                .x
                .get(lo..hi)
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0; n_second]);
            let mut qy = 0.0;
            for j in 0..n_second {
                qy += p.q_second[j] * y_s[j];
            }
            y_by_scenario.push(y_s);
            scenario_values.push(qy);
        }
        let mut c_first_x = 0.0;
        for j in 0..n_first.min(x.len()) {
            c_first_x += p.c_first[j] * x[j];
        }
        let mut expected_q = 0.0;
        for s in 0..n {
            expected_q += scenarios[s].prob.unwrap_or(1.0 / n as f64) * scenario_values[s];
        }

        SLPSolveResult {
            status: slp_status_from_lp(sol.status),
            x,
            objective: sol.objective,
            c_first_x,
            expected_q,
            y_by_scenario,
            scenario_values,
            iterations: sol.iters.unwrap_or(0),
            method: SLPMethod::Monolithic,
            benders_trace: None,
            elapsed_ms: elapsed_ms(t0),
        }
    }
}

/// Deprecated shim: prefer `SLPMonolithicSolver::new(p).transform(scenarios)`.
pub fn solve_slp_monolithic(p: SLPProblem, scenarios: Vec<Scenario>) -> SLPSolveResult {
    SLPMonolithicSolver::new(p).transform(scenarios)
}

// -----------------------------------------------------------------------------
// METHOD 2: Benders Decomposition (L-shaped) AS A DES
// -----------------------------------------------------------------------------

/// Options for [`solve_slp_benders`]. `None` fields take the TS defaults.
#[derive(Clone, Debug, Default)]
pub struct BendersOpts {
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub verbose: Option<bool>,
    /// Optional path to a JSON shaped like `{ x: number[], objective: number }`.
    /// When set, the Benders station auto-attaches an external-reference
    /// validator that compares the first-stage solution element-wise.
    pub reference_path: Option<String>,
    pub reference_tol: Option<f64>,
    pub silent_if_missing: Option<bool>,
}

/// Resolved Benders options (`Required<BendersOpts>`).
#[derive(Clone, Debug)]
struct FilledBendersOpts {
    max_iter: usize,
    tol: f64,
    verbose: bool,
    reference_path: String,
    reference_tol: f64,
    silent_if_missing: bool,
}

/// Persistent status of the Benders loop (TS `BendersIterState.status`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BendersRunStatus {
    Running,
    Optimal,
    Infeasible,
    Unbounded,
    SubproblemError,
}

/// The fixed-point iteration state carried by [`BendersStation`].
#[derive(Clone, Debug)]
struct BendersIterState {
    iter: usize,
    gap: f64,
    upper_bound: f64,
    lower_bound: f64,
    status: BendersRunStatus,
}

/// Final status reported after the loop (TS `finalStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BendersFinalStatus {
    Optimal,
    IterLimit,
    Infeasible,
    Unbounded,
    SubproblemError,
}

/// Concrete leaf of [`FixedPointIterationStation`] running one Benders iteration
/// per tick.
pub struct BendersStation {
    core: StationCore,
    fp: FixedPointCore<BendersIterState>,
    p: SLPProblem,
    scenarios: Vec<Scenario>,
    n: usize,
    n_first: usize,
    m_second: usize,
    verbose: bool,
    master: IncrementalLP,
    pub trace: Vec<BendersIteration>,
    pub best_lower_bound: f64,
    pub best_x: Vec<f64>,
    pub best_y: Vec<Vec<f64>>,
    pub best_scenario_values: Vec<f64>,
    pub pivots_total: usize,
    pub last_scenario_values: Vec<f64>,
    pub last_scenario_y: Vec<Vec<f64>>,
    pub last_x_master: Vec<f64>,
    pub final_status: BendersFinalStatus,
}

fn downcast_benders(s: &dyn DESStation) -> &BendersStation {
    s.as_any()
        .downcast_ref::<BendersStation>()
        .expect("validator received a non-BendersStation station")
}

impl BendersStation {
    fn new(p: SLPProblem, scenarios: Vec<Scenario>, opts: &FilledBendersOpts) -> Self {
        let n = scenarios.len();
        let n_first = p.c_first.len();
        let m_second = p.w_second.len();

        // Build the master LP with variables [x_1..x_n, theta_var].
        let mut master_c = p.c_first.clone();
        master_c.push(1.0);
        let mut master_a: Vec<Vec<f64>> = Vec::new();
        let mut master_b: Vec<f64> = Vec::new();
        for i in 0..p.a_first.len() {
            let mut row = p.a_first[i].clone();
            row.push(0.0);
            master_a.push(row);
            master_b.push(p.b_first[i]);
        }
        let theta_span = p.theta_upper_bound - p.theta_lower_bound;
        if theta_span <= 0.0 {
            panic!("thetaUpperBound must exceed thetaLowerBound");
        }
        let mut theta_row = vec![0.0; n_first];
        theta_row.push(1.0);
        master_a.push(theta_row);
        master_b.push(theta_span);

        let mut var_names: Vec<String> = match &p.var_names {
            Some(v) => v.clone(),
            None => (0..n_first).map(|i| format!("x{}", i + 1)).collect(),
        };
        var_names.push("theta".to_string());

        let master = IncrementalLP::new(IncrementalLPInit {
            sense: IncSense::Max,
            c: master_c,
            a: master_a,
            b: master_b,
            var_names: Some(var_names),
            con_names: None,
        });

        let mut station = BendersStation {
            core: StationCore::new("benders"),
            fp: FixedPointCore::new(FixedPointOptions {
                tol: Some(opts.tol),
                max_iter: Some(opts.max_iter),
                max_history_len: None,
            }),
            p,
            scenarios,
            n,
            n_first,
            m_second,
            verbose: opts.verbose,
            master,
            trace: Vec::new(),
            best_lower_bound: f64::NEG_INFINITY,
            best_x: vec![0.0; n_first],
            best_y: Vec::new(),
            best_scenario_values: Vec::new(),
            pivots_total: 0,
            last_scenario_values: Vec::new(),
            last_scenario_y: Vec::new(),
            last_x_master: vec![0.0; n_first],
            final_status: BendersFinalStatus::IterLimit,
        };
        station.bootstrap();

        // Intrinsic invariants for any Benders run.
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "benders.optimal-implies-gap-le-tol",
                |s: &dyn DESStation| {
                    let st = downcast_benders(s);
                    st.final_status != BendersFinalStatus::Optimal
                        || st.current().gap <= st.fp.tol + 1e-9
                },
                Some("gap ≤ tol when optimal".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_benders(s);
                    format!(
                        "status={:?}  gap={}  tol={}",
                        st.final_status,
                        st.current().gap,
                        st.fp.tol
                    )
                })),
                Some("benders-intrinsic".to_string()),
                Some("optimality declared but UB − LB exceeds tol".to_string()),
            )
            .boxed(),
        );
        station.add_validator(
            intrinsic_check::<dyn DESStation>(
                "benders.lower-bound-le-upper-bound",
                |s: &dyn DESStation| {
                    let st = downcast_benders(s);
                    let cur = st.current();
                    if !cur.upper_bound.is_finite() || !cur.lower_bound.is_finite() {
                        return true;
                    }
                    cur.lower_bound <= cur.upper_bound + 1e-6
                },
                Some("LB ≤ UB".to_string()),
                Some(Box::new(|s: &dyn DESStation| {
                    let st = downcast_benders(s);
                    format!(
                        "LB={}  UB={}",
                        st.current().lower_bound,
                        st.current().upper_bound
                    )
                })),
                Some("benders-intrinsic".to_string()),
                Some("lower bound exceeds upper bound — would indicate a duality bug".to_string()),
            )
            .boxed(),
        );

        // Optional external-reference validator (e.g. scipy extensive-form LP).
        if !opts.reference_path.is_empty() {
            let ref_tol = opts.reference_tol;
            station.add_validator(
                external_reference_validator(
                    "benders.solution-vs-reference",
                    "benders-external",
                    &opts.reference_path,
                    opts.silent_if_missing,
                    move |s: &dyn DESStation, reference: &ReferenceSolution| {
                        let st = downcast_benders(s);
                        let x = &st.last_x_master;
                        let mut out: Vec<ValidationCheck> = Vec::new();
                        if let Some(ref_x) = &reference.x {
                            if ref_x.len() == x.len() {
                                let mut max_abs = 0.0;
                                let mut argmax: isize = -1;
                                for i in 0..x.len() {
                                    let e = (x[i] - ref_x[i]).abs();
                                    if e > max_abs {
                                        max_abs = e;
                                        argmax = i as isize;
                                    }
                                }
                                let passed = max_abs <= ref_tol;
                                out.push(ValidationCheck {
                                    name: "benders.x-vs-reference".to_string(),
                                    passed,
                                    observed: Some(format!("max|Δx|={max_abs:.3e} at i={argmax}")),
                                    expected: Some(format!("≤ {ref_tol}")),
                                    group: Some("benders-external".to_string()),
                                    details: if passed {
                                        None
                                    } else {
                                        let i = argmax as usize;
                                        Some(format!("x[{argmax}]={}  ref={}", x[i], ref_x[i]))
                                    },
                                });
                            }
                        }
                        if let Some(ref_obj) = reference.objective {
                            if ref_obj.is_finite() {
                                let cur = st.current();
                                let obj = if cur.lower_bound.is_finite() {
                                    cur.lower_bound
                                } else {
                                    f64::NAN
                                };
                                let e = (obj - ref_obj).abs() / ref_obj.abs().max(1e-12);
                                let passed = e <= ref_tol;
                                out.push(ValidationCheck {
                                    name: "benders.objective-vs-reference".to_string(),
                                    passed,
                                    observed: Some(format!("{obj:.8e}")),
                                    expected: Some(format!("{ref_obj:.8e}")),
                                    group: Some("benders-external".to_string()),
                                    details: if passed {
                                        None
                                    } else {
                                        Some(format!("rel-err={e:.3e} > {ref_tol}"))
                                    },
                                });
                            }
                        }
                        out
                    },
                )
                .boxed(),
            );
        }

        station
    }
}

impl DESStation for BendersStation {
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
        self.fixed_point_has_work()
    }
    fn run_time_step(&mut self) {
        self.fixed_point_run_time_step();
    }
}

impl FixedPointIterationStation<BendersIterState> for BendersStation {
    fn fp_core(&self) -> &FixedPointCore<BendersIterState> {
        &self.fp
    }
    fn fp_core_mut(&mut self) -> &mut FixedPointCore<BendersIterState> {
        &mut self.fp
    }

    fn initial_state(&self) -> BendersIterState {
        BendersIterState {
            iter: 0,
            gap: f64::INFINITY,
            upper_bound: f64::INFINITY,
            lower_bound: f64::NEG_INFINITY,
            status: BendersRunStatus::Running,
        }
    }

    fn apply_operator(&mut self, prev: &BendersIterState) -> BendersIterState {
        let iter = prev.iter + 1;
        // 1. Master solve.
        let pivots_before = self.master.tick;
        self.master.solve_to_optimum(1000);
        self.pivots_total += self.master.tick - pivots_before;
        if self.master.status != SolverStatus::Optimal {
            let (run, fin) = match self.master.status {
                SolverStatus::Unbounded => {
                    (BendersRunStatus::Unbounded, BendersFinalStatus::Unbounded)
                }
                _ => (BendersRunStatus::Infeasible, BendersFinalStatus::Infeasible),
            };
            self.final_status = fin;
            return BendersIterState {
                iter,
                gap: 0.0,
                upper_bound: f64::NAN,
                lower_bound: self.best_lower_bound,
                status: run,
            };
        }
        let master_x = self.master.get_x();
        let x_master: Vec<f64> = master_x[0..self.n_first].to_vec();
        let theta_var_master = master_x[self.n_first];
        let theta_master = theta_var_master + self.p.theta_lower_bound;
        let mut c_tx = 0.0;
        for (i, &ci) in self.p.c_first.iter().enumerate() {
            c_tx += ci * x_master[i];
        }
        self.last_x_master = x_master.clone();

        // 2. Scenario subproblems.
        let mut scenario_values: Vec<f64> = Vec::new();
        let mut scenario_duals: Vec<Vec<f64>> = Vec::new();
        let mut scenario_y: Vec<Vec<f64>> = Vec::new();
        for s in 0..self.n {
            let sc = &self.scenarios[s];
            let rhs: Vec<f64> =
                sc.h.iter()
                    .enumerate()
                    .map(|(i, &hi)| {
                        let mut v = hi;
                        for j in 0..self.n_first {
                            v -= sc.t[i][j] * x_master[j];
                        }
                        v
                    })
                    .collect();
            let sub = solve_subproblem_with_duals(&self.p.q_second, &self.p.w_second, &rhs);
            if sub.status != SubproblemStatus::Optimal {
                self.final_status = BendersFinalStatus::SubproblemError;
                let stop = BendersIteration {
                    iter,
                    x_master: x_master.clone(),
                    theta_master,
                    scenario_values: scenario_values.clone(),
                    scenario_duals: scenario_duals.clone(),
                    expected_q: f64::NAN,
                    cut_added: None,
                    upper_bound: c_tx + theta_master,
                    lower_bound: self.best_lower_bound,
                    gap: c_tx + theta_master - self.best_lower_bound,
                    stop_reason: Some(BendersStopReason::SubproblemError),
                };
                self.trace.push(stop);
                return BendersIterState {
                    iter,
                    gap: 0.0,
                    upper_bound: f64::NAN,
                    lower_bound: f64::NAN,
                    status: BendersRunStatus::SubproblemError,
                };
            }
            scenario_values.push(sub.obj);
            scenario_duals.push(sub.duals);
            scenario_y.push(sub.y);
        }
        let mut expected_q = 0.0;
        for s in 0..self.n {
            expected_q +=
                self.scenarios[s].prob.unwrap_or(1.0 / self.n as f64) * scenario_values[s];
        }
        let upper_bound = c_tx + theta_master;
        let lower_bound = c_tx + expected_q;
        if lower_bound > self.best_lower_bound {
            self.best_lower_bound = lower_bound;
            self.best_x = x_master.clone();
            self.best_y = scenario_y.iter().cloned().collect();
            self.best_scenario_values = scenario_values.clone();
        }
        self.last_scenario_values = scenario_values.clone();
        self.last_scenario_y = scenario_y.clone();
        let gap = upper_bound - lower_bound;
        if self.verbose {
            eprintln!(
                "[Benders] iter={iter}  x=[{}]  θ={theta_master:.3}  E[Q]={expected_q:.3}  UB={upper_bound:.3}  LB={lower_bound:.3}  gap={gap:.2e}",
                x_master.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(",")
            );
        }

        // 3. Convergence?
        if gap <= self.fp.tol {
            self.final_status = BendersFinalStatus::Optimal;
            self.trace.push(BendersIteration {
                iter,
                x_master,
                theta_master,
                scenario_values,
                scenario_duals,
                expected_q,
                cut_added: None,
                upper_bound,
                lower_bound,
                gap,
                stop_reason: Some(BendersStopReason::Converged),
            });
            return BendersIterState {
                iter,
                gap,
                upper_bound,
                lower_bound,
                status: BendersRunStatus::Optimal,
            };
        }

        // 4. Build the optimality cut and add it to the master.
        let mut pi_h_avg = 0.0;
        let mut pi_t_avg = vec![0.0; self.n_first];
        for s in 0..self.n {
            let w = self.scenarios[s].prob.unwrap_or(1.0 / self.n as f64);
            let pi = &scenario_duals[s];
            let sc = &self.scenarios[s];
            for i in 0..self.m_second {
                pi_h_avg += w * pi[i] * sc.h[i];
            }
            for j in 0..self.n_first {
                let mut s_val = 0.0;
                for i in 0..self.m_second {
                    s_val += pi[i] * sc.t[i][j];
                }
                pi_t_avg[j] += w * s_val;
            }
        }
        let mut cut_coefs = pi_t_avg.clone();
        cut_coefs.push(1.0);
        let cut_rhs = pi_h_avg - self.p.theta_lower_bound;
        self.master
            .apply_add_constraint(&cut_coefs, cut_rhs, Some(format!("cut{iter}")));
        self.trace.push(BendersIteration {
            iter,
            x_master,
            theta_master,
            scenario_values,
            scenario_duals,
            expected_q,
            cut_added: Some(CutAdded {
                coefs: cut_coefs,
                rhs: cut_rhs,
                pi_h: pi_h_avg,
                pi_t: pi_t_avg,
            }),
            upper_bound,
            lower_bound,
            gap,
            stop_reason: None,
        });
        BendersIterState {
            iter,
            gap,
            upper_bound,
            lower_bound,
            status: BendersRunStatus::Running,
        }
    }

    fn delta(&self, _prev: &BendersIterState, next: &BendersIterState) -> f64 {
        next.gap
    }

    fn should_stop(&mut self, iter: usize, last_delta: f64) -> bool {
        let st = self.current().status;
        if st != BendersRunStatus::Running && iter > 0 {
            self.fp_core_mut().convergence_reason = if st == BendersRunStatus::Optimal {
                ConvergenceReason::Converged
            } else {
                ConvergenceReason::MaxIter
            };
            return true;
        }
        // super.shouldStop (the default FixedPointIterationStation behaviour).
        if iter >= self.fp_core().max_iter {
            self.fp_core_mut().convergence_reason = ConvergenceReason::MaxIter;
            return true;
        }
        if iter > 0 && last_delta < self.fp_core().tol {
            self.fp_core_mut().convergence_reason = ConvergenceReason::Converged;
            return true;
        }
        false
    }
}

/// Benders / L-shaped decomposition solve of a two-stage SLP.
pub fn solve_slp_benders(
    p: SLPProblem,
    scenarios: Vec<Scenario>,
    opts: BendersOpts,
) -> SLPSolveResult {
    let t0 = Instant::now();
    let filled = FilledBendersOpts {
        max_iter: opts.max_iter.unwrap_or(100),
        tol: opts.tol.unwrap_or(1e-6),
        verbose: opts.verbose.unwrap_or(false),
        reference_path: opts.reference_path.unwrap_or_default(),
        reference_tol: opts.reference_tol.unwrap_or(1e-3),
        silent_if_missing: opts.silent_if_missing.unwrap_or(true),
    };

    let c_first = p.c_first.clone();
    let station = Rc::new(RefCell::new(BendersStation::new(p, scenarios, &filled)));
    run_iterative_des(
        vec![station.clone() as Rc<RefCell<dyn DESStation>>],
        IterativeRunOptions::default(),
    );

    let st = station.borrow();
    let final_state = st.current().clone();
    let status = st.final_status;
    let dot = |x: &[f64]| -> f64 { c_first.iter().enumerate().map(|(i, ci)| ci * x[i]).sum() };

    match status {
        BendersFinalStatus::Optimal => {
            let c_tx = dot(&st.last_x_master);
            SLPSolveResult {
                status: SLPStatus::Optimal,
                x: st.last_x_master.clone(),
                objective: final_state.lower_bound,
                c_first_x: c_tx,
                expected_q: final_state.lower_bound - c_tx,
                y_by_scenario: st.last_scenario_y.iter().cloned().collect(),
                scenario_values: st.last_scenario_values.clone(),
                iterations: final_state.iter,
                method: SLPMethod::Benders,
                benders_trace: Some(st.trace.clone()),
                elapsed_ms: elapsed_ms(t0),
            }
        }
        BendersFinalStatus::Infeasible | BendersFinalStatus::Unbounded => SLPSolveResult {
            status: if status == BendersFinalStatus::Infeasible {
                SLPStatus::Infeasible
            } else {
                SLPStatus::Unbounded
            },
            x: st.best_x.clone(),
            objective: f64::NAN,
            c_first_x: f64::NAN,
            expected_q: f64::NAN,
            y_by_scenario: vec![],
            scenario_values: vec![],
            iterations: final_state.iter,
            method: SLPMethod::Benders,
            benders_trace: Some(st.trace.clone()),
            elapsed_ms: elapsed_ms(t0),
        },
        BendersFinalStatus::SubproblemError => SLPSolveResult {
            status: SLPStatus::Infeasible,
            x: st.best_x.clone(),
            objective: f64::NAN,
            c_first_x: f64::NAN,
            expected_q: f64::NAN,
            y_by_scenario: st.best_y.clone(),
            scenario_values: st.best_scenario_values.clone(),
            iterations: final_state.iter,
            method: SLPMethod::Benders,
            benders_trace: Some(st.trace.clone()),
            elapsed_ms: elapsed_ms(t0),
        },
        BendersFinalStatus::IterLimit => {
            let c_tx = dot(&st.best_x);
            SLPSolveResult {
                status: SLPStatus::IterLimit,
                x: st.best_x.clone(),
                objective: st.best_lower_bound,
                c_first_x: c_tx,
                expected_q: st.best_lower_bound - c_tx,
                y_by_scenario: st.best_y.clone(),
                scenario_values: st.best_scenario_values.clone(),
                iterations: filled.max_iter,
                method: SLPMethod::Benders,
                benders_trace: Some(st.trace.clone()),
                elapsed_ms: elapsed_ms(t0),
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SCENARIO SAMPLING UTILITIES
// -----------------------------------------------------------------------------

/// Per-product uniform demand ranges plus a random seed.
#[derive(Clone, Debug)]
pub struct UniformDemandSpec {
    /// Range `[a_i, b_i]` per second-stage product.
    pub ranges: Vec<(f64, f64)>,
    /// Random seed.
    pub seed: u32,
}

/// Sample N uniform-demand scenarios for the production-planning template:
/// `W = [I; I]`, `T = [-I; 0]`, `h = [0...0, D_1, ..., D_n]`. The scenario count
/// `N` is configuration; the demand spec is the `transform` input.
pub struct ProductionScenarioBuilder {
    n: usize,
}

impl ProductionScenarioBuilder {
    pub fn new(n: usize) -> Self {
        ProductionScenarioBuilder { n }
    }
}

impl Transform<UniformDemandSpec, Vec<Scenario>> for ProductionScenarioBuilder {
    fn transform(&self, spec: UniformDemandSpec) -> Vec<Scenario> {
        let big_n = self.n;
        // FLAGGED: the TS re-declared `mulberry32`; the shared `SeededRandom`
        // reproduces that exact sequence, so we use it directly here.
        let mut r = SeededRandom::new(spec.seed);
        let n = spec.ranges.len();
        let mut scenarios: Vec<Scenario> = Vec::new();
        for _s in 0..big_n {
            let mut d = vec![0.0; n];
            for i in 0..n {
                let (a, b) = spec.ranges[i];
                d[i] = a + r.next_float() * (b - a);
            }
            // T = [-I; 0]  (capacity rows put -x; demand rows are scenario-independent of x).
            let mut t: Vec<Vec<f64>> = Vec::new();
            for i in 0..n {
                let mut row = vec![0.0; n];
                row[i] = -1.0;
                t.push(row);
            }
            for _i in 0..n {
                t.push(vec![0.0; n]);
            }
            // h = [0...0, D_1, ..., D_n].
            let mut h = vec![0.0; n];
            h.extend_from_slice(&d);
            scenarios.push(Scenario {
                t,
                h,
                prob: Some(1.0 / big_n as f64),
                meta: Some(ScenarioMeta { d }),
            });
        }
        scenarios
    }
}

/// Deprecated shim: prefer `ProductionScenarioBuilder::new(N).transform(spec)`.
pub fn build_production_scenarios(spec: UniformDemandSpec, n: usize) -> Vec<Scenario> {
    ProductionScenarioBuilder::new(n).transform(spec)
}

/// Cost/revenue (and optional budget) for the production-planning template.
#[derive(Clone, Debug)]
pub struct ProductionSLPInput {
    pub c: Vec<f64>,
    pub p: Vec<f64>,
    pub budget: Option<f64>,
}

/// Build the production-planning [`SLPProblem`] template with cost `c` and
/// revenue `p`.
pub struct ProductionSLPBuilder;

impl Transform<ProductionSLPInput, SLPProblem> for ProductionSLPBuilder {
    fn transform(&self, input: ProductionSLPInput) -> SLPProblem {
        let ProductionSLPInput { c, p, budget } = input;
        let n = c.len();
        if p.len() != n {
            panic!("cost and revenue must have same length");
        }
        // First-stage objective is -c · x (cost minimisation in max-form).
        let c_first: Vec<f64> = c.iter().map(|ci| -ci).collect();
        // First-stage constraints: optional budget x_1 + ... + x_n ≤ B.
        let mut a_first: Vec<Vec<f64>> = Vec::new();
        let mut b_first: Vec<f64> = Vec::new();
        if let Some(b) = budget {
            a_first.push(vec![1.0; n]);
            b_first.push(b);
        }
        // Second stage: max p · y, with W = [I; I].
        let mut w: Vec<Vec<f64>> = Vec::new();
        for i in 0..n {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            w.push(row);
        }
        for i in 0..n {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            w.push(row);
        }
        let theta_lb = 0.0;
        let theta_ub: f64 = p.iter().map(|pi| pi * 10000.0).sum();
        SLPProblem {
            c_first,
            a_first,
            b_first,
            q_second: p,
            w_second: w,
            theta_lower_bound: theta_lb,
            theta_upper_bound: theta_ub,
            var_names: Some((0..n).map(|i| format!("x{}", i + 1)).collect()),
        }
    }
}

/// Deprecated shim: prefer `ProductionSLPBuilder.transform({c, p, budget})`.
pub fn build_production_slp(c: Vec<f64>, p: Vec<f64>, budget: Option<f64>) -> SLPProblem {
    ProductionSLPBuilder.transform(ProductionSLPInput { c, p, budget })
}

// -----------------------------------------------------------------------------
// METHOD 3: Closed-form newsvendor solution (validation oracle)
// -----------------------------------------------------------------------------

/// Cost, revenue, and per-product uniform demand ranges for the closed-form oracle.
#[derive(Clone, Debug)]
pub struct ProductionClosedFormInput {
    pub c: Vec<f64>,
    pub p: Vec<f64>,
    pub ranges: Vec<(f64, f64)>,
}

/// Analytical newsvendor-style oracle for the production-planning case.
pub struct ProductionClosedFormSolver;

impl Transform<ProductionClosedFormInput, SLPSolveResult> for ProductionClosedFormSolver {
    fn transform(&self, input: ProductionClosedFormInput) -> SLPSolveResult {
        let ProductionClosedFormInput { c, p, ranges } = input;
        let t0 = Instant::now();
        let n = c.len();
        let mut x = vec![0.0; n];
        let mut z_val = 0.0;
        for i in 0..n {
            let (a, b) = ranges[i];
            if p[i] <= c[i] {
                x[i] = 0.0; // not profitable
            } else {
                let xi = a + (b - a) * (p[i] - c[i]) / p[i];
                x[i] = 0.0_f64.max(xi.min(b));
            }
            // E[min(x_i, D_i)] for uniform [a, b].
            let e_min = if x[i] <= a {
                x[i]
            } else if x[i] >= b {
                (a + b) / 2.0
            } else {
                x[i] - (x[i] - a) * (x[i] - a) / (2.0 * (b - a))
            };
            z_val += -c[i] * x[i] + p[i] * e_min;
        }
        let mut c_first_x = 0.0;
        for i in 0..n {
            c_first_x += -c[i] * x[i];
        }
        SLPSolveResult {
            status: SLPStatus::Optimal,
            x,
            objective: z_val,
            c_first_x,
            expected_q: z_val - c_first_x,
            y_by_scenario: vec![],
            scenario_values: vec![],
            iterations: 0,
            method: SLPMethod::ClosedForm,
            benders_trace: None,
            elapsed_ms: elapsed_ms(t0),
        }
    }
}

/// Deprecated shim: prefer `ProductionClosedFormSolver.transform({c, p, ranges})`.
pub fn solve_production_closed_form(
    c: Vec<f64>,
    p: Vec<f64>,
    ranges: Vec<(f64, f64)>,
) -> SLPSolveResult {
    ProductionClosedFormSolver.transform(ProductionClosedFormInput { c, p, ranges })
}

// -----------------------------------------------------------------------------
// LOCAL external-reference validator (no `des_base` counterpart — see flag above)
// -----------------------------------------------------------------------------

/// The reference solution shape `{ x: number[], objective: number }`.
#[derive(Clone, Debug, Default)]
pub struct ReferenceSolution {
    pub x: Option<Vec<f64>>,
    pub objective: Option<f64>,
}

/// Build a closure-backed validator that loads a JSON reference file and defers
/// to `compare`. When the file is missing it is silent (no checks) if
/// `silent_if_missing`, else emits one failed check.
fn external_reference_validator(
    name: &str,
    group: &str,
    reference_path: &str,
    silent_if_missing: bool,
    compare: impl Fn(&dyn DESStation, &ReferenceSolution) -> Vec<ValidationCheck> + 'static,
) -> FnValidator<dyn DESStation> {
    let name = name.to_string();
    let group = group.to_string();
    let path = reference_path.to_string();
    FnValidator::new(name.clone(), move |s: &dyn DESStation| {
        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                if silent_if_missing {
                    return Vec::new();
                }
                return vec![ValidationCheck {
                    name: format!("{name}/missing-reference"),
                    passed: false,
                    observed: Some(format!("could not read {path}")),
                    group: Some(group.clone()),
                    ..Default::default()
                }];
            }
        };
        let reference = ReferenceSolution {
            x: json_extract_number_array(&contents, "x"),
            objective: json_extract_number(&contents, "objective"),
        };
        compare(s, &reference)
    })
}

/// Extract `"key": <number>` from a flat-ish JSON string. Minimal, sufficient
/// for the `{ x, objective }` reference shape.
fn json_extract_number(src: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let start = src.find(&needle)?;
    let after = &src[start + needle.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

/// Extract `"key": [n0, n1, ...]` from a JSON string.
fn json_extract_number_array(src: &str, key: &str) -> Option<Vec<f64>> {
    let needle = format!("\"{key}\"");
    let start = src.find(&needle)?;
    let after = &src[start + needle.len()..];
    let open = after.find('[')?;
    let close = after[open..].find(']')? + open;
    let inner = &after[open + 1..close];
    let mut out = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<f64>().ok()?);
    }
    Some(out)
}

/// Wall-clock milliseconds since `t0` (the TS `Date.now() - t0`).
fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    //! The three solvers agree on a small two-product production-planning
    //! problem: Benders converges to the SAA optimum (matching the monolithic LP
    //! on the SAME scenario set), and the closed-form oracle lands near both.

    use super::*;

    fn small_problem() -> (Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
        let c = vec![1.0, 1.0];
        let p = vec![3.0, 2.0];
        let ranges = vec![(5.0, 15.0), (10.0, 20.0)];
        (c, p, ranges)
    }

    #[test]
    fn subproblem_recovers_duals() {
        let res = solve_subproblem_with_duals(
            &[3.0, 2.0],
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[4.0, 5.0],
        );
        assert_eq!(res.status, SubproblemStatus::Optimal);
        assert!((res.obj - 22.0).abs() < 1e-6, "obj = {}", res.obj);
        assert!((res.y[0] - 4.0).abs() < 1e-6 && (res.y[1] - 5.0).abs() < 1e-6);
        assert!(
            (res.duals[0] - 3.0).abs() < 1e-6 && (res.duals[1] - 2.0).abs() < 1e-6,
            "duals = {:?}",
            res.duals
        );
    }

    #[test]
    fn benders_matches_monolithic_on_same_scenarios() {
        let (c, p, ranges) = small_problem();
        let problem = build_production_slp(c.clone(), p.clone(), None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: ranges.clone(),
                seed: 42,
            },
            25,
        );

        let mono = solve_slp_monolithic(problem.clone(), scenarios.clone());
        assert_eq!(mono.status, SLPStatus::Optimal);

        let benders = solve_slp_benders(
            problem,
            scenarios,
            BendersOpts {
                tol: Some(1e-6),
                max_iter: Some(200),
                ..Default::default()
            },
        );
        assert_eq!(benders.status, SLPStatus::Optimal);

        // Both solve the same SAA; objectives must agree at convergence.
        let rel = (benders.objective - mono.objective).abs() / mono.objective.abs().max(1e-9);
        assert!(
            rel < 1e-3,
            "benders={} mono={}",
            benders.objective,
            mono.objective
        );
    }

    #[test]
    fn closed_form_is_near_saa_optimum() {
        let (c, p, ranges) = small_problem();
        let problem = build_production_slp(c.clone(), p.clone(), None);
        let scenarios = build_production_scenarios(
            UniformDemandSpec {
                ranges: ranges.clone(),
                seed: 7,
            },
            60,
        );

        let benders = solve_slp_benders(
            problem,
            scenarios,
            BendersOpts {
                tol: Some(1e-6),
                ..Default::default()
            },
        );
        let closed = solve_production_closed_form(c, p, ranges);

        assert_eq!(benders.status, SLPStatus::Optimal);
        // Closed-form first-stage x should be in the same ballpark as the SAA x.
        for i in 0..2 {
            assert!(
                (benders.x[i] - closed.x[i]).abs() < 3.0,
                "x[{i}]: benders={} closed={}",
                benders.x[i],
                closed.x[i]
            );
        }
    }

    #[test]
    fn closed_form_optimum_is_analytic() {
        let (c, p, ranges) = small_problem();
        let res = solve_production_closed_form(c, p, ranges);
        // x0* = 5 + 10·(3-1)/3 = 11.667 ; x1* = 10 + 10·(2-1)/2 = 15.
        assert!((res.x[0] - (5.0 + 10.0 * 2.0 / 3.0)).abs() < 1e-9);
        assert!((res.x[1] - 15.0).abs() < 1e-9);
        assert_eq!(res.method, SLPMethod::ClosedForm);
    }
}
