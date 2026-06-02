//! Port of `src/des/general/lp.ts` — the foundational linear-programming layer
//! for the DES framework.
//!
//! Provides:
//!   1. JSON-describable LP types (`LPProblem`, `LPSolution`, `LPStatus`).
//!   2. A small in-process two-phase simplex (`InternalSimplexSolver` /
//!      `solve_lp_internal`) — an educational fallback, NOT for large problems.
//!   3. A small in-process primal-dual interior-point method
//!      (`InternalInteriorPointSolver` / `solve_lp_internal_ipm`) for smooth
//!      educational solves on medium-small dense LPs.
//!   4. An external scipy.optimize.linprog dispatcher (`ExternalSolver` /
//!      `solve_lp_external`) that shells out via `std::process::Command`.
//!   5. `LPSolver` / `solve_lp`: selects a solver via the `LP_SOLVER` env var,
//!      falling back to the internal simplex if the external bridge is
//!      unavailable (no python / no scipy / parse failure).
//!   6. `LpPrinter` / `lp_to_string`: a human-readable pretty-printer.
//!
//! Mapping notes vs. the TypeScript source:
//!   * `class X extends PureTransform<I,O>` -> struct + `impl Transform`.
//!     Every solver `transform` always returns a fully-populated `LPSolution`
//!     (failure is carried in `status`, not via `Result`), so these are
//!     `Transform`, not `FallibleTransform`.
//!   * `throw new Error(...)` for structural input errors (dimension mismatch,
//!     unknown `LP_SOLVER` value) -> `panic!` (invariant violations).
//!     Recoverable solve failures -> `LPStatus`.
//!   * `console.warn` -> `eprintln!`.
//!   * `number` -> `f64`, indices -> `usize`, `(number|null)[]` -> `Vec<Option<f64>>`.
//!   * `Date.now()` timing -> `std::time::Instant` (reported as `elapsed_ms: f64`).
//!   * The `@deprecated` free-function shims are intentionally KEPT here
//!     (`solve_lp`, `solve_lp_internal`, `solve_lp_external`, `lp_to_string`)
//!     because other modules import them as the stable public API.
//!
//! NOTE on JSON: the TS migration header suggested `serde`, but `serde` /
//! `serde_json` are NOT available deps in this crate, so the external bridge
//! uses a small hand-rolled JSON encoder (request) and parser (response)
//! contained in this file. See the flag in the porting summary.

use std::time::Instant;

use crate::des::shared::linalg::{LinearSystem, Matrix, Vector};
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// LP problem types.
// -----------------------------------------------------------------------------

/// Optimisation direction. TS `'max' | 'min'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Sense {
    #[default]
    Max,
    Min,
}

impl Sense {
    pub fn as_str(self) -> &'static str {
        match self {
            Sense::Max => "max",
            Sense::Min => "min",
        }
    }
}

/// A linear program in canonical form:
///
/// ```text
///     max  c^T x
///     s.t. A_ub  · x ≤ b_ub      (inequality constraints)
///          A_eq  · x  =  b_eq    (equality constraints)
///          lb ≤ x ≤ ub           (per-variable bounds; default [0, +∞))
/// ```
///
/// For minimisation, set `sense: Sense::Min`. `lb`/`ub` entries of `None`
/// mean −∞ / +∞ respectively.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LPProblem {
    pub sense: Sense,
    /// Objective coefficient vector, length n.
    pub c: Vec<f64>,
    /// Inequality LHS, shape m × n (each row a constraint). May be empty.
    pub a_ub: Option<Matrix>,
    /// Inequality RHS, length m.
    pub b_ub: Option<Vector>,
    /// Equality LHS, shape p × n. May be empty.
    pub a_eq: Option<Matrix>,
    /// Equality RHS, length p.
    pub b_eq: Option<Vector>,
    /// Lower bounds, length n. `None` ⇒ −∞. Default 0.
    pub lb: Option<Vec<Option<f64>>>,
    /// Upper bounds, length n. `None` ⇒ +∞. Default +∞.
    pub ub: Option<Vec<Option<f64>>>,
    /// Optional human-readable variable names, length n.
    pub var_names: Option<Vec<String>>,
    /// Optional human-readable constraint names.
    pub con_names: Option<Vec<String>>,
}

/// A source-level LP row bound: `lower <= coefs·x <= upper`.
///
/// Commercial LP/MIP solvers expose this as a natural modelling primitive. The
/// native LP layer compiles it into equality and/or `<=` rows before solving.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LPRowConstraint {
    pub coefs: Vec<f64>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub name: Option<String>,
}

/// LP model with source-level lower/upper row bounds.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GeneralLinearLPProblem {
    pub base: LPProblem,
    pub linear_constraints: Vec<LPRowConstraint>,
}

/// LP model with a constant objective offset: `objective_offset + c^T x`.
///
/// The optimizer argmin/argmax is unchanged by the offset, but reporting the
/// true objective value matters for parity with LP/MPS-style solver surfaces.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ObjectiveOffsetLPProblem {
    pub base: LPProblem,
    pub objective_offset: f64,
}

/// One member of an LP infeasibility conflict.
///
/// This mirrors the row/bound-level IIS surfaces exposed by production LP/MIP
/// solvers: a conflict may include ordinary `<=` rows, equality rows, lower
/// variable bounds, and upper variable bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LPConflictMember {
    UpperRow(usize),
    EqualityRow(usize),
    LowerBound(usize),
    UpperBound(usize),
}

impl LPConflictMember {
    pub fn kind(self) -> &'static str {
        match self {
            LPConflictMember::UpperRow(_) => "upper-row",
            LPConflictMember::EqualityRow(_) => "equality-row",
            LPConflictMember::LowerBound(_) => "lower-bound",
            LPConflictMember::UpperBound(_) => "upper-bound",
        }
    }

    pub fn index(self) -> usize {
        match self {
            LPConflictMember::UpperRow(idx)
            | LPConflictMember::EqualityRow(idx)
            | LPConflictMember::LowerBound(idx)
            | LPConflictMember::UpperBound(idx) => idx,
        }
    }
}

/// Options for the deletion-filter conflict finder.
#[derive(Clone, Copy, Debug, Default)]
pub struct LPConflictOptions {
    pub lp_max_iter: Option<usize>,
    pub tol: Option<f64>,
}

/// A row/bound-level infeasibility conflict.
#[derive(Clone, Debug, PartialEq)]
pub struct LPInfeasibilityConflict {
    /// True when the original model is infeasible and `members` is a conflict.
    pub infeasible: bool,
    /// The row/bound members retained by the deletion filter.
    pub members: Vec<LPConflictMember>,
    /// True when removing any one retained member makes the subsystem feasible.
    pub minimal: bool,
    /// Number of LP feasibility subproblems solved while refining/checking.
    pub checks: usize,
    pub solver: String,
    pub message: Option<String>,
}

/// One relaxable member in a weighted LP feasibility relaxation.
///
/// Equality rows can be violated above or below the target, so they are split
/// into two one-sided members with independent slack variables but the same
/// source-level penalty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LPFeasRelaxMember {
    UpperRow(usize),
    EqualityUpper(usize),
    EqualityLower(usize),
    LowerBound(usize),
    UpperBound(usize),
}

impl LPFeasRelaxMember {
    pub fn kind(self) -> &'static str {
        match self {
            LPFeasRelaxMember::UpperRow(_) => "upper-row",
            LPFeasRelaxMember::EqualityUpper(_) => "equality-upper",
            LPFeasRelaxMember::EqualityLower(_) => "equality-lower",
            LPFeasRelaxMember::LowerBound(_) => "lower-bound",
            LPFeasRelaxMember::UpperBound(_) => "upper-bound",
        }
    }

    pub fn index(self) -> usize {
        match self {
            LPFeasRelaxMember::UpperRow(idx)
            | LPFeasRelaxMember::EqualityUpper(idx)
            | LPFeasRelaxMember::EqualityLower(idx)
            | LPFeasRelaxMember::LowerBound(idx)
            | LPFeasRelaxMember::UpperBound(idx) => idx,
        }
    }
}

/// Options for weighted L1 LP feasibility relaxation.
///
/// Missing penalty vectors default to unit penalties. The objective of the
/// generated relaxation LP is the weighted sum of row and bound violation
/// magnitudes, matching the common FeasRelax/FeasOpt L1 mode.
#[derive(Clone, Debug, Default)]
pub struct LPFeasRelaxOptions {
    pub upper_row_penalties: Option<Vec<f64>>,
    pub equality_row_penalties: Option<Vec<f64>>,
    pub lower_bound_penalties: Option<Vec<f64>>,
    pub upper_bound_penalties: Option<Vec<f64>>,
    pub lp_max_iter: Option<usize>,
    pub tol: Option<f64>,
}

/// One slack variable created by [`build_lp_feasibility_relaxation_problem`].
#[derive(Clone, Debug, PartialEq)]
pub struct LPFeasRelaxSlack {
    pub member: LPFeasRelaxMember,
    pub slack_var: usize,
    pub penalty: f64,
}

/// The ordinary LP generated for a weighted feasibility relaxation.
#[derive(Clone, Debug, PartialEq)]
pub struct LPFeasRelaxModel {
    pub problem: LPProblem,
    pub original_var_count: usize,
    pub slacks: Vec<LPFeasRelaxSlack>,
}

/// One positive violation in a weighted LP feasibility relaxation result.
#[derive(Clone, Debug, PartialEq)]
pub struct LPFeasRelaxViolation {
    pub member: LPFeasRelaxMember,
    pub amount: f64,
    pub penalty: f64,
    pub cost: f64,
}

/// Result of solving a weighted LP feasibility relaxation.
#[derive(Clone, Debug, PartialEq)]
pub struct LPFeasRelaxResult {
    pub status: LPStatus,
    pub x: Vec<f64>,
    pub relaxation_cost: f64,
    pub violations: Vec<LPFeasRelaxViolation>,
    pub relaxation_solution: LPSolution,
    pub solver: String,
    pub message: Option<String>,
}

/// Solve outcome. TS `type LPStatus = 'optimal' | ...`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LPStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterLimit,
    NumericalError,
}

impl LPStatus {
    /// Kebab-case string identical to the TS string-union members.
    pub fn as_str(self) -> &'static str {
        match self {
            LPStatus::Optimal => "optimal",
            LPStatus::Infeasible => "infeasible",
            LPStatus::Unbounded => "unbounded",
            LPStatus::IterLimit => "iter-limit",
            LPStatus::NumericalError => "numerical-error",
        }
    }

    /// Parse the TS string-union members back into an `LPStatus`.
    pub fn from_str(s: &str) -> Option<LPStatus> {
        match s {
            "optimal" => Some(LPStatus::Optimal),
            "infeasible" => Some(LPStatus::Infeasible),
            "unbounded" => Some(LPStatus::Unbounded),
            "iter-limit" => Some(LPStatus::IterLimit),
            "numerical-error" => Some(LPStatus::NumericalError),
            _ => None,
        }
    }
}

/// Result of a solve. Always fully populated; `status` distinguishes success
/// from failure (no `throw`).
#[derive(Clone, Debug, PartialEq)]
pub struct LPSolution {
    pub status: LPStatus,
    /// Optimal x (length n). Empty if status ≠ Optimal.
    pub x: Vec<f64>,
    /// Objective value c^T x. NaN if status ≠ Optimal.
    pub objective: f64,
    /// Dual variables for A_ub rows (shadow prices). May be `None`.
    pub dual_ub: Option<Vec<f64>>,
    /// Dual variables for A_eq rows. May be `None`.
    pub dual_eq: Option<Vec<f64>>,
    /// Reduced costs (for primal variable bounds). May be `None`.
    pub reduced_costs: Option<Vec<f64>>,
    /// Basis status for original variables. Uses normalized labels such as
    /// `basic`, `at_lower`, `at_upper`, `fixed`, `free`, and `nonbasic`.
    pub var_basis: Option<Vec<String>>,
    /// Basis status for original LP rows (`A_ub` rows followed by `A_eq` rows).
    pub row_basis: Option<Vec<String>>,
    /// Iteration count if reported by the solver.
    pub iters: Option<usize>,
    /// Solver name (e.g. `"internal"`, `"scipy:highs"`).
    pub solver: String,
    /// Wall-clock time in milliseconds.
    pub elapsed_ms: f64,
    /// Free-form human-readable message.
    pub message: Option<String>,
}

// -----------------------------------------------------------------------------
// In-process two-phase simplex.
// -----------------------------------------------------------------------------

/// Configuration for the internal simplex. TS `interface InternalSimplexOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct InternalSimplexOptions {
    /// Maximum total simplex iterations across both phases. Default 5000.
    pub max_iter: Option<usize>,
    /// Pivot tolerance. Default 1e-9.
    pub tol: Option<f64>,
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// The vanilla two-phase simplex algorithm (module-private; public entry is
/// `InternalSimplexSolver` / `solve_lp_internal`). Faithful port of TS
/// `runInternalSimplex`.
fn run_internal_simplex(p: &LPProblem, opts: &InternalSimplexOptions) -> LPSolution {
    let t0 = Instant::now();
    let tol = opts.tol.unwrap_or(1e-9);
    let max_iter = opts.max_iter.unwrap_or(5000);
    let n = p.c.len();
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub: &[f64] = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq: &[f64] = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("A_eq / b_eq length mismatch");
    }
    let lb: Vec<Option<f64>> = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub: Vec<Option<f64>> = p.ub.clone().unwrap_or_else(|| vec![None; n]);

    // ---- Standardise: shift finite lower bounds, split free variables ----
    // x_i with finite l_i -> x_i = y_i + l_i (y_i ≥ 0).
    // x_i free            -> x_i = y_i^+ − y_i^- (both ≥ 0).
    // Upper bounds become new ≤ inequality constraints.
    let mut shifts: Vec<f64> = vec![0.0; n];
    let mut free_neg: Vec<isize> = Vec::with_capacity(n); // index of "-" twin, or -1
    let mut y_index_of_pos: Vec<usize> = Vec::with_capacity(n); // y-index of "+" part
    let mut y_count: usize = 0;
    for i in 0..n {
        match lb[i] {
            None => {
                let pos = y_count;
                y_count += 1;
                y_index_of_pos.push(pos);
                let neg = y_count;
                y_count += 1;
                free_neg.push(neg as isize);
                shifts[i] = 0.0;
            }
            Some(l) => {
                let pos = y_count;
                y_count += 1;
                y_index_of_pos.push(pos);
                free_neg.push(-1);
                shifts[i] = l;
            }
        }
    }

    // Convert objective c^T x into y-space.
    let ny = y_count;
    let mut c_y: Vec<f64> = vec![0.0; ny];
    let sign = if p.sense == Sense::Max { 1.0 } else { -1.0 };
    for i in 0..n {
        c_y[y_index_of_pos[i]] += sign * p.c[i];
        if free_neg[i] >= 0 {
            c_y[free_neg[i] as usize] += -sign * p.c[i];
        }
        // (TS also accumulated `constShift` here; it is never read, so dropped.)
    }

    // Build the standard inequality system Ay · y ≤ by in y-space.
    let mut ay: Vec<Vec<f64>> = Vec::new();
    let mut by: Vec<f64> = Vec::new();
    // 1. A_ub · x ≤ b_ub.
    for r in 0..a_ub.len() {
        let mut row = vec![0.0; ny];
        let mut rhs = b_ub[r];
        for i in 0..n {
            row[y_index_of_pos[i]] += a_ub[r][i];
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] += -a_ub[r][i];
            }
            rhs -= a_ub[r][i] * shifts[i];
        }
        ay.push(row);
        by.push(rhs);
    }
    // 2. A_eq · x = b_eq encoded as two ≤ inequalities.
    for r in 0..a_eq.len() {
        let mut row = vec![0.0; ny];
        let mut rhs = b_eq[r];
        for i in 0..n {
            row[y_index_of_pos[i]] += a_eq[r][i];
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] += -a_eq[r][i];
            }
            rhs -= a_eq[r][i] * shifts[i];
        }
        ay.push(row.clone());
        by.push(rhs);
        ay.push(row.iter().map(|v| -v).collect());
        by.push(-rhs);
    }
    // 3. Upper bounds on x.
    for i in 0..n {
        if let Some(u) = ub[i] {
            let mut row = vec![0.0; ny];
            row[y_index_of_pos[i]] = 1.0;
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] = -1.0;
            }
            ay.push(row);
            by.push(u - shifts[i]);
        }
    }

    // ---- Solve  max c_Y^T y  s.t.  Ay · y ≤ by, y ≥ 0  via Big-M two-phase. ----
    let m = ay.len();
    if m == 0 {
        if c_y.iter().any(|&coef| coef > tol) {
            eprintln!(
                "[lp.internal] objective is unbounded in the '{}' direction; no constraints bound an improving ray.",
                p.sense.as_str()
            );
            return LPSolution {
                status: LPStatus::Unbounded,
                x: Vec::new(),
                objective: if p.sense == Sense::Max {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                iters: Some(0),
                solver: "internal".to_string(),
                elapsed_ms: ms_since(t0),
                message: Some("no constraints bound an improving ray".to_string()),
            };
        }

        let x = shifts.clone();
        let objective = p.c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum();
        let (dual_ub, dual_eq, reduced_costs) = recover_lp_certificate(p, &x, tol);
        return LPSolution {
            status: LPStatus::Optimal,
            x,
            objective,
            dual_ub,
            dual_eq,
            reduced_costs,
            var_basis: None,
            row_basis: None,
            iters: Some(0),
            solver: "internal".to_string(),
            elapsed_ms: ms_since(t0),
            message: Some("internal simplex: empty constraint system".to_string()),
        };
    }
    let mut acopy: Vec<Vec<f64>> = ay.iter().cloned().collect();
    let mut bcopy: Vec<f64> = by.clone();
    let total_cols = ny + m;
    let slack_start = ny;
    // Marker convention for artificials: -1 = none, -2 = needs one (resolved below).
    let mut artificial_cols: Vec<isize> = Vec::with_capacity(m);
    let mut slack_sign: Vec<f64> = Vec::with_capacity(m);
    for r in 0..m {
        if bcopy[r] < 0.0 {
            for j in 0..ny {
                acopy[r][j] = -acopy[r][j];
            }
            bcopy[r] = -bcopy[r];
            artificial_cols.push(-2);
            slack_sign.push(-1.0);
        } else {
            artificial_cols.push(-1);
            slack_sign.push(1.0);
        }
    }
    // Allocate artificial column indices.
    let mut art_count = 0;
    for r in 0..m {
        if artificial_cols[r] == -2 {
            artificial_cols[r] = (total_cols + art_count) as isize;
            art_count += 1;
        }
    }
    let full_cols = total_cols + art_count;

    // Build tableau: m rows × (full_cols + 1).
    let mut t: Vec<Vec<f64>> = Vec::with_capacity(m);
    for r in 0..m {
        let mut row = vec![0.0; full_cols + 1];
        for j in 0..ny {
            row[j] = acopy[r][j];
        }
        if artificial_cols[r] == -1 {
            row[slack_start + r] = 1.0;
        } else {
            row[slack_start + r] = -1.0;
        }
        if artificial_cols[r] >= 0 {
            row[artificial_cols[r] as usize] = 1.0;
        }
        row[full_cols] = bcopy[r];
        t.push(row);
    }

    let mut basis: Vec<usize> = Vec::with_capacity(m);
    for r in 0..m {
        basis.push(if artificial_cols[r] >= 0 {
            artificial_cols[r] as usize
        } else {
            slack_start + r
        });
    }

    // ---- Phase 1: minimise sum of artificials (via max −sum). ----
    let mut phase1_cost = vec![0.0; full_cols];
    for r in 0..m {
        if artificial_cols[r] >= 0 {
            phase1_cost[artificial_cols[r] as usize] = -1.0;
        }
    }

    let mut iters = 0usize;
    if art_count > 0 {
        let phase1 = simplex_core(
            &mut t,
            &mut basis,
            &phase1_cost,
            tol,
            max_iter.saturating_sub(iters),
        );
        iters += phase1.iters;
        if phase1.status != LPStatus::Optimal {
            eprintln!(
                "[lp.internal] phase 1 ended '{}' after {} iters (n={}, constraints={}); LP cannot be solved.",
                phase1.status.as_str(),
                iters,
                n,
                m
            );
            return LPSolution {
                status: phase1.status,
                x: Vec::new(),
                objective: f64::NAN,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                iters: Some(iters),
                solver: "internal".to_string(),
                elapsed_ms: ms_since(t0),
                message: Some(format!("phase 1 {}", phase1.status.as_str())),
            };
        }
        // If any artificial remains positive, the LP is infeasible.
        let mut phase1_obj = 0.0;
        for r in 0..m {
            if artificial_cols[r] >= 0
                && t[r][full_cols].abs() > 1e-7
                && basis[r] == artificial_cols[r] as usize
            {
                phase1_obj -= t[r][full_cols];
            }
        }
        if phase1_obj < -1e-7 {
            eprintln!(
                "[lp.internal] infeasible: phase 1 residual sum of artificials = {:.3e} (> 0); feasible region is empty.",
                -phase1_obj
            );
            return LPSolution {
                status: LPStatus::Infeasible,
                x: Vec::new(),
                objective: f64::NAN,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                iters: Some(iters),
                solver: "internal".to_string(),
                elapsed_ms: ms_since(t0),
                message: Some(format!(
                    "phase 1 residual sum of artificials = {:.3e}",
                    -phase1_obj
                )),
            };
        }
        // Drive any artificials still in the basis (value 0) out via degenerate pivots.
        for r in 0..m {
            if basis[r] >= ny + m {
                for j in 0..(ny + m) {
                    if t[r][j].abs() > tol {
                        pivot(&mut t, &mut basis, r, j);
                        break;
                    }
                }
            }
        }
    }

    // ---- Phase 2: maximise c_Y^T y. ----
    let mut phase2_cost = vec![0.0; full_cols];
    for j in 0..ny {
        phase2_cost[j] = c_y[j];
    }
    // Forbid artificials from re-entering by giving them a −∞-like cost.
    for j in (ny + m)..full_cols {
        phase2_cost[j] = -1e15;
    }
    let phase2 = simplex_core(
        &mut t,
        &mut basis,
        &phase2_cost,
        tol,
        max_iter.saturating_sub(iters),
    );
    iters += phase2.iters;
    if phase2.status == LPStatus::Unbounded {
        eprintln!(
            "[lp.internal] objective is unbounded in the '{}' direction after {} iters; check for missing bounding constraints.",
            p.sense.as_str(),
            iters
        );
        return LPSolution {
            status: LPStatus::Unbounded,
            x: Vec::new(),
            objective: if p.sense == Sense::Max {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            },
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            iters: Some(iters),
            solver: "internal".to_string(),
            elapsed_ms: ms_since(t0),
            message: None,
        };
    }
    if phase2.status != LPStatus::Optimal {
        eprintln!(
            "[lp.internal] phase 2 ended '{}' after {} iters (maxIter={}); returning without optimum.",
            phase2.status.as_str(),
            iters,
            max_iter
        );
        return LPSolution {
            status: phase2.status,
            x: Vec::new(),
            objective: f64::NAN,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            iters: Some(iters),
            solver: "internal".to_string(),
            elapsed_ms: ms_since(t0),
            message: None,
        };
    }

    // Extract y from the tableau; reconstruct x.
    let mut y_vals = vec![0.0; ny];
    for r in 0..m {
        if basis[r] < ny {
            y_vals[basis[r]] = t[r][full_cols];
        }
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        let yp = y_vals[y_index_of_pos[i]];
        let yn = if free_neg[i] >= 0 {
            y_vals[free_neg[i] as usize]
        } else {
            0.0
        };
        x[i] = yp - yn + shifts[i];
    }
    let mut obj = 0.0;
    for i in 0..n {
        obj += p.c[i] * x[i];
    }
    let (dual_ub, dual_eq, reduced_costs) =
        simplex_certificates(p, &basis, &t, &phase2_cost, ny, full_cols, &slack_sign);
    let (var_basis, row_basis) =
        lp_basis_statuses_from_basis(p, &basis, &x, &y_index_of_pos, &free_neg, ny, tol);

    LPSolution {
        status: LPStatus::Optimal,
        x,
        objective: obj,
        dual_ub,
        dual_eq,
        reduced_costs,
        var_basis,
        row_basis,
        iters: Some(iters),
        solver: "internal".to_string(),
        elapsed_ms: ms_since(t0),
        message: Some(format!("internal simplex: phase1+phase2, {iters} iters")),
    }
}

fn simplex_certificates(
    p: &LPProblem,
    basis: &[usize],
    t: &[Vec<f64>],
    cost: &[f64],
    ny: usize,
    full_cols: usize,
    slack_sign: &[f64],
) -> (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    let sense_sign = if p.sense == Sense::Max { 1.0 } else { -1.0 };
    let row_duals: Vec<f64> = slack_sign
        .iter()
        .enumerate()
        .map(|(r, &sign)| {
            let slack_col = ny + r;
            let rc = simplex_reduced_cost(t, basis, cost, slack_col, full_cols);
            clean_certificate_value(sense_sign * -sign * rc)
        })
        .collect();

    let dual_ub: Vec<f64> = row_duals.iter().take(a_ub.len()).copied().collect();
    let eq_start = a_ub.len();
    let mut dual_eq = Vec::with_capacity(a_eq.len());
    for r in 0..a_eq.len() {
        let pos = row_duals[eq_start + 2 * r];
        let neg = row_duals[eq_start + 2 * r + 1];
        dual_eq.push(clean_certificate_value(pos - neg));
    }

    let mut reduced_costs = Vec::with_capacity(p.c.len());
    for (j, &coef) in p.c.iter().enumerate() {
        let mut reduced = coef;
        for (row, &dual) in a_ub.iter().zip(&dual_ub) {
            reduced -= dual * row[j];
        }
        for (row, &dual) in a_eq.iter().zip(&dual_eq) {
            reduced -= dual * row[j];
        }
        reduced_costs.push(clean_certificate_value(reduced));
    }

    (Some(dual_ub), Some(dual_eq), Some(reduced_costs))
}

pub(crate) fn lp_basis_statuses_from_basis(
    p: &LPProblem,
    basis: &[usize],
    x: &[f64],
    y_index_of_pos: &[usize],
    free_neg: &[isize],
    y_cols: usize,
    tol: f64,
) -> (Option<Vec<String>>, Option<Vec<String>>) {
    let n = p.c.len();
    if x.len() != n || y_index_of_pos.len() != n || free_neg.len() != n {
        return (None, None);
    }
    let lb: Vec<Option<f64>> = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub: Vec<Option<f64>> = p.ub.clone().unwrap_or_else(|| vec![None; n]);
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub: &[f64] = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);

    let mut var_basis = Vec::with_capacity(n);
    for i in 0..n {
        let pos_basic = basis.iter().any(|&col| col == y_index_of_pos[i]);
        let neg_basic = free_neg[i] >= 0 && basis.iter().any(|&col| col == free_neg[i] as usize);
        let status = if pos_basic || neg_basic {
            "basic"
        } else if ub[i].is_some_and(|upper| (x[i] - upper).abs() <= tol) {
            "at_upper"
        } else if lb[i].is_some_and(|lower| (x[i] - lower).abs() <= tol) {
            "at_lower"
        } else if lb[i].is_none() && x[i].abs() <= tol {
            "free"
        } else {
            "nonbasic"
        };
        var_basis.push(status.to_string());
    }

    let mut row_basis = Vec::with_capacity(a_ub.len() + a_eq.len());
    for (r, (row, rhs)) in a_ub.iter().zip(b_ub).enumerate() {
        let slack_col = y_cols + r;
        let activity = dot_local(row, x);
        let status = if basis.iter().any(|&col| col == slack_col) {
            "basic"
        } else if (activity - rhs).abs() <= tol {
            "at_upper"
        } else {
            "nonbasic"
        };
        row_basis.push(status.to_string());
    }
    row_basis.extend((0..a_eq.len()).map(|_| "fixed".to_string()));

    (Some(var_basis), Some(row_basis))
}

fn simplex_reduced_cost(
    t: &[Vec<f64>],
    basis: &[usize],
    cost: &[f64],
    col: usize,
    rhs_col: usize,
) -> f64 {
    if col >= rhs_col {
        return 0.0;
    }
    let mut rc = cost.get(col).copied().unwrap_or(0.0);
    for (r, &basic_col) in basis.iter().enumerate() {
        let basic_cost = cost.get(basic_col).copied().unwrap_or(0.0);
        rc -= basic_cost * t[r][col];
    }
    rc
}

fn clean_certificate_value(value: f64) -> f64 {
    if value.abs() <= 1e-8 {
        0.0
    } else {
        value
    }
}

/// In-process two-phase simplex as a transform. Config (iteration cap, pivot
/// tolerance) lives on the struct; the LP is the `transform` input. Always
/// returns a fully-populated `LPSolution` (failure via `status`, not panic).
#[derive(Clone, Copy, Debug, Default)]
pub struct InternalSimplexSolver {
    pub opts: InternalSimplexOptions,
}

impl InternalSimplexSolver {
    pub fn new(opts: InternalSimplexOptions) -> Self {
        InternalSimplexSolver { opts }
    }
}

impl Transform<LPProblem, LPSolution> for InternalSimplexSolver {
    fn transform(&self, input: LPProblem) -> LPSolution {
        run_internal_simplex(&input, &self.opts)
    }
}

/// Solve an LP with the internal two-phase simplex. (Kept as the stable public
/// free-function API; prefer `InternalSimplexSolver` for new code.)
pub fn solve_lp_internal(p: &LPProblem, opts: &InternalSimplexOptions) -> LPSolution {
    run_internal_simplex(p, opts)
}

fn add_lp_objective_offset(mut sol: LPSolution, offset: f64) -> LPSolution {
    if offset != 0.0 && sol.objective.is_finite() {
        sol.objective += offset;
    }
    sol
}

/// Solve an LP with a constant objective offset using the internal simplex
/// backend.
pub fn solve_objective_offset_lp_internal(
    problem: &ObjectiveOffsetLPProblem,
    opts: &InternalSimplexOptions,
) -> LPSolution {
    let sol = solve_lp_internal(&problem.base, opts);
    add_lp_objective_offset(sol, problem.objective_offset)
}

fn validate_lp_row_constraint(base: &LPProblem, row: &LPRowConstraint, idx: usize) {
    let n = base.c.len();
    if row.coefs.len() != n {
        panic!(
            "lp row bound {idx}: coefficient length {} != variable count {n}",
            row.coefs.len()
        );
    }
    if row.lower.is_none() && row.upper.is_none() {
        panic!("lp row bound {idx}: at least one side must be finite");
    }
    for (j, &coef) in row.coefs.iter().enumerate() {
        if !coef.is_finite() {
            panic!("lp row bound {idx}: coefficient {j} must be finite");
        }
    }
    if let Some(lower) = row.lower {
        if !lower.is_finite() {
            panic!("lp row bound {idx}: lower bound must be finite");
        }
    }
    if let Some(upper) = row.upper {
        if !upper.is_finite() {
            panic!("lp row bound {idx}: upper bound must be finite");
        }
    }
    if let (Some(lower), Some(upper)) = (row.lower, row.upper) {
        if lower > upper + 1e-9 {
            panic!("lp row bound {idx}: lower bound exceeds upper bound");
        }
    }
}

fn append_lp_ub_row(out: &mut LPProblem, row: Vec<f64>, rhs: f64, name: String) {
    out.a_ub.get_or_insert_with(Vec::new).push(row);
    out.b_ub.get_or_insert_with(Vec::new).push(rhs);
    if let Some(names) = &mut out.con_names {
        names.push(name);
    }
}

fn append_lp_eq_row(out: &mut LPProblem, row: Vec<f64>, rhs: f64, name: String) {
    out.a_eq.get_or_insert_with(Vec::new).push(row);
    out.b_eq.get_or_insert_with(Vec::new).push(rhs);
    if let Some(names) = &mut out.con_names {
        names.push(name);
    }
}

/// Compile source-level LP row bounds into ordinary equality / `<=` rows.
pub fn linearize_general_linear_lp_problem(problem: &GeneralLinearLPProblem) -> LPProblem {
    let mut out = problem.base.clone();
    if out.con_names.is_none() {
        let base_rows = out.a_ub.as_ref().map(|rows| rows.len()).unwrap_or(0)
            + out.a_eq.as_ref().map(|rows| rows.len()).unwrap_or(0);
        out.con_names = Some((0..base_rows).map(|idx| format!("c{idx}")).collect());
    }
    for (idx, row) in problem.linear_constraints.iter().enumerate() {
        validate_lp_row_constraint(&problem.base, row, idx);
        let name = row
            .name
            .clone()
            .unwrap_or_else(|| format!("lp_row_bound_{idx}"));
        match (row.lower, row.upper) {
            (Some(lower), Some(upper)) if (lower - upper).abs() <= 1e-9 => {
                append_lp_eq_row(&mut out, row.coefs.clone(), upper, name);
            }
            (lower, upper) => {
                if let Some(upper) = upper {
                    append_lp_ub_row(&mut out, row.coefs.clone(), upper, format!("{name}_upper"));
                }
                if let Some(lower) = lower {
                    append_lp_ub_row(
                        &mut out,
                        row.coefs.iter().map(|v| -v).collect(),
                        -lower,
                        format!("{name}_lower"),
                    );
                }
            }
        }
    }
    out
}

/// Solve a source-level LP with arbitrary lower/upper row bounds using the
/// internal simplex backend.
pub fn solve_general_linear_lp_internal(
    problem: &GeneralLinearLPProblem,
    opts: &InternalSimplexOptions,
) -> LPSolution {
    let linearized = linearize_general_linear_lp_problem(problem);
    solve_lp_internal(&linearized, opts)
}

fn normalized_lp_bounds(p: &LPProblem) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = p.c.len();
    let lb = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub = p.ub.clone().unwrap_or_else(|| vec![None; n]);
    if lb.len() != n {
        panic!("lb length mismatch: got {}, expected {n}", lb.len());
    }
    if ub.len() != n {
        panic!("ub length mismatch: got {}, expected {n}", ub.len());
    }
    (lb, ub)
}

fn validate_lp_dimensions(p: &LPProblem) {
    let n = p.c.len();
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("A_eq / b_eq length mismatch");
    }
    for (r, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            panic!("A_ub row {r} has length {}, expected {n}", row.len());
        }
    }
    for (r, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            panic!("A_eq row {r} has length {}, expected {n}", row.len());
        }
    }
    normalized_lp_bounds(p);
}

fn validate_lp_feas_relax_penalties(name: &str, penalties: &Option<Vec<f64>>, expected: usize) {
    if let Some(penalties) = penalties {
        if penalties.len() != expected {
            panic!(
                "{name} length mismatch: got {}, expected {expected}",
                penalties.len()
            );
        }
        for (idx, &penalty) in penalties.iter().enumerate() {
            if !penalty.is_finite() || penalty < 0.0 {
                panic!("{name}[{idx}] must be a finite non-negative penalty");
            }
        }
    }
}

fn lp_feas_relax_penalty(penalties: &Option<Vec<f64>>, idx: usize) -> f64 {
    penalties
        .as_ref()
        .and_then(|values| values.get(idx))
        .copied()
        .unwrap_or(1.0)
}

fn lp_source_var_name(p: &LPProblem, j: usize) -> String {
    p.var_names
        .as_ref()
        .and_then(|names| names.get(j))
        .cloned()
        .unwrap_or_else(|| format!("x{j}"))
}

fn lp_source_row_name(p: &LPProblem, offset: usize, fallback: String) -> String {
    p.con_names
        .as_ref()
        .and_then(|names| names.get(offset))
        .cloned()
        .unwrap_or(fallback)
}

fn add_lp_feas_relax_slack(
    c: &mut Vec<f64>,
    lb: &mut Vec<Option<f64>>,
    ub: &mut Vec<Option<f64>>,
    var_names: &mut Vec<String>,
    rows: &mut [Vec<f64>],
    slacks: &mut Vec<LPFeasRelaxSlack>,
    member: LPFeasRelaxMember,
    penalty: f64,
    name: String,
) -> usize {
    let idx = c.len();
    c.push(penalty);
    lb.push(Some(0.0));
    ub.push(None);
    var_names.push(name);
    for row in rows {
        row.push(0.0);
    }
    slacks.push(LPFeasRelaxSlack {
        member,
        slack_var: idx,
        penalty,
    });
    idx
}

fn lp_feas_relax_row(coefs: &[f64], len: usize) -> Vec<f64> {
    let mut row = vec![0.0; len];
    row[..coefs.len()].copy_from_slice(coefs);
    row
}

/// Build an ordinary LP whose optimum is the weighted L1 feasibility relaxation
/// of `p`.
pub fn build_lp_feasibility_relaxation_problem(
    p: &LPProblem,
    opts: &LPFeasRelaxOptions,
) -> LPFeasRelaxModel {
    validate_lp_dimensions(p);
    let n = p.c.len();
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    let (base_lb, base_ub) = normalized_lp_bounds(p);
    validate_lp_feas_relax_penalties("upper_row_penalties", &opts.upper_row_penalties, a_ub.len());
    validate_lp_feas_relax_penalties(
        "equality_row_penalties",
        &opts.equality_row_penalties,
        a_eq.len(),
    );
    validate_lp_feas_relax_penalties("lower_bound_penalties", &opts.lower_bound_penalties, n);
    validate_lp_feas_relax_penalties("upper_bound_penalties", &opts.upper_bound_penalties, n);

    let mut c = vec![0.0; n];
    let mut lb = vec![None; n];
    let mut ub = vec![None; n];
    let mut var_names: Vec<String> = (0..n).map(|j| lp_source_var_name(p, j)).collect();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    let mut con_names = Vec::new();
    let mut slacks = Vec::new();

    for (row_idx, (coefs, &bound)) in a_ub.iter().zip(b_ub).enumerate() {
        let penalty = lp_feas_relax_penalty(&opts.upper_row_penalties, row_idx);
        let slack = add_lp_feas_relax_slack(
            &mut c,
            &mut lb,
            &mut ub,
            &mut var_names,
            &mut rows,
            &mut slacks,
            LPFeasRelaxMember::UpperRow(row_idx),
            penalty,
            format!("fr_ub_row_{row_idx}"),
        );
        let mut row = lp_feas_relax_row(coefs, c.len());
        row[slack] = -1.0;
        rows.push(row);
        rhs.push(bound);
        con_names.push(format!(
            "feasrelax_{}",
            lp_source_row_name(p, row_idx, format!("ub_row_{row_idx}"))
        ));
    }

    for (row_idx, (coefs, &bound)) in a_eq.iter().zip(b_eq).enumerate() {
        let penalty = lp_feas_relax_penalty(&opts.equality_row_penalties, row_idx);
        let source_name = lp_source_row_name(p, a_ub.len() + row_idx, format!("eq_row_{row_idx}"));
        let above_slack = add_lp_feas_relax_slack(
            &mut c,
            &mut lb,
            &mut ub,
            &mut var_names,
            &mut rows,
            &mut slacks,
            LPFeasRelaxMember::EqualityUpper(row_idx),
            penalty,
            format!("fr_eq_row_{row_idx}_above"),
        );
        let mut above = lp_feas_relax_row(coefs, c.len());
        above[above_slack] = -1.0;
        rows.push(above);
        rhs.push(bound);
        con_names.push(format!("feasrelax_{source_name}_above"));

        let below_slack = add_lp_feas_relax_slack(
            &mut c,
            &mut lb,
            &mut ub,
            &mut var_names,
            &mut rows,
            &mut slacks,
            LPFeasRelaxMember::EqualityLower(row_idx),
            penalty,
            format!("fr_eq_row_{row_idx}_below"),
        );
        let neg_coefs: Vec<f64> = coefs.iter().map(|v| -*v).collect();
        let mut below = lp_feas_relax_row(&neg_coefs, c.len());
        below[below_slack] = -1.0;
        rows.push(below);
        rhs.push(-bound);
        con_names.push(format!("feasrelax_{source_name}_below"));
    }

    for j in 0..n {
        let source_name = lp_source_var_name(p, j);
        if let Some(lower) = base_lb[j] {
            let penalty = lp_feas_relax_penalty(&opts.lower_bound_penalties, j);
            let slack = add_lp_feas_relax_slack(
                &mut c,
                &mut lb,
                &mut ub,
                &mut var_names,
                &mut rows,
                &mut slacks,
                LPFeasRelaxMember::LowerBound(j),
                penalty,
                format!("fr_lb_{source_name}"),
            );
            let mut row = vec![0.0; c.len()];
            row[j] = -1.0;
            row[slack] = -1.0;
            rows.push(row);
            rhs.push(-lower);
            con_names.push(format!("feasrelax_lb_{source_name}"));
        }
        if let Some(upper) = base_ub[j] {
            let penalty = lp_feas_relax_penalty(&opts.upper_bound_penalties, j);
            let slack = add_lp_feas_relax_slack(
                &mut c,
                &mut lb,
                &mut ub,
                &mut var_names,
                &mut rows,
                &mut slacks,
                LPFeasRelaxMember::UpperBound(j),
                penalty,
                format!("fr_ub_{source_name}"),
            );
            let mut row = vec![0.0; c.len()];
            row[j] = 1.0;
            row[slack] = -1.0;
            rows.push(row);
            rhs.push(upper);
            con_names.push(format!("feasrelax_ub_{source_name}"));
        }
    }

    LPFeasRelaxModel {
        problem: LPProblem {
            sense: Sense::Min,
            c,
            a_ub: Some(rows),
            b_ub: Some(rhs),
            a_eq: None,
            b_eq: None,
            lb: Some(lb),
            ub: Some(ub),
            var_names: Some(var_names),
            con_names: Some(con_names),
        },
        original_var_count: n,
        slacks,
    }
}

/// Solve a weighted L1 LP feasibility relaxation with the internal simplex
/// backend and decode the positive row/bound violations.
pub fn solve_lp_feasibility_relaxation_internal(
    p: &LPProblem,
    opts: &LPFeasRelaxOptions,
) -> LPFeasRelaxResult {
    let model = build_lp_feasibility_relaxation_problem(p, opts);
    let simplex_opts = InternalSimplexOptions {
        max_iter: opts.lp_max_iter,
        tol: opts.tol,
    };
    let solution = solve_lp_internal(&model.problem, &simplex_opts);
    let status = solution.status;
    let solver = solution.solver.clone();
    let message = solution.message.clone();
    if status != LPStatus::Optimal {
        return LPFeasRelaxResult {
            status,
            x: Vec::new(),
            relaxation_cost: f64::NAN,
            violations: Vec::new(),
            relaxation_solution: solution,
            solver,
            message,
        };
    }

    let tol = opts.tol.unwrap_or(1e-9);
    let x = solution.x[..model.original_var_count].to_vec();
    let mut violations = Vec::new();
    for slack in &model.slacks {
        let amount = solution.x.get(slack.slack_var).copied().unwrap_or(0.0);
        if amount > 10.0 * tol {
            violations.push(LPFeasRelaxViolation {
                member: slack.member,
                amount,
                penalty: slack.penalty,
                cost: amount * slack.penalty,
            });
        }
    }

    LPFeasRelaxResult {
        status,
        x,
        relaxation_cost: solution.objective,
        violations,
        relaxation_solution: solution,
        solver,
        message,
    }
}

/// Return every row and finite variable bound that may participate in an LP
/// infeasibility conflict.
pub fn collect_lp_conflict_members(p: &LPProblem) -> Vec<LPConflictMember> {
    validate_lp_dimensions(p);
    let mut members = Vec::new();
    for r in 0..p.a_ub.as_ref().map(|rows| rows.len()).unwrap_or(0) {
        members.push(LPConflictMember::UpperRow(r));
    }
    for r in 0..p.a_eq.as_ref().map(|rows| rows.len()).unwrap_or(0) {
        members.push(LPConflictMember::EqualityRow(r));
    }
    let (lb, ub) = normalized_lp_bounds(p);
    for i in 0..p.c.len() {
        if lb[i].is_some() {
            members.push(LPConflictMember::LowerBound(i));
        }
        if ub[i].is_some() {
            members.push(LPConflictMember::UpperBound(i));
        }
    }
    members
}

/// Build the feasibility LP induced by a selected row/bound conflict subset.
///
/// The objective is zeroed; only feasibility status is meaningful.
pub fn lp_feasibility_problem_from_conflict_members(
    p: &LPProblem,
    members: &[LPConflictMember],
) -> LPProblem {
    validate_lp_dimensions(p);
    let n = p.c.len();
    let a_ub = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
    let (base_lb, base_ub) = normalized_lp_bounds(p);

    let mut selected_ub_rows = Vec::new();
    let mut selected_ub_rhs = Vec::new();
    let mut selected_eq_rows = Vec::new();
    let mut selected_eq_rhs = Vec::new();
    let mut lb = vec![None; n];
    let mut ub = vec![None; n];
    let mut con_names = Vec::new();

    for &member in members {
        match member {
            LPConflictMember::UpperRow(row) => {
                if row >= a_ub.len() {
                    panic!("LP conflict upper row {row} out of range");
                }
                selected_ub_rows.push(a_ub[row].clone());
                selected_ub_rhs.push(b_ub[row]);
                con_names.push(
                    p.con_names
                        .as_ref()
                        .and_then(|names| names.get(row))
                        .cloned()
                        .unwrap_or_else(|| format!("ub_row_{row}")),
                );
            }
            LPConflictMember::EqualityRow(row) => {
                if row >= a_eq.len() {
                    panic!("LP conflict equality row {row} out of range");
                }
                selected_eq_rows.push(a_eq[row].clone());
                selected_eq_rhs.push(b_eq[row]);
                con_names.push(
                    p.con_names
                        .as_ref()
                        .and_then(|names| names.get(a_ub.len() + row))
                        .cloned()
                        .unwrap_or_else(|| format!("eq_row_{row}")),
                );
            }
            LPConflictMember::LowerBound(var) => {
                if var >= n {
                    panic!("LP conflict lower bound {var} out of range");
                }
                lb[var] = base_lb[var];
            }
            LPConflictMember::UpperBound(var) => {
                if var >= n {
                    panic!("LP conflict upper bound {var} out of range");
                }
                ub[var] = base_ub[var];
            }
        }
    }

    LPProblem {
        sense: Sense::Min,
        c: vec![0.0; n],
        a_ub: Some(selected_ub_rows),
        b_ub: Some(selected_ub_rhs),
        a_eq: Some(selected_eq_rows),
        b_eq: Some(selected_eq_rhs),
        lb: Some(lb),
        ub: Some(ub),
        var_names: p.var_names.clone(),
        con_names: Some(con_names),
    }
}

fn lp_conflict_subset_status(
    p: &LPProblem,
    members: &[LPConflictMember],
    opts: &LPConflictOptions,
) -> LPStatus {
    let subproblem = lp_feasibility_problem_from_conflict_members(p, members);
    solve_lp_internal(
        &subproblem,
        &InternalSimplexOptions {
            max_iter: opts.lp_max_iter,
            tol: opts.tol,
        },
    )
    .status
}

/// Find a minimal row/bound-level infeasibility conflict for a small LP.
///
/// The algorithm is the standard deletion filter used by many IIS/conflict
/// explainers: repeatedly remove a row or bound when the remaining subsystem is
/// still infeasible. The returned conflict is minimal by single deletion, not
/// guaranteed minimum-cardinality.
pub fn find_lp_infeasibility_conflict(
    p: &LPProblem,
    opts: &LPConflictOptions,
) -> LPInfeasibilityConflict {
    let mut members = collect_lp_conflict_members(p);
    let mut checks = 1usize;
    let full_status = lp_conflict_subset_status(p, &members, opts);
    if full_status != LPStatus::Infeasible {
        return LPInfeasibilityConflict {
            infeasible: false,
            members: Vec::new(),
            minimal: false,
            checks,
            solver: "internal-conflict-deletion-filter".to_string(),
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
        if lp_conflict_subset_status(p, &trial, opts) == LPStatus::Infeasible {
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
        if lp_conflict_subset_status(p, &trial, opts) == LPStatus::Infeasible {
            minimal = false;
            break;
        }
    }

    LPInfeasibilityConflict {
        infeasible: true,
        members,
        minimal,
        checks,
        solver: "internal-conflict-deletion-filter".to_string(),
        message: Some("minimal infeasible row/bound subsystem".to_string()),
    }
}

// -----------------------------------------------------------------------------
// In-process primal-dual interior-point method.
// -----------------------------------------------------------------------------

/// Configuration for the dense internal primal-dual interior-point method.
///
/// This is intentionally small and dependency-free: it is useful for DES-native
/// demos, parity checks, and modest LPs where shelling out to SciPy would be
/// awkward. Large production LPs should still use a dedicated solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct InternalInteriorPointOptions {
    /// Maximum Newton iterations. Default 100.
    pub max_iter: Option<usize>,
    /// KKT residual / complementarity tolerance. Default 1e-8.
    pub tol: Option<f64>,
    /// Fraction-to-boundary multiplier for primal/dual steps. Default 0.995.
    pub step_fraction: Option<f64>,
    /// Diagonal regularization added to the normal equations. Default 1e-10.
    pub regularization: Option<f64>,
}

#[derive(Clone, Debug)]
struct StandardInteriorLp {
    q: Vec<f64>,
    a: Matrix,
    b: Vector,
    shifts: Vec<f64>,
    y_index_of_pos: Vec<usize>,
    free_neg: Vec<isize>,
    original_c: Vec<f64>,
    original_n: usize,
    ny: usize,
}

fn empty_lp_solution(
    status: LPStatus,
    solver: &str,
    t0: Instant,
    message: impl Into<Option<String>>,
) -> LPSolution {
    LPSolution {
        status,
        x: Vec::new(),
        objective: if status == LPStatus::Unbounded {
            f64::INFINITY
        } else {
            f64::NAN
        },
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        iters: None,
        solver: solver.to_string(),
        elapsed_ms: ms_since(t0),
        message: message.into(),
    }
}

fn standardize_for_interior_point(
    p: &LPProblem,
    tol: f64,
    t0: Instant,
) -> Result<StandardInteriorLp, Box<LPSolution>> {
    let n = p.c.len();
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub: &[f64] = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq: &[f64] = p.b_eq.as_deref().unwrap_or(&[]);
    if a_ub.len() != b_ub.len() {
        panic!("A_ub / b_ub length mismatch");
    }
    if a_eq.len() != b_eq.len() {
        panic!("A_eq / b_eq length mismatch");
    }
    for (r, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            panic!("A_ub row {r} has length {}, expected {n}", row.len());
        }
    }
    for (r, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            panic!("A_eq row {r} has length {}, expected {n}", row.len());
        }
    }

    let lb: Vec<Option<f64>> = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub: Vec<Option<f64>> = p.ub.clone().unwrap_or_else(|| vec![None; n]);
    if lb.len() != n {
        panic!("lb length mismatch: got {}, expected {n}", lb.len());
    }
    if ub.len() != n {
        panic!("ub length mismatch: got {}, expected {n}", ub.len());
    }

    let mut shifts = vec![0.0; n];
    let mut free_neg: Vec<isize> = Vec::with_capacity(n);
    let mut y_index_of_pos: Vec<usize> = Vec::with_capacity(n);
    let mut y_count = 0usize;
    for i in 0..n {
        match lb[i] {
            None => {
                let pos = y_count;
                y_count += 1;
                let neg = y_count;
                y_count += 1;
                y_index_of_pos.push(pos);
                free_neg.push(neg as isize);
            }
            Some(l) => {
                let pos = y_count;
                y_count += 1;
                y_index_of_pos.push(pos);
                free_neg.push(-1);
                shifts[i] = l;
                if let Some(u) = ub[i] {
                    if u < l - tol {
                        return Err(Box::new(empty_lp_solution(
                            LPStatus::Infeasible,
                            "internal-ipm",
                            t0,
                            Some(format!("inconsistent bounds for x{i}: lb={l}, ub={u}")),
                        )));
                    }
                }
            }
        }
    }

    let ny = y_count;
    let mut q_y = vec![0.0; ny];
    for i in 0..n {
        let coeff = if p.sense == Sense::Max {
            -p.c[i]
        } else {
            p.c[i]
        };
        q_y[y_index_of_pos[i]] += coeff;
        if free_neg[i] >= 0 {
            q_y[free_neg[i] as usize] -= coeff;
        }
    }

    let mut ineq_rows: Vec<Vec<f64>> = Vec::new();
    let mut ineq_rhs: Vec<f64> = Vec::new();
    let mut eq_rows: Vec<Vec<f64>> = Vec::new();
    let mut eq_rhs: Vec<f64> = Vec::new();

    let lift_row = |row_x: &[f64], rhs_x: f64| -> (Vec<f64>, f64) {
        let mut row = vec![0.0; ny];
        let mut rhs = rhs_x;
        for i in 0..n {
            row[y_index_of_pos[i]] += row_x[i];
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] -= row_x[i];
            }
            rhs -= row_x[i] * shifts[i];
        }
        (row, rhs)
    };

    for r in 0..a_ub.len() {
        let (row, rhs) = lift_row(&a_ub[r], b_ub[r]);
        ineq_rows.push(row);
        ineq_rhs.push(rhs);
    }
    for r in 0..a_eq.len() {
        let (row, rhs) = lift_row(&a_eq[r], b_eq[r]);
        eq_rows.push(row);
        eq_rhs.push(rhs);
    }
    for i in 0..n {
        if let Some(u) = ub[i] {
            let mut row = vec![0.0; ny];
            row[y_index_of_pos[i]] = 1.0;
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] = -1.0;
            }
            ineq_rows.push(row);
            ineq_rhs.push(u - shifts[i]);
        }
    }

    let n_slack = ineq_rows.len();
    let n_std = ny + n_slack;
    let mut q = vec![0.0; n_std];
    q[..ny].copy_from_slice(&q_y);

    let mut a: Matrix = Vec::with_capacity(ineq_rows.len() + eq_rows.len());
    let mut b: Vector = Vec::with_capacity(ineq_rows.len() + eq_rows.len());
    for (r, row_y) in ineq_rows.into_iter().enumerate() {
        let mut row = vec![0.0; n_std];
        row[..ny].copy_from_slice(&row_y);
        row[ny + r] = 1.0;
        a.push(row);
        b.push(ineq_rhs[r]);
    }
    for (r, row_y) in eq_rows.into_iter().enumerate() {
        let mut row = vec![0.0; n_std];
        row[..ny].copy_from_slice(&row_y);
        a.push(row);
        b.push(eq_rhs[r]);
    }

    Ok(StandardInteriorLp {
        q,
        a,
        b,
        shifts,
        y_index_of_pos,
        free_neg,
        original_c: p.c.clone(),
        original_n: n,
        ny,
    })
}

fn reconstruct_original_x(std: &StandardInteriorLp, x_std: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; std.original_n];
    for i in 0..std.original_n {
        let yp = x_std[std.y_index_of_pos[i]];
        let yn = if std.free_neg[i] >= 0 {
            x_std[std.free_neg[i] as usize]
        } else {
            0.0
        };
        x[i] = yp - yn + std.shifts[i];
    }
    x
}

fn original_objective(std: &StandardInteriorLp, x: &[f64]) -> f64 {
    std.original_c.iter().zip(x).map(|(c, xi)| c * xi).sum()
}

fn recover_lp_certificate(
    p: &LPProblem,
    x: &[f64],
    tol: f64,
) -> (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let n = p.c.len();
    if x.len() != n {
        return (None, None, None);
    }
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub: &[f64] = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq: &[f64] = p.b_eq.as_deref().unwrap_or(&[]);
    let lb: Vec<Option<f64>> = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub: Vec<Option<f64>> = p.ub.clone().unwrap_or_else(|| vec![None; n]);
    if a_ub.len() != b_ub.len() || a_eq.len() != b_eq.len() || lb.len() != n || ub.len() != n {
        return (None, None, None);
    }
    let mut bound_state = vec![0_i8; n];
    for (j, &xj) in x.iter().enumerate() {
        if let Some(lower) = lb[j] {
            if xj < lower - 10.0 * tol {
                return (None, None, None);
            }
            if (xj - lower).abs() <= 10.0 * tol {
                bound_state[j] = -1;
            }
        }
        if let Some(upper) = ub[j] {
            if xj > upper + 10.0 * tol {
                return (None, None, None);
            }
            if (xj - upper).abs() <= 10.0 * tol {
                bound_state[j] = if bound_state[j] == -1 { 2 } else { 1 };
            }
        }
    }

    let mut active_ub = Vec::new();
    for (i, (row, rhs)) in a_ub.iter().zip(b_ub).enumerate() {
        if row.len() != n {
            return (None, None, None);
        }
        let lhs: f64 = row.iter().zip(x).map(|(a, xj)| a * xj).sum();
        if lhs > rhs + 10.0 * tol {
            return (None, None, None);
        }
        if (lhs - rhs).abs() <= 10.0 * tol {
            active_ub.push(i);
        }
    }
    for (row, rhs) in a_eq.iter().zip(b_eq) {
        if row.len() != n {
            return (None, None, None);
        }
        let lhs: f64 = row.iter().zip(x).map(|(a, xj)| a * xj).sum();
        if (lhs - rhs).abs() > 10.0 * tol {
            return (None, None, None);
        }
    }

    let interior_vars: Vec<usize> = bound_state
        .iter()
        .enumerate()
        .filter_map(|(j, &state)| if state == 0 { Some(j) } else { None })
        .collect();
    let unknowns = active_ub.len() + a_eq.len();
    if unknowns != interior_vars.len() {
        return (None, None, None);
    }
    let mut system = vec![vec![0.0; unknowns]; interior_vars.len()];
    for (col, &row_idx) in active_ub.iter().enumerate() {
        for (eq_row, &j) in interior_vars.iter().enumerate() {
            system[eq_row][col] = a_ub[row_idx][j];
        }
    }
    for (eq_idx, row) in a_eq.iter().enumerate() {
        let col = active_ub.len() + eq_idx;
        for (eq_row, &j) in interior_vars.iter().enumerate() {
            system[eq_row][col] = row[j];
        }
    }
    let gradient: Vec<f64> =
        p.c.iter()
            .map(|&c| if p.sense == Sense::Max { c } else { -c })
            .collect();
    let rhs: Vec<f64> = interior_vars.iter().map(|&j| gradient[j]).collect();
    let solution = if unknowns == 0 {
        Vec::new()
    } else {
        let Some(solution) = LinearSystem::new(&system, &rhs, tol.max(1e-10)).try_solve() else {
            return (None, None, None);
        };
        solution
    };
    let mut dual_ub = vec![0.0; a_ub.len()];
    for (col, &row_idx) in active_ub.iter().enumerate() {
        if solution[col] < -1e-7 {
            return (None, None, None);
        }
        dual_ub[row_idx] = solution[col].max(0.0);
    }
    let dual_eq = solution[active_ub.len()..].to_vec();
    let mut reduced_costs = gradient;
    for (row, &dual) in a_ub.iter().zip(&dual_ub) {
        if dual == 0.0 {
            continue;
        }
        for j in 0..n {
            reduced_costs[j] -= dual * row[j];
        }
    }
    for (row, &dual) in a_eq.iter().zip(&dual_eq) {
        for j in 0..n {
            reduced_costs[j] -= dual * row[j];
        }
    }
    for (j, &state) in bound_state.iter().enumerate() {
        match state {
            0 if reduced_costs[j].abs() > 1e-7 => return (None, None, None),
            -1 if reduced_costs[j] > 1e-7 => return (None, None, None),
            1 if reduced_costs[j] < -1e-7 => return (None, None, None),
            _ => {}
        }
    }
    (Some(dual_ub), Some(dual_eq), Some(reduced_costs))
}

fn vec_inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |acc, x| acc.max(x.abs()))
}

fn mat_vec_local(a: &Matrix, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for r in 0..a.len() {
        let mut s = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            s += a[r][j] * xj;
        }
        out[r] = s;
    }
    out
}

fn trans_mat_vec_local(a: &Matrix, y: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for r in 0..a.len() {
        let yr = y[r];
        if yr == 0.0 {
            continue;
        }
        for j in 0..n {
            out[j] += a[r][j] * yr;
        }
    }
    out
}

fn dot_local(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn fraction_to_boundary(x: &[f64], dx: &[f64], fraction: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for i in 0..x.len() {
        if dx[i] < 0.0 {
            alpha = alpha.min(-x[i] / dx[i]);
        }
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        1e-6
    } else {
        (fraction * alpha).min(1.0)
    }
}

fn solve_ipm_direction(
    a: &Matrix,
    rp: &[f64],
    rd: &[f64],
    x: &[f64],
    z: &[f64],
    rc: &[f64],
    regularization: f64,
    tol: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let m = a.len();
    let n = x.len();
    if m == 0 {
        let mut dx = vec![0.0; n];
        let mut dz = vec![0.0; n];
        for i in 0..n {
            dx[i] = (-rc[i] + x[i] * rd[i]) / z[i].max(1e-300);
            dz[i] = -rd[i];
        }
        return Some((dx, Vec::new(), dz));
    }

    let mut normal = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for i in 0..n {
        let zi = z[i].abs().max(1e-300);
        let d = (x[i] / zi).clamp(1e-14, 1e14);
        let v = (rc[i] - x[i] * rd[i]) / zi;
        for r in 0..m {
            let ari = a[r][i];
            if ari == 0.0 {
                continue;
            }
            rhs[r] += ari * v;
            for c in 0..m {
                let aci = a[c][i];
                if aci != 0.0 {
                    normal[r][c] += ari * d * aci;
                }
            }
        }
    }
    for r in 0..m {
        rhs[r] -= rp[r];
    }

    let base_reg = regularization.max(0.0);
    let scale = normal
        .iter()
        .flat_map(|row| row.iter())
        .fold(1.0_f64, |acc, v| acc.max(v.abs()));
    for k in 0..8 {
        let reg = scale * (base_reg * 10f64.powi(k)).max(1e-14);
        let mut work = normal.clone();
        for (i, row) in work.iter_mut().enumerate() {
            row[i] += reg;
        }
        if let Some(dy) = LinearSystem::new(&work, &rhs, tol.max(1e-12)).try_solve() {
            let at_dy = trans_mat_vec_local(a, &dy, n);
            let mut dx = vec![0.0; n];
            let mut dz = vec![0.0; n];
            for i in 0..n {
                dx[i] = (-rc[i] + x[i] * rd[i] + x[i] * at_dy[i]) / z[i].max(1e-300);
                dz[i] = -rd[i] - at_dy[i];
            }
            return Some((dx, dy, dz));
        }
    }
    None
}

fn run_internal_ipm(p: &LPProblem, opts: &InternalInteriorPointOptions) -> LPSolution {
    let t0 = Instant::now();
    let max_iter = opts.max_iter.unwrap_or(100);
    let tol = opts.tol.unwrap_or(1e-8);
    let step_fraction = opts.step_fraction.unwrap_or(0.995).clamp(0.5, 0.999_999);
    let regularization = opts.regularization.unwrap_or(1e-10);

    let std = match standardize_for_interior_point(p, tol, t0) {
        Ok(std) => std,
        Err(sol) => return *sol,
    };
    let n = std.q.len();
    let m = std.a.len();

    if n == 0 {
        let feasible = std.b.iter().all(|&bi| bi.abs() <= tol);
        return if feasible {
            LPSolution {
                status: LPStatus::Optimal,
                x: Vec::new(),
                objective: 0.0,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                iters: Some(0),
                solver: "internal-ipm".to_string(),
                elapsed_ms: ms_since(t0),
                message: Some("internal primal-dual IPM: empty feasible LP".to_string()),
            }
        } else {
            empty_lp_solution(
                LPStatus::Infeasible,
                "internal-ipm",
                t0,
                Some("empty LP violates equality/inequality constraints".to_string()),
            )
        };
    }

    if m == 0 {
        if std.q.iter().any(|&qi| qi < -tol) {
            let mut sol = empty_lp_solution(
                LPStatus::Unbounded,
                "internal-ipm",
                t0,
                Some("unconstrained LP has an improving nonnegative ray".to_string()),
            );
            sol.objective = if p.sense == Sense::Max {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            return sol;
        }
        let x_std = vec![0.0; n];
        let x = reconstruct_original_x(&std, &x_std);
        let objective = original_objective(&std, &x);
        return LPSolution {
            status: LPStatus::Optimal,
            x,
            objective,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            iters: Some(0),
            solver: "internal-ipm".to_string(),
            elapsed_ms: ms_since(t0),
            message: Some("internal primal-dual IPM: unconstrained optimum".to_string()),
        };
    }

    let mut x = vec![1.0; n];
    for j in std.ny..n {
        let row = j - std.ny;
        if row < std.b.len() {
            x[j] = std.b[row].abs().max(1.0);
        }
    }
    let mut y = vec![0.0; m];
    let mut z = vec![1.0; n];

    let b_scale = 1.0 + vec_inf_norm(&std.b);
    let q_scale = 1.0 + vec_inf_norm(&std.q);

    for iter in 0..=max_iter {
        let ax = mat_vec_local(&std.a, &x);
        let rp: Vec<f64> = ax.iter().zip(&std.b).map(|(axi, bi)| axi - bi).collect();
        let aty = trans_mat_vec_local(&std.a, &y, n);
        let rd: Vec<f64> = aty
            .iter()
            .zip(&z)
            .zip(&std.q)
            .map(|((ati, zi), qi)| ati + zi - qi)
            .collect();
        let mu = dot_local(&x, &z) / n as f64;
        let min_obj = dot_local(&std.q, &x);
        let primal_ok = vec_inf_norm(&rp) <= tol * b_scale;
        let dual_ok = vec_inf_norm(&rd) <= tol * q_scale;
        let gap_ok = mu <= tol * (1.0 + min_obj.abs());
        if primal_ok && dual_ok && gap_ok {
            let x_orig = reconstruct_original_x(&std, &x);
            let objective = original_objective(&std, &x_orig);
            return LPSolution {
                status: LPStatus::Optimal,
                x: x_orig,
                objective,
                dual_ub: None,
                dual_eq: None,
                reduced_costs: None,
                var_basis: None,
                row_basis: None,
                iters: Some(iter),
                solver: "internal-ipm".to_string(),
                elapsed_ms: ms_since(t0),
                message: Some(format!(
                    "internal primal-dual IPM: converged in {iter} iterations"
                )),
            };
        }
        if iter == max_iter {
            break;
        }

        let rc_aff: Vec<f64> = x.iter().zip(&z).map(|(xi, zi)| xi * zi).collect();
        let (dx_aff, _dy_aff, dz_aff) =
            match solve_ipm_direction(&std.a, &rp, &rd, &x, &z, &rc_aff, regularization, tol) {
                Some(dir) => dir,
                None => {
                    return empty_lp_solution(
                        LPStatus::NumericalError,
                        "internal-ipm",
                        t0,
                        Some("normal equations became singular in affine step".to_string()),
                    )
                }
            };

        let alpha_pri_aff = fraction_to_boundary(&x, &dx_aff, 1.0);
        let alpha_dual_aff = fraction_to_boundary(&z, &dz_aff, 1.0);
        let mut mu_aff = 0.0;
        for i in 0..n {
            let xp = (x[i] + alpha_pri_aff * dx_aff[i]).max(0.0);
            let zp = (z[i] + alpha_dual_aff * dz_aff[i]).max(0.0);
            mu_aff += xp * zp;
        }
        mu_aff /= n as f64;
        let sigma = if mu > 0.0 {
            (mu_aff / mu).clamp(0.0, 1.0).powi(3)
        } else {
            0.0
        };
        let rc_corr: Vec<f64> = (0..n)
            .map(|i| x[i] * z[i] + dx_aff[i] * dz_aff[i] - sigma * mu)
            .collect();
        let (dx, dy_step, dz) =
            match solve_ipm_direction(&std.a, &rp, &rd, &x, &z, &rc_corr, regularization, tol) {
                Some(dir) => dir,
                None => {
                    let rc_centered: Vec<f64> = (0..n).map(|i| x[i] * z[i] - 0.1 * mu).collect();
                    match solve_ipm_direction(
                        &std.a,
                        &rp,
                        &rd,
                        &x,
                        &z,
                        &rc_centered,
                        regularization,
                        tol,
                    ) {
                        Some(dir) => dir,
                        None => {
                            return empty_lp_solution(
                                LPStatus::NumericalError,
                                "internal-ipm",
                                t0,
                                Some(
                                    "normal equations became singular in corrector step"
                                        .to_string(),
                                ),
                            )
                        }
                    }
                }
            };

        let alpha_pri = fraction_to_boundary(&x, &dx, step_fraction);
        let alpha_dual = fraction_to_boundary(&z, &dz, step_fraction);
        for i in 0..n {
            x[i] = (x[i] + alpha_pri * dx[i]).max(1e-300);
            z[i] = (z[i] + alpha_dual * dz[i]).max(1e-300);
        }
        for r in 0..m {
            y[r] += alpha_dual * dy_step[r];
        }
    }

    let x_orig = reconstruct_original_x(&std, &x);
    let objective = original_objective(&std, &x_orig);
    LPSolution {
        status: LPStatus::IterLimit,
        x: x_orig,
        objective,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        var_basis: None,
        row_basis: None,
        iters: Some(max_iter),
        solver: "internal-ipm".to_string(),
        elapsed_ms: ms_since(t0),
        message: Some(format!(
            "internal primal-dual IPM hit max_iter={max_iter} before KKT residuals reached tol={tol}"
        )),
    }
}

/// Dense in-process primal-dual interior-point solver as a transform.
#[derive(Clone, Copy, Debug, Default)]
pub struct InternalInteriorPointSolver {
    pub opts: InternalInteriorPointOptions,
}

impl InternalInteriorPointSolver {
    pub fn new(opts: InternalInteriorPointOptions) -> Self {
        InternalInteriorPointSolver { opts }
    }
}

impl Transform<LPProblem, LPSolution> for InternalInteriorPointSolver {
    fn transform(&self, input: LPProblem) -> LPSolution {
        run_internal_ipm(&input, &self.opts)
    }
}

/// Solve an LP with the native dense primal-dual interior-point method.
pub fn solve_lp_internal_ipm(p: &LPProblem, opts: &InternalInteriorPointOptions) -> LPSolution {
    run_internal_ipm(p, opts)
}

// Pivoting machinery. Bland's rule for entering / leaving to guarantee
// finite termination on small problems.
struct SimplexResult {
    status: LPStatus,
    iters: usize,
}

fn simplex_core(
    t: &mut Vec<Vec<f64>>,
    basis: &mut [usize],
    cost: &[f64],
    tol: f64,
    max_iter: usize,
) -> SimplexResult {
    let m = t.len();
    if m == 0 {
        return SimplexResult {
            status: LPStatus::Optimal,
            iters: 0,
        };
    }
    let ncols = t[0].len() - 1;
    let mut iters = 0usize;
    while iters < max_iter {
        iters += 1;
        // Reduced costs: cost[j] − Σ_r cost[basis[r]] · T[r][j].
        let mut entering: Option<usize> = None;
        let best_rc = tol;
        for j in 0..ncols {
            let mut rc = cost[j];
            for r in 0..m {
                rc -= cost[basis[r]] * t[r][j];
            }
            if rc > best_rc {
                // Bland: first strictly-improving column.
                entering = Some(j);
                break;
            }
        }
        let entering = match entering {
            Some(e) => e,
            None => {
                return SimplexResult {
                    status: LPStatus::Optimal,
                    iters,
                }
            }
        };
        // Min-ratio test (Bland tie-break on smallest basis index).
        let mut leaving: Option<usize> = None;
        let mut best_ratio = f64::INFINITY;
        for r in 0..m {
            if t[r][entering] > tol {
                let ratio = t[r][ncols] / t[r][entering];
                if ratio < best_ratio - tol
                    || ((ratio - best_ratio).abs() <= tol
                        && (leaving.is_none() || basis[r] < basis[leaving.unwrap()]))
                {
                    best_ratio = ratio;
                    leaving = Some(r);
                }
            }
        }
        let leaving = match leaving {
            Some(l) => l,
            None => {
                return SimplexResult {
                    status: LPStatus::Unbounded,
                    iters,
                }
            }
        };
        pivot(t, basis, leaving, entering);
    }
    eprintln!(
        "[lp.simplexCore] reached iteration limit ({max_iter}) without optimality — possible cycling or an ill-conditioned tableau."
    );
    SimplexResult {
        status: LPStatus::IterLimit,
        iters,
    }
}

fn pivot(t: &mut Vec<Vec<f64>>, basis: &mut [usize], pivot_row: usize, pivot_col: usize) {
    let ncols = t[0].len();
    let pv = t[pivot_row][pivot_col];
    for j in 0..ncols {
        t[pivot_row][j] /= pv;
    }
    // Snapshot the normalised pivot row (avoids aliasing the borrow below).
    let prow = t[pivot_row].clone();
    for r in 0..t.len() {
        if r == pivot_row {
            continue;
        }
        let factor = t[r][pivot_col];
        if factor == 0.0 {
            continue;
        }
        for j in 0..ncols {
            t[r][j] -= factor * prow[j];
        }
    }
    basis[pivot_row] = pivot_col;
}

// -----------------------------------------------------------------------------
// External-solver dispatcher.
// -----------------------------------------------------------------------------

/// Default path to the repository-local Python LP bridge. The bridge supports
/// SciPy/HiGHS and OR-Tools GLOP, and falls back to dependency-free vertex
/// enumeration for small validation models.
const DEFAULT_SCRIPT: &str = "scripts/lp_solve.py";

fn external_solver_label(method: &str) -> String {
    let normalized = method.to_ascii_lowercase().replace('_', "-");
    if matches!(
        normalized.as_str(),
        "glop" | "ortools-glop" | "ortools:glop"
    ) {
        "ortools:glop".to_string()
    } else {
        format!("scipy:{method}")
    }
}

/// Configuration for the external Python LP bridge. TS `interface ExternalSolverOptions`.
///
/// `method` is modelled as a free `String` (rather than a closed enum) to
/// faithfully reproduce the TS behaviour where external methods can be passed
/// through to the bridge. `max_buffer` is accepted for API parity but unused:
/// `std::process::Command` captures the full output.
#[derive(Clone, Debug, Default)]
pub struct ExternalSolverOptions {
    /// External LP method: SciPy linprog methods (`"highs"`, `"highs-ds"`, `"highs-ipm"`) or OR-Tools `"glop"`. Default `"highs"`.
    pub method: Option<String>,
    /// Override the python executable. Defaults to `PYTHON`, then `PYTHON_BIN`, then `"python3"`.
    pub python: Option<String>,
    /// Override the script path. Defaults to `scripts/lp_solve.py`.
    pub script: Option<String>,
    /// Accepted for parity with the TS `maxBuffer`; unused in the Rust port.
    pub max_buffer: Option<usize>,
}

/// External Python LP dispatcher as a transform. Returns status
/// `NumericalError` if the requested solver / python is unavailable (or the process fails /
/// emits unparseable output) — use `LPSolver` for graceful fallback.
#[derive(Clone, Debug, Default)]
pub struct ExternalSolver {
    pub opts: ExternalSolverOptions,
}

impl ExternalSolver {
    pub fn new(opts: ExternalSolverOptions) -> Self {
        ExternalSolver { opts }
    }

    fn run(&self, p: &LPProblem) -> LPSolution {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let t0 = Instant::now();
        let method = self
            .opts
            .method
            .clone()
            .unwrap_or_else(|| "highs".to_string());
        let requested_solver = external_solver_label(&method);
        let python = self
            .opts
            .python
            .clone()
            .or_else(|| std::env::var("PYTHON").ok())
            .or_else(|| std::env::var("PYTHON_BIN").ok())
            .unwrap_or_else(|| "python3".to_string());
        let script = self
            .opts
            .script
            .clone()
            .unwrap_or_else(|| DEFAULT_SCRIPT.to_string());

        let numerical_error = |msg: String, t0: Instant| LPSolution {
            status: LPStatus::NumericalError,
            x: Vec::new(),
            objective: f64::NAN,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            var_basis: None,
            row_basis: None,
            iters: None,
            solver: requested_solver.clone(),
            elapsed_ms: ms_since(t0),
            message: Some(msg),
        };

        let payload = encode_request(p, &method);

        let mut child = match Command::new(&python)
            .arg(&script)
            .arg("--method")
            .arg(&method)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lp.external] {requested_solver} could not start ({python}): {e}");
                return numerical_error(format!("external solver could not start: {e}"), t0);
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(payload.as_bytes());
            // `stdin` dropped here, closing the pipe so the child sees EOF.
        }

        let out = match child.wait_with_output() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[lp.external] {requested_solver} wait failed: {e}");
                return numerical_error(format!("external solver wait failed: {e}"), t0);
            }
        };

        if out.status.code() != Some(0) {
            let code = out
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "null".to_string());
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stderr = if stderr.is_empty() {
                "(no stderr)".to_string()
            } else {
                stderr.to_string()
            };
            eprintln!("[lp.external] {requested_solver} process exited with code {code}: {stderr}");
            return numerical_error(format!("external solver exited with {code}: {stderr}"), t0);
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed = match json_parse(&stdout) {
            Ok(v) => v,
            Err(e) => {
                let head: String = stdout.chars().take(120).collect();
                eprintln!(
                    "[lp.external] could not parse {requested_solver} stdout as JSON: {e}; stdout head=\"{head}\""
                );
                return numerical_error(
                    format!("failed to parse external solver stdout as JSON: {e}"),
                    t0,
                );
            }
        };

        let status = json_get(&parsed, "status")
            .and_then(json_as_str)
            .and_then(LPStatus::from_str)
            .unwrap_or(LPStatus::NumericalError);
        let x = json_get(&parsed, "x")
            .and_then(json_as_f64_array)
            .unwrap_or_default();
        let objective = json_get(&parsed, "objective")
            .and_then(json_as_f64)
            .unwrap_or(f64::NAN);
        let solver = json_get(&parsed, "solver")
            .and_then(json_as_str)
            .map(|s| s.to_string())
            .unwrap_or_else(|| requested_solver.clone());

        LPSolution {
            status,
            x,
            objective,
            dual_ub: json_get(&parsed, "dualUB").and_then(json_as_f64_array),
            dual_eq: json_get(&parsed, "dualEQ").and_then(json_as_f64_array),
            reduced_costs: json_get(&parsed, "reducedCosts").and_then(json_as_f64_array),
            var_basis: json_get(&parsed, "varBasis").and_then(json_as_string_array),
            row_basis: json_get(&parsed, "rowBasis").and_then(json_as_string_array),
            iters: json_get(&parsed, "iters")
                .and_then(json_as_f64)
                .map(|f| f as usize),
            solver,
            elapsed_ms: ms_since(t0),
            message: json_get(&parsed, "message")
                .and_then(json_as_str)
                .map(|s| s.to_string()),
        }
    }
}

impl Transform<LPProblem, LPSolution> for ExternalSolver {
    fn transform(&self, input: LPProblem) -> LPSolution {
        self.run(&input)
    }
}

/// Solve via an external scipy.optimize.linprog process. (Kept as the stable
/// public free-function API; prefer `ExternalSolver` for new code.)
pub fn solve_lp_external(p: &LPProblem, opts: &ExternalSolverOptions) -> LPSolution {
    ExternalSolver::new(opts.clone()).run(p)
}

/// Solve an LP with a constant objective offset using the external LP bridge.
pub fn solve_objective_offset_lp_external(
    problem: &ObjectiveOffsetLPProblem,
    opts: &ExternalSolverOptions,
) -> LPSolution {
    let sol = solve_lp_external(&problem.base, opts);
    add_lp_objective_offset(sol, problem.objective_offset)
}

/// Solve a source-level LP with arbitrary lower/upper row bounds using the
/// external LP bridge after compiling row bounds.
pub fn solve_general_linear_lp_external(
    problem: &GeneralLinearLPProblem,
    opts: &ExternalSolverOptions,
) -> LPSolution {
    let linearized = linearize_general_linear_lp_problem(problem);
    solve_lp_external(&linearized, opts)
}

/// Combined options for `LPSolver` (TS `ExternalSolverOptions & InternalSimplexOptions`).
#[derive(Clone, Debug, Default)]
pub struct LpSolverOptions {
    pub method: Option<String>,
    pub python: Option<String>,
    pub script: Option<String>,
    pub max_buffer: Option<usize>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

impl LpSolverOptions {
    fn internal(&self) -> InternalSimplexOptions {
        InternalSimplexOptions {
            max_iter: self.max_iter,
            tol: self.tol,
        }
    }

    fn internal_ipm(&self) -> InternalInteriorPointOptions {
        InternalInteriorPointOptions {
            max_iter: self.max_iter,
            tol: self.tol,
            step_fraction: None,
            regularization: None,
        }
    }

    fn external(&self, method: Option<String>) -> ExternalSolverOptions {
        ExternalSolverOptions {
            method,
            python: self.python.clone(),
            script: self.script.clone(),
            max_buffer: self.max_buffer,
        }
    }
}

/// Solve an LP using the solver selected by env var `LP_SOLVER`, falling back
/// to the internal simplex if the external bridge is unavailable.
///
/// ```text
///   LP_SOLVER=internal              in-process two-phase simplex
///   LP_SOLVER=internal-ipm          in-process primal-dual interior-point method
///   LP_SOLVER=scipy:highs           scipy linprog method=highs (DEFAULT)
///   LP_SOLVER=scipy:highs-ipm       scipy interior-point HiGHS
///   LP_SOLVER=scipy:highs-ds        scipy dual simplex HiGHS
///   LP_SOLVER=scipy:simplex         legacy scipy simplex
///   LP_SOLVER=scipy:interior-point  legacy scipy interior-point
///   LP_SOLVER=ortools:glop          OR-Tools GLOP linear solver
/// ```
#[derive(Clone, Debug, Default)]
pub struct LPSolver {
    pub opts: LpSolverOptions,
}

impl LPSolver {
    pub fn new(opts: LpSolverOptions) -> Self {
        LPSolver { opts }
    }
}

impl Transform<LPProblem, LPSolution> for LPSolver {
    fn transform(&self, input: LPProblem) -> LPSolution {
        let choice = std::env::var("LP_SOLVER").unwrap_or_else(|_| "scipy:highs".to_string());
        let choice = choice.trim();
        if choice == "internal" {
            return run_internal_simplex(&input, &self.opts.internal());
        }
        if choice == "internal-ipm" || choice == "internal-interior-point" {
            return run_internal_ipm(&input, &self.opts.internal_ipm());
        }
        if let Some(method) = choice.strip_prefix("scipy:") {
            let ext = ExternalSolver::new(self.opts.external(Some(method.to_string()))).run(&input);
            if ext.status != LPStatus::NumericalError {
                return ext;
            }
            // Fall back to internal if the external bridge failed (no scipy / no python / etc).
            eprintln!(
                "[lp.solveLP] external solver '{choice}' unavailable/failed ({}); falling back to internal simplex.",
                ext.message.as_deref().unwrap_or("unknown")
            );
            let mut fallback = run_internal_simplex(&input, &self.opts.internal());
            let prefix = match &fallback.message {
                Some(m) => format!("{m} | "),
                None => String::new(),
            };
            fallback.message = Some(format!(
                "{prefix}external solver unavailable, fell back to internal: {}",
                ext.message.as_deref().unwrap_or("")
            ));
            return fallback;
        }
        if choice == "ortools:glop" || choice == "glop" {
            let ext = ExternalSolver::new(self.opts.external(Some("glop".to_string()))).run(&input);
            if ext.status != LPStatus::NumericalError {
                return ext;
            }
            eprintln!(
                "[lp.solveLP] external solver '{choice}' unavailable/failed ({}); falling back to internal simplex.",
                ext.message.as_deref().unwrap_or("unknown")
            );
            let mut fallback = run_internal_simplex(&input, &self.opts.internal());
            let prefix = match &fallback.message {
                Some(m) => format!("{m} | "),
                None => String::new(),
            };
            fallback.message = Some(format!(
                "{prefix}external solver unavailable, fell back to internal: {}",
                ext.message.as_deref().unwrap_or("")
            ));
            return fallback;
        }
        panic!("unknown LP_SOLVER value: {choice}");
    }
}

/// Solve an LP via the `LP_SOLVER`-selected solver with internal fallback.
/// (Kept as the stable public free-function API; prefer `LPSolver` for new code.)
pub fn solve_lp(p: &LPProblem, opts: &LpSolverOptions) -> LPSolution {
    LPSolver::new(opts.clone()).transform(p.clone())
}

// -----------------------------------------------------------------------------
// Convenience pretty-printer.
// -----------------------------------------------------------------------------

fn fmt_num(v: f64) -> String {
    format!("{v}")
}

fn term(a: f64, name: &str) -> String {
    if a == 0.0 {
        return String::new();
    }
    let sign = if a >= 0.0 { " + " } else { " − " };
    let mag = a.abs();
    let mag_str = if mag == 1.0 {
        String::new()
    } else {
        fmt_num(mag)
    };
    format!("{sign}{mag_str}{name}")
}

fn strip_leading_plus(s: &str) -> String {
    s.strip_prefix(" + ")
        .map(str::to_string)
        .unwrap_or_else(|| s.to_string())
}

fn render(p: &LPProblem) -> String {
    let mut lines: Vec<String> = Vec::new();
    let n = p.c.len();
    let names: Vec<String> = match &p.var_names {
        Some(v) => v.clone(),
        None => (0..n).map(|i| format!("x{i}")).collect(),
    };

    let obj_line = strip_leading_plus(
        &p.c.iter()
            .enumerate()
            .map(|(i, &a)| term(a, &names[i]))
            .collect::<String>(),
    );
    lines.push(format!("{}  {}", p.sense.as_str(), obj_line));

    if let Some(a_ub) = &p.a_ub {
        if !a_ub.is_empty() {
            lines.push("s.t.".to_string());
            let b_ub = p.b_ub.as_deref().unwrap_or(&[]);
            for (r, row) in a_ub.iter().enumerate() {
                let lhs = strip_leading_plus(
                    &row.iter()
                        .enumerate()
                        .map(|(i, &a)| term(a, &names[i]))
                        .collect::<String>(),
                );
                lines.push(format!("     {} ≤ {}", lhs, fmt_num(b_ub[r])));
            }
        }
    }

    if let Some(a_eq) = &p.a_eq {
        if !a_eq.is_empty() {
            let b_eq = p.b_eq.as_deref().unwrap_or(&[]);
            for (r, row) in a_eq.iter().enumerate() {
                let lhs = strip_leading_plus(
                    &row.iter()
                        .enumerate()
                        .map(|(i, &a)| term(a, &names[i]))
                        .collect::<String>(),
                );
                lines.push(format!("     {} = {}", lhs, fmt_num(b_eq[r])));
            }
        }
    }

    if p.lb.is_some() || p.ub.is_some() {
        for i in 0..n {
            let l: Option<f64> = match &p.lb {
                Some(v) => v[i],
                None => Some(0.0),
            };
            let u: Option<f64> = match &p.ub {
                Some(v) => v[i],
                None => None,
            };
            if l == Some(0.0) && u.is_none() {
                continue;
            }
            let ls = match l {
                None => "−∞".to_string(),
                Some(x) => fmt_num(x),
            };
            let us = match u {
                None => "+∞".to_string(),
                Some(x) => fmt_num(x),
            };
            lines.push(format!("     {} ≤ {} ≤ {}", ls, names[i], us));
        }
    }

    lines.join("\n")
}

/// Render an LP in human-readable form. No config; the LP is the transform input.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpPrinter;

impl Transform<LPProblem, String> for LpPrinter {
    fn transform(&self, input: LPProblem) -> String {
        render(&input)
    }
}

/// Pretty-print an LP. (Kept as the stable public free-function API; prefer
/// `LpPrinter` for new code.)
pub fn lp_to_string(p: &LPProblem) -> String {
    render(p)
}

// -----------------------------------------------------------------------------
// Minimal in-file JSON (no `serde` dependency available in this crate).
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn json_get<'a>(v: &'a Json, key: &str) -> Option<&'a Json> {
    match v {
        Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, vv)| vv),
        _ => None,
    }
}

fn json_as_f64(v: &Json) -> Option<f64> {
    match v {
        Json::Num(n) => Some(*n),
        _ => None,
    }
}

fn json_as_str(v: &Json) -> Option<&str> {
    match v {
        Json::Str(s) => Some(s.as_str()),
        _ => None,
    }
}

fn json_as_f64_array(v: &Json) -> Option<Vec<f64>> {
    match v {
        Json::Arr(items) => items.iter().map(json_as_f64).collect(),
        _ => None,
    }
}

fn json_as_string_array(v: &Json) -> Option<Vec<String>> {
    match v {
        Json::Arr(items) => items
            .iter()
            .map(|item| json_as_str(item).map(str::to_string))
            .collect(),
        _ => None,
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(b'n') => self.null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!(
                "unexpected character '{}' at {}",
                c as char, self.pos
            )),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.pos += 1; // consume '{'
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at {}", self.pos));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.pos));
            }
            self.pos += 1;
            let val = self.value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
        Ok(Json::Obj(entries))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.pos += 1; // consume '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.pos += 1; // consume opening quote
        let mut out = String::new();
        loop {
            match self.bytes.get(self.pos) {
                None => return Err("unterminated string".to_string()),
                Some(&b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(&b'\\') => {
                    self.pos += 1;
                    match self.bytes.get(self.pos) {
                        Some(&b'"') => out.push('"'),
                        Some(&b'\\') => out.push('\\'),
                        Some(&b'/') => out.push('/'),
                        Some(&b'b') => out.push('\u{0008}'),
                        Some(&b'f') => out.push('\u{000C}'),
                        Some(&b'n') => out.push('\n'),
                        Some(&b'r') => out.push('\r'),
                        Some(&b't') => out.push('\t'),
                        Some(&b'u') => {
                            let hex = self
                                .bytes
                                .get(self.pos + 1..self.pos + 5)
                                .ok_or("truncated \\u escape")?;
                            let hex = std::str::from_utf8(hex).map_err(|_| "bad \\u escape")?;
                            let code =
                                u32::from_str_radix(hex, 16).map_err(|_| "bad \\u escape")?;
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                            self.pos += 4;
                        }
                        _ => return Err("bad escape".to_string()),
                    }
                    self.pos += 1;
                }
                Some(_) => {
                    // Copy one UTF-8 codepoint.
                    let start = self.pos;
                    let mut end = self.pos + 1;
                    while end < self.bytes.len() && (self.bytes[end] & 0xC0) == 0x80 {
                        end += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..end]).map_err(|_| "bad utf8")?,
                    );
                    self.pos = end;
                }
            }
        }
        Ok(out)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.' || b == b'e' || b == b'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| "bad number")?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("invalid number '{s}'"))
    }

    fn boolean(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn null(&mut self) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }
}

fn json_parse(s: &str) -> Result<Json, String> {
    let mut p = JsonParser::new(s);
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    Ok(v)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
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
    out
}

fn json_num(v: f64) -> String {
    // JSON has no NaN / Infinity; JSON.stringify maps them to null — mirror that.
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

fn json_num_array(v: &[f64]) -> String {
    let inner: Vec<String> = v.iter().map(|&x| json_num(x)).collect();
    format!("[{}]", inner.join(","))
}

fn json_matrix(m: &[Vec<f64>]) -> String {
    let inner: Vec<String> = m.iter().map(|row| json_num_array(row)).collect();
    format!("[{}]", inner.join(","))
}

fn json_opt_array(v: &[Option<f64>]) -> String {
    let inner: Vec<String> = v
        .iter()
        .map(|x| match x {
            Some(n) => json_num(*n),
            None => "null".to_string(),
        })
        .collect();
    format!("[{}]", inner.join(","))
}

fn json_str_array(v: &[String]) -> String {
    let inner: Vec<String> = v
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect();
    format!("[{}]", inner.join(","))
}

fn encode_lp_json(p: &LPProblem) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("\"sense\":\"{}\"", p.sense.as_str()));
    parts.push(format!("\"c\":{}", json_num_array(&p.c)));
    if let Some(m) = &p.a_ub {
        parts.push(format!("\"A_ub\":{}", json_matrix(m)));
    }
    if let Some(v) = &p.b_ub {
        parts.push(format!("\"b_ub\":{}", json_num_array(v)));
    }
    if let Some(m) = &p.a_eq {
        parts.push(format!("\"A_eq\":{}", json_matrix(m)));
    }
    if let Some(v) = &p.b_eq {
        parts.push(format!("\"b_eq\":{}", json_num_array(v)));
    }
    if let Some(v) = &p.lb {
        parts.push(format!("\"lb\":{}", json_opt_array(v)));
    }
    if let Some(v) = &p.ub {
        parts.push(format!("\"ub\":{}", json_opt_array(v)));
    }
    if let Some(v) = &p.var_names {
        parts.push(format!("\"varNames\":{}", json_str_array(v)));
    }
    if let Some(v) = &p.con_names {
        parts.push(format!("\"conNames\":{}", json_str_array(v)));
    }
    format!("{{{}}}", parts.join(","))
}

fn encode_request(p: &LPProblem, method: &str) -> String {
    format!(
        "{{\"lp\":{},\"method\":\"{}\"}}",
        encode_lp_json(p),
        json_escape(method)
    )
}

// -----------------------------------------------------------------------------
// Tests — internal simplex on small LPs with known optima.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn opts() -> InternalSimplexOptions {
        InternalSimplexOptions::default()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOL,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn finds_minimal_lp_infeasibility_conflict() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0], vec![-1.0], vec![1.0]]),
            b_ub: Some(vec![0.0, -1.0, 5.0]),
            lb: Some(vec![Some(0.0)]),
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_at_most_zero".to_string(),
                "x_at_least_one".to_string(),
                "redundant_cap".to_string(),
            ]),
            ..Default::default()
        };

        let conflict = find_lp_infeasibility_conflict(&p, &LPConflictOptions::default());

        assert!(conflict.infeasible, "{:?}", conflict.message);
        assert!(conflict.minimal, "{:?}", conflict.message);
        assert_eq!(
            conflict.members,
            vec![LPConflictMember::UpperRow(0), LPConflictMember::UpperRow(1)]
        );
        let subsystem = lp_feasibility_problem_from_conflict_members(&p, &conflict.members);
        assert_eq!(
            solve_lp_internal(&subsystem, &opts()).status,
            LPStatus::Infeasible
        );

        for idx in 0..conflict.members.len() {
            let mut trial = conflict.members.clone();
            trial.remove(idx);
            let subsystem = lp_feasibility_problem_from_conflict_members(&p, &trial);
            assert_eq!(
                solve_lp_internal(&subsystem, &opts()).status,
                LPStatus::Optimal
            );
        }
    }

    #[test]
    fn conflict_finder_reports_feasible_models() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0]]),
            b_ub: Some(vec![5.0]),
            ..Default::default()
        };

        let conflict = find_lp_infeasibility_conflict(&p, &LPConflictOptions::default());

        assert!(!conflict.infeasible);
        assert!(!conflict.minimal);
        assert!(conflict.members.is_empty());
    }

    #[test]
    fn feasibility_relaxation_finds_weighted_minimum_violation() {
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![0.0],
            a_ub: Some(vec![vec![1.0], vec![-1.0]]),
            b_ub: Some(vec![0.0, -1.0]),
            lb: Some(vec![None]),
            ub: Some(vec![None]),
            var_names: Some(vec!["x".to_string()]),
            con_names: Some(vec![
                "x_at_most_zero".to_string(),
                "x_at_least_one".to_string(),
            ]),
            ..Default::default()
        };
        let relax = solve_lp_feasibility_relaxation_internal(
            &p,
            &LPFeasRelaxOptions {
                upper_row_penalties: Some(vec![3.0, 1.0]),
                ..Default::default()
            },
        );

        assert_eq!(relax.status, LPStatus::Optimal, "{:?}", relax.message);
        assert!((relax.relaxation_cost - 1.0).abs() < TOL);
        assert!((relax.x[0] - 0.0).abs() < TOL, "x={:?}", relax.x);
        assert_eq!(relax.violations.len(), 1);
        assert_eq!(relax.violations[0].member, LPFeasRelaxMember::UpperRow(1));
        assert!((relax.violations[0].amount - 1.0).abs() < TOL);
    }

    #[test]
    fn maximize_box() {
        // max x + y  s.t.  x ≤ 4, y ≤ 3, x,y ≥ 0  ->  7 at (4,3).
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 3.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert!((sol.objective - 7.0).abs() < TOL, "obj={}", sol.objective);
        assert!((sol.x[0] - 4.0).abs() < TOL, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 3.0).abs() < TOL, "x1={}", sol.x[1]);
    }

    #[test]
    fn internal_simplex_reports_lp_certificates() {
        // max 3x + 4y
        // s.t. x + 2y <= 14, 3x - y >= 0, x - y <= 2
        // has optimum (6, 4). Active rows c0/c2 have duals 7/3 and 2/3.
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 4.0],
            a_ub: Some(vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]]),
            b_ub: Some(vec![14.0, 0.0, 2.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        let dual_ub = sol.dual_ub.as_ref().expect("dual_ub");
        let reduced = sol.reduced_costs.as_ref().expect("reduced_costs");
        assert_close(dual_ub[0], 7.0 / 3.0);
        assert_close(dual_ub[1], 0.0);
        assert_close(dual_ub[2], 2.0 / 3.0);
        assert_close(reduced[0], 0.0);
        assert_close(reduced[1], 0.0);
    }

    #[test]
    fn internal_simplex_reports_basis_statuses() {
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 4.0],
            a_ub: Some(vec![vec![1.0, 2.0], vec![-3.0, 1.0], vec![1.0, -1.0]]),
            b_ub: Some(vec![14.0, 0.0, 2.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert_eq!(
            sol.var_basis.as_deref(),
            Some(&["basic".to_string(), "basic".to_string()][..])
        );
        assert_eq!(
            sol.row_basis.as_deref(),
            Some(
                &[
                    "at_upper".to_string(),
                    "basic".to_string(),
                    "at_upper".to_string()
                ][..]
            )
        );
    }

    #[test]
    fn internal_simplex_reports_equality_dual() {
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            a_eq: Some(vec![vec![1.0]]),
            b_eq: Some(vec![2.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert_close(sol.dual_eq.as_ref().expect("dual_eq")[0], 1.0);
        assert_close(sol.reduced_costs.as_ref().expect("reduced_costs")[0], 0.0);
    }

    #[test]
    fn internal_simplex_reports_bound_reduced_costs() {
        let lower_active = LPProblem {
            sense: Sense::Max,
            c: vec![-1.0],
            a_ub: Some(vec![vec![0.0]]),
            b_ub: Some(vec![0.0]),
            ..Default::default()
        };
        let lower_sol = solve_lp_internal(&lower_active, &opts());
        assert_eq!(lower_sol.status, LPStatus::Optimal);
        assert_close(
            lower_sol.reduced_costs.as_ref().expect("reduced_costs")[0],
            -1.0,
        );

        let upper_active = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            a_ub: Some(vec![vec![0.0]]),
            b_ub: Some(vec![0.0]),
            ub: Some(vec![Some(1.0)]),
            ..Default::default()
        };
        let upper_sol = solve_lp_internal(&upper_active, &opts());
        assert_eq!(upper_sol.status, LPStatus::Optimal);
        assert_close(
            upper_sol.reduced_costs.as_ref().expect("reduced_costs")[0],
            1.0,
        );
    }

    #[test]
    fn solves_source_level_lp_range_rows() {
        let p = GeneralLinearLPProblem {
            base: LPProblem {
                sense: Sense::Max,
                c: vec![3.0, 2.0],
                a_ub: Some(vec![vec![1.0, 0.0]]),
                b_ub: Some(vec![10.0]),
                lb: Some(vec![Some(0.0), Some(0.0)]),
                ub: Some(vec![Some(10.0), Some(10.0)]),
                var_names: Some(vec!["x".to_string(), "y".to_string()]),
                con_names: Some(vec!["x_cap".to_string()]),
                ..Default::default()
            },
            linear_constraints: vec![
                LPRowConstraint {
                    coefs: vec![1.0, 2.0],
                    lower: Some(8.0),
                    upper: Some(8.0),
                    name: Some("balance_eq".to_string()),
                },
                LPRowConstraint {
                    coefs: vec![1.0, -1.0],
                    lower: Some(1.0),
                    upper: None,
                    name: Some("dominance_ge".to_string()),
                },
                LPRowConstraint {
                    coefs: vec![1.0, 1.0],
                    lower: Some(5.0),
                    upper: Some(7.0),
                    name: Some("throughput_range".to_string()),
                },
            ],
        };
        let linearized = linearize_general_linear_lp_problem(&p);
        assert_eq!(linearized.a_eq.as_ref().unwrap().len(), 1);
        assert_eq!(linearized.a_ub.as_ref().unwrap().len(), 4);
        let sol = solve_general_linear_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal, "{sol:?}");
        assert!((sol.objective - 20.0).abs() < TOL, "{sol:?}");
        assert!((sol.x[0] - 6.0).abs() < TOL, "{sol:?}");
        assert!((sol.x[1] - 1.0).abs() < TOL, "{sol:?}");
    }

    #[test]
    fn solves_lp_with_objective_offset() {
        let p = ObjectiveOffsetLPProblem {
            base: LPProblem {
                sense: Sense::Max,
                c: vec![1.0, 1.0],
                a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
                b_ub: Some(vec![4.0, 3.0]),
                ..Default::default()
            },
            objective_offset: 5.5,
        };
        let sol = solve_objective_offset_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert!((sol.objective - 12.5).abs() < TOL, "{sol:?}");
        assert!((sol.x[0] - 4.0).abs() < TOL, "{sol:?}");
        assert!((sol.x[1] - 3.0).abs() < TOL, "{sol:?}");
    }

    #[test]
    fn standard_form_min() {
        // min x + y  s.t.  x + y ≥ 2  (encoded as −x − y ≤ −2), x,y ≥ 0  ->  2.
        // Exercises the b < 0 flip / phase-1 artificial path.
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![-1.0, -1.0]]),
            b_ub: Some(vec![-2.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert!((sol.objective - 2.0).abs() < TOL, "obj={}", sol.objective);
        assert!(sol.x[0] >= -TOL && sol.x[1] >= -TOL);
        assert!((sol.x[0] + sol.x[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn unconstrained_improving_ray_is_unbounded() {
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0],
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Unbounded, "{:?}", sol.message);
        assert!(sol.objective.is_infinite() && sol.objective.is_sign_positive());
    }

    #[test]
    fn equality_constraint() {
        // max 3x + 2y  s.t.  x + y = 4, x ≤ 3, x,y ≥ 0  ->  11 at (3,1).
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 0.0]]),
            b_ub: Some(vec![3.0]),
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![4.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal(&p, &opts());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert!((sol.objective - 11.0).abs() < TOL, "obj={}", sol.objective);
        assert!((sol.x[0] - 3.0).abs() < TOL, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < TOL, "x1={}", sol.x[1]);
    }

    #[test]
    fn interior_point_maximize_box() {
        // Same geometry as `maximize_box`, but through the native primal-dual IPM.
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 3.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective - 7.0).abs() < 1e-5, "obj={}", sol.objective);
        assert!((sol.x[0] - 4.0).abs() < 1e-5, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 3.0).abs() < 1e-5, "x1={}", sol.x[1]);
    }

    #[test]
    fn interior_point_handles_equalities_and_active_bounds() {
        // max 3x + 2y  s.t.  x + y = 4, x ≤ 3, x,y ≥ 0  ->  11 at (3,1).
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a_ub: Some(vec![vec![1.0, 0.0]]),
            b_ub: Some(vec![3.0]),
            a_eq: Some(vec![vec![1.0, 1.0]]),
            b_eq: Some(vec![4.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective - 11.0).abs() < 1e-5, "obj={}", sol.objective);
        assert!((sol.x[0] - 3.0).abs() < 1e-5, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-5, "x1={}", sol.x[1]);
    }

    #[test]
    fn interior_point_min_with_negative_rhs() {
        // min x + y  s.t.  x + y ≥ 2, x,y ≥ 0  -> objective 2.
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![-1.0, -1.0]]),
            b_ub: Some(vec![-2.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective - 2.0).abs() < 1e-5, "obj={}", sol.objective);
        assert!(sol.x[0] >= -1e-6 && sol.x[1] >= -1e-6, "x={:?}", sol.x);
        assert!((sol.x[0] + sol.x[1] - 2.0).abs() < 1e-5, "x={:?}", sol.x);
    }

    #[test]
    fn interior_point_finite_lower_and_upper_bounds() {
        // max 2x + y, with x in [1,4], y in [0,2], x+y≤5 -> (4,1), obj 9.
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![2.0, 1.0],
            a_ub: Some(vec![vec![1.0, 1.0]]),
            b_ub: Some(vec![5.0]),
            lb: Some(vec![Some(1.0), Some(0.0)]),
            ub: Some(vec![Some(4.0), Some(2.0)]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective - 9.0).abs() < 1e-5, "obj={}", sol.objective);
        assert!((sol.x[0] - 4.0).abs() < 1e-5, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-5, "x1={}", sol.x[1]);
    }

    #[test]
    fn interior_point_free_variable_with_two_sided_constraints() {
        // min x, with -2≤x≤3 and x free in the model bounds -> x*=-2.
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0],
            a_ub: Some(vec![vec![1.0], vec![-1.0]]),
            b_ub: Some(vec![3.0, 2.0]),
            lb: Some(vec![None]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective + 2.0).abs() < 1e-5, "obj={}", sol.objective);
        assert!((sol.x[0] + 2.0).abs() < 1e-5, "x0={}", sol.x[0]);
    }

    #[test]
    fn interior_point_transportation_style_lp() {
        // Two plants, two customers. Omit one redundant demand equality so the
        // dense normal equations stay full-row-rank.
        //
        // min 2x11 + 4x12 + 5x21 + x22
        // s.t. x11+x12=20, x21+x22=30, x11+x21=25, x≥0
        // -> x=(20,0,5,25), cost 90.
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![2.0, 4.0, 5.0, 1.0],
            a_eq: Some(vec![
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![1.0, 0.0, 1.0, 0.0],
            ]),
            b_eq: Some(vec![20.0, 30.0, 25.0]),
            ..Default::default()
        };
        let sol = solve_lp_internal_ipm(&p, &InternalInteriorPointOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal, "{:?}", sol.message);
        assert!((sol.objective - 90.0).abs() < 1e-5, "obj={}", sol.objective);
        let expected = [20.0, 0.0, 5.0, 25.0];
        for (i, &want) in expected.iter().enumerate() {
            assert!((sol.x[i] - want).abs() < 1e-5, "x{i}={}", sol.x[i]);
        }
    }

    #[test]
    fn pretty_printer() {
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![1.0, 0.0]]),
            b_ub: Some(vec![4.0]),
            ..Default::default()
        };
        let s = lp_to_string(&p);
        assert!(s.starts_with("max  x0 + x1"));
        assert!(s.contains("s.t."));
    }
}
