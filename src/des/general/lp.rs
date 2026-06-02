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
    let mut acopy: Vec<Vec<f64>> = ay.iter().cloned().collect();
    let mut bcopy: Vec<f64> = by.clone();
    let total_cols = ny + m;
    let slack_start = ny;
    // Marker convention for artificials: -1 = none, -2 = needs one (resolved below).
    let mut artificial_cols: Vec<isize> = Vec::with_capacity(m);
    for r in 0..m {
        if bcopy[r] < 0.0 {
            for j in 0..ny {
                acopy[r][j] = -acopy[r][j];
            }
            bcopy[r] = -bcopy[r];
            artificial_cols.push(-2);
        } else {
            artificial_cols.push(-1);
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

    LPSolution {
        status: LPStatus::Optimal,
        x,
        objective: obj,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        iters: Some(iters),
        solver: "internal".to_string(),
        elapsed_ms: ms_since(t0),
        message: Some(format!("internal simplex: phase1+phase2, {iters} iters")),
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

/// Default path to the repository-local Python LP bridge. The bridge prefers
/// SciPy/HiGHS when installed and falls back to dependency-free vertex
/// enumeration for small validation models.
const DEFAULT_SCRIPT: &str = "scripts/lp_solve.py";

/// Configuration for the external scipy bridge. TS `interface ExternalSolverOptions`.
///
/// `method` is modelled as a free `String` (rather than a closed enum) to
/// faithfully reproduce the TS behaviour where `LP_SOLVER=scipy:<anything>`
/// passes `<anything>` straight through to scipy. `max_buffer` is accepted for
/// API parity but unused: `std::process::Command` captures the full output.
#[derive(Clone, Debug, Default)]
pub struct ExternalSolverOptions {
    /// scipy linprog method: `"highs"`, `"highs-ds"`, `"highs-ipm"`, `"simplex"`, `"interior-point"`. Default `"highs"`.
    pub method: Option<String>,
    /// Override the python executable. Defaults to `PYTHON` env var or `"python3"`.
    pub python: Option<String>,
    /// Override the script path. Defaults to `scripts/lp_solve.py`.
    pub script: Option<String>,
    /// Accepted for parity with the TS `maxBuffer`; unused in the Rust port.
    pub max_buffer: Option<usize>,
}

/// External scipy.optimize.linprog dispatcher as a transform. Returns status
/// `NumericalError` if scipy / python is unavailable (or the process fails /
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
        let python = self
            .opts
            .python
            .clone()
            .or_else(|| std::env::var("PYTHON").ok())
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
            iters: None,
            solver: format!("scipy:{method}"),
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
                eprintln!("[lp.external] scipy:{method} could not start ({python}): {e}");
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
                eprintln!("[lp.external] scipy:{method} wait failed: {e}");
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
            eprintln!("[lp.external] scipy:{method} process exited with code {code}: {stderr}");
            return numerical_error(format!("external solver exited with {code}: {stderr}"), t0);
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed = match json_parse(&stdout) {
            Ok(v) => v,
            Err(e) => {
                let head: String = stdout.chars().take(120).collect();
                eprintln!(
                    "[lp.external] could not parse scipy:{method} stdout as JSON: {e}; stdout head=\"{head}\""
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

        LPSolution {
            status,
            x,
            objective,
            dual_ub: json_get(&parsed, "dualUB").and_then(json_as_f64_array),
            dual_eq: json_get(&parsed, "dualEQ").and_then(json_as_f64_array),
            reduced_costs: json_get(&parsed, "reducedCosts").and_then(json_as_f64_array),
            iters: json_get(&parsed, "iters")
                .and_then(json_as_f64)
                .map(|f| f as usize),
            solver: format!("scipy:{method}"),
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
