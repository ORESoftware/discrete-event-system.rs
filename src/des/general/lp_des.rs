//! Port of `src/des/general/lp-des.ts` — an LP solver implemented as a DES.
//!
//! Two-phase simplex driven as a discrete-event system: simplex walks
//! vertex-to-vertex along the boundary of the feasible polytope, and each PIVOT
//! is a discrete event. The DES runner drives the walk; LP-specific logic is
//! encapsulated in five stationary roles (Observer, PhaseTransition, Entering,
//! Leaving, Pivot). One pivot = one tick. Two phases (phase-1 for feasibility,
//! phase-2 for optimality) share the same DES loop with a different cost row in
//! the tableau. The simulation halts when EnteringStation reports no improving
//! direction (optimality), no positive ratio is available (unboundedness), or
//! the iteration cap is hit.
//!
//! Mapping notes vs. the TypeScript source:
//!   * `interface DESSimplexTrace` -> struct with `Vec` fields; the inline pivot
//!     record object -> `PivotRecord` struct.
//!   * `class SimplexState` -> struct holding the shared mutable tableau + basis
//!     + preprocessing data. Every station reads/writes one shared
//!     `Rc<RefCell<SimplexState>>` (the TS pattern of one external state object
//!     touched by functional-role stations).
//!   * `abstract class SimplexRoleStation extends DESStation` -> each concrete
//!     station embeds `StationCore` and impls the `DESStation` trait; the shared
//!     `hasWork` predicate becomes the free fn `role_has_work`.
//!   * `class Entering/Leaving/Pivot/PhaseTransition/Observer Station` -> structs
//!     `impl DESStation`.
//!   * string-union state `'optimal' | ... | 'in-progress' | 'phase-done'` ->
//!     `SimplexStatus` enum (maps to `lp::LPStatus` at the API boundary).
//!   * `pivotRule: 'dantzig' | 'bland'` -> `PivotRule` enum.
//!   * `phase: 1 | 2` -> `u8` (1 or 2), matching the trace record field.
//!   * `pendingEntering/-Leaving = -1` sentinels -> `Option<usize>`.
//!   * `Set<number>` -> `HashSet<usize>`; `number[][]` -> `Vec<Vec<f64>>`.
//!   * `interface DESSimplexSolution extends LPSolution` -> flattened struct
//!     replicating every `LPSolution` field plus `trace` (no interface
//!     inheritance in Rust; compose by flattening).
//!   * `solveLPViaDES` -> free fn `solve_lp_via_des` plus a `DESSimplexSolver`
//!     `Transform` wrapper (config on the struct, LP as the transform input),
//!     mirroring `lp.rs`'s `InternalSimplexSolver`. Failure is carried in
//!     `status`, never via `throw`/`Result`.
//!   * `Date.now()` timing -> `std::time::Instant` (reported as `elapsed_ms`),
//!     following the precedent set in the ported `lp.rs`.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;

use super::des_base::runner::{run_iterative_des, IterativeRunOptions};
use super::des_base::station::{DESStation, StationCore, StationRef};
use super::lp::{LPProblem, LPStatus, Sense};
use crate::des::shared::transform::Transform;

// -----------------------------------------------------------------------------
// Shared simulation state.
// -----------------------------------------------------------------------------

/// A single recorded pivot. TS inline object in `DESSimplexTrace.pivotHistory`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PivotRecord {
    pub tick: usize,
    /// 1 (phase-1 feasibility) or 2 (phase-2 optimality).
    pub phase: u8,
    pub enter: usize,
    pub leave: usize,
    pub obj: f64,
    pub pivot_elt: f64,
}

/// Per-pivot trace usable for animation / observability. TS `interface DESSimplexTrace`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DESSimplexTrace {
    pub pivot_history: Vec<PivotRecord>,
    pub vertex_history: Vec<Vec<f64>>,
    pub obj_history: Vec<f64>,
}

/// Pivot selection rule. TS `'dantzig' | 'bland'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PivotRule {
    /// Most-negative reduced cost (steepest immediate improvement).
    #[default]
    Dantzig,
    /// First strictly-negative reduced cost (lowest index; anti-cycling).
    Bland,
}

impl PivotRule {
    pub fn as_str(self) -> &'static str {
        match self {
            PivotRule::Dantzig => "dantzig",
            PivotRule::Bland => "bland",
        }
    }
}

/// The status union from TS: the terminal `LPStatus` members plus the two
/// in-flight markers `'in-progress'` and `'phase-done'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimplexStatus {
    InProgress,
    PhaseDone,
    Optimal,
    Infeasible,
    Unbounded,
    IterLimit,
}

/// Shared mutable simulation state. Every station reads / writes this single
/// object (mirrors the TS `class SimplexState`).
struct SimplexState {
    /// Tableau, (m+1) × (ncols+1).
    t: Vec<Vec<f64>>,
    /// Length m; the column index basic in each row.
    basis: Vec<usize>,
    m: usize,
    ncols: usize,
    iters: usize,
    max_iter: usize,
    tol: f64,
    status: SimplexStatus,
    /// 1 or 2.
    phase: u8,
    pivot_rule: PivotRule,
    /// Set by EnteringStation, consumed by LeavingStation then PivotStation.
    pending_entering: Option<usize>,
    pending_leaving: Option<usize>,
    artificial_cols: HashSet<usize>,
    /// Retained for parity with the TS state object; `phase` is derived from it
    /// at construction.
    #[allow(dead_code)]
    has_artificials: bool,
    phase2_cost_row: Vec<f64>,
    n_orig: usize,
    shifts: Vec<f64>,
    y_index_of_pos: Vec<usize>,
    /// `-1` if the variable has no negative twin (matches `lp.rs`).
    free_neg: Vec<isize>,
    sense: Sense,
    trace: DESSimplexTrace,
    snapshot_version: i64,
    observed_snapshot_version: i64,
    skip_selection_this_tick: bool,
}

impl SimplexState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        t: Vec<Vec<f64>>,
        basis: Vec<usize>,
        m: usize,
        ncols: usize,
        max_iter: usize,
        tol: f64,
        phase2_cost_row: Vec<f64>,
        artificial_cols: HashSet<usize>,
        n_orig: usize,
        shifts: Vec<f64>,
        y_index_of_pos: Vec<usize>,
        free_neg: Vec<isize>,
        sense: Sense,
        pivot_rule: PivotRule,
    ) -> Self {
        let has_artificials = !artificial_cols.is_empty();
        let phase = if has_artificials { 1 } else { 2 };
        SimplexState {
            t,
            basis,
            m,
            ncols,
            iters: 0,
            max_iter,
            tol,
            status: SimplexStatus::InProgress,
            phase,
            pivot_rule,
            pending_entering: None,
            pending_leaving: None,
            artificial_cols,
            has_artificials,
            phase2_cost_row,
            n_orig,
            shifts,
            y_index_of_pos,
            free_neg,
            sense,
            trace: DESSimplexTrace::default(),
            snapshot_version: 0,
            observed_snapshot_version: -1,
            skip_selection_this_tick: false,
        }
    }

    /// Reconstruct the original x vector from the current basic feasible solution.
    fn current_vertex(&self) -> Vec<f64> {
        let mut y = vec![0.0; self.ncols];
        for r in 0..self.m {
            y[self.basis[r]] = self.t[r][self.ncols];
        }
        let mut x = vec![0.0; self.n_orig];
        for i in 0..self.n_orig {
            let yp = y[self.y_index_of_pos[i]];
            let yn = if self.free_neg[i] >= 0 {
                y[self.free_neg[i] as usize]
            } else {
                0.0
            };
            x[i] = yp - yn + self.shifts[i];
        }
        x
    }

    /// Current objective in the original (max/min) sense. Phase-1 objective is
    /// the sum-of-artificials, not the user objective, so we return NaN there.
    fn current_objective(&self) -> f64 {
        if self.phase == 1 {
            return f64::NAN;
        }
        let z = self.t[self.m][self.ncols];
        match self.sense {
            Sense::Max => z,
            Sense::Min => -z,
        }
    }

    fn mark_snapshot(&mut self) {
        self.snapshot_version += 1;
    }
}

/// Shared `hasWork` predicate of the TS `abstract class SimplexRoleStation`:
/// there is work while the solve is mid-flight or a phase has just completed.
fn role_has_work(s: &SimplexState) -> bool {
    matches!(s.status, SimplexStatus::InProgress | SimplexStatus::PhaseDone)
}

/// Elementary row operations (Gauss-Jordan on the pivot column). Returns the
/// pivot element. TS free fn `pivotTableau`.
fn pivot_tableau(s: &mut SimplexState, enter: usize, leave: usize) -> f64 {
    let ncols = s.t[0].len();
    let pv = s.t[leave][enter];
    for j in 0..ncols {
        s.t[leave][j] /= pv;
    }
    // Snapshot the normalised pivot row to avoid aliasing the borrow below.
    let prow = s.t[leave].clone();
    let nrows = s.t.len();
    for r in 0..nrows {
        if r == leave {
            continue;
        }
        let factor = s.t[r][enter];
        if factor == 0.0 {
            continue;
        }
        for j in 0..ncols {
            s.t[r][j] -= factor * prow[j];
        }
    }
    s.basis[leave] = enter;
    s.iters += 1;
    s.pending_entering = None;
    s.pending_leaving = None;
    if s.iters >= s.max_iter {
        s.status = SimplexStatus::IterLimit;
    }
    pv
}

// -----------------------------------------------------------------------------
// Stations. Each is a functional role; the per-tick orchestration stays inside
// the DES runner.
// -----------------------------------------------------------------------------

/// Records the current vertex + objective into the trace (read-only snapshot
/// for downstream observability/animation, like Census in the SEIR model).
struct ObserverStation {
    core: StationCore,
    state: Rc<RefCell<SimplexState>>,
}

impl DESStation for ObserverStation {
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
        let s = self.state.borrow();
        role_has_work(&s) || s.observed_snapshot_version < s.snapshot_version
    }
    fn run_time_step(&mut self) {
        let mut s = self.state.borrow_mut();
        if s.observed_snapshot_version == s.snapshot_version {
            return;
        }
        let obj = s.current_objective();
        let vertex = s.current_vertex();
        s.trace.obj_history.push(obj);
        s.trace.vertex_history.push(vertex);
        s.observed_snapshot_version = s.snapshot_version;
    }
}

/// Resolves end-of-phase: declares optimality (phase 2) or transitions from
/// phase 1 to phase 2 (or infeasibility).
struct PhaseTransitionStation {
    core: StationCore,
    state: Rc<RefCell<SimplexState>>,
}

impl DESStation for PhaseTransitionStation {
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
        role_has_work(&self.state.borrow())
    }
    fn run_time_step(&mut self) {
        let mut s = self.state.borrow_mut();
        if s.status != SimplexStatus::PhaseDone {
            return;
        }
        if s.phase == 2 {
            s.status = SimplexStatus::Optimal;
            return;
        }

        let mut sum_art = 0.0;
        for r in 0..s.m {
            if s.artificial_cols.contains(&s.basis[r]) {
                sum_art += s.t[r][s.ncols];
            }
        }
        if sum_art > 1e-7 {
            s.status = SimplexStatus::Infeasible;
            return;
        }

        // Drive any artificials still in the basis (value 0) out of the basis.
        for r in 0..s.m {
            if !s.artificial_cols.contains(&s.basis[r]) {
                continue;
            }
            for j in 0..s.ncols {
                if !s.artificial_cols.contains(&j) && s.t[r][j].abs() > s.tol {
                    pivot_tableau(&mut s, j, r); // &mut RefMut auto-derefs to &mut SimplexState
                    if s.iters >= s.max_iter {
                        return;
                    }
                    break;
                }
            }
        }

        // Install the phase-2 cost row, row-reduced against the basic columns.
        let (m, ncols) = (s.m, s.ncols);
        s.t[m] = s.phase2_cost_row.clone();
        for r in 0..m {
            let cb = s.t[m][s.basis[r]];
            if cb != 0.0 {
                let row_r = s.t[r].clone();
                for j in 0..=ncols {
                    s.t[m][j] -= cb * row_r[j];
                }
            }
        }
        s.phase = 2;
        s.status = SimplexStatus::InProgress;
        s.skip_selection_this_tick = true;
        s.mark_snapshot();
    }
}

/// Scans the cost row for an entering column (negative reduced cost), using the
/// configured pivot rule.
struct EnteringStation {
    core: StationCore,
    state: Rc<RefCell<SimplexState>>,
}

impl DESStation for EnteringStation {
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
        role_has_work(&self.state.borrow())
    }
    fn run_time_step(&mut self) {
        let mut s = self.state.borrow_mut();
        if s.status != SimplexStatus::InProgress {
            return;
        }
        if s.skip_selection_this_tick {
            s.skip_selection_this_tick = false;
            return;
        }
        let m = s.m;
        let ncols = s.ncols;
        let mut entering: Option<usize> = None;
        match s.pivot_rule {
            PivotRule::Dantzig => {
                let mut best_rc = -s.tol;
                for j in 0..ncols {
                    if s.artificial_cols.contains(&j) {
                        continue; // artificials never re-enter, in either phase
                    }
                    if s.t[m][j] < best_rc {
                        best_rc = s.t[m][j];
                        entering = Some(j);
                    }
                }
            }
            PivotRule::Bland => {
                for j in 0..ncols {
                    if s.artificial_cols.contains(&j) {
                        continue;
                    }
                    if s.t[m][j] < -s.tol {
                        entering = Some(j);
                        break;
                    }
                }
            }
        }
        match entering {
            None => {
                // No improving direction → optimal for the current phase.
                s.pending_entering = None;
                s.status = SimplexStatus::PhaseDone;
            }
            Some(e) => {
                s.pending_entering = Some(e);
            }
        }
    }
}

/// Min-ratio test on the entering column (Bland tie-break on lowest basis index).
struct LeavingStation {
    core: StationCore,
    state: Rc<RefCell<SimplexState>>,
}

impl DESStation for LeavingStation {
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
        role_has_work(&self.state.borrow())
    }
    fn run_time_step(&mut self) {
        let mut s = self.state.borrow_mut();
        let entering = match s.pending_entering {
            None => return,
            Some(e) => e,
        };
        let m = s.m;
        let ncols = s.ncols;
        let mut leaving: Option<usize> = None;
        let mut best_ratio = f64::INFINITY;
        for r in 0..m {
            if s.t[r][entering] > s.tol {
                let ratio = s.t[r][ncols] / s.t[r][entering];
                if ratio < best_ratio - s.tol
                    || ((ratio - best_ratio).abs() <= s.tol
                        && (leaving.is_none() || s.basis[r] < s.basis[leaving.unwrap()]))
                {
                    best_ratio = ratio;
                    leaving = Some(r);
                }
            }
        }
        match leaving {
            None => {
                // No positive entry in the entering column ⇒ unbounded direction.
                s.status = SimplexStatus::Unbounded;
                s.pending_leaving = None;
            }
            Some(l) => {
                s.pending_leaving = Some(l);
            }
        }
    }
}

/// Performs the elementary row operations for the pending pivot, then records
/// it into the trace.
struct PivotStation {
    core: StationCore,
    state: Rc<RefCell<SimplexState>>,
}

impl DESStation for PivotStation {
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
        role_has_work(&self.state.borrow())
    }
    fn run_time_step(&mut self) {
        let mut s = self.state.borrow_mut();
        let (enter, leave) = match (s.pending_entering, s.pending_leaving) {
            (Some(e), Some(l)) => (e, l),
            _ => return,
        };
        let pivot_elt = pivot_tableau(&mut s, enter, leave);
        s.mark_snapshot();
        let tick = s.iters;
        let phase = s.phase;
        let obj = s.current_objective();
        s.trace.pivot_history.push(PivotRecord {
            tick,
            phase,
            enter,
            leave,
            obj,
            pivot_elt,
        });
    }
}

// -----------------------------------------------------------------------------
// Build initial tableau in y-space (preprocess: shift bounds, split free
// variables, add slacks / surpluses / artificials, fix b ≥ 0, install phase-1
// cost row).
// -----------------------------------------------------------------------------

struct Preprocessed {
    t: Vec<Vec<f64>>,
    basis: Vec<usize>,
    m: usize,
    ncols: usize,
    artificial_cols: HashSet<usize>,
    phase2_cost_row: Vec<f64>,
    n_orig: usize,
    shifts: Vec<f64>,
    y_index_of_pos: Vec<usize>,
    free_neg: Vec<isize>,
}

fn preprocess(p: &LPProblem) -> Preprocessed {
    let n = p.c.len();
    let a_ub: &[Vec<f64>] = p.a_ub.as_deref().unwrap_or(&[]);
    let b_ub: &[f64] = p.b_ub.as_deref().unwrap_or(&[]);
    let a_eq: &[Vec<f64>] = p.a_eq.as_deref().unwrap_or(&[]);
    let b_eq: &[f64] = p.b_eq.as_deref().unwrap_or(&[]);
    let lb: Vec<Option<f64>> = p.lb.clone().unwrap_or_else(|| vec![Some(0.0); n]);
    let ub: Vec<Option<f64>> = p.ub.clone().unwrap_or_else(|| vec![None; n]);

    // Shift / split: x_i = (y_pos − y_neg) + shift_i.
    let mut shifts: Vec<f64> = vec![0.0; n];
    let mut free_neg: Vec<isize> = Vec::with_capacity(n);
    let mut y_index_of_pos: Vec<usize> = Vec::with_capacity(n);
    let mut y_count: usize = 0;
    for i in 0..n {
        match lb[i] {
            None => {
                y_index_of_pos.push(y_count);
                y_count += 1;
                free_neg.push(y_count as isize);
                y_count += 1;
                shifts[i] = 0.0;
            }
            Some(l) => {
                y_index_of_pos.push(y_count);
                y_count += 1;
                free_neg.push(-1);
                shifts[i] = l;
            }
        }
    }
    let ny = y_count;

    // Cost in y-space (we maximise z = c^T y; minimisation handled by caller).
    let mut c_y: Vec<f64> = vec![0.0; ny];
    for i in 0..n {
        c_y[y_index_of_pos[i]] += p.c[i];
        if free_neg[i] >= 0 {
            c_y[free_neg[i] as usize] += -p.c[i];
        }
    }

    // Build A y ≤ b' and A y = b'' lists.
    let mut ay: Vec<Vec<f64>> = Vec::new();
    let mut by: Vec<f64> = Vec::new();
    let mut eq_rows: Vec<bool> = Vec::new();
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
        eq_rows.push(false);
    }
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
        ay.push(row);
        by.push(rhs);
        eq_rows.push(true);
    }
    // Upper bounds → extra ≤ rows.
    for i in 0..n {
        if let Some(u) = ub[i] {
            let mut row = vec![0.0; ny];
            row[y_index_of_pos[i]] = 1.0;
            if free_neg[i] >= 0 {
                row[free_neg[i] as usize] = -1.0;
            }
            ay.push(row);
            by.push(u - shifts[i]);
            eq_rows.push(false);
        }
    }
    let m = ay.len();

    // Sign-fix b ≥ 0 by flipping rows. slackSign: +1 ≤ row, −1 ≥ row, 0 equality.
    let mut slack_sign: Vec<isize> = Vec::with_capacity(m);
    for r in 0..m {
        if by[r] < 0.0 {
            for j in 0..ny {
                ay[r][j] = -ay[r][j];
            }
            by[r] = -by[r];
            slack_sign.push(if eq_rows[r] { 0 } else { -1 });
        } else {
            slack_sign.push(if eq_rows[r] { 0 } else { 1 });
        }
    }

    // Allocate columns for slacks/surpluses (one per row) and artificials.
    let ny_total = ny + m; // y + slacks/surpluses
    let mut artificial_cols: HashSet<usize> = HashSet::new();
    let mut art_count = 0usize;
    let mut art_col: Vec<isize> = Vec::with_capacity(m); // -1 if no artificial
    for r in 0..m {
        if slack_sign[r] == 1 {
            art_col.push(-1);
        } else {
            let c = ny_total + art_count;
            art_col.push(c as isize);
            artificial_cols.insert(c);
            art_count += 1;
        }
    }
    let total_cols = ny_total + art_count;

    // Build the tableau: m+1 rows × (total_cols+1).
    let mut t: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    for r in 0..m {
        let mut row = vec![0.0; total_cols + 1];
        for j in 0..ny {
            row[j] = ay[r][j];
        }
        if slack_sign[r] == 1 {
            row[ny + r] = 1.0;
        } else if slack_sign[r] == -1 {
            row[ny + r] = -1.0; // surplus
        }
        if art_col[r] >= 0 {
            row[art_col[r] as usize] = 1.0;
        }
        row[total_cols] = by[r];
        t.push(row);
    }

    // Initial basis: artificial if present, else slack.
    let mut basis: Vec<usize> = Vec::with_capacity(m);
    for r in 0..m {
        basis.push(if art_col[r] >= 0 {
            art_col[r] as usize
        } else {
            ny + r
        });
    }

    // Phase-1 cost row: minimise sum of artificials.
    let mut phase1_cost = vec![0.0; total_cols];
    for &c in &artificial_cols {
        phase1_cost[c] = 1.0;
    }
    let mut cost_row = vec![0.0; total_cols + 1];
    for j in 0..total_cols {
        cost_row[j] = phase1_cost[j];
    }
    // Row-reduce: subtract row r from cost row for each artificial in basis.
    for r in 0..m {
        if art_col[r] >= 0 {
            for j in 0..=total_cols {
                cost_row[j] -= t[r][j];
            }
        }
    }
    t.push(cost_row);

    // Phase-2 cost row: c2[j] = -c[j] for original y columns, 0 elsewhere; RHS 0.
    let mut phase2_cost = vec![0.0; total_cols];
    for j in 0..ny {
        phase2_cost[j] = -c_y[j];
    }
    let mut phase2_cost_row = phase2_cost;
    phase2_cost_row.push(0.0);

    Preprocessed {
        t,
        basis,
        m,
        ncols: total_cols,
        artificial_cols,
        phase2_cost_row,
        n_orig: n,
        shifts,
        y_index_of_pos,
        free_neg,
    }
}

// -----------------------------------------------------------------------------
// Public API.
// -----------------------------------------------------------------------------

/// Options for [`solve_lp_via_des`] / [`DESSimplexSolver`]. TS `interface DESSimplexOptions`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DESSimplexOptions {
    /// `Dantzig` (default; steepest reduced cost) or `Bland` (anti-cycling).
    pub pivot_rule: Option<PivotRule>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

/// Result of a DES-driven solve. Flattens every `lp::LPSolution` field (TS
/// `interface DESSimplexSolution extends LPSolution`) and adds the pivot `trace`.
#[derive(Clone, Debug, PartialEq)]
pub struct DESSimplexSolution {
    pub status: LPStatus,
    pub x: Vec<f64>,
    pub objective: f64,
    pub dual_ub: Option<Vec<f64>>,
    pub dual_eq: Option<Vec<f64>>,
    pub reduced_costs: Option<Vec<f64>>,
    pub iters: Option<usize>,
    pub solver: String,
    pub elapsed_ms: f64,
    pub message: Option<String>,
    /// Per-pivot trace usable for animation / observability.
    pub trace: DESSimplexTrace,
}

/// Solve an LP using the DES engine. Each pivot is one tick; the algorithm walks
/// vertex-to-vertex along the steepest improving edge of the feasible polytope,
/// exactly the way classical simplex does — but the orchestration goes through
/// the DES tick loop and five stations.
pub fn solve_lp_via_des(p: &LPProblem, opts: &DESSimplexOptions) -> DESSimplexSolution {
    let t0 = Instant::now();
    let tol = opts.tol.unwrap_or(1e-9);
    let max_iter = opts.max_iter.unwrap_or(5000);
    let pivot_rule = opts.pivot_rule.unwrap_or(PivotRule::Dantzig);

    // Convert min → max by flipping the objective.
    let sense = p.sense;
    let c: Vec<f64> = if sense == Sense::Max {
        p.c.clone()
    } else {
        p.c.iter().map(|v| -v).collect()
    };
    let lp_working = LPProblem {
        sense: Sense::Max,
        c,
        ..p.clone()
    };

    let pp = preprocess(&lp_working);

    let state = Rc::new(RefCell::new(SimplexState::new(
        pp.t,
        pp.basis,
        pp.m,
        pp.ncols,
        max_iter,
        tol,
        pp.phase2_cost_row,
        pp.artificial_cols,
        pp.n_orig,
        pp.shifts,
        pp.y_index_of_pos,
        pp.free_neg,
        sense,
        pivot_rule,
    )));

    // No artificials → skip phase 1; install phase-2 cost row, row-reduced.
    {
        let mut s = state.borrow_mut();
        if s.artificial_cols.is_empty() {
            let (m, ncols) = (s.m, s.ncols);
            s.t[m] = s.phase2_cost_row.clone();
            for r in 0..m {
                let cb = s.t[m][s.basis[r]];
                if cb != 0.0 {
                    let row_r = s.t[r].clone();
                    for j in 0..=ncols {
                        s.t[m][j] -= cb * row_r[j];
                    }
                }
            }
            s.phase = 2;
        }
    }

    let max_ticks = max_iter + state.borrow().m + 10;

    // Build the simplex stations sharing the single tableau state.
    let observer: StationRef = Rc::new(RefCell::new(ObserverStation {
        core: StationCore::new("lp-observer-station"),
        state: state.clone(),
    }));
    let phase: StationRef = Rc::new(RefCell::new(PhaseTransitionStation {
        core: StationCore::new("lp-phase-transition-station"),
        state: state.clone(),
    }));
    let entering: StationRef = Rc::new(RefCell::new(EnteringStation {
        core: StationCore::new("lp-entering-station"),
        state: state.clone(),
    }));
    let leaving: StationRef = Rc::new(RefCell::new(LeavingStation {
        core: StationCore::new("lp-leaving-station"),
        state: state.clone(),
    }));
    let pivot: StationRef = Rc::new(RefCell::new(PivotStation {
        core: StationCore::new("lp-pivot-station"),
        state: state.clone(),
    }));

    run_iterative_des(
        vec![observer, phase, entering, leaving, pivot],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            run_validators: false,
            ..Default::default()
        },
    );

    let s = state.borrow();
    let sol_status = match s.status {
        SimplexStatus::Optimal => LPStatus::Optimal,
        SimplexStatus::Infeasible => LPStatus::Infeasible,
        SimplexStatus::Unbounded => LPStatus::Unbounded,
        SimplexStatus::IterLimit => LPStatus::IterLimit,
        _ => LPStatus::NumericalError,
    };
    let solver = format!("des-simplex({})", pivot_rule.as_str());

    if sol_status != LPStatus::Optimal {
        let message = match sol_status {
            LPStatus::Unbounded => "unbounded direction at LeavingStation".to_string(),
            LPStatus::Infeasible => "phase-1 sum of artificials > 0".to_string(),
            _ => String::new(),
        };
        return DESSimplexSolution {
            status: sol_status,
            x: Vec::new(),
            objective: f64::NAN,
            dual_ub: None,
            dual_eq: None,
            reduced_costs: None,
            iters: Some(s.iters),
            solver,
            elapsed_ms: ms_since(t0),
            message: Some(message),
            trace: s.trace.clone(),
        };
    }

    let x = s.current_vertex();
    let mut obj = 0.0;
    for i in 0..p.c.len() {
        obj += p.c[i] * x[i];
    }
    let message = format!(
        "DES simplex: {} pivots across {}",
        s.trace.pivot_history.len(),
        if s.phase == 2 { "two phases" } else { "phase 2 only" }
    );
    DESSimplexSolution {
        status: LPStatus::Optimal,
        x,
        objective: obj,
        dual_ub: None,
        dual_eq: None,
        reduced_costs: None,
        iters: Some(s.iters),
        solver,
        elapsed_ms: ms_since(t0),
        message: Some(message),
        trace: s.trace.clone(),
    }
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// `Transform` wrapper over [`solve_lp_via_des`] (config on the struct, LP as the
/// transform input), mirroring `lp.rs`'s `InternalSimplexSolver`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DESSimplexSolver {
    pub opts: DESSimplexOptions,
}

impl DESSimplexSolver {
    pub fn new(opts: DESSimplexOptions) -> Self {
        DESSimplexSolver { opts }
    }
}

impl Transform<LPProblem, DESSimplexSolution> for DESSimplexSolver {
    fn transform(&self, input: LPProblem) -> DESSimplexSolution {
        solve_lp_via_des(&input, &self.opts)
    }
}

// -----------------------------------------------------------------------------
// Tests — solve a tiny LP as a DES and match the lp.rs direct solve / known optimum.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::lp::{solve_lp_internal, InternalSimplexOptions};

    const TOL: f64 = 1e-6;

    #[test]
    fn maximize_box_matches_known_optimum() {
        // max x + y  s.t.  x ≤ 4, y ≤ 3, x,y ≥ 0  ->  7 at (4,3).
        let p = LPProblem {
            sense: Sense::Max,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            b_ub: Some(vec![4.0, 3.0]),
            ..Default::default()
        };
        let sol = solve_lp_via_des(&p, &DESSimplexOptions::default());
        assert_eq!(sol.status, LPStatus::Optimal);
        assert!((sol.objective - 7.0).abs() < TOL, "obj={}", sol.objective);
        assert!((sol.x[0] - 4.0).abs() < TOL, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 3.0).abs() < TOL, "x1={}", sol.x[1]);
        // The pivot trace should have recorded at least the initial vertex.
        assert!(!sol.trace.vertex_history.is_empty());
    }

    #[test]
    fn min_with_ge_constraint_matches_internal_simplex() {
        // min x + y  s.t.  x + y ≥ 2  (encoded as −x − y ≤ −2), x,y ≥ 0  ->  2.
        // Exercises the b < 0 flip / phase-1 artificial path.
        let p = LPProblem {
            sense: Sense::Min,
            c: vec![1.0, 1.0],
            a_ub: Some(vec![vec![-1.0, -1.0]]),
            b_ub: Some(vec![-2.0]),
            ..Default::default()
        };
        let des = solve_lp_via_des(&p, &DESSimplexOptions::default());
        let direct = solve_lp_internal(&p, &InternalSimplexOptions::default());
        assert_eq!(des.status, LPStatus::Optimal);
        assert_eq!(direct.status, LPStatus::Optimal);
        assert!((des.objective - 2.0).abs() < TOL, "obj={}", des.objective);
        assert!(
            (des.objective - direct.objective).abs() < TOL,
            "des={} direct={}",
            des.objective,
            direct.objective
        );
        assert!((des.x[0] + des.x[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn equality_constraint_matches_internal_simplex() {
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
        let des = solve_lp_via_des(&p, &DESSimplexOptions { pivot_rule: Some(PivotRule::Bland), ..Default::default() });
        let direct = solve_lp_internal(&p, &InternalSimplexOptions::default());
        assert_eq!(des.status, LPStatus::Optimal);
        assert!((des.objective - 11.0).abs() < TOL, "obj={}", des.objective);
        assert!((des.objective - direct.objective).abs() < TOL);
        assert!((des.x[0] - 3.0).abs() < TOL, "x0={}", des.x[0]);
        assert!((des.x[1] - 1.0).abs() < TOL, "x1={}", des.x[1]);
    }
}
