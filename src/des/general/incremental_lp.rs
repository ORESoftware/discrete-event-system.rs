//! Port of `src/des/general/incremental-lp.ts` — INCREMENTAL Linear Programming
//! solver expressed as a discrete-event SYSTEM. Each pivot is one tick; the
//! solver maintains a warm-startable (parametric) simplex basis across LP
//! modifications:
//!
//!    • Add constraint  a·x ≤ b
//!    • Remove constraint by index
//!    • Change the objective vector c
//!    • Add a new variable (with its column and objective coefficient)
//!    • Remove a variable
//!
//! Every modification breaks at most ONE simplex invariant (primal feasibility
//! `x_B ≥ 0`, dual feasibility `c̄_N ≤ 0` for max), and the right flavour of
//! simplex (primal or dual) is restarted to restore the broken invariant while
//! preserving the other.
//!
//! TABLEAU CONVENTION (max-form throughout):
//!   columns `0 … num_struct-1`            : structural variables x_1..x_n
//!   columns `num_struct … num_struct+m-1` : slack variables       s_1..s_m
//!   column  `num_struct+m`                : RHS column             (last)
//!   row 0                                 : z-row (reduced costs; rightmost = z)
//!   rows `1..=m`                          : constraint rows
//!   `basis[i]`                            : column index basic in row `i+1`
//!
//!   Initial basis = all slack columns (x = 0, z = 0, slacks = b), feasible iff
//!   every `b_i ≥ 0`, which is required at construction (Phase-1 / Big-M is out
//!   of scope here). For min-LPs callers negate `c`/flip `z`; `sense_sign`
//!   handles this transparently.
//!
//! This file was self-contained in TypeScript (no imports). It therefore relies
//! only on `std`; all LP types are defined locally.

const EPS: f64 = 1e-9;

/// Optimisation sense. `Max` keeps `sense_sign = +1`; `Min` uses `-1` (the LP is
/// solved internally as a max LP by negating `c`, then `get_z()` flips back).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Max,
    Min,
}

/// Mode emitted by a single pivot attempt (`PivotEvent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotMode {
    Primal,
    Dual,
    Optimal,
    Infeasible,
    Unbounded,
    Idle,
}

/// Persistent solver status. Mirrors `PivotMode` minus the transient `Idle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus {
    Primal,
    Dual,
    Optimal,
    Infeasible,
    Unbounded,
}

// -----------------------------------------------------------------------------
// MOVABLES — modification events flowing into the LP tableau.
// -----------------------------------------------------------------------------

/// Discriminated union of modification events (`TS LPEvent`).
///
/// `tick` is the conceptual scheduling key carried by the movable; it is not
/// consumed by `apply_event` (kept as `f64` to allow fractional schedules).
#[derive(Clone, Debug, PartialEq)]
pub enum LPEvent {
    AddConstraint {
        tick: f64,
        coefs: Vec<f64>,
        rhs: f64,
        name: Option<String>,
    },
    RemoveConstraint {
        tick: f64,
        index: usize,
        name: Option<String>,
    },
    ChangeObjective {
        tick: f64,
        new_c: Vec<f64>,
        name: Option<String>,
    },
    AddVariable {
        tick: f64,
        column: Vec<f64>,
        c_new: f64,
        name: Option<String>,
    },
    RemoveVariable {
        tick: f64,
        struct_index: usize,
        name: Option<String>,
    },
}

/// Emitted by a pivot, recorded for the trace.
#[derive(Clone, Debug, PartialEq)]
pub struct PivotEvent {
    pub tick: usize,
    pub mode: PivotMode,
    /// Entering column index.
    pub entering: Option<usize>,
    /// Leaving row index (`1..=m`).
    pub leaving: Option<usize>,
    pub entering_name: Option<String>,
    pub leaving_name: Option<String>,
}

impl PivotEvent {
    fn terminal(tick: usize, mode: PivotMode) -> Self {
        PivotEvent {
            tick,
            mode,
            entering: None,
            leaving: None,
            entering_name: None,
            leaving_name: None,
        }
    }
}

// -----------------------------------------------------------------------------
// PROBLEM SETUP
// -----------------------------------------------------------------------------

/// Construction parameters for [`IncrementalLP`].
#[derive(Clone, Debug)]
pub struct IncrementalLPInit {
    pub sense: Sense,
    /// Initial objective. Length `num_struct`.
    pub c: Vec<f64>,
    /// Initial constraints `A·x ≤ b`. Each row length `num_struct`.
    pub a: Vec<Vec<f64>>,
    /// Initial RHS. Length `= a.len()`. Must be non-negative for warm-start.
    pub b: Vec<f64>,
    /// Optional names.
    pub var_names: Option<Vec<String>>,
    pub con_names: Option<Vec<String>>,
}

/// Inspection snapshot at a given tick.
#[derive(Clone, Debug)]
pub struct LPSnapshot {
    pub tick: usize,
    pub num_struct: usize,
    pub num_constraints: usize,
    /// Length m, column indices.
    pub basis: Vec<usize>,
    /// Length `num_struct`.
    pub x: Vec<f64>,
    /// Length m (slack values).
    pub slacks: Vec<f64>,
    /// Current objective value (in original sense).
    pub z: f64,
    /// Length `num_struct + m` (row 0 excluding rhs).
    pub reduced_costs: Vec<f64>,
    /// Length m (current basic feasible values).
    pub rhs: Vec<f64>,
    pub primal_feasible: bool,
    pub dual_feasible: bool,
    pub is_optimal: bool,
    pub var_names: Vec<String>,
    pub con_names: Vec<String>,
    /// The event applied at this tick (if any).
    pub applied_event: Option<LPEvent>,
    /// The pivot that fired at this tick (if any).
    pub pivot: Option<PivotEvent>,
    /// Mode the solver is currently in.
    pub mode: SolverStatus,
}

// -----------------------------------------------------------------------------
// SOLVER
// -----------------------------------------------------------------------------

/// Warm-startable incremental LP solver. See module docs for the tableau layout.
#[derive(Clone, Debug)]
pub struct IncrementalLP {
    /// Tableau: shape `(m+1) × (num_struct + m + 1)`. Last column is rhs; row 0
    /// is the z-row (reduced costs in the variable cells, current z in rhs cell).
    pub tab: Vec<Vec<f64>>,
    /// Length m, each entry is a column index.
    pub basis: Vec<usize>,
    pub num_struct: usize,
    /// `+1.0` for max, `-1.0` for min. We always solve internally as a max LP by
    /// negating `c` at construction; `get_z()` flips the sign back.
    pub sense_sign: f64,
    pub var_names: Vec<String>,
    pub con_names: Vec<String>,
    /// Detected status.
    pub status: SolverStatus,
    /// Current tick counter; advances by exactly one each `step()`.
    pub tick: usize,
}

impl IncrementalLP {
    // ---------------------------------------------------------------------
    pub fn new(init: IncrementalLPInit) -> Self {
        let n = init.c.len();
        let m = init.a.len();
        if m != init.b.len() {
            panic!("A.len() ({}) ≠ b.len() ({})", m, init.b.len());
        }
        for i in 0..m {
            if init.a[i].len() != n {
                panic!("A[{}].len() ({}) ≠ c.len() ({})", i, init.a[i].len(), n);
            }
            if init.b[i] < -EPS {
                panic!(
                    "b[{}] = {} < 0; warm-start requires non-negative RHS (use Phase-1 elsewhere)",
                    i, init.b[i]
                );
            }
        }
        let num_struct = n;
        let sense_sign = if init.sense == Sense::Max { 1.0 } else { -1.0 };

        let mut var_names = init.var_names.clone().unwrap_or_default();
        while var_names.len() < n {
            var_names.push(format!("x{}", var_names.len() + 1));
        }
        let mut con_names = init.con_names.clone().unwrap_or_default();
        while con_names.len() < m {
            con_names.push(format!("c{}", con_names.len() + 1));
        }

        // Build the tableau. n_struct + m slacks + 1 rhs.
        let total_cols = n + m + 1;
        let mut tab: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        // Row 0 (z-row): reduced cost initial z_j − c_j = −c_j for non-basic
        // structural; 0 for basic slacks; 0 for rhs (z = 0).
        let mut z = vec![0.0f64; total_cols];
        for j in 0..n {
            z[j] = -sense_sign * init.c[j];
        }
        tab.push(z);
        // Constraint rows: [A_i | I row | b_i]
        for i in 0..m {
            let mut row = vec![0.0f64; total_cols];
            for j in 0..n {
                row[j] = init.a[i][j];
            }
            row[n + i] = 1.0; // slack column
            row[total_cols - 1] = init.b[i];
            tab.push(row);
        }
        // initial basis = slacks
        let basis: Vec<usize> = (0..m).map(|i| n + i).collect();

        let mut lp = IncrementalLP {
            tab,
            basis,
            num_struct,
            sense_sign,
            var_names,
            con_names,
            status: SolverStatus::Primal,
            tick: 0,
        };
        lp.refresh_status();
        lp
    }

    // ---------------------------------------------------------------------
    // CORE PIVOT MACHINERY
    // ---------------------------------------------------------------------

    /// Primal pivot: most-negative-reduced-cost rule for entering, min-ratio for
    /// leaving. Returns the pivot info or a terminal status.
    fn primal_pivot(&mut self) -> PivotEvent {
        let m = self.tab.len() - 1;
        let total_cols = self.tab[0].len();
        let rhs_col = total_cols - 1;
        // Entering: most negative reduced cost (Dantzig's rule).
        let mut entering: Option<usize> = None;
        let mut most_neg = -EPS;
        for j in 0..rhs_col {
            if self.tab[0][j] < most_neg {
                most_neg = self.tab[0][j];
                entering = Some(j);
            }
        }
        let entering = match entering {
            Some(e) => e,
            None => return PivotEvent::terminal(self.tick, PivotMode::Optimal),
        };
        // Leaving: min ratio.
        let mut leaving: Option<usize> = None;
        let mut min_ratio = f64::INFINITY;
        for i in 1..=m {
            if self.tab[i][entering] > EPS {
                let ratio = self.tab[i][rhs_col] / self.tab[i][entering];
                let tie_better = (ratio - min_ratio).abs() < EPS
                    && match leaving {
                        None => true,
                        Some(l) => self.basis[i - 1] < self.basis[l - 1],
                    };
                if ratio < min_ratio - EPS || tie_better {
                    min_ratio = ratio;
                    leaving = Some(i);
                }
            }
        }
        let leaving = match leaving {
            Some(l) => l,
            None => return PivotEvent::terminal(self.tick, PivotMode::Unbounded),
        };
        // capture BEFORE the pivot updates the basis
        let leaving_col = self.basis[leaving - 1];
        self.do_pivot(leaving, entering);
        self.basis[leaving - 1] = entering;
        PivotEvent {
            tick: self.tick,
            mode: PivotMode::Primal,
            entering: Some(entering),
            leaving: Some(leaving),
            entering_name: Some(self.col_name(entering)),
            leaving_name: Some(self.col_name(leaving_col)),
        }
    }

    /// Dual pivot: most-negative-RHS rule for leaving, then ratio test on the
    /// z-row for entering. Maintains dual feasibility.
    fn dual_pivot(&mut self) -> PivotEvent {
        let m = self.tab.len() - 1;
        let total_cols = self.tab[0].len();
        let rhs_col = total_cols - 1;
        // Leaving: most negative basic value.
        let mut leaving: Option<usize> = None;
        let mut most_neg = -EPS;
        for i in 1..=m {
            if self.tab[i][rhs_col] < most_neg {
                most_neg = self.tab[i][rhs_col];
                leaving = Some(i);
            }
        }
        let leaving = match leaving {
            Some(l) => l,
            None => return PivotEvent::terminal(self.tick, PivotMode::Optimal),
        };
        // Entering: argmin over j with tab[leaving][j] < 0 of (tab[0][j] / -tab[leaving][j]).
        let mut entering: Option<usize> = None;
        let mut min_ratio = f64::INFINITY;
        for j in 0..rhs_col {
            if self.tab[leaving][j] < -EPS {
                let ratio = self.tab[0][j] / -self.tab[leaving][j];
                if ratio < min_ratio - EPS {
                    min_ratio = ratio;
                    entering = Some(j);
                }
            }
        }
        let entering = match entering {
            Some(e) => e,
            None => return PivotEvent::terminal(self.tick, PivotMode::Infeasible),
        };
        // capture BEFORE the pivot updates the basis
        let leaving_col = self.basis[leaving - 1];
        self.do_pivot(leaving, entering);
        self.basis[leaving - 1] = entering;
        PivotEvent {
            tick: self.tick,
            mode: PivotMode::Dual,
            entering: Some(entering),
            leaving: Some(leaving),
            entering_name: Some(self.col_name(entering)),
            leaving_name: Some(self.col_name(leaving_col)),
        }
    }

    /// Gauss–Jordan elimination on (pivot row, pivot column).
    fn do_pivot(&mut self, r: usize, c: usize) {
        let pivot_val = self.tab[r][c];
        if pivot_val.abs() < EPS {
            panic!("degenerate pivot {} at row {} col {}", pivot_val, r, c);
        }
        let ncols = self.tab[r].len();
        for j in 0..ncols {
            self.tab[r][j] /= pivot_val;
        }
        let pivot_row = self.tab[r].clone();
        let nrows = self.tab.len();
        for i in 0..nrows {
            if i == r {
                continue;
            }
            let factor = self.tab[i][c];
            if factor.abs() < EPS {
                continue;
            }
            let row = &mut self.tab[i];
            for j in 0..row.len() {
                row[j] -= factor * pivot_row[j];
            }
        }
    }

    /// One tick of the DES: apply one pivot if not optimal. Returns the pivot
    /// event recorded, or an `Idle` event if optimal/infeasible/unbounded.
    pub fn step(&mut self) -> PivotEvent {
        self.tick += 1;
        if matches!(
            self.status,
            SolverStatus::Optimal | SolverStatus::Infeasible | SolverStatus::Unbounded
        ) {
            return PivotEvent::terminal(self.tick, PivotMode::Idle);
        }
        let pivot = if self.status == SolverStatus::Dual {
            self.dual_pivot()
        } else {
            self.primal_pivot()
        };
        // Update solver mode.
        match pivot.mode {
            PivotMode::Optimal => self.status = SolverStatus::Optimal,
            PivotMode::Infeasible => self.status = SolverStatus::Infeasible,
            PivotMode::Unbounded => self.status = SolverStatus::Unbounded,
            _ => self.refresh_status(),
        }
        pivot
    }

    /// Recompute the primal/dual feasibility flags and choose the mode.
    fn refresh_status(&mut self) {
        let m = self.tab.len() - 1;
        let total_cols = self.tab[0].len();
        let rhs_col = total_cols - 1;
        let mut primal_feas = true;
        let mut dual_feas = true;
        for i in 1..=m {
            if self.tab[i][rhs_col] < -EPS {
                primal_feas = false;
                break;
            }
        }
        for j in 0..rhs_col {
            if self.tab[0][j] < -EPS {
                dual_feas = false;
                break;
            }
        }
        self.status = if primal_feas && dual_feas {
            SolverStatus::Optimal
        } else if primal_feas && !dual_feas {
            SolverStatus::Primal
        } else if !primal_feas && dual_feas {
            SolverStatus::Dual
        } else {
            // both broken: bias to primal
            SolverStatus::Primal
        };
    }

    // ---------------------------------------------------------------------
    // MODIFICATION OPERATIONS
    // ---------------------------------------------------------------------

    /// Add `coefs · x ≤ rhs`. After the call the system is dual-feasible; if the
    /// new slack is negative (primal infeasible) the next `step()`s run dual simplex.
    pub fn apply_add_constraint(&mut self, coefs: &[f64], rhs: f64, name: Option<String>) {
        if coefs.len() != self.num_struct {
            panic!("coefs.len()={}, num_struct={}", coefs.len(), self.num_struct);
        }
        let m = self.tab.len() - 1;
        let old_total = self.tab[0].len();
        // 1. Insert a new slack column at index num_struct + m (just before rhs).
        let new_col = self.num_struct + m;
        for row in self.tab.iter_mut() {
            row.insert(new_col, 0.0);
        }
        // 2. Append a new constraint row in standard form.
        let total_cols = old_total + 1;
        let mut new_row = vec![0.0f64; total_cols];
        for j in 0..self.num_struct {
            new_row[j] = coefs[j];
        }
        new_row[new_col] = 1.0; // its slack
        new_row[total_cols - 1] = rhs;
        // 3. Reduce the new row against all currently basic columns so it has
        //    zeros under every existing basic column.
        for k in 0..self.basis.len() {
            let bj = self.basis[k];
            let factor = new_row[bj];
            if factor.abs() < EPS {
                continue;
            }
            for j in 0..total_cols {
                new_row[j] -= factor * self.tab[k + 1][j];
            }
        }
        self.tab.push(new_row);
        self.basis.push(new_col); // new slack is basic
        let con_name = name.unwrap_or_else(|| format!("c{}", self.con_names.len() + 1));
        self.con_names.push(con_name);
        self.refresh_status();
    }

    /// Remove the constraint at `index` (0-based).
    ///
    /// Locates `r*`, the row where `slack_i` is currently basic. If `slack_i` is
    /// non-basic, pivots it in at any row with a non-zero coefficient in column
    /// `slack_i` (which MAY break primal feasibility — repaired by the next
    /// dual-simplex step). Then drops row `r*` and column `slack_i`.
    pub fn apply_remove_constraint(&mut self, index: usize) {
        if index >= self.tab.len() - 1 {
            panic!("remove-constraint: index {} out of range", index);
        }
        let slack_col = self.num_struct + index;
        let m = self.tab.len() - 1;
        // 1a. Find the row where slack_i is currently basic.
        let mut r_star: Option<usize> = None;
        for r in 1..=m {
            if self.basis[r - 1] == slack_col {
                r_star = Some(r);
                break;
            }
        }
        // 1b. If slack_i is non-basic, force it into the basis.
        if r_star.is_none() {
            for r in 1..=m {
                if self.tab[r][slack_col].abs() > EPS {
                    self.do_pivot(r, slack_col);
                    self.basis[r - 1] = slack_col;
                    r_star = Some(r);
                    break;
                }
            }
            if r_star.is_none() {
                // Column is identically zero in every row — constraint was
                // redundant. Pick row corresponding to `index+1` arbitrarily.
                r_star = Some((index + 1).min(m));
            }
        }
        let r_star = r_star.unwrap();
        // 2. Drop the row and the slack column.
        self.tab.remove(r_star);
        for row in self.tab.iter_mut() {
            row.remove(slack_col);
        }
        self.basis.remove(r_star - 1);
        for k in 0..self.basis.len() {
            if self.basis[k] > slack_col {
                self.basis[k] -= 1;
            }
        }
        self.con_names.remove(index);
        self.refresh_status();
    }

    /// Replace the objective with `new_c`. Dual feasibility may break; the next
    /// `step()`s run primal simplex.
    pub fn apply_change_objective(&mut self, new_c: &[f64]) {
        if new_c.len() != self.num_struct {
            panic!(
                "change-objective: length {}, expected {}",
                new_c.len(),
                self.num_struct
            );
        }
        let total_cols = self.tab[0].len();
        let rhs_col = total_cols - 1;
        // Reset the z-row. New row 0 = -sense_sign * new_c for structural cols,
        // 0 for slack cols, 0 for rhs.
        for j in 0..rhs_col {
            self.tab[0][j] = if j < self.num_struct {
                -self.sense_sign * new_c[j]
            } else {
                0.0
            };
        }
        self.tab[0][rhs_col] = 0.0;
        // Re-zero the z-row beneath every basic column by row-reduction.
        for k in 0..self.basis.len() {
            let bj = self.basis[k];
            let factor = self.tab[0][bj];
            if factor.abs() < EPS {
                continue;
            }
            let src = self.tab[k + 1].clone();
            let row0 = &mut self.tab[0];
            for j in 0..total_cols {
                row0[j] -= factor * src[j];
            }
        }
        self.refresh_status();
    }

    /// Append a new structural variable with column `column` (length m, in
    /// ORIGINAL untransformed standard-form coordinates) and objective
    /// coefficient `c_new`.
    ///
    /// The inserted column must be expressed in the CURRENT tableau coordinate
    /// system, which is `B^{-1}·column`. The slack columns jointly store
    /// `B^{-1}`, so `B^{-1}·column` is a linear combination over those columns.
    /// The row-0 entry is `Σ_k column[k] · tab[0][slack_k] − sense_sign·c_new`.
    pub fn apply_add_variable(&mut self, column: &[f64], c_new: f64, name: Option<String>) {
        let m = self.tab.len() - 1;
        if column.len() != m {
            panic!("add-variable: column length {}, expected {}", column.len(), m);
        }
        // Compute the transformed column = B^{-1} · column, plus the row-0 reduced cost.
        let mut transformed = vec![0.0f64; m + 1];
        for i in 1..=m {
            let mut v = 0.0;
            for k in 0..m {
                v += column[k] * self.tab[i][self.num_struct + k];
            }
            transformed[i] = v;
        }
        let mut z_row_entry = -self.sense_sign * c_new; // start with −c_new term
        for k in 0..m {
            z_row_entry += column[k] * self.tab[0][self.num_struct + k];
        }
        transformed[0] = z_row_entry;
        // Insert the column at position `num_struct` (just after existing
        // structural variables, before the slack block).
        let insert_at = self.num_struct;
        for (i, row) in self.tab.iter_mut().enumerate() {
            row.insert(insert_at, transformed[i]);
        }
        self.num_struct += 1;
        for k in 0..self.basis.len() {
            if self.basis[k] >= insert_at {
                self.basis[k] += 1;
            }
        }
        let var_name = name.unwrap_or_else(|| format!("x{}", insert_at + 1));
        self.var_names.insert(insert_at, var_name);
        self.refresh_status();
    }

    /// Remove a structural variable. If non-basic, drop its column directly; if
    /// basic, force-pivot it out first (in its current basic row), then drop.
    pub fn apply_remove_variable(&mut self, struct_index: usize) {
        if struct_index >= self.num_struct {
            panic!("remove-variable: index {} out of range", struct_index);
        }
        let drop_col = struct_index;
        // Is it basic somewhere?
        let mut basic_row: Option<usize> = None;
        for k in 0..self.basis.len() {
            if self.basis[k] == drop_col {
                basic_row = Some(k + 1);
                break;
            }
        }
        if let Some(br) = basic_row {
            // Find any non-basic column with a non-zero entry in `br` to force a
            // pivot that knocks drop_col out of the basis.
            let total_cols = self.tab[0].len();
            let rhs_col = total_cols - 1;
            let mut entering: Option<usize> = None;
            for j in 0..rhs_col {
                if j == drop_col {
                    continue;
                }
                if self.basis.contains(&j) {
                    continue;
                }
                if self.tab[br][j].abs() > EPS {
                    entering = Some(j);
                    break;
                }
            }
            if let Some(e) = entering {
                self.do_pivot(br, e);
                self.basis[br - 1] = e;
            } else {
                // No pivot possible; the variable is degenerate — drop the row.
                self.tab.remove(br);
                self.basis.remove(br - 1);
            }
        }
        // Drop the column from every row.
        for row in self.tab.iter_mut() {
            row.remove(drop_col);
        }
        self.num_struct -= 1;
        for k in 0..self.basis.len() {
            if self.basis[k] > drop_col {
                self.basis[k] -= 1;
            }
        }
        self.var_names.remove(drop_col);
        self.refresh_status();
    }

    /// Convenience: apply an [`LPEvent`].
    pub fn apply_event(&mut self, e: LPEvent) {
        match e {
            LPEvent::AddConstraint {
                coefs, rhs, name, ..
            } => self.apply_add_constraint(&coefs, rhs, name),
            LPEvent::RemoveConstraint { index, .. } => self.apply_remove_constraint(index),
            LPEvent::ChangeObjective { new_c, .. } => self.apply_change_objective(&new_c),
            LPEvent::AddVariable {
                column, c_new, name, ..
            } => self.apply_add_variable(&column, c_new, name),
            LPEvent::RemoveVariable { struct_index, .. } => {
                self.apply_remove_variable(struct_index)
            }
        }
    }

    /// Pivot until optimal/infeasible/unbounded. Returns the pivot trace.
    /// (`TS` default `max_iters = 1000`.)
    pub fn solve_to_optimum(&mut self, max_iters: usize) -> Vec<PivotEvent> {
        let mut trace: Vec<PivotEvent> = Vec::new();
        for _ in 0..max_iters {
            let ev = self.step();
            let mode = ev.mode;
            trace.push(ev);
            if matches!(
                self.status,
                SolverStatus::Optimal | SolverStatus::Infeasible | SolverStatus::Unbounded
            ) {
                return trace;
            }
            if mode == PivotMode::Idle {
                return trace;
            }
        }
        trace
    }

    // ---------------------------------------------------------------------
    // INSPECTION / SNAPSHOT
    // ---------------------------------------------------------------------

    /// Resolved x for the structural variables.
    pub fn get_x(&self) -> Vec<f64> {
        let mut x = vec![0.0f64; self.num_struct];
        let rhs_col = self.tab[0].len() - 1;
        for k in 0..self.basis.len() {
            let j = self.basis[k];
            if j < self.num_struct {
                x[j] = self.tab[k + 1][rhs_col];
            }
        }
        x
    }

    /// Slack values.
    pub fn get_slacks(&self) -> Vec<f64> {
        let m = self.tab.len() - 1;
        let mut s = vec![0.0f64; m];
        let rhs_col = self.tab[0].len() - 1;
        for k in 0..self.basis.len() {
            let j = self.basis[k];
            if j >= self.num_struct {
                s[j - self.num_struct] = self.tab[k + 1][rhs_col];
            }
        }
        s
    }

    /// Current objective value, in the original sense.
    pub fn get_z(&self) -> f64 {
        let rhs_col = self.tab[0].len() - 1;
        self.sense_sign * self.tab[0][rhs_col]
    }

    /// Reduced-cost vector (length `num_struct + m`).
    pub fn get_reduced_costs(&self) -> Vec<f64> {
        let rhs_col = self.tab[0].len() - 1;
        self.tab[0][..rhs_col].to_vec()
    }

    pub fn col_name(&self, j: usize) -> String {
        if j < self.num_struct {
            self.var_names[j].clone()
        } else {
            format!("{}_slack", self.con_names[j - self.num_struct])
        }
    }

    pub fn snapshot(
        &self,
        applied_event: Option<LPEvent>,
        pivot: Option<PivotEvent>,
    ) -> LPSnapshot {
        let m = self.tab.len() - 1;
        let rhs_col = self.tab[0].len() - 1;
        LPSnapshot {
            tick: self.tick,
            num_struct: self.num_struct,
            num_constraints: m,
            basis: self.basis.clone(),
            x: self.get_x(),
            slacks: self.get_slacks(),
            z: self.get_z(),
            reduced_costs: self.get_reduced_costs(),
            rhs: self.tab[1..].iter().map(|r| r[rhs_col]).collect(),
            primal_feasible: self.tab[1..].iter().all(|r| r[rhs_col] >= -EPS),
            dual_feasible: self.tab[0][..rhs_col].iter().all(|&v| v >= -EPS),
            is_optimal: self.status == SolverStatus::Optimal,
            var_names: self.var_names.clone(),
            con_names: self.con_names.clone(),
            applied_event,
            pivot,
            mode: self.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Basic warm-start solve: maximise 3x1 + 2x2 s.t. x1+x2 ≤ 4, x1+3x2 ≤ 6.
    /// Optimum is z = 12 at (x1, x2) = (4, 0).
    #[test]
    fn solves_basic_max_lp() {
        let mut lp = IncrementalLP::new(IncrementalLPInit {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0], vec![1.0, 3.0]],
            b: vec![4.0, 6.0],
            var_names: None,
            con_names: None,
        });
        lp.solve_to_optimum(1000);
        assert_eq!(lp.status, SolverStatus::Optimal);
        assert!(approx(lp.get_z(), 12.0), "z = {}", lp.get_z());
        let x = lp.get_x();
        assert!(approx(x[0], 4.0) && approx(x[1], 0.0), "x = {:?}", x);
    }

    /// Incremental warm-start: solve, then add a cut x1 ≤ 2 that excludes the
    /// current optimum. The dual simplex must restore feasibility to the new
    /// optimum z = 26/3 at (2, 4/3).
    #[test]
    fn add_constraint_warm_starts_via_dual_simplex() {
        let mut lp = IncrementalLP::new(IncrementalLPInit {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0], vec![1.0, 3.0]],
            b: vec![4.0, 6.0],
            var_names: None,
            con_names: None,
        });
        lp.solve_to_optimum(1000);
        assert!(approx(lp.get_z(), 12.0));

        // Add x1 ≤ 2; the old vertex (4,0) becomes primal-infeasible.
        lp.apply_add_constraint(&[1.0, 0.0], 2.0, None);
        assert_eq!(lp.status, SolverStatus::Dual);
        lp.solve_to_optimum(1000);

        assert_eq!(lp.status, SolverStatus::Optimal);
        assert!(approx(lp.get_z(), 26.0 / 3.0), "z = {}", lp.get_z());
        let x = lp.get_x();
        assert!(approx(x[0], 2.0) && approx(x[1], 4.0 / 3.0), "x = {:?}", x);
    }

    /// Incremental objective change re-optimises via primal simplex.
    /// After solving 3x1+2x2, switch objective to maximise x2 only; new
    /// optimum is z = 2 at (0, 2).
    #[test]
    fn change_objective_warm_starts_via_primal_simplex() {
        let mut lp = IncrementalLP::new(IncrementalLPInit {
            sense: Sense::Max,
            c: vec![3.0, 2.0],
            a: vec![vec![1.0, 1.0], vec![1.0, 3.0]],
            b: vec![4.0, 6.0],
            var_names: None,
            con_names: None,
        });
        lp.solve_to_optimum(1000);
        assert!(approx(lp.get_z(), 12.0));

        lp.apply_change_objective(&[0.0, 1.0]);
        lp.solve_to_optimum(1000);

        assert_eq!(lp.status, SolverStatus::Optimal);
        assert!(approx(lp.get_z(), 2.0), "z = {}", lp.get_z());
        let x = lp.get_x();
        assert!(approx(x[0], 0.0) && approx(x[1], 2.0), "x = {:?}", x);
    }
}
