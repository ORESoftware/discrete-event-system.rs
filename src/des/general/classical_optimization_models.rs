//! Port of `src/des/general/classical-optimization-models.ts`
//! (module `des::general::classical_optimization_models`).
//!
//! Additional classic optimization routines as explicit DES station graphs:
//!   - qp-projected-gradient
//!   - qp-coordinate-descent
//!   - hungarian-assignment
//!   - auction-assignment
//!   - vrp-savings
//!   - vrp-nearest-neighbor
//!   - vrp-exact
//!   - job-shop-dispatch
//!   - job-shop-exact
//!   - flow-shop-neh
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * This file's `AssignmentResult` is distinct from `hungarian.rs`'s; it
//!     stays module-qualified (no global merge).
//!   * The many `*Token` classes become plain structs carried as `Rc<dyn Any>`
//!     (there is no `Token` trait in the ported `station.rs` — FLAGGED); the
//!     `*Station` classes become `struct { core: StationCore, … }` + `impl
//!     DESStation`.
//!   * `cloneMatrix`/`dot`/`norm2`/`zeros` come from the des-base port
//!     ([`learning_optimization`]); `number[][]` → `Vec<Vec<f64>>`.
//!   * `Preconditions` `throw` on bad QP params → guards whose `Err` is turned
//!     into a `panic!` (an invariant for the seed token).
//!   * `'fifo' | 'spt' | 'edd'` string-union → the [`DispatchRule`] enum;
//!     `assignment` indices that may be the `-1` sentinel (auction) → `Vec<i64>`.
//!   * Fully deterministic: no RNG/clock.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    channel_edge, clone_matrix, dot, empty_station_graph, non_empty_array, norm2,
    run_state_loop_pipeline, state_loop_topology, station_graph, zeros, LatestTokenSinkStation,
    SingleTokenSourceStation, StationGraphSummary, StationOrId,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};

/// Panic with the precondition message on a failed guard (TS `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Quadratic programming (projected gradient / coordinate descent)
// =============================================================================

const CH_QP_STATE: &str = "qp-state";
const CH_QP_RESULT: &str = "qp-result";

#[derive(Clone, Debug, Default)]
pub struct QPProjectedGradientParams {
    pub q: Option<Vec<Vec<f64>>>,
    pub c: Option<Vec<f64>>,
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
    pub x0: Option<Vec<f64>>,
    pub step_size: Option<f64>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct QPTraceEntry {
    pub iter: usize,
    pub objective: f64,
    pub gradient_norm: f64,
    pub x: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct QPProjectedGradientResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub gradient_norm: f64,
    pub trace: Vec<QPTraceEntry>,
    pub topology: StationGraphSummary,
}

/// Movable carrying the projected/coordinate walker state.
#[derive(Clone, Debug)]
pub struct QPStateToken {
    pub iter: usize,
    pub x: Vec<f64>,
}

/// Terminal QP result token.
#[derive(Clone, Debug)]
pub struct QPResultToken {
    pub result: QPProjectedGradientResult,
}

/// Box-constrained projected-gradient descent on `½xᵀQx + cᵀx`.
pub struct QPProjectedGradientStation {
    core: StationCore,
    q: Vec<Vec<f64>>,
    c: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    step_size: f64,
    max_iter: usize,
    tol: f64,
    pub trace: Vec<QPTraceEntry>,
}

impl QPProjectedGradientStation {
    pub const CH_STATE: &'static str = CH_QP_STATE;
    pub const CH_RESULT: &'static str = CH_QP_RESULT;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        q: Vec<Vec<f64>>,
        c: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        step_size: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        QPProjectedGradientStation {
            core: StationCore::new(id),
            q,
            c,
            lower,
            upper,
            step_size,
            max_iter,
            tol,
            trace: Vec::new(),
        }
    }
}

impl DESStation for QPProjectedGradientStation {
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
        self.core.inbox_size(Self::CH_STATE) > 0
    }
    fn run_time_step(&mut self) {
        let states = self.core.drain::<QPStateToken>(Self::CH_STATE);
        for state in states {
            let gradient = qp_gradient(&self.q, &self.c, &state.x);
            let gradient_norm = norm2(&gradient);
            let objective = qp_objective(&self.q, &self.c, &state.x);
            self.trace.push(QPTraceEntry {
                iter: state.iter,
                objective,
                gradient_norm,
                x: state.x.clone(),
            });
            if state.iter >= self.max_iter || gradient_norm <= self.tol {
                let result = QPProjectedGradientResult {
                    x: state.x.clone(),
                    objective,
                    iterations: state.iter,
                    gradient_norm,
                    trace: self.trace.clone(),
                    topology: empty_station_graph(),
                };
                self.core
                    .emit(Rc::new(QPResultToken { result }), Self::CH_RESULT);
                continue;
            }
            let next: Vec<f64> = state
                .x
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    self.upper[i].min(self.lower[i].max(v - self.step_size * gradient[i]))
                })
                .collect();
            self.core.emit(
                Rc::new(QPStateToken {
                    iter: state.iter + 1,
                    x: next,
                }),
                Self::CH_STATE,
            );
        }
    }
}

/// Box-constrained coordinate descent on `½xᵀQx + cᵀx`.
pub struct QPCoordinateDescentStation {
    core: StationCore,
    q: Vec<Vec<f64>>,
    c: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    max_iter: usize,
    tol: f64,
    pub trace: Vec<QPTraceEntry>,
}

impl QPCoordinateDescentStation {
    pub const CH_STATE: &'static str = CH_QP_STATE;
    pub const CH_RESULT: &'static str = CH_QP_RESULT;

    pub fn new(
        id: impl Into<String>,
        q: Vec<Vec<f64>>,
        c: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        QPCoordinateDescentStation {
            core: StationCore::new(id),
            q,
            c,
            lower,
            upper,
            max_iter,
            tol,
            trace: Vec::new(),
        }
    }
}

impl DESStation for QPCoordinateDescentStation {
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
        self.core.inbox_size(Self::CH_STATE) > 0
    }
    fn run_time_step(&mut self) {
        let states = self.core.drain::<QPStateToken>(Self::CH_STATE);
        for state in states {
            let gradient = qp_gradient(&self.q, &self.c, &state.x);
            let gradient_norm = norm2(&gradient);
            let objective = qp_objective(&self.q, &self.c, &state.x);
            self.trace.push(QPTraceEntry {
                iter: state.iter,
                objective,
                gradient_norm,
                x: state.x.clone(),
            });
            if state.iter >= self.max_iter || gradient_norm <= self.tol {
                let result = QPProjectedGradientResult {
                    x: state.x.clone(),
                    objective,
                    iterations: state.iter,
                    gradient_norm,
                    trace: self.trace.clone(),
                    topology: empty_station_graph(),
                };
                self.core
                    .emit(Rc::new(QPResultToken { result }), Self::CH_RESULT);
                continue;
            }
            let mut next = state.x.clone();
            for i in 0..next.len() {
                let diag = self.q[i][i];
                if diag.abs() <= 1e-12 {
                    continue;
                }
                let g = dot(&self.q[i], &next) + self.c[i];
                next[i] = self.upper[i].min(self.lower[i].max(next[i] - g / diag));
            }
            self.core.emit(
                Rc::new(QPStateToken {
                    iter: state.iter + 1,
                    x: next,
                }),
                Self::CH_STATE,
            );
        }
    }
}

fn validate_qp_initial_state(
    model: &str,
    token: &QPStateToken,
    n: usize,
    lower: &[f64],
    upper: &[f64],
) {
    require(Preconditions::integer_in_range(
        model,
        "iter",
        token.iter as f64,
        0.0,
        1e9,
    ));
    require(Preconditions::length_eq(model, "x0", &token.x, n));
    require(Preconditions::all_finite(model, "x0", &token.x));
    require(Preconditions::length_eq(model, "lower", lower, n));
    require(Preconditions::length_eq(model, "upper", upper, n));
    for i in 0..n {
        require(Preconditions::check(
            model,
            &format!("lower[{i}] <= x0[{i}] <= upper[{i}]"),
            "hold",
            lower[i] <= token.x[i] && token.x[i] <= upper[i],
            Some(format!("[{}, {}, {}]", lower[i], token.x[i], upper[i])),
        ));
    }
}

pub fn run_qp_projected_gradient(params: QPProjectedGradientParams) -> QPProjectedGradientResult {
    let q = non_empty_array(params.q.as_deref(), &[vec![4.0, 1.0], vec![1.0, 2.0]]);
    let c = non_empty_array(params.c.as_deref(), &[-8.0, -6.0]);
    let n = c.len();
    let lower = non_empty_array(params.lower.as_deref(), &zeros(n));
    let upper = non_empty_array(params.upper.as_deref(), &vec![10.0; n]);
    let x0 = non_empty_array(params.x0.as_deref(), &zeros(n));
    let max_iter = params.max_iter.unwrap_or(200);

    let model = "qp-projected-gradient-source".to_string();
    let lower_v = lower.clone();
    let upper_v = upper.clone();
    let x0_factory = x0.clone();
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "qp-state-source",
        CH_QP_STATE,
        move || QPStateToken {
            iter: 0,
            x: x0_factory.clone(),
        },
        move |t: &QPStateToken| validate_qp_initial_state(&model, t, n, &lower_v, &upper_v),
    )));
    let update = Rc::new(RefCell::new(QPProjectedGradientStation::new(
        "projected-gradient-update",
        q,
        c,
        lower,
        upper,
        params.step_size.unwrap_or(0.12),
        max_iter,
        params.tol.unwrap_or(1e-8),
    )));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<QPResultToken>::new(
        "qp-result-sink",
        CH_QP_RESULT,
    )));

    run_qp(
        "qp-state-source",
        "projected-gradient-update",
        source,
        update,
        sink,
        max_iter,
    )
}

pub fn run_qp_coordinate_descent(params: QPProjectedGradientParams) -> QPProjectedGradientResult {
    let q = non_empty_array(params.q.as_deref(), &[vec![4.0, 1.0], vec![1.0, 2.0]]);
    let c = non_empty_array(params.c.as_deref(), &[-8.0, -6.0]);
    let n = c.len();
    let lower = non_empty_array(params.lower.as_deref(), &zeros(n));
    let upper = non_empty_array(params.upper.as_deref(), &vec![10.0; n]);
    let x0 = non_empty_array(params.x0.as_deref(), &zeros(n));
    let max_iter = params.max_iter.unwrap_or(100);

    let model = "qp-coordinate-descent-source".to_string();
    let lower_v = lower.clone();
    let upper_v = upper.clone();
    let x0_factory = x0.clone();
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        "qp-coordinate-state-source",
        CH_QP_STATE,
        move || QPStateToken {
            iter: 0,
            x: x0_factory.clone(),
        },
        move |t: &QPStateToken| validate_qp_initial_state(&model, t, n, &lower_v, &upper_v),
    )));
    let update = Rc::new(RefCell::new(QPCoordinateDescentStation::new(
        "coordinate-descent-update",
        q,
        c,
        lower,
        upper,
        max_iter,
        params.tol.unwrap_or(1e-8),
    )));
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<QPResultToken>::new(
        "qp-result-sink",
        CH_QP_RESULT,
    )));

    run_qp(
        "qp-coordinate-state-source",
        "coordinate-descent-update",
        source,
        update,
        sink,
        max_iter,
    )
}

fn run_qp<U: DESStation + 'static>(
    _source_id: &str,
    update_id: &str,
    source: Rc<RefCell<SingleTokenSourceStation<QPStateToken>>>,
    update: Rc<RefCell<U>>,
    sink: Rc<RefCell<LatestTokenSinkStation<QPResultToken>>>,
    max_iter: usize,
) -> QPProjectedGradientResult {
    run_state_loop_pipeline(
        source.clone() as StationRef,
        update.clone() as StationRef,
        sink.clone() as StationRef,
        CH_QP_STATE,
        CH_QP_RESULT,
        IterativeRunOptions {
            max_ticks: Some(max_iter + 10),
            ..Default::default()
        },
    );

    let latest = sink
        .borrow()
        .latest
        .clone()
        .unwrap_or_else(|| panic!("{update_id} did not produce a result"));
    let mut result = latest.result.clone();
    result.topology = state_loop_topology(
        &*source.borrow(),
        &*update.borrow(),
        &*sink.borrow(),
        CH_QP_STATE,
        CH_QP_RESULT,
        &["QPStateToken".to_string(), "QPResultToken".to_string()],
    );
    result
}

// =============================================================================
// Assignment (Hungarian DP / auction)
// =============================================================================

const CH_ASSIGNMENT_MATRIX: &str = "assignment-matrix";
const CH_ROW_REDUCED: &str = "row-reduced";
const CH_COLUMN_REDUCED: &str = "column-reduced";
const CH_ASSIGNMENT_RESULT: &str = "assignment-result";
const CH_AUCTION_STATE: &str = "assignment-auction-state";

#[derive(Clone, Debug, Default)]
pub struct AssignmentParams {
    pub cost: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug)]
pub struct AssignmentResult {
    /// `assignment[worker]` = assigned column/job (`-1` if unassigned, auction).
    pub assignment: Vec<i64>,
    pub objective: f64,
    pub row_reductions: Vec<f64>,
    pub col_reductions: Vec<f64>,
    pub topology: StationGraphSummary,
}

#[derive(Clone, Debug)]
pub struct AssignmentMatrixToken {
    pub original: Vec<Vec<f64>>,
    pub reduced: Vec<Vec<f64>>,
    pub row_reductions: Vec<f64>,
    pub col_reductions: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AssignmentResultToken {
    pub result: AssignmentResult,
}

#[derive(Clone, Debug)]
pub struct AssignmentAuctionStateToken {
    pub original: Vec<Vec<f64>>,
    pub prices: Vec<f64>,
    pub assignment: Vec<i64>,
    pub owner: Vec<i64>,
    pub iter: usize,
}

/// Emits the (cloned) cost matrix once.
pub struct AssignmentSourceStation {
    core: StationCore,
    cost: Vec<Vec<f64>>,
    emitted: bool,
}

impl AssignmentSourceStation {
    pub const CH_MATRIX: &'static str = CH_ASSIGNMENT_MATRIX;
    pub fn new(id: impl Into<String>, cost: Vec<Vec<f64>>) -> Self {
        AssignmentSourceStation {
            core: StationCore::new(id),
            cost,
            emitted: false,
        }
    }
}

impl DESStation for AssignmentSourceStation {
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
        let token = AssignmentMatrixToken {
            original: clone_matrix(&self.cost),
            reduced: clone_matrix(&self.cost),
            row_reductions: Vec::new(),
            col_reductions: Vec::new(),
        };
        self.core.emit(Rc::new(token), Self::CH_MATRIX);
        self.emitted = true;
    }
}

/// Subtracts each row's minimum.
pub struct RowReductionStation {
    core: StationCore,
}

impl RowReductionStation {
    pub const CH_MATRIX: &'static str = CH_ASSIGNMENT_MATRIX;
    pub const CH_REDUCED: &'static str = CH_ROW_REDUCED;
    pub fn new(id: impl Into<String>) -> Self {
        RowReductionStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for RowReductionStation {
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
        self.core.inbox_size(Self::CH_MATRIX) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<AssignmentMatrixToken>(Self::CH_MATRIX) {
            let mut reduced = clone_matrix(&token.reduced);
            let rows: Vec<f64> = reduced
                .iter()
                .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                .collect();
            for (i, row) in reduced.iter_mut().enumerate() {
                for v in row.iter_mut() {
                    *v -= rows[i];
                }
            }
            let out = AssignmentMatrixToken {
                original: token.original.clone(),
                reduced,
                row_reductions: rows,
                col_reductions: token.col_reductions.clone(),
            };
            self.core.emit(Rc::new(out), Self::CH_REDUCED);
        }
    }
}

/// Subtracts each column's minimum.
pub struct ColumnReductionStation {
    core: StationCore,
}

impl ColumnReductionStation {
    pub const CH_REDUCED: &'static str = CH_ROW_REDUCED;
    pub const CH_REDUCED2: &'static str = CH_COLUMN_REDUCED;
    pub fn new(id: impl Into<String>) -> Self {
        ColumnReductionStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for ColumnReductionStation {
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
        self.core.inbox_size(Self::CH_REDUCED) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<AssignmentMatrixToken>(Self::CH_REDUCED) {
            let mut reduced = clone_matrix(&token.reduced);
            let mut cols = zeros(reduced[0].len());
            for (j, col) in cols.iter_mut().enumerate() {
                *col = reduced
                    .iter()
                    .map(|row| row[j])
                    .fold(f64::INFINITY, f64::min);
            }
            for row in reduced.iter_mut() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v -= cols[j];
                }
            }
            let out = AssignmentMatrixToken {
                original: token.original.clone(),
                reduced,
                row_reductions: token.row_reductions.clone(),
                col_reductions: cols,
            };
            self.core.emit(Rc::new(out), Self::CH_REDUCED2);
        }
    }
}

/// Solves the (original) cost via an exact bitmask DP.
pub struct AssignmentSolverStation {
    core: StationCore,
}

impl AssignmentSolverStation {
    pub const CH_REDUCED: &'static str = CH_COLUMN_REDUCED;
    pub const CH_RESULT: &'static str = CH_ASSIGNMENT_RESULT;
    pub fn new(id: impl Into<String>) -> Self {
        AssignmentSolverStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for AssignmentSolverStation {
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
        self.core.inbox_size(Self::CH_REDUCED) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<AssignmentMatrixToken>(Self::CH_REDUCED) {
            let (assignment, objective) = solve_assignment_dp(&token.original);
            let result = AssignmentResult {
                assignment,
                objective,
                row_reductions: token.row_reductions.clone(),
                col_reductions: token.col_reductions.clone(),
                topology: empty_station_graph(),
            };
            self.core
                .emit(Rc::new(AssignmentResultToken { result }), Self::CH_RESULT);
        }
    }
}

/// Keeps the latest [`AssignmentResult`].
pub struct AssignmentSinkStation {
    core: StationCore,
    pub result: Option<AssignmentResult>,
}

impl AssignmentSinkStation {
    pub const CH_RESULT: &'static str = CH_ASSIGNMENT_RESULT;
    pub fn new(id: impl Into<String>) -> Self {
        AssignmentSinkStation {
            core: StationCore::new(id),
            result: None,
        }
    }
}

impl DESStation for AssignmentSinkStation {
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
        self.core.inbox_size(Self::CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let results = self.core.drain::<AssignmentResultToken>(Self::CH_RESULT);
        if let Some(last) = results.last() {
            self.result = Some(last.result.clone());
        }
    }
}

/// Emits the initial auction state token once.
pub struct AssignmentAuctionSourceStation {
    core: StationCore,
    cost: Vec<Vec<f64>>,
    emitted: bool,
}

impl AssignmentAuctionSourceStation {
    pub const CH_STATE: &'static str = CH_AUCTION_STATE;
    pub fn new(id: impl Into<String>, cost: Vec<Vec<f64>>) -> Self {
        AssignmentAuctionSourceStation {
            core: StationCore::new(id),
            cost,
            emitted: false,
        }
    }
}

impl DESStation for AssignmentAuctionSourceStation {
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
        let n = self.cost.len();
        let token = AssignmentAuctionStateToken {
            original: clone_matrix(&self.cost),
            prices: zeros(n),
            assignment: vec![-1; n],
            owner: vec![-1; n],
            iter: 0,
        };
        self.core.emit(Rc::new(token), Self::CH_STATE);
        self.emitted = true;
    }
}

/// One auction bidding round (self-loop) on the assignment problem.
pub struct AuctionAssignmentStation {
    core: StationCore,
    epsilon: f64,
    max_iter: usize,
}

impl AuctionAssignmentStation {
    pub const CH_STATE: &'static str = CH_AUCTION_STATE;
    pub const CH_RESULT: &'static str = CH_ASSIGNMENT_RESULT;
    pub fn new(id: impl Into<String>, epsilon: f64, max_iter: usize) -> Self {
        AuctionAssignmentStation {
            core: StationCore::new(id),
            epsilon,
            max_iter,
        }
    }
}

impl DESStation for AuctionAssignmentStation {
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
        self.core.inbox_size(Self::CH_STATE) > 0
    }
    fn run_time_step(&mut self) {
        for state in self
            .core
            .drain::<AssignmentAuctionStateToken>(Self::CH_STATE)
        {
            let unassigned = state.assignment.iter().position(|&job| job < 0);
            if unassigned.is_none() || state.iter >= self.max_iter {
                let objective: f64 = state
                    .assignment
                    .iter()
                    .enumerate()
                    .map(|(worker, &job)| {
                        if job >= 0 {
                            state.original[worker][job as usize]
                        } else {
                            f64::NAN
                        }
                    })
                    .sum();
                let result = AssignmentResult {
                    assignment: state.assignment.clone(),
                    objective,
                    row_reductions: Vec::new(),
                    col_reductions: state.prices.clone(),
                    topology: empty_station_graph(),
                };
                self.core
                    .emit(Rc::new(AssignmentResultToken { result }), Self::CH_RESULT);
                continue;
            }
            let unassigned = unassigned.unwrap();
            let mut nets: Vec<(usize, f64)> = state.original[unassigned]
                .iter()
                .enumerate()
                .map(|(job, &cost)| (job, -cost - state.prices[job]))
                .collect();
            nets.sort_by(|a, b| b.1.total_cmp(&a.1));
            let best = nets[0];
            let second_value = nets.get(1).map(|x| x.1).unwrap_or(best.1 - self.epsilon);
            let mut prices = state.prices.clone();
            let mut assignment = state.assignment.clone();
            let mut owner = state.owner.clone();
            let previous_owner = owner[best.0];
            if previous_owner >= 0 {
                assignment[previous_owner as usize] = -1;
            }
            owner[best.0] = unassigned as i64;
            assignment[unassigned] = best.0 as i64;
            prices[best.0] += best.1 - second_value + self.epsilon;
            let next = AssignmentAuctionStateToken {
                original: state.original.clone(),
                prices,
                assignment,
                owner,
                iter: state.iter + 1,
            };
            self.core.emit(Rc::new(next), Self::CH_STATE);
        }
    }
}

fn default_assignment_cost() -> Vec<Vec<f64>> {
    vec![
        vec![9.0, 2.0, 7.0],
        vec![6.0, 4.0, 3.0],
        vec![5.0, 8.0, 1.0],
    ]
}

pub fn run_hungarian_assignment(params: AssignmentParams) -> AssignmentResult {
    let cost = clone_matrix(&non_empty_array(
        params.cost.as_deref(),
        &default_assignment_cost(),
    ));
    let source = Rc::new(RefCell::new(AssignmentSourceStation::new(
        "assignment-source",
        cost,
    )));
    let row = Rc::new(RefCell::new(RowReductionStation::new("row-reduction")));
    let col = Rc::new(RefCell::new(ColumnReductionStation::new(
        "column-reduction",
    )));
    let solver = Rc::new(RefCell::new(AssignmentSolverStation::new(
        "assignment-builder",
    )));
    let sink = Rc::new(RefCell::new(AssignmentSinkStation::new("assignment-sink")));

    source.borrow_mut().core_mut().pipe(
        row.clone() as StationRef,
        AssignmentSourceStation::CH_MATRIX,
        RowReductionStation::CH_MATRIX,
    );
    row.borrow_mut().core_mut().pipe(
        col.clone() as StationRef,
        RowReductionStation::CH_REDUCED,
        ColumnReductionStation::CH_REDUCED,
    );
    col.borrow_mut().core_mut().pipe(
        solver.clone() as StationRef,
        ColumnReductionStation::CH_REDUCED2,
        AssignmentSolverStation::CH_REDUCED,
    );
    solver.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        AssignmentSolverStation::CH_RESULT,
        AssignmentSinkStation::CH_RESULT,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            row.clone() as StationRef,
            col.clone() as StationRef,
            solver.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("hungarian-assignment did not produce a result"));
    let stations = [
        StationOrId::Id("assignment-source".to_string()),
        StationOrId::Id("row-reduction".to_string()),
        StationOrId::Id("column-reduction".to_string()),
        StationOrId::Id("assignment-builder".to_string()),
        StationOrId::Id("assignment-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(
            &stations[0],
            AssignmentSourceStation::CH_MATRIX,
            &stations[1],
            Some(RowReductionStation::CH_MATRIX),
        ),
        channel_edge(
            &stations[1],
            RowReductionStation::CH_REDUCED,
            &stations[2],
            Some(ColumnReductionStation::CH_REDUCED),
        ),
        channel_edge(
            &stations[2],
            ColumnReductionStation::CH_REDUCED2,
            &stations[3],
            Some(AssignmentSolverStation::CH_REDUCED),
        ),
        channel_edge(
            &stations[3],
            AssignmentSolverStation::CH_RESULT,
            &stations[4],
            Some(AssignmentSinkStation::CH_RESULT),
        ),
    ];
    result.topology = station_graph(
        &stations,
        &[
            "AssignmentMatrixToken".to_string(),
            "AssignmentResultToken".to_string(),
        ],
        &edges,
    );
    result
}

#[derive(Clone, Debug, Default)]
pub struct AuctionAssignmentParams {
    pub cost: Option<Vec<Vec<f64>>>,
    pub epsilon: Option<f64>,
    pub max_iter: Option<usize>,
}

pub fn run_auction_assignment(params: AuctionAssignmentParams) -> AssignmentResult {
    let cost = clone_matrix(&non_empty_array(
        params.cost.as_deref(),
        &default_assignment_cost(),
    ));
    let n = cost.len();
    let max_iter = params.max_iter.unwrap_or_else(|| (20).max(n * n * 20));
    let source = Rc::new(RefCell::new(AssignmentAuctionSourceStation::new(
        "auction-assignment-source",
        cost,
    )));
    let auction = Rc::new(RefCell::new(AuctionAssignmentStation::new(
        "auction-bid-update",
        params.epsilon.unwrap_or(0.01),
        max_iter,
    )));
    let sink = Rc::new(RefCell::new(AssignmentSinkStation::new("assignment-sink")));

    source.borrow_mut().core_mut().pipe(
        auction.clone() as StationRef,
        AssignmentAuctionSourceStation::CH_STATE,
        AuctionAssignmentStation::CH_STATE,
    );
    auction.borrow_mut().core_mut().pipe(
        auction.clone() as StationRef,
        AuctionAssignmentStation::CH_STATE,
        AuctionAssignmentStation::CH_STATE,
    );
    auction.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        AuctionAssignmentStation::CH_RESULT,
        AssignmentSinkStation::CH_RESULT,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            auction.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_iter + 10),
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("auction-assignment did not produce a result"));
    result.topology = state_loop_topology(
        &*source.borrow(),
        &*auction.borrow(),
        &*sink.borrow(),
        CH_AUCTION_STATE,
        CH_ASSIGNMENT_RESULT,
        &[
            "AssignmentAuctionStateToken".to_string(),
            "AssignmentResultToken".to_string(),
        ],
    );
    result
}

// =============================================================================
// Vehicle routing (Clarke-Wright savings / nearest-neighbour)
// =============================================================================

const CH_VRP_PROBLEM: &str = "vrp-problem";
const CH_VRP_SAVINGS: &str = "vrp-savings";
const CH_VRP_RESULT: &str = "vrp-result";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VRPCustomer {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub demand: f64,
}

#[derive(Clone, Debug, Default)]
pub struct VRPSavingsParams {
    pub depot: Option<Point>,
    pub customers: Option<Vec<VRPCustomer>>,
    pub vehicle_capacity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct VRPRoute {
    pub customers: Vec<String>,
    pub load: f64,
    pub distance: f64,
}

#[derive(Clone, Debug)]
pub struct VRPSavingsResult {
    pub routes: Vec<VRPRoute>,
    pub total_distance: f64,
    pub savings_considered: usize,
    pub topology: StationGraphSummary,
}

#[derive(Clone, Debug)]
pub struct Saving {
    pub i: String,
    pub j: String,
    pub saving: f64,
}

#[derive(Clone, Debug)]
pub struct VRPProblemToken {
    pub depot: Point,
    pub customers: Vec<VRPCustomer>,
    pub capacity: f64,
}

#[derive(Clone, Debug)]
pub struct VRPSavingsToken {
    pub problem: VRPProblemToken,
    pub savings: Vec<Saving>,
}

#[derive(Clone, Debug)]
pub struct VRPResultToken {
    pub result: VRPSavingsResult,
}

/// Emits the VRP problem once.
pub struct VRPSourceStation {
    core: StationCore,
    problem: VRPProblemToken,
    emitted: bool,
}

impl VRPSourceStation {
    pub const CH_PROBLEM: &'static str = CH_VRP_PROBLEM;
    pub fn new(id: impl Into<String>, problem: VRPProblemToken) -> Self {
        VRPSourceStation {
            core: StationCore::new(id),
            problem,
            emitted: false,
        }
    }
}

impl DESStation for VRPSourceStation {
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
        self.core
            .emit(Rc::new(self.problem.clone()), Self::CH_PROBLEM);
        self.emitted = true;
    }
}

/// Computes pairwise Clarke-Wright savings, sorted descending.
pub struct SavingsStation {
    core: StationCore,
}

impl SavingsStation {
    pub const CH_PROBLEM: &'static str = CH_VRP_PROBLEM;
    pub const CH_SAVINGS: &'static str = CH_VRP_SAVINGS;
    pub fn new(id: impl Into<String>) -> Self {
        SavingsStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for SavingsStation {
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
        self.core.inbox_size(Self::CH_PROBLEM) > 0
    }
    fn run_time_step(&mut self) {
        for problem in self.core.drain::<VRPProblemToken>(Self::CH_PROBLEM) {
            let problem = (*problem).clone();
            let mut savings: Vec<Saving> = Vec::new();
            for a in 0..problem.customers.len() {
                for b in (a + 1)..problem.customers.len() {
                    let i = &problem.customers[a];
                    let j = &problem.customers[b];
                    savings.push(Saving {
                        i: i.id.clone(),
                        j: j.id.clone(),
                        saving: dist(problem.depot.x, problem.depot.y, i.x, i.y)
                            + dist(problem.depot.x, problem.depot.y, j.x, j.y)
                            - dist(i.x, i.y, j.x, j.y),
                    });
                }
            }
            savings.sort_by(|a, b| b.saving.total_cmp(&a.saving));
            self.core.emit(
                Rc::new(VRPSavingsToken { problem, savings }),
                Self::CH_SAVINGS,
            );
        }
    }
}

/// Greedily merges routes by descending savings, respecting capacity.
pub struct RouteMergeStation {
    core: StationCore,
}

impl RouteMergeStation {
    pub const CH_SAVINGS: &'static str = CH_VRP_SAVINGS;
    pub const CH_RESULT: &'static str = CH_VRP_RESULT;
    pub fn new(id: impl Into<String>) -> Self {
        RouteMergeStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for RouteMergeStation {
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
        self.core.inbox_size(Self::CH_SAVINGS) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<VRPSavingsToken>(Self::CH_SAVINGS) {
            let problem = &token.problem;
            let by_id: HashMap<String, VRPCustomer> = problem
                .customers
                .iter()
                .map(|c| (c.id.clone(), c.clone()))
                .collect();
            let mut routes: Vec<Vec<String>> = problem
                .customers
                .iter()
                .map(|c| vec![c.id.clone()])
                .collect();
            for s in &token.savings {
                let ri_idx = routes.iter().position(|r| r.contains(&s.i));
                let rj_idx = routes.iter().position(|r| r.contains(&s.j));
                let (Some(ri_idx), Some(rj_idx)) = (ri_idx, rj_idx) else {
                    continue;
                };
                if ri_idx == rj_idx {
                    continue;
                }
                let ri = &routes[ri_idx];
                let rj = &routes[rj_idx];
                let i_at_end = ri.last() == Some(&s.i);
                let j_at_start = rj.first() == Some(&s.j);
                let j_at_end = rj.last() == Some(&s.j);
                let i_at_start = ri.first() == Some(&s.i);
                let merged: Option<Vec<String>> = if i_at_end && j_at_start {
                    Some(ri.iter().chain(rj.iter()).cloned().collect())
                } else if j_at_end && i_at_start {
                    Some(rj.iter().chain(ri.iter()).cloned().collect())
                } else {
                    None
                };
                let Some(merged) = merged else { continue };
                let load: f64 = merged
                    .iter()
                    .map(|id| by_id.get(id).map(|c| c.demand).unwrap_or(0.0))
                    .sum();
                if load > problem.capacity {
                    continue;
                }
                let (hi, lo) = (ri_idx.max(rj_idx), ri_idx.min(rj_idx));
                routes.remove(hi);
                routes.remove(lo);
                routes.push(merged);
            }
            let result_routes: Vec<VRPRoute> = routes
                .iter()
                .map(|route| {
                    let customers: Vec<VRPCustomer> = route
                        .iter()
                        .map(|id| by_id.get(id).unwrap().clone())
                        .collect();
                    VRPRoute {
                        customers: route.clone(),
                        load: customers.iter().map(|c| c.demand).sum(),
                        distance: route_distance(problem.depot, &customers),
                    }
                })
                .collect();
            let total_distance = result_routes.iter().map(|r| r.distance).sum();
            let result = VRPSavingsResult {
                routes: result_routes,
                total_distance,
                savings_considered: token.savings.len(),
                topology: empty_station_graph(),
            };
            self.core
                .emit(Rc::new(VRPResultToken { result }), Self::CH_RESULT);
        }
    }
}

/// Keeps the latest [`VRPSavingsResult`].
pub struct VRPSinkStation {
    core: StationCore,
    pub result: Option<VRPSavingsResult>,
}

impl VRPSinkStation {
    pub const CH_RESULT: &'static str = CH_VRP_RESULT;
    pub fn new(id: impl Into<String>) -> Self {
        VRPSinkStation {
            core: StationCore::new(id),
            result: None,
        }
    }
}

impl DESStation for VRPSinkStation {
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
        self.core.inbox_size(Self::CH_RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let results = self.core.drain::<VRPResultToken>(Self::CH_RESULT);
        if let Some(last) = results.last() {
            self.result = Some(last.result.clone());
        }
    }
}

/// Sequential nearest-neighbour route construction with capacity refilling.
pub struct NearestNeighborRouteStation {
    core: StationCore,
}

impl NearestNeighborRouteStation {
    pub const CH_PROBLEM: &'static str = CH_VRP_PROBLEM;
    pub const CH_RESULT: &'static str = CH_VRP_RESULT;
    pub fn new(id: impl Into<String>) -> Self {
        NearestNeighborRouteStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for NearestNeighborRouteStation {
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
        self.core.inbox_size(Self::CH_PROBLEM) > 0
    }
    fn run_time_step(&mut self) {
        for problem in self.core.drain::<VRPProblemToken>(Self::CH_PROBLEM) {
            let problem = (*problem).clone();
            let by_id: HashMap<String, VRPCustomer> = problem
                .customers
                .iter()
                .map(|c| (c.id.clone(), c.clone()))
                .collect();
            let mut served: HashSet<String> = HashSet::new();
            let mut routes: Vec<VRPRoute> = Vec::new();
            while served.len() < problem.customers.len() {
                let mut route: Vec<String> = Vec::new();
                let mut load = 0.0;
                let mut here = problem.depot;
                loop {
                    // Iterate customers in original order (TS Set insertion order),
                    // keep the capacity-feasible ones, then stable-sort by distance.
                    let mut feasible: Vec<&VRPCustomer> = problem
                        .customers
                        .iter()
                        .filter(|c| !served.contains(&c.id) && load + c.demand <= problem.capacity)
                        .collect();
                    feasible.sort_by(|a, b| {
                        dist(here.x, here.y, a.x, a.y).total_cmp(&dist(here.x, here.y, b.x, b.y))
                    });
                    let Some(next) = feasible.first().copied() else {
                        break;
                    };
                    route.push(next.id.clone());
                    load += next.demand;
                    here = Point {
                        x: next.x,
                        y: next.y,
                    };
                    served.insert(next.id.clone());
                }
                let customers: Vec<VRPCustomer> = route
                    .iter()
                    .map(|id| by_id.get(id).unwrap().clone())
                    .collect();
                routes.push(VRPRoute {
                    customers: route,
                    load,
                    distance: route_distance(problem.depot, &customers),
                });
            }
            let total_distance = routes.iter().map(|r| r.distance).sum();
            let result = VRPSavingsResult {
                routes,
                total_distance,
                savings_considered: 0,
                topology: empty_station_graph(),
            };
            self.core
                .emit(Rc::new(VRPResultToken { result }), Self::CH_RESULT);
        }
    }
}

fn default_customers() -> Vec<VRPCustomer> {
    vec![
        VRPCustomer {
            id: "A".to_string(),
            x: 1.0,
            y: 2.0,
            demand: 2.0,
        },
        VRPCustomer {
            id: "B".to_string(),
            x: 2.0,
            y: 1.0,
            demand: 2.0,
        },
        VRPCustomer {
            id: "C".to_string(),
            x: 4.0,
            y: 1.0,
            demand: 2.0,
        },
        VRPCustomer {
            id: "D".to_string(),
            x: 5.0,
            y: 2.0,
            demand: 1.0,
        },
        VRPCustomer {
            id: "E".to_string(),
            x: 3.0,
            y: 4.0,
            demand: 2.0,
        },
    ]
}

fn build_vrp_problem(params: &VRPSavingsParams) -> VRPProblemToken {
    let customers = non_empty_array(params.customers.as_deref(), &default_customers());
    VRPProblemToken {
        depot: params.depot.unwrap_or(Point { x: 0.0, y: 0.0 }),
        customers,
        capacity: params.vehicle_capacity.unwrap_or(5.0),
    }
}

pub fn run_vrp_savings(params: VRPSavingsParams) -> VRPSavingsResult {
    let problem = build_vrp_problem(&params);
    let source = Rc::new(RefCell::new(VRPSourceStation::new("vrp-source", problem)));
    let savings = Rc::new(RefCell::new(SavingsStation::new("savings-calculator")));
    let merge = Rc::new(RefCell::new(RouteMergeStation::new("route-merge")));
    let sink = Rc::new(RefCell::new(VRPSinkStation::new("vrp-sink")));

    source.borrow_mut().core_mut().pipe(
        savings.clone() as StationRef,
        VRPSourceStation::CH_PROBLEM,
        SavingsStation::CH_PROBLEM,
    );
    savings.borrow_mut().core_mut().pipe(
        merge.clone() as StationRef,
        SavingsStation::CH_SAVINGS,
        RouteMergeStation::CH_SAVINGS,
    );
    merge.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        RouteMergeStation::CH_RESULT,
        VRPSinkStation::CH_RESULT,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            savings.clone() as StationRef,
            merge.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("vrp-savings did not produce a result"));
    let stations = [
        StationOrId::Id("vrp-source".to_string()),
        StationOrId::Id("savings-calculator".to_string()),
        StationOrId::Id("route-merge".to_string()),
        StationOrId::Id("vrp-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(
            &stations[0],
            VRPSourceStation::CH_PROBLEM,
            &stations[1],
            Some(SavingsStation::CH_PROBLEM),
        ),
        channel_edge(
            &stations[1],
            SavingsStation::CH_SAVINGS,
            &stations[2],
            Some(RouteMergeStation::CH_SAVINGS),
        ),
        channel_edge(
            &stations[2],
            RouteMergeStation::CH_RESULT,
            &stations[3],
            Some(VRPSinkStation::CH_RESULT),
        ),
    ];
    result.topology = station_graph(
        &stations,
        &[
            "VRPProblemToken".to_string(),
            "VRPSavingsToken".to_string(),
            "VRPResultToken".to_string(),
        ],
        &edges,
    );
    result
}

pub fn run_vrp_nearest_neighbor(params: VRPSavingsParams) -> VRPSavingsResult {
    let problem = build_vrp_problem(&params);
    let source = Rc::new(RefCell::new(VRPSourceStation::new("vrp-source", problem)));
    let route = Rc::new(RefCell::new(NearestNeighborRouteStation::new(
        "nearest-neighbor-route",
    )));
    let sink = Rc::new(RefCell::new(VRPSinkStation::new("vrp-sink")));

    source.borrow_mut().core_mut().pipe(
        route.clone() as StationRef,
        VRPSourceStation::CH_PROBLEM,
        NearestNeighborRouteStation::CH_PROBLEM,
    );
    route.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        NearestNeighborRouteStation::CH_RESULT,
        VRPSinkStation::CH_RESULT,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            route.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("vrp-nearest-neighbor did not produce a result"));
    let stations = [
        StationOrId::Id("vrp-source".to_string()),
        StationOrId::Id("nearest-neighbor-route".to_string()),
        StationOrId::Id("vrp-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(
            &stations[0],
            VRPSourceStation::CH_PROBLEM,
            &stations[1],
            Some(NearestNeighborRouteStation::CH_PROBLEM),
        ),
        channel_edge(
            &stations[1],
            NearestNeighborRouteStation::CH_RESULT,
            &stations[2],
            Some(VRPSinkStation::CH_RESULT),
        ),
    ];
    result.topology = station_graph(
        &stations,
        &["VRPProblemToken".to_string(), "VRPResultToken".to_string()],
        &edges,
    );
    result
}

pub fn run_vrp_exact(params: VRPSavingsParams) -> VRPSavingsResult {
    let problem = build_vrp_problem(&params);
    solve_vrp_exact_problem(&problem)
}

fn solve_vrp_exact_problem(problem: &VRPProblemToken) -> VRPSavingsResult {
    let n = problem.customers.len();
    if n == 0 {
        return VRPSavingsResult {
            routes: Vec::new(),
            total_distance: 0.0,
            savings_considered: 0,
            topology: empty_station_graph(),
        };
    }
    if n > 16 {
        panic!("vrp-exact only practical for n <= 16, got {n}");
    }
    if problem.capacity <= 0.0 {
        panic!("vrp-exact: vehicle capacity must be positive");
    }
    if problem
        .customers
        .iter()
        .any(|customer| customer.demand > problem.capacity + 1e-9)
    {
        panic!("vrp-exact: customer demand exceeds vehicle capacity");
    }

    let full = (1usize << n) - 1;
    let mut demand = vec![0.0; 1usize << n];
    for mask in 1usize..=full {
        let bit = mask & mask.wrapping_neg();
        let idx = bit.trailing_zeros() as usize;
        demand[mask] = demand[mask ^ bit] + problem.customers[idx].demand;
    }

    let mut path_cost = vec![vec![f64::INFINITY; n]; 1usize << n];
    let mut path_parent = vec![vec![None; n]; 1usize << n];
    for i in 0..n {
        let bit = 1usize << i;
        let customer = &problem.customers[i];
        path_cost[bit][i] = dist(problem.depot.x, problem.depot.y, customer.x, customer.y);
    }
    for mask in 1usize..=full {
        for last in 0..n {
            if mask & (1usize << last) == 0 {
                continue;
            }
            let prev_mask = mask ^ (1usize << last);
            if prev_mask == 0 {
                continue;
            }
            let mut best = path_cost[mask][last];
            let mut best_prev = path_parent[mask][last];
            for prev in 0..n {
                if prev_mask & (1usize << prev) == 0 {
                    continue;
                }
                let candidate = path_cost[prev_mask][prev]
                    + dist(
                        problem.customers[prev].x,
                        problem.customers[prev].y,
                        problem.customers[last].x,
                        problem.customers[last].y,
                    );
                if candidate < best {
                    best = candidate;
                    best_prev = Some(prev);
                }
            }
            path_cost[mask][last] = best;
            path_parent[mask][last] = best_prev;
        }
    }

    let mut route_cost = vec![f64::INFINITY; 1usize << n];
    let mut route_last = vec![None; 1usize << n];
    let mut feasible_route_masks = Vec::new();
    for mask in 1usize..=full {
        if demand[mask] > problem.capacity + 1e-9 {
            continue;
        }
        feasible_route_masks.push(mask);
        for last in 0..n {
            if mask & (1usize << last) == 0 {
                continue;
            }
            let last_customer = &problem.customers[last];
            let candidate = path_cost[mask][last]
                + dist(
                    last_customer.x,
                    last_customer.y,
                    problem.depot.x,
                    problem.depot.y,
                );
            if candidate < route_cost[mask] {
                route_cost[mask] = candidate;
                route_last[mask] = Some(last);
            }
        }
    }

    let mut cover_cost = vec![f64::INFINITY; 1usize << n];
    let mut cover_choice = vec![0usize; 1usize << n];
    cover_cost[0] = 0.0;
    for mask in 1usize..=full {
        let mut sub = mask;
        while sub > 0 {
            if route_cost[sub].is_finite() {
                let remaining = mask ^ sub;
                let candidate = cover_cost[remaining] + route_cost[sub];
                if candidate < cover_cost[mask] {
                    cover_cost[mask] = candidate;
                    cover_choice[mask] = sub;
                }
            }
            sub = (sub - 1) & mask;
        }
    }
    if !cover_cost[full].is_finite() {
        panic!("vrp-exact: no feasible route cover found");
    }

    let mut routes = Vec::new();
    let mut mask = full;
    while mask > 0 {
        let route_mask = cover_choice[mask];
        if route_mask == 0 {
            panic!("vrp-exact: failed to reconstruct route cover");
        }
        let order = reconstruct_vrp_route(route_mask, &route_last, &path_parent);
        let customers: Vec<VRPCustomer> = order
            .iter()
            .map(|&idx| problem.customers[idx].clone())
            .collect();
        routes.push(VRPRoute {
            customers: customers
                .iter()
                .map(|customer| customer.id.clone())
                .collect(),
            load: demand[route_mask],
            distance: route_distance(problem.depot, &customers),
        });
        mask ^= route_mask;
    }
    routes.sort_by(|a, b| a.customers.cmp(&b.customers));
    let total_distance = routes.iter().map(|route| route.distance).sum();
    VRPSavingsResult {
        routes,
        total_distance,
        savings_considered: feasible_route_masks.len(),
        topology: empty_station_graph(),
    }
}

fn reconstruct_vrp_route(
    route_mask: usize,
    route_last: &[Option<usize>],
    path_parent: &[Vec<Option<usize>>],
) -> Vec<usize> {
    let mut mask = route_mask;
    let mut last = route_last[route_mask].expect("route last");
    let mut reversed = Vec::new();
    loop {
        reversed.push(last);
        let Some(parent) = path_parent[mask][last] else {
            break;
        };
        mask ^= 1usize << last;
        last = parent;
    }
    reversed.reverse();
    reversed
}

// =============================================================================
// Scheduling (job-shop dispatch / exact job-shop / flow-shop NEH)
// =============================================================================

const CH_JOB: &str = "job";
const CH_SCHEDULE: &str = "schedule";
const CH_FLOW_JOB: &str = "flow-job";
const CH_FLOW_SEQUENCE: &str = "flow-sequence";
const CH_FLOW_SCHEDULE: &str = "flow-schedule";

/// Job-shop dispatch rule (`'fifo' | 'spt' | 'edd'` string-union → enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchRule {
    Fifo,
    Spt,
    Edd,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobOperation {
    pub machine: String,
    pub duration: f64,
}

#[derive(Clone, Debug)]
pub struct JobShopJob {
    pub id: String,
    pub due: Option<f64>,
    pub operations: Vec<JobOperation>,
}

#[derive(Clone, Debug, Default)]
pub struct JobShopDispatchParams {
    pub jobs: Option<Vec<JobShopJob>>,
    pub rule: Option<DispatchRule>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledOperation {
    pub job_id: String,
    pub op_index: usize,
    pub machine: String,
    pub start: f64,
    pub finish: f64,
}

#[derive(Clone, Debug)]
pub struct JobShopDispatchResult {
    pub schedule: Vec<ScheduledOperation>,
    pub makespan: f64,
    pub total_flow_time: f64,
    pub topology: StationGraphSummary,
}

#[derive(Clone, Debug)]
pub struct FlowShopJob {
    pub id: String,
    pub processing_times: Vec<f64>,
    pub due: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct FlowShopNEHParams {
    pub jobs: Option<Vec<FlowShopJob>>,
}

#[derive(Clone, Debug)]
pub struct FlowShopNEHResult {
    pub sequence: Vec<String>,
    pub schedule: Vec<ScheduledOperation>,
    pub makespan: f64,
    pub total_flow_time: f64,
    pub topology: StationGraphSummary,
}

#[derive(Clone, Debug)]
pub struct JobToken {
    pub job: JobShopJob,
}

#[derive(Clone, Debug)]
pub struct ScheduleToken {
    pub result: JobShopDispatchResult,
}

/// Emits each job once.
pub struct JobSourceStation {
    core: StationCore,
    jobs: Vec<JobShopJob>,
    emitted: bool,
}

impl JobSourceStation {
    pub const CH_JOB: &'static str = CH_JOB;
    pub fn new(id: impl Into<String>, jobs: Vec<JobShopJob>) -> Self {
        JobSourceStation {
            core: StationCore::new(id),
            jobs,
            emitted: false,
        }
    }
}

impl DESStation for JobSourceStation {
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
        let jobs = self.jobs.clone();
        for job in jobs {
            self.core.emit(Rc::new(JobToken { job }), Self::CH_JOB);
        }
        self.emitted = true;
    }
}

/// Accumulates jobs, then builds a dispatch schedule under the chosen rule.
pub struct DispatchSchedulerStation {
    core: StationCore,
    rule: DispatchRule,
    jobs: Vec<JobShopJob>,
    scheduled: bool,
}

impl DispatchSchedulerStation {
    pub const CH_JOB: &'static str = CH_JOB;
    pub const CH_SCHEDULE: &'static str = CH_SCHEDULE;
    pub fn new(id: impl Into<String>, rule: DispatchRule) -> Self {
        DispatchSchedulerStation {
            core: StationCore::new(id),
            rule,
            jobs: Vec::new(),
            scheduled: false,
        }
    }
}

impl DESStation for DispatchSchedulerStation {
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
        self.core.inbox_size(Self::CH_JOB) > 0 || (!self.scheduled && !self.jobs.is_empty())
    }
    fn run_time_step(&mut self) {
        let incoming = self.core.drain::<JobToken>(Self::CH_JOB);
        let incoming_len = incoming.len();
        self.jobs
            .extend(incoming.into_iter().map(|t| t.job.clone()));
        if incoming_len > 0 {
            return;
        }
        if self.scheduled || self.jobs.is_empty() {
            return;
        }
        let result = dispatch_schedule(&self.jobs, self.rule);
        self.core
            .emit(Rc::new(ScheduleToken { result }), Self::CH_SCHEDULE);
        self.scheduled = true;
    }
}

/// Keeps the latest [`JobShopDispatchResult`].
pub struct ScheduleSinkStation {
    core: StationCore,
    pub result: Option<JobShopDispatchResult>,
}

impl ScheduleSinkStation {
    pub const CH_SCHEDULE: &'static str = CH_SCHEDULE;
    pub fn new(id: impl Into<String>) -> Self {
        ScheduleSinkStation {
            core: StationCore::new(id),
            result: None,
        }
    }
}

impl DESStation for ScheduleSinkStation {
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
        self.core.inbox_size(Self::CH_SCHEDULE) > 0
    }
    fn run_time_step(&mut self) {
        let schedules = self.core.drain::<ScheduleToken>(Self::CH_SCHEDULE);
        if let Some(last) = schedules.last() {
            self.result = Some(last.result.clone());
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowJobToken {
    pub job: FlowShopJob,
}

#[derive(Clone, Debug)]
pub struct FlowSequenceToken {
    pub jobs: Vec<FlowShopJob>,
}

#[derive(Clone, Debug)]
pub struct FlowScheduleToken {
    pub result: FlowShopNEHResult,
}

/// Emits each flow-shop job once.
pub struct FlowShopJobSourceStation {
    core: StationCore,
    jobs: Vec<FlowShopJob>,
    emitted: bool,
}

impl FlowShopJobSourceStation {
    pub const CH_JOB: &'static str = CH_FLOW_JOB;
    pub fn new(id: impl Into<String>, jobs: Vec<FlowShopJob>) -> Self {
        FlowShopJobSourceStation {
            core: StationCore::new(id),
            jobs,
            emitted: false,
        }
    }
}

impl DESStation for FlowShopJobSourceStation {
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
        let jobs = self.jobs.clone();
        for job in jobs {
            self.core.emit(Rc::new(FlowJobToken { job }), Self::CH_JOB);
        }
        self.emitted = true;
    }
}

/// Accumulates jobs, then computes the NEH insertion sequence.
pub struct NEHSequenceStation {
    core: StationCore,
    jobs: Vec<FlowShopJob>,
    sequenced: bool,
}

impl NEHSequenceStation {
    pub const CH_JOB: &'static str = CH_FLOW_JOB;
    pub const CH_SEQUENCE: &'static str = CH_FLOW_SEQUENCE;
    pub fn new(id: impl Into<String>) -> Self {
        NEHSequenceStation {
            core: StationCore::new(id),
            jobs: Vec::new(),
            sequenced: false,
        }
    }
}

impl DESStation for NEHSequenceStation {
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
        self.core.inbox_size(Self::CH_JOB) > 0 || (!self.sequenced && !self.jobs.is_empty())
    }
    fn run_time_step(&mut self) {
        let incoming = self.core.drain::<FlowJobToken>(Self::CH_JOB);
        let incoming_len = incoming.len();
        self.jobs
            .extend(incoming.into_iter().map(|t| t.job.clone()));
        if incoming_len > 0 {
            return;
        }
        if self.sequenced || self.jobs.is_empty() {
            return;
        }
        self.core.emit(
            Rc::new(FlowSequenceToken {
                jobs: neh_sequence(&self.jobs),
            }),
            Self::CH_SEQUENCE,
        );
        self.sequenced = true;
    }
}

/// Builds the flow-shop schedule (and metrics) for a fixed sequence.
pub struct FlowShopScheduleStation {
    core: StationCore,
}

impl FlowShopScheduleStation {
    pub const CH_SEQUENCE: &'static str = CH_FLOW_SEQUENCE;
    pub const CH_SCHEDULE: &'static str = CH_FLOW_SCHEDULE;
    pub fn new(id: impl Into<String>) -> Self {
        FlowShopScheduleStation {
            core: StationCore::new(id),
        }
    }
}

impl DESStation for FlowShopScheduleStation {
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
        self.core.inbox_size(Self::CH_SEQUENCE) > 0
    }
    fn run_time_step(&mut self) {
        for token in self.core.drain::<FlowSequenceToken>(Self::CH_SEQUENCE) {
            let schedule = build_flow_shop_schedule(&token.jobs);
            let makespan = schedule.iter().map(|op| op.finish).fold(0.0_f64, f64::max);
            let total_flow_time: f64 = token
                .jobs
                .iter()
                .map(|job| {
                    schedule
                        .iter()
                        .filter(|op| op.job_id == job.id)
                        .map(|op| op.finish)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .sum();
            let result = FlowShopNEHResult {
                sequence: token.jobs.iter().map(|job| job.id.clone()).collect(),
                schedule,
                makespan,
                total_flow_time,
                topology: empty_station_graph(),
            };
            self.core
                .emit(Rc::new(FlowScheduleToken { result }), Self::CH_SCHEDULE);
        }
    }
}

/// Keeps the latest [`FlowShopNEHResult`].
pub struct FlowShopSinkStation {
    core: StationCore,
    pub result: Option<FlowShopNEHResult>,
}

impl FlowShopSinkStation {
    pub const CH_SCHEDULE: &'static str = CH_FLOW_SCHEDULE;
    pub fn new(id: impl Into<String>) -> Self {
        FlowShopSinkStation {
            core: StationCore::new(id),
            result: None,
        }
    }
}

impl DESStation for FlowShopSinkStation {
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
        self.core.inbox_size(Self::CH_SCHEDULE) > 0
    }
    fn run_time_step(&mut self) {
        let schedules = self.core.drain::<FlowScheduleToken>(Self::CH_SCHEDULE);
        if let Some(last) = schedules.last() {
            self.result = Some(last.result.clone());
        }
    }
}

fn default_jobs() -> Vec<JobShopJob> {
    vec![
        JobShopJob {
            id: "J1".to_string(),
            due: Some(10.0),
            operations: vec![
                JobOperation {
                    machine: "M1".to_string(),
                    duration: 3.0,
                },
                JobOperation {
                    machine: "M2".to_string(),
                    duration: 2.0,
                },
            ],
        },
        JobShopJob {
            id: "J2".to_string(),
            due: Some(8.0),
            operations: vec![
                JobOperation {
                    machine: "M2".to_string(),
                    duration: 2.0,
                },
                JobOperation {
                    machine: "M1".to_string(),
                    duration: 4.0,
                },
            ],
        },
        JobShopJob {
            id: "J3".to_string(),
            due: Some(12.0),
            operations: vec![
                JobOperation {
                    machine: "M1".to_string(),
                    duration: 2.0,
                },
                JobOperation {
                    machine: "M2".to_string(),
                    duration: 3.0,
                },
            ],
        },
    ]
}

pub fn run_job_shop_dispatch(params: JobShopDispatchParams) -> JobShopDispatchResult {
    let jobs = non_empty_array(params.jobs.as_deref(), &default_jobs());
    validate_job_shop_jobs("job-shop-dispatch", &jobs);
    let source = Rc::new(RefCell::new(JobSourceStation::new("job-source", jobs)));
    let scheduler = Rc::new(RefCell::new(DispatchSchedulerStation::new(
        "dispatch-scheduler",
        params.rule.unwrap_or(DispatchRule::Spt),
    )));
    let sink = Rc::new(RefCell::new(ScheduleSinkStation::new("schedule-sink")));

    source.borrow_mut().core_mut().pipe(
        scheduler.clone() as StationRef,
        JobSourceStation::CH_JOB,
        DispatchSchedulerStation::CH_JOB,
    );
    scheduler.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        DispatchSchedulerStation::CH_SCHEDULE,
        ScheduleSinkStation::CH_SCHEDULE,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            scheduler.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("job-shop-dispatch did not produce a result"));
    let stations = [
        StationOrId::Id("job-source".to_string()),
        StationOrId::Id("dispatch-scheduler".to_string()),
        StationOrId::Id("schedule-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(
            &stations[0],
            JobSourceStation::CH_JOB,
            &stations[1],
            Some(DispatchSchedulerStation::CH_JOB),
        ),
        channel_edge(
            &stations[1],
            DispatchSchedulerStation::CH_SCHEDULE,
            &stations[2],
            Some(ScheduleSinkStation::CH_SCHEDULE),
        ),
    ];
    result.topology = station_graph(
        &stations,
        &["JobToken".to_string(), "ScheduleToken".to_string()],
        &edges,
    );
    result
}

pub fn run_job_shop_exact(params: JobShopDispatchParams) -> JobShopDispatchResult {
    let jobs = non_empty_array(params.jobs.as_deref(), &default_jobs());
    validate_job_shop_jobs("job-shop-exact", &jobs);
    solve_job_shop_exact(&jobs)
}

fn default_flow_shop_jobs() -> Vec<FlowShopJob> {
    vec![
        FlowShopJob {
            id: "F1".to_string(),
            processing_times: vec![2.0, 3.0, 2.0],
            due: None,
        },
        FlowShopJob {
            id: "F2".to_string(),
            processing_times: vec![4.0, 1.0, 3.0],
            due: None,
        },
        FlowShopJob {
            id: "F3".to_string(),
            processing_times: vec![3.0, 2.0, 4.0],
            due: None,
        },
        FlowShopJob {
            id: "F4".to_string(),
            processing_times: vec![2.0, 5.0, 1.0],
            due: None,
        },
    ]
}

pub fn run_flow_shop_neh(params: FlowShopNEHParams) -> FlowShopNEHResult {
    let jobs = non_empty_array(params.jobs.as_deref(), &default_flow_shop_jobs());
    let source = Rc::new(RefCell::new(FlowShopJobSourceStation::new(
        "flow-shop-source",
        jobs,
    )));
    let neh = Rc::new(RefCell::new(NEHSequenceStation::new(
        "neh-sequence-builder",
    )));
    let scheduler = Rc::new(RefCell::new(FlowShopScheduleStation::new(
        "flow-shop-scheduler",
    )));
    let sink = Rc::new(RefCell::new(FlowShopSinkStation::new("flow-shop-sink")));

    source.borrow_mut().core_mut().pipe(
        neh.clone() as StationRef,
        FlowShopJobSourceStation::CH_JOB,
        NEHSequenceStation::CH_JOB,
    );
    neh.borrow_mut().core_mut().pipe(
        scheduler.clone() as StationRef,
        NEHSequenceStation::CH_SEQUENCE,
        FlowShopScheduleStation::CH_SEQUENCE,
    );
    scheduler.borrow_mut().core_mut().pipe(
        sink.clone() as StationRef,
        FlowShopScheduleStation::CH_SCHEDULE,
        FlowShopSinkStation::CH_SCHEDULE,
    );

    run_iterative_des(
        vec![
            source.clone() as StationRef,
            neh.clone() as StationRef,
            scheduler.clone() as StationRef,
            sink.clone() as StationRef,
        ],
        IterativeRunOptions {
            shuffle: false,
            ..Default::default()
        },
    );

    let mut result = sink
        .borrow()
        .result
        .clone()
        .unwrap_or_else(|| panic!("flow-shop-neh did not produce a result"));
    let stations = [
        StationOrId::Id("flow-shop-source".to_string()),
        StationOrId::Id("neh-sequence-builder".to_string()),
        StationOrId::Id("flow-shop-scheduler".to_string()),
        StationOrId::Id("flow-shop-sink".to_string()),
    ];
    let edges = vec![
        channel_edge(
            &stations[0],
            FlowShopJobSourceStation::CH_JOB,
            &stations[1],
            Some(NEHSequenceStation::CH_JOB),
        ),
        channel_edge(
            &stations[1],
            NEHSequenceStation::CH_SEQUENCE,
            &stations[2],
            Some(FlowShopScheduleStation::CH_SEQUENCE),
        ),
        channel_edge(
            &stations[2],
            FlowShopScheduleStation::CH_SCHEDULE,
            &stations[3],
            Some(FlowShopSinkStation::CH_SCHEDULE),
        ),
    ];
    result.topology = station_graph(
        &stations,
        &[
            "FlowJobToken".to_string(),
            "FlowSequenceToken".to_string(),
            "FlowScheduleToken".to_string(),
        ],
        &edges,
    );
    result
}

// =============================================================================
// Math / algorithm helpers
// =============================================================================

fn qp_objective(q: &[Vec<f64>], c: &[f64], x: &[f64]) -> f64 {
    let qx: Vec<f64> = q.iter().map(|row| dot(row, x)).collect();
    0.5 * dot(x, &qx) + dot(c, x)
}

fn qp_gradient(q: &[Vec<f64>], c: &[f64], x: &[f64]) -> Vec<f64> {
    q.iter()
        .enumerate()
        .map(|(i, row)| dot(row, x) + c[i])
        .collect()
}

/// Exact assignment via memoised bitmask DP (TS `solveAssignmentDP`).
fn solve_assignment_dp(cost: &[Vec<f64>]) -> (Vec<i64>, f64) {
    let n = cost.len();
    let mut memo: HashMap<(usize, u32), (f64, Vec<usize>)> = HashMap::new();
    let (objective, assignment) = dp_solve(0, 0, cost, n, &mut memo);
    (
        assignment.into_iter().map(|c| c as i64).collect(),
        objective,
    )
}

fn dp_solve(
    row: usize,
    used_mask: u32,
    cost: &[Vec<f64>],
    n: usize,
    memo: &mut HashMap<(usize, u32), (f64, Vec<usize>)>,
) -> (f64, Vec<usize>) {
    if row == n {
        return (0.0, Vec::new());
    }
    if let Some(hit) = memo.get(&(row, used_mask)) {
        return hit.clone();
    }
    let mut best: (f64, Vec<usize>) = (f64::INFINITY, Vec::new());
    for col in 0..cost[row].len() {
        if used_mask & (1 << col) != 0 {
            continue;
        }
        let (tail_obj, tail_assign) = dp_solve(row + 1, used_mask | (1 << col), cost, n, memo);
        let objective = cost[row][col] + tail_obj;
        if objective < best.0 {
            let mut assignment = vec![col];
            assignment.extend(tail_assign);
            best = (objective, assignment);
        }
    }
    memo.insert((row, used_mask), best.clone());
    best
}

fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (ax - bx).hypot(ay - by)
}

fn route_distance(depot: Point, customers: &[VRPCustomer]) -> f64 {
    if customers.is_empty() {
        return 0.0;
    }
    let mut d = dist(depot.x, depot.y, customers[0].x, customers[0].y);
    for i in 1..customers.len() {
        d += dist(
            customers[i - 1].x,
            customers[i - 1].y,
            customers[i].x,
            customers[i].y,
        );
    }
    d + dist(
        customers[customers.len() - 1].x,
        customers[customers.len() - 1].y,
        depot.x,
        depot.y,
    )
}

/// Internal remaining-job cursor for dispatch scheduling.
struct Remaining {
    job_idx: usize,
    op_index: usize,
}

const JOB_SHOP_EPS: f64 = 1e-9;

fn validate_job_shop_jobs(model: &str, jobs: &[JobShopJob]) {
    if jobs.is_empty() {
        panic!("{model}: jobs must be non-empty");
    }
    let mut ids: HashSet<&str> = HashSet::new();
    for (job_idx, job) in jobs.iter().enumerate() {
        if job.id.trim().is_empty() {
            panic!("{model}: jobs[{job_idx}].id must be non-empty");
        }
        if !ids.insert(job.id.as_str()) {
            panic!("{model}: duplicate job id '{}'", job.id);
        }
        if job.operations.is_empty() {
            panic!("{model}: jobs[{job_idx}].operations must be non-empty");
        }
        for (op_idx, op) in job.operations.iter().enumerate() {
            if op.machine.trim().is_empty() {
                panic!("{model}: jobs[{job_idx}].operations[{op_idx}].machine must be non-empty");
            }
            require(Preconditions::non_negative(
                model,
                &format!("jobs[{job_idx}].operations[{op_idx}].duration"),
                op.duration,
            ));
        }
    }
}

fn job_shop_makespan(schedule: &[ScheduledOperation]) -> f64 {
    schedule.iter().map(|op| op.finish).fold(0.0_f64, f64::max)
}

fn job_shop_total_flow_time(jobs: &[JobShopJob], schedule: &[ScheduledOperation]) -> f64 {
    jobs.iter()
        .map(|job| {
            schedule
                .iter()
                .filter(|op| op.job_id == job.id)
                .map(|op| op.finish)
                .fold(0.0_f64, f64::max)
        })
        .sum()
}

struct JobShopExactState {
    next_ops: Vec<usize>,
    machine_ready: HashMap<String, f64>,
    job_ready: Vec<f64>,
    schedule: Vec<ScheduledOperation>,
}

struct JobShopExactSearch<'a> {
    jobs: &'a [JobShopJob],
    total_ops: usize,
    best_makespan: f64,
    best_total_flow_time: f64,
    best_schedule: Vec<ScheduledOperation>,
}

impl<'a> JobShopExactSearch<'a> {
    fn search(&mut self, state: &mut JobShopExactState) {
        if state.schedule.len() == self.total_ops {
            let makespan = state.job_ready.iter().copied().fold(0.0_f64, f64::max);
            let total_flow_time: f64 = state.job_ready.iter().sum();
            if makespan < self.best_makespan - JOB_SHOP_EPS
                || ((makespan - self.best_makespan).abs() <= JOB_SHOP_EPS
                    && total_flow_time < self.best_total_flow_time - JOB_SHOP_EPS)
            {
                self.best_makespan = makespan;
                self.best_total_flow_time = total_flow_time;
                self.best_schedule = state.schedule.clone();
            }
            return;
        }

        if self.lower_bound(state) > self.best_makespan + JOB_SHOP_EPS {
            return;
        }

        let mut candidates: Vec<(f64, f64, usize)> = Vec::new();
        for (job_idx, job) in self.jobs.iter().enumerate() {
            let op_index = state.next_ops[job_idx];
            if op_index >= job.operations.len() {
                continue;
            }
            let op = &job.operations[op_index];
            let start = state.job_ready[job_idx]
                .max(state.machine_ready.get(&op.machine).copied().unwrap_or(0.0));
            let finish = start + op.duration;
            candidates.push((finish, start, job_idx));
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        for (_, start, job_idx) in candidates {
            let op_index = state.next_ops[job_idx];
            let job = &self.jobs[job_idx];
            let op = &job.operations[op_index];
            let machine = op.machine.clone();
            let finish = start + op.duration;
            let previous_machine_ready = state.machine_ready.insert(machine.clone(), finish);
            let previous_job_ready = state.job_ready[job_idx];
            state.job_ready[job_idx] = finish;
            state.next_ops[job_idx] += 1;
            state.schedule.push(ScheduledOperation {
                job_id: job.id.clone(),
                op_index,
                machine: machine.clone(),
                start,
                finish,
            });

            self.search(state);

            state.schedule.pop();
            state.next_ops[job_idx] -= 1;
            state.job_ready[job_idx] = previous_job_ready;
            match previous_machine_ready {
                Some(value) => {
                    state.machine_ready.insert(machine, value);
                }
                None => {
                    state.machine_ready.remove(&machine);
                }
            }
        }
    }

    fn lower_bound(&self, state: &JobShopExactState) -> f64 {
        let mut bound = state.job_ready.iter().copied().fold(0.0_f64, f64::max);
        for (job_idx, job) in self.jobs.iter().enumerate() {
            let remaining: f64 = job.operations[state.next_ops[job_idx]..]
                .iter()
                .map(|op| op.duration)
                .sum();
            bound = bound.max(state.job_ready[job_idx] + remaining);
        }

        let mut machine_work: HashMap<&str, f64> = HashMap::new();
        for (job_idx, job) in self.jobs.iter().enumerate() {
            for op in &job.operations[state.next_ops[job_idx]..] {
                *machine_work.entry(op.machine.as_str()).or_insert(0.0) += op.duration;
            }
        }
        for (machine, work) in machine_work {
            let ready = state.machine_ready.get(machine).copied().unwrap_or(0.0);
            bound = bound.max(ready + work);
        }
        bound
    }
}

fn solve_job_shop_exact(jobs: &[JobShopJob]) -> JobShopDispatchResult {
    let total_ops: usize = jobs.iter().map(|job| job.operations.len()).sum();
    if total_ops > 20 {
        panic!("job-shop-exact only practical for <= 20 operations, got {total_ops}");
    }

    let incumbent = dispatch_schedule(jobs, DispatchRule::Spt);
    let mut search = JobShopExactSearch {
        jobs,
        total_ops,
        best_makespan: incumbent.makespan,
        best_total_flow_time: incumbent.total_flow_time,
        best_schedule: incumbent.schedule,
    };
    let mut state = JobShopExactState {
        next_ops: vec![0; jobs.len()],
        machine_ready: HashMap::new(),
        job_ready: vec![0.0; jobs.len()],
        schedule: Vec::new(),
    };
    search.search(&mut state);

    JobShopDispatchResult {
        schedule: search.best_schedule,
        makespan: search.best_makespan,
        total_flow_time: search.best_total_flow_time,
        topology: empty_station_graph(),
    }
}

fn dispatch_schedule(jobs: &[JobShopJob], rule: DispatchRule) -> JobShopDispatchResult {
    let mut machine_ready: HashMap<String, f64> = HashMap::new();
    let mut job_ready: HashMap<String, f64> = HashMap::new();
    let mut remaining: Vec<Remaining> = (0..jobs.len())
        .map(|job_idx| Remaining {
            job_idx,
            op_index: 0,
        })
        .collect();
    let mut schedule: Vec<ScheduledOperation> = Vec::new();

    while remaining
        .iter()
        .any(|r| r.op_index < jobs[r.job_idx].operations.len())
    {
        let mut ready: Vec<usize> = (0..remaining.len())
            .filter(|&k| remaining[k].op_index < jobs[remaining[k].job_idx].operations.len())
            .collect();
        ready.sort_by(|&a, &b| {
            let ra = &remaining[a];
            let rb = &remaining[b];
            match rule {
                DispatchRule::Edd => jobs[ra.job_idx]
                    .due
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&jobs[rb.job_idx].due.unwrap_or(f64::INFINITY)),
                DispatchRule::Spt => jobs[ra.job_idx].operations[ra.op_index]
                    .duration
                    .total_cmp(&jobs[rb.job_idx].operations[rb.op_index].duration),
                DispatchRule::Fifo => ra.job_idx.cmp(&rb.job_idx),
            }
        });
        let next_idx = ready[0];
        let job_idx = remaining[next_idx].job_idx;
        let op_index = remaining[next_idx].op_index;
        let op = jobs[job_idx].operations[op_index].clone();
        let start = machine_ready
            .get(&op.machine)
            .copied()
            .unwrap_or(0.0)
            .max(job_ready.get(&jobs[job_idx].id).copied().unwrap_or(0.0));
        let finish = start + op.duration;
        schedule.push(ScheduledOperation {
            job_id: jobs[job_idx].id.clone(),
            op_index,
            machine: op.machine.clone(),
            start,
            finish,
        });
        machine_ready.insert(op.machine.clone(), finish);
        job_ready.insert(jobs[job_idx].id.clone(), finish);
        remaining[next_idx].op_index += 1;
    }

    let makespan = job_shop_makespan(&schedule);
    let total_flow_time = job_shop_total_flow_time(jobs, &schedule);
    JobShopDispatchResult {
        schedule,
        makespan,
        total_flow_time,
        topology: empty_station_graph(),
    }
}

fn neh_sequence(jobs: &[FlowShopJob]) -> Vec<FlowShopJob> {
    let mut ordered = jobs.to_vec();
    ordered.sort_by(|a, b| total_processing_time(b).total_cmp(&total_processing_time(a)));
    let mut sequence: Vec<FlowShopJob> = Vec::new();
    for job in ordered {
        let mut best: Vec<FlowShopJob> = std::iter::once(job.clone())
            .chain(sequence.iter().cloned())
            .collect();
        let mut best_makespan = flow_shop_makespan(&best);
        for pos in 1..=sequence.len() {
            let mut candidate = sequence.clone();
            candidate.insert(pos, job.clone());
            let makespan = flow_shop_makespan(&candidate);
            if makespan < best_makespan {
                best = candidate;
                best_makespan = makespan;
            }
        }
        sequence = best;
    }
    sequence
}

fn build_flow_shop_schedule(sequence: &[FlowShopJob]) -> Vec<ScheduledOperation> {
    if sequence.is_empty() {
        return Vec::new();
    }
    let machines = sequence[0].processing_times.len();
    let mut machine_ready = zeros(machines);
    let mut schedule: Vec<ScheduledOperation> = Vec::new();
    for job in sequence {
        let mut job_ready = 0.0;
        for m in 0..machines {
            let start = machine_ready[m].max(job_ready);
            let finish = start + job.processing_times[m];
            schedule.push(ScheduledOperation {
                job_id: job.id.clone(),
                op_index: m,
                machine: format!("M{}", m + 1),
                start,
                finish,
            });
            machine_ready[m] = finish;
            job_ready = finish;
        }
    }
    schedule
}

fn flow_shop_makespan(sequence: &[FlowShopJob]) -> f64 {
    build_flow_shop_schedule(sequence)
        .iter()
        .map(|op| op.finish)
        .fold(0.0_f64, f64::max)
}

fn total_processing_time(job: &FlowShopJob) -> f64 {
    job.processing_times.iter().sum()
}

#[cfg(test)]
mod tests {
    //! Each model is driven to its known optimum on a small instance: the QP
    //! solvers minimise a strictly-convex box-constrained quadratic toward the
    //! unconstrained optimum Q⁻¹(−c) = (10/7, 16/7); both assignment solvers
    //! recover the cost-9 perfect matching of the default 3x3 matrix
    //! (w0→j1, w1→j0, w2→j2); the VRP and scheduling routines produce feasible,
    //! sensibly-bounded plans.

    use super::*;

    #[test]
    fn qp_projected_gradient_reaches_optimum() {
        let result = run_qp_projected_gradient(QPProjectedGradientParams::default());
        // min of ½xᵀQx + cᵀx for Q=[[4,1],[1,2]], c=[-8,-6] is x=(10/7, 16/7).
        assert!(
            (result.x[0] - 10.0 / 7.0).abs() < 1e-2,
            "x0 = {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 16.0 / 7.0).abs() < 1e-2,
            "x1 = {}",
            result.x[1]
        );
        assert!(
            (result.objective + 12.571).abs() < 1e-1,
            "objective = {}",
            result.objective
        );
    }

    #[test]
    fn qp_coordinate_descent_reaches_optimum() {
        let result = run_qp_coordinate_descent(QPProjectedGradientParams::default());
        assert!(
            (result.x[0] - 10.0 / 7.0).abs() < 1e-2,
            "x0 = {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 16.0 / 7.0).abs() < 1e-2,
            "x1 = {}",
            result.x[1]
        );
        assert!(
            result.gradient_norm <= 1e-6,
            "‖g‖ = {}",
            result.gradient_norm
        );
    }

    #[test]
    fn hungarian_matches_known_optimum() {
        let result = run_hungarian_assignment(AssignmentParams::default());
        // The minimal perfect matching of [[9,2,7],[6,4,3],[5,8,1]] is
        // w0→j1 (2) + w1→j0 (6) + w2→j2 (1) = 9.
        assert!(
            (result.objective - 9.0).abs() < 1e-9,
            "objective = {}",
            result.objective
        );
        assert_eq!(result.assignment, vec![1, 0, 2]);
    }

    #[test]
    fn auction_matches_hungarian_optimum() {
        let result = run_auction_assignment(AuctionAssignmentParams::default());
        // Auction must converge to a complete (no -1) assignment at the
        // ε-optimal cost (within n·ε of the exact optimum 9).
        assert!(
            result.assignment.iter().all(|&j| j >= 0),
            "incomplete: {:?}",
            result.assignment
        );
        assert!(
            result.objective <= 9.0 + 0.1,
            "objective = {}",
            result.objective
        );
    }

    #[test]
    fn vrp_savings_serves_all_customers_feasibly() {
        let result = run_vrp_savings(VRPSavingsParams::default());
        let served: usize = result.routes.iter().map(|r| r.customers.len()).sum();
        assert_eq!(served, 5);
        assert!(
            result.routes.iter().all(|r| r.load <= 5.0 + 1e-9),
            "capacity violated"
        );
        assert!(result.total_distance > 0.0);
    }

    #[test]
    fn vrp_nearest_neighbor_serves_all_customers_feasibly() {
        let result = run_vrp_nearest_neighbor(VRPSavingsParams::default());
        let served: usize = result.routes.iter().map(|r| r.customers.len()).sum();
        assert_eq!(served, 5);
        assert!(
            result.routes.iter().all(|r| r.load <= 5.0 + 1e-9),
            "capacity violated"
        );
    }

    #[test]
    fn job_shop_dispatch_schedules_all_operations() {
        let result = run_job_shop_dispatch(JobShopDispatchParams::default());
        // 3 jobs x 2 operations each = 6 scheduled operations.
        assert_eq!(result.schedule.len(), 6);
        assert!(result.makespan > 0.0);
        assert!(result.total_flow_time >= result.makespan);
    }

    #[test]
    fn flow_shop_neh_minimises_makespan_bound() {
        let result = run_flow_shop_neh(FlowShopNEHParams::default());
        assert_eq!(result.sequence.len(), 4);
        // 4 jobs x 3 machines = 12 scheduled operations.
        assert_eq!(result.schedule.len(), 12);
        // A lower bound on makespan is the busiest single job's total time.
        assert!(result.makespan >= 8.0, "makespan = {}", result.makespan);
    }
}
