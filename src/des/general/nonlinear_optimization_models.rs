//! Port of `src/des/general/nonlinear-optimization-models.ts`
//! (module `des::general::nonlinear_optimization_models`).
//!
//! Newton/quasi-Newton and nonlinear least-squares routines as DES state-token
//! loops. Each model is a flat graph: source → update station (self-loop) →
//! sink, with movable state/result tokens.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * The two abstract update stations are template-method bases (one abstract
//!     `step` hook) → a hook trait with a default driver fn + the required
//!     hooks; concrete stations embed a small `*Core` bundle (the fields the TS
//!     base owned: `trace`, `max_iter`, `tol`, `points`) and implement the
//!     trait.
//!   * All matrices/vectors `number[][]`/`number[]` → `Vec<Vec<f64>>`/`Vec<f64>`
//!     (with [`dot`]/[`norm2`] from the ported des-base).
//!   * `validate*InitialState` `throw` on bad input → [`Preconditions`] guards
//!     whose `Err` is turned into a `panic!` (an invariant for the seed token).
//!   * Fully deterministic: no RNG/clock/Map.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::learning_optimization::{
    dot, empty_station_graph, non_empty_array, norm2, run_state_loop_pipeline, state_loop_topology,
    LatestTokenSinkStation, SingleTokenSourceStation, StationGraphSummary,
};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::IterativeRunOptions;
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};

/// The same `StationGraphSummary` exposed by the des-base port.
pub type NonlinearTopology = StationGraphSummary;

const CH_OPT_STATE: &str = "opt-state";
const CH_OPT_RESULT: &str = "opt-result";
const CH_NLS_STATE: &str = "nls-state";
const CH_NLS_RESULT: &str = "nls-result";

/// Panic with the precondition message on a failed guard (TS `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// ── Unconstrained optimisation ────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct UnconstrainedOptParams {
    pub x0: Option<Vec<f64>>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct UnconstrainedTraceEntry {
    pub iter: usize,
    pub objective: f64,
    pub gradient_norm: f64,
    pub x: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct UnconstrainedOptResult {
    pub x: Vec<f64>,
    pub objective: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub trace: Vec<UnconstrainedTraceEntry>,
    pub topology: NonlinearTopology,
}

/// Movable carrying the walker state. `H` is the optional BFGS inverse-Hessian.
#[derive(Clone, Debug)]
pub struct OptStateToken {
    pub iter: usize,
    pub x: Vec<f64>,
    pub h: Option<Vec<Vec<f64>>>,
}

/// Terminal result token.
#[derive(Clone, Debug)]
pub struct OptResultToken {
    pub result: UnconstrainedOptResult,
}

/// Fields the TS `abstract class UnconstrainedUpdateStation` owned, factored out
/// so concrete stations embed them.
pub struct UnconstrainedUpdateCore {
    pub trace: Vec<UnconstrainedTraceEntry>,
    pub max_iter: usize,
    pub tol: f64,
}

impl UnconstrainedUpdateCore {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        UnconstrainedUpdateCore {
            trace: Vec::new(),
            max_iter,
            tol,
        }
    }
}

/// Template-method base: the `run_update_step` driver calls the abstract hooks
/// (`objective`, `gradient`, `next_state`).
pub trait UnconstrainedUpdateStation: DESStation {
    const CH_STATE: &'static str = CH_OPT_STATE;
    const CH_RESULT: &'static str = CH_OPT_RESULT;

    fn update_core(&self) -> &UnconstrainedUpdateCore;
    fn update_core_mut(&mut self) -> &mut UnconstrainedUpdateCore;

    fn objective(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
    fn next_state(&self, state: &OptStateToken, gradient: &[f64]) -> OptStateToken;

    fn update_has_work(&self) -> bool {
        self.core().inbox_size(CH_OPT_STATE) > 0
    }

    fn run_update_step(&mut self) {
        let states = self.core_mut().drain::<OptStateToken>(CH_OPT_STATE);
        for state in states {
            let gradient = self.gradient(&state.x);
            let gradient_norm = norm2(&gradient);
            let objective = self.objective(&state.x);
            self.update_core_mut().trace.push(UnconstrainedTraceEntry {
                iter: state.iter,
                objective,
                gradient_norm,
                x: state.x.clone(),
            });
            if state.iter >= self.update_core().max_iter || gradient_norm <= self.update_core().tol
            {
                let result = UnconstrainedOptResult {
                    x: state.x.clone(),
                    objective,
                    gradient_norm,
                    iterations: state.iter,
                    trace: self.update_core().trace.clone(),
                    topology: empty_station_graph(),
                };
                self.core_mut()
                    .emit(Rc::new(OptResultToken { result }), CH_OPT_RESULT);
                continue;
            }
            let next = self.next_state(state.as_ref(), &gradient);
            self.core_mut().emit(Rc::new(next), CH_OPT_STATE);
        }
    }
}

/// Newton's method on the 2-D Rosenbrock function.
pub struct NewtonRosenbrockStation {
    core: StationCore,
    update: UnconstrainedUpdateCore,
}

impl NewtonRosenbrockStation {
    pub fn new(id: impl Into<String>, max_iter: usize, tol: f64) -> Self {
        NewtonRosenbrockStation {
            core: StationCore::new(id),
            update: UnconstrainedUpdateCore::new(max_iter, tol),
        }
    }
}

impl DESStation for NewtonRosenbrockStation {
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
        self.update_has_work()
    }
    fn run_time_step(&mut self) {
        self.run_update_step();
    }
}

impl UnconstrainedUpdateStation for NewtonRosenbrockStation {
    fn update_core(&self) -> &UnconstrainedUpdateCore {
        &self.update
    }
    fn update_core_mut(&mut self) -> &mut UnconstrainedUpdateCore {
        &mut self.update
    }
    fn objective(&self, x: &[f64]) -> f64 {
        rosenbrock(x)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        rosenbrock_grad(x)
    }
    fn next_state(&self, state: &OptStateToken, gradient: &[f64]) -> OptStateToken {
        let h = rosenbrock_hessian(&state.x);
        let step = solve2(&h, &[-gradient[0], -gradient[1]]);
        let alpha = backtracking(&state.x, &step, rosenbrock, gradient);
        OptStateToken {
            iter: state.iter + 1,
            x: vec![state.x[0] + alpha * step[0], state.x[1] + alpha * step[1]],
            h: None,
        }
    }
}

/// BFGS quasi-Newton on the 2-D Rosenbrock function.
pub struct BFGSRosenbrockStation {
    core: StationCore,
    update: UnconstrainedUpdateCore,
}

impl BFGSRosenbrockStation {
    pub fn new(id: impl Into<String>, max_iter: usize, tol: f64) -> Self {
        BFGSRosenbrockStation {
            core: StationCore::new(id),
            update: UnconstrainedUpdateCore::new(max_iter, tol),
        }
    }
}

impl DESStation for BFGSRosenbrockStation {
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
        self.update_has_work()
    }
    fn run_time_step(&mut self) {
        self.run_update_step();
    }
}

impl UnconstrainedUpdateStation for BFGSRosenbrockStation {
    fn update_core(&self) -> &UnconstrainedUpdateCore {
        &self.update
    }
    fn update_core_mut(&mut self) -> &mut UnconstrainedUpdateCore {
        &mut self.update
    }
    fn objective(&self, x: &[f64]) -> f64 {
        rosenbrock(x)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        rosenbrock_grad(x)
    }
    fn next_state(&self, state: &OptStateToken, gradient: &[f64]) -> OptStateToken {
        let h = state
            .h
            .clone()
            .unwrap_or_else(|| vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let step = vec![-dot(&h[0], gradient), -dot(&h[1], gradient)];
        let alpha = backtracking(&state.x, &step, rosenbrock, gradient);
        let x_next = vec![state.x[0] + alpha * step[0], state.x[1] + alpha * step[1]];
        let g_next = rosenbrock_grad(&x_next);
        let s = vec![x_next[0] - state.x[0], x_next[1] - state.x[1]];
        let y = vec![g_next[0] - gradient[0], g_next[1] - gradient[1]];
        let h_next = bfgs_inverse_update(&h, &s, &y);
        OptStateToken {
            iter: state.iter + 1,
            x: x_next,
            h: Some(h_next),
        }
    }
}

pub fn run_newton_rosenbrock(params: UnconstrainedOptParams) -> UnconstrainedOptResult {
    let x0 = non_empty_array(params.x0.as_deref(), &[-1.2, 1.0]);
    let update = NewtonRosenbrockStation::new(
        "newton-update",
        params.max_iter.unwrap_or(50),
        params.tol.unwrap_or(1e-8),
    );
    run_unconstrained("newton-state-source", update, x0)
}

pub fn run_bfgs_rosenbrock(params: UnconstrainedOptParams) -> UnconstrainedOptResult {
    let x0 = non_empty_array(params.x0.as_deref(), &[-1.2, 1.0]);
    let update = BFGSRosenbrockStation::new(
        "bfgs-update",
        params.max_iter.unwrap_or(100),
        params.tol.unwrap_or(1e-6),
    );
    run_unconstrained("bfgs-state-source", update, x0)
}

fn run_unconstrained<U: UnconstrainedUpdateStation + 'static>(
    source_id: &str,
    update: U,
    x0: Vec<f64>,
) -> UnconstrainedOptResult {
    let model = source_id.to_string();
    let x0_factory = x0.clone();
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        source_id,
        CH_OPT_STATE,
        move || OptStateToken {
            iter: 0,
            x: x0_factory.clone(),
            h: None,
        },
        move |t: &OptStateToken| validate_opt_initial_state(&model, t),
    )));
    let update_rc = Rc::new(RefCell::new(update));
    let update_id = update_rc.borrow().id().to_string();
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<OptResultToken>::new(
        "opt-result-sink",
        CH_OPT_RESULT,
    )));

    run_state_loop_pipeline(
        source.clone() as StationRef,
        update_rc.clone() as StationRef,
        sink.clone() as StationRef,
        CH_OPT_STATE,
        CH_OPT_RESULT,
        IterativeRunOptions {
            max_ticks: Some(500),
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
        &*update_rc.borrow(),
        &*sink.borrow(),
        CH_OPT_STATE,
        CH_OPT_RESULT,
        &["OptStateToken".to_string(), "OptResultToken".to_string()],
    );
    result
}

// ── Nonlinear least squares ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveFitPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct NonlinearLeastSquaresParams {
    pub points: Option<Vec<CurveFitPoint>>,
    pub initial: Option<Vec<f64>>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
    pub lambda: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct NlsTraceEntry {
    pub iter: usize,
    pub sse: f64,
    pub gradient_norm: f64,
    pub params: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct NonlinearLeastSquaresResult {
    pub params: Vec<f64>,
    pub sse: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub trace: Vec<NlsTraceEntry>,
    pub topology: NonlinearTopology,
}

#[derive(Clone, Debug)]
pub struct NLStateToken {
    pub iter: usize,
    pub params: Vec<f64>,
    pub lambda: f64,
}

#[derive(Clone, Debug)]
pub struct NLResultToken {
    pub result: NonlinearLeastSquaresResult,
}

/// Fields owned by the TS `abstract class NonlinearLeastSquaresStation`.
pub struct NlsCore {
    pub points: Vec<CurveFitPoint>,
    pub max_iter: usize,
    pub tol: f64,
    pub trace: Vec<NlsTraceEntry>,
}

impl NlsCore {
    pub fn new(points: Vec<CurveFitPoint>, max_iter: usize, tol: f64) -> Self {
        NlsCore {
            points,
            max_iter,
            tol,
            trace: Vec::new(),
        }
    }
}

/// Template-method base: the driver calls the abstract `damping` hook.
pub trait NonlinearLeastSquaresStation: DESStation {
    const CH_STATE: &'static str = CH_NLS_STATE;
    const CH_RESULT: &'static str = CH_NLS_RESULT;

    fn nls_core(&self) -> &NlsCore;
    fn nls_core_mut(&mut self) -> &mut NlsCore;

    fn damping(&self, state: &NLStateToken) -> f64;

    fn nls_has_work(&self) -> bool {
        self.core().inbox_size(CH_NLS_STATE) > 0
    }

    fn run_nls_step(&mut self) {
        let states = self.core_mut().drain::<NLStateToken>(CH_NLS_STATE);
        for state in states {
            let damping = self.damping(state.as_ref());
            let system = normal_equations(&state.params, &self.nls_core().points, damping);
            let gradient_norm = norm2(&system.gradient);
            let sse = exp_sse(&state.params, &self.nls_core().points);
            self.nls_core_mut().trace.push(NlsTraceEntry {
                iter: state.iter,
                sse,
                gradient_norm,
                params: state.params.clone(),
            });
            if state.iter >= self.nls_core().max_iter || gradient_norm <= self.nls_core().tol {
                let result = NonlinearLeastSquaresResult {
                    params: state.params.clone(),
                    sse,
                    gradient_norm,
                    iterations: state.iter,
                    trace: self.nls_core().trace.clone(),
                    topology: empty_station_graph(),
                };
                self.core_mut()
                    .emit(Rc::new(NLResultToken { result }), CH_NLS_RESULT);
                continue;
            }
            let step = solve_linear(&system.a, &system.b);
            let next: Vec<f64> = state
                .params
                .iter()
                .enumerate()
                .map(|(i, v)| v + step[i])
                .collect();
            self.core_mut().emit(
                Rc::new(NLStateToken {
                    iter: state.iter + 1,
                    params: next,
                    lambda: state.lambda,
                }),
                CH_NLS_STATE,
            );
        }
    }
}

/// Gauss–Newton (zero damping).
pub struct GaussNewtonStation {
    core: StationCore,
    nls: NlsCore,
}

impl GaussNewtonStation {
    pub fn new(
        id: impl Into<String>,
        points: Vec<CurveFitPoint>,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        GaussNewtonStation {
            core: StationCore::new(id),
            nls: NlsCore::new(points, max_iter, tol),
        }
    }
}

impl DESStation for GaussNewtonStation {
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
        self.nls_has_work()
    }
    fn run_time_step(&mut self) {
        self.run_nls_step();
    }
}

impl NonlinearLeastSquaresStation for GaussNewtonStation {
    fn nls_core(&self) -> &NlsCore {
        &self.nls
    }
    fn nls_core_mut(&mut self) -> &mut NlsCore {
        &mut self.nls
    }
    fn damping(&self, _state: &NLStateToken) -> f64 {
        0.0
    }
}

/// Levenberg–Marquardt (per-state damping `lambda`).
pub struct LevenbergMarquardtStation {
    core: StationCore,
    nls: NlsCore,
}

impl LevenbergMarquardtStation {
    pub fn new(
        id: impl Into<String>,
        points: Vec<CurveFitPoint>,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        LevenbergMarquardtStation {
            core: StationCore::new(id),
            nls: NlsCore::new(points, max_iter, tol),
        }
    }
}

impl DESStation for LevenbergMarquardtStation {
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
        self.nls_has_work()
    }
    fn run_time_step(&mut self) {
        self.run_nls_step();
    }
}

impl NonlinearLeastSquaresStation for LevenbergMarquardtStation {
    fn nls_core(&self) -> &NlsCore {
        &self.nls
    }
    fn nls_core_mut(&mut self) -> &mut NlsCore {
        &mut self.nls
    }
    fn damping(&self, state: &NLStateToken) -> f64 {
        state.lambda
    }
}

pub fn run_gauss_newton_curve_fit(
    params: NonlinearLeastSquaresParams,
) -> NonlinearLeastSquaresResult {
    let points = non_empty_array(params.points.as_deref(), &default_fit_points());
    let initial = non_empty_array(params.initial.as_deref(), &[1.0, -0.2]);
    let update = GaussNewtonStation::new(
        "gauss-newton-update",
        points,
        params.max_iter.unwrap_or(20),
        params.tol.unwrap_or(1e-8),
    );
    run_nls("gauss-newton-source", update, initial, 0.0)
}

pub fn run_levenberg_marquardt_curve_fit(
    params: NonlinearLeastSquaresParams,
) -> NonlinearLeastSquaresResult {
    let points = non_empty_array(params.points.as_deref(), &default_fit_points());
    let initial = non_empty_array(params.initial.as_deref(), &[1.0, -0.2]);
    let update = LevenbergMarquardtStation::new(
        "levenberg-marquardt-update",
        points,
        params.max_iter.unwrap_or(30),
        params.tol.unwrap_or(1e-8),
    );
    run_nls("lm-source", update, initial, params.lambda.unwrap_or(0.1))
}

fn run_nls<U: NonlinearLeastSquaresStation + 'static>(
    source_id: &str,
    update: U,
    initial: Vec<f64>,
    lambda: f64,
) -> NonlinearLeastSquaresResult {
    let model = source_id.to_string();
    let initial_factory = initial.clone();
    let source = Rc::new(RefCell::new(SingleTokenSourceStation::with_validator(
        source_id,
        CH_NLS_STATE,
        move || NLStateToken {
            iter: 0,
            params: initial_factory.clone(),
            lambda,
        },
        move |t: &NLStateToken| validate_nls_initial_state(&model, t),
    )));
    let update_rc = Rc::new(RefCell::new(update));
    let update_id = update_rc.borrow().id().to_string();
    let sink = Rc::new(RefCell::new(LatestTokenSinkStation::<NLResultToken>::new(
        "nls-result-sink",
        CH_NLS_RESULT,
    )));

    run_state_loop_pipeline(
        source.clone() as StationRef,
        update_rc.clone() as StationRef,
        sink.clone() as StationRef,
        CH_NLS_STATE,
        CH_NLS_RESULT,
        IterativeRunOptions {
            max_ticks: Some(200),
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
        &*update_rc.borrow(),
        &*sink.borrow(),
        CH_NLS_STATE,
        CH_NLS_RESULT,
        &["NLStateToken".to_string(), "NLResultToken".to_string()],
    );
    result
}

// ── Validators ────────────────────────────────────────────────────────────────

fn validate_opt_initial_state(model: &str, token: &OptStateToken) {
    require(Preconditions::integer_in_range(
        model,
        "iter",
        token.iter as f64,
        0.0,
        1e9,
    ));
    require(Preconditions::length_eq(model, "x0", &token.x, 2));
    require(Preconditions::all_finite(model, "x0", &token.x));
    if let Some(h) = &token.h {
        require(Preconditions::length_eq(model, "H", h, 2));
        require(Preconditions::length_eq(model, "H[0]", &h[0], 2));
        require(Preconditions::length_eq(model, "H[1]", &h[1], 2));
        require(Preconditions::all_finite(model, "H[0]", &h[0]));
        require(Preconditions::all_finite(model, "H[1]", &h[1]));
    }
}

fn validate_nls_initial_state(model: &str, token: &NLStateToken) {
    require(Preconditions::integer_in_range(
        model,
        "iter",
        token.iter as f64,
        0.0,
        1e9,
    ));
    require(Preconditions::length_eq(model, "initial", &token.params, 2));
    require(Preconditions::all_finite(model, "initial", &token.params));
    require(Preconditions::non_negative(model, "lambda", token.lambda));
}

// ── Math helpers ──────────────────────────────────────────────────────────────

fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    vec![
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ]
}

fn rosenbrock_hessian(x: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0], -400.0 * x[0]],
        vec![-400.0 * x[0], 200.0],
    ]
}

fn solve2(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-12 {
        return b.to_vec();
    }
    vec![
        (b[0] * a[1][1] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - b[0] * a[1][0]) / det,
    ]
}

fn backtracking(x: &[f64], p: &[f64], f: fn(&[f64]) -> f64, g: &[f64]) -> f64 {
    let mut alpha = 1.0;
    let f0 = f(x);
    let slope = dot(g, p);
    while alpha > 1e-8 {
        let next: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + alpha * p[i])
            .collect();
        if f(&next) <= f0 + 1e-4 * alpha * slope {
            return alpha;
        }
        alpha *= 0.5;
    }
    alpha
}

fn bfgs_inverse_update(h: &[Vec<f64>], s: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    let ys = dot(y, s);
    if ys <= 1e-12 {
        return h.iter().map(|row| row.clone()).collect();
    }
    let rho = 1.0 / ys;
    let hy = vec![dot(&h[0], y), dot(&h[1], y)];
    let y_hy = dot(y, &hy);
    let mut out: Vec<Vec<f64>> = h.iter().map(|row| row.clone()).collect();
    for i in 0..2 {
        for j in 0..2 {
            out[i][j] +=
                (1.0 + y_hy * rho) * rho * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
        }
    }
    out
}

fn exp_residuals(params: &[f64], points: &[CurveFitPoint]) -> Vec<f64> {
    let (a, b) = (params[0], params[1]);
    points.iter().map(|p| a * (b * p.x).exp() - p.y).collect()
}

fn exp_jacobian(params: &[f64], points: &[CurveFitPoint]) -> Vec<Vec<f64>> {
    let (a, b) = (params[0], params[1]);
    points
        .iter()
        .map(|p| {
            let e = (b * p.x).exp();
            vec![e, a * p.x * e]
        })
        .collect()
}

fn exp_sse(params: &[f64], points: &[CurveFitPoint]) -> f64 {
    exp_residuals(params, points).iter().map(|r| r * r).sum()
}

/// `A x = b` normal equations for the 2-parameter exponential fit, plus the
/// (unscaled) gradient.
struct NormalSystem {
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    gradient: Vec<f64>,
}

fn normal_equations(params: &[f64], points: &[CurveFitPoint], lambda: f64) -> NormalSystem {
    let r = exp_residuals(params, points);
    let j = exp_jacobian(params, points);
    let mut a = vec![vec![lambda, 0.0], vec![0.0, lambda]];
    let mut b = vec![0.0, 0.0];
    let mut gradient = vec![0.0, 0.0];
    for k in 0..points.len() {
        for i in 0..2 {
            b[i] -= j[k][i] * r[k];
            gradient[i] += 2.0 * j[k][i] * r[k];
            for jj in 0..2 {
                a[i][jj] += j[k][i] * j[k][jj];
            }
        }
    }
    NormalSystem { a, b, gradient }
}

fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    solve2(a, b)
}

fn default_fit_points() -> Vec<CurveFitPoint> {
    vec![
        CurveFitPoint { x: 0.0, y: 2.00 },
        CurveFitPoint { x: 1.0, y: 1.22 },
        CurveFitPoint { x: 2.0, y: 0.74 },
        CurveFitPoint { x: 3.0, y: 0.45 },
        CurveFitPoint { x: 4.0, y: 0.27 },
    ]
}

#[cfg(test)]
mod tests {
    //! Each solver drives a known small problem to its optimum: Newton and BFGS
    //! minimise the 2-D Rosenbrock function (global minimum at (1, 1) with value
    //! 0), and the two least-squares solvers fit a decaying exponential whose
    //! residual sum of squares should fall to near zero.

    use super::*;

    #[test]
    fn newton_reaches_rosenbrock_minimum() {
        let result = run_newton_rosenbrock(UnconstrainedOptParams::default());
        assert!(result.objective < 1e-8, "objective = {}", result.objective);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-4 && (result.x[1] - 1.0).abs() < 1e-4,
            "x = {:?}",
            result.x
        );
        assert!(result.gradient_norm <= 1e-8);
    }

    #[test]
    fn bfgs_reaches_rosenbrock_minimum() {
        let result = run_bfgs_rosenbrock(UnconstrainedOptParams {
            max_iter: Some(200),
            tol: Some(1e-6),
            ..Default::default()
        });
        assert!(result.objective < 1e-4, "objective = {}", result.objective);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-2 && (result.x[1] - 1.0).abs() < 1e-2,
            "x = {:?}",
            result.x
        );
    }

    #[test]
    fn gauss_newton_fits_exponential() {
        let result = run_gauss_newton_curve_fit(NonlinearLeastSquaresParams::default());
        // Default data ~ 2 * exp(-0.5 x); SSE should be tiny.
        assert!(result.sse < 1e-2, "sse = {}", result.sse);
        assert!(
            (result.params[0] - 2.0).abs() < 0.2,
            "a = {}",
            result.params[0]
        );
        assert!(result.params[1] < 0.0, "b = {}", result.params[1]);
    }

    #[test]
    fn levenberg_marquardt_fits_exponential() {
        let result = run_levenberg_marquardt_curve_fit(NonlinearLeastSquaresParams::default());
        assert!(result.sse < 1e-1, "sse = {}", result.sse);
        assert!(
            (result.params[0] - 2.0).abs() < 0.3,
            "a = {}",
            result.params[0]
        );
    }
}
