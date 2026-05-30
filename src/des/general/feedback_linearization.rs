//! Port of `src/des/general/feedback-linearization.ts` — feedback
//! linearization / computed-torque control (Khalil 2002 ch.13) of an inverted
//! pendulum, driven as a DES closed loop.
//!
//! 1:1 behavioural move. The TS file imported `PlantBlock`, `ControllerBlock`,
//! `runClosedLoop` and `ClosedLoopResult` from `des-base/control-blocks`, which
//! is NOT in the allowed dependency list for this migration step. A MINIMAL
//! local equivalent of the lock-step closed-loop driver is defined below and
//! FLAGGED in the migration report; it reproduces the exact stepping semantics
//! of the TS driver (seed u0 = 0, plant tick then controller tick, identity
//! observation). The RK4 integrator takes a derivative closure via a generic
//! `Fn` bound. Deterministic (no RNG/clock).
#![allow(dead_code)]

use std::f64::consts::PI;
use std::rc::Rc;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

// =============================================================================
// LOCAL EQUIVALENT of des-base/control-blocks (FLAGGED: dependency not in the
// allowed list). Reproduces the lightweight lock-step block-diagram driver.
// =============================================================================

/// Composed base result of a closed-loop run (TS `ClosedLoopResult`).
#[derive(Clone, Debug)]
struct ClosedLoopResult {
    trajectory: Vec<Vec<f64>>,
    controls: Vec<Vec<f64>>,
    measurements: Vec<Vec<f64>>,
    num_steps: usize,
}

/// Plant block: owns continuous state advanced by `dynamics` (TS `PlantBlock`).
trait PlantBlock {
    fn dt(&self) -> f64;
    fn dynamics(&mut self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64>;
    fn observe(&self, x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }
}

/// Feedback controller block (TS `ControllerBlock`).
trait ControllerBlock {
    fn m_dim(&self) -> usize;
    fn get_dt(&self) -> f64 {
        1.0
    }
    fn u_bounds(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (None, None)
    }
    fn control_law(&mut self, y: &[f64], tick: usize, t: f64) -> Vec<f64>;
}

fn saturate(u: &mut [f64], u_min: Option<&[f64]>, u_max: Option<&[f64]>) {
    if let Some(lo) = u_min {
        for i in 0..u.len() {
            if u[i] < lo[i] {
                u[i] = lo[i];
            }
        }
    }
    if let Some(hi) = u_max {
        for i in 0..u.len() {
            if u[i] > hi[i] {
                u[i] = hi[i];
            }
        }
    }
}

fn run_closed_loop<P: PlantBlock, C: ControllerBlock>(
    plant: &mut P,
    x0: &[f64],
    controller: &mut C,
    num_steps: usize,
) -> ClosedLoopResult {
    let dt = plant.dt();
    let mut state = x0.to_vec();
    let mut trajectory = vec![state.clone()];
    let mut controls: Vec<Vec<f64>> = Vec::with_capacity(num_steps);
    let mut measurements: Vec<Vec<f64>> = Vec::with_capacity(num_steps);
    let mut last_u = vec![0.0_f64; controller.m_dim()];
    let (u_min, u_max) = controller.u_bounds();
    let mut tick = 0usize;
    for _ in 0..num_steps {
        let x_new = plant.dynamics(&state, &last_u, dt);
        state = x_new.clone();
        trajectory.push(x_new);
        let y = plant.observe(&state);
        measurements.push(y.clone());
        tick += 1;
        let t = tick as f64 * controller.get_dt();
        let mut u = controller.control_law(&y, tick, t);
        if u_min.is_some() || u_max.is_some() {
            saturate(&mut u, u_min.as_deref(), u_max.as_deref());
        }
        controls.push(u.clone());
        last_u = u;
    }
    ClosedLoopResult {
        trajectory,
        controls,
        measurements,
        num_steps,
    }
}

// -----------------------------------------------------------------------------
// PLANT: SIMPLE PENDULUM
// -----------------------------------------------------------------------------

/// Physical parameters of the mass-on-a-rod pendulum (TS `PendulumParams`).
#[derive(Clone, Copy, Debug)]
pub struct PendulumParams {
    /// mass at tip [kg]
    pub m: f64,
    /// length [m]
    pub l: f64,
    /// gravity [m/s²]
    pub g: f64,
    /// viscous damping [N·m·s/rad]
    pub c: f64,
}

/// Inverted-pendulum plant integrated with RK4 (TS `PendulumPlant`).
struct PendulumPlant {
    params: PendulumParams,
    dt: f64,
}

impl PendulumPlant {
    fn new(params: PendulumParams, dt: f64) -> Self {
        PendulumPlant { params, dt }
    }
}

impl PlantBlock for PendulumPlant {
    fn dt(&self) -> f64 {
        self.dt
    }
    fn dynamics(&mut self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
        // RK4 for accuracy. The state is [θ, θ̇].
        let params = self.params;
        let u0 = u[0];
        let f = |xx: &[f64]| -> Vec<f64> {
            let (theta, theta_d) = (xx[0], xx[1]);
            let PendulumParams { m, l, g, c } = params;
            let theta_dd = -(g / l) * theta.sin() - (c / (m * l * l)) * theta_d + (1.0 / (m * l * l)) * u0;
            vec![theta_d, theta_dd]
        };
        rk4(x, f, dt)
    }
}

// -----------------------------------------------------------------------------
// FEEDBACK-LINEARIZATION CONTROLLER (PD ON LINEARISED LOOP)
// -----------------------------------------------------------------------------

/// Desired trajectory sample (θ_d, θ̇_d, θ̈_d) returned by the reference function.
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub theta: f64,
    pub theta_dot: f64,
    pub theta_ddot: f64,
}

/// Computed-torque controller cancelling the pendulum nonlinearity then closing
/// a PD loop on the linearised plant (TS `FeedbackLinearizationController`).
struct FeedbackLinearizationController {
    params: PendulumParams,
    kp: f64,
    kv: f64,
    reference: Rc<dyn Fn(f64) -> Reference>,
    dt_cache: f64,
    u_min: Option<Vec<f64>>,
    u_max: Option<Vec<f64>>,
    theta_d_history: Vec<f64>,
    theta_d_dot_history: Vec<f64>,
    error_history: Vec<f64>,
}

impl FeedbackLinearizationController {
    fn new(
        params: PendulumParams,
        kp: f64,
        kv: f64,
        reference: Rc<dyn Fn(f64) -> Reference>,
        dt: f64,
        u_bound: Option<f64>,
    ) -> Self {
        let (u_min, u_max) = match u_bound {
            Some(b) => (Some(vec![-b]), Some(vec![b])),
            None => (None, None),
        };
        FeedbackLinearizationController {
            params,
            kp,
            kv,
            reference,
            dt_cache: dt,
            u_min,
            u_max,
            theta_d_history: Vec::new(),
            theta_d_dot_history: Vec::new(),
            error_history: Vec::new(),
        }
    }
}

impl ControllerBlock for FeedbackLinearizationController {
    fn m_dim(&self) -> usize {
        1
    }
    fn get_dt(&self) -> f64 {
        self.dt_cache
    }
    fn u_bounds(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (self.u_min.clone(), self.u_max.clone())
    }
    fn control_law(&mut self, y: &[f64], _tick: usize, t: f64) -> Vec<f64> {
        let theta = y[0];
        let theta_d = y[1];
        let rf = (*self.reference)(t);
        let e = theta - rf.theta;
        let ed = theta_d - rf.theta_dot;
        let v = rf.theta_ddot - self.kv * ed - self.kp * e;
        let PendulumParams { m, l, g, c } = self.params;
        let a = -(g / l) * theta.sin() - (c / (m * l * l)) * theta_d;
        let b = 1.0 / (m * l * l);
        let tau = (1.0 / b) * (v - a);
        self.theta_d_history.push(rf.theta);
        self.theta_d_dot_history.push(rf.theta_dot);
        self.error_history.push(e);
        vec![tau]
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for `run_feedback_linearization` (TS `FeedbackLinearizationOpts`).
#[derive(Clone)]
#[derive(Default)]
pub struct FeedbackLinearizationOpts {
    /// Partial override of the pendulum parameters (each defaults individually).
    pub params: Option<PartialPendulumParams>,
    /// Initial angle θ₀ (rad). Default π (downward equilibrium).
    pub theta0: Option<f64>,
    /// Initial angular velocity θ̇₀ (rad/s). Default 0.
    pub theta_dot0: Option<f64>,
    /// Reference (θ_d, θ̇_d, θ̈_d). Default: sinusoid amplitude 1 rad at 0.5 Hz.
    pub reference: Option<Rc<dyn Fn(f64) -> Reference>>,
    /// PD gains. Default kp = 25, kv = 10 (critically damped, ω_n = 5).
    pub kp: Option<f64>,
    pub kv: Option<f64>,
    /// Saturation on torque |τ|. Default 100.
    pub u_bound: Option<f64>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
}


/// Per-field optional override of `PendulumParams` (TS `Partial<PendulumParams>`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PartialPendulumParams {
    pub m: Option<f64>,
    pub l: Option<f64>,
    pub g: Option<f64>,
    pub c: Option<f64>,
}

/// Result of a feedback-linearization run (TS
/// `FeedbackLinearizationResult extends ClosedLoopResult`, flattened).
#[derive(Clone, Debug)]
pub struct FeedbackLinearizationResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
    /// RMS tracking error in steady state.
    pub rms_error_steady_state: f64,
    /// Reference history (desired θ_d) for plotting.
    pub theta_d_history: Vec<f64>,
}

/// Run the feedback-linearization closed loop. Panics (TS `throw`) if any
/// precondition is violated.
pub fn run_feedback_linearization(opts: FeedbackLinearizationOpts) -> FeedbackLinearizationResult {
    run_feedback_linearization_impl(opts).unwrap_or_else(|e| panic!("{e}"))
}

fn run_feedback_linearization_impl(
    opts: FeedbackLinearizationOpts,
) -> Result<FeedbackLinearizationResult, PreconditionError> {
    let p = opts.params.unwrap_or_default();
    let params = PendulumParams {
        m: p.m.unwrap_or(1.0),
        l: p.l.unwrap_or(1.0),
        g: p.g.unwrap_or(9.81),
        c: p.c.unwrap_or(0.1),
    };
    let cls = "runFeedbackLinearization";
    // Mass and length must be positive — they appear as 1/(m l²) in the dynamics.
    Preconditions::positive(cls, "params.m", params.m)?;
    Preconditions::positive(cls, "params.l", params.l)?;
    Preconditions::non_negative(cls, "params.g", params.g)?;
    Preconditions::non_negative(cls, "params.c", params.c)?;
    // PD gains must be positive for closed-loop stability of the linearised system.
    Preconditions::positive(cls, "kp", opts.kp.unwrap_or(25.0))?;
    Preconditions::positive(cls, "kv", opts.kv.unwrap_or(10.0))?;
    if let Some(ub) = opts.u_bound {
        Preconditions::positive(cls, "uBound", ub)?;
    }
    Preconditions::positive(cls, "dt", opts.dt.unwrap_or(0.01))?;
    Preconditions::integer_in_range(cls, "numSteps", opts.num_steps.unwrap_or(1000) as f64, 1.0, 1e9)?;
    if let Some(t0) = opts.theta0 {
        Preconditions::finite(cls, "theta0", t0)?;
    }
    if let Some(td0) = opts.theta_dot0 {
        Preconditions::finite(cls, "thetaDot0", td0)?;
    }

    let theta0 = opts.theta0.unwrap_or(PI); // start hanging down
    let theta_dot0 = opts.theta_dot0.unwrap_or(0.0);
    let reference: Rc<dyn Fn(f64) -> Reference> = opts.reference.clone().unwrap_or_else(|| {
        Rc::new(|t: f64| {
            let w = 2.0 * PI * 0.5;
            Reference {
                theta: (w * t).sin(),
                theta_dot: w * (w * t).cos(),
                theta_ddot: -(w * w) * (w * t).sin(),
            }
        })
    });
    let kp = opts.kp.unwrap_or(25.0);
    let kv = opts.kv.unwrap_or(10.0);
    let dt = opts.dt.unwrap_or(0.01);
    let num_steps = opts.num_steps.unwrap_or(1000);

    let mut plant = PendulumPlant::new(params, dt);
    let mut ctrl = FeedbackLinearizationController::new(
        params,
        kp,
        kv,
        reference,
        dt,
        Some(opts.u_bound.unwrap_or(100.0)),
    );
    let closed = run_closed_loop(&mut plant, &[theta0, theta_dot0], &mut ctrl, num_steps);

    let half = ctrl.error_history.len() / 2;
    let tail = &ctrl.error_history[half..];
    let denom = tail.len().max(1) as f64;
    let rms = (tail.iter().map(|x| x * x).sum::<f64>() / denom).sqrt();

    Ok(FeedbackLinearizationResult {
        trajectory: closed.trajectory,
        controls: closed.controls,
        measurements: closed.measurements,
        num_steps: closed.num_steps,
        rms_error_steady_state: rms,
        theta_d_history: ctrl.theta_d_history.clone(),
    })
}

// -----------------------------------------------------------------------------
// RK4 STEP (used by the pendulum plant for accuracy)
// -----------------------------------------------------------------------------

/// One Runge–Kutta-4 step of `ẋ = f(x)` (TS `rk4`). `f` is a derivative closure.
fn rk4(x: &[f64], f: impl Fn(&[f64]) -> Vec<f64>, dt: f64) -> Vec<f64> {
    let k1 = f(x);
    let x2: Vec<f64> = x.iter().enumerate().map(|(i, xi)| xi + 0.5 * dt * k1[i]).collect();
    let k2 = f(&x2);
    let x3: Vec<f64> = x.iter().enumerate().map(|(i, xi)| xi + 0.5 * dt * k2[i]).collect();
    let k3 = f(&x3);
    let x4: Vec<f64> = x.iter().enumerate().map(|(i, xi)| xi + dt * k3[i]).collect();
    let k4 = f(&x4);
    x.iter()
        .enumerate()
        .map(|(i, xi)| xi + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_sinusoidal_reference_to_small_error() {
        let res = run_feedback_linearization(FeedbackLinearizationOpts::default());
        assert_eq!(res.trajectory.len(), res.num_steps + 1);
        assert_eq!(res.theta_d_history.len(), res.num_steps);
        // Exact cancellation + PD loop drives the steady-state error near zero.
        assert!(
            res.rms_error_steady_state < 0.1,
            "steady-state RMS error {} too large",
            res.rms_error_steady_state
        );
    }

    #[test]
    fn tracks_constant_reference() {
        let res = run_feedback_linearization(FeedbackLinearizationOpts {
            reference: Some(Rc::new(|_t: f64| Reference {
                theta: 0.5,
                theta_dot: 0.0,
                theta_ddot: 0.0,
            })),
            theta0: Some(0.0),
            num_steps: Some(2000),
            ..Default::default()
        });
        // Final angle should settle at the constant set-point 0.5 rad.
        let last = res.trajectory.last().unwrap();
        assert!((last[0] - 0.5).abs() < 1e-3, "final angle {}", last[0]);
    }

    #[test]
    #[should_panic]
    fn rejects_zero_length() {
        run_feedback_linearization(FeedbackLinearizationOpts {
            params: Some(PartialPendulumParams {
                l: Some(0.0),
                ..Default::default()
            }),
            ..Default::default()
        });
    }
}
