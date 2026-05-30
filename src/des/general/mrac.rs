//! Port of `src/des/general/mrac.ts` — Model-Reference Adaptive Control
//! (Whitaker 1958 MIT-rule; Narendra & Annaswamy 1989 Lyapunov MRAC) driven as
//! a DES closed loop.
//!
//! 1:1 behavioural move. The TS file imported `PlantBlock`, `ControllerBlock`,
//! `runClosedLoop` and `ClosedLoopResult` from `des-base/control-blocks`, which
//! is NOT in the allowed dependency list for this migration step. A MINIMAL
//! local equivalent of the lock-step closed-loop driver is defined below and
//! FLAGGED in the migration report; it reproduces the exact stepping semantics
//! of the TS driver (seed u0 = 0, plant tick then controller tick, identity
//! observation). All numerics are `f64`; deterministic (no RNG/clock).
#![allow(dead_code)]

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

/// `Math.sign` semantics (returns 0 for 0, unlike `f64::signum`).
fn math_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// -----------------------------------------------------------------------------
// PLANT (TRUE PARAMETERS HIDDEN FROM CONTROLLER)
// -----------------------------------------------------------------------------

/// First-order plant `ẋ = a x + b u` with hidden `a, b` (TS `UnknownGainPlant`).
struct UnknownGainPlant {
    a: f64,
    b: f64,
    dt: f64,
}

impl UnknownGainPlant {
    fn new(a: f64, b: f64, dt: f64) -> Self {
        UnknownGainPlant { a, b, dt }
    }
}

impl PlantBlock for UnknownGainPlant {
    fn dt(&self) -> f64 {
        self.dt
    }
    fn dynamics(&mut self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
        // Forward Euler.
        vec![x[0] + dt * (self.a * x[0] + self.b * u[0])]
    }
}

// -----------------------------------------------------------------------------
// REFERENCE MODEL
// -----------------------------------------------------------------------------

/// Stable reference model `ẋ_m = a_m x_m + b_m r` (TS `ReferenceModel`).
struct ReferenceModel {
    xm: f64,
    am: f64,
    bm: f64,
    dt: f64,
    history: Vec<f64>,
}

impl ReferenceModel {
    fn new(xm0: f64, am: f64, bm: f64, dt: f64) -> Self {
        ReferenceModel {
            xm: xm0,
            am,
            bm,
            dt,
            history: vec![xm0],
        }
    }
    fn step(&mut self, r: f64) -> f64 {
        self.xm += self.dt * (self.am * self.xm + self.bm * r);
        self.history.push(self.xm);
        self.xm
    }
    fn current(&self) -> f64 {
        self.xm
    }
}

// -----------------------------------------------------------------------------
// MRAC CONTROLLER
// -----------------------------------------------------------------------------

/// MIT-rule MRAC controller. Keeps adaptive gains θ_x, θ_r and advances them
/// each tick (TS `MRACController`).
struct MRACController {
    theta_x: f64,
    theta_r: f64,
    gamma: f64,
    sign_b: f64,
    dt_cache: f64,
    ref_model: ReferenceModel,
    reference: Rc<dyn Fn(f64) -> f64>,
    u_min: Option<Vec<f64>>,
    u_max: Option<Vec<f64>>,
    tracking_error: Vec<f64>,
    theta_x_history: Vec<f64>,
    theta_r_history: Vec<f64>,
}

impl MRACController {
    fn new(
        gamma: f64,
        sign_b: f64,
        dt: f64,
        ref_model: ReferenceModel,
        reference: Rc<dyn Fn(f64) -> f64>,
        u_bound: Option<f64>,
    ) -> Self {
        let (u_min, u_max) = match u_bound {
            Some(b) => (Some(vec![-b]), Some(vec![b])),
            None => (None, None),
        };
        MRACController {
            theta_x: 0.0,
            theta_r: 0.0,
            gamma,
            sign_b,
            dt_cache: dt,
            ref_model,
            reference,
            u_min,
            u_max,
            tracking_error: Vec::new(),
            theta_x_history: Vec::new(),
            theta_r_history: Vec::new(),
        }
    }
}

impl ControllerBlock for MRACController {
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
        let x = y[0];
        let r = (*self.reference)(t);
        // 1. Step the reference model first to compute x_m(t).
        let xm = self.ref_model.step(r);
        // 2. Tracking error e = x - x_m.
        let e = x - xm;
        // 3. Gradient update on θ.
        self.theta_x += -self.gamma * e * x * self.sign_b * self.dt_cache;
        self.theta_r += -self.gamma * e * r * self.sign_b * self.dt_cache;
        // 4. Control u = θ_x x + θ_r r.
        let u = self.theta_x * x + self.theta_r * r;
        self.tracking_error.push(e);
        self.theta_x_history.push(self.theta_x);
        self.theta_r_history.push(self.theta_r);
        vec![u]
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for `run_mrac` (TS `MRACOpts`). `reference` defaults to a square wave
/// of amplitude 1 and period 4 s.
#[derive(Clone)]
#[derive(Default)]
pub struct MRACOpts {
    /// True (hidden) plant parameter a. Default 1 (unstable plant).
    pub a: Option<f64>,
    /// True (hidden) plant parameter b > 0. Default 2.
    pub b: Option<f64>,
    /// Reference model param a_m (must be negative). Default −2.
    pub am: Option<f64>,
    /// Reference model param b_m. Default 2.
    pub bm: Option<f64>,
    pub x0: Option<f64>,
    pub xm0: Option<f64>,
    /// Adaptation gain γ. Default 5.
    pub gamma: Option<f64>,
    /// Reference signal r(t). Default: square wave amplitude 1, period 4 s.
    pub reference: Option<Rc<dyn Fn(f64) -> f64>>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
    pub u_bound: Option<f64>,
}


/// Result of an MRAC run (TS `MRACResult extends ClosedLoopResult`, flattened).
#[derive(Clone, Debug)]
pub struct MRACResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
    pub tracking_error: Vec<f64>,
    pub theta_x_history: Vec<f64>,
    pub theta_r_history: Vec<f64>,
    /// Reference-model trajectory (x_m).
    pub reference_trajectory: Vec<f64>,
    /// Reference signal r(t) at each tick.
    pub r_history: Vec<f64>,
    /// RMS tracking error over the LAST half of the run (steady-state).
    pub rms_error_steady_state: f64,
    /// Final (θ_x, θ_r).
    pub final_theta: [f64; 2],
    /// Ideal θ values (closed-form): θ*_x = (a_m − a)/b, θ*_r = b_m/b.
    pub ideal_theta: [f64; 2],
}

/// Run the MRAC closed loop. Panics (TS `throw`) if any precondition is violated.
pub fn run_mrac(opts: MRACOpts) -> MRACResult {
    run_mrac_impl(opts).unwrap_or_else(|e| panic!("{e}"))
}

fn run_mrac_impl(opts: MRACOpts) -> Result<MRACResult, PreconditionError> {
    let a = opts.a.unwrap_or(1.0);
    let b = opts.b.unwrap_or(2.0);
    let am = opts.am.unwrap_or(-2.0);
    let bm = opts.bm.unwrap_or(2.0);
    let x0 = opts.x0.unwrap_or(0.0);
    let xm0 = opts.xm0.unwrap_or(0.0);
    let gamma = opts.gamma.unwrap_or(5.0);
    let dt = opts.dt.unwrap_or(0.01);
    let num_steps = opts.num_steps.unwrap_or(4000);
    let reference: Rc<dyn Fn(f64) -> f64> = opts
        .reference
        .clone()
        .unwrap_or_else(|| Rc::new(|t: f64| if (t / 2.0).floor() as i64 % 2 == 0 { 1.0 } else { -1.0 }));

    // Pre-run guards.
    let cls = "runMRAC";
    Preconditions::finite(cls, "a", a)?;
    Preconditions::positive(cls, "b", b)?;
    Preconditions::check(cls, "am", "be < 0 (reference model must be stable)", am < 0.0, Some(am.to_string()))?;
    Preconditions::finite(cls, "bm", bm)?;
    Preconditions::finite(cls, "x0", x0)?;
    Preconditions::finite(cls, "xm0", xm0)?;
    Preconditions::positive(cls, "gamma", gamma)?;
    Preconditions::positive(cls, "dt", dt)?;
    Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 1e9)?;
    // Stability margin guard: γ·dt must be moderate for the MIT-rule update.
    Preconditions::check(
        cls,
        "gamma*dt",
        "be <= 1 for numerical stability of the MIT-rule",
        gamma * dt <= 1.0 + 1e-9,
        Some((gamma * dt).to_string()),
    )?;

    let mut plant = UnknownGainPlant::new(a, b, dt);
    let ref_model = ReferenceModel::new(xm0, am, bm, dt);
    let mut ctrl = MRACController::new(gamma, math_sign(b), dt, ref_model, reference.clone(), opts.u_bound);
    let closed = run_closed_loop(&mut plant, &[x0], &mut ctrl, num_steps);

    // RMS tracking error in steady state.
    let half = ctrl.tracking_error.len() / 2;
    let tail = &ctrl.tracking_error[half..];
    let denom = tail.len().max(1) as f64;
    let rms = (tail.iter().map(|x| x * x).sum::<f64>() / denom).sqrt();

    let ideal_theta_x = (am - a) / b;
    let ideal_theta_r = bm / b;

    // Reconstruct r history.
    let mut r_hist: Vec<f64> = Vec::with_capacity(num_steps);
    for k in 0..num_steps {
        r_hist.push((*reference)(k as f64 * dt));
    }

    let final_theta = [
        *ctrl.theta_x_history.last().expect("non-empty history"),
        *ctrl.theta_r_history.last().expect("non-empty history"),
    ];

    Ok(MRACResult {
        trajectory: closed.trajectory,
        controls: closed.controls,
        measurements: closed.measurements,
        num_steps: closed.num_steps,
        tracking_error: ctrl.tracking_error.clone(),
        theta_x_history: ctrl.theta_x_history.clone(),
        theta_r_history: ctrl.theta_r_history.clone(),
        reference_trajectory: ctrl.ref_model.history.clone(),
        r_history: r_hist,
        rms_error_steady_state: rms,
        final_theta,
        ideal_theta: [ideal_theta_x, ideal_theta_r],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_reference_with_small_steady_state_error() {
        let res = run_mrac(MRACOpts::default());
        assert_eq!(res.tracking_error.len(), res.num_steps);
        // Reference-model history has xm0 plus one entry per step.
        assert_eq!(res.reference_trajectory.len(), res.num_steps + 1);
        // Adaptive law drives the steady-state tracking error small.
        assert!(
            res.rms_error_steady_state < 0.5,
            "steady-state RMS error {} too large",
            res.rms_error_steady_state
        );
    }

    #[test]
    fn ideal_theta_is_closed_form() {
        let res = run_mrac(MRACOpts {
            a: Some(1.0),
            b: Some(2.0),
            am: Some(-2.0),
            bm: Some(2.0),
            ..Default::default()
        });
        // θ*_x = (a_m − a)/b = (-2 − 1)/2 = -1.5 ; θ*_r = b_m/b = 1.
        assert!((res.ideal_theta[0] - (-1.5)).abs() < 1e-12);
        assert!((res.ideal_theta[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn rejects_unstable_reference_model() {
        // a_m must be strictly negative (Hurwitz reference model).
        run_mrac(MRACOpts {
            am: Some(1.0),
            ..Default::default()
        });
    }
}
