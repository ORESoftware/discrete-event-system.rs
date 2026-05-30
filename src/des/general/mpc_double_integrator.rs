//! Port of `src/des/general/mpc-double-integrator.ts` — constrained Model
//! Predictive Control (Mayne et al. 2000) for the double integrator, solving a
//! tiny box-constrained QP every tick by projected gradient and applying the
//! FIRST control of the receding horizon.
//!
//! Plant:  ẍ = u,  |u| ≤ u_max.  At each tick the controller solves
//!   min_{u_0..u_{N-1}}  ∑ xᵢᵀQxᵢ + uᵢ²R  +  x_NᵀQ_f x_N
//!   s.t.  x_{i+1} = A x_i + B u_i,   |u_i| ≤ u_max
//! over the horizon N, applies u_0, then rolls forward and re-solves.
//!
//! TS mapping notes:
//!   * `class DoubleIntegratorPlant extends PlantBlock` and
//!     `class MPCDoubleIntegratorController extends ControllerBlock` become
//!     private structs. The `PlantBlock`/`ControllerBlock` template-method bases
//!     live in `des-base/control-blocks.ts`, which is NOT yet ported and is NOT
//!     among the allowed dependencies, so the minimal deterministic lock-step
//!     `runClosedLoop` driver they rely on is reproduced privately here.
//!   * `interface MPCDoubleIntResult extends ClosedLoopResult` is flattened into
//!     one struct (Rust has no interface inheritance).
//!   * all numerics are `f64`; the projected-gradient inner loop is deterministic
//!     (no RNG / clock).
//!   * `Preconditions.*` throws become `Result<_, PreconditionError>` propagation.

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

// -----------------------------------------------------------------------------
// CLOSED-LOOP RESULT (flattened from des-base/control-blocks `ClosedLoopResult`)
// -----------------------------------------------------------------------------

/// Trajectory + control/measurement history produced by the lock-step driver.
///
/// Mirrors `ClosedLoopResult` from `des-base/control-blocks.ts` for the
/// no-estimator path (the optional `estimates` field is omitted because the
/// estimator block is not exercised by this model).
#[derive(Clone, Debug)]
pub struct ClosedLoopResult {
    /// State history `[x0, x1, …, x_numSteps]` (length `num_steps + 1`).
    pub trajectory: Vec<Vec<f64>>,
    /// Applied controls, one per tick (length `num_steps`).
    pub controls: Vec<Vec<f64>>,
    /// Plant measurements `y = h(x)`, one per tick (length `num_steps`).
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
}

// -----------------------------------------------------------------------------
// PLANT
// -----------------------------------------------------------------------------

/// Discrete double integrator with the exact piecewise-constant-`u`
/// discretisation `x' = x + dt v + ½ dt² u`, `v' = v + dt u`.
struct DoubleIntegratorPlant {
    state: Vec<f64>,
    dt: f64,
    tick: usize,
    last_u: Vec<f64>,
    state_history: Vec<Vec<f64>>,
    input_history: Vec<Vec<f64>>,
    output_history: Vec<Vec<f64>>,
}

impl DoubleIntegratorPlant {
    fn new(x0: [f64; 2], dt: f64) -> Self {
        DoubleIntegratorPlant {
            state: vec![x0[0], x0[1]],
            dt,
            tick: 0,
            last_u: vec![0.0],
            state_history: vec![vec![x0[0], x0[1]]],
            input_history: Vec::new(),
            output_history: Vec::new(),
        }
    }

    fn dynamics(x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
        vec![x[0] + dt * x[1] + 0.5 * dt * dt * u[0], x[1] + dt * u[0]]
    }

    /// Drain the latest control, advance one tick, record + return measurement.
    fn step(&mut self, control: Vec<f64>) -> Vec<f64> {
        self.last_u = control;
        let x_new = Self::dynamics(&self.state, &self.last_u, self.dt);
        self.input_history.push(self.last_u.clone());
        self.state = x_new.clone();
        self.state_history.push(x_new.clone());
        self.tick += 1;
        // observe(): identity measurement y = x.
        self.output_history.push(x_new.clone());
        x_new
    }
}

// -----------------------------------------------------------------------------
// MPC CONTROLLER
// -----------------------------------------------------------------------------

struct MpcDoubleIntegratorController {
    n: usize,
    q: [f64; 2],
    qf: [f64; 2],
    r: f64,
    u_max: f64,
    // Sampling period; retained for parity with the TS controller config (the
    // discretized A/B already bake in dt, so the loop does not read it again).
    #[allow(dead_code)]
    dt: f64,
    a: [[f64; 2]; 2],
    b: [f64; 2],
    warm_start: Vec<f64>,
    output_history: Vec<Vec<f64>>,
    input_history: Vec<Vec<f64>>,
}

impl MpcDoubleIntegratorController {
    fn new(
        n: usize,
        q: [f64; 2],
        qf: [f64; 2],
        r: f64,
        u_max: f64,
        dt: f64,
    ) -> Result<Self, PreconditionError> {
        let cls = "MPCDoubleIntegratorController";
        Preconditions::integer_in_range(cls, "N (horizon)", n as f64, 1.0, 1000.0)?;
        Preconditions::arr_non_negative(cls, "Q", &q)?;
        Preconditions::arr_non_negative(cls, "Qf", &qf)?;
        // R MUST be > 0 — appears as 2 R u in the gradient; R = 0 makes the
        // problem ill-posed (control unbounded along the nullspace).
        Preconditions::positive(cls, "R", r)?;
        Preconditions::positive(cls, "uMax", u_max)?;
        Preconditions::positive(cls, "dt", dt)?;
        Ok(MpcDoubleIntegratorController {
            n,
            q,
            qf,
            r,
            u_max,
            dt,
            a: [[1.0, dt], [0.0, 1.0]],
            b: [0.5 * dt * dt, dt],
            warm_start: vec![0.0; n],
            output_history: Vec::new(),
            input_history: Vec::new(),
        })
    }

    /// Project `v` onto `[-u_max, u_max]`.
    fn clip(&self, v: f64) -> f64 {
        if v < -self.u_max {
            -self.u_max
        } else if v > self.u_max {
            self.u_max
        } else {
            v
        }
    }

    /// Compute `J(useq)` and `∇_u J` via a forward rollout + backward sweep.
    fn cost_and_grad(&self, x0: [f64; 2], useq: &[f64]) -> (f64, Vec<f64>) {
        let n = self.n;
        let mut xs: Vec<[f64; 2]> = vec![[0.0, 0.0]; n + 1];
        xs[0] = [x0[0], x0[1]];
        let mut j = 0.0;
        for i in 0..n {
            let x = xs[i];
            let u = useq[i];
            // stage cost xᵀQx + R u²
            j += self.q[0] * x[0] * x[0] + self.q[1] * x[1] * x[1] + self.r * u * u;
            xs[i + 1] = [
                self.a[0][0] * x[0] + self.a[0][1] * x[1] + self.b[0] * u,
                self.a[1][0] * x[0] + self.a[1][1] * x[1] + self.b[1] * u,
            ];
        }
        // terminal cost
        j += self.qf[0] * xs[n][0] * xs[n][0] + self.qf[1] * xs[n][1] * xs[n][1];

        // backward sweep: λ_N = 2 [Qf_0 x_N0; Qf_1 x_N1]
        let mut lambda = [2.0 * self.qf[0] * xs[n][0], 2.0 * self.qf[1] * xs[n][1]];
        let mut grad = vec![0.0; n];
        for i in (0..n).rev() {
            // grad_i = 2 R u_i + Bᵀ λ_{i+1}
            grad[i] = 2.0 * self.r * useq[i] + self.b[0] * lambda[0] + self.b[1] * lambda[1];
            // λ_i = 2 Q x_i + Aᵀ λ_{i+1}
            let lambda_prev = [
                2.0 * self.q[0] * xs[i][0] + self.a[0][0] * lambda[0] + self.a[1][0] * lambda[1],
                2.0 * self.q[1] * xs[i][1] + self.a[0][1] * lambda[0] + self.a[1][1] * lambda[1],
            ];
            lambda = lambda_prev;
        }
        (j, grad)
    }

    /// Projected gradient descent on the box-constrained QP.
    fn solve_qp(&self, x0: [f64; 2]) -> Vec<f64> {
        let mut useq = self.warm_start.clone();
        let mut alpha: f64 = 0.05; // step size
        let mut last_j = f64::INFINITY;
        for _ in 0..200 {
            let (j, grad) = self.cost_and_grad(x0, &useq);
            // Simple step-size adaptation.
            if j > last_j - 1e-10 {
                alpha *= 0.5; // shrink if no progress
            } else {
                alpha = (alpha * 1.05).min(0.1);
            }
            last_j = j;
            for i in 0..self.n {
                useq[i] = self.clip(useq[i] - alpha * grad[i]);
            }
            // Termination on gradient norm (ignoring components pinned at a bound).
            let mut gnorm = 0.0;
            for i in 0..self.n {
                let ui = useq[i];
                let gi = grad[i];
                let at_upper = ui >= self.u_max - 1e-9 && gi < 0.0;
                let at_lower = ui <= -self.u_max + 1e-9 && gi > 0.0;
                if !at_upper && !at_lower {
                    gnorm += gi * gi;
                }
            }
            if gnorm.sqrt() < 1e-3 {
                break;
            }
        }
        useq
    }

    /// Control law: solve the QP, warm-start-shift, apply `u_0`.
    fn control_law(&mut self, y: &[f64]) -> Vec<f64> {
        let x0 = [y[0], y[1]];
        let useq = self.solve_qp(x0);
        for i in 0..self.n - 1 {
            self.warm_start[i] = useq[i + 1];
        }
        self.warm_start[self.n - 1] = 0.0;
        vec![useq[0]]
    }

    /// Saturate the emitted control to `[-u_max, u_max]` (the block's
    /// `setSaturation` bounds, redundant here but faithful to the base class).
    fn saturate(&self, mut u: Vec<f64>) -> Vec<f64> {
        for ui in u.iter_mut() {
            if *ui < -self.u_max {
                *ui = -self.u_max;
            }
            if *ui > self.u_max {
                *ui = self.u_max;
            }
        }
        u
    }
}

// -----------------------------------------------------------------------------
// CLOSED-LOOP DRIVER (faithful port of runClosedLoop, plant+controller only)
// -----------------------------------------------------------------------------

fn run_closed_loop(
    plant: &mut DoubleIntegratorPlant,
    ctrl: &mut MpcDoubleIntegratorController,
    num_steps: usize,
) -> ClosedLoopResult {
    // Seed control u0 = 0, fed to the plant before the controller first fires.
    let mut pending_u = vec![0.0; 1];
    for _ in 0..num_steps {
        // PLANT: drain latest control, advance, emit measurement y.
        let y = plant.step(std::mem::take(&mut pending_u));
        // CONTROLLER: receive y, compute u, saturate, emit to plant.
        let raw_u = ctrl.control_law(&y);
        let u = ctrl.saturate(raw_u);
        ctrl.input_history.push(y.clone());
        ctrl.output_history.push(u.clone());
        pending_u = u;
    }
    ClosedLoopResult {
        trajectory: plant.state_history.clone(),
        controls: ctrl.output_history.clone(),
        measurements: plant.output_history.clone(),
        num_steps,
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for [`run_mpc_double_integrator`]; `None` fields take TS defaults.
#[derive(Clone, Debug, Default)]
pub struct MpcDoubleIntOpts {
    pub x0: Option<[f64; 2]>,
    pub u_max: Option<f64>,
    /// Horizon length. Default 15.
    pub n: Option<usize>,
    /// Stage state weights diag(Q). Default `[10, 1]`.
    pub q: Option<[f64; 2]>,
    /// Terminal weights diag(Qf). Default `[50, 5]`.
    pub qf: Option<[f64; 2]>,
    /// Input weight R. Default 0.1.
    pub r: Option<f64>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
}

/// Result of an MPC closed-loop run (flattened `ClosedLoopResult` plus metrics).
#[derive(Clone, Debug)]
pub struct MpcDoubleIntResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
    /// First tick at which `|x| + |v| < 1e-2`.
    pub arrival_tick: usize,
    /// Maximum `|u|` over the run — should be `≤ u_max`.
    pub max_abs_u: f64,
}

/// Run constrained MPC on the double integrator in closed loop.
pub fn run_mpc_double_integrator(
    opts: MpcDoubleIntOpts,
) -> Result<MpcDoubleIntResult, PreconditionError> {
    let x0 = opts.x0.unwrap_or([3.0, 0.0]);
    let dt = opts.dt.unwrap_or(0.1);
    let num_steps = opts.num_steps.unwrap_or(100);
    let cls = "runMPCDoubleIntegrator";
    Preconditions::length_eq(cls, "x0", &x0, 2)?;
    Preconditions::all_finite(cls, "x0", &x0)?;
    Preconditions::positive(cls, "dt", dt)?;
    Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 1e9)?;

    let mut plant = DoubleIntegratorPlant::new(x0, dt);
    let mut ctrl = MpcDoubleIntegratorController::new(
        opts.n.unwrap_or(15),
        opts.q.unwrap_or([10.0, 1.0]),
        opts.qf.unwrap_or([50.0, 5.0]),
        opts.r.unwrap_or(0.1),
        opts.u_max.unwrap_or(1.0),
        dt,
    )?;
    let out = run_closed_loop(&mut plant, &mut ctrl, num_steps);

    let mut arrival_tick = num_steps;
    for (i, row) in out.trajectory.iter().enumerate() {
        if row[0].abs() + row[1].abs() < 1e-2 {
            arrival_tick = i;
            break;
        }
    }
    let mut max_abs_u = 0.0_f64;
    for u in &out.controls {
        if u[0].abs() > max_abs_u {
            max_abs_u = u[0].abs();
        }
    }

    Ok(MpcDoubleIntResult {
        trajectory: out.trajectory,
        controls: out.controls,
        measurements: out.measurements,
        num_steps: out.num_steps,
        arrival_tick,
        max_abs_u,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drives_double_integrator_to_origin() {
        let r = run_mpc_double_integrator(MpcDoubleIntOpts::default()).unwrap();
        // History shapes: trajectory has one extra (the initial state).
        assert_eq!(r.trajectory.len(), r.num_steps + 1);
        assert_eq!(r.controls.len(), r.num_steps);
        // The closed loop reaches the |x|+|v| < 1e-2 band before the run ends…
        assert!(
            r.arrival_tick < r.num_steps,
            "arrival_tick = {}",
            r.arrival_tick
        );
        // …and finishes essentially at the origin.
        let last = r.trajectory.last().unwrap();
        assert!(last[0].abs() + last[1].abs() < 1e-1, "final = {last:?}");
    }

    #[test]
    fn respects_input_saturation() {
        let r = run_mpc_double_integrator(MpcDoubleIntOpts {
            u_max: Some(0.5),
            num_steps: Some(200),
            ..Default::default()
        })
        .unwrap();
        assert!(r.max_abs_u <= 0.5 + 1e-9, "max_abs_u = {}", r.max_abs_u);
    }

    #[test]
    fn rejects_bad_parameters() {
        // R must be strictly positive.
        assert!(run_mpc_double_integrator(MpcDoubleIntOpts {
            r: Some(0.0),
            ..Default::default()
        })
        .is_err());
        // dt must be positive.
        assert!(run_mpc_double_integrator(MpcDoubleIntOpts {
            dt: Some(-0.1),
            ..Default::default()
        })
        .is_err());
    }
}
