//! Port of `src/des/general/sliding-mode-control.ts` — robust sliding-mode
//! control (Utkin 1977; Edwards & Spurgeon 1998) of an uncertain double
//! integrator, driven as a DES closed loop.
//!
//! 1:1 behavioural move. The TS file imported `PlantBlock`, `ControllerBlock`,
//! `runClosedLoop` and `ClosedLoopResult` from `des-base/control-blocks`, which
//! is NOT in the allowed dependency list for this migration step. A MINIMAL
//! local equivalent of the lock-step closed-loop driver is defined below and
//! FLAGGED in the migration report; it reproduces the exact stepping semantics
//! of the TS driver (seed u0 = 0, plant tick then controller tick, identity
//! observation). The PRNG is threaded through `crate::des::general::prng`
//! (`mulberry32` == `SeededRandom`).
#![allow(dead_code)]

use std::f64::consts::PI;

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

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

/// Drive plant + controller in lockstep for `num_steps` ticks. Seeds u0 = 0,
/// then on each tick advances the plant (consuming the previous control) and
/// runs the controller on the fresh measurement.
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
// PLANT WITH UNKNOWN BOUNDED DISTURBANCE
// -----------------------------------------------------------------------------

/// Disturbance shape `d(t)` with `|d(t)| <= D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisturbanceType {
    Sin,
    Square,
    Random,
}

/// Uncertain double integrator `ẍ = u + d(t)` (TS `UncertainDoubleIntegratorPlant`).
struct UncertainDoubleIntegratorPlant<R: RandomSource> {
    dt: f64,
    amp: f64,
    d_type: DisturbanceType,
    rng: R,
    t: f64,
}

impl<R: RandomSource> UncertainDoubleIntegratorPlant<R> {
    fn new(dt: f64, amp: f64, d_type: DisturbanceType, rng: R) -> Self {
        UncertainDoubleIntegratorPlant {
            dt,
            amp,
            d_type,
            rng,
            t: 0.0,
        }
    }

    /// `d(t)`, advancing the internal clock by `dt` (matches the TS closure that
    /// reads `this.disturbance(this.t)` then `this.t += dt`).
    fn disturbance(&mut self) -> f64 {
        let t = self.t;
        match self.d_type {
            DisturbanceType::Sin => self.amp * (2.0 * PI * 0.5 * t).sin(),
            DisturbanceType::Square => {
                if (t * 2.0).floor() as i64 % 2 == 0 {
                    self.amp
                } else {
                    -self.amp
                }
            }
            DisturbanceType::Random => self.amp * (2.0 * self.rng.next_float() - 1.0),
        }
    }
}

impl<R: RandomSource> PlantBlock for UncertainDoubleIntegratorPlant<R> {
    fn dt(&self) -> f64 {
        self.dt
    }
    fn dynamics(&mut self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
        let d = self.disturbance();
        self.t += dt;
        let ueff = u[0] + d;
        vec![
            x[0] + dt * x[1] + 0.5 * dt * dt * ueff,
            x[1] + dt * ueff,
        ]
    }
}

// -----------------------------------------------------------------------------
// SLIDING-MODE CONTROLLER
// -----------------------------------------------------------------------------

/// Memoryless sliding-mode control law `u = -λ ẋ - η·tanh(s/boundary)`
/// (TS `SlidingModeController`).
struct SlidingModeController {
    lambda: f64,
    eta: f64,
    boundary: f64,
    u_bound: f64,
}

impl SlidingModeController {
    fn new(lambda: f64, eta: f64, boundary: f64, u_bound: f64) -> Result<Self, PreconditionError> {
        let cls = "SlidingModeController";
        Preconditions::positive(cls, "lambda", lambda)?;
        Preconditions::positive(cls, "eta", eta)?;
        Preconditions::positive(cls, "boundary", boundary)?;
        Preconditions::positive(cls, "uBound", u_bound)?;
        Ok(SlidingModeController {
            lambda,
            eta,
            boundary,
            u_bound,
        })
    }
}

impl ControllerBlock for SlidingModeController {
    fn m_dim(&self) -> usize {
        1
    }
    fn u_bounds(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (Some(vec![-self.u_bound]), Some(vec![self.u_bound]))
    }
    fn control_law(&mut self, y: &[f64], _tick: usize, _t: f64) -> Vec<f64> {
        let x = y[0];
        let v = y[1];
        let s = v + self.lambda * x;
        // Smoothed sign: tanh(s / boundary) instead of sign(s) to suppress chatter.
        let sat = (s / self.boundary).tanh();
        let u = -self.lambda * v - self.eta * sat;
        vec![u]
    }
}

// -----------------------------------------------------------------------------
// PUBLIC API
// -----------------------------------------------------------------------------

/// Options for `run_sliding_mode` (TS `SlidingModeOpts`). All fields optional;
/// `Default` yields the TS `?? default` values inside `run_sliding_mode`.
#[derive(Clone, Debug, Default)]
pub struct SlidingModeOpts {
    pub x0: Option<[f64; 2]>,
    pub dt: Option<f64>,
    pub num_steps: Option<usize>,
    /// Sliding-surface gain λ. Default 2.
    pub lambda: Option<f64>,
    /// Reaching gain η. Must exceed disturbance bound D. Default 3.
    pub eta: Option<f64>,
    /// Boundary-layer width (smoothing). Default 0.05.
    pub boundary: Option<f64>,
    /// Bound on |u|. Default 5.
    pub u_bound: Option<f64>,
    /// Disturbance amplitude D. Default 1.
    pub disturbance_amp: Option<f64>,
    /// Disturbance type. Default `Sin`.
    pub disturbance_type: Option<DisturbanceType>,
    pub seed: Option<u32>,
}

/// Result of a sliding-mode run (TS `SlidingModeResult extends ClosedLoopResult`,
/// flattened).
#[derive(Clone, Debug)]
pub struct SlidingModeResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
    /// Final |x| + |v|.
    pub final_distance_from_origin: f64,
    /// First tick at which |s(x, v)| < boundary (sliding surface reached).
    pub reaching_tick: usize,
    /// True iff state stays in a neighbourhood of size ≤ 0.5 after t = numSteps/2.
    pub stayed_near_origin: bool,
}

/// Run the robust sliding-mode closed loop. Panics (TS `throw`) if any
/// precondition is violated.
pub fn run_sliding_mode(opts: SlidingModeOpts) -> SlidingModeResult {
    run_sliding_mode_impl(opts).unwrap_or_else(|e| panic!("{e}"))
}

fn run_sliding_mode_impl(opts: SlidingModeOpts) -> Result<SlidingModeResult, PreconditionError> {
    let x0 = opts.x0.unwrap_or([3.0, 0.0]);
    let dt = opts.dt.unwrap_or(0.05);
    let num_steps = opts.num_steps.unwrap_or(400);
    let lambda = opts.lambda.unwrap_or(2.0);
    let eta = opts.eta.unwrap_or(3.0);
    let boundary = opts.boundary.unwrap_or(0.05);
    let u_bound = opts.u_bound.unwrap_or(5.0);
    let d = opts.disturbance_amp.unwrap_or(1.0);
    let d_type = opts.disturbance_type.unwrap_or(DisturbanceType::Sin);

    // Pre-run guards.
    let cls = "runSlidingMode";
    Preconditions::length_eq(cls, "x0", &x0, 2)?;
    Preconditions::all_finite(cls, "x0", &x0)?;
    Preconditions::positive(cls, "dt", dt)?;
    Preconditions::integer_in_range(cls, "numSteps", num_steps as f64, 1.0, 1e9)?;
    Preconditions::positive(cls, "lambda", lambda)?;
    Preconditions::positive(cls, "eta", eta)?;
    Preconditions::positive(cls, "boundary", boundary)?;
    Preconditions::positive(cls, "uBound", u_bound)?;
    Preconditions::non_negative(cls, "disturbanceAmp", d)?;
    // CORE SMC reaching condition: η must strictly exceed disturbance bound.
    Preconditions::check(
        cls,
        "eta > disturbanceAmp",
        "satisfy the SMC reaching condition (eta strictly > D)",
        eta > d,
        Some(format!("eta={eta}, D={d}")),
    )?;
    // NOTE: the TS `disturbanceType` string check is unnecessary here — the
    // `DisturbanceType` enum makes the value total by construction.

    let rng = mulberry32(opts.seed.unwrap_or(1));
    let mut plant = UncertainDoubleIntegratorPlant::new(dt, d, d_type, rng);
    let mut ctrl = SlidingModeController::new(lambda, eta, boundary, u_bound)?;
    let out = run_closed_loop(&mut plant, &x0, &mut ctrl, num_steps);

    let mut reaching_tick: i64 = -1;
    for (i, st) in out.trajectory.iter().enumerate() {
        let (x, v) = (st[0], st[1]);
        let s = v + lambda * x;
        if s.abs() < boundary {
            reaching_tick = i as i64;
            break;
        }
    }
    let half_point = num_steps / 2;
    let mut stayed_near_origin = true;
    for st in out.trajectory.iter().skip(half_point) {
        let (x, v) = (st[0], st[1]);
        if x.abs() + v.abs() > 0.5 {
            stayed_near_origin = false;
            break;
        }
    }
    let last = out.trajectory.last().expect("trajectory non-empty");
    let final_distance_from_origin = last[0].abs() + last[1].abs();
    Ok(SlidingModeResult {
        trajectory: out.trajectory,
        controls: out.controls,
        measurements: out.measurements,
        num_steps: out.num_steps,
        final_distance_from_origin,
        reaching_tick: if reaching_tick < 0 {
            num_steps
        } else {
            reaching_tick as usize
        },
        stayed_near_origin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drives_state_to_origin_under_sin_disturbance() {
        let res = run_sliding_mode(SlidingModeOpts::default());
        // trajectory has x0 plus one state per step.
        assert_eq!(res.trajectory.len(), res.num_steps + 1);
        assert_eq!(res.controls.len(), res.num_steps);
        // Robust convergence: end up in a small neighbourhood of the origin.
        assert!(
            res.final_distance_from_origin < 0.5,
            "final distance {} too large",
            res.final_distance_from_origin
        );
        assert!(res.stayed_near_origin);
        assert!(res.reaching_tick < res.num_steps);
    }

    #[test]
    fn converges_with_seeded_random_disturbance() {
        let opts = SlidingModeOpts {
            disturbance_type: Some(DisturbanceType::Random),
            disturbance_amp: Some(1.0),
            eta: Some(3.0),
            seed: Some(7),
            ..Default::default()
        };
        let a = run_sliding_mode(opts.clone());
        let b = run_sliding_mode(opts);
        // Deterministic given a fixed seed.
        assert_eq!(a.trajectory, b.trajectory);
        assert!(a.final_distance_from_origin < 0.5);
    }

    #[test]
    #[should_panic]
    fn rejects_reaching_condition_violation() {
        // eta must strictly exceed the disturbance bound D.
        run_sliding_mode(SlidingModeOpts {
            eta: Some(1.0),
            disturbance_amp: Some(2.0),
            ..Default::default()
        });
    }
}
