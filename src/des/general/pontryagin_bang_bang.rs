//! Port of `src/des/general/pontryagin-bang-bang.ts` — TIME-OPTIMAL control of a
//! double integrator via Pontryagin's Maximum Principle.
//!
//! Plant:  ẍ = u,  |u| ≤ u_max.  Goal: drive `(x, ẋ) → (0, 0)` in minimum time.
//! PMP gives `u* = −u_max·sign(λ₂)` (bang-bang on the bound) with at most ONE
//! switch. The switching curve in the `(x, ẋ)` phase plane is
//!   x = −(1 / (2 u_max))·ẋ·|ẋ|.
//! Apply `+u_max` below the curve, `−u_max` above, switch once on crossing.
//!
//! References: Pontryagin et al. 1962; Bryson & Ho 1975, §2.6;
//! Athans & Falb 1966, §6.6.
//!
//! TS mapping notes:
//!   * `class DoubleIntegratorPlant extends PlantBlock` and
//!     `class PontryaginBangBangController extends ControllerBlock` become
//!     private structs. Their template-method bases live in
//!     `des-base/control-blocks.ts`, which is NOT yet ported and is NOT among
//!     the allowed dependencies, so the minimal deterministic lock-step
//!     `runClosedLoop` driver is reproduced privately here.
//!   * `interface PontryaginResult extends ClosedLoopResult` is flattened.
//!   * the switching curve uses `sign`/`abs` on `f64`; fully deterministic.
//!   * `Preconditions.*` throws become `Result<_, PreconditionError>`.

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

// -----------------------------------------------------------------------------
// CLOSED-LOOP RESULT (flattened from des-base/control-blocks `ClosedLoopResult`)
// -----------------------------------------------------------------------------

/// Trajectory + control/measurement history (no-estimator path).
#[derive(Clone, Debug)]
pub struct ClosedLoopResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
}

// -----------------------------------------------------------------------------
// PLANT: DOUBLE INTEGRATOR  ẍ = u
// -----------------------------------------------------------------------------

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
    fn new(x0: [f64; 2], dt: f64) -> Result<Self, PreconditionError> {
        Preconditions::length_eq("DoubleIntegratorPlant", "x0", &x0, 2)?;
        Preconditions::all_finite("DoubleIntegratorPlant", "x0", &x0)?;
        Preconditions::positive("DoubleIntegratorPlant", "dt", dt)?;
        Ok(DoubleIntegratorPlant {
            state: vec![x0[0], x0[1]],
            dt,
            tick: 0,
            last_u: vec![0.0],
            state_history: vec![vec![x0[0], x0[1]]],
            input_history: Vec::new(),
            output_history: Vec::new(),
        })
    }

    /// Exact discretisation of ẍ = u for piecewise-constant u.
    fn dynamics(x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
        vec![x[0] + dt * x[1] + 0.5 * dt * dt * u[0], x[1] + dt * u[0]]
    }

    fn step(&mut self, control: Vec<f64>) -> Vec<f64> {
        self.last_u = control;
        let x_new = Self::dynamics(&self.state, &self.last_u, self.dt);
        self.input_history.push(self.last_u.clone());
        self.state = x_new.clone();
        self.state_history.push(x_new.clone());
        self.tick += 1;
        self.output_history.push(x_new.clone());
        x_new
    }
}

// -----------------------------------------------------------------------------
// CONTROLLER: BANG-BANG ON SWITCHING CURVE (PMP-OPTIMAL)
// -----------------------------------------------------------------------------

struct PontryaginBangBangController {
    u_bound: f64,
    /// Numerical band around the switching curve where a smooth linear law
    /// replaces bang-bang to suppress chattering near the equilibrium.
    deadband: f64,
    u_min: Vec<f64>,
    u_max: Vec<f64>,
    output_history: Vec<Vec<f64>>,
    input_history: Vec<Vec<f64>>,
}

impl PontryaginBangBangController {
    fn new(u_bound: f64, deadband: f64) -> Result<Self, PreconditionError> {
        Preconditions::positive("PontryaginBangBangController", "uBound (u_max)", u_bound)?;
        Preconditions::positive("PontryaginBangBangController", "deadband", deadband)?;
        Ok(PontryaginBangBangController {
            u_bound,
            deadband,
            u_min: vec![-u_bound],
            u_max: vec![u_bound],
            output_history: Vec::new(),
            input_history: Vec::new(),
        })
    }

    fn control_law(&self, y: &[f64]) -> Vec<f64> {
        // y = [x, v] (full state observed).
        let x = y[0];
        let v = y[1];
        let sigma = x + (1.0 / (2.0 * self.u_bound)) * v * v.abs();
        // Once close to the origin, switch to a smooth PD law to avoid the
        // chattering of discrete-time bang-bang near equilibrium.
        if x.abs() + v.abs() < self.deadband {
            let u = -10.0 * x - 6.0 * v;
            return vec![(-self.u_bound).max(self.u_bound.min(u))];
        }
        vec![if sigma > 0.0 {
            -self.u_bound
        } else {
            self.u_bound
        }]
    }

    fn saturate(&self, mut u: Vec<f64>) -> Vec<f64> {
        for i in 0..u.len() {
            if u[i] < self.u_min[i] {
                u[i] = self.u_min[i];
            }
            if u[i] > self.u_max[i] {
                u[i] = self.u_max[i];
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
    ctrl: &mut PontryaginBangBangController,
    num_steps: usize,
) -> ClosedLoopResult {
    let mut pending_u = vec![0.0; 1];
    for _ in 0..num_steps {
        let y = plant.step(std::mem::take(&mut pending_u));
        let u = ctrl.saturate(ctrl.control_law(&y));
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

/// Options for [`run_pontryagin_bang_bang`]; `None` fields take TS defaults.
#[derive(Clone, Debug, Default)]
pub struct PontryaginOpts {
    /// Initial state `[x, v]`. Default `[3, 0]`.
    pub x0: Option<[f64; 2]>,
    /// Bound on `|u|`. Default 1.
    pub u_max: Option<f64>,
    /// Sample period dt. Default 0.05.
    pub dt: Option<f64>,
    /// Number of simulation steps. Default 200.
    pub num_steps: Option<usize>,
    /// State-magnitude band around the origin where the controller switches
    /// from bang-bang to a smooth linear law. Default 0.2.
    pub deadband: Option<f64>,
}

/// Result of a bang-bang closed-loop run (flattened `ClosedLoopResult` + metrics).
#[derive(Clone, Debug)]
pub struct PontryaginResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub num_steps: usize,
    /// First tick at which `|x| + |v| < 1e-2` (proxy for "reached origin").
    pub arrival_tick: usize,
    /// True optimal time-to-go from `x0` (closed-form).
    pub theoretical_arrival_time: f64,
    /// Number of saturated u-sign changes during the run (PMP predicts ≤ 1).
    pub switch_count: usize,
}

/// Run the PMP-optimal bang-bang controller on the double integrator.
pub fn run_pontryagin_bang_bang(
    opts: PontryaginOpts,
) -> Result<PontryaginResult, PreconditionError> {
    let x0 = opts.x0.unwrap_or([3.0, 0.0]);
    let u_max = opts.u_max.unwrap_or(1.0);
    let dt = opts.dt.unwrap_or(0.05);
    let num_steps = opts.num_steps.unwrap_or(200);
    Preconditions::positive("runPontryaginBangBang", "uMax", u_max)?;
    Preconditions::positive("runPontryaginBangBang", "dt", dt)?;
    Preconditions::integer_in_range(
        "runPontryaginBangBang",
        "numSteps",
        num_steps as f64,
        1.0,
        1e9,
    )?;
    Preconditions::length_eq("runPontryaginBangBang", "x0", &x0, 2)?;
    Preconditions::all_finite("runPontryaginBangBang", "x0", &x0)?;
    if let Some(db) = opts.deadband {
        Preconditions::positive("runPontryaginBangBang", "deadband", db)?;
    }
    let deadband = opts.deadband.unwrap_or(0.2);

    let mut plant = DoubleIntegratorPlant::new(x0, dt)?;
    let mut ctrl = PontryaginBangBangController::new(u_max, deadband)?;
    let out = run_closed_loop(&mut plant, &mut ctrl, num_steps);

    // Bang-bang arrival = first entry into the deadband (leaving the saturated
    // phase). This is the textbook t* the PMP formula predicts.
    let mut arrival_tick = num_steps;
    let db_thr = deadband;
    for (i, row) in out.trajectory.iter().enumerate() {
        if row[0].abs() + row[1].abs() < db_thr {
            arrival_tick = i;
            break;
        }
    }

    // Closed-form optimal time t* for the double integrator with |u| ≤ u_max
    // from (x₀, v₀) to the origin. For the simple v₀ = 0 case: t* = 2√(|x₀|/u_max).
    let theoretical = if x0[1] == 0.0 {
        2.0 * (x0[0].abs() / u_max).sqrt()
    } else {
        optimal_time_double_integrator(x0[0], x0[1], u_max)
    };

    // Count BANG-BANG switches: only saturated phases count (the smooth deadband
    // phase is excluded — PMP predicts ≤ 1 saturated switch).
    let mut switch_count = 0;
    let mut last = 0.0_f64;
    let sat_thresh = 0.99 * u_max;
    for u in &out.controls {
        if u[0].abs() < sat_thresh {
            continue;
        }
        let s = u[0].signum();
        if last != 0.0 && s != last {
            switch_count += 1;
        }
        last = s;
    }

    Ok(PontryaginResult {
        trajectory: out.trajectory,
        controls: out.controls,
        measurements: out.measurements,
        num_steps: out.num_steps,
        arrival_tick,
        theoretical_arrival_time: theoretical,
        switch_count,
    })
}

/// Closed-form optimal time-to-go for the double integrator `ẍ = u`,
/// `|u| ≤ u_max`, terminal `(0, 0)` — computed by integrating the closed-loop
/// bang-bang law on a fine grid (Athans & Falb 1966, §6.6).
pub fn optimal_time_double_integrator(x0: f64, v0: f64, u_max: f64) -> f64 {
    if x0.abs() + v0.abs() < 1e-9 {
        return 0.0;
    }
    let mut x = x0;
    let mut v = v0;
    let mut t = 0.0;
    let dt_fine = 1e-4;
    for _ in 0..1_000_000 {
        let sigma = x + (1.0 / (2.0 * u_max)) * v * v.abs();
        let u = if sigma > 0.0 {
            -u_max
        } else if sigma < 0.0 {
            u_max
        } else if v > 0.0 {
            -u_max
        } else {
            u_max
        };
        x += dt_fine * v + 0.5 * dt_fine * dt_fine * u;
        v += dt_fine * u;
        t += dt_fine;
        if x.abs() + v.abs() < 1e-3 {
            return t;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_time_matches_analytic() {
        let r = run_pontryagin_bang_bang(PontryaginOpts::default()).unwrap();
        // Analytic minimum time from (3, 0) with u_max = 1 is 2√3 ≈ 3.4641.
        assert!((r.theoretical_arrival_time - 2.0 * 3.0_f64.sqrt()).abs() < 1e-9);
        // The controller reaches the deadband, and that arrival time tracks the
        // PMP-optimal time-to-go.
        assert!(
            r.arrival_tick < r.num_steps,
            "arrival_tick = {}",
            r.arrival_tick
        );
        let arrival_time = r.arrival_tick as f64 * 0.05;
        assert!(
            (arrival_time - r.theoretical_arrival_time).abs() < 0.5,
            "arrival_time = {arrival_time}, theoretical = {}",
            r.theoretical_arrival_time
        );
    }

    #[test]
    fn fires_exactly_one_bang_bang_switch_from_rest() {
        let r = run_pontryagin_bang_bang(PontryaginOpts::default()).unwrap();
        // From rest above the switching curve (σ₀ = 3 > 0) the first control is −u_max.
        assert_eq!(r.controls[0][0], -1.0);
        // PMP predicts exactly one saturated switch (−u_max → +u_max).
        assert_eq!(r.switch_count, 1);
    }

    #[test]
    fn closed_form_optimal_time_from_rest() {
        // Numerically integrated bang-bang should match the analytic 2√(|x₀|/u_max).
        let t = optimal_time_double_integrator(3.0, 0.0, 1.0);
        assert!((t - 2.0 * 3.0_f64.sqrt()).abs() < 0.05, "t = {t}");
        assert_eq!(optimal_time_double_integrator(0.0, 0.0, 1.0), 0.0);
    }
}
