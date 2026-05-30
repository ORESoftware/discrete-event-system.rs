//! Port of `src/des/general/control-systems/numerical-solvers.ts` — class-only
//! fixed-step ODE integrators for the control-systems family.
//!
//! The general `ode` module exposes the classical solvers as closures taking an
//! `f(t, y)`. The control-systems family is deliberately object-oriented: the
//! thing being integrated is an `OdeSystem` OBJECT whose `derivative(t, state)`
//! encodes the dynamics (and may read mutable conditions such as the latest
//! control input). The integrators are likewise types whose `step` / `integrate`
//! advance such an object.
//!
//! This lets a DES plant station hold an `OdeSystem` field, mutate its inputs on
//! each tick, and call `integrator.step(system, ...)` to advance exactly one
//! numerical step per discrete tick.
//!
//! Rust shape: `interface OdeSystem` becomes the [`OdeSystem`] trait; the
//! `abstract class FixedStepIntegrator` becomes the [`FixedStepIntegrator`]
//! trait with a required `step` plus a provided `integrate`; the concrete
//! integrators are unit structs. `derivative` may read mutable conditions on the
//! system, so callers advance a `&mut` system one step per tick and pass it as
//! `&dyn OdeSystem`. Bad `dt` / `steps` are invariant violations and `panic!`.
#![allow(dead_code)]

/// A first-order vector ODE  dx/dt = f(t, x). Implemented as an OBJECT so the
/// dynamics live in a method (and can read mutable model inputs) rather than in
/// a captured closure.
pub trait OdeSystem {
    /// Dimension n of the state vector.
    fn dimension(&self) -> usize;
    /// The right-hand side f(t, x). Must return a length-n vector.
    fn derivative(&self, t: f64, state: &[f64]) -> Vec<f64>;
}

/// Time grid plus the state recorded at each grid point (TS `{times, states}`).
#[derive(Clone, Debug)]
pub struct IntegrationResult {
    pub times: Vec<f64>,
    /// `states[i]` is the state vector at `times[i]`.
    pub states: Vec<Vec<f64>>,
}

/// `out = a + s·b`, elementwise (TS `axpy`).
fn axpy(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for i in 0..a.len() {
        out[i] = a[i] + s * b[i];
    }
    out
}

/// Common contract for the fixed-step integrators (TS `abstract class
/// FixedStepIntegrator`): a required one-step advance plus a provided multi-step
/// `integrate` that records the trajectory.
pub trait FixedStepIntegrator {
    /// Advance the system by exactly one step of size `dt`, returning the new
    /// state. Pure with respect to `state` (does not mutate the input slice).
    fn step(&self, system: &dyn OdeSystem, t: f64, state: &[f64], dt: f64) -> Vec<f64>;

    /// Integrate from `t0` for `steps` steps of size `dt`. Returns the time grid
    /// and the state at each grid point (including the initial point).
    fn integrate(
        &self,
        system: &dyn OdeSystem,
        t0: f64,
        state0: &[f64],
        dt: f64,
        steps: usize,
    ) -> IntegrationResult {
        if !(dt > 0.0) {
            panic!("FixedStepIntegrator.integrate: dt must be > 0");
        }
        let mut times: Vec<f64> = vec![t0];
        let mut states: Vec<Vec<f64>> = vec![state0.to_vec()];
        let mut t = t0;
        let mut x = state0.to_vec();
        for _ in 0..steps {
            x = self.step(system, t, &x, dt);
            t += dt;
            times.push(t);
            states.push(x.clone());
        }
        IntegrationResult { times, states }
    }
}

/// Forward (explicit) Euler.  x_{n+1} = x_n + dt·f(t_n, x_n).
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEulerIntegrator;

impl ForwardEulerIntegrator {
    pub fn new() -> Self {
        ForwardEulerIntegrator
    }
}

impl FixedStepIntegrator for ForwardEulerIntegrator {
    fn step(&self, system: &dyn OdeSystem, t: f64, state: &[f64], dt: f64) -> Vec<f64> {
        let k1 = system.derivative(t, state);
        axpy(state, &k1, dt)
    }
}

/// Classical fourth-order Runge–Kutta. The workhorse for the control-systems
/// plants (smooth, non-stiff dynamics).
#[derive(Clone, Copy, Debug, Default)]
pub struct RungeKutta4Integrator;

impl RungeKutta4Integrator {
    pub fn new() -> Self {
        RungeKutta4Integrator
    }
}

impl FixedStepIntegrator for RungeKutta4Integrator {
    fn step(&self, system: &dyn OdeSystem, t: f64, state: &[f64], dt: f64) -> Vec<f64> {
        let half = dt / 2.0;
        let k1 = system.derivative(t, state);
        let k2 = system.derivative(t + half, &axpy(state, &k1, half));
        let k3 = system.derivative(t + half, &axpy(state, &k2, half));
        let k4 = system.derivative(t + dt, &axpy(state, &k3, dt));
        let mut out = vec![0.0; state.len()];
        for i in 0..state.len() {
            out[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dy/dt = y, y(0) = 1 ⇒ y(t) = eᵗ.
    struct Exponential;
    impl OdeSystem for Exponential {
        fn dimension(&self) -> usize {
            1
        }
        fn derivative(&self, _t: f64, state: &[f64]) -> Vec<f64> {
            vec![state[0]]
        }
    }

    #[test]
    fn rk4_matches_exponential_growth() {
        let rk4 = RungeKutta4Integrator::new();
        let result = rk4.integrate(&Exponential, 0.0, &[1.0], 0.01, 100);
        let last = *result.states.last().unwrap().first().unwrap();
        assert!((last - std::f64::consts::E).abs() < 1e-6, "RK4 e drift {last}");
        // Includes the initial point: 100 steps -> 101 samples reaching t = 1.
        assert_eq!(result.states.len(), 101);
        assert!((result.times.last().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn euler_single_step() {
        let euler = ForwardEulerIntegrator::new();
        // dy/dt = y, one step of dt = 1 from y0 = 1 -> 2.
        let next = euler.step(&Exponential, 0.0, &[1.0], 1.0);
        assert!((next[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn integrate_rejects_nonpositive_dt() {
        let rk4 = RungeKutta4Integrator::new();
        rk4.integrate(&Exponential, 0.0, &[1.0], 0.0, 10);
    }
}
