//! Port of `src/des/general/des-base/controller.ts` — template-method base for
//! FEEDBACK CONTROL stations (bang-bang / PID / fuzzy / MPC / receding-horizon
//! DP / sliding-mode / LQR-LQG) over a generic observation `O` and control `U`.
//!
//! ## Rust shape
//!
//! TypeScript's `abstract class ControllerStation<O, U> extends DESStation`
//! becomes the [`ControllerStation`] trait extending [`DESStation`]:
//!
//!   * Controller-private state (`ticksProcessed`, `controlHistory`,
//!     `observationHistory`) lives in [`ControllerCore`], embedded by every
//!     concrete controller and exposed via `controller_core{,_mut}`.
//!   * The single abstract hook `controlLaw` → the required
//!     [`ControllerStation::control_law`].
//!   * Optional overrides (`uMin`/`uMax`/`onTick`/`reset`) → provided defaults.
//!   * The *final* template methods (`runTimeStep`, `step`, `clamp`) →
//!     provided defaults; a concrete station's [`DESStation::run_time_step`]
//!     delegates to [`ControllerStation::controller_run_time_step`].
//!   * `uMin()/uMax(): U | null` → `Option<U>`.
//!   * TS `clamp` uses a runtime `typeof u !== 'number'` test that does not
//!     translate. Per the migration header, saturation is modelled as the
//!     [`Saturate`] bound on `U` (implemented for `f64`); non-scalar `U` simply
//!     provides a no-op `Saturate` impl to skip clamping.
//!   * `ControlToken.observation: unknown` → the concrete generic `O`.

use std::rc::Rc;

use super::station::DESStation;

/// Observation inbox channel.
pub const CH_OBSERVATION: &str = "observation";
/// Control outbox channel.
pub const CH_CONTROL: &str = "control";

/// A single sensor reading delivered to the controller.
pub struct ObservationToken<O = f64> {
    pub observation: O,
    pub tick: f64,
    pub time: f64,
}

impl<O> ObservationToken<O> {
    pub fn new(observation: O, tick: f64, time: f64) -> Self {
        ObservationToken {
            observation,
            tick,
            time,
        }
    }
}

/// A single control action sent from the controller.
pub struct ControlToken<O = f64, U = f64> {
    pub control: U,
    pub observation: O,
    pub tick: f64,
    pub time: f64,
}

/// Saturation behaviour for a control type. The default scalar case is `f64`;
/// non-scalar controls supply a no-op impl so the template can still clamp.
pub trait Saturate: Sized {
    fn saturate(self, lo: Option<Self>, hi: Option<Self>) -> Self;
}

impl Saturate for f64 {
    fn saturate(self, lo: Option<f64>, hi: Option<f64>) -> f64 {
        let mut v = self;
        if let Some(lo) = lo {
            if v < lo {
                v = lo;
            }
        }
        if let Some(hi) = hi {
            if v > hi {
                v = hi;
            }
        }
        v
    }
}

/// Controller-private state (the non-shared fields of the TS abstract class).
pub struct ControllerCore<O, U> {
    /// Incremented every time the control law fires.
    pub ticks_processed: usize,
    /// Per-tick history (clamped controls).
    pub control_history: Vec<U>,
    pub observation_history: Vec<O>,
}

impl<O, U> Default for ControllerCore<O, U> {
    fn default() -> Self {
        ControllerCore {
            ticks_processed: 0,
            control_history: Vec::new(),
            observation_history: Vec::new(),
        }
    }
}

impl<O, U> ControllerCore<O, U> {
    pub fn new() -> Self {
        ControllerCore::default()
    }
}

/// Template-method base for feedback controllers.
pub trait ControllerStation<O, U>: DESStation {
    /// Borrow controller-private state.
    fn controller_core(&self) -> &ControllerCore<O, U>;
    /// Mutably borrow controller-private state.
    fn controller_core_mut(&mut self) -> &mut ControllerCore<O, U>;

    // ── HOOK (abstract) ──────────────────────────────────────────────────────

    /// The CONTROL LAW — receive an observation, produce a control output.
    /// Implementors may read/mutate their own persistent state (PID integrator,
    /// MPC plan cache, …) — hence `&mut self`.
    fn control_law(&mut self, observation: &O, tick: f64, time: f64) -> U;

    // ── HOOKS (optional override) ─────────────────────────────────────────────

    /// Lower saturation bound. Default: none (−∞).
    fn u_min(&self) -> Option<U> {
        None
    }
    /// Upper saturation bound. Default: none (+∞).
    fn u_max(&self) -> Option<U> {
        None
    }
    /// Per-tick instrumentation hook.
    fn on_tick(&mut self, _observation: &O, _u: &U, _u_clamped: &U) {}
    /// Reset internal state (e.g. start of a new run).
    fn reset(&mut self) {
        let c = self.controller_core_mut();
        c.ticks_processed = 0;
        c.control_history.clear();
        c.observation_history.clear();
    }

    // ── INTERNAL (final) ───────────────────────────────────────────────────────

    /// Saturation. Works for any `U: Saturate` (scalar `f64` clamps; non-scalar
    /// impls may no-op).
    fn clamp(&self, u: U) -> U
    where
        U: Saturate,
    {
        u.saturate(self.u_min(), self.u_max())
    }

    // ── TEMPLATE METHOD (final) ────────────────────────────────────────────────

    /// Drain observations, apply the control law + saturation, record history,
    /// and emit a [`ControlToken`] per observation. Concrete stations wire this
    /// up from [`DESStation::run_time_step`].
    fn controller_run_time_step(&mut self)
    where
        O: Clone + 'static,
        U: Saturate + Clone + 'static,
    {
        let observations = self.core_mut().drain::<ObservationToken<O>>(CH_OBSERVATION);
        for obs in observations {
            let u = self.control_law(&obs.observation, obs.tick, obs.time);
            let u_clamped = self.clamp(u.clone());
            self.controller_core_mut()
                .observation_history
                .push(obs.observation.clone());
            self.controller_core_mut()
                .control_history
                .push(u_clamped.clone());
            self.on_tick(&obs.observation, &u, &u_clamped);
            let token = ControlToken {
                control: u_clamped.clone(),
                observation: obs.observation.clone(),
                tick: obs.tick,
                time: obs.time,
            };
            self.core_mut().emit(Rc::new(token), CH_CONTROL);
            self.controller_core_mut().ticks_processed += 1;
        }
    }

    /// Synchronous one-shot helper: run the control law on a single observation
    /// and return the clamped control (with history bookkeeping + `on_tick`).
    fn step(&mut self, observation: O, tick: f64, time: f64) -> U
    where
        O: Clone,
        U: Saturate + Clone,
    {
        let u = self.control_law(&observation, tick, time);
        let u_clamped = self.clamp(u.clone());
        self.controller_core_mut()
            .observation_history
            .push(observation.clone());
        self.controller_core_mut()
            .control_history
            .push(u_clamped.clone());
        self.on_tick(&observation, &u, &u_clamped);
        self.controller_core_mut().ticks_processed += 1;
        u_clamped
    }

    /// `hasWork` override: any pending observation counts as work.
    fn controller_has_work(&self) -> bool {
        self.core().inbox_size(CH_OBSERVATION) > 0
    }

    // ── PUBLIC ACCESSOR ────────────────────────────────────────────────────────

    fn ticks_processed(&self) -> usize {
        self.controller_core().ticks_processed
    }
}

#[cfg(test)]
mod tests {
    use super::super::station::{DESStation, StationCore};
    use super::*;
    use std::any::Any;

    /// A textbook PID controller over scalar observation/control.
    struct Pid {
        core: StationCore,
        ctrl: ControllerCore<f64, f64>,
        kp: f64,
        ki: f64,
        kd: f64,
        setpoint: f64,
        integral: f64,
        prev_error: f64,
        lo: Option<f64>,
        hi: Option<f64>,
    }

    impl Pid {
        fn new(kp: f64, ki: f64, kd: f64, setpoint: f64) -> Self {
            Pid {
                core: StationCore::new("pid"),
                ctrl: ControllerCore::new(),
                kp,
                ki,
                kd,
                setpoint,
                integral: 0.0,
                prev_error: 0.0,
                lo: None,
                hi: None,
            }
        }
    }

    impl DESStation for Pid {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn run_time_step(&mut self) {
            self.controller_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.controller_has_work()
        }
    }

    impl ControllerStation<f64, f64> for Pid {
        fn controller_core(&self) -> &ControllerCore<f64, f64> {
            &self.ctrl
        }
        fn controller_core_mut(&mut self) -> &mut ControllerCore<f64, f64> {
            &mut self.ctrl
        }
        fn control_law(&mut self, observation: &f64, _tick: f64, _time: f64) -> f64 {
            let error = self.setpoint - *observation;
            self.integral += error;
            let deriv = error - self.prev_error;
            self.prev_error = error;
            self.kp * error + self.ki * self.integral + self.kd * deriv
        }
        fn u_min(&self) -> Option<f64> {
            self.lo
        }
        fn u_max(&self) -> Option<f64> {
            self.hi
        }
    }

    #[test]
    fn pid_drives_error_down() {
        // Proportional controller on an integrator plant y_{k+1} = y_k + u_k.
        let mut pid = Pid::new(0.5, 0.0, 0.0, 10.0);
        let mut y = 0.0_f64;
        for k in 0..60 {
            let u = pid.step(y, k as f64, k as f64);
            y += u;
        }
        assert!((y - 10.0).abs() < 1e-3, "y did not converge: {y}");
        assert_eq!(pid.ticks_processed(), 60);
    }

    #[test]
    fn clamp_saturates() {
        let mut pid = Pid::new(1.0, 0.0, 0.0, 100.0);
        pid.lo = Some(-2.0);
        pid.hi = Some(2.0);
        // Huge error => raw control 100, clamped to 2.0.
        let u = pid.step(0.0, 0.0, 0.0);
        assert_eq!(u, 2.0);
        assert_eq!(
            pid.controller_core().control_history.last().copied(),
            Some(2.0)
        );
    }

    #[test]
    fn run_time_step_drains_and_records() {
        let mut pid = Pid::new(0.5, 0.0, 0.0, 10.0);
        assert!(!pid.has_work());
        pid.core_mut().take(
            Rc::new(ObservationToken::new(0.0_f64, 0.0, 0.0)),
            CH_OBSERVATION,
        );
        pid.core_mut().take(
            Rc::new(ObservationToken::new(4.0_f64, 1.0, 1.0)),
            CH_OBSERVATION,
        );
        assert!(pid.has_work());
        pid.run_time_step();
        assert_eq!(pid.ticks_processed(), 2);
        assert_eq!(pid.controller_core().observation_history, vec![0.0, 4.0]);
        // controls: 0.5*(10-0)=5, 0.5*(10-4)=3
        assert_eq!(pid.controller_core().control_history, vec![5.0, 3.0]);
        assert!(!pid.has_work());
    }
}
