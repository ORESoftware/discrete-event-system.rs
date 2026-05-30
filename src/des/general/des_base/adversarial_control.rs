//! Port of `src/des/general/des-base/adversarial-control.ts` — shared
//! station/token bases for closed-loop adversarial and stochastic control
//! (plant vs controller vs disturbance).
//!
//! ## Topology
//!
//!   Plant station --StateObservationToken--> controller station
//!   Plant station --StateObservationToken--> disturbance/adversary station
//!   Controller station --ControlMoveToken--> plant station
//!   Adversary station --DisturbanceMoveToken--> plant station
//!
//! The plant owns the continuous state. Policies and adversaries are stationary
//! entities that emit movable command tokens.
//!
//! ## Rust shape
//!
//!   * `const CH_*` → `&str` consts (this module's own; distinct from the
//!     `controller` module's same-named consts).
//!   * `class StateObservationToken / ControlMoveToken / DisturbanceMoveToken`
//!     `implements Token` → plain structs. Tokens flow as `Rc<dyn Any>` (see
//!     `station.rs`), so no `Token` trait is needed; `drain::<T>()` downcasts.
//!   * TEMPLATE METHOD: the abstract `ClosedLoopPlantStation` becomes the
//!     [`ClosedLoopPlantStation`] trait — required `dynamics` hook, provided
//!     `stage_cost`/`terminal` defaults, *final* `plant_run_time_step`. Shared
//!     plant state lives in [`ClosedLoopPlantCore`].
//!   * `FeedbackPolicyStation` / `DisturbancePolicyStation` → traits with a
//!     single required `policy` hook + their own `*Core`.
//!   * `vectors number[]` → `Vec<f64>`; `.slice()` defensive copies → `.clone()`.
//!   * `this.constructor.name` (error model name) → an associated
//!     `const MODEL_NAME: &'static str` per impl.
//!   * `Preconditions.*` throws → `panic!` at the failing edge (invariant).
//!   * `runClosedLoopGame` (`shuffle: false` + `maxTicks`) → an
//!     [`IterativeRunOptions`] with `shuffle = false`.

use super::preconditions::{Check, Preconditions};
use super::runner::{run_iterative_des, IterativeRunOptions, IterativeRunSummary};
use super::station::{DESStation, StationRef};

use std::cell::RefCell;
use std::rc::Rc;

/// Observation channel.
pub const CH_OBSERVATION: &str = "observation";
/// Control channel.
pub const CH_CONTROL: &str = "control";
/// Disturbance channel.
pub const CH_DISTURBANCE: &str = "disturbance";

/// Panic at the failing edge if a precondition guard fails (mirrors TS `throw`).
fn require(c: Check) {
    if let Err(e) = c {
        panic!("{e}");
    }
}

/// State snapshot delivered from the plant to the policies.
pub struct StateObservationToken {
    pub state: Vec<f64>,
    pub tick: f64,
    pub time: f64,
}

impl StateObservationToken {
    pub fn new(state: Vec<f64>, tick: f64, time: f64) -> Self {
        StateObservationToken { state, tick, time }
    }
}

/// A control move emitted by the controller.
pub struct ControlMoveToken {
    pub control: Vec<f64>,
    pub tick: f64,
    pub time: f64,
}

impl ControlMoveToken {
    pub fn new(control: Vec<f64>, tick: f64, time: f64) -> Self {
        ControlMoveToken {
            control,
            tick,
            time,
        }
    }
}

/// A disturbance move emitted by the adversary.
pub struct DisturbanceMoveToken {
    pub disturbance: Vec<f64>,
    pub tick: f64,
    pub time: f64,
}

impl DisturbanceMoveToken {
    pub fn new(disturbance: Vec<f64>, tick: f64, time: f64) -> Self {
        DisturbanceMoveToken {
            disturbance,
            tick,
            time,
        }
    }
}

/// One recorded step of a closed-loop game.
#[derive(Clone, Debug)]
pub struct ClosedLoopGameTraceRow {
    pub tick: usize,
    pub time: f64,
    pub state: Vec<f64>,
    pub control: Vec<f64>,
    pub disturbance: Vec<f64>,
    pub cost: f64,
}

/// Plant construction options.
#[derive(Clone, Debug)]
pub struct ClosedLoopPlantOptions {
    pub x0: Vec<f64>,
    pub dt: f64,
    pub num_steps: usize,
    pub control_dim: usize,
    pub disturbance_dim: usize,
}

/// Shared plant state (the fields of the TS `abstract class`).
pub struct ClosedLoopPlantCore {
    pub state: Vec<f64>,
    pub dt: f64,
    pub num_steps: usize,
    pub control_dim: usize,
    pub disturbance_dim: usize,
    pub control: Vec<f64>,
    pub disturbance: Vec<f64>,
    pub tick: usize,
    pub emitted_initial_observation: bool,
    pub finished: bool,
    pub trace: Vec<ClosedLoopGameTraceRow>,
    pub state_history: Vec<Vec<f64>>,
}

impl ClosedLoopPlantCore {
    /// Validate `opts` and build the initial plant state. Panics on an invalid
    /// spec (TS constructor `throw`).
    pub fn new(id: &str, opts: ClosedLoopPlantOptions) -> Self {
        require(Preconditions::non_empty(id, "x0", &opts.x0));
        require(Preconditions::all_finite(id, "x0", &opts.x0));
        require(Preconditions::positive(id, "dt", opts.dt));
        require(Preconditions::integer_in_range(
            id,
            "numSteps",
            opts.num_steps as f64,
            1.0,
            1e9,
        ));
        require(Preconditions::integer_in_range(
            id,
            "controlDim",
            opts.control_dim as f64,
            1.0,
            1e6,
        ));
        require(Preconditions::integer_in_range(
            id,
            "disturbanceDim",
            opts.disturbance_dim as f64,
            1.0,
            1e6,
        ));
        let state = opts.x0.clone();
        ClosedLoopPlantCore {
            state: state.clone(),
            dt: opts.dt,
            num_steps: opts.num_steps,
            control_dim: opts.control_dim,
            disturbance_dim: opts.disturbance_dim,
            control: vec![0.0; opts.control_dim],
            disturbance: vec![0.0; opts.disturbance_dim],
            tick: 0,
            emitted_initial_observation: false,
            finished: false,
            trace: Vec::new(),
            state_history: vec![state],
        }
    }
}

/// Template-method base for the plant (owns the continuous state).
pub trait ClosedLoopPlantStation: DESStation {
    /// Error-model name (TS `this.constructor.name`).
    const MODEL_NAME: &'static str;

    /// Borrow the shared plant state.
    fn plant_core(&self) -> &ClosedLoopPlantCore;
    /// Mutably borrow the shared plant state.
    fn plant_core_mut(&mut self) -> &mut ClosedLoopPlantCore;

    // ── HOOK (abstract) ───────────────────────────────────────────────────────

    /// Required dynamics hook: x_{k+1} = f(x_k, u_k, w_k, dt).
    fn dynamics(&self, state: &[f64], control: &[f64], disturbance: &[f64], dt: f64) -> Vec<f64>;

    // ── HOOKS (optional override) ─────────────────────────────────────────────

    /// Per-step cost. Default: ‖x'‖² + 0.01‖u‖² + 0.01‖w‖².
    fn stage_cost(
        &self,
        _state: &[f64],
        control: &[f64],
        disturbance: &[f64],
        next_state: &[f64],
    ) -> f64 {
        let state_cost: f64 = next_state.iter().map(|x| x * x).sum();
        let control_cost: f64 = control.iter().map(|u| u * u).sum();
        let disturbance_cost: f64 = disturbance.iter().map(|w| w * w).sum();
        state_cost + 0.01 * control_cost + 0.01 * disturbance_cost
    }

    /// Terminal predicate. Default: never terminal.
    fn terminal(&self, _state: &[f64], _tick: usize) -> bool {
        false
    }

    // ── PRECONDITIONS ─────────────────────────────────────────────────────────

    /// Pre-run guard (TS `assertPreconditions`).
    fn plant_assert_preconditions(&mut self) {
        let cls = Self::MODEL_NAME;
        let c = self.plant_core();
        require(Preconditions::non_empty(cls, "x0", &c.state));
        require(Preconditions::all_finite(cls, "x0", &c.state));
        require(Preconditions::positive(cls, "dt", c.dt));
        require(Preconditions::integer_in_range(
            cls,
            "numSteps",
            c.num_steps as f64,
            1.0,
            1e9,
        ));
        require(Preconditions::integer_in_range(
            cls,
            "controlDim",
            c.control_dim as f64,
            1.0,
            1e6,
        ));
        require(Preconditions::integer_in_range(
            cls,
            "disturbanceDim",
            c.disturbance_dim as f64,
            1.0,
            1e6,
        ));
    }

    // ── TEMPLATE METHOD (final) ───────────────────────────────────────────────

    /// One plant tick: drain the latest control & disturbance, advance the
    /// dynamics, record the trace row, and emit a fresh observation. Concrete
    /// stations wire this up from [`DESStation::run_time_step`].
    fn plant_run_time_step(&mut self) {
        if self.plant_core().finished {
            return;
        }
        if !self.plant_core().emitted_initial_observation {
            self.plant_core_mut().emitted_initial_observation = true;
            self.emit_observation();
            return;
        }

        let controls = self.core_mut().drain::<ControlMoveToken>(CH_CONTROL);
        if let Some(t) = controls.last() {
            self.plant_core_mut().control = t.control.clone();
        }
        let disturbances = self
            .core_mut()
            .drain::<DisturbanceMoveToken>(CH_DISTURBANCE);
        if let Some(t) = disturbances.last() {
            self.plant_core_mut().disturbance = t.disturbance.clone();
        }

        let id = self.id().to_string();
        let control = self.plant_core().control.clone();
        let disturbance = self.plant_core().disturbance.clone();
        let control_dim = self.plant_core().control_dim;
        let disturbance_dim = self.plant_core().disturbance_dim;
        require(Preconditions::length_eq(
            &id,
            "control",
            &control,
            control_dim,
        ));
        require(Preconditions::all_finite(&id, "control", &control));
        require(Preconditions::length_eq(
            &id,
            "disturbance",
            &disturbance,
            disturbance_dim,
        ));
        require(Preconditions::all_finite(&id, "disturbance", &disturbance));

        let tick = self.plant_core().tick;
        let num_steps = self.plant_core().num_steps;
        let prev = self.plant_core().state.clone();
        if tick >= num_steps || self.terminal(&prev, tick) {
            self.plant_core_mut().finished = true;
            return;
        }

        let dt = self.plant_core().dt;
        let next = self.dynamics(&prev, &control, &disturbance, dt);
        require(Preconditions::length_eq(
            &id,
            "next state",
            &next,
            prev.len(),
        ));
        require(Preconditions::all_finite(&id, "next state", &next));
        let cost = self.stage_cost(&prev, &control, &disturbance, &next);
        require(Preconditions::finite(&id, "stage cost", cost));

        {
            let c = self.plant_core_mut();
            c.state = next.clone();
            c.tick += 1;
            c.state_history.push(c.state.clone());
            let row = ClosedLoopGameTraceRow {
                tick: c.tick,
                time: c.tick as f64 * c.dt,
                state: c.state.clone(),
                control,
                disturbance,
                cost,
            };
            c.trace.push(row);
        }
        self.emit_observation();
    }

    /// `hasWork`: the plant runs until it is finished.
    fn plant_has_work(&self) -> bool {
        !self.plant_core().finished
    }

    // ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────

    fn get_state(&self) -> Vec<f64> {
        self.plant_core().state.clone()
    }
    fn get_tick(&self) -> usize {
        self.plant_core().tick
    }
    fn get_dt(&self) -> f64 {
        self.plant_core().dt
    }
    fn get_num_steps(&self) -> usize {
        self.plant_core().num_steps
    }

    /// Emit the current state as a [`StateObservationToken`] on `CH_OBSERVATION`.
    fn emit_observation(&mut self) {
        let tok = {
            let c = self.plant_core();
            StateObservationToken::new(c.state.clone(), c.tick as f64, c.tick as f64 * c.dt)
        };
        self.core_mut().emit(Rc::new(tok), CH_OBSERVATION);
    }
}

/// Shared state for a feedback policy station.
pub struct FeedbackPolicyCore {
    pub control_dim: usize,
    pub control_history: Vec<Vec<f64>>,
}

impl FeedbackPolicyCore {
    pub fn new(id: &str, control_dim: usize) -> Self {
        require(Preconditions::integer_in_range(
            id,
            "controlDim",
            control_dim as f64,
            1.0,
            1e6,
        ));
        FeedbackPolicyCore {
            control_dim,
            control_history: Vec::new(),
        }
    }
}

/// Template-method base for a feedback control policy station.
pub trait FeedbackPolicyStation: DESStation {
    /// Error-model name (TS `this.constructor.name`).
    const MODEL_NAME: &'static str;

    fn feedback_core(&self) -> &FeedbackPolicyCore;
    fn feedback_core_mut(&mut self) -> &mut FeedbackPolicyCore;

    /// Required hook: map an observation to a control move.
    fn policy(&mut self, observation: &StateObservationToken) -> Vec<f64>;

    fn feedback_assert_preconditions(&mut self) {
        require(Preconditions::integer_in_range(
            Self::MODEL_NAME,
            "controlDim",
            self.feedback_core().control_dim as f64,
            1.0,
            1e6,
        ));
    }

    fn feedback_run_time_step(&mut self) {
        let observations = self
            .core_mut()
            .drain::<StateObservationToken>(CH_OBSERVATION);
        let id = self.id().to_string();
        let dim = self.feedback_core().control_dim;
        for obs in observations {
            let u = self.policy(obs.as_ref());
            require(Preconditions::length_eq(&id, "control", &u, dim));
            require(Preconditions::all_finite(&id, "control", &u));
            self.feedback_core_mut().control_history.push(u.clone());
            let tok = ControlMoveToken::new(u, obs.tick, obs.time);
            self.core_mut().emit(Rc::new(tok), CH_CONTROL);
        }
    }

    fn feedback_has_work(&self) -> bool {
        self.core().inbox_size(CH_OBSERVATION) > 0
    }
}

/// Shared state for a disturbance / adversary policy station.
pub struct DisturbancePolicyCore {
    pub disturbance_dim: usize,
    pub disturbance_history: Vec<Vec<f64>>,
}

impl DisturbancePolicyCore {
    pub fn new(id: &str, disturbance_dim: usize) -> Self {
        require(Preconditions::integer_in_range(
            id,
            "disturbanceDim",
            disturbance_dim as f64,
            1.0,
            1e6,
        ));
        DisturbancePolicyCore {
            disturbance_dim,
            disturbance_history: Vec::new(),
        }
    }
}

/// Template-method base for a disturbance / adversary policy station.
pub trait DisturbancePolicyStation: DESStation {
    /// Error-model name (TS `this.constructor.name`).
    const MODEL_NAME: &'static str;

    fn disturbance_core(&self) -> &DisturbancePolicyCore;
    fn disturbance_core_mut(&mut self) -> &mut DisturbancePolicyCore;

    /// Required hook: map an observation to a disturbance move.
    fn policy(&mut self, observation: &StateObservationToken) -> Vec<f64>;

    fn disturbance_assert_preconditions(&mut self) {
        require(Preconditions::integer_in_range(
            Self::MODEL_NAME,
            "disturbanceDim",
            self.disturbance_core().disturbance_dim as f64,
            1.0,
            1e6,
        ));
    }

    fn disturbance_run_time_step(&mut self) {
        let observations = self
            .core_mut()
            .drain::<StateObservationToken>(CH_OBSERVATION);
        let id = self.id().to_string();
        let dim = self.disturbance_core().disturbance_dim;
        for obs in observations {
            let w = self.policy(obs.as_ref());
            require(Preconditions::length_eq(&id, "disturbance", &w, dim));
            require(Preconditions::all_finite(&id, "disturbance", &w));
            self.disturbance_core_mut()
                .disturbance_history
                .push(w.clone());
            let tok = DisturbanceMoveToken::new(w, obs.tick, obs.time);
            self.core_mut().emit(Rc::new(tok), CH_DISTURBANCE);
        }
    }

    fn disturbance_has_work(&self) -> bool {
        self.core().inbox_size(CH_OBSERVATION) > 0
    }
}

/// Wire the plant ↔ controller ↔ adversary edges (by shared handle).
pub fn wire_closed_loop_game(plant: &StationRef, controller: &StationRef, adversary: &StationRef) {
    plant
        .borrow_mut()
        .core_mut()
        .pipe(controller.clone(), CH_OBSERVATION, CH_OBSERVATION);
    plant
        .borrow_mut()
        .core_mut()
        .pipe(adversary.clone(), CH_OBSERVATION, CH_OBSERVATION);
    controller
        .borrow_mut()
        .core_mut()
        .pipe(plant.clone(), CH_CONTROL, CH_CONTROL);
    adversary
        .borrow_mut()
        .core_mut()
        .pipe(plant.clone(), CH_DISTURBANCE, CH_DISTURBANCE);
}

/// Options for [`run_closed_loop_game`].
#[derive(Clone, Debug, Default)]
pub struct ClosedLoopGameRunOptions {
    pub max_ticks: Option<usize>,
    pub run_validators: Option<bool>,
}

/// Wire then run a closed-loop game with deterministic (non-shuffled) order.
///
/// Generic over the concrete station types so `plant.get_num_steps()` is
/// reachable (the runner only sees the erased `dyn DESStation`).
pub fn run_closed_loop_game<P, C, A>(
    plant: Rc<RefCell<P>>,
    controller: Rc<RefCell<C>>,
    adversary: Rc<RefCell<A>>,
    opts: ClosedLoopGameRunOptions,
) -> IterativeRunSummary
where
    P: ClosedLoopPlantStation + 'static,
    C: FeedbackPolicyStation + 'static,
    A: DisturbancePolicyStation + 'static,
{
    let plant_ref: StationRef = plant.clone();
    let controller_ref: StationRef = controller.clone();
    let adversary_ref: StationRef = adversary.clone();
    wire_closed_loop_game(&plant_ref, &controller_ref, &adversary_ref);
    let num_steps = plant.borrow().get_num_steps();
    run_iterative_des(
        vec![plant_ref, controller_ref, adversary_ref],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(opts.max_ticks.unwrap_or(num_steps + 3)),
            run_validators: opts.run_validators.unwrap_or(false),
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::station::StationCore;
    use super::*;
    use std::any::Any;

    /// Scalar plant x_{k+1} = x_k + dt·(u_k + w_k).
    struct ScalarPlant {
        core: StationCore,
        plant: ClosedLoopPlantCore,
    }

    impl ScalarPlant {
        fn new(id: &str, opts: ClosedLoopPlantOptions) -> Self {
            ScalarPlant {
                core: StationCore::new(id),
                plant: ClosedLoopPlantCore::new(id, opts),
            }
        }
    }

    impl DESStation for ScalarPlant {
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
            self.plant_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.plant_has_work()
        }
        fn assert_preconditions(&mut self) {
            self.plant_assert_preconditions();
        }
    }

    impl ClosedLoopPlantStation for ScalarPlant {
        const MODEL_NAME: &'static str = "ScalarPlant";
        fn plant_core(&self) -> &ClosedLoopPlantCore {
            &self.plant
        }
        fn plant_core_mut(&mut self) -> &mut ClosedLoopPlantCore {
            &mut self.plant
        }
        fn dynamics(
            &self,
            state: &[f64],
            control: &[f64],
            disturbance: &[f64],
            dt: f64,
        ) -> Vec<f64> {
            vec![state[0] + dt * (control[0] + disturbance[0])]
        }
    }

    /// Proportional controller u = −k·x (saturating to keep dt·k < 1 stable).
    struct PController {
        core: StationCore,
        fb: FeedbackPolicyCore,
        k: f64,
    }

    impl PController {
        fn new(id: &str, k: f64) -> Self {
            PController {
                core: StationCore::new(id),
                fb: FeedbackPolicyCore::new(id, 1),
                k,
            }
        }
    }

    impl DESStation for PController {
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
            self.feedback_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.feedback_has_work()
        }
        fn assert_preconditions(&mut self) {
            self.feedback_assert_preconditions();
        }
    }

    impl FeedbackPolicyStation for PController {
        const MODEL_NAME: &'static str = "PController";
        fn feedback_core(&self) -> &FeedbackPolicyCore {
            &self.fb
        }
        fn feedback_core_mut(&mut self) -> &mut FeedbackPolicyCore {
            &mut self.fb
        }
        fn policy(&mut self, observation: &StateObservationToken) -> Vec<f64> {
            vec![-self.k * observation.state[0]]
        }
    }

    /// Bounded worst-case adversary: w = bound·sign(x), |w| ≤ bound.
    struct BoundedAdversary {
        core: StationCore,
        dist: DisturbancePolicyCore,
        bound: f64,
    }

    impl BoundedAdversary {
        fn new(id: &str, bound: f64) -> Self {
            BoundedAdversary {
                core: StationCore::new(id),
                dist: DisturbancePolicyCore::new(id, 1),
                bound,
            }
        }
    }

    impl DESStation for BoundedAdversary {
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
            self.disturbance_run_time_step();
        }
        fn has_work(&self) -> bool {
            self.disturbance_has_work()
        }
        fn assert_preconditions(&mut self) {
            self.disturbance_assert_preconditions();
        }
    }

    impl DisturbancePolicyStation for BoundedAdversary {
        const MODEL_NAME: &'static str = "BoundedAdversary";
        fn disturbance_core(&self) -> &DisturbancePolicyCore {
            &self.dist
        }
        fn disturbance_core_mut(&mut self) -> &mut DisturbancePolicyCore {
            &mut self.dist
        }
        fn policy(&mut self, observation: &StateObservationToken) -> Vec<f64> {
            let s = observation.state[0];
            let sign = if s > 0.0 {
                1.0
            } else if s < 0.0 {
                -1.0
            } else {
                0.0
            };
            vec![self.bound * sign]
        }
    }

    fn plant_opts() -> ClosedLoopPlantOptions {
        ClosedLoopPlantOptions {
            x0: vec![1.0],
            dt: 1.0,
            num_steps: 60,
            control_dim: 1,
            disturbance_dim: 1,
        }
    }

    #[test]
    fn robust_to_bounded_disturbance() {
        // dt=1, k=0.5 ⇒ closed loop x_{k+1} = 0.5·x + w with |w| ≤ 0.1.
        // Worst-case adversary drives |x| toward the bound 0.1/(1−0.5) = 0.2.
        let bound = 0.1;
        let k = 0.5;
        let plant = Rc::new(RefCell::new(ScalarPlant::new("plant", plant_opts())));
        let controller = Rc::new(RefCell::new(PController::new("ctrl", k)));
        let adversary = Rc::new(RefCell::new(BoundedAdversary::new("adv", bound)));
        let summary = run_closed_loop_game(
            plant.clone(),
            controller.clone(),
            adversary.clone(),
            ClosedLoopGameRunOptions::default(),
        );
        assert!(summary.ticks > 0);

        let final_state = plant.borrow().get_state()[0].abs();
        let steady = bound / (1.0 - k); // 0.2
        assert!(
            final_state <= steady + 1e-6,
            "state not robustly bounded: {final_state} > {steady}"
        );

        // Every emitted disturbance respected the bound.
        for w in &adversary.borrow().disturbance_core().disturbance_history {
            assert!(
                w[0].abs() <= bound + 1e-12,
                "disturbance exceeded bound: {}",
                w[0]
            );
        }
    }

    #[test]
    fn zero_disturbance_drives_state_to_zero() {
        let plant = Rc::new(RefCell::new(ScalarPlant::new("plant0", plant_opts())));
        let controller = Rc::new(RefCell::new(PController::new("ctrl0", 0.5)));
        // bound = 0 ⇒ no disturbance; pure closed-loop contraction toward 0.
        let adversary = Rc::new(RefCell::new(BoundedAdversary::new("adv0", 0.0)));
        run_closed_loop_game(
            plant.clone(),
            controller.clone(),
            adversary.clone(),
            ClosedLoopGameRunOptions::default(),
        );
        assert!(
            plant.borrow().get_state()[0].abs() < 1e-6,
            "did not converge: {}",
            plant.borrow().get_state()[0]
        );
    }

    #[test]
    fn records_trace_and_history() {
        let plant = Rc::new(RefCell::new(ScalarPlant::new("plantT", plant_opts())));
        let controller = Rc::new(RefCell::new(PController::new("ctrlT", 0.5)));
        let adversary = Rc::new(RefCell::new(BoundedAdversary::new("advT", 0.05)));
        run_closed_loop_game(
            plant.clone(),
            controller.clone(),
            adversary.clone(),
            ClosedLoopGameRunOptions::default(),
        );
        let p = plant.borrow();
        let c = p.plant_core();
        assert_eq!(c.trace.len(), c.num_steps);
        // state_history has the initial state plus one row per advanced step.
        assert_eq!(c.state_history.len(), c.num_steps + 1);
        assert_eq!(c.tick, c.num_steps);
        // Trace ticks are 1..=num_steps in order.
        assert_eq!(c.trace.first().unwrap().tick, 1);
        assert_eq!(c.trace.last().unwrap().tick, c.num_steps);
    }
}
