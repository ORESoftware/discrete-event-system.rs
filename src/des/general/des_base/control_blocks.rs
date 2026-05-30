//! Port of `src/des/general/des-base/control-blocks.ts`
//! (module `des::general::des_base::control_blocks`).
//!
//! Block-diagram control on the heavyweight signal-entity framework: a
//! [`PlantBlock`] owns continuous state and emits measurements `y`, a
//! [`ControllerBlock`] turns `y` into a control `u`, an optional
//! [`EstimatorBlock`] turns `y` (+ `u`) into a state estimate `x̂`, and
//! [`run_closed_loop`] drives the loop in lock-step.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The TS `abstract class`es extend `MultiDirectionalSignalEntity` and carry
//!     both shared fields and one abstract hook. Rust has no inheritance, so —
//!     mirroring the framework's `*Core` + trait split — each role is a trait
//!     ([`PlantBlock`] / [`ControllerBlock`] / [`EstimatorBlock`]) over a
//!     composable field-bag ([`PlantCore`] / [`ControllerCore`] /
//!     [`EstimatorCore`], each embedding a shared [`BlockCore`]). The required
//!     hooks (`dynamics` / `control_law` / `update`) are trait methods; the rest
//!     (`run_time_step`, `assert_preconditions`, …) are provided defaults.
//!   * `VectorSignal extends SignalValue` (a moving entity carrying `number[]` +
//!     a `kind` tag) → the [`VectorSignal`] struct.
//!   * `Preconditions.*` (which `throw`) reuse the engine's ported guards (which
//!     return a `Check = Result<…>`); [`require`] re-introduces the throw-on-fail
//!     behaviour (`panic!`), matching the TS invariant semantics.
//!   * `dt` is plain `f64` (seconds), as in the TS source; the unused
//!     `stepSize: BigNumber` parameter is dropped from the per-tick hook.
//!
//! PORT NOTE: the engine's `MultiDirectionalSignalCore` queue is scalar-only
//! (`dyn MovingEntity` whose `get_value` is a single number), so it cannot carry
//! the *vector* `VectorSignal`s with `kind` tags these blocks exchange. Each
//! block therefore keeps its own typed `VecDeque<VectorSignal>` inbox (exactly
//! the `queue` of `VectorSignal`s the TS used).
//!
//! PORT NOTE: the TS routed emissions through `connectionsOut` →
//! `target.acceptItem/takeItem`. Owned trait objects cannot mint an `Rc`-to-self
//! to build those `EntityConnection`s, so wiring is modelled as a list of target
//! ids ([`BlockCore::out_targets`], set by [`ensure_connected`]) and each tick's
//! emissions are *staged* in [`BlockCore::pending_out`]; [`run_closed_loop`]
//! delivers them to the wired target's inbox. The observable behaviour (drain
//! inbox → compute → emit to connected targets) is identical.

#![allow(dead_code)]

use std::collections::VecDeque;

use crate::des::general::des_base::preconditions::{Check, Preconditions};

/// Re-introduce TS `throw`-on-failed-guard semantics over the engine's
/// `Result`-returning [`Preconditions`] (an invariant violation → `panic!`).
fn require(c: Check) {
    if let Err(e) = c {
        panic!("{e}");
    }
}

// -----------------------------------------------------------------------------
// VECTOR SIGNAL (moving entity)
// -----------------------------------------------------------------------------

/// Moving entity carrying an arbitrary numeric vector — the payload that flows
/// along block-diagram connections (measurement `y`, control `u`, estimate
/// `x̂`, error `e`).
#[derive(Clone, Debug)]
pub struct VectorSignal {
    /// Numeric payload.
    pub vec: Vec<f64>,
    /// Diagnostic kind tag (`"y"`, `"u"`, `"xhat"`, `"e"`, …).
    pub kind: String,
    /// Discrete tick at which this signal was generated.
    pub tick: usize,
}

impl VectorSignal {
    /// `new VectorSignal(vec, kind, tick)` (copies `vec`).
    pub fn new(vec: &[f64], kind: &str, tick: usize) -> Self {
        VectorSignal {
            vec: vec.to_vec(),
            kind: kind.to_string(),
            tick,
        }
    }

    /// `getValue(): number { return this.vec[0]; }`.
    pub fn get_value(&self) -> f64 {
        self.vec[0]
    }
}

// -----------------------------------------------------------------------------
// SHARED BLOCK CORE
// -----------------------------------------------------------------------------

/// The signal-entity surface shared by every block (the part inherited from
/// `MultiDirectionalSignalEntity`): id, typed inbox, wired out-targets, and the
/// staged outgoing signals (see module PORT NOTEs).
#[derive(Default)]
pub struct BlockCore {
    pub id: String,
    /// `queue` of `VectorSignal`s awaiting consumption.
    pub inbox: VecDeque<VectorSignal>,
    /// Ids of wired `connectionsOut` targets.
    pub out_targets: Vec<String>,
    /// Ids of wired `connectionsIn` sources (kept for fidelity; unused in routing).
    pub in_sources: Vec<String>,
    /// Signals emitted this tick, delivered to `out_targets` by the driver.
    pub pending_out: Vec<VectorSignal>,
}

impl BlockCore {
    pub fn new(id: &str) -> Self {
        BlockCore {
            id: id.to_string(),
            ..Default::default()
        }
    }

    /// `takeItem(m) { this.queue.enqueue(m); }`.
    pub fn take_item(&mut self, m: VectorSignal) {
        self.inbox.push_back(m);
    }
}

/// Common accessor trait so the driver can wire / route any block uniformly.
pub trait SignalBlock {
    fn block_core(&self) -> &BlockCore;
    fn block_core_mut(&mut self) -> &mut BlockCore;
    fn id(&self) -> &str {
        &self.block_core().id
    }
    /// `acceptItem(_m): boolean { return true; }` then `takeItem`.
    fn take_item(&mut self, m: VectorSignal) {
        self.block_core_mut().take_item(m);
    }
}

// -----------------------------------------------------------------------------
// PLANT BLOCK
// -----------------------------------------------------------------------------

/// Field-bag for a `PlantBlock`: continuous state, sample period, and the per-
/// tick histories.
pub struct PlantCore {
    pub block: BlockCore,
    /// Continuous state vector.
    pub state: Vec<f64>,
    /// Sample period dt (seconds).
    pub dt: f64,
    pub state_history: Vec<Vec<f64>>,
    pub input_history: Vec<Vec<f64>>,
    pub output_history: Vec<Vec<f64>>,
    /// Discrete tick counter.
    pub tick: usize,
    /// Most recent control received.
    pub last_u: Vec<f64>,
}

impl PlantCore {
    /// `constructor(id, x0, dt, mDim)` — `throw`s if `dt <= 0`.
    pub fn new(id: &str, x0: &[f64], dt: f64, m_dim: usize) -> Self {
        if dt <= 0.0 {
            panic!("PlantBlock {id}: dt must be positive");
        }
        PlantCore {
            block: BlockCore::new(id),
            state: x0.to_vec(),
            dt,
            state_history: vec![x0.to_vec()],
            input_history: Vec::new(),
            output_history: Vec::new(),
            tick: 0,
            last_u: vec![0.0; m_dim],
        }
    }
}

/// `abstract class PlantBlock` — a controllable plant.
pub trait PlantBlock: SignalBlock {
    fn plant_core(&self) -> &PlantCore;
    fn plant_core_mut(&mut self) -> &mut PlantCore;

    /// Plant dynamics `x' = f(x, u, dt)` (required hook).
    fn dynamics(&self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64>;

    /// Measurement equation `y = h(x)`. Default identity.
    fn observe(&self, x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }

    /// Pre-run guard (universal invariants). Override to add specifics, calling
    /// this first.
    fn assert_preconditions(&self) {
        let p = self.plant_core();
        require(Preconditions::positive("PlantBlock", "dt", p.dt));
        require(Preconditions::non_empty("PlantBlock", "state", &p.state));
        require(Preconditions::all_finite("PlantBlock", "state", &p.state));
        require(Preconditions::all_finite("PlantBlock", "lastU", &p.last_u));
    }

    /// Drain incoming controls, advance dynamics, emit measurement.
    fn run_time_step(&mut self) {
        // 1. Drain ALL incoming controls — keep the most recent (last wins).
        let drained: Vec<VectorSignal> = self.block_core_mut().inbox.drain(..).collect();
        for sig in drained {
            if sig.kind == "u" || sig.kind == "control" {
                self.plant_core_mut().last_u = sig.vec.clone();
            }
        }
        // 2. Advance state.
        let (state, last_u, dt) = {
            let p = self.plant_core();
            (p.state.clone(), p.last_u.clone(), p.dt)
        };
        let x_new = self.dynamics(&state, &last_u, dt);
        {
            let p = self.plant_core_mut();
            p.input_history.push(last_u);
            p.state = x_new.clone();
            p.state_history.push(x_new.clone());
            p.tick += 1;
        }
        // 3. Emit measurement.
        let y = self.observe(&self.plant_core().state);
        let tick = self.plant_core().tick;
        {
            let p = self.plant_core_mut();
            p.output_history.push(y.clone());
        }
        self.block_core_mut()
            .pending_out
            .push(VectorSignal::new(&y, "y", tick));
    }
}

// -----------------------------------------------------------------------------
// CONTROLLER BLOCK
// -----------------------------------------------------------------------------

/// Field-bag for a `ControllerBlock`.
pub struct ControllerCore {
    pub block: BlockCore,
    /// Number of control inputs `m`.
    pub m_dim: usize,
    pub tick: usize,
    pub input_history: Vec<Vec<f64>>,
    pub output_history: Vec<Vec<f64>>,
    /// Optional saturation bounds.
    pub u_min: Option<Vec<f64>>,
    pub u_max: Option<Vec<f64>>,
    /// Last received measurement (so the controller can run on empty inboxes).
    pub last_y: Option<Vec<f64>>,
}

impl ControllerCore {
    /// `constructor(id, mDim)`.
    pub fn new(id: &str, m_dim: usize) -> Self {
        ControllerCore {
            block: BlockCore::new(id),
            m_dim,
            tick: 0,
            input_history: Vec::new(),
            output_history: Vec::new(),
            u_min: None,
            u_max: None,
            last_y: None,
        }
    }

    /// `setSaturation(uMin, uMax)`.
    pub fn set_saturation(&mut self, u_min: Option<Vec<f64>>, u_max: Option<Vec<f64>>) {
        self.u_min = u_min;
        self.u_max = u_max;
    }

    /// `saturate(u)` — clamp componentwise to `[uMin, uMax]`.
    pub fn saturate(&self, u: &[f64]) -> Vec<f64> {
        let mut out = u.to_vec();
        if let Some(lo) = &self.u_min {
            for (i, v) in out.iter_mut().enumerate() {
                if i < lo.len() && *v < lo[i] {
                    *v = lo[i];
                }
            }
        }
        if let Some(hi) = &self.u_max {
            for (i, v) in out.iter_mut().enumerate() {
                if i < hi.len() && *v > hi[i] {
                    *v = hi[i];
                }
            }
        }
        out
    }
}

/// `abstract class ControllerBlock` — a feedback controller.
pub trait ControllerBlock: SignalBlock {
    fn controller_core(&self) -> &ControllerCore;
    fn controller_core_mut(&mut self) -> &mut ControllerCore;

    /// `controlLaw(y, tick, t)` (required hook).
    fn control_law(&self, y: &[f64], tick: usize, t: f64) -> Vec<f64>;

    /// `getDt()` — override if the controller needs `dt` for derivative terms.
    fn get_dt(&self) -> f64 {
        1.0
    }

    fn m_dim(&self) -> usize {
        self.controller_core().m_dim
    }

    /// Pre-run guard.
    fn assert_preconditions(&self) {
        let c = self.controller_core();
        require(Preconditions::integer(
            "ControllerBlock",
            "mDim",
            c.m_dim as f64,
        ));
        require(Preconditions::check(
            "ControllerBlock",
            "mDim",
            "be >= 1",
            c.m_dim >= 1,
            Some(c.m_dim.to_string()),
        ));
        if let (Some(lo), Some(hi)) = (&c.u_min, &c.u_max) {
            require(Preconditions::length_eq(
                "ControllerBlock",
                "uMin",
                lo,
                c.m_dim,
            ));
            require(Preconditions::length_eq(
                "ControllerBlock",
                "uMax",
                hi,
                c.m_dim,
            ));
            for i in 0..c.m_dim {
                require(Preconditions::check(
                    "ControllerBlock",
                    &format!("uMin[{i}] <= uMax[{i}]"),
                    "satisfy uMin <= uMax",
                    lo[i] <= hi[i],
                    Some(format!("[{}, {}]", lo[i], hi[i])),
                ));
            }
        }
    }

    fn run_time_step(&mut self) {
        let mut y: Option<Vec<f64>> = self.controller_core().last_y.clone();
        let drained: Vec<VectorSignal> = self.block_core_mut().inbox.drain(..).collect();
        for sig in drained {
            if sig.kind == "y" || sig.kind == "meas" {
                y = Some(sig.vec.clone());
            }
        }
        let Some(y) = y else {
            return;
        };
        let tick = {
            let c = self.controller_core_mut();
            c.last_y = Some(y.clone());
            c.tick += 1;
            c.tick
        };
        let t = tick as f64 * self.get_dt();
        let mut u = self.control_law(&y, tick, t);
        let needs_sat = {
            let c = self.controller_core();
            c.u_min.is_some() || c.u_max.is_some()
        };
        if needs_sat {
            u = self.controller_core().saturate(&u);
        }
        {
            let c = self.controller_core_mut();
            c.input_history.push(y);
            c.output_history.push(u.clone());
        }
        self.block_core_mut()
            .pending_out
            .push(VectorSignal::new(&u, "u", tick));
    }
}

// -----------------------------------------------------------------------------
// ESTIMATOR BLOCK
// -----------------------------------------------------------------------------

/// Field-bag for an `EstimatorBlock`.
pub struct EstimatorCore {
    pub block: BlockCore,
    pub tick: usize,
    pub estimate_history: Vec<Vec<f64>>,
    pub measurement_history: Vec<Vec<f64>>,
    pub last_u: Option<Vec<f64>>,
}

impl EstimatorCore {
    /// `constructor(id)`.
    pub fn new(id: &str) -> Self {
        EstimatorCore {
            block: BlockCore::new(id),
            tick: 0,
            estimate_history: Vec::new(),
            measurement_history: Vec::new(),
            last_u: None,
        }
    }
}

/// `abstract class EstimatorBlock` — a state estimator (observer / Kalman).
pub trait EstimatorBlock: SignalBlock {
    fn estimator_core(&self) -> &EstimatorCore;
    fn estimator_core_mut(&mut self) -> &mut EstimatorCore;

    /// One filter step `(y, u) → x̂` (required hook).
    fn update(&mut self, y: &[f64], u: Option<&[f64]>) -> Vec<f64>;

    /// Current estimate (required hook).
    fn get_estimate(&self) -> Vec<f64>;

    fn assert_preconditions(&self) {
        let e = self.get_estimate();
        require(Preconditions::all_finite("EstimatorBlock", "estimate", &e));
    }

    fn run_time_step(&mut self) {
        let mut y: Option<Vec<f64>> = None;
        let drained: Vec<VectorSignal> = self.block_core_mut().inbox.drain(..).collect();
        for sig in drained {
            if sig.kind == "y" || sig.kind == "meas" {
                y = Some(sig.vec.clone());
            } else if sig.kind == "u" || sig.kind == "control" {
                self.estimator_core_mut().last_u = Some(sig.vec.clone());
            }
        }
        let Some(y) = y else {
            return;
        };
        let tick = {
            let e = self.estimator_core_mut();
            e.tick += 1;
            e.measurement_history.push(y.clone());
            e.tick
        };
        let last_u = self.estimator_core().last_u.clone();
        let xhat = self.update(&y, last_u.as_deref());
        self.estimator_core_mut()
            .estimate_history
            .push(xhat.clone());
        self.block_core_mut()
            .pending_out
            .push(VectorSignal::new(&xhat, "xhat", tick));
    }
}

// -----------------------------------------------------------------------------
// CLOSED-LOOP DRIVER
// -----------------------------------------------------------------------------

/// `interface ClosedLoopOpts` (the optional `estimator` is passed separately to
/// [`run_closed_loop`] — see its signature note).
#[derive(Clone, Debug, Default)]
pub struct ClosedLoopOpts {
    pub num_steps: usize,
    /// Seed control `u0` fed to the plant before any controller fires.
    pub u0: Option<Vec<f64>>,
}

/// `interface ClosedLoopResult`.
#[derive(Clone, Debug)]
pub struct ClosedLoopResult {
    pub trajectory: Vec<Vec<f64>>,
    pub controls: Vec<Vec<f64>>,
    pub measurements: Vec<Vec<f64>>,
    pub estimates: Option<Vec<Vec<f64>>>,
    pub num_steps: usize,
}

/// `ensureConnected(src, tgt)` — wire `src → tgt` if not already wired. Operates
/// on the shared [`BlockCore`] so it works for every block type without trait
/// upcasting.
fn ensure_connected(src: &mut BlockCore, tgt_id: &str) {
    if src.out_targets.iter().any(|t| t == tgt_id) {
        return;
    }
    src.out_targets.push(tgt_id.to_string());
}

/// Record the reverse (in) edge for fidelity.
fn add_in_edge(tgt: &mut BlockCore, src_id: &str) {
    if !tgt.in_sources.iter().any(|s| s == src_id) {
        tgt.in_sources.push(src_id.to_string());
    }
}

/// Deliver a block's staged emissions to the inbox of whichever of the wired
/// blocks matches each out-target id (the routing the TS did inside
/// `runTimeStep` via `connectionsOut`).
fn deliver(
    pending: Vec<VectorSignal>,
    targets: &[String],
    plant: &mut dyn PlantBlock,
    controller: &mut dyn ControllerBlock,
    mut estimator: Option<&mut dyn EstimatorBlock>,
) {
    let plant_id = plant.id().to_string();
    let controller_id = controller.id().to_string();
    let estimator_id = estimator.as_deref().map(|e| e.id().to_string());
    for sig in pending {
        for target_id in targets {
            if *target_id == plant_id {
                plant.take_item(sig.clone());
            } else if *target_id == controller_id {
                controller.take_item(sig.clone());
            } else if estimator_id.as_deref() == Some(target_id.as_str()) {
                if let Some(est) = estimator.as_deref_mut() {
                    est.take_item(sig.clone());
                }
            }
        }
    }
}

/// `runClosedLoop(plant, controller, opts)` — drive plant + controller (+ an
/// optional estimator) in lock-step for `numSteps` ticks.
///
/// Signature note: the TS `opts.estimator` is hoisted to a dedicated parameter
/// here to keep the `&mut` borrows of all three blocks disjoint and explicit.
pub fn run_closed_loop(
    plant: &mut dyn PlantBlock,
    controller: &mut dyn ControllerBlock,
    mut estimator: Option<&mut dyn EstimatorBlock>,
    opts: ClosedLoopOpts,
) -> ClosedLoopResult {
    require(Preconditions::integer(
        "runClosedLoop",
        "numSteps",
        opts.num_steps as f64,
    ));
    require(Preconditions::check(
        "runClosedLoop",
        "numSteps",
        "be >= 1",
        opts.num_steps >= 1,
        Some(opts.num_steps.to_string()),
    ));

    let plant_id = plant.id().to_string();
    let controller_id = controller.id().to_string();
    let estimator_id = estimator.as_deref().map(|e| e.id().to_string());

    // Auto-wire: plant → (estimator ?? controller); estimator → controller;
    // controller → plant; controller → estimator.
    let plant_target = estimator_id
        .clone()
        .unwrap_or_else(|| controller_id.clone());
    ensure_connected(plant.block_core_mut(), &plant_target);
    if let Some(eid) = &estimator_id {
        if let Some(est) = estimator.as_deref_mut() {
            ensure_connected(est.block_core_mut(), &controller_id);
            add_in_edge(est.block_core_mut(), &plant_id);
        }
        add_in_edge(controller.block_core_mut(), eid);
    } else {
        add_in_edge(controller.block_core_mut(), &plant_id);
    }
    ensure_connected(controller.block_core_mut(), &plant_id);
    add_in_edge(plant.block_core_mut(), &controller_id);
    if let Some(eid) = &estimator_id {
        ensure_connected(controller.block_core_mut(), eid);
    }

    // Pre-run guards.
    plant.assert_preconditions();
    controller.assert_preconditions();
    if let Some(est) = estimator.as_deref() {
        est.assert_preconditions();
    }

    // Seed control.
    let m_dim = controller.m_dim();
    let u0 = opts.u0.clone().unwrap_or_else(|| vec![0.0; m_dim]);
    require(Preconditions::length_eq("runClosedLoop", "u0", &u0, m_dim));
    require(Preconditions::all_finite("runClosedLoop", "u0", &u0));
    plant.take_item(VectorSignal::new(&u0, "u", 0));

    // Step loop. `stepBN = bgn(plant.dt)` was computed in TS but ignored by the
    // tick hooks, so it is omitted here.
    let estimates = if let Some(est) = estimator.as_deref_mut() {
        for _k in 0..opts.num_steps {
            plant.run_time_step();
            {
                let pending: Vec<VectorSignal> =
                    std::mem::take(&mut plant.block_core_mut().pending_out);
                let targets = plant.block_core().out_targets.clone();
                deliver(pending, &targets, plant, controller, Some(&mut *est));
            }

            est.run_time_step();
            {
                let pending: Vec<VectorSignal> =
                    std::mem::take(&mut est.block_core_mut().pending_out);
                let targets = est.block_core().out_targets.clone();
                deliver(pending, &targets, plant, controller, None);
            }

            controller.run_time_step();
            {
                let pending: Vec<VectorSignal> =
                    std::mem::take(&mut controller.block_core_mut().pending_out);
                let targets = controller.block_core().out_targets.clone();
                deliver(pending, &targets, plant, controller, Some(&mut *est));
            }
        }
        Some(est.estimator_core().estimate_history.clone())
    } else {
        for _k in 0..opts.num_steps {
            plant.run_time_step();
            {
                let pending: Vec<VectorSignal> =
                    std::mem::take(&mut plant.block_core_mut().pending_out);
                let targets = plant.block_core().out_targets.clone();
                deliver(pending, &targets, plant, controller, None);
            }

            controller.run_time_step();
            {
                let pending: Vec<VectorSignal> =
                    std::mem::take(&mut controller.block_core_mut().pending_out);
                let targets = controller.block_core().out_targets.clone();
                deliver(pending, &targets, plant, controller, None);
            }
        }
        None
    };

    ClosedLoopResult {
        trajectory: plant.plant_core().state_history.clone(),
        controls: controller.controller_core().output_history.clone(),
        measurements: plant.plant_core().output_history.clone(),
        estimates,
        num_steps: opts.num_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A double-integrator plant: x = [pos, vel], u = [accel].
    struct DoubleIntegrator {
        core: PlantCore,
    }
    impl DoubleIntegrator {
        fn new() -> Self {
            DoubleIntegrator {
                core: PlantCore::new("plant", &[1.0, 0.0], 0.1, 1),
            }
        }
    }
    impl SignalBlock for DoubleIntegrator {
        fn block_core(&self) -> &BlockCore {
            &self.core.block
        }
        fn block_core_mut(&mut self) -> &mut BlockCore {
            &mut self.core.block
        }
    }
    impl PlantBlock for DoubleIntegrator {
        fn plant_core(&self) -> &PlantCore {
            &self.core
        }
        fn plant_core_mut(&mut self) -> &mut PlantCore {
            &mut self.core
        }
        fn dynamics(&self, x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
            let a = u.first().copied().unwrap_or(0.0);
            vec![x[0] + dt * x[1], x[1] + dt * a]
        }
    }

    /// PD controller: u = -k_p * pos - k_d * vel.
    struct PController {
        core: ControllerCore,
        kp: f64,
        kd: f64,
    }
    impl PController {
        fn new(kp: f64, kd: f64) -> Self {
            PController {
                core: ControllerCore::new("ctrl", 1),
                kp,
                kd,
            }
        }
    }
    impl SignalBlock for PController {
        fn block_core(&self) -> &BlockCore {
            &self.core.block
        }
        fn block_core_mut(&mut self) -> &mut BlockCore {
            &mut self.core.block
        }
    }
    impl ControllerBlock for PController {
        fn controller_core(&self) -> &ControllerCore {
            &self.core
        }
        fn controller_core_mut(&mut self) -> &mut ControllerCore {
            &mut self.core
        }
        fn control_law(&self, y: &[f64], _tick: usize, _t: f64) -> Vec<f64> {
            let pos = y.first().copied().unwrap_or(0.0);
            let vel = y.get(1).copied().unwrap_or(0.0);
            vec![-self.kp * pos - self.kd * vel]
        }
    }

    #[test]
    fn closed_loop_drives_state_toward_zero() {
        let mut plant = DoubleIntegrator::new();
        let mut ctrl = PController::new(0.5, 1.0);
        let result = run_closed_loop(
            &mut plant,
            &mut ctrl,
            None,
            ClosedLoopOpts {
                num_steps: 50,
                u0: None,
            },
        );
        assert_eq!(result.num_steps, 50);
        // One seed state + one per step.
        assert_eq!(result.trajectory.len(), 51);
        assert_eq!(result.measurements.len(), 50);
        assert!(result.estimates.is_none());
        // The proportional loop should pull the position below its start.
        let final_pos = result.trajectory.last().unwrap()[0];
        assert!(final_pos.abs() <= 1.0, "final pos {final_pos}");
    }

    #[test]
    fn vector_signal_get_value_is_first_component() {
        let s = VectorSignal::new(&[3.0, 9.0], "y", 2);
        assert_eq!(s.get_value(), 3.0);
        assert_eq!(s.kind, "y");
        assert_eq!(s.tick, 2);
    }
}
