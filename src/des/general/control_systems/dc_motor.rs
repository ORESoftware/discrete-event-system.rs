//! Port of `src/des/general/control-systems/dc-motor.ts` — separately-excited /
//! permanent-magnet DC motor modelled as a two-state ODE system with explicit
//! back-EMF coupling, driven and regulated inside the lightweight DES graph.
//!
//! State x = [i, ω] (armature current, rotor speed). The electrical equation is
//! L di/dt = V − R i − E with back-EMF E = K_e ω; the mechanical equation is
//! J dω/dt = K_t i − B ω − T_L. In state-space form (input u = V, output y = ω)
//! the system matrices are exposed by `state_space` so the same plant can feed
//! the observability / controllability evaluator.
//!
//! DES structure: a self-clocking `DcMotorPlantStation` integrates the ODE one
//! RK4 step per tick and emits a `MotorStateToken` carrying the back-EMF; the
//! `SpeedPiVoltageController` is a queue-backed `MemoryTransformEntity` whose
//! integral accumulator lives in the memory field; a `DcMotorSinkStation`
//! collects the trajectory. `throw` invariants become `panic!`; all numerics are
//! `f64`.
#![allow(dead_code)]

use std::any::Any;
use std::rc::Rc;

use super::linear_algebra::Matrix;
use super::numerical_solvers::{FixedStepIntegrator, OdeSystem, RungeKutta4Integrator};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::transform_entity::{
    MemoryTransformEntity, OutputChannel, TransformContext, TransformEntity, TransformEntityCore,
    TransformEntityOptions, TransformResult,
};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// -----------------------------------------------------------------------------
// CHANNELS
// -----------------------------------------------------------------------------

pub struct DcMotorChannels;

impl DcMotorChannels {
    pub const STATE: &'static str = "motor-state";
    pub const VOLTAGE: &'static str = "armature-voltage";
}

// -----------------------------------------------------------------------------
// TOKENS
// -----------------------------------------------------------------------------

/// Measured motor state emitted once per discrete tick.
#[derive(Clone, Debug)]
pub struct MotorStateToken {
    pub tick: usize,
    pub time: f64,
    /// armature current i [A]
    pub current: f64,
    /// rotor speed ω [rad/s]
    pub omega: f64,
    /// back-EMF E = K_e·ω [V]
    pub back_emf: f64,
    /// electromagnetic torque K_t·i [N·m]
    pub torque: f64,
    /// applied armature voltage this step [V]
    pub voltage: f64,
    /// load torque this step [N·m]
    pub load_torque: f64,
}

impl MotorStateToken {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tick: usize,
        time: f64,
        current: f64,
        omega: f64,
        back_emf: f64,
        torque: f64,
        voltage: f64,
        load_torque: f64,
    ) -> Self {
        MotorStateToken { tick, time, current, omega, back_emf, torque, voltage, load_torque }
    }
}

/// Armature voltage command produced by the controller.
#[derive(Clone, Debug)]
pub struct VoltageToken {
    pub tick: usize,
    pub voltage: f64,
}

impl VoltageToken {
    pub fn new(tick: usize, voltage: f64) -> Self {
        VoltageToken { tick, voltage }
    }
}

// -----------------------------------------------------------------------------
// MOTOR PARAMETERS + DYNAMICS (ODE)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DcMotorParams {
    /// armature resistance R [Ω]
    pub resistance: f64,
    /// armature inductance L [H]
    pub inductance: f64,
    /// back-EMF constant K_e [V·s/rad]
    pub back_emf_constant: f64,
    /// torque constant K_t [N·m/A]
    pub torque_constant: f64,
    /// rotor inertia J [kg·m²]
    pub inertia: f64,
    /// viscous friction B [N·m·s]
    pub friction: f64,
}

/// Continuous-time state-space matrices (input u = V, output y = ω).
#[derive(Clone, Debug)]
pub struct StateSpaceMatrices {
    pub a: Matrix,
    pub b: Matrix,
    pub c: Matrix,
    pub d: Matrix,
}

/// Two-state DC-motor ODE. The applied voltage and load torque are MUTABLE
/// conditions set by the plant station before each numerical step.
pub struct DcMotorDynamics {
    pub params: DcMotorParams,
    voltage: f64,
    load_torque: f64,
}

impl DcMotorDynamics {
    pub fn new(params: DcMotorParams) -> Self {
        let cls = "DcMotorDynamics";
        require(Preconditions::positive(cls, "resistance", params.resistance));
        require(Preconditions::positive(cls, "inductance", params.inductance));
        require(Preconditions::positive(cls, "backEmfConstant", params.back_emf_constant));
        require(Preconditions::positive(cls, "torqueConstant", params.torque_constant));
        require(Preconditions::positive(cls, "inertia", params.inertia));
        require(Preconditions::non_negative(cls, "friction", params.friction));
        DcMotorDynamics { params, voltage: 0.0, load_torque: 0.0 }
    }

    /// Set the inputs for the upcoming numerical step.
    pub fn set_inputs(&mut self, voltage: f64, load_torque: f64) {
        self.voltage = voltage;
        self.load_torque = load_torque;
    }

    /// Back-EMF E = K_e·ω.
    pub fn back_emf(&self, omega: f64) -> f64 {
        self.params.back_emf_constant * omega
    }

    /// Electromagnetic torque T_e = K_t·i.
    pub fn electromagnetic_torque(&self, current: f64) -> f64 {
        self.params.torque_constant * current
    }

    /// Continuous-time state-space matrices (input u = V, output y = ω).
    pub fn state_space(&self) -> StateSpaceMatrices {
        let p = &self.params;
        StateSpaceMatrices {
            a: vec![
                vec![-p.resistance / p.inductance, -p.back_emf_constant / p.inductance],
                vec![p.torque_constant / p.inertia, -p.friction / p.inertia],
            ],
            b: vec![vec![1.0 / p.inductance], vec![0.0]],
            c: vec![vec![0.0, 1.0]],
            d: vec![vec![0.0]],
        }
    }
}

impl OdeSystem for DcMotorDynamics {
    fn dimension(&self) -> usize {
        2
    }

    fn derivative(&self, _t: f64, state: &[f64]) -> Vec<f64> {
        let i = state[0];
        let omega = state[1];
        let p = &self.params;
        let e = self.back_emf(omega);
        let di = (self.voltage - p.resistance * i - e) / p.inductance;
        let domega = (p.torque_constant * i - p.friction * omega - self.load_torque) / p.inertia;
        vec![di, domega]
    }
}

// -----------------------------------------------------------------------------
// LOAD-TORQUE PROFILE
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LoadSegment {
    pub from_time: f64,
    pub torque: f64,
}

/// Piecewise-constant load-torque schedule T_L(t).
#[derive(Clone, Debug)]
pub struct LoadProfile {
    segments: Vec<LoadSegment>,
}

impl LoadProfile {
    pub fn new(segments: &[LoadSegment]) -> Self {
        let mut segs: Vec<LoadSegment> = if segments.is_empty() {
            vec![LoadSegment { from_time: 0.0, torque: 0.0 }]
        } else {
            segments.to_vec()
        };
        segs.sort_by(|a, b| a.from_time.partial_cmp(&b.from_time).unwrap());
        LoadProfile { segments: segs }
    }

    pub fn torque_at(&self, time: f64) -> f64 {
        let mut t = self.segments[0].torque;
        for s in &self.segments {
            if time + 1e-12 >= s.from_time {
                t = s.torque;
            } else {
                break;
            }
        }
        t
    }
}

// -----------------------------------------------------------------------------
// PLANT STATION (self-clocking ODE integrator)
// -----------------------------------------------------------------------------

pub struct DcMotorPlantOpts {
    pub params: DcMotorParams,
    /// integration / sample step dt [s]
    pub dt: f64,
    /// number of discrete ticks to simulate
    pub steps: usize,
    /// initial state [i₀, ω₀]. `None` -> [0, 0].
    pub initial_state: Option<Vec<f64>>,
    /// load-torque schedule. `None` -> zero load.
    pub load: Option<LoadProfile>,
}

/// The DC-motor PLANT. Self-clocks for `steps` ticks; each tick it drains the
/// latest armature voltage, advances the 2-state ODE one RK4 step, and emits a
/// `MotorStateToken` carrying the back-EMF.
pub struct DcMotorPlantStation {
    core: StationCore,
    dynamics: DcMotorDynamics,
    integrator: RungeKutta4Integrator,
    dt: f64,
    steps: usize,
    load: LoadProfile,
    state: Vec<f64>,
    tick: usize,
    last_voltage: f64,
    pub trace: Vec<MotorStateToken>,
}

impl DcMotorPlantStation {
    pub fn new(id: &str, opts: DcMotorPlantOpts) -> Self {
        require(Preconditions::positive("DcMotorPlantStation", "dt", opts.dt));
        require(Preconditions::integer_in_range(
            "DcMotorPlantStation",
            "steps",
            opts.steps as f64,
            1.0,
            10_000_000.0,
        ));
        let dynamics = DcMotorDynamics::new(opts.params);
        let load = opts
            .load
            .unwrap_or_else(|| LoadProfile::new(&[LoadSegment { from_time: 0.0, torque: 0.0 }]));
        let state = opts.initial_state.unwrap_or_else(|| vec![0.0, 0.0]);
        require(Preconditions::length_eq("DcMotorPlantStation", "initialState", &state, 2));
        require(Preconditions::all_finite("DcMotorPlantStation", "initialState", &state));
        DcMotorPlantStation {
            core: StationCore::new(id),
            dynamics,
            integrator: RungeKutta4Integrator::new(),
            dt: opts.dt,
            steps: opts.steps,
            load,
            state,
            tick: 0,
            last_voltage: 0.0,
            trace: Vec::new(),
        }
    }

    /// Override the constant open-loop drive voltage (used when no controller is
    /// wired — the motor is driven by a fixed armature voltage).
    pub fn set_open_loop_voltage(&mut self, voltage: f64) {
        self.last_voltage = voltage;
    }

    pub fn get_state(&self) -> Vec<f64> {
        self.state.clone()
    }

    pub fn get_dynamics(&self) -> &DcMotorDynamics {
        &self.dynamics
    }

    pub fn get_trace(&self) -> &[MotorStateToken] {
        &self.trace
    }
}

impl DESStation for DcMotorPlantStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.tick < self.steps
    }
    fn assert_preconditions(&mut self) {
        require(Preconditions::positive("DcMotorPlantStation", "dt", self.dt));
        require(Preconditions::all_finite("DcMotorPlantStation", "state", &self.state));
    }
    fn run_time_step(&mut self) {
        if self.tick >= self.steps {
            return;
        }
        // 1. Drain voltage commands — last write wins.
        for cmd in self.core.drain::<VoltageToken>(DcMotorChannels::VOLTAGE) {
            self.last_voltage = cmd.voltage;
        }
        // 2. Advance the 2-state ODE one RK4 step.
        let time = self.tick as f64 * self.dt;
        let load_torque = self.load.torque_at(time);
        self.dynamics.set_inputs(self.last_voltage, load_torque);
        self.state = self.integrator.step(&self.dynamics, time, &self.state, self.dt);
        // 3. Emit the measured state (back-EMF included).
        let current = self.state[0];
        let omega = self.state[1];
        let token = MotorStateToken::new(
            self.tick,
            (self.tick + 1) as f64 * self.dt,
            current,
            omega,
            self.dynamics.back_emf(omega),
            self.dynamics.electromagnetic_torque(current),
            self.last_voltage,
            load_torque,
        );
        self.trace.push(token.clone());
        self.core.emit(Rc::new(token), DcMotorChannels::STATE);
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// SPEED PI CONTROLLER
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SpeedReferenceSegment {
    pub from_time: f64,
    pub speed: f64,
}

pub struct SpeedPiVoltageOpts {
    /// proportional gain [V·s/rad]
    pub kp: f64,
    /// integral gain [V/rad]
    pub ki: f64,
    /// sample step dt [s]
    pub dt: f64,
    /// reference-speed schedule ω*(t) [rad/s]
    pub reference: Vec<SpeedReferenceSegment>,
    /// armature-voltage saturation magnitude [V]. `None` -> ±∞.
    pub max_voltage: Option<f64>,
}

/// PI speed controller: V = K_p·e + K_i·∫e with e = ω* − ω. The reference
/// schedule lives inside the controller; the integral accumulator is the
/// `MemoryTransformEntity` memory field (with anti-windup clamping).
pub struct SpeedPiVoltageController {
    tcore: TransformEntityCore<MotorStateToken, VoltageToken>,
    /// integral accumulator (TS `MemoryTransformEntity.previous`).
    previous: f64,
    kp: f64,
    ki: f64,
    dt: f64,
    max_voltage: f64,
    reference: Vec<SpeedReferenceSegment>,
}

impl SpeedPiVoltageController {
    pub fn new(id: &str, opts: SpeedPiVoltageOpts) -> Self {
        let cls = "SpeedPiVoltageController";
        require(Preconditions::non_negative(cls, "kp", opts.kp));
        require(Preconditions::non_negative(cls, "ki", opts.ki));
        require(Preconditions::positive(cls, "dt", opts.dt));
        require(Preconditions::non_empty(cls, "reference", &opts.reference));
        let mut reference = opts.reference;
        reference.sort_by(|a, b| a.from_time.partial_cmp(&b.from_time).unwrap());
        SpeedPiVoltageController {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![DcMotorChannels::STATE.to_string()],
                    output_channel: OutputChannel::Fixed(DcMotorChannels::VOLTAGE.to_string()),
                    ..Default::default()
                },
            ),
            previous: 0.0,
            kp: opts.kp,
            ki: opts.ki,
            dt: opts.dt,
            max_voltage: opts.max_voltage.unwrap_or(f64::INFINITY),
            reference,
        }
    }

    /// Reference speed ω*(t) from the schedule.
    pub fn reference_at(&self, time: f64) -> f64 {
        let mut r = self.reference[0].speed;
        for s in &self.reference {
            if time + 1e-12 >= s.from_time {
                r = s.speed;
            } else {
                break;
            }
        }
        r
    }
}

impl TransformEntity<MotorStateToken, VoltageToken> for SpeedPiVoltageController {
    fn tcore(&self) -> &TransformEntityCore<MotorStateToken, VoltageToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<MotorStateToken, VoltageToken> {
        &mut self.tcore
    }
}

impl MemoryTransformEntity<MotorStateToken, VoltageToken> for SpeedPiVoltageController {
    fn transform_queued(
        &mut self,
        token: &MotorStateToken,
        _ctx: &mut TransformContext<VoltageToken>,
    ) -> TransformResult<VoltageToken> {
        let reference = self.reference_at(token.time);
        let error = reference - token.omega;
        let candidate_integral = self.previous + error * self.dt;
        let mut voltage = self.kp * error + self.ki * candidate_integral;
        if voltage > self.max_voltage {
            voltage = self.max_voltage;
        } else if voltage < -self.max_voltage {
            voltage = -self.max_voltage;
        } else {
            self.previous = candidate_integral;
        }
        TransformResult::One(VoltageToken::new(token.tick, voltage))
    }
}

impl DESStation for SpeedPiVoltageController {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.run_queued();
    }
    fn has_work(&self) -> bool {
        self.tcore().has_queued_input()
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

// -----------------------------------------------------------------------------
// SINK
// -----------------------------------------------------------------------------

/// Collects the motor-state trajectory for analysis / validation.
pub struct DcMotorSinkStation {
    core: StationCore,
    pub samples: Vec<Rc<MotorStateToken>>,
}

impl DcMotorSinkStation {
    pub fn new(id: &str) -> Self {
        DcMotorSinkStation { core: StationCore::new(id), samples: Vec::new() }
    }

    pub fn final_state(&self) -> Option<&MotorStateToken> {
        self.samples.last().map(|r| &**r)
    }

    pub fn final_omega(&self) -> f64 {
        self.final_state().map(|s| s.omega).unwrap_or(0.0)
    }

    pub fn final_back_emf(&self) -> f64 {
        self.final_state().map(|s| s.back_emf).unwrap_or(0.0)
    }
}

impl DESStation for DcMotorSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        self.core.inbox_size(DcMotorChannels::STATE) > 0
    }
    fn run_time_step(&mut self) {
        let drained = self.core.drain::<MotorStateToken>(DcMotorChannels::STATE);
        self.samples.extend(drained);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::des::general::des_base::station::StationRef;

    fn params() -> DcMotorParams {
        DcMotorParams {
            resistance: 1.0,
            inductance: 0.5,
            back_emf_constant: 0.01,
            torque_constant: 0.01,
            inertia: 0.01,
            friction: 0.1,
        }
    }

    #[test]
    fn dynamics_derivative_and_state_space() {
        let mut dyn_ = DcMotorDynamics::new(params());
        dyn_.set_inputs(12.0, 0.0);
        let d = dyn_.derivative(0.0, &[0.0, 0.0]);
        // di = V/L, dω = 0 at the origin with no load.
        assert!((d[0] - 12.0 / 0.5).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);

        let ss = dyn_.state_space();
        assert!((ss.a[0][0] - (-1.0 / 0.5)).abs() < 1e-12);
        assert!((ss.b[0][0] - 1.0 / 0.5).abs() < 1e-12);
        assert_eq!(ss.c, vec![vec![0.0, 1.0]]);
    }

    #[test]
    fn load_profile_is_piecewise_constant() {
        let lp = LoadProfile::new(&[
            LoadSegment { from_time: 1.0, torque: 5.0 },
            LoadSegment { from_time: 0.0, torque: 1.0 },
        ]);
        assert_eq!(lp.torque_at(0.0), 1.0);
        assert_eq!(lp.torque_at(0.5), 1.0);
        assert_eq!(lp.torque_at(1.0), 5.0);
        assert_eq!(lp.torque_at(2.0), 5.0);
    }

    #[test]
    fn plant_spins_up_under_open_loop_voltage() {
        let mut plant = DcMotorPlantStation::new(
            "motor",
            DcMotorPlantOpts {
                params: params(),
                dt: 0.001,
                steps: 200,
                initial_state: None,
                load: None,
            },
        );
        plant.set_open_loop_voltage(12.0);
        for _ in 0..200 {
            plant.run_time_step();
        }
        assert_eq!(plant.get_trace().len(), 200);
        let final_omega = plant.get_state()[1];
        assert!(final_omega > 0.0, "omega = {final_omega}");
        // current should be positive while accelerating from rest.
        assert!(plant.get_trace()[0].current > 0.0);
    }

    struct VoltageSink {
        core: StationCore,
        got: Vec<Rc<VoltageToken>>,
    }
    impl DESStation for VoltageSink {
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
            let d = self.core.drain::<VoltageToken>(DcMotorChannels::VOLTAGE);
            self.got.extend(d);
        }
    }

    #[test]
    fn pi_controller_drives_voltage_toward_reference() {
        let sink = Rc::new(RefCell::new(VoltageSink {
            core: StationCore::new("v-sink"),
            got: Vec::new(),
        }));
        let mut ctrl = SpeedPiVoltageController::new(
            "pi",
            SpeedPiVoltageOpts {
                kp: 1.0,
                ki: 0.0,
                dt: 0.001,
                reference: vec![SpeedReferenceSegment { from_time: 0.0, speed: 100.0 }],
                max_voltage: None,
            },
        );
        ctrl.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            DcMotorChannels::VOLTAGE,
            DcMotorChannels::VOLTAGE,
        );
        // Measured omega = 0 -> error = 100 -> V = kp*100 = 100.
        ctrl.take(
            Rc::new(MotorStateToken::new(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DcMotorChannels::STATE,
        );
        ctrl.run_time_step();
        sink.borrow_mut().run_time_step();
        let got = &sink.borrow().got;
        assert_eq!(got.len(), 1);
        assert!((got[0].voltage - 100.0).abs() < 1e-9, "voltage {}", got[0].voltage);
    }
}
