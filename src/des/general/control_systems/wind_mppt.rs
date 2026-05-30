//! Port of `src/des/general/control-systems/wind-mppt.ts` — Maximum Power Point
//! Tracking (MPPT) for a variable-speed wind-energy conversion system (WECS)
//! with a permanent-magnet synchronous generator (PMSG).
//!
//! Reference: K. K. Pandey & A. N. Tiwari, "Maximum Power Point Tracking of Wind
//! Energy Conversion System with Permanent Magnet Synchronous Generator", IJERT
//! Vol. 1 Issue 5, July 2012.
//!
//! Physics: wind power through swept area A = πR² is P_wind = ½ρA V³; captured
//! power via the power coefficient is P_mech = ½ρA C_p(λ, β) V³. The tip-speed
//! ratio is λ = ω_r R / V and C_p follows the Heier coefficient model. The rotor
//! mechanical ODE has a single state ω_r: J dω_r/dt = T_aero − T_gen − B ω_r with
//! T_aero = P_mech / ω_r.
//!
//! Two MPPT controllers: an optimal-torque law T_gen = K_opt ω_r² and a speed-loop
//! PI controller tracking ω* = λ* V / R. DES structure: a self-clocking
//! `WindTurbinePlantStation` integrates the rotor ODE one RK4 step per tick and
//! emits a `TurbineStateToken`; the controllers are zero-backlog / queue-backed
//! transforms. `throw` invariants become `panic!`; all numerics are `f64`. The
//! lazy optimum cache becomes eager computation in the aerodynamics constructor.
#![allow(dead_code)]

use std::any::Any;
use std::rc::Rc;

use super::numerical_solvers::{FixedStepIntegrator, OdeSystem, RungeKutta4Integrator};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::transform_entity::{
    MemoryTransformEntity, OutputChannel, PureTransformEntity, TransformContext, TransformEntity,
    TransformEntityCore, TransformEntityOptions, TransformResult,
};

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// -----------------------------------------------------------------------------
// CHANNELS
// -----------------------------------------------------------------------------

pub struct WindMpptChannels;

impl WindMpptChannels {
    pub const STATE: &'static str = "turbine-state";
    pub const TORQUE: &'static str = "gen-torque";
}

// -----------------------------------------------------------------------------
// TOKENS
// -----------------------------------------------------------------------------

/// Snapshot of the turbine emitted once per discrete tick.
#[derive(Clone, Debug)]
pub struct TurbineStateToken {
    pub tick: usize,
    pub time: f64,
    /// rotor speed ω_r [rad/s]
    pub omega: f64,
    /// wind speed V [m/s] used this step
    pub wind_speed: f64,
    /// tip-speed ratio λ
    pub lambda: f64,
    /// power coefficient C_p
    pub cp: f64,
    /// captured mechanical power P_mech [W]
    pub mech_power: f64,
    /// generator (load) torque applied this step [N·m]
    pub gen_torque: f64,
}

impl TurbineStateToken {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tick: usize,
        time: f64,
        omega: f64,
        wind_speed: f64,
        lambda: f64,
        cp: f64,
        mech_power: f64,
        gen_torque: f64,
    ) -> Self {
        TurbineStateToken {
            tick,
            time,
            omega,
            wind_speed,
            lambda,
            cp,
            mech_power,
            gen_torque,
        }
    }
}

/// Generator electromagnetic torque command produced by the MPPT controller.
#[derive(Clone, Debug)]
pub struct GenTorqueToken {
    pub tick: usize,
    pub torque: f64,
}

impl GenTorqueToken {
    pub fn new(tick: usize, torque: f64) -> Self {
        GenTorqueToken { tick, torque }
    }
}

// -----------------------------------------------------------------------------
// AERODYNAMICS
// -----------------------------------------------------------------------------

pub struct WindTurbineAeroOpts {
    /// air density ρ [kg/m³]. `None` -> 1.225.
    pub air_density: Option<f64>,
    /// blade radius R [m].
    pub blade_radius: f64,
    /// blade pitch angle β [deg]. `None` -> 0.
    pub pitch_deg: Option<f64>,
}

/// Aerodynamic model: C_p(λ, β), captured power, aero torque, and the optimal
/// operating point used by the MPPT controllers. The optimal tip-speed ratio /
/// C_p,max are scanned once in the constructor (the TS class cached them lazily).
#[derive(Clone, Debug)]
pub struct WindTurbineAerodynamics {
    pub air_density: f64,
    pub blade_radius: f64,
    pub pitch_deg: f64,
    opt_lambda: f64,
    opt_cp: f64,
}

impl WindTurbineAerodynamics {
    pub fn new(opts: WindTurbineAeroOpts) -> Self {
        let air_density = opts.air_density.unwrap_or(1.225);
        let blade_radius = opts.blade_radius;
        let pitch_deg = opts.pitch_deg.unwrap_or(0.0);
        let cls = "WindTurbineAerodynamics";
        require(Preconditions::positive(cls, "airDensity", air_density));
        require(Preconditions::positive(cls, "bladeRadius", blade_radius));
        require(Preconditions::non_negative(cls, "pitchDeg", pitch_deg));
        let mut aero = WindTurbineAerodynamics {
            air_density,
            blade_radius,
            pitch_deg,
            opt_lambda: 0.0,
            opt_cp: 0.0,
        };
        aero.compute_optimum();
        aero
    }

    /// Swept area A = πR².
    pub fn swept_area(&self) -> f64 {
        std::f64::consts::PI * self.blade_radius * self.blade_radius
    }

    /// Tip-speed ratio λ = ωR/V (guards V≈0).
    pub fn tip_speed_ratio(&self, omega: f64, wind_speed: f64) -> f64 {
        let v = wind_speed.max(1e-6);
        (omega * self.blade_radius) / v
    }

    /// Heier C_p(λ, β) model. Clamped at 0 (no negative capture).
    pub fn power_coefficient(&self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        let beta = self.pitch_deg;
        let inv_li = 1.0 / (lambda + 0.08 * beta) - 0.035 / (beta * beta * beta + 1.0);
        let cp =
            0.5176 * (116.0 * inv_li - 0.4 * beta - 5.0) * (-21.0 * inv_li).exp() + 0.0068 * lambda;
        if cp > 0.0 {
            cp
        } else {
            0.0
        }
    }

    /// Captured mechanical power P_mech = ½ρA·C_p·V³.
    pub fn mechanical_power(&self, wind_speed: f64, omega: f64) -> f64 {
        let lambda = self.tip_speed_ratio(omega, wind_speed);
        let cp = self.power_coefficient(lambda);
        0.5 * self.air_density * self.swept_area() * cp * wind_speed * wind_speed * wind_speed
    }

    /// Aerodynamic torque T_aero = P_mech / ω (guards ω≈0 with the C_p/λ form).
    pub fn aero_torque(&self, wind_speed: f64, omega: f64) -> f64 {
        let lambda = self.tip_speed_ratio(omega, wind_speed);
        let cp = self.power_coefficient(lambda);
        let power =
            0.5 * self.air_density * self.swept_area() * cp * wind_speed * wind_speed * wind_speed;
        if omega > 1e-3 {
            return power / omega;
        }
        // ω → 0 limit: T = ½ρA·R·(C_p/λ)·V²  (finite startup torque).
        if lambda <= 1e-9 {
            return 0.0;
        }
        0.5 * self.air_density
            * self.swept_area()
            * self.blade_radius
            * (cp / lambda)
            * wind_speed
            * wind_speed
    }

    /// Optimal tip-speed ratio λ* maximising C_p (scanned + cached eagerly).
    pub fn optimal_tip_speed_ratio(&self) -> f64 {
        self.opt_lambda
    }

    /// Maximum power coefficient C_p,max (cached).
    pub fn max_power_coefficient(&self) -> f64 {
        self.opt_cp
    }

    /// Optimal-torque gain K_opt with T_opt = K_opt·ω².
    /// K_opt = ½·ρ·π·R⁵·C_p,max / λ*³.
    pub fn optimal_torque_gain(&self) -> f64 {
        let lambda_star = self.optimal_tip_speed_ratio();
        let cp_max = self.max_power_coefficient();
        let r5 = self.blade_radius.powi(5);
        (0.5 * self.air_density * std::f64::consts::PI * r5 * cp_max) / lambda_star.powi(3)
    }

    /// Power gain K_p with P_opt = K_p·ω³ at λ*.
    pub fn optimal_power_gain(&self) -> f64 {
        let lambda_star = self.optimal_tip_speed_ratio();
        let cp_max = self.max_power_coefficient();
        let r5 = self.blade_radius.powi(5);
        (0.5 * self.air_density * std::f64::consts::PI * r5 * cp_max) / lambda_star.powi(3)
    }

    fn compute_optimum(&mut self) {
        let mut best_lambda = 0.0;
        let mut best_cp = f64::NEG_INFINITY;
        let mut lambda = 0.1;
        while lambda <= 20.0 {
            let cp = self.power_coefficient(lambda);
            if cp > best_cp {
                best_cp = cp;
                best_lambda = lambda;
            }
            lambda += 0.001;
        }
        self.opt_lambda = best_lambda;
        self.opt_cp = best_cp;
    }
}

// -----------------------------------------------------------------------------
// WIND PROFILE + ROTOR DYNAMICS (ODE)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WindProfileSegment {
    /// start time of this segment [s]
    pub from_time: f64,
    /// wind speed during this segment [m/s]
    pub speed: f64,
}

/// Piecewise-constant wind speed schedule, V(t).
#[derive(Clone, Debug)]
pub struct WindProfile {
    segments: Vec<WindProfileSegment>,
}

impl WindProfile {
    pub fn new(segments: &[WindProfileSegment]) -> Self {
        if segments.is_empty() {
            panic!("WindProfile: at least one segment required");
        }
        let mut segs = segments.to_vec();
        segs.sort_by(|a, b| a.from_time.partial_cmp(&b.from_time).unwrap());
        for s in &segs {
            require(Preconditions::non_negative("WindProfile", "speed", s.speed));
        }
        WindProfile { segments: segs }
    }

    /// Wind speed at time t.
    pub fn speed_at(&self, time: f64) -> f64 {
        let mut speed = self.segments[0].speed;
        for s in &self.segments {
            if time + 1e-12 >= s.from_time {
                speed = s.speed;
            } else {
                break;
            }
        }
        speed
    }
}

/// Single-state rotor ODE  J·dω/dt = T_aero(V, ω) − T_gen − B·ω. The wind speed
/// and generator torque are MUTABLE conditions set by the plant station before
/// each numerical step.
pub struct RotorDynamics {
    aero: WindTurbineAerodynamics,
    inertia: f64,
    friction: f64,
    wind_speed: f64,
    gen_torque: f64,
}

impl RotorDynamics {
    pub fn new(aero: WindTurbineAerodynamics, inertia: f64, friction: f64) -> Self {
        require(Preconditions::positive("RotorDynamics", "inertia", inertia));
        require(Preconditions::non_negative(
            "RotorDynamics",
            "friction",
            friction,
        ));
        RotorDynamics {
            aero,
            inertia,
            friction,
            wind_speed: 0.0,
            gen_torque: 0.0,
        }
    }

    /// Set the operating conditions for the upcoming numerical step.
    pub fn set_conditions(&mut self, wind_speed: f64, gen_torque: f64) {
        self.wind_speed = wind_speed;
        self.gen_torque = gen_torque;
    }
}

impl OdeSystem for RotorDynamics {
    fn dimension(&self) -> usize {
        1
    }

    fn derivative(&self, _t: f64, state: &[f64]) -> Vec<f64> {
        let omega = state[0].max(0.0);
        let t_aero = self.aero.aero_torque(self.wind_speed, omega);
        let domega = (t_aero - self.gen_torque - self.friction * omega) / self.inertia;
        vec![domega]
    }
}

// -----------------------------------------------------------------------------
// PLANT STATION (self-clocking ODE integrator)
// -----------------------------------------------------------------------------

pub struct WindTurbinePlantOpts {
    pub aero: WindTurbineAerodynamics,
    pub wind_profile: WindProfile,
    /// rotor inertia J [kg·m²]
    pub inertia: f64,
    /// viscous friction B [N·m·s]
    pub friction: f64,
    /// integration / sample step dt [s]
    pub dt: f64,
    /// number of discrete ticks to simulate
    pub steps: usize,
    /// initial rotor speed ω₀ [rad/s]
    pub initial_omega: f64,
}

/// The turbine PLANT. Self-clocks for `steps` ticks. Each tick it drains the
/// latest generator-torque command, advances the rotor ODE one RK4 step, and
/// emits a `TurbineStateToken`.
pub struct WindTurbinePlantStation {
    core: StationCore,
    dynamics: RotorDynamics,
    integrator: RungeKutta4Integrator,
    wind_profile: WindProfile,
    aero: WindTurbineAerodynamics,
    dt: f64,
    steps: usize,
    omega: f64,
    tick: usize,
    last_gen_torque: f64,
    pub trace: Vec<TurbineStateToken>,
}

impl WindTurbinePlantStation {
    pub fn new(id: &str, opts: WindTurbinePlantOpts) -> Self {
        let cls = "WindTurbinePlantStation";
        require(Preconditions::positive(cls, "dt", opts.dt));
        require(Preconditions::integer_in_range(
            cls,
            "steps",
            opts.steps as f64,
            1.0,
            10_000_000.0,
        ));
        require(Preconditions::non_negative(
            cls,
            "initialOmega",
            opts.initial_omega,
        ));
        let aero = opts.aero;
        let dynamics = RotorDynamics::new(aero.clone(), opts.inertia, opts.friction);
        WindTurbinePlantStation {
            core: StationCore::new(id),
            dynamics,
            integrator: RungeKutta4Integrator::new(),
            wind_profile: opts.wind_profile,
            aero,
            dt: opts.dt,
            steps: opts.steps,
            omega: opts.initial_omega,
            tick: 0,
            last_gen_torque: 0.0,
            trace: Vec::new(),
        }
    }

    pub fn get_omega(&self) -> f64 {
        self.omega
    }

    pub fn get_trace(&self) -> &[TurbineStateToken] {
        &self.trace
    }
}

impl DESStation for WindTurbinePlantStation {
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
        require(Preconditions::positive(
            "WindTurbinePlantStation",
            "dt",
            self.dt,
        ));
        require(Preconditions::finite(
            "WindTurbinePlantStation",
            "initialOmega",
            self.omega,
        ));
    }
    fn run_time_step(&mut self) {
        if self.tick >= self.steps {
            return;
        }
        // 1. Drain torque commands — keep the most recent (last write wins).
        for cmd in self.core.drain::<GenTorqueToken>(WindMpptChannels::TORQUE) {
            self.last_gen_torque = cmd.torque;
        }
        // 2. Advance the rotor ODE one RK4 step under the current conditions.
        let time = self.tick as f64 * self.dt;
        let wind_speed = self.wind_profile.speed_at(time);
        self.dynamics
            .set_conditions(wind_speed, self.last_gen_torque);
        let next = self
            .integrator
            .step(&self.dynamics, time, &[self.omega], self.dt);
        self.omega = next[0].max(0.0);
        // 3. Emit the measured turbine state.
        let lambda = self.aero.tip_speed_ratio(self.omega, wind_speed);
        let cp = self.aero.power_coefficient(lambda);
        let mech_power = self.aero.mechanical_power(wind_speed, self.omega);
        let token = TurbineStateToken::new(
            self.tick,
            (self.tick + 1) as f64 * self.dt,
            self.omega,
            wind_speed,
            lambda,
            cp,
            mech_power,
            self.last_gen_torque,
        );
        self.trace.push(token.clone());
        self.core.emit(Rc::new(token), WindMpptChannels::STATE);
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// MPPT CONTROLLERS
// -----------------------------------------------------------------------------

/// Optimal-torque MPPT: T_gen = K_opt·ω². A memoryless control law, so it is a
/// zero-backlog `PureTransformEntity` from turbine state → torque command.
pub struct OptimalTorqueMpptController {
    tcore: TransformEntityCore<TurbineStateToken, GenTorqueToken>,
    k_opt: f64,
}

impl OptimalTorqueMpptController {
    pub fn new(id: &str, aero: &WindTurbineAerodynamics) -> Self {
        OptimalTorqueMpptController {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![WindMpptChannels::STATE.to_string()],
                    output_channel: OutputChannel::Fixed(WindMpptChannels::TORQUE.to_string()),
                    ..Default::default()
                },
            ),
            k_opt: aero.optimal_torque_gain(),
        }
    }

    pub fn get_optimal_torque_gain(&self) -> f64 {
        self.k_opt
    }
}

impl TransformEntity<TurbineStateToken, GenTorqueToken> for OptimalTorqueMpptController {
    fn tcore(&self) -> &TransformEntityCore<TurbineStateToken, GenTorqueToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<TurbineStateToken, GenTorqueToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<TurbineStateToken, GenTorqueToken> for OptimalTorqueMpptController {
    fn transform(
        &mut self,
        token: &TurbineStateToken,
        _ctx: &mut TransformContext<GenTorqueToken>,
    ) -> TransformResult<GenTorqueToken> {
        let torque = self.k_opt * token.omega * token.omega;
        TransformResult::One(GenTorqueToken::new(token.tick, torque))
    }
}

impl DESStation for OptimalTorqueMpptController {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

pub struct SpeedPiMpptOpts {
    /// proportional gain on speed error [N·m·s/rad]
    pub kp: f64,
    /// integral gain on speed error [N·m/rad]
    pub ki: f64,
    /// sample step dt [s] (for the integrator accumulation)
    pub dt: f64,
    /// torque saturation [N·m]. `None` -> no clamp.
    pub max_torque: Option<f64>,
}

/// Speed-loop MPPT (paper §3): track the optimal speed reference ω* = λ*·V / R
/// and let a PI regulator on the speed error e = ω − ω* set the generator
/// braking torque. When ω < ω* the (negative) command is clamped to zero;
/// the PI integral removes the steady-state offset. Queue-backed
/// `MemoryTransformEntity` whose memory field carries the integral accumulator.
pub struct SpeedPiMpptController {
    tcore: TransformEntityCore<TurbineStateToken, GenTorqueToken>,
    /// integral accumulator (TS `MemoryTransformEntity.previous`).
    previous: f64,
    kp: f64,
    ki: f64,
    dt: f64,
    max_torque: f64,
    lambda_star: f64,
    blade_radius: f64,
}

impl SpeedPiMpptController {
    pub fn new(id: &str, aero: &WindTurbineAerodynamics, opts: SpeedPiMpptOpts) -> Self {
        let cls = "SpeedPiMpptController";
        require(Preconditions::non_negative(cls, "kp", opts.kp));
        require(Preconditions::non_negative(cls, "ki", opts.ki));
        require(Preconditions::positive(cls, "dt", opts.dt));
        SpeedPiMpptController {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![WindMpptChannels::STATE.to_string()],
                    output_channel: OutputChannel::Fixed(WindMpptChannels::TORQUE.to_string()),
                    ..Default::default()
                },
            ),
            previous: 0.0,
            kp: opts.kp,
            ki: opts.ki,
            dt: opts.dt,
            max_torque: opts.max_torque.unwrap_or(f64::INFINITY),
            lambda_star: aero.optimal_tip_speed_ratio(),
            blade_radius: aero.blade_radius,
        }
    }
}

impl TransformEntity<TurbineStateToken, GenTorqueToken> for SpeedPiMpptController {
    fn tcore(&self) -> &TransformEntityCore<TurbineStateToken, GenTorqueToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<TurbineStateToken, GenTorqueToken> {
        &mut self.tcore
    }
}

impl MemoryTransformEntity<TurbineStateToken, GenTorqueToken> for SpeedPiMpptController {
    fn transform_queued(
        &mut self,
        token: &TurbineStateToken,
        _ctx: &mut TransformContext<GenTorqueToken>,
    ) -> TransformResult<GenTorqueToken> {
        let reference_speed = (self.lambda_star * token.wind_speed) / self.blade_radius;
        let error = token.omega - reference_speed;
        let candidate_integral = self.previous + error * self.dt;
        let mut torque = self.kp * error + self.ki * candidate_integral;
        if torque < 0.0 {
            torque = 0.0;
        } else if torque > self.max_torque {
            torque = self.max_torque;
        } else {
            // Anti-windup: only accumulate the integral while unsaturated.
            self.previous = candidate_integral;
        }
        TransformResult::One(GenTorqueToken::new(token.tick, torque))
    }
}

impl DESStation for SpeedPiMpptController {
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

/// Collects the turbine-state trajectory for analysis / validation.
pub struct WindMpptSinkStation {
    core: StationCore,
    pub samples: Vec<Rc<TurbineStateToken>>,
}

impl WindMpptSinkStation {
    pub fn new(id: &str) -> Self {
        WindMpptSinkStation {
            core: StationCore::new(id),
            samples: Vec::new(),
        }
    }

    /// Final captured power [W].
    pub fn final_power(&self) -> f64 {
        self.samples.last().map(|s| s.mech_power).unwrap_or(0.0)
    }

    /// Final tip-speed ratio λ.
    pub fn final_lambda(&self) -> f64 {
        self.samples.last().map(|s| s.lambda).unwrap_or(0.0)
    }

    /// Final power coefficient C_p.
    pub fn final_cp(&self) -> f64 {
        self.samples.last().map(|s| s.cp).unwrap_or(0.0)
    }
}

impl DESStation for WindMpptSinkStation {
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
        self.core.inbox_size(WindMpptChannels::STATE) > 0
    }
    fn run_time_step(&mut self) {
        let drained = self
            .core
            .drain::<TurbineStateToken>(WindMpptChannels::STATE);
        self.samples.extend(drained);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::des::general::des_base::station::StationRef;

    fn aero() -> WindTurbineAerodynamics {
        WindTurbineAerodynamics::new(WindTurbineAeroOpts {
            air_density: None,
            blade_radius: 2.0,
            pitch_deg: None,
        })
    }

    #[test]
    fn aero_optimum_and_coefficient_are_physical() {
        let a = aero();
        assert_eq!(a.power_coefficient(0.0), 0.0);
        let lstar = a.optimal_tip_speed_ratio();
        let cpmax = a.max_power_coefficient();
        assert!((5.0..12.0).contains(&lstar), "lambda* = {lstar}");
        assert!((0.4..0.55).contains(&cpmax), "Cp,max = {cpmax}");
        assert!(a.optimal_torque_gain() > 0.0);
    }

    #[test]
    fn wind_profile_is_piecewise_constant() {
        let wp = WindProfile::new(&[
            WindProfileSegment {
                from_time: 2.0,
                speed: 10.0,
            },
            WindProfileSegment {
                from_time: 0.0,
                speed: 6.0,
            },
        ]);
        assert_eq!(wp.speed_at(0.0), 6.0);
        assert_eq!(wp.speed_at(1.9), 6.0);
        assert_eq!(wp.speed_at(2.0), 10.0);
    }

    #[test]
    fn plant_rotor_accelerates_in_wind() {
        let mut plant = WindTurbinePlantStation::new(
            "turbine",
            WindTurbinePlantOpts {
                aero: aero(),
                wind_profile: WindProfile::new(&[WindProfileSegment {
                    from_time: 0.0,
                    speed: 12.0,
                }]),
                inertia: 5.0,
                friction: 0.1,
                dt: 0.01,
                steps: 300,
                initial_omega: 1.0,
            },
        );
        for _ in 0..300 {
            plant.run_time_step();
        }
        assert_eq!(plant.get_trace().len(), 300);
        assert!(plant.get_omega() > 1.0, "omega = {}", plant.get_omega());
    }

    struct TorqueSink {
        core: StationCore,
        got: Vec<Rc<GenTorqueToken>>,
    }
    impl DESStation for TorqueSink {
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
            let d = self.core.drain::<GenTorqueToken>(WindMpptChannels::TORQUE);
            self.got.extend(d);
        }
    }

    #[test]
    fn optimal_torque_controller_applies_k_omega_squared() {
        let a = aero();
        let k = a.optimal_torque_gain();
        let sink = Rc::new(RefCell::new(TorqueSink {
            core: StationCore::new("t-sink"),
            got: Vec::new(),
        }));
        let mut ctrl = OptimalTorqueMpptController::new("mppt", &a);
        ctrl.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            WindMpptChannels::TORQUE,
            WindMpptChannels::TORQUE,
        );
        ctrl.take(
            Rc::new(TurbineStateToken::new(
                0, 0.0, 5.0, 12.0, 0.0, 0.0, 0.0, 0.0,
            )),
            WindMpptChannels::STATE,
        );
        sink.borrow_mut().run_time_step();
        let got = &sink.borrow().got;
        assert_eq!(got.len(), 1);
        assert!(
            (got[0].torque - k * 25.0).abs() < 1e-9,
            "torque {}",
            got[0].torque
        );
    }
}
