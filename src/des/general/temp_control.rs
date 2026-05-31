//! Port of `src/des/general/temp-control.ts` — INDOOR TEMPERATURE CONTROL as a
//! discrete-event system, with four interchangeable controllers
//! (bang-bang / PID / fuzzy / MDP-MPC) compared on the same physical house and
//! the same 24-hour outdoor temperature trajectory.
//!
//! Keep indoor temperature within ±2°F of a target while minimising HVAC
//! energy. The default run is heating-only, while mixed-season runs can opt into
//! bidirectional heat-pump behavior by setting `Q_min < 0`. The outside
//! temperature follows a known diurnal pattern plus noise;
//! a noisy forecast of the next H hours feeds the MPC-style controller (that is
//! where the partial observability lives). Each tick is one minute of simulated
//! time. The house integrates a first-order thermal ODE
//! dT_in/dt = (T_out − T_in) / tau + Q · G.
//!
//! ## Rust shape
//!
//! `ControllerSpec` (a discriminated union) → the [`ControllerSpec`] enum, and
//! `makeTempController` becomes a constructor that stores the spec. The fuzzy
//! linguistic terms / output levels (string unions) → the [`Term`] / [`OutLevel`]
//! enums. The RNG (the TS re-declared `mulberry32`) is injected as the shared
//! [`SeededRandom`] capability rather than a second copy.
//!
//! FLAGGED faithful simplification (no unported dep involved): the four TS leaf
//! classes `BangBang/PID/Fuzzy/MdpMpcController extends TempControllerBase`
//! differ ONLY by the `ControllerSpec` they pass to `controllerStep`. Per the
//! migration header's sanctioned "enum of controllers" option they collapse into
//! the single [`TempControllerBase`] struct carrying that spec; the four named
//! constructors are preserved as functions.

use std::any::Any;

use crate::des::general::des_base::controller::{ControllerCore, ControllerStation};
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::validation::intrinsic_check;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// PHYSICAL HOUSE MODEL
// -----------------------------------------------------------------------------

/// Physical parameters of the house thermal model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HouseParams {
    /// Thermal time constant tau (hours). dT/dt has a (T_out − T_in)/tau term.
    pub tau: f64,
    /// Heater gain G (°F per kW per hour). dT/dt has a Q · G term.
    pub g: f64,
    /// Maximum cooling power (kW), encoded as a negative command. Defaults to
    /// zero for the original heating-only scenario.
    pub q_min: f64,
    /// Maximum heater power Q_max (kW).
    pub q_max: f64,
    /// Initial indoor temperature (°F).
    pub t_init: f64,
}

pub const DEFAULT_HOUSE: HouseParams = HouseParams {
    tau: 12.0,
    g: 1.0,
    q_min: 0.0,
    q_max: 5.0,
    t_init: 70.0,
};

/// Partial house override (TS `Partial<HouseParams>`).
#[derive(Clone, Copy, Debug, Default)]
pub struct HouseParamsPartial {
    pub tau: Option<f64>,
    pub g: Option<f64>,
    pub q_min: Option<f64>,
    pub q_max: Option<f64>,
    pub t_init: Option<f64>,
}

impl HouseParams {
    fn merged(self, p: &HouseParamsPartial) -> HouseParams {
        HouseParams {
            tau: p.tau.unwrap_or(self.tau),
            g: p.g.unwrap_or(self.g),
            q_min: p.q_min.unwrap_or(self.q_min),
            q_max: p.q_max.unwrap_or(self.q_max),
            t_init: p.t_init.unwrap_or(self.t_init),
        }
    }
}

/// Diurnal outdoor-temperature pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutdoorPattern {
    /// Mean outdoor temperature over a day (°F).
    pub mean: f64,
    /// Diurnal swing amplitude (°F).
    pub amp: f64,
    /// Phase shift in hours (peak temperature occurs at hour `phase + 6`).
    pub phase: f64,
    /// Standard deviation of additive noise (°F).
    pub noise_std: f64,
}

// Default: cold winter day. Peak at 3 PM (phase = 9), minimum at 3 AM.
pub const DEFAULT_OUTDOOR: OutdoorPattern = OutdoorPattern {
    mean: 25.0,
    amp: 15.0,
    phase: 9.0,
    noise_std: 1.5,
};

/// Partial outdoor override (TS `Partial<OutdoorPattern>`).
#[derive(Clone, Copy, Debug, Default)]
pub struct OutdoorPatternPartial {
    pub mean: Option<f64>,
    pub amp: Option<f64>,
    pub phase: Option<f64>,
    pub noise_std: Option<f64>,
}

impl OutdoorPattern {
    fn merged(self, p: &OutdoorPatternPartial) -> OutdoorPattern {
        OutdoorPattern {
            mean: p.mean.unwrap_or(self.mean),
            amp: p.amp.unwrap_or(self.amp),
            phase: p.phase.unwrap_or(self.phase),
            noise_std: p.noise_std.unwrap_or(self.noise_std),
        }
    }
}

/// True outside temperature at simulation time `t_hours`, with optional rng noise.
pub fn true_outdoor_temp(
    t_hours: f64,
    pattern: &OutdoorPattern,
    rng: Option<&mut dyn RandomSource>,
) -> f64 {
    let periodic = pattern.mean
        + pattern.amp * (2.0 * std::f64::consts::PI * (t_hours - pattern.phase) / 24.0).sin();
    match rng {
        None => periodic,
        Some(_) if pattern.noise_std == 0.0 => periodic,
        Some(r) => {
            // Approx. Gaussian via sum-of-uniforms (mean 0, std ≈ 0.577).
            let u = r.next_float() + r.next_float() + r.next_float() + r.next_float() - 2.0;
            periodic + pattern.noise_std * (u / 0.577)
        }
    }
}

/// Forward-Euler step of the first-order thermal ODE
/// dT_in/dt = (T_out − T_in) / tau + Q · G.
pub fn house_step(t_in: f64, t_out: f64, q: f64, dt_h: f64, h: &HouseParams) -> f64 {
    let d_t = (t_out - t_in) / h.tau + q * h.g;
    t_in + d_t * dt_h
}

// -----------------------------------------------------------------------------
// PRNG (mulberry32) — reproducible noise
// -----------------------------------------------------------------------------

/// The TS file re-declared `mulberry32`; here it is the shared [`SeededRandom`]
/// capability (mulberry32) — not a second copy.
pub fn mulberry32(seed: u32) -> SeededRandom {
    SeededRandom::new(seed)
}

// -----------------------------------------------------------------------------
// CONTROLLERS
// -----------------------------------------------------------------------------

/// `type ControllerSpec` discriminated union → enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ControllerSpec {
    BangBang,
    Pid {
        kp: f64,
        ki: f64,
        kd: f64,
    },
    Fuzzy,
    MdpMpc {
        horizon_h: f64,
        n_levels: usize,
        comfort_penalty: f64,
        cost_per_kwh: f64,
        /// Soft tracking weight inside the band (default 1.0).
        track_weight: Option<f64>,
    },
}

/// Persistent state owned by a controller across ticks.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControllerState {
    /// PID integral accumulator and fuzzy-PI integrated output.
    pub integral: Option<f64>,
    /// Previous error (PID derivative term + fuzzy error-rate).
    pub prev_error: Option<f64>,
    /// Low-pass-filtered error derivative (PID, °F/h).
    pub d_err_filt: Option<f64>,
    /// Fuzzy-PI integrated Q command (kW), held across ticks.
    pub fuzzy_q: Option<f64>,
}

/// What every controller observes on each tick (TS `interface TempObs`). This is
/// also the `ctx` consumed by [`controller_step`] (the TS `ctx` had identical
/// fields).
#[derive(Clone, Debug)]
pub struct TempObs {
    pub t_target: f64,
    pub t_in_meas: f64,
    pub forecast: Vec<f64>,
    pub dt_h: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub house: HouseParams,
}

/// Compute heater command Q ∈ [0, Q_max] given the current observation context.
pub fn controller_step(spec: &ControllerSpec, state: &mut ControllerState, ctx: &TempObs) -> f64 {
    let e = ctx.t_target - ctx.t_in_meas;
    match *spec {
        ControllerSpec::BangBang => {
            if e > 0.0 {
                ctx.q_max
            } else if e < 0.0 {
                ctx.q_min
            } else {
                0.0
            }
        }
        ControllerSpec::Pid { kp, ki, kd } => {
            // Conditional integration anti-windup + first-order low-pass filter
            // on the derivative term. Filter time constant tau_d = 5 ticks.
            let i_prev = state.integral.unwrap_or(0.0);
            let de_raw = (e - state.prev_error.unwrap_or(e)) / ctx.dt_h;
            let alpha = 1.0 / 6.0;
            let d_err = (1.0 - alpha) * state.d_err_filt.unwrap_or(0.0) + alpha * de_raw;
            state.d_err_filt = Some(d_err);
            let u_pre = kp * e + ki * i_prev + kd * d_err;
            let sat_high = u_pre >= ctx.q_max && e > 0.0;
            let sat_low = u_pre <= ctx.q_min && e < 0.0;
            state.integral = Some(if sat_high || sat_low {
                i_prev
            } else {
                i_prev + e * ctx.dt_h
            });
            state.prev_error = Some(e);
            let u = kp * e + ki * state.integral.unwrap() + kd * d_err;
            u.clamp(ctx.q_min, ctx.q_max)
        }
        ControllerSpec::Fuzzy => {
            // Fuzzy-PI: the rule base outputs Δ-Q normalised to [-1,+1] (units of
            // Q_max per hour); integrate it over time for an offset-free command.
            let de_dt = (e - state.prev_error.unwrap_or(e)) / ctx.dt_h;
            state.prev_error = Some(e);
            let dq_norm = fuzzy_delta_controller(e, de_dt);
            let actuator_scale = if dq_norm >= 0.0 {
                ctx.q_max
            } else if ctx.q_min < 0.0 {
                -ctx.q_min
            } else {
                ctx.q_max
            };
            let dq = dq_norm * actuator_scale * ctx.dt_h * 6.0; // 6/h gain factor — empirical
            let q_prev = state.fuzzy_q.unwrap_or(0.0);
            let q = (q_prev + dq).clamp(ctx.q_min, ctx.q_max);
            state.fuzzy_q = Some(q);
            q
        }
        ControllerSpec::MdpMpc {
            horizon_h,
            n_levels,
            comfort_penalty,
            cost_per_kwh,
            track_weight,
        } => mdp_mpc_controller_with_bounds(
            ctx.t_in_meas,
            &ctx.forecast,
            horizon_h,
            n_levels,
            ctx.t_target,
            ctx.dt_h,
            ctx.q_min,
            ctx.q_max,
            &ctx.house,
            comfort_penalty,
            cost_per_kwh,
            track_weight.unwrap_or(1.0),
        ),
    }
}

// ── Fuzzy logic controller (Fuzzy-PI form) ───────────────────────────────────
//
// Inputs:  e = T_target − T_in_meas (°F; positive = room is cold)
//          de/dt = derivative of e  (°F/h; positive = still cooling)
// Output:  Δ-Q ∈ [-1, +1] (normalised), integrated externally.
//
// Linguistic terms on each input: NL, NS, Z, PS, PL. Triangular membership
// functions evenly spaced; centre-of-gravity defuzzification.
//
//      e\de    NL     NS    Z     PS    PL
//      NL      ND     ND    NS    NS    Z
//      NS      ND     NS    NS    Z     PS
//      Z       NS     NS    Z     PS    PS
//      PS      NS     Z     PS    PS    PD
//      PL      Z      PS    PS    PD    PD
//   ND/PD = ±1.0 (drive), NS/PS = ±0.5 (small), Z = 0.0

fn tri(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if x <= a || x >= c {
        return 0.0;
    }
    if x == b {
        return 1.0;
    }
    if x < b {
        (x - a) / (b - a)
    } else {
        (c - x) / (c - b)
    }
}

const E_RANGE: f64 = 6.0; // °F — saturate above this
const DE_RANGE: f64 = 4.0; // °F/h — saturate above this

/// Membership degrees for the error input, indexed by [`Term`] order
/// (`[NL, NS, Z, PS, PL]`).
fn mu_e(x: f64) -> [f64; 5] {
    let e = x.clamp(-E_RANGE, E_RANGE);
    [
        tri(e, -E_RANGE * 1.5, -E_RANGE, -E_RANGE / 2.0),
        tri(e, -E_RANGE, -E_RANGE / 2.0, 0.0),
        tri(e, -E_RANGE / 2.0, 0.0, E_RANGE / 2.0),
        tri(e, 0.0, E_RANGE / 2.0, E_RANGE),
        tri(e, E_RANGE / 2.0, E_RANGE, E_RANGE * 1.5),
    ]
}

/// Membership degrees for the error-rate input, indexed by [`Term`] order.
fn mu_de(x: f64) -> [f64; 5] {
    let xx = x.clamp(-DE_RANGE, DE_RANGE);
    [
        tri(xx, -DE_RANGE * 1.5, -DE_RANGE, -DE_RANGE / 2.0),
        tri(xx, -DE_RANGE, -DE_RANGE / 2.0, 0.0),
        tri(xx, -DE_RANGE / 2.0, 0.0, DE_RANGE / 2.0),
        tri(xx, 0.0, DE_RANGE / 2.0, DE_RANGE),
        tri(xx, DE_RANGE / 2.0, DE_RANGE, DE_RANGE * 1.5),
    ]
}

/// Fuzzy linguistic term (negative-large … positive-large).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    NL,
    NS,
    Z,
    PS,
    PL,
}

const TERMS: [Term; 5] = [Term::NL, Term::NS, Term::Z, Term::PS, Term::PL];

impl Term {
    fn index(self) -> usize {
        match self {
            Term::NL => 0,
            Term::NS => 1,
            Term::Z => 2,
            Term::PS => 3,
            Term::PL => 4,
        }
    }
}

/// Fuzzy output level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutLevel {
    ND,
    NS,
    Z,
    PS,
    PD,
}

impl OutLevel {
    fn value(self) -> f64 {
        match self {
            OutLevel::ND => -1.0,
            OutLevel::NS => -0.5,
            OutLevel::Z => 0.0,
            OutLevel::PS => 0.5,
            OutLevel::PD => 1.0,
        }
    }
}

/// The 5×5 rule base: `RULES[e][de]` (TS `RULES[i][j]`).
fn rule(e: Term, de: Term) -> OutLevel {
    use OutLevel as O;
    use Term::*;
    match (e, de) {
        (NL, NL) => O::ND,
        (NL, NS) => O::ND,
        (NL, Z) => O::NS,
        (NL, PS) => O::NS,
        (NL, PL) => O::Z,
        (NS, NL) => O::ND,
        (NS, NS) => O::NS,
        (NS, Z) => O::NS,
        (NS, PS) => O::Z,
        (NS, PL) => O::PS,
        (Z, NL) => O::NS,
        (Z, NS) => O::NS,
        (Z, Z) => O::Z,
        (Z, PS) => O::PS,
        (Z, PL) => O::PS,
        (PS, NL) => O::NS,
        (PS, NS) => O::Z,
        (PS, Z) => O::PS,
        (PS, PS) => O::PS,
        (PS, PL) => O::PD,
        (PL, NL) => O::Z,
        (PL, NS) => O::PS,
        (PL, Z) => O::PS,
        (PL, PS) => O::PD,
        (PL, PL) => O::PD,
    }
}

/// Mamdani fuzzy controller: returns Δ-Q normalised to [-1, +1].
pub fn fuzzy_delta_controller(e: f64, de_dt: f64) -> f64 {
    let me = mu_e(e);
    let md = mu_de(de_dt);
    let mut num = 0.0;
    let mut den = 0.0;
    for &i in &TERMS {
        for &j in &TERMS {
            let w = me[i.index()].min(md[j.index()]);
            if w == 0.0 {
                continue;
            }
            let out = rule(i, j).value();
            num += w * out;
            den += w;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

// ── MDP-MPC controller (receding-horizon DP) ─────────────────────────────────
//
// At each tick the controller looks at forecasts for the next H ticks, builds a
// finite-horizon discrete MDP on a grid of (indoor temperature × time), runs
// backward induction to compute the optimal action sequence, and executes the
// FIRST action. Linear interpolation of V[k+1] removes grid-quantisation.

#[allow(clippy::too_many_arguments)]
pub fn mdp_mpc_controller(
    t_in_now: f64,
    forecast: &[f64],
    horizon_h: f64,
    n_levels: usize,
    t_target: f64,
    dt_h: f64,
    q_max: f64,
    house: &HouseParams,
    comfort_penalty: f64,
    cost_per_kwh: f64,
    track_weight: f64,
) -> f64 {
    mdp_mpc_controller_with_bounds(
        t_in_now,
        forecast,
        horizon_h,
        n_levels,
        t_target,
        dt_h,
        0.0,
        q_max,
        house,
        comfort_penalty,
        cost_per_kwh,
        track_weight,
    )
}

/// Bidirectional variant of [`mdp_mpc_controller`]. `q_min < 0` lets the action
/// grid include cooling commands; `q_min = 0` is the original heating-only MDP.
#[allow(clippy::too_many_arguments)]
pub fn mdp_mpc_controller_with_bounds(
    t_in_now: f64,
    forecast: &[f64],
    horizon_h: f64,
    n_levels: usize,
    t_target: f64,
    dt_h: f64,
    q_min: f64,
    q_max: f64,
    house: &HouseParams,
    comfort_penalty: f64,
    cost_per_kwh: f64,
    track_weight: f64,
) -> f64 {
    let h = forecast.len().min((horizon_h / dt_h).round() as usize);
    // T_in grid covering [T_target − 10, T_target + 10] in fine 0.1°F steps.
    let t_lo = t_target - 10.0;
    let t_hi = t_target + 10.0;
    let t_step = 0.1;
    let n_t = ((t_hi - t_lo) / t_step).round() as usize + 1;
    let t_val = |i: usize| t_lo + i as f64 * t_step;
    // Linear-interpolate V[k+1] at continuous T_next, clamped to the grid.
    let interp_v = |v_row: &[f64], t: f64| -> f64 {
        let x = (t - t_lo) / t_step;
        if x <= 0.0 {
            return v_row[0];
        }
        if x >= (n_t - 1) as f64 {
            return v_row[n_t - 1];
        }
        let i0 = x.floor() as usize;
        let i1 = i0 + 1;
        let w = x - i0 as f64;
        (1.0 - w) * v_row[i0] + w * v_row[i1]
    };
    // Action grid.
    let mut actions: Vec<f64> = (0..n_levels)
        .map(|k| q_min + (q_max - q_min) * k as f64 / (n_levels as f64 - 1.0))
        .collect();
    if q_min < 0.0 && q_max > 0.0 && !actions.iter().any(|q| q.abs() < 1e-12) {
        actions.push(0.0);
        actions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    // Backward induction. V[k][i] = optimal cost-to-go from tick k with T_in in bin i.
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; n_t]; h + 1];
    let mut pi: Vec<Vec<f64>> = vec![vec![0.0; n_t]; h];
    if h >= 1 {
        for k in (0..h).rev() {
            let t_out_k = forecast[k];
            for i in 0..n_t {
                let mut best_q = 0.0;
                let mut best_val = f64::INFINITY;
                for &q in &actions {
                    let t_now = t_val(i);
                    let t_next = t_now + ((t_out_k - t_now) / house.tau + q * house.g) * dt_h;
                    let future_v = interp_v(&v[k + 1], t_next);
                    let energy_cost = cost_per_kwh * q.abs() * dt_h;
                    let track_err = t_now - t_target;
                    let dev = (track_err.abs() - 2.0).max(0.0);
                    let comfort_cost =
                        (track_weight * track_err * track_err + comfort_penalty * dev * dev) * dt_h;
                    let total = energy_cost + comfort_cost + future_v;
                    if total < best_val {
                        best_val = total;
                        best_q = q;
                    }
                }
                v[k][i] = best_val;
                pi[k][i] = best_q;
            }
        }
    }
    // Look up the policy at the continuous initial T_in via the dominant cell.
    if h == 0 {
        return 0.0;
    }
    let x = (t_in_now - t_lo) / t_step;
    if x <= 0.0 {
        return pi[0][0];
    }
    if x >= (n_t - 1) as f64 {
        return pi[0][n_t - 1];
    }
    let i0 = x.floor() as usize;
    let i1 = i0 + 1;
    let w = x - i0 as f64;
    if w < 0.5 {
        pi[0][i0]
    } else {
        pi[0][i1]
    }
}

// -----------------------------------------------------------------------------
// CONTROLLER STATION — concrete leaf of ControllerStation<TempObs, f64>
// -----------------------------------------------------------------------------

/// Common base for the four temperature controllers — owns the
/// [`ControllerState`] across ticks and clamps the heater command to
/// `[0, Q_max]`. The concrete control law dispatches on the stored
/// [`ControllerSpec`] (see the module-level faithful-simplification flag).
pub struct TempControllerBase {
    core: StationCore,
    ctrl: ControllerCore<TempObs, f64>,
    spec: ControllerSpec,
    ctrl_state: ControllerState,
    q_min_cached: f64,
    q_max_cached: f64,
}

impl TempControllerBase {
    pub fn new(id: &str, q_max: f64, spec: ControllerSpec) -> Self {
        Self::new_with_bounds(id, 0.0, q_max, spec)
    }

    pub fn new_with_bounds(id: &str, q_min: f64, q_max: f64, spec: ControllerSpec) -> Self {
        let mut st = TempControllerBase {
            core: StationCore::new(id),
            ctrl: ControllerCore::new(),
            spec,
            ctrl_state: ControllerState::default(),
            q_min_cached: q_min,
            q_max_cached: q_max,
        };
        // Intrinsic invariant: every emitted control must lie in [Q_min, Q_max].
        let v = intrinsic_check::<dyn DESStation>(
            "temp-control.u-in-saturation",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<TempControllerBase>().unwrap();
                let lo = st.q_min_cached;
                let hi = st.q_max_cached;
                for &u in &st.ctrl.control_history {
                    if u < lo - 1e-9 || u > hi + 1e-9 {
                        return false;
                    }
                }
                true
            },
            Some("0 ≤ u ≤ Q_max".to_string()),
            Some(Box::new(|s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<TempControllerBase>().unwrap();
                format!(
                    "n={}  Q_min={}  Q_max={}",
                    st.ctrl.control_history.len(),
                    st.q_min_cached,
                    st.q_max_cached
                )
            })),
            Some("temp-control-intrinsic".to_string()),
            Some("controller emitted a u outside its saturation band".to_string()),
        );
        st.add_validator(v.boxed());
        st
    }
}

impl DESStation for TempControllerBase {
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

impl ControllerStation<TempObs, f64> for TempControllerBase {
    fn controller_core(&self) -> &ControllerCore<TempObs, f64> {
        &self.ctrl
    }
    fn controller_core_mut(&mut self) -> &mut ControllerCore<TempObs, f64> {
        &mut self.ctrl
    }
    fn control_law(&mut self, observation: &TempObs, _tick: f64, _time: f64) -> f64 {
        let spec = self.spec;
        controller_step(&spec, &mut self.ctrl_state, observation)
    }
    fn u_min(&self) -> Option<f64> {
        Some(self.q_min_cached)
    }
    fn u_max(&self) -> Option<f64> {
        Some(self.q_max_cached)
    }
    fn reset(&mut self) {
        let c = self.controller_core_mut();
        c.ticks_processed = 0;
        c.control_history.clear();
        c.observation_history.clear();
        self.ctrl_state = ControllerState::default();
    }
}

/// `class BangBangController` — heater FULL ON below target, OFF otherwise.
pub fn bang_bang_controller(id: &str, q_max: f64) -> TempControllerBase {
    TempControllerBase::new(id, q_max, ControllerSpec::BangBang)
}

/// `class PIDController` — classical PID feedback.
pub fn pid_controller(id: &str, q_max: f64, kp: f64, ki: f64, kd: f64) -> TempControllerBase {
    TempControllerBase::new(id, q_max, ControllerSpec::Pid { kp, ki, kd })
}

/// `class FuzzyController` — Mamdani fuzzy-PI controller.
pub fn fuzzy_controller(id: &str, q_max: f64) -> TempControllerBase {
    TempControllerBase::new(id, q_max, ControllerSpec::Fuzzy)
}

/// `class MdpMpcController` — receding-horizon DP. `spec` must be
/// [`ControllerSpec::MdpMpc`]; any other variant panics (TS used a narrowed type).
pub fn mdp_mpc_controller_station(
    id: &str,
    q_max: f64,
    spec: ControllerSpec,
) -> TempControllerBase {
    assert!(
        matches!(spec, ControllerSpec::MdpMpc { .. }),
        "mdp-mpc controller requires an MdpMpc spec"
    );
    TempControllerBase::new(id, q_max, spec)
}

/// Factory: build the right controller leaf for a spec.
pub fn make_temp_controller(spec: ControllerSpec, q_max: f64, id: &str) -> TempControllerBase {
    TempControllerBase::new(id, q_max, spec)
}

/// Factory for a heat-pump style controller that can cool (`q_min < 0`) as well
/// as heat (`q_max > 0`).
pub fn make_temp_controller_with_bounds(
    spec: ControllerSpec,
    q_min: f64,
    q_max: f64,
    id: &str,
) -> TempControllerBase {
    TempControllerBase::new_with_bounds(id, q_min, q_max, spec)
}

// -----------------------------------------------------------------------------
// SIMULATION RUNNER — orchestrates the stations on each tick.
// -----------------------------------------------------------------------------

/// Episode configuration.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Target indoor temperature (°F).
    pub t_target: f64,
    /// Comfort band (±band in °F). `None` ⇒ 2.
    pub band: Option<f64>,
    /// Total simulated time (hours).
    pub duration_h: f64,
    /// Tick length (minutes).
    pub dt_min: f64,
    /// Controller specification.
    pub controller: ControllerSpec,
    /// House parameters override.
    pub house: Option<HouseParamsPartial>,
    /// Outdoor temperature pattern override.
    pub outdoor: Option<OutdoorPatternPartial>,
    /// Cost per kWh ($).
    pub cost_per_kwh: f64,
    /// Comfort penalty ($ per (°F)² per hour outside the band).
    pub comfort_penalty: f64,
    /// Sensor noise std (°F).
    pub sensor_noise_std: Option<f64>,
    /// Forecast noise std (°F).
    pub forecast_noise_std: Option<f64>,
    /// Forecast horizon (hours) used by mpc-style controllers.
    pub forecast_horizon_h: Option<f64>,
    /// PRNG seed.
    pub seed: Option<u32>,
}

/// One per-tick record.
#[derive(Clone, Debug)]
pub struct TickRecord {
    pub tick: usize,
    pub t_h: f64,
    pub t_out_true: f64,
    pub t_out_meas: f64,
    pub t_in_true: f64,
    pub t_in_meas: f64,
    pub error: f64,
    pub q: f64,
    pub energy_cum_kwh: f64,
    pub in_band: bool,
    pub violation_fh: f64,
}

/// Result of a single controller episode.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub cfg: SimConfig,
    pub trace: Vec<TickRecord>,
    pub energy_kwh: f64,
    pub comfort_pct: f64,
    pub violation_fh: f64,
    pub cost: f64,
    // Convenience time series for plotting / animation.
    pub ticks: Vec<usize>,
    pub t_in: Vec<f64>,
    pub t_out: Vec<f64>,
    pub q: Vec<f64>,
    pub energy: Vec<f64>,
}

/// Run a single controller through a full episode.
pub fn run_temp_control(cfg: SimConfig) -> RunResult {
    let cls = "runTempControl";
    Preconditions::positive(cls, "cfg.dt_min", cfg.dt_min).expect("cfg.dt_min");
    Preconditions::positive(cls, "cfg.duration_h", cfg.duration_h).expect("cfg.duration_h");
    Preconditions::finite(cls, "cfg.T_target", cfg.t_target).expect("cfg.T_target");
    if let Some(band) = cfg.band {
        Preconditions::positive(cls, "cfg.band", band).expect("cfg.band");
    }
    if let Some(s) = cfg.sensor_noise_std {
        Preconditions::non_negative(cls, "cfg.sensorNoiseStd", s).expect("cfg.sensorNoiseStd");
    }
    if let Some(s) = cfg.forecast_noise_std {
        Preconditions::non_negative(cls, "cfg.forecastNoiseStd", s).expect("cfg.forecastNoiseStd");
    }
    if let Some(s) = cfg.forecast_horizon_h {
        Preconditions::positive(cls, "cfg.forecastHorizon_h", s).expect("cfg.forecastHorizon_h");
    }
    Preconditions::non_negative(cls, "cfg.cost_per_kWh", cfg.cost_per_kwh)
        .expect("cfg.cost_per_kWh");
    Preconditions::non_negative(cls, "cfg.comfort_penalty", cfg.comfort_penalty)
        .expect("cfg.comfort_penalty");
    if let Some(h) = &cfg.house {
        if let Some(q) = h.q_max {
            Preconditions::positive(cls, "cfg.house.Q_max", q).expect("cfg.house.Q_max");
        }
        if let Some(q) = h.q_min {
            Preconditions::finite(cls, "cfg.house.Q_min", q).expect("cfg.house.Q_min");
        }
        if let Some(t) = h.tau {
            Preconditions::positive(cls, "cfg.house.tau", t).expect("cfg.house.tau");
        }
    }

    let house = DEFAULT_HOUSE.merged(&cfg.house.unwrap_or_default());
    Preconditions::check(
        cls,
        "cfg.house.Q_min/Q_max",
        "satisfy Q_min < Q_max",
        house.q_min < house.q_max,
        Some(format!("{}/{}", house.q_min, house.q_max)),
    )
    .expect("cfg.house.Q_min/Q_max");
    let outdoor = DEFAULT_OUTDOOR.merged(&cfg.outdoor.unwrap_or_default());
    let dt_h = cfg.dt_min / 60.0;
    let n = (cfg.duration_h / dt_h).round() as usize;
    let t_target = cfg.t_target;
    let band = cfg.band.unwrap_or(2.0);
    let sensor_std = cfg.sensor_noise_std.unwrap_or(0.0);
    let forecast_std = cfg.forecast_noise_std.unwrap_or(0.0);
    let horizon_h = cfg.forecast_horizon_h.unwrap_or(6.0);
    let horizon_ticks = (horizon_h / dt_h).round() as usize;
    let cost_per_kwh = cfg.cost_per_kwh;
    let comfort_penalty = cfg.comfort_penalty;
    let seed = cfg.seed.unwrap_or(12345);
    let mut rng = SeededRandom::new(seed);
    let mut fc_rng = SeededRandom::new(seed ^ 0xa5a5_a5a5);
    let mut sensor_rng = SeededRandom::new(seed ^ 0x5a5a_5a5a);

    let mut t_in = house.t_init;
    let mut energy = 0.0;
    let mut violation = 0.0;
    let mut in_band = 0usize;
    let mut trace: Vec<TickRecord> = Vec::new();
    let mut ticks: Vec<usize> = Vec::new();
    let mut t_in_trace: Vec<f64> = Vec::new();
    let mut t_out_trace: Vec<f64> = Vec::new();
    let mut q_trace: Vec<f64> = Vec::new();
    let mut energy_trace: Vec<f64> = Vec::new();

    let mut ctrl =
        make_temp_controller_with_bounds(cfg.controller, house.q_min, house.q_max, "tempctrl");
    for k in 0..n {
        let t_h = k as f64 * dt_h;
        let t_out_true = true_outdoor_temp(t_h, &outdoor, Some(&mut rng as &mut dyn RandomSource));
        let t_in_meas = if sensor_std > 0.0 {
            t_in + sensor_std
                * (sensor_rng.next_float()
                    + sensor_rng.next_float()
                    + sensor_rng.next_float()
                    + sensor_rng.next_float()
                    - 2.0)
                / 0.577
        } else {
            t_in
        };
        // Forecast: peek ahead at the noiseless mean trajectory + Gaussian noise.
        let mut forecast: Vec<f64> = Vec::with_capacity(horizon_ticks);
        for i in 0..horizon_ticks {
            let t_future = t_h + i as f64 * dt_h;
            let periodic = outdoor.mean
                + outdoor.amp
                    * (2.0 * std::f64::consts::PI * (t_future - outdoor.phase) / 24.0).sin();
            let fc_noise = if forecast_std > 0.0 {
                forecast_std
                    * (fc_rng.next_float()
                        + fc_rng.next_float()
                        + fc_rng.next_float()
                        + fc_rng.next_float()
                        - 2.0)
                    / 0.577
            } else {
                0.0
            };
            forecast.push(periodic + fc_noise);
        }
        let obs = TempObs {
            t_target,
            t_in_meas,
            forecast,
            dt_h,
            q_min: house.q_min,
            q_max: house.q_max,
            house,
        };
        let q = ctrl.step(obs, k as f64, t_h);
        // House physics + energy + comfort.
        let t_next = house_step(t_in, t_out_true, q, dt_h, &house);
        energy += q.abs() * dt_h;
        let dev = (t_in - t_target).abs();
        let in_b = dev <= band;
        if in_b {
            in_band += 1;
        }
        violation += (dev - band).max(0.0) * dt_h;
        trace.push(TickRecord {
            tick: k,
            t_h,
            t_out_true,
            t_out_meas: t_out_true,
            t_in_true: t_in,
            t_in_meas,
            error: t_target - t_in_meas,
            q,
            energy_cum_kwh: energy,
            in_band: in_b,
            violation_fh: violation,
        });
        ticks.push(k);
        t_in_trace.push(t_in);
        t_out_trace.push(t_out_true);
        q_trace.push(q);
        energy_trace.push(energy);
        t_in = t_next;
    }
    let comfort_pct = in_band as f64 / n as f64;
    let cost = cost_per_kwh * energy + comfort_penalty * violation;
    RunResult {
        cfg,
        trace,
        energy_kwh: energy,
        comfort_pct,
        violation_fh: violation,
        cost,
        ticks,
        t_in: t_in_trace,
        t_out: t_out_trace,
        q: q_trace,
        energy: energy_trace,
    }
}

#[cfg(test)]
mod tests {
    //! Temp-control tests: a PID controller must settle the indoor temperature
    //! to the setpoint band, and the bang-bang baseline must keep the heater
    //! command within its saturation limits.

    use super::*;

    fn base_cfg(controller: ControllerSpec) -> SimConfig {
        SimConfig {
            t_target: 70.0,
            band: Some(2.0),
            duration_h: 12.0,
            dt_min: 1.0,
            controller,
            house: None,
            outdoor: None,
            cost_per_kwh: 0.15,
            comfort_penalty: 1.0,
            sensor_noise_std: None,
            forecast_noise_std: None,
            forecast_horizon_h: None,
            seed: Some(7),
        }
    }

    #[test]
    fn pid_settles_to_setpoint() {
        let res = run_temp_control(base_cfg(ControllerSpec::Pid {
            kp: 4.0,
            ki: 2.0,
            kd: 0.5,
        }));
        // After a 12-hour run the final indoor temperature should be close to the
        // 70°F setpoint.
        let final_t_in = *res.t_in.last().unwrap();
        assert!(
            (final_t_in - 70.0).abs() < 2.0,
            "did not settle: T_in={final_t_in}"
        );
        // And the controller should spend most of the run inside the comfort band.
        assert!(
            res.comfort_pct > 0.5,
            "comfort too low: {}",
            res.comfort_pct
        );
    }

    #[test]
    fn bang_bang_respects_saturation() {
        let res = run_temp_control(base_cfg(ControllerSpec::BangBang));
        for rec in &res.trace {
            assert!(
                rec.q >= -1e-9 && rec.q <= DEFAULT_HOUSE.q_max + 1e-9,
                "Q out of band: {}",
                rec.q
            );
        }
    }

    #[test]
    fn house_step_relaxes_toward_outside_without_heat() {
        // With no heat, an over-warm house cools toward the outside temperature.
        let h = DEFAULT_HOUSE;
        let next = house_step(70.0, 30.0, 0.0, 1.0 / 60.0, &h);
        assert!(next < 70.0 && next > 30.0, "unexpected relaxation: {next}");
    }
}
